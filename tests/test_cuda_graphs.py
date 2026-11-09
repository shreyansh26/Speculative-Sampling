import copy
import unittest

import torch

from autoregressive_sampling import AutoregressiveDecoder, autoregressive_sampling
from benchmarks.benchmark import check_graph_reuse
from qwen3_cache import Qwen3Cache, kv_splits_for_prompt
from speculative_sampling import SpeculativeDecoder, speculative_sampling
from tests.test_sampling import full_prefix_greedy, storage_pointers, tiny_qwen


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA graph tests require a GPU')
class CudaGraphTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    @torch.inference_mode()
    def test_autoregressive_replay_matches_eager_and_preserves_rng(self):
        for temperature in (0, 0.5):
            for prompt_length in (1, 3):
                model = tiny_qwen().cuda()
                prompt = torch.tensor([[1, 5, 4]], device='cuda')[:, :prompt_length]
                eager = AutoregressiveDecoder(model, prompt, 18, temperature=temperature)
                graphed = AutoregressiveDecoder(model, prompt, 18, temperature=temperature, cuda_graphs=True)
                for seed in (42, 5, 17):
                    changed_prompt = prompt + seed % 3
                    torch.manual_seed(seed)
                    expected = eager.generate(changed_prompt)
                    expected_rng = torch.cuda.get_rng_state()
                    torch.manual_seed(seed)
                    actual = graphed.generate(changed_prompt)
                    torch.testing.assert_close(actual, expected)
                    torch.testing.assert_close(torch.cuda.get_rng_state(), expected_rng)
                    self.assertEqual(set(graphed.graphs), {'decode'})
                    pointers = [graphed.tokens.data_ptr(), graphed.token.data_ptr(), graphed.length.data_ptr()]
                    for layer in graphed.cache.cache.layers:
                        pointers.extend((layer.keys.data_ptr(), layer.values.data_ptr(), layer.cumulative_length.data_ptr()))
                    if seed == 42:
                        initial_pointers = pointers
                        graph = graphed.graphs['decode']
                    else:
                        self.assertEqual(pointers, initial_pointers)
                        self.assertIs(graphed.graphs['decode'], graph)
                calls = []
                handle = model.register_forward_pre_hook(
                    lambda module, args: calls.append((args[0].shape[1], torch.cuda.is_current_stream_capturing()))
                )
                graphed.generate(prompt)
                handle.remove()
                self.assertEqual(calls, [(prompt_length, False)])
                with self.assertRaises(ValueError):
                    graphed.generate(torch.ones((1, 19), dtype=torch.long, device='cuda'))

    def test_autoregressive_short_output_budget(self):
        model = tiny_qwen().cuda()
        prompt = torch.tensor([[1]], device='cuda')
        for new_tokens in (0, 1, 2):
            actual = autoregressive_sampling(model, prompt, 1 + new_tokens, temperature=0, cuda_graphs=True)
            torch.testing.assert_close(actual, full_prefix_greedy(model, prompt, 1 + new_tokens))
        decoder = AutoregressiveDecoder(model, prompt, 2, temperature=0, cuda_graphs=True)
        with self.assertRaises(ValueError):
            decoder.capture_decode()
        decoder.generate()
        decoder.capture_decode()
        self.assertFalse(decoder.graphs)
        self.assertEqual(decoder.capture_seconds, 0)

    @torch.inference_mode()
    def test_replay_rng_and_new_prompt_match_eager(self):
        for temperature in (0, 0.5):
            for prompt_length in (1, 3):
                with self.subTest(temperature=temperature, prompt_length=prompt_length):
                    target = tiny_qwen().cuda()
                    draft = copy.deepcopy(target)
                    draft.lm_head.weight.neg_()
                    prompt = torch.tensor([[1, 5, 4]], device='cuda')[:, :prompt_length]
                    eager = SpeculativeDecoder(target, draft, prompt, 18, temperature=temperature)
                    graphed = SpeculativeDecoder(target, draft, prompt, 18, temperature=temperature, cuda_graphs=True)
                    for seed in (42, 5, 17):
                        changed_prompt = prompt + seed % 3
                        torch.manual_seed(seed)
                        expected = eager.generate(changed_prompt)
                        expected_rng = torch.cuda.get_rng_state()
                        torch.manual_seed(seed)
                        actual = graphed.generate(changed_prompt)
                        torch.testing.assert_close(actual, expected)
                        torch.testing.assert_close(torch.cuda.get_rng_state(), expected_rng)
                        self.assertEqual(set(graphed.graphs), {'draft', 'verify', 'append'})
                        if seed == 42:
                            pointers = storage_pointers(graphed)
                            graphs = dict(graphed.graphs)
                        else:
                            self.assertEqual(storage_pointers(graphed), pointers)
                            self.assertEqual(graphed.graphs, graphs)

                    calls = []
                    handle = target.register_forward_pre_hook(
                        lambda module, args: calls.append((args[0].shape[1], torch.cuda.is_current_stream_capturing()))
                    )
                    graphed.generate(prompt)
                    handle.remove()
                    self.assertEqual(calls, [(prompt_length, False)])

    @torch.inference_mode()
    def test_replay_rejection_rollback_and_full_acceptance(self):
        target = tiny_qwen().to(device='cuda', dtype=torch.bfloat16)
        draft = copy.deepcopy(target)
        prompt = torch.tensor([[1, 5, 4]], device='cuda')
        decoder = SpeculativeDecoder(target, draft, prompt, 15, cuda_graphs=True, temperature=0)
        expected = full_prefix_greedy(target, prompt, 15)
        decoder.prefill()
        rejected_at = torch.tensor(3, device='cuda')
        step = 0

        def force_rejection(module, args, output):
            nonlocal step
            scores = output.logits[:, -1, :]
            wrong = (scores.argmax(-1, keepdim=True) + 1) % scores.shape[-1]
            forced = torch.full_like(scores, -1000).scatter_(1, wrong, 1000)
            scores.copy_(torch.where(rejected_at == step, forced, scores))
            step = (step + 1) % decoder.lookahead

        handle = draft.register_forward_hook(force_rejection)
        decoder.capture_stages()
        handle.remove()
        pointers = storage_pointers(decoder)
        for rejection in (0, 1, 3):
            with self.subTest(rejected_at=rejection):
                rejected_at.fill_(rejection)
                decoder.prefill()
                while decoder.length.item() < 15:
                    decoder.draft()
                    decoder.verify()
                    decoder.append()
                    self.assertEqual(decoder.accepted.item(), rejection)
                    length = decoder.length.item()
                    torch.testing.assert_close(decoder.tokens[:, :length], expected[:, :length])
                    for runner, cursor in ((decoder.target, length - 1), (decoder.proposer, length - 2)):
                        for layer in runner.cache.layers:
                            self.assertEqual(layer.cumulative_length.item(), cursor)
                            layer.keys[:, :, cursor:, :].fill_(100)
                            layer.values[:, :, cursor:, :].fill_(100)
                self.assertEqual(storage_pointers(decoder), pointers)

    @torch.inference_mode()
    def test_flash_attention_graphs_reuse_across_prompt_and_output_lengths(self):
        target = tiny_qwen().to(device='cuda', dtype=torch.bfloat16)
        draft = copy.deepcopy(target)
        draft.lm_head.weight.neg_()
        initial = torch.tensor([[1, 5, 4]], device='cuda')
        for temperature in (0, 0.5):
            for decoder_type in (AutoregressiveDecoder, SpeculativeDecoder):
                models = (target,) if decoder_type is AutoregressiveDecoder else (target, draft)
                eager = decoder_type(*models, initial, 12, temperature=temperature, max_seq_len=48)
                graphed = decoder_type(
                    *models, initial, 12, temperature=temperature, max_seq_len=48, cuda_graphs=True,
                )
                cache = graphed.cache if decoder_type is AutoregressiveDecoder else graphed.target
                self.assertIsNotNone(cache.kernel)
                for index, (prompt_length, new_tokens) in enumerate(((3, 9), (17, 20), (1, 2), (7, 1), (2, 15))):
                    prompt = (torch.arange(prompt_length, device='cuda')[None] + 1) % 31
                    target_len = prompt_length + new_tokens
                    torch.manual_seed(index)
                    expected = eager.generate(prompt, target_len=target_len)
                    rng = torch.cuda.get_rng_state()
                    torch.manual_seed(index)
                    actual = graphed.generate(prompt, target_len=target_len)
                    torch.testing.assert_close(actual, expected)
                    torch.testing.assert_close(torch.cuda.get_rng_state(), rng)
                    self.assertEqual(actual.shape[1], target_len)
                    pointers = [graphed.prompt.data_ptr(), graphed.tokens.data_ptr(),
                                cache.valid_length.data_ptr()]
                    for runner in ((graphed.cache,) if decoder_type is AutoregressiveDecoder
                                   else (graphed.target, graphed.proposer)):
                        for layer in runner.cache.layers:
                            pointers.extend((layer.keys.data_ptr(), layer.values.data_ptr()))
                    if index == 0:
                        graphs, original_pointers = dict(graphed.graphs), pointers
                    else:
                        self.assertEqual(graphed.graphs, graphs)
                        self.assertEqual(pointers, original_pointers)
                with self.assertRaises(ValueError):
                    graphed.generate(initial, target_len=49)
                with self.assertRaises(ValueError):
                    graphed.generate(torch.ones((1, 49), device='cuda', dtype=torch.long))

    @torch.inference_mode()
    def test_flash_attention_logits_ignore_unused_capacity(self):
        model = tiny_qwen().to(device='cuda', dtype=torch.bfloat16)
        for capacity, prompt_lengths in (
            (64, (1, 17, 45)), (2048, (1, 17, 45)), (32832, (1024, 8192, 32768)),
        ):
            cache = Qwen3Cache(model, capacity)
            for prompt_length in prompt_lengths:
                for query_length in (1, 2, 4):
                    prefix = (torch.arange(prompt_length, device='cuda')[None] + 1) % 31
                    query = (torch.arange(query_length, device='cuda')[None] + 5) % 31
                    if cache.cache.layers[0].is_initialized:
                        cache.rewind(torch.zeros((), dtype=torch.long, device='cuda'))
                    cache.num_splits = kv_splits_for_prompt(prompt_length)
                    prefill_logits = cache.forward(prefix, prefill=True)
                    reference = model(prefix, use_cache=False).logits[:, -1:]
                    torch.testing.assert_close(prefill_logits, reference, rtol=0.02, atol=0.002)
                    for layer in cache.cache.layers:
                        layer.keys[:, :, prompt_length:].fill_(100)
                        layer.values[:, :, prompt_length:].fill_(100)
                    expected = model(torch.cat((prefix, query), dim=1), use_cache=False).logits[:, -query_length:]
                    actual = cache.forward(query, logits_to_keep=0)
                    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.002)
                    self.assertEqual(cache.valid_length.item(), prompt_length + query_length)
                    self.assertEqual(model.config._attn_implementation, 'sdpa')

    @torch.inference_mode()
    def test_long_context_replay(self):
        target = tiny_qwen().to(device='cuda', dtype=torch.bfloat16)
        draft = copy.deepcopy(target)
        draft.lm_head.weight.neg_()
        inputs = [(torch.arange(length, device='cuda')[None] % 30 + 1)
                  for length in (511, 512, 1024, 4095, 4096, 8192, 32768)]
        for temperature in (0, 0.5):
            for cls, models in ((AutoregressiveDecoder, (target,)), (SpeculativeDecoder, (target, draft))):
                decoder = cls(*models, inputs[0], 1032, temperature=temperature,
                              max_seq_len=32776, cuda_graphs=True)
                check_graph_reuse(decoder, inputs, new_tokens=8)
                self.assertEqual(set(decoder.graph_cache), {1, 4, 16})

    def test_short_output_budget(self):
        model = tiny_qwen().cuda()
        prompt = torch.tensor([[1]], device='cuda')
        for new_tokens in (0, 1, 2):
            actual = speculative_sampling(
                model, model, prompt, 1 + new_tokens, None, temperature=0,
                debug=False, cuda_graphs=True,
            )
            torch.testing.assert_close(actual, full_prefix_greedy(model, prompt, 1 + new_tokens))
        decoder = SpeculativeDecoder(model, model, prompt, 2, temperature=0, cuda_graphs=True)
        with self.assertRaises(ValueError):
            decoder.capture_stages()
        decoder.generate()
        decoder.capture_stages()
        self.assertFalse(decoder.graphs)
        self.assertEqual(decoder.capture_seconds, 0)


if __name__ == '__main__':
    unittest.main()
