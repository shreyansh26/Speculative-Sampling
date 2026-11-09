import copy
import unittest
from unittest.mock import patch

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from autoregressive_sampling import autoregressive_sampling
from speculative_sampling import SpeculativeDecoder, speculative_sampling
from utils import sample_probabilities


def tiny_qwen(vocab_size=31):
    torch.manual_seed(7)
    config = Qwen3Config(
        vocab_size=vocab_size, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=8,
    )
    return Qwen3ForCausalLM(config).eval()


@torch.inference_mode()
def full_prefix_greedy(model, prompt, length):
    tokens = prompt.clone()
    while tokens.shape[1] < length:
        token = model(tokens, use_cache=False).logits[:, -1, :].argmax(-1, keepdim=True)
        tokens = torch.cat((tokens, token), dim=1)
    return tokens


def storage_pointers(decoder):
    tensors = [value for value in vars(decoder).values() if isinstance(value, torch.Tensor)]
    for runner in (decoder.target, decoder.proposer):
        for layer in runner.cache.layers:
            tensors.extend((layer.keys, layer.values, layer.cumulative_length))
    return [tensor.data_ptr() for tensor in tensors]


class SamplingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_cached_autoregression_matches_full_prefix(self):
        model = tiny_qwen()
        prompt = torch.tensor([[1, 5, 4]])
        expected = full_prefix_greedy(model, prompt, 20)
        query_lengths = []
        handle = model.register_forward_pre_hook(lambda module, args: query_lengths.append(args[0].shape[1]))
        actual = autoregressive_sampling(model, prompt, 20, temperature=0)
        handle.remove()
        torch.testing.assert_close(actual, expected)
        self.assertEqual(query_lengths, [3] + [1] * 16)

    @torch.inference_mode()
    def test_rejection_rollback_and_full_acceptance(self):
        for prompt_length in (1, 3):
            for rejected_at in (0, 2, 4):
                with self.subTest(prompt_length=prompt_length, rejected_at=rejected_at):
                    target = tiny_qwen()
                    draft = copy.deepcopy(target)
                    prompt = torch.tensor([[1, 5, 4]])[:, :prompt_length]
                    expected = full_prefix_greedy(target, prompt, 19)
                    decoder = SpeculativeDecoder(target, draft, prompt, 19, lookahead=4, temperature=0)
                    decoder.prefill()
                    pointers = storage_pointers(decoder)
                    draft_queries = []
                    target_queries = []

                    def force_rejection(module, args, output):
                        step = len(draft_queries)
                        draft_queries.append(args[0].shape[1])
                        if step == rejected_at:
                            scores = output.logits[:, -1, :]
                            wrong = (scores.argmax(-1, keepdim=True) + 1) % scores.shape[-1]
                            scores.fill_(-1000).scatter_(1, wrong, 1000)

                    draft_hook = draft.register_forward_hook(force_rejection)
                    target_hook = target.register_forward_pre_hook(
                        lambda module, args: target_queries.append(args[0].shape[1])
                    )
                    with patch.object(torch.Tensor, 'item', side_effect=AssertionError('Host scalar read')), \
                         patch.object(torch.Tensor, '__bool__', side_effect=AssertionError('Host branch')):
                        decoder.draft()
                        decoder.verify()
                        decoder.append()
                    draft_hook.remove()
                    self.assertEqual(decoder.accepted.item(), rejected_at)
                    self.assertEqual(draft_queries, [2, 1, 1, 1])
                    length = decoder.length.item()
                    torch.testing.assert_close(decoder.tokens[:, :length], expected[:, :length])

                    # Poison abandoned slots so subsequent rounds exercise attention masking.
                    for runner, cached_length in ((decoder.target, length - 1), (decoder.proposer, length - 2)):
                        for layer in runner.cache.layers:
                            self.assertEqual(layer.cumulative_length.item(), cached_length)
                            layer.keys[:, :, cached_length:, :].fill_(100)
                            layer.values[:, :, cached_length:, :].fill_(100)
                    while decoder.length.item() < 19:
                        decoder.draft()
                        decoder.verify()
                        decoder.append()
                        length = decoder.length.item()
                        torch.testing.assert_close(decoder.tokens[:, :length], expected[:, :length])
                    target_hook.remove()
                    self.assertTrue(all(length == 5 for length in target_queries))
                    self.assertEqual(storage_pointers(decoder), pointers)
                    self.assertEqual(decoder.length.item(), 19)

    def test_output_budget_and_lookahead(self):
        model = tiny_qwen()
        prompt = torch.tensor([[1, 5, 4]])
        for new_tokens in (0, 1, 2, 6, 7):
            for lookahead in (1, 3, 4, 8):
                with self.subTest(new_tokens=new_tokens, lookahead=lookahead):
                    length = prompt.shape[1] + new_tokens
                    actual = speculative_sampling(
                        model, model, prompt, length, None, lookahead, temperature=0, debug=False
                    )
                    torch.testing.assert_close(actual, full_prefix_greedy(model, prompt, length))
                    self.assertEqual(actual.shape[1], length)

    @torch.inference_mode()
    def test_stochastic_cached_logits_match_full_prefix(self):
        target = tiny_qwen()
        draft = copy.deepcopy(target)
        draft.lm_head.weight.neg_()
        decoder = SpeculativeDecoder(target, draft, torch.tensor([[1, 5, 4]]), 21, temperature=0.5)
        decoder.prefill()
        while decoder.length.item() < decoder.target_len:
            length = decoder.length.item()
            decoder.draft()
            prefix = torch.cat((decoder.tokens[:, :length - 1], decoder.block), dim=1)
            expected = target(prefix, use_cache=False).logits[:, -(decoder.lookahead + 1):, :]
            for step in range(decoder.lookahead):
                draft_logits = draft(prefix[:, :length + step], use_cache=False).logits[:, -1, :]
                torch.testing.assert_close(
                    decoder.draft_probs[:, step, :], torch.softmax(draft_logits.float() / 0.5, -1),
                    rtol=1e-4, atol=1e-6,
                )
            outputs = []
            handle = target.register_forward_hook(lambda module, args, result: outputs.append(result.logits))
            decoder.verify()
            handle.remove()
            torch.testing.assert_close(outputs[0], expected, rtol=1e-4, atol=1e-6)
            decoder.append()

    @torch.inference_mode()
    def test_stochastic_verification_preserves_target_distribution(self):
        target = tiny_qwen(vocab_size=3)
        p = torch.tensor([0.1, 0.2, 0.7])
        for q in (torch.tensor([0.6, 0.3, 0.1]), p):
            with self.subTest(q=q.tolist()):
                decoder = SpeculativeDecoder(target, target, torch.tensor([[1]]), 8, lookahead=2)
                decoder.prefill()
                decoder.draft_probs[:, :2, :].copy_(q)
                logits = p.log().view(1, 1, 3).expand(1, 3, 3)
                counts = torch.zeros(3)
                bonus_counts = torch.zeros(3)
                with patch.object(decoder.target, 'forward', return_value=logits):
                    for _ in range(4096):
                        decoder.block[:, 1:].copy_(sample_probabilities(q.expand(1, 2, 3)).squeeze(-1))
                        with patch.object(torch.Tensor, 'item', side_effect=AssertionError('Host scalar read')), \
                             patch.object(torch.Tensor, '__bool__', side_effect=AssertionError('Host branch')):
                            decoder.verify()
                        counts[decoder.commit_tokens[0, 0]] += 1
                        if torch.equal(q, p):
                            self.assertEqual(decoder.accepted.item(), 2)
                            bonus_counts[decoder.commit_tokens[0, 2]] += 1
                torch.testing.assert_close(counts / 4096, p, rtol=0, atol=0.025)
                if torch.equal(q, p):
                    torch.testing.assert_close(bonus_counts / 4096, p, rtol=0, atol=0.025)

    def test_invalid_requests(self):
        model = tiny_qwen()
        for prompt, length, temperature in (
            (torch.ones((2, 3), dtype=torch.long), 10, 0),
            (torch.empty((1, 0), dtype=torch.long), 10, 0),
            (torch.tensor([[31]]), 10, 0),
            (torch.tensor([[1.0]]), 10, 0),
            (torch.tensor([[1]]), -1, 0),
            (torch.tensor([[1]]), 10, -0.5),
            (torch.tensor([[1]]), 10, float('nan')),
        ):
            with self.subTest(prompt=prompt, length=length, temperature=temperature):
                with self.assertRaises(ValueError):
                    autoregressive_sampling(model, prompt, length, temperature)
                with self.assertRaises(ValueError):
                    speculative_sampling(model, model, prompt, length, None, temperature=temperature, debug=False)
        with self.assertRaises(ValueError):
            SpeculativeDecoder(model, model, torch.tensor([[1]]), 10, lookahead=0)
        with self.assertRaises(ValueError):
            speculative_sampling(model, model, torch.tensor([[1]]), 1, None, lookahead=0)
        with self.assertRaises(ValueError):
            SpeculativeDecoder(model, tiny_qwen(vocab_size=32), torch.tensor([[1]]), 10)


if __name__ == '__main__':
    unittest.main()
