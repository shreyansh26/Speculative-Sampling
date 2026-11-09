"""Benchmark cached Qwen3 with an FP8 target and BF16/FP8 drafts; write CSV."""

import argparse
from contextlib import ExitStack
import csv
from pathlib import Path
import time
from unittest.mock import patch
import torch
import transformers
from qwen3_cache import FLASH_ATTENTION_REVISION, kv_splits_for_prompt
from transformers import AutoTokenizer, AutoModelForCausalLM
from autoregressive_sampling import AutoregressiveDecoder
from speculative_sampling import SpeculativeDecoder

prompts_sample_1 = [
    'What did Rutherford discover?\n',
    'The key to the mysterious chest had been missing for generations, until today.',
    'When the rain started falling upwards, Lily knew something was terribly wrong.',
    'A single photograph discovered in an old album unveiled a family secret that had been buried for decades.',
    'The old lighthouse had been abandoned for years, but its beam of light suddenly flickered to life one stormy night.',
    'As the last leaf fell from the ancient tree, a long-forgotten prophecy began to unfold.',
    'In a world of constant silence, a deaf musician discovered a hidden language in the patterns of the stars.',
    'The message written in a bottle that washed ashore carried a plea for help from a distant, unknown island.',
    "When the town's clock tower chimed 13 times, the residents realized they were trapped in a time loop.",
    "The antique mirror reflected a room that didn't exist, and it beckoned Sarah to step through.",
    "In a city where emotions could be bought and sold, Ella's heart was the only one immune to the trade.",
    'These shorter beginnings should still provide a great foundation for your storytelling prompts.'
  ]

prompts_sample_2 = [
    'What did Rutherford discover?\n',
    "Emily found a mysterious letter on her doorstep one sunny morning.",
    "On a rainy afternoon, Max stumbled upon an old treasure map in the attic.",
    "A friendly stray cat showed up at Lisa's doorstep, leading her to a hidden garden.",
    "Jake's new neighbor had a strange habit of disappearing into the woods every night.",
    "While cleaning out the garage, Mia discovered a box of her grandfather's old inventions.",
    "At the county fair, Tom won a goldfish that seemed to have an uncanny ability.",
    "Amelia woke up one day to find her bedroom ceiling covered in glowing stars.",
    "In a dusty antique shop, Sarah found a vintage camera with peculiar abilities.",
    "During a family camping trip, they stumbled upon an unusual rock formation.",
    "A peculiar antique shop opened in town, and its owner seemed to know everyone's deepest secrets."
  ]


def benchmark_inputs(tokenizer, prompt_tokens=None):
    if prompt_tokens is None:
        prompts = [prompts_sample_1[i] for i in (0, 1, 3, 10)] + [prompts_sample_2[i] for i in (2, 8)]
        return [tokenizer(prompt, return_tensors='pt').input_ids for prompt in prompts]
    corpus = tokenizer('\n'.join(prompts_sample_1 + prompts_sample_2),
                       return_tensors='pt', add_special_tokens=False).input_ids
    copies = (prompt_tokens + corpus.shape[1] - 1) // corpus.shape[1]
    return [corpus.repeat(1, copies)[:, :prompt_tokens].contiguous()]


@torch.inference_mode()
def check_graph_reuse(decoder, inputs, new_tokens):
    """Check replay within split buckets and reuse when returning to earlier buckets."""
    short = inputs[0][:, :1]
    decoder.generate(short, target_len=3)
    expected_stages = {'decode'} if isinstance(decoder, AutoregressiveDecoder) else {'draft', 'verify', 'append'}
    assert set(decoder.graphs) == expected_stages
    variants = dict(decoder.graph_cache)
    runners = ((decoder.cache,) if isinstance(decoder, AutoregressiveDecoder)
               else (decoder.target, decoder.proposer))

    def storage_addresses():
        tensors = [value for value in vars(decoder).values() if isinstance(value, torch.Tensor)]
        for runner in runners:
            tensors.append(runner.valid_length)
            for layer in runner.cache.layers:
                tensors.extend((layer.keys, layer.values, layer.cumulative_length))
        return [tensor.data_ptr() for tensor in tensors]

    addresses = storage_addresses()
    for index, ids in enumerate([*inputs, short]):
        splits = kv_splits_for_prompt(ids.shape[1])
        if splits not in variants:
            capture_length = {1: 1, 4: 512, 16: 4096}[splits]
            decoder.generate(ids[:, :capture_length], target_len=capture_length + 2)
            variants[splits] = decoder.graph_cache[splits]
        graphs = variants[splits]
        assert set(graphs) == expected_stages
        target_len = ids.shape[1] + new_tokens
        torch.manual_seed(index)
        decoder.cuda_graphs, decoder.graphs = False, {}
        try:
            expected = decoder.generate(ids, target_len=target_len)
            rng = torch.cuda.get_rng_state(ids.device)
        finally:
            decoder.cuda_graphs, decoder.graphs = True, dict(graphs)
        calls = []
        with ExitStack() as stack:
            for model in {runner.model for runner in runners}:
                handle = model.register_forward_pre_hook(
                    lambda module, args: calls.append((id(module), args[0].shape[1],
                                                      torch.cuda.is_current_stream_capturing()))
                )
                stack.callback(handle.remove)
            torch.manual_seed(index)
            actual = decoder.generate(ids, target_len=target_len)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(torch.cuda.get_rng_state(ids.device), rng, rtol=0, atol=0)
        assert decoder.graphs == graphs and storage_addresses() == addresses
        assert decoder.graph_cache == variants
        assert sorted(calls) == sorted((id(runner.model), ids.shape[1], False) for runner in runners)
        print(f'    {type(decoder).__name__}: {ids.shape[1]} prompt tokens, {splits} KV splits, '
              'eager/graph tokens and RNG match; stable buffers; decode replay only', flush=True)


def timed_sample(sampler, input_ids, target_len, temperature):
    if input_ids.is_cuda:
        torch.cuda.synchronize(input_ids.device)
    start = time.perf_counter()
    tokens = sampler(initial_prompt_seq=input_ids, target_len=target_len, temperature=temperature)
    if input_ids.is_cuda:
        torch.cuda.synchronize(input_ids.device)
    elapsed = time.perf_counter() - start
    if tokens.shape[-1] != target_len:
        raise RuntimeError(f"Sampler returned {tokens.shape[-1]} tokens, expected {target_len}")
    return tokens, elapsed


def profile_stages(sampler, inputs, new_tokens, temperature):
    stages = ('prefill', 'draft', 'verify', 'append')
    milliseconds = dict.fromkeys(stages, 0.0)
    counts = dict.fromkeys(stages, 0)
    wall_seconds = 0.0
    accepted_total = 0
    for index, ids in enumerate(inputs):
        print(f'    profile prompt {index + 1}/{len(inputs)}', flush=True)
        accepted_lengths = torch.empty(new_tokens - 1, dtype=torch.long, device=ids.device)
        # Allocate and initialize event handles before the measured request.
        events = {
            name: [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                   for _ in range(1 if name == 'prefill' else new_tokens - 1)]
            for name in stages
        }
        for pairs in events.values():
            for start, end in pairs:
                start.record()
                end.record()
        torch.cuda.synchronize()
        used = dict.fromkeys(stages, 0)

        def instrument(name, method):
            def measured(decoder):
                if decoder.capturing:
                    return method(decoder)
                start, end = events[name][used[name]]
                used[name] += 1
                start.record()
                result = method(decoder)
                end.record()
                if name == 'verify':
                    accepted_lengths[used[name] - 1].copy_(decoder.accepted)
                return result
            return measured

        torch.manual_seed(index)
        with ExitStack() as stack:
            for name in stages:
                stack.enter_context(patch.object(
                    SpeculativeDecoder, name, instrument(name, getattr(SpeculativeDecoder, name))
                ))
            _, seconds = timed_sample(sampler, ids, ids.shape[1] + new_tokens, temperature)
        wall_seconds += seconds
        assert used['prefill'] == 1
        assert used['draft'] == used['verify'] == used['append']
        accepted_total += accepted_lengths[:used['verify']].sum().item()
        for name in stages:
            counts[name] += used[name]
            milliseconds[name] += sum(start.elapsed_time(end) for start, end in events[name][:used[name]])
    return {
        'wall_seconds': wall_seconds,
        'calls': counts,
        'total_ms': milliseconds,
        'mean_ms_per_call': {name: milliseconds[name] / counts[name] if counts[name] else 0.0
                             for name in stages},
        'outside_stages_ms': wall_seconds * 1000 - sum(milliseconds.values()),
        'committed_tokens_per_round': ((new_tokens - 1) * len(inputs) / counts['append']
                                       if counts['append'] else 0.0),
        'accepted_draft_tokens_per_round': accepted_total / counts['verify'] if counts['verify'] else 0.0,
    }


def benchmark_experiment(target, draft, inputs, temperature, new_tokens, repeats, lookahead=3,
                         cuda_graphs=False, check_graphs=False):
    max_seq_len = max(ids.shape[1] for ids in inputs) + new_tokens
    first_target_len = inputs[0].shape[1] + new_tokens
    ar_decoder = AutoregressiveDecoder(
        target, inputs[0], first_target_len, temperature, cuda_graphs, max_seq_len=max_seq_len,
    )
    decoder = SpeculativeDecoder(
        target, draft, inputs[0], first_target_len, lookahead, temperature, cuda_graphs,
        max_seq_len=max_seq_len,
    )

    def sample_ar(initial_prompt_seq, target_len, temperature):
        return ar_decoder.generate(initial_prompt_seq, target_len=target_len)

    def sample_spec(initial_prompt_seq, target_len, temperature):
        return decoder.generate(initial_prompt_seq, target_len=target_len)

    samplers = {
        'ar': sample_ar,
        'spec': sample_spec,
    }
    if check_graphs:
        if not cuda_graphs or new_tokens < 2:
            raise ValueError('Graph validation requires CUDA graphs and at least two new tokens')
        print('  Graph validation (excluded from timings)', flush=True)
        for reusable in (ar_decoder, decoder):
            check_graph_reuse(reusable, inputs, new_tokens)
    torch.cuda.reset_peak_memory_stats()
    print(f'  Warmup: shared buffers for all prompt lengths, max_seq_len={max_seq_len}', flush=True)
    for index, ids in enumerate(inputs):
        for sampler in samplers.values():
            torch.manual_seed(index)
            timed_sample(sampler, ids, ids.shape[1] + new_tokens, temperature)
        print(f'    warmup prompt {index + 1}/{len(inputs)}', flush=True)

    print('  End-to-end timing (no profiling hooks)', flush=True)
    seconds = dict.fromkeys(samplers, 0.0)
    greedy_matches = 0
    for repeat in range(repeats):
        for index, ids in enumerate(inputs):
            outputs = {}
            order = ('ar', 'spec') if (repeat + index) % 2 == 0 else ('spec', 'ar')
            for name in order:
                torch.manual_seed(repeat * len(inputs) + index)
                outputs[name], elapsed = timed_sample(
                    samplers[name], ids, ids.shape[1] + new_tokens, temperature
                )
                seconds[name] += elapsed
            if temperature == 0:
                greedy_matches += int(torch.equal(outputs['ar'], outputs['spec']))
            print(f'    repeat {repeat + 1}/{repeats}, prompt {index + 1}/{len(inputs)}', flush=True)

    print('  Separate stage/acceptance profiling pass (first repetition seeds)', flush=True)
    profile = profile_stages(samplers['spec'], inputs, new_tokens, temperature)
    row = summarize(seconds, len(inputs), new_tokens, repeats, profile, greedy_matches, temperature)
    row['attention_backend'] = 'flash_attention_3_kvcache'
    row['prefill_backend'] = 'flash_attention_3'
    row['attention_kernel_revision'] = FLASH_ATTENTION_REVISION
    row['max_seq_len'] = max_seq_len
    row['spec_graph_setup_s'] = decoder.capture_seconds
    row['ar_graph_setup_s'] = ar_decoder.capture_seconds
    row['spec_graph_count'] = sum(map(len, decoder.graph_cache.values()))
    row['ar_graph_count'] = sum(map(len, ar_decoder.graph_cache.values()))
    row['kv_splits'] = ','.join(str(splits) for splits in sorted({kv_splits_for_prompt(ids.shape[1]) for ids in inputs}))
    row['graph_reuse_checked'] = check_graphs
    row['peak_allocated_gib'] = torch.cuda.max_memory_allocated() / 2**30
    return row


def summarize(seconds, prompt_count, new_tokens, repeats, profile, greedy_matches, temperature):
    requests = prompt_count * repeats
    row = {
        'ar_total_e2e_s': seconds['ar'], 'spec_total_e2e_s': seconds['spec'],
        'ar_mean_e2e_ms': 1000 * seconds['ar'] / requests,
        'spec_mean_e2e_ms': 1000 * seconds['spec'] / requests,
        'ar_tokens_per_s': requests * new_tokens / seconds['ar'],
        'spec_tokens_per_s': requests * new_tokens / seconds['spec'],
        'speedup': seconds['ar'] / seconds['spec'],
        'greedy_exact_matches': greedy_matches if temperature == 0 else '',
        'profile_requests': prompt_count,
        'profile_rounds': profile['calls']['verify'],
        'profile_total_e2e_s': profile['wall_seconds'],
        'profile_avg_accepted_draft_tokens': profile['accepted_draft_tokens_per_round'],
        'profile_avg_committed_tokens': profile['committed_tokens_per_round'],
    }
    for stage in ('prefill', 'draft', 'verify', 'append'):
        unit = 'request' if stage == 'prefill' else 'round'
        row[f'profile_{stage}_total_ms'] = profile['total_ms'][stage]
        row[f'profile_{stage}_ms_per_{unit}'] = profile['mean_ms_per_call'][stage]
    row['profile_outside_stages_ms'] = profile['outside_stages_ms']
    return row


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target-model', default='Qwen/Qwen3-14B-FP8')
    parser.add_argument('--bf16-draft-model', default='Qwen/Qwen3-1.7B')
    parser.add_argument('--fp8-draft-model', default='Qwen/Qwen3-1.7B-FP8')
    parser.add_argument('--max-new-tokens', type=int, default=64)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--lookahead', type=int, default=3)
    parser.add_argument('--prompt-lengths', type=int, nargs='+',
                        help='Benchmark one synthetic repeated-text prompt per specified token length')
    parser.add_argument('--check-graphs', action='store_true',
                        help='Check graph reuse within split buckets against eager output/RNG before timing')
    parser.add_argument('--cuda-graphs', action='store_true',
                        help='Capture AR decode and speculative draft/verify/append; prefill stays eager (default: all eager)')
    parser.add_argument('--output', type=Path, default=Path('outputs/benchmark.csv'))
    args = parser.parse_args()
    if args.max_new_tokens < 1 or args.repeats < 1 or args.lookahead < 1:
        parser.error('Token count, repeats, and lookahead must be positive')
    if args.prompt_lengths is not None and any(length < 1 for length in args.prompt_lengths):
        parser.error('Prompt lengths must be positive')
    if args.check_graphs and (not args.cuda_graphs or args.max_new_tokens < 2):
        parser.error('--check-graphs requires --cuda-graphs and at least two new tokens')
    return args


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for stage timing')
    print(f'Loading FP8 target: {args.target_model}', flush=True)
    target = AutoModelForCausalLM.from_pretrained(
        args.target_model, dtype='auto', device_map={'': 'cuda'}
    )
    if target.model.layers[0].self_attn.q_proj.weight.dtype != torch.float8_e4m3fn:
        raise ValueError('The target must use FP8 weights')
    tokenizer = AutoTokenizer.from_pretrained(args.target_model)
    prompt_lengths = args.prompt_lengths or [None]
    metadata = {
        'target_model': args.target_model, 'target_dtype': 'fp8',
        'gpu': torch.cuda.get_device_name(), 'torch_version': torch.__version__,
        'transformers_version': transformers.__version__, 'batch_size': 1, 'lookahead': args.lookahead,
        'new_tokens_per_request': args.max_new_tokens,
        'repeats': args.repeats,
        'graph_mode': args.cuda_graphs, 'spec_decoder_reused': True,
        'ar_decoder_reused': True,
    }
    print(f"{metadata['gpu']} | prompt lengths {prompt_lengths} | {args.max_new_tokens} new tokens | "
          f'{args.repeats} repetitions | lookahead {args.lookahead} | '
          f'CUDA graphs {args.cuda_graphs}', flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', newline='') as output:
        writer = None
        experiment = 0
        for label, model_name, expected_dtype in (
            ('bf16', args.bf16_draft_model, torch.bfloat16),
            ('fp8', args.fp8_draft_model, torch.float8_e4m3fn),
        ):
            print(f'Loading {label.upper()} draft: {model_name}', flush=True)
            draft_tokenizer = AutoTokenizer.from_pretrained(model_name)
            if tokenizer.get_vocab() != draft_tokenizer.get_vocab():
                raise ValueError('Target and draft tokenizers must map tokens to the same IDs')
            draft = AutoModelForCausalLM.from_pretrained(
                model_name, dtype='auto', device_map={'': 'cuda'}
            )
            if draft.model.layers[0].self_attn.q_proj.weight.dtype != expected_dtype:
                raise ValueError(f'The {label} draft has an unexpected weight dtype')
            for temperature in (0, 0.5):
                for prompt_tokens in prompt_lengths:
                    inputs = [ids.to('cuda') for ids in benchmark_inputs(tokenizer, prompt_tokens)]
                    experiment += 1
                    print(f'\nExperiment {experiment}/{4 * len(prompt_lengths)}: FP8 target + {label.upper()} draft, '
                          f'temperature={temperature}, lookahead={args.lookahead}, '
                          f'prompt_tokens={prompt_tokens}, CUDA graphs={args.cuda_graphs}', flush=True)
                    row = {**metadata, 'draft_model': model_name, 'draft_dtype': label,
                           'temperature': temperature, 'prompt_count': len(inputs),
                           'prompt_tokens': prompt_tokens if prompt_tokens is not None else '',
                           'prompt_set': 'synthetic_repeated_prompts' if prompt_tokens is not None else 'short_prompts'}
                    row.update(benchmark_experiment(
                        target, draft, inputs, temperature, args.max_new_tokens, args.repeats,
                        args.lookahead, args.cuda_graphs, args.check_graphs,
                    ))
                    if writer is None:
                        writer = csv.DictWriter(output, fieldnames=list(row), lineterminator='\n')
                        writer.writeheader()
                    writer.writerow(row)
                    output.flush()
                    print(f"  AR {row['ar_tokens_per_s']:.2f} tok/s | speculative "
                          f"{row['spec_tokens_per_s']:.2f} tok/s | speedup {row['speedup']:.3f}x | "
                          f"accepted draft length {row['profile_avg_accepted_draft_tokens']:.3f}", flush=True)
                    print(f'  Saved CSV row to {args.output}', flush=True)
            del draft
            torch.cuda.empty_cache()
    print(f'\nCompleted all {experiment} experiments: {args.output}', flush=True)


if __name__ == '__main__':
    main()
