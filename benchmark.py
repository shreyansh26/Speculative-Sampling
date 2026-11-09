"""Benchmark cached Qwen3 with an FP8 target and BF16/FP8 drafts; write CSV."""

import argparse
from contextlib import ExitStack
import csv
from functools import partial
from pathlib import Path
import time
from unittest.mock import patch
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from autoregressive_sampling import autoregressive_sampling
from speculative_sampling import SpeculativeDecoder, speculative_sampling

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


def benchmark_experiment(target, draft, inputs, temperature, new_tokens, repeats, lookahead=3):
    samplers = {
        'ar': partial(autoregressive_sampling, target),
        'spec': partial(speculative_sampling, target, draft, tokenizer=None, debug=False, lookahead=lookahead),
    }
    print('  Warmup', flush=True)
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
    return summarize(seconds, len(inputs), new_tokens, repeats, profile, greedy_matches, temperature)


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
    parser.add_argument('--output', type=Path, default=Path('outputs/benchmark.csv'))
    args = parser.parse_args()
    if args.max_new_tokens < 1 or args.repeats < 1 or args.lookahead < 1:
        parser.error('Token count, repeats, and lookahead must be positive')
    return args


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for stage timing')
    print(f'Loading FP8 target: {args.target_model}', flush=True)
    target = AutoModelForCausalLM.from_pretrained(
        args.target_model, dtype='auto', device_map={'': 'cuda'}, attn_implementation='sdpa'
    )
    if target.model.layers[0].self_attn.q_proj.weight.dtype != torch.float8_e4m3fn:
        raise ValueError('The target must use FP8 weights')
    tokenizer = AutoTokenizer.from_pretrained(args.target_model)
    prompts = [prompts_sample_1[i] for i in (0, 1, 3, 10)] + [prompts_sample_2[i] for i in (2, 8)]
    inputs = [tokenizer(prompt, return_tensors='pt').input_ids.to('cuda') for prompt in prompts]
    metadata = {
        'target_model': args.target_model, 'target_dtype': 'fp8',
        'gpu': torch.cuda.get_device_name(), 'torch_version': torch.__version__,
        'transformers_version': transformers.__version__, 'batch_size': 1, 'lookahead': args.lookahead,
        'prompt_count': len(inputs), 'new_tokens_per_request': args.max_new_tokens,
        'repeats': args.repeats,
    }
    print(f"{metadata['gpu']} | {len(inputs)} prompts | {args.max_new_tokens} new tokens | "
          f'{args.repeats} repetitions | lookahead {args.lookahead}', flush=True)
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
                model_name, dtype='auto', device_map={'': 'cuda'}, attn_implementation='sdpa'
            )
            if draft.model.layers[0].self_attn.q_proj.weight.dtype != expected_dtype:
                raise ValueError(f'The {label} draft has an unexpected weight dtype')
            for temperature in (0, 0.5):
                experiment += 1
                print(f'\nExperiment {experiment}/4: FP8 target + {label.upper()} draft, '
                      f'temperature={temperature}, lookahead={args.lookahead}', flush=True)
                row = {**metadata, 'draft_model': model_name, 'draft_dtype': label,
                       'temperature': temperature}
                row.update(benchmark_experiment(
                    target, draft, inputs, temperature, args.max_new_tokens, args.repeats, args.lookahead
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
    print(f'\nCompleted all four experiments: {args.output}', flush=True)


if __name__ == '__main__':
    main()
