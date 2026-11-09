import argparse
from functools import partial
import time
import torch
from tqdm import tqdm
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from autoregressive_sampling import autoregressive_sampling
from speculative_sampling import speculative_sampling

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


def main():
    parser = argparse.ArgumentParser(description="Benchmark both README prompt sets at temperatures 0 and 0.5.")
    parser.add_argument("--target-model", default="facebook/opt-13b")
    parser.add_argument("--draft-model", default="facebook/opt-1.3b")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    args = parser.parse_args()
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be positive")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.target_model)
    draft_tokenizer = AutoTokenizer.from_pretrained(args.draft_model)
    if tokenizer.get_vocab() != draft_tokenizer.get_vocab():
        raise ValueError("Target and draft tokenizers must map tokens to the same IDs")
    target_model = AutoModelForCausalLM.from_pretrained(args.target_model, dtype="auto", device_map={"": device})
    draft_model = AutoModelForCausalLM.from_pretrained(args.draft_model, dtype="auto", device_map={"": device})
    for label, name, model in (("Target", args.target_model, target_model), ("Draft", args.draft_model, draft_model)):
        print(f"{label}: {name}")
        print(f"{label} weight dtypes: {', '.join(sorted({str(p.dtype) for p in model.parameters()}))}")
    print(f"Device: {torch.cuda.get_device_name() if device == 'cuda' else device}")
    print(f"PyTorch: {torch.__version__} | Transformers: {transformers.__version__}")
    print(f"New tokens per prompt: {args.max_new_tokens}")
    print("Lookahead: 4 | Batch size: 1 | KV cache: off | Seed: 0")

    baseline = partial(autoregressive_sampling, target_model)
    speculative = partial(speculative_sampling, target_model, draft_model, tokenizer=tokenizer, debug=False)
    for temperature in (0, 0.5):
        warmup = tokenizer(prompts_sample_1[0], return_tensors="pt").input_ids.to(device)
        for sampler in (baseline, speculative):
            timed_sample(sampler, warmup, warmup.shape[-1] + args.max_new_tokens, temperature)
        for set_number, prompts in enumerate((prompts_sample_1, prompts_sample_2), start=1):
            torch.manual_seed(0)
            baseline_seconds = speculative_seconds = 0.0
            greedy_matches = 0
            for prompt in tqdm(prompts, desc=f"Set {set_number}, temperature {temperature}"):
                input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
                target_len = input_ids.shape[-1] + args.max_new_tokens
                baseline_tokens, elapsed = timed_sample(baseline, input_ids, target_len, temperature)
                baseline_seconds += elapsed
                speculative_tokens, elapsed = timed_sample(speculative, input_ids, target_len, temperature)
                speculative_seconds += elapsed
                if temperature == 0:
                    greedy_matches += torch.equal(baseline_tokens, speculative_tokens)

            token_count = len(prompts) * args.max_new_tokens
            print(f"Set {set_number}, temperature {temperature}, {token_count} tokens per method")
            print(f"Autoregressive: {token_count / baseline_seconds:.2f} tok/s ({baseline_seconds:.3f} s)")
            print(f"Speculative:    {token_count / speculative_seconds:.2f} tok/s ({speculative_seconds:.3f} s)")
            print(f"Speedup: {baseline_seconds / speculative_seconds:.3f}x")
            if temperature == 0:
                print(f"Greedy exact matches: {greedy_matches}/{len(prompts)}")


if __name__ == "__main__":
    main()
