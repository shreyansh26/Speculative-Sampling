import sys
import time
import random
import argparse
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from autoregressive_sampling import autoregressive_sampling
from speculative_sampling import speculative_sampling

parser = argparse.ArgumentParser(description='Speculative Sampling')
parser.add_argument('--method', default="speculative", help='Sampling Method (autogressive / speculative)')
parser.add_argument('--prompt', required=True, help='Input prompt')
parser.add_argument('--max_new_tokens', type=int, required=True, help='No. of max new tokens')
parser.add_argument('--target_model', default="Qwen/Qwen3-14B", help='Target Qwen3 model')
parser.add_argument('--draft_model', required=False, help='Draft Qwen3 model')
parser.add_argument('--temperature', default=0, type=float, help='Temperature')
parser.add_argument('--cuda-graphs', action='store_true', help='Capture decode stages for either method; prefill stays eager')
args = parser.parse_args()
if args.max_new_tokens < 1:
    parser.error('--max_new_tokens must be positive')

if not torch.cuda.is_available():
    parser.error('FlashAttention requires a supported CUDA GPU')
device = "cuda"

if args.method == "speculative":
    if args.draft_model is None:
        print("Draft model should be specified for Speculative Sampling")
        sys.exit(1)

    print("Using target model:", args.target_model)
    print("Using draft model:", args.draft_model)

    target_model = AutoModelForCausalLM.from_pretrained(args.target_model, dtype="auto", device_map={"": device})
    draft_model = AutoModelForCausalLM.from_pretrained(args.draft_model, dtype="auto", device_map={"": device})
    tokenizer = AutoTokenizer.from_pretrained(args.target_model)
    draft_tokenizer = AutoTokenizer.from_pretrained(args.draft_model)
    if tokenizer.get_vocab() != draft_tokenizer.get_vocab():
        raise ValueError('Target and draft tokenizers must map tokens to the same IDs')

    inputs = tokenizer(args.prompt, return_tensors="pt").to(device)
    prompt_len = inputs.input_ids.shape[-1]

    if device == 'cuda':
        torch.cuda.synchronize()
    start_time = time.time_ns()
    tokens = speculative_sampling(target_model, draft_model, initial_prompt_seq=inputs.input_ids, target_len=prompt_len+args.max_new_tokens, tokenizer=tokenizer, temperature=args.temperature, debug=False, cuda_graphs=args.cuda_graphs)
    if device == 'cuda':
        torch.cuda.synchronize()
    end_time = time.time_ns()

    new_tokens = tokens.shape[-1] - prompt_len
    time_taken = (end_time - start_time) / 1_000_000_000

    print(tokenizer.decode(tokens[0]))
    print()
    print("New tokens:", new_tokens)
    print(f"Latency (Speculative Sampling): {new_tokens/time_taken:.2f} tok/s")

elif args.method == "autoregressive":
    print("Using target model:", args.target_model)

    target_model = AutoModelForCausalLM.from_pretrained(args.target_model, dtype="auto", device_map={"": device})
    tokenizer = AutoTokenizer.from_pretrained(args.target_model)

    inputs = tokenizer(args.prompt, return_tensors="pt").to(device)
    prompt_len = inputs.input_ids.shape[-1]

    if device == 'cuda':
        torch.cuda.synchronize()
    start_time = time.time_ns()
    tokens = autoregressive_sampling(target_model, initial_prompt_seq=inputs.input_ids, target_len=prompt_len+args.max_new_tokens, temperature=args.temperature, cuda_graphs=args.cuda_graphs)
    if device == 'cuda':
        torch.cuda.synchronize()
    end_time = time.time_ns()

    new_tokens = tokens.shape[-1] - prompt_len
    time_taken = (end_time - start_time) / 1_000_000_000

    print(tokenizer.decode(tokens[0]))
    print()
    print("New tokens:", new_tokens)
    print(f"Latency (Cached Autoregressive Sampling): {new_tokens/time_taken:.2f} tok/s")

else:
    print("Method should be either autoregressive / speculative")
