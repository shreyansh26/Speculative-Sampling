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
parser.add_argument('--target_model', default="facebook/opt-13b", help='Target model')
parser.add_argument('--draft_model', required=False, help='Draft model')
parser.add_argument('--temperature', default=0, type=float, help='Temperature')
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"

if args.method == "speculative":
    if args.draft_model is None:
        print("Draft model should be specified for Speculative Sampling")
        sys.exit(1)

    print("Using target model:", args.target_model)
    print("Using draft model:", args.draft_model)

    target_model = AutoModelForCausalLM.from_pretrained(args.target_model).to(device)
    draft_model = AutoModelForCausalLM.from_pretrained(args.draft_model).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.target_model)

    inputs = tokenizer(args.prompt, return_tensors="pt").to(device)

    start_time = time.time_ns()
    tokens = speculative_sampling(target_model, draft_model, initial_prompt_seq=inputs.input_ids, max_new_tokens=args.max_new_tokens, tokenizer=tokenizer, temperature=args.temperature, debug=False)
    end_time = time.time_ns()

    new_tokens = len(tokens[0]) - len(inputs.input_ids)
    time_taken = (end_time - start_time) / 1_000_000_000

    print(tokenizer.decode(tokens[0]))
    print()
    print(f"Latency (Speculative Sampling): {new_tokens/time_taken:.2f} tok/s")

elif args.method == "autoregressive":
    print("Using target model:", args.target_model)

    target_model = AutoModelForCausalLM.from_pretrained(args.target_model).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.target_model)

    inputs = tokenizer(args.prompt, return_tensors="pt").to(device)

    start_time = time.time_ns()
    tokens = autoregressive_sampling(target_model, initial_prompt_seq=inputs.input_ids, target_len=args.max_new_tokens+len(inputs.input_ids), temperature=args.temperature)
    end_time = time.time_ns()

    new_tokens = len(tokens[0]) - len(inputs.input_ids)
    time_taken = (end_time - start_time) / 1_000_000_000

    print(tokenizer.decode(tokens[0]))
    print()
    print(f"Latency (Naive Autoregressive Sampling): {new_tokens/time_taken:.2f} tok/s")

else:
    print("Method should be either autoregressive / speculative")