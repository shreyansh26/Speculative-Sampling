import time
import random
import argparse
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, GPTNeoXForCausalLM
from sampling import autoregressive_sampling
from speculative_sampling import speculative_sampling

device = "cuda" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser(description='Speculative Sampling')
parser.add_argument('-b', '--benchmark', action='store_true', help='Benchmark tokens/sec')
args = parser.parse_args()

target_model = AutoModelForCausalLM.from_pretrained(
  "facebook/opt-13b",
  cache_dir="./opt-13b",
).to(device)

draft_model = AutoModelForCausalLM.from_pretrained(
  "facebook/opt-1.3b",
  cache_dir="./opt-1.3b",
).to(device)

tokenizer = AutoTokenizer.from_pretrained(
  "facebook/opt-13b",
  cache_dir="./opt-13b",
)

# target_model = GPTNeoXForCausalLM.from_pretrained(
#   "EleutherAI/pythia-410m",
#   cache_dir="./pythia-410m",
# ).to(device)

# draft_model = GPTNeoXForCausalLM.from_pretrained(
#   "EleutherAI/pythia-160m",
#   cache_dir="./pythia-160m",
# ).to(device)

# tokenizer = AutoTokenizer.from_pretrained(
#   "EleutherAI/pythia-410m",
#   cache_dir="./pythia-410m",
# )


texts = [
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

MAX_NEW_TOKENS = 64
TEMPERATURE = 0 # Deterministic

print("Target Model -", target_model.config._name_or_path)
print("Draft Model -", draft_model.config._name_or_path)
print("************\n")

inputs_sample = tokenizer(random.choice(texts), return_tensors="pt").to(device)
tokens = target_model.generate(**inputs_sample, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
print("HF's generate")
print("Count of new tokens:", len(tokens[0]) - len(inputs_sample.input_ids))
print(tokenizer.decode(tokens[0]))
print("******")

tokens = autoregressive_sampling(target_model, initial_prompt_seq=inputs_sample.input_ids, target_len=MAX_NEW_TOKENS+len(inputs_sample.input_ids), temperature=TEMPERATURE)
print("Naive Autoregressive with temperature")
print("Count of new tokens:", len(tokens[0]) - len(inputs_sample.input_ids))
print(tokenizer.decode(tokens[0]))
print("******")

tokens = speculative_sampling(target_model, draft_model, initial_prompt_seq=inputs_sample.input_ids, max_new_tokens=MAX_NEW_TOKENS, tokenizer=tokenizer, temperature=TEMPERATURE, debug=False)
print("Speculative Sampling with temperature")
print("Count of new tokens:", len(tokens[0]) - len(inputs_sample.input_ids))
print(tokenizer.decode(tokens[0]))
print("******")

if args.benchmark:
  print("Benchmarking naive Autoregressive Sampling...")
  ## Autoregressive
  # Warmup
  tokens = autoregressive_sampling(target_model, initial_prompt_seq=inputs_sample.input_ids, target_len=MAX_NEW_TOKENS-len(inputs_sample.input_ids), temperature=TEMPERATURE)

  time_taken = 0
  new_tokens = 0
  for i in tqdm(range(len(texts))):
    text = texts[i]
    inputs = tokenizer(text, return_tensors="pt").to(device)
    start_len = len(inputs.input_ids)

    start_time = time.time_ns()
    tokens = autoregressive_sampling(target_model, initial_prompt_seq=inputs.input_ids, target_len=MAX_NEW_TOKENS-len(inputs.input_ids), temperature=TEMPERATURE)
    end_time = time.time_ns()

    new_tokens += len(tokens[0]) - start_len
    time_taken += (end_time - start_time) / 1_000_000_000

  print(f"Time per token (Autoregressive Sampling): {new_tokens/time_taken:.2f} tok/s")

  ## Speculative Sampling
  # Warmup
  print("Benchmarking Speculative Sampling...")
  tokens = speculative_sampling(target_model, draft_model, initial_prompt_seq=inputs_sample.input_ids, max_new_tokens=MAX_NEW_TOKENS, tokenizer=tokenizer, temperature=TEMPERATURE, debug=False)

  time_taken = 0
  new_tokens = 0
  for i in tqdm(range(len(texts))):
    text = texts[i]
    inputs = tokenizer(text, return_tensors="pt").to(device)
    start_len = len(inputs.input_ids)

    start_time = time.time_ns()
    tokens = speculative_sampling(target_model, draft_model, initial_prompt_seq=inputs.input_ids, max_new_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE, tokenizer=tokenizer, debug=False)
    end_time = time.time_ns()

    new_tokens += len(tokens[0]) - start_len
    time_taken += (end_time - start_time) / 1_000_000_000

  print(f"Time per token (Speculative Sampling): {new_tokens/time_taken:.2f} tok/s")