import time
import random
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from autoregressive_sampling import autoregressive_sampling
from speculative_sampling import speculative_sampling

device = "cuda" if torch.cuda.is_available() else "cpu"

target_model = AutoModelForCausalLM.from_pretrained("facebook/opt-13b").to(device)

draft_model = AutoModelForCausalLM.from_pretrained("facebook/opt-1.3b").to(device)

tokenizer = AutoTokenizer.from_pretrained("facebook/opt-13b")

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

texts = prompts_sample_1

MAX_NEW_TOKENS = 64
TEMPERATURE = 0 # 0 for Deterministic

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
print()

print("Benchmarking naive Autoregressive Sampling...")
## Autoregressive
# Warmup
tokens = autoregressive_sampling(target_model, initial_prompt_seq=inputs_sample.input_ids, target_len=MAX_NEW_TOKENS+len(inputs_sample.input_ids), temperature=TEMPERATURE)

time_taken = 0
new_tokens = 0
for i in tqdm(range(len(texts))):
  text = texts[i]
  inputs = tokenizer(text, return_tensors="pt").to(device)
  start_len = len(inputs.input_ids)

  start_time = time.time_ns()
  tokens = autoregressive_sampling(target_model, initial_prompt_seq=inputs.input_ids, target_len=MAX_NEW_TOKENS+len(inputs.input_ids), temperature=TEMPERATURE)
  end_time = time.time_ns()

  new_tokens += len(tokens[0]) - start_len
  time_taken += (end_time - start_time) / 1_000_000_000

print(f"Latency (Autoregressive Sampling): {new_tokens/time_taken:.2f} tok/s")

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

print(f"Latency (Speculative Sampling): {new_tokens/time_taken:.2f} tok/s")