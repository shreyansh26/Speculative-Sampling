import time
import random
import argparse
from transformers import GPTNeoXForCausalLM, AutoTokenizer
from sampling import autoregressive_sampling
from speculative_sampling import speculative_sampling

parser = argparse.ArgumentParser(description='Speculative Sampling')
parser.add_argument('-t', '--time', help='Benchmark tokens/sec')
args = parser.parse_args()

model = GPTNeoXForCausalLM.from_pretrained(
  "EleutherAI/pythia-160m",
  cache_dir="./pythia-160m",
)

tokenizer = AutoTokenizer.from_pretrained(
  "EleutherAI/pythia-160m",
  cache_dir="./pythia-160m",
)

MAX_NEW_TOKENS = 64

texts = [
    'In which country is Hamburg?\n',
    'How are you doing today?\n',
    'It was a dark and stormy night.',
    'The sun rose slowly over the horizon, casting a warm glow on the world below.',
    'I never believed in ghosts until the day I met one.',
    'The sound of the train whistle echoed through the valley as I stood at the station, waiting.',
    'She walked into the room and everything changed.',
    'The smell of freshly baked bread filled the air as I entered the bakery.',
    'The first time I saw her, I knew she was trouble.'
    'The world was ending, and I was the only one who knew.',
    'It was the best of times, it was the worst of times.',
    'The forest was alive with the sound of animals as I walked deeper into the woods.',
    'As I looked out over the city, I knew that anything was possible.',
    'The sound of gunfire echoed through the streets as I ran for cover.',
    'The waves crashed against the shore, a never-ending cycle of destruction and creation.',
    'I woke up to find myself in a strange place, with no memory of how I got there.',
    'The clock struck midnight, and I knew that my life would never be the same.',]

inputs_sample = tokenizer(random.choice(texts), return_tensors="pt")
# tokens = model.generate(**inputs, max_new_tokens=20, do_sample=True)
# print(tokenizer.decode(tokens[0]))

tokens = autoregressive_sampling(model, initial_prompt_seq=inputs_sample.input_ids, target_len=MAX_NEW_TOKENS-len(inputs_sample.input_ids), temperature=0.001)
print(tokenizer.decode(tokens[0]))

tokens = speculative_sampling(model, model, initial_prompt_seq=inputs_sample.input_ids, max_new_tokens=MAX_NEW_TOKENS, temperature=0.001, debug=False)
print(tokenizer.decode(tokens[0]))


if args.time:
  ## Autoregressive
  # Warmup
  tokens = autoregressive_sampling(model, initial_prompt_seq=inputs_sample.input_ids, target_len=MAX_NEW_TOKENS-len(inputs_sample.input_ids), temperature=0.001)

  time_taken = 0
  new_tokens = 0
  for i in range(len(texts)):
    text = texts[i]
    inputs = tokenizer(text, return_tensors="pt")
    start_len = len(inputs.input_ids)

    start_time = time.time_ns()
    tokens = autoregressive_sampling(model, initial_prompt_seq=inputs.input_ids, target_len=MAX_NEW_TOKENS-len(inputs.input_ids), temperature=0.001)
    end_time = time.time_ns()

    new_tokens += tokens[0].shape[1] - start_len
    time_taken += (end_time - start_time) / 1_000_000

  print(f"Time per token: {new_tokens/time_taken:02f}ms")

  ## Speculative Sampling
  # Warmup
  tokens = speculative_sampling(model, model, initial_prompt_seq=inputs_sample.input_ids, max_new_tokens=MAX_NEW_TOKENS, temperature=0.001, debug=False)

  time_taken = 0
  new_tokens = 0
  for i in range(len(texts)):
    text = texts[i]
    inputs = tokenizer(text, return_tensors="pt")
    start_len = len(inputs.input_ids)

    start_time = time.time_ns()
    tokens = speculative_sampling(model, model, initial_prompt_seq=inputs.input_ids, max_new_tokens=MAX_NEW_TOKENS, temperature=0.001, debug=False)
    end_time = time.time_ns()

    new_tokens += tokens[0].shape[1] - start_len
    time_taken += (end_time - start_time) / 1_000_000

  print(f"Time per token: {new_tokens/time_taken:02f}ms")