# Speculative Sampling

An implementation of speculative sampling as described in the paper [Accelerating Large Language Model Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318) by Deepmind.

## Setup

```bash
uv sync --locked
```

The project uses Python 3.12 by default. On Linux x86-64, uv selects the
CUDA 12.6 PyTorch build for compatibility with the H100 host's NVIDIA 570 driver.

## To run

```bash
uv run generate.py -h
```

Run the above command to see the parameter options and their description.

**Speculative Sampling**
```bash
uv run generate.py  --method speculative \
                    --prompt "Emily found a mysterious letter on her doorstep one sunny morning." \
                    --max_new_tokens 64 \
                    --target_model facebook/opt-13b \
                    --draft_model facebook/opt-1.3b \
                    --temperature 0.5
```

**Naive Autoregressive Sampling**
```bash
uv run generate.py  --method autoregressive \
                    --prompt "Emily found a mysterious letter on her doorstep one sunny morning." \
                    --max_new_tokens 64 \
                    --target_model facebook/opt-13b \
                    --temperature 0.5
```

**Benchmarking**
```bash
uv run benchmark.py
```

To run the same workload with the official Qwen3 FP8 checkpoints on an H100:

```bash
TRANSFORMERS_DISABLE_DEEPGEMM_LINEAR=1 uv run --locked --extra fp8 benchmark.py \
  --target-model Qwen/Qwen3-32B-FP8 --draft-model Qwen/Qwen3-1.7B-FP8
```

The benchmark runs both prompt sets at temperatures 0 and 0.5. Each method
generates exactly 64 tokens per prompt with batch size 1 and full-prefix forwards.

## Results

Results showing the speedup (as ratio) of speculative sampling over naive autoregressive sampling. These results are from different benchmarking runs and the logs can be found in the [outputs](outputs/) directory.

**Target Model - `facebook/opt-13b`**  
**Draft Model - `facebook/opt-1.3b`**

| Config            | Speedup (Set 1) | Speedup (Set 2) | Average Speedup |
|-------------------|-----------------|-----------------|-----------------|
| Temperature = 0   | 1.83            | 1.73            | 1.78            |
| Temperature = 0.5 | 1.68            | 1.81            | 1.75            |

**Target Model - `facebook/opt-6.7b`**  
**Draft Model - `facebook/opt-1.3b`**

| Config            | Speedup (Set 1) | Speedup (Set 2) | Average Speedup |
|-------------------|-----------------|-----------------|-----------------|
| Temperature = 0   | 1.55            | 1.38            | 1.46            |
| Temperature = 0.5 | 1.53            | 1.49            | 1.51            |

The speedup ratio seems to increase as the target model size increases (and when the draft model is also relatively big enough). So, the speedup ratio of 2-2.5x mentioned in the Deepmind paper, could also be true for a 70B target model and a 7B draft model (which they use).

### Qwen3-32B-FP8 target and Qwen3-1.7B-FP8 draft

Measured on one NVIDIA H100 80 GB with Transformers' Triton FP8 kernels.
The speedup is speculative throughput divided by autoregressive throughput.

| Temperature | Prompt set | Autoregressive (tok/s) | Speculative (tok/s) | Speedup |
|-------------|------------|-----------------------:|--------------------:|--------:|
| 0           | 1          | 11.30                  | 9.53                | 0.843×  |
| 0           | 2          | 11.61                  | 10.17               | 0.877×  |
| 0.5         | 1          | 11.58                  | 9.61                | 0.830×  |
| 0.5         | 2          | 11.63                  | 10.27               | 0.884×  |
| Overall     | Both       | 11.52                  | 9.87                | 0.857×  |

Both methods generated 2,944 tokens across the four cases. The Qwen3 FP8
ratios are below the historical OPT ratios of 1.38×–1.83×. The runs differ in
models, runtime, and token accounting. See the [benchmark log](outputs/qwen3_fp8_h100.txt).

Greedy outputs matched on 21 of 23 prompts. A replay of one divergence
confirmed that the teacher's single-step and block forwards selected different
argmax tokens for the same causal prefix.
