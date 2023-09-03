# Speculative Sampling

An implementation of speculative sampling as described in the paper [Accelerating Large Language Model Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318) by Deepmind.

## To run

```
python generate.py -h
```

Run the above command to see the parameter options and their description.

**Speculative Sampling**
```
python generate.py  --method speculative \
                    --prompt "Emily found a mysterious letter on her doorstep one sunny morning." \
                    --max_new_tokens 64 \
                    --target_model facebook/opt-13b \
                    --draft_model facebook/opt-1.3b \
                    --temperature 0.5
```

**Naive Autoregressive Sampling**
```
python generate.py  --method autoregressive \
                    --prompt "Emily found a mysterious letter on her doorstep one sunny morning." \
                    --max_new_tokens 64 \
                    --target_model facebook/opt-13b \
                    --temperature 0.5
```

**Benchmarking**
```
python benchmark.py
```

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