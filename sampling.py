import torch
from utils import sample

def autoregressive_sampling(model, initial_prompt_seq, target_len, temperature=1.0):
    n = initial_prompt_seq.shape[-1]
    fin_prompt_seq = initial_prompt_seq.detach().clone()

    while n < target_len:
        sample_token_logits = model(fin_prompt_seq).logits[:, -1, :]
        sample_token = sample(sample_token_logits, temperature=temperature)
        fin_prompt_seq = torch.concat([fin_prompt_seq, sample_token[None,...]], dim=-1)
        n += 1
    return fin_prompt_seq