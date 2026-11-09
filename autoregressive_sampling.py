import torch
from qwen3_cache import Qwen3Cache, validate_request
from utils import sample

@torch.inference_mode()
def autoregressive_sampling(model, initial_prompt_seq, target_len, temperature=1.0):
    validate_request(model, initial_prompt_seq, target_len, temperature)
    prompt_len = initial_prompt_seq.shape[1]
    if target_len <= prompt_len:
        return initial_prompt_seq.detach().clone()
    cache = Qwen3Cache(model, target_len)
    tokens = torch.empty((1, target_len), dtype=torch.long, device=initial_prompt_seq.device)
    tokens[:, :prompt_len].copy_(initial_prompt_seq)
    logits = cache.forward(initial_prompt_seq)[:, -1, :]
    token = torch.empty((1, 1), dtype=torch.long, device=tokens.device)
    position = torch.tensor([[prompt_len]], device=tokens.device)
    for step in range(target_len - prompt_len):
        token.copy_(sample(logits, temperature))
        tokens.scatter_(1, position, token)
        position.add_(1)
        if step + 1 < target_len - prompt_len:
            logits = cache.forward(token)[:, -1, :]
    return tokens
