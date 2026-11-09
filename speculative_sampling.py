import torch
from utils import sample_from_draft_model, get_distribution, sample


@torch.inference_mode()
def speculative_sampling(target_model, draft_model, initial_prompt_seq, target_len, tokenizer, lookahead=4, temperature=1.0, debug=True):
    """Run Algorithm 2 (https://arxiv.org/abs/2302.01318) with a shared token vocabulary."""
    assert initial_prompt_seq.shape[0] == 1, 'Batch size should be 1'
    if lookahead < 1:
        raise ValueError('Lookahead must be at least 1')

    fin_prompt_seq = initial_prompt_seq.detach().clone()
    while fin_prompt_seq.shape[-1] < target_len:
        start = fin_prompt_seq.shape[-1]
        remaining = target_len - start
        draft_len = min(lookahead, remaining)
        draft_outputs, draft_logits = sample_from_draft_model(
            draft_model, fin_prompt_seq, new_tokens=draft_len, temperature=temperature
        )
        if debug:
            print(f"Possible continuations: {tokenizer.decode(draft_outputs[0, start:], skip_special_tokens=True)}")

        target_logits = target_model(draft_outputs, use_cache=False).logits[:, -draft_len-1:, :]
        if target_logits.shape[-1] != draft_logits.shape[-1]:
            raise ValueError('Target and draft models must have the same logit vocabulary size')
        target_probs = get_distribution(target_logits, temperature)
        draft_probs = get_distribution(draft_logits, temperature)

        for step in range(draft_len):
            token = draft_outputs[:, start+step:start+step+1]
            p = target_probs[:, step, :]
            q = draft_probs[:, step, :]
            p_token = p.gather(-1, token)
            q_token = q.gather(-1, token)
            # Equivalent to accepting with min(1, p_token / q_token).
            if torch.rand_like(p_token) * q_token >= p_token:
                replacement = torch.multinomial((p - q).clamp_min(0), num_samples=1)
                fin_prompt_seq = torch.cat((fin_prompt_seq, replacement), dim=-1)
                break
            fin_prompt_seq = torch.cat((fin_prompt_seq, token), dim=-1)
        else:
            if draft_len < remaining:
                bonus = sample(target_logits[:, -1, :], temperature=temperature)
                fin_prompt_seq = torch.cat((fin_prompt_seq, bonus[None, ...]), dim=-1)

        if debug:
            print(f"Accepted continuations: {tokenizer.decode(fin_prompt_seq[0, start:], skip_special_tokens=True)}")

    return fin_prompt_seq
