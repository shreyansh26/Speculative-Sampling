import torch
from utils import sample_from_draft_model, get_distribution, sample
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
  "EleutherAI/pythia-160m",
  cache_dir="./pythia-160m",
)

def speculative_sampling(target_model, draft_model, initial_prompt_seq, max_new_tokens,  lookahead=4, temperature=1.0, debug=True):
    assert initial_prompt_seq.shape[0] == 1, 'Batch size should be 1'

    n = initial_prompt_seq.shape[-1]
    fin_prompt_seq = initial_prompt_seq.detach().clone()

    while n < max_new_tokens:
        n_orig = n
        N = fin_prompt_seq.shape[-1]
        draft_outputs, draft_logits = sample_from_draft_model(draft_model, fin_prompt_seq, new_tokens=lookahead, temperature=temperature)
        
        if debug:
            print(f"Possible continuations: {tokenizer.decode(draft_outputs[0,n_orig:], skip_special_tokens=True)}")

        target_logits = target_model(draft_outputs).logits[:, -lookahead-1:, :] # B x (lookahead+1) x V

        target_model_distribution = get_distribution(target_logits, temperature)
        draft_model_distribution = get_distribution(draft_logits, temperature)
        
        accepted_flag = 1
        
        for t in range(lookahead):
            # print("n+t", n+t)
            numerator = target_model_distribution[:, t, draft_outputs[0, N+t]]
            denominator = draft_model_distribution[:, t, draft_outputs[0, N+t]]
            ratio = (numerator / denominator)
            uniform_distribution = torch.rand_like(numerator)
            ones_tensor = torch.ones_like(numerator)

            if (uniform_distribution < torch.min(ones_tensor, ratio)).any():
                fin_prompt_seq = torch.concat([fin_prompt_seq, draft_outputs[:, N+t].unsqueeze(dim=-1)], dim=-1)
                n += 1
            else:
                new_dist = (numerator - denominator)
                new_dist = torch.max(torch.zeros_like(new_dist), new_dist)
                new_dist = new_dist / new_dist.sum(dim=-1, keepdim=True)
                torch.set_printoptions(profile="full")
                print(new_dist) # prints the whole tensor
                torch.set_printoptions(profile="default") # reset
                token_id = torch.multinomial(new_dist, num_samples=1)[0]
                fin_prompt_seq = torch.concat([fin_prompt_seq, token_id[None,...]], dim=-1)
                accepted_flag = 0
                break

        if accepted_flag == 1:
            sample_token = sample(target_logits[:, -1, :], temperature=temperature)
            fin_prompt_seq = torch.concat([fin_prompt_seq, sample_token[None,...]], dim=-1)
        
        if debug:
            print(f"Accepted continuations: {tokenizer.decode(fin_prompt_seq[0,n_orig:], skip_special_tokens=True)}")

        n += 1

    return fin_prompt_seq