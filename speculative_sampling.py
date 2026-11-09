import torch

from qwen3_cache import Qwen3Cache, validate_request
from utils import get_distribution, sample, sample_probabilities


class SpeculativeDecoder:
    """Batch-one stages sharing fixed buffers and device-resident decoding state.

    After prefill/append, target KV covers tokens through length - 1 and draft
    KV through length - 2. The remaining committed tokens are consumed next round.
    Call prefill once, then draft, verify, and append in that order.
    """

    def __init__(self, target_model, draft_model, initial_prompt_seq, target_len,
                 lookahead=3, temperature=1.0):
        validate_request(target_model, initial_prompt_seq, target_len, temperature)
        validate_request(draft_model, initial_prompt_seq, target_len, temperature)
        if target_model.config.vocab_size != draft_model.config.vocab_size:
            raise ValueError('Target and draft must have the same vocabulary size')
        if not isinstance(lookahead, int) or lookahead < 1:
            raise ValueError('Lookahead must be a positive integer')
        if target_len <= initial_prompt_seq.shape[1]:
            raise ValueError('A decoder needs at least one new token to generate')

        self.prompt = initial_prompt_seq.detach().clone()
        self.target_len = target_len
        self.lookahead = min(lookahead, target_len - self.prompt.shape[1])
        self.temperature = temperature
        capacity = target_len + self.lookahead
        self.target = Qwen3Cache(target_model, capacity)
        self.proposer = Qwen3Cache(draft_model, capacity)
        device = self.prompt.device
        self.tokens = torch.zeros((1, capacity), dtype=torch.long, device=device)
        self.tokens[:, :self.prompt.shape[1]].copy_(self.prompt)
        self.length = torch.tensor(self.prompt.shape[1], device=device)
        self.accepted = torch.zeros((), dtype=torch.long, device=device)
        self.block = torch.zeros((1, self.lookahead + 1), dtype=torch.long, device=device)
        self.commit_tokens = torch.zeros_like(self.block)
        self.draft_input = torch.zeros((1, 2), dtype=torch.long, device=device)
        self.draft_probs = torch.zeros(
            (1, self.lookahead + 1, draft_model.config.vocab_size), device=device,
        )
        self.block_offsets = torch.arange(self.lookahead + 1, device=device)[None, :]
        self.catchup_offsets = torch.tensor([[-2, -1]], device=device)

    @torch.inference_mode()
    def prefill(self):
        logits = self.target.forward(self.prompt)[:, -1, :]
        self.proposer.forward(self.prompt)
        self.tokens[:, self.prompt.shape[1]:self.prompt.shape[1] + 1].copy_(
            sample(logits, self.temperature)
        )
        self.length.add_(1)
        self.proposer.rewind(self.length - 2)

    @torch.inference_mode()
    def draft(self):
        torch.gather(self.tokens, 1, self.length + self.catchup_offsets, out=self.draft_input)
        self.block[:, :1].copy_(self.draft_input[:, -1:])
        for step in range(self.lookahead):
            # Two tokens also catch up the uncached final proposal after full acceptance.
            inputs = self.draft_input if step == 0 else self.draft_input[:, -1:]
            logits = self.proposer.forward(inputs)[:, -1, :]
            if self.temperature == 0:
                token = logits.argmax(dim=-1, keepdim=True)
            else:
                probs = get_distribution(logits, self.temperature)
                self.draft_probs[:, step, :].copy_(probs)
                token = sample_probabilities(probs)
            self.block[:, step + 1:step + 2].copy_(token)
            self.draft_input[:, -1:].copy_(token)

    @torch.inference_mode()
    def verify(self):
        logits = self.target.forward(self.block, logits_to_keep=0)
        proposals = self.block[:, 1:]
        if self.temperature == 0:
            target_tokens = logits.argmax(dim=-1)
            matches = target_tokens[:, :-1] == proposals
            self.accepted.copy_(matches.long().cumprod(dim=-1).sum())
            replacement = target_tokens.gather(1, self.accepted.view(1, 1))
        else:
            probs = get_distribution(logits, self.temperature)
            p = probs[:, :-1, :].gather(-1, proposals.unsqueeze(-1)).squeeze(-1)
            q = self.draft_probs[:, :-1, :].gather(-1, proposals.unsqueeze(-1)).squeeze(-1)
            matches = torch.rand_like(p) * q < p
            self.accepted.copy_(matches.long().cumprod(dim=-1).sum())
            index = self.accepted.view(1)
            # The final draft-probability row stays zero, giving p for the bonus token.
            residual = (probs.index_select(1, index) - self.draft_probs.index_select(1, index))
            replacement = sample_probabilities(residual[:, 0, :].clamp_min(0))
        self.commit_tokens[:, :-1].copy_(proposals)
        self.commit_tokens.scatter_(1, self.accepted.view(1, 1), replacement)

    @torch.inference_mode()
    def append(self):
        self.tokens.scatter_(1, self.length + self.block_offsets, self.commit_tokens)
        self.length.add_(self.accepted + 1).clamp_(max=self.target_len)
        self.target.rewind(self.length - 1)
        self.proposer.rewind(self.length - 2)


@torch.inference_mode()
def speculative_sampling(target_model, draft_model, initial_prompt_seq, target_len,
                         tokenizer, lookahead=3, temperature=1.0, debug=True):
    validate_request(target_model, initial_prompt_seq, target_len, temperature)
    if target_len <= initial_prompt_seq.shape[-1]:
        if not isinstance(lookahead, int) or lookahead < 1:
            raise ValueError('Lookahead must be a positive integer')
        return initial_prompt_seq.detach().clone()
    decoder = SpeculativeDecoder(
        target_model, draft_model, initial_prompt_seq, target_len, lookahead, temperature
    )
    decoder.prefill()
    length = initial_prompt_seq.shape[1] + 1
    while length < target_len:
        start = length
        decoder.draft()
        decoder.verify()
        decoder.append()
        # Completion checks and text decoding stay outside the capturable stages.
        length = decoder.length.item()
        if debug:
            print(f"Possible continuations: {tokenizer.decode(decoder.block[0, 1:], skip_special_tokens=True)}")
            print(f"Accepted continuations: {tokenizer.decode(decoder.tokens[0, start:length], skip_special_tokens=True)}")
    return decoder.tokens[:, :target_len].clone()
