import torch

from cuda_graphs import capture_cuda_graphs
from qwen3_cache import Qwen3Cache, kv_splits_for_length, kv_splits_for_range, validate_request
from utils import get_distribution, sample, sample_probabilities


class SpeculativeDecoder:
    """Batch-one stages sharing fixed buffers and device-resident decoding state.

    After prefill or append, if another round will run, let L be the committed
    token count. The committed sequence is tokens[:, :L], the target KV
    prefix is tokens[:, :L-1], and the draft KV prefix is tokens[:, :L-2].
    Slice ends are exclusive. The next draft forward consumes the last two
    committed tokens. Verify consumes the last committed token and the proposals.
    Call prefill once, then draft, verify, and append in that order. When
    driving stages directly, select the split at each committed-length boundary.
    """

    def __init__(self, target_model, draft_model, initial_prompt_seq, target_len,
                 lookahead=3, temperature=1.0, cuda_graphs=False, max_seq_len=None):
        self.max_seq_len = target_len if max_seq_len is None else max_seq_len
        validate_request(target_model, initial_prompt_seq, target_len, temperature, self.max_seq_len)
        validate_request(draft_model, initial_prompt_seq, target_len, temperature, self.max_seq_len)
        if target_model.config.vocab_size != draft_model.config.vocab_size:
            raise ValueError('Target and draft must have the same vocabulary size')
        if not isinstance(lookahead, int) or lookahead < 1:
            raise ValueError('Lookahead must be a positive integer')
        if target_len <= initial_prompt_seq.shape[1]:
            raise ValueError('A decoder needs at least one new token to generate')

        self.prompt = torch.empty((1, self.max_seq_len), dtype=torch.long, device=initial_prompt_seq.device)
        self.prompt_length = initial_prompt_seq.shape[1]
        self.prompt[:, :self.prompt_length].copy_(initial_prompt_seq)
        self.target_len = target_len
        self.lookahead = lookahead
        self.temperature = temperature
        self.cuda_graphs = cuda_graphs
        self.active_graphs = {}
        self.graphs_by_split = {}
        self.capturing = False
        self.capture_seconds = 0.0
        capacity = self.max_seq_len + self.lookahead
        self.target = Qwen3Cache(target_model, capacity)
        self.proposer = Qwen3Cache(draft_model, capacity)
        device = self.prompt.device
        self.tokens = torch.zeros((1, capacity), dtype=torch.long, device=device)
        self.tokens[:, :self.prompt_length].copy_(initial_prompt_seq)
        self.length = torch.tensor(self.prompt_length, device=device)
        self.stop_length = torch.tensor(target_len, device=device)
        self.accepted = torch.zeros((), dtype=torch.long, device=device)
        self.block = torch.zeros((1, self.lookahead + 1), dtype=torch.long, device=device)
        self.commit_tokens = torch.zeros_like(self.block)
        self.draft_input = torch.zeros((1, 2), dtype=torch.long, device=device)
        self.draft_probs = None
        if self.temperature != 0:
            self.draft_probs = torch.zeros(
                (1, self.lookahead + 1, draft_model.config.vocab_size),
                dtype=torch.float32, device=device,
            )
        self.block_offsets = torch.arange(self.lookahead + 1, device=device)[None, :]
        self.catchup_offsets = torch.tensor([[-2, -1]], device=device)

    def select_split(self, splits):
        self.target.num_splits = self.proposer.num_splits = splits
        self.active_graphs = self.graphs_by_split.get(splits, {}) if self.cuda_graphs else {}

    @torch.inference_mode()
    def prefill(self):
        self.select_split(kv_splits_for_length(self.prompt_length + 1))
        self.accepted.zero_()
        prompt = self.prompt[:, :self.prompt_length]
        self.tokens[:, :self.prompt_length].copy_(prompt)
        logits = self.target.forward(prompt, prefill=True)[:, -1, :]
        self.tokens[:, self.prompt_length:self.prompt_length + 1].copy_(
            sample(logits, self.temperature)
        )
        self.length.fill_(self.prompt_length + 1)
        self.stop_length.fill_(self.target_len)
        if self.prompt_length + 1 < self.target_len:
            self.proposer.forward(prompt, prefill=True, logits_to_keep=None)
            self.proposer.rewind(self.length - 2)

    @torch.inference_mode()
    def draft(self):
        if self.active_graphs:
            self.active_graphs['draft'].replay()
            return
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
        if self.active_graphs:
            self.active_graphs['verify'].replay()
            return
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
        if self.active_graphs:
            self.active_graphs['append'].replay()
            return
        self.tokens.scatter_(1, self.length + self.block_offsets, self.commit_tokens)
        self.length.add_(self.accepted + 1).clamp_(max=self.stop_length)
        self.target.rewind(self.length - 1)
        self.proposer.rewind(self.length - 2)

    @torch.inference_mode()
    def capture_stages(self, length=None):
        """Prepare missing variants against the prefilled prefix and preserve RNG."""
        device = self.tokens.device
        if device.type != 'cuda':
            raise ValueError('CUDA graph capture requires CUDA')
        if not self.target.is_initialized:
            raise ValueError('Prefill before capturing decode stages')
        length = self.length.item() if length is None else length
        missing = [splits for splits in kv_splits_for_range(length, self.target_len)
                   if splits not in self.graphs_by_split]
        if not missing:
            return
        saved_length = self.length.clone()

        def restore():
            self.length.copy_(saved_length)
            self.target.rewind(saved_length - 1)
            self.proposer.rewind(saved_length - 2)
            self.accepted.zero_()

        stages = {name: getattr(self, name) for name in ('draft', 'verify', 'append')}
        self.capturing = True
        try:
            for splits in missing:
                self.select_split(splits)
                graphs, seconds = capture_cuda_graphs(stages, restore, device)
                self.graphs_by_split[splits] = graphs
                self.capture_seconds += seconds
        finally:
            self.select_split(kv_splits_for_length(length))
            self.capturing = False

    @torch.inference_mode()
    def generate(self, initial_prompt_seq=None, tokenizer=None, debug=False, target_len=None):
        """Reuse buffers and graphs for any request that fits max_seq_len."""
        target_len = self.target_len if target_len is None else target_len
        prompt = self.prompt[:, :self.prompt_length] if initial_prompt_seq is None else initial_prompt_seq
        validate_request(self.target.model, prompt, target_len, self.temperature, self.max_seq_len)
        self.target_len = target_len
        if initial_prompt_seq is not None:
            self.prompt_length = initial_prompt_seq.shape[1]
            self.prompt[:, :self.prompt_length].copy_(initial_prompt_seq)
        if target_len <= self.prompt_length:
            return self.prompt[:, :self.prompt_length].clone()
        self.prefill()
        length = self.prompt_length + 1
        if self.cuda_graphs and length < self.target_len:
            self.capture_stages(length)
        while length < self.target_len:
            splits = kv_splits_for_length(length)
            if splits != self.target.num_splits:
                self.select_split(splits)
            start = length
            self.draft()
            self.verify()
            self.append()
            # Completion checks and text decoding stay outside the captured stages.
            length = self.length.item()
            if debug:
                print(f"Possible continuations: {tokenizer.decode(self.block[0, 1:], skip_special_tokens=True)}")
                print(f"Accepted continuations: {tokenizer.decode(self.tokens[0, start:length], skip_special_tokens=True)}")
        return self.tokens[:, :self.target_len].clone()


@torch.inference_mode()
def speculative_sampling(target_model, draft_model, initial_prompt_seq, target_len,
                         tokenizer, lookahead=3, temperature=1.0, debug=True, cuda_graphs=False):
    validate_request(target_model, initial_prompt_seq, target_len, temperature)
    if target_len <= initial_prompt_seq.shape[-1]:
        if not isinstance(lookahead, int) or lookahead < 1:
            raise ValueError('Lookahead must be a positive integer')
        return initial_prompt_seq.detach().clone()
    decoder = SpeculativeDecoder(
        target_model, draft_model, initial_prompt_seq, target_len, lookahead, temperature, cuda_graphs
    )
    return decoder.generate(tokenizer=tokenizer, debug=debug)
