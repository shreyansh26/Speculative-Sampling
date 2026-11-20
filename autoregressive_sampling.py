import torch

from cuda_graphs import capture_cuda_graphs
from qwen3_cache import Qwen3Cache, kv_splits_for_length, kv_splits_for_range, validate_request
from utils import sample


class AutoregressiveDecoder:
    """Reusable eager prefill and one-token decode, optionally captured on CUDA."""

    def __init__(self, model, initial_prompt_seq, target_len, temperature=1.0, cuda_graphs=False,
                 max_seq_len=None):
        self.max_seq_len = target_len if max_seq_len is None else max_seq_len
        validate_request(model, initial_prompt_seq, target_len, temperature, self.max_seq_len)
        if target_len <= initial_prompt_seq.shape[1]:
            raise ValueError('A decoder needs at least one new token to generate')
        self.prompt = torch.empty((1, self.max_seq_len), dtype=torch.long, device=initial_prompt_seq.device)
        self.prompt_length = initial_prompt_seq.shape[1]
        self.prompt[:, :self.prompt_length].copy_(initial_prompt_seq)
        self.target_len = target_len
        self.temperature = temperature
        self.cuda_graphs = cuda_graphs
        self.cache = Qwen3Cache(model, self.max_seq_len)
        self.tokens = torch.empty_like(self.prompt)
        self.token = torch.empty((1, 1), dtype=torch.long, device=self.prompt.device)
        self.length = torch.tensor(self.prompt_length, device=self.prompt.device)
        self.active_graphs = {}
        self.graphs_by_split = {}
        self.capture_seconds = 0.0

    def select_split(self, splits):
        self.cache.num_splits = splits
        self.active_graphs = self.graphs_by_split.get(splits, {}) if self.cuda_graphs else {}

    @torch.inference_mode()
    def prefill(self):
        self.select_split(kv_splits_for_length(self.prompt_length + 1))
        prompt = self.prompt[:, :self.prompt_length]
        self.tokens[:, :self.prompt_length].copy_(prompt)
        logits = self.cache.forward(prompt, prefill=True)[:, -1, :]
        self.token.copy_(sample(logits, self.temperature))
        self.tokens[:, self.prompt_length:self.prompt_length + 1].copy_(self.token)
        self.length.fill_(self.prompt_length + 1)

    @torch.inference_mode()
    def decode(self):
        if self.active_graphs:
            self.active_graphs['decode'].replay()
            return
        logits = self.cache.forward(self.token)[:, -1, :]
        self.token.copy_(sample(logits, self.temperature))
        self.tokens.scatter_(1, self.length.view(1, 1), self.token)
        self.length.add_(1)

    @torch.inference_mode()
    def capture_decode(self, length=None):
        device = self.tokens.device
        if device.type != 'cuda':
            raise ValueError('CUDA graph capture requires CUDA')
        if not self.cache.is_initialized:
            raise ValueError('Prefill before capturing decode')
        length = self.length.item() if length is None else length
        missing = [splits for splits in kv_splits_for_range(length, self.target_len)
                   if splits not in self.graphs_by_split]
        if not missing:
            return
        saved_length = self.length.clone()
        token = self.token.clone()

        def restore():
            self.length.copy_(saved_length)
            self.token.copy_(token)
            self.cache.rewind(saved_length - 1)

        try:
            for splits in missing:
                self.select_split(splits)
                graphs, seconds = capture_cuda_graphs({'decode': self.decode}, restore, device)
                self.graphs_by_split[splits] = graphs
                self.capture_seconds += seconds
        finally:
            self.select_split(kv_splits_for_length(length))

    @torch.inference_mode()
    def generate(self, initial_prompt_seq=None, target_len=None):
        target_len = self.target_len if target_len is None else target_len
        prompt = self.prompt[:, :self.prompt_length] if initial_prompt_seq is None else initial_prompt_seq
        validate_request(self.cache.model, prompt, target_len, self.temperature, self.max_seq_len)
        self.target_len = target_len
        if initial_prompt_seq is not None:
            self.prompt_length = initial_prompt_seq.shape[1]
            self.prompt[:, :self.prompt_length].copy_(initial_prompt_seq)
        if target_len <= self.prompt_length:
            return self.prompt[:, :self.prompt_length].clone()
        self.prefill()
        start = self.prompt_length + 1
        if self.cuda_graphs and start < self.target_len:
            self.capture_decode(start)
        for length in range(start, self.target_len):
            splits = kv_splits_for_length(length)
            if splits != self.cache.num_splits:
                self.select_split(splits)
            self.decode()
        return self.tokens[:, :self.target_len].clone()


@torch.inference_mode()
def autoregressive_sampling(model, initial_prompt_seq, target_len, temperature=1.0, cuda_graphs=False):
    validate_request(model, initial_prompt_seq, target_len, temperature)
    prompt_len = initial_prompt_seq.shape[1]
    if target_len <= prompt_len:
        return initial_prompt_seq.detach().clone()
    decoder = AutoregressiveDecoder(model, initial_prompt_seq, target_len, temperature, cuda_graphs)
    return decoder.generate()
