import torch

from cuda_graphs import capture_cuda_graphs
from qwen3_cache import Qwen3Cache, kv_splits_for_prompt, validate_request
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
        self.graphs = {}
        self.graph_cache = {}
        self.capture_seconds = 0.0

    @torch.inference_mode()
    def prefill(self):
        self.cache.num_splits = kv_splits_for_prompt(self.prompt_length)
        self.graphs = self.graph_cache.get(self.cache.num_splits, {}) if self.cuda_graphs else {}
        self.length.zero_()
        if self.cache.cache.layers[0].is_initialized:
            self.cache.rewind(self.length)
        prompt = self.prompt[:, :self.prompt_length]
        self.tokens[:, :self.prompt_length].copy_(prompt)
        logits = self.cache.forward(prompt, prefill=True)[:, -1, :]
        self.token.copy_(sample(logits, self.temperature))
        self.tokens[:, self.prompt_length:self.prompt_length + 1].copy_(self.token)
        self.length.fill_(self.prompt_length + 1)

    @torch.inference_mode()
    def decode(self):
        if self.graphs:
            self.graphs['decode'].replay()
            return
        logits = self.cache.forward(self.token)[:, -1, :]
        self.token.copy_(sample(logits, self.temperature))
        self.tokens.scatter_(1, self.length.view(1, 1), self.token)
        self.length.add_(1)

    @torch.inference_mode()
    def capture_decode(self):
        if self.graphs:
            return
        device = self.tokens.device
        if device.type != 'cuda':
            raise ValueError('CUDA graph capture requires CUDA')
        if not self.cache.cache.layers[0].is_initialized:
            raise ValueError('Prefill before capturing decode')
        if self.length.item() >= self.target_len:
            return
        length = self.length.clone()
        token = self.token.clone()

        def restore():
            self.length.copy_(length)
            self.token.copy_(token)
            self.cache.rewind(length - 1)

        self.graphs, seconds = capture_cuda_graphs(
            {'decode': self.decode}, restore, device,
        )
        self.graph_cache[self.cache.num_splits] = self.graphs
        self.capture_seconds += seconds

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
        remaining = self.target_len - self.prompt_length - 1
        if self.cuda_graphs and remaining and not self.graphs:
            self.capture_decode()
        for _ in range(remaining):
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
