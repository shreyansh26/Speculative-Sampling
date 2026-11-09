import math

import torch
from transformers import StaticCache


def validate_request(model, input_ids, target_len, temperature):
    if model.config.model_type != 'qwen3':
        raise ValueError('Only Qwen3 models are supported')
    if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] == 0:
        raise ValueError('Expected a nonempty prompt with batch size 1')
    if input_ids.dtype != torch.long:
        raise ValueError('Prompt token IDs must have dtype torch.long')
    if any(parameter.device != input_ids.device for parameter in model.parameters()):
        raise ValueError('The model and prompt must be on the same device')
    if ((input_ids < 0) | (input_ids >= model.config.vocab_size)).any():
        raise ValueError('Prompt token IDs are outside the model vocabulary')
    if not isinstance(target_len, int) or target_len < 0:
        raise ValueError('Target length must be a nonnegative integer')
    if not math.isfinite(temperature) or temperature < 0:
        raise ValueError('Temperature must be finite and nonnegative')
    if model.training:
        raise ValueError('Call model.eval() before sampling')
    if any(layer_type != 'full_attention' for layer_type in model.config.layer_types):
        raise ValueError('Only full-attention Qwen3 models are supported')
    if model.config._attn_implementation not in ('sdpa', 'eager'):
        raise ValueError('Use Qwen3 with sdpa or eager attention')
    if model.config.rope_parameters['rope_type'] != 'default':
        raise ValueError('Only the default Qwen3 rotary embeddings are supported')


class Qwen3Cache:
    """Incremental Qwen3 forwards with fixed KV storage and a device-side cursor."""

    def __init__(self, model, capacity):
        self.model = model
        self.cache = StaticCache(config=model.config, max_cache_len=capacity)
        self.key_positions = torch.arange(capacity, device=model.device)

    def forward(self, input_ids, logits_to_keep=1):
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)
        positions = positions + self.cache.get_seq_length()
        mask = torch.zeros(
            (1, 1, input_ids.shape[1], self.key_positions.numel()),
            dtype=self.model.dtype, device=input_ids.device,
        )
        mask.masked_fill_(
            self.key_positions[None, :] > positions[:, None], torch.finfo(mask.dtype).min
        )
        return self.model(
            input_ids, position_ids=positions[None, :],
            attention_mask={'full_attention': mask},
            past_key_values=self.cache, use_cache=True, logits_to_keep=logits_to_keep,
        ).logits

    def rewind(self, length):
        # Keep the allocation and hide stale slots with the next forward's causal mask.
        for layer in self.cache.layers:
            layer.cumulative_length.copy_(length)
