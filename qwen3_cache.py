import math
from functools import lru_cache

import torch
from transformers import AttentionInterface

FLASH_ATTENTION_REVISION = '0b1e31f22cc994ed4cafd8bb09c23eefa798f24d'


# shortcut: H100-oriented thresholds; retune for other GPUs.
KV_SPLIT_BUCKETS = ((1, 1), (512, 4), (1024, 8), (4096, 16))


def kv_splits_for_length(length):
    for minimum, splits in reversed(KV_SPLIT_BUCKETS):
        if length >= minimum:
            return splits


def kv_splits_for_range(start, stop):
    """Yield split counts used at committed lengths in [start, stop)."""
    if start < stop:
        yield kv_splits_for_length(start)
        for minimum, splits in KV_SPLIT_BUCKETS:
            if start < minimum < stop:
                yield splits


@lru_cache(maxsize=1)
def flash_attention_kernel():
    from kernels import get_kernel
    return get_kernel('kernels-community/flash-attn3', revision=FLASH_ATTENTION_REVISION)


def cached_attention(module, query, key, value, attention_mask, *,
                     qwen3_cache, prefill, scaling, **kwargs):
    layer_idx = module.layer_idx
    if qwen3_cache.layers[layer_idx] is None:
        shape = (key.shape[0], key.shape[1], qwen3_cache.capacity, key.shape[-1])
        qwen3_cache.layers[layer_idx] = (key.new_zeros(shape), value.new_zeros(shape))
    keys, values = qwen3_cache.layers[layer_idx]
    if prefill:
        length = key.shape[2]
        keys[:, :, :length].copy_(key)
        values[:, :, :length].copy_(value)
        output = qwen3_cache.kernel.flash_attn_func(
            query.transpose(1, 2), keys[:, :, :length].transpose(1, 2),
            values[:, :, :length].transpose(1, 2), causal=True, softmax_scale=scaling,
        )
        return output, None
    pre_append_length = qwen3_cache.valid_length
    output = qwen3_cache.kernel.flash_attn_with_kvcache(
        query.transpose(1, 2), keys.transpose(1, 2), values.transpose(1, 2),
        k=key.transpose(1, 2), v=value.transpose(1, 2), cache_seqlens=pre_append_length,
        causal=True, softmax_scale=scaling,
        num_splits=qwen3_cache.num_splits,
    )
    return output, None


AttentionInterface.register('qwen3_kvcache', cached_attention)


def validate_request(model, input_ids, target_len, temperature, max_seq_len=None):
    if model.config.model_type != 'qwen3':
        raise ValueError('Only Qwen3 models are supported')
    if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] == 0:
        raise ValueError('Expected a nonempty prompt with batch size 1')
    if input_ids.dtype != torch.long:
        raise ValueError('Prompt token IDs must have dtype torch.long')
    if input_ids.device.type != 'cuda' or model.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError('FlashAttention requires CUDA with FP16 or BF16 activations (FP8 weights are supported)')
    if any(parameter.device != input_ids.device for parameter in model.parameters()):
        raise ValueError('The model and prompt must be on the same device')
    if ((input_ids < 0) | (input_ids >= model.config.vocab_size)).any():
        raise ValueError('Prompt token IDs are outside the model vocabulary')
    if not isinstance(target_len, int) or target_len < 0:
        raise ValueError('Target length must be a nonnegative integer')
    if max_seq_len is not None:
        if not isinstance(max_seq_len, int) or max_seq_len < max(target_len, input_ids.shape[1]):
            raise ValueError('Prompt and target length must fit max_seq_len')
    if not math.isfinite(temperature) or temperature < 0:
        raise ValueError('Temperature must be finite and nonnegative')
    if model.training:
        raise ValueError('Call model.eval() before sampling')
    if any(layer_type != 'full_attention' for layer_type in model.config.layer_types):
        raise ValueError('Only full-attention Qwen3 models are supported')
    if model.config.rope_parameters['rope_type'] != 'default':
        raise ValueError('Only the default Qwen3 rotary embeddings are supported')


class Qwen3Cache:
    """Incremental Qwen3 forwards with fixed KV storage and a device-side cursor."""

    def __init__(self, model, capacity):
        self.model = model
        self.capacity = capacity
        self.layers = [None] * model.config.num_hidden_layers
        self.kernel = flash_attention_kernel()
        self.num_splits = 1
        self.valid_length = torch.zeros(1, dtype=torch.int32, device=model.device)

    @property
    def is_initialized(self):
        return self.layers[0] is not None

    def forward(self, input_ids, logits_to_keep=1, prefill=False):
        query_length = input_ids.shape[1]
        positions = torch.arange(query_length, device=input_ids.device)
        if not prefill:
            positions.add_(self.valid_length)
        implementation = self.model.config._attn_implementation
        # Select this backend only for our forward; ordinary model calls retain theirs.
        self.model.config._attn_implementation = 'qwen3_kvcache'
        try:
            kwargs = dict(
                position_ids=positions[None, :],
                attention_mask={'full_attention': None},
                use_cache=False,
                qwen3_cache=self, prefill=prefill,
            )
            if logits_to_keep is None:
                self.model.base_model(input_ids, **kwargs)
                logits = None
            else:
                logits = self.model(input_ids, logits_to_keep=logits_to_keep, **kwargs).logits
        finally:
            self.model.config._attn_implementation = implementation
        if prefill:
            self.valid_length.fill_(query_length)
        else:
            self.valid_length.add_(query_length)
        return logits

    def rewind(self, length):
        self.valid_length.copy_(length)
