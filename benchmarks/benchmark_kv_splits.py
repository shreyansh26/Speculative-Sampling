"""Measure fixed FlashAttention KV splits and validate graph replay on Qwen3 shapes."""

import csv
from pathlib import Path

import torch
from cuda_graphs import capture_cuda_graphs
from qwen3_cache import FLASH_ATTENTION_REVISION, flash_attention_kernel


@torch.inference_mode()
def main():
    torch.manual_seed(7)
    kernel = flash_attention_kernel()
    capacity = 32768 + 64 + 3
    length = torch.tensor([64], dtype=torch.int32, device='cuda')
    # Match StaticCache's head-major layout, presented through sequence-major views.
    k = torch.randn(1, 8, capacity, 128, dtype=torch.bfloat16, device='cuda').transpose(1, 2)
    v = torch.randn_like(k.transpose(1, 2)).transpose(1, 2)
    path = Path('outputs/kv_split_benchmark.csv')
    with path.open('w', newline='') as output:
        writer = None
        for model, heads, query_lengths in (('qwen3_14b', 40, (1, 4)), ('qwen3_1.7b', 16, (1, 2))):
            for query_length in query_lengths:
                q = torch.randn(1, query_length, heads, 128, dtype=torch.bfloat16, device='cuda')
                for splits in (1, 4, 8, 16):
                    length.fill_(64)
                    held = {}
                    def attention():
                        return kernel.flash_attn_with_kvcache(
                            q, k, v, cache_seqlens=length, causal=True, num_splits=splits,
                        )
                    def repeated_attention():
                        for _ in range(100):
                            held['out'] = attention()
                    graphs, _ = capture_cuda_graphs({'attention': repeated_attention}, lambda: None, q.device)
                    for valid in (1024, 8192, 32768):
                        length.fill_(valid)
                        eager = attention()
                        graphs['attention'].replay()
                        torch.testing.assert_close(held['out'], eager, rtol=0, atol=0)
                        reference = kernel.flash_attn_with_kvcache(
                            q, k, v, cache_seqlens=length, causal=True, num_splits=1,
                        )
                        torch.testing.assert_close(eager, reference, rtol=0.02, atol=0.002)
                        error = (eager - reference).abs().max().item()
                        for _ in range(5):
                            graphs['attention'].replay()
                        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        start.record()
                        for _ in range(20):
                            graphs['attention'].replay()
                        end.record()
                        end.synchronize()
                        row = dict(model_shape=model, query_tokens=query_length, valid_kv_tokens=valid,
                                   capacity=capacity, kv_splits=splits, graph_mode=True,
                                   attention_us=start.elapsed_time(end) * 1000 / 2000,
                                   capture_kv_tokens=64, graph_eager_exact=True,
                                   max_abs_diff_from_one_split=error,
                                   gpu=torch.cuda.get_device_name(), torch_version=torch.__version__,
                                   attention_kernel_revision=FLASH_ATTENTION_REVISION)
                        if writer is None:
                            writer = csv.DictWriter(output, fieldnames=list(row), lineterminator='\n')
                            writer.writeheader()
                        writer.writerow(row)
                        output.flush()
                        print(model, query_length, splits, valid, row['attention_us'], flush=True)
                    del graphs, held


if __name__ == '__main__':
    main()
