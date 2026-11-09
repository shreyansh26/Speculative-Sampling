import time

import torch


@torch.inference_mode()
def capture_cuda_graphs(stages, restore, device):
    """Warm and capture stages separately, preserving request state and CUDA RNG."""
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    rng_state = torch.cuda.get_rng_state(device)
    with torch.cuda.device(device):
        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(stream):
            for _ in range(3):
                for stage in stages.values():
                    stage()
                restore()
        torch.cuda.current_stream(device).wait_stream(stream)
        graphs = {}
        for name, stage in stages.items():
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                stage()
            graphs[name] = graph
        restore()
        torch.cuda.set_rng_state(rng_state, device)
        torch.cuda.synchronize(device)
    return graphs, time.perf_counter() - start
