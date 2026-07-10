"""ReduceScatter benchmark for all backends."""

from __future__ import annotations

import gc
import time
import torch
import torch.distributed as dist

from ..report import BenchResult, compute_bandwidth


def bench_reducescatter_nccl(
    size_bytes: int, dtype: torch.dtype, device: torch.device,
    iters: int, warmup: int, nranks: int, cudagraph: int = 10000,
) -> BenchResult:
    """Benchmark NCCL reduce_scatter."""
    element_size = torch.tensor(0, dtype=dtype).element_size()
    total_count = size_bytes // element_size
    chunk_size = total_count // nranks
    input_tensor = torch.randn(total_count, dtype=dtype, device=device)
    output_tensor = torch.empty(chunk_size, dtype=dtype, device=device)

    for _ in range(warmup):
        dist.reduce_scatter_tensor(output_tensor, input_tensor)
    torch.cuda.synchronize()

    if cudagraph > 0:
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        gc_was_enabled = gc.isenabled()
        if gc_was_enabled:
            gc.disable()
        with torch.cuda.stream(stream):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                for _ in range(cudagraph):
                    dist.reduce_scatter_tensor(output_tensor, input_tensor)
        if gc_was_enabled:
            gc.enable()
            gc.collect()
        torch.cuda.current_stream().wait_stream(stream)

        for _ in range(warmup):
            graph.replay()
        torch.cuda.synchronize()

        times = []
        for _ in range(iters):
            torch.cuda.synchronize()
            start = time.perf_counter()
            graph.replay()
            torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1e6 / cudagraph)

        del graph
    else:
        times = []
        for _ in range(iters):
            torch.cuda.synchronize()
            start = time.perf_counter()
            dist.reduce_scatter_tensor(output_tensor, input_tensor)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1e6)

    times.sort()
    time_us = times[len(times) // 2]
    algbw, busbw = compute_bandwidth(size_bytes, time_us, nranks, "reduce_scatter")

    dtype_str = {torch.bfloat16: "bf16", torch.float16: "fp16", torch.float32: "fp32"}
    return BenchResult(
        size_bytes=size_bytes, count=total_count,
        dtype=dtype_str.get(dtype, str(dtype)), redop="sum",
        time_us=time_us, algbw_gbs=algbw, busbw_gbs=busbw,
    )
