"""AllGather benchmark for all backends."""

from __future__ import annotations

import gc
import os
import time
import torch
import torch.distributed as dist

from ..report import BenchResult, compute_bandwidth


def bench_allgather_nccl(
    size_bytes: int, dtype: torch.dtype, device: torch.device,
    iters: int, warmup: int, nranks: int, cudagraph: int = 10000,
) -> BenchResult:
    """Benchmark NCCL allgather."""
    element_size = torch.tensor(0, dtype=dtype).element_size()
    count = size_bytes // element_size
    local = torch.randn(count, dtype=dtype, device=device)
    output_list = [torch.empty_like(local) for _ in range(nranks)]

    for _ in range(warmup):
        dist.all_gather(output_list, local)
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
                    dist.all_gather(output_list, local)
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
            dist.all_gather(output_list, local)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1e6)

    times.sort()
    time_us = times[len(times) // 2]
    algbw, busbw = compute_bandwidth(size_bytes, time_us, nranks, "all_gather")

    dtype_str = {torch.bfloat16: "bf16", torch.float16: "fp16", torch.float32: "fp32"}
    return BenchResult(
        size_bytes=size_bytes, count=count,
        dtype=dtype_str.get(dtype, str(dtype)), redop="sum",
        time_us=time_us, algbw_gbs=algbw, busbw_gbs=busbw,
    )


def bench_allgather_ubx(
    size_bytes: int, dtype: torch.dtype, device: torch.device,
    iters: int, warmup: int, nranks: int,
    smlimit: int = 0, group=None, cudagraph: int = 10000,
    kernel: str = "auto",
) -> BenchResult:
    """Benchmark UB-X allgather.

    ``kernel``:
      auto  — multicast when available, UC fallback otherwise (default)
      mc    — force multicast (errors on no-multicast rendezvous)
      uc    — force unicast (works everywhere; required for large EP
              groups (>36 ranks) in environments where multicast
              hardware is unavailable)
    """
    from ubx import SymmAllocator

    element_size = torch.tensor(0, dtype=dtype).element_size()
    count = size_bytes // element_size

    if group is None:
        group = dist.group.WORLD

    if cudagraph > 0:
        os.environ["UBX_GRAPH_POOL_SHARE"] = "0.5"
        # Allgather output is nranks * input, so need larger pool
        pool_size = max(size_bytes * (nranks + 1) * 6, 64 * 1024 * 1024)
    else:
        os.environ["UBX_GRAPH_POOL_SHARE"] = "0.1"
        pool_size = size_bytes * (nranks + 1) * 6

    allocator = SymmAllocator(pool_size, device, group)
    tensor = allocator.create_tensor(torch.Size([count]), dtype)
    tensor.fill_(1.0)

    if kernel == "uc":
        op_fn = lambda t: allocator.allgather_uc(t, smlimit=smlimit)
    else:
        # auto and mc both go through allgather() — auto falls back to UC
        # when mc0_ptr is unset; mc raises an obvious error if it doesn't.
        op_fn = lambda t: allocator.allgather(t, smlimit=smlimit)

    if cudagraph > 0:
        for _ in range(warmup):
            result = op_fn(tensor)
            del result
        torch.cuda.synchronize()

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        gc_was_enabled = gc.isenabled()
        if gc_was_enabled:
            gc.disable()
        with torch.cuda.stream(stream):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                for _ in range(cudagraph):
                    result = op_fn(tensor)
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
        for _ in range(warmup):
            result = op_fn(tensor)
            del result
        torch.cuda.synchronize()

        times = []
        for _ in range(iters):
            torch.cuda.synchronize()
            start = time.perf_counter()
            result = op_fn(tensor)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1e6)
            del result

    times.sort()
    time_us = times[len(times) // 2]
    algbw, busbw = compute_bandwidth(size_bytes, time_us, nranks, "all_gather")

    dtype_str = {torch.bfloat16: "bf16", torch.float16: "fp16", torch.float32: "fp32"}
    return BenchResult(
        size_bytes=size_bytes, count=count,
        dtype=dtype_str.get(dtype, str(dtype)), redop="sum",
        time_us=time_us, algbw_gbs=algbw, busbw_gbs=busbw,
    )
