"""AllReduce benchmark for all backends."""

from __future__ import annotations

import gc
import os
import time
import torch
import torch.distributed as dist
from typing import Optional

from ..report import BenchResult, compute_bandwidth


def bench_allreduce_nccl(
    size_bytes: int, dtype: torch.dtype, device: torch.device,
    iters: int, warmup: int, nranks: int, cudagraph: int = 10000,
) -> BenchResult:
    """Benchmark NCCL allreduce."""
    element_size = torch.tensor(0, dtype=dtype).element_size()
    count = size_bytes // element_size
    tensor = torch.randn(count, dtype=dtype, device=device)

    for _ in range(warmup):
        dist.all_reduce(tensor)
    torch.cuda.synchronize()

    if cudagraph > 0:
        # Capture a single NCCL allreduce in a CUDA graph
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                dist.all_reduce(tensor)
        torch.cuda.current_stream().wait_stream(stream)

        # Graph warmup
        for _ in range(warmup):
            graph.replay()
        torch.cuda.synchronize()

        # Timed: replay cudagraph times per measurement
        times = []
        for _ in range(iters):
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(cudagraph):
                graph.replay()
            torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1e6 / cudagraph)

        del graph
    else:
        times = []
        for _ in range(iters):
            torch.cuda.synchronize()
            start = time.perf_counter()
            dist.all_reduce(tensor)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1e6)

    times.sort()
    time_us = times[len(times) // 2]
    algbw, busbw = compute_bandwidth(size_bytes, time_us, nranks, "all_reduce")

    dtype_str = {torch.bfloat16: "bf16", torch.float16: "fp16", torch.float32: "fp32"}
    return BenchResult(
        size_bytes=size_bytes, count=count,
        dtype=dtype_str.get(dtype, str(dtype)), redop="sum",
        time_us=time_us, algbw_gbs=algbw, busbw_gbs=busbw,
    )


def bench_allreduce_ubx(
    size_bytes: int, dtype: torch.dtype, device: torch.device,
    iters: int, warmup: int, nranks: int,
    kernel: str = "auto", smlimit: int = 0, cgasize: int = 0,
    group=None, cudagraph: int = 10000,
) -> BenchResult:
    """Benchmark UB-X allreduce.

    Args:
        cudagraph: Number of allreduce ops to batch into a single CUDA graph.
            0 = eager mode (default). When > 0, captures `cudagraph` iterations
            in a graph, replays it `iters` times, and reports the median
            per-op latency (replay_time / cudagraph).
    """
    from ubx import SymmAllocator

    element_size = torch.tensor(0, dtype=dtype).element_size()
    count = size_bytes // element_size

    if group is None:
        group = dist.group.WORLD

    if cudagraph > 0:
        # Graph mode: 90% of pool for graph allocations, 10% for eager.
        os.environ["UBX_GRAPH_POOL_SHARE"] = "0.9"
        pool_size = max(size_bytes * 12, 64 * 1024 * 1024)
    else:
        # Eager mode: 10% for graph (unused), 90% for eager allocations.
        os.environ["UBX_GRAPH_POOL_SHARE"] = "0.1"
        pool_size = size_bytes * 6

    allocator = SymmAllocator(pool_size, device, group)
    tensor = allocator.create_tensor(torch.Size([count]), dtype)
    tensor.fill_(1.0)

    # Select kernel variant
    if kernel == "mc":
        op_fn = lambda t: allocator.allreduce_mc(t, smlimit=smlimit, cgasize=cgasize)
    elif kernel == "lamport":
        op_fn = lambda t: allocator.allreduce_lamport(t, smlimit=smlimit, cgasize=cgasize)
    elif kernel == "uc":
        op_fn = lambda t: allocator.allreduce_uc(t, smlimit=smlimit, cgasize=cgasize)
    else:
        op_fn = lambda t: allocator.allreduce(t, smlimit=smlimit, cgasize=cgasize)

    if cudagraph > 0:
        # --- CUDA Graph mode ---
        # Capture `cudagraph` ops in a single graph and replay once per
        # measurement (zero per-op overhead). The single-op-graph approach
        # adds ~4-5us per replay; the N-ops approach captures all ops in
        # one graph and avoids that overhead.

        # Eager warmup to stabilize allocator state
        for _ in range(warmup):
            tensor = op_fn(tensor)
        torch.cuda.synchronize()

        # Capture `cudagraph` ops in a single graph (zero per-op overhead).
        # Disable GC during capture to prevent Python's cyclic garbage
        # collector from destroying SymmTensors, which triggers PyTorch's
        # AllocationRef destructor (a CUDA op forbidden during capture).
        # See https://github.com/pytorch/pytorch/pull/161037
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        gc_was_enabled = gc.isenabled()
        if gc_was_enabled:
            gc.disable()
        with torch.cuda.stream(stream):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                for _ in range(cudagraph):
                    tensor = op_fn(tensor)
        if gc_was_enabled:
            gc.enable()
            gc.collect()
        torch.cuda.current_stream().wait_stream(stream)

        # Graph warmup replays
        for _ in range(warmup):
            graph.replay()
        torch.cuda.synchronize()

        # Timed: single replay per measurement (graph contains cudagraph ops)
        times = []
        for _ in range(iters):
            torch.cuda.synchronize()
            start = time.perf_counter()
            graph.replay()
            torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1e6 / cudagraph)

        del graph
    else:
        # --- Eager mode ---
        for _ in range(warmup):
            op_fn(tensor)
        torch.cuda.synchronize()

        times = []
        for _ in range(iters):
            torch.cuda.synchronize()
            start = time.perf_counter()
            op_fn(tensor)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1e6)

    times.sort()
    time_us = times[len(times) // 2]
    algbw, busbw = compute_bandwidth(size_bytes, time_us, nranks, "all_reduce")

    dtype_str = {torch.bfloat16: "bf16", torch.float16: "fp16", torch.float32: "fp32"}
    return BenchResult(
        size_bytes=size_bytes, count=count,
        dtype=dtype_str.get(dtype, str(dtype)), redop="sum",
        time_us=time_us, algbw_gbs=algbw, busbw_gbs=busbw,
    )
