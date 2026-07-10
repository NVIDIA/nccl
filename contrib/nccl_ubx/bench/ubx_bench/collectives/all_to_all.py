"""AllToAll benchmark for all backends."""

from __future__ import annotations

import gc
import os
import time
import torch
import torch.distributed as dist

from ..report import BenchResult, compute_bandwidth


def bench_alltoall_nccl(
    size_bytes: int, dtype: torch.dtype, device: torch.device,
    iters: int, warmup: int, nranks: int, cudagraph: int = 10000,
) -> BenchResult:
    """Benchmark NCCL all_to_all."""
    element_size = torch.tensor(0, dtype=dtype).element_size()
    total_count = size_bytes // element_size
    chunk_size = total_count // nranks
    input_tensor = torch.randn(total_count, dtype=dtype, device=device)
    output_tensor = torch.empty(total_count, dtype=dtype, device=device)

    send_chunks = list(input_tensor.split(chunk_size))
    recv_chunks = list(output_tensor.split(chunk_size))

    for _ in range(warmup):
        dist.all_to_all(recv_chunks, send_chunks)
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
                    dist.all_to_all(recv_chunks, send_chunks)
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
            dist.all_to_all(recv_chunks, send_chunks)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1e6)

    times.sort()
    time_us = times[len(times) // 2] if times else 0.0  # iters=0 -> empty
    algbw, busbw = compute_bandwidth(size_bytes, time_us, nranks, "all_to_all")

    dtype_str = {torch.bfloat16: "bf16", torch.float16: "fp16", torch.float32: "fp32"}
    return BenchResult(
        size_bytes=size_bytes, count=total_count,
        dtype=dtype_str.get(dtype, str(dtype)), redop="sum",
        time_us=time_us, algbw_gbs=algbw, busbw_gbs=busbw,
    )


def bench_alltoall_ubx(
    size_bytes: int, dtype: torch.dtype, device: torch.device,
    iters: int, warmup: int, nranks: int,
    smlimit: int = 0, nthreads_per_block: int = 0,
    group=None, cudagraph: int = 10000,
    kernel: str = "auto",
) -> BenchResult:
    """Benchmark UB-X alltoall. kernel='lamport' uses barrier-free Lamport polling.

    `nthreads_per_block` is forwarded to all alltoall variants
    (uc / lamport / auto). 0 = launcher default (1024 threads/block).
    """
    from ubx import SymmAllocator

    element_size = torch.tensor(0, dtype=dtype).element_size()
    count = size_bytes // element_size
    # `auto` may pick Lamport at small sizes, so size the pool for Lamport's triple buffer.
    lamport = kernel in ("lamport", "auto")

    if group is None:
        group = dist.group.WORLD

    if cudagraph > 0:
        os.environ["UBX_GRAPH_POOL_SHARE"] = "0.5"
        # Lamport needs extra pool for triple buffering
        pool_multiplier = 20 if lamport else 12
        pool_size = max(size_bytes * pool_multiplier, 64 * 1024 * 1024)
    else:
        os.environ["UBX_GRAPH_POOL_SHARE"] = "0.1"
        pool_multiplier = 16 if lamport else 6
        pool_size = max(size_bytes * pool_multiplier, 64 * 1024 * 1024)

    allocator = SymmAllocator(pool_size, device, group)
    tensor = allocator.create_tensor(torch.Size([count]), dtype)
    tensor.fill_(1.0)

    if kernel == "lamport":
        op_fn = lambda t: allocator.alltoall_lamport(
            t, smlimit=smlimit, nthreads=nthreads_per_block)
    elif kernel == "auto":
        op_fn = lambda t: allocator.alltoall_auto(
            t, smlimit=smlimit, nthreads=nthreads_per_block)
    else:
        op_fn = lambda t: allocator.alltoall(
            t, smlimit=smlimit, nthreads=nthreads_per_block)

    if cudagraph > 0:
        for _ in range(warmup):
            tensor = op_fn(tensor)
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
                    tensor = op_fn(tensor)
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
    time_us = times[len(times) // 2] if times else 0.0  # iters=0 -> empty
    algbw, busbw = compute_bandwidth(size_bytes, time_us, nranks, "all_to_all")

    dtype_str = {torch.bfloat16: "bf16", torch.float16: "fp16", torch.float32: "fp32"}
    return BenchResult(
        size_bytes=size_bytes, count=count,
        dtype=dtype_str.get(dtype, str(dtype)), redop="sum",
        time_us=time_us, algbw_gbs=algbw, busbw_gbs=busbw,
    )
