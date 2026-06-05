"""Alltoallv benchmark: variable-length all-to-all with power-law distribution."""

from __future__ import annotations

import gc
import os
import time
import torch
import torch.distributed as dist

from ..report import BenchResult, compute_bandwidth


def powerlaw_split(total_elems, nranks, alpha, seed, device):
    """Generate power-law distributed split sizes (uint4-aligned).

    alpha=0: uniform
    alpha=0.5: moderately skewed (default)
    alpha=1.0: Zipf-like
    """
    torch.manual_seed(seed)
    if alpha == 0:
        weights = torch.ones(nranks, device=device)
    else:
        weights = torch.arange(1, nranks + 1, dtype=torch.float32, device=device) ** (-alpha)
        perm = torch.randperm(nranks, device=device)
        weights = weights[perm]

    raw = (weights / weights.sum() * total_elems).floor().to(torch.int32)
    remainder = total_elems - int(raw.sum().item())
    for i in range(remainder):
        raw[i % nranks] += 1

    # Align to 8 elements (16 bytes / 2 bytes per bf16 = 8 elems per uint4 line)
    align = 8
    return ((raw + align - 1) // align) * align


def bench_alltoallv_nccl(
    size_bytes: int, dtype: torch.dtype, device: torch.device,
    iters: int, warmup: int, nranks: int, alpha: float = 0.5,
) -> BenchResult:
    """Benchmark NCCL alltoallv with power-law split sizes."""
    element_size = torch.tensor(0, dtype=dtype).element_size()
    total_count = size_bytes // element_size

    rank = dist.get_rank()

    # Build split matrix (same on all ranks)
    split_matrix = torch.zeros(nranks, nranks, dtype=torch.int32, device=device)
    for src in range(nranks):
        split_matrix[src] = powerlaw_split(total_count, nranks, alpha, 42 + src, device)

    input_splits = split_matrix[rank]
    output_splits = split_matrix[:, rank]
    total_send = int(input_splits.sum().item())
    total_recv = int(output_splits.sum().item())

    input_tensor = torch.randn(total_send, dtype=dtype, device=device)
    send_chunks = list(input_tensor.split(input_splits.tolist()))
    recv_chunks = [torch.empty(s, dtype=dtype, device=device)
                   for s in output_splits.tolist()]

    for _ in range(warmup):
        dist.all_to_all(recv_chunks, send_chunks)
    torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        torch.cuda.synchronize()
        start = time.perf_counter()
        dist.all_to_all(recv_chunks, send_chunks)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1e6)

    times.sort()
    time_us = times[len(times) // 2]
    # Report size as total bytes sent by this rank
    report_bytes = total_send * element_size
    algbw, busbw = compute_bandwidth(report_bytes, time_us, nranks, "all_to_all")

    dtype_str = {torch.bfloat16: "bf16", torch.float16: "fp16", torch.float32: "fp32"}
    return BenchResult(
        size_bytes=report_bytes, count=total_send,
        dtype=dtype_str.get(dtype, str(dtype)), redop="none",
        time_us=time_us, algbw_gbs=algbw, busbw_gbs=busbw,
    )


def bench_alltoallv_ubx(
    size_bytes: int, dtype: torch.dtype, device: torch.device,
    iters: int, warmup: int, nranks: int,
    smlimit: int = 0, group=None, alpha: float = 0.5,
) -> BenchResult:
    """Benchmark UB-X alltoallv with power-law split sizes."""
    from ubx import SymmAllocator

    element_size = torch.tensor(0, dtype=dtype).element_size()
    total_count = size_bytes // element_size

    if group is None:
        group = dist.group.WORLD
    rank = dist.get_rank(group)

    # Build split matrix (same on all ranks)
    split_matrix = torch.zeros(nranks, nranks, dtype=torch.int32, device=device)
    for src in range(nranks):
        split_matrix[src] = powerlaw_split(total_count, nranks, alpha, 42 + src, device)

    input_splits = split_matrix[rank]
    output_splits = split_matrix[:, rank]
    total_send = int(input_splits.sum().item())
    total_recv = int(output_splits.sum().item())

    os.environ["UBX_GRAPH_POOL_SHARE"] = "0.1"
    # Pool size must be identical on all ranks for symmetric memory rendezvous.
    # Use total_count (same on all ranks) as the base for sizing.
    pool_size = max(total_count * element_size * 12, 64 * 1024 * 1024)
    allocator = SymmAllocator(pool_size, device, group)

    tensor = allocator.create_tensor(torch.Size([total_send]), dtype)
    tensor.fill_(1.0)

    # Precompute offsets and output buffer (once, outside timing loop)
    state = allocator.alltoallv_prepare(output_splits, input_splits, dtype)

    def op_fn():
        allocator.alltoallv_run(tensor, state, smlimit=smlimit)

    for _ in range(warmup):
        op_fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        torch.cuda.synchronize()
        start = time.perf_counter()
        op_fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1e6)

    times.sort()
    time_us = times[len(times) // 2]
    report_bytes = total_send * element_size
    algbw, busbw = compute_bandwidth(report_bytes, time_us, nranks, "all_to_all")

    dtype_str = {torch.bfloat16: "bf16", torch.float16: "fp16", torch.float32: "fp32"}
    return BenchResult(
        size_bytes=report_bytes, count=total_send,
        dtype=dtype_str.get(dtype, str(dtype)), redop="none",
        time_us=time_us, algbw_gbs=algbw, busbw_gbs=busbw,
    )
