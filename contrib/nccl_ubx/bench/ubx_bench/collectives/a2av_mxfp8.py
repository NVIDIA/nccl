"""AllToAll-v token dispatch with mxfp8 quantization benchmark (UB-X only)."""

from __future__ import annotations

import gc
import os
import time
import torch
import torch.distributed as dist

from ..report import BenchResult, compute_bandwidth


def _make_routing(ntokens: int, total_experts: int, topk: int,
                  device: torch.device, alpha: float = 0.0) -> torch.Tensor:
    """Generate a deterministic top-k routing matrix.

    Each token is routed to exactly *topk* experts chosen round-robin so that
    every rank gets the same routing table (seeded from token index, not RNG).

    Returns:
        routing: [ntokens, total_experts] uint8, non-zero = routed.
    """
    routing = torch.zeros(ntokens, total_experts, dtype=torch.uint8, device=device)
    for t in range(ntokens):
        for k in range(topk):
            e = (t * topk + k) % total_experts
            routing[t, e] = 1
    return routing


def bench_a2av_mxfp8_ubx(
    ntokens: int,
    hidden: int,
    experts_per_rank: int,
    topk: int,
    device: torch.device,
    iters: int,
    warmup: int,
    nranks: int,
    smlimit: int = 0,
    group=None,
    cudagraph: int = 10000,
    routing_alpha: float = 0.0,
) -> BenchResult:
    """Benchmark UB-X a2av_token_bf16_mxfp8 (MoE token dispatch)."""
    from ubx import SymmAllocator
    from ubx.ops import compute_token_offsets

    if group is None:
        group = dist.group.WORLD

    rank = dist.get_rank(group)
    total_experts = nranks * experts_per_rank

    # Build routing matrix (same on all ranks)
    routing = _make_routing(ntokens, total_experts, topk, device, alpha=routing_alpha)
    token_offsets, max_tokens_per_rank, _, _ = compute_token_offsets(
        routing, experts_per_rank, rank, nranks,
    )

    local_ntokens = ntokens // nranks

    # Size the pool: output is mxfp8 blocked tensor, input is bf16
    # mxfp8 output: data = max_tokens_per_rank * hidden bytes (fp8)
    #               scales = max_tokens_per_rank * (hidden // 32) bytes
    output_data_bytes = max_tokens_per_rank * hidden
    output_scale_bytes = max_tokens_per_rank * (hidden // 32)
    # Align data region to 4096
    output_data_aligned = ((output_data_bytes + 4095) // 4096) * 4096
    output_total = output_data_aligned + output_scale_bytes

    # Pool needs space for output tensor on each rank + overhead
    pool_size = max(output_total * 6, 64 * 1024 * 1024)

    if cudagraph > 0:
        os.environ["UBX_GRAPH_POOL_SHARE"] = "0.5"
        pool_size = max(pool_size, output_total * 12)
    else:
        os.environ["UBX_GRAPH_POOL_SHARE"] = "0.1"

    allocator = SymmAllocator(pool_size, device, group)

    # Input: bf16 tokens (not in symmetric memory — regular GPU tensor)
    tokens_bf16 = torch.randn(local_ntokens, hidden, dtype=torch.bfloat16, device=device)

    # Output: mxfp8 blocked SymmTensor
    output = allocator.create_tensor(
        [max_tokens_per_rank, hidden], torch.float8_e4m3fn, blocked="mxfp8",
    )

    def op_fn():
        allocator.a2av_token_bf16_mxfp8(
            tokens_bf16, token_offsets, experts_per_rank, output, smlimit=smlimit,
        )

    if cudagraph > 0:
        for _ in range(warmup):
            op_fn()
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
                    op_fn()
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

    # Each token is sent to topk experts; actual wire bytes = tokens * topk * hidden * fp8_size
    # Use fp8 output bytes (what's actually written over NVLink) for bandwidth
    size_bytes = local_ntokens * topk * hidden  # fp8 = 1 byte per element
    algbw, busbw = compute_bandwidth(size_bytes, time_us, nranks, "a2av_mxfp8")

    return BenchResult(
        size_bytes=size_bytes, count=local_ntokens * topk * hidden,
        dtype="bf16→mxfp8", redop="none",
        time_us=time_us, algbw_gbs=algbw, busbw_gbs=busbw,
    )
