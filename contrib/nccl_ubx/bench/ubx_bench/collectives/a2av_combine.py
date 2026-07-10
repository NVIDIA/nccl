"""Benchmark for the combine collective (reverse of a2av_token dispatch).

Compares per-backend median combine latency:
  - ubx_bf16    : UBX bf16-wire combine (allocator.combine_bf16_bf16)
  - ubx_mxfp8   : UBX mxfp8-wire combine (allocator.combine_mxfp8_bf16)
  - nccl_ep_ll  : NCCL EP low-latency mode (ncclEpCombine)
  - nccl_ep_ht  : NCCL EP high-throughput mode (ncclEpCombine)

Setup runs dispatch once to populate symmetric/EP-group expert outputs
(identity-FFN assumption: combine result = top-K weighted sum of original
tokens). Only the combine call itself is timed.
"""

from __future__ import annotations

import ctypes
import gc
import os
import time

import torch
import torch.distributed as dist

from ..report import BenchResult, compute_bandwidth
from . import a2av_token_dispatch as _td
from .a2av_token_dispatch import (
    _make_routing,
    _NCCL_EP_OK,
    _get_nccl_lib,
    _bootstrap_comm,
    _tensor_create,
    _tensor_destroy,
    _tensor_data,
    _skipped,
    _CUDA_MEMCPY_D2D,
    _NCCL_EP_AUTO,
)
if _NCCL_EP_OK:
    from nccl_ep.nccl_wrapper import (  # type: ignore
        ncclDataTypeEnum,
        ncclEpAlgorithm_t,
        ncclEpGroupConfig_t,
        ncclEpTensorTag_t,
        ncclNDTensor_t,
    )


def _bench_combine_ubx(
    ntokens, hidden, experts_per_rank, topk,
    device, iters, warmup, nranks,
    wire: str,
    smlimit=0, group=None, cudagraph=0, routing_alpha: float = 0.0,
    sync: bool = True,
    kernel: str = "auto",
) -> BenchResult:
    """Run UBX combine bench. wire ∈ {'bf16', 'mxfp8'}.

    sync=True (default): single kernel does Phase 1 + barrier + Phase 2.
    sync=False: split — main kernel does Phase 1 only, combine_wait runs
                Phase 2. Two kernel launches per call.
    kernel='lamport': bf16 wire only — Lamport-poll combine with triple-buffered
                      pre-poisoned temp. No cross-rank barrier in steady state.
                      Ignored (falls back to default) for wire='mxfp8'.
    """
    use_lamport_push = (kernel == "lamport_push" and wire == "bf16")
    use_push         = (kernel == "push"         and wire == "bf16")
    from ubx import SymmAllocator
    from ubx.ops import compute_token_offsets

    assert wire in ("bf16", "mxfp8"), f"unsupported wire format: {wire}"

    if group is None:
        group = dist.group.WORLD

    rank = dist.get_rank(group)
    total_experts = nranks * experts_per_rank
    local_ntokens = ntokens // nranks

    routing = _make_routing(ntokens, total_experts, topk, device,
                            alpha=routing_alpha)
    token_offsets, max_tokens_per_rank, _, expert_offsets = compute_token_offsets(
        routing, experts_per_rank, rank, nranks,
    )
    inverse_map = None
    topk_idx_t = None
    if use_lamport_push or use_push:
        from ubx import compute_combine_push_map
        inverse_map, topk_idx_t, _ = compute_combine_push_map(
            routing, experts_per_rank, rank, nranks,
        )
    # Actual received-token count for this rank (≤ max_tokens_per_rank,
    # which is the symmetric ceiling). Phase 1 iterates n_recv slots, not
    # the inflated max under skewed routing.
    n_recv = int(expert_offsets[-1].item())

    # Pool sizing: dispatch output (bf16, [max_tpr, hidden]) + combine temp
    # (bf16 or mxfp8 [max_tpr, hidden]) + REG0 + alignment headroom.
    bytes_dispatch = max_tokens_per_rank * hidden * 2  # bf16
    bytes_temp = (max_tokens_per_rank * hidden        # fp8 1B
                  + max_tokens_per_rank * (hidden // 32))  # E8M0 scales
    if wire == "bf16":
        bytes_temp = max_tokens_per_rank * hidden * 2
    pool_size = max(8 * (bytes_dispatch + bytes_temp), 64 * 1024 * 1024)
    if cudagraph > 0:
        os.environ["UBX_GRAPH_POOL_SHARE"] = "0.5"
        pool_size = max(pool_size, 16 * (bytes_dispatch + bytes_temp))
    else:
        os.environ["UBX_GRAPH_POOL_SHARE"] = "0.1"

    allocator = SymmAllocator(pool_size, device, group)

    # ---- Setup: run dispatch once to populate "expert outputs" ----
    tokens_bf16 = torch.randn(local_ntokens, hidden,
                              dtype=torch.bfloat16, device=device)
    dispatch_out = allocator.create_tensor(
        [max_tokens_per_rank, hidden], torch.bfloat16)
    allocator.a2av_token_bf16_bf16(
        tokens_bf16, token_offsets, experts_per_rank, dispatch_out)
    torch.cuda.synchronize()

    # Materialize as a regular tensor — combine API takes torch.Tensor (not
    # SymmTensor) for inputs. Detach + clone so the input lives outside the
    # symm pool (matches the user-facing API contract: caller passes a normal
    # torch.Tensor of FFN outputs).
    #
    # We pass the FULL [max_tokens_per_rank, hidden] tensor — NOT the first
    # n_recv rows — because compute_token_offsets places per-local-expert
    # data at SCATTERED slot ranges (`[e_local * max_slots,
    # e_local * max_slots + k)` per expert), not dense `[0, n_recv)`. Slicing
    # to `[:n_recv]` would drop data for experts ≥ 1 whenever
    # max_slots_per_expert > n_recv / experts_per_rank, which always happens
    # under skewed routing. The kernel's Phase 1 iterates the full range so
    # peers reading at any slot index in `token_offsets[t,e]` find data.
    expert_outputs = dispatch_out.detach().clone()
    n_recv = max_tokens_per_rank

    if use_push:
        def op_fn():
            allocator.combine_bf16_bf16_push(
                expert_outputs, inverse_map, topk_idx_t,
                experts_per_rank, max_tokens_per_rank, smlimit=smlimit)
    elif use_lamport_push:
        def op_fn():
            allocator.combine_bf16_bf16_lamport_push(
                expert_outputs, inverse_map, topk_idx_t,
                experts_per_rank, max_tokens_per_rank, smlimit=smlimit)
    elif wire == "bf16" and sync:
        def op_fn():
            allocator.combine_bf16_bf16(
                expert_outputs, token_offsets, experts_per_rank,
                max_tokens_per_rank, smlimit=smlimit)
    elif wire == "bf16":  # async: Phase 1 in main kernel, Phase 2 in combine_wait
        def op_fn():
            allocator.combine_bf16_bf16(
                expert_outputs, token_offsets, experts_per_rank,
                max_tokens_per_rank, smlimit=smlimit, sync=False)
            allocator.combine_wait()
    elif wire == "mxfp8" and sync:
        def op_fn():
            allocator.combine_mxfp8_bf16(
                expert_outputs, token_offsets, experts_per_rank,
                max_tokens_per_rank, smlimit=smlimit)
    else:  # mxfp8 async
        def op_fn():
            allocator.combine_mxfp8_bf16(
                expert_outputs, token_offsets, experts_per_rank,
                max_tokens_per_rank, smlimit=smlimit, sync=False)
            allocator.combine_wait()

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
            t0 = time.perf_counter()
            graph.replay()
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1e6 / cudagraph)
        del graph
    else:
        for _ in range(warmup):
            op_fn()
        torch.cuda.synchronize()

        times = []
        for _ in range(iters):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            op_fn()
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1e6)

    times.sort()
    time_us = times[len(times) // 2]

    # Wire payload (per-rank): combine moves the same data as dispatch in
    # reverse — local_ntokens * topk tokens-worth of contributions enter
    # this rank from peers.
    bytes_per_elem = 2 if wire == "bf16" else 1
    overhead_bytes = 0 if wire == "bf16" else (
        local_ntokens * topk * (hidden // 32))  # 1 byte E8M0 per block
    size_bytes = local_ntokens * topk * hidden * bytes_per_elem + overhead_bytes
    algbw, busbw = compute_bandwidth(size_bytes, time_us, nranks, "a2av_dispatch")

    if use_push:
        redop = f"ubx_{wire}_push"
    elif use_lamport_push:
        redop = f"ubx_{wire}_lamport_push"
    else:
        redop = f"ubx_{wire}" + ("_async" if not sync else "")
    return BenchResult(
        size_bytes=size_bytes,
        count=local_ntokens * topk * hidden,
        dtype=("bf16->bf16" if wire == "bf16" else "bf16->mxfp8->bf16"),
        redop=redop,
        time_us=time_us, algbw_gbs=algbw, busbw_gbs=busbw,
    )


def bench_a2av_combine_ubx_bf16(*args, **kwargs):
    return _bench_combine_ubx(*args, wire="bf16", **kwargs)


def bench_a2av_combine_ubx_mxfp8(*args, **kwargs):
    return _bench_combine_ubx(*args, wire="mxfp8", **kwargs)


def bench_a2av_combine_ubx_bf16_async(*args, **kwargs):
    return _bench_combine_ubx(*args, wire="bf16", sync=False, **kwargs)


def bench_a2av_combine_ubx_mxfp8_async(*args, **kwargs):
    return _bench_combine_ubx(*args, wire="mxfp8", sync=False, **kwargs)


# ---------------------------------------------------------------------------
# NCCL EP combine — uses the same ncclEpHandle as dispatch. We do dispatch once
# in setup to populate expert outputs, then time only ncclEpCombine.
# ---------------------------------------------------------------------------

def bench_a2av_combine_nccl_ep(
    ntokens, hidden, experts_per_rank, topk,
    device, iters, warmup, nranks,
    mode: str = "ll", group=None, cudagraph: int = 0,
    routing_alpha: float = 0.0,
) -> BenchResult:
    """NCCL EP combine row (mode='ll' or 'ht'), single NVLink domain.

    Setup mirrors bench_a2av_dispatch_nccl_ep through ncclEpDispatch
    (run once outside the timing loop). Then allocates combine output
    tensors (always 2D [local_ntokens, hidden] regardless of mode) and
    times ncclEpCombine() in eager or graph mode.
    """
    if not _NCCL_EP_OK:
        return _skipped(f"nccl_ep_{mode}_combine", ntokens, reason="NO-NCCL_EP")
    if group is None:
        group = dist.group.WORLD

    rank = dist.get_rank(group)
    local_ntokens = ntokens // nranks
    total_experts = nranks * experts_per_rank
    num_local_experts = experts_per_rank

    nccl = _get_nccl_lib()
    comm = _bootstrap_comm(nranks, rank, group=group)

    algorithm = (ncclEpAlgorithm_t.NCCL_EP_ALGO_LOW_LATENCY if mode == "ll"
                 else ncclEpAlgorithm_t.NCCL_EP_ALGO_HIGH_THROUGHPUT)
    is_ll = (mode == "ll")

    ep_stream = torch.cuda.Stream(device=device)
    stream_ptr = ctypes.c_void_p(ep_stream.cuda_stream)

    # ---- EP group ---------------------------------------------------------
    cfg = ncclEpGroupConfig_t()
    cfg.version = 1
    cfg.algorithm = algorithm
    cfg.num_experts = total_experts
    cfg.max_tokens_per_rank = local_ntokens
    cfg.token_size_bytes = hidden * 2
    cfg.rdma_buffer_size = _NCCL_EP_AUTO
    cfg.num_qp_per_rank = _NCCL_EP_AUTO
    cfg.num_channels = _NCCL_EP_AUTO
    ep_group = nccl.ncclEpCreateGroup(comm, cfg, stream_ptr,
                                       _td._ALLOC_FN, _td._FREE_FN)

    # ---- topk_idx + handle ------------------------------------------------
    topk_idx_t = _tensor_create(
        nccl, ep_group, 2, ncclDataTypeEnum.ncclInt64,
        ncclEpTensorTag_t.NCCL_EP_TENSOR_TAG_TOPK_IDX,
        None, local_ntokens, topk,
    )
    routing = _make_routing(ntokens, total_experts, topk, device, alpha=routing_alpha)
    routing_local = routing[rank * local_ntokens:(rank + 1) * local_ntokens]
    expert_ids = (routing_local.nonzero()[:, 1]
                  .view(local_ntokens, topk)
                  .to(torch.int64)
                  .contiguous())
    _td._CUDA_RT.cudaMemcpyAsync(
        _tensor_data(nccl, topk_idx_t), ctypes.c_void_p(expert_ids.data_ptr()),
        local_ntokens * topk * 8, _CUDA_MEMCPY_D2D, stream_ptr,
    )

    ep_handle = nccl.ncclEpCreateHandle(ep_group, topk_idx_t, None, stream_ptr)
    ep_stream.synchronize()

    # ---- dispatch tensors (used to populate expert outputs once) ---------
    input_t = _tensor_create(
        nccl, ep_group, 2, ncclDataTypeEnum.ncclBfloat16,
        ncclEpTensorTag_t.NCCL_EP_TENSOR_TAG_TOKENS,
        None, local_ntokens, hidden,
    )
    tokens_bf16 = torch.randn(local_ntokens, hidden,
                              dtype=torch.bfloat16, device=device)
    _td._CUDA_RT.cudaMemcpyAsync(
        _tensor_data(nccl, input_t), ctypes.c_void_p(tokens_bf16.data_ptr()),
        local_ntokens * hidden * 2, _CUDA_MEMCPY_D2D, stream_ptr,
    )

    # Dispatch output (= combine input). Shape depends on mode.
    if is_ll:
        expert_outputs_t = _tensor_create(
            nccl, ep_group, 3, ncclDataTypeEnum.ncclBfloat16,
            ncclEpTensorTag_t.NCCL_EP_TENSOR_TAG_TOKENS,
            None, num_local_experts, local_ntokens * nranks, hidden,
        )
        recv_count_t = _tensor_create(
            nccl, ep_group, 1, ncclDataTypeEnum.ncclInt32,
            ncclEpTensorTag_t.NCCL_EP_TENSOR_TAG_RECV_EXPERT_COUNTER_DEVICE,
            None, num_local_experts,
        )
        d_topk_weights_t = None
        d_out_tw_t = None
        d_out_ti_t = None
    else:
        num_recv_tokens = local_ntokens * num_local_experts
        expert_outputs_t = _tensor_create(
            nccl, ep_group, 2, ncclDataTypeEnum.ncclBfloat16,
            ncclEpTensorTag_t.NCCL_EP_TENSOR_TAG_TOKENS,
            None, num_recv_tokens, hidden,
        )
        d_topk_weights_t = _tensor_create(
            nccl, ep_group, 2, ncclDataTypeEnum.ncclFloat32,
            ncclEpTensorTag_t.NCCL_EP_TENSOR_TAG_TOPK_WEIGHTS,
            None, local_ntokens, topk,
        )
        tw = torch.full((local_ntokens, topk), 1.0 / topk,
                        dtype=torch.float32, device=device)
        _td._CUDA_RT.cudaMemcpyAsync(
            _tensor_data(nccl, d_topk_weights_t), ctypes.c_void_p(tw.data_ptr()),
            local_ntokens * topk * 4, _CUDA_MEMCPY_D2D, stream_ptr,
        )
        d_out_tw_t = _tensor_create(
            nccl, ep_group, 2, ncclDataTypeEnum.ncclFloat32,
            ncclEpTensorTag_t.NCCL_EP_TENSOR_TAG_TOPK_WEIGHTS,
            None, num_recv_tokens, topk,
        )
        d_out_ti_t = _tensor_create(
            nccl, ep_group, 2, ncclDataTypeEnum.ncclInt64,
            ncclEpTensorTag_t.NCCL_EP_TENSOR_TAG_TOPK_IDX,
            None, num_recv_tokens, topk,
        )
        recv_count_t = None

    # ---- combine output tensor (always 2D [local_ntokens, hidden]) -------
    combine_out_t = _tensor_create(
        nccl, ep_group, 2, ncclDataTypeEnum.ncclBfloat16,
        ncclEpTensorTag_t.NCCL_EP_TENSOR_TAG_TOKENS,
        None, local_ntokens, hidden,
    )

    # LL mode: optional TOP_K_WEIGHTS local tensor for the combine.
    if is_ll:
        c_topk_weights_t = _tensor_create(
            nccl, ep_group, 2, ncclDataTypeEnum.ncclFloat32,
            ncclEpTensorTag_t.NCCL_EP_TENSOR_TAG_TOPK_WEIGHTS,
            None, local_ntokens, topk,
        )
        tw_combine = torch.full((local_ntokens, topk), 1.0 / topk,
                                dtype=torch.float32, device=device)
        _td._CUDA_RT.cudaMemcpyAsync(
            _tensor_data(nccl, c_topk_weights_t),
            ctypes.c_void_p(tw_combine.data_ptr()),
            local_ntokens * topk * 4, _CUDA_MEMCPY_D2D, stream_ptr,
        )
    else:
        c_topk_weights_t = None

    ep_stream.synchronize()

    # ---- assemble dispatch arrays (used once) ----------------------------
    d_num_inputs  = 1 if is_ll else 3
    d_num_outputs = 1 if is_ll else 3
    d_num_local   = 1 if is_ll else 0
    d_inputs_arr  = (ncclNDTensor_t * d_num_inputs)()
    d_inputs_arr[0] = input_t
    if not is_ll:
        d_inputs_arr[1] = d_topk_weights_t
        d_inputs_arr[2] = topk_idx_t
    d_outputs_arr = (ncclNDTensor_t * d_num_outputs)()
    d_outputs_arr[0] = expert_outputs_t
    if not is_ll:
        d_outputs_arr[1] = d_out_tw_t
        d_outputs_arr[2] = d_out_ti_t
    d_local_arr = (ncclNDTensor_t * max(d_num_local, 1))()
    if is_ll:
        d_local_arr[0] = recv_count_t
    d_inputs_p  = ctypes.cast(d_inputs_arr,  ctypes.POINTER(ncclNDTensor_t))
    d_outputs_p = ctypes.cast(d_outputs_arr, ctypes.POINTER(ncclNDTensor_t))
    d_local_p   = ctypes.cast(d_local_arr,   ctypes.POINTER(ncclNDTensor_t)) \
                  if d_num_local else None

    from nccl_ep.nccl_wrapper import ncclEpDispatchConfig_t  # type: ignore
    dispatch_cfg = ncclEpDispatchConfig_t()
    dispatch_cfg.round_scales = 0

    # ---- assemble combine arrays (used N times) --------------------------
    c_inputs_arr = (ncclNDTensor_t * 1)()
    c_inputs_arr[0] = expert_outputs_t  # dispatch's output is combine's input
    c_outputs_arr = (ncclNDTensor_t * 1)()
    c_outputs_arr[0] = combine_out_t
    c_inputs_p  = ctypes.cast(c_inputs_arr,  ctypes.POINTER(ncclNDTensor_t))
    c_outputs_p = ctypes.cast(c_outputs_arr, ctypes.POINTER(ncclNDTensor_t))

    if is_ll:
        c_local_arr = (ncclNDTensor_t * 1)()
        c_local_arr[0] = c_topk_weights_t
        c_local_p = ctypes.cast(c_local_arr, ctypes.POINTER(ncclNDTensor_t))
        c_num_local = 1
    else:
        c_local_p = None
        c_num_local = 0

    # ---- prime expert outputs once (dispatch) ----------------------------
    # NB: NCCL EP HT dispatch is broken on single-node at v2.30.4-1 — the
    # underlying ncclEpDispatch call calls std::terminate, which Python
    # cannot catch. We omit nccl_ep_ht from the default backend list to
    # avoid the surprise crash; if the user explicitly opts in via
    # --backend nccl_ep_ht, the bench will abort here.
    with torch.cuda.stream(ep_stream):
        nccl.ncclEpDispatch(
            ep_handle, d_inputs_p, d_num_inputs, d_outputs_p, d_num_outputs,
            d_local_p, d_num_local, 0, dispatch_cfg, stream_ptr,
        )
        nccl.ncclEpComplete(ep_handle, None, stream_ptr)
    ep_stream.synchronize()

    # ---- combine op ------------------------------------------------------
    def op_fn():
        nccl.ncclEpCombine(
            ep_handle, c_inputs_p, 1, c_outputs_p, 1,
            c_local_p, c_num_local, 0, None, stream_ptr,
        )

    fallback_dtype = "bf16->bf16"
    times: list = []
    eager_fallback = False

    try:
        if cudagraph > 0:
            with torch.cuda.stream(ep_stream):
                for _ in range(warmup):
                    op_fn()
            ep_stream.synchronize()

            cap_stream = torch.cuda.Stream(device=device)
            cap_stream.wait_stream(ep_stream)
            gc_was_enabled = gc.isenabled()
            if gc_was_enabled:
                gc.disable()
            with torch.cuda.stream(cap_stream):
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=cap_stream):
                    cap_stream_ptr = ctypes.c_void_p(cap_stream.cuda_stream)
                    for _ in range(cudagraph):
                        nccl.ncclEpCombine(
                            ep_handle, c_inputs_p, 1, c_outputs_p, 1,
                            c_local_p, c_num_local, 0, None, cap_stream_ptr,
                        )
            if gc_was_enabled:
                gc.enable()
                gc.collect()
            torch.cuda.current_stream().wait_stream(cap_stream)

            for _ in range(warmup):
                graph.replay()
            torch.cuda.synchronize()
            for _ in range(iters):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                graph.replay()
                torch.cuda.synchronize()
                times.append((time.perf_counter() - t0) * 1e6 / cudagraph)
            del graph
        else:
            with torch.cuda.stream(ep_stream):
                for _ in range(warmup):
                    op_fn()
            ep_stream.synchronize()
            for _ in range(iters):
                ep_stream.synchronize()
                t0 = time.perf_counter()
                with torch.cuda.stream(ep_stream):
                    op_fn()
                ep_stream.synchronize()
                times.append((time.perf_counter() - t0) * 1e6)
    except Exception:
        if cudagraph > 0 and not eager_fallback:
            eager_fallback = True
            fallback_dtype = "bf16->bf16(eager)"
            times = []
            with torch.cuda.stream(ep_stream):
                for _ in range(warmup):
                    op_fn()
            ep_stream.synchronize()
            for _ in range(iters):
                ep_stream.synchronize()
                t0 = time.perf_counter()
                with torch.cuda.stream(ep_stream):
                    op_fn()
                ep_stream.synchronize()
                times.append((time.perf_counter() - t0) * 1e6)
        else:
            _cleanup_combine_state(
                nccl, ep_group, ep_handle, stream_ptr,
                topk_idx_t, input_t, expert_outputs_t,
                recv_count_t, d_topk_weights_t, d_out_tw_t, d_out_ti_t,
                combine_out_t, c_topk_weights_t,
            )
            raise

    times.sort()
    time_us = times[len(times) // 2]

    _cleanup_combine_state(
        nccl, ep_group, ep_handle, stream_ptr,
        topk_idx_t, input_t, expert_outputs_t,
        recv_count_t, d_topk_weights_t, d_out_tw_t, d_out_ti_t,
        combine_out_t, c_topk_weights_t,
    )

    # Wire payload mirrors dispatch (combine moves the same data in reverse).
    size_bytes = local_ntokens * topk * hidden * 2
    algbw, busbw = compute_bandwidth(size_bytes, time_us, nranks, "a2av_dispatch")

    return BenchResult(
        size_bytes=size_bytes,
        count=local_ntokens * topk * hidden,
        dtype=fallback_dtype,
        redop=f"nccl_ep_{mode}",
        time_us=time_us, algbw_gbs=algbw, busbw_gbs=busbw,
    )


def _cleanup_combine_state(nccl, ep_group, ep_handle, stream_ptr,
                            topk_idx_t, input_t, expert_outputs_t,
                            recv_count_t, d_topk_weights_t, d_out_tw_t, d_out_ti_t,
                            combine_out_t, c_topk_weights_t):
    """Tear down everything created in bench_a2av_combine_nccl_ep."""
    for t in (input_t, expert_outputs_t, recv_count_t,
              d_topk_weights_t, d_out_tw_t, d_out_ti_t,
              combine_out_t, c_topk_weights_t, topk_idx_t):
        if t is not None:
            _tensor_destroy(nccl, ep_group, t)
    if ep_handle is not None:
        nccl.ncclEpHandleDestroy(ep_handle)
    if ep_group is not None:
        nccl.ncclEpGroupDestroy(ep_group, stream_ptr)
