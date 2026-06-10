"""AllToAll-v token dispatch benchmark — multi-implementation comparison.

Compares MoE token-dispatch latency/bandwidth across:
  - ubx_bf16    : UBX bf16 -> bf16 dispatch (Phase 3)
  - ubx_mxfp8   : UBX bf16 -> mxfp8 dispatch (existing kernel)
  - nccl_ep_ll  : NCCL EP low-latency mode
  - nccl_ep_ht  : NCCL EP high-throughput mode

Headline metric is median dispatch latency (us). Bandwidth is reported
per-backend using each backend's own wire payload size (mxfp8 ~1.03 B/elem,
bf16 = 2 B/elem), so latency is the apples-to-apples comparison.

NCCL EP backends require building libnccl_ep from
github.com/NVIDIA/nccl/contrib/nccl_ep and exporting NCCL_EP_PATH to point at
the cloned `nccl` repo (we expect $NCCL_EP_PATH/contrib/nccl_ep/python on
sys.path). NCCL EP itself is bootstrapped without MPI: rank 0 calls
nccl.ncclGetUniqueId() and broadcasts the bytes via torch.distributed; every
rank then builds a fresh ncclComm_t alongside torch's existing PG.
"""

from __future__ import annotations

import ctypes
import gc
import os
import sys
import time

import torch
import torch.distributed as dist

from ..report import BenchResult, compute_bandwidth


_NCCL_EP_PATH = os.environ.get("NCCL_EP_PATH")
if _NCCL_EP_PATH:
    sys.path.insert(0, os.path.join(_NCCL_EP_PATH, "contrib", "nccl_ep", "python"))

try:
    from nccl_ep.nccl_wrapper import (  # type: ignore
        NCCLLibrary,
        ncclDataTypeEnum,
        ncclEpAlgorithm_t,
        ncclEpAllocFn_t,
        ncclEpDispatchConfig_t,
        ncclEpFreeFn_t,
        ncclEpGroupConfig_t,
        ncclEpTensorTag_t,
        ncclNDTensor_t,
        ncclUniqueId,
    )
    _NCCL_EP_OK = True
except ImportError:
    _NCCL_EP_OK = False


_CUDA_MEMCPY_D2D = 3
_NCCL_EP_AUTO = 0

# Module-level cache so we don't pay communicator setup per ntokens step.
_NCCL_LIB = None
_CUDA_RT = None
_ALLOC_FN = None
_FREE_FN = None
_NCCL_COMM_CACHE: dict = {}  # keyed by (world_size, rank)


def _make_routing(ntokens: int, total_experts: int, topk: int,
                  device: torch.device, alpha: float = 0.0,
                  seed: int = 0) -> torch.Tensor:
    """Build a [ntokens, total_experts] uint8 routing matrix.

    alpha == 0.0  -> deterministic round-robin (uniform load per expert).
    alpha  > 0.0  -> Zipfian: expert at popularity rank i gets weight
                     proportional to 1/(i+1)^alpha; the popularity-rank
                     ordering is a fixed permutation seeded by `seed`, so
                     all ranks see the same matrix.

    Each token gets `topk` distinct experts (sampled without replacement
    from the categorical distribution).
    """
    if alpha == 0.0:
        routing = torch.zeros(ntokens, total_experts, dtype=torch.uint8, device=device)
        for t in range(ntokens):
            for k in range(topk):
                e = (t * topk + k) % total_experts
                routing[t, e] = 1
        return routing

    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    rank_perm = torch.randperm(total_experts, generator=g)
    weights = (torch.arange(total_experts, dtype=torch.float64) + 1).pow(-alpha)
    weights = weights / weights.sum()
    perm_weights = torch.empty_like(weights)
    perm_weights[rank_perm] = weights
    perm_weights = perm_weights.to(torch.float32)

    g_sample = torch.Generator(device="cpu")
    g_sample.manual_seed(seed + 1)
    chosen = torch.multinomial(
        perm_weights.expand(ntokens, -1).contiguous(),
        topk, replacement=False, generator=g_sample,
    )  # [ntokens, topk] int64

    routing = torch.zeros(ntokens, total_experts, dtype=torch.uint8, device=device)
    routing.scatter_(1, chosen.to(device), 1)
    return routing


# ---------------------------------------------------------------------------
# UBX rows
# ---------------------------------------------------------------------------

def bench_a2av_dispatch_ubx_mxfp8(
    ntokens, hidden, experts_per_rank, topk,
    device, iters, warmup, nranks,
    smlimit=0, group=None, cudagraph=10000, routing_alpha: float = 0.0,
) -> BenchResult:
    """UBX bf16 -> mxfp8 dispatch row. Wraps existing bench_a2av_mxfp8_ubx."""
    from .a2av_mxfp8 import bench_a2av_mxfp8_ubx
    return bench_a2av_mxfp8_ubx(
        ntokens, hidden, experts_per_rank, topk,
        device, iters, warmup, nranks,
        smlimit=smlimit, group=group, cudagraph=cudagraph,
        routing_alpha=routing_alpha,
    )


def bench_a2av_dispatch_ubx_bf16(
    ntokens, hidden, experts_per_rank, topk,
    device, iters, warmup, nranks,
    smlimit=0, group=None, cudagraph=10000, routing_alpha: float = 0.0,
) -> BenchResult:
    """UBX bf16 -> bf16 dispatch row.

    Mirrors bench_a2av_mxfp8_ubx but allocates a plain bf16 SymmTensor (no
    blocked='mxfp8') and calls SymmAllocator.a2av_token_bf16_bf16.
    """
    from ubx import SymmAllocator
    from ubx.ops import compute_token_offsets

    if group is None:
        group = dist.group.WORLD

    rank = dist.get_rank(group)
    total_experts = nranks * experts_per_rank

    routing = _make_routing(ntokens, total_experts, topk, device, alpha=routing_alpha)
    token_offsets, max_tokens_per_rank, _, _ = compute_token_offsets(
        routing, experts_per_rank, rank, nranks,
    )

    local_ntokens = ntokens // nranks

    # Output: plain bf16 SymmTensor [max_tokens_per_rank, hidden]. 2 B/elem,
    # no scales region.
    output_bytes = max_tokens_per_rank * hidden * 2
    output_aligned = ((output_bytes + 4095) // 4096) * 4096

    pool_size = max(output_aligned * 6, 64 * 1024 * 1024)
    if cudagraph > 0:
        os.environ["UBX_GRAPH_POOL_SHARE"] = "0.5"
        pool_size = max(pool_size, output_aligned * 12)
    else:
        os.environ["UBX_GRAPH_POOL_SHARE"] = "0.1"

    allocator = SymmAllocator(pool_size, device, group)

    tokens_bf16 = torch.randn(local_ntokens, hidden, dtype=torch.bfloat16, device=device)
    output = allocator.create_tensor([max_tokens_per_rank, hidden], torch.bfloat16)

    def op_fn():
        allocator.a2av_token_bf16_bf16(
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

    # Wire payload: each rank sends local_ntokens * topk tokens to remote
    # ranks; payload per element is bf16 (2 B).
    size_bytes = local_ntokens * topk * hidden * 2
    algbw, busbw = compute_bandwidth(size_bytes, time_us, nranks, "a2av_dispatch")

    return BenchResult(
        size_bytes=size_bytes,
        count=local_ntokens * topk * hidden,
        dtype="bf16->bf16",
        redop="ubx_bf16",
        time_us=time_us, algbw_gbs=algbw, busbw_gbs=busbw,
    )


def bench_a2av_dispatch_ubx_bf16_topk(
    ntokens, hidden, experts_per_rank, topk,
    device, iters, warmup, nranks,
    smlimit=0, group=None, cudagraph=10000, routing_alpha: float = 0.0,
) -> BenchResult:
    """UBX bf16 -> bf16 dispatch via the top-K LUT kernel
    (``ubx_kernel_a2av_token_bf16_bf16_topk``).

    Mirrors bench_a2av_dispatch_ubx_bf16 but precomputes the
    [ntokens, topk_max] LUT with ubx.ops.compute_dispatch_topk_map and
    feeds the topk launcher. Lets us see whether the K-loop replaces the
    96 %-dead-check total_experts loop in the headline number.
    """
    from ubx import SymmAllocator
    from ubx.ops import compute_token_offsets, compute_dispatch_topk_map

    if group is None:
        group = dist.group.WORLD

    rank = dist.get_rank(group)
    total_experts = nranks * experts_per_rank

    routing = _make_routing(ntokens, total_experts, topk, device, alpha=routing_alpha)
    token_offsets, max_tokens_per_rank, _, _ = compute_token_offsets(
        routing, experts_per_rank, rank, nranks,
    )
    topk_expert, topk_slot, _ = compute_dispatch_topk_map(
        routing, token_offsets, experts_per_rank, rank, nranks,
    )

    local_ntokens = ntokens // nranks
    output_bytes  = max_tokens_per_rank * hidden * 2
    output_aligned = ((output_bytes + 4095) // 4096) * 4096
    pool_size = max(output_aligned * 6, 64 * 1024 * 1024)
    if cudagraph > 0:
        os.environ["UBX_GRAPH_POOL_SHARE"] = "0.5"
        pool_size = max(pool_size, output_aligned * 12)
    else:
        os.environ["UBX_GRAPH_POOL_SHARE"] = "0.1"

    allocator = SymmAllocator(pool_size, device, group)
    tokens_bf16 = torch.randn(local_ntokens, hidden, dtype=torch.bfloat16, device=device)
    output = allocator.create_tensor([max_tokens_per_rank, hidden], torch.bfloat16)

    def op_fn():
        allocator.a2av_token_bf16_bf16_topk(
            tokens_bf16, topk_expert, topk_slot, experts_per_rank,
            output, smlimit=smlimit,
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
    size_bytes = local_ntokens * topk * hidden * 2
    algbw, busbw = compute_bandwidth(size_bytes, time_us, nranks, "a2av_dispatch")
    return BenchResult(
        size_bytes=size_bytes,
        count=local_ntokens * topk * hidden,
        dtype="bf16->bf16",
        redop="ubx_bf16_topk",
        time_us=time_us, algbw_gbs=algbw, busbw_gbs=busbw,
    )


# ---------------------------------------------------------------------------
# NCCL EP support
# ---------------------------------------------------------------------------

def _skipped(implementation: str, ntokens: int, reason: str = "SKIPPED") -> BenchResult:
    return BenchResult(
        size_bytes=0, count=ntokens, dtype=reason, redop=implementation,
        time_us=float("nan"), algbw_gbs=0.0, busbw_gbs=0.0, errors=0,
    )


def _get_nccl_lib():
    """Initialise the NCCL EP library + ctypes CUDA runtime + alloc/free callbacks.

    Cached at module level — alloc/free callbacks must outlive every group
    (Python would gc them otherwise and the C side would hold dangling fn ptrs).
    """
    global _NCCL_LIB, _CUDA_RT, _ALLOC_FN, _FREE_FN
    if _NCCL_LIB is not None:
        return _NCCL_LIB

    _NCCL_LIB = NCCLLibrary()

    _CUDA_RT = ctypes.CDLL("libcudart.so")
    _CUDA_RT.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
    _CUDA_RT.cudaMalloc.restype = ctypes.c_int
    _CUDA_RT.cudaFree.argtypes = [ctypes.c_void_p]
    _CUDA_RT.cudaFree.restype = ctypes.c_int
    _CUDA_RT.cudaMemcpyAsync.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p,
    ]
    _CUDA_RT.cudaMemcpyAsync.restype = ctypes.c_int

    @ncclEpAllocFn_t
    def _alloc(ptr, size):
        return _CUDA_RT.cudaMalloc(ptr, size)

    @ncclEpFreeFn_t
    def _free(ptr):
        return _CUDA_RT.cudaFree(ptr)

    _ALLOC_FN = _alloc
    _FREE_FN = _free
    return _NCCL_LIB


def _bootstrap_comm(world_size: int, rank: int, group=None):
    """Build (and cache) a fresh ncclComm_t via torch.distributed broadcast."""
    if (world_size, rank) in _NCCL_COMM_CACHE:
        return _NCCL_COMM_CACHE[(world_size, rank)]

    nccl = _get_nccl_lib()
    if rank == 0:
        unique_id = nccl.ncclGetUniqueId()
    else:
        unique_id = ncclUniqueId()

    # Marshall the 128-byte unique_id through torch.distributed.
    if rank == 0:
        id_list = list(bytes(unique_id.internal))
    else:
        id_list = [0] * 128
    buf = torch.tensor(id_list, dtype=torch.uint8, device="cuda")
    dist.broadcast(buf, src=0, group=group)
    out = bytes(buf.cpu().tolist())
    for i in range(128):
        unique_id.internal[i] = out[i]

    comm = nccl.ncclCommInitRank(world_size, unique_id, rank)
    _NCCL_COMM_CACHE[(world_size, rank)] = comm
    return comm


def _tensor_create(nccl, ep_group, ndim, dtype, tag, data, *sizes):
    """Wrap ncclEpTensorCreate (mirrors helper in upstream ep_test.py)."""
    tensor = ncclNDTensor_t()
    padded = list(sizes) + [1] * (5 - len(sizes))
    nccl.NCCL_CHECK(nccl._funcs["ncclEpTensorCreate"](
        ep_group, ctypes.byref(tensor),
        ctypes.c_uint(ndim), ctypes.c_int(dtype), ctypes.c_int(tag),
        data,
        ctypes.c_uint(padded[0]), ctypes.c_uint(padded[1]),
        ctypes.c_uint(padded[2]), ctypes.c_uint(padded[3]),
        ctypes.c_uint(padded[4]),
    ))
    return tensor


def _tensor_destroy(nccl, ep_group, tensor):
    nccl.NCCL_CHECK(nccl._funcs["ncclEpTensorDestroy"](ep_group, tensor))


def _tensor_data(nccl, tensor) -> ctypes.c_void_p:
    data = ctypes.c_void_p()
    nccl.NCCL_CHECK(nccl._funcs["ncclEpTensorGetData"](tensor, ctypes.byref(data)))
    return data


def bench_a2av_dispatch_nccl_ep(
    ntokens, hidden, experts_per_rank, topk,
    device, iters, warmup, nranks,
    mode: str = "ll", group=None, cudagraph: int = 10000,
    routing_alpha: float = 0.0,
) -> BenchResult:
    """NCCL EP dispatch row (mode='ll' or 'ht'), single NVLink domain.

    Both modes are timed on the same NVLink-only domain (no GIN/RDMA exit
    even for HT — set NCCL_GIN_TYPE=0 in the launch env on multi-node
    setups to keep HT inside the LSA).

    Cudagraph mode is best-effort: if upstream ncclEpDispatch isn't capturable
    on the supplied stream we fall back to eager timing and tag the result.
    """
    if not _NCCL_EP_OK:
        return _skipped(f"nccl_ep_{mode}", ntokens, reason="NO-NCCL_EP")
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

    # Use a torch-managed CUDA stream so graph capture can see EP kernels.
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
    ep_group = nccl.ncclEpCreateGroup(comm, cfg, stream_ptr, _ALLOC_FN, _FREE_FN)

    # ---- topk_idx tensor [local_ntokens, topk] int64 ----------------------
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
    _CUDA_RT.cudaMemcpyAsync(
        _tensor_data(nccl, topk_idx_t), ctypes.c_void_p(expert_ids.data_ptr()),
        local_ntokens * topk * 8, _CUDA_MEMCPY_D2D, stream_ptr,
    )

    # ---- handle (handle creation reads topk_idx) --------------------------
    ep_handle = nccl.ncclEpCreateHandle(ep_group, topk_idx_t, None, stream_ptr)
    ep_stream.synchronize()

    # ---- input tokens [local_ntokens, hidden] bf16 -----------------------
    input_t = _tensor_create(
        nccl, ep_group, 2, ncclDataTypeEnum.ncclBfloat16,
        ncclEpTensorTag_t.NCCL_EP_TENSOR_TAG_TOKENS,
        None, local_ntokens, hidden,
    )
    tokens_bf16 = torch.randn(local_ntokens, hidden, dtype=torch.bfloat16, device=device)
    _CUDA_RT.cudaMemcpyAsync(
        _tensor_data(nccl, input_t), ctypes.c_void_p(tokens_bf16.data_ptr()),
        local_ntokens * hidden * 2, _CUDA_MEMCPY_D2D, stream_ptr,
    )

    # ---- output tokens (shape depends on algorithm) ----------------------
    if is_ll:
        output_t = _tensor_create(
            nccl, ep_group, 3, ncclDataTypeEnum.ncclBfloat16,
            ncclEpTensorTag_t.NCCL_EP_TENSOR_TAG_TOKENS,
            None, num_local_experts, local_ntokens * nranks, hidden,
        )
        recv_count_t = _tensor_create(
            nccl, ep_group, 1, ncclDataTypeEnum.ncclInt32,
            ncclEpTensorTag_t.NCCL_EP_TENSOR_TAG_RECV_EXPERT_COUNTER_DEVICE,
            None, num_local_experts,
        )
        topk_weights_t = None
        out_tw_t = None
        out_ti_t = None
    else:
        num_recv_tokens = local_ntokens * num_local_experts
        output_t = _tensor_create(
            nccl, ep_group, 2, ncclDataTypeEnum.ncclBfloat16,
            ncclEpTensorTag_t.NCCL_EP_TENSOR_TAG_TOKENS,
            None, num_recv_tokens, hidden,
        )
        topk_weights_t = _tensor_create(
            nccl, ep_group, 2, ncclDataTypeEnum.ncclFloat32,
            ncclEpTensorTag_t.NCCL_EP_TENSOR_TAG_TOPK_WEIGHTS,
            None, local_ntokens, topk,
        )
        tw = torch.full((local_ntokens, topk), 1.0 / topk,
                        dtype=torch.float32, device=device)
        _CUDA_RT.cudaMemcpyAsync(
            _tensor_data(nccl, topk_weights_t), ctypes.c_void_p(tw.data_ptr()),
            local_ntokens * topk * 4, _CUDA_MEMCPY_D2D, stream_ptr,
        )
        out_tw_t = _tensor_create(
            nccl, ep_group, 2, ncclDataTypeEnum.ncclFloat32,
            ncclEpTensorTag_t.NCCL_EP_TENSOR_TAG_TOPK_WEIGHTS,
            None, num_recv_tokens, topk,
        )
        out_ti_t = _tensor_create(
            nccl, ep_group, 2, ncclDataTypeEnum.ncclInt64,
            ncclEpTensorTag_t.NCCL_EP_TENSOR_TAG_TOPK_IDX,
            None, num_recv_tokens, topk,
        )
        recv_count_t = None

    ep_stream.synchronize()

    # ---- assemble dispatch arg arrays ------------------------------------
    num_inputs = 1 if is_ll else 3
    num_outputs = 1 if is_ll else 3
    num_local = 1 if is_ll else 0

    inputs_arr = (ncclNDTensor_t * num_inputs)()
    inputs_arr[0] = input_t
    if not is_ll:
        inputs_arr[1] = topk_weights_t
        inputs_arr[2] = topk_idx_t

    outputs_arr = (ncclNDTensor_t * num_outputs)()
    outputs_arr[0] = output_t
    if not is_ll:
        outputs_arr[1] = out_tw_t
        outputs_arr[2] = out_ti_t

    local_arr = (ncclNDTensor_t * max(num_local, 1))()
    if is_ll:
        local_arr[0] = recv_count_t

    inputs_p = ctypes.cast(inputs_arr, ctypes.POINTER(ncclNDTensor_t))
    outputs_p = ctypes.cast(outputs_arr, ctypes.POINTER(ncclNDTensor_t))
    local_p = ctypes.cast(local_arr, ctypes.POINTER(ncclNDTensor_t)) if num_local else None

    dispatch_cfg = ncclEpDispatchConfig_t()
    dispatch_cfg.round_scales = 0

    def op_fn():
        nccl.ncclEpDispatch(
            ep_handle, inputs_p, num_inputs, outputs_p, num_outputs,
            local_p, num_local, 0, dispatch_cfg, stream_ptr,
        )
        nccl.ncclEpComplete(ep_handle, None, stream_ptr)

    # ---- timing loop -----------------------------------------------------
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
                    # Capture stream must match the stream NCCL EP launches on.
                    cap_stream_ptr = ctypes.c_void_p(cap_stream.cuda_stream)
                    for _ in range(cudagraph):
                        nccl.ncclEpDispatch(
                            ep_handle, inputs_p, num_inputs, outputs_p, num_outputs,
                            local_p, num_local, 0, dispatch_cfg, cap_stream_ptr,
                        )
                        nccl.ncclEpComplete(ep_handle, None, cap_stream_ptr)
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
    except Exception as e:  # noqa: BLE001
        # Cudagraph not supported by upstream → fall back to eager for this point.
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
            _cleanup_nccl_ep_state(
                nccl, ep_group, ep_handle, stream_ptr,
                topk_idx_t, input_t, output_t,
                recv_count_t, topk_weights_t, out_tw_t, out_ti_t,
            )
            raise

    times.sort()
    time_us = times[len(times) // 2]

    # ---- cleanup ----------------------------------------------------------
    _cleanup_nccl_ep_state(
        nccl, ep_group, ep_handle, stream_ptr,
        topk_idx_t, input_t, output_t,
        recv_count_t, topk_weights_t, out_tw_t, out_ti_t,
    )

    # Wire payload (bf16 = 2 B/elem, K writes per token to remote ranks).
    size_bytes = local_ntokens * topk * hidden * 2
    algbw, busbw = compute_bandwidth(size_bytes, time_us, nranks, "a2av_dispatch")

    return BenchResult(
        size_bytes=size_bytes,
        count=local_ntokens * topk * hidden,
        dtype=fallback_dtype,
        redop=f"nccl_ep_{mode}",
        time_us=time_us, algbw_gbs=algbw, busbw_gbs=busbw,
    )


def _cleanup_nccl_ep_state(nccl, ep_group, ep_handle, stream_ptr,
                           topk_idx_t, input_t, output_t,
                           recv_count_t, topk_weights_t, out_tw_t, out_ti_t):
    """Tear down everything created in bench_a2av_dispatch_nccl_ep.

    Order matters: tensor destroys before ncclEpGroupDestroy because the
    group's free_fn is consulted by tensor destroy.
    """
    if input_t is not None:
        _tensor_destroy(nccl, ep_group, input_t)
    if output_t is not None:
        _tensor_destroy(nccl, ep_group, output_t)
    if recv_count_t is not None:
        _tensor_destroy(nccl, ep_group, recv_count_t)
    if topk_weights_t is not None:
        _tensor_destroy(nccl, ep_group, topk_weights_t)
    if out_tw_t is not None:
        _tensor_destroy(nccl, ep_group, out_tw_t)
    if out_ti_t is not None:
        _tensor_destroy(nccl, ep_group, out_ti_t)
    if topk_idx_t is not None:
        _tensor_destroy(nccl, ep_group, topk_idx_t)
    if ep_handle is not None:
        nccl.ncclEpHandleDestroy(ep_handle)
    if ep_group is not None:
        nccl.ncclEpGroupDestroy(ep_group, stream_ptr)
