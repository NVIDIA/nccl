#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# See LICENSE.txt for more license information.

"""Python EP test — replicates ep_test.cu using the nccl_wrapper.py API.

Usage:
    mpirun -np <N> python ep_test.py [OPTIONS]

Options (identical to ep_test.cu):
    -a {ll,ht}                          Algorithm mode (default: ll)
    -m                                   Disable max_tokens_per_rank (HT only, not yet supported)
    -s {none,dispatch,combine,both}      Send-only mode (default: none)
    -c                                   Enable cached mode (HT only)
    -r                                   Enable random mode
    -t NUM                               Number of tokens (default: 50)
    -d NUM                               Hidden dimension size (default: 7168)
"""

import argparse
import ctypes
import os
import random
import struct
import sys

from mpi4py import MPI

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "python"))

from nccl_ep.nccl_wrapper import (
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

# ---------------------------------------------------------------------------
# CUDA runtime helpers (ctypes bindings for the small subset we need)
# ---------------------------------------------------------------------------

_cuda_rt = ctypes.CDLL("libcudart.so")

# Declare argtypes/restype for every CUDA runtime function we use.
# Without these, ctypes defaults to passing Python ints as 32-bit C ints,
# which silently truncates 64-bit device pointers — especially fatal
# inside CFUNCTYPE callbacks (the allocator/free callbacks).
_cuda_rt.cudaSetDevice.argtypes = [ctypes.c_int]
_cuda_rt.cudaSetDevice.restype = ctypes.c_int
_cuda_rt.cudaStreamCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
_cuda_rt.cudaStreamCreate.restype = ctypes.c_int
_cuda_rt.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
_cuda_rt.cudaMalloc.restype = ctypes.c_int
_cuda_rt.cudaFree.argtypes = [ctypes.c_void_p]
_cuda_rt.cudaFree.restype = ctypes.c_int
_cuda_rt.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
_cuda_rt.cudaMemcpy.restype = ctypes.c_int
_cuda_rt.cudaStreamSynchronize.argtypes = [ctypes.c_void_p]
_cuda_rt.cudaStreamSynchronize.restype = ctypes.c_int
_cuda_rt.cudaHostAlloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t, ctypes.c_uint]
_cuda_rt.cudaHostAlloc.restype = ctypes.c_int
_cuda_rt.cudaFreeHost.argtypes = [ctypes.c_void_p]
_cuda_rt.cudaFreeHost.restype = ctypes.c_int
_cuda_rt.cudaDeviceReset.argtypes = []
_cuda_rt.cudaDeviceReset.restype = ctypes.c_int

CUDA_MEMCPY_H2D = 1
CUDA_MEMCPY_D2H = 2
CUDA_MEMCPY_D2D = 3
CUDA_HOST_ALLOC_MAPPED = 2
NCCL_EP_AUTO = 0


def _cuda_check(ret, name="CUDA"):
    if ret != 0:
        raise RuntimeError(f"{name} failed with error code {ret}")


def cuda_set_device(dev):
    _cuda_check(_cuda_rt.cudaSetDevice(dev), "cudaSetDevice")


def cuda_stream_create():
    s = ctypes.c_void_p()
    _cuda_check(_cuda_rt.cudaStreamCreate(ctypes.byref(s)), "cudaStreamCreate")
    return s


def cuda_malloc(size):
    ptr = ctypes.c_void_p()
    _cuda_check(_cuda_rt.cudaMalloc(ctypes.byref(ptr), size), "cudaMalloc")
    return ptr


def cuda_free(ptr):
    _cuda_check(_cuda_rt.cudaFree(ptr), "cudaFree")


def cuda_memcpy(dst, src, size, kind):
    _cuda_check(_cuda_rt.cudaMemcpy(dst, src, size, kind), "cudaMemcpy")


def cuda_stream_synchronize(stream):
    _cuda_check(_cuda_rt.cudaStreamSynchronize(stream), "cudaStreamSynchronize")


def cuda_host_alloc(size):
    ptr = ctypes.c_void_p()
    _cuda_check(_cuda_rt.cudaHostAlloc(ctypes.byref(ptr), size, CUDA_HOST_ALLOC_MAPPED), "cudaHostAlloc")
    return ptr


def cuda_free_host(ptr):
    _cuda_check(_cuda_rt.cudaFreeHost(ptr), "cudaFreeHost")


def cuda_device_reset():
    _cuda_check(_cuda_rt.cudaDeviceReset(), "cudaDeviceReset")


# ---------------------------------------------------------------------------
# Tensor helpers (ncclEpTensor* not exposed as high-level wrapper methods)
# ---------------------------------------------------------------------------

def tensor_create(nccl, ep_group, ndim, datatype, tag, data, *sizes):
    tensor = ncclNDTensor_t()
    padded = list(sizes) + [1] * (5 - len(sizes))
    nccl.NCCL_CHECK(nccl._funcs["ncclEpTensorCreate"](
        ep_group, ctypes.byref(tensor),
        ctypes.c_uint(ndim), ctypes.c_int(datatype), ctypes.c_int(tag),
        data,
        ctypes.c_uint(padded[0]), ctypes.c_uint(padded[1]),
        ctypes.c_uint(padded[2]), ctypes.c_uint(padded[3]),
        ctypes.c_uint(padded[4]),
    ))
    return tensor


def tensor_destroy(nccl, ep_group, tensor):
    nccl.NCCL_CHECK(nccl._funcs["ncclEpTensorDestroy"](ep_group, tensor))


def tensor_get_data(nccl, tensor):
    data = ctypes.c_void_p()
    nccl.NCCL_CHECK(nccl._funcs["ncclEpTensorGetData"](tensor, ctypes.byref(data)))
    return data


# ---------------------------------------------------------------------------
# bfloat16 conversion (matches the C++ float_to_bf16 exactly)
# ---------------------------------------------------------------------------

def float_to_bf16(f):
    x = struct.unpack("I", struct.pack("f", f))[0]
    rounding_bias = 0x00007FFF + ((x >> 16) & 1)
    return ((x + rounding_bias) >> 16) & 0xFFFF


# ---------------------------------------------------------------------------
# Custom allocator callbacks (matching torchMalloc / torchFree in ep_test.cu)
# ---------------------------------------------------------------------------

@ncclEpAllocFn_t
def _alloc_fn(ptr, size):
    return _cuda_rt.cudaMalloc(ptr, size)


@ncclEpFreeFn_t
def _free_fn(ptr):
    # ptr arrives as a Python int from the CFUNCTYPE c_void_p parameter.
    # With argtypes set on cudaFree, ctypes auto-converts to c_void_p.
    return _cuda_rt.cudaFree(ptr)


# ---------------------------------------------------------------------------
# DJB2 hostname hash (matches getHostHash in ep_test.cu)
# ---------------------------------------------------------------------------

def _host_hash(name):
    h = 5381
    for c in name:
        h = ((h << 5) + h + ord(c)) & 0xFFFFFFFFFFFFFFFF
    return h


# ---------------------------------------------------------------------------
# Main test
# ---------------------------------------------------------------------------

def main():  # noqa: C901 — intentionally kept as a single function to mirror ep_test.cu
    mpi_comm = MPI.COMM_WORLD
    my_rank = mpi_comm.Get_rank()
    n_ranks = mpi_comm.Get_size()

    # -- argument parsing (same flags as the C++ test) ----------------------
    parser = argparse.ArgumentParser(description="EP Test (Python)")
    parser.add_argument("-a", choices=["ll", "ht"], default="ll", help="Algorithm mode")
    parser.add_argument("-m", action="store_true", help="Disable max_tokens_per_rank (HT only)")
    parser.add_argument("-s", choices=["none", "dispatch", "combine", "both"], default="none",
                        help="Send-only mode")
    parser.add_argument("-c", action="store_true", help="Enable cached mode (HT only)")
    parser.add_argument("-r", action="store_true", help="Enable random mode")
    parser.add_argument("-t", type=int, default=50, help="Number of tokens")
    parser.add_argument("-d", type=int, default=7168, help="Hidden dimension size")
    args = parser.parse_args()

    algorithm = (ncclEpAlgorithm_t.NCCL_EP_ALGO_LOW_LATENCY
                 if args.a == "ll" else ncclEpAlgorithm_t.NCCL_EP_ALGO_HIGH_THROUGHPUT)
    disable_max_tokens = args.m
    dispatch_send_only = args.s in ("dispatch", "both")
    combine_send_only = args.s in ("combine", "both")
    cached_mode = args.c
    random_mode = args.r
    num_tokens = args.t
    hidden = args.d

    if n_ranks not in (2, 4) and n_ranks % 8 != 0:
        if my_rank == 0:
            print("Error: nRanks must be 2, 4 or multiple of 8 for this test")
        sys.exit(1)

    if disable_max_tokens:
        if my_rank == 0:
            if algorithm != ncclEpAlgorithm_t.NCCL_EP_ALGO_HIGH_THROUGHPUT:
                print("Error: -m is only applicable to HT mode (-a ht)")
            else:
                print("Error: -m (NCCL_EP_AUTO for max_tokens_per_rank) is not yet supported.\n"
                      "       This feature will be available in a future release for HT mode.")
        sys.exit(1)

    # -- derived parameters -------------------------------------------------
    ELEMENTS_TESTED_PER_TOKEN = 10
    top_k = min(8, n_ranks)
    num_experts = min(256, top_k * n_ranks)
    num_local_experts = num_experts // n_ranks
    local_experts_start = num_local_experts * my_rank
    local_experts_end = local_experts_start + num_local_experts

    if num_experts % n_ranks != 0:
        if my_rank == 0:
            print(f"Error: num_experts ({num_experts}) must be divisible by nRanks ({n_ranks})")
        sys.exit(1)
    if top_k > num_local_experts:
        if my_rank == 0:
            print(f"Error: top_k ({top_k}) must be <= num_local_experts ({num_local_experts})")
        sys.exit(1)

    # -- calculate localRank via hostname hash (same as C++ test) -----------
    hostname = MPI.Get_processor_name().split(".")[0]
    my_hash = _host_hash(hostname)
    all_hashes = mpi_comm.allgather(my_hash)
    local_rank = sum(1 for p in range(my_rank) if all_hashes[p] == my_hash)

    # -- CUDA setup ---------------------------------------------------------
    cuda_set_device(local_rank)
    stream = cuda_stream_create()

    # -- NCCL setup ---------------------------------------------------------
    nccl = NCCLLibrary()

    if my_rank == 0:
        unique_id = nccl.ncclGetUniqueId()
    else:
        unique_id = ncclUniqueId()

    id_bytes = bytearray(unique_id.internal)
    id_bytes = mpi_comm.bcast(id_bytes, root=0)
    for i in range(128):
        unique_id.internal[i] = id_bytes[i]

    comm = nccl.ncclCommInitRank(n_ranks, unique_id, my_rank)

    # -- EP group -----------------------------------------------------------
    config = ncclEpGroupConfig_t()
    config.version = 1
    config.algorithm = algorithm
    config.num_experts = num_experts
    config.max_tokens_per_rank = NCCL_EP_AUTO if disable_max_tokens else num_tokens
    config.token_size_bytes = hidden * 2  # bfloat16
    config.rdma_buffer_size = NCCL_EP_AUTO
    config.num_qp_per_rank = NCCL_EP_AUTO
    config.num_channels = NCCL_EP_AUTO

    algorithm_name = "LOW_LATENCY" if algorithm == ncclEpAlgorithm_t.NCCL_EP_ALGO_LOW_LATENCY else "HIGH_THROUGHPUT"
    extra = " (no max_tokens_per_rank)" if disable_max_tokens else ""
    print(f"Rank {my_rank}: Testing ncclEpCreateGroup with algorithm: {algorithm_name}{extra}")

    ep_group = nccl.ncclEpCreateGroup(comm, config, stream, _alloc_fn, _free_fn)

    # -- topk_idx tensor [num_tokens, top_k] int64 --------------------------
    topk_idx = tensor_create(
        nccl, ep_group, 2, ncclDataTypeEnum.ncclInt64,
        ncclEpTensorTag_t.NCCL_EP_TENSOR_TAG_TOPK_IDX,
        None, num_tokens, top_k,
    )

    topk_idx_host = (ctypes.c_int64 * (num_tokens * top_k))()
    if random_mode:
        random.seed(my_rank + 42)
        for i in range(num_tokens):
            first_expert = random.randint(0, num_experts - 1)
            topk_idx_host[i * top_k] = first_expert
            for j in range(1, top_k):
                topk_idx_host[i * top_k + j] = (first_expert + j) % num_experts
        if my_rank == 0:
            print("Random mode enabled: first expert random, rest deterministic (no repetitions)")
    else:
        for i in range(num_tokens):
            for j in range(top_k):
                topk_idx_host[i * top_k + j] = (local_experts_end + j) % num_experts

    topk_idx_data = tensor_get_data(nccl, topk_idx)
    cuda_memcpy(topk_idx_data, topk_idx_host, num_tokens * top_k * 8, CUDA_MEMCPY_H2D)

    # -- recv_expert_counter for handle (only when disable_max_tokens) ------
    handle_local_tensors = []
    handle_recv_expert_counter = None
    recv_counter_host_ptr = None
    if disable_max_tokens:
        recv_counter_host_ptr = cuda_host_alloc(num_local_experts * 4)
        handle_recv_expert_counter = tensor_create(
            nccl, ep_group, 1, ncclDataTypeEnum.ncclInt32,
            ncclEpTensorTag_t.NCCL_EP_TENSOR_TAG_RECV_EXPERT_COUNTER_HOST,
            recv_counter_host_ptr, num_local_experts,
        )
        handle_local_tensors.append(handle_recv_expert_counter)

    # -- EP handle ----------------------------------------------------------
    print(f"Rank {my_rank}: Testing ncclEpCreateHandle")
    ep_handle = nccl.ncclEpCreateHandle(
        ep_group, topk_idx, None, stream,
        local_tensors=handle_local_tensors if handle_local_tensors else None,
    )
    cuda_stream_synchronize(stream)

    if disable_max_tokens:
        num_recv_tokens = nccl.ncclEpHandleGetNumRecvTokens(ep_handle)
    else:
        num_recv_tokens = config.max_tokens_per_rank * num_local_experts
    assert num_recv_tokens > 0

    # -- dispatch config ----------------------------------------------------
    dispatch_config = ncclEpDispatchConfig_t()
    dispatch_config.round_scales = 0

    is_ll = algorithm == ncclEpAlgorithm_t.NCCL_EP_ALGO_LOW_LATENCY
    num_inputs = 1 if is_ll else 3
    num_outputs = 1 if is_ll else 3
    num_local = 1 if is_ll else 0

    # -- input/output tensors for dispatch ----------------------------------
    input_tokens = tensor_create(
        nccl, ep_group, 2, ncclDataTypeEnum.ncclBfloat16,
        ncclEpTensorTag_t.NCCL_EP_TENSOR_TAG_TOKENS,
        None, num_tokens, hidden,
    )

    topk_weights = tensor_create(
        nccl, ep_group, 2, ncclDataTypeEnum.ncclFloat32,
        ncclEpTensorTag_t.NCCL_EP_TENSOR_TAG_TOPK_WEIGHTS,
        None, num_tokens, top_k,
    )

    if is_ll:
        output_tokens = tensor_create(
            nccl, ep_group, 3, ncclDataTypeEnum.ncclBfloat16,
            ncclEpTensorTag_t.NCCL_EP_TENSOR_TAG_TOKENS,
            None, num_local_experts, config.max_tokens_per_rank * n_ranks, hidden,
        )
    else:
        output_tokens = tensor_create(
            nccl, ep_group, 2, ncclDataTypeEnum.ncclBfloat16,
            ncclEpTensorTag_t.NCCL_EP_TENSOR_TAG_TOKENS,
            None, num_recv_tokens, hidden,
        )

    local_tensor_recv_count = None
    if is_ll:
        local_tensor_recv_count = tensor_create(
            nccl, ep_group, 1, ncclDataTypeEnum.ncclInt32,
            ncclEpTensorTag_t.NCCL_EP_TENSOR_TAG_RECV_EXPERT_COUNTER_DEVICE,
            None, num_local_experts,
        )

    output_topk_weights = None
    output_topk_idx = None
    if not is_ll:
        output_topk_weights = tensor_create(
            nccl, ep_group, 2, ncclDataTypeEnum.ncclFloat32,
            ncclEpTensorTag_t.NCCL_EP_TENSOR_TAG_TOPK_WEIGHTS,
            None, num_recv_tokens, top_k,
        )
        output_topk_idx = tensor_create(
            nccl, ep_group, 2, ncclDataTypeEnum.ncclInt64,
            ncclEpTensorTag_t.NCCL_EP_TENSOR_TAG_TOPK_IDX,
            None, num_recv_tokens, top_k,
        )

    # -- fill input tokens: first ELEMENTS_TESTED_PER_TOKEN elems = 0x1000 + rank
    input_host = (ctypes.c_uint16 * (num_tokens * hidden))()
    for i in range(num_tokens):
        for j in range(ELEMENTS_TESTED_PER_TOKEN):
            input_host[i * hidden + j] = 0x1000 + my_rank
    input0_data = tensor_get_data(nccl, input_tokens)
    cuda_memcpy(input0_data, input_host, num_tokens * hidden * 2, CUDA_MEMCPY_H2D)

    # -- fill topk_weights: 1.0 / top_k
    tw_host = (ctypes.c_float * (num_tokens * top_k))()
    for i in range(num_tokens * top_k):
        tw_host[i] = 1.0 / top_k
    tw_data = tensor_get_data(nccl, topk_weights)
    cuda_memcpy(tw_data, tw_host, num_tokens * top_k * 4, CUDA_MEMCPY_H2D)

    # -- build dispatch tensor arrays ----------------------------------------
    inputs_arr = (ncclNDTensor_t * num_inputs)()
    inputs_arr[0] = input_tokens
    if not is_ll:
        inputs_arr[1] = topk_weights
        inputs_arr[2] = topk_idx

    outputs_arr = (ncclNDTensor_t * num_outputs)()
    outputs_arr[0] = output_tokens
    if not is_ll:
        outputs_arr[1] = output_topk_weights
        outputs_arr[2] = output_topk_idx

    local_arr = (ncclNDTensor_t * max(num_local, 1))()
    if is_ll:
        local_arr[0] = local_tensor_recv_count

    # -- dispatch -----------------------------------------------------------
    print(f"Rank {my_rank}: Testing ncclEpDispatch (send_only={'true' if dispatch_send_only else 'false'})")
    nccl.ncclEpDispatch(
        ep_handle,
        ctypes.cast(inputs_arr, ctypes.POINTER(ncclNDTensor_t)), num_inputs,
        ctypes.cast(outputs_arr, ctypes.POINTER(ncclNDTensor_t)), num_outputs,
        ctypes.cast(local_arr, ctypes.POINTER(ncclNDTensor_t)) if num_local > 0 else None,
        num_local,
        int(dispatch_send_only),
        dispatch_config,
        stream,
    )

    print(f"Rank {my_rank}: Testing ncclEpComplete")
    nccl.ncclEpComplete(ep_handle, None, stream)
    cuda_stream_synchronize(stream)

    # -- read recv_count ----------------------------------------------------
    recv_count_host = None
    should_free_recv_count = False
    if is_ll:
        recv_count_host = (ctypes.c_int * num_local_experts)()
        lt0_data = tensor_get_data(nccl, local_tensor_recv_count)
        cuda_memcpy(recv_count_host, lt0_data, num_local_experts * 4, CUDA_MEMCPY_D2H)
        should_free_recv_count = True
    elif disable_max_tokens and handle_recv_expert_counter is not None:
        hlt0_data = tensor_get_data(nccl, handle_recv_expert_counter)
        recv_count_host = ctypes.cast(hlt0_data, ctypes.POINTER(ctypes.c_int))

    recv_from_expert_start = (local_experts_start + num_experts - num_local_experts) % num_experts
    recv_rank = recv_from_expert_start // num_local_experts

    # -- verify dispatch output ---------------------------------------------
    dispatch_check_passed = True
    first_dispatch_output0_host = None

    if not random_mode and is_ll and recv_count_host is not None:
        total_elems = num_local_experts * config.max_tokens_per_rank * n_ranks * hidden
        output_host = (ctypes.c_uint16 * total_elems)()
        out0_data = tensor_get_data(nccl, output_tokens)
        cuda_memcpy(output_host, out0_data, total_elems * 2, CUDA_MEMCPY_D2H)

        for e in range(num_local_experts):
            expected_count = num_tokens
            if recv_count_host[e] != expected_count:
                print(f"Recv_count check failed! Rank {my_rank}, expert {e}: "
                      f"expected {expected_count}, got {recv_count_host[e]}")
                dispatch_check_passed = False
                break
            max_t = config.max_tokens_per_rank * n_ranks
            for t in range(min(recv_count_host[e], max_t)):
                token_off = (e * max_t + t) * hidden
                for j in range(ELEMENTS_TESTED_PER_TOKEN):
                    expected = 0x1000 + recv_rank
                    actual = output_host[token_off + j]
                    if actual != expected:
                        print(f"Dispatch data check failed! Rank {my_rank}, expert {e}, "
                              f"token {t}, element {j}: expected {expected}, got {actual}")
                        dispatch_check_passed = False
                        break
                if not dispatch_check_passed:
                    break
            if not dispatch_check_passed:
                break

    elif not random_mode:
        if recv_count_host is not None:
            for e in range(num_local_experts):
                expected_count = num_tokens
                if recv_count_host[e] != expected_count:
                    print(f"Recv_count check failed! Rank {my_rank}, expert {e}: "
                          f"expected {expected_count}, got {recv_count_host[e]}")
                    dispatch_check_passed = False
                    break

        output_host = (ctypes.c_uint16 * (num_recv_tokens * hidden))()
        out0_data = tensor_get_data(nccl, output_tokens)
        cuda_memcpy(output_host, out0_data, num_recv_tokens * hidden * 2, CUDA_MEMCPY_D2H)
        check_count = num_recv_tokens if disable_max_tokens else num_tokens
        for i in range(check_count):
            for j in range(ELEMENTS_TESTED_PER_TOKEN):
                expected = 0x1000 + recv_rank
                actual = output_host[i * hidden + j]
                if actual != expected:
                    print(f"Dispatch check failed! Rank {my_rank}, token {i}, "
                          f"element {j}: expected {expected}, got {actual}")
                    dispatch_check_passed = False
                    break
            if not dispatch_check_passed:
                break
        first_dispatch_output0_host = output_host

    # recv_count cleanup (matches C++ should_free_recv_count logic)
    del should_free_recv_count

    # -- verify HT recv_topk_weights / recv_topk_idx -----------------------
    if not random_mode and not is_ll:
        recv_tw = (ctypes.c_float * (num_recv_tokens * top_k))()
        out1_data = tensor_get_data(nccl, output_topk_weights)
        cuda_memcpy(recv_tw, out1_data, num_recv_tokens * top_k * 4, CUDA_MEMCPY_D2H)

        recv_ti = (ctypes.c_int64 * (num_recv_tokens * top_k))()
        out2_data = tensor_get_data(nccl, output_topk_idx)
        cuda_memcpy(recv_ti, out2_data, num_recv_tokens * top_k * 8, CUDA_MEMCPY_D2H)

        ht_outputs_valid = True
        print(f"Rank {my_rank}: Verifying recv_topk_weights and recv_topk_idx")
        expected_weight = 1.0 / top_k
        weight_errors = 0
        idx_errors = 0

        for i in range(num_tokens):
            for j in range(top_k):
                off = i * top_k + j
                if recv_tw[off] != expected_weight:
                    if weight_errors < 5:
                        print(f"Rank {my_rank}: recv_topk_weights[{i}][{j}] = {recv_tw[off]}, "
                              f"expected {expected_weight}")
                    weight_errors += 1
                    ht_outputs_valid = False
                idx_val = recv_ti[off]
                if idx_val < 0 or idx_val >= num_experts:
                    if idx_errors < 5:
                        print(f"Rank {my_rank}: recv_topk_idx[{i}][{j}] = {idx_val}, "
                              f"expected range [0, {num_experts})")
                    idx_errors += 1
                    ht_outputs_valid = False

        if weight_errors > 0:
            print(f"Rank {my_rank}: recv_topk_weights verification failed with {weight_errors} errors")
        if idx_errors > 0:
            print(f"Rank {my_rank}: recv_topk_idx verification failed with {idx_errors} errors")
        if ht_outputs_valid:
            print(f"Rank {my_rank}: {algorithm_name} mode recv_topk_weights and recv_topk_idx verification passed")
        else:
            dispatch_check_passed = False

    if random_mode:
        print(f"Rank {my_rank}: {algorithm_name} Dispatch flow completed (random mode, checks skipped)")
    elif dispatch_check_passed:
        print(f"Rank {my_rank}: {algorithm_name} Dispatch flow passed successfully")
    else:
        print(f"Rank {my_rank}: Exiting test due to dispatch failure")
        sys.exit(1)

    # ===================================================================
    # Combine phase
    # ===================================================================
    print(f"Rank {my_rank}: Testing {algorithm_name} Combine flow")

    if is_ll:
        expert_outputs = tensor_create(
            nccl, ep_group, 3, ncclDataTypeEnum.ncclBfloat16,
            ncclEpTensorTag_t.NCCL_EP_TENSOR_TAG_TOKENS,
            None, num_local_experts, config.max_tokens_per_rank * n_ranks, hidden,
        )
        eo_host = (ctypes.c_uint16 * (config.max_tokens_per_rank * hidden))()
        for t in range(config.max_tokens_per_rank):
            for j in range(ELEMENTS_TESTED_PER_TOKEN):
                eo_host[t * hidden + j] = float_to_bf16(float((j + 1) * 2))
        eo_data = tensor_get_data(nccl, expert_outputs)
        for e in range(num_local_experts):
            offset_bytes = e * config.max_tokens_per_rank * hidden * n_ranks * 2
            dst = ctypes.c_void_p(eo_data.value + offset_bytes)
            cuda_memcpy(dst, eo_host, config.max_tokens_per_rank * hidden * 2, CUDA_MEMCPY_H2D)
    else:
        expert_outputs = tensor_create(
            nccl, ep_group, 2, ncclDataTypeEnum.ncclBfloat16,
            ncclEpTensorTag_t.NCCL_EP_TENSOR_TAG_TOKENS,
            None, num_recv_tokens, hidden,
        )
        eo_host = (ctypes.c_uint16 * (num_recv_tokens * hidden))()
        for t in range(num_recv_tokens):
            for j in range(ELEMENTS_TESTED_PER_TOKEN):
                eo_host[t * hidden + j] = float_to_bf16(float((j + 1) * 2))
        eo_data = tensor_get_data(nccl, expert_outputs)
        cuda_memcpy(eo_data, eo_host, num_recv_tokens * hidden * 2, CUDA_MEMCPY_H2D)

    combined_output = tensor_create(
        nccl, ep_group, 2, ncclDataTypeEnum.ncclBfloat16,
        ncclEpTensorTag_t.NCCL_EP_TENSOR_TAG_TOKENS,
        None, num_tokens, hidden,
    )

    combine_in = (ncclNDTensor_t * 1)()
    combine_in[0] = expert_outputs

    combine_out = (ncclNDTensor_t * 1)()
    combine_out[0] = combined_output

    combine_local = (ncclNDTensor_t * 1)()
    combine_num_local = 0
    if is_ll:
        combine_local[0] = topk_weights
        combine_num_local = 1

    print(f"Rank {my_rank}: Testing ncclEpCombine (send_only={'true' if combine_send_only else 'false'})")
    nccl.ncclEpCombine(
        ep_handle,
        ctypes.cast(combine_in, ctypes.POINTER(ncclNDTensor_t)), 1,
        ctypes.cast(combine_out, ctypes.POINTER(ncclNDTensor_t)), 1,
        ctypes.cast(combine_local, ctypes.POINTER(ncclNDTensor_t)) if combine_num_local > 0 else None,
        combine_num_local,
        int(combine_send_only),
        None,
        stream,
    )

    nccl.ncclEpComplete(ep_handle, None, stream)
    cuda_stream_synchronize(stream)

    # -- verify combine output ---------------------------------------------
    combine_errors = 0
    if not random_mode:
        co_host = (ctypes.c_uint16 * (num_tokens * hidden))()
        co_data = tensor_get_data(nccl, combined_output)
        cuda_memcpy(co_host, co_data, num_tokens * hidden * 2, CUDA_MEMCPY_D2H)

        for i in range(num_tokens):
            for j in range(ELEMENTS_TESTED_PER_TOKEN):
                expected = float_to_bf16(float((j + 1) * 2))
                actual = co_host[i * hidden + j]
                if actual != expected:
                    print(f"Combine check failed! Rank {my_rank}, token {i}, "
                          f"element {j}: expected {expected}, got {actual}")
                    combine_errors += 1
                    if combine_errors >= 5:
                        break
            if combine_errors >= 5:
                break

    if random_mode:
        print(f"Rank {my_rank}: Combine flow completed (random mode, checks skipped)")
    elif combine_errors == 0:
        print(f"Rank {my_rank}: Combine verification PASSED! "
              f"All {num_tokens} tokens with {hidden} elements each correctly combined")
    else:
        print(f"Rank {my_rank}: Combine verification FAILED with {combine_errors} errors")
        print(f"Rank {my_rank}: Exiting test due to combine failure")
        sys.exit(1)

    # ===================================================================
    # Cached mode test (HT only — repeat dispatch + combine, compare)
    # ===================================================================
    if cached_mode:
        if is_ll:
            print(f"Rank {my_rank}: Error - cached mode is only supported in HT modes (not LL)")
            sys.exit(1)

        print(f"Rank {my_rank}: Testing cached mode ({algorithm_name})")

        # save first-phase outputs
        first_d0 = first_d1 = first_d2 = first_co = None
        if not random_mode:
            first_d0 = (ctypes.c_uint16 * (num_recv_tokens * hidden))()
            d0 = tensor_get_data(nccl, output_tokens)
            cuda_memcpy(first_d0, d0, num_recv_tokens * hidden * 2, CUDA_MEMCPY_D2H)

            first_d1 = (ctypes.c_float * (num_recv_tokens * top_k))()
            d1 = tensor_get_data(nccl, output_topk_weights)
            cuda_memcpy(first_d1, d1, num_recv_tokens * top_k * 4, CUDA_MEMCPY_D2H)

            first_d2 = (ctypes.c_int64 * (num_recv_tokens * top_k))()
            d2 = tensor_get_data(nccl, output_topk_idx)
            cuda_memcpy(first_d2, d2, num_recv_tokens * top_k * 8, CUDA_MEMCPY_D2H)

            first_co = (ctypes.c_uint16 * (num_tokens * hidden))()
            co = tensor_get_data(nccl, combined_output)
            cuda_memcpy(first_co, co, num_tokens * hidden * 2, CUDA_MEMCPY_D2H)

        # new output tensors for second dispatch
        cached_out_tokens = tensor_create(
            nccl, ep_group, 2, ncclDataTypeEnum.ncclBfloat16,
            ncclEpTensorTag_t.NCCL_EP_TENSOR_TAG_TOKENS,
            None, num_recv_tokens, hidden,
        )
        cached_combined_output = tensor_create(
            nccl, ep_group, 2, ncclDataTypeEnum.ncclBfloat16,
            ncclEpTensorTag_t.NCCL_EP_TENSOR_TAG_TOKENS,
            None, num_tokens, hidden,
        )
        cached_combined_tw = tensor_create(
            nccl, ep_group, 2, ncclDataTypeEnum.ncclFloat32,
            ncclEpTensorTag_t.NCCL_EP_TENSOR_TAG_TOPK_WEIGHTS,
            None, num_tokens, top_k,
        )

        cached_comb_outs = (ncclNDTensor_t * 2)()
        cached_comb_outs[0] = cached_combined_output
        cached_comb_outs[1] = cached_combined_tw

        # topk_weights input for combine (copy from dispatch output)
        cached_ctw_in = tensor_create(
            nccl, ep_group, 2, ncclDataTypeEnum.ncclFloat32,
            ncclEpTensorTag_t.NCCL_EP_TENSOR_TAG_TOPK_WEIGHTS,
            None, num_recv_tokens, top_k,
        )
        ctw_dst = tensor_get_data(nccl, cached_ctw_in)
        ctw_src = tensor_get_data(nccl, output_topk_weights)
        cuda_memcpy(ctw_dst, ctw_src, num_recv_tokens * top_k * 4, CUDA_MEMCPY_D2D)

        cached_comb_ins = (ncclNDTensor_t * 2)()
        cached_comb_ins[0] = expert_outputs
        cached_comb_ins[1] = cached_ctw_in

        # second dispatch (cached — only tokens, 1 input / 1 output)
        cached_disp_outs = (ncclNDTensor_t * 1)()
        cached_disp_outs[0] = cached_out_tokens

        print(f"Rank {my_rank}: Testing cached mode - second ncclEpDispatch call "
              f"(send_only={'true' if dispatch_send_only else 'false'})")
        nccl.ncclEpDispatch(
            ep_handle,
            ctypes.cast(inputs_arr, ctypes.POINTER(ncclNDTensor_t)), 1,
            ctypes.cast(cached_disp_outs, ctypes.POINTER(ncclNDTensor_t)), 1,
            None, 0,
            int(dispatch_send_only),
            dispatch_config,
            stream,
        )

        print(f"Rank {my_rank}: Testing cached mode - second ncclEpComplete (dispatch)")
        nccl.ncclEpComplete(ep_handle, None, stream)
        cuda_stream_synchronize(stream)

        print(f"Rank {my_rank}: Testing cached mode - second ncclEpCombine call "
              f"(send_only={'true' if combine_send_only else 'false'})")
        nccl.ncclEpCombine(
            ep_handle,
            ctypes.cast(cached_comb_ins, ctypes.POINTER(ncclNDTensor_t)), 2,
            ctypes.cast(cached_comb_outs, ctypes.POINTER(ncclNDTensor_t)), 2,
            None, 0,
            int(combine_send_only),
            None,
            stream,
        )

        print(f"Rank {my_rank}: Testing cached mode - second ncclEpComplete (combine)")
        nccl.ncclEpComplete(ep_handle, None, stream)
        cuda_stream_synchronize(stream)

        # compare first vs second phase
        cached_dispatch_errors = 0
        cached_combine_errors = 0

        if not random_mode:
            sec_d0 = (ctypes.c_uint16 * (num_recv_tokens * hidden))()
            sd0 = tensor_get_data(nccl, cached_out_tokens)
            cuda_memcpy(sec_d0, sd0, num_recv_tokens * hidden * 2, CUDA_MEMCPY_D2H)

            sec_co = (ctypes.c_uint16 * (num_tokens * hidden))()
            sco = tensor_get_data(nccl, cached_combined_output)
            cuda_memcpy(sec_co, sco, num_tokens * hidden * 2, CUDA_MEMCPY_D2H)

            sec_tw = (ctypes.c_float * (num_tokens * top_k))()
            stw = tensor_get_data(nccl, cached_combined_tw)
            cuda_memcpy(sec_tw, stw, num_tokens * top_k * 4, CUDA_MEMCPY_D2H)

            for i in range(num_recv_tokens * hidden):
                if first_d0[i] != sec_d0[i]:
                    if cached_dispatch_errors < 5:
                        print(f"Rank {my_rank}: Cached dispatch output0 mismatch at {i}: "
                              f"first={first_d0[i]}, second={sec_d0[i]}")
                    cached_dispatch_errors += 1

            for i in range(num_tokens * hidden):
                if first_co[i] != sec_co[i]:
                    if cached_combine_errors < 5:
                        print(f"Rank {my_rank}: Cached combine output mismatch at {i}: "
                              f"first={first_co[i]}, second={sec_co[i]}")
                    cached_combine_errors += 1

            expected_w = 1.0 / top_k
            for i in range(num_tokens * top_k):
                if sec_tw[i] != expected_w:
                    if cached_combine_errors < 5:
                        print(f"Rank {my_rank}: Cached combine topk_weights mismatch at {i}: "
                              f"expected={expected_w}, got={sec_tw[i]}")
                    cached_combine_errors += 1

        if random_mode:
            print(f"Rank {my_rank}: Cached mode completed (random mode, checks skipped)")
        elif cached_dispatch_errors == 0 and cached_combine_errors == 0:
            print(f"Rank {my_rank}: Cached mode verification PASSED - dispatch and combine outputs match")
        else:
            print(f"Rank {my_rank}: Cached mode verification FAILED - "
                  f"dispatch errors: {cached_dispatch_errors}, combine errors: {cached_combine_errors}")
            sys.exit(1)

        # cleanup cached tensors
        tensor_destroy(nccl, ep_group, cached_out_tokens)
        tensor_destroy(nccl, ep_group, cached_combined_output)
        tensor_destroy(nccl, ep_group, cached_combined_tw)
        tensor_destroy(nccl, ep_group, cached_ctw_in)

        print(f"Rank {my_rank}: Cached mode - second dispatch and combine calls completed successfully")

    # ===================================================================
    # Cleanup — all tensor destroys must happen before ncclEpGroupDestroy
    # because GroupDestroy free()s the ep_group struct whose free_fn is
    # needed by TensorDestroy to release owned device memory.
    # ===================================================================
    tensor_destroy(nccl, ep_group, expert_outputs)
    tensor_destroy(nccl, ep_group, topk_weights)
    tensor_destroy(nccl, ep_group, combined_output)
    tensor_destroy(nccl, ep_group, topk_idx)
    if disable_max_tokens and handle_recv_expert_counter is not None:
        rc_data = tensor_get_data(nccl, handle_recv_expert_counter)
        cuda_free_host(rc_data)
        tensor_destroy(nccl, ep_group, handle_recv_expert_counter)
    tensor_destroy(nccl, ep_group, input_tokens)
    tensor_destroy(nccl, ep_group, output_tokens)
    if not is_ll:
        tensor_destroy(nccl, ep_group, output_topk_weights)
        tensor_destroy(nccl, ep_group, output_topk_idx)
    if is_ll:
        tensor_destroy(nccl, ep_group, local_tensor_recv_count)

    nccl.ncclEpHandleDestroy(ep_handle)
    nccl.ncclEpGroupDestroy(ep_group, stream)

    # ncclCommDestroy (not exposed in wrapper — call underlying library)
    _nccl_base = NCCLLibrary._nccl_base_lib or nccl.lib
    _nccl_base.ncclCommDestroy(comm)

    cuda_device_reset()
    print(f"[MPI Rank {my_rank}] Success ")


if __name__ == "__main__":
    main()
