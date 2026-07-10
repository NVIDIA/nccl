# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0
#
# This code was automatically generated with version 0.0.1. Do not modify it directly.


from libc.stdint cimport uint64_t

from nccl.bindings.cynccl cimport ncclResult_t, ncclDataType_t, _NCCLRESULT_T_INTERNAL_LOADING_ERROR



###############################################################################
# Types (structs, enums, ...)
###############################################################################

# enums
ctypedef enum ncclEpAlgorithm_t "ncclEpAlgorithm_t":
    NCCL_EP_ALGO_LOW_LATENCY "NCCL_EP_ALGO_LOW_LATENCY" = 0
    NCCL_EP_ALGO_HIGH_THROUGHPUT "NCCL_EP_ALGO_HIGH_THROUGHPUT" = 1

ctypedef enum ncclEpLayout_t "ncclEpLayout_t":
    NCCL_EP_LAYOUT_UNSET "NCCL_EP_LAYOUT_UNSET" = 0
    NCCL_EP_LAYOUT_EXPERT_MAJOR "NCCL_EP_LAYOUT_EXPERT_MAJOR"
    NCCL_EP_LAYOUT_RANK_MAJOR "NCCL_EP_LAYOUT_RANK_MAJOR"
    NCCL_EP_LAYOUT_FLAT "NCCL_EP_LAYOUT_FLAT"

ctypedef enum ncclEpPassDir_t "ncclEpPassDir_t":
    NCCL_EP_FWD_PASS "NCCL_EP_FWD_PASS" = 0
    NCCL_EP_BWD_PASS "NCCL_EP_BWD_PASS" = 1


# types
cdef extern from *:
    """
    #include <driver_types.h>
    #include <library_types.h>
    #include <cuComplex.h>
    """
    ctypedef void* cudaStream_t 'cudaStream_t'
    ctypedef int cudaError_t 'cudaError_t'


ctypedef void* ncclComm_t 'ncclComm_t'

ctypedef void* ncclWindow_t 'ncclWindow_t'

ctypedef void* ncclEpGroup_t 'ncclEpGroup_t'

ctypedef void* ncclEpHandle_t 'ncclEpHandle_t'

ctypedef struct ncclEpTensorAllocConfig_t 'ncclEpTensorAllocConfig_t':
    unsigned int size
    unsigned int magic

ctypedef cudaError_t (*ncclEpAllocFn_t 'ncclEpAllocFn_t')(
    void** ptr,
    size_t size,
    void* context
)

ctypedef cudaError_t (*ncclEpFreeFn_t 'ncclEpFreeFn_t')(
    void* ptr,
    void* context
)

ctypedef struct ncclEpHandleConfig_t 'ncclEpHandleConfig_t':
    unsigned int size
    unsigned int magic
    size_t dispatch_output_per_expert_alignment

ctypedef struct ncclEpDispatchConfig_t 'ncclEpDispatchConfig_t':
    unsigned int size
    unsigned int magic
    unsigned int send_only
    unsigned int round_scales
    ncclEpPassDir_t pass_direction

ctypedef struct ncclEpCombineConfig_t 'ncclEpCombineConfig_t':
    unsigned int size
    unsigned int magic
    unsigned int send_only
    ncclEpPassDir_t pass_direction

ctypedef struct ncclEpCompleteConfig_t 'ncclEpCompleteConfig_t':
    unsigned int size
    unsigned int magic

ctypedef struct ncclEpTensor_t 'ncclEpTensor_t':
    unsigned int size
    unsigned int magic
    unsigned int ndim
    ncclDataType_t datatype
    void* data
    ncclWindow_t win_hdl
    uint64_t win_offset
    size_t* sizes

ctypedef struct ncclEpAllocConfig_t 'ncclEpAllocConfig_t':
    ncclEpAllocFn_t alloc_fn
    ncclEpFreeFn_t free_fn
    void* context

ctypedef struct ncclEpLayoutInfo_t 'ncclEpLayoutInfo_t':
    unsigned int size
    unsigned int magic
    ncclEpTensor_t* expert_counters
    ncclEpTensor_t* src_rank_counters
    ncclEpTensor_t* expert_offsets
    ncclEpTensor_t* recv_total_counter

ctypedef struct ncclEpDispatchInputs_t 'ncclEpDispatchInputs_t':
    unsigned int size
    unsigned int magic
    ncclEpTensor_t* tokens
    ncclEpTensor_t* topk_weights
    ncclEpTensor_t* scales

ctypedef struct ncclEpDispatchOutputs_t 'ncclEpDispatchOutputs_t':
    unsigned int size
    unsigned int magic
    ncclEpTensor_t* tokens
    ncclEpTensor_t* topk_weights
    ncclEpTensor_t* scales
    ncclEpTensor_t* topk_idx

ctypedef struct ncclEpCombineInputs_t 'ncclEpCombineInputs_t':
    unsigned int size
    unsigned int magic
    ncclEpTensor_t* tokens
    ncclEpTensor_t* topk_weights

ctypedef struct ncclEpCombineOutputs_t 'ncclEpCombineOutputs_t':
    unsigned int size
    unsigned int magic
    ncclEpTensor_t* tokens
    ncclEpTensor_t* topk_weights

ctypedef struct ncclEpGroupConfig_t 'ncclEpGroupConfig_t':
    unsigned int size
    unsigned int magic
    unsigned int version
    ncclEpAlgorithm_t algorithm
    unsigned int num_experts
    unsigned int max_dispatch_tokens_per_rank
    unsigned int max_recv_tokens_per_rank
    unsigned int max_token_bytes
    unsigned long int rdma_buffer_size
    unsigned int num_qp_per_rank
    unsigned int num_channels
    unsigned int max_num_sms
    ncclEpAllocConfig_t alloc
    unsigned int enable_mask
    uint64_t timeout_ns


###############################################################################
# Functions
###############################################################################

cdef ncclResult_t ncclEpGetVersion(int* version) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil
cdef ncclResult_t ncclEpTensorAlloc(ncclEpTensor_t** tensor, unsigned int ndim, ncclDataType_t datatype, const size_t* sizes, const ncclEpTensorAllocConfig_t* config) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil
cdef ncclResult_t ncclEpTensorDestroy(ncclEpTensor_t* tensor) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil
cdef ncclResult_t ncclEpCreateGroup(ncclEpGroup_t* ep_group, ncclComm_t comm, const ncclEpGroupConfig_t* config) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil
cdef ncclResult_t ncclEpGroupDestroy(ncclEpGroup_t ep_group) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil
cdef ncclResult_t ncclEpCreateHandle(ncclEpHandle_t* handle, ncclEpGroup_t ep_group, ncclEpLayout_t layout, const ncclEpTensor_t* topk_idx, const ncclEpLayoutInfo_t* layout_info, const ncclEpHandleConfig_t* config, cudaStream_t stream) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil
cdef ncclResult_t ncclEpHandleDestroy(ncclEpHandle_t handle) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil
cdef ncclResult_t ncclEpHandleMemSize(ncclEpGroup_t ep_group, ncclEpLayout_t layout, const ncclEpHandleConfig_t* config, size_t* size_out, int num_topk) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil
cdef ncclResult_t ncclEpInitHandle(ncclEpHandle_t* handle, ncclEpGroup_t ep_group, ncclEpLayout_t layout, const ncclEpHandleConfig_t* config, int num_topk, const ncclEpTensor_t* handle_mem) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil
cdef ncclResult_t ncclEpUpdateHandle(ncclEpHandle_t handle, const ncclEpTensor_t* topk_idx, const ncclEpLayoutInfo_t* layout_info, cudaStream_t stream) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil
cdef ncclResult_t ncclEpDispatch(ncclEpHandle_t handle, const ncclEpDispatchInputs_t* inputs, const ncclEpDispatchOutputs_t* outputs, const ncclEpLayoutInfo_t* layout_info, const ncclEpDispatchConfig_t* config, cudaStream_t stream) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil
cdef ncclResult_t ncclEpCombine(ncclEpHandle_t handle, const ncclEpCombineInputs_t* inputs, const ncclEpCombineOutputs_t* outputs, const ncclEpCombineConfig_t* config, cudaStream_t stream) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil
cdef ncclResult_t ncclEpComplete(ncclEpHandle_t handle, const ncclEpCompleteConfig_t* config, cudaStream_t stream) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil
