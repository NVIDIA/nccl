# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0
#
# This code was automatically generated with version 0.0.1. Do not modify it directly.

from ._internal cimport nccl_ep as _nccl_ep


###############################################################################
# Wrapper functions
###############################################################################

cdef ncclResult_t ncclEpGetVersion(int* version) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    return _nccl_ep._ncclEpGetVersion(version)


cdef ncclResult_t ncclEpTensorAlloc(ncclEpTensor_t** tensor, unsigned int ndim, ncclDataType_t datatype, const size_t* sizes, const ncclEpTensorAllocConfig_t* config) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    return _nccl_ep._ncclEpTensorAlloc(tensor, ndim, datatype, sizes, config)


cdef ncclResult_t ncclEpTensorDestroy(ncclEpTensor_t* tensor) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    return _nccl_ep._ncclEpTensorDestroy(tensor)


cdef ncclResult_t ncclEpCreateGroup(ncclEpGroup_t* ep_group, ncclComm_t comm, const ncclEpGroupConfig_t* config) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    return _nccl_ep._ncclEpCreateGroup(ep_group, comm, config)


cdef ncclResult_t ncclEpGroupDestroy(ncclEpGroup_t ep_group) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    return _nccl_ep._ncclEpGroupDestroy(ep_group)


cdef ncclResult_t ncclEpCreateHandle(ncclEpHandle_t* handle, ncclEpGroup_t ep_group, ncclEpLayout_t layout, const ncclEpTensor_t* topk_idx, const ncclEpLayoutInfo_t* layout_info, const ncclEpHandleConfig_t* config, cudaStream_t stream) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    return _nccl_ep._ncclEpCreateHandle(handle, ep_group, layout, topk_idx, layout_info, config, stream)


cdef ncclResult_t ncclEpHandleDestroy(ncclEpHandle_t handle) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    return _nccl_ep._ncclEpHandleDestroy(handle)


cdef ncclResult_t ncclEpHandleMemSize(ncclEpGroup_t ep_group, ncclEpLayout_t layout, const ncclEpHandleConfig_t* config, size_t* size_out, int num_topk) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    return _nccl_ep._ncclEpHandleMemSize(ep_group, layout, config, size_out, num_topk)


cdef ncclResult_t ncclEpInitHandle(ncclEpHandle_t* handle, ncclEpGroup_t ep_group, ncclEpLayout_t layout, const ncclEpHandleConfig_t* config, int num_topk, const ncclEpTensor_t* handle_mem) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    return _nccl_ep._ncclEpInitHandle(handle, ep_group, layout, config, num_topk, handle_mem)


cdef ncclResult_t ncclEpUpdateHandle(ncclEpHandle_t handle, const ncclEpTensor_t* topk_idx, const ncclEpLayoutInfo_t* layout_info, cudaStream_t stream) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    return _nccl_ep._ncclEpUpdateHandle(handle, topk_idx, layout_info, stream)


cdef ncclResult_t ncclEpDispatch(ncclEpHandle_t handle, const ncclEpDispatchInputs_t* inputs, const ncclEpDispatchOutputs_t* outputs, const ncclEpLayoutInfo_t* layout_info, const ncclEpDispatchConfig_t* config, cudaStream_t stream) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    return _nccl_ep._ncclEpDispatch(handle, inputs, outputs, layout_info, config, stream)


cdef ncclResult_t ncclEpCombine(ncclEpHandle_t handle, const ncclEpCombineInputs_t* inputs, const ncclEpCombineOutputs_t* outputs, const ncclEpCombineConfig_t* config, cudaStream_t stream) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    return _nccl_ep._ncclEpCombine(handle, inputs, outputs, config, stream)


cdef ncclResult_t ncclEpComplete(ncclEpHandle_t handle, const ncclEpCompleteConfig_t* config, cudaStream_t stream) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    return _nccl_ep._ncclEpComplete(handle, config, stream)
