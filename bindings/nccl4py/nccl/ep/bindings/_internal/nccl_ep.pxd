# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0
#
# This code was automatically generated with version 1.0.0. Do not modify it directly.

from ..cynccl_ep cimport *


###############################################################################
# Wrapper functions
###############################################################################

cdef ncclResult_t _ncclEpCreateGroup(ncclEpGroup_t* ep_group, ncclComm_t comm, const ncclEpGroupConfig_t* config) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil
cdef ncclResult_t _ncclEpGroupDestroy(ncclEpGroup_t ep_group) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil
cdef ncclResult_t _ncclEpTensorCreate(ncclNDTensor_t* tensor, unsigned int ndim, ncclDataType_t datatype, void* data, const size_t* sizes) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil
cdef ncclResult_t _ncclEpTensorCreateFromWindow(ncclNDTensor_t* tensor, unsigned int ndim, ncclDataType_t datatype, ncclWindow_t win, uint64_t win_offset, const size_t* sizes) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil
cdef ncclResult_t _ncclEpTensorDestroy(ncclNDTensor_t tensor) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil
cdef ncclResult_t _ncclEpCreateHandle(ncclEpHandle_t* handle, ncclEpGroup_t ep_group, ncclNDTensor_t topk_idx, const ncclEpLayoutInfo_t* layout_info, const ncclEpHandleConfig_t* config, cudaStream_t stream) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil
cdef ncclResult_t _ncclEpHandleDestroy(ncclEpHandle_t handle) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil
cdef ncclResult_t _ncclEpHandleMemSize(ncclEpGroup_t ep_group, const ncclEpHandleConfig_t* config, size_t* size_out, int num_topk) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil
cdef ncclResult_t _ncclEpInitHandle(ncclEpHandle_t* handle, ncclEpGroup_t ep_group, const ncclEpHandleConfig_t* config, int num_topk, ncclNDTensor_t handle_mem) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil
cdef ncclResult_t _ncclEpUpdateHandle(ncclEpHandle_t handle, ncclNDTensor_t topk_idx, const ncclEpLayoutInfo_t* layout_info, cudaStream_t stream) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil
cdef ncclResult_t _ncclEpDispatch(ncclEpHandle_t handle, const ncclEpDispatchInputs_t* inputs, const ncclEpDispatchOutputs_t* outputs, const ncclEpLayoutInfo_t* layout_info, const ncclEpDispatchConfig_t* config, cudaStream_t stream) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil
cdef ncclResult_t _ncclEpCombine(ncclEpHandle_t handle, const ncclEpCombineInputs_t* inputs, const ncclEpCombineOutputs_t* outputs, const ncclEpCombineConfig_t* config, cudaStream_t stream) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil
cdef ncclResult_t _ncclEpComplete(ncclEpHandle_t handle, const ncclEpCompleteConfig_t* config, cudaStream_t stream) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil
cdef ncclResult_t _ncclEpTensorGetData(ncclNDTensor_t tensor, void** data) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil
cdef ncclResult_t _ncclEpTensorGetSizes(ncclNDTensor_t tensor, const size_t** sizes, unsigned int* ndim) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil
