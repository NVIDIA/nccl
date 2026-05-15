# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0
#
# This code was automatically generated with version 1.0.0. Do not modify it directly.

from libc.stdint cimport intptr_t

from .cynccl_ep cimport *


###############################################################################
# Types
###############################################################################

ctypedef ncclComm_t Comm
ctypedef ncclWindow_t Window
ctypedef ncclNDTensor_t NDTensor
ctypedef ncclEpGroup_t EpGroup
ctypedef ncclEpHandle_t EpHandle
ctypedef ncclEpAllocFn_t EpAllocFn
ctypedef ncclEpFreeFn_t EpFreeFn

ctypedef cudaStream_t Stream


###############################################################################
# Enum
###############################################################################

ctypedef ncclResult_t _Result
ctypedef ncclDataType_t _DataType
ctypedef ncclEpAlgorithm_t _EpAlgorithm
ctypedef ncclEpLayout_t _EpLayout


###############################################################################
# Functions
###############################################################################

cpdef intptr_t create_group(intptr_t comm, intptr_t config) except? 0
cpdef group_destroy(intptr_t ep_group)
cpdef intptr_t tensor_create(unsigned int ndim, int datatype, intptr_t data, intptr_t sizes) except? 0
cpdef intptr_t tensor_create_from_window(unsigned int ndim, int datatype, intptr_t win, uint64_t win_offset, intptr_t sizes) except? 0
cpdef tensor_destroy(intptr_t tensor)
cpdef intptr_t create_handle(intptr_t ep_group, intptr_t topk_idx, intptr_t layout_info, intptr_t config, intptr_t stream) except? 0
cpdef handle_destroy(intptr_t handle)
cpdef size_t handle_mem_size(intptr_t ep_group, intptr_t config, int num_topk) except? -1
cpdef intptr_t init_handle(intptr_t ep_group, intptr_t config, int num_topk, intptr_t handle_mem) except? 0
cpdef update_handle(intptr_t handle, intptr_t topk_idx, intptr_t layout_info, intptr_t stream)
cpdef dispatch(intptr_t handle, intptr_t inputs, intptr_t outputs, intptr_t layout_info, intptr_t config, intptr_t stream)
cpdef combine(intptr_t handle, intptr_t inputs, intptr_t outputs, intptr_t config, intptr_t stream)
cpdef complete(intptr_t handle, intptr_t config, intptr_t stream)
cpdef intptr_t tensor_get_data(intptr_t tensor) except? 0
cpdef tensor_get_sizes(intptr_t tensor, intptr_t sizes, intptr_t ndim)
