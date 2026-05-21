# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0
#
# This code was automatically generated with version 0.1.0. Do not modify it directly.

from libc.stdint cimport intptr_t

from .cynccl_ep cimport *


###############################################################################
# Types
###############################################################################

ctypedef ncclComm_t Comm
ctypedef ncclWindow_t Window
ctypedef ncclEpGroup_t Group
ctypedef ncclEpHandle_t Handle
ctypedef ncclEpAllocFn_t AllocFn
ctypedef ncclEpFreeFn_t FreeFn

ctypedef cudaStream_t Stream


###############################################################################
# Enum
###############################################################################

ctypedef ncclEpAlgorithm_t _Algorithm
ctypedef ncclEpLayout_t _Layout
ctypedef ncclEpPassDir_t _PassDir


###############################################################################
# Functions
###############################################################################

cpdef int get_version() except? -1
cpdef object tensor_alloc(unsigned int ndim, ncclDataType_t datatype, intptr_t sizes, intptr_t config)
cpdef tensor_destroy(intptr_t tensor)
cpdef intptr_t create_group(intptr_t comm, intptr_t config) except? 0
cpdef group_destroy(intptr_t ep_group)
cpdef intptr_t create_handle(intptr_t ep_group, int layout, intptr_t topk_idx, intptr_t layout_info, intptr_t config, intptr_t stream) except? 0
cpdef handle_destroy(intptr_t handle)
cpdef size_t handle_mem_size(intptr_t ep_group, int layout, intptr_t config, int num_topk) except? -1
cpdef intptr_t init_handle(intptr_t ep_group, int layout, intptr_t config, int num_topk, intptr_t handle_mem) except? 0
cpdef update_handle(intptr_t handle, intptr_t topk_idx, intptr_t layout_info, intptr_t stream)
cpdef dispatch(intptr_t handle, intptr_t inputs, intptr_t outputs, intptr_t layout_info, intptr_t config, intptr_t stream)
cpdef combine(intptr_t handle, intptr_t inputs, intptr_t outputs, intptr_t config, intptr_t stream)
cpdef complete(intptr_t handle, intptr_t config, intptr_t stream)
