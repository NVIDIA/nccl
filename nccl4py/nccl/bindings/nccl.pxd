# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0
#
# This code was automatically generated with version 2.28.0. Do not modify it directly.

from libc.stdint cimport intptr_t

from .cynccl cimport *


###############################################################################
# Types
###############################################################################

ctypedef ncclComm_t Comm
ctypedef ncclWindow_t Window

ctypedef cudaStream_t Stream


###############################################################################
# Enum
###############################################################################

ctypedef ncclResult_t _Result
ctypedef ncclRedOp_dummy_t _RedOp_dummy
ctypedef ncclRedOp_t _RedOp
ctypedef ncclDataType_t _DataType
ctypedef ncclScalarResidence_t _ScalarResidence


###############################################################################
# Functions
###############################################################################

cpdef intptr_t mem_alloc(size_t size) except? 0
cpdef mem_free(intptr_t ptr)
cpdef int get_version() except? -1
cpdef get_unique_id(intptr_t unique_id)
cpdef intptr_t comm_init_rank_config(int nranks, comm_id, int rank, intptr_t config) except? 0
cpdef intptr_t comm_init_rank(int nranks, comm_id, int rank) except? 0
cpdef comm_init_all(intptr_t comm, int ndev, devlist)
cpdef comm_finalize(intptr_t comm)
cpdef comm_destroy(intptr_t comm)
cpdef comm_abort(intptr_t comm)
cpdef intptr_t comm_split(intptr_t comm, int color, int key, intptr_t config) except? 0
cpdef intptr_t comm_shrink(intptr_t comm, exclude_ranks_list, int exclude_ranks_count, intptr_t config, int shrink_flags) except? 0
cpdef intptr_t comm_init_rank_scalable(int nranks, int myrank, int n_id, comm_ids, intptr_t config) except? 0
cpdef str get_error_string(int result)
cpdef str get_last_error(intptr_t comm)
cpdef int comm_get_async_error(intptr_t comm) except? -1
cpdef int comm_count(intptr_t comm) except? -1
cpdef int comm_cu_device(intptr_t comm) except? -1
cpdef int comm_user_rank(intptr_t comm) except? -1
cpdef intptr_t comm_register(intptr_t comm, intptr_t buff, size_t size) except? 0
cpdef comm_deregister(intptr_t comm, intptr_t handle)
cpdef intptr_t comm_window_register(intptr_t comm, intptr_t buff, size_t size, int win_flags) except? 0
cpdef comm_window_deregister(intptr_t comm, intptr_t win)
cpdef int red_op_create_pre_mul_sum(intptr_t scalar, int datatype, int residence, intptr_t comm) except? -1
cpdef red_op_destroy(int op, intptr_t comm)
cpdef reduce(intptr_t sendbuff, intptr_t recvbuff, size_t count, int datatype, int op, int root, intptr_t comm, intptr_t stream)
cpdef bcast(intptr_t buff, size_t count, int datatype, int root, intptr_t comm, intptr_t stream)
cpdef broadcast(intptr_t sendbuff, intptr_t recvbuff, size_t count, int datatype, int root, intptr_t comm, intptr_t stream)
cpdef all_reduce(intptr_t sendbuff, intptr_t recvbuff, size_t count, int datatype, int op, intptr_t comm, intptr_t stream)
cpdef reduce_scatter(intptr_t sendbuff, intptr_t recvbuff, size_t recvcount, int datatype, int op, intptr_t comm, intptr_t stream)
cpdef all_gather(intptr_t sendbuff, intptr_t recvbuff, size_t sendcount, int datatype, intptr_t comm, intptr_t stream)
cpdef allto_all(intptr_t sendbuff, intptr_t recvbuff, size_t count, int datatype, intptr_t comm, intptr_t stream)
cpdef gather(intptr_t sendbuff, intptr_t recvbuff, size_t count, int datatype, int root, intptr_t comm, intptr_t stream)
cpdef scatter(intptr_t sendbuff, intptr_t recvbuff, size_t count, int datatype, int root, intptr_t comm, intptr_t stream)
cpdef send(intptr_t sendbuff, size_t count, int datatype, int peer, intptr_t comm, intptr_t stream)
cpdef recv(intptr_t recvbuff, size_t count, int datatype, int peer, intptr_t comm, intptr_t stream)
cpdef group_start()
cpdef group_end()
cpdef group_simulate_end(intptr_t sim_info)
