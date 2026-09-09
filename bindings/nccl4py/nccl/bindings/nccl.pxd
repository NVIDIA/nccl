# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0
#
# This code was automatically generated with version 2.31.2. Do not modify it directly.



# <<<< PREAMBLE CONTENT >>>>

from libc.stdint cimport (
    intptr_t,
    uint64_t,
)


# <<<< END OF PREAMBLE CONTENT >>>>

from .cynccl cimport *


###############################################################################
# Types
###############################################################################

ctypedef ncclParamHandle_t ParamHandle
ctypedef ncclDevCommWindowTable_t DevCommWindowTable

ctypedef cudaStream_t Stream


###############################################################################
# Enum
###############################################################################

ctypedef ncclResult_t _Result
ctypedef ncclHostCftMode_t _HostCftMode
ctypedef ncclCommMemStat_t _CommMemStat
ctypedef ncclRedOp_dummy_t _RedOpDummy
ctypedef ncclRedOp_t _RedOp
ctypedef ncclDataType_t _DataType
ctypedef ncclScalarResidence_t _ScalarResidence
ctypedef ncclGinType_t _GinType
ctypedef ncclGinConnectionType_t _GinConnectionType
ctypedef ncclCftTeamMode_t _CftTeamMode


###############################################################################
# Functions
###############################################################################

cpdef intptr_t mem_alloc(size_t size) except? 0
cpdef mem_free(intptr_t ptr)
cpdef int get_version() except? -1
cpdef object get_unique_id()
cpdef object comm_init_rank_config(int nranks, comm_id, int rank, intptr_t config)
cpdef object comm_init_rank(int nranks, comm_id, int rank)
cpdef object comm_init_all(int ndev, devlist)
cpdef comm_finalize(object comm)
cpdef comm_destroy(object comm)
cpdef comm_abort(object comm)
cpdef comm_revoke(object comm, int revoke_flags)
cpdef object comm_split(object comm, int color, int key, intptr_t config)
cpdef object comm_shrink(object comm, exclude_ranks_list, int exclude_ranks_count, intptr_t config, int shrink_flags)
cpdef object comm_get_unique_id(object comm)
cpdef object comm_grow(object comm, int n_ranks, intptr_t unique_id, int rank, intptr_t config)
cpdef object comm_init_rank_scalable(int nranks, int myrank, int n_id, comm_ids, intptr_t config)
cpdef str get_error_string(int result)
cpdef str get_last_error(object comm)
cpdef int comm_get_async_error(object comm) except? -1
cpdef int comm_count(object comm) except? -1
cpdef int comm_cu_device(object comm) except? -1
cpdef int comm_user_rank(object comm) except? -1
cpdef intptr_t comm_register(object comm, intptr_t buff, size_t size) except? 0
cpdef comm_deregister(object comm, intptr_t handle)
cpdef comm_suspend(object comm, int flags)
cpdef comm_resume(object comm)
cpdef uint64_t comm_mem_stats(object comm, int stat) except? -1
cpdef object comm_window_register(object comm, intptr_t buff, size_t size, int win_flags)
cpdef comm_window_deregister(object comm, object win)
cpdef intptr_t win_get_user_ptr(object comm, object win) except? 0
cpdef int red_op_create_pre_mul_sum(intptr_t scalar, int datatype, int residence, object comm) except? -1
cpdef red_op_destroy(int op, object comm)
cpdef reduce(intptr_t sendbuff, intptr_t recvbuff, size_t count, int datatype, int op, int root, object comm, intptr_t stream)
cpdef bcast(intptr_t buff, size_t count, int datatype, int root, object comm, intptr_t stream)
cpdef broadcast(intptr_t sendbuff, intptr_t recvbuff, size_t count, int datatype, int root, object comm, intptr_t stream)
cpdef all_reduce(intptr_t sendbuff, intptr_t recvbuff, size_t count, int datatype, int op, object comm, intptr_t stream)
cpdef reduce_scatter(intptr_t sendbuff, intptr_t recvbuff, size_t recvcount, int datatype, int op, object comm, intptr_t stream)
cpdef all_gather(intptr_t sendbuff, intptr_t recvbuff, size_t sendcount, int datatype, object comm, intptr_t stream)
cpdef allto_all(intptr_t sendbuff, intptr_t recvbuff, size_t count, int datatype, object comm, intptr_t stream)
cpdef gather(intptr_t sendbuff, intptr_t recvbuff, size_t count, int datatype, int root, object comm, intptr_t stream)
cpdef scatter(intptr_t sendbuff, intptr_t recvbuff, size_t count, int datatype, int root, object comm, intptr_t stream)
cpdef all_reduce_config(intptr_t sendbuff, intptr_t recvbuff, size_t count, int datatype, int op, object comm, intptr_t stream, intptr_t config)
cpdef broadcast_config(intptr_t sendbuff, intptr_t recvbuff, size_t count, int datatype, int root, object comm, intptr_t stream, intptr_t config)
cpdef reduce_config(intptr_t sendbuff, intptr_t recvbuff, size_t count, int datatype, int op, int root, object comm, intptr_t stream, intptr_t config)
cpdef all_gather_config(intptr_t sendbuff, intptr_t recvbuff, size_t sendcount, int datatype, object comm, intptr_t stream, intptr_t config)
cpdef reduce_scatter_config(intptr_t sendbuff, intptr_t recvbuff, size_t recvcount, int datatype, int op, object comm, intptr_t stream, intptr_t config)
cpdef allto_all_config(intptr_t sendbuff, intptr_t recvbuff, size_t count, int datatype, object comm, intptr_t stream, intptr_t config)
cpdef gather_config(intptr_t sendbuff, intptr_t recvbuff, size_t count, int datatype, int root, object comm, intptr_t stream, intptr_t config)
cpdef scatter_config(intptr_t sendbuff, intptr_t recvbuff, size_t count, int datatype, int root, object comm, intptr_t stream, intptr_t config)
cpdef send(intptr_t sendbuff, size_t count, int datatype, int peer, object comm, intptr_t stream)
cpdef recv(intptr_t recvbuff, size_t count, int datatype, int peer, object comm, intptr_t stream)
cpdef put_signal(intptr_t localbuff, size_t count, int datatype, int peer, object peer_win, size_t peer_win_offset, int sig_idx, int ctx, unsigned int flags, object comm, intptr_t stream)
cpdef signal(int peer, int sig_idx, int ctx, unsigned int flags, object comm, intptr_t stream)
cpdef wait_signal(int n_desc, signal_descs, object comm, intptr_t stream)
cpdef group_start()
cpdef group_end()
cpdef object group_simulate_end()
cpdef object comm_query_properties(object comm)
cpdef object dev_comm_create(object comm, intptr_t reqs)
cpdef dev_comm_destroy(object comm, intptr_t dev_comm)
cpdef intptr_t get_lsa_multimem_device_pointer(object window, size_t offset) except? 0
cpdef intptr_t get_lsa_device_pointer(object window, size_t offset, int lsa_rank) except? 0
cpdef intptr_t get_multimem_device_pointer(object window, size_t offset, multimem) except? 0
cpdef intptr_t get_peer_device_pointer(object window, size_t offset, int peer) except? 0
cpdef tuple get_multimem_device_le_info(object window, size_t offset)
cpdef tuple get_cft_device_le_info(object window, size_t offset, int peer_cft, cft_team)
cpdef tuple get_peer_device_le_info(object window, size_t offset, int peer_world)
cpdef lsa_barrier_create_requirement(team, int n_barriers, intptr_t out_handle, intptr_t out_req)
cpdef gin_barrier_create_requirement(object comm, team, int n_barriers, intptr_t out_handle, intptr_t out_req)
cpdef ll_a2a_create_requirement(int n_blocks, int n_slots, intptr_t out_handle, intptr_t out_req)

# Hand-written: the team getters return ncclTeam_t by value, which cybind cannot
# emit; the rank mappers return int rather than ncclResult_t.
cpdef object team_world(object comm)
cpdef object team_lsa(object comm)
cpdef object team_rail(object comm)
cpdef object team_cft(object comm, int mode)
cpdef object team_cft_multimem(object comm)
cpdef int team_rank_to_world(object comm, intptr_t team, int rank)
cpdef int team_rank_to_lsa(object comm, intptr_t team, int rank)

# Hand-written: Param API (SKIP_LOWPP in nccl.cybind.yaml).
cpdef str param_get_parameter(str key)
cpdef list param_get_all_keys()
cpdef param_dump_all()

# Hand-written: not an NCCL entry point; reports the path of the loaded DSO.
cpdef object get_library_path()
