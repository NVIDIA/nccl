// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "nccl4rust_device.h"

#include <cstddef>

// Keep the flattened Rust ABI assumptions adjacent to the NCCL types that
// define them. The Rust crate has matching compile-time size/alignment checks.
static_assert(sizeof(int) == 4);
static_assert(sizeof(ncclTeam_t) == 3 * sizeof(int));
static_assert(alignof(ncclTeam_t) == alignof(int));
static_assert(offsetof(ncclTeam_t, nRanks) == 0);
static_assert(offsetof(ncclTeam_t, rank) == sizeof(int));
static_assert(offsetof(ncclTeam_t, stride) == 2 * sizeof(int));
static_assert(sizeof(ncclMultimemHandle_t) == sizeof(void*));
static_assert(alignof(ncclMultimemHandle_t) == alignof(void*));
static_assert(offsetof(ncclMultimemHandle_t, mcBasePtr) == 0);
static_assert(sizeof(bool) == 1);

// Preserve a stable C symbol for CUDA-Oxide/nvJitLink. The called NCCL public
// API remains inline and is compiled into this contribution-owned LTOIR.
#define NCCL4RUST_EXPORT extern "C" __device__ __noinline__

NCCL4RUST_EXPORT int nccl4rust_dev_comm_rank(ncclDevComm_t const* comm) {
  return comm->rank;
}

NCCL4RUST_EXPORT int nccl4rust_dev_comm_n_ranks(ncclDevComm_t const* comm) {
  return comm->nRanks;
}

NCCL4RUST_EXPORT int nccl4rust_dev_comm_lsa_rank(ncclDevComm_t const* comm) {
  return comm->lsaRank;
}

NCCL4RUST_EXPORT int nccl4rust_dev_comm_lsa_size(ncclDevComm_t const* comm) {
  return comm->lsaSize;
}

NCCL4RUST_EXPORT int nccl4rust_team_rank_to_world(
    ncclDevComm_t const* comm, int teamNRanks, int teamRank, int teamStride, int rank) {
  ncclTeam_t team = {teamNRanks, teamRank, teamStride};
  return ncclTeamRankToWorld(*comm, team, rank);
}

NCCL4RUST_EXPORT int nccl4rust_team_rank_to_lsa(
    ncclDevComm_t const* comm, int teamNRanks, int teamRank, int teamStride, int rank) {
  ncclTeam_t team = {teamNRanks, teamRank, teamStride};
  return ncclTeamRankToLsa(*comm, team, rank);
}

NCCL4RUST_EXPORT void* nccl4rust_get_local_pointer(ncclWindow_t window, size_t offset) {
  return ncclGetLocalPointer(window, offset);
}

NCCL4RUST_EXPORT void* nccl4rust_get_lsa_pointer(ncclWindow_t window, size_t offset, int peer) {
  return ncclGetLsaPointer(window, offset, peer);
}

NCCL4RUST_EXPORT void* nccl4rust_get_peer_pointer(ncclWindow_t window, size_t offset, int peer) {
  return ncclGetPeerPointer(window, offset, peer);
}

NCCL4RUST_EXPORT void* nccl4rust_get_peer_pointer_team(
    ncclWindow_t window, size_t offset,
    int teamNRanks, int teamRank, int teamStride, int peer) {
  ncclTeam_t team = {teamNRanks, teamRank, teamStride};
  return ncclGetPeerPointer(window, offset, team, peer);
}

NCCL4RUST_EXPORT void* nccl4rust_get_multimem_pointer(
    ncclWindow_t window, size_t offset, void* mcBasePtr) {
  ncclMultimemHandle_t handle = {mcBasePtr};
  return ncclGetMultimemPointer(window, offset, handle);
}

NCCL4RUST_EXPORT void* nccl4rust_get_lsa_multimem_pointer(
    ncclWindow_t window, size_t offset, ncclDevComm_t const* comm) {
  return ncclGetLsaMultimemPointer(window, offset, *comm);
}

template <typename Coop>
__device__ __forceinline__ void nccl4rust_lsa_barrier(
    Coop coop, ncclDevComm_t const* comm, uint32_t index, bool multimem) {
  ncclLsaBarrierSession<Coop> session(coop, *comm, ncclTeamTagLsa{}, index, multimem);
  session.sync(coop, cuda::memory_order_acq_rel);
}

NCCL4RUST_EXPORT void nccl4rust_lsa_barrier_thread(
    ncclDevComm_t const* comm, uint32_t index, bool multimem) {
  ncclCoopThread coop;
  nccl4rust_lsa_barrier(coop, comm, index, multimem);
}

NCCL4RUST_EXPORT void nccl4rust_lsa_barrier_warp(
    ncclDevComm_t const* comm, uint32_t index, bool multimem) {
  ncclCoopWarp coop;
  nccl4rust_lsa_barrier(coop, comm, index, multimem);
}

NCCL4RUST_EXPORT void nccl4rust_lsa_barrier_cta(
    ncclDevComm_t const* comm, uint32_t index, bool multimem) {
  ncclCoopCta coop;
  nccl4rust_lsa_barrier(coop, comm, index, multimem);
}

NCCL4RUST_EXPORT void nccl4rust_lsa_reduce_sum_copy_f32_cta(
    ncclDevComm_t const* comm, ncclWindow_t srcWindow, size_t srcOffset,
    ncclWindow_t dstWindow, size_t dstOffset, size_t count) {
  ncclCoopCta coop;
  ncclLsaReduceSumCopy<float>(
    coop, srcWindow, srcOffset, dstWindow, dstOffset, count, *comm);
}
