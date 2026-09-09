// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef NCCL4RUST_DEVICE_H_
#define NCCL4RUST_DEVICE_H_

#include <nccl_device.h>

extern "C" {

__device__ int nccl4rust_dev_comm_rank(ncclDevComm_t const* comm);
__device__ int nccl4rust_dev_comm_n_ranks(ncclDevComm_t const* comm);
__device__ int nccl4rust_dev_comm_lsa_rank(ncclDevComm_t const* comm);
__device__ int nccl4rust_dev_comm_lsa_size(ncclDevComm_t const* comm);

__device__ int nccl4rust_team_rank_to_world(
  ncclDevComm_t const* comm, int teamNRanks, int teamRank, int teamStride, int rank);
__device__ int nccl4rust_team_rank_to_lsa(
  ncclDevComm_t const* comm, int teamNRanks, int teamRank, int teamStride, int rank);

__device__ void* nccl4rust_get_local_pointer(ncclWindow_t window, size_t offset);
__device__ void* nccl4rust_get_lsa_pointer(ncclWindow_t window, size_t offset, int peer);
__device__ void* nccl4rust_get_peer_pointer(ncclWindow_t window, size_t offset, int peer);
__device__ void* nccl4rust_get_peer_pointer_team(
  ncclWindow_t window, size_t offset, int teamNRanks, int teamRank, int teamStride, int peer);
__device__ void* nccl4rust_get_multimem_pointer(
  ncclWindow_t window, size_t offset, void* mcBasePtr);
__device__ void* nccl4rust_get_lsa_multimem_pointer(
  ncclWindow_t window, size_t offset, ncclDevComm_t const* comm);

__device__ void nccl4rust_lsa_barrier_thread(ncclDevComm_t const* comm, uint32_t index, bool multimem);
__device__ void nccl4rust_lsa_barrier_warp(ncclDevComm_t const* comm, uint32_t index, bool multimem);
__device__ void nccl4rust_lsa_barrier_cta(ncclDevComm_t const* comm, uint32_t index, bool multimem);

__device__ void nccl4rust_lsa_reduce_sum_copy_f32_cta(
  ncclDevComm_t const* comm, ncclWindow_t srcWindow, size_t srcOffset,
  ncclWindow_t dstWindow, size_t dstOffset, size_t count);

} // extern "C"

#endif // NCCL4RUST_DEVICE_H_
