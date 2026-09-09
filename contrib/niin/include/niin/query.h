/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NIIN_QUERY_H_
#define NIIN_QUERY_H_

#include "niin/context.h"

// Device-only query helpers (used by device-side nvshmem_* implementations)
__device__ __forceinline__ int niin_device_my_pe() {
  return niin_comm().rank;
}

__device__ __forceinline__ int niin_device_n_pes() {
  return niin_comm().nRanks;
}

// nvshmem_my_pe: returns this PE's rank in TEAM_WORLD
// __host__ __device__ — on device reads from niin_g_ctx, on host reads from global state
__host__ __device__ __forceinline__ int nvshmem_my_pe() {
#ifdef __CUDA_ARCH__
  return niin_comm().rank;
#else
  // Host implementation provided by nvshmem_host.h via niin::detail::state()
  // We can't call it here (circular include), so return from the extern state.
  extern int niin_host_my_pe();
  return niin_host_my_pe();
#endif
}

__host__ __device__ __forceinline__ int nvshmem_n_pes() {
#ifdef __CUDA_ARCH__
  return niin_comm().nRanks;
#else
  extern int niin_host_n_pes();
  return niin_host_n_pes();
#endif
}

// nvshmem_ptr: returns a local pointer to pe's copy of a symmetric address.
// Returns nullptr if ptr is not in the symmetric heap or pe is not accessible.
__host__ __device__ __forceinline__ void* nvshmem_ptr(void* ptr, int pe) {
#ifdef __CUDA_ARCH__
  // Bounds check: ptr must be within the symmetric heap
  uintptr_t p = reinterpret_cast<uintptr_t>(ptr);
  uintptr_t base = reinterpret_cast<uintptr_t>(niin_heap_base());
  if (p < base || p >= base + niin_heap_size()) return nullptr;
  if (pe == niin_device_my_pe()) return ptr;
  if (!niin_is_lsa_peer(pe)) return nullptr;
  size_t offset = niin_sym_offset(ptr);
  return niin_get_peer_ptr(offset, pe);
#else
  extern void* niin_host_ptr(void*, int);
  return niin_host_ptr(ptr, pe);
#endif
}

// Team queries — all predefined teams supported on device
__host__ __device__ __forceinline__ int nvshmem_team_my_pe(nvshmem_team_t team) {
#ifdef __CUDA_ARCH__
  ncclDevComm const& c = niin_comm();
  switch (team) {
    case NVSHMEM_TEAM_WORLD:           return c.rank;
    case NVSHMEM_TEAM_SHARED:          return c.lsaRank;
    case NVSHMEMX_TEAM_NODE:           return c.lsaRank;
    case NVSHMEMX_TEAM_SAME_MYPE_NODE: return c.rank / c.lsaSize;  // rail rank
    case NVSHMEMI_TEAM_SAME_GPU:       return 0;                    // always rank 0 (size=1)
    case NVSHMEMI_TEAM_GPU_LEADERS:    return c.rank;               // same as WORLD
    case NVSHMEM_TEAM_INVALID:         return -1;
    default: return -1;  // unknown team
  }
#else
  extern int niin_host_team_my_pe(int);
  return niin_host_team_my_pe(team);
#endif
}

__host__ __device__ __forceinline__ int nvshmem_team_n_pes(nvshmem_team_t team) {
#ifdef __CUDA_ARCH__
  ncclDevComm const& c = niin_comm();
  switch (team) {
    case NVSHMEM_TEAM_WORLD:           return c.nRanks;
    case NVSHMEM_TEAM_SHARED:          return c.lsaSize;
    case NVSHMEMX_TEAM_NODE:           return c.lsaSize;
    case NVSHMEMX_TEAM_SAME_MYPE_NODE: return (c.nRanks + c.lsaSize - 1) / c.lsaSize; // num nodes
    case NVSHMEMI_TEAM_SAME_GPU:       return 1;
    case NVSHMEMI_TEAM_GPU_LEADERS:    return c.nRanks;             // same as WORLD
    case NVSHMEM_TEAM_INVALID:         return -1;
    default: return -1;  // unknown team
  }
#else
  extern int niin_host_team_n_pes(int);
  return niin_host_team_n_pes(team);
#endif
}

#endif // NIIN_QUERY_H_
