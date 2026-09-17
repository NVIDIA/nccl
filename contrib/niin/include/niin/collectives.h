/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NIIN_COLLECTIVES_H_
#define NIIN_COLLECTIVES_H_

#include "niin/context.h"

// ---------------------------------------------------------------------------
// Device-side barrier implementation.
// Uses LSA-only barrier when all PEs are on the same node (lsaSize == nRanks),
// falls back to full world barrier (LSA + GIN) when multi-node.
// ---------------------------------------------------------------------------
__device__ __forceinline__ void niin_device_barrier_all() {
  ncclDevComm const& comm = niin_comm();
  if (comm.lsaSize == comm.nRanks || !niin_has_gin()) {
    // All PEs on same node — use LSA barrier only (no GIN overhead)
    ncclLsaBarrierSession<ncclCoopThread> bar(ncclCoopThread{}, comm, ncclTeamTagLsa{}, 0);
    bar.sync(ncclCoopThread{}, cuda::memory_order_acq_rel);
  } else {
    // Multi-node — need GIN barrier for cross-node sync
    // RMA operations are distributed across GIN contexts by CTA.  The
    // barrier must therefore fence all of them, not only this CTA's context.
    ncclGinAllContexts gin(comm);
    ncclGinBarrierSession<ncclCoopThread> bar(ncclCoopThread{}, gin, ncclTeamTagWorld{}, 0);
    // A barrier must make prior remote puts visible after it returns.  Ask
    // NCCL's existing GIN barrier to fence both the producer and consumer
    // sides instead of using its synchronization-only mode.
    bar.sync(ncclCoopThread{}, cuda::memory_order_acq_rel,
             ncclGinFenceLevel::Put | ncclGinFenceLevel::Get);
  }
}

// The block-scoped NVSHMEMX barriers have a stronger participation contract:
// every thread in one matching CTA per PE enters the operation.  Use NCCL's
// CTA cooperative form for those wrappers rather than letting one thread use
// the scalar barrier form.  In particular, the CTA form owns the required
// intra-CTA rendezvous while advancing the shared NCCL barrier session.
__device__ __forceinline__ void niin_device_barrier_all_cta() {
  ncclDevComm const& comm = niin_comm();
  ncclCoopCta coop;
  if (comm.lsaSize == comm.nRanks || !niin_has_gin()) {
    ncclLsaBarrierSession<ncclCoopCta> bar(
        coop, comm, ncclTeamLsa(comm), comm.lsaBarrier, 0);
    bar.sync(coop, cuda::memory_order_acq_rel);
  } else {
    ncclGinAllContexts gin(comm);
    ncclGinBarrierSession<ncclCoopCta> bar(coop, gin, ncclTeamTagWorld{}, 0);
    bar.sync(coop, cuda::memory_order_acq_rel,
             ncclGinFenceLevel::Put | ncclGinFenceLevel::Get);
  }
}

// ---------------------------------------------------------------------------
// __host__ __device__ wrappers that dispatch to device or host impl
// ---------------------------------------------------------------------------

__host__ __device__ __forceinline__ void nvshmem_barrier_all() {
#ifdef __CUDA_ARCH__
  niin_device_barrier_all();
#else
  extern void niin_host_barrier_all();
  niin_host_barrier_all();
#endif
}

__host__ __device__ __forceinline__ void nvshmem_sync_all() {
  nvshmem_barrier_all();
}

// ---------------------------------------------------------------------------
// Block-scoped extended barriers.
//
// NVSHMEMX requires every thread in the CTA to participate in the block
// variants.  One representative thread performs NIIN's existing global
// barrier/sync while the CTA-level barriers make the operation collective at
// block scope.  Barrier adds the system-scope release/acquire ordering that
// sync intentionally does not provide.
// ---------------------------------------------------------------------------
#define NIIN_HAS_NVSHMEMX_BLOCK_COLLECTIVES 1

__device__ __forceinline__ void nvshmemx_sync_all_block() {
  __syncthreads();
  niin_device_barrier_all_cta();
  __syncthreads();
}

__device__ __forceinline__ void nvshmemx_barrier_all_block() {
  cuda::atomic_thread_fence(cuda::memory_order_release, cuda::thread_scope_system);
  __syncthreads();
  niin_device_barrier_all_cta();
  __syncthreads();
  cuda::atomic_thread_fence(cuda::memory_order_acquire, cuda::thread_scope_system);
}

__host__ __device__ __forceinline__ void nvshmem_barrier(nvshmem_team_t team) {
  if (team == NVSHMEM_TEAM_WORLD) {
    nvshmem_barrier_all();
    return;
  }
#ifdef __CUDA_ARCH__
  NIIN_NOT_IMPLEMENTED_VOID("nvshmem_barrier (non-WORLD team)");
#endif
}

__host__ __device__ __forceinline__ void nvshmem_sync(nvshmem_team_t team) {
  if (team == NVSHMEM_TEAM_WORLD) {
    nvshmem_sync_all();
    return;
  }
#ifdef __CUDA_ARCH__
  NIIN_NOT_IMPLEMENTED_VOID("nvshmem_sync (non-WORLD team)");
#endif
}

// ---------------------------------------------------------------------------
// Collective reductions, broadcast, alltoall, fcollect: NOT_IMPLEMENTED
// ---------------------------------------------------------------------------

#define NIIN_DEFINE_REDUCE_STUB(TYPENAME, TYPE)                               \
__device__ __forceinline__ int nvshmem_##TYPENAME##_sum_reduce(               \
    nvshmem_team_t team, TYPE* dest, const TYPE* src, size_t nreduce) {       \
  NIIN_NOT_IMPLEMENTED_RETURN("nvshmem_" #TYPENAME "_sum_reduce", -1);       \
}                                                                              \
__device__ __forceinline__ int nvshmem_##TYPENAME##_max_reduce(               \
    nvshmem_team_t team, TYPE* dest, const TYPE* src, size_t nreduce) {       \
  NIIN_NOT_IMPLEMENTED_RETURN("nvshmem_" #TYPENAME "_max_reduce", -1);       \
}                                                                              \
__device__ __forceinline__ int nvshmem_##TYPENAME##_min_reduce(               \
    nvshmem_team_t team, TYPE* dest, const TYPE* src, size_t nreduce) {       \
  NIIN_NOT_IMPLEMENTED_RETURN("nvshmem_" #TYPENAME "_min_reduce", -1);       \
}                                                                              \
__device__ __forceinline__ int nvshmem_##TYPENAME##_prod_reduce(              \
    nvshmem_team_t team, TYPE* dest, const TYPE* src, size_t nreduce) {       \
  NIIN_NOT_IMPLEMENTED_RETURN("nvshmem_" #TYPENAME "_prod_reduce", -1);      \
}

NIIN_STANDARD_RMA_TYPES(NIIN_DEFINE_REDUCE_STUB)
#undef NIIN_DEFINE_REDUCE_STUB

__device__ __forceinline__ int nvshmem_broadcastmem(nvshmem_team_t team,
    void* dest, const void* src, size_t nelems, int pe_root) {
  NIIN_NOT_IMPLEMENTED_RETURN("nvshmem_broadcastmem", -1);
}

__device__ __forceinline__ int nvshmem_alltoallmem(nvshmem_team_t team,
    void* dest, const void* src, size_t nelems) {
  NIIN_NOT_IMPLEMENTED_RETURN("nvshmem_alltoallmem", -1);
}

__device__ __forceinline__ int nvshmem_fcollectmem(nvshmem_team_t team,
    void* dest, const void* src, size_t nelems) {
  NIIN_NOT_IMPLEMENTED_RETURN("nvshmem_fcollectmem", -1);
}

#endif // NIIN_COLLECTIVES_H_
