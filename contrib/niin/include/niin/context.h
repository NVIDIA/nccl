/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NIIN_CONTEXT_H_
#define NIIN_CONTEXT_H_

#include "niin/config.h"
#include "niin/types.h"
#include "niin/context_abi.h"

// Device-side global context pointer used by the header-only device API.
// Relocatable device code lets `inline` coalesce this header-defined variable
// across consumer translation units. Whole-program CUDA compilation cannot use
// an externally linked inline device variable, so a single-TU consumer gets an
// internal definition instead.
#if defined(__CUDACC_RDC__)
inline __device__ niinContext* niin_g_ctx;
inline __constant__ niinContext niin_g_ctx_constant;
#else
static __device__ niinContext* niin_g_ctx;
static __constant__ niinContext niin_g_ctx_constant;
#endif

#ifndef NIIN_USE_DEVICE_CONTEXT_POINTER
#define NIIN_USE_DEVICE_CONTEXT_POINTER 0
#endif

__device__ __forceinline__ niinContext const& niin_ctx() {
#if NIIN_USE_DEVICE_CONTEXT_POINTER
  return *niin_g_ctx;
#else
  return niin_g_ctx_constant;
#endif
}

// ---------------------------------------------------------------------------
// Internal helpers for accessing context fields
// ---------------------------------------------------------------------------

__device__ __forceinline__ ncclDevComm const& niin_comm() {
#if NIIN_USE_DEVICE_CONTEXT_POINTER
  return *niin_g_ctx->comm;
#else
  return niin_g_ctx_constant.commValue;
#endif
}

__device__ __forceinline__ ncclWindow_t niin_heap_window() {
  return niin_ctx().heapWindow;
}

__device__ __forceinline__ void* niin_heap_base() {
  return niin_ctx().heapBase;
}

__device__ __forceinline__ size_t niin_heap_size() {
  return niin_ctx().heapSize;
}

__device__ __forceinline__ void** niin_peer_heap_base_p2p() {
  return niin_ctx().peerHeapBaseP2p;
}

__device__ __forceinline__ int niin_rank() {
  return niin_ctx().rank;
}

__device__ __forceinline__ int niin_n_ranks() {
  return niin_ctx().nRanks;
}

__device__ __forceinline__ int niin_lsa_rank() {
  return niin_ctx().lsaRank;
}

__device__ __forceinline__ int niin_lsa_size() {
  return niin_ctx().lsaSize;
}

// Number of GIN contexts on this comm. Each context owns one QP per peer, and
// NCCL derives connectionId from the context index, so consecutive indices also
// spread across NICs. NCCL rounds the requested count up to a multiple of the
// connection count, so read the count back from the devComm rather than trusting
// what NIIN asked for at init.
__device__ __forceinline__ uint32_t niin_gin_context_count() {
  uint32_t n = niin_comm().ginContextCount;
  return n > 0 ? n : 1;
}

// Rotating cursor behind niin_gin_next_context(). Header-static, so each
// translation unit rotates its own copy; the sequence only has to spread
// operations over contexts, not be globally ordered.
static __device__ unsigned int niin_gin_context_cursor = 0;

// Take the next GIN context for one network operation, round-robin over the
// available contexts, the way NVSHMEM's IBGDA and IBRC transports pick a QP.
//
// Selection is per operation, so a thread's consecutive puts can land on
// different QPs. Puts on different QPs are unordered against each other at the
// receiving NIC, so every operation that has to stay ordered with a later flush
// or signal reuses the context returned here rather than calling this again,
// and nvshmem_fence()/nvshmem_quiet() drain all contexts.
__device__ __forceinline__ int niin_gin_next_context() {
  uint32_t n = niin_gin_context_count();
  if (n == 1) return 0;
  uint32_t ticket = atomicAdd(&niin_gin_context_cursor, 1u);
  return (int)((n & (n - 1)) == 0 ? (ticket & (n - 1)) : (ticket % n));
}

// Resolve an operation's context: a caller that already reserved one passes it
// through so the whole operation stays on a single QP.
__device__ __forceinline__ int niin_gin_context_or_next(int ctx) {
  return ctx >= 0 ? ctx : niin_gin_next_context();
}

// Context reserved for collectives. GIN barriers exchange indexed signals whose
// storage and shadow counters are per-context, so a barrier only converges if
// every PE drives it from the same context index.
__device__ __forceinline__ int niin_gin_collective_context_index() {
  return 0;
}

__device__ __forceinline__ const struct niinGpunetioAtomicContext* niin_gpunetio_atomic_context() {
  return niin_ctx().gpunetioAtomicContext;
}

__device__ __forceinline__ bool niin_has_gin() {
  return niin_ctx().ginConnectionCount > 0;
}

__device__ __forceinline__ bool niin_world_is_lsa_only() {
  return niin_ctx().worldIsLsaOnly;
}

__device__ __forceinline__ size_t niin_linear_block_id() {
  return (size_t)blockIdx.x + (size_t)blockIdx.y * gridDim.x +
         (size_t)blockIdx.z * gridDim.x * gridDim.y;
}

__device__ __forceinline__ void niin_note_lsa_store_pending() {
  niinContext const& ctx = niin_ctx();
  unsigned int* pending = ctx.lsaStorePending;
  size_t blockId = niin_linear_block_id();
  if (pending != nullptr && blockId < ctx.lsaStorePendingLen) pending[blockId] = 1u;
}

__device__ __forceinline__ bool niin_consume_lsa_store_pending() {
  niinContext const& ctx = niin_ctx();
  unsigned int* pending = ctx.lsaStorePending;
  size_t blockId = niin_linear_block_id();
  if (pending == nullptr || blockId >= ctx.lsaStorePendingLen) return true;
  if (*(volatile unsigned int*)&pending[blockId] == 0u) return false;
  pending[blockId] = 0u;
  return true;
}

__device__ __forceinline__ void niin_clear_lsa_store_pending() {
  niinContext const& ctx = niin_ctx();
  unsigned int* pending = ctx.lsaStorePending;
  size_t blockId = niin_linear_block_id();
  if (pending != nullptr && blockId < ctx.lsaStorePendingLen) pending[blockId] = 0u;
}

__device__ __forceinline__ void niin_note_gin_op_pending() {
  unsigned int* pending = niin_ctx().ginPendingOps;
  if (pending != nullptr) atomicAdd(pending, 1u);
}

__device__ __forceinline__ void niin_note_gin_op_complete() {
  unsigned int* pending = niin_ctx().ginPendingOps;
  if (pending == nullptr) return;
  unsigned int old = atomicAdd(pending, 0u);
  while (old != 0u) {
    unsigned int prev = atomicCAS(pending, old, old - 1u);
    if (prev == old) return;
    old = prev;
  }
}

__device__ __forceinline__ bool niin_has_gin_ops_pending() {
  unsigned int* pending = niin_ctx().ginPendingOps;
  if (pending == nullptr) return niin_has_gin();
  return *(volatile unsigned int*)pending != 0u;
}

__device__ __forceinline__ unsigned int niin_take_gin_ops_pending() {
  unsigned int* pending = niin_ctx().ginPendingOps;
  if (pending == nullptr) return niin_has_gin() ? 1u : 0u;
  return atomicExch(pending, 0u);
}

// Compute byte offset of a symmetric pointer relative to the heap base.
__device__ __forceinline__ size_t niin_sym_offset(const void* symPtr) {
  return (size_t)((const char*)symPtr - (const char*)niin_ctx().heapBase);
}

// Check whether pe is reachable via NVLink (i.e., is in our LSA team).
__device__ __forceinline__ bool niin_is_lsa_peer(int pe) {
  if (niin_world_is_lsa_only()) return true;
  ncclDevComm const& c = niin_comm();
  if (c.lsaSize <= 1) return false;
  ncclTeam lsa = ncclTeamLsa(c);
  ncclTeam world = ncclTeamWorld(c);
  return ncclTeamRankIsMember(lsa, world, pe);
}

// Check whether peer GPUs support native system-scope atomics.
__device__ __forceinline__ bool niin_peer_native_atomic() {
  return niin_ctx().peerNativeAtomic;
}

__device__ __forceinline__ bool niin_force_separate_put_signal() {
  return niin_ctx().forceSeparatePutSignal;
}

// TMA policy selected at initialization (NVSHMEM_TMA_POLICY).
__device__ __forceinline__ int niin_tma_policy() {
  return niin_ctx().tmaPolicy;
}

// Get a peer pointer for an LSA peer at the given symmetric offset.
__device__ __forceinline__ void* niin_get_peer_ptr(size_t offset, int pe) {
  void** peerHeapBaseP2p = niin_peer_heap_base_p2p();
  if (peerHeapBaseP2p != nullptr) {
    auto base = (char*)__ldg((const unsigned long long*)peerHeapBaseP2p + pe);
    if (base != nullptr) return base + offset;
  }
  return ncclGetPeerPointer(niin_heap_window(), offset, pe);
}

#endif // NIIN_CONTEXT_H_
