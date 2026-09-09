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

// The niinContext holds all state needed by device-side NVSHMEM API functions.
// The NVSHMEM host API publishes this pointer with cudaMemcpyToSymbol during
// initialization; low-level NIIN users can still populate and pass it manually.
struct niinContext {
  ncclDevComm const* comm;       // NCCL device communicator
  ncclWindow_t heapWindow;       // The single symmetric heap window
  void* heapBase;                // Local base pointer of the heap
  size_t heapSize;               // Size of the symmetric heap
  int ginContextIndex;           // Which GIN context index to use
  bool peerNativeAtomic;         // True if peer GPUs support native system-scope atomics
  bool forceSeparatePutSignal;   // Force put+fence+signal instead of fused put_signal
};

// Device-side global context pointer used by the header-only device API.
__device__ niinContext* niin_g_ctx;

// ---------------------------------------------------------------------------
// Internal helpers for accessing context fields
// ---------------------------------------------------------------------------

__device__ __forceinline__ ncclDevComm const& niin_comm() {
  return *niin_g_ctx->comm;
}

__device__ __forceinline__ ncclWindow_t niin_heap_window() {
  return niin_g_ctx->heapWindow;
}

__device__ __forceinline__ void* niin_heap_base() {
  return niin_g_ctx->heapBase;
}

__device__ __forceinline__ size_t niin_heap_size() {
  return niin_g_ctx->heapSize;
}

__device__ __forceinline__ int niin_gin_context_index() {
  return niin_g_ctx->ginContextIndex;
}

__device__ __forceinline__ bool niin_has_gin() {
  return niin_comm().ginConnectionCount > 0;
}

// Compute byte offset of a symmetric pointer relative to the heap base.
__device__ __forceinline__ size_t niin_sym_offset(const void* symPtr) {
  return (size_t)((const char*)symPtr - (const char*)niin_g_ctx->heapBase);
}

// Check whether pe is reachable via NVLink (i.e., is in our LSA team).
__device__ __forceinline__ bool niin_is_lsa_peer(int pe) {
  ncclDevComm const& c = niin_comm();
  if (c.lsaSize <= 1) return false;
  ncclTeam lsa = ncclTeamLsa(c);
  ncclTeam world = ncclTeamWorld(c);
  return ncclTeamRankIsMember(lsa, world, pe);
}

// Check whether peer GPUs support native system-scope atomics.
__device__ __forceinline__ bool niin_peer_native_atomic() {
  return niin_g_ctx->peerNativeAtomic;
}

__device__ __forceinline__ bool niin_force_separate_put_signal() {
  return niin_g_ctx->forceSeparatePutSignal;
}

// Get a peer pointer for an LSA peer at the given symmetric offset.
__device__ __forceinline__ void* niin_get_peer_ptr(size_t offset, int pe) {
  return ncclGetPeerPointer(niin_heap_window(), offset, pe);
}

#endif // NIIN_CONTEXT_H_
