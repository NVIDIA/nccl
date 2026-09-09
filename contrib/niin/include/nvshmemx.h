/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

// NIIN: NVSHMEM Implemented In NCCL
//
// Drop-in replacement for nvshmemx.h (extended API).
// nvshmemx_init_attr, nvshmemx_init_attr_t, etc. are defined in nvshmem_host.h
// (included via nvshmem.h).

#ifndef NVSHMEMX_H_NIIN_
#define NVSHMEMX_H_NIIN_

#include "nvshmem.h"

// Warp/block threadgroup RMA variants
#include "niin/threadgroup.h"

// Stream-based operations (kernel-launch wrappers for the device API)
#include "niin/stream.h"

// ---------------------------------------------------------------------------
// nvshmemx_mc_ptr — multicast pointer for a symmetric address within a team.
// Maps to ncclGetLsaMultimemPointer for LSA teams on device.
// Returns nullptr if multimem is not available or team is not LSA-based.
// ---------------------------------------------------------------------------
__host__ __device__ __forceinline__ void* nvshmemx_mc_ptr(nvshmem_team_t team, const void* ptr) {
#ifdef __CUDA_ARCH__
  // Bounds check
  uintptr_t p = reinterpret_cast<uintptr_t>(ptr);
  uintptr_t base = reinterpret_cast<uintptr_t>(niin_heap_base());
  if (p < base || p >= base + niin_heap_size()) return nullptr;
  size_t offset = (size_t)(p - base);

  // Multicast pointers require LSA multimem support
  ncclDevComm const& comm = niin_comm();
  if (comm.lsaMultimem.mcBasePtr == nullptr) return nullptr;

  if (team == NVSHMEM_TEAM_WORLD && comm.lsaSize == comm.nRanks) {
    return ncclGetLsaMultimemPointer(niin_heap_window(), offset, comm);
  }
  if (team == NVSHMEM_TEAM_SHARED || team == NVSHMEMX_TEAM_NODE) {
    return ncclGetLsaMultimemPointer(niin_heap_window(), offset, comm);
  }
  return nullptr;
#else
  // Host-side: multicast pointers not available from host
  (void)team; (void)ptr;
  return nullptr;
#endif
}

// ---------------------------------------------------------------------------
// Stream-based barrier/quiet
// ---------------------------------------------------------------------------
inline int nvshmemx_quiet_on_stream(cudaStream_t stream) {
  niin_stream_quiet_kernel<<<1, 1, 0, stream>>>();
  return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

inline int nvshmemx_barrier_all_on_stream(cudaStream_t stream) {
  auto& s = niin::detail::state();
  if (!s.initialized) return -1;
  ncclAllReduce(s.barrierScratch, s.barrierScratch, 1, ncclInt32, ncclSum, s.comm, stream);
  return 0;
}

// ---------------------------------------------------------------------------
// CUBIN/module management (no-op stubs)
// ---------------------------------------------------------------------------
inline int nvshmemx_cumodule_init(void* module) { (void)module; return 0; }
inline int nvshmemx_cumodule_finalize(void* module) { (void)module; return 0; }
inline int nvshmemx_culibrary_init(void* library) { (void)library; return 0; }
inline int nvshmemx_culibrary_finalize(void* library) { (void)library; return 0; }

// ---------------------------------------------------------------------------
// Collective launch
// ---------------------------------------------------------------------------
inline int nvshmemx_collective_launch(const void* func, dim3 gridDim, dim3 blockDim,
                                       void** args, size_t sharedMem, cudaStream_t stream) {
  return cudaLaunchKernel(func, gridDim, blockDim, args, sharedMem, stream);
}

inline int nvshmemx_collective_launch_query_gridsize(const void* func, dim3 blockDim,
                                                      void** args, size_t sharedMem,
                                                      int* gridSize) {
  *gridSize = 1;
  return 0;
}

// ---------------------------------------------------------------------------
// Buffer registration (stubs — NIIN's heap model handles the common case)
// ---------------------------------------------------------------------------
inline int nvshmemx_buffer_register(void* addr, size_t length) {
  (void)addr; (void)length; return 0;
}
inline int nvshmemx_buffer_unregister(void* addr) {
  (void)addr; return 0;
}
inline void nvshmemx_buffer_unregister_all() {}
inline void* nvshmemx_buffer_register_symmetric(void* buf_ptr, size_t size, int flags) {
  (void)size; (void)flags; return buf_ptr;
}
inline void* nvshmemx_buffer_register_symmetric_at_preferred_address(
    void* buf_ptr, size_t size, void* preferred_addr, int flags) {
  (void)size; (void)preferred_addr; (void)flags; return buf_ptr;
}
inline int nvshmemx_buffer_unregister_symmetric(void* mmap_ptr, size_t size) {
  (void)mmap_ptr; (void)size; return 0;
}

// ---------------------------------------------------------------------------
// nvshmem_global_exit
// ---------------------------------------------------------------------------
inline void nvshmem_global_exit(int status) {
  nvshmem_finalize();
  exit(status);
}

#endif // NVSHMEMX_H_NIIN_
