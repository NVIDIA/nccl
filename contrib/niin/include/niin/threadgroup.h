/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

// Warp/block threadgroup RMA variants (nvshmemx_*_warp / nvshmemx_*_block).
//
// LSA (NVLink) path: all threads cooperatively copy data via peer pointers
// using vectorized int4 (16-byte) loads/stores for maximum bandwidth.
// Network (GIN) path: thread 0 issues the GIN put/get (single-thread API).

#ifndef NIIN_THREADGROUP_H_
#define NIIN_THREADGROUP_H_

#include "niin/rma.h"
#include "niin/signaling.h"

// ===========================================================================
// Internal: cooperative memcpy kernels using all threads in a group.
// Uses int4 (16-byte) vectorized copies for NVLink bandwidth.
// ===========================================================================

// Internal: cooperative vectorized memcpy with 4x unroll for ILP.
// Each unrolled access is offset by nthreads to maintain coalescing:
//   iter 0: thread k accesses [k], [k+nthreads], [k+2*nthreads], [k+3*nthreads]
// All 4 accesses within an iteration are independently coalesced across the warp.
#define NIIN_COOP_UNROLL 4

__device__ __forceinline__ void niin_coop_copy(
    void* __restrict__ dst, const void* __restrict__ src, size_t bytes,
    int tid, int nthreads) {
  char* d = static_cast<char*>(dst);
  const char* s = static_cast<const char*>(src);
  uintptr_t da = reinterpret_cast<uintptr_t>(d);
  uintptr_t sa = reinterpret_cast<uintptr_t>(s);

  // Only use 16B vector path if src/dst have matching alignment mod 16,
  // so a single head adjustment can align both simultaneously.
  if (((da ^ sa) & 0xF) == 0) {
    // Phase 1: head bytes — align both dst and src to 16 bytes.
    size_t head = (16 - (da & 0xF)) & 0xF;
    if (head > bytes) head = bytes;
    for (size_t j = tid; j < head; j += nthreads)
      d[j] = s[j];
    d += head;
    s += head;
    size_t remaining = bytes - head;

    // Phase 2: vectorized int4 with 4x unroll, all accesses coalesced.
    // Both d and s are now 16-byte aligned.
    size_t n16 = remaining / 16;
    size_t stride = (size_t)nthreads;
    auto* d16 = reinterpret_cast<int4*>(d);
    auto const* s16 = reinterpret_cast<int4 const*>(s);

    size_t i = tid;
    size_t unrolled_end = (n16 >= stride * NIIN_COOP_UNROLL)
        ? n16 - stride * (NIIN_COOP_UNROLL - 1) : 0;
    for (; i < unrolled_end; i += stride * NIIN_COOP_UNROLL) {
      d16[i + stride * 0] = s16[i + stride * 0];
      d16[i + stride * 1] = s16[i + stride * 1];
      d16[i + stride * 2] = s16[i + stride * 2];
      d16[i + stride * 3] = s16[i + stride * 3];
    }
    for (; i < n16; i += stride)
      d16[i] = s16[i];

    d += n16 * 16;
    s += n16 * 16;
    bytes = remaining - n16 * 16;
  }

  // Phase 3: tail bytes, or full byte copy if alignments didn't match.
  for (size_t j = tid; j < bytes; j += nthreads)
    d[j] = s[j];
}

// Cooperative put: all threads copy src -> peer dest via NVLink
__device__ __forceinline__ void niin_coop_put(
    void* dest, const void* src, size_t bytes, int pe,
    int tid, int nthreads) {
  if (pe == niin_device_my_pe()) {
    niin_coop_copy(dest, src, bytes, tid, nthreads);
    return;
  }

  size_t offset = niin_sym_offset(dest);
  if (niin_is_lsa_peer(pe)) {
    void* peerDst = niin_get_peer_ptr(offset, pe);
    niin_coop_copy(peerDst, src, bytes, tid, nthreads);
    // Release fence at system scope to ensure all peer stores are visible
    cuda::atomic_thread_fence(cuda::memory_order_release, cuda::thread_scope_system);
    return;
  }

  // Network: thread 0 does GIN put
  if (tid == 0)
    niin_gin_put(offset, src, bytes, pe);
}

// Cooperative get: all threads copy peer src -> local dest via NVLink
__device__ __forceinline__ void niin_coop_get(
    void* dest, const void* src, size_t bytes, int pe,
    int tid, int nthreads) {
  if (pe == niin_device_my_pe()) {
    niin_coop_copy(dest, src, bytes, tid, nthreads);
    return;
  }

  size_t offset = niin_sym_offset(src);
  if (niin_is_lsa_peer(pe)) {
    const void* peerSrc = niin_get_peer_ptr(offset, pe);
    niin_coop_copy(dest, peerSrc, bytes, tid, nthreads);
    return;
  }

  // Network: thread 0 does GIN get
  if (tid == 0)
    niin_gin_get(dest, offset, bytes, pe);
}

// ===========================================================================
// Typed put warp/block — cooperative NVLink copy
// ===========================================================================
#define NIIN_DEFINE_PUT_WARP(TYPENAME, TYPE)                                   \
__device__ __forceinline__ void nvshmemx_##TYPENAME##_put_warp(               \
    TYPE* dest, const TYPE* src, size_t nelems, int pe) {                     \
  niin_coop_put(dest, src, nelems * sizeof(TYPE), pe,                         \
                nccl::utility::lane(), 32);                                    \
  __syncwarp();                                                                \
}

NIIN_STANDARD_RMA_TYPES(NIIN_DEFINE_PUT_WARP)
#undef NIIN_DEFINE_PUT_WARP

#define NIIN_DEFINE_PUT_BLOCK(TYPENAME, TYPE)                                  \
__device__ __forceinline__ void nvshmemx_##TYPENAME##_put_block(              \
    TYPE* dest, const TYPE* src, size_t nelems, int pe) {                     \
  niin_coop_put(dest, src, nelems * sizeof(TYPE), pe,                         \
                threadIdx.x, blockDim.x);                                      \
  __syncthreads();                                                             \
}

NIIN_STANDARD_RMA_TYPES(NIIN_DEFINE_PUT_BLOCK)
#undef NIIN_DEFINE_PUT_BLOCK

// ===========================================================================
// Typed get warp/block — cooperative NVLink copy
// ===========================================================================
#define NIIN_DEFINE_GET_WARP(TYPENAME, TYPE)                                   \
__device__ __forceinline__ void nvshmemx_##TYPENAME##_get_warp(               \
    TYPE* dest, const TYPE* src, size_t nelems, int pe) {                     \
  niin_coop_get(dest, src, nelems * sizeof(TYPE), pe,                         \
                nccl::utility::lane(), 32);                                    \
  __syncwarp();                                                                \
}

NIIN_STANDARD_RMA_TYPES(NIIN_DEFINE_GET_WARP)
#undef NIIN_DEFINE_GET_WARP

#define NIIN_DEFINE_GET_BLOCK(TYPENAME, TYPE)                                  \
__device__ __forceinline__ void nvshmemx_##TYPENAME##_get_block(              \
    TYPE* dest, const TYPE* src, size_t nelems, int pe) {                     \
  niin_coop_get(dest, src, nelems * sizeof(TYPE), pe,                         \
                threadIdx.x, blockDim.x);                                      \
  __syncthreads();                                                             \
}

NIIN_STANDARD_RMA_TYPES(NIIN_DEFINE_GET_BLOCK)
#undef NIIN_DEFINE_GET_BLOCK

// ===========================================================================
// Sized put/get warp/block
// ===========================================================================
#define NIIN_DEFINE_PUT_SIZED_WARP(SIZE, NBYTES)                               \
__device__ __forceinline__ void nvshmemx_put##SIZE##_warp(                    \
    void* dest, const void* src, size_t nelems, int pe) {                     \
  niin_coop_put(dest, src, nelems * NBYTES, pe,                               \
                nccl::utility::lane(), 32);                                    \
  __syncwarp();                                                                \
}

NIIN_SIZED_RMA(NIIN_DEFINE_PUT_SIZED_WARP)
#undef NIIN_DEFINE_PUT_SIZED_WARP

#define NIIN_DEFINE_PUT_SIZED_BLOCK(SIZE, NBYTES)                              \
__device__ __forceinline__ void nvshmemx_put##SIZE##_block(                   \
    void* dest, const void* src, size_t nelems, int pe) {                     \
  niin_coop_put(dest, src, nelems * NBYTES, pe,                               \
                threadIdx.x, blockDim.x);                                      \
  __syncthreads();                                                             \
}

NIIN_SIZED_RMA(NIIN_DEFINE_PUT_SIZED_BLOCK)
#undef NIIN_DEFINE_PUT_SIZED_BLOCK

#define NIIN_DEFINE_GET_SIZED_WARP(SIZE, NBYTES)                               \
__device__ __forceinline__ void nvshmemx_get##SIZE##_warp(                    \
    void* dest, const void* src, size_t nelems, int pe) {                     \
  niin_coop_get(dest, src, nelems * NBYTES, pe,                               \
                nccl::utility::lane(), 32);                                    \
  __syncwarp();                                                                \
}

NIIN_SIZED_RMA(NIIN_DEFINE_GET_SIZED_WARP)
#undef NIIN_DEFINE_GET_SIZED_WARP

#define NIIN_DEFINE_GET_SIZED_BLOCK(SIZE, NBYTES)                              \
__device__ __forceinline__ void nvshmemx_get##SIZE##_block(                   \
    void* dest, const void* src, size_t nelems, int pe) {                     \
  niin_coop_get(dest, src, nelems * NBYTES, pe,                               \
                threadIdx.x, blockDim.x);                                      \
  __syncthreads();                                                             \
}

NIIN_SIZED_RMA(NIIN_DEFINE_GET_SIZED_BLOCK)
#undef NIIN_DEFINE_GET_SIZED_BLOCK

// ===========================================================================
// putmem/getmem warp/block
// ===========================================================================
__device__ __forceinline__ void nvshmemx_putmem_warp(void* dest, const void* src, size_t bytes, int pe) {
  niin_coop_put(dest, src, bytes, pe, nccl::utility::lane(), 32);
  __syncwarp();
}
__device__ __forceinline__ void nvshmemx_putmem_block(void* dest, const void* src, size_t bytes, int pe) {
  niin_coop_put(dest, src, bytes, pe, threadIdx.x, blockDim.x);
  __syncthreads();
}
__device__ __forceinline__ void nvshmemx_getmem_warp(void* dest, const void* src, size_t bytes, int pe) {
  niin_coop_get(dest, src, bytes, pe, nccl::utility::lane(), 32);
  __syncwarp();
}
__device__ __forceinline__ void nvshmemx_getmem_block(void* dest, const void* src, size_t bytes, int pe) {
  niin_coop_get(dest, src, bytes, pe, threadIdx.x, blockDim.x);
  __syncthreads();
}

// ===========================================================================
// NBI put/get warp/block (same as blocking — already async for NVLink)
// ===========================================================================
#define NIIN_DEFINE_PUT_NBI_WARP(TYPENAME, TYPE)                               \
__device__ __forceinline__ void nvshmemx_##TYPENAME##_put_nbi_warp(           \
    TYPE* dest, const TYPE* src, size_t nelems, int pe) {                     \
  nvshmemx_##TYPENAME##_put_warp(dest, src, nelems, pe);                      \
}

NIIN_STANDARD_RMA_TYPES(NIIN_DEFINE_PUT_NBI_WARP)
#undef NIIN_DEFINE_PUT_NBI_WARP

#define NIIN_DEFINE_PUT_NBI_BLOCK(TYPENAME, TYPE)                              \
__device__ __forceinline__ void nvshmemx_##TYPENAME##_put_nbi_block(          \
    TYPE* dest, const TYPE* src, size_t nelems, int pe) {                     \
  nvshmemx_##TYPENAME##_put_block(dest, src, nelems, pe);                     \
}

NIIN_STANDARD_RMA_TYPES(NIIN_DEFINE_PUT_NBI_BLOCK)
#undef NIIN_DEFINE_PUT_NBI_BLOCK

#define NIIN_DEFINE_GET_NBI_WARP(TYPENAME, TYPE)                               \
__device__ __forceinline__ void nvshmemx_##TYPENAME##_get_nbi_warp(           \
    TYPE* dest, const TYPE* src, size_t nelems, int pe) {                     \
  nvshmemx_##TYPENAME##_get_warp(dest, src, nelems, pe);                      \
}

NIIN_STANDARD_RMA_TYPES(NIIN_DEFINE_GET_NBI_WARP)
#undef NIIN_DEFINE_GET_NBI_WARP

#define NIIN_DEFINE_GET_NBI_BLOCK(TYPENAME, TYPE)                              \
__device__ __forceinline__ void nvshmemx_##TYPENAME##_get_nbi_block(          \
    TYPE* dest, const TYPE* src, size_t nelems, int pe) {                     \
  nvshmemx_##TYPENAME##_get_block(dest, src, nelems, pe);                     \
}

NIIN_STANDARD_RMA_TYPES(NIIN_DEFINE_GET_NBI_BLOCK)
#undef NIIN_DEFINE_GET_NBI_BLOCK

// NBI putmem/getmem warp/block
__device__ __forceinline__ void nvshmemx_putmem_nbi_warp(void* dest, const void* src, size_t bytes, int pe) {
  nvshmemx_putmem_warp(dest, src, bytes, pe);
}
__device__ __forceinline__ void nvshmemx_putmem_nbi_block(void* dest, const void* src, size_t bytes, int pe) {
  nvshmemx_putmem_block(dest, src, bytes, pe);
}
__device__ __forceinline__ void nvshmemx_getmem_nbi_warp(void* dest, const void* src, size_t bytes, int pe) {
  nvshmemx_getmem_warp(dest, src, bytes, pe);
}
__device__ __forceinline__ void nvshmemx_getmem_nbi_block(void* dest, const void* src, size_t bytes, int pe) {
  nvshmemx_getmem_block(dest, src, bytes, pe);
}

// NBI sized put/get warp/block
#define NIIN_DEFINE_PUT_SIZED_NBI_WARP(SIZE, NBYTES)                           \
__device__ __forceinline__ void nvshmemx_put##SIZE##_nbi_warp(                \
    void* dest, const void* src, size_t nelems, int pe) {                     \
  nvshmemx_put##SIZE##_warp(dest, src, nelems, pe);                           \
}

NIIN_SIZED_RMA(NIIN_DEFINE_PUT_SIZED_NBI_WARP)
#undef NIIN_DEFINE_PUT_SIZED_NBI_WARP

#define NIIN_DEFINE_PUT_SIZED_NBI_BLOCK(SIZE, NBYTES)                          \
__device__ __forceinline__ void nvshmemx_put##SIZE##_nbi_block(               \
    void* dest, const void* src, size_t nelems, int pe) {                     \
  nvshmemx_put##SIZE##_block(dest, src, nelems, pe);                          \
}

NIIN_SIZED_RMA(NIIN_DEFINE_PUT_SIZED_NBI_BLOCK)
#undef NIIN_DEFINE_PUT_SIZED_NBI_BLOCK

#define NIIN_DEFINE_GET_SIZED_NBI_WARP(SIZE, NBYTES)                           \
__device__ __forceinline__ void nvshmemx_get##SIZE##_nbi_warp(                \
    void* dest, const void* src, size_t nelems, int pe) {                     \
  nvshmemx_get##SIZE##_warp(dest, src, nelems, pe);                           \
}

NIIN_SIZED_RMA(NIIN_DEFINE_GET_SIZED_NBI_WARP)
#undef NIIN_DEFINE_GET_SIZED_NBI_WARP

#define NIIN_DEFINE_GET_SIZED_NBI_BLOCK(SIZE, NBYTES)                          \
__device__ __forceinline__ void nvshmemx_get##SIZE##_nbi_block(               \
    void* dest, const void* src, size_t nelems, int pe) {                     \
  nvshmemx_get##SIZE##_block(dest, src, nelems, pe);                          \
}

NIIN_SIZED_RMA(NIIN_DEFINE_GET_SIZED_NBI_BLOCK)
#undef NIIN_DEFINE_GET_SIZED_NBI_BLOCK

// ===========================================================================
// Strided iput/iget warp/block — cooperative strided copy
// ===========================================================================
#define NIIN_DEFINE_IPUT_WARP(TYPENAME, TYPE)                                  \
__device__ __forceinline__ void nvshmemx_##TYPENAME##_iput_warp(              \
    TYPE* dest, const TYPE* src, ptrdiff_t dst, ptrdiff_t sst,                \
    size_t nelems, int pe) {                                                   \
  if (nccl::utility::lane() == 0) nvshmem_##TYPENAME##_iput(dest, src, dst, sst, nelems, pe); \
  __syncwarp();                                                                \
}

NIIN_STANDARD_RMA_TYPES(NIIN_DEFINE_IPUT_WARP)
#undef NIIN_DEFINE_IPUT_WARP

#define NIIN_DEFINE_IPUT_BLOCK(TYPENAME, TYPE)                                 \
__device__ __forceinline__ void nvshmemx_##TYPENAME##_iput_block(             \
    TYPE* dest, const TYPE* src, ptrdiff_t dst, ptrdiff_t sst,                \
    size_t nelems, int pe) {                                                   \
  if (threadIdx.x == 0) nvshmem_##TYPENAME##_iput(dest, src, dst, sst, nelems, pe); \
  __syncthreads();                                                             \
}

NIIN_STANDARD_RMA_TYPES(NIIN_DEFINE_IPUT_BLOCK)
#undef NIIN_DEFINE_IPUT_BLOCK

#define NIIN_DEFINE_IGET_WARP(TYPENAME, TYPE)                                  \
__device__ __forceinline__ void nvshmemx_##TYPENAME##_iget_warp(              \
    TYPE* dest, const TYPE* src, ptrdiff_t dst, ptrdiff_t sst,                \
    size_t nelems, int pe) {                                                   \
  if (nccl::utility::lane() == 0) nvshmem_##TYPENAME##_iget(dest, src, dst, sst, nelems, pe); \
  __syncwarp();                                                                \
}

NIIN_STANDARD_RMA_TYPES(NIIN_DEFINE_IGET_WARP)
#undef NIIN_DEFINE_IGET_WARP

#define NIIN_DEFINE_IGET_BLOCK(TYPENAME, TYPE)                                 \
__device__ __forceinline__ void nvshmemx_##TYPENAME##_iget_block(             \
    TYPE* dest, const TYPE* src, ptrdiff_t dst, ptrdiff_t sst,                \
    size_t nelems, int pe) {                                                   \
  if (threadIdx.x == 0) nvshmem_##TYPENAME##_iget(dest, src, dst, sst, nelems, pe); \
  __syncthreads();                                                             \
}

NIIN_STANDARD_RMA_TYPES(NIIN_DEFINE_IGET_BLOCK)
#undef NIIN_DEFINE_IGET_BLOCK

// ===========================================================================
// put_signal warp/block — cooperative copy + signal
// ===========================================================================
#define NIIN_DEFINE_PUT_SIGNAL_WARP(TYPENAME, TYPE)                            \
__device__ __forceinline__ void nvshmemx_##TYPENAME##_put_signal_warp(        \
    TYPE* dest, const TYPE* src, size_t nelems,                               \
    uint64_t* sig, uint64_t signal, int sig_op, int pe) {                    \
  /* Re-converge the warp before reading src: callers often have only lane 0 */ \
  /* wait on the inbound signal before forwarding the buffer. */              \
  __syncwarp();                                                                \
  cuda::atomic_thread_fence(cuda::memory_order_acquire, cuda::thread_scope_system); \
  niin_coop_put(dest, src, nelems * sizeof(TYPE), pe,                         \
                nccl::utility::lane(), 32);                                    \
  if (nccl::utility::lane() == 0 && !niin_is_lsa_peer(pe)) niin_gin_flush_thread(); \
  cuda::atomic_thread_fence(cuda::memory_order_release, cuda::thread_scope_system); \
  __syncwarp();                                                                \
  if (nccl::utility::lane() == 0) niin_deliver_signal(sig, signal, sig_op, pe); \
  __syncwarp();                                                                \
}

NIIN_STANDARD_RMA_TYPES(NIIN_DEFINE_PUT_SIGNAL_WARP)
#undef NIIN_DEFINE_PUT_SIGNAL_WARP

#define NIIN_DEFINE_PUT_SIGNAL_BLOCK(TYPENAME, TYPE)                           \
__device__ __forceinline__ void nvshmemx_##TYPENAME##_put_signal_block(       \
    TYPE* dest, const TYPE* src, size_t nelems,                               \
    uint64_t* sig, uint64_t signal, int sig_op, int pe) {                    \
  /* Re-converge the CTA before reading src: callers often have only thread 0 */ \
  /* wait on the inbound signal before forwarding the buffer. */              \
  __syncthreads();                                                             \
  cuda::atomic_thread_fence(cuda::memory_order_acquire, cuda::thread_scope_system); \
  niin_coop_put(dest, src, nelems * sizeof(TYPE), pe,                         \
                threadIdx.x, blockDim.x);                                      \
  if (threadIdx.x == 0 && !niin_is_lsa_peer(pe)) niin_gin_flush_thread();     \
  cuda::atomic_thread_fence(cuda::memory_order_release, cuda::thread_scope_system); \
  __syncthreads();                                                             \
  if (threadIdx.x == 0) niin_deliver_signal(sig, signal, sig_op, pe);         \
  __syncthreads();                                                             \
}

NIIN_STANDARD_RMA_TYPES(NIIN_DEFINE_PUT_SIGNAL_BLOCK)
#undef NIIN_DEFINE_PUT_SIGNAL_BLOCK

// putmem_signal warp/block
__device__ __forceinline__ void nvshmemx_putmem_signal_warp(
    void* dest, const void* src, size_t bytes,
    uint64_t* sig, uint64_t signal, int sig_op, int pe) {
  __syncwarp();
  cuda::atomic_thread_fence(cuda::memory_order_acquire, cuda::thread_scope_system);
  niin_coop_put(dest, src, bytes, pe, nccl::utility::lane(), 32);
  if (nccl::utility::lane() == 0 && !niin_is_lsa_peer(pe)) niin_gin_flush_thread();
  cuda::atomic_thread_fence(cuda::memory_order_release, cuda::thread_scope_system);
  __syncwarp();
  if (nccl::utility::lane() == 0) niin_deliver_signal(sig, signal, sig_op, pe);
  __syncwarp();
}
__device__ __forceinline__ void nvshmemx_putmem_signal_block(
    void* dest, const void* src, size_t bytes,
    uint64_t* sig, uint64_t signal, int sig_op, int pe) {
  __syncthreads();
  cuda::atomic_thread_fence(cuda::memory_order_acquire, cuda::thread_scope_system);
  niin_coop_put(dest, src, bytes, pe, threadIdx.x, blockDim.x);
  if (threadIdx.x == 0 && !niin_is_lsa_peer(pe)) niin_gin_flush_thread();
  cuda::atomic_thread_fence(cuda::memory_order_release, cuda::thread_scope_system);
  __syncthreads();
  if (threadIdx.x == 0) niin_deliver_signal(sig, signal, sig_op, pe);
  __syncthreads();
}

// put_signal_nbi warp/block (same as blocking)
#define NIIN_DEFINE_PUT_SIGNAL_NBI_WARP(TYPENAME, TYPE)                        \
__device__ __forceinline__ void nvshmemx_##TYPENAME##_put_signal_nbi_warp(    \
    TYPE* dest, const TYPE* src, size_t nelems,                               \
    uint64_t* sig, uint64_t signal, int sig_op, int pe) {                    \
  nvshmemx_##TYPENAME##_put_signal_warp(dest, src, nelems, sig, signal, sig_op, pe); \
}

NIIN_STANDARD_RMA_TYPES(NIIN_DEFINE_PUT_SIGNAL_NBI_WARP)
#undef NIIN_DEFINE_PUT_SIGNAL_NBI_WARP

#define NIIN_DEFINE_PUT_SIGNAL_NBI_BLOCK(TYPENAME, TYPE)                       \
__device__ __forceinline__ void nvshmemx_##TYPENAME##_put_signal_nbi_block(   \
    TYPE* dest, const TYPE* src, size_t nelems,                               \
    uint64_t* sig, uint64_t signal, int sig_op, int pe) {                    \
  nvshmemx_##TYPENAME##_put_signal_block(dest, src, nelems, sig, signal, sig_op, pe); \
}

NIIN_STANDARD_RMA_TYPES(NIIN_DEFINE_PUT_SIGNAL_NBI_BLOCK)
#undef NIIN_DEFINE_PUT_SIGNAL_NBI_BLOCK

__device__ __forceinline__ void nvshmemx_putmem_signal_nbi_warp(
    void* dest, const void* src, size_t bytes,
    uint64_t* sig, uint64_t signal, int sig_op, int pe) {
  nvshmemx_putmem_signal_warp(dest, src, bytes, sig, signal, sig_op, pe);
}
__device__ __forceinline__ void nvshmemx_putmem_signal_nbi_block(
    void* dest, const void* src, size_t bytes,
    uint64_t* sig, uint64_t signal, int sig_op, int pe) {
  nvshmemx_putmem_signal_block(dest, src, bytes, sig, signal, sig_op, pe);
}

// Sized put_signal warp/block
#define NIIN_DEFINE_PUT_SIGNAL_SIZED_WARP(SIZE, NBYTES)                        \
__device__ __forceinline__ void nvshmemx_put##SIZE##_signal_warp(              \
    void* dest, const void* src, size_t nelems,                               \
    uint64_t* sig, uint64_t signal, int sig_op, int pe) {                    \
  nvshmemx_putmem_signal_warp(dest, src, nelems * NBYTES, sig, signal, sig_op, pe); \
}

NIIN_SIZED_RMA(NIIN_DEFINE_PUT_SIGNAL_SIZED_WARP)
#undef NIIN_DEFINE_PUT_SIGNAL_SIZED_WARP

#define NIIN_DEFINE_PUT_SIGNAL_SIZED_BLOCK(SIZE, NBYTES)                       \
__device__ __forceinline__ void nvshmemx_put##SIZE##_signal_block(             \
    void* dest, const void* src, size_t nelems,                               \
    uint64_t* sig, uint64_t signal, int sig_op, int pe) {                    \
  nvshmemx_putmem_signal_block(dest, src, nelems * NBYTES, sig, signal, sig_op, pe); \
}

NIIN_SIZED_RMA(NIIN_DEFINE_PUT_SIGNAL_SIZED_BLOCK)
#undef NIIN_DEFINE_PUT_SIGNAL_SIZED_BLOCK

// Sized put_signal_nbi warp/block
#define NIIN_DEFINE_PUT_SIGNAL_SIZED_NBI_WARP(SIZE, NBYTES)                    \
__device__ __forceinline__ void nvshmemx_put##SIZE##_signal_nbi_warp(          \
    void* dest, const void* src, size_t nelems,                               \
    uint64_t* sig, uint64_t signal, int sig_op, int pe) {                    \
  nvshmemx_put##SIZE##_signal_warp(dest, src, nelems, sig, signal, sig_op, pe); \
}

NIIN_SIZED_RMA(NIIN_DEFINE_PUT_SIGNAL_SIZED_NBI_WARP)
#undef NIIN_DEFINE_PUT_SIGNAL_SIZED_NBI_WARP

#define NIIN_DEFINE_PUT_SIGNAL_SIZED_NBI_BLOCK(SIZE, NBYTES)                   \
__device__ __forceinline__ void nvshmemx_put##SIZE##_signal_nbi_block(         \
    void* dest, const void* src, size_t nelems,                               \
    uint64_t* sig, uint64_t signal, int sig_op, int pe) {                    \
  nvshmemx_put##SIZE##_signal_block(dest, src, nelems, sig, signal, sig_op, pe); \
}

NIIN_SIZED_RMA(NIIN_DEFINE_PUT_SIGNAL_SIZED_NBI_BLOCK)
#undef NIIN_DEFINE_PUT_SIGNAL_SIZED_NBI_BLOCK

#endif // NIIN_THREADGROUP_H_
