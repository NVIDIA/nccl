/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

// Host-side stream-based NVSHMEM operations (the nvshmemx_*_on_stream family).
//
// Callers supply only a CUDA stream, matching NVSHMEM's _on_stream contract.
// NIIN owns the internal launch geometry: block RMA fallbacks choose a
// size-based number of eight-warp CTAs for self/LSA transfers, while scalar,
// strided, signal, and wait operations use compact control kernels.

#ifndef NIIN_STREAM_H_
#define NIIN_STREAM_H_

#include <climits>
#include <cstdlib>

#include "niin/context.h"
#include "niin/rma.h"
#include "niin/signaling.h"
#include "niin/sync.h"
#include "niin/threadgroup.h"

static constexpr int kNiinStreamThreadsPerCta = 256;
static constexpr int kNiinStreamDefaultMaxCtas = 16;
static constexpr size_t kNiinStreamTargetBytesPerCta = 64 * 1024;

inline int niin_stream_max_ctas() {
  static int maxCtas = []() {
    const char* env = std::getenv("NVSHMEM_MAX_CTAS");
    if (env == nullptr || *env == '\0') return kNiinStreamDefaultMaxCtas;
    char* end = nullptr;
    long val = std::strtol(env, &end, 10);
    if (end == env || val <= 0) return kNiinStreamDefaultMaxCtas;
    if (val > INT_MAX) return INT_MAX;
    return static_cast<int>(val);
  }();
  return maxCtas;
}

inline int niin_stream_cta_count(size_t bytes) {
  if (bytes == 0) return 1;
  int maxCtas = niin_stream_max_ctas();
  size_t ctas = (bytes + kNiinStreamTargetBytesPerCta - 1) /
                kNiinStreamTargetBytesPerCta;
  if (ctas < 1) ctas = 1;
  if (ctas > static_cast<size_t>(maxCtas)) ctas = static_cast<size_t>(maxCtas);
  return static_cast<int>(ctas);
}

__device__ __forceinline__ bool niin_stream_cta_chunk(
    size_t bytes, int pe, size_t* offset, size_t* chunkBytes) {
  if (bytes == 0) {
    *offset = 0;
    *chunkBytes = 0;
    return false;
  }

  // One GIN context is currently requested, so keep non-LSA network transfers
  // single-CTA until NIIN can assign independent GIN contexts per CTA.
  bool canSplit = (pe == nvshmem_my_pe()) || niin_is_lsa_peer(pe);
  if (!canSplit) {
    if (blockIdx.x != 0) return false;
    *offset = 0;
    *chunkBytes = bytes;
    return true;
  }

  size_t nCtas = static_cast<size_t>(gridDim.x);
  size_t cta = static_cast<size_t>(blockIdx.x);
  size_t base = bytes / nCtas;
  size_t rem = bytes - base * nCtas;
  *offset = cta * base + (cta < rem ? cta : rem);
  *chunkBytes = base + (cta < rem ? 1 : 0);
  return *chunkBytes != 0;
}

// ===========================================================================
// Internal kernel templates
// ===========================================================================

// ---------------------------------------------------------------------------
// Scalar p (put one element)
// ---------------------------------------------------------------------------
template<typename T>
__global__ void niin_stream_p_kernel(T* dest, T value, int pe) {
  nvshmem_putmem(dest, &value, sizeof(T), pe);
}

// ---------------------------------------------------------------------------
// Scalar g (get one element, store to output buffer)
// ---------------------------------------------------------------------------
template<typename T>
__global__ void niin_stream_g_kernel(const T* src, int pe, T* out) {
  nvshmem_getmem(out, src, sizeof(T), pe);
}

// ---------------------------------------------------------------------------
// Put / Get (block transfer)
// ---------------------------------------------------------------------------
template<typename T>
__global__ void niin_stream_put_kernel(T* dest, const T* src, size_t nelems, int pe) {
  size_t offset, chunkBytes;
  size_t bytes = nelems * sizeof(T);
  if (!niin_stream_cta_chunk(bytes, pe, &offset, &chunkBytes)) return;
  nvshmemx_putmem_block(
      reinterpret_cast<char*>(dest) + offset,
      reinterpret_cast<const char*>(src) + offset,
      chunkBytes, pe);
}

template<typename T>
__global__ void niin_stream_get_kernel(T* dest, const T* src, size_t nelems, int pe) {
  size_t offset, chunkBytes;
  size_t bytes = nelems * sizeof(T);
  if (!niin_stream_cta_chunk(bytes, pe, &offset, &chunkBytes)) return;
  nvshmemx_getmem_block(
      reinterpret_cast<char*>(dest) + offset,
      reinterpret_cast<const char*>(src) + offset,
      chunkBytes, pe);
}

// ---------------------------------------------------------------------------
// Sized put / get (untyped, nbytes = nelems * element_size)
// Dummy template parameter avoids ODR issues in header-only usage.
// ---------------------------------------------------------------------------
template<int = 0>
__global__ void niin_stream_putmem_kernel(void* dest, const void* src, size_t bytes, int pe) {
  size_t offset, chunkBytes;
  if (!niin_stream_cta_chunk(bytes, pe, &offset, &chunkBytes)) return;
  nvshmemx_putmem_block(
      static_cast<char*>(dest) + offset,
      static_cast<const char*>(src) + offset,
      chunkBytes, pe);
}

template<int = 0>
__global__ void niin_stream_getmem_kernel(void* dest, const void* src, size_t bytes, int pe) {
  size_t offset, chunkBytes;
  if (!niin_stream_cta_chunk(bytes, pe, &offset, &chunkBytes)) return;
  nvshmemx_getmem_block(
      static_cast<char*>(dest) + offset,
      static_cast<const char*>(src) + offset,
      chunkBytes, pe);
}

// ---------------------------------------------------------------------------
// Strided iput / iget
// ---------------------------------------------------------------------------
template<typename T>
__global__ void niin_stream_iput_kernel(T* dest, const T* src,
                                        ptrdiff_t dst, ptrdiff_t sst,
                                        size_t nelems, int pe) {
  size_t offset = niin_sym_offset(dest);
  int myPe = nvshmem_my_pe();
  if (pe == myPe) {
    for (size_t i = 0; i < nelems; i++)
      ((volatile T*)dest)[i * dst] = src[i * sst];
    return;
  }
  if (niin_is_lsa_peer(pe)) {
    volatile T* d = (volatile T*)niin_get_peer_ptr(offset, pe);
    for (size_t i = 0; i < nelems; i++)
      d[i * dst] = src[i * sst];
    return;
  }
  // Fall back to element-wise puts for network path
  for (size_t i = 0; i < nelems; i++) {
    T val = src[i * sst];
    nvshmem_putmem(&dest[i * dst], &val, sizeof(T), pe);
  }
}

template<typename T>
__global__ void niin_stream_iget_kernel(T* dest, const T* src,
                                        ptrdiff_t dst, ptrdiff_t sst,
                                        size_t nelems, int pe) {
  size_t offset = niin_sym_offset(src);
  int myPe = nvshmem_my_pe();
  if (pe == myPe) {
    for (size_t i = 0; i < nelems; i++)
      dest[i * dst] = ((volatile const T*)src)[i * sst];
    return;
  }
  if (niin_is_lsa_peer(pe)) {
    volatile const T* s = (volatile const T*)niin_get_peer_ptr(offset, pe);
    for (size_t i = 0; i < nelems; i++)
      dest[i * dst] = s[i * sst];
    return;
  }
  // Fall back to element-wise gets for network path
  for (size_t i = 0; i < nelems; i++) {
    nvshmem_getmem(&dest[i * dst], &src[i * sst], sizeof(T), pe);
  }
}

// ---------------------------------------------------------------------------
// Put signal
// ---------------------------------------------------------------------------
template<typename T>
__global__ void niin_stream_put_signal_kernel(T* dest, const T* src, size_t nelems, int pe) {
  size_t offset, chunkBytes;
  size_t bytes = nelems * sizeof(T);
  if (!niin_stream_cta_chunk(bytes, pe, &offset, &chunkBytes)) return;
  nvshmemx_putmem_block(
      reinterpret_cast<char*>(dest) + offset,
      reinterpret_cast<const char*>(src) + offset,
      chunkBytes, pe);
}

template<int = 0>
__global__ void niin_stream_putmem_signal_kernel(void* dest, const void* src, size_t bytes, int pe) {
  size_t offset, chunkBytes;
  if (!niin_stream_cta_chunk(bytes, pe, &offset, &chunkBytes)) return;
  nvshmemx_putmem_block(
      static_cast<char*>(dest) + offset,
      static_cast<const char*>(src) + offset,
      chunkBytes, pe);
}

template<int = 0>
__global__ void niin_stream_put_signal_complete_kernel(uint64_t* sig_addr, uint64_t signal,
                                                       int sig_op, int pe) {
  if (threadIdx.x == 0) {
    __threadfence_system();
    if (pe != nvshmem_my_pe() && !niin_is_lsa_peer(pe))
      niin_gin_flush_thread();
    niin_deliver_signal(sig_addr, signal, sig_op, pe);
  }
}

// ---------------------------------------------------------------------------
// Signal op
// ---------------------------------------------------------------------------
template<int = 0>
__global__ void niin_stream_signal_op_kernel(uint64_t* sig_addr, uint64_t val,
                                             int sig_op, int pe) {
  size_t sigOffset = niin_sym_offset(sig_addr);
  int myPe = nvshmem_my_pe();
  uint64_t* target;
  if (pe == myPe) {
    target = sig_addr;
  } else if (niin_is_lsa_peer(pe)) {
    target = (uint64_t*)niin_get_peer_ptr(sigOffset, pe);
  } else {
    target = sig_addr; // fallback
  }
  if (sig_op == NVSHMEM_SIGNAL_SET)
    atomicExch((unsigned long long*)target, val);
  else
    atomicAdd((unsigned long long*)target, val);
}

// ---------------------------------------------------------------------------
// Quiet
// ---------------------------------------------------------------------------
template<int = 0>
__global__ void niin_stream_quiet_kernel() {
  niin_device_quiet();
}

// ---------------------------------------------------------------------------
// Signal wait until
// ---------------------------------------------------------------------------
template<int = 0>
__global__ void niin_stream_signal_wait_until_kernel(uint64_t* sig_addr, int cmp,
                                                     uint64_t cmp_value, uint64_t* out) {
  *out = nvshmem_signal_wait_until(sig_addr, cmp, cmp_value);
}

// ---------------------------------------------------------------------------
// Wait until (typed)
// ---------------------------------------------------------------------------
template<typename T>
__global__ void niin_stream_wait_until_kernel(T* ivar, int cmp, T cmp_value) {
  volatile T* v = (volatile T*)ivar;
  while (!niin_cmp_eval(cmp, (long long)*v, (long long)cmp_value)) {}
}

// ---------------------------------------------------------------------------
// Wait until all (typed)
// ---------------------------------------------------------------------------
template<typename T>
__global__ void niin_stream_wait_until_all_kernel(T* ivars, size_t nelems,
                                                  const int* status,
                                                  int cmp, T cmp_value) {
  for (size_t i = 0; i < nelems; i++) {
    if (status != nullptr && status[i] != 0) continue;
    volatile T* v = (volatile T*)&ivars[i];
    while (!niin_cmp_eval(cmp, (long long)*v, (long long)cmp_value)) {}
  }
}

// ---------------------------------------------------------------------------
// Wait until all vector (typed)
// ---------------------------------------------------------------------------
template<typename T>
__global__ void niin_stream_wait_until_all_vector_kernel(T* ivars, size_t nelems,
                                                         const int* status,
                                                         int cmp, T* cmp_values) {
  for (size_t i = 0; i < nelems; i++) {
    if (status != nullptr && status[i] != 0) continue;
    volatile T* v = (volatile T*)&ivars[i];
    while (!niin_cmp_eval(cmp, (long long)*v, (long long)cmp_values[i])) {}
  }
}

// ===========================================================================
// Host-side _on_stream wrappers
//
// When host RMA is available (ncclPutSignal), puts go directly through NCCL's
// host RMA path for best performance. Gets and waits still use kernel launches.
// ===========================================================================

// ---------------------------------------------------------------------------
// Internal: host RMA put helper. Uses ncclPutSignal when available,
// falls back to kernel launch.
// ---------------------------------------------------------------------------
inline void niin_host_rma_put(void* dest, const void* src, size_t bytes,
                               int pe, cudaStream_t stream) {
  auto& s = niin::detail::state();
  if (s.hostRmaAvail) {
    size_t offset = (size_t)((char*)dest - (char*)s.heapBase);
    ncclPutSignal(src, bytes, ncclUint8, pe,
                  s.hostCtx.heapWindow, offset,
                  -1 /*no signal*/, s.rmaCtx, 0 /*flags*/, s.comm, stream);
  } else {
    int ctas = niin_stream_cta_count(bytes);
    niin_stream_putmem_kernel<<<ctas, kNiinStreamThreadsPerCta, 0, stream>>>(
        (void*)dest, src, bytes, pe);
  }
}

// Internal: host RMA put + signal helper.
inline void niin_host_rma_put_signal(void* dest, const void* src, size_t bytes,
                                      uint64_t* sig_addr, uint64_t signal,
                                      int sig_op, int pe, cudaStream_t stream) {
  // Host RMA only supports indexed signals, while NVSHMEM put_signal requires
  // arbitrary signal addresses and operations. Use the device path for
  // correctness until a matching host-side VA signal path exists.
  int ctas = niin_stream_cta_count(bytes);
  niin_stream_putmem_signal_kernel<<<ctas, kNiinStreamThreadsPerCta, 0, stream>>>(
      (void*)dest, src, bytes, pe);
  niin_stream_put_signal_complete_kernel<<<1, 1, 0, stream>>>(
      sig_addr, signal, sig_op, pe);
}

// ---------------------------------------------------------------------------
// 1. Scalar p on stream (typed)
// ---------------------------------------------------------------------------
#define NIIN_DEFINE_P_ON_STREAM(TYPENAME, TYPE)                                \
inline void nvshmemx_##TYPENAME##_p_on_stream(TYPE* dest, TYPE value,          \
                                               int pe, cudaStream_t stream) {  \
  auto& s = niin::detail::state();                                             \
  if (s.hostRmaAvail) {                                                        \
    size_t offset = (size_t)((char*)dest - (char*)s.heapBase);                \
    TYPE tmp = value;                                                           \
    ncclPutSignal(&tmp, sizeof(TYPE), ncclUint8, pe,                          \
                  s.hostCtx.heapWindow, offset,                                \
                  -1, s.rmaCtx, 0, s.comm, stream);                           \
  } else {                                                                     \
    niin_stream_p_kernel<TYPE><<<1, 1, 0, stream>>>(dest, value, pe);         \
  }                                                                            \
}

NIIN_STANDARD_RMA_TYPES(NIIN_DEFINE_P_ON_STREAM)
#undef NIIN_DEFINE_P_ON_STREAM

// ---------------------------------------------------------------------------
// 2. Scalar g on stream (typed)
//    Synchronous: allocates temp device buffer, launches kernel, copies back.
// ---------------------------------------------------------------------------
#define NIIN_DEFINE_G_ON_STREAM(TYPENAME, TYPE)                                \
inline TYPE nvshmemx_##TYPENAME##_g_on_stream(const TYPE* src, int pe,         \
                                               cudaStream_t stream) {          \
  TYPE result;                                                                  \
  TYPE* d_tmp;                                                                  \
  cudaMalloc(&d_tmp, sizeof(TYPE));                                            \
  niin_stream_g_kernel<TYPE><<<1, 1, 0, stream>>>(src, pe, d_tmp);            \
  cudaMemcpyAsync(&result, d_tmp, sizeof(TYPE), cudaMemcpyDeviceToHost, stream); \
  cudaStreamSynchronize(stream);                                               \
  cudaFree(d_tmp);                                                             \
  return result;                                                                \
}

NIIN_STANDARD_RMA_TYPES(NIIN_DEFINE_G_ON_STREAM)
#undef NIIN_DEFINE_G_ON_STREAM

// ---------------------------------------------------------------------------
// 3. Typed put on stream
// ---------------------------------------------------------------------------
#define NIIN_DEFINE_PUT_ON_STREAM(TYPENAME, TYPE)                              \
inline void nvshmemx_##TYPENAME##_put_on_stream(TYPE* dest, const TYPE* src,   \
    size_t nelems, int pe, cudaStream_t stream) {                              \
  niin_host_rma_put(dest, src, nelems * sizeof(TYPE), pe, stream);            \
}

NIIN_STANDARD_RMA_TYPES(NIIN_DEFINE_PUT_ON_STREAM)
#undef NIIN_DEFINE_PUT_ON_STREAM

// ---------------------------------------------------------------------------
// 4. Sized put on stream (put8, put16, put32, put64, put128)
// ---------------------------------------------------------------------------
#define NIIN_DEFINE_PUT_SIZED_ON_STREAM(SIZE, NBYTES)                          \
inline void nvshmemx_put##SIZE##_on_stream(void* dest, const void* src,        \
    size_t nelems, int pe, cudaStream_t stream) {                              \
  niin_host_rma_put(dest, src, nelems * NBYTES, pe, stream);                  \
}

NIIN_SIZED_RMA(NIIN_DEFINE_PUT_SIZED_ON_STREAM)
#undef NIIN_DEFINE_PUT_SIZED_ON_STREAM

// ---------------------------------------------------------------------------
// 5. putmem on stream
// ---------------------------------------------------------------------------
inline void nvshmemx_putmem_on_stream(void* dest, const void* src,
    size_t bytes, int pe, cudaStream_t stream) {
  niin_host_rma_put(dest, src, bytes, pe, stream);
}

// ---------------------------------------------------------------------------
// 6. Typed get on stream
// ---------------------------------------------------------------------------
#define NIIN_DEFINE_GET_ON_STREAM(TYPENAME, TYPE)                              \
inline void nvshmemx_##TYPENAME##_get_on_stream(TYPE* dest, const TYPE* src,   \
    size_t nelems, int pe, cudaStream_t stream) {                              \
  int ctas = niin_stream_cta_count(nelems * sizeof(TYPE));                    \
  niin_stream_get_kernel<TYPE><<<ctas, kNiinStreamThreadsPerCta, 0, stream>>>( \
      dest, src, nelems, pe);                                                  \
}

NIIN_STANDARD_RMA_TYPES(NIIN_DEFINE_GET_ON_STREAM)
#undef NIIN_DEFINE_GET_ON_STREAM

// ---------------------------------------------------------------------------
// 7. Sized get on stream (get8, get16, get32, get64, get128)
// ---------------------------------------------------------------------------
#define NIIN_DEFINE_GET_SIZED_ON_STREAM(SIZE, NBYTES)                          \
inline void nvshmemx_get##SIZE##_on_stream(void* dest, const void* src,        \
    size_t nelems, int pe, cudaStream_t stream) {                              \
  size_t bytes = nelems * NBYTES;                                              \
  int ctas = niin_stream_cta_count(bytes);                                     \
  niin_stream_getmem_kernel<<<ctas, kNiinStreamThreadsPerCta, 0, stream>>>(    \
      dest, src, bytes, pe);                                                    \
}

NIIN_SIZED_RMA(NIIN_DEFINE_GET_SIZED_ON_STREAM)
#undef NIIN_DEFINE_GET_SIZED_ON_STREAM

// ---------------------------------------------------------------------------
// 8. getmem on stream
// ---------------------------------------------------------------------------
inline void nvshmemx_getmem_on_stream(void* dest, const void* src,
    size_t bytes, int pe, cudaStream_t stream) {
  int ctas = niin_stream_cta_count(bytes);
  niin_stream_getmem_kernel<<<ctas, kNiinStreamThreadsPerCta, 0, stream>>>(
      dest, src, bytes, pe);
}

// ---------------------------------------------------------------------------
// 9. NBI put on stream (same as blocking — ncclPutSignal is already async)
// ---------------------------------------------------------------------------
#define NIIN_DEFINE_PUT_NBI_ON_STREAM(TYPENAME, TYPE)                          \
inline void nvshmemx_##TYPENAME##_put_nbi_on_stream(TYPE* dest, const TYPE* src, \
    size_t nelems, int pe, cudaStream_t stream) {                              \
  niin_host_rma_put(dest, src, nelems * sizeof(TYPE), pe, stream);            \
}

NIIN_STANDARD_RMA_TYPES(NIIN_DEFINE_PUT_NBI_ON_STREAM)
#undef NIIN_DEFINE_PUT_NBI_ON_STREAM

#define NIIN_DEFINE_PUT_SIZED_NBI_ON_STREAM(SIZE, NBYTES)                      \
inline void nvshmemx_put##SIZE##_nbi_on_stream(void* dest, const void* src,    \
    size_t nelems, int pe, cudaStream_t stream) {                              \
  niin_host_rma_put(dest, src, nelems * NBYTES, pe, stream);                  \
}

NIIN_SIZED_RMA(NIIN_DEFINE_PUT_SIZED_NBI_ON_STREAM)
#undef NIIN_DEFINE_PUT_SIZED_NBI_ON_STREAM

inline void nvshmemx_putmem_nbi_on_stream(void* dest, const void* src,
    size_t bytes, int pe, cudaStream_t stream) {
  niin_host_rma_put(dest, src, bytes, pe, stream);
}

// ---------------------------------------------------------------------------
// 10. NBI get on stream (already async, same as blocking)
// ---------------------------------------------------------------------------
#define NIIN_DEFINE_GET_NBI_ON_STREAM(TYPENAME, TYPE)                          \
inline void nvshmemx_##TYPENAME##_get_nbi_on_stream(TYPE* dest, const TYPE* src, \
    size_t nelems, int pe, cudaStream_t stream) {                              \
  int ctas = niin_stream_cta_count(nelems * sizeof(TYPE));                    \
  niin_stream_get_kernel<TYPE><<<ctas, kNiinStreamThreadsPerCta, 0, stream>>>( \
      dest, src, nelems, pe);                                                  \
}

NIIN_STANDARD_RMA_TYPES(NIIN_DEFINE_GET_NBI_ON_STREAM)
#undef NIIN_DEFINE_GET_NBI_ON_STREAM

#define NIIN_DEFINE_GET_SIZED_NBI_ON_STREAM(SIZE, NBYTES)                      \
inline void nvshmemx_get##SIZE##_nbi_on_stream(void* dest, const void* src,    \
    size_t nelems, int pe, cudaStream_t stream) {                              \
  size_t bytes = nelems * NBYTES;                                              \
  int ctas = niin_stream_cta_count(bytes);                                     \
  niin_stream_getmem_kernel<<<ctas, kNiinStreamThreadsPerCta, 0, stream>>>(    \
      dest, src, bytes, pe);                                                    \
}

NIIN_SIZED_RMA(NIIN_DEFINE_GET_SIZED_NBI_ON_STREAM)
#undef NIIN_DEFINE_GET_SIZED_NBI_ON_STREAM

inline void nvshmemx_getmem_nbi_on_stream(void* dest, const void* src,
    size_t bytes, int pe, cudaStream_t stream) {
  int ctas = niin_stream_cta_count(bytes);
  niin_stream_getmem_kernel<<<ctas, kNiinStreamThreadsPerCta, 0, stream>>>(
      dest, src, bytes, pe);
}

// ---------------------------------------------------------------------------
// 11. Strided iput on stream
// ---------------------------------------------------------------------------
#define NIIN_DEFINE_IPUT_ON_STREAM(TYPENAME, TYPE)                             \
inline void nvshmemx_##TYPENAME##_iput_on_stream(TYPE* dest, const TYPE* src,  \
    ptrdiff_t dst, ptrdiff_t sst, size_t nelems, int pe,                       \
    cudaStream_t stream) {                                                     \
  niin_stream_iput_kernel<TYPE><<<1, 1, 0, stream>>>(dest, src, dst, sst,     \
      nelems, pe);                                                             \
}

NIIN_STANDARD_RMA_TYPES(NIIN_DEFINE_IPUT_ON_STREAM)
#undef NIIN_DEFINE_IPUT_ON_STREAM

// ---------------------------------------------------------------------------
// 12. Strided iget on stream
// ---------------------------------------------------------------------------
#define NIIN_DEFINE_IGET_ON_STREAM(TYPENAME, TYPE)                             \
inline void nvshmemx_##TYPENAME##_iget_on_stream(TYPE* dest, const TYPE* src,  \
    ptrdiff_t dst, ptrdiff_t sst, size_t nelems, int pe,                       \
    cudaStream_t stream) {                                                     \
  niin_stream_iget_kernel<TYPE><<<1, 1, 0, stream>>>(dest, src, dst, sst,     \
      nelems, pe);                                                             \
}

NIIN_STANDARD_RMA_TYPES(NIIN_DEFINE_IGET_ON_STREAM)
#undef NIIN_DEFINE_IGET_ON_STREAM

// ---------------------------------------------------------------------------
// 13. Typed put signal on stream
// ---------------------------------------------------------------------------
#define NIIN_DEFINE_PUT_SIGNAL_ON_STREAM(TYPENAME, TYPE)                       \
inline void nvshmemx_##TYPENAME##_put_signal_on_stream(TYPE* dest,             \
    const TYPE* src, size_t nelems, uint64_t* sig_addr, uint64_t signal,       \
    int sig_op, int pe, cudaStream_t stream) {                                 \
  niin_host_rma_put_signal(dest, src, nelems * sizeof(TYPE),                  \
      sig_addr, signal, sig_op, pe, stream);                                   \
}

NIIN_STANDARD_RMA_TYPES(NIIN_DEFINE_PUT_SIGNAL_ON_STREAM)
#undef NIIN_DEFINE_PUT_SIGNAL_ON_STREAM

// ---------------------------------------------------------------------------
// 14. Sized put signal on stream
// ---------------------------------------------------------------------------
#define NIIN_DEFINE_PUT_SIGNAL_SIZED_ON_STREAM(SIZE, NBYTES)                   \
inline void nvshmemx_put##SIZE##_signal_on_stream(void* dest, const void* src, \
    size_t nelems, uint64_t* sig_addr, uint64_t signal, int sig_op,            \
    int pe, cudaStream_t stream) {                                             \
  niin_host_rma_put_signal(dest, src, nelems * NBYTES,                        \
      sig_addr, signal, sig_op, pe, stream);                                   \
}

NIIN_SIZED_RMA(NIIN_DEFINE_PUT_SIGNAL_SIZED_ON_STREAM)
#undef NIIN_DEFINE_PUT_SIGNAL_SIZED_ON_STREAM

// ---------------------------------------------------------------------------
// 15. putmem signal on stream
// ---------------------------------------------------------------------------
inline void nvshmemx_putmem_signal_on_stream_impl(void* dest, const void* src,
    size_t bytes, uint64_t* sig_addr, uint64_t signal, int sig_op,
    int pe, cudaStream_t stream) {
  niin_host_rma_put_signal(dest, src, bytes, sig_addr, signal, sig_op, pe, stream);
}

// ---------------------------------------------------------------------------
// 16. NBI put signal on stream (same as blocking)
// ---------------------------------------------------------------------------
#define NIIN_DEFINE_PUT_SIGNAL_NBI_ON_STREAM(TYPENAME, TYPE)                   \
inline void nvshmemx_##TYPENAME##_put_signal_nbi_on_stream(TYPE* dest,         \
    const TYPE* src, size_t nelems, uint64_t* sig_addr, uint64_t signal,       \
    int sig_op, int pe, cudaStream_t stream) {                                 \
  niin_host_rma_put_signal(dest, src, nelems * sizeof(TYPE),                  \
      sig_addr, signal, sig_op, pe, stream);                                   \
}

NIIN_STANDARD_RMA_TYPES(NIIN_DEFINE_PUT_SIGNAL_NBI_ON_STREAM)
#undef NIIN_DEFINE_PUT_SIGNAL_NBI_ON_STREAM

inline void nvshmemx_putmem_signal_nbi_on_stream(void* dest, const void* src,
    size_t bytes, uint64_t* sig_addr, uint64_t signal, int sig_op,
    int pe, cudaStream_t stream) {
  niin_host_rma_put_signal(dest, src, bytes, sig_addr, signal, sig_op, pe, stream);
}

// ---------------------------------------------------------------------------
// 17. Signal op on stream
// ---------------------------------------------------------------------------
inline void nvshmemx_signal_op_on_stream(uint64_t* sig_addr, uint64_t val,
    int sig_op, int pe, cudaStream_t stream) {
  niin_stream_signal_op_kernel<<<1, 1, 0, stream>>>(sig_addr, val, sig_op, pe);
}

// ---------------------------------------------------------------------------
// 18. Signal wait until on stream (synchronous -- must return value)
// ---------------------------------------------------------------------------
inline uint64_t nvshmemx_signal_wait_until_on_stream(uint64_t* sig_addr,
    int cmp, uint64_t cmp_value, cudaStream_t stream) {
  uint64_t result;
  uint64_t* d_tmp;
  cudaMalloc(&d_tmp, sizeof(uint64_t));
  niin_stream_signal_wait_until_kernel<<<1, 1, 0, stream>>>(sig_addr, cmp,
      cmp_value, d_tmp);
  cudaMemcpyAsync(&result, d_tmp, sizeof(uint64_t), cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);
  cudaFree(d_tmp);
  return result;
}

// ---------------------------------------------------------------------------
// 19. Typed wait_until on stream
// ---------------------------------------------------------------------------
#define NIIN_DEFINE_WAIT_UNTIL_ON_STREAM(TYPENAME, TYPE)                       \
inline void nvshmemx_##TYPENAME##_wait_until_on_stream(TYPE* ivar, int cmp,    \
    TYPE cmp_value, cudaStream_t stream) {                                     \
  niin_stream_wait_until_kernel<TYPE><<<1, 1, 0, stream>>>(ivar, cmp,         \
      cmp_value);                                                              \
}

NIIN_WAIT_TYPES(NIIN_DEFINE_WAIT_UNTIL_ON_STREAM)
#undef NIIN_DEFINE_WAIT_UNTIL_ON_STREAM

// ---------------------------------------------------------------------------
// 20. Typed wait_until_all on stream
// ---------------------------------------------------------------------------
#define NIIN_DEFINE_WAIT_UNTIL_ALL_ON_STREAM(TYPENAME, TYPE)                   \
inline void nvshmemx_##TYPENAME##_wait_until_all_on_stream(TYPE* ivars,        \
    size_t nelems, const int* status, int cmp, TYPE cmp_value,                 \
    cudaStream_t stream) {                                                     \
  niin_stream_wait_until_all_kernel<TYPE><<<1, 1, 0, stream>>>(ivars, nelems,  \
      status, cmp, cmp_value);                                                 \
}

NIIN_WAIT_TYPES(NIIN_DEFINE_WAIT_UNTIL_ALL_ON_STREAM)
#undef NIIN_DEFINE_WAIT_UNTIL_ALL_ON_STREAM

// ---------------------------------------------------------------------------
// 21. Typed wait_until_all_vector on stream
// ---------------------------------------------------------------------------
#define NIIN_DEFINE_WAIT_UNTIL_ALL_VECTOR_ON_STREAM(TYPENAME, TYPE)            \
inline void nvshmemx_##TYPENAME##_wait_until_all_vector_on_stream(TYPE* ivars, \
    size_t nelems, const int* status, int cmp, TYPE* cmp_values,               \
    cudaStream_t stream) {                                                     \
  niin_stream_wait_until_all_vector_kernel<TYPE><<<1, 1, 0, stream>>>(ivars,   \
      nelems, status, cmp, cmp_values);                                        \
}

NIIN_WAIT_TYPES(NIIN_DEFINE_WAIT_UNTIL_ALL_VECTOR_ON_STREAM)
#undef NIIN_DEFINE_WAIT_UNTIL_ALL_VECTOR_ON_STREAM

#endif // NIIN_STREAM_H_
