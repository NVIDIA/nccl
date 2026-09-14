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
#include "niin/sync.h"

// ===========================================================================
// Internal: cooperative memcpy kernels using all threads in a group.
// Uses int4 (16-byte) vectorized copies for NVLink bandwidth.
// ===========================================================================

NIIN_NOINLINE_DEVICE void niin_coop_copy_slow(
    void* __restrict__ dst, const void* __restrict__ src, size_t bytes,
    int tid, int nthreads) {
  char* d = static_cast<char*>(dst);
  const char* s = static_cast<const char*>(src);

  if ((reinterpret_cast<uintptr_t>(d) % 16) == 0 &&
      (reinterpret_cast<uintptr_t>(s) % 16) == 0) {
    size_t n16 = bytes / 16;
    auto* d16 = reinterpret_cast<int4*>(d);
    auto const* s16 = reinterpret_cast<int4 const*>(s);
    for (size_t i = static_cast<size_t>(tid); i < n16; i += static_cast<size_t>(nthreads)) {
      d16[i] = s16[i];
    }
    d += n16 * 16;
    s += n16 * 16;
    bytes -= n16 * 16;
    if (bytes == 0) return;
  }

  if ((reinterpret_cast<uintptr_t>(d) % 8) == 0 &&
      (reinterpret_cast<uintptr_t>(s) % 8) == 0) {
    size_t n8 = bytes / 8;
    auto* d8 = reinterpret_cast<uint64_t*>(d);
    auto const* s8 = reinterpret_cast<uint64_t const*>(s);
    for (size_t i = static_cast<size_t>(tid); i < n8; i += static_cast<size_t>(nthreads)) {
      d8[i] = s8[i];
    }
    d += n8 * 8;
    s += n8 * 8;
    bytes -= n8 * 8;
    if (bytes == 0) return;
  }

  if ((reinterpret_cast<uintptr_t>(d) % 4) == 0 &&
      (reinterpret_cast<uintptr_t>(s) % 4) == 0) {
    size_t n4 = bytes / 4;
    auto* d4 = reinterpret_cast<uint32_t*>(d);
    auto const* s4 = reinterpret_cast<uint32_t const*>(s);
    for (size_t i = static_cast<size_t>(tid); i < n4; i += static_cast<size_t>(nthreads)) {
      d4[i] = s4[i];
    }
    d += n4 * 4;
    s += n4 * 4;
    bytes -= n4 * 4;
    if (bytes == 0) return;
  }

  if ((reinterpret_cast<uintptr_t>(d) % 2) == 0 &&
      (reinterpret_cast<uintptr_t>(s) % 2) == 0) {
    size_t n2 = bytes / 2;
    auto* d2 = reinterpret_cast<uint16_t*>(d);
    auto const* s2 = reinterpret_cast<uint16_t const*>(s);
    for (size_t i = static_cast<size_t>(tid); i < n2; i += static_cast<size_t>(nthreads)) {
      d2[i] = s2[i];
    }
    d += n2 * 2;
    s += n2 * 2;
    bytes -= n2 * 2;
    if (bytes == 0) return;
  }

  for (size_t i = static_cast<size_t>(tid); i < bytes; i += static_cast<size_t>(nthreads)) {
    d[i] = s[i];
  }
}

__device__ __forceinline__ void niin_coop_copy(
    void* __restrict__ dst, const void* __restrict__ src, size_t bytes,
    int tid, int nthreads) {
  uintptr_t daddr = reinterpret_cast<uintptr_t>(dst);
  uintptr_t saddr = reinterpret_cast<uintptr_t>(src);
  if (((daddr | saddr | bytes) & 0xF) != 0) {
    niin_coop_copy_slow(dst, src, bytes, tid, nthreads);
    return;
  }

  auto* d16 = reinterpret_cast<int4*>(dst);
  auto const* s16 = reinterpret_cast<int4 const*>(src);
  size_t n16 = bytes / 16;
  for (size_t i = static_cast<size_t>(tid); i < n16; i += static_cast<size_t>(nthreads)) {
    d16[i] = s16[i];
  }
}

template<int SCOPE, bool BLOCKING = true>
__device__ __forceinline__ void niin_coop_put(
    void* dest, const void* src, size_t bytes, int pe,
    int tid, int nthreads, int ctx = -1, bool markLsaPending = true);

template<int SCOPE>
__device__ __forceinline__ void niin_coop_get(
    void* dest, const void* src, size_t bytes, int pe,
    int tid, int nthreads, int ctx = -1);

template<int SCOPE>
__device__ __forceinline__ void niin_coop_put_lsa_ldst(
    void* dest, const void* src, size_t bytes, int pe,
    int tid, int nthreads, bool markLsaPending) {
  void* target = dest;
  if (pe != niin_device_my_pe()) {
    target = niin_get_peer_ptr_lsa_only(niin_sym_offset(dest), pe);
  }
  niin_coop_copy(target, src, bytes, tid, nthreads);
  if (markLsaPending && tid == 0) niin_note_lsa_store_pending();
}

template<int SCOPE>
__device__ __forceinline__ void niin_coop_get_lsa_ldst(
    void* dest, const void* src, size_t bytes, int pe,
    int tid, int nthreads) {
  const void* source = src;
  if (pe != niin_device_my_pe()) source = niin_get_peer_ptr_lsa_only(niin_sym_offset(src), pe);
  niin_coop_copy(dest, source, bytes, tid, nthreads);
}

template<int SCOPE, bool BLOCKING = true>
__device__ __forceinline__ void niin_coop_put_lsa(
    void* dest, const void* src, size_t bytes, int pe,
    int tid, int nthreads, bool markLsaPending) {
  void* target = dest;
  if (pe != niin_device_my_pe()) {
    target = niin_get_peer_ptr(niin_sym_offset(dest), pe);
  }
  if (niin_tma_try_copy<SCOPE, BLOCKING>(target, src, bytes) == 0) {
    if (markLsaPending && tid == 0) niin_note_lsa_store_pending();
    return;
  }
  niin_coop_copy(target, src, bytes, tid, nthreads);
  if (markLsaPending && tid == 0) niin_note_lsa_store_pending();
}

template<int SCOPE>
__device__ __forceinline__ void niin_coop_get_lsa(
    void* dest, const void* src, size_t bytes, int pe,
    int tid, int nthreads) {
  const void* source = src;
  if (pe != niin_device_my_pe()) source = niin_get_peer_ptr(niin_sym_offset(src), pe);
  if (niin_tma_try_copy<SCOPE>(dest, source, bytes) == 0) return;
  niin_coop_copy(dest, source, bytes, tid, nthreads);
}

template<int SCOPE, bool BLOCKING = true>
NIIN_NOINLINE_DEVICE void niin_coop_put_mixed(
    void* dest, const void* src, size_t bytes, int pe,
    int tid, int nthreads, int ctx, bool markLsaPending) {
  if (pe == niin_device_my_pe() || niin_is_lsa_peer(pe)) {
    niin_coop_put_lsa<SCOPE, BLOCKING>(dest, src, bytes, pe, tid, nthreads, markLsaPending);
    return;
  }

  if (tid == 0) {
    niin_gin_put(niin_sym_offset(dest), src, bytes, pe, ctx);
  }
}

template<int SCOPE>
NIIN_NOINLINE_DEVICE void niin_coop_get_mixed(
    void* dest, const void* src, size_t bytes, int pe,
    int tid, int nthreads, int ctx) {
  if (pe == niin_device_my_pe() || niin_is_lsa_peer(pe)) {
    niin_coop_get_lsa<SCOPE>(dest, src, bytes, pe, tid, nthreads);
    return;
  }

  if (tid == 0) {
    niin_gin_get(dest, niin_sym_offset(src), bytes, pe, ctx);
  }
}

// Cooperative put: all threads copy src -> peer dest via NVLink.
// SCOPE tells the TMA path whether the caller is a warp or a whole CTA; when
// this CTA has lent NIIN shared memory the copy goes out over cp.async.bulk
// instead of the cooperative int4 loop. niin_tma_try_copy returns -1 for every
// transfer TMA cannot take, and it does so uniformly across the threadgroup,
// so either all threads take the TMA path or none do.
template<int SCOPE, bool BLOCKING>
__device__ __forceinline__ void niin_coop_put(
    void* dest, const void* src, size_t bytes, int pe,
    int tid, int nthreads, int ctx, bool markLsaPending) {
  if (niin_world_is_lsa_only()) {
    niin_coop_put_lsa<SCOPE, BLOCKING>(dest, src, bytes, pe, tid, nthreads, markLsaPending);
    return;
  }
  niin_coop_put_mixed<SCOPE, BLOCKING>(dest, src, bytes, pe, tid, nthreads, ctx, markLsaPending);
}

// Cooperative get: all threads copy peer src -> local dest via NVLink.
// Takes the TMA path on the same terms as niin_coop_put; a destination in
// shared memory routes straight into smem instead of staging through it.
template<int SCOPE>
__device__ __forceinline__ void niin_coop_get(
    void* dest, const void* src, size_t bytes, int pe,
    int tid, int nthreads, int ctx) {
  if (niin_world_is_lsa_only()) {
    niin_coop_get_lsa<SCOPE>(dest, src, bytes, pe, tid, nthreads);
    return;
  }
  niin_coop_get_mixed<SCOPE>(dest, src, bytes, pe, tid, nthreads, ctx);
}

template<int SCOPE, bool BLOCKING = true>
NIIN_NOINLINE_DEVICE void niin_coop_put_generic_body(
    void* dest, const void* src, size_t bytes, int pe,
    int tid, int nthreads, int ctx, bool markLsaPending) {
  niin_coop_put<SCOPE, BLOCKING>(dest, src, bytes, pe, tid, nthreads, ctx, markLsaPending);
}

template<int SCOPE>
NIIN_NOINLINE_DEVICE void niin_coop_get_generic_body(
    void* dest, const void* src, size_t bytes, int pe,
    int tid, int nthreads, int ctx) {
  niin_coop_get<SCOPE>(dest, src, bytes, pe, tid, nthreads, ctx);
}

template<int SCOPE>
__device__ __forceinline__ void niin_threadgroup_sync() {
  if constexpr (SCOPE == NIIN_TMA_WARP) {
    __syncwarp();
  } else if constexpr (SCOPE == NIIN_TMA_BLOCK) {
    __syncthreads();
  }
}

template<int SCOPE>
__device__ __forceinline__ bool niin_threadgroup_is_leader() {
  if constexpr (SCOPE == NIIN_TMA_THREAD) {
    return true;
  } else if constexpr (SCOPE == NIIN_TMA_WARP) {
    return nccl::utility::lane() == 0;
  } else {
    return threadIdx.x == 0;
  }
}

template<int SCOPE>
__device__ __forceinline__ void niin_threadgroup_quiet() {
  if constexpr (SCOPE == NIIN_TMA_THREAD) {
    niin_device_quiet();
    return;
  }

  const bool hadTma = niin_tma_smem_registered();
  niin_tma_drain_if_registered();

  if (niin_threadgroup_is_leader<SCOPE>()) {
    if (niin_world_is_lsa_only() && niin_tma_policy() == NVSHMEMX_TMA_DISABLE) {
      if (niin_consume_lsa_store_pending()) __threadfence_system();
    } else {
      const bool hadLsaStores = niin_consume_lsa_store_pending();
      const bool hadGinOps = !niin_world_is_lsa_only() && (niin_take_gin_ops_pending() != 0u);
      if (hadGinOps) niin_device_drain_all_contexts();
      if (hadTma || hadLsaStores || hadGinOps) __threadfence_system();
    }
  }

  niin_threadgroup_sync<SCOPE>();
}

__device__ __forceinline__ void nvshmemx_qp_quiet(
    int pe, nvshmemx_qp_handle_t* qp_handle, int num_qps) {
  (void)pe; (void)qp_handle; (void)num_qps;
  niin_threadgroup_quiet<NIIN_TMA_THREAD>();
}

__device__ __forceinline__ void nvshmemx_qp_quiet_warp(
    int pe, nvshmemx_qp_handle_t* qp_handle, int num_qps) {
  (void)pe; (void)qp_handle; (void)num_qps;
  niin_threadgroup_quiet<NIIN_TMA_WARP>();
}

__device__ __forceinline__ void nvshmemx_qp_quiet_block(
    int pe, nvshmemx_qp_handle_t* qp_handle, int num_qps) {
  (void)pe; (void)qp_handle; (void)num_qps;
  niin_threadgroup_quiet<NIIN_TMA_BLOCK>();
}

__device__ __forceinline__ void niin_threadgroup_deliver_signal(
    bool lsa, uint64_t* sig, uint64_t signal, int sig_op, int pe, int ctx) {
  if (lsa) {
    __threadfence_system();
    niin_clear_lsa_store_pending();
    if (pe == niin_rank() || niin_peer_native_atomic()) {
      niin_deliver_signal_lsa_fast(sig, signal, sig_op, pe);
      return;
    }
  }
  niin_deliver_signal(sig, signal, sig_op, pe, ctx);
}

// NBI calls are nonblocking with respect to RMA completion, but the warp/block
// NVSHMEM APIs are collective over the threadgroup.
template<int SCOPE, bool BLOCKING = false>
__device__ __forceinline__ void niin_coop_put_nbi(
    void* dest, const void* src, size_t bytes, int pe,
    int tid, int nthreads) {
  niin_threadgroup_sync<SCOPE>();
  if (niin_world_is_lsa_only() && niin_tma_policy() == NVSHMEMX_TMA_DISABLE) {
    niin_coop_put_lsa_ldst<SCOPE>(dest, src, bytes, pe, tid, nthreads, true);
  } else {
    niin_coop_put_generic_body<SCOPE, BLOCKING>(dest, src, bytes, pe, tid, nthreads, -1, true);
  }
  niin_threadgroup_sync<SCOPE>();
}

template<int SCOPE>
__device__ __forceinline__ void niin_coop_get_nbi(
    void* dest, const void* src, size_t bytes, int pe,
    int tid, int nthreads) {
  niin_threadgroup_sync<SCOPE>();
  if (niin_world_is_lsa_only() && niin_tma_policy() == NVSHMEMX_TMA_DISABLE) {
    niin_coop_get_lsa_ldst<SCOPE>(dest, src, bytes, pe, tid, nthreads);
  } else {
    niin_coop_get_generic_body<SCOPE>(dest, src, bytes, pe, tid, nthreads, -1);
  }
  niin_threadgroup_sync<SCOPE>();
}

// ===========================================================================
// Typed put warp/block — cooperative NVLink copy
// ===========================================================================
#define NIIN_DEFINE_PUT_WARP(TYPENAME, TYPE)                                   \
__device__ __forceinline__ void nvshmemx_##TYPENAME##_put_warp(               \
    TYPE* dest, const TYPE* src, size_t nelems, int pe) {                     \
  niin_coop_put<NIIN_TMA_WARP>(dest, src, nelems * sizeof(TYPE), pe,           \
                               nccl::utility::lane(), 32);                     \
  __syncwarp();                                                                \
}

NIIN_STANDARD_RMA_TYPES(NIIN_DEFINE_PUT_WARP)
#undef NIIN_DEFINE_PUT_WARP

#define NIIN_DEFINE_PUT_BLOCK(TYPENAME, TYPE)                                  \
__device__ __forceinline__ void nvshmemx_##TYPENAME##_put_block(              \
    TYPE* dest, const TYPE* src, size_t nelems, int pe) {                     \
  niin_coop_put<NIIN_TMA_BLOCK>(dest, src, nelems * sizeof(TYPE), pe,          \
                                threadIdx.x, blockDim.x);                      \
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
  niin_coop_get<NIIN_TMA_WARP>(dest, src, nelems * sizeof(TYPE), pe,           \
                               nccl::utility::lane(), 32);                     \
  __syncwarp();                                                                \
}

NIIN_STANDARD_RMA_TYPES(NIIN_DEFINE_GET_WARP)
#undef NIIN_DEFINE_GET_WARP

#define NIIN_DEFINE_GET_BLOCK(TYPENAME, TYPE)                                  \
__device__ __forceinline__ void nvshmemx_##TYPENAME##_get_block(              \
    TYPE* dest, const TYPE* src, size_t nelems, int pe) {                     \
  niin_coop_get<NIIN_TMA_BLOCK>(dest, src, nelems * sizeof(TYPE), pe,          \
                                threadIdx.x, blockDim.x);                      \
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
  niin_coop_put<NIIN_TMA_WARP>(dest, src, nelems * NBYTES, pe,                 \
                               nccl::utility::lane(), 32);                     \
  __syncwarp();                                                                \
}

NIIN_SIZED_RMA(NIIN_DEFINE_PUT_SIZED_WARP)
#undef NIIN_DEFINE_PUT_SIZED_WARP

#define NIIN_DEFINE_PUT_SIZED_BLOCK(SIZE, NBYTES)                              \
__device__ __forceinline__ void nvshmemx_put##SIZE##_block(                   \
    void* dest, const void* src, size_t nelems, int pe) {                     \
  niin_coop_put<NIIN_TMA_BLOCK>(dest, src, nelems * NBYTES, pe,                \
                                threadIdx.x, blockDim.x);                      \
  __syncthreads();                                                             \
}

NIIN_SIZED_RMA(NIIN_DEFINE_PUT_SIZED_BLOCK)
#undef NIIN_DEFINE_PUT_SIZED_BLOCK

#define NIIN_DEFINE_GET_SIZED_WARP(SIZE, NBYTES)                               \
__device__ __forceinline__ void nvshmemx_get##SIZE##_warp(                    \
    void* dest, const void* src, size_t nelems, int pe) {                     \
  niin_coop_get<NIIN_TMA_WARP>(dest, src, nelems * NBYTES, pe,                 \
                               nccl::utility::lane(), 32);                     \
  __syncwarp();                                                                \
}

NIIN_SIZED_RMA(NIIN_DEFINE_GET_SIZED_WARP)
#undef NIIN_DEFINE_GET_SIZED_WARP

#define NIIN_DEFINE_GET_SIZED_BLOCK(SIZE, NBYTES)                              \
__device__ __forceinline__ void nvshmemx_get##SIZE##_block(                   \
    void* dest, const void* src, size_t nelems, int pe) {                     \
  niin_coop_get<NIIN_TMA_BLOCK>(dest, src, nelems * NBYTES, pe,                \
                                threadIdx.x, blockDim.x);                      \
  __syncthreads();                                                             \
}

NIIN_SIZED_RMA(NIIN_DEFINE_GET_SIZED_BLOCK)
#undef NIIN_DEFINE_GET_SIZED_BLOCK

// ===========================================================================
// putmem/getmem warp/block
// ===========================================================================
__device__ __forceinline__ void nvshmemx_putmem_warp(void* dest, const void* src, size_t bytes, int pe) {
  niin_coop_put<NIIN_TMA_WARP>(dest, src, bytes, pe, nccl::utility::lane(), 32);
  __syncwarp();
}
__device__ __forceinline__ void nvshmemx_putmem_block(void* dest, const void* src, size_t bytes, int pe) {
  niin_coop_put<NIIN_TMA_BLOCK>(dest, src, bytes, pe, threadIdx.x, blockDim.x);
  __syncthreads();
}
__device__ __forceinline__ void nvshmemx_getmem_warp(void* dest, const void* src, size_t bytes, int pe) {
  niin_coop_get<NIIN_TMA_WARP>(dest, src, bytes, pe, nccl::utility::lane(), 32);
  __syncwarp();
}
__device__ __forceinline__ void nvshmemx_getmem_block(void* dest, const void* src, size_t bytes, int pe) {
  niin_coop_get<NIIN_TMA_BLOCK>(dest, src, bytes, pe, threadIdx.x, blockDim.x);
  __syncthreads();
}

// ===========================================================================
// NBI put/get warp/block (same as blocking — already async for NVLink)
// ===========================================================================
#define NIIN_DEFINE_PUT_NBI_WARP(TYPENAME, TYPE)                               \
__device__ __forceinline__ void nvshmemx_##TYPENAME##_put_nbi_warp(           \
    TYPE* dest, const TYPE* src, size_t nelems, int pe) {                     \
  niin_coop_put_nbi<NIIN_TMA_WARP>(dest, src, nelems * sizeof(TYPE), pe,       \
                                   nccl::utility::lane(), 32);                 \
}

NIIN_STANDARD_RMA_TYPES(NIIN_DEFINE_PUT_NBI_WARP)
#undef NIIN_DEFINE_PUT_NBI_WARP

#define NIIN_DEFINE_PUT_NBI_BLOCK(TYPENAME, TYPE)                              \
__device__ __forceinline__ void nvshmemx_##TYPENAME##_put_nbi_block(          \
    TYPE* dest, const TYPE* src, size_t nelems, int pe) {                     \
  niin_coop_put_nbi<NIIN_TMA_BLOCK>(dest, src, nelems * sizeof(TYPE), pe,      \
                                    threadIdx.x, blockDim.x);                  \
}

NIIN_STANDARD_RMA_TYPES(NIIN_DEFINE_PUT_NBI_BLOCK)
#undef NIIN_DEFINE_PUT_NBI_BLOCK

#define NIIN_DEFINE_GET_NBI_WARP(TYPENAME, TYPE)                               \
__device__ __forceinline__ void nvshmemx_##TYPENAME##_get_nbi_warp(           \
    TYPE* dest, const TYPE* src, size_t nelems, int pe) {                     \
  niin_coop_get_nbi<NIIN_TMA_WARP>(dest, src, nelems * sizeof(TYPE), pe,       \
                                   nccl::utility::lane(), 32);                 \
}

NIIN_STANDARD_RMA_TYPES(NIIN_DEFINE_GET_NBI_WARP)
#undef NIIN_DEFINE_GET_NBI_WARP

#define NIIN_DEFINE_GET_NBI_BLOCK(TYPENAME, TYPE)                              \
__device__ __forceinline__ void nvshmemx_##TYPENAME##_get_nbi_block(          \
    TYPE* dest, const TYPE* src, size_t nelems, int pe) {                     \
  niin_coop_get_nbi<NIIN_TMA_BLOCK>(dest, src, nelems * sizeof(TYPE), pe,      \
                                    threadIdx.x, blockDim.x);                  \
}

NIIN_STANDARD_RMA_TYPES(NIIN_DEFINE_GET_NBI_BLOCK)
#undef NIIN_DEFINE_GET_NBI_BLOCK

// NBI putmem/getmem warp/block
__device__ __forceinline__ void nvshmemx_putmem_nbi_warp(void* dest, const void* src, size_t bytes, int pe) {
  niin_coop_put_nbi<NIIN_TMA_WARP>(dest, src, bytes, pe, nccl::utility::lane(), 32);
}
__device__ __forceinline__ void nvshmemx_putmem_nbi_block(void* dest, const void* src, size_t bytes, int pe) {
  niin_coop_put_nbi<NIIN_TMA_BLOCK>(dest, src, bytes, pe, threadIdx.x, blockDim.x);
}
__device__ __forceinline__ void nvshmemx_getmem_nbi_warp(void* dest, const void* src, size_t bytes, int pe) {
  niin_coop_get_nbi<NIIN_TMA_WARP>(dest, src, bytes, pe, nccl::utility::lane(), 32);
}
__device__ __forceinline__ void nvshmemx_getmem_nbi_block(void* dest, const void* src, size_t bytes, int pe) {
  niin_coop_get_nbi<NIIN_TMA_BLOCK>(dest, src, bytes, pe, threadIdx.x, blockDim.x);
}

// NBI sized put/get warp/block
#define NIIN_DEFINE_PUT_SIZED_NBI_WARP(SIZE, NBYTES)                           \
__device__ __forceinline__ void nvshmemx_put##SIZE##_nbi_warp(                \
    void* dest, const void* src, size_t nelems, int pe) {                     \
  niin_coop_put_nbi<NIIN_TMA_WARP>(dest, src, nelems * NBYTES, pe,             \
                                   nccl::utility::lane(), 32);                 \
}

NIIN_SIZED_RMA(NIIN_DEFINE_PUT_SIZED_NBI_WARP)
#undef NIIN_DEFINE_PUT_SIZED_NBI_WARP

#define NIIN_DEFINE_PUT_SIZED_NBI_BLOCK(SIZE, NBYTES)                          \
__device__ __forceinline__ void nvshmemx_put##SIZE##_nbi_block(               \
    void* dest, const void* src, size_t nelems, int pe) {                     \
  niin_coop_put_nbi<NIIN_TMA_BLOCK>(dest, src, nelems * NBYTES, pe,            \
                                    threadIdx.x, blockDim.x);                  \
}

NIIN_SIZED_RMA(NIIN_DEFINE_PUT_SIZED_NBI_BLOCK)
#undef NIIN_DEFINE_PUT_SIZED_NBI_BLOCK

#define NIIN_DEFINE_GET_SIZED_NBI_WARP(SIZE, NBYTES)                           \
__device__ __forceinline__ void nvshmemx_get##SIZE##_nbi_warp(                \
    void* dest, const void* src, size_t nelems, int pe) {                     \
  niin_coop_get_nbi<NIIN_TMA_WARP>(dest, src, nelems * NBYTES, pe,             \
                                   nccl::utility::lane(), 32);                 \
}

NIIN_SIZED_RMA(NIIN_DEFINE_GET_SIZED_NBI_WARP)
#undef NIIN_DEFINE_GET_SIZED_NBI_WARP

#define NIIN_DEFINE_GET_SIZED_NBI_BLOCK(SIZE, NBYTES)                          \
__device__ __forceinline__ void nvshmemx_get##SIZE##_nbi_block(               \
    void* dest, const void* src, size_t nelems, int pe) {                     \
  niin_coop_get_nbi<NIIN_TMA_BLOCK>(dest, src, nelems * NBYTES, pe,            \
                                    threadIdx.x, blockDim.x);                  \
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
  const bool lsa = niin_world_is_lsa_only() || niin_is_lsa_peer(pe);           \
  int ctx = -1;                                                                \
  if (nccl::utility::lane() == 0 && !lsa) ctx = niin_gin_next_context();       \
  __syncwarp();                                                                \
  if (!lsa) cuda::atomic_thread_fence(cuda::memory_order_acquire, cuda::thread_scope_system); \
  niin_coop_put<NIIN_TMA_WARP>(dest, src, nelems * sizeof(TYPE), pe,           \
                               nccl::utility::lane(), 32, ctx, false);         \
  if (ctx >= 0) niin_gin_flush_thread(ctx);                                    \
  if (!lsa) cuda::atomic_thread_fence(cuda::memory_order_release, cuda::thread_scope_system); \
  __syncwarp();                                                                \
  if (nccl::utility::lane() == 0) {                                            \
    niin_threadgroup_deliver_signal(lsa, sig, signal, sig_op, pe, ctx);         \
  }                                                                            \
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
  const bool lsa = niin_world_is_lsa_only() || niin_is_lsa_peer(pe);           \
  int ctx = -1;                                                                \
  if (threadIdx.x == 0 && !lsa) ctx = niin_gin_next_context();                 \
  __syncthreads();                                                             \
  if (!lsa) cuda::atomic_thread_fence(cuda::memory_order_acquire, cuda::thread_scope_system); \
  niin_coop_put<NIIN_TMA_BLOCK>(dest, src, nelems * sizeof(TYPE), pe,          \
                                threadIdx.x, blockDim.x, ctx, false);          \
  if (ctx >= 0) niin_gin_flush_thread(ctx);                                    \
  if (!lsa) cuda::atomic_thread_fence(cuda::memory_order_release, cuda::thread_scope_system); \
  __syncthreads();                                                             \
  if (threadIdx.x == 0) {                                                      \
    niin_threadgroup_deliver_signal(lsa, sig, signal, sig_op, pe, ctx);         \
  }                                                                            \
  __syncthreads();                                                             \
}

NIIN_STANDARD_RMA_TYPES(NIIN_DEFINE_PUT_SIGNAL_BLOCK)
#undef NIIN_DEFINE_PUT_SIGNAL_BLOCK

// putmem_signal warp/block
__device__ __forceinline__ void nvshmemx_putmem_signal_warp(
    void* dest, const void* src, size_t bytes,
    uint64_t* sig, uint64_t signal, int sig_op, int pe) {
  const bool lsa = niin_world_is_lsa_only() || niin_is_lsa_peer(pe);
  int ctx = -1;
  if (nccl::utility::lane() == 0 && !lsa) ctx = niin_gin_next_context();
  __syncwarp();
  if (!lsa) cuda::atomic_thread_fence(cuda::memory_order_acquire, cuda::thread_scope_system);
  niin_coop_put<NIIN_TMA_WARP>(dest, src, bytes, pe, nccl::utility::lane(), 32, ctx, false);
  if (ctx >= 0) niin_gin_flush_thread(ctx);
  if (!lsa) cuda::atomic_thread_fence(cuda::memory_order_release, cuda::thread_scope_system);
  __syncwarp();
  if (nccl::utility::lane() == 0) {
    niin_threadgroup_deliver_signal(lsa, sig, signal, sig_op, pe, ctx);
  }
  __syncwarp();
}
__device__ __forceinline__ void nvshmemx_putmem_signal_block(
    void* dest, const void* src, size_t bytes,
    uint64_t* sig, uint64_t signal, int sig_op, int pe) {
  const bool lsa = niin_world_is_lsa_only() || niin_is_lsa_peer(pe);
  int ctx = -1;
  if (threadIdx.x == 0 && !lsa) ctx = niin_gin_next_context();
  __syncthreads();
  if (!lsa) cuda::atomic_thread_fence(cuda::memory_order_acquire, cuda::thread_scope_system);
  niin_coop_put<NIIN_TMA_BLOCK>(dest, src, bytes, pe, threadIdx.x, blockDim.x, ctx, false);
  if (ctx >= 0) niin_gin_flush_thread(ctx);
  if (!lsa) cuda::atomic_thread_fence(cuda::memory_order_release, cuda::thread_scope_system);
  __syncthreads();
  if (threadIdx.x == 0) {
    niin_threadgroup_deliver_signal(lsa, sig, signal, sig_op, pe, ctx);
  }
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
