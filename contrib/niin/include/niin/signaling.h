/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NIIN_SIGNALING_H_
#define NIIN_SIGNALING_H_

#include "niin/context.h"

__device__ __forceinline__ void niin_gin_flush_thread() {
  if (!niin_has_gin()) return;
  ncclGin gin(niin_comm(), niin_gin_context_index());
  gin.flush(ncclCoopThread{});
}

__device__ __forceinline__ void niin_deliver_signal_remote_va(
    size_t sigOffset, uint64_t signal, int sig_op, int pe) {
  if (!niin_has_gin()) {
    NIIN_NOT_IMPLEMENTED_VOID("nvshmem_put_signal (network without GIN)");
    return;
  }
  ncclDevComm const& comm = niin_comm();
  ncclTeam world = ncclTeamWorld(comm);
  ncclGin gin(comm, niin_gin_context_index());
  if (sig_op == NVSHMEM_SIGNAL_SET) {
    gin.putValue<uint64_t>(world, pe, niin_heap_window(), sigOffset, signal);
  } else {
    gin.signal(world, pe, ncclGin_VASignalAdd{niin_heap_window(), sigOffset, signal});
  }
  gin.flush(ncclCoopThread{});
}

__device__ __forceinline__ void niin_deliver_signal(
    uint64_t* sig_addr, uint64_t signal, int sig_op, int pe) {
  size_t sigOffset = niin_sym_offset(sig_addr);
  if (pe == nvshmem_my_pe()) {
    if (sig_op == NVSHMEM_SIGNAL_SET) atomicExch((unsigned long long*)sig_addr, signal);
    else atomicAdd((unsigned long long*)sig_addr, signal);
    __threadfence_system();
    return;
  }
  if (niin_is_lsa_peer(pe) && niin_peer_native_atomic()) {
    uint64_t* peerSig = (uint64_t*)niin_get_peer_ptr(sigOffset, pe);
    if (sig_op == NVSHMEM_SIGNAL_SET) atomicExch((unsigned long long*)peerSig, signal);
    else atomicAdd((unsigned long long*)peerSig, signal);
    __threadfence_system();
    return;
  }
  niin_deliver_signal_remote_va(sigOffset, signal, sig_op, pe);
}

__device__ __forceinline__ bool niin_use_separate_put_signal(int pe, int sig_op) {
  if (sig_op == NVSHMEM_SIGNAL_SET) return true;
  if (niin_force_separate_put_signal()) return true;
  if (!niin_is_lsa_peer(pe)) return true;
  return niin_is_lsa_peer(pe) && !niin_peer_native_atomic();
}

__device__ __forceinline__ void niin_putmem_signal_impl(
    void* dest, const void* src, size_t bytes,
    uint64_t* sig_addr, uint64_t signal, int sig_op, int pe) {
  size_t dstOffset = niin_sym_offset(dest);
  if (pe == nvshmem_my_pe()) {
    niin_memcpy_to_peer(dest, src, bytes);
    __threadfence_system();
    niin_deliver_signal(sig_addr, signal, sig_op, pe);
    return;
  }
  if (niin_is_lsa_peer(pe)) {
    void* peerDst = niin_get_peer_ptr(dstOffset, pe);
    niin_memcpy_to_peer(peerDst, src, bytes);
    __threadfence_system();
    niin_deliver_signal(sig_addr, signal, sig_op, pe);
    return;
  }

  if (niin_use_separate_put_signal(pe, sig_op)) {
    niin_gin_put(dstOffset, src, bytes, pe);
    niin_gin_flush_thread();
    niin_deliver_signal(sig_addr, signal, sig_op, pe);
    return;
  }

  size_t sigOffset = niin_sym_offset(sig_addr);
  ncclDevComm const& comm = niin_comm();
  ncclTeam world = ncclTeamWorld(comm);
  ncclGin gin(comm, niin_gin_context_index());
  ncclGin_VASignalAdd sigAction;
  sigAction.signalWindow = niin_heap_window();
  sigAction.signalOffset = sigOffset;
  sigAction.value = signal;
  gin.put<ncclGin_VASignalAdd>(
    world, pe,
    niin_heap_window(), dstOffset,
    niin_heap_window(), niin_sym_offset(src), bytes,
    sigAction
  );
}

// ---------------------------------------------------------------------------
// nvshmem_TYPE_put_signal: put data + signal on completion
//
// Native-atomic LSA path: store data, then peer-pointer atomic on the signal.
// Non-native-atomic or forced-split path: put, threadfence/flush, then signal.
// Network path: fused VASignalAdd where possible, otherwise split put+signal.
// ---------------------------------------------------------------------------

#define NIIN_DEFINE_PUT_SIGNAL(TYPENAME, TYPE)                                \
__device__ __forceinline__ void nvshmem_##TYPENAME##_put_signal(              \
    TYPE* dest, const TYPE* src, size_t nelems,                               \
    uint64_t* sig_addr, uint64_t signal, int sig_op, int pe) {               \
  size_t bytes = nelems * sizeof(TYPE);                                        \
  niin_putmem_signal_impl(dest, src, bytes, sig_addr, signal, sig_op, pe);    \
}

NIIN_STANDARD_RMA_TYPES(NIIN_DEFINE_PUT_SIGNAL)
#undef NIIN_DEFINE_PUT_SIGNAL

// Untyped variant
__device__ __forceinline__ void nvshmem_putmem_signal(
    void* dest, const void* src, size_t bytes,
    uint64_t* sig_addr, uint64_t signal, int sig_op, int pe) {
  niin_putmem_signal_impl(dest, src, bytes, sig_addr, signal, sig_op, pe);
}

// NBI variant
#define NIIN_DEFINE_PUT_SIGNAL_NBI(TYPENAME, TYPE)                            \
__device__ __forceinline__ void nvshmem_##TYPENAME##_put_signal_nbi(          \
    TYPE* dest, const TYPE* src, size_t nelems,                               \
    uint64_t* sig_addr, uint64_t signal, int sig_op, int pe) {               \
  nvshmem_##TYPENAME##_put_signal(dest, src, nelems, sig_addr, signal, sig_op, pe); \
}

NIIN_STANDARD_RMA_TYPES(NIIN_DEFINE_PUT_SIGNAL_NBI)
#undef NIIN_DEFINE_PUT_SIGNAL_NBI

__device__ __forceinline__ void nvshmem_putmem_signal_nbi(
    void* dest, const void* src, size_t bytes,
    uint64_t* sig_addr, uint64_t signal, int sig_op, int pe) {
  nvshmem_putmem_signal(dest, src, bytes, sig_addr, signal, sig_op, pe);
}

// ---------------------------------------------------------------------------
// Sized put_signal (put8_signal, put16_signal, ..., put128_signal)
// ---------------------------------------------------------------------------
#define NIIN_DEFINE_PUT_SIGNAL_SIZED(SIZE, NBYTES)                             \
__device__ __forceinline__ void nvshmem_put##SIZE##_signal(                    \
    void* dest, const void* src, size_t nelems,                               \
    uint64_t* sig_addr, uint64_t signal, int sig_op, int pe) {               \
  nvshmem_putmem_signal(dest, src, nelems * NBYTES, sig_addr, signal, sig_op, pe); \
}

NIIN_SIZED_RMA(NIIN_DEFINE_PUT_SIGNAL_SIZED)
#undef NIIN_DEFINE_PUT_SIGNAL_SIZED

#define NIIN_DEFINE_PUT_SIGNAL_SIZED_NBI(SIZE, NBYTES)                         \
__device__ __forceinline__ void nvshmem_put##SIZE##_signal_nbi(                \
    void* dest, const void* src, size_t nelems,                               \
    uint64_t* sig_addr, uint64_t signal, int sig_op, int pe) {               \
  nvshmem_putmem_signal(dest, src, nelems * NBYTES, sig_addr, signal, sig_op, pe); \
}

NIIN_SIZED_RMA(NIIN_DEFINE_PUT_SIGNAL_SIZED_NBI)
#undef NIIN_DEFINE_PUT_SIGNAL_SIZED_NBI

// ---------------------------------------------------------------------------
// nvshmem_signal_fetch: read a signal value atomically
// ---------------------------------------------------------------------------
__device__ __forceinline__ uint64_t nvshmem_signal_fetch(const uint64_t* sig_addr) {
  return atomicAdd((unsigned long long*)sig_addr, 0ULL);
}

// ---------------------------------------------------------------------------
// nvshmem_signal_wait_until: spin-wait until signal meets condition
// ---------------------------------------------------------------------------
__device__ __forceinline__ uint64_t nvshmem_signal_wait_until(
    uint64_t* sig_addr, int cmp, uint64_t cmp_value) {
  uint64_t val;
  do {
    val = atomicAdd((unsigned long long*)sig_addr, 0ULL);
  } while (!niin_cmp_eval_u(cmp, val, cmp_value));
  // Acquire fence: data written before the signal must be visible
  cuda::atomic_thread_fence(cuda::memory_order_acquire, cuda::thread_scope_system);
  return val;
}

#endif // NIIN_SIGNALING_H_
