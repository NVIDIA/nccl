/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NIIN_SYNC_H_
#define NIIN_SYNC_H_

#include "niin/context.h"
#include "niin/tma.h"

static __device__ unsigned long long niin_test_any_cursor = 0;

// ---------------------------------------------------------------------------
// Device-side fence/quiet implementations
// ---------------------------------------------------------------------------
// Complete every context. A thread cannot know which QPs carried its own
// operations once selection is per operation, and the on-stream and host paths
// complete work issued by other threads entirely.
__device__ __forceinline__ void niin_device_drain_all_contexts() {
  ncclDevComm const& comm = niin_comm();
  if (niin_world_is_lsa_only() || !niin_has_gin()) return;
  uint32_t nCtx = niin_gin_context_count();
  if (nCtx == 1) {  // the common single-QP case: one flush, no loop
    ncclGin gin(comm, 0);
    gin.flush(ncclCoopThread{});
    return;
  }
  for (uint32_t ctx = 0; ctx < nCtx; ctx++) {
    ncclGin gin(comm, (int)ctx);
    gin.flush(ncclCoopThread{});
  }
}

// fence orders operations; it does not have to complete them. On a single GIN
// context the transport already provides that ordering: every operation goes to
// the same RC QP per peer, and an RC QP processes its work queue in order, so
// consecutive puts to a PE land in order with no completion wait. A local
// memory fence is then all fence owes the caller.
//
// Completion is only needed when something outside that one queue can reorder
// against it:
//   - several contexts, because operations rotate across QPs and two QPs are
//     unordered against each other at the receiving NIC;
//   - a bound GPUNetIO atomic sidecar, whose QPs are a separate network domain
//     from GIN, so a pending put is not ordered against a following AMO.
// Either of those turns fence into the same all-contexts drain quiet does.
__device__ __forceinline__ bool niin_fence_needs_completion() {
  if (niin_world_is_lsa_only() || !niin_has_gin()) return false;
  if (niin_gin_context_count() > 1) return true;
  return niin_gpunetio_atomic_context() != nullptr;
}

NIIN_NOINLINE_DEVICE void niin_device_fence_slow() {
  // Drain this thread's TMA bulk ops first. Unlike the context drain below this
  // is not gated on connectivity: in a mixed topology, TMA puts to LSA peers
  // are invisible to GIN, so a fence after a TMA put must order it either way.
  const bool hadTma = niin_tma_smem_registered();
  niin_tma_drain_if_registered();
  const bool hadLsaStores = niin_consume_lsa_store_pending();
  const bool hadGinOps = !niin_world_is_lsa_only() && niin_has_gin_ops_pending();
  if (hadGinOps && niin_fence_needs_completion()) {
    (void)niin_take_gin_ops_pending();
    niin_device_drain_all_contexts();
  }
  if (hadTma || hadLsaStores || hadGinOps) __threadfence_system();
}

__device__ __forceinline__ void niin_device_fence() {
  if (niin_world_is_lsa_only() && niin_tma_policy() == NVSHMEMX_TMA_DISABLE) {
    if (niin_consume_lsa_store_pending()) __threadfence_system();
    return;
  }
  niin_device_fence_slow();
}

NIIN_NOINLINE_DEVICE void niin_device_quiet_slow() {
  const bool hadTma = niin_tma_smem_registered();
  niin_tma_drain_if_registered();
  const bool hadLsaStores = niin_consume_lsa_store_pending();
  const bool hadGinOps = !niin_world_is_lsa_only() && (niin_take_gin_ops_pending() != 0u);
  if (hadGinOps) niin_device_drain_all_contexts();
  if (hadTma || hadLsaStores || hadGinOps) __threadfence_system();
}

__device__ __forceinline__ void niin_device_quiet() {
  if (niin_world_is_lsa_only() && niin_tma_policy() == NVSHMEMX_TMA_DISABLE) {
    if (niin_consume_lsa_store_pending()) __threadfence_system();
    return;
  }
  niin_device_quiet_slow();
}

// Kernel form of quiet for host-side paths. Templated so each translation unit
// including NIIN gets its own weak instantiation rather than a duplicate symbol
// at link time.
template<int = 0>
__global__ void niin_quiet_kernel() {
  niin_device_quiet();
}

// ---------------------------------------------------------------------------
// __host__ __device__ wrappers
// ---------------------------------------------------------------------------
__host__ __device__ __forceinline__ void nvshmem_fence() {
#ifdef __CUDA_ARCH__
  niin_device_fence();
#else
  extern void niin_host_fence();
  niin_host_fence();
#endif
}

__host__ __device__ __forceinline__ void nvshmem_quiet() {
#ifdef __CUDA_ARCH__
  niin_device_quiet();
#else
  extern void niin_host_quiet();
  niin_host_quiet();
#endif
}

// ---------------------------------------------------------------------------
// nvshmem_TYPE_wait_until: spin-wait on a local symmetric variable
// ---------------------------------------------------------------------------
#define NIIN_DEFINE_WAIT_UNTIL(TYPENAME, TYPE)                                \
__device__ __forceinline__ void nvshmem_##TYPENAME##_wait_until(              \
    TYPE* ivar, int cmp, TYPE cmp_value) {                                    \
  volatile TYPE* v = (volatile TYPE*)ivar;                                    \
  while (!niin_cmp_eval(cmp, (long long)*v, (long long)cmp_value)) {}        \
  /* Acquire fence: ensure all remote stores ordered before the signal */     \
  /* are visible to this thread after wait_until returns. */                   \
  cuda::atomic_thread_fence(cuda::memory_order_acquire, cuda::thread_scope_system); \
}

NIIN_WAIT_TYPES(NIIN_DEFINE_WAIT_UNTIL)
#undef NIIN_DEFINE_WAIT_UNTIL

// Unsigned specializations that use unsigned comparison
__device__ __forceinline__ void nvshmem_uint_wait_until_u(
    unsigned int* ivar, int cmp, unsigned int cmp_value) {
  volatile unsigned int* v = (volatile unsigned int*)ivar;
  while (!niin_cmp_eval_u(cmp, (unsigned long long)*v, (unsigned long long)cmp_value)) {}
}

// ---------------------------------------------------------------------------
// nvshmem_TYPE_test: non-blocking test on a local symmetric variable
// ---------------------------------------------------------------------------
#define NIIN_DEFINE_TEST(TYPENAME, TYPE)                                      \
__device__ __forceinline__ int nvshmem_##TYPENAME##_test(                     \
    TYPE* ivar, int cmp, TYPE cmp_value) {                                    \
  volatile TYPE* v = (volatile TYPE*)ivar;                                    \
  return niin_cmp_eval(cmp, (long long)*v, (long long)cmp_value) ? 1 : 0;   \
}

NIIN_WAIT_TYPES(NIIN_DEFINE_TEST)
#undef NIIN_DEFINE_TEST

// ---------------------------------------------------------------------------
// nvshmem_TYPE_wait_until_all: wait for all elements to meet condition
// ---------------------------------------------------------------------------
#define NIIN_DEFINE_WAIT_UNTIL_ALL(TYPENAME, TYPE)                            \
__device__ __forceinline__ void nvshmem_##TYPENAME##_wait_until_all(          \
    TYPE* ivars, size_t nelems, const int* status, int cmp, TYPE cmp_value) { \
  for (size_t i = 0; i < nelems; i++) {                                       \
    if (status != nullptr && status[i] != 0) continue;                        \
    nvshmem_##TYPENAME##_wait_until(&ivars[i], cmp, cmp_value);              \
  }                                                                            \
}

NIIN_WAIT_TYPES(NIIN_DEFINE_WAIT_UNTIL_ALL)
#undef NIIN_DEFINE_WAIT_UNTIL_ALL

// ---------------------------------------------------------------------------
// nvshmem_TYPE_wait_until_any: wait for any element to meet condition
// Returns index of the first element that satisfies the condition.
// ---------------------------------------------------------------------------
#define NIIN_DEFINE_WAIT_UNTIL_ANY(TYPENAME, TYPE)                            \
__device__ __forceinline__ size_t nvshmem_##TYPENAME##_wait_until_any(        \
    TYPE* ivars, size_t nelems, const int* status, int cmp, TYPE cmp_value) { \
  while (true) {                                                               \
    bool any_pending = false;                                                  \
    for (size_t i = 0; i < nelems; i++) {                                     \
      if (status != nullptr && status[i] != 0) continue;                      \
      any_pending = true;                                                      \
      if (nvshmem_##TYPENAME##_test(&ivars[i], cmp, cmp_value)) {            \
        cuda::atomic_thread_fence(cuda::memory_order_acquire,                 \
                                  cuda::thread_scope_system);                 \
        return i;                                                              \
      }                                                                        \
    }                                                                          \
    if (!any_pending) return SIZE_MAX;                                         \
  }                                                                            \
}

NIIN_WAIT_TYPES(NIIN_DEFINE_WAIT_UNTIL_ANY)
#undef NIIN_DEFINE_WAIT_UNTIL_ANY

// ---------------------------------------------------------------------------
// nvshmem_TYPE_wait_until_some: wait for at least one, return count
// ---------------------------------------------------------------------------
#define NIIN_DEFINE_WAIT_UNTIL_SOME(TYPENAME, TYPE)                           \
__device__ __forceinline__ size_t nvshmem_##TYPENAME##_wait_until_some(       \
    TYPE* ivars, size_t nelems, size_t* indices, const int* status,           \
    int cmp, TYPE cmp_value) {                                                \
  while (true) {                                                               \
    size_t count = 0;                                                          \
    bool any_pending = false;                                                  \
    for (size_t i = 0; i < nelems; i++) {                                     \
      if (status != nullptr && status[i] != 0) continue;                      \
      any_pending = true;                                                      \
      if (nvshmem_##TYPENAME##_test(&ivars[i], cmp, cmp_value)) {            \
        if (indices != nullptr) indices[count] = i;                           \
        count++;                                                               \
      }                                                                        \
    }                                                                          \
    if (count != 0) {                                                          \
      cuda::atomic_thread_fence(cuda::memory_order_acquire,                   \
                                cuda::thread_scope_system);                   \
      return count;                                                            \
    }                                                                          \
    if (!any_pending) return 0;                                                \
  }                                                                            \
}

NIIN_WAIT_TYPES(NIIN_DEFINE_WAIT_UNTIL_SOME)
#undef NIIN_DEFINE_WAIT_UNTIL_SOME

// ---------------------------------------------------------------------------
// nvshmem_TYPE_test_all / test_any / test_some
// ---------------------------------------------------------------------------
#define NIIN_DEFINE_TEST_ALL(TYPENAME, TYPE)                                  \
__device__ __forceinline__ int nvshmem_##TYPENAME##_test_all(                 \
    TYPE* ivars, size_t nelems, const int* status, int cmp, TYPE cmp_value) { \
  for (size_t i = 0; i < nelems; i++) {                                       \
    if (status != nullptr && status[i] != 0) continue;                        \
    if (!nvshmem_##TYPENAME##_test(&ivars[i], cmp, cmp_value))               \
      return 0;                                                                \
  }                                                                            \
  cuda::atomic_thread_fence(cuda::memory_order_acquire,                       \
                            cuda::thread_scope_system);                       \
  return 1;                                                                    \
}

NIIN_WAIT_TYPES(NIIN_DEFINE_TEST_ALL)
#undef NIIN_DEFINE_TEST_ALL

#define NIIN_DEFINE_TEST_ANY(TYPENAME, TYPE)                                  \
__device__ __forceinline__ size_t nvshmem_##TYPENAME##_test_any(              \
    TYPE* ivars, size_t nelems, const int* status, int cmp, TYPE cmp_value) { \
  size_t start = 0;                                                            \
  if (status == nullptr && nelems != 0) {                                      \
    start = atomicAdd((unsigned long long*)&niin_test_any_cursor, 1ULL) % nelems; \
  }                                                                            \
  for (size_t step = 0; step < nelems; step++) {                              \
    size_t i = (start + step) % nelems;                                        \
    if (status != nullptr && status[i] != 0) continue;                        \
    if (nvshmem_##TYPENAME##_test(&ivars[i], cmp, cmp_value)) {              \
      cuda::atomic_thread_fence(cuda::memory_order_acquire,                   \
                                cuda::thread_scope_system);                   \
      return i;                                                                \
    }                                                                          \
  }                                                                            \
  return SIZE_MAX;                                                             \
}

NIIN_WAIT_TYPES(NIIN_DEFINE_TEST_ANY)
#undef NIIN_DEFINE_TEST_ANY

#define NIIN_DEFINE_TEST_SOME(TYPENAME, TYPE)                                 \
__device__ __forceinline__ size_t nvshmem_##TYPENAME##_test_some(             \
    TYPE* ivars, size_t nelems, size_t* indices, const int* status,           \
    int cmp, TYPE cmp_value) {                                                \
  size_t count = 0;                                                            \
  for (size_t i = 0; i < nelems; i++) {                                       \
    if (status != nullptr && status[i] != 0) continue;                        \
      if (nvshmem_##TYPENAME##_test(&ivars[i], cmp, cmp_value)) {              \
        if (indices != nullptr) indices[count] = i;                             \
        count++;                                                                 \
      }                                                                          \
  }                                                                            \
  if (count != 0) {                                                            \
    cuda::atomic_thread_fence(cuda::memory_order_acquire,                     \
                              cuda::thread_scope_system);                     \
  }                                                                            \
  return count;                                                                \
}

NIIN_WAIT_TYPES(NIIN_DEFINE_TEST_SOME)
#undef NIIN_DEFINE_TEST_SOME

// ---------------------------------------------------------------------------
// Vector variants: per-element comparison values (TYPE* cmp_values array)
// ---------------------------------------------------------------------------

#define NIIN_DEFINE_WAIT_UNTIL_ALL_VECTOR(TYPENAME, TYPE)                     \
__device__ __forceinline__ void nvshmem_##TYPENAME##_wait_until_all_vector(   \
    TYPE* ivars, size_t nelems, const int* status, int cmp, TYPE* cmp_values) { \
  for (size_t i = 0; i < nelems; i++) {                                       \
    if (status != nullptr && status[i] != 0) continue;                        \
    nvshmem_##TYPENAME##_wait_until(&ivars[i], cmp, cmp_values[i]);          \
  }                                                                            \
}

NIIN_WAIT_TYPES(NIIN_DEFINE_WAIT_UNTIL_ALL_VECTOR)
#undef NIIN_DEFINE_WAIT_UNTIL_ALL_VECTOR

#define NIIN_DEFINE_WAIT_UNTIL_ANY_VECTOR(TYPENAME, TYPE)                     \
__device__ __forceinline__ size_t nvshmem_##TYPENAME##_wait_until_any_vector( \
    TYPE* ivars, size_t nelems, const int* status, int cmp, TYPE* cmp_values) { \
  while (true) {                                                               \
    bool any_pending = false;                                                  \
    for (size_t i = 0; i < nelems; i++) {                                     \
      if (status != nullptr && status[i] != 0) continue;                      \
      any_pending = true;                                                      \
      if (nvshmem_##TYPENAME##_test(&ivars[i], cmp, cmp_values[i])) {        \
        cuda::atomic_thread_fence(cuda::memory_order_acquire,                 \
                                  cuda::thread_scope_system);                 \
        return i;                                                              \
      }                                                                        \
    }                                                                          \
    if (!any_pending) return SIZE_MAX;                                         \
  }                                                                            \
}

NIIN_WAIT_TYPES(NIIN_DEFINE_WAIT_UNTIL_ANY_VECTOR)
#undef NIIN_DEFINE_WAIT_UNTIL_ANY_VECTOR

#define NIIN_DEFINE_WAIT_UNTIL_SOME_VECTOR(TYPENAME, TYPE)                    \
__device__ __forceinline__ size_t nvshmem_##TYPENAME##_wait_until_some_vector( \
    TYPE* ivars, size_t nelems, size_t* indices, const int* status,           \
    int cmp, TYPE* cmp_values) {                                              \
  while (true) {                                                               \
    size_t count = 0;                                                          \
    bool any_pending = false;                                                  \
    for (size_t i = 0; i < nelems; i++) {                                     \
      if (status != nullptr && status[i] != 0) continue;                      \
      any_pending = true;                                                      \
      if (nvshmem_##TYPENAME##_test(&ivars[i], cmp, cmp_values[i])) {        \
        if (indices != nullptr) indices[count] = i;                           \
        count++;                                                               \
      }                                                                        \
    }                                                                          \
    if (count != 0) {                                                          \
      cuda::atomic_thread_fence(cuda::memory_order_acquire,                   \
                                cuda::thread_scope_system);                   \
      return count;                                                            \
    }                                                                          \
    if (!any_pending) return 0;                                                \
  }                                                                            \
}

NIIN_WAIT_TYPES(NIIN_DEFINE_WAIT_UNTIL_SOME_VECTOR)
#undef NIIN_DEFINE_WAIT_UNTIL_SOME_VECTOR

#define NIIN_DEFINE_TEST_ALL_VECTOR(TYPENAME, TYPE)                           \
__device__ __forceinline__ int nvshmem_##TYPENAME##_test_all_vector(          \
    TYPE* ivars, size_t nelems, const int* status, int cmp, TYPE* cmp_values) { \
  for (size_t i = 0; i < nelems; i++) {                                       \
    if (status != nullptr && status[i] != 0) continue;                        \
    if (!nvshmem_##TYPENAME##_test(&ivars[i], cmp, cmp_values[i]))           \
      return 0;                                                                \
  }                                                                            \
  cuda::atomic_thread_fence(cuda::memory_order_acquire,                       \
                            cuda::thread_scope_system);                       \
  return 1;                                                                    \
}

NIIN_WAIT_TYPES(NIIN_DEFINE_TEST_ALL_VECTOR)
#undef NIIN_DEFINE_TEST_ALL_VECTOR

#define NIIN_DEFINE_TEST_ANY_VECTOR(TYPENAME, TYPE)                           \
__device__ __forceinline__ size_t nvshmem_##TYPENAME##_test_any_vector(       \
    TYPE* ivars, size_t nelems, const int* status, int cmp, TYPE* cmp_values) { \
  size_t start = 0;                                                            \
  if (status == nullptr && nelems != 0) {                                      \
    start = atomicAdd((unsigned long long*)&niin_test_any_cursor, 1ULL) % nelems; \
  }                                                                            \
  for (size_t step = 0; step < nelems; step++) {                              \
    size_t i = (start + step) % nelems;                                        \
    if (status != nullptr && status[i] != 0) continue;                        \
    if (nvshmem_##TYPENAME##_test(&ivars[i], cmp, cmp_values[i])) {          \
      cuda::atomic_thread_fence(cuda::memory_order_acquire,                   \
                                cuda::thread_scope_system);                   \
      return i;                                                                \
    }                                                                          \
  }                                                                            \
  return SIZE_MAX;                                                             \
}

NIIN_WAIT_TYPES(NIIN_DEFINE_TEST_ANY_VECTOR)
#undef NIIN_DEFINE_TEST_ANY_VECTOR

#define NIIN_DEFINE_TEST_SOME_VECTOR(TYPENAME, TYPE)                          \
__device__ __forceinline__ size_t nvshmem_##TYPENAME##_test_some_vector(      \
    TYPE* ivars, size_t nelems, size_t* indices, const int* status,           \
    int cmp, TYPE* cmp_values) {                                              \
  size_t count = 0;                                                            \
  for (size_t i = 0; i < nelems; i++) {                                       \
    if (status != nullptr && status[i] != 0) continue;                        \
      if (nvshmem_##TYPENAME##_test(&ivars[i], cmp, cmp_values[i])) {          \
        if (indices != nullptr) indices[count] = i;                             \
        count++;                                                                 \
      }                                                                          \
  }                                                                            \
  if (count != 0) {                                                            \
    cuda::atomic_thread_fence(cuda::memory_order_acquire,                     \
                              cuda::thread_scope_system);                     \
  }                                                                            \
  return count;                                                                \
}

NIIN_WAIT_TYPES(NIIN_DEFINE_TEST_SOME_VECTOR)
#undef NIIN_DEFINE_TEST_SOME_VECTOR

#endif // NIIN_SYNC_H_
