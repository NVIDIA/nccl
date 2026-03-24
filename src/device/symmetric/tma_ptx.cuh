/*
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#ifndef NCCL_PTX_CUH
#define NCCL_PTX_CUH

#include <cstdint>
#include <cstdlib>
#include <cassert>

#if __CUDA_ARCH__ >= 1000 && defined(ENABLE_TMA)

#include <cuda_awbarrier_primitives.h> // __mbarrier_*

inline __device__ void cp_async_bulk_global_to_shared(void* dest, void* src, __mbarrier_t* barrier, int size) {
  // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk
  uint32_t smem_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(dest));
  uint64_t gmem_ptr = static_cast<uint64_t>(__cvta_generic_to_global(src));
  uint32_t smem_barrier_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(barrier));

  asm volatile(
    "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
    :
    : "r"(smem_ptr),
      "l"(gmem_ptr),
      "r"(size),
      "r"(smem_barrier_ptr)
    : "memory");
}

inline __device__ void fence_proxy_async(void) { asm volatile("fence.proxy.async.shared::cta;" ::: "memory");}

inline __device__ void cp_async_bulk_shared_to_global(void* dest, void* src, int size) {
  // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk
  uint64_t dest_gmem_ptr = static_cast<uint64_t>(__cvta_generic_to_global(dest));
  uint32_t src_smem_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(src));

  asm volatile(
    "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;\n"
    :
    : "l"(dest_gmem_ptr),
      "r"(src_smem_ptr),
      "r"(size)
    : "memory");
}

inline __device__ void multimem_cp_async_bulk_shared_to_global(void* dest, void* src, int size) {
  // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk
  uint64_t dest_gmem_ptr = static_cast<uint64_t>(__cvta_generic_to_global(dest));
  uint32_t src_smem_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(src));

  asm volatile(
#if CUDART_VERSION >= 13100
    "multimem.cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;\n"
#else
    "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;\n"
#endif
    :
    : "l"(dest_gmem_ptr),
      "r"(src_smem_ptr),
      "r"(size)
    : "memory");
}

inline __device__ void cp_async_bulk_commit_group() {
  // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-commit-group
  asm volatile("cp.async.bulk.commit_group;\n" ::: "memory");
}

inline __device__ void cp_async_bulk_wait_all() {
  // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-wait-group
  asm volatile("cp.async.bulk.wait_group 0; \n" ::: "memory");
}

inline __device__ void cp_async_bulk_wait_all_read() {
  // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-wait-group
  asm volatile("cp.async.bulk.wait_group.read 0; \n" ::: "memory");
}

inline __device__ __mbarrier_token_t barrier_arrive1_tx_relaxed(__mbarrier_t* barrier, uint32_t expected_tx_count) {
  // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier-arrive
  __mbarrier_token_t token;

  asm volatile("mbarrier.arrive.expect_tx.relaxed.cta.shared::cta.b64 %0, [%1], %2;"
               : "=l"(token)
               : "r"(static_cast<unsigned int>(__cvta_generic_to_shared(barrier))), "r"(expected_tx_count)
               : "memory");
  return token;
}

inline __device__ bool barrier_try_wait_token_relaxed(__mbarrier_t* barrier, __mbarrier_token_t token) {
  // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier-test-wait-try-wait
  int __ready;
  asm volatile("{\n\t"
               ".reg .pred p;\n\t"
               "mbarrier.try_wait.relaxed.cta.shared::cta.b64 p, [%1], %2;\n\t"
               "selp.b32 %0, 1, 0, p;\n\t"
               "}"
               : "=r"(__ready)
               : "r"(static_cast<unsigned int>(__cvta_generic_to_shared(barrier))),
                 "l"(token)
               : "memory");
  return __ready;
}

#else

#define __mbarrier_t int
#define __mbarrier_token_t int

inline __device__ void cp_async_bulk_global_to_shared(void* dest, void* src, __mbarrier_t* barrier, int size) {
  return;
}

inline __device__ void fence_proxy_async(void) { return; }

inline __device__ void cp_async_bulk_shared_to_global(void* dest, void* src, int size) {
  return;
}

inline __device__ void multimem_cp_async_bulk_shared_to_global(void* dest, void* src, int size) {
  return;
}

inline __device__ void cp_async_bulk_commit_group() {
  return;
}

inline __device__ void cp_async_bulk_wait_all() {
  return;
}

inline __device__ void cp_async_bulk_wait_all_read() {
  return;
}

inline __device__ __mbarrier_token_t barrier_arrive1_tx_relaxed(__mbarrier_t* barrier, uint32_t expected_tx_count) {
  return 0;
}

inline __device__ bool barrier_try_wait_token_relaxed(__mbarrier_t* barrier, __mbarrier_token_t token) {
  return false;
}

inline __device__ void __mbarrier_init(__mbarrier_t* barrier, int i) {
  return;
}
#endif
#endif /*! NCCL_PTX_CUH */
