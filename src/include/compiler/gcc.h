/*************************************************************************
 * Copyright (c) 2019-2025, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_COMPILER_GCC_H
#define NCCL_COMPILER_GCC_H

// Helper macros to convert C++ memory ordering to GCC atomic ordering
#define NCCL_CONVERT_ORDER(order) \
  ((order) == std::memory_order_relaxed ? __ATOMIC_RELAXED : \
   (order) == std::memory_order_consume ? __ATOMIC_CONSUME : \
   (order) == std::memory_order_acquire ? __ATOMIC_ACQUIRE : \
   (order) == std::memory_order_release ? __ATOMIC_RELEASE : \
   (order) == std::memory_order_acq_rel ? __ATOMIC_ACQ_REL : \
   (order) == std::memory_order_seq_cst ? __ATOMIC_SEQ_CST : \
   __ATOMIC_SEQ_CST)

#define COMPILER_ATOMIC_LOAD(ptr, order) \
  __atomic_load_n((ptr), NCCL_CONVERT_ORDER(order))
#define COMPILER_ATOMIC_LOAD_DEST(ptr, dest, order) do { \
  __atomic_load((ptr), (dest), NCCL_CONVERT_ORDER(order)); \
} while(0)
#define COMPILER_ATOMIC_STORE(ptr, val, order) \
  __atomic_store_n((ptr), (val), NCCL_CONVERT_ORDER(order))
#define COMPILER_ATOMIC_EXCHANGE(ptr, val, order) \
  __atomic_exchange_n((ptr), (val), NCCL_CONVERT_ORDER(order))
#define COMPILER_ATOMIC_COMPARE_EXCHANGE(ptr, expected, desired, success_order, failure_order) \
  __atomic_compare_exchange_n((ptr), (expected), (desired), true, NCCL_CONVERT_ORDER(success_order), NCCL_CONVERT_ORDER(failure_order))

#define COMPILER_ATOMIC_FETCH_ADD(ptr, val, order) __atomic_fetch_add((ptr), (val), NCCL_CONVERT_ORDER(order))
#define COMPILER_ATOMIC_ADD_FETCH(ptr, val, order) __atomic_add_fetch((ptr), (val), NCCL_CONVERT_ORDER(order))
#define COMPILER_ATOMIC_SUB_FETCH(ptr, val, order) __atomic_sub_fetch((ptr), (val), NCCL_CONVERT_ORDER(order))

#define COMPILER_PREFETCH(addr) __builtin_prefetch((addr))

#define COMPILER_POPCOUNT32(x) __builtin_popcount(x)
#define COMPILER_POPCOUNT64(x) __builtin_popcountll(x)

#define COMPILER_EXPECT(x, v) __builtin_expect((x), (v))

// Find First Set (FFS) - returns index of first set bit (1-indexed), 0 if no bits set
#define COMPILER_FFS(x) __builtin_ffs(x)
#define COMPILER_FFSL(x) __builtin_ffsl(x)
#define COMPILER_FFSLL(x) __builtin_ffsll(x)

// Count Leading Zeros (CLZ) - undefined behavior if x == 0
#define COMPILER_CLZ(x) __builtin_clz(x)
#define COMPILER_CLZL(x) __builtin_clzl(x)
#define COMPILER_CLZLL(x) __builtin_clzll(x)

// Byte Swap
#define COMPILER_BSWAP16(x) __builtin_bswap16(x)
#define COMPILER_BSWAP32(x) __builtin_bswap32(x)
#define COMPILER_BSWAP64(x) __builtin_bswap64(x)

// Compiler hints
#define COMPILER_ASSUME_ALIGNED(ptr, alignment) __builtin_assume_aligned((ptr), (alignment))

#endif // NCCL_COMPILER_GCC_H
