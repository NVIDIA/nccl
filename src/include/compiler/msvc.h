/*************************************************************************
 * Copyright (c) 2019-2025, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_COMPILER_MSVC_H
#define NCCL_COMPILER_MSVC_H

#include <intrin.h>
#include <emmintrin.h>

// TODO: Check the memory orders and the fences
static inline long COMPILER_ATOMIC_LOAD_impl(volatile long* ptr, std::memory_order order) {
  long result;
  if (order == std::memory_order_relaxed) {
    result = InterlockedAdd64NoFence(ptr, 0);
  } else if (order == std::memory_order_acquire) {
    result = InterlockedAdd64Acquire(ptr, 0);
  } else if (order == std::memory_order_release) {
    result = InterlockedAdd64Release(ptr, 0);
  } else if (order == std::memory_order_acq_rel) {
    result = InterlockedAdd64AcquireRelease(ptr, 0);
  } else if (order == std::memory_order_seq_cst) {
    result = InterlockedAdd64(ptr, 0);
  }
  return result;
}

#define COMPILER_ATOMIC_LOAD(ptr, order) COMPILER_ATOMIC_LOAD_impl((volatile long*)(ptr), (order))
#define COMPILER_ATOMIC_LOAD_DEST(ptr, dest, order) do { \
  if (order == std::memory_order_relaxed) { \
    *(dest) = InterlockedAdd64NoFence((volatile long*)(ptr), 0); \
  } else if (order == std::memory_order_acquire) { \
    *(dest) = InterlockedAdd64Acquire((volatile long*)(ptr), 0); \
  } else if (order == std::memory_order_release) { \
    *(dest) = InterlockedAdd64Release((volatile long*)(ptr), 0); \
  } else if (order == std::memory_order_acq_rel) { \
    *(dest) = InterlockedAdd64AcquireRelease((volatile long*)(ptr), 0); \
  } else if (order == std::memory_order_seq_cst) { \
    *(dest) = InterlockedAdd64((volatile long*)(ptr), 0); \
  } \
} while(0)

#define COMPILER_ATOMIC_STORE(ptr, val, order) do { \
  if (order == std::memory_order_relaxed) { \
    InterlockedExchange64NoFence((volatile long*)(ptr), (val)); \
  } else if (order == std::memory_order_acquire) { \
    InterlockedExchange64Acquire((volatile long*)(ptr), (val)); \
  } else if (order == std::memory_order_release) { \
    InterlockedExchange64Release((volatile long*)(ptr), (val)); \
  } else if (order == std::memory_order_acq_rel) { \
    InterlockedExchange64AcquireRelease((volatile long*)(ptr), (val)); \
  } else if (order == std::memory_order_seq_cst) { \
    InterlockedExchange64((volatile long*)(ptr), (val)); \
  } \
} while(0)

// COMPILER_ATOMIC_EXCHANGE - Atomic exchange, returns old value
static inline long COMPILER_ATOMIC_EXCHANGE_impl(volatile long* ptr, long val, std::memory_order order) {
  long result;
  if (order == std::memory_order_relaxed) {
    result = InterlockedExchange64NoFence(ptr, val);
  } else if (order == std::memory_order_acquire) {
    result = InterlockedExchange64Acquire(ptr, val);
  } else if (order == std::memory_order_release) {
    result = InterlockedExchange64Release(ptr, val);
  } else if (order == std::memory_order_acq_rel) {
    result = InterlockedExchange64AcquireRelease(ptr, val);
  } else if (order == std::memory_order_seq_cst) {
    result = InterlockedExchange64(ptr, val);
  }
  return result;
}
#define COMPILER_ATOMIC_EXCHANGE(ptr, val, order) COMPILER_ATOMIC_EXCHANGE_impl((volatile long*)(ptr), (val), (order))

// COMPILER_ATOMIC_COMPARE_EXCHANGE - Compare and exchange
// Note: To match GCC API, 'desired' is passed by value, 'expected' is a pointer
// Returns true if exchange succeeded, updates *expected with old value on failure
static inline bool COMPILER_ATOMIC_COMPARE_EXCHANGE_impl(
  volatile long* ptr,
  long* expected,
  long desired,
  std::memory_order success_order,
  std::memory_order failure_order) {
  long old;
  if (success_order == std::memory_order_relaxed) {
    old = InterlockedCompareExchange64NoFence(ptr, desired, *expected);
  } else if (success_order == std::memory_order_acquire) {
    old = InterlockedCompareExchange64Acquire(ptr, desired, *expected);
  } else if (success_order == std::memory_order_release) {
    old = InterlockedCompareExchange64Release(ptr, desired, *expected);
  } else if (success_order == std::memory_order_acq_rel) {
    old = InterlockedCompareExchange64AcquireRelease(ptr, desired, *expected);
  } else if (success_order == std::memory_order_seq_cst) {
    old = InterlockedCompareExchange64(ptr, desired, *expected);
  }
  bool success = (old == *expected);
  if (!success) {
    *expected = old;
  }
  // Use appropriate fence based on success/failure
  std::atomic_thread_fence(success ? success_order : failure_order);
  return success;
}
#define COMPILER_ATOMIC_COMPARE_EXCHANGE(ptr, expected, desired, success_order, failure_order) \
  COMPILER_ATOMIC_COMPARE_EXCHANGE_impl((volatile long*)(ptr), (expected), (desired), (success_order), (failure_order))

// COMPILER_ATOMIC_ADD_FETCH - Add and return new value
static inline long COMPILER_ATOMIC_ADD_FETCH_impl(volatile long* ptr, long val, std::memory_order order) {
  long result;
  if (order == std::memory_order_relaxed) {
    result = InterlockedAdd64NoFence(ptr, val);
  } else if (order == std::memory_order_acquire) {
    result = InterlockedAdd64Acquire(ptr, val);
  } else if (order == std::memory_order_release) {
    result = InterlockedAdd64Release(ptr, val);
  } else if (order == std::memory_order_acq_rel) {
    result = InterlockedAdd64AcquireRelease(ptr, val);
  } else if (order == std::memory_order_seq_cst) {
    result = InterlockedAdd64(ptr, val);
  }
  return result;
}

// COMPILER_ATOMIC_FETCH_ADD - Fetch old value then add
static inline long COMPILER_ATOMIC_FETCH_ADD_impl(volatile long* ptr, long val, std::memory_order order) {
  long result;
  if (order == std::memory_order_relaxed) {
    result = InterlockedExchangeAdd64NoFence(ptr, val);
  } else if (order == std::memory_order_acquire) {
    result = InterlockedExchangeAdd64Acquire(ptr, val);
  } else if (order == std::memory_order_release) {
    result = InterlockedExchangeAdd64Release(ptr, val);
  } else if (order == std::memory_order_acq_rel) {
    result = InterlockedExchangeAdd64AcquireRelease(ptr, val);
  } else if (order == std::memory_order_seq_cst) {
    result = InterlockedExchangeAdd64(ptr, val);
  }
  return result;
}
#define COMPILER_ATOMIC_FETCH_ADD(ptr, val, order) COMPILER_ATOMIC_FETCH_ADD_impl((volatile long*)(ptr), (val), (order))
#define COMPILER_ATOMIC_ADD_FETCH(ptr, val, order) COMPILER_ATOMIC_ADD_FETCH_impl((volatile long*)(ptr), (val), (order))

// COMPILER_ATOMIC_SUB_FETCH - Subtract and return new value
static inline long COMPILER_ATOMIC_SUB_FETCH_impl(volatile long* ptr, long val, std::memory_order order) {
  long result;
  if (order == std::memory_order_relaxed) {
    result = InterlockedAdd64NoFence(ptr, -val);
  } else if (order == std::memory_order_acquire) {
    result = InterlockedAdd64Acquire(ptr, -val);
  } else if (order == std::memory_order_release) {
    result = InterlockedAdd64Release(ptr, -val);
  } else if (order == std::memory_order_acq_rel) {
    result = InterlockedAdd64AcquireRelease(ptr, -val);
  } else if (order == std::memory_order_seq_cst) {
    result = InterlockedAdd64(ptr, -val);
  }
  return result;
}
#define COMPILER_ATOMIC_SUB_FETCH(ptr, val, order) COMPILER_ATOMIC_SUB_FETCH_impl((volatile long*)(ptr), (val), (order))

// Prefetch
#define COMPILER_PREFETCH(addr) _mm_prefetch((const char*)(addr), _MM_HINT_T0)

// Population count
#define COMPILER_POPCOUNT32(x) __popcnt(x)
#define COMPILER_POPCOUNT64(x) __popcnt64(x)

// Branch prediction hints (MSVC doesn't support these, so no-op)
#define COMPILER_EXPECT(x, v) (x)

// Find First Set (FFS) - returns index of first set bit (1-indexed), 0 if no bits set
// MSVC uses _BitScanForward which gives 0-indexed position
inline int NCCL_FFS_impl(int x) {
  unsigned long index;
  return _BitScanForward(&index, (unsigned long)x) ? (int)(index + 1) : 0;
}
inline int NCCL_FFSl_impl(long x) {
  unsigned long index;
  return _BitScanForward(&index, (unsigned long)x) ? (int)(index + 1) : 0;
}
inline int NCCL_FFSll_impl(long long x) {
  unsigned long index;
#ifdef _WIN64
  return _BitScanForward64(&index, (unsigned __int64)x) ? (int)(index + 1) : 0;
#else
  // For 32-bit, check low 32 bits first, then high 32 bits
  if (_BitScanForward(&index, (unsigned long)(x & 0xFFFFFFFF))) return (int)(index + 1);
  if (_BitScanForward(&index, (unsigned long)(x >> 32))) return (int)(index + 33);
  return 0;
#endif
}
#define COMPILER_FFS(x) NCCL_FFS_impl(x)
#define COMPILER_FFSL(x) NCCL_FFSl_impl(x)
#define COMPILER_FFSLL(x) NCCL_FFSll_impl(x)

// Count Leading Zeros (CLZ) - undefined behavior if x == 0 (to match GCC behavior)
// MSVC uses _BitScanReverse which gives position of highest bit
inline int nccl_clz_impl(unsigned int x) {
  unsigned long index;
  _BitScanReverse(&index, x);
  return 31 - (int)index;
}
inline int nccl_clzl_impl(unsigned long x) {
  unsigned long index;
  _BitScanReverse(&index, x);
  return (8 * sizeof(unsigned long) - 1) - (int)index;
}
inline int nccl_clzll_impl(unsigned long long x) {
  unsigned long index;
#ifdef _WIN64
  _BitScanReverse64(&index, (unsigned __int64)x);
  return 63 - (int)index;
#else
  // For 32-bit, check high 32 bits first, then low 32 bits
  if (_BitScanReverse(&index, (unsigned long)(x >> 32))) return 31 - (int)index;
  _BitScanReverse(&index, (unsigned long)(x & 0xFFFFFFFF));
  return 63 - (int)index;
#endif
}
#define COMPILER_CLZ(x) nccl_clz_impl(x)
#define COMPILER_CLZL(x) nccl_clzl_impl(x)
#define COMPILER_CLZLL(x) nccl_clzll_impl(x)

// Byte Swap
#define COMPILER_BSWAP16(x) _byteswap_ushort(x)
#define COMPILER_BSWAP32(x) _byteswap_ulong(x)
#define COMPILER_BSWAP64(x) _byteswap_uint64(x)

// Compiler hints
// TODO: Check if __declspec(align(alignment)) can be used
#define COMPILER_ASSUME_ALIGNED(ptr, alignment) (ptr)

#endif // NCCL_COMPILER_MSVC_H
