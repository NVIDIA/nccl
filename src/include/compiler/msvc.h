/*************************************************************************
 * Copyright (c) 2019-2026, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_COMPILER_MSVC_H
#define NCCL_COMPILER_MSVC_H

#include <intrin.h>
#include <emmintrin.h>
#include <atomic>

// Use standard C++ atomics via reinterpret_cast - this is safe for primitive types
// since std::atomic<T> has the same size and alignment as T

template<typename T>
static inline T COMPILER_ATOMIC_LOAD_impl(volatile T* ptr, std::memory_order order) {
  return std::atomic_load_explicit(reinterpret_cast<volatile std::atomic<T>*>(ptr), order);
}
#define COMPILER_ATOMIC_LOAD(ptr, order) COMPILER_ATOMIC_LOAD_impl(ptr, order)

template<typename T>
static inline void COMPILER_ATOMIC_LOAD_DEST_impl(volatile T* ptr, T* dest, std::memory_order order) {
  *dest = std::atomic_load_explicit(reinterpret_cast<volatile std::atomic<T>*>(ptr), order);
}
#define COMPILER_ATOMIC_LOAD_DEST(ptr, dest, order) COMPILER_ATOMIC_LOAD_DEST_impl(ptr, dest, order)

template<typename T>
static inline void COMPILER_ATOMIC_STORE_impl(volatile T* ptr, T val, std::memory_order order) {
  std::atomic_store_explicit(reinterpret_cast<volatile std::atomic<T>*>(ptr), val, order);
}
#define COMPILER_ATOMIC_STORE(ptr, val, order) COMPILER_ATOMIC_STORE_impl(ptr, val, order)

// Explicit 32-bit variants (same as generic but fixed type for uint32_t* pointers)
#define COMPILER_ATOMIC_LOAD_32(ptr, order) \
  COMPILER_ATOMIC_LOAD_impl(reinterpret_cast<volatile uint32_t*>(ptr), order)
#define COMPILER_ATOMIC_STORE_32(ptr, val, order) \
  COMPILER_ATOMIC_STORE_impl(reinterpret_cast<volatile uint32_t*>(ptr), static_cast<uint32_t>(val), order)

template<typename T>
static inline T COMPILER_ATOMIC_EXCHANGE_impl(volatile T* ptr, T val, std::memory_order order) {
  return std::atomic_exchange_explicit(reinterpret_cast<volatile std::atomic<T>*>(ptr), val, order);
}
#define COMPILER_ATOMIC_EXCHANGE(ptr, val, order) COMPILER_ATOMIC_EXCHANGE_impl(ptr, val, order)

template<typename T>
static inline bool COMPILER_ATOMIC_COMPARE_EXCHANGE_impl(volatile T* ptr, T* expected, T desired,
    std::memory_order success_order, std::memory_order failure_order) {
  return std::atomic_compare_exchange_strong_explicit(
    reinterpret_cast<volatile std::atomic<T>*>(ptr), expected, desired, success_order, failure_order);
}
#define COMPILER_ATOMIC_COMPARE_EXCHANGE(ptr, expected, desired, success_order, failure_order) \
  COMPILER_ATOMIC_COMPARE_EXCHANGE_impl(ptr, expected, desired, success_order, failure_order)

template<typename T>
static inline T COMPILER_ATOMIC_FETCH_ADD_impl(volatile T* ptr, T val, std::memory_order order) {
  return std::atomic_fetch_add_explicit(reinterpret_cast<volatile std::atomic<T>*>(ptr), val, order);
}
#define COMPILER_ATOMIC_FETCH_ADD(ptr, val, order) COMPILER_ATOMIC_FETCH_ADD_impl(ptr, val, order)

template<typename T>
static inline T COMPILER_ATOMIC_ADD_FETCH_impl(volatile T* ptr, T val, std::memory_order order) {
  return std::atomic_fetch_add_explicit(reinterpret_cast<volatile std::atomic<T>*>(ptr), val, order) + val;
}
#define COMPILER_ATOMIC_ADD_FETCH(ptr, val, order) COMPILER_ATOMIC_ADD_FETCH_impl(ptr, val, order)

template<typename T>
static inline T COMPILER_ATOMIC_SUB_FETCH_impl(volatile T* ptr, T val, std::memory_order order) {
  return std::atomic_fetch_sub_explicit(reinterpret_cast<volatile std::atomic<T>*>(ptr), val, order) - val;
}
#define COMPILER_ATOMIC_SUB_FETCH(ptr, val, order) COMPILER_ATOMIC_SUB_FETCH_impl(ptr, val, order)

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

// Unused variable/parameter attribute (MSVC doesn't have __attribute__((unused)), use empty macro)
#define COMPILER_ATTRIBUTE_UNUSED

#endif // NCCL_COMPILER_MSVC_H
