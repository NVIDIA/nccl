/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_COMPILER_MSVC_H
#define NCCL_COMPILER_MSVC_H

#include <intrin.h>
#include <emmintrin.h>
#include <cstring>  // memcpy

/* Use the underscore-prefix intrinsic forms (_InterlockedXxx64) which are
 * declared in <intrin.h> and do not require winnt.h or architecture macros.
 * On MSVC x64 these map directly to lock-prefixed instructions.
 * 'long' is 32-bit on MSVC; the intrinsics take __int64. */
typedef volatile __int64* _vLONG64;

/* Helpers: reinterpret bits of a __int64 as type T without strict-aliasing UB. */
template<typename T> static inline T _nccl_bits_to(__int64 bits) {
  T v; memcpy(&v, &bits, sizeof(T)); return v;
}
template<typename T> static inline __int64 _nccl_to_bits(T v) {
  __int64 bits = 0; memcpy(&bits, &v, sizeof(T)); return bits;
}

/* Atomic load: 8-byte uses full-barrier fetch-add 0; smaller use volatile
 * read with compiler barrier (x86-64 TSO: aligned loads are naturally atomic). */
template<typename T>
static inline T COMPILER_ATOMIC_LOAD_impl(T volatile* ptr, std::memory_order) {
  if constexpr (sizeof(T) == 8) {
    return _nccl_bits_to<T>(_InterlockedExchangeAdd64((_vLONG64)ptr, 0));
  } else {
    _ReadWriteBarrier();
    T v = *ptr;
    _ReadWriteBarrier();
    return v;
  }
}
/* Cast ptr to volatile to allow template deduction from non-volatile pointers.
 * Use std::remove_reference_t to strip the lvalue-ref that decltype(*ptr) produces. */
#define COMPILER_ATOMIC_LOAD(ptr, order) \
  COMPILER_ATOMIC_LOAD_impl((std::remove_reference_t<decltype(*(ptr))> volatile*)(ptr), \
    static_cast<std::memory_order>(order))

#define COMPILER_ATOMIC_LOAD_DEST(ptr, dest, order) do { \
  *(dest) = COMPILER_ATOMIC_LOAD_impl((std::remove_reference_t<decltype(*(ptr))> volatile*)(ptr), \
    static_cast<std::memory_order>(order)); \
} while(0)

template<typename T>
static inline void COMPILER_ATOMIC_STORE_impl(T volatile* ptr, T val, std::memory_order) {
  if constexpr (sizeof(T) == 8) {
    _InterlockedExchange64((_vLONG64)ptr, _nccl_to_bits(val));
  } else {
    _ReadWriteBarrier();
    *ptr = val;
    _ReadWriteBarrier();
  }
}
#define COMPILER_ATOMIC_STORE(ptr, val, order) \
  COMPILER_ATOMIC_STORE_impl((std::remove_reference_t<decltype(*(ptr))> volatile*)(ptr), \
    (std::remove_reference_t<decltype(*(ptr))>)(val), static_cast<std::memory_order>(order))

template<typename T>
static inline T COMPILER_ATOMIC_EXCHANGE_impl(T volatile* ptr, T val, std::memory_order) {
  if constexpr (sizeof(T) == 8) {
    return _nccl_bits_to<T>(_InterlockedExchange64((_vLONG64)ptr, _nccl_to_bits(val)));
  } else if constexpr (sizeof(T) == 4) {
    long bits = (long)_nccl_to_bits(val);
    return _nccl_bits_to<T>((long)_InterlockedExchange((volatile long*)ptr, bits));
  } else {
    /* Small types (bool, char, short): non-atomic swap, but adequate for current uses */
    _ReadWriteBarrier();
    T old = *ptr;
    *ptr = val;
    _ReadWriteBarrier();
    return old;
  }
}
#define COMPILER_ATOMIC_EXCHANGE(ptr, val, order) \
  COMPILER_ATOMIC_EXCHANGE_impl((std::remove_reference_t<decltype(*(ptr))> volatile*)(ptr), \
    (std::remove_reference_t<decltype(*(ptr))>)(val), static_cast<std::memory_order>(order))

template<typename T>
static inline bool COMPILER_ATOMIC_COMPARE_EXCHANGE_impl(
  T volatile* ptr, T* expected, T desired,
  std::memory_order, std::memory_order) {
  if constexpr (sizeof(T) == 8) {
    __int64 exp_bits = _nccl_to_bits(*expected);
    __int64 old = _InterlockedCompareExchange64((_vLONG64)ptr, _nccl_to_bits(desired), exp_bits);
    bool success = (old == exp_bits);
    if (!success) *expected = _nccl_bits_to<T>(old);
    return success;
  } else if constexpr (sizeof(T) == 4) {
    long exp_bits = (long)_nccl_to_bits(*expected);
    long old = _InterlockedCompareExchange((volatile long*)ptr, (long)_nccl_to_bits(desired), exp_bits);
    bool success = (old == exp_bits);
    if (!success) *expected = _nccl_bits_to<T>(old);
    return success;
  } else {
    /* Small types: use 4-byte aligned CAS (safe for naturally-aligned small members) */
    __int64 exp_bits = _nccl_to_bits(*expected);
    __int64 old = _InterlockedCompareExchange64((_vLONG64)ptr, _nccl_to_bits(desired), exp_bits);
    bool success = (old == exp_bits);
    if (!success) *expected = _nccl_bits_to<T>(old);
    return success;
  }
}
#define COMPILER_ATOMIC_COMPARE_EXCHANGE(ptr, expected, desired, success_order, failure_order) \
  COMPILER_ATOMIC_COMPARE_EXCHANGE_impl((std::remove_reference_t<decltype(*(ptr))> volatile*)(ptr), expected, \
    (std::remove_reference_t<decltype(*(ptr))>)(desired), \
    static_cast<std::memory_order>(success_order), static_cast<std::memory_order>(failure_order))

static inline long long COMPILER_ATOMIC_ADD_FETCH_impl(volatile long long* ptr, long long val, std::memory_order) {
  /* _InterlockedExchangeAdd64 returns old; add val to get new value */
  return _InterlockedExchangeAdd64((_vLONG64)ptr, val) + val;
}
static inline long long COMPILER_ATOMIC_FETCH_ADD_impl(volatile long long* ptr, long long val, std::memory_order) {
  return _InterlockedExchangeAdd64((_vLONG64)ptr, val);
}
#define COMPILER_ATOMIC_FETCH_ADD(ptr, val, order) \
  COMPILER_ATOMIC_FETCH_ADD_impl((volatile long long*)(ptr), (long long)(val), (order))
#define COMPILER_ATOMIC_ADD_FETCH(ptr, val, order) \
  COMPILER_ATOMIC_ADD_FETCH_impl((volatile long long*)(ptr), (long long)(val), (order))

static inline long long COMPILER_ATOMIC_SUB_FETCH_impl(volatile long long* ptr, long long val, std::memory_order) {
  return _InterlockedExchangeAdd64((_vLONG64)ptr, -val) - val;
}
#define COMPILER_ATOMIC_SUB_FETCH(ptr, val, order) \
  COMPILER_ATOMIC_SUB_FETCH_impl((volatile long long*)(ptr), (long long)(val), (order))

// Prefetch
#define COMPILER_PREFETCH(addr) _mm_prefetch((const char*)(addr), _MM_HINT_T0)

// Population count
#define COMPILER_POPCOUNT32(x) __popcnt(x)
#define COMPILER_POPCOUNT64(x) __popcnt64(x)

// Branch prediction hints (MSVC doesn't support these, so no-op)
#define COMPILER_EXPECT(x, v) (x)

// Find First Set (FFS) - returns index of first set bit (1-indexed), 0 if no bits set
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
  if (_BitScanForward(&index, (unsigned long)(x & 0xFFFFFFFF))) return (int)(index + 1);
  if (_BitScanForward(&index, (unsigned long)(x >> 32))) return (int)(index + 33);
  return 0;
#endif
}
#define COMPILER_FFS(x) NCCL_FFS_impl(x)
#define COMPILER_FFSL(x) NCCL_FFSl_impl(x)
#define COMPILER_FFSLL(x) NCCL_FFSll_impl(x)

// Count Leading Zeros (CLZ) - undefined behavior if x == 0 (to match GCC behavior)
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
#define COMPILER_ASSUME_ALIGNED(ptr, alignment) (ptr)

/* wc_store_fence: Write-combining store fence (needed after stores to WC-mapped memory).
 * Defined here so any file that includes compiler.h gets the MSVC implementation. */
#include <immintrin.h>
static inline void wc_store_fence(void) { _mm_sfence(); }

/* GCC atomic builtins for MSVC host code.
 * Defined as inline functions (NOT macros) so CCCL's msvc_to_builtins.h can declare
 * its own template functions with the same names without triggering macro expansion. */
template<typename T>
static inline T __atomic_load_n(T const* ptr, int /*order*/) {
  return COMPILER_ATOMIC_LOAD_impl(const_cast<T volatile*>(ptr), std::memory_order_seq_cst);
}
template<typename T>
static inline void __atomic_store_n(T* ptr, T val, int /*order*/) {
  COMPILER_ATOMIC_STORE_impl((T volatile*)ptr, val, std::memory_order_seq_cst);
}
template<typename T>
static inline T __atomic_fetch_add(T* ptr, T val, int /*order*/) {
  return COMPILER_ATOMIC_FETCH_ADD_impl((volatile long long*)ptr, (long long)val, std::memory_order_seq_cst);
}
template<typename T>
static inline T __atomic_fetch_sub(T* ptr, T val, int /*order*/) {
  return COMPILER_ATOMIC_FETCH_ADD_impl((volatile long long*)ptr, -(long long)val, std::memory_order_seq_cst);
}

#endif // NCCL_COMPILER_MSVC_H
