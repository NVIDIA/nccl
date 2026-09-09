/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NIIN_ATOMICS_H_
#define NIIN_ATOMICS_H_

#include "niin/context.h"
#include <cstring>

// ---------------------------------------------------------------------------
// Internal: helpers to dispatch CUDA atomics for the supported types.
// CUDA atomicAdd supports int, unsigned int, unsigned long long, float, double.
// For other integer types we cast through a compatible type.
// ---------------------------------------------------------------------------

// Template-based atomic dispatchers. Uses sizeof(T) to route to the correct
// CUDA atomic intrinsic, avoiding LP64 type-alias collisions (e.g. int32_t==int).

template<typename T>
__device__ __forceinline__ T niin_atomicAdd(T* addr, T val) {
  static_assert(sizeof(T) == 4 || sizeof(T) == 8, "niin_atomicAdd: unsupported type size");
  if constexpr (sizeof(T) == 4) {
    unsigned int r = atomicAdd((unsigned int*)addr, *(unsigned int*)&val);
    T result; memcpy(&result, &r, 4); return result;
  } else {
    unsigned long long r = atomicAdd((unsigned long long*)addr, *(unsigned long long*)&val);
    T result; memcpy(&result, &r, 8); return result;
  }
}

template<typename T>
__device__ __forceinline__ T niin_atomicCAS(T* addr, T cmp, T val) {
  static_assert(sizeof(T) == 4 || sizeof(T) == 8, "niin_atomicCAS: unsupported type size");
  if constexpr (sizeof(T) == 4) {
    unsigned int r = atomicCAS((unsigned int*)addr, *(unsigned int*)&cmp, *(unsigned int*)&val);
    T result; memcpy(&result, &r, 4); return result;
  } else {
    unsigned long long r = atomicCAS((unsigned long long*)addr, *(unsigned long long*)&cmp, *(unsigned long long*)&val);
    T result; memcpy(&result, &r, 8); return result;
  }
}

template<typename T>
__device__ __forceinline__ T niin_atomicExch(T* addr, T val) {
  static_assert(sizeof(T) == 4 || sizeof(T) == 8, "niin_atomicExch: unsupported type size");
  if constexpr (sizeof(T) == 4) {
    unsigned int r = atomicExch((unsigned int*)addr, *(unsigned int*)&val);
    T result; memcpy(&result, &r, 4); return result;
  } else {
    unsigned long long r = atomicExch((unsigned long long*)addr, *(unsigned long long*)&val);
    T result; memcpy(&result, &r, 8); return result;
  }
}

template<typename T>
__device__ __forceinline__ T niin_atomicAnd(T* addr, T val) {
  static_assert(sizeof(T) == 4 || sizeof(T) == 8, "niin_atomicAnd: unsupported type size");
  if constexpr (sizeof(T) == 4) {
    unsigned int r = atomicAnd((unsigned int*)addr, *(unsigned int*)&val);
    T result; memcpy(&result, &r, 4); return result;
  } else {
    unsigned long long r = atomicAnd((unsigned long long*)addr, *(unsigned long long*)&val);
    T result; memcpy(&result, &r, 8); return result;
  }
}

template<typename T>
__device__ __forceinline__ T niin_atomicOr(T* addr, T val) {
  static_assert(sizeof(T) == 4 || sizeof(T) == 8, "niin_atomicOr: unsupported type size");
  if constexpr (sizeof(T) == 4) {
    unsigned int r = atomicOr((unsigned int*)addr, *(unsigned int*)&val);
    T result; memcpy(&result, &r, 4); return result;
  } else {
    unsigned long long r = atomicOr((unsigned long long*)addr, *(unsigned long long*)&val);
    T result; memcpy(&result, &r, 8); return result;
  }
}

template<typename T>
__device__ __forceinline__ T niin_atomicXor(T* addr, T val) {
  static_assert(sizeof(T) == 4 || sizeof(T) == 8, "niin_atomicXor: unsupported type size");
  if constexpr (sizeof(T) == 4) {
    unsigned int r = atomicXor((unsigned int*)addr, *(unsigned int*)&val);
    T result; memcpy(&result, &r, 4); return result;
  } else {
    unsigned long long r = atomicXor((unsigned long long*)addr, *(unsigned long long*)&val);
    T result; memcpy(&result, &r, 8); return result;
  }
}

// ---------------------------------------------------------------------------
// nvshmem_TYPE_atomic_fetch_add: NVLink = atomicAdd via peer ptr, Network = NOT_IMPLEMENTED
// ---------------------------------------------------------------------------
#define NIIN_DEFINE_ATOMIC_FETCH_ADD(TYPENAME, TYPE)                          \
__device__ __forceinline__ TYPE nvshmem_##TYPENAME##_atomic_fetch_add(        \
    TYPE* dest, TYPE value, int pe) {                                         \
  if (pe == nvshmem_my_pe())                                                  \
    return niin_atomicAdd(dest, value);                                       \
  size_t offset = niin_sym_offset(dest);                                      \
  if (niin_is_lsa_peer(pe))                                                   \
    return niin_atomicAdd((TYPE*)niin_get_peer_ptr(offset, pe), value);       \
  NIIN_NOT_IMPLEMENTED_RETURN("nvshmem_" #TYPENAME "_atomic_fetch_add (network)", (TYPE)0); \
}

NIIN_AMO_STANDARD_TYPES(NIIN_DEFINE_ATOMIC_FETCH_ADD)
#undef NIIN_DEFINE_ATOMIC_FETCH_ADD

// ---------------------------------------------------------------------------
// nvshmem_TYPE_atomic_add (non-fetching)
// ---------------------------------------------------------------------------
#define NIIN_DEFINE_ATOMIC_ADD(TYPENAME, TYPE)                                \
__device__ __forceinline__ void nvshmem_##TYPENAME##_atomic_add(              \
    TYPE* dest, TYPE value, int pe) {                                         \
  (void)nvshmem_##TYPENAME##_atomic_fetch_add(dest, value, pe);              \
}

NIIN_AMO_STANDARD_TYPES(NIIN_DEFINE_ATOMIC_ADD)
#undef NIIN_DEFINE_ATOMIC_ADD

// ---------------------------------------------------------------------------
// nvshmem_TYPE_atomic_compare_swap
// ---------------------------------------------------------------------------
#define NIIN_DEFINE_ATOMIC_COMPARE_SWAP(TYPENAME, TYPE)                       \
__device__ __forceinline__ TYPE nvshmem_##TYPENAME##_atomic_compare_swap(     \
    TYPE* dest, TYPE cond, TYPE value, int pe) {                              \
  if (pe == nvshmem_my_pe())                                                  \
    return niin_atomicCAS(dest, cond, value);                                 \
  size_t offset = niin_sym_offset(dest);                                      \
  if (niin_is_lsa_peer(pe))                                                   \
    return niin_atomicCAS((TYPE*)niin_get_peer_ptr(offset, pe), cond, value); \
  NIIN_NOT_IMPLEMENTED_RETURN("nvshmem_" #TYPENAME "_atomic_compare_swap (network)", (TYPE)0); \
}

NIIN_AMO_STANDARD_TYPES(NIIN_DEFINE_ATOMIC_COMPARE_SWAP)
#undef NIIN_DEFINE_ATOMIC_COMPARE_SWAP

// ---------------------------------------------------------------------------
// nvshmem_TYPE_atomic_swap
// ---------------------------------------------------------------------------
#define NIIN_DEFINE_ATOMIC_SWAP(TYPENAME, TYPE)                               \
__device__ __forceinline__ TYPE nvshmem_##TYPENAME##_atomic_swap(             \
    TYPE* dest, TYPE value, int pe) {                                         \
  if (pe == nvshmem_my_pe())                                                  \
    return niin_atomicExch(dest, value);                                      \
  size_t offset = niin_sym_offset(dest);                                      \
  if (niin_is_lsa_peer(pe))                                                   \
    return niin_atomicExch((TYPE*)niin_get_peer_ptr(offset, pe), value);      \
  NIIN_NOT_IMPLEMENTED_RETURN("nvshmem_" #TYPENAME "_atomic_swap (network)", (TYPE)0); \
}

NIIN_AMO_STANDARD_TYPES(NIIN_DEFINE_ATOMIC_SWAP)
#undef NIIN_DEFINE_ATOMIC_SWAP

// ---------------------------------------------------------------------------
// nvshmem_TYPE_atomic_fetch: atomicAdd(ptr, 0) to read atomically
// ---------------------------------------------------------------------------
#define NIIN_DEFINE_ATOMIC_FETCH(TYPENAME, TYPE)                              \
__device__ __forceinline__ TYPE nvshmem_##TYPENAME##_atomic_fetch(            \
    const TYPE* dest, int pe) {                                               \
  if (pe == nvshmem_my_pe())                                                  \
    return niin_atomicAdd((TYPE*)dest, (TYPE)0);                              \
  size_t offset = niin_sym_offset(dest);                                      \
  if (niin_is_lsa_peer(pe))                                                   \
    return niin_atomicAdd((TYPE*)niin_get_peer_ptr(offset, pe), (TYPE)0);    \
  NIIN_NOT_IMPLEMENTED_RETURN("nvshmem_" #TYPENAME "_atomic_fetch (network)", (TYPE)0); \
}

NIIN_AMO_STANDARD_TYPES(NIIN_DEFINE_ATOMIC_FETCH)
#undef NIIN_DEFINE_ATOMIC_FETCH

// ---------------------------------------------------------------------------
// nvshmem_TYPE_atomic_set: atomicExch(ptr, val), discard old
// ---------------------------------------------------------------------------
#define NIIN_DEFINE_ATOMIC_SET(TYPENAME, TYPE)                                \
__device__ __forceinline__ void nvshmem_##TYPENAME##_atomic_set(              \
    TYPE* dest, TYPE value, int pe) {                                         \
  (void)nvshmem_##TYPENAME##_atomic_swap(dest, value, pe);                   \
}

NIIN_AMO_STANDARD_TYPES(NIIN_DEFINE_ATOMIC_SET)
#undef NIIN_DEFINE_ATOMIC_SET

// ---------------------------------------------------------------------------
// nvshmem_TYPE_atomic_inc / fetch_inc: add 1
// ---------------------------------------------------------------------------
#define NIIN_DEFINE_ATOMIC_INC(TYPENAME, TYPE)                                \
__device__ __forceinline__ void nvshmem_##TYPENAME##_atomic_inc(              \
    TYPE* dest, int pe) {                                                     \
  nvshmem_##TYPENAME##_atomic_add(dest, (TYPE)1, pe);                        \
}

NIIN_AMO_STANDARD_TYPES(NIIN_DEFINE_ATOMIC_INC)
#undef NIIN_DEFINE_ATOMIC_INC

#define NIIN_DEFINE_ATOMIC_FETCH_INC(TYPENAME, TYPE)                          \
__device__ __forceinline__ TYPE nvshmem_##TYPENAME##_atomic_fetch_inc(        \
    TYPE* dest, int pe) {                                                     \
  return nvshmem_##TYPENAME##_atomic_fetch_add(dest, (TYPE)1, pe);           \
}

NIIN_AMO_STANDARD_TYPES(NIIN_DEFINE_ATOMIC_FETCH_INC)
#undef NIIN_DEFINE_ATOMIC_FETCH_INC

// ---------------------------------------------------------------------------
// Bitwise atomics: and, or, xor, fetch_and, fetch_or, fetch_xor
// NVLink only -- network = NOT_IMPLEMENTED
// ---------------------------------------------------------------------------

#define NIIN_DEFINE_ATOMIC_FETCH_AND(TYPENAME, TYPE)                          \
__device__ __forceinline__ TYPE nvshmem_##TYPENAME##_atomic_fetch_and(        \
    TYPE* dest, TYPE value, int pe) {                                         \
  if (pe == nvshmem_my_pe())                                                  \
    return niin_atomicAnd(dest, value);                                       \
  size_t offset = niin_sym_offset(dest);                                      \
  if (niin_is_lsa_peer(pe))                                                   \
    return niin_atomicAnd((TYPE*)niin_get_peer_ptr(offset, pe), value);      \
  NIIN_NOT_IMPLEMENTED_RETURN("nvshmem_" #TYPENAME "_atomic_fetch_and (network)", (TYPE)0); \
}

NIIN_AMO_BITWISE_TYPES(NIIN_DEFINE_ATOMIC_FETCH_AND)
#undef NIIN_DEFINE_ATOMIC_FETCH_AND

#define NIIN_DEFINE_ATOMIC_AND(TYPENAME, TYPE)                                \
__device__ __forceinline__ void nvshmem_##TYPENAME##_atomic_and(              \
    TYPE* dest, TYPE value, int pe) {                                         \
  (void)nvshmem_##TYPENAME##_atomic_fetch_and(dest, value, pe);              \
}

NIIN_AMO_BITWISE_TYPES(NIIN_DEFINE_ATOMIC_AND)
#undef NIIN_DEFINE_ATOMIC_AND

#define NIIN_DEFINE_ATOMIC_FETCH_OR(TYPENAME, TYPE)                           \
__device__ __forceinline__ TYPE nvshmem_##TYPENAME##_atomic_fetch_or(         \
    TYPE* dest, TYPE value, int pe) {                                         \
  if (pe == nvshmem_my_pe())                                                  \
    return niin_atomicOr(dest, value);                                        \
  size_t offset = niin_sym_offset(dest);                                      \
  if (niin_is_lsa_peer(pe))                                                   \
    return niin_atomicOr((TYPE*)niin_get_peer_ptr(offset, pe), value);       \
  NIIN_NOT_IMPLEMENTED_RETURN("nvshmem_" #TYPENAME "_atomic_fetch_or (network)", (TYPE)0); \
}

NIIN_AMO_BITWISE_TYPES(NIIN_DEFINE_ATOMIC_FETCH_OR)
#undef NIIN_DEFINE_ATOMIC_FETCH_OR

#define NIIN_DEFINE_ATOMIC_OR(TYPENAME, TYPE)                                 \
__device__ __forceinline__ void nvshmem_##TYPENAME##_atomic_or(               \
    TYPE* dest, TYPE value, int pe) {                                         \
  (void)nvshmem_##TYPENAME##_atomic_fetch_or(dest, value, pe);               \
}

NIIN_AMO_BITWISE_TYPES(NIIN_DEFINE_ATOMIC_OR)
#undef NIIN_DEFINE_ATOMIC_OR

#define NIIN_DEFINE_ATOMIC_FETCH_XOR(TYPENAME, TYPE)                          \
__device__ __forceinline__ TYPE nvshmem_##TYPENAME##_atomic_fetch_xor(        \
    TYPE* dest, TYPE value, int pe) {                                         \
  if (pe == nvshmem_my_pe())                                                  \
    return niin_atomicXor(dest, value);                                       \
  size_t offset = niin_sym_offset(dest);                                      \
  if (niin_is_lsa_peer(pe))                                                   \
    return niin_atomicXor((TYPE*)niin_get_peer_ptr(offset, pe), value);      \
  NIIN_NOT_IMPLEMENTED_RETURN("nvshmem_" #TYPENAME "_atomic_fetch_xor (network)", (TYPE)0); \
}

NIIN_AMO_BITWISE_TYPES(NIIN_DEFINE_ATOMIC_FETCH_XOR)
#undef NIIN_DEFINE_ATOMIC_FETCH_XOR

#define NIIN_DEFINE_ATOMIC_XOR(TYPENAME, TYPE)                                \
__device__ __forceinline__ void nvshmem_##TYPENAME##_atomic_xor(              \
    TYPE* dest, TYPE value, int pe) {                                         \
  (void)nvshmem_##TYPENAME##_atomic_fetch_xor(dest, value, pe);              \
}

NIIN_AMO_BITWISE_TYPES(NIIN_DEFINE_ATOMIC_XOR)
#undef NIIN_DEFINE_ATOMIC_XOR

#endif // NIIN_ATOMICS_H_
