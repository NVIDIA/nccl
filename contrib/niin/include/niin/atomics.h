/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NIIN_ATOMICS_H_
#define NIIN_ATOMICS_H_

#include "niin/context.h"
#include "niin/gpunetio/atomics.h"
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
__device__ __forceinline__ void niin_atomicAddNoReturn(T* addr, T val) {
  static_assert(sizeof(T) == 4 || sizeof(T) == 8, "niin_atomicAddNoReturn: unsupported type size");
  if constexpr (sizeof(T) == 4) {
    (void)atomicAdd((unsigned int*)addr, *(unsigned int*)&val);
  } else {
    (void)atomicAdd((unsigned long long*)addr, *(unsigned long long*)&val);
  }
}

template<typename T>
__device__ __forceinline__ T niin_atomicAddSystem(T* addr, T val) {
  static_assert(sizeof(T) == 4 || sizeof(T) == 8, "niin_atomicAddSystem: unsupported type size");
  if constexpr (sizeof(T) == 4) {
    unsigned int r = atomicAdd_system((unsigned int*)addr, *(unsigned int*)&val);
    T result; memcpy(&result, &r, 4); return result;
  } else {
    unsigned long long r = atomicAdd_system((unsigned long long*)addr, *(unsigned long long*)&val);
    T result; memcpy(&result, &r, 8); return result;
  }
}

template<typename T>
__device__ __forceinline__ void niin_atomicAddSystemNoReturn(T* addr, T val) {
  static_assert(sizeof(T) == 4 || sizeof(T) == 8, "niin_atomicAddSystemNoReturn: unsupported type size");
  if constexpr (sizeof(T) == 4) {
    (void)atomicAdd_system((unsigned int*)addr, *(unsigned int*)&val);
  } else {
    (void)atomicAdd_system((unsigned long long*)addr, *(unsigned long long*)&val);
  }
}

template<typename T>
NIIN_NOINLINE_DEVICE void niin_atomic_add_slow(T* dest, T value, int pe, size_t offset) {
  const niinGpunetioAtomicContext* atomicContext = niin_gpunetio_atomic_context();
  if (niin_gpunetio_proxy_atomic_active(atomicContext)) {
    if (niin_gpunetio_atomic_add_try(atomicContext, offset, value, pe)) return;
    NIIN_NOT_IMPLEMENTED_VOID("nvshmem_atomic_add (proxy)");
    return;
  }
  if (pe == nvshmem_my_pe()) {
    niin_atomicAddNoReturn(dest, value);
    return;
  }
  if (niin_is_lsa_peer(pe)) {
    niin_atomicAddSystemNoReturn((T*)niin_get_peer_ptr(offset, pe), value);
    return;
  }
  if (niin_gpunetio_atomic_add_try(atomicContext, offset, value, pe)) return;
  NIIN_NOT_IMPLEMENTED_VOID("nvshmem_atomic_add (network)");
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
// nvshmem_TYPE_atomic_fetch_add: self/LSA use CUDA atomics; a bound NIIN
// GPUNetIO provider handles the network case. An unbound provider retains the
// existing fail-closed network behavior.
// ---------------------------------------------------------------------------
#define NIIN_DEFINE_ATOMIC_FETCH_ADD(TYPENAME, TYPE)                          \
__device__ __forceinline__ TYPE nvshmem_##TYPENAME##_atomic_fetch_add(        \
    TYPE* dest, TYPE value, int pe) {                                         \
  const niinGpunetioAtomicContext* atomicContext = niin_gpunetio_atomic_context(); \
  size_t offset = niin_sym_offset(dest);                                      \
  if (niin_gpunetio_proxy_atomic_active(atomicContext)) {                     \
    TYPE previous;                                                              \
    if (niin_gpunetio_atomic_fetch_add_try(atomicContext, offset, value, pe, &previous)) \
      return previous;                                                          \
    NIIN_NOT_IMPLEMENTED_RETURN("nvshmem_" #TYPENAME "_atomic_fetch_add (proxy)", (TYPE)0); \
  }                                                                             \
  if (pe == nvshmem_my_pe())                                                  \
    return niin_atomicAdd(dest, value);                                       \
  if (niin_is_lsa_peer(pe))                                                   \
    return niin_atomicAddSystem((TYPE*)niin_get_peer_ptr(offset, pe), value); \
  TYPE previous;                                                              \
  if (niin_gpunetio_atomic_fetch_add_try(atomicContext,                       \
                                         offset, value, pe, &previous))       \
    return previous;                                                          \
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
  size_t offset = niin_sym_offset(dest);                                      \
  if (niin_world_is_lsa_only()) {                                             \
    if (pe == niin_rank())                                                    \
      niin_atomicAddNoReturn(dest, value);                                    \
    else                                                                      \
      niin_atomicAddSystemNoReturn((TYPE*)niin_get_peer_ptr_lsa_only(offset, pe), value); \
    return;                                                                    \
  }                                                                            \
  niin_atomic_add_slow(dest, value, pe, offset);                              \
}

NIIN_AMO_STANDARD_TYPES(NIIN_DEFINE_ATOMIC_ADD)
#undef NIIN_DEFINE_ATOMIC_ADD

// ---------------------------------------------------------------------------
// nvshmem_TYPE_atomic_compare_swap
// ---------------------------------------------------------------------------
#define NIIN_DEFINE_ATOMIC_COMPARE_SWAP(TYPENAME, TYPE)                       \
__device__ __forceinline__ TYPE nvshmem_##TYPENAME##_atomic_compare_swap(     \
    TYPE* dest, TYPE cond, TYPE value, int pe) {                              \
  const niinGpunetioAtomicContext* atomicContext = niin_gpunetio_atomic_context(); \
  size_t offset = niin_sym_offset(dest);                                      \
  if (niin_gpunetio_proxy_atomic_active(atomicContext)) {                     \
    TYPE previous;                                                              \
    if (niin_gpunetio_atomic_compare_swap_try(atomicContext, offset, cond, value, pe, &previous)) \
      return previous;                                                          \
    NIIN_NOT_IMPLEMENTED_RETURN("nvshmem_" #TYPENAME "_atomic_compare_swap (proxy)", (TYPE)0); \
  }                                                                             \
  if (pe == nvshmem_my_pe())                                                  \
    return niin_atomicCAS(dest, cond, value);                                 \
  if (niin_is_lsa_peer(pe))                                                   \
    return niin_atomicCAS((TYPE*)niin_get_peer_ptr(offset, pe), cond, value); \
  TYPE previous;                                                              \
  if (niin_gpunetio_atomic_compare_swap_try(atomicContext,                    \
                                            offset, cond, value, pe, &previous)) \
    return previous;                                                          \
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
  const niinGpunetioAtomicContext* atomicContext = niin_gpunetio_atomic_context(); \
  size_t offset = niin_sym_offset(dest);                                      \
  if (niin_gpunetio_proxy_atomic_active(atomicContext)) {                     \
    TYPE previous;                                                              \
    if (niin_gpunetio_atomic_swap_try(atomicContext, offset, value, pe, &previous)) \
      return previous;                                                          \
    NIIN_NOT_IMPLEMENTED_RETURN("nvshmem_" #TYPENAME "_atomic_swap (proxy)", (TYPE)0); \
  }                                                                             \
  if (pe == nvshmem_my_pe())                                                  \
    return niin_atomicExch(dest, value);                                      \
  if (niin_is_lsa_peer(pe))                                                   \
    return niin_atomicExch((TYPE*)niin_get_peer_ptr(offset, pe), value);      \
  TYPE previous;                                                              \
  if (niin_gpunetio_atomic_swap_try(atomicContext, offset,                    \
                                    value, pe, &previous))                     \
    return previous;                                                          \
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
  const niinGpunetioAtomicContext* atomicContext = niin_gpunetio_atomic_context(); \
  size_t offset = niin_sym_offset(dest);                                      \
  if (niin_gpunetio_proxy_atomic_active(atomicContext)) {                     \
    TYPE previous;                                                              \
    if (niin_gpunetio_atomic_fetch_try(atomicContext, offset, pe, &previous)) \
      return previous;                                                          \
    NIIN_NOT_IMPLEMENTED_RETURN("nvshmem_" #TYPENAME "_atomic_fetch (proxy)", (TYPE)0); \
  }                                                                             \
  if (pe == nvshmem_my_pe())                                                  \
    return niin_atomicAdd((TYPE*)dest, (TYPE)0);                              \
  if (niin_is_lsa_peer(pe))                                                   \
    return niin_atomicAdd((TYPE*)niin_get_peer_ptr(offset, pe), (TYPE)0);    \
  TYPE previous;                                                              \
  if (niin_gpunetio_atomic_fetch_try(atomicContext, offset,                   \
                                     pe, &previous))                           \
    return previous;                                                          \
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
  const niinGpunetioAtomicContext* atomicContext = niin_gpunetio_atomic_context(); \
  size_t offset = niin_sym_offset(dest);                                      \
  if (niin_gpunetio_proxy_atomic_active(atomicContext)) {                     \
    if (niin_gpunetio_atomic_set_try(atomicContext, offset, value, pe)) return; \
    NIIN_NOT_IMPLEMENTED_VOID("nvshmem_" #TYPENAME "_atomic_set (proxy)");  \
    return;                                                                    \
  }                                                                             \
  if (pe == nvshmem_my_pe()) {                                                \
    (void)niin_atomicExch(dest, value);                                       \
    return;                                                                    \
  }                                                                            \
  if (niin_is_lsa_peer(pe)) {                                                 \
    (void)niin_atomicExch((TYPE*)niin_get_peer_ptr(offset, pe), value);       \
    return;                                                                    \
  }                                                                            \
  if (niin_gpunetio_atomic_set_try(atomicContext, offset,                     \
                                   value, pe))                                 \
    return;                                                                    \
  NIIN_NOT_IMPLEMENTED_VOID("nvshmem_" #TYPENAME "_atomic_set (network)");  \
}

NIIN_AMO_STANDARD_TYPES(NIIN_DEFINE_ATOMIC_SET)
#undef NIIN_DEFINE_ATOMIC_SET

// ---------------------------------------------------------------------------
// nvshmem_TYPE_atomic_inc / fetch_inc: add 1
// ---------------------------------------------------------------------------
#define NIIN_DEFINE_ATOMIC_INC(TYPENAME, TYPE)                                \
__device__ __forceinline__ void nvshmem_##TYPENAME##_atomic_inc(              \
    TYPE* dest, int pe) {                                                     \
  size_t offset = niin_sym_offset(dest);                                      \
  if (niin_world_is_lsa_only()) {                                             \
    if (pe == niin_rank())                                                    \
      niin_atomicAddNoReturn(dest, (TYPE)1);                                  \
    else                                                                      \
      niin_atomicAddSystemNoReturn((TYPE*)niin_get_peer_ptr_lsa_only(offset, pe), (TYPE)1); \
    return;                                                                    \
  }                                                                            \
  niin_atomic_add_slow(dest, (TYPE)1, pe, offset);                            \
}

NIIN_AMO_STANDARD_TYPES(NIIN_DEFINE_ATOMIC_INC)
#undef NIIN_DEFINE_ATOMIC_INC

#define NIIN_DEFINE_ATOMIC_FETCH_INC(TYPENAME, TYPE)                          \
__device__ __forceinline__ TYPE nvshmem_##TYPENAME##_atomic_fetch_inc(        \
    TYPE* dest, int pe) {                                                     \
  const niinGpunetioAtomicContext* atomicContext = niin_gpunetio_atomic_context(); \
  size_t offset = niin_sym_offset(dest);                                      \
  if (niin_gpunetio_proxy_atomic_active(atomicContext)) {                     \
    TYPE previous;                                                              \
    if (niin_gpunetio_atomic_fetch_inc_try(atomicContext, offset, pe, &previous)) \
      return previous;                                                          \
    NIIN_NOT_IMPLEMENTED_RETURN("nvshmem_" #TYPENAME "_atomic_fetch_inc (proxy)", (TYPE)0); \
  }                                                                             \
  if (pe == nvshmem_my_pe())                                                  \
    return niin_atomicAdd(dest, (TYPE)1);                                     \
  if (niin_is_lsa_peer(pe))                                                   \
    return niin_atomicAddSystem((TYPE*)niin_get_peer_ptr(offset, pe), (TYPE)1); \
  TYPE previous;                                                              \
  if (niin_gpunetio_atomic_fetch_inc_try(atomicContext,                       \
                                         offset, pe, &previous))              \
    return previous;                                                          \
  NIIN_NOT_IMPLEMENTED_RETURN("nvshmem_" #TYPENAME "_atomic_fetch_inc (network)", (TYPE)0); \
}

NIIN_AMO_STANDARD_TYPES(NIIN_DEFINE_ATOMIC_FETCH_INC)
#undef NIIN_DEFINE_ATOMIC_FETCH_INC

// ---------------------------------------------------------------------------
// Bitwise atomics: and, or, xor, fetch_and, fetch_or, fetch_xor. The network
// case uses NIIN's optional direct GPUNetIO provider, which emits masked AMOs.
// ---------------------------------------------------------------------------

#define NIIN_DEFINE_ATOMIC_FETCH_AND(TYPENAME, TYPE)                          \
__device__ __forceinline__ TYPE nvshmem_##TYPENAME##_atomic_fetch_and(        \
    TYPE* dest, TYPE value, int pe) {                                         \
  const niinGpunetioAtomicContext* atomicContext = niin_gpunetio_atomic_context(); \
  size_t offset = niin_sym_offset(dest);                                      \
  if (niin_gpunetio_proxy_atomic_active(atomicContext)) {                     \
    TYPE previous;                                                              \
    if (niin_gpunetio_atomic_fetch_and_try(atomicContext, offset, value, pe, &previous)) \
      return previous;                                                          \
    NIIN_NOT_IMPLEMENTED_RETURN("nvshmem_" #TYPENAME "_atomic_fetch_and (proxy)", (TYPE)0); \
  }                                                                             \
  if (pe == nvshmem_my_pe())                                                  \
    return niin_atomicAnd(dest, value);                                       \
  if (niin_is_lsa_peer(pe))                                                   \
    return niin_atomicAnd((TYPE*)niin_get_peer_ptr(offset, pe), value);      \
  TYPE previous;                                                              \
  if (niin_gpunetio_atomic_fetch_and_try(atomicContext,                       \
                                         offset, value, pe, &previous))       \
    return previous;                                                          \
  NIIN_NOT_IMPLEMENTED_RETURN("nvshmem_" #TYPENAME "_atomic_fetch_and (network)", (TYPE)0); \
}

NIIN_AMO_BITWISE_TYPES(NIIN_DEFINE_ATOMIC_FETCH_AND)
#undef NIIN_DEFINE_ATOMIC_FETCH_AND

#define NIIN_DEFINE_ATOMIC_AND(TYPENAME, TYPE)                                \
__device__ __forceinline__ void nvshmem_##TYPENAME##_atomic_and(              \
    TYPE* dest, TYPE value, int pe) {                                         \
  const niinGpunetioAtomicContext* atomicContext = niin_gpunetio_atomic_context(); \
  size_t offset = niin_sym_offset(dest);                                      \
  if (niin_gpunetio_proxy_atomic_active(atomicContext)) {                     \
    if (niin_gpunetio_atomic_and_try(atomicContext, offset, value, pe)) return; \
    NIIN_NOT_IMPLEMENTED_VOID("nvshmem_" #TYPENAME "_atomic_and (proxy)");  \
    return;                                                                    \
  }                                                                             \
  if (pe == nvshmem_my_pe()) {                                                \
    (void)niin_atomicAnd(dest, value);                                        \
    return;                                                                    \
  }                                                                            \
  if (niin_is_lsa_peer(pe)) {                                                 \
    (void)niin_atomicAnd((TYPE*)niin_get_peer_ptr(offset, pe), value);        \
    return;                                                                    \
  }                                                                            \
  if (niin_gpunetio_atomic_and_try(atomicContext, offset,                     \
                                   value, pe))                                 \
    return;                                                                    \
  NIIN_NOT_IMPLEMENTED_VOID("nvshmem_" #TYPENAME "_atomic_and (network)");  \
}

NIIN_AMO_BITWISE_TYPES(NIIN_DEFINE_ATOMIC_AND)
#undef NIIN_DEFINE_ATOMIC_AND

#define NIIN_DEFINE_ATOMIC_FETCH_OR(TYPENAME, TYPE)                           \
__device__ __forceinline__ TYPE nvshmem_##TYPENAME##_atomic_fetch_or(         \
    TYPE* dest, TYPE value, int pe) {                                         \
  const niinGpunetioAtomicContext* atomicContext = niin_gpunetio_atomic_context(); \
  size_t offset = niin_sym_offset(dest);                                      \
  if (niin_gpunetio_proxy_atomic_active(atomicContext)) {                     \
    TYPE previous;                                                              \
    if (niin_gpunetio_atomic_fetch_or_try(atomicContext, offset, value, pe, &previous)) \
      return previous;                                                          \
    NIIN_NOT_IMPLEMENTED_RETURN("nvshmem_" #TYPENAME "_atomic_fetch_or (proxy)", (TYPE)0); \
  }                                                                             \
  if (pe == nvshmem_my_pe())                                                  \
    return niin_atomicOr(dest, value);                                        \
  if (niin_is_lsa_peer(pe))                                                   \
    return niin_atomicOr((TYPE*)niin_get_peer_ptr(offset, pe), value);       \
  TYPE previous;                                                              \
  if (niin_gpunetio_atomic_fetch_or_try(atomicContext,                        \
                                        offset, value, pe, &previous))        \
    return previous;                                                          \
  NIIN_NOT_IMPLEMENTED_RETURN("nvshmem_" #TYPENAME "_atomic_fetch_or (network)", (TYPE)0); \
}

NIIN_AMO_BITWISE_TYPES(NIIN_DEFINE_ATOMIC_FETCH_OR)
#undef NIIN_DEFINE_ATOMIC_FETCH_OR

#define NIIN_DEFINE_ATOMIC_OR(TYPENAME, TYPE)                                 \
__device__ __forceinline__ void nvshmem_##TYPENAME##_atomic_or(               \
    TYPE* dest, TYPE value, int pe) {                                         \
  const niinGpunetioAtomicContext* atomicContext = niin_gpunetio_atomic_context(); \
  size_t offset = niin_sym_offset(dest);                                      \
  if (niin_gpunetio_proxy_atomic_active(atomicContext)) {                     \
    if (niin_gpunetio_atomic_or_try(atomicContext, offset, value, pe)) return; \
    NIIN_NOT_IMPLEMENTED_VOID("nvshmem_" #TYPENAME "_atomic_or (proxy)");   \
    return;                                                                    \
  }                                                                             \
  if (pe == nvshmem_my_pe()) {                                                \
    (void)niin_atomicOr(dest, value);                                         \
    return;                                                                    \
  }                                                                            \
  if (niin_is_lsa_peer(pe)) {                                                 \
    (void)niin_atomicOr((TYPE*)niin_get_peer_ptr(offset, pe), value);         \
    return;                                                                    \
  }                                                                            \
  if (niin_gpunetio_atomic_or_try(atomicContext, offset,                      \
                                  value, pe))                                  \
    return;                                                                    \
  NIIN_NOT_IMPLEMENTED_VOID("nvshmem_" #TYPENAME "_atomic_or (network)");   \
}

NIIN_AMO_BITWISE_TYPES(NIIN_DEFINE_ATOMIC_OR)
#undef NIIN_DEFINE_ATOMIC_OR

#define NIIN_DEFINE_ATOMIC_FETCH_XOR(TYPENAME, TYPE)                          \
__device__ __forceinline__ TYPE nvshmem_##TYPENAME##_atomic_fetch_xor(        \
    TYPE* dest, TYPE value, int pe) {                                         \
  const niinGpunetioAtomicContext* atomicContext = niin_gpunetio_atomic_context(); \
  size_t offset = niin_sym_offset(dest);                                      \
  if (niin_gpunetio_proxy_atomic_active(atomicContext)) {                     \
    TYPE previous;                                                              \
    if (niin_gpunetio_atomic_fetch_xor_try(atomicContext, offset, value, pe, &previous)) \
      return previous;                                                          \
    NIIN_NOT_IMPLEMENTED_RETURN("nvshmem_" #TYPENAME "_atomic_fetch_xor (proxy)", (TYPE)0); \
  }                                                                             \
  if (pe == nvshmem_my_pe())                                                  \
    return niin_atomicXor(dest, value);                                       \
  if (niin_is_lsa_peer(pe))                                                   \
    return niin_atomicXor((TYPE*)niin_get_peer_ptr(offset, pe), value);      \
  TYPE previous;                                                              \
  if (niin_gpunetio_atomic_fetch_xor_try(atomicContext,                       \
                                         offset, value, pe, &previous))       \
    return previous;                                                          \
  NIIN_NOT_IMPLEMENTED_RETURN("nvshmem_" #TYPENAME "_atomic_fetch_xor (network)", (TYPE)0); \
}

NIIN_AMO_BITWISE_TYPES(NIIN_DEFINE_ATOMIC_FETCH_XOR)
#undef NIIN_DEFINE_ATOMIC_FETCH_XOR

#define NIIN_DEFINE_ATOMIC_XOR(TYPENAME, TYPE)                                \
__device__ __forceinline__ void nvshmem_##TYPENAME##_atomic_xor(              \
    TYPE* dest, TYPE value, int pe) {                                         \
  const niinGpunetioAtomicContext* atomicContext = niin_gpunetio_atomic_context(); \
  size_t offset = niin_sym_offset(dest);                                      \
  if (niin_gpunetio_proxy_atomic_active(atomicContext)) {                     \
    if (niin_gpunetio_atomic_xor_try(atomicContext, offset, value, pe)) return; \
    NIIN_NOT_IMPLEMENTED_VOID("nvshmem_" #TYPENAME "_atomic_xor (proxy)");  \
    return;                                                                    \
  }                                                                             \
  if (pe == nvshmem_my_pe()) {                                                \
    (void)niin_atomicXor(dest, value);                                        \
    return;                                                                    \
  }                                                                            \
  if (niin_is_lsa_peer(pe)) {                                                 \
    (void)niin_atomicXor((TYPE*)niin_get_peer_ptr(offset, pe), value);        \
    return;                                                                    \
  }                                                                            \
  if (niin_gpunetio_atomic_xor_try(atomicContext, offset,                     \
                                   value, pe))                                 \
    return;                                                                    \
  NIIN_NOT_IMPLEMENTED_VOID("nvshmem_" #TYPENAME "_atomic_xor (network)");  \
}

NIIN_AMO_BITWISE_TYPES(NIIN_DEFINE_ATOMIC_XOR)
#undef NIIN_DEFINE_ATOMIC_XOR

// ---------------------------------------------------------------------------
// Floating AMOs require the NIIN SRQ proxy for a remote target. GPUNetIO's
// native RC atomic WQEs are integral only. The float/double base operations
// are raw-bit fetch/swap/set; the six NVSHMEMX add/fetch-add spellings perform
// real floating-point addition at the target proxy. Do not route any of these
// through niin_atomicAdd<T>: its generic implementation intentionally performs
// modulo integer arithmetic on raw bits for the integral AMO family.
// ---------------------------------------------------------------------------

template <typename T>
__device__ __forceinline__ T niin_atomicFloatingAdd(T* addr, T value) {
  if constexpr (std::is_same<T, float>::value) {
    return atomicAdd(addr, value);
  } else if constexpr (std::is_same<T, double>::value) {
    return atomicAdd(addr, value);
  } else {
    static_assert(std::is_same<T, __half>::value,
                  "NIIN floating AMO add supports float, double, and __half only");
    return atomicAdd(addr, value);
  }
}

// atomicOr(x, 0) reads the original 32/64-bit representation without changing
// it. Unlike atomicAdd(x, 0), it preserves NaN payloads and signed zero.
template <typename T>
__device__ __forceinline__ T niin_atomicFloatingFetch(const T* addr) {
  static_assert(std::is_same<T, float>::value || std::is_same<T, double>::value,
                "NIIN floating raw fetch supports float/double only");
  return niin_atomicOr(const_cast<T*>(addr), T{});
}

#define NIIN_DEFINE_FLOATING_ATOMIC_SWAP(TYPENAME, TYPE)                       \
__device__ __forceinline__ TYPE nvshmem_##TYPENAME##_atomic_swap(              \
    TYPE* dest, TYPE value, int pe) {                                          \
  const niinGpunetioAtomicContext* atomicContext = niin_gpunetio_atomic_context(); \
  size_t offset = niin_sym_offset(dest);                                       \
  if (niin_gpunetio_proxy_atomic_active(atomicContext)) {                      \
    TYPE previous;                                                               \
    if (niin_gpunetio_proxy_atomic_try(atomicContext,                           \
                                       NIIN_GPUNETIO_PROXY_ATOMIC_OP_SWAP, offset, value, TYPE{}, pe, &previous)) \
      return previous;                                                           \
    NIIN_NOT_IMPLEMENTED_RETURN("nvshmem_" #TYPENAME "_atomic_swap (proxy)", (TYPE)0); \
  }                                                                              \
  if (pe == nvshmem_my_pe()) return niin_atomicExch(dest, value);              \
  if (niin_is_lsa_peer(pe))                                                     \
    return niin_atomicExch((TYPE*)niin_get_peer_ptr(offset, pe), value);       \
  NIIN_NOT_IMPLEMENTED_RETURN("nvshmem_" #TYPENAME "_atomic_swap (network)", (TYPE)0); \
}

NIIN_DEFINE_FLOATING_ATOMIC_SWAP(float, float)
NIIN_DEFINE_FLOATING_ATOMIC_SWAP(double, double)
#undef NIIN_DEFINE_FLOATING_ATOMIC_SWAP

#define NIIN_DEFINE_FLOATING_ATOMIC_FETCH(TYPENAME, TYPE)                      \
__device__ __forceinline__ TYPE nvshmem_##TYPENAME##_atomic_fetch(             \
    const TYPE* dest, int pe) {                                                \
  const niinGpunetioAtomicContext* atomicContext = niin_gpunetio_atomic_context(); \
  size_t offset = niin_sym_offset(dest);                                       \
  if (niin_gpunetio_proxy_atomic_active(atomicContext)) {                      \
    TYPE previous;                                                               \
    if (niin_gpunetio_proxy_atomic_try(atomicContext,                           \
                                       NIIN_GPUNETIO_PROXY_ATOMIC_OP_FETCH, offset, TYPE{}, TYPE{}, pe, &previous)) \
      return previous;                                                           \
    NIIN_NOT_IMPLEMENTED_RETURN("nvshmem_" #TYPENAME "_atomic_fetch (proxy)", (TYPE)0); \
  }                                                                              \
  if (pe == nvshmem_my_pe()) return niin_atomicFloatingFetch(dest);            \
  if (niin_is_lsa_peer(pe))                                                     \
    return niin_atomicFloatingFetch((TYPE*)niin_get_peer_ptr(offset, pe));     \
  NIIN_NOT_IMPLEMENTED_RETURN("nvshmem_" #TYPENAME "_atomic_fetch (network)", (TYPE)0); \
}

NIIN_DEFINE_FLOATING_ATOMIC_FETCH(float, float)
NIIN_DEFINE_FLOATING_ATOMIC_FETCH(double, double)
#undef NIIN_DEFINE_FLOATING_ATOMIC_FETCH

#define NIIN_DEFINE_FLOATING_ATOMIC_SET(TYPENAME, TYPE)                        \
__device__ __forceinline__ void nvshmem_##TYPENAME##_atomic_set(               \
    TYPE* dest, TYPE value, int pe) {                                          \
  const niinGpunetioAtomicContext* atomicContext = niin_gpunetio_atomic_context(); \
  size_t offset = niin_sym_offset(dest);                                       \
  if (niin_gpunetio_proxy_atomic_active(atomicContext)) {                      \
    if (niin_gpunetio_proxy_atomic_try(atomicContext,                           \
                                       NIIN_GPUNETIO_PROXY_ATOMIC_OP_SET, offset, value, TYPE{}, pe)) return; \
    NIIN_NOT_IMPLEMENTED_VOID("nvshmem_" #TYPENAME "_atomic_set (proxy)");   \
    return;                                                                     \
  }                                                                              \
  if (pe == nvshmem_my_pe()) { (void)niin_atomicExch(dest, value); return; }   \
  if (niin_is_lsa_peer(pe)) {                                                   \
    (void)niin_atomicExch((TYPE*)niin_get_peer_ptr(offset, pe), value);        \
    return;                                                                     \
  }                                                                              \
  NIIN_NOT_IMPLEMENTED_VOID("nvshmem_" #TYPENAME "_atomic_set (network)");   \
}

NIIN_DEFINE_FLOATING_ATOMIC_SET(float, float)
NIIN_DEFINE_FLOATING_ATOMIC_SET(double, double)
#undef NIIN_DEFINE_FLOATING_ATOMIC_SET

#define NIIN_DEFINE_EXTENDED_ATOMIC_FETCH_ADD(TYPENAME, TYPE)                  \
__device__ __forceinline__ TYPE nvshmemx_##TYPENAME##_atomic_fetch_add(        \
    TYPE* dest, TYPE value, int pe) {                                          \
  const niinGpunetioAtomicContext* atomicContext = niin_gpunetio_atomic_context(); \
  size_t offset = niin_sym_offset(dest);                                       \
  if (niin_gpunetio_proxy_atomic_active(atomicContext)) {                      \
    TYPE previous;                                                               \
    if (niin_gpunetio_proxy_atomic_try(atomicContext,                           \
                                       NIIN_GPUNETIO_PROXY_ATOMIC_OP_FETCH_ADD, offset, value, TYPE{}, pe, &previous)) \
      return previous;                                                           \
    NIIN_NOT_IMPLEMENTED_RETURN("nvshmemx_" #TYPENAME "_atomic_fetch_add (proxy)", (TYPE)0); \
  }                                                                              \
  if (pe == nvshmem_my_pe()) return niin_atomicFloatingAdd(dest, value);       \
  if (niin_is_lsa_peer(pe))                                                     \
    return niin_atomicFloatingAdd((TYPE*)niin_get_peer_ptr(offset, pe), value); \
  NIIN_NOT_IMPLEMENTED_RETURN("nvshmemx_" #TYPENAME "_atomic_fetch_add (network)", (TYPE)0); \
}

#define NIIN_DEFINE_EXTENDED_ATOMIC_ADD(TYPENAME, TYPE)                        \
__device__ __forceinline__ void nvshmemx_##TYPENAME##_atomic_add(              \
    TYPE* dest, TYPE value, int pe) {                                          \
  const niinGpunetioAtomicContext* atomicContext = niin_gpunetio_atomic_context(); \
  size_t offset = niin_sym_offset(dest);                                       \
  if (niin_gpunetio_proxy_atomic_active(atomicContext)) {                      \
    if (niin_gpunetio_proxy_atomic_try(atomicContext,                           \
                                       NIIN_GPUNETIO_PROXY_ATOMIC_OP_ADD, offset, value, TYPE{}, pe)) return; \
    NIIN_NOT_IMPLEMENTED_VOID("nvshmemx_" #TYPENAME "_atomic_add (proxy)");  \
    return;                                                                     \
  }                                                                              \
  if (pe == nvshmem_my_pe()) { (void)niin_atomicFloatingAdd(dest, value); return; } \
  if (niin_is_lsa_peer(pe)) {                                                   \
    (void)niin_atomicFloatingAdd((TYPE*)niin_get_peer_ptr(offset, pe), value); \
    return;                                                                     \
  }                                                                              \
  NIIN_NOT_IMPLEMENTED_VOID("nvshmemx_" #TYPENAME "_atomic_add (network)");  \
}

NIIN_DEFINE_EXTENDED_ATOMIC_FETCH_ADD(half, __half)
NIIN_DEFINE_EXTENDED_ATOMIC_FETCH_ADD(float, float)
NIIN_DEFINE_EXTENDED_ATOMIC_FETCH_ADD(double, double)
NIIN_DEFINE_EXTENDED_ATOMIC_ADD(half, __half)
NIIN_DEFINE_EXTENDED_ATOMIC_ADD(float, float)
NIIN_DEFINE_EXTENDED_ATOMIC_ADD(double, double)
#undef NIIN_DEFINE_EXTENDED_ATOMIC_FETCH_ADD
#undef NIIN_DEFINE_EXTENDED_ATOMIC_ADD

#endif // NIIN_ATOMICS_H_
