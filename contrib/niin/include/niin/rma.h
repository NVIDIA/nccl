/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NIIN_RMA_H_
#define NIIN_RMA_H_

#include "niin/context.h"

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

// Single-thread vectorized memcpy for LSA peer transfers.
// Uses int4 (16-byte) loads/stores when src and dst have matching alignment
// mod 16 (so a single head adjustment aligns both). Falls back to byte
// copy when alignments differ.
__device__ __forceinline__ void niin_memcpy_to_peer(
    void* __restrict__ dst, const void* __restrict__ src, size_t bytes) {
  char* d = static_cast<char*>(dst);
  const char* s = static_cast<const char*>(src);
  uintptr_t da = reinterpret_cast<uintptr_t>(d);
  uintptr_t sa = reinterpret_cast<uintptr_t>(s);

  // Vectorize only when src/dst have matching alignment mod 16
  if (((da ^ sa) & 0xF) == 0) {
    // Head: align both to 16 bytes
    size_t head = (16 - (da & 0xF)) & 0xF;
    if (head > bytes) head = bytes;
    #pragma unroll
    for (size_t i = 0; i < head; i++) d[i] = s[i];
    d += head; s += head;
    size_t remaining = bytes - head;

    // Body: int4 stores with unroll for ILP
    size_t n16 = remaining / 16;
    auto* d16 = reinterpret_cast<int4*>(d);
    auto const* s16 = reinterpret_cast<int4 const*>(s);
    #pragma unroll 4
    for (size_t i = 0; i < n16; i++) d16[i] = s16[i];

    d += n16 * 16; s += n16 * 16;
    bytes = remaining - n16 * 16;
  }

  // Tail (or full copy if alignment didn't match)
  #pragma unroll
  for (size_t i = 0; i < bytes; i++) d[i] = s[i];
}

// Single-thread vectorized memcpy from an LSA peer.
__device__ __forceinline__ void niin_memcpy_from_peer(
    void* __restrict__ dst, const void* __restrict__ src, size_t bytes) {
  char* d = static_cast<char*>(dst);
  const char* s = static_cast<const char*>(src);
  uintptr_t da = reinterpret_cast<uintptr_t>(d);
  uintptr_t sa = reinterpret_cast<uintptr_t>(s);

  if (((da ^ sa) & 0xF) == 0) {
    size_t head = (16 - (sa & 0xF)) & 0xF;
    if (head > bytes) head = bytes;
    #pragma unroll
    for (size_t i = 0; i < head; i++) d[i] = s[i];
    d += head; s += head;
    size_t remaining = bytes - head;

    size_t n16 = remaining / 16;
    auto* d16 = reinterpret_cast<int4*>(d);
    auto const* s16 = reinterpret_cast<int4 const*>(s);
    #pragma unroll 4
    for (size_t i = 0; i < n16; i++) d16[i] = s16[i];

    d += n16 * 16; s += n16 * 16;
    bytes = remaining - n16 * 16;
  }

  #pragma unroll
  for (size_t i = 0; i < bytes; i++) d[i] = s[i];
}

// Helper: GIN put (network path for arbitrary data)
__device__ __forceinline__ void niin_gin_put(size_t dstOffset, const void* src, size_t bytes, int pe) {
  if (!niin_has_gin()) {
    NIIN_NOT_IMPLEMENTED_VOID("nvshmem_put (network without GIN)");
    return;
  }
  ncclDevComm const& comm = niin_comm();
  ncclTeam world = ncclTeamWorld(comm);
  ncclGin gin(comm, niin_gin_context_index());
  // For GIN put, both src and dst must be in registered windows.
  // src is in our local heap, dst is at dstOffset in peer's heap.
  gin.put(
    world, pe,
    niin_heap_window(), dstOffset,
    niin_heap_window(), niin_sym_offset(src), bytes
  );
}

// Helper: GIN get (network path for block data into the symmetric heap)
__device__ __forceinline__ void niin_gin_get(void* dest, size_t srcOffset, size_t bytes, int pe) {
  if (!niin_has_gin()) {
    NIIN_NOT_IMPLEMENTED_VOID("nvshmem_get (network without GIN)");
    return;
  }
  ncclDevComm const& comm = niin_comm();
  ncclTeam world = ncclTeamWorld(comm);
  ncclGin gin(comm, niin_gin_context_index());
  gin.get(
    world, pe,
    niin_heap_window(), srcOffset,
    niin_heap_window(), niin_sym_offset(dest), bytes
  );
  gin.flush(ncclCoopThread{});
}

// Helper: GIN putValue (network path for small values <= 8 bytes)
template<typename T>
__device__ __forceinline__ void niin_gin_put_value(size_t dstOffset, T value, int pe) {
  if (!niin_has_gin()) {
    NIIN_NOT_IMPLEMENTED_VOID("nvshmem_p (network without GIN)");
    return;
  }
  ncclDevComm const& comm = niin_comm();
  ncclTeam world = ncclTeamWorld(comm);
  ncclGin gin(comm, niin_gin_context_index());
  gin.putValue<T>(
    world, pe,
    niin_heap_window(), dstOffset, value
  );
}

// ---------------------------------------------------------------------------
// nvshmem_TYPE_p: scalar put
// ---------------------------------------------------------------------------
#define NIIN_DEFINE_P(TYPENAME, TYPE)                                          \
__device__ __forceinline__ void nvshmem_##TYPENAME##_p(TYPE* dest, TYPE value, int pe) { \
  size_t offset = niin_sym_offset(dest);                                      \
  if (pe == nvshmem_my_pe()) {                                                \
    *(volatile TYPE*)dest = value;                                             \
    return;                                                                    \
  }                                                                            \
  if (niin_is_lsa_peer(pe)) {                                                 \
    *(volatile TYPE*)niin_get_peer_ptr(offset, pe) = value;                   \
    return;                                                                    \
  }                                                                            \
  if (sizeof(TYPE) <= 8) {                                                     \
    niin_gin_put_value(offset, value, pe);                                    \
  } else {                                                                     \
    niin_gin_put(offset, &value, sizeof(TYPE), pe);                           \
  }                                                                            \
}

NIIN_STANDARD_RMA_TYPES(NIIN_DEFINE_P)
#undef NIIN_DEFINE_P

// ---------------------------------------------------------------------------
// nvshmem_TYPE_g: scalar get
// ---------------------------------------------------------------------------
#define NIIN_DEFINE_G(TYPENAME, TYPE)                                          \
__device__ __forceinline__ TYPE nvshmem_##TYPENAME##_g(const TYPE* src, int pe) { \
  if (pe == nvshmem_my_pe()) {                                                \
    return *(volatile const TYPE*)src;                                         \
  }                                                                            \
  size_t offset = niin_sym_offset(src);                                       \
  if (niin_is_lsa_peer(pe)) {                                                 \
    return *(volatile const TYPE*)niin_get_peer_ptr(offset, pe);              \
  }                                                                            \
  NIIN_NOT_IMPLEMENTED_RETURN("nvshmem_" #TYPENAME "_g (network)", (TYPE)0);  \
}

NIIN_STANDARD_RMA_TYPES(NIIN_DEFINE_G)
#undef NIIN_DEFINE_G

// ---------------------------------------------------------------------------
// nvshmem_TYPE_put / nvshmem_TYPE_get: block transfers
// ---------------------------------------------------------------------------
#define NIIN_DEFINE_PUT(TYPENAME, TYPE)                                        \
__device__ __forceinline__ void nvshmem_##TYPENAME##_put(TYPE* dest, const TYPE* src, \
                                                          size_t nelems, int pe) { \
  size_t bytes = nelems * sizeof(TYPE);                                        \
  size_t offset = niin_sym_offset(dest);                                      \
  if (pe == nvshmem_my_pe()) {                                                \
    niin_memcpy_to_peer(dest, src, bytes);                                    \
    return;                                                                    \
  }                                                                            \
  if (niin_is_lsa_peer(pe)) {                                                 \
    niin_memcpy_to_peer(niin_get_peer_ptr(offset, pe), src, bytes);           \
    return;                                                                    \
  }                                                                            \
  niin_gin_put(offset, src, bytes, pe);                                       \
}

NIIN_STANDARD_RMA_TYPES(NIIN_DEFINE_PUT)
#undef NIIN_DEFINE_PUT

#define NIIN_DEFINE_GET(TYPENAME, TYPE)                                        \
__device__ __forceinline__ void nvshmem_##TYPENAME##_get(TYPE* dest, const TYPE* src, \
                                                          size_t nelems, int pe) { \
  size_t bytes = nelems * sizeof(TYPE);                                        \
  if (pe == nvshmem_my_pe()) {                                                \
    niin_memcpy_from_peer(dest, src, bytes);                                  \
    return;                                                                    \
  }                                                                            \
  size_t offset = niin_sym_offset(src);                                       \
  if (niin_is_lsa_peer(pe)) {                                                 \
    niin_memcpy_from_peer(dest, niin_get_peer_ptr(offset, pe), bytes);        \
    return;                                                                    \
  }                                                                            \
  niin_gin_get(dest, offset, bytes, pe);                                       \
}

NIIN_STANDARD_RMA_TYPES(NIIN_DEFINE_GET)
#undef NIIN_DEFINE_GET

// ---------------------------------------------------------------------------
// nvshmem_putmem / nvshmem_getmem: untyped byte transfers
// ---------------------------------------------------------------------------
__device__ __forceinline__ void nvshmem_putmem(void* dest, const void* src,
                                                size_t bytes, int pe) {
  size_t offset = niin_sym_offset(dest);
  if (pe == nvshmem_my_pe()) {
    niin_memcpy_to_peer(dest, src, bytes);
    return;
  }
  if (niin_is_lsa_peer(pe)) {
    niin_memcpy_to_peer(niin_get_peer_ptr(offset, pe), src, bytes);
    return;
  }
  niin_gin_put(offset, src, bytes, pe);
}

__device__ __forceinline__ void nvshmem_getmem(void* dest, const void* src,
                                                size_t bytes, int pe) {
  if (pe == nvshmem_my_pe()) {
    niin_memcpy_from_peer(dest, src, bytes);
    return;
  }
  size_t offset = niin_sym_offset(src);
  if (niin_is_lsa_peer(pe)) {
    niin_memcpy_from_peer(dest, niin_get_peer_ptr(offset, pe), bytes);
    return;
  }
  niin_gin_get(dest, offset, bytes, pe);
}

// ---------------------------------------------------------------------------
// nvshmem_putSIZE / nvshmem_getSIZE: sized transfers (8, 16, 32, 64, 128)
// ---------------------------------------------------------------------------
#define NIIN_DEFINE_PUT_SIZED(SIZE, NBYTES)                                   \
__device__ __forceinline__ void nvshmem_put##SIZE(void* dest, const void* src, \
                                                   size_t nelems, int pe) {   \
  nvshmem_putmem(dest, src, nelems * NBYTES, pe);                             \
}

NIIN_SIZED_RMA(NIIN_DEFINE_PUT_SIZED)
#undef NIIN_DEFINE_PUT_SIZED

#define NIIN_DEFINE_GET_SIZED(SIZE, NBYTES)                                   \
__device__ __forceinline__ void nvshmem_get##SIZE(void* dest, const void* src, \
                                                   size_t nelems, int pe) {   \
  nvshmem_getmem(dest, src, nelems * NBYTES, pe);                             \
}

NIIN_SIZED_RMA(NIIN_DEFINE_GET_SIZED)
#undef NIIN_DEFINE_GET_SIZED

// ---------------------------------------------------------------------------
// nvshmem_TYPE_put_nbi / nvshmem_TYPE_get_nbi: non-blocking variants
// On LSA peers the operations are already non-blocking (direct stores).
// On GIN, NIIN maps nbi variants to the blocking implementation.
// ---------------------------------------------------------------------------
#define NIIN_DEFINE_PUT_NBI(TYPENAME, TYPE)                                   \
__device__ __forceinline__ void nvshmem_##TYPENAME##_put_nbi(TYPE* dest, const TYPE* src, \
                                                              size_t nelems, int pe) { \
  nvshmem_##TYPENAME##_put(dest, src, nelems, pe);                            \
}

NIIN_STANDARD_RMA_TYPES(NIIN_DEFINE_PUT_NBI)
#undef NIIN_DEFINE_PUT_NBI

#define NIIN_DEFINE_GET_NBI(TYPENAME, TYPE)                                   \
__device__ __forceinline__ void nvshmem_##TYPENAME##_get_nbi(TYPE* dest, const TYPE* src, \
                                                              size_t nelems, int pe) { \
  nvshmem_##TYPENAME##_get(dest, src, nelems, pe);                            \
}

NIIN_STANDARD_RMA_TYPES(NIIN_DEFINE_GET_NBI)
#undef NIIN_DEFINE_GET_NBI

// Untyped nbi variants
__device__ __forceinline__ void nvshmem_putmem_nbi(void* dest, const void* src,
                                                    size_t bytes, int pe) {
  nvshmem_putmem(dest, src, bytes, pe);
}

__device__ __forceinline__ void nvshmem_getmem_nbi(void* dest, const void* src,
                                                    size_t bytes, int pe) {
  nvshmem_getmem(dest, src, bytes, pe);
}

// Sized nbi variants
#define NIIN_DEFINE_PUT_SIZED_NBI(SIZE, NBYTES)                               \
__device__ __forceinline__ void nvshmem_put##SIZE##_nbi(void* dest, const void* src, \
                                                         size_t nelems, int pe) { \
  nvshmem_put##SIZE(dest, src, nelems, pe);                                   \
}

NIIN_SIZED_RMA(NIIN_DEFINE_PUT_SIZED_NBI)
#undef NIIN_DEFINE_PUT_SIZED_NBI

#define NIIN_DEFINE_GET_SIZED_NBI(SIZE, NBYTES)                               \
__device__ __forceinline__ void nvshmem_get##SIZE##_nbi(void* dest, const void* src, \
                                                         size_t nelems, int pe) { \
  nvshmem_get##SIZE(dest, src, nelems, pe);                                   \
}

NIIN_SIZED_RMA(NIIN_DEFINE_GET_SIZED_NBI)
#undef NIIN_DEFINE_GET_SIZED_NBI

// ---------------------------------------------------------------------------
// nvshmem_TYPE_iput / nvshmem_TYPE_iget: strided transfers
// ---------------------------------------------------------------------------
#define NIIN_DEFINE_IPUT(TYPENAME, TYPE)                                      \
__device__ __forceinline__ void nvshmem_##TYPENAME##_iput(TYPE* dest, const TYPE* src, \
                                                           ptrdiff_t dst, ptrdiff_t sst, \
                                                           size_t nelems, int pe) { \
  if (pe == nvshmem_my_pe()) {                                                \
    for (size_t i = 0; i < nelems; i++)                                       \
      ((volatile TYPE*)dest)[i * dst] = src[i * sst];                         \
    return;                                                                    \
  }                                                                            \
  size_t offset = niin_sym_offset(dest);                                      \
  if (niin_is_lsa_peer(pe)) {                                                 \
    volatile TYPE* d = (volatile TYPE*)niin_get_peer_ptr(offset, pe);         \
    for (size_t i = 0; i < nelems; i++)                                       \
      d[i * dst] = src[i * sst];                                             \
    return;                                                                    \
  }                                                                            \
  NIIN_NOT_IMPLEMENTED_VOID("nvshmem_" #TYPENAME "_iput (network)");          \
}

NIIN_STANDARD_RMA_TYPES(NIIN_DEFINE_IPUT)
#undef NIIN_DEFINE_IPUT

#define NIIN_DEFINE_IGET(TYPENAME, TYPE)                                      \
__device__ __forceinline__ void nvshmem_##TYPENAME##_iget(TYPE* dest, const TYPE* src, \
                                                           ptrdiff_t dst, ptrdiff_t sst, \
                                                           size_t nelems, int pe) { \
  if (pe == nvshmem_my_pe()) {                                                \
    for (size_t i = 0; i < nelems; i++)                                       \
      dest[i * dst] = ((volatile const TYPE*)src)[i * sst];                   \
    return;                                                                    \
  }                                                                            \
  size_t offset = niin_sym_offset(src);                                       \
  if (niin_is_lsa_peer(pe)) {                                                 \
    volatile const TYPE* s = (volatile const TYPE*)niin_get_peer_ptr(offset, pe); \
    for (size_t i = 0; i < nelems; i++)                                       \
      dest[i * dst] = s[i * sst];                                            \
    return;                                                                    \
  }                                                                            \
  NIIN_NOT_IMPLEMENTED_VOID("nvshmem_" #TYPENAME "_iget (network)");          \
}

NIIN_STANDARD_RMA_TYPES(NIIN_DEFINE_IGET)
#undef NIIN_DEFINE_IGET

#endif // NIIN_RMA_H_
