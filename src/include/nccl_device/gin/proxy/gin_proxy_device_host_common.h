/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/
#ifndef GIN_PROXY_DEFS_H
#define GIN_PROXY_DEFS_H

#include <stdint.h>
#include <stddef.h>

// MSVC uses #pragma pack(push/pop) wrapping the union/struct declarations
// below; NCCL_GIN_PACKED_ATTR is a GCC/Clang extension that nvcc/cl.exe
// rejects.  Define it away on MSVC — the packing is equivalent.
#ifdef _MSC_VER
# define NCCL_GIN_PACKED_ATTR
#else
# define NCCL_GIN_PACKED_ATTR __attribute__((packed))
#endif

#define NCCL_GIN_PROXY_VERSION 100

typedef enum {
  ncclGinProxyOpPut = 1 << 0,
  ncclGinProxyOpBaseMask = 1 << 0,
  ncclGinProxyOpWithInline = 1 << 1,
  ncclGinProxyOpWithCounter = 1 << 2,
  ncclGinProxyOpWithSignalInc = 1 << 3,
  ncclGinProxyOpWithSignalAdd = 1 << 4,
  ncclGinProxyOpVASignal = 1 << 5, // VA signals do not include put.
} ncclGinProxyOp_t;

static_assert(sizeof(void *) == sizeof(uint64_t) && sizeof(size_t) == sizeof(uint64_t),
              "The proxy code is built on the assumption that the pointer size is 64 bits and at "
              "most 57 bits are used for the actual pointer.");

#ifdef _MSC_VER
#pragma pack(push, 1)
#endif
typedef union {
  uint64_t raw;
  struct {
    uint64_t v : 1;
    uint64_t resv : 63;
  } NCCL_GIN_PACKED_ATTR flag;
  struct {
    uint64_t flag : 1;
    uint64_t op : 6;
    uint64_t size : 57;
  } NCCL_GIN_PACKED_ATTR header;
  struct {
    // the last bit is the flag, so we support 63 bit VAs
    uint64_t flag : 1;
    uint64_t srcOff : 63;
  } NCCL_GIN_PACKED_ATTR srcOff;
  struct {
    // the last bit is the flag, so we support 63 bit VAs
    uint64_t flag : 1;
    uint64_t srcHandle : 63;
  } NCCL_GIN_PACKED_ATTR srcHandle;
  struct {
    // the last bit is the flag, so we support 63 bit VAs
    uint64_t flag : 1;
    uint64_t vaSignalOff : 63;
  } NCCL_GIN_PACKED_ATTR vaSignalOff;
  struct {
    // the last bit is the flag, so we support 63 bit VAs
    uint64_t flag : 1;
    uint64_t vaSignalHandle : 63;
  } NCCL_GIN_PACKED_ATTR vaSignalHandle;
  struct {
    uint8_t flag : 1;
    uint8_t resv : 7;
    uint32_t inlineValLow;
    uint16_t inlineValLow2;
  } NCCL_GIN_PACKED_ATTR inlineLow;
  // inline supports a max of 96 bit / 12 byte values
  struct {
    uint8_t flag : 1;
    uint8_t resv : 7;
    uint16_t inlineValHigh;
    uint8_t resv1;
    uint32_t resv2;
  } NCCL_GIN_PACKED_ATTR inlineHigh;
  struct {
    // the last bit is the flag, so we support 63 bit VAs
    uint64_t flag : 1;
    uint64_t dstOff : 63;
  } NCCL_GIN_PACKED_ATTR dstOff;
  struct {
    // the last bit is the flag, so we support 63 bit VAs
    uint64_t flag : 1;
    uint64_t dstHandle : 63;
  } NCCL_GIN_PACKED_ATTR dstHandle;
  struct {
    // Use uint64_t for all fields so this struct occupies exactly one 64-bit storage
    // unit on both GCC and MSVC.  GCC with NCCL_GIN_PACKED_ATTR packs cross-type
    // bitfields contiguously; MSVC does not — it starts a new storage unit for each
    // new underlying type, making a mixed uint8_t/uint32_t/uint16_t struct >8 bytes
    // even with #pragma pack(1).  Bit layout is identical to the original GCC packed
    // version on little-endian x86-64.
    uint64_t flag : 1;
    // We need to keep the size of counterId and signalId in sync with the
    // NCCL_GIN_COUNTER_POOL_SIZE / NCCL_GIN_SIGNAL_POOL_SIZE upper limits
    // in gin_host.cc.
    // must be non-zero if WITH_COUNTER is set
    uint64_t counterId : 23;
    // must be non-zero if WITH_SIGNAL_INC, WITH_SIGNAL_ADD, or WITH_SIGNAL_SET is set
    uint64_t signalId : 24;
    uint64_t signalValLow : 16;
  } completion;
  struct {
    uint8_t flag : 1;
    uint8_t resv : 7;
    uint16_t signalValLow2;
    uint32_t signalValHigh;
  } NCCL_GIN_PACKED_ATTR signalVal;
} ncclGinProxyQword_t;
#ifdef _MSC_VER
#pragma pack(pop)
#endif
static_assert(sizeof(ncclGinProxyQword_t) == sizeof(uint64_t),
              "sizeof(ncclGinProxyQword_t) != sizeof(uint64_t)");

typedef enum {
  ncclGinProxyGfdHeader = 0,
  ncclGinProxyGfdInlineLow = 1,
  ncclGinProxyGfdInlineHigh = 2,
  ncclGinProxyGfdSrcOff = 1, // re-uses the inline word
  ncclGinProxyGfdSrcHandle = 2, // re-uses the inline word
  ncclGinProxyGfdVASignalOff = 1, // re-uses the inline word, VA signals with PUT must be split into two GFDs
  ncclGinProxyGfdVASignalHandle = 2, // re-uses the inline word, VA signals with PUT must be split into two GFDs
  ncclGinProxyGfdDstOff = 3,
  ncclGinProxyGfdDstHandle = 4,
  ncclGinProxyGfdCompletion = 5,
  ncclGinProxyGfdSignalVal = 6,
  ncclGinProxyGfdReserved = 7,
  ncclGinProxyGfdQwords = 8,
} ncclGinProxyGfdQwordIdx_t;

#ifdef _MSC_VER
#pragma pack(push, 1)
#endif
typedef struct NCCL_GIN_PACKED_ATTR {
  ncclGinProxyQword_t qword[ncclGinProxyGfdQwords];
} ncclGinProxyGfd_t;
#ifdef _MSC_VER
#pragma pack(pop)
#endif
static_assert(sizeof(ncclGinProxyGfd_t) == 64,
              "sizeof(ncclGinProxyGfd_t) != 64 - it is crucial the GFD is 64 bytes!");

typedef struct {
  int nranks;
  uint32_t queueSize;
  ncclGinProxyGfd_t *queues;
  uint32_t *pis;
  // The consumer indices will reside in CPU or GPU memory depending on the availability of GDR
  uint32_t *cis;

  uint64_t *counters;
  uint64_t *signals;
} ncclGinProxyGpuCtx_t;

#endif
