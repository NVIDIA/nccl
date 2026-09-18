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

#define NCCL_GIN_PROXY_VERSION 100
#define NCCL_GIN_PROXY_GFD_SHORT_VERSION 3
#define NCCL_GIN_PROXY_GFD_VERSION 3

typedef enum {
  ncclGinProxyOpPut = 1 << 0,
  ncclGinProxyOpBaseMask = 1 << 0,
  ncclGinProxyOpWithInline = 1 << 1,
  ncclGinProxyOpWithCounter = 1 << 2,
  ncclGinProxyOpWithSignalInc = 1 << 3,
  ncclGinProxyOpWithSignalAdd = 1 << 4,
  ncclGinProxyOpVASignal = 1 << 5, // VA signals do not include put.
  ncclGinProxyOpGet = 1 << 6,
  ncclGinProxyOpFlush = 1 << 7,
} ncclGinProxyOp_t;

static_assert(sizeof(void*) == sizeof(uint64_t) && sizeof(size_t) == sizeof(uint64_t),
              "The proxy code is built on the assumption that the pointer size is 64 bits and at "
              "most 57 bits are used for the actual pointer.");

typedef union {
  uint64_t raw;
  struct {
    uint64_t v:1;
    uint64_t resv:63;
  } __attribute__((packed)) flag;
  struct {
    uint64_t flag:1;
    uint64_t version:4;
    uint64_t resv:2;
    uint64_t size:57;
  } __attribute__((packed)) header;
  struct {
    // the last bit is the flag, so we support 63 bit VAs
    uint64_t flag:1;
    uint64_t srcOff:63;
  } __attribute__((packed)) srcOff;
  struct {
    // the last bit is the flag, so we support 63 bit VAs
    uint64_t flag:1;
    uint64_t srcHandle:63;
  } __attribute__((packed)) srcHandle;
  struct {
    // the last bit is the flag, so we support 63 bit VAs
    uint64_t flag:1;
    uint64_t vaSignalOff:63;
  } __attribute__((packed)) vaSignalOff;
  struct {
    // the last bit is the flag, so we support 63 bit VAs
    uint64_t flag:1;
    uint64_t vaSignalHandle:63;
  } __attribute__((packed)) vaSignalHandle;
  struct {
    uint8_t flag:1;
    uint8_t resv:7;
    uint32_t inlineValLow;
    uint16_t inlineValLow2;
  } __attribute__((packed)) inlineLow;
  // inline supports a max of 96 bit / 12 byte values
  struct {
    uint8_t flag:1;
    uint8_t resv:7;
    uint16_t inlineValHigh;
    uint8_t resv1;
    uint32_t resv2;
  } __attribute__((packed)) inlineHigh;
  struct {
    // the last bit is the flag, so we support 63 bit VAs
    uint64_t flag:1;
    uint64_t dstOff:63;
  } __attribute__((packed)) dstOff;
  struct {
    // the last bit is the flag, so we support 63 bit VAs
    uint64_t flag:1;
    uint64_t dstHandle:63;
  } __attribute__((packed)) dstHandle;
  struct {
    uint8_t flag:1;
    // We need to keep the size of counterId and signalId in sync with the
    // NCCL_GIN_COUNTER_POOL_SIZE / NCCL_GIN_SIGNAL_POOL_SIZE upper limits
    // in gin_host.cc.
    // must be non-zero if WITH_COUNTER is set
    uint32_t counterId:23;
    // must be non-zero if WITH_SIGNAL_INC, WITH_SIGNAL_ADD, or WITH_SIGNAL_SET is set
    uint32_t signalId:24;
    uint16_t signalValLow;
  } __attribute__((packed)) completion;
  struct {
    uint8_t flag:1;
    uint8_t isStrongSignal:1;
    uint8_t resv:6;
    uint16_t signalValLow2;
    uint32_t signalValHigh;
  } __attribute__((packed)) signalVal;
  struct {
    uint8_t flag:1;
    uint8_t resv:7;
    uint16_t op;
    uint8_t resv2;
    uint32_t resv3;
  } __attribute__((packed)) headerExt;
} ncclGinProxyQword_t;
static_assert(sizeof(ncclGinProxyQword_t) == sizeof(uint64_t), "sizeof(ncclGinProxyQword_t) != sizeof(uint64_t)");
static_assert(NCCL_GIN_PROXY_GFD_VERSION < (1 << 4), "NCCL_GIN_PROXY_GFD_VERSION must be less than 2^4");
static_assert(NCCL_GIN_PROXY_GFD_SHORT_VERSION <= NCCL_GIN_PROXY_GFD_VERSION,
              "The current GFD version must support short descriptors");

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
  ncclGinProxyGfdHeaderExt = 7,
  ncclGinProxyGfdShortQwords = 8,
  ncclGinProxyGfdQwords = 16,
} ncclGinProxyGfdQwordIdx_t;
static_assert(ncclGinProxyGfdHeaderExt + 1 == ncclGinProxyGfdShortQwords,
              "All GFD fields must fit in a short descriptor");

// Queue slots remain 128 bytes so device code using older GFD versions keeps
// the same stride. GFD versions NCCL_GIN_PROXY_GFD_SHORT_VERSION and later use
// only the first 64 bytes of each slot. aligned(16) is required because
// gin_proxy.h casts descriptors to uint4. packed preserves the no-padding
// guarantee used by the bitfield layouts.
typedef struct __attribute__((packed, aligned(16))) {
  ncclGinProxyQword_t qword[ncclGinProxyGfdShortQwords];
} ncclGinProxyGfdShort_t;
static_assert(sizeof(ncclGinProxyGfdShort_t) == 64, "sizeof(ncclGinProxyGfdShort_t) != 64");
static_assert(alignof(ncclGinProxyGfdShort_t) >= 16, "ncclGinProxyGfdShort_t must be at least 16-byte aligned");

typedef struct __attribute__((packed, aligned(16))) {
  ncclGinProxyQword_t qword[ncclGinProxyGfdQwords];
} ncclGinProxyGfd_t;
static_assert(sizeof(ncclGinProxyGfd_t) == 128, "sizeof(ncclGinProxyGfd_t) != 128");
static_assert(alignof(ncclGinProxyGfd_t) >= 16, "ncclGinProxyGfd_t must be at least 16-byte aligned");

typedef enum {
  ncclGinProxyGpuPeerQueuePi = 0,
  ncclGinProxyGpuPeerQueueCachedCi = 1,
  ncclGinProxyGpuPeerQueueIndexCount = 2,
} ncclGinProxyGpuPeerQueueIndex_t;

typedef struct {
  int nranks;
  uint32_t queueSize;
  ncclGinProxyGfd_t* queues;
  // GPU-resident producer indices followed by cached copies of cis. Keeping
  // both arrays in this allocation preserves the context layout.
  uint32_t* pis;
  // The consumer indices reside in CPU or GPU memory depending on the
  // availability of GDR.
  uint32_t* cis;

  uint64_t* counters;
  uint64_t* signals;
  uint64_t* signalOffsets;

  uint32_t* lastIssuedGet; // per-peer index of most recent get
  uint32_t* lastVisibleGet; // per-peer index of last get for which the payload is guaranteed visible (via flush GFD)

  bool flushesAllPutsOnAnySignal;
} ncclGinProxyGpuCtx_t;

#endif
