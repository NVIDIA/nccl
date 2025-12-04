/*************************************************************************
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/
#ifndef GIN_PROXY_DEFS_H
#define GIN_PROXY_DEFS_H

#include <stdint.h>
#include <stddef.h>

/* Cross-platform packed struct attribute */
#ifdef _MSC_VER
#define NCCL_PACKED_STRUCT_BEGIN __pragma(pack(push, 1))
#define NCCL_PACKED_STRUCT_END __pragma(pack(pop))
#define NCCL_PACKED_ATTR
#else
#define NCCL_PACKED_STRUCT_BEGIN
#define NCCL_PACKED_STRUCT_END
#define NCCL_PACKED_ATTR __attribute__((packed))
#endif

#define NCCL_GIN_PROXY_VERSION 100

typedef enum
{
  ncclGinProxyOpPut = 1 << 0,
  ncclGinProxyOpBaseMask = 1 << 0,
  ncclGinProxyOpWithInline = 1 << 1,
  ncclGinProxyOpWithCounter = 1 << 2,
  ncclGinProxyOpWithSignalInc = 1 << 3,
  ncclGinProxyOpWithSignalAdd = 1 << 4,
  ncclGinProxyOpComplMask = ~ncclGinProxyOpPut,
} ncclGinProxyOp_t;

static_assert(sizeof(void *) == sizeof(uint64_t) && sizeof(size_t) == sizeof(uint64_t),
              "The proxy code is built on the assumption that the pointer size is 64 bits and at "
              "most 57 bits are used for the actual pointer.");

NCCL_PACKED_STRUCT_BEGIN
typedef union
{
  uint64_t raw;
  struct
  {
    uint64_t v : 1;
    uint64_t resv : 63;
  } NCCL_PACKED_ATTR flag;
  struct
  {
    uint64_t flag : 1;
    uint64_t op : 6;
    uint64_t size : 57;
  } NCCL_PACKED_ATTR header;
  struct
  {
    // the last bit is the flag, so we support 63 bit VAs
    uint64_t flag : 1;
    uint64_t srcOff : 63;
  } NCCL_PACKED_ATTR srcOff;
  struct
  {
    // the last bit is the flag, so we support 63 bit VAs
    uint64_t flag : 1;
    uint64_t srcHandle : 63;
  } NCCL_PACKED_ATTR srcHandle;
  struct
  {
    uint8_t flag : 1;
    uint8_t resv : 7;
    uint32_t inlineValLow;
    uint16_t inlineValLow2;
  } NCCL_PACKED_ATTR inlineLow;
  // inline supports a max of 96 bit / 12 byte values
  struct
  {
    uint8_t flag : 1;
    uint8_t resv : 7;
    uint16_t inlineValHigh;
    uint8_t resv1;
    uint32_t resv2;
  } NCCL_PACKED_ATTR inlineHigh;
  struct
  {
    // the last bit is the flag, so we support 63 bit VAs
    uint64_t flag : 1;
    uint64_t dstOff : 63;
  } NCCL_PACKED_ATTR dstOff;
  struct
  {
    // the last bit is the flag, so we support 63 bit VAs
    uint64_t flag : 1;
    uint64_t dstHandle : 63;
  } NCCL_PACKED_ATTR dstHandle;
  struct
  {
    uint8_t flag : 1;
    uint8_t resv1 : 7;
    // must be non-zero if WITH_COUNTER is set
    uint16_t counterId;
    // must be non-zero if WITH_SIGNAL_INC, WITH_SIGNAL_ADD, or WITH_SIGNAL_SET is set
    uint16_t signalId;
    uint16_t signalValLow;
    uint8_t resv2;
  } NCCL_PACKED_ATTR completion;
  struct
  {
    uint8_t flag : 1;
    uint8_t resv : 7;
    uint16_t signalValLow2;
    uint32_t signalValHigh;
  } NCCL_PACKED_ATTR signalVal;
} ncclGinProxyQword_t;
NCCL_PACKED_STRUCT_END
static_assert(sizeof(ncclGinProxyQword_t) == sizeof(uint64_t),
              "sizeof(ncclGinProxyQword_t) != sizeof(uint64_t)");

typedef enum
{
  ncclGinProxyGfdHeader = 0,
  ncclGinProxyGfdInlineLow = 1,
  ncclGinProxyGfdInlineHigh = 2,
  ncclGinProxyGfdSrcOff = 1,
  ncclGinProxyGfdSrcHandle = 2,
  ncclGinProxyGfdDstOff = 3,
  ncclGinProxyGfdDstHandle = 4,
  ncclGinProxyGfdCompletion = 5,
  ncclGinProxyGfdSignalVal = 6,
  ncclGinProxyGfdReserved = 7,
  ncclGinProxyGfdQwords = 8,
} ncclGinProxyGfdQwordIdx_t;

NCCL_PACKED_STRUCT_BEGIN
typedef struct NCCL_PACKED_ATTR
{
  ncclGinProxyQword_t qword[ncclGinProxyGfdQwords];
} ncclGinProxyGfd_t;
NCCL_PACKED_STRUCT_END
static_assert(sizeof(ncclGinProxyGfd_t) == 64,
              "sizeof(ncclGinProxyGfd_t) != 64 - it is crucial the GFD is 64 bytes!");

typedef struct
{
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
