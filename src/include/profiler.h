/*************************************************************************
 * Copyright (c) 2016-2023, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_PROFILER_H_
#define NCCL_PROFILER_H_
#include "nvtx.h"

// NVTX Categories
#define NVTX_CATEGORY_STATE 123
#define NVTX_CATEGORY_RECV  124
#define NVTX_CATEGORY_SEND  125

// Colors for profiling
#define ARGB_RED 0xffff0000
#define ARGB_BLUE 0xff91d2ff
#define ARGB_YELLOW 0xffffcc00
#define ARGB_YELLOW1 0xffc8cc00
#define ARGB_YELLOW2 0xfffac864
#define ARGB_YELLOW3 0xffcd7864
#define ARGB_GREEN 0xff0aff32
#define ARGB_GREEN1 0xff00c800
#define ARGB_GREEN2 0xff19a019
#define ARGB_GREEN3 0xff3c7819
#define ARGB_PURPLE 0xffff80ff
#define ARGB_PURPLE1 0xffe56edc
#define ARGB_PURPLE2 0xffc864d2

struct ncclProxyStateNvtx {
  nvtxRangeId_t rangeStateId;
  nvtxDomainHandle_t domain;

  nvtxEventAttributes_t sleep;
  nvtxEventAttributes_t append;
  nvtxEventAttributes_t active;
  nvtxEventAttributes_t idle;
  nvtxEventAttributes_t wakeup;

  nvtxEventAttributes_t recvBegin;
  nvtxEventAttributes_t recvNetWait;
  nvtxEventAttributes_t recvFlushWait;
  nvtxEventAttributes_t recvGpuWait;

  nvtxEventAttributes_t sendBegin;
  nvtxEventAttributes_t sendGpuWait;
  nvtxEventAttributes_t sendNetPost;
  nvtxEventAttributes_t sendNetWait;
};

void ncclProxyInitNvtx(struct ncclProxyState* proxyState);
void ncclProxySubArgsInitNvtx(struct ncclProxySubArgs* sub, nvtxDomainHandle_t domain, nvtxEventAttributes_t* event, uint64_t opCount);
void ncclProxySubArgsTraceNvtx(struct ncclProxySubArgs* sub, uint64_t opCount, uint64_t step, nvtxDomainHandle_t domain, nvtxEventAttributes_t* event, int size);
void ncclProxySubArgsFreeNvtx(struct ncclProxyArgs* args);
void ncclProxySubArgsStopNvtx(struct ncclProxySubArgs* sub, uint64_t step);

void ncclProxyStateTraceNvtx(struct ncclProxyState* sub, nvtxEventAttributes_t* event);

#endif
