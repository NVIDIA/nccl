/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef GIN_PROXY_GPUCONTEXT_V2_H_
#define GIN_PROXY_GPUCONTEXT_V2_H_

#include <cstdint>
#include "nccl_device/gin/proxy/gin_proxy_device_host_common.h"

typedef struct {
  int nranks;
  uint32_t queueSize;
  ncclGinProxyGfd_t* queues;
  uint32_t* pis;
  // The consumer indices will reside in CPU or GPU memory depending on the availability of GDR
  uint32_t* cis;

  uint64_t* counters;
  uint64_t* signals;
  uint64_t* signalOffsets;

  uint32_t* lastIssuedGet; // per-peer index of most recent get
  uint32_t* lastVisibleGet; // per-peer index of last get for which the payload is guaranteed visible (via flush GFD)
} ncclGinProxyGpuCtx_v2_t;

static_assert(sizeof(ncclGinProxyGpuCtx_v2_t) == 72);

void ncclGinProxyGpuCtx_v2_init(void* ctxArray, int idx, int nRanks, uint32_t queueSize, ncclGinProxyGfd_t* queues,
                                uint32_t* pis, uint32_t* cis, uint64_t* counters, uint64_t* signals,
                                uint64_t* signalOffsets, uint32_t* lastIssuedGet, uint32_t* lastVisibleGet);

#endif // GIN_PROXY_GPUCONTEXT_V2_H_
