/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef GIN_PROXY_GPUCONTEXT_V1_H_
#define GIN_PROXY_GPUCONTEXT_V1_H_

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
} ncclGinProxyGpuCtx_v1_t;

static_assert(sizeof(ncclGinProxyGpuCtx_v1_t) == 48);

void ncclGinProxyGpuCtx_v1_init(void* ctxArray, int idx, int nRanks, uint32_t queueSize, ncclGinProxyGfd_t* queues,
                                uint32_t* pis, uint32_t* cis, uint64_t* counters, uint64_t* signals);

#endif // GIN_PROXY_GPUCONTEXT_V1_H_
