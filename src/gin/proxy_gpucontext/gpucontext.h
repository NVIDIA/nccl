/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef GIN_PROXY_GPUCONTEXT_H_
#define GIN_PROXY_GPUCONTEXT_H_

#include "nccl.h"
#include "gpucontext_v1.h"
#include "nccl_device/gin/proxy/gin_proxy_device_host_common.h"
#include <algorithm>

constexpr size_t NCCL_GIN_PROXY_GPU_CONTEXT_MAX_SIZE =
  std::max({sizeof(ncclGinProxyGpuCtx_v1_t), sizeof(ncclGinProxyGpuCtx_t)});

const int NCCL_GIN_PROXY_GPU_CONTEXT_VERSION = 2;

void ncclGinProxyGpuCtx_initCurrent(void* ctxArray, int idx, int nranks, uint32_t queueSize, ncclGinProxyGfd_t* queues,
                                    uint32_t* pis, uint32_t* cis, uint64_t* counters, uint64_t* signals,
                                    uint64_t* signalOffsets, uint32_t* lastIssuedGet, uint32_t* lastVisibleGet);

ncclResult_t ncclGinProxyGpuCtx_init(int version, void* ctxArray, int idx, int nranks, uint32_t queueSize,
                                     ncclGinProxyGfd_t* queues, uint32_t* pis, uint32_t* cis, uint64_t* counters,
                                     uint64_t* signals, uint64_t* signalOffsets, uint32_t* lastIssuedGet,
                                     uint32_t* lastVisibleGet);

#endif // GIN_PROXY_GPUCONTEXT_H_
