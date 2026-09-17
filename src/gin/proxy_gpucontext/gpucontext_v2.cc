/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "gpucontext_v2.h"

void ncclGinProxyGpuCtx_v2_init(void* ctxArray, int idx, int nRanks, uint32_t queueSize, ncclGinProxyGfd_t* queues,
                                uint32_t* pis, uint32_t* cis, uint64_t* counters, uint64_t* signals,
                                uint64_t* signalOffsets, uint32_t* lastIssuedGet, uint32_t* lastVisibleGet) {
  ncclGinProxyGpuCtx_v2_t* ctx = (ncclGinProxyGpuCtx_v2_t*)ctxArray + idx;
  ctx->nranks = nRanks;
  ctx->queueSize = queueSize;
  ctx->queues = queues;
  ctx->pis = pis;
  ctx->cis = cis;
  ctx->counters = counters;
  ctx->signals = signals;
  ctx->signalOffsets = signalOffsets;
  ctx->lastIssuedGet = lastIssuedGet;
  ctx->lastVisibleGet = lastVisibleGet;
}
