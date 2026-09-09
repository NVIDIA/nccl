/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "gpucontext_v1.h"

void ncclGinProxyGpuCtx_v1_init(void* ctxArray, int idx, int nRanks, uint32_t queueSize, ncclGinProxyGfd_t* queues,
                                uint32_t* pis, uint32_t* cis, uint64_t* counters, uint64_t* signals) {
  ncclGinProxyGpuCtx_v1_t* ctx = (ncclGinProxyGpuCtx_v1_t*)ctxArray + idx;
  ctx->nranks = nRanks;
  ctx->queueSize = queueSize;
  ctx->queues = queues;
  ctx->pis = pis;
  ctx->cis = cis;
  ctx->counters = counters;
  ctx->signals = signals;
}
