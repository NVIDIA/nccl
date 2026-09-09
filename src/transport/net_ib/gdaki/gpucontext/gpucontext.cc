/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "gpucontext.h"
#include "alloc.h"

void ncclGinGdakiGPUContext_initCurrent(void* ctxArray, int idx, struct doca_gpu_dev_verbs_qp* gdqp,
                                        struct doca_gpu_dev_verbs_qp* companion_gdqp,
                                        struct ncclGinGdakiGlobalGPUBufferTable<uint64_t> counters_table,
                                        struct ncclGinGdakiGlobalGPUBufferTable<uint64_t> signals_table,
                                        __be32 sink_buffer_lkey, uint64_t* last_issued_get,
                                        uint64_t* last_visible_get) {
  ncclGinGdakiGPUContext* ctx = (ncclGinGdakiGPUContext*)ctxArray + idx;
  ctx->gdqp = gdqp;
  ctx->companion_gdqp = companion_gdqp;
  ctx->counters_table = counters_table;
  ctx->signals_table = signals_table;
  ctx->sink_buffer_lkey = sink_buffer_lkey;
  ctx->last_issued_get = last_issued_get;
  ctx->last_visible_get = last_visible_get;
}

ncclResult_t ncclGinGdakiGPUContext_init(int version, void* ctxArray, int idx, struct doca_gpu_dev_verbs_qp* gdqp,
                                         struct doca_gpu_dev_verbs_qp* companion_gdqp,
                                         struct ncclGinGdakiGlobalGPUBufferTable<uint64_t> counters_table,
                                         struct ncclGinGdakiGlobalGPUBufferTable<uint64_t> signals_table,
                                         __be32 sink_buffer_lkey, uint64_t* last_issued_get,
                                         uint64_t* last_visible_get) {
  switch (version) {
  case 1:
    ncclGinGdakiGPUContext_v1_init(ctxArray, idx, gdqp, companion_gdqp, counters_table, signals_table,
                                   sink_buffer_lkey);
    break;
  case NCCL_GIN_GDAKI_GPU_CONTEXT_VERSION:
    ncclGinGdakiGPUContext_initCurrent(ctxArray, idx, gdqp, companion_gdqp, counters_table, signals_table,
                                       sink_buffer_lkey, last_issued_get, last_visible_get);
    break;
  default:
    WARN("Invalid GIN gdaki backend version %d", version);
    return ncclInternalError;
  }
  return ncclSuccess;
}
