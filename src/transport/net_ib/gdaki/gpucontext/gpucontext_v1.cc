/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "gpucontext_v1.h"

void ncclGinGdakiGPUContext_v1_init(void* ctxArray, int idx, struct doca_gpu_dev_verbs_qp* gdqp,
                                    struct doca_gpu_dev_verbs_qp* companion_gdqp,
                                    struct ncclGinGdakiGlobalGPUBufferTable<uint64_t> counters_table,
                                    struct ncclGinGdakiGlobalGPUBufferTable<uint64_t> signals_table,
                                    __be32 sink_buffer_lkey) {
  ncclGinGdakiGPUContext_v1* ctx = (ncclGinGdakiGPUContext_v1*)ctxArray + idx;
  ctx->gdqp = gdqp;
  ctx->companion_gdqp = companion_gdqp;
  ctx->counters_table = counters_table;
  ctx->signals_table = signals_table;
  ctx->sink_buffer_lkey = sink_buffer_lkey;
}
