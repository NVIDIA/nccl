/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef GIN_GDAKI_GPUCONTEXT_H_
#define GIN_GDAKI_GPUCONTEXT_H_

#include "gpucontext_v1.h"
#include "nccl.h"
#include "nccl_device/gin/gdaki/gin_gdaki_device_host_common.h"
#include <algorithm>

constexpr size_t NCCL_GIN_GDAKI_GPU_CONTEXT_MAX_SIZE =
  std::max({sizeof(ncclGinGdakiGPUContext_v1), sizeof(ncclGinGdakiGPUContext)});

const int NCCL_GIN_GDAKI_GPU_CONTEXT_VERSION = 2;

void ncclGinGdakiGPUContext_initCurrent(void* ctxArray, int idx, struct doca_gpu_dev_verbs_qp* gdqp,
                                        struct doca_gpu_dev_verbs_qp* companion_gdqp,
                                        struct ncclGinGdakiGlobalGPUBufferTable<uint64_t> counters_table,
                                        struct ncclGinGdakiGlobalGPUBufferTable<uint64_t> signals_table,
                                        __be32 sink_buffer_lkey, uint64_t* last_issued_get, uint64_t* last_visible_get);

ncclResult_t ncclGinGdakiGPUContext_init(int version, void* ctxArray, int idx, struct doca_gpu_dev_verbs_qp* gdqp,
                                         struct doca_gpu_dev_verbs_qp* companion_gdqp,
                                         struct ncclGinGdakiGlobalGPUBufferTable<uint64_t> counters_table,
                                         struct ncclGinGdakiGlobalGPUBufferTable<uint64_t> signals_table,
                                         __be32 sink_buffer_lkey, uint64_t* last_issued_get,
                                         uint64_t* last_visible_get);

#endif // GIN_GDAKI_GPUCONTEXT_H_
