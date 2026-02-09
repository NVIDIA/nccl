/*************************************************************************
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef _NCCL_DEVICE_GIN_GDAKI_DEVICE_HOST_COMMON_H_
#define _NCCL_DEVICE_GIN_GDAKI_DEVICE_HOST_COMMON_H_

#include <linux/types.h>

#define NCCL_GIN_GDAKI_VERSION 100

template <typename T>
struct ncclGinGdakiGlobalGPUBufferTable {
  T *buffer;
  __be32 *rkeys;
  __be32 lkey;
};

struct ncclGinGdakiGPUContext {
  struct doca_gpu_dev_verbs_qp *gdqp;
  struct doca_gpu_dev_verbs_qp *companion_gdqp;
  struct ncclGinGdakiGlobalGPUBufferTable<uint64_t> counters_table;
  struct ncclGinGdakiGlobalGPUBufferTable<uint64_t> signals_table;

  // Local buffer we don't consume but is required for some operations.
  __be32 sink_buffer_lkey;
};

struct ncclGinGdakiMemHandle {
  __be32 *rkeys;
  __be32 lkey;
};

#endif /* _NCCL_DEVICE_GIN_GDAKI_DEVICE_HOST_COMMON_H_ */
