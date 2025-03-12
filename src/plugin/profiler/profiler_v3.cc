/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "comm.h"
#include "nccl_profiler.h"

static ncclProfiler_v3_t* ncclProfiler_v3;

ncclProfiler_t* getNcclProfiler_v3(void* lib) {
  ncclProfiler_v3 = (ncclProfiler_v3_t*)dlsym(lib, "ncclProfiler_v3");
  if (ncclProfiler_v3) {
    INFO(NCCL_INIT|NCCL_ENV, "PROFILER/Plugin: loaded %s", ncclProfiler_v3->name);
    return ncclProfiler_v3;
  }
  INFO(NCCL_INIT|NCCL_ENV, "PROFILER/Plugin: failed to find ncclProfiler_v3");
  return NULL;
}
