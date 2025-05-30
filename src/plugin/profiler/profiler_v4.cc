/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "comm.h"
#include "nccl_profiler.h"
#include "checks.h"

static ncclProfiler_v4_t* ncclProfiler_v4;

ncclProfiler_t* getNcclProfiler_v4(void* lib) {
  ncclProfiler_v4 = (ncclProfiler_v4_t*)dlsym(lib, "ncclProfiler_v4");
  if (ncclProfiler_v4) {
    INFO(NCCL_INIT|NCCL_ENV, "PROFILER/Plugin: loaded %s", ncclProfiler_v4->name);
    return ncclProfiler_v4;
  }
  INFO(NCCL_INIT|NCCL_ENV, "PROFILER/Plugin: failed to find ncclProfiler_v4");
  return NULL;
}
