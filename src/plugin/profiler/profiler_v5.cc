/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "comm.h"
#include "nccl_profiler.h"
#include "checks.h"
#include <dlfcn.h>

static ncclProfiler_v5_t* ncclProfiler_v5;

ncclProfiler_t* getNcclProfiler_v5(void* lib) {
  ncclProfiler_v5 = (ncclProfiler_v5_t*)dlsym(lib, "ncclProfiler_v5");
  if (ncclProfiler_v5) {
    INFO(NCCL_INIT, "PROFILER/Plugin: Loaded %s (v5)", ncclProfiler_v5->name);
    return ncclProfiler_v5;
  }
  return NULL;
}
