/*************************************************************************
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "comm.h"
#include "nccl_profiler.h"
#include "plugin/profiler/profiler_v6.h"
#include "checks.h"
#include <dlfcn.h>

static ncclProfiler_v6_t* ncclProfiler_v6;

ncclProfiler_t* getNcclProfiler_v6(void* lib) {
  ncclProfiler_v6 = (ncclProfiler_v6_t*)dlsym(lib, "ncclProfiler_v6");
  if (ncclProfiler_v6) {
    INFO(NCCL_INIT, "PROFILER/Plugin: Loaded %s (v6)", ncclProfiler_v6->name);
    return ncclProfiler_v6;
  }
  return NULL;
}

