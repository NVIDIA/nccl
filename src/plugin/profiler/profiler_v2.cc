/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "comm.h"
#include "nccl_profiler.h"
#include "checks.h"

static ncclProfiler_t ncclProfiler;
static ncclProfiler_v2_t* ncclProfiler_v2;

static ncclResult_t ncclProfiler_startEvent(void* context, void** eHandle, ncclProfilerEventDescr_t* eDescr) {
  if (eDescr->type == ncclProfileKernelCh || eDescr->type == ncclProfileNetPlugin) {
    *eHandle = NULL;
    return ncclSuccess;
  }
  return ncclProfiler_v2->startEvent(context, eHandle, (ncclProfilerEventDescr_v2_t *)eDescr);
}

static ncclResult_t ncclProfiler_recordEventState(void* eHandle, ncclProfilerEventState_t eState, ncclProfilerEventStateArgs_t* eStateArgs) {
  return ncclProfiler_v2->recordEventState(eHandle, eState, (ncclProfilerEventStateArgs_v2_t *)eStateArgs);
}

static ncclResult_t ncclProfiler_init(void** context, int* eActivationMask) {
  NCCLCHECK(ncclProfiler_v2->init(context, eActivationMask));
  ncclProfiler.startEvent = ncclProfiler_startEvent;
  ncclProfiler.stopEvent = ncclProfiler_v2->stopEvent;
  ncclProfiler.recordEventState = ncclProfiler_recordEventState;
  ncclProfiler.finalize = ncclProfiler_v2->finalize;
  return ncclSuccess;
}

ncclProfiler_t* getNcclProfiler_v2(void* lib) {
  ncclProfiler_v2 = (ncclProfiler_v2_t*)dlsym(lib, "ncclProfiler_v2");
  if (ncclProfiler_v2) {
    ncclProfiler.name = ncclProfiler_v2->name;
    ncclProfiler.init = ncclProfiler_init;
    INFO(NCCL_INIT|NCCL_ENV, "PROFILER/Plugin: loaded %s", ncclProfiler_v2->name);
    return &ncclProfiler;
  }
  INFO(NCCL_INIT|NCCL_ENV, "PROFILER/Plugin: failed to find ncclProfiler_v2");
  return NULL;
}
