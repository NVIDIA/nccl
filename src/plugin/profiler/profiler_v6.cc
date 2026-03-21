/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "comm.h"
#include "nccl_profiler.h"
#include "plugin/profiler/profiler_v6.h"
#include "checks.h"
#include <dlfcn.h>

static ncclProfiler_v6_t* ncclProfiler_v6;
static ncclProfiler_t ncclProfiler;

static ncclResult_t ncclProfiler_startEvent(void* ctx, void** eHandle, ncclProfilerEventDescr_t* eDescr) {
  // Discard v7 UserTag events - not supported in v6
  if (eDescr->type == ncclProfileUserTag) {
    *eHandle = NULL;
    return ncclSuccess;
  }
  // v6 and v7 descriptors are layout-compatible for all non-UserTag events
  return ncclProfiler_v6->startEvent(ctx, eHandle, (ncclProfilerEventDescr_v6_t*)eDescr);
}

static ncclResult_t ncclProfiler_init(void** ctx, uint64_t commId, int* eActivationMask, const char* commName, int nNodes, int nRanks, int rank, ncclDebugLogger_t logfn) {
  NCCLCHECK(ncclProfiler_v6->init(ctx, commId, eActivationMask, commName, nNodes, nRanks, rank, logfn));

  // Clear v7 UserTag bit from activation mask since v6 doesn't support it
  if (eActivationMask) {
    *eActivationMask &= ~ncclProfileUserTag;
  }

  ncclProfiler.startEvent = ncclProfiler_startEvent;
  ncclProfiler.recordEventState = ncclProfiler_v6->recordEventState;
  ncclProfiler.stopEvent = ncclProfiler_v6->stopEvent;
  ncclProfiler.finalize = ncclProfiler_v6->finalize;
  return ncclSuccess;
}

ncclProfiler_t* getNcclProfiler_v6(void* lib) {
  ncclProfiler_v6 = (ncclProfiler_v6_t*)dlsym(lib, "ncclProfiler_v6");
  if (ncclProfiler_v6) {
    ncclProfiler.name = ncclProfiler_v6->name;
    ncclProfiler.init = ncclProfiler_init;
    INFO(NCCL_INIT, "PROFILER/Plugin: Loaded %s (v6)", ncclProfiler_v6->name);
    return &ncclProfiler;
  }
  return NULL;
}
