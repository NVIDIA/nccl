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
#include "os.h"

static ncclProfiler_v6_t* ncclProfiler_v6;
static ncclProfiler_t ncclProfiler;

ncclProfiler_t* getNcclProfiler_v6(void* lib) {
  ncclProfiler_v6 = (ncclProfiler_v6_t*)ncclOsDlsym(lib, "ncclProfiler_v6");
  if (ncclProfiler_v6) {
    // v7 reuses the v6 event descriptor / state args, so the host-side callbacks
    // pass straight through; v6 plugins carry no device hook.
    ncclProfiler.name = ncclProfiler_v6->name;
    ncclProfiler.init = ncclProfiler_v6->init;
    ncclProfiler.startEvent = ncclProfiler_v6->startEvent;
    ncclProfiler.stopEvent = ncclProfiler_v6->stopEvent;
    ncclProfiler.recordEventState = ncclProfiler_v6->recordEventState;
    ncclProfiler.finalize = ncclProfiler_v6->finalize;
    ncclProfiler.getDeviceHook = nullptr;
    INFO(NCCL_INIT, "PROFILER/Plugin: Loaded %s (v6)", ncclProfiler_v6->name);
    return &ncclProfiler;
  }
  return NULL;
}
