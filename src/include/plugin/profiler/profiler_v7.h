/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef PROFILER_V7_H_
#define PROFILER_V7_H_

#include "profiler_v6.h"

// v7 reuses the v6 event descriptor / state args verbatim (no new host-side
// events) and only adds an optional device-side profiler hook getter.
typedef ncclProfilerEventDescr_v6_t ncclProfilerEventDescr_v7_t;
typedef ncclProfilerEventStateArgs_v6_t ncclProfilerEventStateArgs_v7_t;

typedef struct {
  const char* name;

  // init - initialize the profiler plugin
  ncclResult_t (*init)(void** context, uint64_t commId, int* eActivationMask, const char* commName, int nNodes,
                       int nranks, int rank, ncclDebugLogger_t logfn);

  // startEvent - initialize and start a new event
  ncclResult_t (*startEvent)(void* context, void** eHandle, ncclProfilerEventDescr_v7_t* eDescr);

  // stopEvent - stop/finalize an event
  ncclResult_t (*stopEvent)(void* eHandle);

  // recordEventState - record event state transitions and updates
  ncclResult_t (*recordEventState)(void* eHandle, ncclProfilerEventState_v7_t eState,
                                   ncclProfilerEventStateArgs_v7_t* eStateArgs);

  // finalize - finalize the profiler plugin
  ncclResult_t (*finalize)(void* context);

  // getDeviceHook - optional device-side profiler hook
  ncclResult_t (*getDeviceHook)(int cudaDev, void** devHook, void** devCtx);
} ncclProfiler_v7_t;

#endif // PROFILER_V7_H_
