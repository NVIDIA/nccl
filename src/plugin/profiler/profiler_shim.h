/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_PROFILER_SHIM_H_
#define NCCL_PROFILER_SHIM_H_

#include "debug.h"
#include "os.h"

// v1-v7 shims share one adapter function table per ABI version, so stopEvent
// and recordEventState -- which receive only eHandle -- cannot tell which
// plugin an event belongs to. The shim holds a thread-local pointer to the
// plugin currently being dispatched; profilerActivate() in profiler.cc sets it
// before every callback, including init() and finalize(). Only native v8 needs
// no shim.
//
// Plugins must not call back into NCCL profiler APIs from startEvent,
// stopEvent, or recordEventState: that would switch the pointer mid-call.

#define NCCL_PROFILER_SHIM_BEGIN(VER) \
  static ncclProfiler_t ncclProfiler; \
  typedef ncclProfiler_v##VER##_t ncclProfilerShim_t; \
  static thread_local ncclProfilerShim_t* ncclProfiler_v##VER##Active; \
\
  static ncclProfilerShim_t* getActiveProfiler() { \
    return ncclProfiler_v##VER##Active; \
  }

#define NCCL_PROFILER_SHIM_CALL(method, ...) \
  do { \
    ncclProfilerShim_t* _p = getActiveProfiler(); \
    if (_p == nullptr) return ncclSuccess; \
    return _p->method(__VA_ARGS__); \
  } while (0)

#define NCCL_PROFILER_SHIM_END(VER) \
  void setNcclProfiler_v##VER(void* target) { \
    ncclProfiler_v##VER##Active = (ncclProfiler_v##VER##_t*)target; \
  } \
\
  ncclProfiler_t* getNcclProfiler_v##VER(void* lib, void** target) { \
    ncclProfiler_v##VER##_t* profiler = (ncclProfiler_v##VER##_t*)ncclOsDlsym(lib, "ncclProfiler_v" #VER); \
    *target = profiler; \
    if (profiler) { \
      ncclProfiler.name = profiler->name; \
      ncclProfiler.init = ncclProfiler_init; \
      ncclProfiler.startEvent = ncclProfiler_startEvent; \
      ncclProfiler.stopEvent = ncclProfiler_stopEvent; \
      ncclProfiler.recordEventState = ncclProfiler_recordEventState; \
      ncclProfiler.finalize = ncclProfiler_finalize; \
      INFO(NCCL_INIT, "PROFILER/Plugin: Loaded %s (v" #VER ")", profiler->name); \
      return &ncclProfiler; \
    } \
    return NULL; \
  }

#endif
