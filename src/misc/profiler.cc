/*************************************************************************
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

/* A thin layer connecting internal profiling calls (ncclProfiling*) to external profiling APIs */

#include "profiler.h"
#include "checks.h"
#include <errno.h>
#include <dlfcn.h>

// External profiler
ncclProfiler_t* ncclProfiler;

// Initialize external profiler by dynamically loading plugin symbol
ncclResult_t ncclProfilerInit() {
  ncclProfiler = nullptr;

  char ncclProfilerPluginName[128] = "libnccl-profiler.so";
  void* profilerPluginLib = dlopen(ncclProfilerPluginName, RTLD_NOW | RTLD_LOCAL);
  if (profilerPluginLib == nullptr) {
    // dlopen does not guarantee to set errno, but dlerror only gives us a
    // string, so checking errno doesn't hurt to try to provide a better
    // error message
    if (errno == ENOENT) {
      INFO(NCCL_INIT, "Profiler : No plugin found (%s), profiling is disabled", ncclProfilerPluginName);
    } else {
      INFO(NCCL_INIT, "Profiler : Plugin load returned %d : %s.", errno, dlerror());
    }
    return ncclSuccess;
  }

  ncclProfiler = (ncclProfiler_t*)dlsym(profilerPluginLib, "NCCL_PROFILER_SYMBOL");
  if (ncclProfiler == nullptr) {
    INFO(NCCL_INIT, "Profiler: Failed to find NCCL_PROFILER_SYMBOL.");
    return ncclSuccess;
  }

  INFO(NCCL_INIT, "Using profiler : %s", ncclProfiler->name);
  return ncclSuccess;
}

/* Translation to external plugin APIs */

ncclResult_t ncclProfilingRecord(struct ncclProxyArgs* args, int sub, int step, int state) {
  // No profiler present, return
  if (!ncclProfiler)
    return ncclSuccess;

  // Formatting proxy operation information
  ncclProxyProfileInfo_t info;
  info.opCount = args->opCount;
  info.channel = args->subs[sub].channelId;
  info.peer = args->subs[sub].peer;
  info.type = args->pattern == ncclPatternSend ? ncclProxySend : ncclProxyRecv;
  info.step = step;
  info.opIndex = (((uint64_t)args)/sizeof(struct ncclProxyArgs))%256;
  void** eventPtr = args->subs[sub].profilingEvents + step%NCCL_STEPS;

  return ncclProfiler->profilingRecord(&info, state, eventPtr);
}

ncclResult_t ncclProfilingDump() {
  // No profiler present, return
  if (!ncclProfiler)
    return ncclSuccess;

  return ncclProfiler->profilingDump();
}
