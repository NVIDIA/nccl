/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/
#include "transport.h"
#include "proxy.h"
#include "profiler.h"
#include "device.h"
#include "param.h"

NCCL_PARAM(ProfilerKernelChPollSleepUsec, "PROFILER_KERNEL_CH_POLL_SLEEP_USEC", 1);

static ncclResult_t profilerProxyConnect(struct ncclProxyConnection* connection, struct ncclProxyState* proxyState,
                                         void* reqBuff, int reqSize, void* respBuff, int respSize, int* done) {
  connection->proxyAppendPtr = &connection->proxyAppend;
  connection->shared = 0;
  return ncclSuccess;
}

// The following ncclProxySubArgs are overloaded by the profiler progress function:
// - base       : is set to the current value of workCounter[channelId]
// - posted     : is set to 1 to indicate that the profiler has started the event
// - transmitted: is set to 1 to indicate that the profiler has stopped the event
static ncclResult_t profilerProxyProgress(struct ncclProxyState* proxyState, struct ncclProxyArgs* args) {
  if (args->state == ncclProxyOpReady) {
    for (int s = 0; s < args->nsubs; s++) {
      struct ncclProxySubArgs* sub = args->subs + s;
      sub->base = sub->workCounter;
      sub->posted = sub->transmitted = 0;
    }
    args->state = ncclProxyOpProgress;
  }
  args->idle = 1;
  args->pollDelayUsec = 0;
  if (args->state == ncclProxyOpProgress) {
    for (int s = 0; s < args->nsubs; s++) {
      struct ncclProxySubArgs* sub = args->subs + s;
      struct ncclDevProfiler* workStarted = (struct ncclDevProfiler*)sub->sendbuff;
      struct ncclDevProfiler* workCompleted = (struct ncclDevProfiler*)sub->recvbuff;
      if (sub->posted == 0 &&
          sub->base <= workStarted[sub->channelId].data[sub->base % MAX_PROFILER_EVENTS_PER_CHANNEL].counter) {
        ncclProfilerStartKernelChEvent(
          args, s, workStarted[sub->channelId].data[sub->base % MAX_PROFILER_EVENTS_PER_CHANNEL].timestamp);
        sub->posted = 1;
        args->idle = 0;
        continue; // allow events on every channel to start
      }
      if (sub->transmitted == 0 &&
          sub->base <= workCompleted[sub->channelId].data[sub->base % MAX_PROFILER_EVENTS_PER_CHANNEL].counter) {
        ncclProfilerStopKernelChEvent(
          args, s, workCompleted[sub->channelId].data[sub->base % MAX_PROFILER_EVENTS_PER_CHANNEL].timestamp);
        sub->transmitted = 1;
        args->idle = 0;
        args->done++;
      }
    }
    if (args->done == args->nsubs) args->state = ncclProxyOpNone;
    if (args->idle && args->state == ncclProxyOpProgress) {
      args->pollDelayUsec = ncclParamProfilerKernelChPollSleepUsec();
    }
  }
  return ncclSuccess;
}

struct ncclTransport profilerTransport = {"Prof",
                                          NULL,
                                          {NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL},
                                          {NULL, NULL, NULL, NULL, NULL, profilerProxyConnect, NULL,
                                           profilerProxyProgress, NULL, NULL}};
