/*************************************************************************
 * Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/
#include "transport.h"
#include "proxy.h"
#include "profiler.h"

static ncclResult_t profilerProxyConnect(struct ncclProxyConnection* connection, struct ncclProxyState* proxyState, void* reqBuff, int reqSize, void* respBuff, int respSize, int* done) {
  connection->proxyAppendPtr = &connection->proxyAppend;
  connection->shared = 1;
  return ncclSuccess;
}

// The following ncclProxySubArgs are overloaded by the profiler progress function:
// - base       : is set to the current value of workCounter[channelId]
// - posted     : is set to sub->nsteps to indicate that the profiler has started the event
// - transmitted: is set to sub->nsteps to indicate that the profiler has stopped the event
static ncclResult_t profilerProxyProgress(struct ncclProxyState* proxyState, struct ncclProxyArgs* args) {
  if (args->state == ncclProxyOpReady) {
    for (int s = 0; s < args->nsubs; s++) {
      struct ncclProxySubArgs* sub = args->subs + s;
      sub->base = sub->workCounter;
      sub->posted = sub->transmitted = 0;
    }
    args->state = ncclProxyOpProgress;
  }
  if (args->state == ncclProxyOpProgress) {
    for (int s = 0; s < args->nsubs; s++) {
      struct ncclProxySubArgs* sub = args->subs + s;
      uint64_t* workStarted = (uint64_t *)sub->sendbuff;
      uint64_t* workCompleted = (uint64_t *)sub->recvbuff;
      if (sub->posted < sub->nsteps && sub->base <= workStarted[sub->channelId]) {
        ncclProfilerStartKernelChEvent(args, s);
        sub->posted = sub->nsteps;
        continue; // allow events on every channel to start
      }
      if (sub->transmitted < sub->nsteps && sub->base <= workCompleted[sub->channelId]) {
        ncclProfilerStopKernelChEvent(args, s);
        sub->transmitted = sub->nsteps;
        args->done++;
      }
    }
    if (args->done == args->nsubs) args->state = ncclProxyOpNone;
  }
  return ncclSuccess;
}

struct ncclTransport profilerTransport = {
  "Prof",
  NULL,
  { NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL },
  { NULL, NULL, NULL, NULL, NULL, profilerProxyConnect, NULL, profilerProxyProgress, NULL, NULL }
};
