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
#include "comm.h"

static ncclResult_t profilerProxyConnect(struct ncclProxyConnection* connection, struct ncclProxyState* proxyState,
                                         void* reqBuff, int reqSize, void* respBuff, int respSize, int* done) {
  connection->proxyAppendPtr = &connection->proxyAppend;
  connection->shared = 0;
  return ncclSuccess;
}

// Drain completed KernelStep start/end pairs for one channel.
static void profilerDrainKernelSteps(struct ncclProxyArgs* args, int s, struct ncclComm* comm) {
  struct ncclProxySubArgs* sub = args->subs + s;
  if (!(sub->eActivationMask & ncclProfileKernelStep) || sub->stepStarted == nullptr ||
      sub->stepCompleted == nullptr || comm == nullptr || comm->profiler.kernelStepHandles == nullptr) {
    return;
  }

  int ch = sub->channelId;
  uint64_t drained = comm->profiler.stepCounter[ch];
  uint64_t produced = comm->profiler.stepSeq ? comm->profiler.stepSeq[ch] : 0;
  uint64_t limit = produced > drained ? produced : drained + 1;

  while (drained < limit) {
    uint64_t next = drained + 1;
    int slot = (int)(next % MAX_KERNEL_STEP_EVENTS_PER_CHANNEL);
    struct ncclDevKernelStepEvent* st = &sub->stepStarted[ch].data[slot];
    struct ncclDevKernelStepEvent* co = &sub->stepCompleted[ch].data[slot];
    void** handleSlot = &comm->profiler.kernelStepHandles[ch * MAX_KERNEL_STEP_EVENTS_PER_CHANNEL + slot];

    if (st->counter != next) break;

    if (*handleSlot == nullptr) {
      (void)ncclProfilerStartKernelStepEvent(args, s, st, handleSlot);
    }

    if (co->counter != next) {
      // Start visible but end not yet; stop until produced catches up after work complete.
      if (produced <= drained) break;
      // If produced advanced past next but complete missing, keep waiting for this slot.
      break;
    }

    if (*handleSlot) {
      (void)ncclProfilerStopKernelStepEvent(*handleSlot, co);
      *handleSlot = nullptr;
    }
    drained = next;

    if (comm->profiler.stepSeq) produced = comm->profiler.stepSeq[ch];
    if (limit < produced) limit = produced;
  }

  comm->profiler.stepCounter[ch] = drained;
}

// The following ncclProxySubArgs are overloaded by the profiler progress function:
// - base       : is set to the current value of workCounter[channelId]
// - posted     : is set to sub->nsteps to indicate that the profiler has started the event
// - transmitted: is set to sub->nsteps to indicate that the profiler has stopped the event
static ncclResult_t profilerProxyProgress(struct ncclProxyState* proxyState, struct ncclProxyArgs* args) {
  struct ncclComm* comm = proxyState ? proxyState->comm : nullptr;
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
      struct ncclDevProfiler* workStarted = (struct ncclDevProfiler*)sub->sendbuff;
      struct ncclDevProfiler* workCompleted = (struct ncclDevProfiler*)sub->recvbuff;

      profilerDrainKernelSteps(args, s, comm);

      if (sub->posted < sub->nsteps &&
          sub->base <= workStarted[sub->channelId].data[sub->base % MAX_PROFILER_EVENTS_PER_CHANNEL].counter) {
        ncclProfilerStartKernelChEvent(
          args, s, workStarted[sub->channelId].data[sub->base % MAX_PROFILER_EVENTS_PER_CHANNEL].timestamp);
        sub->posted = sub->nsteps;
        profilerDrainKernelSteps(args, s, comm);
        continue; // allow events on every channel to start
      }
      if (sub->transmitted < sub->nsteps &&
          sub->base <= workCompleted[sub->channelId].data[sub->base % MAX_PROFILER_EVENTS_PER_CHANNEL].counter) {
        profilerDrainKernelSteps(args, s, comm);
        ncclProfilerStopKernelChEvent(
          args, s, workCompleted[sub->channelId].data[sub->base % MAX_PROFILER_EVENTS_PER_CHANNEL].timestamp);
        profilerDrainKernelSteps(args, s, comm);
        sub->transmitted = sub->nsteps;
        args->done++;
      }
    }
    if (args->done == args->nsubs) args->state = ncclProxyOpNone;
  }
  return ncclSuccess;
}

struct ncclTransport profilerTransport = {"Prof",
                                          NULL,
                                          {NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL},
                                          {NULL, NULL, NULL, NULL, NULL, profilerProxyConnect, NULL,
                                           profilerProxyProgress, NULL, NULL}};
