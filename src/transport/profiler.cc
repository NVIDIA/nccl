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

// Try one start/end pair. Sequence allocation and CUDA-role completion order
// are independent, so callers must not assume that sequence N completes before N+1.
static bool profilerTryDrainKernelStep(struct ncclProxySubArgs* sub, struct ncclComm* comm, int ch, uint64_t seq) {
  int slot = (int)(seq % MAX_KERNEL_STEP_EVENTS_PER_CHANNEL);
  struct ncclDevKernelStepEvent* st = &sub->stepStarted[ch].data[slot];
  struct ncclDevKernelStepEvent* co = &sub->stepCompleted[ch].data[slot];
  if (st->counter != seq || co->counter != seq) return false;

  int dir = (st->flags & NCCL_KERNEL_STEP_FLAG_SEND) ? 1 : 0;
  int parentSlot = (int)(st->work_tag % MAX_KERNEL_STEP_PARENT_EVENTS);
  size_t parentIndex = ((size_t)ch * 2 + dir) * MAX_KERNEL_STEP_PARENT_EVENTS + parentSlot;
  struct ncclKernelStepParent* parent = comm->profiler.kernelStepParents + parentIndex;
  if ((uint16_t)parent->workCounter != st->work_tag) return false;

  struct ncclProxyArgs routedArgs = {};
  routedArgs.subs[0].eActivationMask = parent->eActivationMask;
  routedArgs.subs[0].taskEventHandle = parent->taskEventHandle;
  routedArgs.subs[0].profilerContext = parent->profilerContext;
  routedArgs.subs[0].rank = parent->rank;
  routedArgs.subs[0].channelId = ch;
  void* handle = nullptr;
  (void)ncclProfilerStartKernelStepEvent(&routedArgs, 0, st, &handle);
  if (handle) (void)ncclProfilerStopKernelStepEvent(handle, co);
  return true;
}

// Drain completed pairs immediately and retain only unresolved sequence IDs.
// This is O(new events + unresolved events), not O(the full device ring).
static void profilerDrainKernelSteps(struct ncclProxyArgs* args, int s, struct ncclComm* comm) {
  struct ncclProxySubArgs* sub = args->subs + s;
  if (!(sub->eActivationMask & ncclProfileKernelStep) || sub->stepStarted == nullptr ||
      sub->stepCompleted == nullptr || comm == nullptr || comm->profiler.stepSeq == nullptr ||
      comm->profiler.kernelStepPending == nullptr || comm->profiler.kernelStepParents == nullptr) return;

  int ch = sub->channelId;
  uint64_t discovered = comm->profiler.stepCounter[ch];
  uint64_t produced = comm->profiler.stepSeq[ch];
  uint64_t* pending = comm->profiler.kernelStepPending + (size_t)ch * MAX_KERNEL_STEP_EVENTS_PER_CHANNEL;
  int count = comm->profiler.kernelStepPendingCount[ch];
  int keep = 0;

  for (int i = 0; i < count; i++) {
    uint64_t seq = pending[i];
    if (produced >= seq && produced - seq >= MAX_KERNEL_STEP_EVENTS_PER_CHANNEL) continue; // slot overwritten
    if (!profilerTryDrainKernelStep(sub, comm, ch, seq)) pending[keep++] = seq;
  }

  uint64_t first = discovered + 1;
  if (produced - discovered > MAX_KERNEL_STEP_EVENTS_PER_CHANNEL)
    first = produced - MAX_KERNEL_STEP_EVENTS_PER_CHANNEL + 1;
  for (uint64_t seq = first; seq <= produced; seq++) {
    if (!profilerTryDrainKernelStep(sub, comm, ch, seq) && keep < MAX_KERNEL_STEP_EVENTS_PER_CHANNEL)
      pending[keep++] = seq;
  }

  comm->profiler.kernelStepPendingCount[ch] = keep;
  comm->profiler.stepCounter[ch] = produced;
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
