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

// KernelStep is intra-host only (P2P/SHM/NVLS). Drop NET/inter-host leftovers
// so CoMMA never sees a KernelStep whose peer lives on another host.
static bool ncclKernelStepSameHost(struct ncclComm* comm, int peer) {
  if (comm == nullptr || comm->peerInfo == nullptr) return false;
  if (peer < 0 || peer >= comm->nRanks) return false;
  int rank = comm->rank;
  if (rank < 0 || rank >= comm->nRanks || peer == rank) return false;
  return comm->peerInfo[rank].hostHash == comm->peerInfo[peer].hostHash;
}

// Publish a KernelStep start as soon as the GPU start ring is visible.  The
// previous implementation waited for the completion ring and invoked start
// and stop back-to-back, which made in-flight localization impossible.
static void profilerStartKernelStep(struct ncclProxySubArgs* sub, struct ncclComm* comm, int ch,
                                    uint64_t seq) {
  int slot = (int)(seq % MAX_KERNEL_STEP_EVENTS_PER_CHANNEL);
  struct ncclDevKernelStepEvent* st = &sub->stepStarted[ch].data[slot];
  size_t handleIndex = (size_t)ch * MAX_KERNEL_STEP_EVENTS_PER_CHANNEL + slot;
  if (comm->profiler.kernelStepHandleSeq[handleIndex] == seq) return;
  comm->profiler.kernelStepHandleSeq[handleIndex] = seq;
  comm->profiler.kernelStepHandles[handleIndex] = nullptr;

  if (!ncclKernelStepSameHost(comm, (int)st->peer)) return;

  int dir = (st->flags & NCCL_KERNEL_STEP_FLAG_SEND) ? 1 : 0;
  int parentSlot = (int)(st->work_tag % MAX_KERNEL_STEP_PARENT_EVENTS);
  size_t parentIndex = ((size_t)ch * 2 + dir) * MAX_KERNEL_STEP_PARENT_EVENTS + parentSlot;
  struct ncclKernelStepParent* parent = comm->profiler.kernelStepParents + parentIndex;
  // Unroutable (parent slot reused / never saved): drop so the drain cursor can advance.
  if ((uint16_t)parent->workCounter != st->work_tag) return;

  struct ncclProxyArgs routedArgs = {};
  routedArgs.subs[0].eActivationMask = parent->eActivationMask;
  routedArgs.subs[0].taskEventHandle = parent->taskEventHandle;
  routedArgs.subs[0].profilerContext = parent->profilerContext;
  routedArgs.subs[0].rank = parent->rank;
  routedArgs.subs[0].channelId = ch;
  void* handle = nullptr;
  (void)ncclProfilerStartKernelStepEvent(&routedArgs, 0, st, &handle);
  comm->profiler.kernelStepHandles[handleIndex] = handle;
}

static void profilerStopKernelStep(struct ncclProxySubArgs* sub, struct ncclComm* comm, int ch,
                                   uint64_t seq) {
  int slot = (int)(seq % MAX_KERNEL_STEP_EVENTS_PER_CHANNEL);
  size_t handleIndex = (size_t)ch * MAX_KERNEL_STEP_EVENTS_PER_CHANNEL + slot;
  profilerStartKernelStep(sub, comm, ch, seq);
  void* handle = comm->profiler.kernelStepHandles[handleIndex];
  if (handle != nullptr) {
    struct ncclDevKernelStepEvent* co = &sub->stepCompleted[ch].data[slot];
    (void)ncclProfilerStopKernelStepEvent(handle, co);
  }
  comm->profiler.kernelStepHandles[handleIndex] = nullptr;
  comm->profiler.kernelStepHandleSeq[handleIndex] = 0;
}

// Sequential KernelStep drain (pre-sample-rate style) with work_tag parent routing.
//
// The sample-rate commit replaced this with a pending-list that jumped stepCounter to
// `produced` and retried all unresolved seqs every progress call. Incomplete pairs
// (or briefly not-yet-visible slots after atomicAdd on stepSeq) then made every NET
// proxy tick O(pending). With sample rate > 1 that contamination persisted into later
// send/recv in the same process (~10x ProxyStep latency). Without deferred-last,
// start/stop pairs complete in order, so a sequential cursor is correct and O(1) when
// the next seq is not ready yet.
static void profilerDrainKernelSteps(struct ncclProxyArgs* args, int s, struct ncclComm* comm) {
  struct ncclProxySubArgs* sub = args->subs + s;
  if (!(sub->eActivationMask & ncclProfileKernelStep) || sub->stepStarted == nullptr ||
      sub->stepCompleted == nullptr || comm == nullptr || comm->profiler.stepSeq == nullptr ||
      comm->profiler.kernelStepParents == nullptr) return;

  int ch = sub->channelId;
  uint64_t drained = comm->profiler.stepCounter[ch];
  uint64_t produced = comm->profiler.stepSeq[ch];

  while (drained < produced) {
    uint64_t next = drained + 1;
    int slot = (int)(next % MAX_KERNEL_STEP_EVENTS_PER_CHANNEL);
    struct ncclDevKernelStepEvent* st = &sub->stepStarted[ch].data[slot];
    struct ncclDevKernelStepEvent* co = &sub->stepCompleted[ch].data[slot];

    if (st->counter != next) {
      // Start not visible yet (slot fill lags atomicAdd on stepSeq), or lost.
      if (produced - next >= MAX_KERNEL_STEP_EVENTS_PER_CHANNEL) {
        drained = next;
        continue;
      }
      // If a later seq is already visible, this one was skipped/lost — advance.
      if (next < produced) {
        int slotN = (int)((next + 1) % MAX_KERNEL_STEP_EVENTS_PER_CHANNEL);
        uint64_t c1 = sub->stepStarted[ch].data[slotN].counter;
        uint64_t c2 = sub->stepCompleted[ch].data[slotN].counter;
        if (c1 == next + 1 || c2 == next + 1) {
          drained = next;
          continue;
        }
      }
      break;
    }
    // Start is observable independently of completion.  Keep the sequential
    // cursor on this sequence until its stop arrives; checking one slot per
    // proxy tick is O(1) and does not pollute the NET progress loop.
    profilerStartKernelStep(sub, comm, ch, next);
    if (co->counter != next) {
      if (produced - next >= MAX_KERNEL_STEP_EVENTS_PER_CHANNEL) {
        size_t handleIndex = (size_t)ch * MAX_KERNEL_STEP_EVENTS_PER_CHANNEL + slot;
        comm->profiler.kernelStepHandles[handleIndex] = nullptr;
        comm->profiler.kernelStepHandleSeq[handleIndex] = 0;
        drained = next;
        continue;
      }
      break;
    }

    profilerStopKernelStep(sub, comm, ch, next);
    drained = next;
    produced = comm->profiler.stepSeq[ch];
  }

  comm->profiler.stepCounter[ch] = drained;
  // Pending list unused by sequential drain; keep count clear so any stale state is inert.
  if (comm->profiler.kernelStepPendingCount) comm->profiler.kernelStepPendingCount[ch] = 0;
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
