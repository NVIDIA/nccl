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

// Started-but-uncompleted distance after which a KernelStep completion is
// treated as lost (see profilerDrainKernelSteps). A channel's thread block
// runs slices in order, so only about one slice per peer is ever in flight;
// 64 later starts take a few ms. At 1024, every missing completion held all
// later KernelStep stops on its channel back by about one training step
// (DP2/TP2/PP2), too late for per-iteration copy-rate checks.
#define KERNEL_STEP_COMPLETION_LOST_LAG 64

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
  if ((uint32_t)parent->workCounter != st->work_tag) return;

  void* handle = nullptr;
  (void)ncclProfilerStartKernelStepEvent(parent, ch, st, &handle);
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

// Close an unresolved seq so CoMMA is not left holding a start-only handle.
static void profilerAbandonKernelStep(struct ncclProxySubArgs* sub, struct ncclComm* comm, int ch,
                                      uint64_t seq) {
  int slot = (int)(seq % MAX_KERNEL_STEP_EVENTS_PER_CHANNEL);
  size_t handleIndex = (size_t)ch * MAX_KERNEL_STEP_EVENTS_PER_CHANNEL + slot;
  void* handle = comm->profiler.kernelStepHandles[handleIndex];
  if (handle != nullptr) {
    struct ncclDevKernelStepEvent* co = &sub->stepCompleted[ch].data[slot];
    struct ncclDevKernelStepEvent* ev =
      (co->counter == seq) ? co : &sub->stepStarted[ch].data[slot];
    (void)ncclProfilerStopKernelStepEvent(handle, ev);
  }
  comm->profiler.kernelStepHandles[handleIndex] = nullptr;
  comm->profiler.kernelStepHandleSeq[handleIndex] = 0;
}

// Drain starts and completions with independent sequential cursors. A completion
// can lag an older step for a long time, but it must not prevent later starts from
// reaching the online monitor. Conversely, seeing a later sequence is not proof
// that an earlier GPU writer was lost: sequence reservation happens before the
// payload's system-scope publication, so writers may become visible out of order.
// Only skip an entry after its ring slot has necessarily wrapped.
static void profilerDrainKernelSteps(struct ncclProxyArgs* args, int s, struct ncclComm* comm) {
  struct ncclProxySubArgs* sub = args->subs + s;
  if (!(sub->eActivationMask & ncclProfileKernelStep) || sub->stepStarted == nullptr ||
      sub->stepCompleted == nullptr || comm == nullptr || comm->profiler.stepSeq == nullptr ||
      comm->profiler.kernelStepParents == nullptr) return;

  int ch = sub->channelId;
  uint64_t produced = comm->profiler.stepSeq[ch];

  uint64_t starts = comm->profiler.stepStartCounter[ch];
  while (starts < produced) {
    uint64_t next = starts + 1;
    int slot = (int)(next % MAX_KERNEL_STEP_EVENTS_PER_CHANNEL);
    struct ncclDevKernelStepEvent* st = &sub->stepStarted[ch].data[slot];
    if (st->counter != next) {
      if (produced - next < MAX_KERNEL_STEP_EVENTS_PER_CHANNEL) break;
      profilerAbandonKernelStep(sub, comm, ch, next);
    } else {
      profilerStartKernelStep(sub, comm, ch, next);
    }
    starts = next;
  }
  comm->profiler.stepStartCounter[ch] = starts;

  uint64_t drained = comm->profiler.stepCounter[ch];
  while (drained < starts) {
    uint64_t next = drained + 1;
    int slot = (int)(next % MAX_KERNEL_STEP_EVENTS_PER_CHANNEL);
    struct ncclDevKernelStepEvent* co = &sub->stepCompleted[ch].data[slot];
    if (co->counter != next) {
      // One thread block runs a channel's slices in order (wait, copy, post),
      // so a completion still missing after KERNEL_STEP_COMPLETION_LOST_LAG
      // later steps started was never written. Waiting for a full ring wrap
      // instead froze every later KernelStep stop on this channel.
      if (produced - next < KERNEL_STEP_COMPLETION_LOST_LAG) break;
      profilerAbandonKernelStep(sub, comm, ch, next);
    } else {
      profilerStopKernelStep(sub, comm, ch, next);
    }
    drained = next;
    produced = comm->profiler.stepSeq[ch];
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
        continue; // allow events on every channel to start
      }
      if (sub->transmitted < sub->nsteps &&
          sub->base <= workCompleted[sub->channelId].data[sub->base % MAX_PROFILER_EVENTS_PER_CHANNEL].counter) {
        // Drain children before KernelCh stop can release/reclaim their parent
        // in the plugin. Terminal device work publication is ordered after the
        // final KernelStep stop by ncclKernelMain's profiling barrier.
        profilerDrainKernelSteps(args, s, comm);
        ncclProfilerStopKernelChEvent(
          args, s, workCompleted[sub->channelId].data[sub->base % MAX_PROFILER_EVENTS_PER_CHANNEL].timestamp);
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
