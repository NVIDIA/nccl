/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef PROFILER_H_
#define PROFILER_H_

#include <cuda_runtime.h>
#include "nccl_profiler.h"
#include "profiler_event.h"

struct ncclProxyArgs;
struct ncclKernelPlan;
struct ncclTaskColl;
struct ncclTaskP2p;
struct ncclInfo;
struct ncclComm;
struct ncclProxyOp;
struct ncclDevProfiler;
struct ncclDevProfilerPhases;

struct ncclProfilerWorkOp {
  int channelId;
  uint64_t workCounter;
  int eActivationMask;
  struct ncclProfilerEventRef taskEventHandle;
  void* profilerContext;
  struct ncclProfilerEventRef kernelEventHandle;
  // Captured per-op so a profiler thread shared across comm-split children
  // polls the correct (originating comm) buffers.
  struct ncclDevProfiler* workStarted;
  struct ncclDevProfiler* workCompleted;
  struct ncclDevProfilerPhases* workPhases;
  bool started;
  bool completed;
  struct ncclProfilerWorkOp* next;
};

struct ncclProfilerThread;

// Per-comm profiler state polled by the dedicated profiler thread.
struct ncclProfilerCommState {
  struct ncclDevProfiler* workStarted /*[MAXCHANNELS]*/;
  struct ncclDevProfiler* workCompleted /*[MAXCHANNELS]*/;
  uint64_t workCounter[MAXCHANNELS];
  // Dedicated counter/buffers for symmetric collectives: sym kernels read a fixed
  // counter from the args buffer and never advance the device channels[].workCounter,
  // so sharing the above with regular/p2p kernels would desync the two sequences.
  struct ncclDevProfiler* symWorkStarted /*[MAXCHANNELS]*/;
  struct ncclDevProfiler* symWorkCompleted /*[MAXCHANNELS]*/;
  struct ncclDevProfilerPhases* symWorkPhases /*[MAXCHANNELS]*/;
  uint64_t symWorkCounter[MAXCHANNELS];
  // Shared with comm-split children when shareResources is set.
  struct ncclProfilerThread* profilerThread;
  // What the plugin asked for during init on this communicator.
  ncclProfilerConfig_t config;
  // Whether this comm's profiler init() succeeded, and so whether its plugin
  // reference must be released. Not inferrable from profilerContext itself: a
  // stateless plugin may return NULL and succeed, and a plugin can stay loaded
  // for other comms after failing this one.
  bool contextValid;
};

enum groupApiState {
  ncclProfilerGroupApiStartStateReset = 0,
  ncclProfilerGroupApiStartStateStarted = 1,
  ncclProfilerGroupApiStartStateStopped = 2,
};

// Used by the profiler to track state for API events
typedef struct ncclProfilerApiState {
  int profilerGroupDepth;
  int eActivationMask;
  groupApiState state;
  struct ncclProfilerEventRef groupApiEventHandle;
  // Tracks the latest API event handles for p2p/collectives
  struct ncclProfilerEventRef p2pApiEventHandle;
  struct ncclProfilerEventRef collApiEventHandle;
} ncclProfilerApiState_t;

extern thread_local ncclProfilerApiState_t ncclProfilerApiState;

// Profiler linkage of one API call, snapshotted on the calling thread by ncclApiProfiler.
// Task preparation may run on another thread, so it must not read the thread-local
// ncclProfilerApiState; it carries this snapshot instead.
struct ncclProfilerApiCtx {
  int eActivationMask;        // event mask snapshot for the enclosing group
  struct ncclProfilerEventRef groupApiEventHandle;  // ncclProfileGroupApi event this call belongs to
  struct ncclProfilerEventRef apiEventHandle;       // ncclProfileCollApi / ncclProfileP2pApi event for this call
  uint64_t userTag;           // per-call user profiler tag from the CollConfig (0 == untagged)
};

// Profiler API-event bracket for one collective/p2p/RMA API call. ncclApiProfilerStart opens
// the group API event and the Coll/P2p API event for info->coll and snapshots their linkage in
// ctx; ncclApiProfilerStop closes the API event. The destructor calls stop if it was not called
// explicitly, so every early exit closes what start opened. Call ncclPlannerSetCapturingGraph
// for the call's stream before start so the group API event reports graph capture correctly.
//
//   struct ncclApiProfiler profiler = {comm, info};
//   NCCLCHECK(ncclApiProfilerStart(&profiler));
//   ...
//   NCCLCHECK(ncclApiProfilerStop(&profiler));
struct ncclApiProfiler;
ncclResult_t ncclApiProfilerStart(struct ncclApiProfiler* profiler);
ncclResult_t ncclApiProfilerStop(struct ncclApiProfiler* profiler);

struct ncclApiProfiler {
  struct ncclComm* comm;
  struct ncclInfo* info;
  bool started;
  struct ncclProfilerApiCtx ctx; // valid after ncclApiProfilerStart succeeds

  ~ncclApiProfiler() {
    (void)ncclApiProfilerStop(this);
  }
};

// Copy an API call's profiler linkage onto a task (ncclTaskColl, ncclTaskBcast, ncclTaskP2p, ...).
template <typename TaskT>
inline void ncclTaskCaptureProfilerCtx(TaskT* task, const struct ncclProfilerApiCtx& ctx) {
  task->eActivationMask = ctx.eActivationMask;
  task->groupApiEventHandle = ctx.groupApiEventHandle;
  task->apiEventHandle = ctx.apiEventHandle;
  task->profilerTag = ctx.userTag;
}

extern int ncclProfilerEventMask;

// Rebuilds the union of per-plugin masks and returns it. Plugins may write the
// pointer init() gave them (exampleProfilerStart); this copies that into the
// process-wide mask enqueue snapshots onto tasks.
int ncclProfilerCurrentEventMask(void);

// Out-of-band (RAS) override of the event mask; see profiler.cc.
void ncclProfilerSetRasOverride(int mask);

// Plugin Init/Finalize Wrappers
ncclResult_t ncclProfilerPluginInit(struct ncclComm* comm);
ncclResult_t ncclProfilerPluginFinalize(struct ncclComm* comm);

// Profiler Start/Stop/Record wrappers for ncclGroupStart and ncclGroupEnd API calls
ncclResult_t ncclProfilerStartGroupApiEvent(struct ncclInfo* info, bool isGraphCaptured);
ncclResult_t ncclProfilerStopGroupApiEvent();
ncclResult_t ncclProfilerRecordGroupApiEventState(ncclProfilerEventState_t eState);

// Profiler Start/Stop wrappers for P2p API calls
ncclResult_t ncclProfilerStartP2pApiEvent(struct ncclInfo* info, bool isGraphCaptured);
ncclResult_t ncclProfilerStopP2pApiEvent();

// Profiler Start/Stop wrappers for Collective API calls
ncclResult_t ncclProfilerStartCollApiEvent(struct ncclInfo* info, bool isGraphCaptured);
ncclResult_t ncclProfilerStopCollApiEvent();

// Kernel Launch Start/Stop Event Wrappers
ncclResult_t ncclProfilerStartKernelLaunchEvent(struct ncclKernelPlan* plan, cudaStream_t stream);
ncclResult_t ncclProfilerStopKernelLaunchEvent(struct ncclKernelPlan* plan);

// Profiler Start/Stop Group Wrappers
ncclResult_t ncclProfilerStartGroupEvent(struct ncclKernelPlan* plan);
ncclResult_t ncclProfilerStopGroupEvent(struct ncclKernelPlan* plan);

// Profiler Start/Stop Task Events Wrappers
ncclResult_t ncclProfilerStartTaskEvents(struct ncclKernelPlan* plan);
ncclResult_t ncclProfilerStopTaskEvents(struct ncclKernelPlan* plan);

// Launch events for the rearch schedulers (enqueue/task_sched), keyed by an
// ncclSchedProfilerLaunch (enqueue/sched_profiler.h). Both accept a null launch.
struct ncclSchedProfilerLaunch;
// Before the driver launch: reserve the sym KernelCh counters in the kernel args, start the
// ncclProfileGroup and ncclProfileColl events of an eager launch and post its KernelCh work, then
// start ncclProfileKernelLaunch.
ncclResult_t ncclProfilerSchedPreLaunch(struct ncclSchedProfilerLaunch* launch, cudaStream_t stream);
// After the driver launch: stop ncclProfileKernelLaunch, then the group and collective events of
// an eager launch. When a captured launch succeeded, register its per-replay callback.
ncclResult_t ncclProfilerSchedPostLaunch(struct ncclSchedProfilerLaunch* launch, cudaStream_t stream, bool launched);
// Proxy Op Start/Stop Event Wrappers
ncclResult_t ncclProfilerStartProxyOpEvent(int sub, struct ncclProxyArgs* args);
ncclResult_t ncclProfilerStopProxyOpEvent(int sub, struct ncclProxyArgs* args);

// Proxy Step Start/Stop Event Wrappers
ncclResult_t ncclProfilerStartSendProxyStepEvent(int sub, struct ncclProxyArgs* args, int stepId);
ncclResult_t ncclProfilerStartRecvProxyStepEvent(int sub, struct ncclProxyArgs* args, int stepId);
ncclResult_t ncclProfilerStopProxyStepEvent(int sub, struct ncclProxyArgs* args, int stepId);

// Proxy Control Start/Stop Events Wrappers
ncclResult_t ncclProfilerStartProxyCtrlEvent(void* profilerContext, struct ncclProfilerEventRef* eHandle);
ncclResult_t ncclProfilerStopProxyCtrlEvent(struct ncclProfilerEventRef* eHandle);

// Kernel Channel Start/Stop Event Wrappers
ncclResult_t ncclProfilerStartKernelChEvent(struct ncclProfilerWorkOp* op, uint64_t start);
ncclResult_t ncclProfilerStopKernelChEvent(struct ncclProfilerWorkOp* op, uint64_t stop);

// Record Event Wrappers
ncclResult_t ncclProfilerRecordProxyOpEventState(int sub, struct ncclProxyArgs* args, ncclProfilerEventState_t eState);
ncclResult_t ncclProfilerRecordProxyStepEventState(int sub, struct ncclProxyArgs* args, int stepId,
                                                   ncclProfilerEventState_t eState);
ncclResult_t ncclProfilerRecordProxyCtrlEventState(struct ncclProfilerEventRef* eHandle, int appended,
                                                   ncclProfilerEventState_t eState);

// Profiler utility functions
ncclResult_t ncclProfilerAddPidToProxyOp(struct ncclProxyOp* op);
bool ncclProfilerPluginLoaded(void);
uint8_t ncclProfilerDeviceMode(int eActivationMask);

// GPU-clock probe (defined in device/onerank.cu): launches a 1-thread kernel
// that writes the GPU globaltimer into *devOut (host-visible). Used by the
// profiler thread's GPU->CPU time calibration when a plugin has asked for
// NCCL-sourced timings.
ncclResult_t ncclProfilerProbeGpuClock(unsigned long long* devOut, cudaStream_t stream);

// Dedicated profiler thread API
ncclResult_t ncclProfilerThreadCreate(struct ncclComm* comm, struct ncclComm* parent);
ncclResult_t ncclProfilerThreadDestroy(struct ncclComm* comm);
// Post KernelCh work per (task, channel) for every KernelCh-enabled task in
// plan. Called from hostStreamPlanCallback so the host workCounter stays in
// lock-step with the device on every (graph-captured) replay.
ncclResult_t ncclProfilerPostPlanWork(struct ncclComm* comm, struct ncclKernelPlan* plan);
// Reserve the per-channel sym workCounters into the plan's args buffer before
// cuLaunchKernel snapshots it (called pre-launch). The matching KernelCh ops are
// posted later by ncclProfilerPostPlanWork() from the host callback.
void ncclProfilerReserveSymCounters(struct ncclComm* comm, struct ncclKernelPlan* plan);

// Profiler callback for network plugin
ncclResult_t ncclProfilerCallback(void** eHandle, int type, void* pHandle, int64_t pluginId, void* extData);

// ============================================================================
// CE Profiler Declarations
// ============================================================================

// Forward declarations for CE types
struct ncclCeCollArgs;
struct ncclCeBatchOpsParams;

// CE profiler event start/stop functions (simple wrappers that call plugin callbacks)
ncclResult_t ncclProfilerStartCeCollEvent(struct ncclComm* comm, struct ncclCeCollArgs* args, cudaStream_t stream);
ncclResult_t ncclProfilerStopCeCollEvent(struct ncclComm* comm, struct ncclCeCollArgs* args, cudaStream_t stream);
ncclResult_t ncclProfilerStartCeSyncEvent(struct ncclComm* comm, struct ncclCeCollArgs* args, cudaStream_t stream,
                                          struct ncclProfilerEventRef* ceSyncHandle);
ncclResult_t ncclProfilerStopCeSyncEvent(struct ncclComm* comm, struct ncclProfilerEventRef* ceSyncHandle,
                                         cudaStream_t stream);
ncclResult_t ncclProfilerStartCeBatchEvent(struct ncclComm* comm, struct ncclCeCollArgs* args,
                                           struct ncclCeBatchOpsParams* params, cudaStream_t stream,
                                           struct ncclProfilerEventRef* ceBatchHandle);
ncclResult_t ncclProfilerStopCeBatchEvent(struct ncclComm* comm, struct ncclProfilerEventRef* ceBatchHandle,
                                          cudaStream_t stream);

#endif
