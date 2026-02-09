/*************************************************************************
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef PROFILER_H_
#define PROFILER_H_

#include <cuda_runtime.h>
#include "nccl_profiler.h"

struct ncclProxyArgs;
struct ncclKernelPlan;
struct ncclTaskColl;
struct ncclTaskP2p;
struct ncclInfo;
struct ncclComm;
struct ncclProxyOp;
struct ncclProxyConnector;

struct ncclProfilerProxy {
  bool initialized;
  struct ncclDevProfiler* workStarted/*[MAXCHANNELS]*/;
  struct ncclDevProfiler* workCompleted/*[MAXCHANNELS]*/;
  uint64_t workCounter[MAXCHANNELS]; // host work counter
  struct ncclProxyConnector sendProxyConn[MAXCHANNELS];
  struct ncclProxyConnector recvProxyConn[MAXCHANNELS];
};

enum groupApiState {
  ncclProfilerGroupApiStartStateReset   = 0,
  ncclProfilerGroupApiStartStateStarted = 1,
  ncclProfilerGroupApiStartStateStopped = 2,
};

// Used by the profiler to track state for API events
typedef struct ncclProfilerApiState {
  int profilerGroupDepth;
  int eActivationMask;
  groupApiState state;
  void *groupApiEventHandle;
  // Tracks the latest API event handles for p2p/collectives
  void* p2pApiEventHandle;
  void *collApiEventHandle;
} ncclProfilerApiState_t;

extern thread_local ncclProfilerApiState_t ncclProfilerApiState;

extern int ncclProfilerEventMask;

// Plugin Init/Finalize Wrappers
ncclResult_t ncclProfilerPluginInit(struct ncclComm* comm);
ncclResult_t ncclProfilerPluginFinalize(struct ncclComm* comm);

// Profiler Start/Stop/Record wrappers for ncclGroupStart and ncclGroupEnd API calls
ncclResult_t ncclProfilerStartGroupApiEvent(struct ncclInfo *info, bool isGraphCaptured);
ncclResult_t ncclProfilerStopGroupApiEvent();
ncclResult_t ncclProfilerRecordGroupApiEventState(ncclProfilerEventState_t eState);

//Profiler Start/Stop wrappers for P2p API calls
ncclResult_t ncclProfilerStartP2pApiEvent(struct ncclInfo *info, bool isGraphCaptured);
ncclResult_t ncclProfilerStopP2pApiEvent();

//Profiler Start/Stop wrappers for Collective API calls
ncclResult_t ncclProfilerStartCollApiEvent(struct ncclInfo *info, bool isGraphCaptured);
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

// Proxy Op Start/Stop Event Wrappers
ncclResult_t ncclProfilerStartProxyOpEvent(int sub, struct ncclProxyArgs* args);
ncclResult_t ncclProfilerStopProxyOpEvent(int sub, struct ncclProxyArgs* args);

// Proxy Step Start/Stop Event Wrappers
ncclResult_t ncclProfilerStartSendProxyStepEvent(int sub, struct ncclProxyArgs* args, int stepId);
ncclResult_t ncclProfilerStartRecvProxyStepEvent(int sub, struct ncclProxyArgs* args, int stepId);
ncclResult_t ncclProfilerStopProxyStepEvent(int sub, struct ncclProxyArgs* args, int stepId);

// Proxy Control Start/Stop Events Wrappers
ncclResult_t ncclProfilerStartProxyCtrlEvent(void* profilerContext, void** eHandle);
ncclResult_t ncclProfilerStopProxyCtrlEvent(void* eHandle);

// Kernel Channel Start/Stop Event Wrappers
ncclResult_t ncclProfilerStartKernelChEvent(struct ncclProxyArgs* args, int s, uint64_t start);
ncclResult_t ncclProfilerStopKernelChEvent(struct ncclProxyArgs* args, int s, uint64_t stop);

// Record Event Wrappers
ncclResult_t ncclProfilerRecordProxyOpEventState(int sub, struct ncclProxyArgs* args, ncclProfilerEventState_t eState);
ncclResult_t ncclProfilerRecordProxyStepEventState(int sub, struct ncclProxyArgs* args, int stepId, ncclProfilerEventState_t eState);
ncclResult_t ncclProfilerRecordProxyCtrlEventState(void*eHandle, int appended, ncclProfilerEventState_t eState);

// Profiler utility functions
ncclResult_t ncclProfilerAddPidToProxyOp(struct ncclProxyOp* op);
bool ncclProfilerNeedsProxy(struct ncclComm* comm, struct ncclProxyOp* op);
bool ncclProfilerPluginLoaded(void);

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
ncclResult_t ncclProfilerStartCeSyncEvent(struct ncclComm* comm, struct ncclCeCollArgs* args, cudaStream_t stream, void** ceSyncHandle);
ncclResult_t ncclProfilerStopCeSyncEvent(struct ncclComm* comm, void* ceSyncHandle, cudaStream_t stream);
ncclResult_t ncclProfilerStartCeBatchEvent(struct ncclComm* comm, struct ncclCeCollArgs* args, struct ncclCeBatchOpsParams* params, cudaStream_t stream, void** ceBatchHandle);
ncclResult_t ncclProfilerStopCeBatchEvent(struct ncclComm* comm, void* ceBatchHandle, cudaStream_t stream);

#endif
