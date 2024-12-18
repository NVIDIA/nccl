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

// Plugin Init/Finalize Wrappers
ncclResult_t ncclProfilerPluginInit(struct ncclComm* comm);
ncclResult_t ncclProfilerPluginFinalize(struct ncclComm* comm);

// Profiler Start/Stop Group Wrappers
ncclResult_t ncclProfilerStartGroupEvent(struct ncclKernelPlan* plan);
ncclResult_t ncclProfilerStopGroupEvent(struct ncclKernelPlan* plan);

// Profiler Start/Stop Task Events Wrappers
ncclResult_t ncclProfilerStartTaskEvents(struct ncclKernelPlan* plan);
ncclResult_t ncclProfilerStopTaskEvents(struct ncclKernelPlan* plan);

// Proxy Op Start/Stop Event Wrappers
ncclResult_t ncclProfilerStartSendProxyOpEvent(int sub, struct ncclProxyArgs* args);
ncclResult_t ncclProfilerStartRecvProxyOpEvent(int sub, struct ncclProxyArgs* args);
ncclResult_t ncclProfilerStopProxyOpEvent(int sub, struct ncclProxyArgs* args);

// Proxy Step Start/Stop Event Wrappers
ncclResult_t ncclProfilerStartSendProxyStepEvent(int sub, struct ncclProxyArgs* args, int stepId);
ncclResult_t ncclProfilerStartRecvProxyStepEvent(int sub, struct ncclProxyArgs* args, int stepId);
ncclResult_t ncclProfilerStopProxyStepEvent(int sub, struct ncclProxyArgs* args, int stepId);

// Proxy Control Start/Stop Events Wrappers
ncclResult_t ncclProfilerStartProxyCtrlEvent(void* profilerContext, void** eHandle);
ncclResult_t ncclProfilerStopProxyCtrlEvent(void* eHandle);

// Record Event Wrappers
ncclResult_t ncclProfilerRecordProxyOpEventState(int sub, struct ncclProxyArgs* args, int steps, size_t transSize, ncclProfilerEventState_t eState);
ncclResult_t ncclProfilerRecordProxyStepEventState(int sub, struct ncclProxyArgs* args, int stepId, ncclProfilerEventState_t eState);
ncclResult_t ncclProfilerRecordProxyCtrlEventState(void*eHandle, int appended, ncclProfilerEventState_t eState);

// Profiler utility functions
ncclResult_t ncclProfilerAddPidToProxyOp(struct ncclProxyOp* op);

#endif
