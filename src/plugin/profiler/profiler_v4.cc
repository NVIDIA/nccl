/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "comm.h"
#include "nccl_profiler.h"
#include "checks.h"
#include "os.h"
#include "profiler_shim.h"

NCCL_PROFILER_SHIM_BEGIN(4)

static ncclResult_t ncclProfiler_startEvent(void* ctx, void** eHandle, ncclProfilerEventDescr_t* eDescr) {
  ncclProfilerEventDescr_v4_t eDescr_v4;
  eDescr_v4.type = eDescr->type;
  eDescr_v4.parentObj = eDescr->parentObj;
  eDescr_v4.rank = eDescr->rank;
  switch (eDescr->type) {
  case ncclProfileGroup:
    break;
  case ncclProfileColl:
    {
      eDescr_v4.coll.seqNumber = eDescr->coll.seqNumber;
      eDescr_v4.coll.func = eDescr->coll.func;
      eDescr_v4.coll.sendBuff = eDescr->coll.sendBuff;
      eDescr_v4.coll.recvBuff = eDescr->coll.recvBuff;
      eDescr_v4.coll.count = eDescr->coll.count;
      eDescr_v4.coll.root = eDescr->coll.root;
      eDescr_v4.coll.datatype = eDescr->coll.datatype;
      eDescr_v4.coll.nChannels = eDescr->coll.nChannels;
      eDescr_v4.coll.nWarps = eDescr->coll.nWarps;
      eDescr_v4.coll.algo = eDescr->coll.algo;
      eDescr_v4.coll.proto = eDescr->coll.proto;
      eDescr_v4.parentObj = eDescr->coll.parentGroup;
    }
    break;
  case ncclProfileP2p:
    {
      eDescr_v4.p2p.func = eDescr->p2p.func;
      eDescr_v4.p2p.buff = eDescr->p2p.buff;
      eDescr_v4.p2p.count = eDescr->p2p.count;
      eDescr_v4.p2p.datatype = eDescr->p2p.datatype;
      eDescr_v4.p2p.peer = eDescr->p2p.peer;
      eDescr_v4.parentObj = eDescr->p2p.parentGroup;
    }
    break;
  case ncclProfileProxyOp:
    {
      eDescr_v4.proxyOp.pid = eDescr->proxyOp.pid;
      eDescr_v4.proxyOp.channelId = eDescr->proxyOp.channelId;
      eDescr_v4.proxyOp.peer = eDescr->proxyOp.peer;
      eDescr_v4.proxyOp.nSteps = eDescr->proxyOp.nSteps;
      eDescr_v4.proxyOp.chunkSize = eDescr->proxyOp.chunkSize;
      eDescr_v4.proxyOp.isSend = eDescr->proxyOp.isSend;
    }
    break;
  case ncclProfileProxyStep:
    {
      eDescr_v4.proxyStep.step = eDescr->proxyStep.step;
    }
    break;
  case ncclProfileProxyCtrl:
    break;
  case ncclProfileKernelCh:
    {
      eDescr_v4.kernelCh.channelId = eDescr->kernelCh.channelId;
      eDescr_v4.kernelCh.pTimer = eDescr->kernelCh.pTimer;
    }
    break;
  case ncclProfileNetPlugin:
    {
      eDescr_v4.netPlugin.id = eDescr->netPlugin.id;
      eDescr_v4.netPlugin.data = eDescr->netPlugin.data;
    }
    break;
  default:
    return ncclSuccess;
  }
  NCCL_PROFILER_SHIM_CALL(startEvent, ctx, eHandle, &eDescr_v4);
}

static ncclResult_t ncclProfiler_recordEventState(void* eHandle, ncclProfilerEventState_t eState,
                                                  ncclProfilerEventStateArgs_t* eStateArgs) {
  ncclProfilerEventStateArgs_v4_t eStateArgs_v4;
  switch (eState) {
  case ncclProfilerProxyOpInProgress_v4:
    break;
  case ncclProfilerProxyStepSendGPUWait:
  case ncclProfilerProxyStepSendPeerWait_v4:
  case ncclProfilerProxyStepSendWait:
  case ncclProfilerProxyStepRecvWait:
  case ncclProfilerProxyStepRecvFlushWait:
  case ncclProfilerProxyStepRecvGPUWait:
    eStateArgs_v4.proxyStep.transSize = eStateArgs->proxyStep.transSize;
    break;
  case ncclProfilerNetPluginUpdate:
    eStateArgs_v4.netPlugin.data = eStateArgs->netPlugin.data;
    break;
  case ncclProfilerKernelChStop:
    eStateArgs_v4.kernelCh.pTimer = eStateArgs->kernelCh.pTimer;
    break;
  case ncclProfilerProxyCtrlIdle:
  case ncclProfilerProxyCtrlActive:
  case ncclProfilerProxyCtrlSleep:
  case ncclProfilerProxyCtrlWakeup:
  case ncclProfilerProxyCtrlAppend:
  case ncclProfilerProxyCtrlAppendEnd:
    eStateArgs_v4.proxyCtrl.appendedProxyOps = eStateArgs->proxyCtrl.appendedProxyOps;
    break;
  default:
    return ncclSuccess;
  }
  NCCL_PROFILER_SHIM_CALL(recordEventState, eHandle, (ncclProfilerEventState_v4_t)eState, &eStateArgs_v4);
}

static ncclResult_t ncclProfiler_finalize(void* context) {
  NCCL_PROFILER_SHIM_CALL(finalize, context);
}

static ncclResult_t ncclProfiler_stopEvent(void* eHandle, ncclProfilerEventDescr_t* eStopDescr) {
  // v4 stopEvent takes no stop timestamps; drop the v8 stop descriptor.
  (void)eStopDescr;
  NCCL_PROFILER_SHIM_CALL(stopEvent, eHandle);
}

static ncclResult_t ncclProfiler_init(void** ctx, uint64_t commId, int* eActivationMask, const char* commName,
                                      int nNodes, int nRanks, int rank, ncclDebugLogger_t logfn,
                                      ncclProfilerConfig_t* config __attribute__((unused))) {
  ncclProfilerShim_t* p = getActiveProfiler();
  if (p == nullptr) return ncclInternalError;
  NCCLCHECK(p->init(ctx, eActivationMask, commName, commId, nNodes, nRanks, rank, logfn));
  return ncclSuccess;
}

NCCL_PROFILER_SHIM_END(4)
