/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "comm.h"
#include "nccl_profiler.h"
#include "plugin/profiler/profiler_v6.h"
#include "checks.h"
#include "os.h"

static ncclProfiler_v6_t* ncclProfiler_v6;
static ncclProfiler_t ncclProfiler;

static ncclResult_t ncclProfiler_startEvent(void* ctx, void** eHandle, ncclProfilerEventDescr_t* eDescr) {
  // v6 plugins don't understand kernel phase events; silently drop.
  if (eDescr->type == ncclProfileKernelPhase) {
    *eHandle = nullptr;
    return ncclSuccess;
  }

  ncclProfilerEventDescr_v6_t eDescr_v6 = {};
  eDescr_v6.type = eDescr->type;
  eDescr_v6.parentObj = eDescr->parentObj;
  eDescr_v6.rank = eDescr->rank;

  switch (eDescr->type) {
  case ncclProfileGroup:
    break;
  case ncclProfileGroupApi:
    {
      eDescr_v6.groupApi.graphCaptured = eDescr->groupApi.graphCaptured;
      eDescr_v6.groupApi.groupDepth = eDescr->groupApi.groupDepth;
    }
    break;
  case ncclProfileCollApi:
    {
      eDescr_v6.collApi.func = eDescr->collApi.func;
      eDescr_v6.collApi.count = eDescr->collApi.count;
      eDescr_v6.collApi.datatype = eDescr->collApi.datatype;
      eDescr_v6.collApi.root = eDescr->collApi.root;
      eDescr_v6.collApi.stream = eDescr->collApi.stream;
      eDescr_v6.collApi.graphCaptured = eDescr->collApi.graphCaptured;
    }
    break;
  case ncclProfileP2pApi:
    {
      eDescr_v6.p2pApi.func = eDescr->p2pApi.func;
      eDescr_v6.p2pApi.count = eDescr->p2pApi.count;
      eDescr_v6.p2pApi.datatype = eDescr->p2pApi.datatype;
      eDescr_v6.p2pApi.stream = eDescr->p2pApi.stream;
      eDescr_v6.p2pApi.graphCaptured = eDescr->p2pApi.graphCaptured;
    }
    break;
  case ncclProfileKernelLaunch:
    {
      eDescr_v6.kernelLaunch.stream = eDescr->kernelLaunch.stream;
    }
    break;
  case ncclProfileColl:
    {
    // v7-only fields (kernelVariant, isSymColl) are intentionally dropped.
      eDescr_v6.coll.seqNumber = eDescr->coll.seqNumber;
      eDescr_v6.coll.func = eDescr->coll.func;
      eDescr_v6.coll.sendBuff = eDescr->coll.sendBuff;
      eDescr_v6.coll.recvBuff = eDescr->coll.recvBuff;
      eDescr_v6.coll.count = eDescr->coll.count;
      eDescr_v6.coll.root = eDescr->coll.root;
      eDescr_v6.coll.datatype = eDescr->coll.datatype;
      eDescr_v6.coll.nChannels = eDescr->coll.nChannels;
      eDescr_v6.coll.nWarps = eDescr->coll.nWarps;
      eDescr_v6.coll.algo = eDescr->coll.algo;
      eDescr_v6.coll.proto = eDescr->coll.proto;
      eDescr_v6.coll.parentGroup = eDescr->coll.parentGroup;
    }
    break;
  case ncclProfileP2p:
    {
      eDescr_v6.p2p.func = eDescr->p2p.func;
      eDescr_v6.p2p.buff = eDescr->p2p.buff;
      eDescr_v6.p2p.datatype = eDescr->p2p.datatype;
      eDescr_v6.p2p.count = eDescr->p2p.count;
      eDescr_v6.p2p.peer = eDescr->p2p.peer;
      eDescr_v6.p2p.nChannels = eDescr->p2p.nChannels;
      eDescr_v6.p2p.parentGroup = eDescr->p2p.parentGroup;
    }
    break;
  case ncclProfileProxyOp:
    {
      eDescr_v6.proxyOp.pid = eDescr->proxyOp.pid;
      eDescr_v6.proxyOp.channelId = eDescr->proxyOp.channelId;
      eDescr_v6.proxyOp.peer = eDescr->proxyOp.peer;
      eDescr_v6.proxyOp.nSteps = eDescr->proxyOp.nSteps;
      eDescr_v6.proxyOp.chunkSize = eDescr->proxyOp.chunkSize;
      eDescr_v6.proxyOp.isSend = eDescr->proxyOp.isSend;
    }
    break;
  case ncclProfileProxyStep:
    {
      eDescr_v6.proxyStep.step = eDescr->proxyStep.step;
    }
    break;
  case ncclProfileKernelCh:
    {
      eDescr_v6.kernelCh.channelId = eDescr->kernelCh.channelId;
      eDescr_v6.kernelCh.pTimer = eDescr->kernelCh.pTimer;
    }
    break;
  case ncclProfileNetPlugin:
    {
      eDescr_v6.netPlugin.id = eDescr->netPlugin.id;
      eDescr_v6.netPlugin.data = eDescr->netPlugin.data;
    }
    break;
  case ncclProfileCeColl:
    {
      eDescr_v6.ceColl.seqNumber = eDescr->ceColl.seqNumber;
      eDescr_v6.ceColl.func = eDescr->ceColl.func;
      eDescr_v6.ceColl.sendBuff = eDescr->ceColl.sendBuff;
      eDescr_v6.ceColl.recvBuff = eDescr->ceColl.recvBuff;
      eDescr_v6.ceColl.count = eDescr->ceColl.count;
      eDescr_v6.ceColl.root = eDescr->ceColl.root;
      eDescr_v6.ceColl.datatype = eDescr->ceColl.datatype;
      eDescr_v6.ceColl.syncStrategy = eDescr->ceColl.syncStrategy;
      eDescr_v6.ceColl.intraBatchSync = eDescr->ceColl.intraBatchSync;
      eDescr_v6.ceColl.batchSize = eDescr->ceColl.batchSize;
      eDescr_v6.ceColl.numBatches = eDescr->ceColl.numBatches;
      eDescr_v6.ceColl.ceSeqNum = eDescr->ceColl.ceSeqNum;
      eDescr_v6.ceColl.stream = eDescr->ceColl.stream;
    }
    break;
  case ncclProfileCeSync:
    {
      eDescr_v6.ceCollSync.isComplete = eDescr->ceCollSync.isComplete;
      eDescr_v6.ceCollSync.nRanks = eDescr->ceCollSync.nRanks;
    }
    break;
  case ncclProfileCeBatch:
    {
      eDescr_v6.ceCollBatch.numOps = eDescr->ceCollBatch.numOps;
      eDescr_v6.ceCollBatch.totalBytes = eDescr->ceCollBatch.totalBytes;
      eDescr_v6.ceCollBatch.useIntraSync = eDescr->ceCollBatch.useIntraSync;
    }
    break;
  default:
    break;
  }
  return ncclProfiler_v6->startEvent(ctx, eHandle, &eDescr_v6);
}

static ncclResult_t ncclProfiler_recordEventState(void* eHandle, ncclProfilerEventState_t eState,
                                                  ncclProfilerEventStateArgs_t* eStateArgs) {
  // Drop v7-only state transitions.
  if (eState == ncclProfilerKernelPhaseStop) return ncclSuccess;

  ncclProfilerEventStateArgs_v6_t eStateArgs_v6 = {};
  if (eStateArgs) {
    switch (eState) {
    case ncclProfilerProxyStepSendGPUWait:
    case ncclProfilerProxyStepSendPeerWait_v4:
    case ncclProfilerProxyStepSendWait:
    case ncclProfilerProxyStepRecvWait:
    case ncclProfilerProxyStepRecvFlushWait:
    case ncclProfilerProxyStepRecvGPUWait:
      eStateArgs_v6.proxyStep.transSize = eStateArgs->proxyStep.transSize;
      break;
    case ncclProfilerProxyCtrlIdle:
    case ncclProfilerProxyCtrlActive:
    case ncclProfilerProxyCtrlSleep:
    case ncclProfilerProxyCtrlWakeup:
    case ncclProfilerProxyCtrlAppend:
    case ncclProfilerProxyCtrlAppendEnd:
      eStateArgs_v6.proxyCtrl.appendedProxyOps = eStateArgs->proxyCtrl.appendedProxyOps;
      break;
    case ncclProfilerNetPluginUpdate:
      eStateArgs_v6.netPlugin.data = eStateArgs->netPlugin.data;
      break;
    case ncclProfilerKernelChStop:
      eStateArgs_v6.kernelCh.pTimer = eStateArgs->kernelCh.pTimer;
      break;
    default:
      break;
    }
  }
  return ncclProfiler_v6->recordEventState(eHandle, (ncclProfilerEventState_v6_t)eState, &eStateArgs_v6);
}

static ncclResult_t ncclProfiler_init(void** ctx, uint64_t commId, int* eActivationMask, const char* commName,
                                      int nNodes, int nranks, int rank, ncclDebugLogger_t logfn) {
  ncclResult_t ret = ncclProfiler_v6->init(ctx, commId, eActivationMask, commName, nNodes, nranks, rank, logfn);
  // v6 plugins don't know about kernel phases; strip the bit so the core never
  // generates phase work for them.
  if (eActivationMask) {
    *eActivationMask &= ~ncclProfileKernelPhase;
  }
  ncclProfiler.startEvent = ncclProfiler_startEvent;
  ncclProfiler.recordEventState = ncclProfiler_recordEventState;
  ncclProfiler.stopEvent = ncclProfiler_v6->stopEvent;
  ncclProfiler.finalize = ncclProfiler_v6->finalize;
  return ret;
}

ncclProfiler_t* getNcclProfiler_v6(void* lib) {
  ncclProfiler_v6 = (ncclProfiler_v6_t*)ncclOsDlsym(lib, "ncclProfiler_v6");
  if (ncclProfiler_v6) {
    ncclProfiler.name = ncclProfiler_v6->name;
    ncclProfiler.init = ncclProfiler_init;
    INFO(NCCL_INIT, "PROFILER/Plugin: Loaded %s (v6)", ncclProfiler_v6->name);
    return &ncclProfiler;
  }
  return NULL;
}
