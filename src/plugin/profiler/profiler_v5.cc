/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "comm.h"
#include "nccl_profiler.h"
#include "checks.h"
#include <dlfcn.h>

static ncclProfiler_v5_t* ncclProfiler_v5;
static ncclProfiler_t ncclProfiler;

static ncclResult_t ncclProfiler_startEvent(void* ctx, void** eHandle, ncclProfilerEventDescr_t* eDescr) {
  ncclProfilerEventDescr_v5_t eDescr_v5;
  eDescr_v5.type = eDescr->type;
  eDescr_v5.parentObj = eDescr->parentObj;
  eDescr_v5.rank = eDescr->rank;

  switch(eDescr->type) {
  case ncclProfileGroup: break;
  case ncclProfileGroupApi: {
    eDescr_v5.groupApi.graphCaptured = eDescr->groupApi.graphCaptured;
    eDescr_v5.groupApi.groupDepth = eDescr->groupApi.groupDepth;
  } break;
  case ncclProfileCollApi: {
    eDescr_v5.collApi.func = eDescr->collApi.func;
    eDescr_v5.collApi.count = eDescr->collApi.count;
    eDescr_v5.collApi.datatype = eDescr->collApi.datatype;
    eDescr_v5.collApi.root = eDescr->collApi.root;
    eDescr_v5.collApi.stream = eDescr->collApi.stream;
    eDescr_v5.collApi.graphCaptured = eDescr->collApi.graphCaptured;
  } break;
  case ncclProfileP2pApi: {
    eDescr_v5.p2pApi.func = eDescr->p2pApi.func;
    eDescr_v5.p2pApi.count = eDescr->p2pApi.count;
    eDescr_v5.p2pApi.datatype = eDescr->p2pApi.datatype;
    eDescr_v5.p2pApi.stream = eDescr->p2pApi.stream;
    eDescr_v5.p2pApi.graphCaptured = eDescr->p2pApi.graphCaptured;
  } break;
  case ncclProfileKernelLaunch: {
    eDescr_v5.kernelLaunch.stream = eDescr->kernelLaunch.stream;
  } break;
  case ncclProfileColl: {
    eDescr_v5.coll.seqNumber = eDescr->coll.seqNumber;
    eDescr_v5.coll.func = eDescr->coll.func;
    eDescr_v5.coll.sendBuff = eDescr->coll.sendBuff;
    eDescr_v5.coll.recvBuff = eDescr->coll.recvBuff;
    eDescr_v5.coll.count = eDescr->coll.count;
    eDescr_v5.coll.root = eDescr->coll.root;
    eDescr_v5.coll.datatype = eDescr->coll.datatype;
    eDescr_v5.coll.nChannels = eDescr->coll.nChannels;
    eDescr_v5.coll.nWarps = eDescr->coll.nWarps;
    eDescr_v5.coll.algo = eDescr->coll.algo;
    eDescr_v5.coll.proto = eDescr->coll.proto;
    eDescr_v5.coll.parentGroup = eDescr->coll.parentGroup;
  } break;
  case ncclProfileP2p: {
    eDescr_v5.p2p.func = eDescr->p2p.func;
    eDescr_v5.p2p.buff = eDescr->p2p.buff;
    eDescr_v5.p2p.datatype = eDescr->p2p.datatype;
    eDescr_v5.p2p.count = eDescr->p2p.count;
    eDescr_v5.p2p.peer = eDescr->p2p.peer;
    eDescr_v5.p2p.nChannels = eDescr->p2p.nChannels;
    eDescr_v5.p2p.parentGroup = eDescr->p2p.parentGroup;
  } break;
  case ncclProfileProxyOp: {
    eDescr_v5.proxyOp.pid = eDescr->proxyOp.pid;
    eDescr_v5.proxyOp.channelId = eDescr->proxyOp.channelId;
    eDescr_v5.proxyOp.peer = eDescr->proxyOp.peer;
    eDescr_v5.proxyOp.nSteps = eDescr->proxyOp.nSteps;
    eDescr_v5.proxyOp.chunkSize = eDescr->proxyOp.chunkSize;
    eDescr_v5.proxyOp.isSend = eDescr->proxyOp.isSend;
  } break;
  case ncclProfileProxyStep: {
    eDescr_v5.proxyStep.step = eDescr->proxyStep.step;
  } break;
  case ncclProfileProxyCtrl: break;
  case ncclProfileKernelCh: {
    eDescr_v5.kernelCh.channelId = eDescr->kernelCh.channelId;
    eDescr_v5.kernelCh.pTimer = eDescr->kernelCh.pTimer;
  } break;
  case ncclProfileNetPlugin: {
    eDescr_v5.netPlugin.id = eDescr->netPlugin.id;
    eDescr_v5.netPlugin.data = eDescr->netPlugin.data;
  } break;
    // v6 CE events - not supported in v5, discard them
  case ncclProfileCeColl:
  case ncclProfileCeSync:
  case ncclProfileCeBatch:
    *eHandle = NULL;
    return ncclSuccess;
  default:
    return ncclSuccess;
  }

  return ncclProfiler_v5->startEvent(ctx, eHandle, &eDescr_v5);
}

static ncclResult_t ncclProfiler_recordEventState(void* eHandle, ncclProfilerEventState_t eState, ncclProfilerEventStateArgs_t* eStateArgs) {
  // Discard v6-specific CE event states
  switch(eState) {
  case ncclProfilerCeCollStart:
  case ncclProfilerCeCollComplete:
  case ncclProfilerCeSyncStart:
  case ncclProfilerCeSyncComplete:
  case ncclProfilerCeBatchStart:
  case ncclProfilerCeBatchComplete:
    return ncclSuccess;
  default:
    break;
  }

  // v5 uses the same state args structure as v6
  ncclProfilerEventStateArgs_v5_t eStateArgs_v5;
  switch(eState) {
  case ncclProfilerProxyStepSendGPUWait:
  case ncclProfilerProxyStepSendPeerWait_v4:
  case ncclProfilerProxyStepSendWait:
  case ncclProfilerProxyStepRecvWait:
  case ncclProfilerProxyStepRecvFlushWait:
  case ncclProfilerProxyStepRecvGPUWait:
    eStateArgs_v5.proxyStep.transSize = eStateArgs->proxyStep.transSize;
    break;
  case ncclProfilerProxyCtrlIdle:
  case ncclProfilerProxyCtrlActive:
  case ncclProfilerProxyCtrlSleep:
  case ncclProfilerProxyCtrlWakeup:
  case ncclProfilerProxyCtrlAppend:
  case ncclProfilerProxyCtrlAppendEnd:
    eStateArgs_v5.proxyCtrl.appendedProxyOps = eStateArgs->proxyCtrl.appendedProxyOps;
    break;
  case ncclProfilerNetPluginUpdate:
    eStateArgs_v5.netPlugin.data = eStateArgs->netPlugin.data;
    break;
  case ncclProfilerKernelChStop:
    eStateArgs_v5.kernelCh.pTimer = eStateArgs->kernelCh.pTimer;
    break;
  case ncclProfilerProxyOpInProgress_v4:
  case ncclProfilerGroupStartApiStop:
  case ncclProfilerGroupEndApiStart:
    break;
  default:
    return ncclSuccess;
  }

  return ncclProfiler_v5->recordEventState(eHandle, (ncclProfilerEventState_v5_t)eState, &eStateArgs_v5);
}

static ncclResult_t ncclProfiler_init(void** ctx, uint64_t commId, int* eActivationMask, const char* commName, int nNodes, int nRanks, int rank, ncclDebugLogger_t logfn) {
  NCCLCHECK(ncclProfiler_v5->init(ctx, commId, eActivationMask, commName, nNodes, nRanks, rank, logfn));

  // Clear v6 CE event bits from the activation mask since v5 doesn't support them
  if (eActivationMask) {
    *eActivationMask &= ~(ncclProfileCeColl | ncclProfileCeSync | ncclProfileCeBatch);
  }

  ncclProfiler.startEvent = ncclProfiler_startEvent;
  ncclProfiler.recordEventState = ncclProfiler_recordEventState;
  ncclProfiler.stopEvent = ncclProfiler_v5->stopEvent;
  ncclProfiler.finalize = ncclProfiler_v5->finalize;
  return ncclSuccess;
}

ncclProfiler_t* getNcclProfiler_v5(void* lib) {
  ncclProfiler_v5 = (ncclProfiler_v5_t*)dlsym(lib, "ncclProfiler_v5");
  if (ncclProfiler_v5) {
    ncclProfiler.name = ncclProfiler_v5->name;
    ncclProfiler.init = ncclProfiler_init;
    INFO(NCCL_INIT, "PROFILER/Plugin: Loaded %s (v5)", ncclProfiler_v5->name);
    return &ncclProfiler;
  }
  return NULL;
}
