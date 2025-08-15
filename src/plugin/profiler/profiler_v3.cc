/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "comm.h"
#include "nccl_profiler.h"
#include "checks.h"
#include <dlfcn.h>

static ncclProfiler_t ncclProfiler;
static ncclProfiler_v3_t* ncclProfiler_v3;

static ncclResult_t ncclProfiler_startEvent(void* context, void** eHandle, ncclProfilerEventDescr_t* eDescr) {
  *eHandle = nullptr;
  ncclProfilerEventDescr_v3_t eDescr_v3 = { };
  eDescr_v3.type = eDescr->type;
  eDescr_v3.parentObj = eDescr->parentObj;
  eDescr_v3.rank = eDescr->rank;
  switch(eDescr->type) {
    case ncclProfileGroup: break;
    case ncclProfileColl: {
      eDescr_v3.coll.name = nullptr; // removed in v4
      eDescr_v3.coll.commHash = 0; // removed in v4
      eDescr_v3.parentObj = eDescr->coll.parentGroup; // Hierarchy changed in v5
      eDescr_v3.coll.seqNumber = eDescr->coll.seqNumber;
      eDescr_v3.coll.func = eDescr->coll.func;
      eDescr_v3.coll.sendBuff = eDescr->coll.sendBuff;
      eDescr_v3.coll.recvBuff = eDescr->coll.recvBuff;
      eDescr_v3.coll.count = eDescr->coll.count;
      eDescr_v3.coll.root = eDescr->coll.root;
      eDescr_v3.coll.datatype = eDescr->coll.datatype;
      eDescr_v3.coll.nMaxChannels = eDescr->coll.nChannels;
      eDescr_v3.coll.nWarps = eDescr->coll.nWarps;
      eDescr_v3.coll.algo = eDescr->coll.algo;
      eDescr_v3.coll.proto = eDescr->coll.proto;
    } break;
    case ncclProfileP2p: {
      eDescr_v3.p2p.name = nullptr; // removed in v4
      eDescr_v3.p2p.commHash = 0; // removed in v4
      eDescr_v3.parentObj = eDescr->p2p.parentGroup; // Hierarchy changed in v5
      eDescr_v3.p2p.func = eDescr->p2p.func;
      eDescr_v3.p2p.buff = eDescr->p2p.buff;
      eDescr_v3.p2p.count = eDescr->p2p.count;
      eDescr_v3.p2p.datatype = eDescr->p2p.datatype;
      eDescr_v3.p2p.peer = eDescr->p2p.peer;
    } break;
    case ncclProfileProxyOp: {
      eDescr_v3.proxyOp.pid = eDescr->proxyOp.pid;
      eDescr_v3.proxyOp.channelId = eDescr->proxyOp.channelId;
      eDescr_v3.proxyOp.peer = eDescr->proxyOp.peer;
      eDescr_v3.proxyOp.nSteps = eDescr->proxyOp.nSteps;
      eDescr_v3.proxyOp.chunkSize = eDescr->proxyOp.chunkSize;
      eDescr_v3.proxyOp.isSend = eDescr->proxyOp.isSend;
    } break;
    case ncclProfileProxyStep: {
      eDescr_v3.proxyStep.step = eDescr->proxyStep.step;
    } break;
    case ncclProfileProxyCtrl: break;
    case ncclProfileKernelCh: {
      eDescr_v3.kernelCh.channelId = eDescr->kernelCh.channelId;
    } break;
    case ncclProfileNetPlugin: {
      eDescr_v3.netPlugin.id = eDescr->netPlugin.id;
      eDescr_v3.netPlugin.data = eDescr->netPlugin.data;
    } break;
    default: return ncclSuccess;
  }
  return ncclProfiler_v3->startEvent(context, eHandle, &eDescr_v3);
}

static ncclResult_t ncclProfiler_recordEventState(void* eHandle, ncclProfilerEventState_t eState, ncclProfilerEventStateArgs_t* eStateArgs) {
  ncclProfilerEventStateArgs_v3_t args = { };
  switch (eState) {
    case ncclProfilerProxyCtrlIdle:
    case ncclProfilerProxyCtrlActive:
    case ncclProfilerProxyCtrlSleep:
    case ncclProfilerProxyCtrlWakeup:
    case ncclProfilerProxyCtrlAppend:
    case ncclProfilerProxyCtrlAppendEnd:
      args.proxyCtrl.appendedProxyOps = eStateArgs->proxyCtrl.appendedProxyOps;
      break;
    case ncclProfilerProxyStepSendGPUWait:
    case ncclProfilerProxyStepSendWait:
    case ncclProfilerProxyStepRecvWait:
    case ncclProfilerProxyStepRecvFlushWait:
    case ncclProfilerProxyStepRecvGPUWait:
      break;
    default: return ncclSuccess;
  }
  return ncclProfiler_v3->recordEventState(eHandle, eState, &args);
}

static ncclResult_t ncclProfiler_init(void** context,
    uint64_t commId __attribute__((unused)),
    int* eActivationMask __attribute__((unused)),
    const char* commName __attribute__((unused)),
    int nNodes __attribute__((unused)),
    int nranks __attribute__((unused)),
    int rank __attribute__((unused)),
    ncclDebugLogger_t logfn __attribute__((unused))
  ) {
  NCCLCHECK(ncclProfiler_v3->init(context, eActivationMask));
  ncclProfiler.startEvent = ncclProfiler_startEvent;
  ncclProfiler.stopEvent = ncclProfiler_v3->stopEvent;
  ncclProfiler.recordEventState = ncclProfiler_recordEventState;
  ncclProfiler.finalize = ncclProfiler_v3->finalize;
  return ncclSuccess;
}

ncclProfiler_t* getNcclProfiler_v3(void* lib) {
  ncclProfiler_v3 = (ncclProfiler_v3_t*)dlsym(lib, "ncclProfiler_v3");
  if (ncclProfiler_v3) {
    ncclProfiler.name = ncclProfiler_v3->name;
    ncclProfiler.init = ncclProfiler_init;
    INFO(NCCL_INIT, "PROFILER/Plugin: Loaded %s (v3)", ncclProfiler_v3->name);
    return &ncclProfiler;
  }
  return NULL;
}
