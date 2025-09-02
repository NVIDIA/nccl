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
static ncclProfiler_v2_t* ncclProfiler_v2;

static ncclResult_t ncclProfiler_startEvent(void* context, void** eHandle, ncclProfilerEventDescr_t* eDescr) {
  *eHandle = nullptr;
  ncclProfilerEventDescr_v2_t eDescr_v2 = { };
  eDescr_v2.type = eDescr->type;
  eDescr_v2.parentObj = eDescr->parentObj;
  eDescr_v2.rank = eDescr->rank;
  switch(eDescr->type) {
    case ncclProfileGroup: break;
    case ncclProfileColl: {
      eDescr_v2.parentObj = eDescr->coll.parentGroup; // Hierarchy changed in v5
      eDescr_v2.coll.name = nullptr; // removed in v4
      eDescr_v2.coll.commHash = 0; // removed in v4
      eDescr_v2.coll.seqNumber = eDescr->coll.seqNumber;
      eDescr_v2.coll.func = eDescr->coll.func;
      eDescr_v2.coll.sendBuff = eDescr->coll.sendBuff;
      eDescr_v2.coll.recvBuff = eDescr->coll.recvBuff;
      eDescr_v2.coll.count = eDescr->coll.count;
      eDescr_v2.coll.root = eDescr->coll.root;
      eDescr_v2.coll.datatype = eDescr->coll.datatype;
      eDescr_v2.coll.trafficBytes = 0; // removed in v3
      eDescr_v2.coll.nMaxChannels = eDescr->coll.nChannels;
      eDescr_v2.coll.nWarps = eDescr->coll.nWarps;
      eDescr_v2.coll.algo = eDescr->coll.algo;
      eDescr_v2.coll.proto = eDescr->coll.proto;
    } break;
    case ncclProfileP2p: {
      eDescr_v2.p2p.name = nullptr; // removed in v4
      eDescr_v2.p2p.commHash = 0; // removed in v4
      eDescr_v2.parentObj = eDescr->p2p.parentGroup; // Hierarchy changed in v5
      eDescr_v2.p2p.func = eDescr->p2p.func;
      eDescr_v2.p2p.buff = eDescr->p2p.buff;
      eDescr_v2.p2p.count = eDescr->p2p.count;
      eDescr_v2.p2p.datatype = eDescr->p2p.datatype;
      eDescr_v2.p2p.peer = eDescr->p2p.peer;
    } break;
    case ncclProfileProxyOp: {
      eDescr_v2.proxyOp.pid = eDescr->proxyOp.pid;
      eDescr_v2.proxyOp.channelId = eDescr->proxyOp.channelId;
      eDescr_v2.proxyOp.peer = eDescr->proxyOp.peer;
      eDescr_v2.proxyOp.nSteps = eDescr->proxyOp.nSteps;
      eDescr_v2.proxyOp.chunkSize = eDescr->proxyOp.chunkSize;
      eDescr_v2.proxyOp.isSend = eDescr->proxyOp.isSend;
    } break;
    case ncclProfileProxyStep: {
      eDescr_v2.proxyStep.step = eDescr->proxyStep.step;
    } break;
    case ncclProfileProxyCtrl: break;
    default: return ncclSuccess;
  }
  return ncclProfiler_v2->startEvent(context, eHandle, &eDescr_v2);
}

static ncclResult_t ncclProfiler_recordEventState(void* eHandle, ncclProfilerEventState_t eState, ncclProfilerEventStateArgs_t* eStateArgs) {
  ncclProfilerEventStateArgs_v2_t args = { };
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
  return ncclProfiler_v2->recordEventState(eHandle, eState, &args);
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
  NCCLCHECK(ncclProfiler_v2->init(context, eActivationMask));
  ncclProfiler.startEvent = ncclProfiler_startEvent;
  ncclProfiler.stopEvent = ncclProfiler_v2->stopEvent;
  ncclProfiler.recordEventState = ncclProfiler_recordEventState;
  ncclProfiler.finalize = ncclProfiler_v2->finalize;
  return ncclSuccess;
}

ncclProfiler_t* getNcclProfiler_v2(void* lib) {
  ncclProfiler_v2 = (ncclProfiler_v2_t*)dlsym(lib, "ncclProfiler_v2");
  if (ncclProfiler_v2) {
    ncclProfiler.name = ncclProfiler_v2->name;
    ncclProfiler.init = ncclProfiler_init;
    INFO(NCCL_INIT, "PROFILER/Plugin: Loaded %s (v2)", ncclProfiler_v2->name);
    return &ncclProfiler;
  }
  return NULL;
}
