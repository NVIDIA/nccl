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
static ncclProfiler_v1_t* ncclProfiler_v1;

static uint8_t ncclStringToFunc(const char* func) {
  if (0 == strcmp(func, "AllGather")) return ncclFuncAllGather;
  if (0 == strcmp(func, "AllReduce")) return ncclFuncAllReduce;
  if (0 == strcmp(func, "Broadcast")) return ncclFuncBroadcast;
  if (0 == strcmp(func, "Recv")) return ncclFuncRecv;
  if (0 == strcmp(func, "Reduce")) return ncclFuncReduce;
  if (0 == strcmp(func, "ReduceScatter")) return ncclFuncReduceScatter;
  if (0 == strcmp(func, "SendRecv")) return ncclFuncSendRecv;
  return ncclFuncSend;
}

static uint8_t ncclStringToAlgo(const char* algo) {
  if (0 == strcmp(algo, "TREE")) return NCCL_ALGO_TREE;
  if (0 == strcmp(algo, "RING")) return NCCL_ALGO_RING;
  if (0 == strcmp(algo, "COLLNET_DIRECT")) return NCCL_ALGO_COLLNET_DIRECT;
  if (0 == strcmp(algo, "COLLNET_CHAIN")) return NCCL_ALGO_COLLNET_CHAIN;
  if (0 == strcmp(algo, "NVLS")) return NCCL_ALGO_NVLS;
  if (0 == strcmp(algo, "NVLS_TREE")) return NCCL_ALGO_NVLS_TREE;
  return NCCL_ALGO_PAT;
}

static uint8_t ncclStringToProto(const char* proto) {
  if (0 == strcmp(proto, "LL")) return NCCL_PROTO_LL;
  if (0 == strcmp(proto, "LL128")) return NCCL_PROTO_LL128;
  return NCCL_PROTO_SIMPLE;
}

static uint8_t ncclStringToDatatype(const char* dt) {
  if (0 == strcmp(dt, "ncclInt8")) return ncclInt8;
  if (0 == strcmp(dt, "ncclInt32")) return ncclInt32;
  if (0 == strcmp(dt, "ncclUint32")) return ncclUint32;
  if (0 == strcmp(dt, "ncclInt64")) return ncclInt64;
  if (0 == strcmp(dt, "ncclUint64")) return ncclUint64;
  if (0 == strcmp(dt, "ncclFloat16")) return ncclFloat16;
  if (0 == strcmp(dt, "ncclFloat32")) return ncclFloat32;
#if defined(__CUDA_BF16_TYPES_EXIST__)
  if (0 == strcmp(dt, "ncclBfloat16")) return ncclBfloat16;
#endif
  return ncclFloat64;
}

static ncclResult_t ncclProfiler_startEvent(void* context, void** eHandle, ncclProfilerEventDescr_t* eDescr) {
  *eHandle = NULL;
  ncclProfilerEventDescr_v1_t eDescr_v1 = { 0 };
  eDescr_v1.type = eDescr->type;
  eDescr_v1.parentObj = eDescr->parentObj;
  eDescr_v1.rank = eDescr->rank;
  switch(eDescr->type) {
    case ncclProfileGroup: break;
    case ncclProfileColl: {
      eDescr_v1.coll.name = nullptr; // removed in v4
      eDescr_v1.coll.commHash = 0; // removed in v4
      eDescr_v1.parentObj = eDescr->coll.parentGroup; // Hierarchy changed in v5
      eDescr_v1.coll.seqNumber = eDescr->coll.seqNumber;
      eDescr_v1.coll.func = ncclStringToFunc(eDescr->coll.func);
      eDescr_v1.coll.sendBuff = eDescr->coll.sendBuff;
      eDescr_v1.coll.recvBuff = eDescr->coll.recvBuff;
      eDescr_v1.coll.count = eDescr->coll.count;
      eDescr_v1.coll.root = eDescr->coll.root;
      eDescr_v1.coll.datatype = ncclStringToDatatype(eDescr->coll.datatype);
      eDescr_v1.coll.op = 0; // removed in v2
      eDescr_v1.coll.trafficBytes = 0; // removed in v3
      eDescr_v1.coll.nMaxChannels = eDescr->coll.nChannels;
      eDescr_v1.coll.nWarps = eDescr->coll.nWarps;
      eDescr_v1.coll.algo = ncclStringToAlgo(eDescr->coll.algo);
      eDescr_v1.coll.proto = ncclStringToProto(eDescr->coll.proto);
    } break;
    case ncclProfileP2p: {
      eDescr_v1.p2p.name = nullptr; // removed in v4
      eDescr_v1.p2p.commHash = 0; // removed in v4
      eDescr_v1.parentObj = eDescr->p2p.parentGroup; // Hierarchy changed in v5
      eDescr_v1.p2p.func = ncclStringToFunc(eDescr->p2p.func);
      eDescr_v1.p2p.buff = eDescr->p2p.buff;
      eDescr_v1.p2p.count = eDescr->p2p.count;
      eDescr_v1.p2p.datatype = ncclStringToDatatype(eDescr->p2p.datatype);
      eDescr_v1.p2p.peer = eDescr->p2p.peer;
    } break;
    case ncclProfileProxyOp: {
      eDescr_v1.proxyOp.pid = eDescr->proxyOp.pid;
      eDescr_v1.proxyOp.channelId = eDescr->proxyOp.channelId;
      eDescr_v1.proxyOp.peer = eDescr->proxyOp.peer;
      eDescr_v1.proxyOp.nSteps = eDescr->proxyOp.nSteps;
      eDescr_v1.proxyOp.chunkSize = eDescr->proxyOp.chunkSize;
      eDescr_v1.proxyOp.isSend = eDescr->proxyOp.isSend;
    } break;
    case ncclProfileProxyStep: {
      eDescr_v1.proxyStep.step = eDescr->proxyStep.step;
    } break;
    case ncclProfileProxyCtrl: break;
    default: return ncclSuccess;
  }
  return ncclProfiler_v1->startEvent(context, eHandle, &eDescr_v1);
}

static ncclResult_t ncclProfiler_recordEventState(void* eHandle, ncclProfilerEventState_t eState, ncclProfilerEventStateArgs_t* eStateArgs) {
  ncclProfilerEventStateArgs_v1_t args = { };
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
  return ncclProfiler_v1->recordEventState(eHandle, eState, &args);
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
  NCCLCHECK(ncclProfiler_v1->init(context, eActivationMask));
  ncclProfiler.startEvent = ncclProfiler_startEvent;
  ncclProfiler.stopEvent = ncclProfiler_v1->stopEvent;
  ncclProfiler.recordEventState = ncclProfiler_recordEventState;
  ncclProfiler.finalize = ncclProfiler_v1->finalize;
  return ncclSuccess;
}

ncclProfiler_t* getNcclProfiler_v1(void* lib) {
  ncclProfiler_v1 = (ncclProfiler_v1_t*)dlsym(lib, "ncclProfiler_v1");
  if (ncclProfiler_v1) {
    ncclProfiler.name = ncclProfiler_v1->name;
    ncclProfiler.init = ncclProfiler_init;
    INFO(NCCL_INIT, "PROFILER/Plugin: Loaded %s (v1)", ncclProfiler_v1->name);
    return &ncclProfiler;
  }
  return NULL;
}
