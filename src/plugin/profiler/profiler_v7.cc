/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "comm.h"
#include "nccl_profiler.h"
#include "plugin/profiler/profiler_v7.h"
#include "checks.h"
#include "os.h"

// Adapter: the internal interface is v8, which can stamp NCCL timestamps on
// every event. v7 plugins predate that, so this adapter forwards the v8 calls
// to a v7 plugin by dropping the timestamp fields on startEvent, the stop
// descriptor argument on stopEvent, and the config argument on init -- with no
// config to write, the plugin leaves ncclSourcedTimings at its default of off
// and keeps computing its own timestamps internally, exactly as before.
// Everything else -- including the v7 userTag, kernelVariant/isSymColl and the
// kernel phase sub-event -- is carried through unchanged, since the descriptor
// union is identical in the two versions.

static ncclProfiler_v7_t* ncclProfiler_v7;
static ncclProfiler_t ncclProfiler;

static ncclResult_t ncclProfiler_startEvent(void* ctx, void** eHandle, ncclProfilerEventDescr_t* eDescr) {
  ncclProfilerEventDescr_v7_t eDescr_v7 = {};
  eDescr_v7.type = eDescr->type;
  eDescr_v7.parentObj = eDescr->parentObj;
  eDescr_v7.rank = eDescr->rank;
  // v8-only top-level timestamp fields (tsGpu/tsCpuMono/tsCpuReal) are dropped.

  switch (eDescr->type) {
  case ncclProfileGroup:
    break;
  case ncclProfileGroupApi:
    eDescr_v7.groupApi.graphCaptured = eDescr->groupApi.graphCaptured;
    eDescr_v7.groupApi.groupDepth = eDescr->groupApi.groupDepth;
    break;
  case ncclProfileCollApi:
    eDescr_v7.collApi.func = eDescr->collApi.func;
    eDescr_v7.collApi.count = eDescr->collApi.count;
    eDescr_v7.collApi.datatype = eDescr->collApi.datatype;
    eDescr_v7.collApi.root = eDescr->collApi.root;
    eDescr_v7.collApi.stream = eDescr->collApi.stream;
    eDescr_v7.collApi.graphCaptured = eDescr->collApi.graphCaptured;
    eDescr_v7.collApi.userTag = eDescr->collApi.userTag;
    break;
  case ncclProfileP2pApi:
    eDescr_v7.p2pApi.func = eDescr->p2pApi.func;
    eDescr_v7.p2pApi.count = eDescr->p2pApi.count;
    eDescr_v7.p2pApi.datatype = eDescr->p2pApi.datatype;
    eDescr_v7.p2pApi.stream = eDescr->p2pApi.stream;
    eDescr_v7.p2pApi.graphCaptured = eDescr->p2pApi.graphCaptured;
    eDescr_v7.p2pApi.userTag = eDescr->p2pApi.userTag;
    break;
  case ncclProfileKernelLaunch:
    eDescr_v7.kernelLaunch.stream = eDescr->kernelLaunch.stream;
    break;
  case ncclProfileColl:
    eDescr_v7.coll.seqNumber = eDescr->coll.seqNumber;
    eDescr_v7.coll.func = eDescr->coll.func;
    eDescr_v7.coll.sendBuff = eDescr->coll.sendBuff;
    eDescr_v7.coll.recvBuff = eDescr->coll.recvBuff;
    eDescr_v7.coll.count = eDescr->coll.count;
    eDescr_v7.coll.root = eDescr->coll.root;
    eDescr_v7.coll.datatype = eDescr->coll.datatype;
    eDescr_v7.coll.nChannels = eDescr->coll.nChannels;
    eDescr_v7.coll.nWarps = eDescr->coll.nWarps;
    eDescr_v7.coll.algo = eDescr->coll.algo;
    eDescr_v7.coll.proto = eDescr->coll.proto;
    eDescr_v7.coll.parentGroup = eDescr->coll.parentGroup;
    eDescr_v7.coll.kernelVariant = eDescr->coll.kernelVariant;
    eDescr_v7.coll.isSymColl = eDescr->coll.isSymColl;
    eDescr_v7.coll.userTag = eDescr->coll.userTag;
    break;
  case ncclProfileP2p:
    eDescr_v7.p2p.func = eDescr->p2p.func;
    eDescr_v7.p2p.buff = eDescr->p2p.buff;
    eDescr_v7.p2p.datatype = eDescr->p2p.datatype;
    eDescr_v7.p2p.count = eDescr->p2p.count;
    eDescr_v7.p2p.peer = eDescr->p2p.peer;
    eDescr_v7.p2p.nChannels = eDescr->p2p.nChannels;
    eDescr_v7.p2p.parentGroup = eDescr->p2p.parentGroup;
    eDescr_v7.p2p.userTag = eDescr->p2p.userTag;
    break;
  case ncclProfileProxyOp:
    eDescr_v7.proxyOp.pid = eDescr->proxyOp.pid;
    eDescr_v7.proxyOp.channelId = eDescr->proxyOp.channelId;
    eDescr_v7.proxyOp.peer = eDescr->proxyOp.peer;
    eDescr_v7.proxyOp.nSteps = eDescr->proxyOp.nSteps;
    eDescr_v7.proxyOp.chunkSize = eDescr->proxyOp.chunkSize;
    eDescr_v7.proxyOp.isSend = eDescr->proxyOp.isSend;
    break;
  case ncclProfileProxyStep:
    eDescr_v7.proxyStep.step = eDescr->proxyStep.step;
    break;
  case ncclProfileProxyCtrl:
    break;
  case ncclProfileKernelCh:
    eDescr_v7.kernelCh.channelId = eDescr->kernelCh.channelId;
    eDescr_v7.kernelCh.pTimer = eDescr->kernelCh.pTimer;
    break;
  case ncclProfileNetPlugin:
    eDescr_v7.netPlugin.id = eDescr->netPlugin.id;
    eDescr_v7.netPlugin.data = eDescr->netPlugin.data;
    break;
  case ncclProfileCeColl:
    eDescr_v7.ceColl.seqNumber = eDescr->ceColl.seqNumber;
    eDescr_v7.ceColl.func = eDescr->ceColl.func;
    eDescr_v7.ceColl.sendBuff = eDescr->ceColl.sendBuff;
    eDescr_v7.ceColl.recvBuff = eDescr->ceColl.recvBuff;
    eDescr_v7.ceColl.count = eDescr->ceColl.count;
    eDescr_v7.ceColl.root = eDescr->ceColl.root;
    eDescr_v7.ceColl.datatype = eDescr->ceColl.datatype;
    eDescr_v7.ceColl.syncStrategy = eDescr->ceColl.syncStrategy;
    eDescr_v7.ceColl.intraBatchSync = eDescr->ceColl.intraBatchSync;
    eDescr_v7.ceColl.batchSize = eDescr->ceColl.batchSize;
    eDescr_v7.ceColl.numBatches = eDescr->ceColl.numBatches;
    eDescr_v7.ceColl.ceSeqNum = eDescr->ceColl.ceSeqNum;
    eDescr_v7.ceColl.stream = eDescr->ceColl.stream;
    eDescr_v7.ceColl.userTag = eDescr->ceColl.userTag;
    break;
  case ncclProfileCeSync:
    eDescr_v7.ceCollSync.isComplete = eDescr->ceCollSync.isComplete;
    eDescr_v7.ceCollSync.nRanks = eDescr->ceCollSync.nRanks;
    break;
  case ncclProfileCeBatch:
    eDescr_v7.ceCollBatch.numOps = eDescr->ceCollBatch.numOps;
    eDescr_v7.ceCollBatch.totalBytes = eDescr->ceCollBatch.totalBytes;
    eDescr_v7.ceCollBatch.useIntraSync = eDescr->ceCollBatch.useIntraSync;
    break;
  case ncclProfileKernelPhase:
    eDescr_v7.kernelPhase.channelId = eDescr->kernelPhase.channelId;
    eDescr_v7.kernelPhase.phaseId = eDescr->kernelPhase.phaseId;
    eDescr_v7.kernelPhase.phaseName = eDescr->kernelPhase.phaseName;
    eDescr_v7.kernelPhase.pTimer = eDescr->kernelPhase.pTimer;
    break;
  default:
    return ncclSuccess;
  }

  return ncclProfiler_v7->startEvent(ctx, eHandle, &eDescr_v7);
}

static ncclResult_t ncclProfiler_stopEvent(void* eHandle, ncclProfilerEventStopDescr_t* eStopDescr) {
  // v7 plugins don't take stop timestamps; drop the stop descriptor.
  (void)eStopDescr;
  return ncclProfiler_v7->stopEvent(eHandle);
}

static ncclResult_t ncclProfiler_recordEventState(void* eHandle, ncclProfilerEventState_t eState,
                                                  ncclProfilerEventStateArgs_t* eStateArgs) {
  // v8 state args are identical to v7 -- forward unchanged.
  return ncclProfiler_v7->recordEventState(eHandle, (ncclProfilerEventState_v7_t)eState,
                                           (ncclProfilerEventStateArgs_v7_t*)eStateArgs);
}

static ncclResult_t ncclProfiler_init(void** ctx, uint64_t commId, int* eActivationMask, const char* commName,
                                      int nNodes, int nRanks, int rank, ncclDebugLogger_t logfn,
                                      ncclProfilerConfig_t* config __attribute__((unused))) {
  NCCLCHECK(ncclProfiler_v7->init(ctx, commId, eActivationMask, commName, nNodes, nRanks, rank, logfn));
  ncclProfiler.startEvent = ncclProfiler_startEvent;
  ncclProfiler.stopEvent = ncclProfiler_stopEvent;
  ncclProfiler.recordEventState = ncclProfiler_recordEventState;
  ncclProfiler.finalize = ncclProfiler_v7->finalize;
  return ncclSuccess;
}

ncclProfiler_t* getNcclProfiler_v7(void* lib) {
  ncclProfiler_v7 = (ncclProfiler_v7_t*)ncclOsDlsym(lib, "ncclProfiler_v7");
  if (ncclProfiler_v7) {
    ncclProfiler.name = ncclProfiler_v7->name;
    ncclProfiler.init = ncclProfiler_init;
    INFO(NCCL_INIT, "PROFILER/Plugin: Loaded %s (v7)", ncclProfiler_v7->name);
    return &ncclProfiler;
  }
  return NULL;
}
