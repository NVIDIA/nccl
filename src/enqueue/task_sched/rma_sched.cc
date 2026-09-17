/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include <thread>

#include "comm.h"
#include "alloc.h"
#include "checks.h"
#include "compiler.h"
#include "profiler.h"
#include "strongstream.h"
#include "cudawrap.h"
#include "ce_coll.h"
#include "dev_runtime.h"
#include "gdrwrap.h"
#include "rma/rma.h"
#include "rma/rma_proxy.h"
#include "enqueue/task_sched.h"


struct ncclRmaSchedCtxPlan {
  struct ncclIntruQueue<struct ncclTaskTuningInfo, &ncclTaskTuningInfo::next> waitSignalCE;
  struct ncclIntruQueue<struct ncclTaskTuningInfo, &ncclTaskTuningInfo::next> waitSignalProxy;
  struct ncclIntruQueue<struct ncclTaskTuningInfo, &ncclTaskTuningInfo::next> putSignalCE;
  struct ncclIntruQueue<struct ncclTaskTuningInfo, &ncclTaskTuningInfo::next> putSignalProxy;
};

struct ncclRmaSchedStreamPlan {
  struct ncclRmaSchedStreamPlan* next;
  cudaStream_t stream;
  bool captured;
  struct ncclCudaGraph graph;
  struct ncclRmaSchedCtxPlan* ctxPlans;
};

struct ncclRmaSchedPlan {
  struct ncclRmaSchedStreamPlan* noncapturedStreamPlans;
  struct ncclRmaSchedStreamPlan* capturedStreamPlans;
};

static bool isLsaAccessible(struct ncclComm* comm, int rank) {
  for (int i = 0; i < comm->devrState.lsaTeamSize; i++) {
    if (comm->devrState.lsaTeamRankList[i] == rank) {
      return true;
    }
  }
  return false;
}

static int rmaTuningInfoCtx(struct ncclTaskTuningInfo* tInfo) {
  struct ncclRawTask* raw = tInfo->raw;
  switch (raw->rma.func) {
  case ncclFuncPutSignal:
    return raw->rma.rmaOp.putSignal.ctx;
  case ncclFuncSignal:
    return raw->rma.rmaOp.signal.ctx;
  case ncclFuncWaitSignal:
    return raw->rma.rmaOp.waitSignal.signalDesc.ctx;
  default:
    return 0;
  }
}

static ncclFunc_t rmaTuningInfoFunc(struct ncclTaskTuningInfo* tInfo) {
  return tInfo->raw->rma.func;
}

static int rmaTuningInfoPeer(struct ncclTaskTuningInfo* tInfo) {
  struct ncclRawTask* raw = tInfo->raw;
  switch (raw->rma.func) {
  case ncclFuncPutSignal:
    return raw->rma.rmaOp.putSignal.peer;
  case ncclFuncSignal:
    return raw->rma.rmaOp.signal.peer;
  case ncclFuncWaitSignal:
    return raw->rma.rmaOp.waitSignal.signalDesc.peer;
  default:
    return -1;
  }
}

static cudaStream_t rmaTuningInfoStream(struct ncclTaskTuningInfo* tInfo) {
  struct ncclRawTask* raw = tInfo->raw;
  switch (raw->rma.func) {
  case ncclFuncPutSignal:
    return raw->rma.rmaOp.putSignal.stream;
  case ncclFuncSignal:
    return raw->rma.rmaOp.signal.stream;
  case ncclFuncWaitSignal:
    return raw->rma.rmaOp.waitSignal.stream;
  default:
    return nullptr;
  }
}

static ncclResult_t rmaSchedInitStreamPlan(
    struct ncclComm* comm, struct ncclRmaSchedStreamPlan* streamPlan
  ) {
  int nRmaCtx = comm->config.numRmaCtx;
  streamPlan->ctxPlans = ncclMemoryStackAlloc<struct ncclRmaSchedCtxPlan>(&comm->memScoped, nRmaCtx);
  for (int ctx = 0; ctx < nRmaCtx; ctx++) {
    ncclIntruQueueConstruct(&streamPlan->ctxPlans[ctx].waitSignalCE);
    ncclIntruQueueConstruct(&streamPlan->ctxPlans[ctx].waitSignalProxy);
    ncclIntruQueueConstruct(&streamPlan->ctxPlans[ctx].putSignalCE);
    ncclIntruQueueConstruct(&streamPlan->ctxPlans[ctx].putSignalProxy);
  }
  return ncclSuccess;
}

static struct ncclRmaSchedStreamPlan* rmaSchedFindStreamPlan(
    struct ncclRmaSchedStreamPlan* head, cudaStream_t stream
  ) {
  for (struct ncclRmaSchedStreamPlan* plan = head; plan != nullptr; plan = plan->next) {
    if (plan->stream == stream) return plan;
  }
  return nullptr;
}

static ncclResult_t rmaSchedGetStreamPlan(
    struct ncclComm* comm,
    struct ncclRmaSchedStreamPlan** head,
    cudaStream_t stream,
    bool captured,
    const struct ncclCudaGraph& graph,
    struct ncclRmaSchedStreamPlan** streamPlanOut
  ) {
  struct ncclRmaSchedStreamPlan* streamPlan = rmaSchedFindStreamPlan(*head, stream);
  if (streamPlan != nullptr) {
    *streamPlanOut = streamPlan;
    return ncclSuccess;
  }

  streamPlan = ncclMemoryStackAlloc<struct ncclRmaSchedStreamPlan>(&comm->memScoped);
  streamPlan->stream = stream;
  streamPlan->captured = captured;
  streamPlan->graph = graph;
  streamPlan->next = *head;
  *head = streamPlan;
  NCCLCHECK(rmaSchedInitStreamPlan(comm, streamPlan));
  *streamPlanOut = streamPlan;
  return ncclSuccess;
}

static ncclResult_t rmaSchedStreamCaptured(
    struct ncclComm* comm, cudaStream_t stream,
    bool* capturedOut, struct ncclCudaGraph* graphOut
  ) {
  NCCLCHECK(ncclCudaGetCapturingGraph(graphOut, stream, comm->config.graphUsageMode));
  *capturedOut = ncclCudaGraphValid(*graphOut);
  return ncclSuccess;
}

static bool rmaCtxPlanHasTasks(struct ncclRmaSchedCtxPlan* ctxPlan) {
  return ctxPlan->waitSignalCE.nElems + ctxPlan->waitSignalProxy.nElems +
    ctxPlan->putSignalCE.nElems + ctxPlan->putSignalProxy.nElems > 0;
}

static bool rmaStreamPlanHasTasks(struct ncclRmaSchedStreamPlan* streamPlan, int nRmaCtx) {
  for (int ctx = 0; ctx < nRmaCtx; ctx++) {
    if (rmaCtxPlanHasTasks(&streamPlan->ctxPlans[ctx])) return true;
  }
  return false;
}

static void rmaSchedFreeTuningInfoRaw(struct ncclComm* comm, struct ncclTaskTuningInfo* taskInfo) {
  if (taskInfo->raw != nullptr) {
    ncclMemoryPoolFree(&comm->memPool_ncclRawTask, taskInfo->raw);
    taskInfo->raw = nullptr;
  }
}

static ncclResult_t rmaSchedProxyWaitBuildDesc(
    struct ncclComm* comm, struct ncclRmaProxyCtx* rmaProxyCtx,
    struct ncclRmaSchedStreamPlan* streamPlan,
    int npeers, int** peers, int** nsignals, int** signalIdxs,
    struct ncclRmaProxyDesc* desc) {
  ncclResult_t ret = ncclSuccess;
  bool captured = streamPlan->captured;

  desc->rmaDescType = ncclRmaDescTypeWaitSignal;
  desc->rmaDescState = ncclRmaDescStateReady;
  desc->waitSignal.npeers = npeers;
  desc->waitSignal.waitPeers = *peers;
  desc->waitSignal.waitSignals = *nsignals;
  desc->waitSignal.waitSignalIdxs = *signalIdxs;
  *peers = nullptr;
  *nsignals = nullptr;
  *signalIdxs = nullptr;
  desc->persistPlan = nullptr;
  desc->persistDescValid = false;
  if (captured) {
    desc->opSeq = 1;
    desc->waitSignal.needFlush = (rmaProxyCtx->cpuAccessSignalsGdrHandle == NULL);
    NCCLCHECKGOTO(allocMemCPUAccessible(&desc->readySeq, &desc->readySeqDev, 1, 0, &desc->readySeqGdrHandle, comm->memManager), ret, fail);
    NCCLCHECKGOTO(allocMemCPUAccessible(&desc->doneSeq, &desc->doneSeqDev, 1, 0, &desc->doneSeqGdrHandle, comm->memManager), ret, fail);
    desc->graph = streamPlan->graph;
    desc->captured = streamPlan->captured;
  }
exit:
  return ret;
fail:
  if (desc->readySeq) {
    freeMemCPUAccessible(desc->readySeq, desc->readySeqGdrHandle, comm->memManager);
    desc->readySeq = nullptr;
    desc->readySeqDev = nullptr;
    desc->readySeqGdrHandle = nullptr;
  }
  if (desc->doneSeq) {
    freeMemCPUAccessible(desc->doneSeq, desc->doneSeqGdrHandle, comm->memManager);
    desc->doneSeq = nullptr;
    desc->doneSeqDev = nullptr;
    desc->doneSeqGdrHandle = nullptr;
  }
  goto exit;
}

static ncclResult_t rmaSchedProxyWaitLaunch(
    struct ncclComm* comm,
    int ctx,
    struct ncclIntruQueue<struct ncclTaskTuningInfo, &ncclTaskTuningInfo::next>* queue,
    struct ncclRmaSchedStreamPlan* streamPlan
  ) {
  ncclResult_t ret = ncclSuccess;
  struct ncclRmaProxyDesc* desc = nullptr;
  int* peers = nullptr;
  int* nsignals = nullptr;
  int* signalIdxs = nullptr;
  CUstreamBatchMemOpParams* batchParams = nullptr;
  int nElems = queue->nElems;
  cudaStream_t stream = streamPlan->stream;

  if (nElems == 0) return ncclSuccess;

  if (!comm->rmaState.rmaProxyState.connected) {
    WARN("RMA proxy is not connected");
    return ncclInternalError;
  }

  struct ncclRmaProxyCtx* rmaProxyCtx = (struct ncclRmaProxyCtx*)comm->rmaState.rmaProxyState.rmaProxyCtxs[ctx];

  NCCLCHECKGOTO(ncclCalloc(&desc, 1), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&peers, nElems), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&nsignals, nElems), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&signalIdxs, nElems), ret, fail);

  for (int i = 0; i < nElems; i++) {
    struct ncclTaskTuningInfo* taskInfo = ncclIntruQueueDequeue(queue);
    struct ncclRawTaskWaitSignal* wait = &taskInfo->raw->rma.rmaOp.waitSignal;

    peers[i] = wait->signalDesc.peer;
    nsignals[i] = wait->signalDesc.opCnt;
    signalIdxs[i] = wait->signalDesc.sigIdx;

    rmaSchedFreeTuningInfoRaw(comm, taskInfo);
  }

  NCCLCHECKGOTO(rmaSchedProxyWaitBuildDesc(
      comm, rmaProxyCtx, streamPlan, nElems, &peers, &nsignals, &signalIdxs, desc), ret, fail);

  {
    int nOps = ncclRmaProxyWaitNumStreamOps(desc);
    NCCLCHECKGOTO(ncclCalloc(&batchParams, nOps), ret, fail);
    NCCLCHECKGOTO(ncclRmaProxyWaitParams(rmaProxyCtx, desc, batchParams), ret, fail);
    NCCLCHECKGOTO(ncclRmaProxyEnqueueDesc(rmaProxyCtx, &desc), ret, fail);
    NCCLCHECKGOTO(ncclCuStreamBatchMemOp(stream, nOps, batchParams), ret, fail);
  }

exit:
  if (desc != nullptr) {
    (void)ncclRmaProxyDestroyDesc(comm, &desc);
  }
  free(batchParams);
  return ret;
fail:
  free(peers);
  free(nsignals);
  free(signalIdxs);
  goto exit;
}

struct rmaSchedPutFields {
  ncclFunc_t func;
  const void* srcBuff;
  size_t count;
  ncclDataType_t datatype;
  int peer;
  int ctx;
  int sigIdx;
  size_t peerWinOffset;
  ncclSignalMode_t signalMode;
  struct ncclDevrWindow* srcWinHost;
  size_t srcWinOffset;
  struct ncclDevrWindow* peerWinHost;
};

static ncclResult_t rmaSchedPutFieldsFromTuningInfo(
    struct ncclComm* comm, struct ncclTaskTuningInfo* tInfo, struct rmaSchedPutFields* fields) {
  struct ncclRawTask* raw = tInfo->raw;

  fields->func = raw->rma.func;
  fields->signalMode = NCCL_SIGNAL;
  fields->srcWinHost = NULL;
  fields->srcWinOffset = 0;
  fields->peerWinHost = NULL;

  if (raw->rma.func == ncclFuncPutSignal) {
    struct ncclRawTaskPutSignal* put = &raw->rma.rmaOp.putSignal;
    fields->srcBuff = put->localbuff;
    fields->count = put->count;
    fields->datatype = put->datatype;
    fields->peer = put->peer;
    fields->ctx = put->ctx;
    fields->sigIdx = put->sigIdx;
    fields->peerWinOffset = put->peerWinOffset;
    NCCLCHECK(ncclDevrFindWindow(comm, put->localbuff, &fields->srcWinHost));
    if (fields->srcWinHost != NULL) {
      fields->srcWinOffset = (char*)put->localbuff - (char*)fields->srcWinHost->userPtr;
    }
    struct ncclWindow_vidmem* peerWinDevHost = NULL;
    NCCLCHECK(ncclShadowPoolToHost(&comm->devrState.shadows, put->peerWin, &peerWinDevHost));
    fields->peerWinHost = (struct ncclDevrWindow*)peerWinDevHost->winHost;
  } else if (raw->rma.func == ncclFuncSignal) {
    struct ncclRawTaskSignal* sig = &raw->rma.rmaOp.signal;
    fields->srcBuff = NULL;
    fields->count = 0;
    fields->datatype = ncclInt8;
    fields->peer = sig->peer;
    fields->ctx = sig->ctx;
    fields->sigIdx = sig->sigIdx;
    fields->peerWinOffset = 0;
  } else {
    return ncclInternalError;
  }
  return ncclSuccess;
}

struct rmaSchedCeWaitFields {
  int peer;
  int opCnt;
  int sigIdx;
  ncclSignalMode_t signalMode;
};

static ncclResult_t rmaSchedCeWaitFieldsFromTuningInfo(
    struct ncclTaskTuningInfo* tInfo, struct rmaSchedCeWaitFields* fields) {
  struct ncclRawTask* raw = tInfo->raw;

  if (raw->rma.func != ncclFuncWaitSignal) return ncclInternalError;

  struct ncclRawTaskWaitSignal* wait = &raw->rma.rmaOp.waitSignal;
  fields->peer = wait->signalDesc.peer;
  fields->opCnt = wait->signalDesc.opCnt;
  fields->sigIdx = wait->signalDesc.sigIdx;
  fields->signalMode = NCCL_SIGNAL;
  return ncclSuccess;
}

static ncclResult_t rmaSchedProxyPutBuildOp(
    struct ncclComm* comm, struct ncclRmaProxyCtx* rmaProxyCtx,
    int ctx, bool captured,
    struct ncclDevrWindow* srcWin, size_t srcOff,
    struct ncclDevrWindow* peerWin, size_t peerOff,
    size_t size, int peer, int signalIdx,
    ncclSignalMode_t signalMode,
    struct ncclRmaPutSignalOp* op) {
  op->srcOff = srcOff;
  op->srcHandle = ncclDevrGetRmaWin(srcWin, ctx);
  op->dstOff = peerOff;
  op->dstHandle = ncclDevrGetRmaWin(peerWin, ctx);
  op->size = size;
  op->targetRank = peer;
  op->request = nullptr;

  if (signalMode == NCCL_SIGNAL_NONE) {
    op->signal.op = 0;
  } else if (signalMode == NCCL_SIGNAL) {
    op->signal.op = NCCL_NET_SIGNAL_OP_ADD;
    op->signal.offset = ncclRmaSignalOffset(comm->nRanks, signalIdx, comm->rank);
    op->signal.signalMhandle = captured ? rmaProxyCtx->cpuAccessSignalsMhandle
                                          : rmaProxyCtx->signalsMhandle;
    op->signal.val = 1;
  }
  return ncclSuccess;
}

static ncclResult_t rmaSchedProxyPutBuildDesc(
    struct ncclComm* comm, struct ncclRmaProxyCtx* rmaProxyCtx,
    struct ncclRmaSchedStreamPlan* streamPlan,
    struct ncclDevrWindow* srcWinHost, size_t srcWinOffset,
    struct ncclDevrWindow* peerWinHost, size_t peerWinOffset,
    size_t size, int peer, int ctx, int signalIdx,
    ncclSignalMode_t signalMode,
    struct ncclRmaProxyDesc* desc) {
  ncclResult_t ret = ncclSuccess;
  bool captured = streamPlan->captured;

  desc->rmaDescType = ncclRmaDescTypePutSignal;
  desc->rmaDescState = ncclRmaDescStateReady;
  desc->persistPlan = nullptr;
  desc->persistDescValid = false;

  NCCLCHECKGOTO(rmaSchedProxyPutBuildOp(
      comm, rmaProxyCtx, ctx, captured,
      srcWinHost, srcWinOffset, peerWinHost, peerWinOffset,
      size, peer, signalIdx, signalMode, &desc->putSignal), ret, fail);

  if (!captured) {
    desc->opSeq = ++rmaProxyCtx->opSeqs[peer];
    desc->readySeq = &rmaProxyCtx->readySeqs[peer];
    desc->readySeqDev = &rmaProxyCtx->readySeqsDev[peer];
    desc->readySeqGdrHandle = rmaProxyCtx->readySeqsGdrHandle;
    desc->doneSeq = &rmaProxyCtx->doneSeqs[peer];
    desc->doneSeqDev = &rmaProxyCtx->doneSeqsDev[peer];
    desc->doneSeqGdrHandle = rmaProxyCtx->doneSeqsGdrHandle;
  } else {
    desc->opSeq = 1;
    NCCLCHECKGOTO(allocMemCPUAccessible(&desc->readySeq, &desc->readySeqDev, 1, 0, &desc->readySeqGdrHandle, comm->memManager), ret, fail);
    NCCLCHECKGOTO(allocMemCPUAccessible(&desc->doneSeq, &desc->doneSeqDev, 1, 0, &desc->doneSeqGdrHandle, comm->memManager), ret, fail);
    desc->graph = streamPlan->graph;
    desc->captured = streamPlan->captured;
  }
exit:
  return ret;
fail:
  if (desc->readySeq) {
    freeMemCPUAccessible(desc->readySeq, desc->readySeqGdrHandle, comm->memManager);
    desc->readySeq = nullptr;
    desc->readySeqDev = nullptr;
    desc->readySeqGdrHandle = nullptr;
  }
  if (desc->doneSeq) {
    freeMemCPUAccessible(desc->doneSeq, desc->doneSeqGdrHandle, comm->memManager);
    desc->doneSeq = nullptr;
    desc->doneSeqDev = nullptr;
    desc->doneSeqGdrHandle = nullptr;
  }
  goto exit;
}

static ncclResult_t rmaSchedProxyPutLaunch(
    struct ncclComm* comm, int ctx,
    struct ncclIntruQueue<struct ncclTaskTuningInfo, &ncclTaskTuningInfo::next>* queue,
    struct ncclRmaSchedStreamPlan* streamPlan) {
  ncclResult_t ret = ncclSuccess;
  int nRmaTasksProxy = queue->nElems;
  cudaStream_t stream = streamPlan->stream;

  if (nRmaTasksProxy == 0) return ncclSuccess;

  if (!comm->rmaState.rmaProxyState.connected) {
    WARN("RMA proxy is not connected");
    return ncclInternalError;
  }

  bool captured = streamPlan->captured;
  struct ncclRmaProxyCtx* rmaProxyCtx = (struct ncclRmaProxyCtx*)comm->rmaState.rmaProxyState.rmaProxyCtxs[ctx];

  int startOps = ncclRmaProxyPutStartNumOps(captured);
  int doneOps = ncclRmaProxyPutDoneNumOps(captured);
  int opsPerTask = startOps + doneOps;

  struct ncclRmaProxyDesc** descs = nullptr;
  CUstreamBatchMemOpParams* batchParams = nullptr;
  NCCLCHECK(ncclCalloc(&descs, nRmaTasksProxy));
  NCCLCHECK(ncclCalloc(&batchParams, opsPerTask * nRmaTasksProxy));

  for (int i = 0; i < nRmaTasksProxy; i++) {
    struct ncclTaskTuningInfo* taskInfo = ncclIntruQueueDequeue(queue);
    struct rmaSchedPutFields fields = {};

    NCCLCHECKGOTO(ncclCalloc(&descs[i], 1), ret, fail);
    NCCLCHECKGOTO(rmaSchedPutFieldsFromTuningInfo(comm, taskInfo, &fields), ret, fail);
    NCCLCHECKGOTO(rmaSchedProxyPutBuildDesc(
        comm, rmaProxyCtx, streamPlan,
        fields.srcWinHost, fields.srcWinOffset,
        fields.peerWinHost, fields.peerWinOffset,
        fields.count * ncclTypeSize(fields.datatype), fields.peer, fields.ctx, fields.sigIdx,
        fields.signalMode, descs[i]), ret, fail);
    NCCLCHECKGOTO(ncclRmaProxyPutStartParams(descs[i], &batchParams[i * startOps]), ret, fail);
    NCCLCHECKGOTO(ncclRmaProxyPutDoneParams(descs[i], &batchParams[nRmaTasksProxy * startOps + i * doneOps]), ret, fail);

    INFO(NCCL_COLL, "rmaSchedProxyPutLaunch enqueued Desc: rank=%d peer=%d ctx=%d size=%ld signalMode=%d readySeq=%lu doneSeq=%lu captured=%d",
      comm->rank, fields.peer, ctx, fields.count * ncclTypeSize(fields.datatype), fields.signalMode,
      (uint64_t)descs[i]->opSeq, (uint64_t)descs[i]->opSeq, captured);

    rmaSchedFreeTuningInfoRaw(comm, taskInfo);
  }

  {
    int count = 0;
    while (count < nRmaTasksProxy) {
      int i = count;
      for (; i < nRmaTasksProxy; i++) {
        if (ncclRmaProxyEnqueueFull(rmaProxyCtx, descs[i])) {
          break;
        }
        NCCLCHECKGOTO(ncclRmaProxyEnqueueDesc(rmaProxyCtx, &descs[i]), ret, fail);
      }

      int pending = i - count;
      if (pending == 0) {
        std::this_thread::yield();
        continue;
      }

      if (count == 0 && i == nRmaTasksProxy) {
        NCCLCHECKGOTO(ncclCuStreamBatchMemOp(stream,
                          opsPerTask * nRmaTasksProxy,
                          batchParams),
                      ret, fail);
      } else {
        NCCLCHECKGOTO(ncclCuStreamBatchMemOp(stream,
                          pending * startOps,
                          &batchParams[count * startOps]),
                      ret, fail);
        NCCLCHECKGOTO(ncclCuStreamBatchMemOp(stream,
                          pending * doneOps,
                          &batchParams[nRmaTasksProxy * startOps + count * doneOps]),
                      ret, fail);
      }
      count = i;
    }
  }

exit:
  free(batchParams);
  free(descs);
  return ret;
fail:
  if (descs != nullptr) {
    for (int i = 0; i < nRmaTasksProxy; i++) {
      if (descs[i] == nullptr) continue;
      (void)ncclRmaProxyDestroyDesc(comm, &descs[i]);
    }
  }
  goto exit;
}

static ncclResult_t rmaSchedCePutLaunchPersist(
    struct ncclComm* comm, int ctx,
    struct ncclIntruQueue<struct ncclTaskTuningInfo, &ncclTaskTuningInfo::next>* queue,
    cudaStream_t stream) {
  ncclResult_t ret = ncclSuccess;
  int nRmaTasksCe = queue->nElems;
  struct ncclRmaCeCtx* ceCtx = (struct ncclRmaCeCtx*)comm->rmaState.rmaCeState.rmaCeCtxs[ctx];
  int lsaSize = comm->devrState.lsaTeamSize;
  int lsaSelf = comm->devrState.lsaTeamSelf;
  // A batched memcpy does not guarantee order of execution within the batch, so the signal
  // must not share a batch with its payload: it could land first and release the peer's
  // wait before the data arrives.
  struct ncclCeBatchOpsParams dataParams = {};
  struct ncclCeBatchOpsParams signalParams = {};

  if (nRmaTasksCe == 0) return ncclSuccess;

  NCCLCHECKGOTO(ncclCeInitBatchOpsParams(&dataParams, 1), ret, fail);
  NCCLCHECKGOTO(ncclCeInitBatchOpsParams(&signalParams, 1), ret, fail);

  for (int i = 0; i < nRmaTasksCe; i++) {
    struct ncclTaskTuningInfo* taskInfo = ncclIntruQueueDequeue(queue);
    struct rmaSchedPutFields fields = {};

    NCCLCHECKGOTO(rmaSchedPutFieldsFromTuningInfo(comm, taskInfo, &fields), ret, fail);
    rmaSchedFreeTuningInfoRaw(comm, taskInfo);

    int peerLsaRank;
    NCCLCHECKGOTO(ncclDevrWorldToLsaRank(comm, fields.peer, &peerLsaRank), ret, fail);
    size_t bytes = fields.count * ncclTypeSize(fields.datatype);

    if (fields.signalMode != NCCL_SIGNAL_NONE) {
      size_t ackSlot = ncclRmaSignalSlot(lsaSize, fields.sigIdx, peerLsaRank);
      CUdeviceptr ackAddr = (CUdeviceptr)&ceCtx->graphAckDev[ackSlot];
      CUstreamBatchMemOpParams ackOps[2] = {};
      ackOps[0].waitValue.operation = CU_STREAM_MEM_OP_WAIT_VALUE_64;
      ackOps[0].waitValue.address = ackAddr;
      ackOps[0].waitValue.value64 = 1;
      ackOps[0].waitValue.flags = CU_STREAM_WAIT_VALUE_GEQ;
      ackOps[1].writeValue.operation = CU_STREAM_MEM_OP_WRITE_VALUE_64;
      ackOps[1].writeValue.address = ackAddr;
      ackOps[1].writeValue.value64 = 0;
      ackOps[1].writeValue.flags = CU_STREAM_WRITE_VALUE_DEFAULT;
      NCCLCHECKGOTO(ncclCuStreamBatchMemOp(stream, 2, ackOps), ret, fail);
    }

    dataParams.numOps = 0;
    signalParams.numOps = 0;

    if (bytes > 0) {
      void* peerBuff;
      NCCLCHECKGOTO(ncclDevrGetLsaRankPtr(comm, fields.peerWinHost, fields.peerWinOffset, peerLsaRank, &peerBuff), ret, fail);
      if (peerBuff == NULL) {
        WARN("RMA CE: peerBuff is NULL after ncclDevrGetLsaRankPtr");
        ret = ncclInvalidArgument;
        goto fail;
      }
      dataParams.srcs[dataParams.numOps]  = const_cast<void*>(fields.srcBuff);
      dataParams.dsts[dataParams.numOps]  = peerBuff;
      dataParams.sizes[dataParams.numOps] = bytes;
      dataParams.numOps++;
    }

    if (fields.signalMode != NCCL_SIGNAL_NONE) {
      size_t signalOffset = ncclRmaSignalOffset(lsaSize, fields.sigIdx, lsaSelf);
      void* peerGraphSignal;
      NCCLCHECKGOTO(ncclDevrGetLsaRankPtr(comm, ceCtx->signalsWin, ceCtx->graphSignalOffset + signalOffset, peerLsaRank, &peerGraphSignal), ret, fail);
      signalParams.srcs[signalParams.numOps]  = ceCtx->signalConstOneDev;
      signalParams.dsts[signalParams.numOps]  = peerGraphSignal;
      signalParams.sizes[signalParams.numOps] = sizeof(uint64_t);
      signalParams.numOps++;
    }

    NCCLCHECKGOTO(ncclCeLaunchBatchOps(comm, &dataParams, stream), ret, fail);
    NCCLCHECKGOTO(ncclCeLaunchBatchOps(comm, &signalParams, stream), ret, fail);
  }

exit:
  ncclCeFreeBatchOpsParams(&dataParams);
  ncclCeFreeBatchOpsParams(&signalParams);
  return ret;
fail:
  goto exit;
}

static ncclResult_t rmaSchedCePutLaunchNonPersist(
    struct ncclComm* comm, int ctx,
    struct ncclIntruQueue<struct ncclTaskTuningInfo, &ncclTaskTuningInfo::next>* queue,
    cudaStream_t stream) {
  ncclResult_t ret = ncclSuccess;
  struct ncclRmaCeCtx* ceCtx = (struct ncclRmaCeCtx*)comm->rmaState.rmaCeState.rmaCeCtxs[ctx];
  int lsaSize = comm->devrState.lsaTeamSize;
  int lsaSelf = comm->devrState.lsaTeamSelf;
  // A batched memcpy does not guarantee order of execution within the batch. Data and signal
  // ops therefore go in separate batches (data first), and each batch carries at most one task
  // per peer so no two sequence-number writes target the same signal slot in one batch.
  struct ncclIntruQueue<struct ncclTaskTuningInfo, &ncclTaskTuningInfo::next>* peerTaskQueues = nullptr;
  int* activePeers = nullptr;
  struct ncclCeBatchOpsParams dataParams = {};
  struct ncclCeBatchOpsParams signalParams = {};
  CUstreamBatchMemOpParams* seqStageOps = nullptr;
  int nActivePeers = 0;
  int nQueuedPeers = 0;
  int selfPeer = -1;

  if (queue->nElems == 0) goto exit;

  NCCLCHECKGOTO(ncclCalloc(&peerTaskQueues, comm->nRanks), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&activePeers, comm->nRanks), ret, fail);
  NCCLCHECKGOTO(ncclCeInitBatchOpsParams(&dataParams, comm->nRanks), ret, fail);
  NCCLCHECKGOTO(ncclCeInitBatchOpsParams(&signalParams, comm->nRanks), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&seqStageOps, comm->nRanks), ret, fail);

  while (!ncclIntruQueueEmpty(queue)) {
    struct ncclTaskTuningInfo* taskInfo = ncclIntruQueueDequeue(queue);
    int peer = rmaTuningInfoPeer(taskInfo);
    if (peer < 0 || peer >= comm->nRanks) {
      WARN("RMA CE: task has invalid peer %d", peer);
      rmaSchedFreeTuningInfoRaw(comm, taskInfo);
      ret = ncclInternalError;
      goto fail;
    }
    if (ncclIntruQueueEmpty(&peerTaskQueues[peer])) nQueuedPeers++;
    ncclIntruQueueEnqueue(&peerTaskQueues[peer], taskInfo);
  }

  // Arrange independent peer heads in LSA-relative cyclic order. For a dense all-to-all,
  // every batch position then targets a different rank on every sender, avoiding an
  // application-order-induced destination incast.
  selfPeer = comm->devrState.lsaTeamRankList[lsaSelf];
  for (int step = 1; step < lsaSize; step++) {
    int peerLsaRank = (lsaSelf + step) % lsaSize;
    int peer = comm->devrState.lsaTeamRankList[peerLsaRank];
    if (!ncclIntruQueueEmpty(&peerTaskQueues[peer])) activePeers[nActivePeers++] = peer;
  }
  if (!ncclIntruQueueEmpty(&peerTaskQueues[selfPeer])) activePeers[nActivePeers++] = selfPeer;
  if (nActivePeers != nQueuedPeers) {
    WARN("RMA CE: cyclic peer ordering lost active peers (%d of %d)", nActivePeers, nQueuedPeers);
    ret = ncclInternalError;
    goto fail;
  }

  while (nActivePeers > 0) {
    int nNextActivePeers = 0;
    int nSeqStageOps = 0;
    dataParams.numOps = 0;
    signalParams.numOps = 0;

    for (int i = 0; i < nActivePeers; i++) {
      int peer = activePeers[i];
      struct ncclTaskTuningInfo* taskInfo = ncclIntruQueueDequeue(&peerTaskQueues[peer]);
      struct rmaSchedPutFields fields = {};

      NCCLCHECKGOTO(rmaSchedPutFieldsFromTuningInfo(comm, taskInfo, &fields), ret, fail);
      rmaSchedFreeTuningInfoRaw(comm, taskInfo);

      int peerLsaRank;
      NCCLCHECKGOTO(ncclDevrWorldToLsaRank(comm, fields.peer, &peerLsaRank), ret, fail);
      size_t bytes = fields.count * ncclTypeSize(fields.datatype);

      if (bytes > 0) {
        void* peerBuff;
        NCCLCHECKGOTO(ncclDevrGetLsaRankPtr(comm, fields.peerWinHost, fields.peerWinOffset, peerLsaRank, &peerBuff), ret, fail);
        if (peerBuff == NULL) {
          WARN("RMA CE: peerBuff is NULL after ncclDevrGetLsaRankPtr");
          ret = ncclInvalidArgument;
          goto fail;
        }
        dataParams.srcs[dataParams.numOps]  = const_cast<void*>(fields.srcBuff);
        dataParams.dsts[dataParams.numOps]  = peerBuff;
        dataParams.sizes[dataParams.numOps] = bytes;
        dataParams.numOps++;
      }

      if (fields.signalMode != NCCL_SIGNAL_NONE) {
        size_t signalOffset = ncclRmaSignalOffset(lsaSize, fields.sigIdx, lsaSelf);
        size_t signalSlot = ncclRmaSignalSlot(lsaSize, fields.sigIdx, peerLsaRank);
        void* peerSignal;
        NCCLCHECKGOTO(ncclDevrGetLsaRankPtr(comm, ceCtx->signalsWin, ceCtx->signalOffset + signalOffset, peerLsaRank, &peerSignal), ret, fail);
        ceCtx->signalOpSeqs[signalSlot]++;

        seqStageOps[nSeqStageOps].writeValue.operation = CU_STREAM_MEM_OP_WRITE_VALUE_64;
        seqStageOps[nSeqStageOps].writeValue.address   = (CUdeviceptr)&ceCtx->signalOpSeqsDev[nSeqStageOps];
        seqStageOps[nSeqStageOps].writeValue.value64   = ceCtx->signalOpSeqs[signalSlot];
        seqStageOps[nSeqStageOps].writeValue.flags     = CU_STREAM_WRITE_VALUE_DEFAULT;

        signalParams.srcs[signalParams.numOps]  = &ceCtx->signalOpSeqsDev[nSeqStageOps];
        signalParams.dsts[signalParams.numOps]  = peerSignal;
        signalParams.sizes[signalParams.numOps] = sizeof(uint64_t);
        signalParams.numOps++;

        nSeqStageOps++;
      }

      if (!ncclIntruQueueEmpty(&peerTaskQueues[peer])) activePeers[nNextActivePeers++] = peer;
    }

    // Issue in stream order: staging writes feed the signal memcpys, and the data batch
    // must complete before the signals that advertise it.
    if (nSeqStageOps > 0) {
      NCCLCHECKGOTO(ncclCuStreamBatchMemOp(stream, nSeqStageOps, seqStageOps), ret, fail);
    }
    NCCLCHECKGOTO(ncclCeLaunchBatchOps(comm, &dataParams, stream), ret, fail);
    NCCLCHECKGOTO(ncclCeLaunchBatchOps(comm, &signalParams, stream), ret, fail);
    nActivePeers = nNextActivePeers;
  }

exit:
  if (peerTaskQueues != nullptr) {
    for (int peer = 0; peer < comm->nRanks; peer++) {
      while (!ncclIntruQueueEmpty(&peerTaskQueues[peer])) {
        struct ncclTaskTuningInfo* taskInfo = ncclIntruQueueDequeue(&peerTaskQueues[peer]);
        rmaSchedFreeTuningInfoRaw(comm, taskInfo);
      }
    }
  }
  ncclCeFreeBatchOpsParams(&dataParams);
  ncclCeFreeBatchOpsParams(&signalParams);
  free(peerTaskQueues);
  free(activePeers);
  free(seqStageOps);
  return ret;
fail:
  goto exit;
}

static ncclResult_t rmaSchedCePutLaunch(
    struct ncclComm* comm, int ctx,
    struct ncclIntruQueue<struct ncclTaskTuningInfo, &ncclTaskTuningInfo::next>* queue,
    struct ncclRmaSchedStreamPlan* streamPlan) {
  if (queue->nElems == 0) return ncclSuccess;

  if (!comm->rmaState.rmaCeState.initialized) {
    WARN("RMA CE is not initialized");
    return ncclInternalError;
  }

  if (streamPlan->captured) {
    return rmaSchedCePutLaunchPersist(comm, ctx, queue, streamPlan->stream);
  }
  return rmaSchedCePutLaunchNonPersist(comm, ctx, queue, streamPlan->stream);
}

static ncclResult_t rmaSchedProxyCEPutLaunch(
    struct ncclComm* comm, int ctx,
    struct ncclIntruQueue<struct ncclTaskTuningInfo, &ncclTaskTuningInfo::next>* proxyQueue,
    struct ncclIntruQueue<struct ncclTaskTuningInfo, &ncclTaskTuningInfo::next>* ceQueue,
    struct ncclRmaSchedStreamPlan* streamPlan) {
  ncclResult_t ret = ncclSuccess;

  if (!comm->rmaState.rmaProxyState.connected) {
    WARN("RMA proxy is not connected");
    return ncclInternalError;
  }
  if (!comm->rmaState.rmaCeState.initialized) {
    WARN("RMA CE is not initialized");
    return ncclInternalError;
  }
  // need to merge these 2 functions in the future
  NCCLCHECKGOTO(rmaSchedProxyPutLaunch(comm, ctx, proxyQueue, streamPlan), ret, fail);
  NCCLCHECKGOTO(rmaSchedCePutLaunch(comm, ctx, ceQueue, streamPlan), ret, fail);

exit:
  return ret;
fail:
  goto exit;
}

static ncclResult_t rmaSchedCeWaitLaunch(
    struct ncclComm* comm,
    int ctx,
    struct ncclIntruQueue<struct ncclTaskTuningInfo, &ncclTaskTuningInfo::next>* queue,
    struct ncclRmaSchedStreamPlan* streamPlan
  ) {
  ncclResult_t ret = ncclSuccess;
  CUstreamBatchMemOpParams* batchParams = nullptr;
  struct ncclCeBatchOpsParams ceParams = {};
  int nElems = queue->nElems;
  cudaStream_t stream = streamPlan->stream;
  bool captured = streamPlan->captured;

  if (nElems == 0) return ncclSuccess;

  if (!comm->rmaState.rmaCeState.initialized) {
    WARN("RMA CE is not initialized");
    return ncclInternalError;
  }

  struct ncclRmaCeCtx* ceCtx = (struct ncclRmaCeCtx*)comm->rmaState.rmaCeState.rmaCeCtxs[ctx];
  int lsaSize = comm->devrState.lsaTeamSize;
  int lsaSelf = comm->devrState.lsaTeamSelf;

  if (!captured) {
    NCCLCHECKGOTO(ncclCalloc(&batchParams, nElems), ret, fail);
    size_t opIdx = 0;
    for (int i = 0; i < nElems; i++) {
      struct ncclTaskTuningInfo* taskInfo = ncclIntruQueueDequeue(queue);
      struct rmaSchedCeWaitFields fields = {};

      NCCLCHECKGOTO(rmaSchedCeWaitFieldsFromTuningInfo(taskInfo, &fields), ret, fail);
      rmaSchedFreeTuningInfoRaw(comm, taskInfo);

      if (fields.signalMode == NCCL_SIGNAL) {
        int peerLsaRank;
        NCCLCHECKGOTO(ncclDevrWorldToLsaRank(comm, fields.peer, &peerLsaRank), ret, fail);
        size_t signalSlot = ncclRmaSignalSlot(lsaSize, fields.sigIdx, peerLsaRank);
        uint64_t waitValue = ceCtx->signalsHost[signalSlot] + fields.opCnt;
        ceCtx->signalsHost[signalSlot] = waitValue;

        CUdeviceptr signalAddr = (CUdeviceptr)&ceCtx->signalsDev[signalSlot];
        batchParams[opIdx].waitValue.operation = CU_STREAM_MEM_OP_WAIT_VALUE_64;
        batchParams[opIdx].waitValue.address = signalAddr;
        batchParams[opIdx].waitValue.value64 = waitValue;
        batchParams[opIdx].waitValue.flags = CU_STREAM_WAIT_VALUE_GEQ;
        opIdx++;
      }
    }
    if (opIdx > 0) {
      NCCLCHECKGOTO(ncclCuStreamBatchMemOp(stream, opIdx, batchParams), ret, fail);
    }
  } else {
    NCCLCHECKGOTO(ncclCeInitBatchOpsParams(&ceParams, 1), ret, fail);

    for (int i = 0; i < nElems; i++) {
      struct ncclTaskTuningInfo* taskInfo = ncclIntruQueueDequeue(queue);
      struct rmaSchedCeWaitFields fields = {};

      NCCLCHECKGOTO(rmaSchedCeWaitFieldsFromTuningInfo(taskInfo, &fields), ret, fail);
      rmaSchedFreeTuningInfoRaw(comm, taskInfo);

      if (fields.signalMode == NCCL_SIGNAL) {
        int peerLsaRank;
        NCCLCHECKGOTO(ncclDevrWorldToLsaRank(comm, fields.peer, &peerLsaRank), ret, fail);
        size_t signalSlot = ncclRmaSignalSlot(lsaSize, fields.sigIdx, peerLsaRank);

        for (int s = 0; s < fields.opCnt; s++) {
          CUdeviceptr graphSignalAddr = (CUdeviceptr)&ceCtx->graphSignalsDev[signalSlot];
          CUstreamBatchMemOpParams signalOps[2] = {};
          signalOps[0].waitValue.operation = CU_STREAM_MEM_OP_WAIT_VALUE_64;
          signalOps[0].waitValue.address = graphSignalAddr;
          signalOps[0].waitValue.value64 = 1;
          signalOps[0].waitValue.flags = CU_STREAM_WAIT_VALUE_GEQ;
          signalOps[1].writeValue.operation = CU_STREAM_MEM_OP_WRITE_VALUE_64;
          signalOps[1].writeValue.address = graphSignalAddr;
          signalOps[1].writeValue.value64 = 0;
          signalOps[1].writeValue.flags = CU_STREAM_WRITE_VALUE_DEFAULT;
          NCCLCHECKGOTO(ncclCuStreamBatchMemOp(stream, 2, signalOps), ret, fail);

          void* peerAck;
          size_t ackOffset = ncclRmaSignalOffset(lsaSize, fields.sigIdx, lsaSelf);
          NCCLCHECKGOTO(ncclDevrGetLsaRankPtr(comm, ceCtx->signalsWin, ceCtx->graphAckOffset + ackOffset, peerLsaRank, &peerAck), ret, fail);
          ceParams.numOps = 0;
          ceParams.srcs[ceParams.numOps]  = ceCtx->signalConstOneDev;
          ceParams.dsts[ceParams.numOps]  = peerAck;
          ceParams.sizes[ceParams.numOps] = sizeof(uint64_t);
          ceParams.numOps++;
          NCCLCHECKGOTO(ncclCeLaunchBatchOps(comm, &ceParams, stream), ret, fail);
        }
      }
    }
  }

exit:
  if (batchParams) free(batchParams);
  ncclCeFreeBatchOpsParams(&ceParams);
  return ret;
fail:
  if (batchParams) free(batchParams);
  ncclCeFreeBatchOpsParams(&ceParams);
  goto exit;
}

static ncclResult_t rmaPutLaunch(
    struct ncclComm* comm, int ctx, struct ncclRmaSchedStreamPlan* streamPlan) {
  ncclResult_t ret = ncclSuccess;
  struct ncclRmaSchedCtxPlan* ctxPlan = &streamPlan->ctxPlans[ctx];

  if (ctxPlan->putSignalProxy.nElems + ctxPlan->putSignalCE.nElems == 0) return ncclSuccess;

  if (ctxPlan->putSignalProxy.nElems > 0 && ctxPlan->putSignalCE.nElems > 0) {
    NCCLCHECKGOTO(rmaSchedProxyCEPutLaunch(comm, ctx, &ctxPlan->putSignalProxy, &ctxPlan->putSignalCE, streamPlan), ret, fail);
  } else if (ctxPlan->putSignalProxy.nElems > 0) {
    NCCLCHECKGOTO(rmaSchedProxyPutLaunch(comm, ctx, &ctxPlan->putSignalProxy, streamPlan), ret, fail);
  } else {
    NCCLCHECKGOTO(rmaSchedCePutLaunch(comm, ctx, &ctxPlan->putSignalCE, streamPlan), ret, fail);
  }

exit:
  return ret;
fail:
  goto exit;
}

static ncclResult_t rmaWaitSignalLaunch(
    struct ncclComm* comm, int ctx, struct ncclRmaSchedStreamPlan* streamPlan) {
  struct ncclRmaSchedCtxPlan* ctxPlan = &streamPlan->ctxPlans[ctx];

  if (ctxPlan->waitSignalCE.nElems + ctxPlan->waitSignalProxy.nElems == 0) return ncclSuccess;

  if (ctxPlan->waitSignalProxy.nElems > 0) {
    NCCLCHECK(rmaSchedProxyWaitLaunch(comm, ctx, &ctxPlan->waitSignalProxy, streamPlan));
  }
  if (ctxPlan->waitSignalCE.nElems > 0) {
    NCCLCHECK(rmaSchedCeWaitLaunch(comm, ctx, &ctxPlan->waitSignalCE, streamPlan));
  }
  return ncclSuccess;
}

static void CUDART_CB rmaSchedStreamPlanDestructor(void* graph_) {
  struct ncclCudaGraph* graph = (struct ncclCudaGraph*)graph_;
  (void)graph;
  free(graph);
}

static ncclResult_t launchRmaStreamPlan(
    struct ncclComm* comm,
    struct ncclRmaSchedStreamPlan* streamPlans
  ) {
  ncclResult_t ret = ncclSuccess;
  int nRmaCtx = comm->config.numRmaCtx;

  for (struct ncclRmaSchedStreamPlan* streamPlan = streamPlans; streamPlan != nullptr; streamPlan = streamPlan->next) {
    for (int ctx = 0; ctx < nRmaCtx; ctx++) {
      NCCLCHECKGOTO(rmaPutLaunch(comm, ctx, streamPlan), ret, fail);
    }

    for (int ctx = 0; ctx < nRmaCtx; ctx++) {
      NCCLCHECKGOTO(rmaWaitSignalLaunch(comm, ctx, streamPlan), ret, fail);
    }

    if (streamPlan->captured) {
      struct ncclCudaGraph* graphArg = nullptr;
      NCCLCHECKGOTO(ncclCalloc(&graphArg, 1), ret, fail);
      *graphArg = streamPlan->graph;
      ret = ncclCudaGraphAddDestructor(
          streamPlan->graph, rmaSchedStreamPlanDestructor, (void*)graphArg);
      if (ret != ncclSuccess) {
        free(graphArg);
        goto fail;
      }
    }
  }

exit:
  return ret;
fail:
  goto exit;
}

static ncclResult_t launchRmaTasks(
    struct ncclComm* comm,
    struct ncclRmaSchedPlan* rmaPlan
  ) {
  ncclResult_t ret = ncclSuccess;

  NCCLCHECKGOTO(launchRmaStreamPlan(comm, rmaPlan->noncapturedStreamPlans), ret, fail);
  NCCLCHECKGOTO(launchRmaStreamPlan(comm, rmaPlan->capturedStreamPlans), ret, fail);

exit:
  return ret;
fail:
  goto exit;
}

static ncclResult_t scheduleRmaTasks(
    struct ncclComm* comm,
    struct ncclIntruQueue<struct ncclTaskTuningInfo, &ncclTaskTuningInfo::next>* queue,
    struct ncclRmaSchedPlan* rmaPlanOut
  ) {
  rmaPlanOut->noncapturedStreamPlans = nullptr;
  rmaPlanOut->capturedStreamPlans = nullptr;

  while (!ncclIntruQueueEmpty(queue)) {
    struct ncclTaskTuningInfo* taskInfo = ncclIntruQueueDequeue(queue);
    cudaStream_t stream = rmaTuningInfoStream(taskInfo);
    int ctx = rmaTuningInfoCtx(taskInfo);
    struct ncclRmaSchedStreamPlan* streamPlan = nullptr;
    struct ncclRmaSchedCtxPlan* ctxPlan = nullptr;
    struct ncclRmaSchedStreamPlan** streamListHead = nullptr;
    bool captured = false;
    struct ncclCudaGraph graph = {};
    bool lsaAccessible = isLsaAccessible(comm, rmaTuningInfoPeer(taskInfo));

    NCCLCHECK(rmaSchedStreamCaptured(comm, stream, &captured, &graph));
    streamListHead = captured ? &rmaPlanOut->capturedStreamPlans : &rmaPlanOut->noncapturedStreamPlans;
    NCCLCHECK(rmaSchedGetStreamPlan(comm, streamListHead, stream, captured, graph, &streamPlan));
    ctxPlan = &streamPlan->ctxPlans[ctx];

    switch (rmaTuningInfoFunc(taskInfo)) {
      case ncclFuncWaitSignal:
        if (lsaAccessible) {
          ncclIntruQueueEnqueue(&ctxPlan->waitSignalCE, taskInfo);
        } else {
          ncclIntruQueueEnqueue(&ctxPlan->waitSignalProxy, taskInfo);
        }
        break;
      case ncclFuncPutSignal:
      case ncclFuncSignal:
        if (lsaAccessible) {
          ncclIntruQueueEnqueue(&ctxPlan->putSignalCE, taskInfo);
        } else {
          ncclIntruQueueEnqueue(&ctxPlan->putSignalProxy, taskInfo);
        }
        break;
      default:
        continue;
    }
  }

  return ncclSuccess;
}

ncclResult_t ncclScheduleRmaTasks(
    struct ncclComm* comm,
    struct ncclIntruQueue<struct ncclTaskTuningInfo, &ncclTaskTuningInfo::next>* queue,
    struct ncclStrongStream* depStream,
    struct ncclStrongStream* launchOrderStream
  ) {
  (void)depStream;
  (void)launchOrderStream;

  if (!ncclIntruQueueEmpty(queue) && comm->config.numRmaCtx > 0) {
    struct ncclRmaSchedPlan rmaPlan = {};

    NCCLCHECK(scheduleRmaTasks(comm, queue, &rmaPlan));
    NCCLCHECK(launchRmaTasks(comm, &rmaPlan));
  }

  return ncclSuccess;
}
