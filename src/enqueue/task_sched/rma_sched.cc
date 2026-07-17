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
  // Set while a captured launch owns proxy descriptors. Moved to the graph
  // destructor, which frees those descriptors.
  struct ncclRmaProxyReclaimToken* reclaimToken;
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
  streamPlan->reclaimToken = nullptr;
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

static ncclResult_t rmaSchedCapturedToken(struct ncclComm* comm, struct ncclRmaSchedStreamPlan* streamPlan) {
  if (streamPlan->reclaimToken != nullptr) return ncclSuccess;
  NCCLCHECK(ncclCalloc(&streamPlan->reclaimToken, 1));
  streamPlan->reclaimToken->comm = comm;
  return ncclSuccess;
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
    NCCLCHECKGOTO(rmaSchedCapturedToken(comm, streamPlan), ret, fail);
    desc->reclaimToken = streamPlan->reclaimToken;
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

// Legacy task append splits a put at 1GiB and signals only the last piece.
static constexpr size_t rmaSchedPutChunkSize = 1ULL << 30;

static int rmaSchedPutChunkCount(size_t bytes) {
  if (bytes <= rmaSchedPutChunkSize) return 1;
  return (int)((bytes + rmaSchedPutChunkSize - 1) / rmaSchedPutChunkSize);
}

static size_t rmaSchedPutChunkBytes(size_t bytes, int chunkIdx) {
  size_t off = (size_t)chunkIdx * rmaSchedPutChunkSize;
  size_t remain = bytes - off;
  return remain > rmaSchedPutChunkSize ? rmaSchedPutChunkSize : remain;
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

  // Shared with the legacy proxy path: window offset is relative to the MR, and
  // the MR table is indexed by the context's physical NIC (collCommIdx), not ctx.
  NCCLCHECKGOTO(ncclRmaProxyPutBuildOp(
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
    NCCLCHECKGOTO(rmaSchedCapturedToken(comm, streamPlan), ret, fail);
    desc->reclaimToken = streamPlan->reclaimToken;
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
    struct ncclComm* comm, struct ncclRmaSchedStreamPlan* streamPlan) {
  ncclResult_t ret = ncclSuccess;
  int nRmaCtx = comm->config.numRmaCtx;
  int nTasks = 0;
  int nDesc = 0;
  int startOps = 0;
  int doneOps = 0;
  int opsPerTask = 0;
  bool captured = streamPlan->captured;
  cudaStream_t stream = streamPlan->stream;
  void** rmaProxyCtxs = nullptr;
  struct ncclTaskTuningInfo** tasks = nullptr;
  struct ncclRmaProxyDesc** descs = nullptr;
  struct ncclRmaProxyCtx** descCtxs = nullptr;
  CUstreamBatchMemOpParams* batchParams = nullptr;

  for (int ctx = 0; ctx < nRmaCtx; ctx++) nTasks += streamPlan->ctxPlans[ctx].putSignalProxy.nElems;
  if (nTasks == 0) return ncclSuccess;

  if (!comm->rmaState.rmaProxyState.connected) {
    WARN("RMA proxy is not connected");
    return ncclInternalError;
  }

  rmaProxyCtxs = comm->rmaState.rmaProxyState.rmaProxyCtxs;
  NCCLCHECKGOTO(ncclCalloc(&tasks, nTasks), ret, fail);
  {
    int n = 0;
    // Drain contexts in index order. That is the order the legacy plan pulls puts
    // before bucketing them into one launch.
    for (int ctx = 0; ctx < nRmaCtx; ctx++) {
      struct ncclIntruQueue<struct ncclTaskTuningInfo, &ncclTaskTuningInfo::next>* queue =
        &streamPlan->ctxPlans[ctx].putSignalProxy;
      while (!ncclIntruQueueEmpty(queue)) tasks[n++] = ncclIntruQueueDequeue(queue);
    }
  }

  for (int t = 0; t < nTasks; t++) {
    struct rmaSchedPutFields fields = {};
    NCCLCHECKGOTO(rmaSchedPutFieldsFromTuningInfo(comm, tasks[t], &fields), ret, fail);
    nDesc += rmaSchedPutChunkCount(fields.count * ncclTypeSize(fields.datatype));
  }

  startOps = ncclRmaProxyPutStartNumOps(captured);
  doneOps = ncclRmaProxyPutDoneNumOps(captured);
  opsPerTask = startOps + doneOps;
  NCCLCHECKGOTO(ncclCalloc(&descs, nDesc), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&descCtxs, nDesc), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&batchParams, (size_t)opsPerTask * nDesc), ret, fail);

  {
    int d = 0;
    for (int t = 0; t < nTasks; t++) {
      struct rmaSchedPutFields fields = {};
      size_t bytes;
      int nChunks;
      struct ncclRmaProxyCtx* taskCtx;

      NCCLCHECKGOTO(rmaSchedPutFieldsFromTuningInfo(comm, tasks[t], &fields), ret, fail);
      if (fields.ctx < 0 || fields.ctx >= nRmaCtx) {
        WARN("RMA proxy: task has invalid context %d", fields.ctx);
        ret = ncclInternalError;
        goto fail;
      }
      bytes = fields.count * ncclTypeSize(fields.datatype);
      nChunks = rmaSchedPutChunkCount(bytes);
      // A single launch batches puts from every context. Each descriptor is built
      // on its own task's context so every start is posted before any blocking done.
      taskCtx = (struct ncclRmaProxyCtx*)rmaProxyCtxs[fields.ctx];
      for (int c = 0; c < nChunks; c++) {
        size_t byteOff = (size_t)c * rmaSchedPutChunkSize;
        ncclSignalMode_t signalMode = (c == nChunks - 1) ? fields.signalMode : NCCL_SIGNAL_NONE;
        NCCLCHECKGOTO(ncclCalloc(&descs[d], 1), ret, fail);
        descCtxs[d] = taskCtx;
        NCCLCHECKGOTO(rmaSchedProxyPutBuildDesc(
            comm, taskCtx, streamPlan,
            fields.srcWinHost, fields.srcWinOffset + byteOff,
            fields.peerWinHost, fields.peerWinOffset + byteOff,
            rmaSchedPutChunkBytes(bytes, c), fields.peer, fields.ctx, fields.sigIdx,
            signalMode, descs[d]), ret, fail);
        NCCLCHECKGOTO(ncclRmaProxyPutStartParams(descs[d], &batchParams[d * startOps]), ret, fail);
        NCCLCHECKGOTO(ncclRmaProxyPutDoneParams(descs[d], &batchParams[nDesc * startOps + d * doneOps]), ret, fail);

        INFO(NCCL_COLL, "rmaSchedProxyPutLaunch enqueued Desc: rank=%d peer=%d ctx=%d size=%zu signalMode=%d "
          "readySeq=%lu doneSeq=%lu captured=%d",
          comm->rank, fields.peer, fields.ctx, rmaSchedPutChunkBytes(bytes, c), signalMode,
          (uint64_t)descs[d]->opSeq, (uint64_t)descs[d]->opSeq, captured);
        d++;
      }
      rmaSchedFreeTuningInfoRaw(comm, tasks[t]);
      tasks[t] = nullptr;
    }
  }

  {
    int count = 0;
    while (count < nDesc) {
      int i = count;
      for (; i < nDesc; i++) {
        if (ncclRmaProxyEnqueueFull(descCtxs[i], descs[i])) break;
        NCCLCHECKGOTO(ncclRmaProxyEnqueueDesc(descCtxs[i], &descs[i]), ret, fail);
      }

      int pending = i - count;
      if (pending == 0) {
        std::this_thread::yield();
        continue;
      }

      if (count == 0 && i == nDesc) {
        NCCLCHECKGOTO(ncclCuStreamBatchMemOp(stream, opsPerTask * nDesc, batchParams), ret, fail);
      } else {
        NCCLCHECKGOTO(ncclCuStreamBatchMemOp(stream, pending * startOps, &batchParams[count * startOps]), ret, fail);
        NCCLCHECKGOTO(ncclCuStreamBatchMemOp(stream, pending * doneOps,
                          &batchParams[nDesc * startOps + count * doneOps]), ret, fail);
      }
      count = i;
    }
  }

exit:
  free(tasks);
  free(batchParams);
  free(descCtxs);
  free(descs);
  return ret;
fail:
  if (tasks != nullptr) {
    for (int t = 0; t < nTasks; t++) {
      if (tasks[t] != nullptr) rmaSchedFreeTuningInfoRaw(comm, tasks[t]);
    }
  }
  if (descs != nullptr) {
    for (int i = 0; i < nDesc; i++) {
      if (descs[i] == nullptr) continue;
      (void)ncclRmaProxyDestroyDesc(comm, &descs[i]);
    }
  }
  goto exit;
}

// Profiler args for one CE launch of RMA tasks. The CE batches it issues report as
// ncclProfileCeBatch children of a ncclProfileCeColl event describing the whole launch,
// parented to the API event of the first task. Wait launches carry no payload.
static void rmaSchedCeCollArgsInit(ncclFunc_t func, struct ncclCeCollArgs* args) {
  memset(args, 0, sizeof(*args));
  args->func = func;
  args->rootRank = -1;
  args->datatype = ncclInt8;
  args->eltSize = 1;
}

static void rmaSchedCeCollArgsAddQueue(
  struct ncclIntruQueue<struct ncclTaskTuningInfo, &ncclTaskTuningInfo::next>* queue, struct ncclCeCollArgs* args) {
  struct ncclTaskTuningInfo* head = ncclIntruQueueHead(queue);
  if (head == nullptr) return;
  if (args->apiEventHandle == nullptr) {
    args->apiEventHandle = head->raw->profiler.apiEventHandle;
    args->userTag = head->raw->profiler.userTag;
    if (args->func == ncclFuncPutSignal) {
      args->datatype = head->raw->rma.rmaOp.putSignal.datatype;
      args->eltSize = ncclTypeSize(args->datatype);
    }
  }
  for (struct ncclTaskTuningInfo* t = head; t != nullptr; t = t->next) {
    args->nElts += args->func == ncclFuncPutSignal ? t->raw->rma.rmaOp.putSignal.count : 1;
  }
}

static ncclResult_t rmaSchedCePutLaunchPersist(struct ncclComm* comm, struct ncclRmaSchedStreamPlan* streamPlan,
                                               cudaStream_t stream, struct ncclCeCollArgs* profArgs) {
  ncclResult_t ret = ncclSuccess;
  int nRmaCtx = comm->config.numRmaCtx;
  int nRmaTasksCe = 0;
  int lsaSize = comm->devrState.lsaTeamSize;
  int lsaSelf = comm->devrState.lsaTeamSelf;
  // A batched memcpy does not guarantee order of execution within the batch, so the signal
  // must not share a batch with its payload: it could land first and release the peer's
  // wait before the data arrives.
  struct ncclCeBatchOpsParams dataParams = {};
  struct ncclCeBatchOpsParams signalParams = {};

  for (int ctx = 0; ctx < nRmaCtx; ctx++) nRmaTasksCe += streamPlan->ctxPlans[ctx].putSignalCE.nElems;
  if (nRmaTasksCe == 0) return ncclSuccess;

  NCCLCHECKGOTO(ncclCeInitBatchOpsParams(&dataParams, 1), ret, fail);
  NCCLCHECKGOTO(ncclCeInitBatchOpsParams(&signalParams, 1), ret, fail);

  for (int ctx = 0; ctx < nRmaCtx; ctx++) {
    struct ncclIntruQueue<struct ncclTaskTuningInfo, &ncclTaskTuningInfo::next>* queue =
      &streamPlan->ctxPlans[ctx].putSignalCE;
    while (!ncclIntruQueueEmpty(queue)) {
      struct ncclTaskTuningInfo* taskInfo = ncclIntruQueueDequeue(queue);
      struct rmaSchedPutFields fields = {};
      struct ncclRmaCeCtx* ceCtx;
      int peerLsaRank;
      size_t bytes;
      int nChunks;

      NCCLCHECKGOTO(rmaSchedPutFieldsFromTuningInfo(comm, taskInfo, &fields), ret, fail);
      rmaSchedFreeTuningInfoRaw(comm, taskInfo);
      if (fields.ctx < 0 || fields.ctx >= nRmaCtx) {
        WARN("RMA CE: task has invalid context %d", fields.ctx);
        ret = ncclInternalError;
        goto fail;
      }
      // Each task signals the peer buffer of its own context.
      ceCtx = (struct ncclRmaCeCtx*)comm->rmaState.rmaCeState.rmaCeCtxs[fields.ctx];

      NCCLCHECKGOTO(ncclDevrWorldToLsaRank(comm, fields.peer, &peerLsaRank), ret, fail);
      bytes = fields.count * ncclTypeSize(fields.datatype);
      nChunks = rmaSchedPutChunkCount(bytes);

      for (int c = 0; c < nChunks; c++) {
        size_t byteOff = (size_t)c * rmaSchedPutChunkSize;
        size_t chunkBytes = rmaSchedPutChunkBytes(bytes, c);
        ncclSignalMode_t signalMode = (c == nChunks - 1) ? fields.signalMode : NCCL_SIGNAL_NONE;

        if (signalMode != NCCL_SIGNAL_NONE) {
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

        if (chunkBytes > 0) {
          void* peerBuff;
          NCCLCHECKGOTO(ncclDevrGetLsaRankPtr(
              comm, fields.peerWinHost, fields.peerWinOffset + byteOff, peerLsaRank, &peerBuff), ret, fail);
          if (peerBuff == NULL) {
            WARN("RMA CE: peerBuff is NULL after ncclDevrGetLsaRankPtr");
            ret = ncclInvalidArgument;
            goto fail;
          }
          dataParams.srcs[dataParams.numOps]  = (char*)fields.srcBuff + byteOff;
          dataParams.dsts[dataParams.numOps]  = peerBuff;
          dataParams.sizes[dataParams.numOps] = chunkBytes;
          dataParams.numOps++;
        }

        if (signalMode != NCCL_SIGNAL_NONE) {
          size_t signalOffset = ncclRmaSignalOffset(lsaSize, fields.sigIdx, lsaSelf);
          void* peerGraphSignal;
          NCCLCHECKGOTO(ncclDevrGetLsaRankPtr(comm, ceCtx->signalsWin, ceCtx->graphSignalOffset + signalOffset,
                                            peerLsaRank, &peerGraphSignal), ret, fail);
          signalParams.srcs[signalParams.numOps]  = ceCtx->signalConstOneDev;
          signalParams.dsts[signalParams.numOps]  = peerGraphSignal;
          signalParams.sizes[signalParams.numOps] = sizeof(uint64_t);
          signalParams.numOps++;
        }

        NCCLCHECKGOTO(ncclCeLaunchBatchOps(comm, &dataParams, stream, profArgs), ret, fail);
        NCCLCHECKGOTO(ncclCeLaunchBatchOps(comm, &signalParams, stream, profArgs), ret, fail);
      }
    }
  }

exit:
  ncclCeFreeBatchOpsParams(&dataParams);
  ncclCeFreeBatchOpsParams(&signalParams);
  return ret;
fail:
  goto exit;
}

struct rmaSchedCeOp {
  struct rmaSchedCeOp* next;
  int ctx;
  int peer;
  int sigIdx;
  const void* srcBuff;
  struct ncclDevrWindow* peerWinHost;
  size_t peerWinOffset;
  size_t byteOff;
  size_t nbytes;
  ncclSignalMode_t signalMode;
};

static ncclResult_t rmaSchedCePutLaunchNonPersist(struct ncclComm* comm, struct ncclRmaSchedStreamPlan* streamPlan,
                                                  cudaStream_t stream, struct ncclCeCollArgs* profArgs) {
  ncclResult_t ret = ncclSuccess;
  int nRmaCtx = comm->config.numRmaCtx;
  int lsaSize = comm->devrState.lsaTeamSize;
  int lsaSelf = comm->devrState.lsaTeamSelf;
  int nOps = 0;
  int opi = 0;
  // A batched memcpy does not guarantee order of execution within the batch. Data and signal
  // ops therefore go in separate batches (data first), and each batch carries at most one task
  // per peer so no two sequence-number writes target the same signal slot in one batch.
  struct ncclIntruQueue<struct rmaSchedCeOp, &rmaSchedCeOp::next>* peerTaskQueues = nullptr;
  struct rmaSchedCeOp* ops = nullptr;
  int* activePeers = nullptr;
  struct ncclCeBatchOpsParams dataParams = {};
  struct ncclCeBatchOpsParams signalParams = {};
  CUstreamBatchMemOpParams* seqStageOps = nullptr;
  int nActivePeers = 0;
  int nQueuedPeers = 0;
  int selfPeer = -1;

  for (int ctx = 0; ctx < nRmaCtx; ctx++) {
    struct ncclIntruQueue<struct ncclTaskTuningInfo, &ncclTaskTuningInfo::next>* queue =
      &streamPlan->ctxPlans[ctx].putSignalCE;
    for (struct ncclTaskTuningInfo* taskInfo = ncclIntruQueueHead(queue); taskInfo != nullptr;
         taskInfo = taskInfo->next) {
      struct rmaSchedPutFields fields = {};
      NCCLCHECKGOTO(rmaSchedPutFieldsFromTuningInfo(comm, taskInfo, &fields), ret, fail);
      nOps += rmaSchedPutChunkCount(fields.count * ncclTypeSize(fields.datatype));
    }
  }
  if (nOps == 0) goto exit;

  NCCLCHECKGOTO(ncclCalloc(&peerTaskQueues, comm->nRanks), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&ops, nOps), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&activePeers, comm->nRanks), ret, fail);
  NCCLCHECKGOTO(ncclCeInitBatchOpsParams(&dataParams, comm->nRanks), ret, fail);
  NCCLCHECKGOTO(ncclCeInitBatchOpsParams(&signalParams, comm->nRanks), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&seqStageOps, comm->nRanks), ret, fail);

  for (int ctx = 0; ctx < nRmaCtx; ctx++) {
    struct ncclIntruQueue<struct ncclTaskTuningInfo, &ncclTaskTuningInfo::next>* queue =
      &streamPlan->ctxPlans[ctx].putSignalCE;
    while (!ncclIntruQueueEmpty(queue)) {
      struct ncclTaskTuningInfo* taskInfo = ncclIntruQueueDequeue(queue);
      struct rmaSchedPutFields fields = {};
      size_t bytes;
      int nChunks;

      ret = rmaSchedPutFieldsFromTuningInfo(comm, taskInfo, &fields);
      rmaSchedFreeTuningInfoRaw(comm, taskInfo);
      if (ret != ncclSuccess) goto fail;
      if (fields.peer < 0 || fields.peer >= comm->nRanks) {
        WARN("RMA CE: task has invalid peer %d", fields.peer);
        ret = ncclInternalError;
        goto fail;
      }
      if (fields.ctx < 0 || fields.ctx >= nRmaCtx) {
        WARN("RMA CE: task has invalid context %d", fields.ctx);
        ret = ncclInternalError;
        goto fail;
      }
      bytes = fields.count * ncclTypeSize(fields.datatype);
      nChunks = rmaSchedPutChunkCount(bytes);
      for (int c = 0; c < nChunks; c++) {
        struct rmaSchedCeOp* op = &ops[opi++];
        op->ctx = fields.ctx;
        op->peer = fields.peer;
        op->sigIdx = fields.sigIdx;
        op->srcBuff = fields.srcBuff;
        op->peerWinHost = fields.peerWinHost;
        op->peerWinOffset = fields.peerWinOffset;
        op->byteOff = (size_t)c * rmaSchedPutChunkSize;
        op->nbytes = rmaSchedPutChunkBytes(bytes, c);
        op->signalMode = (c == nChunks - 1) ? fields.signalMode : NCCL_SIGNAL_NONE;
        if (ncclIntruQueueEmpty(&peerTaskQueues[op->peer])) nQueuedPeers++;
        ncclIntruQueueEnqueue(&peerTaskQueues[op->peer], op);
      }
    }
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
    int nRemotePeers = 0;
    size_t minRemoteBytes = (size_t)-1;
    size_t maxRemoteBytes = 0;
    dataParams.numOps = 0;
    dataParams.chunking = false;
    signalParams.numOps = 0;

    for (int i = 0; i < nActivePeers; i++) {
      int peer = activePeers[i];
      struct rmaSchedCeOp* op = ncclIntruQueueDequeue(&peerTaskQueues[peer]);
      struct ncclRmaCeCtx* ceCtx = (struct ncclRmaCeCtx*)comm->rmaState.rmaCeState.rmaCeCtxs[op->ctx];
      int peerLsaRank;

      NCCLCHECKGOTO(ncclDevrWorldToLsaRank(comm, op->peer, &peerLsaRank), ret, fail);
      if (peer != selfPeer) {
        nRemotePeers++;
        if (op->nbytes < minRemoteBytes) minRemoteBytes = op->nbytes;
        if (op->nbytes > maxRemoteBytes) maxRemoteBytes = op->nbytes;
      }

      if (op->nbytes > 0) {
        void* peerBuff;
        NCCLCHECKGOTO(ncclDevrGetLsaRankPtr(
            comm, op->peerWinHost, op->peerWinOffset + op->byteOff, peerLsaRank, &peerBuff), ret, fail);
        if (peerBuff == NULL) {
          WARN("RMA CE: peerBuff is NULL after ncclDevrGetLsaRankPtr");
          ret = ncclInvalidArgument;
          goto fail;
        }
        dataParams.srcs[dataParams.numOps]  = (char*)op->srcBuff + op->byteOff;
        dataParams.dsts[dataParams.numOps]  = peerBuff;
        dataParams.sizes[dataParams.numOps] = op->nbytes;
        dataParams.numOps++;
      }

      if (op->signalMode != NCCL_SIGNAL_NONE) {
        size_t signalOffset = ncclRmaSignalOffset(lsaSize, op->sigIdx, lsaSelf);
        size_t signalSlot = ncclRmaSignalSlot(lsaSize, op->sigIdx, peerLsaRank);
        void* peerSignal;
        if (nSeqStageOps >= comm->nRanks) {
          WARN("RMA CE: staged signal ops (%d) exceed staging capacity (%d)", nSeqStageOps, comm->nRanks);
          ret = ncclInternalError;
          goto fail;
        }
        NCCLCHECKGOTO(ncclDevrGetLsaRankPtr(
            comm, ceCtx->signalsWin, ceCtx->signalOffset + signalOffset, peerLsaRank, &peerSignal), ret, fail);
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

    // Only dense heterogeneous RMA data batches request CE chunking. The CE
    // launcher owns the chunk size and round-robin wave construction.
    dataParams.chunking = nRemotePeers > 0 && (nRemotePeers == lsaSize - 1) && minRemoteBytes != maxRemoteBytes;

    // Issue in stream order: staging writes feed the signal memcpys, and the data batch
    // must complete before the signals that advertise it.
    if (nSeqStageOps > 0) {
      NCCLCHECKGOTO(ncclCuStreamBatchMemOp(stream, nSeqStageOps, seqStageOps), ret, fail);
    }
    NCCLCHECKGOTO(ncclCeLaunchBatchOps(comm, &dataParams, stream, profArgs), ret, fail);
    NCCLCHECKGOTO(ncclCeLaunchBatchOps(comm, &signalParams, stream, profArgs), ret, fail);
    nActivePeers = nNextActivePeers;
  }

exit:
  ncclCeFreeBatchOpsParams(&dataParams);
  ncclCeFreeBatchOpsParams(&signalParams);
  free(ops);
  free(peerTaskQueues);
  free(activePeers);
  free(seqStageOps);
  return ret;
fail:
  for (int ctx = 0; ctx < nRmaCtx; ctx++) {
    struct ncclIntruQueue<struct ncclTaskTuningInfo, &ncclTaskTuningInfo::next>* queue =
      &streamPlan->ctxPlans[ctx].putSignalCE;
    while (!ncclIntruQueueEmpty(queue)) {
      struct ncclTaskTuningInfo* taskInfo = ncclIntruQueueDequeue(queue);
      rmaSchedFreeTuningInfoRaw(comm, taskInfo);
    }
  }
  goto exit;
}

static ncclResult_t rmaSchedCePutLaunch(
    struct ncclComm* comm, struct ncclRmaSchedStreamPlan* streamPlan, cudaStream_t stream) {
  int nRmaCtx = comm->config.numRmaCtx;
  int nTasks = 0;

  for (int ctx = 0; ctx < nRmaCtx; ctx++) nTasks += streamPlan->ctxPlans[ctx].putSignalCE.nElems;
  if (nTasks == 0) return ncclSuccess;

  if (!comm->rmaState.rmaCeState.initialized) {
    WARN("RMA CE is not initialized");
    return ncclInternalError;
  }

  ncclResult_t ret = ncclSuccess;
  struct ncclCeCollArgs profArgs;
  rmaSchedCeCollArgsInit(ncclFuncPutSignal, &profArgs);
  for (int ctx = 0; ctx < nRmaCtx; ctx++) rmaSchedCeCollArgsAddQueue(&streamPlan->ctxPlans[ctx].putSignalCE, &profArgs);
  NCCLCHECK(ncclProfilerStartCeCollEvent(comm, &profArgs, stream));
  if (streamPlan->captured) {
    ret = rmaSchedCePutLaunchPersist(comm, streamPlan, stream, &profArgs);
  } else {
    ret = rmaSchedCePutLaunchNonPersist(comm, streamPlan, stream, &profArgs);
  }
  ncclProfilerStopCeCollEvent(comm, &profArgs, stream);
  return ret;
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
  struct ncclCeCollArgs profArgs;
  int nElems = queue->nElems;
  cudaStream_t stream = streamPlan->stream;
  bool captured = streamPlan->captured;

  if (nElems == 0) return ncclSuccess;

  if (!comm->rmaState.rmaCeState.initialized) {
    WARN("RMA CE is not initialized");
    return ncclInternalError;
  }

  rmaSchedCeCollArgsInit(ncclFuncWaitSignal, &profArgs);
  rmaSchedCeCollArgsAddQueue(queue, &profArgs);
  NCCLCHECK(ncclProfilerStartCeCollEvent(comm, &profArgs, stream));

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
          NCCLCHECKGOTO(ncclCeLaunchBatchOps(comm, &ceParams, stream, &profArgs), ret, fail);
        }
      }
    }
  }

exit:
  ncclProfilerStopCeCollEvent(comm, &profArgs, stream);
  if (batchParams) free(batchParams);
  ncclCeFreeBatchOpsParams(&ceParams);
  return ret;
fail:
  if (batchParams) free(batchParams);
  ncclCeFreeBatchOpsParams(&ceParams);
  goto exit;
}

static ncclResult_t rmaPutLaunch(
    struct ncclComm* comm, struct ncclRmaSchedStreamPlan* streamPlan) {
  ncclResult_t ret = ncclSuccess;
  int nRmaCtx = comm->config.numRmaCtx;
  int nProxy = 0;
  int nCe = 0;

  for (int ctx = 0; ctx < nRmaCtx; ctx++) {
    nProxy += streamPlan->ctxPlans[ctx].putSignalProxy.nElems;
    nCe += streamPlan->ctxPlans[ctx].putSignalCE.nElems;
  }
  if (nProxy == 0 && nCe == 0) return ncclSuccess;

  // Proxy and CE puts in one group run together. A captured launch stays on the
  // captured stream so both sides are recorded in the graph. An eager launch
  // forks CE onto its own stream, matching ncclRmaPut.
  if (nProxy > 0 && nCe > 0 && !streamPlan->captured) {
    cudaStream_t ceStream = comm->rmaState.rmaCeState.ceStream;
    cudaEvent_t ceEvent = comm->rmaState.rmaCeState.ceEvent;
    CUDACHECKGOTO(cudaEventRecord(ceEvent, streamPlan->stream), ret, fail);
    CUDACHECKGOTO(cudaStreamWaitEvent(ceStream, ceEvent, 0), ret, fail);
    NCCLCHECKGOTO(rmaSchedProxyPutLaunch(comm, streamPlan), ret, fail);
    NCCLCHECKGOTO(rmaSchedCePutLaunch(comm, streamPlan, ceStream), ret, fail);
    CUDACHECKGOTO(cudaEventRecord(ceEvent, ceStream), ret, fail);
    CUDACHECKGOTO(cudaStreamWaitEvent(streamPlan->stream, ceEvent, 0), ret, fail);
  } else {
    if (nProxy > 0) NCCLCHECKGOTO(rmaSchedProxyPutLaunch(comm, streamPlan), ret, fail);
    if (nCe > 0) NCCLCHECKGOTO(rmaSchedCePutLaunch(comm, streamPlan, streamPlan->stream), ret, fail);
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

// CUDA user-object destructors run on an internal thread and cannot call CUDA.
// Enqueue the token; ncclScheduleRmaTasks reclaims it on an NCCL thread.
static void CUDART_CB rmaSchedStreamPlanDestructor(void* token_) {
  struct ncclRmaProxyReclaimToken* token = (struct ncclRmaProxyReclaimToken*)token_;
  if (token == nullptr || token->comm == nullptr) return;
  ncclIntruQueueMpscEnqueue(&token->comm->rmaSchedCallbackQueue, token);
}

ncclResult_t ncclRmaSchedPollCallbacks(struct ncclComm* comm) {
  ncclResult_t result = ncclSuccess;
  struct ncclRmaProxyReclaimToken* token =
    ncclIntruQueueMpscDequeueAll(&comm->rmaSchedCallbackQueue, /*waitSome=*/false);
  while (token != nullptr) {
    struct ncclRmaProxyReclaimToken* next = token->next;
    ncclResult_t ret = ncclRmaProxyReclaimDesc(token->comm, token);
    free(token);
    if (ret != ncclSuccess) result = ret;
    token = next;
  }
  NCCLCHECK(result);
  return ncclSuccess;
}

static ncclResult_t launchRmaStreamPlan(
    struct ncclComm* comm,
    struct ncclRmaSchedStreamPlan* streamPlans
  ) {
  ncclResult_t ret = ncclSuccess;
  int nRmaCtx = comm->config.numRmaCtx;
  struct ncclRmaSchedStreamPlan* streamPlan = nullptr;

  for (streamPlan = streamPlans; streamPlan != nullptr; streamPlan = streamPlan->next) {
    NCCLCHECKGOTO(rmaPutLaunch(comm, streamPlan), ret, fail);

    for (int ctx = 0; ctx < nRmaCtx; ctx++) {
      NCCLCHECKGOTO(rmaWaitSignalLaunch(comm, ctx, streamPlan), ret, fail);
    }

    if (streamPlan->captured && streamPlan->reclaimToken != nullptr) {
      struct ncclRmaProxyReclaimToken* token = streamPlan->reclaimToken;
      streamPlan->reclaimToken = nullptr;
      ret = ncclCudaGraphAddDestructor(streamPlan->graph, rmaSchedStreamPlanDestructor, token);
      if (ret != ncclSuccess) {
        (void)ncclRmaProxyReclaimDesc(comm, token);
        free(token);
        goto fail;
      }
    }
  }

exit:
  return ret;
fail:
  if (streamPlan != nullptr && streamPlan->reclaimToken != nullptr) {
    (void)ncclRmaProxyReclaimDesc(comm, streamPlan->reclaimToken);
    free(streamPlan->reclaimToken);
    streamPlan->reclaimToken = nullptr;
  }
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

  NCCLCHECK(ncclRmaSchedPollCallbacks(comm));

  if (!ncclIntruQueueEmpty(queue) && comm->config.numRmaCtx > 0) {
    struct ncclRmaSchedPlan rmaPlan = {};

    NCCLCHECK(scheduleRmaTasks(comm, queue, &rmaPlan));
    NCCLCHECK(launchRmaTasks(comm, &rmaPlan));
  }

  return ncclSuccess;
}
