/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "comm.h"
#include "alloc.h"
#include "checks.h"
#include "collectives.h"
#include "enqueue.h"
#include "strongstream.h"
#include "cudawrap.h"
#include "ce_coll.h"
#include "dev_runtime.h"
#include "gdrwrap.h"
#include "profiler.h"
#include "rma/rma_proxy.h"
#include "enqueue/task_sched.h"

static constexpr size_t CE_SCHED_HIER_MAX_CHUNK_SIZE = 64 * 1024 * 1024;

struct ncclCeSchedStreamPlan {
  struct ncclCeSchedStreamPlan* next;
  cudaStream_t stream;
  bool captured;
  struct ncclCudaGraph graph;
  struct ncclIntruQueue<struct ncclTaskTuningInfo, &ncclTaskTuningInfo::next> taskQueue;
};

struct ncclCeSchedPlan {
  struct ncclCeSchedStreamPlan* noncapturedStreamPlans;
  struct ncclCeSchedStreamPlan* capturedStreamPlans;
};

struct ceSchedHierChunkPlan {
  int nPeers;
  int* chunkStart;
  size_t* chunkBytes;
  size_t* chunkOff;
};

static void ceSchedFreeTuningInfoRaw(struct ncclComm* comm, struct ncclTaskTuningInfo* taskInfo) {
  if (taskInfo->raw != nullptr) {
    ncclMemoryPoolFree(&comm->memPool_ncclRawTask, taskInfo->raw);
    taskInfo->raw = nullptr;
  }
}

static cudaStream_t ceSchedTuningInfoStream(struct ncclTaskTuningInfo* tInfo) {
  return tInfo->raw->coll.stream;
}

static ncclResult_t ceSchedStreamCaptured(
    struct ncclComm* comm, cudaStream_t stream,
    bool* capturedOut, struct ncclCudaGraph* graphOut
  ) {
  NCCLCHECK(ncclCudaGetCapturingGraph(graphOut, stream, comm->config.graphUsageMode));
  *capturedOut = ncclCudaGraphValid(*graphOut);
  return ncclSuccess;
}

static struct ncclCeSchedStreamPlan* ceSchedFindStreamPlan(
    struct ncclCeSchedStreamPlan* head, cudaStream_t stream
  ) {
  for (struct ncclCeSchedStreamPlan* plan = head; plan != nullptr; plan = plan->next) {
    if (plan->stream == stream) return plan;
  }
  return nullptr;
}

static ncclResult_t ceSchedGetStreamPlan(
    struct ncclComm* comm,
    struct ncclCeSchedStreamPlan** head,
    cudaStream_t stream,
    bool captured,
    const struct ncclCudaGraph& graph,
    struct ncclCeSchedStreamPlan** streamPlanOut
  ) {
  struct ncclCeSchedStreamPlan* streamPlan = ceSchedFindStreamPlan(*head, stream);
  if (streamPlan != nullptr) {
    *streamPlanOut = streamPlan;
    return ncclSuccess;
  }

  streamPlan = ncclMemoryStackAlloc<struct ncclCeSchedStreamPlan>(&comm->memScoped);
  streamPlan->stream = stream;
  streamPlan->captured = captured;
  streamPlan->graph = graph;
  streamPlan->next = *head;
  *head = streamPlan;
  ncclIntruQueueConstruct(&streamPlan->taskQueue);
  *streamPlanOut = streamPlan;
  return ncclSuccess;
}

static ncclResult_t ceSchedCollArgsFromTuningInfo(
    struct ncclComm* comm, struct ncclTaskTuningInfo* tInfo,
    struct ncclCeCollArgs* args
  ) {
  struct ncclRawTaskColl* raw = &tInfo->raw->coll;

  if (tInfo->raw->kind != ncclTaskKindColl) return ncclInternalError;

  NCCLCHECK(ncclDevrFindWindow(comm, raw->sendbuff, &args->sendWin));
  NCCLCHECK(ncclDevrFindWindow(comm, raw->recvbuff, &args->recvWin));

  args->func = raw->func;
  args->rootRank = raw->root;
  args->datatype = raw->datatype;
  args->nElts = raw->count;
  args->eltSize = ncclTypeSize(raw->datatype);
  if (raw->func == ncclFuncAllGather || raw->func == ncclFuncBroadcast) {
    args->nElts *= args->eltSize;
    args->datatype = ncclInt8;
    args->eltSize = 1;
  }
  args->sendBuff = (uint8_t*)raw->sendbuff;
  args->recvBuff = (uint8_t*)raw->recvbuff;
  args->apiEventHandle = tInfo->raw->profiler.apiEventHandle;
  args->userTag = tInfo->raw->profiler.userTag;
  args->ceCollProfHandle = {};
  return ncclSuccess;
}

static ncclResult_t ceSchedProxyWaitBuildDesc(
    struct ncclComm* comm, struct ncclRmaProxyCtx* rmaProxyCtx,
    struct ncclCeSchedStreamPlan* streamPlan,
    int npeers, int** peers, int** nsignals,
    struct ncclRmaProxyDesc* desc) {
  ncclResult_t ret = ncclSuccess;
  bool captured = streamPlan->captured;

  desc->rmaDescType = ncclRmaDescTypeWaitSignal;
  desc->rmaDescState = ncclRmaDescStateReady;
  desc->waitSignal.npeers = npeers;
  // The internal collectives always put on signal index 0; the waiter polls the
  // matching slot, so the per-peer index array is all zeros.
  NCCLCHECKGOTO(ncclCalloc(&desc->waitSignal.waitSignalIdxs, npeers), ret, fail);
  desc->waitSignal.waitPeers = *peers;
  desc->waitSignal.waitSignals = *nsignals;
  *peers = nullptr;
  *nsignals = nullptr;
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

static ncclResult_t ceSchedProxyPutGroupBuildDesc(
    struct ncclComm* comm, struct ncclRmaProxyCtx* rmaProxyCtx,
    struct ncclCeSchedStreamPlan* streamPlan,
    int nOps, struct ncclRmaPutSignalOp** ops,
    int ctx,
    struct ncclRmaProxyDesc* desc) {
  ncclResult_t ret = ncclSuccess;
  bool captured = streamPlan->captured;
  int slot = comm->rank;
  (void)ctx;

  desc->rmaDescType = ncclRmaDescTypePutSignalGroup;
  desc->rmaDescState = ncclRmaDescStateReady;
  desc->persistPlan = nullptr;
  desc->persistDescValid = false;
  desc->putSignalGroup.nOps = nOps;
  desc->putSignalGroup.ops = *ops;
  desc->putSignalGroup.nIssued = 0;
  desc->putSignalGroup.nCompleted = 0;
  *ops = nullptr;

  if (!captured) {
    desc->opSeq = ++rmaProxyCtx->opSeqs[slot];
    desc->readySeq = &rmaProxyCtx->readySeqs[slot];
    desc->readySeqDev = &rmaProxyCtx->readySeqsDev[slot];
    desc->readySeqGdrHandle = rmaProxyCtx->readySeqsGdrHandle;
    desc->doneSeq = &rmaProxyCtx->doneSeqs[slot];
    desc->doneSeqDev = &rmaProxyCtx->doneSeqsDev[slot];
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

static ncclResult_t ceSchedHierCollBuildChunk(
    size_t perRankBytes, int nPeers, size_t maxChunk,
    struct ceSchedHierChunkPlan* outPlan) {
  ncclResult_t ret = ncclSuccess;
  const size_t align = 8 * 1024;

  outPlan->nPeers = nPeers;
  outPlan->chunkStart = nullptr;
  outPlan->chunkBytes = nullptr;
  outPlan->chunkOff = nullptr;

  int numChunks;
  size_t uniformSize, lastChunk;
  if (perRankBytes == 0 || maxChunk == 0 || perRankBytes <= maxChunk) {
    numChunks = 1;
    uniformSize = perRankBytes;
    lastChunk = perRankBytes;
  } else {
    numChunks = (int)((perRankBytes + maxChunk - 1) / maxChunk);
    uniformSize = (perRankBytes / numChunks / align) * align;
    if (uniformSize < align) uniformSize = align;
    lastChunk = perRankBytes - uniformSize * (numChunks - 1);
  }

  NCCLCHECKGOTO(ncclCalloc(&outPlan->chunkStart, nPeers + 1), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&outPlan->chunkBytes, nPeers * numChunks), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&outPlan->chunkOff, nPeers * numChunks), ret, fail);

  for (int p = 0; p <= nPeers; p++) {
    outPlan->chunkStart[p] = p * numChunks;
  }
  for (int p = 0; p < nPeers; p++) {
    size_t off = 0;
    for (int c = 0; c < numChunks; c++) {
      int idx = p * numChunks + c;
      size_t sz = (c == numChunks - 1) ? lastChunk : uniformSize;
      outPlan->chunkBytes[idx] = sz;
      outPlan->chunkOff[idx] = off;
      off += sz;
    }
  }
exit:
  return ret;
fail:
  free(outPlan->chunkStart);
  outPlan->chunkStart = nullptr;
  free(outPlan->chunkBytes);
  outPlan->chunkBytes = nullptr;
  free(outPlan->chunkOff);
  outPlan->chunkOff = nullptr;
  goto exit;
}

static void ceSchedHierCollFreeChunkPlan(struct ceSchedHierChunkPlan* plan) {
  if (plan == nullptr) return;
  free(plan->chunkStart);
  free(plan->chunkBytes);
  free(plan->chunkOff);
  plan->chunkStart = nullptr;
  plan->chunkBytes = nullptr;
  plan->chunkOff = nullptr;
  plan->nPeers = 0;
}

static ncclResult_t ceSchedRailSync(
    struct ncclComm* comm,
    struct ncclRmaProxyCtx* rmaProxyCtx,
    struct ncclCeSchedStreamPlan* streamPlan,
    int ctx,
    cudaStream_t stream) {
  ncclResult_t ret = ncclSuccess;
  int localRank = comm->localRank;
  int nNodes = comm->nNodes;
  int nRemoteNodes = nNodes - 1;
  bool captured = streamPlan->captured;

  if (nRemoteNodes <= 0) return ncclSuccess;

  int* railPeers = nullptr;
  int* railSigOnes = nullptr;
  struct ncclRmaPutSignalOp* groupOps = nullptr;
  struct ncclRmaProxyDesc* groupDesc = nullptr;
  struct ncclRmaProxyDesc* waitDesc = nullptr;
  CUstreamBatchMemOpParams* putBatch = nullptr;
  CUstreamBatchMemOpParams* waitBatch = nullptr;

  NCCLCHECKGOTO(ncclCalloc(&railPeers, nRemoteNodes), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&railSigOnes, nRemoteNodes), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&groupOps, nRemoteNodes), ret, fail);

  {
    int idx = 0;
    for (int n = 0; n < nNodes; n++) {
      if (n == comm->node) continue;
      int railPeer = comm->nodeRanks[n].localRankToRank[localRank];
      railPeers[idx] = railPeer;
      railSigOnes[idx] = 1;

      NCCLCHECKGOTO(ncclRmaProxyPutBuildOp(
          comm, rmaProxyCtx, ctx, captured,
          nullptr, 0, nullptr, 0,
          0, railPeer, /*signalIdx=*/0, NCCL_SIGNAL,
          &groupOps[idx]), ret, fail);
      idx++;
    }
  }

  NCCLCHECKGOTO(ncclCalloc(&groupDesc, 1), ret, fail);
  NCCLCHECKGOTO(ceSchedProxyPutGroupBuildDesc(
      comm, rmaProxyCtx, streamPlan, nRemoteNodes, &groupOps, ctx, groupDesc), ret, fail);

  NCCLCHECKGOTO(ncclCalloc(&waitDesc, 1), ret, fail);
  NCCLCHECKGOTO(ceSchedProxyWaitBuildDesc(
      comm, rmaProxyCtx, streamPlan, nRemoteNodes, &railPeers, &railSigOnes, waitDesc), ret, fail);

  {
    int startOps = ncclRmaProxyPutGroupStartNumOps(captured);
    int doneOps = ncclRmaProxyPutGroupDoneNumOps(captured);
    int putBatchOps = startOps + doneOps;

    NCCLCHECKGOTO(ncclCalloc(&putBatch, putBatchOps), ret, fail);
    NCCLCHECKGOTO(ncclRmaProxyPutGroupStartParams(groupDesc, &putBatch[0]), ret, fail);
    NCCLCHECKGOTO(ncclRmaProxyPutGroupDoneParams(groupDesc, &putBatch[startOps]), ret, fail);

    NCCLCHECKGOTO(ncclRmaProxyEnqueueDesc(rmaProxyCtx, &groupDesc), ret, fail);
    NCCLCHECKGOTO(ncclCuStreamBatchMemOp(stream, putBatchOps, putBatch), ret, fail);
  }

  {
    int waitOps = ncclRmaProxyWaitNumStreamOps(waitDesc);
    NCCLCHECKGOTO(ncclCalloc(&waitBatch, waitOps), ret, fail);
    NCCLCHECKGOTO(ncclRmaProxyWaitParams(rmaProxyCtx, waitDesc, waitBatch), ret, fail);
    NCCLCHECKGOTO(ncclRmaProxyEnqueueDesc(rmaProxyCtx, &waitDesc), ret, fail);
    NCCLCHECKGOTO(ncclCuStreamBatchMemOp(stream, waitOps, waitBatch), ret, fail);
  }

exit:
  free(putBatch);
  free(waitBatch);
  if (groupDesc != nullptr) (void)ncclRmaProxyDestroyDesc(comm, &groupDesc);
  if (waitDesc != nullptr) (void)ncclRmaProxyDestroyDesc(comm, &waitDesc);
  free(groupOps);
  free(railPeers);
  free(railSigOnes);
  return ret;
fail:
  goto exit;
}

static ncclResult_t ceSchedProxyWaitOnePeer(
    struct ncclComm* comm,
    struct ncclRmaProxyCtx* rmaProxyCtx,
    struct ncclCeSchedStreamPlan* streamPlan,
    int ctx,
    cudaStream_t stream,
    int peer,
    int nsignals) {
  ncclResult_t ret = ncclSuccess;
  int* waitPeers = nullptr;
  int* waitSigCounts = nullptr;
  struct ncclRmaProxyDesc* waitDesc = nullptr;
  CUstreamBatchMemOpParams* waitBatch = nullptr;

  NCCLCHECKGOTO(ncclCalloc(&waitPeers, 1), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&waitSigCounts, 1), ret, fail);
  waitPeers[0] = peer;
  waitSigCounts[0] = nsignals;

  NCCLCHECKGOTO(ncclCalloc(&waitDesc, 1), ret, fail);
  NCCLCHECKGOTO(ceSchedProxyWaitBuildDesc(
      comm, rmaProxyCtx, streamPlan, 1, &waitPeers, &waitSigCounts, waitDesc), ret, fail);

  {
    int waitOps = ncclRmaProxyWaitNumStreamOps(waitDesc);
    NCCLCHECKGOTO(ncclCalloc(&waitBatch, waitOps), ret, fail);
    NCCLCHECKGOTO(ncclRmaProxyWaitParams(rmaProxyCtx, waitDesc, waitBatch), ret, fail);
    NCCLCHECKGOTO(ncclRmaProxyEnqueueDesc(rmaProxyCtx, &waitDesc), ret, fail);
    NCCLCHECKGOTO(ncclCuStreamBatchMemOp(stream, waitOps, waitBatch), ret, fail);
  }

exit:
  free(waitBatch);
  if (waitDesc != nullptr) (void)ncclRmaProxyDestroyDesc(comm, &waitDesc);
  free(waitPeers);
  free(waitSigCounts);
  return ret;
fail:
  goto exit;
}

static ncclResult_t ceSchedHierCeAllGather(
    struct ncclComm* comm, struct ncclCeSchedStreamPlan* streamPlan,
    struct ncclCeCollArgs* args, cudaStream_t stream) {
  ncclResult_t ret = ncclSuccess;

  int ctx = 0;
  int myRank = comm->rank;
  int localRank = comm->localRank;
  int nNodes = comm->nNodes;
  int nRemoteNodes = nNodes - 1;
  int myLsaRank = comm->devrState.lsaTeamSelf;
  int lsaSize = comm->devrState.lsaTeamSize;
  bool captured = streamPlan->captured;

  const void* sendbuff = args->sendBuff;
  void* recvbuff = args->recvBuff;
  struct ncclDevrWindow* sendWin = args->sendWin;
  struct ncclDevrWindow* recvWin = args->recvWin;
  size_t perRankBytes = args->nElts * args->eltSize;

  struct ncclRmaProxyCtx* rmaProxyCtx =
      (struct ncclRmaProxyCtx*)comm->rmaState.rmaProxyState.rmaProxyCtxs[ctx];

  struct ceSchedHierChunkPlan chunkPlan = {};
  struct ncclRmaProxyDesc* groupDesc = nullptr;
  struct ncclRmaPutSignalOp* groupOps = nullptr;
  CUstreamBatchMemOpParams* groupStartParam = nullptr;
  CUstreamBatchMemOpParams* groupDoneParam = nullptr;
  struct ncclCeBatchOpsParams ceBcastOps = {};
  struct ncclCeBatchOpsParams ceScatterOps = {};

  NCCLCHECKGOTO(ceSchedRailSync(comm, rmaProxyCtx, streamPlan, ctx, stream), ret, fail);

  {
    NCCLCHECKGOTO(ceSchedHierCollBuildChunk(
        perRankBytes, nRemoteNodes, CE_SCHED_HIER_MAX_CHUNK_SIZE, &chunkPlan), ret, fail);
    int totalOps = chunkPlan.chunkStart[chunkPlan.nPeers];

    int startOps = ncclRmaProxyPutGroupStartNumOps(captured);
    int doneOps = ncclRmaProxyPutGroupDoneNumOps(captured);
    NCCLCHECKGOTO(ncclCalloc(&groupStartParam, startOps), ret, fail);
    NCCLCHECKGOTO(ncclCalloc(&groupDoneParam, doneOps), ret, fail);

    size_t srcWinOffset = (const uint8_t*)sendbuff - (const uint8_t*)sendWin->userPtr;
    size_t peerWinOffset = ((const uint8_t*)recvbuff + myRank * perRankBytes) - (const uint8_t*)recvWin->userPtr;

    NCCLCHECKGOTO(ncclCalloc(&groupDesc, 1), ret, fail);
    NCCLCHECKGOTO(ncclCalloc(&groupOps, totalOps), ret, fail);

    for (int s = 1; s < nNodes; s++) {
      int p = s - 1;
      int n = (comm->node + s) % nNodes;
      int railPeer = comm->nodeRanks[n].localRankToRank[localRank];

      for (int c = chunkPlan.chunkStart[p]; c < chunkPlan.chunkStart[p+1]; c++) {
        size_t subBytes = chunkPlan.chunkBytes[c];
        size_t off = chunkPlan.chunkOff[c];

        NCCLCHECKGOTO(ncclRmaProxyPutBuildOp(
            comm, rmaProxyCtx, ctx, captured,
            sendWin, srcWinOffset + off,
            recvWin, peerWinOffset + off,
            subBytes, railPeer, /*signalIdx=*/0, NCCL_SIGNAL,
            &groupOps[c]), ret, fail);
      }
    }

    NCCLCHECKGOTO(ceSchedProxyPutGroupBuildDesc(
        comm, rmaProxyCtx, streamPlan, totalOps, &groupOps, ctx, groupDesc), ret, fail);

    NCCLCHECKGOTO(ncclRmaProxyPutGroupStartParams(groupDesc, groupStartParam), ret, fail);
    NCCLCHECKGOTO(ncclRmaProxyPutGroupDoneParams(groupDesc, groupDoneParam), ret, fail);

    NCCLCHECKGOTO(ncclRmaProxyEnqueueDesc(rmaProxyCtx, &groupDesc), ret, fail);
    NCCLCHECKGOTO(ncclCuStreamBatchMemOp(stream, startOps, groupStartParam), ret, fail);
  }

  NCCLCHECKGOTO(ncclMemOpSync(comm, stream, args), ret, fail);

  NCCLCHECKGOTO(ncclCeInitBatchOpsParams(&ceBcastOps, lsaSize), ret, fail);
  {
    uint8_t* myRecvSlot = (uint8_t*)recvbuff + myRank * perRankBytes;
    size_t offset = myRecvSlot - (uint8_t*)recvWin->userPtr;

    if (myRecvSlot != (const uint8_t*)sendbuff) {
      ceBcastOps.srcs[ceBcastOps.numOps] = (void*)sendbuff;
      ceBcastOps.dsts[ceBcastOps.numOps] = (void*)myRecvSlot;
      ceBcastOps.sizes[ceBcastOps.numOps] = perRankBytes;
      ceBcastOps.numOps++;
    }

    for (int r = 1; r < lsaSize; r++) {
      int targetLsaRank = (myLsaRank + r) % lsaSize;
      void* peerBuf;
      NCCLCHECKGOTO(ncclDevrGetLsaRankPtr(comm, recvWin, offset, targetLsaRank, &peerBuf), ret, fail);
      ceBcastOps.srcs[ceBcastOps.numOps] = (void*)sendbuff;
      ceBcastOps.dsts[ceBcastOps.numOps] = peerBuf;
      ceBcastOps.sizes[ceBcastOps.numOps] = perRankBytes;
      ceBcastOps.numOps++;
    }

    NCCLCHECKGOTO(ncclCeLaunchBatchOps(comm, &ceBcastOps, stream, args), ret, fail);
  }

  {
    for (int s = 1; s < nNodes; s++) {
      int p = s - 1;
      int n = (comm->node - s + nNodes) % nNodes;
      int railPeer = comm->nodeRanks[n].localRankToRank[localRank];
      size_t peerSliceOffset = railPeer * perRankBytes;

      for (int c = chunkPlan.chunkStart[p]; c < chunkPlan.chunkStart[p+1]; c++) {
        size_t subBytes = chunkPlan.chunkBytes[c];
        size_t off = chunkPlan.chunkOff[c];

        uint8_t* chunkSlot = (uint8_t*)recvbuff + peerSliceOffset + off;
        size_t winOffset = chunkSlot - (uint8_t*)recvWin->userPtr;

        NCCLCHECKGOTO(ceSchedProxyWaitOnePeer(
            comm, rmaProxyCtx, streamPlan, ctx, stream, railPeer, 1), ret, fail);

        NCCLCHECKGOTO(ncclCeInitBatchOpsParams(&ceScatterOps, lsaSize), ret, fail);
        for (int r = 1; r < lsaSize; r++) {
          int targetLsaRank = (myLsaRank + r) % lsaSize;
          void* peerBuf;
          NCCLCHECKGOTO(ncclDevrGetLsaRankPtr(comm, recvWin, winOffset, targetLsaRank, &peerBuf), ret, fail);
          ceScatterOps.srcs[ceScatterOps.numOps] = chunkSlot;
          ceScatterOps.dsts[ceScatterOps.numOps] = peerBuf;
          ceScatterOps.sizes[ceScatterOps.numOps] = subBytes;
          ceScatterOps.numOps++;
        }

        NCCLCHECKGOTO(ncclCeLaunchBatchOps(comm, &ceScatterOps, stream, args), ret, fail);
        ncclCeFreeBatchOpsParams(&ceScatterOps);
      }
    }
  }

  {
    int doneOps = ncclRmaProxyPutGroupDoneNumOps(captured);
    NCCLCHECKGOTO(ncclCuStreamBatchMemOp(stream, doneOps, groupDoneParam), ret, fail);
  }

  NCCLCHECKGOTO(ncclMemOpSync(comm, stream, args), ret, fail);

exit:
  ncclCeFreeBatchOpsParams(&ceBcastOps);
  ncclCeFreeBatchOpsParams(&ceScatterOps);
  free(groupStartParam);
  free(groupDoneParam);
  free(groupOps);
  if (groupDesc != nullptr) (void)ncclRmaProxyDestroyDesc(comm, &groupDesc);
  ceSchedHierCollFreeChunkPlan(&chunkPlan);
  return ret;
fail:
  goto exit;
}

static ncclResult_t ceSchedHierCeAlltoAll(
    struct ncclComm* comm, struct ncclCeSchedStreamPlan* streamPlan,
    struct ncclCeCollArgs* args, cudaStream_t stream) {
  ncclResult_t ret = ncclSuccess;

  int ctx = 0;
  int myRank = comm->rank;
  int myNode = comm->node;
  int nNodes = comm->nNodes;
  int localRanks = comm->localRanks;
  int myLsaRank = comm->devrState.lsaTeamSelf;
  int lsaSize = comm->devrState.lsaTeamSize;
  int numRemotePeers = (nNodes - 1) * localRanks;
  bool captured = streamPlan->captured;

  const void* sendbuff = args->sendBuff;
  void* recvbuff = args->recvBuff;
  struct ncclDevrWindow* sendWin = args->sendWin;
  struct ncclDevrWindow* recvWin = args->recvWin;
  size_t perPeerBytes = args->nElts * args->eltSize;
  bool inPlace = (sendbuff == recvbuff);

  struct ncclRmaProxyCtx* rmaProxyCtx =
      (struct ncclRmaProxyCtx*)comm->rmaState.rmaProxyState.rmaProxyCtxs[ctx];

  struct ceSchedHierChunkPlan chunkPlan = {};
  struct ncclRmaProxyDesc* groupDesc = nullptr;
  struct ncclRmaPutSignalOp* groupOps = nullptr;
  CUstreamBatchMemOpParams* groupStartParam = nullptr;
  CUstreamBatchMemOpParams* groupDoneParam = nullptr;
  int* waitPeers = nullptr;
  int* waitSigCounts = nullptr;
  struct ncclRmaProxyDesc* waitDesc = nullptr;
  CUstreamBatchMemOpParams* waitBatch = nullptr;
  struct ncclCeBatchOpsParams ceLocalA2A = {};

  NCCLCHECKGOTO(ceSchedRailSync(comm, rmaProxyCtx, streamPlan, ctx, stream), ret, fail);
  NCCLCHECKGOTO(ncclMemOpSync(comm, stream, args), ret, fail);

  {
    NCCLCHECKGOTO(ceSchedHierCollBuildChunk(
        perPeerBytes, numRemotePeers, CE_SCHED_HIER_MAX_CHUNK_SIZE, &chunkPlan), ret, fail);
    int totalOps = chunkPlan.chunkStart[chunkPlan.nPeers];

    int startOps = ncclRmaProxyPutGroupStartNumOps(captured);
    int doneOps = ncclRmaProxyPutGroupDoneNumOps(captured);
    NCCLCHECKGOTO(ncclCalloc(&groupStartParam, startOps), ret, fail);
    NCCLCHECKGOTO(ncclCalloc(&groupDoneParam, doneOps), ret, fail);

    NCCLCHECKGOTO(ncclCalloc(&groupDesc, 1), ret, fail);
    NCCLCHECKGOTO(ncclCalloc(&groupOps, totalOps), ret, fail);

    int p = 0;
    for (int s = 1; s < nNodes; s++) {
      int n = (myNode + s) % nNodes;
      for (int lr = 0; lr < localRanks; lr++) {
        int peer = comm->nodeRanks[n].localRankToRank[lr];
        size_t srcWinOffset = ((const uint8_t*)sendbuff + (size_t)peer * perPeerBytes) - (const uint8_t*)sendWin->userPtr;
        size_t peerWinOffset = ((const uint8_t*)recvbuff + (size_t)myRank * perPeerBytes) - (const uint8_t*)recvWin->userPtr;

        for (int c = chunkPlan.chunkStart[p]; c < chunkPlan.chunkStart[p+1]; c++) {
          size_t subBytes = chunkPlan.chunkBytes[c];
          size_t off = chunkPlan.chunkOff[c];

          NCCLCHECKGOTO(ncclRmaProxyPutBuildOp(
              comm, rmaProxyCtx, ctx, captured,
              sendWin, srcWinOffset + off,
              recvWin, peerWinOffset + off,
              subBytes, peer, /*signalIdx=*/0, NCCL_SIGNAL,
              &groupOps[c]), ret, fail);
        }
        p++;
      }
    }

    NCCLCHECKGOTO(ceSchedProxyPutGroupBuildDesc(
        comm, rmaProxyCtx, streamPlan, totalOps, &groupOps, ctx, groupDesc), ret, fail);

    NCCLCHECKGOTO(ncclRmaProxyPutGroupStartParams(groupDesc, groupStartParam), ret, fail);
    NCCLCHECKGOTO(ncclRmaProxyPutGroupDoneParams(groupDesc, groupDoneParam), ret, fail);

    NCCLCHECKGOTO(ncclRmaProxyEnqueueDesc(rmaProxyCtx, &groupDesc), ret, fail);
    NCCLCHECKGOTO(ncclCuStreamBatchMemOp(stream, startOps, groupStartParam), ret, fail);
  }

  NCCLCHECKGOTO(ncclCeInitBatchOpsParams(&ceLocalA2A, lsaSize), ret, fail);
  {
    size_t myRecvOffset = ((const uint8_t*)recvbuff + (size_t)myRank * perPeerBytes) - (const uint8_t*)recvWin->userPtr;

    for (int k = 0; k < lsaSize; k++) {
      int targetLsa = (myLsaRank + k) % lsaSize;
      int targetWorldRank = comm->nodeRanks[myNode].localRankToRank[targetLsa];

      if (inPlace && targetLsa == myLsaRank) continue;

      void* peerRecvSlot;
      NCCLCHECKGOTO(ncclDevrGetLsaRankPtr(comm, recvWin, myRecvOffset, targetLsa, &peerRecvSlot), ret, fail);

      ceLocalA2A.srcs[ceLocalA2A.numOps] = (void*)((const uint8_t*)sendbuff + (size_t)targetWorldRank * perPeerBytes);
      ceLocalA2A.dsts[ceLocalA2A.numOps] = peerRecvSlot;
      ceLocalA2A.sizes[ceLocalA2A.numOps] = perPeerBytes;
      ceLocalA2A.numOps++;
    }

    NCCLCHECKGOTO(ncclCeLaunchBatchOps(comm, &ceLocalA2A, stream, args), ret, fail);
  }

  {
    NCCLCHECKGOTO(ncclCalloc(&waitPeers, numRemotePeers), ret, fail);
    NCCLCHECKGOTO(ncclCalloc(&waitSigCounts, numRemotePeers), ret, fail);

    int p = 0;
    for (int s = 1; s < nNodes; s++) {
      int n = (myNode - s + nNodes) % nNodes;
      for (int lr = 0; lr < localRanks; lr++) {
        waitPeers[p] = comm->nodeRanks[n].localRankToRank[lr];
        waitSigCounts[p] = chunkPlan.chunkStart[p+1] - chunkPlan.chunkStart[p];
        p++;
      }
    }

    NCCLCHECKGOTO(ncclCalloc(&waitDesc, 1), ret, fail);
    NCCLCHECKGOTO(ceSchedProxyWaitBuildDesc(
        comm, rmaProxyCtx, streamPlan, numRemotePeers, &waitPeers, &waitSigCounts, waitDesc), ret, fail);

    int waitOps = ncclRmaProxyWaitNumStreamOps(waitDesc);
    NCCLCHECKGOTO(ncclCalloc(&waitBatch, waitOps), ret, fail);
    NCCLCHECKGOTO(ncclRmaProxyWaitParams(rmaProxyCtx, waitDesc, waitBatch), ret, fail);
    NCCLCHECKGOTO(ncclRmaProxyEnqueueDesc(rmaProxyCtx, &waitDesc), ret, fail);
    NCCLCHECKGOTO(ncclCuStreamBatchMemOp(stream, waitOps, waitBatch), ret, fail);
  }

  {
    int doneOps = ncclRmaProxyPutGroupDoneNumOps(captured);
    NCCLCHECKGOTO(ncclCuStreamBatchMemOp(stream, doneOps, groupDoneParam), ret, fail);
  }

  NCCLCHECKGOTO(ncclMemOpSync(comm, stream, args), ret, fail);

exit:
  ncclCeFreeBatchOpsParams(&ceLocalA2A);
  free(groupStartParam);
  free(groupDoneParam);
  free(groupOps);
  if (groupDesc != nullptr) (void)ncclRmaProxyDestroyDesc(comm, &groupDesc);
  free(waitBatch);
  if (waitDesc != nullptr) (void)ncclRmaProxyDestroyDesc(comm, &waitDesc);
  free(waitPeers);
  free(waitSigCounts);
  ceSchedHierCollFreeChunkPlan(&chunkPlan);
  return ret;
fail:
  goto exit;
}

static ncclResult_t ceSchedLaunchColl(
    struct ncclComm* comm, struct ncclCeSchedStreamPlan* streamPlan,
    struct ncclCeCollArgs* args) {
  ncclResult_t ret = ncclSuccess;
  cudaStream_t stream = streamPlan->stream;

  NCCLCHECKGOTO(ncclProfilerStartCeCollEvent(comm, args, stream), ret, fail);

  if (!ncclDevrIsOneLsaTeam(comm)) {
    if (!comm->rmaState.rmaProxyState.connected) {
      WARN("RMA proxy is not connected");
      ret = ncclInternalError;
      goto fail;
    }
    switch (args->func) {
      case ncclFuncAllGather:
        NCCLCHECKGOTO(ceSchedHierCeAllGather(comm, streamPlan, args, stream), ret, fail);
        break;
      case ncclFuncAlltoAll:
        NCCLCHECKGOTO(ceSchedHierCeAlltoAll(comm, streamPlan, args, stream), ret, fail);
        break;
      default:
        WARN("Hierarchical CE collective not supported for %s", ncclFuncToString(args->func));
        ret = ncclInvalidUsage;
    }
  } else {
    switch (args->func) {
      case ncclFuncAllGather:
        NCCLCHECKGOTO(ncclCeAllGather(comm, args, stream), ret, fail);
        break;
      case ncclFuncAlltoAll:
        NCCLCHECKGOTO(ncclCeAlltoAll(comm, args, stream), ret, fail);
        break;
      case ncclFuncScatter:
        NCCLCHECKGOTO(ncclCeScatter(comm, args, stream), ret, fail);
        break;
      case ncclFuncGather:
        NCCLCHECKGOTO(ncclCeGather(comm, args, stream), ret, fail);
        break;
      default:
        WARN("CE collective not supported for %s", ncclFuncToString(args->func));
        ret = ncclInvalidUsage;
    }
  }

exit:
  ncclProfilerStopCeCollEvent(comm, args, stream);
  return ret;
fail:
  goto exit;
}

static void CUDART_CB ceSchedStreamPlanDestructor(void* graph_) {
  struct ncclCudaGraph* graph = (struct ncclCudaGraph*)graph_;
  (void)graph;
  free(graph);
}

static ncclResult_t launchCeStreamPlan(
    struct ncclComm* comm,
    struct ncclCeSchedStreamPlan* streamPlans,
    struct ncclStrongStream* depStream,
    struct ncclStrongStream* launchOrderStream
  ) {
  ncclResult_t ret = ncclSuccess;

  for (struct ncclCeSchedStreamPlan* streamPlan = streamPlans; streamPlan != nullptr; streamPlan = streamPlan->next) {
    cudaStream_t launchStream = streamPlan->stream;
    cudaEvent_t finishedEvent = comm->sharedRes->scratchEvent;
    cudaStream_t acqDepStream;
    cudaStream_t acqLaunchOrderStream;

    if (depStream) {
      NCCLCHECKGOTO(ncclStrongStreamAcquire(streamPlan->graph, depStream, false, &acqDepStream), ret, fail);
      NCCLCHECKGOTO(ncclStreamWaitStream(launchStream, acqDepStream, finishedEvent), ret, fail);
    }

    if (launchOrderStream) {
      NCCLCHECKGOTO(ncclStrongStreamAcquire(streamPlan->graph, launchOrderStream, false, &acqLaunchOrderStream), ret, fail);
      NCCLCHECKGOTO(ncclStreamWaitStream(launchStream, acqLaunchOrderStream, finishedEvent), ret, fail);
    }

    while (!ncclIntruQueueEmpty(&streamPlan->taskQueue)) {
      struct ncclTaskTuningInfo* taskInfo = ncclIntruQueueDequeue(&streamPlan->taskQueue);
      struct ncclCeCollArgs args = {};

      NCCLCHECKGOTO(ceSchedCollArgsFromTuningInfo(comm, taskInfo, &args), ret, fail);
      NCCLCHECKGOTO(ceSchedLaunchColl(comm, streamPlan, &args), ret, fail);
      ceSchedFreeTuningInfoRaw(comm, taskInfo);
    }

    if (depStream || launchOrderStream) {
      CUDACHECKGOTO(cudaEventRecord(finishedEvent, launchStream), ret, fail);
    }
    if (depStream) {
      NCCLCHECKGOTO(ncclStreamAdvanceToEvent(streamPlan->graph, acqDepStream, finishedEvent), ret, fail);
      NCCLCHECKGOTO(ncclStrongStreamRelease(streamPlan->graph, depStream, false), ret, fail);
    }
    if (launchOrderStream) {
      NCCLCHECKGOTO(ncclStreamAdvanceToEvent(streamPlan->graph, acqLaunchOrderStream, finishedEvent), ret, fail);
      NCCLCHECKGOTO(ncclStrongStreamRelease(streamPlan->graph, launchOrderStream, false), ret, fail);
    }
    if (streamPlan->captured) {
      struct ncclCudaGraph* graphArg = nullptr;
      NCCLCHECKGOTO(ncclCalloc(&graphArg, 1), ret, fail);
      *graphArg = streamPlan->graph;
      ret = ncclCudaGraphAddDestructor(
          streamPlan->graph, ceSchedStreamPlanDestructor, (void*)graphArg);
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

static ncclResult_t launchCeTasks(
    struct ncclComm* comm,
    struct ncclCeSchedPlan* cePlan,
    struct ncclStrongStream* depStream,
    struct ncclStrongStream* launchOrderStream
  ) {
  ncclResult_t ret = ncclSuccess;

  NCCLCHECKGOTO(launchCeStreamPlan(comm, cePlan->noncapturedStreamPlans, depStream, launchOrderStream), ret, fail);
  NCCLCHECKGOTO(launchCeStreamPlan(comm, cePlan->capturedStreamPlans, depStream, launchOrderStream), ret, fail);

exit:
  return ret;
fail:
  goto exit;
}

static ncclResult_t scheduleCeTasks(
    struct ncclComm* comm,
    struct ncclIntruQueue<struct ncclTaskTuningInfo, &ncclTaskTuningInfo::next>* queue,
    struct ncclCeSchedPlan* cePlanOut
  ) {
  cePlanOut->noncapturedStreamPlans = nullptr;
  cePlanOut->capturedStreamPlans = nullptr;

  while (!ncclIntruQueueEmpty(queue)) {
    struct ncclTaskTuningInfo* taskInfo = ncclIntruQueueDequeue(queue);
    cudaStream_t stream = ceSchedTuningInfoStream(taskInfo);
    struct ncclCeSchedStreamPlan* streamPlan = nullptr;
    struct ncclCeSchedStreamPlan** streamListHead = nullptr;
    bool captured = false;
    struct ncclCudaGraph graph = {};

    NCCLCHECK(ceSchedStreamCaptured(comm, stream, &captured, &graph));
    streamListHead = captured ? &cePlanOut->capturedStreamPlans : &cePlanOut->noncapturedStreamPlans;
    NCCLCHECK(ceSchedGetStreamPlan(comm, streamListHead, stream, captured, graph, &streamPlan));
    ncclIntruQueueEnqueue(&streamPlan->taskQueue, taskInfo);
  }

  return ncclSuccess;
}

ncclResult_t ncclScheduleCeTasks(
    struct ncclComm* comm,
    struct ncclIntruQueue<struct ncclTaskTuningInfo, &ncclTaskTuningInfo::next>* queue,
    struct ncclStrongStream* depStream,
    struct ncclStrongStream* launchOrderStream
  ) {
  if (ncclIntruQueueEmpty(queue)) return ncclSuccess;

  if (!comm->ceColl.initialized) {
    WARN("CE collective is not initialized");
    return ncclInternalError;
  }

  struct ncclCeSchedPlan cePlan = {};
  NCCLCHECK(scheduleCeTasks(comm, queue, &cePlan));
  NCCLCHECK(launchCeTasks(comm, &cePlan, depStream, launchOrderStream));
  return ncclSuccess;
}
