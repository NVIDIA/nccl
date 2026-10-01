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
static constexpr size_t CE_SCHED_HIER_CHUNK_ALIGN = 8 * 1024;
static constexpr size_t CE_SCHED_HIER_AG_RING_HALF_AWARE_THRESHOLD = 128 * 1024 * 1024;
static constexpr size_t CE_SCHED_HIER_AG_RING_MIN_HALF_AWARE_CHUNK = 8 * 1024 * 1024;
static constexpr int64_t CE_SCHED_HIER_AG_RING_MULTI_CTX_THRESHOLD_DEFAULT = 4 * 1024 * 1024;

// Defined in ce_coll.cc so both launch paths read the same settings.
int64_t ncclParamHierCeCollAgRailRingEnable();
int64_t ncclParamRmaMultiCtxThreshold();
int64_t ncclParamHierCeCollNumCtx();

struct ncclCeSchedStreamPlan {
  struct ncclCeSchedStreamPlan* next;
  cudaStream_t stream;
  bool captured;
  struct ncclCudaGraph graph;
  // Set while a captured launch owns proxy descriptors. Moved to the graph
  // destructor, which frees those descriptors.
  struct ncclRmaProxyReclaimToken* reclaimToken;
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

static ncclResult_t ceSchedStreamCaptured(struct ncclComm* comm, cudaStream_t stream, bool* capturedOut,
                                          struct ncclCudaGraph* graphOut) {
  NCCLCHECK(ncclCudaGetCapturingGraph(graphOut, stream, comm->config.graphUsageMode));
  *capturedOut = ncclCudaGraphValid(*graphOut);
  return ncclSuccess;
}

static struct ncclCeSchedStreamPlan* ceSchedFindStreamPlan(struct ncclCeSchedStreamPlan* head, cudaStream_t stream) {
  for (struct ncclCeSchedStreamPlan* plan = head; plan != nullptr; plan = plan->next) {
    if (plan->stream == stream) return plan;
  }
  return nullptr;
}

static ncclResult_t ceSchedGetStreamPlan(struct ncclComm* comm, struct ncclCeSchedStreamPlan** head,
                                         cudaStream_t stream, bool captured, const struct ncclCudaGraph& graph,
                                         struct ncclCeSchedStreamPlan** streamPlanOut) {
  struct ncclCeSchedStreamPlan* streamPlan = ceSchedFindStreamPlan(*head, stream);
  if (streamPlan != nullptr) {
    *streamPlanOut = streamPlan;
    return ncclSuccess;
  }

  streamPlan = ncclMemoryStackAlloc<struct ncclCeSchedStreamPlan>(&comm->memScoped);
  streamPlan->stream = stream;
  streamPlan->captured = captured;
  streamPlan->graph = graph;
  streamPlan->reclaimToken = nullptr;
  streamPlan->next = *head;
  *head = streamPlan;
  ncclIntruQueueConstruct(&streamPlan->taskQueue);
  *streamPlanOut = streamPlan;
  return ncclSuccess;
}

static ncclResult_t ceSchedCollArgsFromTuningInfo(struct ncclComm* comm, struct ncclTaskTuningInfo* tInfo,
                                                  struct ncclCeCollArgs* args) {
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

static ncclResult_t ceSchedCapturedToken(struct ncclComm* comm, struct ncclCeSchedStreamPlan* streamPlan) {
  if (streamPlan->reclaimToken != nullptr) return ncclSuccess;
  NCCLCHECK(ncclCalloc(&streamPlan->reclaimToken, 1));
  streamPlan->reclaimToken->comm = comm;
  return ncclSuccess;
}

static ncclResult_t ceSchedProxyWaitBuildDesc(struct ncclComm* comm, struct ncclRmaProxyCtx* rmaProxyCtx,
                                              struct ncclCeSchedStreamPlan* streamPlan, int npeers, int** peers,
                                              int** nsignals, struct ncclRmaProxyDesc* desc) {
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
    NCCLCHECKGOTO(allocMemCPUAccessible(&desc->readySeq, &desc->readySeqDev, 1, 0, &desc->readySeqGdrHandle,
                                        comm->memManager),
                  ret, fail);
    NCCLCHECKGOTO(allocMemCPUAccessible(&desc->doneSeq, &desc->doneSeqDev, 1, 0, &desc->doneSeqGdrHandle,
                                        comm->memManager),
                  ret, fail);
    NCCLCHECKGOTO(ceSchedCapturedToken(comm, streamPlan), ret, fail);
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

static ncclResult_t ceSchedProxyPutGroupBuildDesc(struct ncclComm* comm, struct ncclRmaProxyCtx* rmaProxyCtx,
                                                  struct ncclCeSchedStreamPlan* streamPlan, int nOps,
                                                  struct ncclRmaPutSignalOp** ops, int ctx,
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
    NCCLCHECKGOTO(allocMemCPUAccessible(&desc->readySeq, &desc->readySeqDev, 1, 0, &desc->readySeqGdrHandle,
                                        comm->memManager),
                  ret, fail);
    NCCLCHECKGOTO(allocMemCPUAccessible(&desc->doneSeq, &desc->doneSeqDev, 1, 0, &desc->doneSeqGdrHandle,
                                        comm->memManager),
                  ret, fail);
    NCCLCHECKGOTO(ceSchedCapturedToken(comm, streamPlan), ret, fail);
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

static ncclResult_t ceSchedHierCollBuildChunk(size_t perRankBytes, int nPeers, size_t maxChunk,
                                              struct ceSchedHierChunkPlan* outPlan) {
  ncclResult_t ret = ncclSuccess;
  const size_t align = CE_SCHED_HIER_CHUNK_ALIGN;

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

static ncclResult_t ceSchedRailSync(struct ncclComm* comm, struct ncclRmaProxyCtx* rmaProxyCtx,
                                    struct ncclCeSchedStreamPlan* streamPlan, int ctx, cudaStream_t stream) {
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

      NCCLCHECKGOTO(ncclRmaProxyPutBuildOp(comm, rmaProxyCtx, ctx, captured, nullptr, 0, nullptr, 0, 0, railPeer,
                                           /*signalIdx=*/0, NCCL_SIGNAL, &groupOps[idx]),
                    ret, fail);
      idx++;
    }
  }

  NCCLCHECKGOTO(ncclCalloc(&groupDesc, 1), ret, fail);
  NCCLCHECKGOTO(ceSchedProxyPutGroupBuildDesc(comm, rmaProxyCtx, streamPlan, nRemoteNodes, &groupOps, ctx, groupDesc),
                ret, fail);

  NCCLCHECKGOTO(ncclCalloc(&waitDesc, 1), ret, fail);
  NCCLCHECKGOTO(ceSchedProxyWaitBuildDesc(comm, rmaProxyCtx, streamPlan, nRemoteNodes, &railPeers, &railSigOnes,
                                          waitDesc),
                ret, fail);

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

// AllToAll's chunk -> context, keyed on the rank pair so both ends agree.
static inline int ceSchedHierA2aChunkCtx(int myRank, int peer, int c, int peerBase, int numCtx) {
  if (numCtx <= 1) return 0;
  return (int)(((unsigned)(myRank + peer) + (unsigned)(c - peerBase)) % (unsigned)numCtx);
}

// AllGather's chunk -> context: the global chunk index, symmetric because both
// ends of a rail pair label it with the same peer slot.
static inline int ceSchedHierChunkCtx(int c, int numCtx) {
  if (numCtx <= 1) return 0;
  return c % numCtx;
}

// Internal RMA contexts used by a hierarchical collective's rail. nPeers is the
// number of distinct rail peers; it is identical on all ranks.
static int ceSchedHierNumCtx(struct ncclRmaProxyState* rmaProxyState, size_t perPeerBytes, int nPeers) {
  int numCtx = rmaProxyState->numIntCtx;
  int64_t numCtxOverride = ncclParamHierCeCollNumCtx();
  if (numCtxOverride > 0) {
    if (numCtxOverride > numCtx) {
      WARN("NCCL_HIER_CE_COLL_NUM_CTX=%lld exceeds provisioned contexts; using %d. "
           "Increase NCCL_NUM_RMA_INT_CTX before init.",
           (long long)numCtxOverride, numCtx);
    }
    return numCtxOverride < numCtx ? (int)numCtxOverride : numCtx;
  }
  if (nPeers < numCtx) numCtx = nPeers > 1 ? nPeers : 1;
  int64_t threshold = ncclParamRmaMultiCtxThreshold();
  if (threshold >= 0 && perPeerBytes < (size_t)threshold) numCtx = 1;
  return numCtx;
}

// Context count for the rail-ring path, which only reaches extra contexts by
// narrowing chunks: stay on one context under capture and for small messages.
static int ceSchedHierRingNumCtx(struct ncclRmaProxyState* rmaProxyState, size_t perRankBytes, bool captured) {
  int64_t numCtxOverride = ncclParamHierCeCollNumCtx();
  int numCtx = ceSchedHierNumCtx(rmaProxyState, perRankBytes, rmaProxyState->numIntCtx);
  if (numCtxOverride > 0) return numCtx;
  if (captured) return 1;
  int64_t threshold = ncclParamRmaMultiCtxThreshold();
  if (threshold < 0) threshold = CE_SCHED_HIER_AG_RING_MULTI_CTX_THRESHOLD_DEFAULT;
  if (perRankBytes < (size_t)threshold) return 1;
  return numCtx;
}

// Max chunk width for the inter-node put-signal-group. Deterministic in
// (perPeerBytes, numCtx), so sender and receiver derive the same chunking.
static size_t ceSchedHierChunkWidth(size_t perPeerBytes, int numCtx) {
  size_t maxChunk = CE_SCHED_HIER_MAX_CHUNK_SIZE;
  if (numCtx > 1) {
    size_t perCtx = DIVUP(perPeerBytes, (size_t)numCtx);
    size_t chunksPerCtx = DIVUP(perCtx, CE_SCHED_HIER_MAX_CHUNK_SIZE);
    size_t target = alignUp(DIVUP(perPeerBytes, numCtx * chunksPerCtx), CE_SCHED_HIER_CHUNK_ALIGN);
    if (target < maxChunk) maxChunk = target;
  }
  return maxChunk;
}

static size_t ceSchedHierRingChunkWidth(size_t perRankBytes, size_t cwBytes, size_t ccwBytes, int numCtx) {
  size_t maxChunk = ceSchedHierChunkWidth(perRankBytes, numCtx);
  if (numCtx < 4 || perRankBytes < CE_SCHED_HIER_AG_RING_HALF_AWARE_THRESHOLD) {
    return maxChunk;
  }

  size_t halfBytes = std::max(cwBytes, ccwBytes);
  if (halfBytes == 0) return maxChunk;

  size_t halfAware = alignUp(DIVUP(halfBytes, (size_t)numCtx), CE_SCHED_HIER_CHUNK_ALIGN);
  halfAware = std::min(halfAware, CE_SCHED_HIER_MAX_CHUNK_SIZE);
  halfAware = std::max(halfAware, CE_SCHED_HIER_AG_RING_MIN_HALF_AWARE_CHUNK);
  return std::min(maxChunk, halfAware);
}

// Ring wait for up to two peers, merging the clockwise and counterclockwise
// entries when they resolve to the same peer.
static ncclResult_t ceSchedProxyBuildWaitPeersRing(struct ncclComm* comm, struct ncclRmaProxyCtx* rmaProxyCtx,
                                                   struct ncclCeSchedStreamPlan* streamPlan, int npeers,
                                                   const int* peersIn, const int* nsignalsIn,
                                                   struct ncclRmaProxyDesc** waitDescOut) {
  ncclResult_t ret = ncclSuccess;

  int realPeers = 0;
  int mergedPeers[2] = {0, 0};
  int mergedSignals[2] = {0, 0};
  int* waitPeers = nullptr;
  int* waitSigCounts = nullptr;
  struct ncclRmaProxyDesc* waitDesc = nullptr;

  *waitDescOut = nullptr;
  if (npeers <= 0) return ncclSuccess;
  if (npeers > 2) return ncclInternalError;

  if (npeers == 1) {
    if (nsignalsIn[0] > 0) {
      mergedPeers[0] = peersIn[0];
      mergedSignals[0] = nsignalsIn[0];
      realPeers = 1;
    }
  } else {
    int sig0 = nsignalsIn[0];
    int sig1 = nsignalsIn[1];
    if (sig0 > 0) {
      mergedPeers[0] = peersIn[0];
      mergedSignals[0] = sig0;
      realPeers = 1;
    }
    if (sig1 > 0) {
      if (realPeers > 0 && mergedPeers[0] == peersIn[1]) {
        mergedSignals[0] += sig1;
      } else {
        mergedPeers[realPeers] = peersIn[1];
        mergedSignals[realPeers] = sig1;
        realPeers++;
      }
    }
  }
  if (realPeers == 0) goto exit;

  NCCLCHECKGOTO(ncclCalloc(&waitPeers, realPeers), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&waitSigCounts, realPeers), ret, fail);
  for (int i = 0; i < realPeers; i++) {
    waitPeers[i] = mergedPeers[i];
    waitSigCounts[i] = mergedSignals[i];
  }

  NCCLCHECKGOTO(ncclCalloc(&waitDesc, 1), ret, fail);
  NCCLCHECKGOTO(ceSchedProxyWaitBuildDesc(comm, rmaProxyCtx, streamPlan, realPeers, &waitPeers, &waitSigCounts,
                                          waitDesc),
                ret, fail);
  *waitDescOut = waitDesc;
  waitDesc = nullptr;

exit:
  if (waitDesc != nullptr) (void)ncclRmaProxyDestroyDesc(comm, &waitDesc);
  free(waitPeers);
  free(waitSigCounts);
  return ret;
fail:
  goto exit;
}

// Hierarchical AllGather: railed all-to-all inter-node + intra-node CE scatter.
// Inter-node puts and waits run on the internal RMA contexts
// [numRmaCtx, numRmaCtx + numIntCtx); user contexts are never touched.
//
// DAG on the user stream:
//   RailSync                    // cross-node entry barrier (net + wait)
//   PutGroupSubmit              // one memop fires all network puts in parallel
//   IntraNodeBarrier            // gates LSA peers' recvbuf writes
//   SelfBcast                   // CE scatter of own slice to LSA peers
//   for (peer, chunk) in shift order:
//     wait for chunk's signal; CE-scatter it to local peers via LSA
//   PutGroupDone                // blocks until all network puts complete
//   IntraNodeBarrier            // gates user code reading recvbuf
static ncclResult_t ceSchedHierCeAllGatherDirect(struct ncclComm* comm, struct ncclCeSchedStreamPlan* streamPlan,
                                                 struct ncclCeCollArgs* args, cudaStream_t stream) {
  ncclResult_t ret = ncclSuccess;

  struct ncclRmaProxyState* rmaProxyState = &comm->rmaState.rmaProxyState;
  int baseCtx = comm->config.numRmaCtx;
  int railCtx = baseCtx;
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
  int numCtx = ceSchedHierNumCtx(rmaProxyState, perRankBytes, nRemoteNodes);

  struct ncclRmaProxyCtx* railProxyCtx = (struct ncclRmaProxyCtx*)rmaProxyState->rmaProxyCtxs[railCtx];

  struct ceSchedHierChunkPlan chunkPlan = {};
  int startOps = ncclRmaProxyPutGroupStartNumOps(captured);
  int doneOps = ncclRmaProxyPutGroupDoneNumOps(captured);
  int* ctxOps = nullptr;
  int* ctxFill = nullptr;
  struct ncclRmaProxyDesc** groupDesc = nullptr;
  struct ncclRmaPutSignalOp** groupOps = nullptr;
  int nActiveCtx = 0;
  CUstreamBatchMemOpParams* groupStartParams = nullptr;
  CUstreamBatchMemOpParams* groupDoneParams = nullptr;
  int nWaitTotal = 0;
  struct ncclRmaProxyDesc** waitDesc = nullptr;
  struct ncclRmaProxyCtx** waitCtx = nullptr;
  int* waitPeers = nullptr;
  int* waitSigCounts = nullptr;
  int waitStartOps = 0;
  int waitDoneOps = 0;
  CUstreamBatchMemOpParams* waitStartParams = nullptr;
  CUstreamBatchMemOpParams* waitDoneParams = nullptr;
  struct ncclCeBatchOpsParams ceBcastOps = {};
  struct ncclCeBatchOpsParams ceScatterOps = {};
  bool agUseMulticast;

  NCCLCHECKGOTO(ncclCalloc(&ctxOps, numCtx), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&ctxFill, numCtx), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&groupDesc, numCtx), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&groupOps, numCtx), ret, fail);

  // Phase 1: rail sync (cross-node entry barrier)
  NCCLCHECKGOTO(ceSchedRailSync(comm, railProxyCtx, streamPlan, railCtx, stream), ret, fail);

  // Phase 2: start all inter-node puts, one group descriptor per context
  {
    size_t maxChunk = ceSchedHierChunkWidth(perRankBytes, 1);
    NCCLCHECKGOTO(ceSchedHierCollBuildChunk(perRankBytes, nRemoteNodes, maxChunk, &chunkPlan), ret, fail);

    size_t srcWinOffset = (const uint8_t*)sendbuff - (const uint8_t*)sendWin->userPtr;
    size_t peerWinOffset = ((const uint8_t*)recvbuff + myRank * perRankBytes) - (const uint8_t*)recvWin->userPtr;

    for (int s = 1; s < nNodes; s++) {
      int p = s - 1;
      for (int c = chunkPlan.chunkStart[p]; c < chunkPlan.chunkStart[p + 1]; c++) {
        ctxOps[ceSchedHierChunkCtx(c, numCtx)]++;
      }
    }

    for (int k = 0; k < numCtx; k++) {
      if (ctxOps[k] == 0) continue;
      NCCLCHECKGOTO(ncclCalloc(&groupOps[k], ctxOps[k]), ret, fail);
      NCCLCHECKGOTO(ncclCalloc(&groupDesc[k], 1), ret, fail);
      nActiveCtx++;
    }
    NCCLCHECKGOTO(ncclCalloc(&groupStartParams, (size_t)nActiveCtx * startOps), ret, fail);
    NCCLCHECKGOTO(ncclCalloc(&groupDoneParams, (size_t)nActiveCtx * doneOps), ret, fail);

    for (int s = 1; s < nNodes; s++) {
      int p = s - 1;
      int n = (comm->node + s) % nNodes;
      int railPeer = comm->nodeRanks[n].localRankToRank[localRank];

      for (int c = chunkPlan.chunkStart[p]; c < chunkPlan.chunkStart[p + 1]; c++) {
        int k = ceSchedHierChunkCtx(c, numCtx);
        size_t subBytes = chunkPlan.chunkBytes[c];
        size_t off = chunkPlan.chunkOff[c];

        NCCLCHECKGOTO(ncclRmaProxyPutBuildOp(comm, (struct ncclRmaProxyCtx*)rmaProxyState->rmaProxyCtxs[baseCtx + k],
                                             baseCtx + k, captured, sendWin, srcWinOffset + off, recvWin,
                                             peerWinOffset + off, subBytes, railPeer, /*signalIdx=*/0, NCCL_SIGNAL,
                                             &groupOps[k][ctxFill[k]++]),
                      ret, fail);
      }
    }

    int a = 0;
    for (int k = 0; k < numCtx; k++) {
      if (ctxOps[k] == 0) continue;
      struct ncclRmaProxyCtx* pc = (struct ncclRmaProxyCtx*)rmaProxyState->rmaProxyCtxs[baseCtx + k];
      NCCLCHECKGOTO(ceSchedProxyPutGroupBuildDesc(comm, pc, streamPlan, ctxOps[k], &groupOps[k], baseCtx + k,
                                                  groupDesc[k]),
                    ret, fail);
      NCCLCHECKGOTO(ncclRmaProxyPutGroupStartParams(groupDesc[k], groupStartParams + a * startOps), ret, fail);
      NCCLCHECKGOTO(ncclRmaProxyPutGroupDoneParams(groupDesc[k], groupDoneParams + a * doneOps), ret, fail);
      NCCLCHECKGOTO(ncclRmaProxyEnqueueDesc(pc, &groupDesc[k]), ret, fail);
      a++;
    }
    NCCLCHECKGOTO(ncclCuStreamBatchMemOp(stream, nActiveCtx * startOps, groupStartParams), ret, fail);
  }

  // Phase 3: initial intra-node barrier
  NCCLCHECKGOTO(ncclMemOpSync(comm, stream, args), ret, fail);

  agUseMulticast = ncclCeAllGatherUseMulticast(comm, perRankBytes, captured,
                                               ncclAllGatherIsInPlace(sendbuff, recvbuff, myRank, perRankBytes));

  // Phase 4: self-broadcast of own slice
  NCCLCHECKGOTO(ncclCeInitBatchOpsParams(&ceBcastOps, lsaSize), ret, fail);
  {
    uint8_t* myRecvSlot = (uint8_t*)recvbuff + myRank * perRankBytes;
    size_t offset = myRecvSlot - (uint8_t*)recvWin->userPtr;

    if (agUseMulticast) {
      void* mcDstPtr;
      NCCLCHECKGOTO(ncclDevrGetLsaTeamPtrMC(comm, recvWin, offset, ncclTeamLsa(comm), &mcDstPtr), ret, fail);
      ceBcastOps.srcs[ceBcastOps.numOps] = (void*)sendbuff;
      ceBcastOps.dsts[ceBcastOps.numOps] = mcDstPtr;
      ceBcastOps.sizes[ceBcastOps.numOps] = perRankBytes;
      ceBcastOps.numOps++;
    } else {
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
    }

    NCCLCHECKGOTO(ncclCeLaunchBatchOps(comm, &ceBcastOps, stream, args), ret, fail);
  }

  // Phase 5: wait for each (peer, chunk) on its context, then CE-scatter it
  {
    // Build every wait first and fire all start memops together so each context
    // starts polling before the first blocking done wait.
    nWaitTotal = chunkPlan.chunkStart[nRemoteNodes];
    NCCLCHECKGOTO(ncclCalloc(&waitDesc, nWaitTotal), ret, fail);
    NCCLCHECKGOTO(ncclCalloc(&waitCtx, nWaitTotal), ret, fail);

    for (int s = 1; s < nNodes; s++) {
      int p = s - 1;
      int n = (comm->node - s + nNodes) % nNodes;
      int railPeer = comm->nodeRanks[n].localRankToRank[localRank];
      for (int c = chunkPlan.chunkStart[p]; c < chunkPlan.chunkStart[p + 1]; c++) {
        int k = ceSchedHierChunkCtx(c, numCtx);
        waitCtx[c] = (struct ncclRmaProxyCtx*)rmaProxyState->rmaProxyCtxs[baseCtx + k];
        NCCLCHECKGOTO(ncclCalloc(&waitPeers, 1), ret, fail);
        NCCLCHECKGOTO(ncclCalloc(&waitSigCounts, 1), ret, fail);
        waitPeers[0] = railPeer;
        waitSigCounts[0] = 1;
        NCCLCHECKGOTO(ncclCalloc(&waitDesc[c], 1), ret, fail);
        NCCLCHECKGOTO(ceSchedProxyWaitBuildDesc(comm, waitCtx[c], streamPlan, 1, &waitPeers, &waitSigCounts,
                                                waitDesc[c]),
                      ret, fail);
      }
    }

    if (nWaitTotal > 0) {
      waitStartOps = ncclRmaProxyWaitStartNumOps(waitDesc[0]);
      waitDoneOps = ncclRmaProxyWaitDoneNumOps(waitDesc[0]);
      NCCLCHECKGOTO(ncclCalloc(&waitStartParams, (size_t)nWaitTotal * waitStartOps), ret, fail);
      NCCLCHECKGOTO(ncclCalloc(&waitDoneParams, (size_t)nWaitTotal * waitDoneOps), ret, fail);
    }
    // Done params consume each context's expected-signal counters in call order.
    for (int c = 0; c < nWaitTotal; c++) {
      if (waitStartParams != nullptr) {
        NCCLCHECKGOTO(ncclRmaProxyWaitStartParams(waitDesc[c], waitStartParams + (size_t)c * waitStartOps), ret, fail);
      }
      if (waitDoneParams != nullptr) {
        NCCLCHECKGOTO(ncclRmaProxyWaitDoneParams(waitCtx[c], waitDesc[c], waitDoneParams + (size_t)c * waitDoneOps),
                      ret, fail);
      }
      NCCLCHECKGOTO(ncclRmaProxyEnqueueDesc(waitCtx[c], &waitDesc[c]), ret, fail);
    }

    NCCLCHECKGOTO(ncclCuStreamBatchMemOp(stream, nWaitTotal * waitStartOps, waitStartParams), ret, fail);

    for (int s = 1; s < nNodes; s++) {
      int p = s - 1;
      int n = (comm->node - s + nNodes) % nNodes;
      int railPeer = comm->nodeRanks[n].localRankToRank[localRank];
      size_t peerSliceOffset = railPeer * perRankBytes;

      for (int c = chunkPlan.chunkStart[p]; c < chunkPlan.chunkStart[p + 1]; c++) {
        size_t subBytes = chunkPlan.chunkBytes[c];
        size_t off = chunkPlan.chunkOff[c];

        uint8_t* chunkSlot = (uint8_t*)recvbuff + peerSliceOffset + off;
        size_t winOffset = chunkSlot - (uint8_t*)recvWin->userPtr;

        if (waitDoneOps > 0) {
          NCCLCHECKGOTO(ncclCuStreamBatchMemOp(stream, waitDoneOps, waitDoneParams + (size_t)c * waitDoneOps), ret,
                        fail);
        }

        NCCLCHECKGOTO(ncclCeInitBatchOpsParams(&ceScatterOps, lsaSize), ret, fail);
        if (agUseMulticast) {
          void* mcDstPtr;
          NCCLCHECKGOTO(ncclDevrGetLsaTeamPtrMC(comm, recvWin, winOffset, ncclTeamLsa(comm), &mcDstPtr), ret, fail);
          ceScatterOps.srcs[ceScatterOps.numOps] = chunkSlot;
          ceScatterOps.dsts[ceScatterOps.numOps] = mcDstPtr;
          ceScatterOps.sizes[ceScatterOps.numOps] = subBytes;
          ceScatterOps.numOps++;
        } else {
          for (int r = 1; r < lsaSize; r++) {
            int targetLsaRank = (myLsaRank + r) % lsaSize;
            void* peerBuf;
            NCCLCHECKGOTO(ncclDevrGetLsaRankPtr(comm, recvWin, winOffset, targetLsaRank, &peerBuf), ret, fail);
            ceScatterOps.srcs[ceScatterOps.numOps] = chunkSlot;
            ceScatterOps.dsts[ceScatterOps.numOps] = peerBuf;
            ceScatterOps.sizes[ceScatterOps.numOps] = subBytes;
            ceScatterOps.numOps++;
          }
        }

        NCCLCHECKGOTO(ncclCeLaunchBatchOps(comm, &ceScatterOps, stream, args), ret, fail);
        ncclCeFreeBatchOpsParams(&ceScatterOps);
      }
    }
  }

  // Phase 6: wait for all outgoing puts, all contexts in one batch
  NCCLCHECKGOTO(ncclCuStreamBatchMemOp(stream, nActiveCtx * doneOps, groupDoneParams), ret, fail);

  // Phase 7: final intra-node barrier
  NCCLCHECKGOTO(ncclMemOpSync(comm, stream, args), ret, fail);

exit:
  ncclCeFreeBatchOpsParams(&ceBcastOps);
  ncclCeFreeBatchOpsParams(&ceScatterOps);
  for (int k = 0; groupOps && k < numCtx; k++) free(groupOps[k]);
  if (groupDesc) {
    for (int k = 0; k < numCtx; k++) {
      if (groupDesc[k] != nullptr) (void)ncclRmaProxyDestroyDesc(comm, &groupDesc[k]);
    }
  }
  free(groupOps);
  free(groupStartParams);
  free(groupDoneParams);
  free(groupDesc);
  free(ctxOps);
  free(ctxFill);
  if (waitDesc != nullptr) {
    for (int c = 0; c < nWaitTotal; c++) {
      if (waitDesc[c] != nullptr) (void)ncclRmaProxyDestroyDesc(comm, &waitDesc[c]);
    }
  }
  free(waitDesc);
  free(waitCtx);
  free(waitPeers);
  free(waitSigCounts);
  free(waitStartParams);
  free(waitDoneParams);
  ceSchedHierCollFreeChunkPlan(&chunkPlan);
  return ret;
fail:
  goto exit;
}

struct ceSchedHierAgRingRound {
  int originRankSendCw;
  int originRankSendCcw;
  int originRankRecvCw;
  int originRankRecvCcw;
  struct ncclDevrWindow* srcWinHostCw;
  struct ncclDevrWindow* srcWinHostCcw;
  size_t srcBaseOffsetCw;
  size_t dstBaseOffsetCw;
  size_t srcBaseOffsetCcw;
  size_t dstBaseOffsetCcw;
};

struct ceSchedHierAgRingIssueState {
  int numCtx;
  int baseCtx;
  int nActiveCtx;
  int sendRailPeerCw;
  int sendRailPeerCcw;
  int* ctxOps;
  int* ctxFill;
  int* activeCtxs;
  struct ceSchedHierChunkPlan* cwChunkPlan;
  struct ceSchedHierChunkPlan* ccwChunkPlan;
  struct ncclRmaProxyDesc** groupDesc;
  struct ncclRmaPutSignalOp** groupOps;
  CUstreamBatchMemOpParams** groupStartParams;
  CUstreamBatchMemOpParams** groupDoneParams;
  struct ceSchedHierAgRingRound* rounds;
};

// Build and submit one bidirectional forwarding round. The caller alternates
// two parameter slots so the next round can be submitted before the current
// round's local fanout and completion retirement.
static ncclResult_t ceSchedHierAgRingIssueRound(struct ncclComm* comm, struct ncclCeSchedStreamPlan* streamPlan,
                                                struct ncclCeCollArgs* args, cudaStream_t stream,
                                                struct ceSchedHierAgRingIssueState* st, int step, int slot) {
  ncclResult_t ret = ncclSuccess;
  struct ncclRmaProxyState* rmaProxyState = &comm->rmaState.rmaProxyState;
  struct ncclDevrWindow* recvWin = args->recvWin;
  struct ceSchedHierAgRingRound* round = &st->rounds[step];
  bool captured = streamPlan->captured;
  int startOps = ncclRmaProxyPutGroupStartNumOps(captured);
  int doneOps = ncclRmaProxyPutGroupDoneNumOps(captured);

  memset(st->ctxFill, 0, (size_t)st->numCtx * sizeof(int));
  for (int ai = 0; ai < st->nActiveCtx; ai++) {
    int k = st->activeCtxs[ai];
    NCCLCHECKGOTO(ncclCalloc(&st->groupOps[k], st->ctxOps[k]), ret, fail);
    NCCLCHECKGOTO(ncclCalloc(&st->groupDesc[k], 1), ret, fail);
  }

  if (st->cwChunkPlan->chunkStart != nullptr) {
    for (int c = st->cwChunkPlan->chunkStart[0]; c < st->cwChunkPlan->chunkStart[1]; c++) {
      int k = c % st->numCtx;
      size_t subBytes = st->cwChunkPlan->chunkBytes[c];
      size_t off = st->cwChunkPlan->chunkOff[c];
      NCCLCHECKGOTO(ncclRmaProxyPutBuildOp(comm, (struct ncclRmaProxyCtx*)rmaProxyState->rmaProxyCtxs[st->baseCtx + k],
                                           st->baseCtx + k, captured, round->srcWinHostCw, round->srcBaseOffsetCw + off,
                                           recvWin, round->dstBaseOffsetCw + off, subBytes, st->sendRailPeerCw,
                                           /*signalIdx=*/0, NCCL_SIGNAL, &st->groupOps[k][st->ctxFill[k]++]),
                    ret, fail);
    }
  }
  if (st->ccwChunkPlan->chunkStart != nullptr) {
    for (int c = st->ccwChunkPlan->chunkStart[0]; c < st->ccwChunkPlan->chunkStart[1]; c++) {
      int k = c % st->numCtx;
      size_t subBytes = st->ccwChunkPlan->chunkBytes[c];
      size_t off = st->ccwChunkPlan->chunkOff[c];
      NCCLCHECKGOTO(ncclRmaProxyPutBuildOp(comm, (struct ncclRmaProxyCtx*)rmaProxyState->rmaProxyCtxs[st->baseCtx + k],
                                           st->baseCtx + k, captured, round->srcWinHostCcw,
                                           round->srcBaseOffsetCcw + off, recvWin, round->dstBaseOffsetCcw + off,
                                           subBytes, st->sendRailPeerCcw, /*signalIdx=*/0, NCCL_SIGNAL,
                                           &st->groupOps[k][st->ctxFill[k]++]),
                    ret, fail);
    }
  }

  for (int ai = 0; ai < st->nActiveCtx; ai++) {
    int k = st->activeCtxs[ai];
    struct ncclRmaProxyCtx* proxyCtx = (struct ncclRmaProxyCtx*)rmaProxyState->rmaProxyCtxs[st->baseCtx + k];
    NCCLCHECKGOTO(ceSchedProxyPutGroupBuildDesc(comm, proxyCtx, streamPlan, st->ctxOps[k], &st->groupOps[k],
                                                st->baseCtx + k, st->groupDesc[k]),
                  ret, fail);
    NCCLCHECKGOTO(ncclRmaProxyPutGroupStartParams(st->groupDesc[k], st->groupStartParams[slot] + ai * startOps), ret,
                  fail);
    NCCLCHECKGOTO(ncclRmaProxyPutGroupDoneParams(st->groupDesc[k], st->groupDoneParams[slot] + ai * doneOps), ret,
                  fail);
    NCCLCHECKGOTO(ncclRmaProxyEnqueueDesc(proxyCtx, &st->groupDesc[k]), ret, fail);
  }
  if (st->nActiveCtx > 0) {
    NCCLCHECKGOTO(ncclCuStreamBatchMemOp(stream, st->nActiveCtx * startOps, st->groupStartParams[slot]), ret, fail);
  }

exit:
  return ret;
fail:
  goto exit;
}

// Bidirectional ring variant of hierarchical CE allgather.
//
// DAG on the user stream:
//   RailSync                    // cross-node entry barrier (net + wait)
//   IntraNodeBarrier            // gates LSA peers' recvbuf writes
//   SelfBcast                   // CE scatter of own slice to LSA peers
//   Round1PutGroupSubmit        // prime clockwise and counterclockwise halves
//   for each ring round:
//     wait per active context   // covers that context's cw + ccw arrivals
//     NextRoundPutGroupSubmit   // submit the next round before local fanout
//     CE-scatter arrived halves // local fanout via LSA
//     PutGroupDone              // retire the current round's outbound puts
//   IntraNodeBarrier            // gates user code reading recvbuf
static ncclResult_t ceSchedHierCeAllGatherRing(struct ncclComm* comm, struct ncclCeSchedStreamPlan* streamPlan,
                                               struct ncclCeCollArgs* args, cudaStream_t stream) {
  ncclResult_t ret = ncclSuccess;

  struct ncclRmaProxyState* rmaProxyState = &comm->rmaState.rmaProxyState;
  int baseCtx = comm->config.numRmaCtx;
  int railCtx = baseCtx;
  int myRank = comm->rank;
  int localRank = comm->localRank;
  int myNode = comm->node;
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
  size_t cwBytes = perRankBytes <= 1 ? perRankBytes : perRankBytes / 2;
  size_t ccwBytes = perRankBytes - cwBytes;
  bool agUseMulticast = ncclCeAllGatherUseMulticast(
    comm, perRankBytes, captured, (const uint8_t*)sendbuff == (const uint8_t*)recvbuff + myRank * perRankBytes);
  int numCtx = ceSchedHierRingNumCtx(rmaProxyState, perRankBytes, captured);

  struct ncclRmaProxyCtx* railProxyCtx = (struct ncclRmaProxyCtx*)rmaProxyState->rmaProxyCtxs[railCtx];
  struct ceSchedHierChunkPlan cwChunkPlan = {};
  struct ceSchedHierChunkPlan ccwChunkPlan = {};
  int startOps = ncclRmaProxyPutGroupStartNumOps(captured);
  int doneOps = ncclRmaProxyPutGroupDoneNumOps(captured);
  int* ctxOps = nullptr;
  int* cwCtxOps = nullptr;
  int* ccwCtxOps = nullptr;
  int* ctxFill = nullptr;
  int* cwChunkFill = nullptr;
  int* ccwChunkFill = nullptr;
  int* activeCtxs = nullptr;
  int** cwChunksByCtx = nullptr;
  int** ccwChunksByCtx = nullptr;
  uint8_t** scatterPeerBases = nullptr;
  int* waitNPeersByActiveCtx = nullptr;
  int** waitPeersByActiveCtx = nullptr;
  int** waitSignalsByActiveCtx = nullptr;
  struct ncclRmaProxyDesc** waitDesc = nullptr;
  struct ncclRmaProxyDesc** groupDesc = nullptr;
  struct ncclRmaPutSignalOp** groupOps = nullptr;
  int nActiveCtx = 0;
  CUstreamBatchMemOpParams* groupStartParams[2] = {nullptr, nullptr};
  CUstreamBatchMemOpParams* groupDoneParams[2] = {nullptr, nullptr};
  CUstreamBatchMemOpParams* waitStartParams = nullptr;
  CUstreamBatchMemOpParams* waitDoneParams = nullptr;
  struct ncclCeBatchOpsParams ceBcastOps = {};
  struct ncclCeBatchOpsParams ceScatterOps = {};
  uint8_t* scatterMcBase = nullptr;
  struct ceSchedHierAgRingRound* roundMeta = nullptr;
  struct ceSchedHierAgRingIssueState issueState = {};

  NCCLCHECKGOTO(ncclCalloc(&ctxOps, numCtx), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&cwCtxOps, numCtx), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&ccwCtxOps, numCtx), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&ctxFill, numCtx), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&cwChunkFill, numCtx), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&ccwChunkFill, numCtx), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&activeCtxs, numCtx), ret, fail);
  if (lsaSize > 1) {
    NCCLCHECKGOTO(ncclCalloc(&scatterPeerBases, lsaSize - 1), ret, fail);
  }
  NCCLCHECKGOTO(ncclCalloc(&cwChunksByCtx, numCtx), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&ccwChunksByCtx, numCtx), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&waitNPeersByActiveCtx, numCtx), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&waitPeersByActiveCtx, numCtx), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&waitSignalsByActiveCtx, numCtx), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&waitDesc, numCtx), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&groupDesc, numCtx), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&groupOps, numCtx), ret, fail);

  // Phase 1: rail sync (cross-node entry barrier)
  NCCLCHECKGOTO(ceSchedRailSync(comm, railProxyCtx, streamPlan, railCtx, stream), ret, fail);

  // One-peer chunking reused by every ring round.
  if (nRemoteNodes > 0) {
    size_t maxChunk = ceSchedHierRingChunkWidth(perRankBytes, cwBytes, ccwBytes, numCtx);
    if (cwBytes > 0) {
      NCCLCHECKGOTO(ceSchedHierCollBuildChunk(cwBytes, 1, maxChunk, &cwChunkPlan), ret, fail);
      for (int c = cwChunkPlan.chunkStart[0]; c < cwChunkPlan.chunkStart[1]; c++) {
        int k = c % numCtx;
        ctxOps[k]++;
        cwCtxOps[k]++;
      }
    }
    if (ccwBytes > 0) {
      NCCLCHECKGOTO(ceSchedHierCollBuildChunk(ccwBytes, 1, maxChunk, &ccwChunkPlan), ret, fail);
      for (int c = ccwChunkPlan.chunkStart[0]; c < ccwChunkPlan.chunkStart[1]; c++) {
        int k = c % numCtx;
        ctxOps[k]++;
        ccwCtxOps[k]++;
      }
    }
    for (int k = 0; k < numCtx; k++) {
      if (ctxOps[k] == 0) continue;
      activeCtxs[nActiveCtx] = k;
      nActiveCtx++;
    }

    for (int k = 0; k < numCtx; k++) {
      if (cwCtxOps[k] > 0) NCCLCHECKGOTO(ncclCalloc(&cwChunksByCtx[k], cwCtxOps[k]), ret, fail);
      if (ccwCtxOps[k] > 0) NCCLCHECKGOTO(ncclCalloc(&ccwChunksByCtx[k], ccwCtxOps[k]), ret, fail);
    }
    if (cwBytes > 0) {
      for (int c = cwChunkPlan.chunkStart[0]; c < cwChunkPlan.chunkStart[1]; c++) {
        int k = c % numCtx;
        cwChunksByCtx[k][cwChunkFill[k]++] = c;
      }
    }
    if (ccwBytes > 0) {
      for (int c = ccwChunkPlan.chunkStart[0]; c < ccwChunkPlan.chunkStart[1]; c++) {
        int k = c % numCtx;
        ccwChunksByCtx[k][ccwChunkFill[k]++] = c;
      }
    }

    for (int i = 0; i < 2; i++) {
      NCCLCHECKGOTO(ncclCalloc(&groupStartParams[i], (size_t)nActiveCtx * startOps), ret, fail);
      NCCLCHECKGOTO(ncclCalloc(&groupDoneParams[i], (size_t)nActiveCtx * doneOps), ret, fail);
    }
  }

  // Phase 2: initial intra-node barrier
  NCCLCHECKGOTO(ncclMemOpSync(comm, stream, args), ret, fail);

  if (agUseMulticast) {
    void* mcBase = nullptr;
    NCCLCHECKGOTO(ncclDevrGetLsaTeamPtrMC(comm, recvWin, 0, ncclTeamLsa(comm), &mcBase), ret, fail);
    scatterMcBase = (uint8_t*)mcBase;
  } else if (lsaSize > 1) {
    for (int r = 1; r < lsaSize; r++) {
      int targetLsaRank = (myLsaRank + r) % lsaSize;
      void* peerBase = nullptr;
      NCCLCHECKGOTO(ncclDevrGetLsaRankPtr(comm, recvWin, 0, targetLsaRank, &peerBase), ret, fail);
      scatterPeerBases[r - 1] = (uint8_t*)peerBase;
    }
  }

  // Phase 3: self-broadcast of own slice
  NCCLCHECKGOTO(ncclCeInitBatchOpsParams(&ceBcastOps, lsaSize), ret, fail);
  {
    uint8_t* myRecvSlot = (uint8_t*)recvbuff + myRank * perRankBytes;
    size_t offset = myRecvSlot - (uint8_t*)recvWin->userPtr;

    if (agUseMulticast) {
      ceBcastOps.srcs[ceBcastOps.numOps] = (void*)sendbuff;
      ceBcastOps.dsts[ceBcastOps.numOps] = (void*)(scatterMcBase + offset);
      ceBcastOps.sizes[ceBcastOps.numOps] = perRankBytes;
      ceBcastOps.numOps++;
    } else {
      if (myRecvSlot != (const uint8_t*)sendbuff) {
        ceBcastOps.srcs[ceBcastOps.numOps] = (void*)sendbuff;
        ceBcastOps.dsts[ceBcastOps.numOps] = (void*)myRecvSlot;
        ceBcastOps.sizes[ceBcastOps.numOps] = perRankBytes;
        ceBcastOps.numOps++;
      }
      for (int r = 1; r < lsaSize; r++) {
        ceBcastOps.srcs[ceBcastOps.numOps] = (void*)sendbuff;
        ceBcastOps.dsts[ceBcastOps.numOps] = (void*)(scatterPeerBases[r - 1] + offset);
        ceBcastOps.sizes[ceBcastOps.numOps] = perRankBytes;
        ceBcastOps.numOps++;
      }
    }

    NCCLCHECKGOTO(ncclCeLaunchBatchOps(comm, &ceBcastOps, stream, args), ret, fail);
  }

  // Phase 4: ring rounds (next-node send, previous-node receive)
  if (nRemoteNodes > 0 && nActiveCtx > 0) {
    int nRounds = nNodes - 1;
    int nextNode = (myNode + 1) % nNodes;
    int prevNode = (myNode - 1 + nNodes) % nNodes;
    int sendRailPeerCw = comm->nodeRanks[nextNode].localRankToRank[localRank];
    int recvRailPeerCw = comm->nodeRanks[prevNode].localRankToRank[localRank];
    int sendRailPeerCcw = comm->nodeRanks[prevNode].localRankToRank[localRank];
    int recvRailPeerCcw = comm->nodeRanks[nextNode].localRankToRank[localRank];
    int nRoundChunks = (cwChunkPlan.chunkStart ? (cwChunkPlan.chunkStart[1] - cwChunkPlan.chunkStart[0]) : 0) +
                       (ccwChunkPlan.chunkStart ? (ccwChunkPlan.chunkStart[1] - ccwChunkPlan.chunkStart[0]) : 0);
    int scatterCapacity = agUseMulticast ? nRoundChunks : nRoundChunks * (lsaSize - 1);

    for (int ai = 0; ai < nActiveCtx; ai++) {
      int k = activeCtxs[ai];
      int waitNPeers = 0;
      NCCLCHECKGOTO(ncclCalloc(&waitPeersByActiveCtx[k], 2), ret, fail);
      NCCLCHECKGOTO(ncclCalloc(&waitSignalsByActiveCtx[k], 2), ret, fail);
      if (cwCtxOps[k] > 0) {
        waitPeersByActiveCtx[k][waitNPeers] = recvRailPeerCw;
        waitSignalsByActiveCtx[k][waitNPeers] = cwCtxOps[k];
        waitNPeers++;
      }
      if (ccwCtxOps[k] > 0) {
        waitPeersByActiveCtx[k][waitNPeers] = recvRailPeerCcw;
        waitSignalsByActiveCtx[k][waitNPeers] = ccwCtxOps[k];
        waitNPeers++;
      }
      waitNPeersByActiveCtx[k] = waitNPeers;
    }

    NCCLCHECKGOTO(ncclCalloc(&roundMeta, nRounds + 1), ret, fail);
    for (int step = 1; step <= nRounds; step++) {
      int originNodeSendCw = (myNode - (step - 1) + nNodes) % nNodes;
      int originNodeSendCcw = (myNode + (step - 1)) % nNodes;
      int originNodeRecvCw = (myNode - step + nNodes) % nNodes;
      int originNodeRecvCcw = (myNode + step) % nNodes;
      struct ceSchedHierAgRingRound* meta = &roundMeta[step];
      meta->originRankSendCw = comm->nodeRanks[originNodeSendCw].localRankToRank[localRank];
      meta->originRankSendCcw = comm->nodeRanks[originNodeSendCcw].localRankToRank[localRank];
      meta->originRankRecvCw = comm->nodeRanks[originNodeRecvCw].localRankToRank[localRank];
      meta->originRankRecvCcw = comm->nodeRanks[originNodeRecvCcw].localRankToRank[localRank];
      meta->srcWinHostCw = step == 1 ? sendWin : recvWin;
      meta->srcWinHostCcw = step == 1 ? sendWin : recvWin;
      meta->srcBaseOffsetCw = step == 1 ? (const uint8_t*)sendbuff - (const uint8_t*)sendWin->userPtr :
                                          ((const uint8_t*)recvbuff + meta->originRankSendCw * perRankBytes) -
                                            (const uint8_t*)recvWin->userPtr;
      meta->dstBaseOffsetCw =
        ((const uint8_t*)recvbuff + meta->originRankSendCw * perRankBytes) - (const uint8_t*)recvWin->userPtr;
      meta->srcBaseOffsetCcw = (step == 1 ? (const uint8_t*)sendbuff - (const uint8_t*)sendWin->userPtr :
                                            ((const uint8_t*)recvbuff + meta->originRankSendCcw * perRankBytes) -
                                              (const uint8_t*)recvWin->userPtr) +
                               cwBytes;
      meta->dstBaseOffsetCcw = ((const uint8_t*)recvbuff + meta->originRankSendCcw * perRankBytes) -
                               (const uint8_t*)recvWin->userPtr + cwBytes;
    }

    issueState.numCtx = numCtx;
    issueState.baseCtx = baseCtx;
    issueState.nActiveCtx = nActiveCtx;
    issueState.sendRailPeerCw = sendRailPeerCw;
    issueState.sendRailPeerCcw = sendRailPeerCcw;
    issueState.ctxOps = ctxOps;
    issueState.ctxFill = ctxFill;
    issueState.activeCtxs = activeCtxs;
    issueState.cwChunkPlan = &cwChunkPlan;
    issueState.ccwChunkPlan = &ccwChunkPlan;
    issueState.groupDesc = groupDesc;
    issueState.groupOps = groupOps;
    issueState.groupStartParams = groupStartParams;
    issueState.groupDoneParams = groupDoneParams;
    issueState.rounds = roundMeta;

    if (scatterCapacity > 0) NCCLCHECKGOTO(ncclCeInitBatchOpsParams(&ceScatterOps, scatterCapacity), ret, fail);

    NCCLCHECKGOTO(ceSchedHierAgRingIssueRound(comm, streamPlan, args, stream, &issueState, /*step=*/1, /*slot=*/0), ret,
                  fail);

    for (int step = 1; step <= nRounds; step++) {
      struct ceSchedHierAgRingRound* meta = &roundMeta[step];
      int slot = (step - 1) & 1;

      // Start polling on every context before the first blocking done wait.
      int waitStartOpsTotal = 0;
      int waitDoneOpsTotal = 0;
      for (int ai = 0; ai < nActiveCtx; ai++) {
        int k = activeCtxs[ai];
        struct ncclRmaProxyCtx* pc = (struct ncclRmaProxyCtx*)rmaProxyState->rmaProxyCtxs[baseCtx + k];
        NCCLCHECKGOTO(ceSchedProxyBuildWaitPeersRing(comm, pc, streamPlan, waitNPeersByActiveCtx[k],
                                                     waitPeersByActiveCtx[k], waitSignalsByActiveCtx[k], &waitDesc[k]),
                      ret, fail);
        if (waitDesc[k] != nullptr) {
          waitStartOpsTotal += ncclRmaProxyWaitStartNumOps(waitDesc[k]);
          waitDoneOpsTotal += ncclRmaProxyWaitDoneNumOps(waitDesc[k]);
        }
      }

      // The active contexts and their wait peer sets are invariant across rounds.
      if (waitStartOpsTotal > 0 && waitStartParams == nullptr) {
        NCCLCHECKGOTO(ncclCalloc(&waitStartParams, waitStartOpsTotal), ret, fail);
      }
      if (waitDoneOpsTotal > 0 && waitDoneParams == nullptr) {
        NCCLCHECKGOTO(ncclCalloc(&waitDoneParams, waitDoneOpsTotal), ret, fail);
      }
      int waitStartOff = 0;
      int waitDoneOff = 0;
      for (int ai = 0; ai < nActiveCtx; ai++) {
        int k = activeCtxs[ai];
        if (waitDesc[k] == nullptr) continue;
        struct ncclRmaProxyCtx* pc = (struct ncclRmaProxyCtx*)rmaProxyState->rmaProxyCtxs[baseCtx + k];
        int nStart = ncclRmaProxyWaitStartNumOps(waitDesc[k]);
        int nDone = ncclRmaProxyWaitDoneNumOps(waitDesc[k]);
        if (waitStartParams != nullptr) {
          NCCLCHECKGOTO(ncclRmaProxyWaitStartParams(waitDesc[k], waitStartParams + waitStartOff), ret, fail);
        }
        if (waitDoneParams != nullptr) {
          NCCLCHECKGOTO(ncclRmaProxyWaitDoneParams(pc, waitDesc[k], waitDoneParams + waitDoneOff), ret, fail);
        }
        NCCLCHECKGOTO(ncclRmaProxyEnqueueDesc(pc, &waitDesc[k]), ret, fail);
        waitStartOff += nStart;
        waitDoneOff += nDone;
      }
      NCCLCHECKGOTO(ncclCuStreamBatchMemOp(stream, waitStartOpsTotal, waitStartParams), ret, fail);
      NCCLCHECKGOTO(ncclCuStreamBatchMemOp(stream, waitDoneOpsTotal, waitDoneParams), ret, fail);

      if (step < nRounds) {
        NCCLCHECKGOTO(ceSchedHierAgRingIssueRound(comm, streamPlan, args, stream, &issueState, step + 1, slot ^ 1), ret,
                      fail);
      }

      ceScatterOps.numOps = 0;
      for (int ai = 0; ai < nActiveCtx; ai++) {
        int k = activeCtxs[ai];
        if (cwBytes > 0) {
          for (int ci = 0; ci < cwCtxOps[k]; ci++) {
            int c = cwChunksByCtx[k][ci];
            size_t subBytes = cwChunkPlan.chunkBytes[c];
            size_t off = cwChunkPlan.chunkOff[c];
            uint8_t* chunkSlot = (uint8_t*)recvbuff + meta->originRankRecvCw * perRankBytes + off;
            size_t winOffset = chunkSlot - (uint8_t*)recvWin->userPtr;

            if (agUseMulticast) {
              ceScatterOps.srcs[ceScatterOps.numOps] = chunkSlot;
              ceScatterOps.dsts[ceScatterOps.numOps] = (void*)(scatterMcBase + winOffset);
              ceScatterOps.sizes[ceScatterOps.numOps] = subBytes;
              ceScatterOps.numOps++;
            } else {
              for (int r = 1; r < lsaSize; r++) {
                ceScatterOps.srcs[ceScatterOps.numOps] = chunkSlot;
                ceScatterOps.dsts[ceScatterOps.numOps] = (void*)(scatterPeerBases[r - 1] + winOffset);
                ceScatterOps.sizes[ceScatterOps.numOps] = subBytes;
                ceScatterOps.numOps++;
              }
            }
          }
        }
        if (ccwBytes > 0) {
          for (int ci = 0; ci < ccwCtxOps[k]; ci++) {
            int c = ccwChunksByCtx[k][ci];
            size_t subBytes = ccwChunkPlan.chunkBytes[c];
            size_t off = ccwChunkPlan.chunkOff[c];
            uint8_t* chunkSlot = (uint8_t*)recvbuff + meta->originRankRecvCcw * perRankBytes + cwBytes + off;
            size_t winOffset = chunkSlot - (uint8_t*)recvWin->userPtr;

            if (agUseMulticast) {
              ceScatterOps.srcs[ceScatterOps.numOps] = chunkSlot;
              ceScatterOps.dsts[ceScatterOps.numOps] = (void*)(scatterMcBase + winOffset);
              ceScatterOps.sizes[ceScatterOps.numOps] = subBytes;
              ceScatterOps.numOps++;
            } else {
              for (int r = 1; r < lsaSize; r++) {
                ceScatterOps.srcs[ceScatterOps.numOps] = chunkSlot;
                ceScatterOps.dsts[ceScatterOps.numOps] = (void*)(scatterPeerBases[r - 1] + winOffset);
                ceScatterOps.sizes[ceScatterOps.numOps] = subBytes;
                ceScatterOps.numOps++;
              }
            }
          }
        }
      }

      if (ceScatterOps.numOps > 0) {
        NCCLCHECKGOTO(ncclCeLaunchBatchOps(comm, &ceScatterOps, stream, args), ret, fail);
      }

      NCCLCHECKGOTO(ncclCuStreamBatchMemOp(stream, nActiveCtx * doneOps, groupDoneParams[slot]), ret, fail);
    }
  }

  // Phase 5: final intra-node barrier
  NCCLCHECKGOTO(ncclMemOpSync(comm, stream, args), ret, fail);

exit:
  ncclCeFreeBatchOpsParams(&ceBcastOps);
  ncclCeFreeBatchOpsParams(&ceScatterOps);
  if (groupDesc) {
    for (int k = 0; k < numCtx; k++) {
      if (groupDesc[k] != nullptr) (void)ncclRmaProxyDestroyDesc(comm, &groupDesc[k]);
    }
  }
  if (groupOps) {
    for (int k = 0; k < numCtx; k++) free(groupOps[k]);
  }
  free(groupOps);
  for (int i = 0; i < 2; i++) {
    free(groupStartParams[i]);
    free(groupDoneParams[i]);
  }
  free(groupDesc);
  free(ctxOps);
  free(cwCtxOps);
  free(ccwCtxOps);
  free(ctxFill);
  free(cwChunkFill);
  free(ccwChunkFill);
  free(activeCtxs);
  free(scatterPeerBases);
  free(roundMeta);
  if (waitPeersByActiveCtx) {
    for (int k = 0; k < numCtx; k++) free(waitPeersByActiveCtx[k]);
  }
  if (waitSignalsByActiveCtx) {
    for (int k = 0; k < numCtx; k++) free(waitSignalsByActiveCtx[k]);
  }
  free(waitNPeersByActiveCtx);
  free(waitPeersByActiveCtx);
  free(waitSignalsByActiveCtx);
  if (waitDesc) {
    for (int k = 0; k < numCtx; k++) {
      if (waitDesc[k] != nullptr) (void)ncclRmaProxyDestroyDesc(comm, &waitDesc[k]);
    }
  }
  free(waitDesc);
  free(waitStartParams);
  free(waitDoneParams);
  if (cwChunksByCtx) {
    for (int k = 0; k < numCtx; k++) free(cwChunksByCtx[k]);
  }
  if (ccwChunksByCtx) {
    for (int k = 0; k < numCtx; k++) free(ccwChunksByCtx[k]);
  }
  free(cwChunksByCtx);
  free(ccwChunksByCtx);
  ceSchedHierCollFreeChunkPlan(&cwChunkPlan);
  ceSchedHierCollFreeChunkPlan(&ccwChunkPlan);
  return ret;
fail:
  goto exit;
}

static ncclResult_t ceSchedHierCeAllGather(struct ncclComm* comm, struct ncclCeSchedStreamPlan* streamPlan,
                                           struct ncclCeCollArgs* args, cudaStream_t stream) {
  size_t perRankBytes = args->nElts * args->eltSize;
  bool useRing = ncclParamHierCeCollAgRailRingEnable() > 0;
  if (comm->rank == 0) {
    INFO(NCCL_TUNING, "AllGather [Hierarchical CE]: %zu Bytes -> Rail %s RMA proxy + CE", perRankBytes,
         useRing ? "Ring" : "Direct");
  }
  if (useRing) return ceSchedHierCeAllGatherRing(comm, streamPlan, args, stream);
  return ceSchedHierCeAllGatherDirect(comm, streamPlan, args, stream);
}

// Hierarchical AlltoAll: alltoall inter-node + intra-node CE alltoall.
// Inter-node puts and waits run on the internal RMA contexts, keyed on the
// rank pair so both ends agree on each chunk's context.
//
// DAG on the user stream:
//   RailSync                    // rail-only entry barrier
//   IntraNodeBarrier #1         // all ranks in sync
//   PutGroupSubmit              // one memop fires all put operations
//   IntraNodeAlltoAll           // batched CE alltoall
//   AggregateWait               // per-context wait covering all remote peers
//   PutGroupDone                // blocks until outbound puts done
//   IntraNodeBarrier #2         // all ranks in sync
static ncclResult_t ceSchedHierCeAlltoAll(struct ncclComm* comm, struct ncclCeSchedStreamPlan* streamPlan,
                                          struct ncclCeCollArgs* args, cudaStream_t stream) {
  ncclResult_t ret = ncclSuccess;

  struct ncclRmaProxyState* rmaProxyState = &comm->rmaState.rmaProxyState;
  int baseCtx = comm->config.numRmaCtx;
  int railCtx = baseCtx;
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
  int numCtx = ceSchedHierNumCtx(rmaProxyState, perPeerBytes, numRemotePeers);
  bool inPlace = (sendbuff == recvbuff);

  struct ncclRmaProxyCtx* railProxyCtx = (struct ncclRmaProxyCtx*)rmaProxyState->rmaProxyCtxs[railCtx];

  struct ceSchedHierChunkPlan chunkPlan = {};
  int startOps = ncclRmaProxyPutGroupStartNumOps(captured);
  int doneOps = ncclRmaProxyPutGroupDoneNumOps(captured);
  int* ctxOps = nullptr;
  int* ctxFill = nullptr;
  struct ncclRmaProxyDesc** groupDesc = nullptr;
  struct ncclRmaPutSignalOp** groupOps = nullptr;
  int nActiveCtx = 0;
  CUstreamBatchMemOpParams* groupStartParams = nullptr;
  CUstreamBatchMemOpParams* groupDoneParams = nullptr;
  int** waitPeers = nullptr;
  int** waitSigCounts = nullptr;
  struct ncclRmaProxyDesc** waitDesc = nullptr;
  int waitStartOpsTotal = 0;
  int waitDoneOpsTotal = 0;
  CUstreamBatchMemOpParams* waitStartParams = nullptr;
  CUstreamBatchMemOpParams* waitDoneParams = nullptr;
  struct ncclCeBatchOpsParams ceLocalA2A = {};

  NCCLCHECKGOTO(ncclCalloc(&ctxOps, numCtx), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&ctxFill, numCtx), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&groupDesc, numCtx), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&groupOps, numCtx), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&waitPeers, numCtx), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&waitSigCounts, numCtx), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&waitDesc, numCtx), ret, fail);

  // Phase 1: rail sync (rail-only cross-node entry barrier)
  NCCLCHECKGOTO(ceSchedRailSync(comm, railProxyCtx, streamPlan, railCtx, stream), ret, fail);

  // Phase 2: intra-node barrier
  NCCLCHECKGOTO(ncclMemOpSync(comm, stream, args), ret, fail);

  // Phase 3: build and submit the put-signal-groups
  {
    size_t maxChunk = ceSchedHierChunkWidth(perPeerBytes, 1);
    NCCLCHECKGOTO(ceSchedHierCollBuildChunk(perPeerBytes, numRemotePeers, maxChunk, &chunkPlan), ret, fail);

    {
      int p = 0;
      for (int s = 1; s < nNodes; s++) {
        int n = (myNode + s) % nNodes;
        for (int lr = 0; lr < localRanks; lr++) {
          int peer = comm->nodeRanks[n].localRankToRank[lr];
          for (int c = chunkPlan.chunkStart[p]; c < chunkPlan.chunkStart[p + 1]; c++) {
            ctxOps[ceSchedHierA2aChunkCtx(myRank, peer, c, chunkPlan.chunkStart[p], numCtx)]++;
          }
          p++;
        }
      }
    }

    for (int k = 0; k < numCtx; k++) {
      if (ctxOps[k] == 0) continue;
      NCCLCHECKGOTO(ncclCalloc(&groupOps[k], ctxOps[k]), ret, fail);
      NCCLCHECKGOTO(ncclCalloc(&groupDesc[k], 1), ret, fail);
      nActiveCtx++;
    }
    NCCLCHECKGOTO(ncclCalloc(&groupStartParams, (size_t)nActiveCtx * startOps), ret, fail);
    NCCLCHECKGOTO(ncclCalloc(&groupDoneParams, (size_t)nActiveCtx * doneOps), ret, fail);

    int p = 0;
    for (int s = 1; s < nNodes; s++) {
      int n = (myNode + s) % nNodes;
      for (int lr = 0; lr < localRanks; lr++) {
        int peer = comm->nodeRanks[n].localRankToRank[lr];
        size_t srcWinOffset =
          ((const uint8_t*)sendbuff + (size_t)peer * perPeerBytes) - (const uint8_t*)sendWin->userPtr;
        size_t peerWinOffset =
          ((const uint8_t*)recvbuff + (size_t)myRank * perPeerBytes) - (const uint8_t*)recvWin->userPtr;

        for (int c = chunkPlan.chunkStart[p]; c < chunkPlan.chunkStart[p + 1]; c++) {
          int k = ceSchedHierA2aChunkCtx(myRank, peer, c, chunkPlan.chunkStart[p], numCtx);
          size_t subBytes = chunkPlan.chunkBytes[c];
          size_t off = chunkPlan.chunkOff[c];

          NCCLCHECKGOTO(ncclRmaProxyPutBuildOp(comm, (struct ncclRmaProxyCtx*)rmaProxyState->rmaProxyCtxs[baseCtx + k],
                                               baseCtx + k, captured, sendWin, srcWinOffset + off, recvWin,
                                               peerWinOffset + off, subBytes, peer, /*signalIdx=*/0, NCCL_SIGNAL,
                                               &groupOps[k][ctxFill[k]++]),
                        ret, fail);
        }
        p++;
      }
    }

    int a = 0;
    for (int k = 0; k < numCtx; k++) {
      if (ctxOps[k] == 0) continue;
      struct ncclRmaProxyCtx* pc = (struct ncclRmaProxyCtx*)rmaProxyState->rmaProxyCtxs[baseCtx + k];
      NCCLCHECKGOTO(ceSchedProxyPutGroupBuildDesc(comm, pc, streamPlan, ctxOps[k], &groupOps[k], baseCtx + k,
                                                  groupDesc[k]),
                    ret, fail);
      NCCLCHECKGOTO(ncclRmaProxyPutGroupStartParams(groupDesc[k], groupStartParams + a * startOps), ret, fail);
      NCCLCHECKGOTO(ncclRmaProxyPutGroupDoneParams(groupDesc[k], groupDoneParams + a * doneOps), ret, fail);
      NCCLCHECKGOTO(ncclRmaProxyEnqueueDesc(pc, &groupDesc[k]), ret, fail);
      a++;
    }
    NCCLCHECKGOTO(ncclCuStreamBatchMemOp(stream, nActiveCtx * startOps, groupStartParams), ret, fail);
  }

  // Phase 4: intra-node alltoall (batched CE memcpy over LSA)
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

  // Phase 5: per-context wait for all remote peers. Skipping contexts with no
  // outbound puts is valid only because the chunk->context mapping commutes in
  // the rank pair, so the contexts carrying our puts to a peer also carry its
  // puts to us.
  {
    for (int k = 0; k < numCtx; k++) {
      if (ctxOps[k] == 0) continue;

      NCCLCHECKGOTO(ncclCalloc(&waitPeers[k], numRemotePeers), ret, fail);
      NCCLCHECKGOTO(ncclCalloc(&waitSigCounts[k], numRemotePeers), ret, fail);

      int wp = 0;
      int p = 0;
      for (int s = 1; s < nNodes; s++) {
        int n = (myNode - s + nNodes) % nNodes;
        for (int lr = 0; lr < localRanks; lr++) {
          int sig = 0;
          int rpeer = comm->nodeRanks[n].localRankToRank[lr];
          for (int c = chunkPlan.chunkStart[p]; c < chunkPlan.chunkStart[p + 1]; c++) {
            if (ceSchedHierA2aChunkCtx(myRank, rpeer, c, chunkPlan.chunkStart[p], numCtx) == k) sig++;
          }
          if (sig > 0) {
            waitPeers[k][wp] = rpeer;
            waitSigCounts[k][wp] = sig;
            wp++;
          }
          p++;
        }
      }

      struct ncclRmaProxyCtx* pc = (struct ncclRmaProxyCtx*)rmaProxyState->rmaProxyCtxs[baseCtx + k];
      NCCLCHECKGOTO(ncclCalloc(&waitDesc[k], 1), ret, fail);
      NCCLCHECKGOTO(ceSchedProxyWaitBuildDesc(comm, pc, streamPlan, wp, &waitPeers[k], &waitSigCounts[k], waitDesc[k]),
                    ret, fail);
      waitStartOpsTotal += ncclRmaProxyWaitStartNumOps(waitDesc[k]);
      waitDoneOpsTotal += ncclRmaProxyWaitDoneNumOps(waitDesc[k]);
    }

    NCCLCHECKGOTO(ncclCalloc(&waitStartParams, waitStartOpsTotal), ret, fail);
    NCCLCHECKGOTO(ncclCalloc(&waitDoneParams, waitDoneOpsTotal), ret, fail);

    int waitStartOff = 0;
    int waitDoneOff = 0;
    for (int k = 0; k < numCtx; k++) {
      if (ctxOps[k] == 0) continue;
      struct ncclRmaProxyCtx* pc = (struct ncclRmaProxyCtx*)rmaProxyState->rmaProxyCtxs[baseCtx + k];
      int nStart = ncclRmaProxyWaitStartNumOps(waitDesc[k]);
      int nDone = ncclRmaProxyWaitDoneNumOps(waitDesc[k]);
      if (waitStartParams != nullptr) {
        NCCLCHECKGOTO(ncclRmaProxyWaitStartParams(waitDesc[k], waitStartParams + waitStartOff), ret, fail);
      }
      if (waitDoneParams != nullptr) {
        NCCLCHECKGOTO(ncclRmaProxyWaitDoneParams(pc, waitDesc[k], waitDoneParams + waitDoneOff), ret, fail);
      }
      NCCLCHECKGOTO(ncclRmaProxyEnqueueDesc(pc, &waitDesc[k]), ret, fail);
      waitStartOff += nStart;
      waitDoneOff += nDone;
    }

    NCCLCHECKGOTO(ncclCuStreamBatchMemOp(stream, waitStartOpsTotal, waitStartParams), ret, fail);
    NCCLCHECKGOTO(ncclCuStreamBatchMemOp(stream, waitDoneOpsTotal, waitDoneParams), ret, fail);
  }

  // Phase 6: outbound puts complete, all contexts in one batch
  NCCLCHECKGOTO(ncclCuStreamBatchMemOp(stream, nActiveCtx * doneOps, groupDoneParams), ret, fail);

  // Phase 7: intra-node barrier
  NCCLCHECKGOTO(ncclMemOpSync(comm, stream, args), ret, fail);

exit:
  ncclCeFreeBatchOpsParams(&ceLocalA2A);
  for (int k = 0; groupOps && k < numCtx; k++) free(groupOps[k]);
  if (groupDesc) {
    for (int k = 0; k < numCtx; k++) {
      if (groupDesc[k] != nullptr) (void)ncclRmaProxyDestroyDesc(comm, &groupDesc[k]);
    }
  }
  if (waitDesc) {
    for (int k = 0; k < numCtx; k++) {
      if (waitDesc[k] != nullptr) (void)ncclRmaProxyDestroyDesc(comm, &waitDesc[k]);
    }
  }
  for (int k = 0; waitPeers && k < numCtx; k++) free(waitPeers[k]);
  for (int k = 0; waitSigCounts && k < numCtx; k++) free(waitSigCounts[k]);
  free(groupOps);
  free(groupStartParams);
  free(groupDoneParams);
  free(groupDesc);
  free(waitStartParams);
  free(waitDoneParams);
  free(waitDesc);
  free(waitPeers);
  free(waitSigCounts);
  free(ctxOps);
  free(ctxFill);
  ceSchedHierCollFreeChunkPlan(&chunkPlan);
  return ret;
fail:
  goto exit;
}

static ncclResult_t ceSchedLaunchColl(struct ncclComm* comm, struct ncclCeSchedStreamPlan* streamPlan,
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

// CUDA user-object destructors run on an internal thread and cannot call CUDA.
// Enqueue the token; ncclScheduleCeTasks reclaims it on an NCCL thread.
static void CUDART_CB ceSchedStreamPlanDestructor(void* token_) {
  struct ncclRmaProxyReclaimToken* token = (struct ncclRmaProxyReclaimToken*)token_;
  if (token == nullptr || token->comm == nullptr) return;
  ncclIntruQueueMpscEnqueue(&token->comm->ceSchedCallbackQueue, token);
}

ncclResult_t ncclCeSchedPollCallbacks(struct ncclComm* comm) {
  ncclResult_t result = ncclSuccess;
  struct ncclRmaProxyReclaimToken* token =
    ncclIntruQueueMpscDequeueAll(&comm->ceSchedCallbackQueue, /*waitSome=*/false);
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

static ncclResult_t launchCeStreamPlan(struct ncclComm* comm, struct ncclCeSchedStreamPlan* streamPlans,
                                       struct ncclStrongStream* depStream) {
  ncclResult_t ret = ncclSuccess;
  struct ncclCeSchedStreamPlan* streamPlan = nullptr;

  for (streamPlan = streamPlans; streamPlan != nullptr; streamPlan = streamPlan->next) {
    cudaStream_t launchStream = streamPlan->stream;
    cudaEvent_t finishedEvent = comm->sharedRes->scratchEvent;
    cudaStream_t acqDepStream;

    if (depStream) {
      NCCLCHECKGOTO(ncclStrongStreamAcquire(streamPlan->graph, depStream, false, &acqDepStream), ret, fail);
      NCCLCHECKGOTO(ncclStreamWaitStream(launchStream, acqDepStream, finishedEvent), ret, fail);
    }

    while (!ncclIntruQueueEmpty(&streamPlan->taskQueue)) {
      struct ncclTaskTuningInfo* taskInfo = ncclIntruQueueDequeue(&streamPlan->taskQueue);
      struct ncclCeCollArgs args = {};

      NCCLCHECKGOTO(ceSchedCollArgsFromTuningInfo(comm, taskInfo, &args), ret, fail);
      NCCLCHECKGOTO(ceSchedLaunchColl(comm, streamPlan, &args), ret, fail);
      ceSchedFreeTuningInfoRaw(comm, taskInfo);
    }

    if (depStream) {
      CUDACHECKGOTO(cudaEventRecord(finishedEvent, launchStream), ret, fail);
      NCCLCHECKGOTO(ncclStreamAdvanceToEvent(streamPlan->graph, acqDepStream, finishedEvent), ret, fail);
      NCCLCHECKGOTO(ncclStrongStreamRelease(streamPlan->graph, depStream, false), ret, fail);
    }
    if (streamPlan->captured && streamPlan->reclaimToken != nullptr) {
      struct ncclRmaProxyReclaimToken* token = streamPlan->reclaimToken;
      streamPlan->reclaimToken = nullptr;
      ret = ncclCudaGraphAddDestructor(streamPlan->graph, ceSchedStreamPlanDestructor, token);
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

static ncclResult_t launchCeTasks(struct ncclComm* comm, struct ncclCeSchedPlan* cePlan,
                                  struct ncclStrongStream* depStream) {
  ncclResult_t ret = ncclSuccess;

  NCCLCHECKGOTO(launchCeStreamPlan(comm, cePlan->noncapturedStreamPlans, depStream), ret, fail);
  NCCLCHECKGOTO(launchCeStreamPlan(comm, cePlan->capturedStreamPlans, depStream), ret, fail);

exit:
  return ret;
fail:
  goto exit;
}

static ncclResult_t scheduleCeTasks(struct ncclComm* comm,
                                    struct ncclIntruQueue<struct ncclTaskTuningInfo, &ncclTaskTuningInfo::next>* queue,
                                    struct ncclCeSchedPlan* cePlanOut) {
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

ncclResult_t ncclScheduleCeTasks(struct ncclComm* comm,
                                 struct ncclIntruQueue<struct ncclTaskTuningInfo, &ncclTaskTuningInfo::next>* queue,
                                 struct ncclStrongStream* depStream) {
  NCCLCHECK(ncclCeSchedPollCallbacks(comm));
  if (ncclIntruQueueEmpty(queue)) return ncclSuccess;

  if (!comm->ceColl.initialized) {
    WARN("CE collective is not initialized");
    return ncclInternalError;
  }

  struct ncclCeSchedPlan cePlan = {};
  NCCLCHECK(scheduleCeTasks(comm, queue, &cePlan));
  NCCLCHECK(launchCeTasks(comm, &cePlan, depStream));
  return ncclSuccess;
}
