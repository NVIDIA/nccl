/*************************************************************************
 * Copyright (c) 2017-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "enqueue.h"
#include "argcheck.h"
#include "coll_net.h"
#include "gdrwrap.h"
#include "bootstrap.h"
#include "channel.h"
#include "cudawrap.h"
#include "transport.h"
#include <cassert>
#include <cstring> // std::memcpy
#include <cinttypes> // PRIx64

NCCL_PARAM(L1SharedMemoryCarveout, "L1_SHARED_MEMORY_CARVEOUT", 0);

static ncclResult_t initCollWorkElem(struct ncclInfo* collInfo, struct ncclWorkElem* work);
static ncclResult_t setCollWorkElem(uint64_t workCount, uint64_t workOffset, size_t lastChunkCount, struct ncclWorkElem* work);
static ncclResult_t initCollWorkElemReg(struct ncclComm* comm, struct ncclWorkElem* work, struct ncclChannel* channel, ncclRegBufferType regBufType, void* regBufSend[], void* regBufRecv[], struct ncclWorkElemReg* workElemReg);
static ncclResult_t computeCollChunkInfo(struct ncclInfo* collInfo, size_t nBytes, int nChannels);
static ncclResult_t initCollProxyOp(struct ncclInfo* collInfo, int channelId, uint64_t opCount, uint32_t nsteps, struct ncclProxyOp* proxyOp);
static ncclResult_t getTunerInfo(struct ncclInfo* collInfo, int collNetSupport, int nvlsSupport, int numPipeOps);
static ncclResult_t topoGetAlgoInfo(struct ncclInfo* collInfo, int collNetSupport, int nvlsSupport, int numPipeOps);
static ncclResult_t getChannnelThreadInfo(struct ncclInfo* collInfo);
static ncclResult_t computeCollWorkFunc(struct ncclInfo* collInfo);
static ncclResult_t getPatternInfo(struct ncclInfo* collInfo);
static ncclResult_t getLoopInfo(struct ncclInfo* collInfo);
static ncclResult_t getCollNetSupport(struct ncclInfo* info, int* collNetSupport);

// Returns maximum kernel stack size of all CUDA kernels
ncclResult_t ncclInitKernelsForDevice(int cudaArch, size_t* maxStackSize) {
  ncclResult_t result = ncclSuccess;

  if (maxStackSize) *maxStackSize = 0;
  int carveout = ncclParamL1SharedMemoryCarveout();

  for (int k=0; k < ncclDevKernelCount; k++) {
    void* fn = ncclDevKernelList[k];
    if (fn == nullptr) continue;

    if (maxStackSize) {
      cudaFuncAttributes attr = {0};
      CUDACHECKGOTO(cudaFuncGetAttributes(&attr, fn), result, ignore0);
      if (attr.localSizeBytes > *maxStackSize) *maxStackSize = attr.localSizeBytes;
    ignore0:;
    }
    if (carveout) {
      CUDACHECKGOTO(cudaFuncSetAttribute(fn,
        cudaFuncAttributePreferredSharedMemoryCarveout, carveout),
        result, ignore1);
    ignore1:;
    }
    if (ncclShmemDynamicSize(cudaArch) != 0) {
      CUDACHECKGOTO(cudaFuncSetAttribute(fn,
        cudaFuncAttributeMaxDynamicSharedMemorySize, ncclShmemDynamicSize(cudaArch)),
        result, next_kernel);
    }
  next_kernel:;
  }
  return result;
}

/*****************************************************************************/
/*       Launch system : synchronization and CUDA kernel launch              */
/*****************************************************************************/

static void appendWorkElemColl(
    struct ncclComm* comm, struct ncclKernelPlan* plan, int channelId,
    int funcIndex, struct ncclWorkElem const *elem) {
  struct ncclKernelPlan::Channel* chan = &plan->channels[channelId];
  struct ncclWorkList* q = ncclIntruQueueTail(&chan->workQueue);
  if (q && funcIndex == q->work.header.funcIndex
        && elem->nWarps == q->work.elems[0].nWarps
        && chan->nWorkElem < NCCL_MAX_WORK_ELEMENTS
        && ncclWorkTypeColl == q->work.header.type) {
    int e = chan->nWorkElem++;
    q->work.elems[e] = *elem; // C++ struct assignment
    return;
  }
  q = ncclMemoryStackAlloc<struct ncclWorkList>(&comm->memScoped);
  q->work.header.type = ncclWorkTypeColl;
  q->work.header.funcIndex = funcIndex;
  q->work.elems[0] = *elem; // C++ struct assignment
  chan->nWorkElem = 1;
  chan->nWork += 1;
  ncclIntruQueueEnqueue(&chan->workQueue, q);
}

static void appendWorkElemColl(
    struct ncclComm* comm, struct ncclKernelPlan* plan, int channelId,
    int funcIndex, struct ncclWorkElemReg const *elem) {
  struct ncclKernelPlan::Channel* chan = &plan->channels[channelId];
  struct ncclWorkList* q = ncclIntruQueueTail(&chan->workQueue);
  if (q && funcIndex == q->work.header.funcIndex
        && elem->elem.nWarps == q->work.regElems[0].elem.nWarps
        && chan->nWorkElem < NCCL_MAX_WORK_ELEMENTS_REG
        && ncclWorkTypeRegColl == q->work.header.type) {
    int e = chan->nWorkElem++;
    q->work.regElems[e] = *elem; // C++ struct assignment
    q->work.regElems[e].elem.isUsed = 1;
    return;
  }
  q = ncclMemoryStackAlloc<struct ncclWorkList>(&comm->memScoped);
  q->work.header.type = ncclWorkTypeRegColl;
  q->work.header.funcIndex = funcIndex;
  q->work.regElems[0] = *elem; // C++ struct assignment
  q->work.regElems[0].elem.isUsed = 1;
  chan->nWorkElem = 1;
  chan->nWork += 1;
  ncclIntruQueueEnqueue(&chan->workQueue, q);
}

static void finishWorkP2p(struct ncclWork* work) {
  int nElem = 0;
  for (int e=0; e < NCCL_MAX_WORK_ELEMENTS_P2P; e++) {
    if (work->p2pElems[e].p2pType != ncclWorkP2pTypeUnused)
      nElem = e+1;
  }
  int nGroup = 1;
  while (nGroup < nElem) nGroup *= 2;
  int nWarp = 1;
  while (nWarp*nGroup <= (NCCL_MAX_NTHREADS/WARP_SIZE)/2) nWarp *= 2;
  for (int i=0; i < nGroup; i++) {
    work->p2pElems[i].ngroups = nGroup;
    work->p2pElems[i].warpStart = i*(NCCL_MAX_NTHREADS/WARP_SIZE)/nGroup;
    int extraWarp = nWarp >= 2 ? i%2 : 0;
    work->p2pElems[i].nWarps = nWarp + extraWarp;
  }
}

static void finishWork(struct ncclWork* work) {
  if (work->header.type == ncclWorkTypeP2p) {
    finishWorkP2p(work);
  }
}

static void appendWorkElemP2p(
    struct ncclComm* comm, struct ncclKernelPlan* plan, int channelId,
    struct ncclWorkElemP2p const *elem, bool fuseOk
  ) {
  int funcIndex = ncclDevFuncId_P2p();
  struct ncclKernelPlan::Channel* chan = &plan->channels[channelId];
  struct ncclWorkList* q = ncclIntruQueueTail(&chan->workQueue);
  if (q && funcIndex == q->work.header.funcIndex) {
    if (!fuseOk) goto NewWork;
    if (chan->p2pTailElem[elem->p2pType-1] < NCCL_MAX_WORK_ELEMENTS_P2P) {
      for (int e = -2 + chan->p2pTailElem[elem->p2pType-1]; e >= 0; e -= 2) {
        // Can't have multiple elements of the same ncclWork communicate with the
        // same peer otherwise they would attempt to use that connection concurrently.
        if (q->work.p2pElems[e].peer == elem->peer)
          goto NewWork;
      }
      int e = chan->p2pTailElem[elem->p2pType-1];
      q->work.p2pElems[e] = *elem; // C++ struct assignment
      chan->p2pTailElem[elem->p2pType-1] += 2;
      return;
    }
  NewWork:
    finishWorkP2p(&q->work);
  }
  q = ncclMemoryStackAlloc<struct ncclWorkList>(&comm->memScoped);
  q->work.header.type = ncclWorkTypeP2p;
  q->work.header.funcIndex = ncclDevFuncId_P2p();
  chan->p2pTailElem[ncclWorkP2pTypeRecv-1] = 0;
  chan->p2pTailElem[ncclWorkP2pTypeSend-1] = 1;
  q->work.p2pElems[chan->p2pTailElem[elem->p2pType-1]] = *elem; // C++ struct assignment
  chan->p2pTailElem[elem->p2pType-1] += 2;
  chan->nWork += 1;
  ncclIntruQueueEnqueue(&chan->workQueue, q);
}

static ncclResult_t addProxyOpIfNeeded(struct ncclComm* comm, struct ncclKernelPlan* plan, struct ncclProxyOp* op) {
  bool needed = true;
  NCCLCHECK(ncclProxySaveOp(comm, op, &needed));
  if (needed) {
    struct ncclProxyOp* q = ncclMemoryPoolAlloc<struct ncclProxyOp>(&comm->memPool_ncclProxyOp, &comm->memPermanent);
    *q = *op; // C++ struct assignment
    ncclIntruQueueEnqueue(&plan->channels[op->channelId].proxyOpQueue, q);
  }
  return ncclSuccess;
}

static ncclResult_t computeCollSteps(struct ncclInfo* collInfo, size_t workCount, uint32_t* steps) {
  struct ncclComm* comm = collInfo->comm;
  if (collInfo->coll == ncclFuncAllReduce) {
    if (collInfo->algorithm == NCCL_ALGO_RING)
      *steps = DIVUP(workCount, comm->nRanks * collInfo->chunkCount) * (comm->nRanks - 1) * 2 * collInfo->chunkSteps;
    else if (collInfo->algorithm == NCCL_ALGO_COLLNET_DIRECT)
      *steps = DIVUP(workCount, comm->channels[0].collnetDirect.nHeads * collInfo->chunkCount) * collInfo->chunkSteps;
    else if (collInfo->algorithm == NCCL_ALGO_NVLS || collInfo->algorithm == NCCL_ALGO_NVLS_TREE)
      *steps = DIVUP(workCount, comm->channels[0].nvls.nHeads * collInfo->chunkCount) * collInfo->chunkSteps;
    else
      *steps = DIVUP(workCount, collInfo->chunkCount) * collInfo->chunkSteps;
  } else if (collInfo->coll == ncclFuncReduceScatter) {
    if (collInfo->algorithm == NCCL_ALGO_RING)
      *steps = DIVUP(workCount, collInfo->chunkCount) * (comm->nRanks - 1) * collInfo->chunkSteps;
    else
      *steps = DIVUP(workCount, collInfo->chunkCount) * collInfo->chunkSteps;
  } else if (collInfo->coll == ncclFuncAllGather) {
    if (collInfo->algorithm == NCCL_ALGO_RING)
      *steps = DIVUP(workCount, collInfo->chunkCount) * (comm->nRanks - 1) * collInfo->chunkSteps;
    else
      *steps = DIVUP(workCount, collInfo->chunkCount) * collInfo->chunkSteps;
  } else {
    *steps = DIVUP(workCount, collInfo->chunkCount) * collInfo->chunkSteps;
  }
  return ncclSuccess;
}

static ncclResult_t computeCollAlignCount(struct ncclInfo* collInfo, size_t* alignCount) {
  if (collInfo->protocol == NCCL_PROTO_SIMPLE) {
    *alignCount = NCCL_SIMPLE_ALIGNMENT / ncclTypeSize(collInfo->datatype);
  } else if (collInfo->protocol == NCCL_PROTO_LL128) {
    *alignCount = NCCL_LL128_ALIGNMENT_PER_WARP / ncclTypeSize(collInfo->datatype) * (collInfo->nThreads / WARP_SIZE);
  } else {
    *alignCount = NCCL_LL_ALIGNMENT_PER_THREAD / ncclTypeSize(collInfo->datatype) * collInfo->nThreads;
  }
  return ncclSuccess;
}

static ncclResult_t computeCollLastChunkInfo(struct ncclInfo* collInfo, size_t workCount, size_t alignCount, size_t* lastChunkCount) {
  struct ncclComm* comm = collInfo->comm;

  if (collInfo->coll == ncclFuncAllReduce) {
    if (collInfo->algorithm == NCCL_ALGO_RING) {
      size_t remCount = workCount % (comm->nRanks * collInfo->chunkCount);
      *lastChunkCount = DIVUP(DIVUP(remCount, comm->nRanks), alignCount) * alignCount;
    } else if (collInfo->algorithm == NCCL_ALGO_NVLS || collInfo->algorithm == NCCL_ALGO_NVLS_TREE) {
      size_t remCount = workCount % (comm->channels[0].nvls.nHeads * collInfo->chunkCount);
      *lastChunkCount = DIVUP(DIVUP(remCount, comm->channels[0].nvls.nHeads), alignCount) * alignCount;
    } else if (collInfo->algorithm == NCCL_ALGO_COLLNET_DIRECT) {
      size_t remCount = workCount % (comm->channels[0].collnetDirect.nHeads * collInfo->chunkCount);
      *lastChunkCount = DIVUP(DIVUP(remCount, comm->channels[0].collnetDirect.nHeads), alignCount) * alignCount;
    } else {
      *lastChunkCount = collInfo->chunkCount;
    }
  } else {
    *lastChunkCount = collInfo->chunkCount;
  }
  return ncclSuccess;
}

static ncclResult_t getCollnetLoopInfo(struct ncclInfo* collInfo, int* nstepsPerLoop, int* nchunksPerLoop) {
  switch (collInfo->pattern) {
    case ncclPatternCollnetChain:
      *nstepsPerLoop = *nchunksPerLoop = 1; break;
    case ncclPatternNvls:
      *nstepsPerLoop = 1; *nchunksPerLoop = collInfo->comm->channels[0].nvls.nHeads; break;
    case ncclPatternCollnetDirect:
      *nstepsPerLoop = 1; *nchunksPerLoop = collInfo->comm->channels[0].collnetDirect.nHeads; break;
    default:
      WARN("Unknown collnet pattern %d", collInfo->pattern);
      return ncclInternalError;
  }
  return ncclSuccess;
}

static ncclResult_t addCollnetCollToPlan(
    struct ncclComm* comm, struct ncclKernelPlan* plan, int usableChannels,
    struct ncclInfo* collInfo, int* nWorkBudget
  ) {
  ncclResult_t ret = ncclSuccess;
  struct ncclKernelPlan::Channel *chans = plan->channels;
  struct ncclWorkElem workElem;
  uint64_t opCount = uint64_t(plan->collOpCount++) << 1 | 0;
  ncclRegBufferType regBufType = collInfo->regBufType;
  int nChannels = std::min(collInfo->nChannels, usableChannels);
  size_t countPerChannel = DIVUP(collInfo->count, nChannels);
  uint32_t typeSize = ncclTypeSize(collInfo->datatype);
  int steps, nchunksPerLoop, nstepsPerLoop, nLoop;

  NCCLCHECK(computeCollChunkInfo(collInfo, collInfo->nBytes, collInfo->nChannels));
  NCCLCHECKGOTO(initCollWorkElem(collInfo, &workElem), ret, fail);
  workElem.nChannels = nChannels;

  NCCLCHECKGOTO(getCollnetLoopInfo(collInfo, &nstepsPerLoop, &nchunksPerLoop), ret, fail);
  nLoop = (int)DIVUP(collInfo->nBytes, (size_t)nChannels * nchunksPerLoop * collInfo->chunkSize);
  steps = nstepsPerLoop * nLoop * collInfo->chunkSteps;

  for (int bid = 0; bid < nChannels; bid++) {
    workElem.bid = bid;
    // Add work elem
    *nWorkBudget += chans[bid].nWork;
    if (regBufType == NCCL_REGULAR_BUFFER) {
      appendWorkElemColl(comm, plan, bid, collInfo->workFuncIndex, &workElem);
    } else {
      struct ncclWorkElemReg workElemReg;
      NCCLCHECKGOTO(initCollWorkElemReg(comm, &workElem, &comm->channels[bid], regBufType, collInfo->regBufSend, collInfo->regBufRecv, &workElemReg), ret, fail);
      appendWorkElemColl(comm, plan, bid, collInfo->workFuncIndex, &workElemReg);
    }
    *nWorkBudget -= chans[bid].nWork; // subtract delta of chans[c].nWork

    // Add proxy task. Empty collectives do not make it to the proxy thread
    // since they don't imply synchronization for the user like p2p.
    if (collInfo->nBytes != 0) {
      struct ncclProxyOp proxyOp;
      NCCLCHECKGOTO(initCollProxyOp(collInfo, bid, opCount, steps, &proxyOp), ret, fail);
      NCCLCHECKGOTO(addProxyOpIfNeeded(comm, plan, &proxyOp), ret, fail);
    }

    chans[bid].collBytes += countPerChannel * typeSize;
  }

  plan->threadPerBlock = std::max(plan->threadPerBlock, collInfo->nThreads);
  if (!plan->kernelSpecialized) {
    plan->kernelFn = ncclDevKernelForFunc[collInfo->workFuncIndex];
    plan->kernelSpecialized = ncclDevKernelForFuncIsSpecialized[collInfo->workFuncIndex];
  }

  if (comm->rank == 0) {
    TRACE(NCCL_COLL, "collnetColl enqueue coll %s(%s, %s, %s, %s), nChannels %d, count %ld (nbytes %ld), usableChannel %d, chunkCount %d, funcIndex %d, nThreads %d", collInfo->opName, ncclOpToString(collInfo->op), ncclDatatypeToString(collInfo->datatype), ncclAlgoToString(collInfo->algorithm), ncclProtoToString(collInfo->protocol), collInfo->nChannels, collInfo->count, collInfo->workBytes, usableChannels, collInfo->chunkCount, collInfo->workFuncIndex, collInfo->nThreads);
  }

exit:
  return ret;
fail:
  goto exit;
}

static ncclResult_t addTunedCollToPlan(
    struct ncclComm* comm, struct ncclKernelPlan* plan, int usableChannels,
    struct ncclInfo* collInfo, int* nWorkBudget
  ) {
  ncclResult_t ret = ncclSuccess;
  struct ncclKernelPlan::Channel *chans = plan->channels;
  struct ncclWorkElem workElem;
  uint64_t opCount = uint64_t(plan->collOpCount++) << 1 | 0;
  uint64_t workCount;
  uint64_t workOffset = 0;
  uint32_t typeSize = ncclTypeSize(collInfo->datatype);
  ncclRegBufferType regBufType = collInfo->regBufType;
  size_t alignCount, lastChunkCount;
  int least[/*nBid*/MAXCHANNELS];
  int maxIndexInLeast;
  size_t maxBytesInLeast;
  int nChannels = std::min(collInfo->nChannels, usableChannels);
  int rnChannels = 0;
  size_t countPerChannels;
  size_t remCount = collInfo->count;

  NCCLCHECKGOTO(computeCollAlignCount(collInfo, &alignCount), ret, fail);
  countPerChannels = DIVUP(DIVUP(collInfo->count, nChannels), alignCount) * alignCount;
  nChannels = DIVUP(collInfo->count, countPerChannels);
  NCCLCHECKGOTO(computeCollChunkInfo(collInfo, collInfo->nBytes, nChannels), ret, fail);
  NCCLCHECKGOTO(initCollWorkElem(collInfo, &workElem), ret, fail);

  // Choose the `nBid` least loaded channels to do the work. This ensures
  // all bids go to different channels in case they need to synchronize.
  least[0] = 0;
  maxIndexInLeast = 0;
  maxBytesInLeast = chans[0].collBytes;
  // Initialize least[] such that the first nBid channels are accounted for.
  for (int b = 1; b < nChannels; b++) {
    least[b] = b;
    if (maxBytesInLeast < chans[b].collBytes) {
      maxIndexInLeast = b;
      maxBytesInLeast = chans[b].collBytes;
    }
  }
  // Sort in the rest of the channels. If a channel has less work than the max
  // member of least[], replace that member and compute the new max. We only
  // sort channels when coll algo is not collnet.
  for (int c = nChannels; c < usableChannels; c++) {
    if (chans[c].collBytes < maxBytesInLeast) {
      least[maxIndexInLeast] = c;
      maxBytesInLeast = chans[least[0]].collBytes;
      maxIndexInLeast = 0;
      for (int b = 1; b < nChannels; b++) {
        if (maxBytesInLeast < chans[least[b]].collBytes) {
          maxIndexInLeast = b;
          maxBytesInLeast = chans[least[b]].collBytes;
        }
      }
    }
  }

  for (int bid = 0; bid < nChannels && remCount > 0; bid++) {
    int c = least[bid];

    workCount = std::min(countPerChannels, remCount);
    NCCLCHECKGOTO(computeCollLastChunkInfo(collInfo, workCount, alignCount, &lastChunkCount), ret, fail);
    NCCLCHECKGOTO(setCollWorkElem(workCount, workOffset, lastChunkCount, &workElem), ret, fail);

    // Add work elem
    *nWorkBudget += chans[c].nWork;
    if (regBufType == NCCL_REGULAR_BUFFER) {
      appendWorkElemColl(comm, plan, c, collInfo->workFuncIndex, &workElem);
    } else {
      struct ncclWorkElemReg workElemReg;
      NCCLCHECKGOTO(initCollWorkElemReg(comm, &workElem, &comm->channels[c], regBufType, collInfo->regBufSend, collInfo->regBufRecv, &workElemReg), ret, fail);
      appendWorkElemColl(comm, plan, c, collInfo->workFuncIndex, &workElemReg);
    }
    *nWorkBudget -= chans[c].nWork; // subtract delta of chans[c].nWork

    // Add proxy task. Empty collectives do not make it to the proxy thread
    // since they don't imply synchronization for the user like p2p.
    if (collInfo->nBytes != 0) {
      uint32_t steps;
      struct ncclProxyOp proxyOp;
      NCCLCHECKGOTO(computeCollSteps(collInfo, workCount, &steps), ret, fail);
      NCCLCHECKGOTO(initCollProxyOp(collInfo, c, opCount, steps, &proxyOp), ret, fail);
      NCCLCHECKGOTO(addProxyOpIfNeeded(comm, plan, &proxyOp), ret, fail);
    }

    remCount -= workCount;
    chans[c].collBytes += workCount * typeSize;
    workOffset += workCount;
    rnChannels++;
  }

  plan->threadPerBlock = std::max(plan->threadPerBlock, collInfo->nThreads);
  if (!plan->kernelSpecialized) {
    plan->kernelFn = ncclDevKernelForFunc[collInfo->workFuncIndex];
    plan->kernelSpecialized = ncclDevKernelForFuncIsSpecialized[collInfo->workFuncIndex];
  }

  if (comm->rank == 0) {
    TRACE(NCCL_COLL, "tunedColl enqueue coll %s(%s, %s, %s, %s), nChannels %d, count %ld (nbytes %ld), usableChannel %d, chunkCount %d, lastChunkCount %ld, funcIndex %d, nThreads %d", collInfo->opName, ncclOpToString(collInfo->op), ncclDatatypeToString(collInfo->datatype), ncclAlgoToString(collInfo->algorithm), ncclProtoToString(collInfo->protocol), rnChannels, collInfo->count, collInfo->workBytes, usableChannels, collInfo->chunkCount, lastChunkCount, collInfo->workFuncIndex, collInfo->nThreads);
  }

exit:
  return ret;
fail:
  goto exit;
}

static ncclResult_t addCBDCollToPlan(
    struct ncclComm* comm, struct ncclKernelPlan* plan, int usableChannels,
    struct ncclInfo* collInfo, int* nWorkBudget
  ) {
  ncclResult_t ret = ncclSuccess;
  struct ncclKernelPlan::Channel *chans = plan->channels;
  size_t enqBytes;
  uint64_t opCount = uint64_t(plan->collOpCount++) << 1 | 0;
  size_t typeSize = ncclTypeSize(collInfo->datatype);
  size_t workBytesTotal = collInfo->count * typeSize;
  size_t workCountTotal = collInfo->count;
  struct ncclWorkElem workElem;
  size_t workOffset = 0;
  size_t workCount;
  ncclRegBufferType regBufType = collInfo->regBufType;
  size_t alignCount;
  size_t lastChunkCount;
  int rnChannel = 0;

  NCCLCHECKGOTO(computeCollChunkInfo(collInfo, collInfo->aggnBytes, collInfo->nChannels), ret, fail);
  NCCLCHECKGOTO(computeCollAlignCount(collInfo, &alignCount), ret, fail);
  NCCLCHECKGOTO(initCollWorkElem(collInfo, &workElem), ret, fail);
  for (int c = 0; c < usableChannels; c++) {
    if (plan->maxBytesPerChannel <= chans[c].collBytes) continue;
    if (workBytesTotal == 0) break;
    enqBytes = std::min(plan->maxBytesPerChannel - chans[c].collBytes, workBytesTotal);
    workCount = std::min(DIVUP(DIVUP(enqBytes, typeSize), alignCount) * alignCount, workCountTotal);
    enqBytes = workCount * typeSize;

    NCCLCHECKGOTO(computeCollLastChunkInfo(collInfo, workCount, alignCount, &lastChunkCount), ret, fail);
    NCCLCHECKGOTO(setCollWorkElem(workCount, workOffset, lastChunkCount, &workElem), ret, fail);

    // Add work elem
    *nWorkBudget += chans[c].nWork;
    if (regBufType == NCCL_REGULAR_BUFFER) {
      appendWorkElemColl(comm, plan, c, collInfo->workFuncIndex, &workElem);
    } else {
      struct ncclWorkElemReg workElemReg;
      NCCLCHECKGOTO(initCollWorkElemReg(comm, &workElem, &comm->channels[c], regBufType, collInfo->regBufSend, collInfo->regBufRecv, &workElemReg), ret, fail);
      appendWorkElemColl(comm, plan, c, collInfo->workFuncIndex, &workElemReg);
    }
    *nWorkBudget -= chans[c].nWork; // subtract delta of chans[c].nWork

    // Add proxy task. Empty collectives do not make it to the proxy thread
    // since they don't imply synchronization for the user like p2p.
    if (collInfo->nBytes != 0) {
      uint32_t steps;
      struct ncclProxyOp proxyOp;
      NCCLCHECKGOTO(computeCollSteps(collInfo, workCount, &steps), ret, fail);
      NCCLCHECKGOTO(initCollProxyOp(collInfo, c, opCount, steps, &proxyOp), ret, fail);
      NCCLCHECKGOTO(addProxyOpIfNeeded(comm, plan, &proxyOp), ret, fail);
    }

    workBytesTotal -= enqBytes;
    workCountTotal -= workCount;
    chans[c].collBytes += enqBytes;
    workOffset += workCount;
    rnChannel++;
  }

  plan->threadPerBlock = std::max(plan->threadPerBlock, collInfo->nThreads);
  if (!plan->kernelSpecialized) {
    plan->kernelFn = ncclDevKernelForFunc[collInfo->workFuncIndex];
    plan->kernelSpecialized = ncclDevKernelForFuncIsSpecialized[collInfo->workFuncIndex];
  }

  if (comm->rank == 0) {
    TRACE(NCCL_COLL, "CBDColl enqueue coll %s(%s, %s, %s, %s), nChannels %d, count %ld (nbytes %ld), usableChannel %d, maxBytesPerChannel %ld, chunkCount %d, lastChunkCount %ld, funcIndex %d, nThreads %d", collInfo->opName, ncclOpToString(collInfo->op), ncclDatatypeToString(collInfo->datatype), ncclAlgoToString(collInfo->algorithm), ncclProtoToString(collInfo->protocol), rnChannel, collInfo->count, collInfo->workBytes, usableChannels, plan->maxBytesPerChannel, collInfo->chunkCount, lastChunkCount, collInfo->workFuncIndex, collInfo->nThreads);
  }

exit:
  return ret;
fail:
  goto exit;
}

NCCL_PARAM(P2pLLThreshold, "P2P_LL_THRESHOLD", 16384);

// Put p2p op in plan assuming there is space in nWorkBudget, so you must
// ensure *nWorkBudget >= 1 upon entry.
static ncclResult_t addP2pToPlan(
    struct ncclComm* comm, struct ncclKernelPlan* plan, int* nWorkBudget,
    bool isSendNotRecv, int peer, int chunk, void *addr, size_t bytes, bool fuseOk
  ) {
  struct ncclInfo info = {
    isSendNotRecv ? ncclFuncSend : ncclFuncRecv,
    isSendNotRecv ? "Send" : "Recv",
    nullptr, addr, bytes, ncclInt8, ncclSum, peer, comm, (cudaStream_t)0,
    /*Args*/1, 1
  };

  int channelId;
  NCCLCHECK(ncclChannelCompute(comm, peer, chunk%comm->p2pnChannelsPerPeer, info.coll, &channelId));
  info.channelId = channelId;

  // 1 is connIndex
  struct ncclConnInfo* conn = isSendNotRecv ?
    &comm->channels[channelId].peers[peer]->send[1].conn : &comm->channels[channelId].peers[peer]->recv[1].conn;
  info.protocol = ((conn->buffs[NCCL_PROTO_LL] != nullptr) && bytes <= ncclParamP2pLLThreshold()) ? NCCL_PROTO_LL : NCCL_PROTO_SIMPLE;

  int reg = 0;
  if (info.protocol == NCCL_PROTO_SIMPLE) {
    struct ncclReg* regRecord;
    NCCLCHECK(ncclRegFind(comm, addr, bytes, &regRecord));
    reg = regRecord && regRecord->nDevs ? 1 : 0;
  }

  struct ncclProxyOp proxyOp = {};
  // May tune chunksize and set proxyOp.reg=0 if not using the network.
  NCCLCHECK(ncclProxyComputeP2p(&info, &proxyOp, reg));

  struct ncclWorkElemP2p elem = {0};
  elem.proto = info.protocol;
  elem.peer = peer;
  elem.nWarps = NCCL_MAX_NTHREADS/WARP_SIZE;
  elem.reg = proxyOp.reg;
  elem.p2pType = isSendNotRecv ? ncclWorkP2pTypeSend : ncclWorkP2pTypeRecv;
  elem.buffLo32 = uint32_t(reinterpret_cast<uintptr_t>(addr));
  elem.buffHi32 = reinterpret_cast<uintptr_t>(addr)>>32;
  elem.countLo32 = uint32_t(bytes);
  elem.countHi32 = bytes>>32;
  elem.chunkSize = info.chunkSize; // computed by ncclProxyComputeP2p

  *nWorkBudget += plan->channels[channelId].nWork;
  appendWorkElemP2p(comm, plan, channelId, &elem, fuseOk);
  *nWorkBudget -= plan->channels[channelId].nWork;

  // Calculate the opCount after appendWorkElemP2p since it will always return
  // with channel->nWork equal to one plus the work index this p2p settled in.
  proxyOp.opCount = uint64_t(plan->channels[channelId].nWork)<<1 | 1;
  NCCLCHECK(addProxyOpIfNeeded(comm, plan, &proxyOp));
  return ncclSuccess;
}

static void finishPlan(struct ncclKernelPlan* plan) {
  int channelUbound = 0;
  int channelCount = 0;
  uint64_t channelMask = 0;
  bool hasProxyOps = false;
  for (int c=0; c < MAXCHANNELS; c++) {
    struct ncclWorkList* tail = ncclIntruQueueTail(&plan->channels[c].workQueue);
    if (tail != nullptr) {
      channelUbound = c+1;
      channelCount += 1;
      channelMask |= 1ull<<c;
      tail->work.header.isLast = 1;
      finishWork(&tail->work);
    }
    hasProxyOps |= !ncclIntruQueueEmpty(&plan->channels[c].proxyOpQueue);
  }
  plan->channelUbound = channelUbound;
  plan->channelCount = channelCount;
  plan->channelMask = channelMask;
  plan->hasProxyOps = hasProxyOps;
  plan->threadPerBlock = std::max(plan->threadPerBlock, 3*WARP_SIZE);
}

int64_t ncclParamLocalRegister();
NCCL_PARAM(GraphRegister, "GRAPH_REGISTER", 1);

static ncclResult_t registerIntraNodeBuffers(
    struct ncclComm* comm, struct ncclKernelPlan* plan, struct ncclInfo* info
  ) {
  ncclResult_t result = ncclSuccess;

  info->regBufType = NCCL_REGULAR_BUFFER;
#if CUDART_VERSION >= 11030
  if ((info->algorithm == NCCL_ALGO_NVLS || info->algorithm == NCCL_ALGO_NVLS_TREE) && comm->nvlsRegSupport) {
    bool regBufUsed = false;
    const void *sendbuff = info->sendbuff;
    void *recvbuff = info->recvbuff;

    if (info->coll == ncclFuncAllGather)
      sendbuff = NULL;
    else if (info->coll == ncclFuncReduceScatter)
      recvbuff = NULL;

    /* first try local registration. */
    if (ncclParamLocalRegister()) {
      ncclNvlsLocalRegisterBuffer(comm, sendbuff, recvbuff, info->sendbuffSize, info->recvbuffSize, &regBufUsed, info->regBufSend, info->regBufRecv);
    }

    if (regBufUsed == false && plan->persistent && ncclParamGraphRegister()) {
      ncclNvlsGraphRegisterBuffer(comm, plan, sendbuff, recvbuff, info->sendbuffSize, info->recvbuffSize, &regBufUsed, info->regBufSend, info->regBufRecv);
    }

    if (regBufUsed) {
      /* tweak NVLS channels usage; for registered NVLS buffer, we only need 4/5 channels to
       * saturate bandwidth. */
      if (comm->nNodes == 1) {
        if (info->coll == ncclFuncReduceScatter)
          info->nChannels = std::max(comm->config.minCTAs, std::min(comm->config.maxCTAs, 5));
        else
          info->nChannels = std::max(comm->config.minCTAs, std::min(comm->config.maxCTAs, 4));
      } else {
        info->nChannels = std::max(comm->config.minCTAs, std::min(comm->config.maxCTAs, 6));
      }

      info->regBufType = NCCL_NVLS_REG_BUFFER;
    }
  } else if (info->algorithm == NCCL_ALGO_COLLNET_DIRECT &&   // limited to CollNetDirect for now
    comm->intraHighestTransportType == TRANSPORT_P2P && // only when all ranks can p2p each other
    comm->intraRanks < comm->localRanks &&  // only with inter-process & intra-node peers
    plan->persistent && 0) {
    /* Disable CollnetDirect registration since it does not support cuMem* allocated memory. */
    int localRank = comm->localRank;
    cudaPointerAttributes sattr, rattr;

    CUDACHECK(cudaPointerGetAttributes(&sattr, info->sendbuff));
    CUDACHECK(cudaPointerGetAttributes(&rattr, info->recvbuff));
    if (sattr.type != cudaMemoryTypeDevice || rattr.type != cudaMemoryTypeDevice) return ncclSuccess;

    if (CUPFN(cuMemGetAddressRange) == nullptr) return ncclSuccess;

    struct HandlePair {
      cudaIpcMemHandle_t ipc[2]; // {send, recv}
      size_t offset[2]; // {send, recv}
    };
    struct HandlePair handles[NCCL_MAX_LOCAL_RANKS];

    CUDACHECKGOTO(cudaIpcGetMemHandle(&handles[localRank].ipc[0], (void*)info->sendbuff), result, fallback);
    CUDACHECKGOTO(cudaIpcGetMemHandle(&handles[localRank].ipc[1], (void*)info->recvbuff), result, fallback);

    void *baseSend, *baseRecv;
    size_t size;
    CUCHECK(cuMemGetAddressRange((CUdeviceptr *)&baseSend, &size, (CUdeviceptr)info->sendbuff));
    handles[localRank].offset[0] = (char*)info->sendbuff - (char*)baseSend;
    CUCHECK(cuMemGetAddressRange((CUdeviceptr *)&baseRecv, &size, (CUdeviceptr)info->recvbuff));
    handles[localRank].offset[1] = (char*)info->recvbuff - (char*)baseRecv;

    NCCLCHECK(bootstrapIntraNodeAllGather(comm->bootstrap, comm->localRankToRank, comm->localRank, comm->localRanks, handles, sizeof(struct HandlePair)));

    // Open handles locally
    for (int i=0; i < comm->localRanks; i++) {
      if (i == localRank) { // Skip self
        info->regBufSend[i] = nullptr;
        info->regBufRecv[i] = nullptr;
      } else {
        for (int sr=0; sr < 2; sr++) {
          // Get base address of mapping
          void* base;
          CUDACHECK(cudaIpcOpenMemHandle(&base, handles[i].ipc[sr], cudaIpcMemLazyEnablePeerAccess));
          // Get real buffer address by adding offset in the mapping
          (sr == 0 ? info->regBufSend : info->regBufRecv)[i] = (char*)base + handles[i].offset[sr];
          // Enqueue reminder to close memory handle
          struct ncclPointerList* q = ncclMemoryPoolAlloc<struct ncclPointerList>(&comm->memPool_ncclPointerList, &comm->memPermanent);
          q->ptr = base;
          ncclIntruQueueEnqueue(&plan->ipcMemQueue, q);
        }
      }
    }
    info->regBufType = NCCL_IPC_REG_BUFFER;
  }
fallback:
#endif
  return result;
}

static ncclResult_t getCBDCollnChannel(struct ncclKernelPlan* plan, struct ncclInfo* collInfo, int usableChannels) {
  size_t firstEnqBytes;
  size_t workBytesTotal = collInfo->workBytes;
  struct ncclKernelPlan::Channel *chans = plan->channels;
  int typeSize = ncclTypeSize(collInfo->datatype);
  size_t maxCount = DIVUP(plan->maxBytesPerChannel, typeSize);

  if (workBytesTotal == 0) {
    collInfo->nChannels = 1;
    goto exit;
  }

  for (int c = 0; c < usableChannels; c++) {
    if (plan->maxBytesPerChannel <= chans[c].collBytes) continue;
    firstEnqBytes = std::min(plan->maxBytesPerChannel - chans[c].collBytes, workBytesTotal);
    firstEnqBytes = DIVUP(firstEnqBytes, typeSize) * typeSize;
    collInfo->nChannels = 1 + DIVUP((workBytesTotal - firstEnqBytes) / typeSize, maxCount);
    break;
  }

exit:
  return ncclSuccess;
}

static ncclResult_t scheduleCollTasksToPlan(
    struct ncclComm* comm, struct ncclKernelPlan* plan, int* nWorkBudget
  ) {
  struct ncclTasks* tasks = &comm->tasks;
  size_t totalCBDBytes = tasks->workBytesTotal;
  struct ncclInfo* collInfo;

  if (!ncclIntruQueueEmpty(&tasks->collQueue)) {
    int usableChannels = 0, accChannels = 0;

    tasks->usableChannels = 1;
    while (!ncclIntruQueueEmpty(&tasks->collQueue)) {
      collInfo = ncclIntruQueueDequeue(&tasks->collQueue);
      if (collInfo->count == 0) continue;
      if (collInfo->algorithm == NCCL_ALGO_UNDEF) {
        struct ncclInfo* aggInfo = ncclMemoryStackAlloc<struct ncclInfo>(&comm->memScoped);
        struct ncclInfo* nextInfo = collInfo->next;
        int nvlsSupport;
        int collNetSupport;

        memcpy(aggInfo, collInfo, sizeof(struct ncclInfo));
        while (nextInfo) {
          if (nextInfo->coll == aggInfo->coll && nextInfo->opFull.op == aggInfo->opFull.op && nextInfo->datatype == aggInfo->datatype) {
            aggInfo->count += nextInfo->count;
            nextInfo = nextInfo->next;
          } else {
            break;
          }
        }

        nvlsSupport = comm->nvlsSupport && ncclNvlsSupported(aggInfo->opFull.op, aggInfo->datatype);
        NCCLCHECK(getCollNetSupport(aggInfo, &collNetSupport));
        NCCLCHECK(ncclInfoSetDerived(aggInfo, comm->nRanks));
        NCCLCHECK(getTunerInfo(aggInfo, collNetSupport, nvlsSupport, 1));
        NCCLCHECK(topoGetAlgoInfo(aggInfo, collNetSupport, nvlsSupport, 1));
        NCCLCHECK(getChannnelThreadInfo(aggInfo));
        NCCLCHECK(computeCollWorkFunc(aggInfo));
        NCCLCHECK(getPatternInfo(aggInfo));

        // Try to assign algo and proto to all possible collectives
        nextInfo = collInfo;
        while (nextInfo) {
          if (nextInfo->coll == aggInfo->coll && nextInfo->opFull.op == aggInfo->opFull.op && nextInfo->datatype == aggInfo->datatype) {
            NCCLCHECK(ncclInfoSetDerived(nextInfo, comm->nRanks));
            NCCLCHECK(getTunerInfo(nextInfo, collNetSupport, nvlsSupport, 1));
            nextInfo->algorithm = aggInfo->algorithm;
            nextInfo->protocol = aggInfo->protocol;
            nextInfo->nThreads = aggInfo->nThreads;
            nextInfo->pattern = aggInfo->pattern;
            nextInfo->workFuncIndex = aggInfo->workFuncIndex;
            nextInfo->aggnBytes = aggInfo->nBytes;

            NCCLCHECK(getChannnelThreadInfo(nextInfo));
            // if possible, start registration
            registerIntraNodeBuffers(comm, plan, nextInfo);
            // accumulate channels
            accChannels += nextInfo->nChannels;
            nextInfo = nextInfo->next;
          } else {
            break;
          }
        }
      } // end of aggInfo

      if (collInfo->algorithm == NCCL_ALGO_NVLS || collInfo->algorithm == NCCL_ALGO_NVLS_TREE) {
        usableChannels = std::max(usableChannels, comm->nvlsChannels);
      } else {
        usableChannels = std::max(usableChannels, comm->collChannels);
      }

      if (collInfo->algorithm == NCCL_ALGO_COLLNET_DIRECT || collInfo->algorithm == NCCL_ALGO_COLLNET_CHAIN || (collInfo->algorithm == NCCL_ALGO_NVLS && comm->nNodes > 1)) {
        // substract collective which needs to be executed separately
        totalCBDBytes -= collInfo->workBytes;
        tasks->workBytesTotal -= collInfo->workBytes;
        ncclIntruQueueEnqueue(&tasks->collnetQueue, collInfo);
      } else if (collInfo->userTuned) {
        // substract collective which needs to be executed separately
        totalCBDBytes -= collInfo->workBytes;
        tasks->workBytesTotal -= collInfo->workBytes;
        ncclIntruQueueEnqueue(&tasks->collTunedQueue, collInfo);
      } else {
        ncclIntruQueueEnqueue(&tasks->collCBDQueue, collInfo);
      }
    }

    tasks->usableChannels = std::min(usableChannels, accChannels);
  }

  /* Calculate maxBytesPerChannel for CBD colls and it should be 16 bytes aligned
   * Note: it it not hard upper bound for maxBytes, we can relax it if any optimization
   * is needed */
  plan->maxBytesPerChannel = DIVUP(DIVUP(totalCBDBytes, tasks->usableChannels), NCCL_BYTES_ALIGNMENT) * NCCL_BYTES_ALIGNMENT;
  // First enqueue CBD colls
  while (!ncclIntruQueueEmpty(&tasks->collCBDQueue)) {
    // Get nChannels and peek whether the budget allows before we enqueue
    collInfo = ncclIntruQueueHead(&tasks->collCBDQueue);
    collInfo->nChannels = DIVUP(collInfo->aggnBytes * tasks->usableChannels, totalCBDBytes);
    // Haven't got nChannels info yet, relax the budget boundary a bit.
    if (*nWorkBudget < collInfo->nChannels) return ncclSuccess;

    collInfo = ncclIntruQueueDequeue(&tasks->collCBDQueue);
    NCCLCHECK(addCBDCollToPlan(comm, plan, tasks->usableChannels, collInfo, nWorkBudget));
    tasks->nTasksColl -= 1;
    tasks->workBytesTotal -= collInfo->count * ncclTypeSize(collInfo->datatype);
  }

  // Then enqueue collnet colls
  while (!ncclIntruQueueEmpty(&tasks->collnetQueue)) {
    collInfo = ncclIntruQueueHead(&tasks->collnetQueue);
    if (*nWorkBudget < collInfo->nChannels) return ncclSuccess;

    collInfo = ncclIntruQueueDequeue(&tasks->collnetQueue);
    NCCLCHECK(addCollnetCollToPlan(comm, plan, tasks->usableChannels, collInfo, nWorkBudget));
    tasks->nTasksColl -= 1;
  }

  // Finally enqueue user-tuned colls
  while (!ncclIntruQueueEmpty(&tasks->collTunedQueue)) {
    collInfo = ncclIntruQueueHead(&tasks->collTunedQueue);
    if (*nWorkBudget < collInfo->nChannels) return ncclSuccess;

    collInfo = ncclIntruQueueDequeue(&tasks->collTunedQueue);
    NCCLCHECK(addTunedCollToPlan(comm, plan, tasks->usableChannels, collInfo, nWorkBudget));
    tasks->nTasksColl -= 1;
  }

  return ncclSuccess;
}

static size_t calcP2pChunkSize(size_t totalSize, int minChannels, int maxChannels, size_t minSize, size_t maxSize) {
  size_t size = std::max(minSize, divUp(totalSize, minChannels));
  int nChannels = minChannels;
  while (size > maxSize && nChannels <= maxChannels/2) {
    nChannels *= 2;
    size = divUp(totalSize, nChannels);
  }
  return alignUp(size, minSize);
}

static ncclResult_t scheduleP2pTasksToPlan(
    struct ncclComm* comm, struct ncclKernelPlan* plan, int* nWorkBudget
  ) {
  struct ncclTasks* tasks = &comm->tasks;
  int nRanks = comm->nRanks;
  struct ncclTasks::Peer* peers = tasks->peers;
  int const *sendOrder = tasks->p2pSendOrder;
  int const *recvOrder = tasks->p2pRecvOrder;

  plan->threadPerBlock = std::max(plan->threadPerBlock, NCCL_MAX_NTHREADS);
  if (!plan->kernelSpecialized) {
    plan->kernelFn = ncclDevKernelForFunc[ncclDevFuncId_P2p()];
    plan->kernelSpecialized = ncclDevKernelForFuncIsSpecialized[ncclDevFuncId_P2p()];
  }

  // Compute how much to split operations
  // Natural step size matching buffer steps.
  ssize_t stepSize = comm->p2pChunkSize;
  // Try to use all channels
  int nChannelsMax = comm->p2pnChannelsPerPeer;
  int nChannelsMin = nChannelsMax;
  // Try to use all channels, but one channel per operation.
  while (nChannelsMin*nRanks > comm->p2pnChannels && nChannelsMin > 1) nChannelsMin /= 2;

  bool fuseOk = false;
  // We can perform 8 send/recv per round per CTA. Make sure we jump between fused blocks at node boundaries.
  while (tasks->nTasksP2p != 0) {
    for (int i=0; i < tasks->p2pOrderSteps; i++) {
      int sendPeer = sendOrder[i];
      int recvPeer = recvOrder[i];
      struct ncclTaskP2p* send = sendPeer != -1 ? ncclIntruQueueHead(&peers[sendPeer].sendQueue) : NULL;
      struct ncclTaskP2p* recv = recvPeer != -1 ? ncclIntruQueueHead(&peers[recvPeer].recvQueue) : NULL;
      if (sendPeer == comm->rank) {
        if (recvPeer != comm->rank) {
          WARN("Sendrecv plan not aligned for self");
          return ncclInternalError;
        }
        if (send && recv == nullptr) {
          WARN("Trying to send to self without a matching recv");
          return ncclInvalidUsage;
        }
        if (send == nullptr && recv) {
          WARN("Trying to recv to self without a matching send");
          return ncclInvalidUsage;
        }
      }
      if (send != nullptr || recv != nullptr) {
        char* recvPtr = recv ? (char*)recv->buff : nullptr;
        char* sendPtr = send ? (char*)send->buff : nullptr;
        ssize_t recvBytes = recv ? recv->bytes : 0;
        ssize_t sendBytes = send ? send->bytes : 0;
        ssize_t minSize = comm->nNodes > 1 ? stepSize/2 : stepSize/8;
        ssize_t maxSize = comm->nNodes > 1 ? stepSize : stepSize*32;
        ssize_t recvChunkBytesMax = calcP2pChunkSize(recvBytes, nChannelsMin, nChannelsMax, minSize, maxSize);
        ssize_t sendChunkBytesMax = calcP2pChunkSize(sendBytes, nChannelsMin, nChannelsMax, minSize, maxSize);
        // Zero size send/recv are syncs, encode here with -1.
        recvBytes = recv && recvBytes == 0 ? -1 : recvBytes;
        sendBytes = send && sendBytes == 0 ? -1 : sendBytes;
        // Advance to current chunk. Syncs will always have chunk=0 so no effect on the -1.
        if (recv) recvPtr   += recv->chunk*recvChunkBytesMax;
        if (recv) recvBytes -= recv->chunk*recvChunkBytesMax;
        if (send) sendPtr   += send->chunk*sendChunkBytesMax;
        if (send) sendBytes -= send->chunk*sendChunkBytesMax;

        do {
          if ((i % (NCCL_MAX_WORK_ELEMENTS_P2P/2)) == 0) fuseOk = false;
          ssize_t recvChunkBytes = std::min(recvBytes, recvChunkBytesMax); // -1 preserved
          ssize_t sendChunkBytes = std::min(sendBytes, sendChunkBytesMax);
          if (recvChunkBytes != 0) {
            if (recvChunkBytes == -1) recvChunkBytes = 0;
            if (*nWorkBudget < 1) return ncclSuccess; // ensure room in budget
            NCCLCHECK(addP2pToPlan(comm, plan, nWorkBudget, /*isSendNotRecv=*/false, recvPeer, recv->chunk, recvPtr, recvChunkBytes, fuseOk));
            fuseOk = true;
            recvPtr += recvChunkBytes;
            recvBytes -= recvChunkBytes;
            recv->chunk += 1;
            if (recvBytes <= 0) {
              recvBytes = 0; // in case still -1
              ncclIntruQueueDequeue(&peers[recvPeer].recvQueue);
              tasks->nTasksP2p -= 1;
            }
          }
          if (sendChunkBytes != 0) {
            if (sendChunkBytes == -1) sendChunkBytes = 0;
            if (*nWorkBudget < 1) return ncclSuccess; // ensure room in budget
            NCCLCHECK(addP2pToPlan(comm, plan, nWorkBudget, /*isSendNotRecv=*/true, sendPeer, send->chunk, sendPtr, sendChunkBytes, fuseOk));
            fuseOk = true;
            sendPtr += sendChunkBytes;
            sendBytes -= sendChunkBytes;
            send->chunk += 1;
            if (sendBytes <= 0) {
              sendBytes = 0; // in case still -1
              ncclIntruQueueDequeue(&peers[sendPeer].sendQueue);
              tasks->nTasksP2p -= 1;
            }
          }
        } while (sendBytes != 0 || recvBytes != 0);
      }
    }
  }
  return ncclSuccess;
}

// Comparison of monotonic rolling counters.
static inline bool rollingLess32(uint32_t a, uint32_t b) {
  constexpr uint32_t PositiveMax = uint32_t(-1)>>1;
  return a-b > PositiveMax;
}
static inline uint32_t rollingMin32(uint32_t a, uint32_t b) {
  constexpr uint32_t PositiveMax = uint32_t(-1)>>1;
  return (b-a <= PositiveMax) ? a : b;
}

// Spin until its safe to increase comm->workFifoSent to desiredSent.
static void waitWorkFifoAvailable(struct ncclComm* comm, uint32_t desiredSent) {
  if (__builtin_expect(rollingLess32(comm->workFifoAckdMin + comm->workFifoDepth, desiredSent), false)) {
    while (1) {
      // We have to poll for notifications from device.
      uint32_t* doneLive = comm->workFifoDone;
      uint32_t ackd[MAXCHANNELS];
      for (int c=0; c < MAXCHANNELS; c++) {
        ackd[c] = __atomic_load_n(&doneLive[c], __ATOMIC_RELAXED);
      }
      // Compiler-only fence to prevent fusion of loops to encourage dense loads.
      __atomic_signal_fence(__ATOMIC_SEQ_CST);

      uint32_t ackdAll = comm->workFifoSent;
      for (int c=0; c < MAXCHANNELS; c++) {
        // ackdAll is min over all non-quiesced channels
        if (ackd[c] != comm->channels[c].workFifoSent)
          ackdAll = rollingMin32(ackdAll, ackd[c]);
      }

      // Compiler only fence to prevent fusion of loops to encourage dense stores.
      __atomic_signal_fence(__ATOMIC_SEQ_CST);

      for (int c=0; c < MAXCHANNELS; c++) {
        // Advance counter on quiesced channels so they don't lag behind
        // too far where they could get lost in 32-bit wraparound.
        if (ackd[c] == comm->channels[c].workFifoSent) {
          comm->channels[c].workFifoSent = ackdAll;
          __atomic_store_n(&doneLive[c], ackdAll, __ATOMIC_RELAXED);
        }
      }
      comm->workFifoAckdMin = ackdAll;

      // See if that was enough.
      if (!rollingLess32(comm->workFifoAckdMin + comm->workFifoDepth, desiredSent)) break;
      sched_yield();
    }
  }
}

static ncclResult_t uploadWork(struct ncclComm* comm, struct ncclKernelPlan* plan) {
  bool persistent = plan->persistent;
  int channelUbound = plan->channelUbound;
  int nWork = 0;
  for (int c=0; c < channelUbound; c++) nWork += plan->channels[c].nWork;

  struct ncclWork* workHeap;
  if (!persistent) {
    workHeap = comm->workFifoHeap;
  } else {
    workHeap = ncclMemoryStackAlloc<struct ncclWork>(&comm->memScoped, nWork);
  }
  uint32_t ixMask = persistent ? ~uint32_t(0) : comm->workFifoDepth-1;
  uint32_t ixSent;
  if (persistent) {
    ixSent = 0;
  } else {
    ixSent = comm->workFifoSent;
    // First work for a channel has to be at workHeap+blockIdx.x which means
    // we cannot tolerate fifo wraparound. So round up to the wrap boundary
    // if not doing so would incur crossing it.
    if (((ixSent + plan->channelCount-1) & ixMask) < (ixSent & ixMask)) {
      ixSent = (ixSent + ixMask) & ~ixMask;
      // Need to update workFifoSent so waitWorkFifoAvailable() knows we've
      // skipped those elements. Consider if all the channels report quiesced,
      // this way the skipped slots will be considered consumed as well.
      comm->workFifoSent = ixSent;
    }
    waitWorkFifoAvailable(comm, ixSent + nWork);
  }
  uint32_t ixHead = ixSent;
  ixSent += plan->channelCount;
  int channelsWithWork = 0; // number of channels below `c` with work structs.
  for (int c=0; c < channelUbound; c++) {
    struct ncclWorkList* q = ncclIntruQueueHead(&plan->channels[c].workQueue);
    // Offset of first work equals number of channels below with work.
    uint32_t ix = ixHead + channelsWithWork;
    channelsWithWork += q != nullptr ? 1 : 0;
    while (q != nullptr) {
      if (q->next != nullptr) {
        q->work.header.workNext = int32_t(ixSent & ixMask) - int32_t(ixHead & ixMask);
      } else {
        q->work.header.inFifo = !persistent ? 1 : 0;
        // Tell channel to ack us back ix+1 indicating that all slots up to and
        // including ix have been consumed.
        q->work.header.doneAcks = ix+1;
        comm->channels[c].workFifoSent = ix+1;
      }
      workHeap[ix & ixMask] = q->work; // C++ struct assignment
      q = q->next;
      if (q != nullptr) ix = ixSent++;
    }
  }

  if (!persistent) {
    comm->workFifoSent = ixSent;
    if (comm->workFifoHeapGdrHandle != nullptr) wc_store_fence();
    plan->workHead = &comm->devWorkFifoHeap[ixHead & ixMask];
  } else {
    NCCLCHECK(ncclCudaMalloc(&plan->workHead, nWork));
    NCCLCHECK(ncclCudaMemcpy(plan->workHead, workHeap, nWork));
  }
  return ncclSuccess;
}

static ncclResult_t uploadProxyOps(struct ncclComm* comm, struct ncclKernelPlan* plan) {
  uint64_t collOpCount = comm->sharedRes->collOpCount;
  // Advance comm's collOpCount by number of colls in this plan.
  comm->sharedRes->collOpCount += plan->collOpCount;

  uint64_t p2pOpBump[MAXCHANNELS];
  struct ncclProxyOp* heads[MAXCHANNELS];
  uint64_t headIds[MAXCHANNELS];
  int nHeads = 0;
  for (int c=0; c < plan->channelUbound; c++) {
    p2pOpBump[c] = 0;
    heads[c] = ncclIntruQueueHead(&plan->channels[c].proxyOpQueue);
    nHeads += (heads[c] != nullptr) ? 1 : 0;
    headIds[c] = (heads[c] != nullptr) ? heads[c]->opCount : uint64_t(-1);
  }

  while (nHeads != 0) {
    int minChan = -1;
    uint64_t minId = uint64_t(-1);
    // We store the heads[c]->opCount in headIds[c] specifically to remove indirect
    // loads from this loop which speeds it up considerably.
    for (int c=0; c < plan->channelUbound; c++) {
      uint64_t id = headIds[c];
      id = (id>>1 | id<<63); // Move tag bit to order collectives before p2p's
      if (id < minId) { minChan = c; minId = id; }
    }

    struct ncclProxyOp* q = heads[minChan];
    uint64_t oldId = headIds[minChan]; // same as q->opCount
    // Advance heads[c]
    heads[minChan] = q->enqNext;
    if (q->enqNext == nullptr) nHeads -= 1;
    headIds[minChan] = (q->enqNext != nullptr) ? q->enqNext->opCount : uint64_t(-1);

    // Ignoring the bottom tag bit, opCount's are zero-based within plan so
    // translate them to the tip of the comm's history.
    if (oldId & 1) { // p2p
      // opCount is monotonic increasing within a plan's channel so just
      // remember last value to compute max.
      p2pOpBump[minChan] = (oldId>>1) + 1; // +1 to ensure next plan doesn't collide
      q->opCount = (comm->sharedRes->p2pOpCount[minChan]<<1) + oldId;
    } else { // coll
      q->opCount = (collOpCount<<1) + oldId;
    }

    NCCLCHECK(ncclProxySaveOp(comm, q, nullptr));
    q->opCount = oldId; // Restore for next uploadProxyOps()
    if (!plan->persistent) {
      // Non-persistent kernels upload ops only once so can be free'd here.
      ncclMemoryPoolFree(&comm->memPool_ncclProxyOp, q);
    }
  }

  for (int c=0; c < plan->channelUbound; c++) {
    // Erase proxyOpQueue since all ops were free'd back to mempool.
    if (!plan->persistent) ncclIntruQueueConstruct(&plan->channels[c].proxyOpQueue);
    // Advance channel's p2pOpCount by number of p2p's in this plan channel.
    comm->sharedRes->p2pOpCount[c] += p2pOpBump[c];
  }
  return ncclSuccess;
}

static ncclResult_t hostStreamPlanTask(struct ncclComm* comm, struct ncclKernelPlan* plan) {
  NCCLCHECK(uploadProxyOps(comm, plan));
  NCCLCHECK(ncclProxyStart(comm));
  if (!plan->persistent) {
    // Notify main thread of our reclaiming. This will reclaim plan concurrently.
    ncclIntruQueueMpscEnqueue(&comm->callbackQueue, &plan->reclaimer);
  }
  return ncclSuccess;
}

static void CUDART_CB hostStreamPlanCallback(void *plan_) {
  NVTX3_FUNC_RANGE_IN(nccl_domain);
  struct ncclKernelPlan* plan = (struct ncclKernelPlan*)plan_;
  ncclResult_t result = hostStreamPlanTask(plan->comm, plan);
  if (result != ncclSuccess) {
    WARN("hostStreamPlanCallback() failed : %s", ncclGetErrorString(result));
  }
}

static ncclResult_t reclaimPlan(struct ncclComm* comm, struct ncclCommCallback* me) {
  struct ncclKernelPlan* plan = (struct ncclKernelPlan*)me; // cast from first member `reclaim`
  if (plan->persistent) {
    comm->persistentRefs -= 1;
    NCCLCHECK(ncclCudaFree(plan->workHead));
    for (int c=0; c < plan->channelUbound; c++) {
      struct ncclProxyOp* q = ncclIntruQueueHead(&plan->channels[c].proxyOpQueue);
      while (q != nullptr) {
        struct ncclProxyOp* q1 = q->enqNext;
        ncclMemoryPoolFree(&comm->memPool_ncclProxyOp, q);
        q = q1;
      }
    }
    while (!ncclIntruQueueEmpty(&plan->ipcMemQueue)) {
      struct ncclPointerList* q = ncclIntruQueueDequeue(&plan->ipcMemQueue);
      CUDACHECKIGNORE(cudaIpcCloseMemHandle(q->ptr));
      ncclMemoryPoolFree(&comm->memPool_ncclPointerList, q);
    }
    /* free mcHandle */
    while (!ncclIntruQueueEmpty(&plan->nvlsMcHandleQueue)) {
      struct ncclNvlsMcHandleList* obj = ncclIntruQueueDequeue(&plan->nvlsMcHandleQueue);
      NCCLCHECK(ncclNvlsDeregBuffer(&obj->mcHandle, obj->ptr, obj->dev, obj->size));
      INFO(NCCL_NVLS, "rank %d - deregistered buffer %p on device %d, size %ld", comm->rank, (void*)obj->ptr, obj->dev, obj->size);
      ncclMemoryPoolFree(&comm->memPool_ncclNvlsHandleList, obj);
    }
  }
  ncclMemoryPoolFree(&comm->memPool_ncclKernelPlan, plan);
  return ncclSuccess;
}

static void persistentDestructor(void* plans_) {
  struct ncclKernelPlan* plan = (struct ncclKernelPlan*)plans_;
  struct ncclComm* comm = plan->comm;
  while (plan != nullptr) {
    struct ncclKernelPlan* next = plan->next;
    ncclIntruQueueMpscEnqueue(&comm->callbackQueue, &plan->reclaimer);
    plan = next;
  }
}

ncclResult_t ncclLaunchPrepare(struct ncclComm* comm) {
  ncclResult_t result = ncclSuccess;
  struct ncclTasks* tasks = &comm->tasks;
  bool persistent = ncclCudaGraphValid(tasks->capturingGraph);
  int nPlans = 0;

  // Poll for callbacks sent to us from other threads. Typically these free
  // resources from to our memory pools.
  NCCLCHECK(ncclCommPollCallbacks(comm, /*waitSome=*/false));

  // We already have one frame present which holds all of our tasks (which we
  // are about to schedule). Now push an additional frame for allocating
  // work structs (see appendWorkElem() variants all use scoped allocation).
  ncclMemoryStackPush(&comm->memScoped);

  if (tasks->nTasksColl + tasks->nTasksP2p != 0) {
    do {
      struct ncclKernelPlan* plan = ncclMemoryPoolAlloc<struct ncclKernelPlan>(&comm->memPool_ncclKernelPlan, &comm->memPermanent);
      ncclIntruQueueEnqueue(&comm->planQueue, plan);
      nPlans += 1;
      plan->comm = comm;
      plan->reclaimer.fn = reclaimPlan;
      plan->persistent = persistent;

      // Non-persistent kernels fill up at most half of our fifo per kernel.
      int nWorkBudget = plan->persistent ? INT_MAX : comm->workFifoDepth/2;
      int nWorkBudgetOld = nWorkBudget;

      // Drain coll tasks first. This is essential since we partition tasks based
      // on the work budget and p2p work isn't collective. If we were to drain p2p
      // first, the place where we cut the kernel could vary by rank which would
      // cause the "shortest channel first" channel picker to have divergent results.
      if (tasks->nTasksColl != 0) {
        NCCLCHECKGOTO(scheduleCollTasksToPlan(comm, plan, &nWorkBudget), result, failure);
      }
      // And only drain p2p tasks once colls are depleted.
      if (tasks->nTasksColl == 0 && tasks->nTasksP2p != 0) {
        NCCLCHECKGOTO(scheduleP2pTasksToPlan(comm, plan, &nWorkBudget), result, failure);
      }
      if (nWorkBudget == nWorkBudgetOld) {
        // We weren't able to fit any tasks into our budget which means now we're
        // stuck in an infinite loop. We defer this check until here, instead of
        // doing it in comm init, to permit testing with insanely shallow queues
        // for cases where that's expected to still work (e.g. few channels).
        WARN("'NCCL_WORK_FIFO_DEPTH=%d' is too small. Minimum value is %d", comm->workFifoDepth, 2*MAXCHANNELS);
        result = ncclInvalidUsage;
        goto failure;
      }
      finishPlan(plan);
    } while (tasks->nTasksColl + tasks->nTasksP2p != 0);

    struct ncclKernelPlan* planHead = ncclIntruQueueHead(&comm->planQueue);
    comm->unlaunchedPlansHead = planHead;

    // Semantically we want these dependencies for the kernels launched:
    //   1. Launch host task on hostStream.
    //   2. Launch kernel, depends on all of {deviceStream, hostStream, userStream[i]...}
    //   3. {deviceStream, userStream[i]...} depend on kernel.
    // We achieve this by:
    //   1. userStream[0] waits on deviceStream
    //   2. deviceStream waits on each of userStream[1...]
    //   3. host task launch on hostStream
    //   4. userStream[0] waits on hostStream
    //   5. kernel launch on userStream[0]
    //   6. deviceStream waits on userStream[0]
    //   7. userStream[1...] each waits on deviceStream
    // The two-level fan-in fan-out is because ncclStrongStreamWaitStream() requires
    // at least one of the two streams to be strong-stream.
    cudaStream_t launchStream = tasks->streams->stream;
    NCCLCHECKGOTO(ncclStrongStreamAcquire(tasks->capturingGraph, &comm->sharedRes->deviceStream), result, failure);

    // Create dependency for device stream on user streams. First from extra user
    // streams to deviceStream. Then deviceStream to first user stream.
    for (struct ncclCudaStreamList* l=tasks->streams->next; l != nullptr; l = l->next) {
      NCCLCHECKGOTO(ncclStrongStreamWaitStream(tasks->capturingGraph, &comm->sharedRes->deviceStream, l->stream), result, failure);
    }
    NCCLCHECKGOTO(ncclStrongStreamWaitStream(tasks->capturingGraph, launchStream, &comm->sharedRes->deviceStream), result, failure);

    if (persistent || comm->persistentRefs != 0 || ncclCudaLaunchBlocking) {
      // We have to launch host tasks to push proxy args. We are careful to only
      // do this if necessary since host tasks impose a high performance cost in CUDA.
      bool acquired = false;
      for (struct ncclKernelPlan* plan=planHead; plan != nullptr; plan = plan->next) {
        if (plan->hasProxyOps) {
          if (!acquired) {
            acquired = true;
            NCCLCHECKGOTO(ncclStrongStreamAcquire(tasks->capturingGraph, &comm->sharedRes->hostStream), result, failure);
          }
          NCCLCHECKGOTO(ncclStrongStreamLaunchHost(tasks->capturingGraph, &comm->sharedRes->hostStream, hostStreamPlanCallback, plan), result, failure);
        }
      }
      if (acquired) {
        // Make to-be-launched kernels dependent on just-launched host stream tasks.
        NCCLCHECKGOTO(ncclStrongStreamWaitStream(tasks->capturingGraph, launchStream, &comm->sharedRes->hostStream), result, failure);
        NCCLCHECKGOTO(ncclStrongStreamRelease(tasks->capturingGraph, &comm->sharedRes->hostStream), result, failure);
      }
    }

    if (persistent) {
      comm->persistentRefs += nPlans;
      NCCLCHECKGOTO(ncclCudaGraphAddDestructor(tasks->capturingGraph, persistentDestructor, (void*)planHead), result, failure);
    }
  }

  if (false) {
  failure:
    ncclMemoryStackPop(&comm->memScoped); // deallocate ncclWork's
  }
  return result;
}

ncclResult_t ncclLaunchKernelBefore_NoUncapturedCuda(struct ncclComm* comm, struct ncclKernelPlan* plan) {
  // This code is called after we've checked in to the intra-process barrier
  // but before launching the kernel. We are not allowed to call CUDA unless the
  // kernel launch is captured.
  NCCLCHECK(uploadWork(comm, plan));
  return ncclSuccess;
}

#if CUDART_VERSION >= 12000
// NCCL uses the "Remote" Mem Sync domain by default
NCCL_PARAM(MemSyncDomain, "MEM_SYNC_DOMAIN", cudaLaunchMemSyncDomainRemote);
#endif

ncclResult_t ncclLaunchKernel(struct ncclComm* comm, struct ncclKernelPlan* plan) {
  struct ncclTasks* tasks = &comm->tasks;
  void *fn = plan->kernelFn;
  cudaStream_t launchStream = tasks->streams->stream;
  dim3 grid = {(unsigned)plan->channelCount, 1, 1};
  dim3 block = {(unsigned)plan->threadPerBlock, 1, 1};
  size_t smem = ncclShmemDynamicSize(comm->cudaArch);
  void *args[3] = {&comm->devComm, &plan->channelMask, &plan->workHead};

  #if CUDART_VERSION >= 11080
  int driverVersion;
  NCCLCHECK(ncclCudaDriverVersion(&driverVersion));
  if (driverVersion >= 11080) {
    int compCap = comm->compCap;
    unsigned int clusterSize = (compCap == 90) ? comm->config.cgaClusterSize : 0;

    cudaLaunchConfig_t launchConfig = {0};
    cudaLaunchAttribute launchAttrs[3];
    int attrs = 0;
    /* Cooperative Group Array (CGA)
     * On sm90 and later we have an extra level of hierarchy where we
     * can group together several blocks within the Grid, called
     * Thread Block Clusters.
     * Clusters enable multiple thread blocks running concurrently
     * across multiple SMs to synchronize and collaboratively fetch
     * and exchange data. A cluster of blocks are guaranteed to be
     * concurrently scheduled onto a group of SMs.
     * The maximum value is 8 and it must be divisible into the grid dimensions
     */
    if (clusterSize) {
      // Grid dimension must be divisible by clusterSize
      if (grid.x % clusterSize) clusterSize = 1;
      launchAttrs[attrs].id = cudaLaunchAttributeClusterDimension;
      launchAttrs[attrs++].val.clusterDim = {clusterSize, 1, 1};
      launchAttrs[attrs].id = cudaLaunchAttributeClusterSchedulingPolicyPreference;
      launchAttrs[attrs++].val.clusterSchedulingPolicyPreference = cudaClusterSchedulingPolicySpread;
    }
    #if CUDART_VERSION >= 12000
    if (compCap >= 90 && driverVersion >= 12000) {
      // Set the NCCL Mem Sync domain on CUDA 12.0 and later (sm90)
      launchAttrs[attrs].id = cudaLaunchAttributeMemSyncDomain;
      launchAttrs[attrs++].val.memSyncDomain = (cudaLaunchMemSyncDomain) ncclParamMemSyncDomain();
    }
    #endif
    launchConfig.gridDim = grid;
    launchConfig.blockDim = block;
    launchConfig.dynamicSmemBytes = smem;
    launchConfig.attrs = launchAttrs;
    launchConfig.numAttrs = attrs;
    launchConfig.stream = launchStream;

    CUDACHECK(cudaLaunchKernelExC(&launchConfig, fn, args));
    return ncclSuccess;
  }
  #endif
  // Standard kernel launch
  CUDACHECK(cudaLaunchKernel(fn, grid, block, args, smem, launchStream));
  return ncclSuccess;
}

ncclResult_t ncclLaunchKernelAfter_NoCuda(struct ncclComm* comm, struct ncclKernelPlan* plan) {
  if (!(plan->persistent || comm->persistentRefs != 0 || ncclCudaLaunchBlocking)) {
    // We are not using the host stream for proxy ops and reclaimation submission.
    NCCLCHECK(hostStreamPlanTask(comm, plan));
  } else {
    // We are using the host stream for proxy ops and reclaimation submission.
    // Only plans with proxy ops have a callback pushed by ncclLaunchPrepare.
    // Since non-persistent plans also require reclaimation, we have to do it
    // here.
    if (!plan->persistent && !plan->hasProxyOps) {
      ncclIntruQueueMpscEnqueue(&comm->callbackQueue, &plan->reclaimer);
    }
  }
  return ncclSuccess;
}

ncclResult_t ncclLaunchFinish(struct ncclComm* comm) {
  ncclResult_t result = ncclSuccess;
  struct ncclTasks* tasks = &comm->tasks;
  tasks->workBytesTotal = 0; // Just in case subtraction during scheduleCollTasksToPlan() doesn't get to 0

  // Deallocate ncclWork's. This frame exists so long as ncclLaunchPrepare
  // succeeded, and if it ncclLaunchPrepare didn't succeed we wouldn't be here.
  ncclMemoryStackPop(&comm->memScoped);

  if (!ncclIntruQueueEmpty(&comm->planQueue)) {
    // Reset queue to empty without destroying plans since those will be sent
    // back to us for reclaiming via callbackQueue.
    ncclIntruQueueConstruct(&comm->planQueue);
    cudaStream_t launchStream = tasks->streams->stream; // First user stream gets launch
    // Create dependency for deviceStream on launchStream. We know that deviceStream
    // hasn't been modified since launchStream waited on it (in ncclLaunchPrepare),
    // so we can say that launchStream subsumes it.
    NCCLCHECKGOTO(ncclStrongStreamWaitStream(tasks->capturingGraph, &comm->sharedRes->deviceStream, launchStream, /*b_subsumes_a=*/true), result, resume1);
  resume1:
    // Create dependency for other user streams (skip launch stream) on deviceStream.
    // Again, the user streams haven't been touched since deviceStream waited on them
    // so we can say they are subsumed by deviceStream.
    struct ncclCudaStreamList* sl = tasks->streams->next;
    tasks->streams = nullptr; // Reset comm->tasks.streams to empty.
    while (sl != nullptr) {
      NCCLCHECKGOTO(ncclStrongStreamWaitStream(tasks->capturingGraph, sl->stream, &comm->sharedRes->deviceStream, /*b_subsumes_a=*/true), result, resume2);
    resume2:
      sl = sl->next;
    }
    // Release device stream as acquired in ncclLaunchPrepare()
    NCCLCHECKGOTO(ncclStrongStreamRelease(tasks->capturingGraph, &comm->sharedRes->deviceStream), result, resume3);
  resume3:;
  }
  return result;
}

/*****************************************************************************/
/* Enqueueing system : computation of kernel and proxy operations parameters */
/*****************************************************************************/

static inline ncclResult_t getCollNetSupport(struct ncclInfo* info, int* collNetSupport) {
  // Translate ncclAvg and PreMulSum
  ncclRedOp_t netOp = info->op == ncclAvg || info->op >= ncclNumOps ? ncclSum : info->op;
  *collNetSupport = info->comm->collNetSupport;
  switch (info->coll) {
  case ncclFuncAllReduce:
  case ncclFuncReduce:
  case ncclFuncReduceScatter:
    *collNetSupport &= info->comm->collNetSupportMatrix[netOp][info->datatype];
    break;
  default:
    break;
  }
  return ncclSuccess;
}

// numPipeOps: number of pipelined ops. Can be greater than 1 in aggregation mode. Used to adjust latency.
static ncclResult_t topoGetAlgoInfo(struct ncclInfo* collInfo, int collNetSupport, int nvlsSupport, int numPipeOps) {
  struct ncclComm* comm = collInfo->comm;
  if (comm->nRanks == 1) {
    collInfo->algorithm = NCCL_ALGO_RING;
    collInfo->protocol = NCCL_PROTO_SIMPLE;
  }
  else if (collInfo->algorithm == NCCL_ALGO_UNDEF || collInfo->protocol == NCCL_PROTO_UNDEF) {
    float minTime = 3600000000.0; // Hopefully no operation will take an hour to complete.
    float backupMinTime = 3600000000.0;
    bool backup = false;
    int backupAlgo = NCCL_ALGO_UNDEF; // back up algo and proto if no algo/proto is picked up.
    int backupProto = NCCL_PROTO_UNDEF;
    // Find algorithm / protocol.
    collInfo->algorithm = -1;
    collInfo->protocol = -1;
    int nAlgos = NCCL_NUM_ALGORITHMS;
    for (int a=0; a<nAlgos; a++) {
      if ((a == NCCL_ALGO_COLLNET_DIRECT || a == NCCL_ALGO_COLLNET_CHAIN) && collNetSupport != 1) continue;
      if ((a == NCCL_ALGO_NVLS || a == NCCL_ALGO_NVLS_TREE) && nvlsSupport != 1) continue;
      if (a == NCCL_ALGO_NVLS && collNetSupport != 1 && comm->nNodes > 1) continue;
      /* now we only support single-node NVLS allgather and reducescatter */
      if (a == NCCL_ALGO_NVLS && (collInfo->coll == ncclFuncAllGather || collInfo->coll == ncclFuncReduceScatter) && comm->nNodes > 1) continue;

      for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
        float time;
        NCCLCHECK(ncclTopoGetAlgoTime(collInfo, a, p, numPipeOps, &time, &backup));
        if (!backup) {
          if (time >= 0 && time < minTime) {
            collInfo->algorithm = a;
            collInfo->protocol = p;
            minTime = time;
          }
        } else {
          if (time >= 0 && time < backupMinTime) {
            backupAlgo = a;
            backupProto = p;
            backupMinTime = time;
          }
        }
      }
    }

    if (collInfo->algorithm == NCCL_ALGO_UNDEF || collInfo->protocol == NCCL_PROTO_UNDEF) {
      if (backupAlgo == NCCL_ALGO_UNDEF || backupProto == NCCL_PROTO_UNDEF) {
        WARN("Error : no algorithm/protocol available");
        return ncclInternalError;
      }
      collInfo->algorithm = backupAlgo;
      collInfo->protocol = backupProto;
    }
    if (comm->rank == 0) INFO(NCCL_TUNING, "%ld Bytes -> Algo %d proto %d time %f", collInfo->nBytes, collInfo->algorithm, collInfo->protocol, minTime);
    TRACE(NCCL_COLL, "%ld Bytes -> Algo %d proto %d time %f", collInfo->nBytes, collInfo->algorithm, collInfo->protocol, minTime);
  }

  return ncclSuccess;
}

// Use the default topo-based tuner if tuner plugin is not successful.
// Call the plugin first. Let it set algo+proto, and/or nChannels.
// Then, topoGetAlgoInfo will set algo/proto if not set, then nChannels and nThreads based on algo/proto.
// Finally, nChannels will be overriden by the plugin setting.
static ncclResult_t getTunerInfo(struct ncclInfo* collInfo, int collNetSupport, int nvlsSupport, int numPipeOps) {
  collInfo->algorithm = NCCL_ALGO_UNDEF;
  collInfo->protocol = NCCL_PROTO_UNDEF;
  collInfo->nChannels = 0;
  if (collInfo->comm->tuner != NULL) {
    NCCLCHECK(collInfo->comm->tuner->getCollInfo(
          collInfo->coll, collInfo->nBytes,
          collNetSupport, nvlsSupport, numPipeOps,
          &collInfo->algorithm, &collInfo->protocol, &collInfo->nChannels));
  }

  /* We only honor nChannels decision when user sets the nChannels by tuner plugin or the coll picks
   * collnet algorithm. For other cases, we need to decide nChannels based on the maxBytesPerChannel */
  if (collInfo->nChannels != 0)
    collInfo->userTuned = true;
  else
    collInfo->userTuned = false;
  return ncclSuccess;
}

/* Compute nChannels and nThreads. */
static ncclResult_t getChannnelThreadInfo(struct ncclInfo* collInfo) {
  struct ncclComm *comm = collInfo->comm;
  int nc = comm->collChannels;
  int nt = comm->maxThreads[collInfo->algorithm][collInfo->protocol];
  int threadThreshold = comm->threadThresholds[collInfo->algorithm][collInfo->protocol];

  if (collInfo->nChannels == 0) {
    /* not preset by users */
    if (collInfo->algorithm == NCCL_ALGO_COLLNET_DIRECT) {
      // CollNet channel tuning
      int ncSwitch = 16;
      bool flag = true;
      while (ncSwitch >= 1 && flag) {
        while ((flag = collInfo->nBytes < nc * nt * collInfo->comm->channels[0].collnetDirect.nHeads * threadThreshold) && nc > ncSwitch) {
          if (nc == ncSwitch + ncSwitch / 2) threadThreshold /= 2;
          nc--;
        }
        ncSwitch /= 2;
      }
    } else if (collInfo->algorithm == NCCL_ALGO_NVLS || collInfo->algorithm == NCCL_ALGO_NVLS_TREE) {
      // NVLS should not need more than 16 channels to get peak BW.
      nc = comm->nvlsChannels;
    } else {
      // Ring/Tree channel tuning
      while (collInfo->nBytes < nc * nt * threadThreshold) {
        if (nc >= 2) nc--;
        else break;
      }
    }
    collInfo->nChannels = nc;
  } else {
    nc = collInfo->nChannels;
  }

  if (collInfo->nThreads == 0) {
    if (collInfo->algorithm != NCCL_ALGO_NVLS && collInfo->algorithm != NCCL_ALGO_NVLS_TREE &&
      collInfo->algorithm != NCCL_ALGO_COLLNET_DIRECT) {
      while (collInfo->nBytes < nc * nt * threadThreshold) {
        if (nt % 128 == 0) nt /= 2;
        else break;
      }
    }

    if (collInfo->protocol == NCCL_PROTO_SIMPLE) {
      if (collInfo->algorithm == NCCL_ALGO_RING) nt += WARP_SIZE; // Extra warp for sync
      // More threads or sync warps needed due to split thread model
      if (collInfo->algorithm == NCCL_ALGO_TREE) nt += 4*WARP_SIZE;
    }
    nt = nt / WARP_SIZE < 3 ? 3 * WARP_SIZE : nt;
    collInfo->nThreads = nt;
  }

  return ncclSuccess;
}

static ncclResult_t getPatternInfo(struct ncclInfo* collInfo) {
  switch (collInfo->coll) {
    case ncclFuncBroadcast:
      collInfo->pattern = collInfo->algorithm == NCCL_ALGO_TREE ? ncclPatternTreeDown : ncclPatternPipelineFrom; break;
    case ncclFuncReduce:
      collInfo->pattern = collInfo->algorithm == NCCL_ALGO_TREE ? ncclPatternTreeUp : ncclPatternPipelineTo; break;
    case ncclFuncReduceScatter:
    case ncclFuncAllGather:
      collInfo->pattern =
        collInfo->algorithm == NCCL_ALGO_NVLS ? ncclPatternNvls :
        collInfo->algorithm == NCCL_ALGO_COLLNET_DIRECT ? ncclPatternCollnetDirect :
        ncclPatternRing; break;
    case ncclFuncAllReduce:
      collInfo->pattern =
        collInfo->algorithm == NCCL_ALGO_NVLS ? ncclPatternNvls :
        collInfo->algorithm == NCCL_ALGO_NVLS_TREE ? ncclPatternNvlsTree :
        collInfo->algorithm == NCCL_ALGO_COLLNET_DIRECT ? ncclPatternCollnetDirect :
        collInfo->algorithm == NCCL_ALGO_COLLNET_CHAIN ? ncclPatternCollnetChain :
        collInfo->algorithm == NCCL_ALGO_TREE ? ncclPatternTreeUpDown :
        ncclPatternRingTwice; break;
    default:
      WARN("Unknown pattern for collective %d algorithm %d", collInfo->coll, collInfo->algorithm);
      return ncclInternalError;
  }
  return ncclSuccess;
}

static ncclResult_t computeCollWorkFunc(struct ncclInfo* collInfo) {
  collInfo->workFuncIndex = ncclDevFuncId(collInfo->coll, collInfo->opFull.op, collInfo->datatype, collInfo->algorithm, collInfo->protocol);
  return ncclSuccess;
}

static ncclResult_t initCollWorkElem(struct ncclInfo* collInfo, struct ncclWorkElem* work) {
  work->sendbuff = collInfo->sendbuff;
  work->recvbuff = collInfo->recvbuff;
  work->root = collInfo->root;
  work->count = collInfo->count;
  work->nWarps = collInfo->nThreads / WARP_SIZE;
  work->redOpArg = collInfo->opFull.scalarArg;
  work->redOpArgIsPtr = collInfo->opFull.scalarArgIsPtr;
  work->chunkCount = collInfo->chunkCount;
  work->regUsed = 0;
  work->isUsed = 1;

  if (collInfo->comm->nNodes == 1)
    work->oneNode = 1;
  else
    work->oneNode = 0;
  if (collInfo->algorithm == NCCL_ALGO_COLLNET_DIRECT) {
    // Set direct direction for broadcast-gather (read or write)
    work->direct = (collInfo->nBytes / collInfo->nChannels <= 1024 * 1024) ? NCCL_DIRECT_WRITE : NCCL_DIRECT_READ;
  } else {
    work->direct = 0;
  }
  return ncclSuccess;
}

static ncclResult_t setCollWorkElem(uint64_t workCount, uint64_t workOffset, size_t lastChunkCount, struct ncclWorkElem* work) {
  work->workCount = workCount;
  work->workOffset = workOffset;
  work->lastChunkCount = lastChunkCount;
  return ncclSuccess;
}

static ncclResult_t initCollWorkElemReg(struct ncclComm* comm, struct ncclWorkElem* work, struct ncclChannel* channel, ncclRegBufferType regBufType, void* regBufSend[], void* regBufRecv[], struct ncclWorkElemReg* workElemReg) {
  if (regBufType == NCCL_IPC_REG_BUFFER) {
    workElemReg->elem = *work;
    workElemReg->elem.regUsed = 1;
    for (int i = 0; i < NCCL_MAX_DIRECT_ARITY; i++) {
      int peer = channel->collnetDirect.down[i];
      if (peer == -1) break;
      int j = comm->rankToLocalRank[peer]; // Get intra-node slot
      workElemReg->dnInputs[i] = regBufSend[j]; // Input buffer of leaf peer
      workElemReg->dnOutputs[i] = regBufRecv[j]; // Output buffer of leaf peer
    }
    for (int i = 0; i < NCCL_MAX_DIRECT_ARITY; i++) {
      int peer = channel->collnetDirect.up[i];
      if (peer == -1) break;
      int j = comm->rankToLocalRank[peer];
      // Output buffer of root peer
      workElemReg->upOutputs[i] = regBufRecv[j];
    }
  } else if (regBufType == NCCL_NVLS_REG_BUFFER) {
    workElemReg->elem = *work;
    workElemReg->elem.regUsed = 1;
    /* NVLS only has one send and recv buffer registered */
    workElemReg->dnInputs[0] = regBufSend[0];
    workElemReg->dnOutputs[0] = regBufRecv[0];
  } else {
    /* impossible value */
    WARN("Invalid regBufType %d\n", regBufType);
    return ncclInvalidArgument;
  }
  return ncclSuccess;
}

NCCL_PARAM(NvlsTreeChunkSize, "NVLSTREE_MAX_CHUNKSIZE", -2);

static ncclResult_t computeCollChunkInfo(struct ncclInfo* collInfo, size_t nBytes, int nChannels) {
  int stepSize = collInfo->comm->buffSizes[collInfo->protocol] / NCCL_STEPS;
  int chunkSteps = (collInfo->protocol == NCCL_PROTO_SIMPLE && collInfo->algorithm == NCCL_ALGO_RING) ? collInfo->chunkSteps : 1;
  int sliceSteps = (collInfo->protocol == NCCL_PROTO_SIMPLE && collInfo->algorithm == NCCL_ALGO_RING) ? collInfo->sliceSteps : 1;
  int chunkSize = stepSize * chunkSteps;

  if (collInfo->protocol == NCCL_PROTO_LL) chunkSize /= 2;
  if (collInfo->protocol == NCCL_PROTO_LL128) chunkSize = (chunkSize / NCCL_LL128_LINEELEMS) * NCCL_LL128_DATAELEMS;

  if (collInfo->algorithm == NCCL_ALGO_COLLNET_DIRECT) {
    // Optimize chunkSize / nSteps
    while (nBytes / (nChannels * collInfo->comm->channels[0].collnetDirect.nHeads * chunkSize) < collInfo->comm->channels[0].collnetDirect.depth * 64 && chunkSize > 131072) chunkSize /= 2;
    while (nBytes / (nChannels * collInfo->comm->channels[0].collnetDirect.nHeads * chunkSize) < collInfo->comm->channels[0].collnetDirect.depth * 8 && chunkSize > 65536) chunkSize /= 2;
    while (nBytes / (nChannels * collInfo->comm->channels[0].collnetDirect.nHeads * chunkSize) < collInfo->comm->channels[0].collnetDirect.depth * 8 && chunkSize > 32768) chunkSize /= 2;
  } else if (collInfo->algorithm == NCCL_ALGO_COLLNET_CHAIN) {
    stepSize = collInfo->comm->buffSizes[NCCL_PROTO_SIMPLE] / NCCL_STEPS;
    chunkSize = std::min(256 * 1024, stepSize * chunkSteps);
    while (nBytes / (nChannels * chunkSize) < collInfo->comm->channels[0].collnetChain.depth * 64 && chunkSize > 131072) chunkSize /= 2;
    while (nBytes / (nChannels * chunkSize) < collInfo->comm->channels[0].collnetChain.depth * 8 && chunkSize > 65536) chunkSize /= 2;
    while (nBytes / (nChannels * chunkSize) < collInfo->comm->channels[0].collnetChain.depth && chunkSize > 32768) chunkSize /= 2;
  } else if (collInfo->algorithm == NCCL_ALGO_NVLS) {
    int maxChunkSize = 131072;
    if (collInfo->comm->nNodes > 1 && collInfo->comm->bandwidths[ncclFuncAllReduce][NCCL_ALGO_NVLS][NCCL_PROTO_SIMPLE] < 150) maxChunkSize = 32768;
    if (chunkSize > maxChunkSize) chunkSize = maxChunkSize;
    // Use uint64_t so that concurrentOps*chunkSize*X does not overflow
    uint64_t concurrentOps = nChannels * collInfo->comm->channels[0].nvls.nHeads;
    if ((nBytes < (64 * (concurrentOps * chunkSize))) && (chunkSize > 65536)) chunkSize = 65536;
    if ((nBytes < (8 * (concurrentOps * chunkSize))) && (chunkSize > 32768)) chunkSize = 32768;
    if ((nBytes < (2 * (concurrentOps * chunkSize))) && (chunkSize > 16384)) chunkSize = 16384;
  } else if (collInfo->algorithm == NCCL_ALGO_NVLS_TREE) {
    // Use uint64_t so that concurrentOps*chunkSize*X does not overflow
    uint64_t concurrentOps = nChannels * collInfo->comm->channels[0].nvls.nHeads;
    int maxChunkSize = ncclParamNvlsTreeChunkSize();
    if (maxChunkSize == -2) maxChunkSize = collInfo->comm->nNodes >= 4 ? 65536 : chunkSize;
    chunkSize = std::min(chunkSize, maxChunkSize);
    if ((nBytes < (32 * (concurrentOps * chunkSize))) && (chunkSize > 262144)) chunkSize = 262144;
    if ((nBytes < (16 * (concurrentOps * chunkSize))) && (chunkSize > 131072)) chunkSize = 131072;
    if ((nBytes < (4 * (concurrentOps * chunkSize))) && (chunkSize > 65536)) chunkSize = 65536;
    if ((nBytes < (1 * (concurrentOps * chunkSize))) && (chunkSize > 32768)) chunkSize = 32768;
  } else if (collInfo->algorithm == NCCL_ALGO_TREE && collInfo->protocol == NCCL_PROTO_LL128) {
    int nNodes = collInfo->comm->nNodes;
    float ppn = collInfo->comm->nRanks / (float)nNodes;
    float nstepsLL128 = 1+log2i(nNodes) + 0.1*ppn;
    while (nBytes / (nChannels*chunkSize) < nstepsLL128*64/ppn && chunkSize > 131072) chunkSize /= 2;
    while (nBytes / (nChannels*chunkSize) < nstepsLL128*16/ppn && chunkSize > 32768) chunkSize /= 2;
  }

  collInfo->chunkSize = chunkSize;
  collInfo->chunkCount = chunkSize / ncclTypeSize(collInfo->datatype);
  collInfo->chunkSteps = chunkSteps;
  collInfo->sliceSteps = sliceSteps;
  collInfo->stepSize = stepSize;
  return ncclSuccess;
}

static ncclResult_t initCollProxyOp(struct ncclInfo* collInfo, int channelId, uint64_t opCount, uint32_t nsteps, struct ncclProxyOp* proxyOp) {
  proxyOp->nsteps = nsteps;
  proxyOp->sliceSteps = collInfo->sliceSteps;
  proxyOp->chunkSteps = collInfo->chunkSteps;
  proxyOp->chunkSize = collInfo->chunkSize;
  proxyOp->protocol = collInfo->protocol;
  proxyOp->dtype = collInfo->datatype;
  // Network sees avg as sum
  proxyOp->redOp = collInfo->opFull.op == ncclDevPreMulSum || collInfo->opFull.op == ncclDevSumPostDiv ? ncclSum : collInfo->opFull.proxyOp;
  proxyOp->pattern = collInfo->pattern;
  proxyOp->coll = collInfo->coll;
  proxyOp->root = collInfo->root;
  proxyOp->reg = 0;
  // This is used by P2P to reduce the receive buffer size. We don't use it in collectives
  // because some protocols need to transmit more than the total size, plus they sometimes
  // round up
  proxyOp->nbytes = collInfo->stepSize * proxyOp->sliceSteps;
  proxyOp->channelId = channelId;
  proxyOp->opCount = opCount;

  if (collInfo->pattern == ncclPatternCollnetDirect) {
    proxyOp->specifics.collnetDirect.nNodes = collInfo->comm->nNodes;
    proxyOp->specifics.collnetDirect.node = collInfo->comm->node;
    if (collInfo->coll == ncclFuncAllGather || collInfo->coll == ncclFuncReduceScatter) {
      proxyOp->specifics.collnetDirect.sizePerRank = collInfo->count * ncclTypeSize(collInfo->datatype);
    }
  }
  return ncclSuccess;
}

static ncclResult_t hostToDevRedOp(
    ncclDevRedOpFull *opFull, ncclRedOp_t op, ncclDataType_t datatype, ncclComm *comm
  ) {
  union {
    int8_t   i8; uint8_t   u8;
    int32_t i32; uint32_t u32;
    int64_t i64; uint64_t u64;
    half f16; float f32; double f64;
    #if defined(__CUDA_BF16_TYPES_EXIST__)
      __nv_bfloat16 bf16;
    #endif
    void *ptr;
  };
  u64 = 0;
  opFull->scalarArgIsPtr = false;
  opFull->proxyOp = op;

  int nbits = 8*ncclTypeSize(datatype);
  uint64_t allBits = uint64_t(-1)>>(64-nbits);
  uint64_t signBit = allBits^(allBits>>1);

  switch (int(op)) {
  case ncclSum:  opFull->op = ncclDevSum;  break;
  case ncclProd: opFull->op = ncclDevProd; break;
  case ncclMin:
  case ncclMax:
    opFull->op = ncclDevMinMax;
    opFull->scalarArg = 0;
    // The xormask used by ncclFuncMinMax<[u]int> is the XOR of the sign bit
    // for signed (opposed to unsigned) types and all the bits for max (opposed to min).
    if (datatype==ncclInt8 || datatype==ncclInt32 || datatype==ncclInt64) {
      opFull->scalarArg ^= signBit;
    }
    opFull->scalarArg ^= (op == ncclMax) ? allBits : 0;
    break;
  case ncclAvg:
    switch ((int)datatype) {
    case ncclInt8:  case ncclInt32:  case ncclInt64:
    case ncclUint8: case ncclUint32: case ncclUint64:
      opFull->op = ncclDevSumPostDiv;
      u64 = comm->nRanks;
      break;
    case ncclFloat16:
      opFull->op = ncclDevPreMulSum;
      f16 = __float2half(float(1.0/comm->nRanks)); // __double2half not supported pre CUDA 11.x
      break;
    #if defined(__CUDA_BF16_TYPES_EXIST__)
    case ncclBfloat16:
      opFull->op = ncclDevPreMulSum;
      bf16 = __float2bfloat16(float(1.0/comm->nRanks));
      break;
    #endif
    case ncclFloat32:
      opFull->op = ncclDevPreMulSum;
      f32 = float(1.0/comm->nRanks);
      break;
    case ncclFloat64:
      opFull->op = ncclDevPreMulSum;
      f64 = 1.0/comm->nRanks;
      break;
    }
    opFull->scalarArgIsPtr = false;
    opFull->scalarArg = u64;
    break;
  default: // user created
    int ix = int(ncclUserRedOpMangle(comm, op)) - int(ncclNumOps);
    ncclUserRedOp *user = &comm->userRedOps[ix];
    if (datatype != user->datatype) {
      WARN("Data type supplied to user-created ncclRedOp_t does not match type "
           "given to reduction operation");
      return ncclInvalidArgument;
    }
    *opFull = user->opFull;
    break;
  }
  return ncclSuccess;
}

static int collCmp(struct ncclInfo *a, struct ncclInfo *b) {
  if (a->coll > b->coll)
    return 1;
  else if (a->coll == b->coll && a->datatype > b->datatype)
    return 1;
  else if (a->coll == b->coll && a->datatype == b->datatype && a->opFull.op > b->opFull.op)
    return 1;
  else if (a->coll == b->coll && a->datatype == b->datatype && a->opFull.op == b->opFull.op && a->count > b->count)
    return 1;
  else
    return -1;
}

// Converts `info` to a task and adds it to `comm->tasks`. The exception is with
// single rank communicators, collectives are issued as `ncclMemcpyAsync`s and
// thus don't need a task.
static ncclResult_t taskAppend(struct ncclComm* comm, struct ncclInfo* info) {
  ncclTasks *tasks = &comm->tasks;

  if (info->count == 0 && info->coll != ncclFuncSend && info->coll != ncclFuncRecv) return ncclSuccess;
  if (info->coll == ncclFuncSend || info->coll == ncclFuncRecv) {
    int peer = info->root;
    ssize_t nBytes = info->count*ncclTypeSize(info->datatype);
    bool isSendNotRecv = info->coll == ncclFuncSend;

    // Must be in thread local group before tasks can be alloc'd in `comm->memScoped`.
    ncclGroupCommJoin(info->comm);
    struct ncclTaskP2p* p2p = ncclMemoryStackAlloc<struct ncclTaskP2p>(&comm->memScoped);
    p2p->buff = (void*)info->recvbuff;
    p2p->bytes = nBytes;
    p2p->chunk = 0;
    ncclIntruQueueEnqueue(
      isSendNotRecv ? &tasks->peers[peer].sendQueue : &tasks->peers[peer].recvQueue,
      p2p);
    tasks->nTasksP2p += 1;

    // Mark channels that need pre-connect
    if (comm->rank != peer) {
      int channelBaseId;
      NCCLCHECK(ncclChannelComputeBase(comm, peer, info->coll, &channelBaseId));
      if (!(isSendNotRecv ? tasks->peers[peer].sendSeen : tasks->peers[peer].recvSeen)) {
        (isSendNotRecv ? tasks->peers[peer].sendSeen : tasks->peers[peer].recvSeen) = true;
        for (int c=0; c < comm->p2pnChannelsPerPeer; c++) {
          int channelId;
          NCCLCHECK(ncclChannelComputeFromBase(comm, channelBaseId, c, &channelId));
          if (isSendNotRecv) {
            if (comm->channels[channelId].peers[peer]->send[1].connected == 0) { // P2P uses only 1 connector
              comm->connectSend[peer] |= (1UL<<channelId);
              ncclGroupCommPreconnect(comm);
            }
          } else {
            if (comm->channels[channelId].peers[peer]->recv[1].connected == 0) { // P2P uses only 1 connector
              comm->connectRecv[peer] |= (1UL<<channelId);
              ncclGroupCommPreconnect(comm);
            }
          }
        }
      }
    }
  } else {
    // Copy reduction op state from op handle into info struct here since the
    // op handle may be destroyed before ncclGroupEnd().
    NCCLCHECK(hostToDevRedOp(&info->opFull, info->op, info->datatype, comm));

    if (comm->nRanks == 1) {
      NCCLCHECK(ncclLaunchOneRank(info->recvbuff, info->sendbuff, info->count, info->opFull, info->datatype, info->stream));
      return ncclSuccess;
    } else {
      // Must be in thread local group before tasks can be alloc'd in `comm->memScoped`.
      ncclGroupCommJoin(info->comm);
      struct ncclInfo* t = ncclMemoryStackAlloc<struct ncclInfo>(&comm->memScoped);
      info->nChannels = 0;
      info->nThreads = 0;
      info->algorithm = NCCL_ALGO_UNDEF;
      info->protocol = NCCL_PROTO_UNDEF;
      info->userTuned = false;
      memcpy(t, info, sizeof(struct ncclInfo));
      ncclIntruQueueSortEnqueue(&tasks->collQueue, t, collCmp);
      tasks->workBytesTotal += info->count * ncclTypeSize(info->datatype);
      tasks->nTasksColl += 1;
    }
  }

  if (info->stream != tasks->streamRecent || tasks->streams == nullptr) {
    tasks->streamRecent = info->stream;
    struct ncclCudaStreamList* l = tasks->streams;
    while (true) {
      if (l == nullptr) { // Got to the end, this must be a new stream.
        struct ncclCudaGraph graph;
        NCCLCHECK(ncclCudaGetCapturingGraph(&graph, info->stream))
        if (tasks->streams != nullptr && !ncclCudaGraphSame(tasks->capturingGraph, graph)) {
          WARN("Streams given to a communicator within a NCCL group must either be all uncaptured or all captured by the same graph.");
          return ncclInvalidUsage;
        }
        tasks->capturingGraph = graph; // C++ struct assignment
        // Add stream to list
        l = ncclMemoryStackAlloc<struct ncclCudaStreamList>(&comm->memScoped);
        l->stream = info->stream;
        l->next = tasks->streams;
        tasks->streams = l;
        break;
      }
      if (l->stream == info->stream)
        break; // Already seen stream.
      l = l->next;
    }
  }
  return ncclSuccess;
}

ncclResult_t ncclEnqueueCheck(struct ncclInfo* info) {
  NCCLCHECK(ncclGroupStartInternal());
  ncclResult_t ret = ncclSuccess;
  int devOld = -1;

  NCCLCHECKGOTO(PtrCheck(info->comm, info->opName, "comm"), ret, fail);
  // Check whether communicator is ready to communicate
  NCCLCHECKGOTO(ncclCommEnsureReady(info->comm), ret, fail);

  if (info->comm->checkPointers) {
    CUDACHECKGOTO(cudaGetDevice(&devOld), ret, fail);
    CUDACHECKGOTO(cudaSetDevice(info->comm->cudaDev), ret, fail);
  }
  NCCLCHECKGOTO(ArgsCheck(info), ret, fail);

  INFO(NCCL_COLL,"%s: opCount %lx sendbuff %p recvbuff %p count %zi datatype %d op %d root %d comm %p [nranks=%d] stream %p",
        info->opName, info->comm->opCount, info->sendbuff, info->recvbuff, info->count,
        info->datatype, info->op, info->root, info->comm, info->comm->nRanks, info->stream);
  TRACE_CALL("nccl%s(%" PRIx64 ",%" PRIx64 ",%zi,%d,%d,%d,%p,%p)", info->opName, reinterpret_cast<int64_t>(info->sendbuff), reinterpret_cast<int64_t>(info->recvbuff), info->count, info->datatype, info->op, info->root, info->comm, info->stream);

  NCCLCHECKGOTO(taskAppend(info->comm, info), ret, fail);

exit:
  if (devOld != -1) CUDACHECK(cudaSetDevice(devOld));
  ncclGroupErrCheck(ret);
  NCCLCHECK(ncclGroupEndInternal());
  /* if depth is 1, ncclGroupEndInternal() will trigger group ops. The state can change
   * so we have to check state here. */
  if (info->comm && !info->comm->config.blocking) { NCCLCHECK(ncclCommGetAsyncError(info->comm, &ret)) };
  return ret;
fail:
  if (info->comm && !info->comm->config.blocking) (void) ncclCommSetAsyncError(info->comm, ret);
  goto exit;
}

NCCL_API(ncclResult_t, ncclRedOpCreatePreMulSum, ncclRedOp_t *op, void *scalar, ncclDataType_t datatype, ncclScalarResidence_t residence, ncclComm_t comm);
ncclResult_t ncclRedOpCreatePreMulSum(ncclRedOp_t *op, void *scalar, ncclDataType_t datatype, ncclScalarResidence_t residence, ncclComm_t comm) {
  NCCLCHECK(PtrCheck(comm, "ncclRedOpCreatePreMulSum", "comm"));
  /* join init thread before creating PreMulSum op. */
  NCCLCHECK(ncclCommEnsureReady(comm));

  if (comm->userRedOpFreeHead == comm->userRedOpCapacity) {
    // double capacity and resize
    int cap = 2*comm->userRedOpCapacity;
    if (cap < 4) cap = 4;
    ncclUserRedOp *ops = new ncclUserRedOp[cap];
    std::memcpy(ops, comm->userRedOps, comm->userRedOpCapacity*sizeof(ncclUserRedOp));
    for(int ix=comm->userRedOpCapacity; ix < cap; ix++)
      ops[ix].freeNext = ix + 1;
    delete[] comm->userRedOps;
    comm->userRedOps = ops;
    comm->userRedOpCapacity = cap;
  }
  // pop from free list
  int ix = comm->userRedOpFreeHead;
  ncclUserRedOp *user = &comm->userRedOps[ix];
  comm->userRedOpFreeHead = user->freeNext;

  user->freeNext = -1; // allocated
  user->datatype = datatype;
  user->opFull.op = ncclDevPreMulSum;
  if (residence == ncclScalarHostImmediate) {
    user->opFull.scalarArgIsPtr = false;
    std::memcpy(&user->opFull.scalarArg, scalar, ncclTypeSize(datatype));
  } else {
    user->opFull.scalarArgIsPtr = true;
    user->opFull.scalarArg = reinterpret_cast<uint64_t>(scalar);
  }
  *op = ncclRedOp_t(int(ncclNumOps) + ix);
  *op = ncclUserRedOpMangle(comm, *op);
  TRACE_CALL("ncclRedOpCreatePreMulSum(%d,%p,%d,%d,%p)", *op, scalar, datatype, residence, comm);
  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclRedOpDestroy, ncclRedOp_t op, ncclComm_t comm);
ncclResult_t ncclRedOpDestroy(ncclRedOp_t op, ncclComm_t comm) {
  if (0 <= int(op) && int(op) < int(ncclNumOps)) {
    WARN("ncclRedOpDestroy : operator is a NCCL builtin.");
    return ncclInvalidArgument;
  }
  if (int(op) < 0 || int(ncclMaxRedOp) < int(op)) {
    WARN("ncclRedOpDestroy :  operator is garbage.");
    return ncclInvalidArgument;
  }
  if (comm == NULL) {
    WARN("ncclRedOpDestroy : invalid communicator passed.");
    return ncclInvalidArgument;
  }

  int ix = int(ncclUserRedOpMangle(comm, op)) - int(ncclNumOps);
  if (comm->userRedOpCapacity <= ix || comm->userRedOps[ix].freeNext != -1) {
    WARN("ncclRedOpDestroy : operator unknown to this communicator.");
    return ncclInvalidArgument;
  }
  // push to free list
  comm->userRedOps[ix].freeNext = comm->userRedOpFreeHead;
  comm->userRedOpFreeHead = ix;
  TRACE_CALL("ncclRedOpDestroy(%d,%p)", op, comm);
  return ncclSuccess;
}
