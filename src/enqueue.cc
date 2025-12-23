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
#include "profiler.h"
#include "transport.h"
#include "register_inline.h"
#include "ce_coll.h"
#include "nvtx.h"
#include "scheduler.h"
#include "compiler.h"
#include "rma/rma.h"

#include <cstring> // std::memcpy
#include <cinttypes> // PRIx64
#include <cassert>
#include <cfloat> // FLT_MAX

NCCL_PARAM(L1SharedMemoryCarveout, "L1_SHARED_MEMORY_CARVEOUT", 0);
NCCL_PARAM(AllgathervEnable, "ALLGATHERV_ENABLE", 1);
NCCL_PARAM(SymCeThreshold, "SYM_CE_THRESHOLD", 8*1024*1024);

// Returns maximum kernel stack size of all CUDA kernels
ncclResult_t ncclInitKernelsForDevice(int cudaArch, int maxSharedMem, size_t* maxStackSize) {
  ncclResult_t result = ncclSuccess;

  if (maxStackSize) *maxStackSize = 0;
  int carveout = ncclParamL1SharedMemoryCarveout();
  int ncclMaxSharedMem = ncclShmemDynamicSize(cudaArch);

  int driverVersion;
  NCCLCHECK(ncclCudaDriverVersion(&driverVersion));

  for (int sym=0; sym <= 1; sym++) {
    int kcount = sym==0 ? ncclDevKernelCount : ncclSymkKernelCount;
    void** kptrs = sym==0 ? ncclDevKernelList : ncclSymkKernelList;
    int* krequires = sym==0 ? ncclDevKernelRequirements : ncclSymkKernelRequirements;
    for (int k=0; k < kcount; k++) {
      if (kptrs[k] != nullptr && driverVersion < krequires[k]) {
        INFO(NCCL_INIT, "Skipping %skernel %d which requires driver %d",
             sym ? "symmetric " : "", k, krequires[k]);
        kptrs[k] = nullptr;
      }
      void* fn = kptrs[k];
      cudaFuncAttributes attr = {0};
      if (fn == nullptr) continue;

      if (!CUDASUCCESS(cudaFuncGetAttributes(&attr, fn))) continue; // Silently ignore failures

      if (maxStackSize) {
        if (attr.localSizeBytes > *maxStackSize) *maxStackSize = attr.localSizeBytes;
      }
      if (carveout) {
        CUDACHECKGOTO(cudaFuncSetAttribute(fn,
          cudaFuncAttributePreferredSharedMemoryCarveout, carveout),
          result, ignore1);
      ignore1:;
      }
      if (ncclMaxSharedMem != 0) {
        int sharedMemSize = ncclMaxSharedMem;
        if (sharedMemSize > (maxSharedMem-attr.sharedSizeBytes)) {
          WARN("cudaArch %d ncclMaxSharedMem %d exceeds device/fn maxSharedMem %zu",
               cudaArch, sharedMemSize, maxSharedMem-attr.sharedSizeBytes);
          return ncclSystemError;
        }
        CUDACHECKGOTO(cudaFuncSetAttribute(fn,
          cudaFuncAttributeMaxDynamicSharedMemorySize, sharedMemSize),
          result, next_kernel);
      }
    next_kernel:;
    }
  }
  return result;
}

////////////////////////////////////////////////////////////////////////////////
// Data movement metrics.

static inline int ncclFuncTrafficPerByte(ncclFunc_t func, int nRanks) {
  switch (func) {
  case ncclFuncAllReduce: return 2;
  case ncclFuncAllGather: return nRanks;
  case ncclFuncReduceScatter: return nRanks;
  default: return 1;
  }
}

/*****************************************************************************/
/*       Launch system : synchronization and CUDA kernel launch              */
/*****************************************************************************/

ncclResult_t ncclAddProxyOpIfNeeded(struct ncclComm* comm, struct ncclKernelPlan* plan, struct ncclProxyOp* op) {
  bool needed = true;
  NCCLCHECK(ncclProxySaveOp(comm, op, &needed));
  if (needed) {
    struct ncclProxyOp* q = ncclMemoryPoolAlloc<struct ncclProxyOp>(&comm->memPool_ncclProxyOp, &comm->memPermanent);
    *q = *op; // C++ struct assignment
    ncclIntruQueueEnqueue(&comm->planner.wipPlan.channels[op->channelId].proxyOpQueue, q);
  }
  return ncclSuccess;
}

NCCL_PARAM(P2pEpochEnable, "P2P_EPOCH_ENABLE", 1);

void ncclAddWorkBatchToPlan(
    struct ncclComm* comm, struct ncclKernelPlan* plan, int channelId,
    enum ncclDevWorkType workType, int devFuncId, uint32_t workOffset,
    int p2pEpoch, int p2pRound, bool newBatch
  ) {
  size_t workSize = ncclDevWorkSize(workType);
  ncclKernelPlanner::WipPlan::Channel* chan = &comm->planner.wipPlan.channels[channelId];
  // Conditions causing us to create a new blank batch.
  newBatch = (chan->workBatchQueue.tail == nullptr);
  struct ncclDevWorkBatch* batch = nullptr;
  if (!newBatch) {
    batch = &chan->workBatchQueue.tail->batch;
    // All of the conditions that prevent us from appending to current batch.
    newBatch |= batch->workType != (uint8_t)workType;
    newBatch |= batch->funcId != devFuncId;
    // The following ensure the device can handle a batch this large. They have to
    // account for all extension batches being fused together which is why
    // wipBatch.workBytes and wipBatch.nP2ps aren't reset to 0 for a new extension
    // batch further down.
    if (workType == ncclDevWorkTypeP2p) {
      if (ncclParamP2pEpochEnable()) newBatch |= chan->wipBatch.p2pEpoch != p2pEpoch;
      // We only allow NCCL_MAX_DEV_WORK_P2P_PER_BATCH ops per batch.
      newBatch |= chan->wipBatch.nP2ps == NCCL_MAX_DEV_WORK_P2P_PER_BATCH;
      for (int i = 0; i < chan->wipBatch.nP2ps; i++) {
        // Do not allow the same round twice in the same batch, it would use the same connection.
        newBatch |= p2pRound == chan->wipBatch.p2pRounds[i];
        // Make sure we only aggregate p2p operations within the same p2p group (one group is NCCL_MAX_DEV_WORK_P2P_PER_BATCH ops).
        // This enforces uniform batching accross ranks in the communicator and prevents hangs.
        newBatch |= (p2pRound / NCCL_MAX_DEV_WORK_P2P_PER_BATCH) != (chan->wipBatch.p2pRounds[i] / NCCL_MAX_DEV_WORK_P2P_PER_BATCH);
      }
    }
    if (workType == ncclDevWorkTypeBcast) {
      int maxitem = ncclMaxDevWorkBatchBytes(comm->cudaArch) / sizeof(ncclDevWorkBcast);
      newBatch |= chan->wipBatch.nBcasts == maxitem;
    } else {
      newBatch |= NCCL_MAX_DEV_WORK_BATCH_BYTES < chan->wipBatch.workBytes + workSize;
    }
  }
  // Conditions causing us to create an extension batch (prev->nextExtends=1)
  uint32_t offset = newBatch ? 0 : (workOffset - batch->offsetBase);
  bool extendBatch = 63*workSize < offset;
  extendBatch |= 0 != offset%workSize;
  if (newBatch || extendBatch) {
    if (!newBatch) batch->nextExtends = extendBatch; // Extending the previous batch.
    struct ncclWorkBatchList* batchNode = ncclMemoryStackAlloc<ncclWorkBatchList>(&comm->memScoped);
    // Coverity thinks that ncclIntruQueueEnqueue will access chan->workBatchQueue->tail, which might
    // be NULL.  But that code is guarded by chan->workBatchQueue->head not being NULL, in which
    // case tail won't be NULL either.
    // coverity[var_deref_model:FALSE]
    ncclIntruQueueEnqueue(&chan->workBatchQueue, batchNode);
    batch = &batchNode->batch;
    batch->nextExtends = 0;
    batch->workType = (uint32_t)workType;
    batch->funcId = devFuncId;
    batch->offsetBase = workOffset;
    batch->offsetBitset = 0;
    offset = 0;
    if (newBatch) {
      // Since extension batches are fused together on the device, and these values
      // account for constraints on the fused batch, we only reset the values on
      // a new batch
      chan->wipBatch.workBytes = 0;
      chan->wipBatch.nP2ps = 0;
      chan->wipBatch.nBcasts = 0;
      // We don't count extension batches since this is used to derive a proxyOpCount,
      // and we wan't all ops which are fused together to have the same value.
      chan->nWorkBatchesP2p += (workType == ncclDevWorkTypeP2p ? 1 : 0);
      chan->nWorkBatchesBcast += (workType == ncclDevWorkTypeBcast ? 1 : 0);
    }
    plan->nWorkBatches += 1;
  }
  batch->offsetBitset |= 1ull<<(offset/workSize);
  chan->wipBatch.workBytes += workSize;
  if (workType == ncclDevWorkTypeP2p) {
    chan->wipBatch.p2pEpoch = p2pEpoch;
    chan->wipBatch.p2pRounds[chan->wipBatch.nP2ps++] = p2pRound;
  }
  if (workType == ncclDevWorkTypeBcast) {
    chan->wipBatch.nBcasts += 1;
  }
}

static void finishPlan(struct ncclComm* comm, struct ncclKernelPlan* plan) {
  ncclKernelPlanner::WipPlan::Channel* wipChannels = comm->planner.wipPlan.channels;
  size_t workBytes = plan->workBytes;
  size_t batchBytes = plan->nWorkBatches*sizeof(struct ncclDevWorkBatch);

  if (plan->isSymColl) return;
  plan->threadPerBlock = std::max(plan->threadPerBlock, NCCL_MIN_NTHREADS);

  // If we can fit everything into the kernel args we do so.
  if (sizeof(ncclDevKernelArgs) + batchBytes + workBytes <= comm->workArgsBytes) {
    plan->workStorageType = ncclDevWorkStorageTypeArgs;
  }
  plan->kernelArgsSize = sizeof(struct ncclDevKernelArgs) + batchBytes;
  plan->kernelArgsSize += (plan->workStorageType == ncclDevWorkStorageTypeArgs) ? workBytes : 0;
  plan->kernelArgsSize = alignUp(plan->kernelArgsSize, 16);
  plan->kernelArgs = (struct ncclDevKernelArgs*)ncclMemoryStackAlloc(&comm->memScoped, plan->kernelArgsSize, /*align=*/16);
  plan->kernelArgs->comm = comm->devComm;
  plan->kernelArgs->channelMask = plan->channelMask;
  plan->kernelArgs->workStorageType = plan->workStorageType;

  // Put batches into the kernel arguments. The first batch for each channel
  // must be located at batchZero[blockIdx.x]. To achieve this we round robin
  // over the channels in ascending order until they're exhausted.
  uint64_t hasBatchMask = plan->channelMask;
  struct ncclDevWorkBatch* batchPrev[MAXCHANNELS] = {}; // {0...}
  struct ncclDevWorkBatch* batchZero = (struct ncclDevWorkBatch*)(plan->kernelArgs+1);
  int batchIx = 0;
  while (hasBatchMask != 0) {
    uint64_t tmpMask = hasBatchMask; // channels with a batch for this round.
    do {
      int c = popFirstOneBit(&tmpMask);
      if (!ncclIntruQueueEmpty(&wipChannels[c].workBatchQueue)) {
        struct ncclWorkBatchList* batchNode = ncclIntruQueueDequeue(&wipChannels[c].workBatchQueue);
        if (batchPrev[c] != nullptr) {
          batchPrev[c]->nextJump = int(&batchZero[batchIx] - batchPrev[c]);
        }
        batchPrev[c] = &batchZero[batchIx];
        batchZero[batchIx++] = batchNode->batch;
      }
      if (ncclIntruQueueEmpty(&wipChannels[c].workBatchQueue)) {
        hasBatchMask ^= 1ull<<c;
      }
    } while (tmpMask != 0);
  }

  // Merge-sort per-channel proxy-op lists by opCount when merging them into plan->proxyOpQueue
  // Phase 1: scan first op of each channel, store opCount in headIds[c].
  uint64_t headIds[MAXCHANNELS];
  int nHeads = 0;
  int channelUbound = 0;
  for (int c=0; c < MAXCHANNELS; c++) {
    struct ncclProxyOp* op = ncclIntruQueueHead(&wipChannels[c].proxyOpQueue);
    headIds[c] = op ? op->opCount : uint64_t(-1);
    if (op) nHeads += 1;
    if (op) plan->hasProxyOps = true;
    if (op) channelUbound = c+1;
  }
  // Phase 2: Dequeue from planner->channels[c], enqueue in merged order to plan
  while (nHeads != 0) {
    int c = -1;
    uint64_t minId = uint64_t(-1);
    // Find channel with least proxy-op id. We store the heads[c]->opCount in
    // headIds[c] to remove indirect loads from this loop.
    for (int c1=0; c1 < channelUbound; c1++) {
      uint64_t id = headIds[c1];
      id = (id>>1 | id<<63); // Move tag bit to order collectives before p2p's
      if (id < minId) { c = c1; minId = id; }
    }
    struct ncclProxyOp* op = ncclIntruQueueDequeue(&wipChannels[c].proxyOpQueue);
    struct ncclProxyOp* opNext = ncclIntruQueueHead(&wipChannels[c].proxyOpQueue);
    headIds[c] = opNext ? opNext->opCount : uint64_t(-1);
    nHeads -= opNext ? 0 : 1;
    ncclIntruQueueEnqueue(&plan->proxyOpQueue, op);
  }
}

NCCL_PARAM(GraphRegister, "GRAPH_REGISTER", 1);

static ncclResult_t calcCollChunking(
  struct ncclComm* comm, struct ncclTaskColl* task, int nChannels, size_t nBytes,
  /*outputs*/uint32_t* outChunkSize, uint32_t* outDirectFlags, struct ncclProxyOp* proxyOp
);

struct ncclKernelPlanBudget {
  ssize_t inArgsBytes; // Space available within kernel args struct
  ssize_t outArgsBytes; // Space available outside of args struct (fifo or persistent buf)
};

bool ncclTestBudget(
    struct ncclKernelPlanBudget* budget, int nWorkBatches, ssize_t workBytes
  ) {
  ssize_t batchBytes = nWorkBatches*sizeof(struct ncclDevWorkBatch);
  bool ok = false;
  ok |= (batchBytes + workBytes <= budget->inArgsBytes);
  ok |= (batchBytes <= budget->inArgsBytes) && (workBytes <= budget->outArgsBytes);
  return ok;
}

ncclResult_t ncclTasksRegAndEnqueue(struct ncclComm* comm) {
  struct ncclKernelPlanner* planner = &comm->planner;
  struct ncclTaskColl *task;
  task = ncclIntruQueueHead(&planner->collTaskQueue);
  while (task != nullptr) {
    // Build a ncclDevWorkColl[Reg?] struct for each task.
    void* regBufSend[NCCL_MAX_LOCAL_RANKS];
    void* regBufRecv[NCCL_MAX_LOCAL_RANKS];
    bool regNeedConnect = true;
    struct ncclWorkList* workNode = NULL;
    struct ncclDevWorkColl devWork = {};

    if (task->algorithm == NCCL_ALGO_NVLS_TREE || task->algorithm == NCCL_ALGO_NVLS) {
      workNode = ncclIntruQueueDequeue(&planner->tmpCollWorkQueue);
      goto next;
    }
    ncclRegisterCollBuffers(comm, task, regBufSend, regBufRecv, &planner->collCleanupQueue, &regNeedConnect);

    devWork.sendbuff = (void*)task->sendbuff;
    devWork.recvbuff = (void*)task->recvbuff;
    devWork.sendbuffOffset = task->sendbuffOffset;
    devWork.recvbuffOffset = task->recvbuffOffset;
    devWork.sendbuffRmtAddrs = task->sendbuffRmtAddrs;
    devWork.recvbuffRmtAddrs = task->recvbuffRmtAddrs;
    devWork.root = task->root;
    devWork.nWarps = task->nWarps;
    devWork.redOpArg = task->opDev.scalarArg;
    devWork.redOpArgIsPtr = task->opDev.scalarArgIsPtr;
    devWork.oneNode = (comm->nNodes == 1);
    devWork.isOneRPN = comm->isOneRPN;
    devWork.netRegUsed = devWork.regUsed = 0;
    devWork.profilerEnabled = ncclProfilerPluginLoaded() && (task->eActivationMask & ncclProfileKernelCh);
    if (task->regBufType & NCCL_NET_REG_BUFFER)
      devWork.netRegUsed = 1;
    if (task->regBufType & (NCCL_IPC_REG_BUFFER | NCCL_NVLS_REG_BUFFER))
      devWork.regUsed = 1;

    if (task->regBufType & NCCL_NVLS_REG_BUFFER) {
      struct ncclDevWorkCollReg workReg = {};
      workReg.coll = devWork; // C++ struct assignment
      /* NVLS only has one send and recv buffer registered */
      workReg.dnInputs[0] = regBufSend[0];
      workReg.dnOutputs[0] = regBufRecv[0];
      workNode = ncclMemoryStackAllocInlineArray<ncclWorkList, ncclDevWorkCollReg>(&comm->memScoped, 1);
      workNode->workType = ncclDevWorkTypeCollReg;
      workNode->size = sizeof(struct ncclDevWorkCollReg);
      memcpy((void*)(workNode+1), (void*)&workReg, workNode->size);
    } else {
      workNode = ncclMemoryStackAllocInlineArray<ncclWorkList, ncclDevWorkColl>(&comm->memScoped, 1);
      workNode->workType = ncclDevWorkTypeColl;
      workNode->size = sizeof(struct ncclDevWorkColl);
      memcpy((void*)(workNode+1), (void*)&devWork, workNode->size);
    }
next:
    ncclIntruQueueEnqueue(&planner->collWorkQueue, workNode);
    task = task->next;
  }
  assert(ncclIntruQueueEmpty(&planner->tmpCollWorkQueue));
  return ncclSuccess;
}

// Called once per ncclGroup to organize the user submitted tasks in
// comm->planner so that they can be peeled off into plans.
ncclResult_t ncclPrepareTasks(struct ncclComm* comm, bool* algoNeedConnect, bool* needConnect, ncclSimInfo_t* simInfo) {
  struct ncclKernelPlanner* planner = &comm->planner;
  planner->persistent = ncclCudaGraphValid(planner->capturingGraph);

  // Put bcast tasks into collSorter if there's only one bcast peer
  if (planner->bcast_info.BcastPeers == 1) {
    while (!ncclIntruQueueEmpty(&planner->peers[planner->bcast_info.minBcastPeer].bcastQueue)) {
      struct ncclTaskBcast* bcastTask = ncclIntruQueueDequeue(&planner->peers[planner->bcast_info.minBcastPeer].bcastQueue);
      struct ncclTaskColl *t = ncclMemoryPoolAlloc<struct ncclTaskColl>(&comm->memPool_ncclTaskColl, &comm->memPermanent);
      t->func = ncclFuncBroadcast;
      t->sendbuff = bcastTask->sendbuff;
      t->recvbuff = bcastTask->recvbuff;
      t->count = bcastTask->count;
      t->root = bcastTask->root;
      t->datatype = bcastTask->datatype;
      t->trafficBytes = t->count*ncclFuncTrafficPerByte(t->func, comm->nRanks);
      t->chunkSteps = BROADCAST_CHUNKSTEPS;
      t->sliceSteps = BROADCAST_SLICESTEPS;
      ncclTaskCollSorterInsert(&planner->collSorter, t, t->trafficBytes);
      planner->nTasksColl += 1;
      ncclMemoryPoolFree(&comm->memPool_ncclTaskBcast, bcastTask);
    }
    // reset bcast info
    planner->nTasksBcast = 0;
    planner->bcast_info.BcastPeers = 0;
  }

  // Tasks from the sorter come out ordered size descending.
  struct ncclTaskColl* task = ncclTaskCollSorterDequeueAll(&planner->collSorter);
  // Tasks are assembled by (fn,op,ty) size ascending.
  struct ncclTaskColl* tasksByFnOpTy[ncclNumFuncs*ncclNumDevRedOps*ncclNumTypes];
  memset(tasksByFnOpTy, 0, sizeof(tasksByFnOpTy));
  int fnOpTyIndices[ncclNumFuncs*ncclNumDevRedOps*ncclNumTypes];
  int fnOpTyCount = 0;

  if (comm->symmetricSupport) {
    NCCLCHECK(ncclMakeSymmetricTaskList(comm, task, &planner->collSymTaskQueue, &task));
  }

  // Walk the size sorted tasks, binning them by (fn,op,ty).
  while (task != nullptr) {
    struct ncclTaskColl* next = task->next;
    int index = ((int)task->func*ncclNumDevRedOps + (int)task->opDev.op)*ncclNumTypes + (int)task->datatype;
    // Add to set of (fn,op,ty) indices on first occurrence
    if (tasksByFnOpTy[index] == nullptr) fnOpTyIndices[fnOpTyCount++] = index;
    // Add to LIFO for this (fn,op,ty)
    task->next = tasksByFnOpTy[index];
    tasksByFnOpTy[index] = task;
    // Next task
    task = next;
  }

  // Walk (fn,op,ty) bins, compute algo and proto etc. Then bin them by their
  // scheduling constraints (collnet x nvls).
  struct ncclIntruQueue<struct ncclTaskColl, &ncclTaskColl::next> collBins[2][2] = {};
  for (int cursor=0; cursor < fnOpTyCount; cursor++) {
    struct ncclTaskColl* aggBeg = tasksByFnOpTy[fnOpTyIndices[cursor]];
    int collNetSupport = 0;
    NCCLCHECK(ncclGetCollNetSupport(comm, aggBeg, &collNetSupport));
    int nvlsSupport = comm->nvlsSupport && (ncclNvlsSupported(aggBeg->opDev.op, aggBeg->datatype) || aggBeg->func == ncclFuncAllGather);
    // Crudely estimate number of tasks per channel. This is using the wrong number
    // of channels for NVLS algos, but knowing the algo requires having this value,
    // so either be crude our iterate until fixed point, we chose the former.
    int nTasksPerChannel = divUp(comm->planner.nTasksColl, comm->nChannels);
    do {
      struct ncclTaskColl* aggEnd = aggBeg->next;
      struct ncclTaskColl agg = *aggBeg;
      // We aggregate operations that are within 4X size of each other.
      while (aggEnd != nullptr && aggEnd->trafficBytes < 4*aggBeg->trafficBytes) {
        agg.count += aggEnd->count;
        agg.trafficBytes += aggEnd->trafficBytes;
        aggEnd = aggEnd->next;
      }

      NCCLCHECK(ncclGetAlgoInfo(comm, &agg, collNetSupport, nvlsSupport, nTasksPerChannel, simInfo));
      agg.devFuncId = ncclDevFuncId(agg.func, agg.opDev.op, agg.datatype, agg.algorithm, agg.protocol);

      int isCollnet=0, isNvls=0;
      switch (agg.algorithm) {
      case NCCL_ALGO_NVLS:
      case NCCL_ALGO_NVLS_TREE:
        isNvls = 1;
        isCollnet = agg.algorithm == NCCL_ALGO_NVLS && comm->nNodes > 1;
        break;
      case NCCL_ALGO_COLLNET_CHAIN:
      case NCCL_ALGO_COLLNET_DIRECT:
        isCollnet = 1;
        break;
      }
      // Update the aggregated tasks with the computed values.
      do {
        struct ncclTaskColl* next = aggBeg->next;
        aggBeg->algorithm = agg.algorithm;
        aggBeg->protocol = agg.protocol;
        if (aggBeg->protocol == NCCL_PROTO_LL) aggBeg->trafficBytes *= 4;
        aggBeg->nMaxChannels = agg.nMaxChannels;
        aggBeg->nWarps = agg.nWarps;
        aggBeg->devFuncId = agg.devFuncId;
        aggBeg->isCollnet = isCollnet;
        aggBeg->isNvls = isNvls;
        ncclIntruQueueEnqueue(&collBins[isCollnet][isNvls], aggBeg);
        aggBeg = next;
      } while (aggBeg != aggEnd);
    } while (aggBeg != nullptr);
  }

  // Concatenate `collBins[*][*]` together into final list `planner->collTaskQueue`.
  // Collnet is the outer dimension since that affects how we divide over the
  // channels.
  for (int isCollnet=0; isCollnet <= 1; isCollnet++) {
    for (int isNvls=0; isNvls <= 1; isNvls++) {
      ncclIntruQueueTransfer(&planner->collTaskQueue, &collBins[isCollnet][isNvls]);
    }
  }

  // Walk tasks again to:
  // 1. Possibly register buffers.
  // 2. Build ncclDevWorkColl structs.
  // 3. Bin the work structs according to the number of valid channels they
  //    may be assigned to {collnet, nvls, standard}
  task = ncclIntruQueueHead(&planner->collTaskQueue);
  while (task != nullptr) {
    // Build a ncclDevWorkColl[Reg?] struct for each task.
    void* regBufSend[NCCL_MAX_LOCAL_RANKS];
    void* regBufRecv[NCCL_MAX_LOCAL_RANKS];
    bool regNeedConnect = true;
    ncclRegisterCollNvlsBuffers(comm, task, regBufSend, regBufRecv, &planner->collCleanupQueue, &regNeedConnect);

    if (comm->runtimeConn && comm->initAlgoChannels[task->algorithm] == false) {
      if (task->algorithm == NCCL_ALGO_NVLS_TREE && comm->initAlgoChannels[NCCL_ALGO_NVLS] == false && regNeedConnect == true) {
        comm->initAlgoChannels[NCCL_ALGO_NVLS] = true;
        algoNeedConnect[NCCL_ALGO_NVLS] = true;
      }
      if (task->algorithm != NCCL_ALGO_NVLS || regNeedConnect == true) {
        comm->initAlgoChannels[task->algorithm] = true;
        algoNeedConnect[task->algorithm] = true;
        *needConnect = true;
      }
    }

    if (task->algorithm == NCCL_ALGO_NVLS_TREE || task->algorithm == NCCL_ALGO_NVLS) {
      struct ncclDevWorkColl devWork = {};
      devWork.sendbuff = (void*)task->sendbuff;
      devWork.recvbuff = (void*)task->recvbuff;
      devWork.sendbuffOffset = task->sendbuffOffset;
      devWork.recvbuffOffset = task->recvbuffOffset;
      devWork.sendbuffRmtAddrs = task->sendbuffRmtAddrs;
      devWork.recvbuffRmtAddrs = task->recvbuffRmtAddrs;
      devWork.root = task->root;
      devWork.nWarps = task->nWarps;
      devWork.redOpArg = task->opDev.scalarArg;
      devWork.redOpArgIsPtr = task->opDev.scalarArgIsPtr;
      devWork.oneNode = (comm->nNodes == 1);
      devWork.netRegUsed = devWork.regUsed = 0;
      devWork.profilerEnabled = ncclProfilerPluginLoaded() && (task->eActivationMask & ncclProfileKernelCh);
      if (task->regBufType & NCCL_NET_REG_BUFFER)
        devWork.netRegUsed = 1;
      if (task->regBufType & (NCCL_IPC_REG_BUFFER | NCCL_NVLS_REG_BUFFER))
        devWork.regUsed = 1;

      struct ncclWorkList* workNode;
      if (task->regBufType & NCCL_NVLS_REG_BUFFER) {
        struct ncclDevWorkCollReg workReg = {};
        workReg.coll = devWork; // C++ struct assignment
        /* NVLS only has one send and recv buffer registered */
        workReg.dnInputs[0] = regBufSend[0];
        workReg.dnOutputs[0] = regBufRecv[0];
        workNode = ncclMemoryStackAllocInlineArray<ncclWorkList, ncclDevWorkCollReg>(&comm->memScoped, 1);
        workNode->workType = ncclDevWorkTypeCollReg;
        workNode->size = sizeof(struct ncclDevWorkCollReg);
        memcpy((void*)(workNode + 1), (void*)&workReg, workNode->size);
      } else {
        workNode = ncclMemoryStackAllocInlineArray<ncclWorkList, ncclDevWorkColl>(&comm->memScoped, 1);
        workNode->workType = ncclDevWorkTypeColl;
        workNode->size = sizeof(struct ncclDevWorkColl);
        memcpy((void*)(workNode + 1), (void*)&devWork, workNode->size);
      }

      ncclIntruQueueEnqueue(&planner->tmpCollWorkQueue, workNode);
    }
    task = task->next;
  }

  // Process broadcast tasks for runtimeConn
  if (comm->runtimeConn && planner->nTasksBcast > 0) {
    for (int peer = planner->bcast_info.minBcastPeer; peer <= planner->bcast_info.maxBcastPeer; peer++) {
      struct ncclTaskBcast* bcastTask = ncclIntruQueueHead(&planner->peers[peer].bcastQueue);
      while (bcastTask != nullptr) {
        if (comm->initAlgoChannels[NCCL_ALGO_RING] == false) {
          comm->initAlgoChannels[NCCL_ALGO_RING] = true;
          algoNeedConnect[NCCL_ALGO_RING] = true;
          *needConnect = true;
        }
        bcastTask = bcastTask->next;
      }
    }
  }

  return ncclSuccess;
}

static ncclResult_t addProfilerProxyOpIfNeeded(struct ncclComm* comm, struct ncclKernelPlan* plan, struct ncclProxyOp* op) {
  int tmp = op->pattern;
  op->pattern = ncclPatternProfiler;
  ncclResult_t ret = ncclAddProxyOpIfNeeded(comm, plan, op);
  op->pattern = tmp;
  return ret;
}

static ncclResult_t scheduleCollTasksToPlan(
    struct ncclComm* comm, struct ncclKernelPlan* plan, struct ncclKernelPlanBudget* budget
  ) {
  struct ncclKernelPlanner* planner = &comm->planner;
  // Estimate number of tasks that will fit in this plan.
  int nPlanColls = 0;
  size_t trafficBytes[2*2] = {0, 0, 0, 0}; // [collnet][nvls]
  int nChannels[2*2] = {0, 0, 0, 0}; // [collnet][nvls]
  int const nMaxChannels[2*2] = {comm->nChannels, comm->nvlsChannels, // [collnet][nvls]
                                 comm->nChannels, std::min(comm->nChannels, comm->nvlsChannels)};
  constexpr size_t MinTrafficPerChannel = 32 << 10; // 32K traffic as minimal
  do {
    size_t workBytes = 0;
    struct ncclTaskColl* task = ncclIntruQueueHead(&planner->collTaskQueue);
    struct ncclWorkList* workNode = ncclIntruQueueHead(&planner->collWorkQueue);
    while (task != nullptr) {
      int nBatches = divUp(nPlanColls, 4); // Rough guess: 4 colls per batch.
      if (!ncclTestBudget(budget, nBatches, workBytes + workNode->size)) goto plan_full;

      nPlanColls += 1;
      workBytes += workNode->size;
      int kind = 2*task->isCollnet + task->isNvls;
      trafficBytes[kind] += std::max(MinTrafficPerChannel, task->trafficBytes);
      nChannels[kind] += task->nMaxChannels;
      nChannels[kind] = std::min(nChannels[kind], nMaxChannels[kind]);
      task = task->next;
      workNode = workNode->next;
    }
  plan_full:;
  } while (0);

  int kindPrev = -1;
  size_t trafficPerChannel = 0;
  int channelId = 0;
  size_t currentTraffic = 0;
  while (nPlanColls!=0 && !ncclIntruQueueEmpty(&planner->collTaskQueue)) {
    struct ncclTaskColl* task = ncclIntruQueueHead(&planner->collTaskQueue);
    struct ncclWorkList* workNode = ncclIntruQueueHead(&planner->collWorkQueue);
    struct ncclDevWorkColl* devWork = (struct ncclDevWorkColl*)(workNode+1);
    size_t elementSize = ncclTypeSize(task->datatype);

    int kind = 2*task->isCollnet + task->isNvls;
    if (kind != kindPrev) {
      trafficPerChannel = divUp(trafficBytes[kind] / nChannels[kind], 16) * 16;
      kindPrev = kind;
      channelId = 0;
      currentTraffic = 0;
    }

    if (task->isCollnet) {
      int nChannels = task->nMaxChannels;
      // Ensure room for worst case of one new batch per channel
      if (!ncclTestBudget(budget, plan->nWorkBatches + nChannels, plan->workBytes + workNode->size)) {
        return ncclSuccess;
      }

      size_t globalBytesPerElement = elementSize*ncclFuncMaxSendRecvCount(task->func, comm->nRanks, 1);
      struct ncclProxyOp proxyOp;
      uint32_t chunkSize, directFlags=0;
      NCCLCHECK(calcCollChunking(comm, task, nChannels, globalBytesPerElement*task->count, &chunkSize, &directFlags, &proxyOp));
      devWork->channelLo = 0;
      devWork->channelHi = nChannels-1;
      devWork->collnet.count = task->count;
      devWork->collnet.chunkCount = chunkSize/ncclTypeSize(task->datatype);
      devWork->direct = directFlags;

      uint64_t proxyOpId = uint64_t(plan->collOpCount++)<<1 | 0;
      for (int c=devWork->channelLo; c <= (int)devWork->channelHi; c++) {
        proxyOp.channelId = c;
        proxyOp.opCount = proxyOpId;
        proxyOp.task.coll = task;
        proxyOp.rank = comm->rank;
        proxyOp.eActivationMask = task->eActivationMask;
        proxyOp.incWorkCounter = true;
        ncclAddWorkBatchToPlan(comm, plan, c, workNode->workType, task->devFuncId, plan->workBytes);
        // Set pattern to profiler to add a proxy profiler for kernel events
        NCCLCHECK(ncclAddProxyOpIfNeeded(comm, plan, &proxyOp));
        NCCLCHECK(addProfilerProxyOpIfNeeded(comm, plan, &proxyOp));
      }
    } else { // not task->isCollnet
      int trafficPerByte = ncclFuncTrafficPerByte(task->func, comm->nRanks);
      if (task->protocol == NCCL_PROTO_LL) trafficPerByte *= 4;
      size_t cellSize = divUp(divUp(MinTrafficPerChannel, (size_t)trafficPerByte), 16) * 16;
      int elementsPerCell = cellSize/elementSize;
      size_t cells = divUp(task->count*elementSize, cellSize);
      size_t trafficPerElement = elementSize*trafficPerByte;
      size_t trafficPerCell = cellSize*trafficPerByte;
      size_t cellsPerChannel = std::min(cells, divUp(trafficPerChannel, trafficPerCell));
      size_t cellsLo;
      if (channelId+1 == nMaxChannels[kind]) { // On last channel everything goes to "lo"
        cellsLo = cells;
      } else {
        cellsLo = std::min(cells, divUp((trafficPerChannel-currentTraffic),trafficPerCell));
      }
      int nMidChannels = (cells-cellsLo)/cellsPerChannel;
      size_t cellsHi = (cells-cellsLo)%cellsPerChannel;
      int nChannels = (cellsLo!=0 ? 1 : 0) + nMidChannels + (cellsHi!=0 ? 1 : 0);
      if (nMaxChannels[kind] < channelId + nChannels) { // Overflowed available channels
        nMidChannels = nMaxChannels[kind] - channelId - 2;
        cellsPerChannel = (cells-cellsLo)/(nMidChannels+1);
        cellsHi = cellsPerChannel + (cells-cellsLo)%(nMidChannels+1);
      }
      if (cellsHi == 0 && nMidChannels != 0) {
        cellsHi = cellsPerChannel;
        nMidChannels -= 1;
      }
      if (cellsLo == 0) { // Least channel skipped. Make the next channel the new least.
        channelId += 1;
        if (nMidChannels == 0) { cellsLo = cellsHi; cellsHi = 0; }
        else { cellsLo = cellsPerChannel; nMidChannels -= 1; }
      }
      size_t countMid = nMidChannels!=0 ? cellsPerChannel*elementsPerCell : 0;
      size_t countLo = cellsLo*elementsPerCell;
      size_t countHi = cellsHi*elementsPerCell;
      (countHi != 0 ? countHi : countLo) -= cells*elementsPerCell - task->count;

      nChannels = (countLo!=0 ? 1 : 0) + nMidChannels + (cellsHi!=0 ? 1 : 0);

      // Update number of channels propagated to the profiler
      task->nChannels = (uint8_t)nChannels;

      // Ensure room for worst case of one new batch per channel
      if (!ncclTestBudget(budget, plan->nWorkBatches + nChannels, plan->workBytes + workNode->size)) {
        return ncclSuccess;
      }

      devWork->channelLo = channelId;
      devWork->channelHi = channelId + nChannels-1;
      devWork->cbd.countLo = countLo;
      devWork->cbd.countMid = countMid;
      devWork->cbd.countHi = countHi;

      // calcCollChunking() uses global bytes instead of traffic which differs
      // in that allreduce isn't multiplied by 2.
      size_t globalBytesPerElement = elementSize*ncclFuncMaxSendRecvCount(task->func, comm->nRanks, 1);
      struct ncclProxyOp proxyOpLo, proxyOpMid, proxyOpHi;

      uint32_t chunkSize, directFlags=0;
      size_t grainSize = ncclProtoGrainSize(task->protocol);
      if (countLo != 0) {
        NCCLCHECK(calcCollChunking(comm, task, /*nChannels=*/1, globalBytesPerElement*countLo, &chunkSize, &directFlags, &proxyOpLo));
        devWork->cbd.chunkGrainsLo = chunkSize/grainSize;
      }
      if (countHi != 0) {
        NCCLCHECK(calcCollChunking(comm, task, /*nChannels=*/1, globalBytesPerElement*countHi, &chunkSize, &directFlags, &proxyOpHi));
        devWork->cbd.chunkGrainsHi = chunkSize/grainSize;
      }
      if (nMidChannels != 0) {
        NCCLCHECK(calcCollChunking(comm, task, /*nChannels=*/1, globalBytesPerElement*countMid, &chunkSize, &directFlags, &proxyOpMid));
        devWork->cbd.chunkGrainsMid = chunkSize/grainSize;
      }
      devWork->direct = directFlags;

      // Update the current channel and vacant traffic budget.
      if (countHi != 0) {
        channelId += nChannels-1;
        currentTraffic = cellsHi*elementsPerCell*trafficPerElement;
      } else if (nMidChannels != 0) {
        channelId += nChannels;
        currentTraffic = 0;
      } else {
        currentTraffic += cellsLo*elementsPerCell*trafficPerElement;
      }

      if (currentTraffic >= trafficPerChannel && channelId+1 != nMaxChannels[kind]) {
        channelId += 1;
        currentTraffic = 0;
      }

      uint64_t proxyOpId = uint64_t(plan->collOpCount++)<<1 | 0;
      for (int c=devWork->channelLo; c <= (int)devWork->channelHi; c++) {
        struct ncclProxyOp* proxyOp;
        if (c == (int)devWork->channelLo) {
          proxyOp = &proxyOpLo;
          proxyOp->loopOffset = 0;
          proxyOp->channelSize = countLo * elementSize;
        } else if (c == (int)devWork->channelHi) {
          proxyOp = &proxyOpHi;
          proxyOp->loopOffset = (countLo + nMidChannels * countMid) * elementSize;
          proxyOp->channelSize = countHi * elementSize;
        } else {
          proxyOp = &proxyOpMid;
          proxyOp->loopOffset = (countLo + (c - devWork->channelLo - 1) * countMid) * elementSize;
          proxyOp->channelSize = countMid * elementSize;
        }
        proxyOp->channelId = c;
        proxyOp->opCount = proxyOpId;
        proxyOp->task.coll = task;
        proxyOp->rank = comm->rank;
        proxyOp->ringAlgo = NULL;
        if (proxyOp->reg && task->algorithm == NCCL_ALGO_RING && (task->recvNetHandles[c] || task->sendNetHandles[c])) {
          if (task->func == ncclFuncAllGather) {
            proxyOp->ringAlgo = new RingAGAlgorithm(task->sendbuff, task->recvbuff, comm->nRanks, comm->channels[c].ring.userRanks, proxyOp->chunkSteps, proxyOp->sliceSteps, proxyOp->chunkSize, proxyOp->sliceSize, proxyOp->loopOffset, proxyOp->channelSize, elementSize, task->count * elementSize, task->sendNetHandles[c], task->recvNetHandles[c], task->srecvNetHandles[c]);
          } else if (task->func == ncclFuncAllReduce) {
            proxyOp->ringAlgo = new RingARAlgorithm(task->sendbuff, task->recvbuff, comm->nRanks, comm->channels[c].ring.index, proxyOp->chunkSteps, proxyOp->sliceSteps, proxyOp->chunkSize, proxyOp->sliceSize, proxyOp->loopOffset, proxyOp->channelSize, elementSize, task->sendNetHandles[c], task->recvNetHandles[c], task->srecvNetHandles[c]);
          } else if (task->func == ncclFuncBroadcast) {
            proxyOp->ringAlgo = new RingBCAlgorithm(task->sendbuff, task->recvbuff, comm->rank, task->root, comm->nRanks, comm->channels[c].ring.userRanks, proxyOp->chunkSteps, proxyOp->sliceSteps, proxyOp->chunkSize, proxyOp->sliceSize, proxyOp->loopOffset, proxyOp->channelSize, task->sendNetHandles[c], task->recvNetHandles[c], task->srecvNetHandles[c]);
          }
          proxyOp->ringAlgo->incRefCount();
        }
        proxyOp->eActivationMask = task->eActivationMask;
        proxyOp->incWorkCounter = true;
        proxyOp->nChannels = nChannels;
        ncclAddWorkBatchToPlan(comm, plan, c, workNode->workType, task->devFuncId, plan->workBytes);
        // Coverity reports "proxyOp->connection" as being possibly uninitialized.  It's hard to
        // determine if that's actually true but it's also not clear if that would be an issue.
        // coverity[uninit_use_in_call:FALSE]
        NCCLCHECK(ncclAddProxyOpIfNeeded(comm, plan, proxyOp));
        NCCLCHECK(addProfilerProxyOpIfNeeded(comm, plan, proxyOp));
      }
    }

    plan->channelMask |= (2ull<<devWork->channelHi) - (1ull<<devWork->channelLo);
    plan->threadPerBlock = std::max(plan->threadPerBlock, task->nWarps*WARP_SIZE);
    if (!plan->kernelSpecialized) {
      plan->kernelFn = ncclDevKernelForFunc[task->devFuncId];
      plan->kernelSpecialized = ncclDevKernelForFuncIsSpecialized[task->devFuncId];
    }
    // Profiler
    plan->groupApiEventHandle = task->groupApiEventHandle;

    if (comm->rank == 0) {
      INFO(NCCL_TUNING, "%s: %ld Bytes -> Algo %s proto %s channel{Lo..Hi}={%d..%d}",
        ncclFuncToString(task->func), task->count * ncclTypeSize(task->datatype), ncclAlgoToString(task->algorithm),
        ncclProtoToString(task->protocol), devWork->channelLo, devWork->channelHi);

      if (task->isCollnet) {
        TRACE(NCCL_COLL, "Collective %s(%s, %s, %s, %s) count=%ld devFuncId=%d channel{Lo..Hi}={%d..%d} count=%ld chunkCount=%d",
          ncclFuncToString(task->func), ncclDevRedOpToString(task->opDev.op),
          ncclDatatypeToString(task->datatype), ncclAlgoToString(task->algorithm),
          ncclProtoToString(task->protocol),
          (long)task->count, task->devFuncId, devWork->channelLo, devWork->channelHi,
          (long)devWork->collnet.count, devWork->collnet.chunkCount);
      } else {
        TRACE(NCCL_COLL, "Collective %s(%s, %s, %s, %s) count=%ld devFuncId=%d channel{Lo..Hi}={%d..%d} count{Lo,Mid,Hi}={%ld,%ld,%ld} chunkBytes{Lo,Mid,Hi}={%d,%d,%d}",
          ncclFuncToString(task->func), ncclDevRedOpToString(task->opDev.op),
          ncclDatatypeToString(task->datatype), ncclAlgoToString(task->algorithm),
          ncclProtoToString(task->protocol),
          (long)task->count, task->devFuncId, devWork->channelLo, devWork->channelHi,
          (long)devWork->cbd.countLo, (long)devWork->cbd.countMid, (long)devWork->cbd.countHi,
          int(devWork->cbd.chunkGrainsLo*ncclProtoGrainSize(task->protocol)),
          int(devWork->cbd.chunkGrainsMid*ncclProtoGrainSize(task->protocol)),
          int(devWork->cbd.chunkGrainsHi*ncclProtoGrainSize(task->protocol)));
      }
    }

    for (int i=0; i < task->nCleanupQueueElts; i++) {
      ncclIntruQueueEnqueue(&plan->cleanupQueue, ncclIntruQueueDequeue(&planner->collCleanupQueue));
    }
    ncclIntruQueueDequeue(&planner->collTaskQueue);
    ncclIntruQueueDequeue(&planner->collWorkQueue);
    nPlanColls -= 1;
    planner->nTasksColl -= 1;
    ncclIntruQueueEnqueue(&plan->collTaskQueue, task);
    ncclIntruQueueEnqueue(&plan->workQueue, workNode);
    plan->workBytes += workNode->size;
  }
  return ncclSuccess;
}

NCCL_PARAM(P2pLLThreshold, "P2P_LL_THRESHOLD", 16384);
NCCL_PARAM(ChunkSize, "CHUNK_SIZE", 0);

// Put p2p op in plan assuming there is sizeof(ncclDevWorkBatch) in batch budget
// and sizeof(ncclDevWorkP2p) in work budget. "sendRank" and "recvRank" must
// match the corresponding values for this round of the p2p schedule (no -1's).
// No-op's are encoded with a -1 size.
static ncclResult_t addP2pToPlan(
    struct ncclComm* comm, struct ncclKernelPlan* plan,
    int nChannelsMin, int nChannelsMax, int p2pEpoch, int p2pRound,
    int sendRank, void* sendAddr, ssize_t sendBytes,
    int recvRank, void* recvAddr, ssize_t recvBytes,
    const int planTotalTasks[], struct ncclTaskP2p** p2pTasks
  ) {
  ncclResult_t ret = ncclSuccess;
  constexpr int connIndex = 1;
  bool selfSend = (sendRank == comm->rank);
  // recv: dir=0, send: dir=1
  void* addrs[2] = {recvAddr, sendAddr};
  ssize_t bytes[2] = {recvBytes, sendBytes};
  bool protoLL[2] = {!selfSend, !selfSend};
  bool network[2] = {false, false};
  bool proxySameProcess[2] = {true, true};
  void** handles[2] = {NULL, NULL};
  uint8_t base = ncclP2pChannelBaseForRound(comm, p2pRound);
  struct ncclProxyOp proxyOps[2] = {};
  int nProxyOps = selfSend ? 0 : 2;
  if (!selfSend) {
    for (int part=0; part < nChannelsMax; part++) {
      int channelId = ncclP2pChannelForPart(comm->p2pnChannels, base, part);
      struct ncclChannelPeer** channelPeers = comm->channels[channelId].peers;
      for (int dir=0; dir <= 1; dir++) {
        int peerRank = dir ? sendRank : recvRank;
        struct ncclConnector* conn = dir ? &channelPeers[peerRank]->send[connIndex]
                                         : &channelPeers[peerRank]->recv[connIndex];
        protoLL[dir] &= conn->conn.buffs[NCCL_PROTO_LL] != nullptr;
        network[dir] |= conn->transportComm == (dir ? &netTransport.send : &netTransport.recv);
        proxySameProcess[dir] &= conn->proxyConn.sameProcess;
      }
    }
  }

  ssize_t paramChunkSize = ncclParamChunkSize();
  // Arrays indexed by dir where recv=0, send=1:
  int nChannels[2];
  int protocol[2];
  int stepSize[2];
  int chunkSize[2];
  int chunkDataSize[2];
  int chunkDataSize_u32fp8[2];
  bool netRegistered[2] = {false, false};
  bool ipcRegistered[2] = {false, false};

  for (int dir=0; dir < 2; dir++) { // 0=recv, 1=send
    // Assume SIMPLE protocol to start with to determine number of channels
    stepSize[dir] = comm->p2pChunkSize;

    if (bytes[dir] == -1) nChannels[dir] = 0;
    else if (bytes[dir] == 0) nChannels[dir] = 1;
    else {
      ssize_t minPartSize = comm->nNodes > 1 ? stepSize[dir]/2 : stepSize[dir]/8;
      ssize_t maxPartSize = comm->nNodes > 1 ? stepSize[dir]   : stepSize[dir]*32;
      nChannels[dir] = std::min<int>(nChannelsMin, divUp(bytes[dir], minPartSize));
      size_t partSize = std::max(minPartSize, divUp(bytes[dir], nChannels[dir]));
      while (partSize > maxPartSize && nChannels[dir] <= nChannelsMax/2) {
        nChannels[dir] *= 2;
        partSize = divUp(bytes[dir], nChannels[dir]);
      }
    }
    // Update number of channels propagated to the profiler
    if (p2pTasks[dir]) p2pTasks[dir]->nChannels = nChannels[dir];

    // Select protocol (LL vs SIMPLE) used based on payload per channel
    if (bytes[dir] != -1)
      protoLL[dir] &= bytes[dir] <= nChannels[dir] * ncclParamP2pLLThreshold();
    protocol[dir] = protoLL[dir] ? NCCL_PROTO_LL : NCCL_PROTO_SIMPLE;

    stepSize[dir] = comm->buffSizes[protocol[dir]]/NCCL_STEPS;
    if (protocol[dir] == NCCL_PROTO_SIMPLE) stepSize[dir] = comm->p2pChunkSize;
    chunkSize[dir] = stepSize[dir];
    if (paramChunkSize != 0) {
      chunkSize[dir] = paramChunkSize;
    } else if (network[dir]) {
      // Tune chunk size for the network
      if (protocol[dir] == NCCL_PROTO_SIMPLE && bytes[dir] < stepSize[dir]) chunkSize[dir] /= 4;
      else if (bytes[dir] < 8*stepSize[dir]) chunkSize[dir] /= 2;
    }

    chunkDataSize[dir] = chunkSize[dir];
    if (protocol[dir] == NCCL_PROTO_LL) chunkDataSize[dir] /= 2;
    chunkDataSize_u32fp8[dir] = u32fp8Encode(chunkDataSize[dir]);
    chunkDataSize[dir] = u32fp8Decode(chunkDataSize_u32fp8[dir]);
    chunkSize[dir] = chunkDataSize[dir];
    if (protocol[dir] == NCCL_PROTO_LL) chunkSize[dir] *= 2;

    if (p2pTasks[dir] && p2pTasks[dir]->allowUB) {
      if (network[dir]) {
        bool pxnUsed = !ncclPxnDisable(comm) && comm->isAllNvlink && comm->maxLocalRanks > 1;
        if (bytes[dir] > 0 && proxySameProcess[dir] && protocol[dir] == NCCL_PROTO_SIMPLE && (!pxnUsed)) {
          int regFlag = 0;
          NCCLCHECKGOTO(ncclCalloc(&handles[dir], nChannelsMax), ret, cleanup);
          for (int part = 0; part < nChannelsMax; part++) {
            int channelId = ncclP2pChannelForPart(comm->p2pnChannels, base, part);
            struct ncclChannelPeer** channelPeers = comm->channels[channelId].peers;
            int peerRank = dir ? sendRank : recvRank;
            struct ncclConnector* conn = dir ? &channelPeers[peerRank]->send[connIndex]
              : &channelPeers[peerRank]->recv[connIndex];
              if (conn->conn.flags & NCCL_DIRECT_NIC)
                ncclRegisterP2pNetBuffer(comm, addrs[dir], bytes[dir], conn, &regFlag, &handles[dir][part], &plan->cleanupQueue);
              if (!regFlag) break;
          }
          netRegistered[dir] = regFlag ? true : false;
        }
      } else if (bytes[dir] > 0 && addrs[dir] && protocol[dir] == NCCL_PROTO_SIMPLE && !selfSend) {
        int peerRank = dir ? sendRank : recvRank;
        int regFlag = 0;
        int channelId = ncclP2pChannelForPart(comm->p2pnChannels, base, 0);
        struct ncclChannelPeer** channelPeers = comm->channels[channelId].peers;
        struct ncclConnector* conn = dir ? &channelPeers[peerRank]->send[connIndex]
          : &channelPeers[peerRank]->recv[connIndex];
          void* regAddr = NULL;
          if (conn->conn.flags & (NCCL_P2P_WRITE | NCCL_P2P_READ)) {
            // We require users registering buffers on both sides
            NCCLCHECKGOTO(ncclRegisterP2pIpcBuffer(comm, addrs[dir], bytes[dir], peerRank, &regFlag, &regAddr, &plan->cleanupQueue), ret, cleanup);
            if (regFlag) {
              if (dir == 0 && (conn->conn.flags & NCCL_P2P_WRITE)) recvAddr = regAddr;
              else if (dir == 1 && (conn->conn.flags & NCCL_P2P_READ)) sendAddr = regAddr;
            }
          }
          ipcRegistered[dir] = regFlag ? true : false;
      }
    }
  }

  struct ncclWorkList* workNode;
  workNode = ncclMemoryStackAllocInlineArray<ncclWorkList, ncclDevWorkP2p>(&comm->memScoped, 1);
  workNode->workType = ncclDevWorkTypeP2p;
  workNode->size = sizeof(struct ncclDevWorkP2p);
  ncclIntruQueueEnqueue(&plan->workQueue, workNode);
  uint32_t workOffset;
  workOffset = plan->workBytes;
  plan->workBytes += sizeof(struct ncclDevWorkP2p);

  struct ncclDevWorkP2p* work;
  work = (struct ncclDevWorkP2p*)(workNode+1);
  work->nP2pChannels = comm->p2pnChannels;
  work->channelBase = base;
  work->nSendChannels = nChannels[1];
  work->sendProtoLL = protoLL[1];
  work->sendNetReg = netRegistered[1];
  work->sendIpcReg = ipcRegistered[1];
  work->sendChunkSize_u32fp8 = chunkDataSize_u32fp8[1];
  work->sendRank = sendRank;
  work->sendAddr = sendAddr;
  work->sendBytes = sendBytes==-1 ? 0 : sendBytes;
  work->nRecvChannels = nChannels[0];
  work->recvProtoLL = protoLL[0];
  work->recvNetReg = netRegistered[0];
  work->recvIpcReg = ipcRegistered[0];
  work->recvChunkSize_u32fp8 = chunkDataSize_u32fp8[0];
  work->recvRank = recvRank;
  work->recvAddr = recvAddr;
  work->recvBytes = recvBytes==-1 ? 0 : recvBytes;
  work->profilerEnabled = ncclProfilerPluginLoaded() && ((p2pTasks[0] ? p2pTasks[0] : p2pTasks[1])->eActivationMask & ncclProfileKernelCh);

  for (int dir=0; dir < nProxyOps; dir++) {
    struct ncclProxyOp* op = &proxyOps[dir];
    op->root = dir ? sendRank : recvRank;
    op->sliceSteps = 1;
    op->chunkSteps = 1;
    op->dtype = ncclInt8;
    op->redOp = ncclSum;
    op->protocol = protocol[dir];
    op->pattern = dir ? ncclPatternSend : ncclPatternRecv;
    op->chunkSize = chunkSize[dir];
    op->reg = netRegistered[dir];
    op->coll = p2pTasks[dir] ? p2pTasks[dir]->func : 0;
    op->collAPI = p2pTasks[dir] ? p2pTasks[dir]->collAPI : 0;
    op->task.p2p = p2pTasks[dir];
    op->rank = comm->rank;
    op->eActivationMask = p2pTasks[dir] ? p2pTasks[dir]->eActivationMask : 0;
    // The following are modified per channel part in addWorkToChannels():
    // op->buffer, op->nbytes, op->nsteps = ...;
  }

  nChannelsMax = std::max(nChannels[0], nChannels[1]);
  // Determine how many peers this plan will target concurrently. Make a
  // simplifying assumption that each task targets a different peer.
  // Each task is striped across 'nChannelsMax' of 'p2pnChannels' channels.
  // Each channel runs up to NCCL_MAX_DEV_WORK_P2P_PER_BATCH tasks concurrently.
  int maxConcurrent;
  int concurrentTasks[2];
  maxConcurrent = comm->p2pnChannels / nChannelsMax * NCCL_MAX_DEV_WORK_P2P_PER_BATCH;
  concurrentTasks[0] = std::min(planTotalTasks[0], maxConcurrent);
  concurrentTasks[1] = std::min(planTotalTasks[1], maxConcurrent);
  for (int part=0; part < nChannelsMax; part++) {
    int incWorkCounter = -1;
    int channelId = ncclP2pChannelForPart(comm->p2pnChannels, base, part);
    plan->channelMask |= uint64_t(1)<<channelId;
    // Add batch first.
    ncclAddWorkBatchToPlan(comm, plan, channelId, ncclDevWorkTypeP2p, ncclDevFuncId_P2p(), workOffset, p2pEpoch, p2pRound);
    for (int dir=0; dir < nProxyOps; dir++) {
      // Partition steps across channels.
      int nParts = dir ? work->nSendChannels : work->nRecvChannels;
      void* addr = dir ? work->sendAddr : work->recvAddr;
      size_t bytes = dir ? work->sendBytes : work->recvBytes;

      proxyOps[dir].recvbuff = nullptr;
      if (nParts <= part) {
        proxyOps[dir].nsteps = 0;
      } else if (bytes == 0) {
        proxyOps[dir].nsteps = 1;
        proxyOps[dir].nbytes = 0;
      } else {
        size_t chunkDataSize = u32fp8Decode(dir ? work->sendChunkSize_u32fp8 : work->recvChunkSize_u32fp8);
        size_t partBeg, partEnd;
        ncclP2pPartBounds(nParts, part, bytes, &partBeg, &partEnd);
        if (proxyOps[dir].reg) {
          (dir ? proxyOps[dir].sendbuff : proxyOps[dir].recvbuff) = (uint8_t*)addr + partBeg;
          (dir ? proxyOps[dir].sendMhandle : proxyOps[dir].recvMhandle) = handles[dir][part];
          proxyOps[dir].nbytes = partEnd - partBeg;
          proxyOps[dir].nsteps = DIVUP(proxyOps[dir].nbytes, NCCL_MAX_NET_SIZE);
        } else {
          proxyOps[dir].nsteps = divUp(partEnd-partBeg, chunkDataSize);
          proxyOps[dir].nbytes = std::min(partEnd-partBeg, chunkDataSize);
        }
        if (proxyOps[dir].protocol == NCCL_PROTO_LL) {
          proxyOps[dir].nbytes *= 2;
          proxyOps[dir].nbytes = roundUp(proxyOps[dir].nbytes, sizeof(union ncclLLFifoLine));
        }
      }

      // Increment work counter for <send, recv> pair rather than individual p2p
      if (proxyOps[dir].nsteps && incWorkCounter < 0) {
        proxyOps[dir].incWorkCounter = true;
        incWorkCounter = dir;
      }

      if (proxyOps[dir].nsteps != 0) {
        // Calculate the opCount after adding batch since then the batch count will
        // equal one plus the batch index this p2p settled in.
        proxyOps[dir].channelId = channelId;
        proxyOps[dir].opCount = uint64_t(comm->planner.wipPlan.channels[channelId].nWorkBatchesP2p)<<1 | 1;
        proxyOps[dir].nChannels = nChannels[dir];
        proxyOps[dir].nPeers = concurrentTasks[dir];
        NCCLCHECKGOTO(ncclAddProxyOpIfNeeded(comm, plan, &proxyOps[dir]), ret, cleanup);
        NCCLCHECKGOTO(addProfilerProxyOpIfNeeded(comm, plan, &proxyOps[dir]), ret, cleanup);
      }
    }
  }
cleanup:
  free(handles[0]);
  free(handles[1]);
  return ret;
}

static int calcP2pChannelCount(size_t totalSize, int minChannels, int maxChannels, size_t minSize, size_t maxSize) {
  size_t size = std::max(minSize, divUp(totalSize, minChannels));
  int nChannels = minChannels;
  while (size > maxSize && nChannels <= maxChannels/2) {
    nChannels *= 2;
    size = divUp(totalSize, nChannels);
  }
  return nChannels;
}

static ncclResult_t scheduleP2pTasksToPlan(struct ncclComm* comm, int* p2pEpoch, struct ncclKernelPlan* plan, struct ncclKernelPlanBudget* budget) {
  int nRanks = comm->nRanks;
  struct ncclKernelPlanner::Peer* peers = comm->planner.peers;

  plan->threadPerBlock = std::max(plan->threadPerBlock, NCCL_MAX_NTHREADS);
  if (!plan->kernelSpecialized) {
    plan->kernelFn = ncclDevKernelForFunc[ncclDevFuncId_P2p()];
    plan->kernelSpecialized = ncclDevKernelForFuncIsSpecialized[ncclDevFuncId_P2p()];
  }

  // Compute how much to split operations
  // Try to use all channels
  int nChannelsMax = comm->p2pnChannelsPerPeer;
  int nChannelsMin = nChannelsMax;
  // Try to use all channels, but one channel per operation.
  while (nChannelsMin*nRanks > comm->p2pnChannels && nChannelsMin > 1) nChannelsMin /= 2;

  // Save the total count of send/recv tasks in the plan
  int planTotalTasks[2] = {comm->planner.nTasksP2pRecv, comm->planner.nTasksP2pSend};
  while (comm->planner.nTasksP2p != 0) {
    // increment before to avoid issue when the budget is too small and we exit early
    (*p2pEpoch)++;
    for (int round=0; round < nRanks; round++) {
      int sendRank = comm->p2pSchedule[round].sendRank;
      int recvRank = comm->p2pSchedule[round].recvRank;
      struct ncclTaskP2p* send = ncclIntruQueueHead(&peers[sendRank].sendQueue);
      struct ncclTaskP2p* recv = ncclIntruQueueHead(&peers[recvRank].recvQueue);
      if (send == nullptr && recv == nullptr) continue;

      if (sendRank == comm->rank) {
        if (send != nullptr && recv == nullptr) {
          WARN("Trying to send to self without a matching recv");
          return ncclInvalidUsage;
        }
        if (send == nullptr && recv != nullptr) {
          WARN("Trying to recv to self without a matching send");
          return ncclInvalidUsage;
        }
      }
      ssize_t sendBytes = send ? send->bytes : -1;
      ssize_t recvBytes = recv ? recv->bytes : -1;
      void* sendBuff = send ? send->buff : nullptr;
      void* recvBuff = recv ? recv->buff : nullptr;

      if (sendRank == comm->rank && send->buff == recv->buff) {
        // Skip send to self in-place (we don't need to support this).
        ncclIntruQueueDequeue(&peers[sendRank].sendQueue);
        ncclIntruQueueDequeue(&peers[recvRank].recvQueue);
        ncclMemoryPoolFree(&comm->memPool_ncclTaskP2p, send);
        ncclMemoryPoolFree(&comm->memPool_ncclTaskP2p, recv);
        comm->planner.nTasksP2p -= 2;
        comm->planner.nTasksP2pSend -= 1;
        comm->planner.nTasksP2pRecv -= 1;
      } else {
        // Ensure room for worst case of one new batch per channel.
        if (!ncclTestBudget(budget, plan->nWorkBatches+nChannelsMax, plan->workBytes + sizeof(struct ncclDevWorkP2p))) {
          return ncclSuccess;
        }
        struct ncclTaskP2p* p2pTasks[2] = { recv, send };
        NCCLCHECK(addP2pToPlan(comm, plan, nChannelsMin, nChannelsMax, *p2pEpoch, round, sendRank, sendBuff, sendBytes, recvRank, recvBuff, recvBytes, planTotalTasks, p2pTasks));
        if (send != nullptr) {
          ncclIntruQueueDequeue(&peers[sendRank].sendQueue);
          // Profiler - We can overwrite groupAPI event handles here since all operations here belong to the same group
          plan->groupApiEventHandle = send->groupApiEventHandle;
          ncclIntruQueueEnqueue(&plan->p2pTaskQueue, send);
          comm->planner.nTasksP2p -= 1;
          comm->planner.nTasksP2pSend -= 1;
        }
        if (recv != nullptr) {
          ncclIntruQueueDequeue(&peers[recvRank].recvQueue);
          // Profiler - We can overwrite groupAPI event handles here since all operations here belong to the same group
          plan->groupApiEventHandle = recv->groupApiEventHandle;
          ncclIntruQueueEnqueue(&plan->p2pTaskQueue, recv);
          comm->planner.nTasksP2p -= 1;
          comm->planner.nTasksP2pRecv -= 1;
        }
      }
    }
  }
  return ncclSuccess;
}

// Spin until its safe to increase comm->workFifoProduced to desiredProduced.
static ncclResult_t waitWorkFifoAvailable(struct ncclComm* comm, uint32_t desiredProduced) {
  bool hasRoom = (desiredProduced - comm->workFifoConsumed) <= comm->workFifoBytes;
  if (!hasRoom) {
    while (true) {
      // Check abort flag to break deadlock when abort is signaled
      if (__atomic_load_n(comm->abortFlag, __ATOMIC_ACQUIRE)) {
        return ncclInternalError;
      }

      NCCLCHECK(ncclCommPollEventCallbacks(comm, /*waitSome=*/true));
      hasRoom = (desiredProduced - comm->workFifoConsumed) <= comm->workFifoBytes;
      if (hasRoom) break;
      std::this_thread::yield();
    }
  }
  return ncclSuccess;
}

namespace {
  struct uploadWork_cleanup_t {
    struct ncclCommEventCallback base;
    void *hostBuf;
  };
  ncclResult_t uploadWork_cleanup_fn(
      struct ncclComm* comm, struct ncclCommEventCallback* cb
    ) {
    struct uploadWork_cleanup_t* me = (struct uploadWork_cleanup_t*)cb;
    free(me->hostBuf);
    CUDACHECK(cudaEventDestroy(me->base.event));
    free(me);
    return ncclSuccess;
  }
}

static ncclResult_t uploadWork(struct ncclComm* comm, struct ncclKernelPlan* plan) {
  if (plan->isSymColl || plan->isCeColl || plan->isRma) return ncclSuccess;

  size_t workBytes = plan->workBytes;
  size_t batchBytes = plan->nWorkBatches*sizeof(struct ncclDevWorkBatch);
  void* fifoBufHost;
  uint32_t fifoCursor, fifoMask;

  switch (plan->workStorageType) {
  case ncclDevWorkStorageTypeArgs:
    plan->kernelArgs->workBuf = nullptr;
    fifoBufHost = (void*)plan->kernelArgs;
    fifoCursor = sizeof(ncclDevKernelArgs) + batchBytes;
    fifoMask = ~0u;
    break;
  case ncclDevWorkStorageTypeFifo:
    fifoBufHost = comm->workFifoBuf;
    fifoCursor = comm->workFifoProduced;
    fifoMask = comm->workFifoBytes-1;
    NCCLCHECK(waitWorkFifoAvailable(comm, fifoCursor + workBytes));
    plan->kernelArgs->workBuf = comm->workFifoBufDev;
    break;
  case ncclDevWorkStorageTypePersistent:
    // We rely on 16-byte alignment
    #if __cplusplus >= 201103L
    fifoBufHost = aligned_alloc(16, ROUNDUP(workBytes, 16));
    #else
    static_assert(16 <= alignof(max_align_t), "We rely on 16-byte alignment.");
    fifoBufHost = malloc(workBytes);
    #endif
    fifoCursor = 0;
    fifoMask = ~0u;
    break;
  default:
    return ncclInternalError;
  }
  plan->kernelArgs->workMask = fifoMask;

  // Batches were placed after kernelArgs by finishPlan(). Only thing left to
  // do is translate the work offset from zero based (in plan) to:
  //  ncclDevWorkStorageTypeArgs: offset from beginning of kernel args
  //  ncclDevWorkStorageTypeFifo: offset from base of fifo
  //  ncclDevWorkStorageTypePersistent: no translation since our dedicated buffer will also begin at zero.
  struct ncclDevWorkBatch* batchZero = (struct ncclDevWorkBatch*)(plan->kernelArgs+1);
  for (int b=0; b < plan->nWorkBatches; b++) {
    batchZero[b].offsetBase += fifoCursor;
  }

  // Write the channel-shared work structs.
  struct ncclWorkList* workNode = ncclIntruQueueHead(&plan->workQueue);
  while (workNode != nullptr) {
    char* dst = (char*)fifoBufHost;
    char* src = (char*)(workNode+1);
    for (int n = workNode->size; n != 0; n -= 16) {
      memcpy(
        COMPILER_ASSUME_ALIGNED(dst + (fifoCursor & fifoMask), 16),
        COMPILER_ASSUME_ALIGNED(src, 16),
        16
      );
      fifoCursor += 16;
      src += 16;
    }
    workNode = workNode->next;
  }

  switch (plan->workStorageType) {
  case ncclDevWorkStorageTypeFifo:
    comm->workFifoProduced = fifoCursor;
    if (comm->workFifoBufGdrHandle != nullptr) wc_store_fence();
    break;
  case ncclDevWorkStorageTypePersistent:
    { ncclResult_t result = ncclSuccess;
      struct uploadWork_cleanup_t* cleanup = nullptr;
      cudaStreamCaptureMode mode = cudaStreamCaptureModeRelaxed;
      void* fifoBufDev = nullptr;
      cudaStream_t deviceStream;

      CUDACHECKGOTO(cudaThreadExchangeStreamCaptureMode(&mode), result, fail);

      // Acquire deviceStream. Since the user's graph will be launched later and it also
      // acquires the deviceStream, it will observe this upload.
      NCCLCHECKGOTO(ncclStrongStreamAcquire(ncclCudaGraphNone(comm->config.graphUsageMode), &comm->sharedRes->deviceStream, /*concurrent=*/false, &deviceStream), result, fail);

      CUDACHECKGOTO(cudaMallocAsync(&fifoBufDev, workBytes, comm->memPool, deviceStream), result, fail);
      plan->workBufPersistent = fifoBufDev;
      plan->kernelArgs->workBuf = fifoBufDev;

      // coverity[uninit_use_in_call:FALSE] => fifoBufHost is never NULL
      CUDACHECKGOTO(cudaMemcpyAsync(fifoBufDev, fifoBufHost, workBytes, cudaMemcpyDefault, deviceStream), result, fail);
      cudaEvent_t memcpyDone;
      CUDACHECKGOTO(cudaEventCreateWithFlags(&memcpyDone, cudaEventDisableTiming), result, fail);
      CUDACHECKGOTO(cudaEventRecord(memcpyDone, deviceStream), result, fail);

      NCCLCHECKGOTO(ncclCalloc(&cleanup, 1), result, fail);
      cleanup->base.fn = uploadWork_cleanup_fn;
      cleanup->base.event = memcpyDone;
      cleanup->hostBuf = fifoBufHost;
      ncclIntruQueueEnqueue(&comm->eventCallbackQueue, (struct ncclCommEventCallback *)cleanup);

      NCCLCHECKGOTO(ncclStrongStreamRelease(ncclCudaGraphNone(comm->config.graphUsageMode), &comm->sharedRes->deviceStream, /*concurrent=*/false), result, fail);
      NCCLCHECKGOTO(ncclCommPollEventCallbacks(comm, /*waitSome=*/false), result, fail);

    finish_scope:
      if (mode != cudaStreamCaptureModeRelaxed) (void)cudaThreadExchangeStreamCaptureMode(&mode);
      return result;
    fail:
      if (!cleanup) free(fifoBufHost);
      goto finish_scope;
    } break;
  default: break;
  }
  return ncclSuccess;
}

static int geteActivationMask(struct ncclProxyOp * op) {
  if (ncclFuncSendRecv <= op->coll && op->coll <= ncclFuncRecv) {
    return op->task.p2p->eActivationMask;
  }
  if (op->coll == ncclFuncAllGatherV) {
    return 0;
  }
  return op->task.coll->eActivationMask;
}

static void *gettaskEventHandle(struct ncclProxyOp * op) {
  if (ncclFuncSendRecv <= op->coll && op->coll <= ncclFuncRecv) {
    return op->task.p2p->eventHandle;
  }
  if (op->coll == ncclFuncAllGatherV) {
    return nullptr;
  }
  return op->task.coll->eventHandle;
}

static ncclResult_t uploadProxyOps(struct ncclComm* comm, struct ncclKernelPlan* plan) {
  uint64_t collOpCount = comm->sharedRes->collOpCount;
  uint64_t p2pOpBump[MAXCHANNELS] = {/*0...*/};
  // Advance comm's collOpCount by number of colls in this plan.
  int hasp2p = 0;
  comm->sharedRes->collOpCount += plan->collOpCount;
  comm->collOpCount += plan->collOpCount;

  struct ncclProxyOp* op = ncclIntruQueueHead(&plan->proxyOpQueue);
  while (op != nullptr) {
    op->profilerContext = comm->profilerContext;
    op->eActivationMask = geteActivationMask(op);
    op->taskEventHandle = gettaskEventHandle(op);
    ncclProfilerAddPidToProxyOp(op);

    uint64_t oldId = op->opCount;
    // Ignoring the bottom tag bit, opCount's are zero-based within plan so
    // translate them to the tip of the comm's history.
    if (oldId & 1) { // p2p
      // opCount is monotonic increasing within a plan's channel so just
      // remember last value to compute max.
      p2pOpBump[op->channelId] = (oldId>>1) + 1; // +1 to ensure next plan doesn't collide
      op->opCount = (comm->sharedRes->p2pOpCount[op->channelId]<<1) + oldId;
      hasp2p = 1;
    } else { // coll
      op->opCount = (collOpCount<<1) + oldId;
    }

    NCCLCHECK(ncclProxySaveOp(comm, op, nullptr));
    op->opCount = oldId; // Restore for next uploadProxyOps()
    op = op->enqNext;
  }

  if (hasp2p) {
    for (int c=0; c < MAXCHANNELS; c++) {
      // Advance channel's p2pOpCount by number of p2p's in this plan channel.
      comm->sharedRes->p2pOpCount[c] += p2pOpBump[c];
    }
  }
  return ncclSuccess;
}

static ncclResult_t hostStreamPlanTask(struct ncclComm* comm, struct ncclKernelPlan* plan) {
  NCCLCHECK(ncclProfilerStartGroupEvent(plan));
  NCCLCHECK(ncclProfilerStartTaskEvents(plan));
  if (ncclIntruQueueHead(&plan->proxyOpQueue)) {
    NCCLCHECK(uploadProxyOps(comm, plan));
    NCCLCHECK(ncclProxyStart(comm));
  }
  NCCLCHECK(ncclProfilerStopTaskEvents(plan));
  NCCLCHECK(ncclProfilerStopGroupEvent(plan));
  if (!plan->persistent) {
    // Notify main thread of our reclaiming. This will reclaim plan concurrently.
    ncclIntruQueueMpscEnqueue(&comm->callbackQueue, &plan->reclaimer);
  }
  return ncclSuccess;
}

static void CUDART_CB hostStreamPlanCallback(void *plan_) {
  NCCL_NVTX3_FUNC_RANGE;
  struct ncclKernelPlan* plan = (struct ncclKernelPlan*)plan_;
  ncclResult_t result = hostStreamPlanTask(plan->comm, plan);
  if (result != ncclSuccess) {
    WARN("hostStreamPlanCallback() failed : %s", ncclGetErrorString(result));
  }
  return;
}

static ncclResult_t reclaimPlan(struct ncclComm* comm, struct ncclCommCallback* me) {
  struct ncclKernelPlan* plan = (struct ncclKernelPlan*)me; // cast from first member `reclaim`
  if (plan->persistent) {
    comm->sharedRes->persistentRefs -= 1;
    comm->localPersistentRefs -= 1;
    if (plan->workStorageType == ncclDevWorkStorageTypePersistent) {
      cudaStreamCaptureMode mode = cudaStreamCaptureModeRelaxed;
      CUDACHECK(cudaThreadExchangeStreamCaptureMode(&mode));
      CUDACHECK(cudaFree(plan->workBufPersistent));
      CUDACHECK(cudaThreadExchangeStreamCaptureMode(&mode));
    }
  }
  if (plan->isSymColl) {
    free(plan->kernelSymArgs);
  }
  // Free coll tasks
  struct ncclTaskColl* ct = ncclIntruQueueHead(&plan->collTaskQueue);
  while (ct != nullptr) {
    struct ncclTaskColl* ct1 = ct->next;
    free(ct->sendNetHandles);
    free(ct->recvNetHandles);
    free(ct->srecvNetHandles);
    ncclMemoryPoolFree(&comm->memPool_ncclTaskColl, ct);
    ct = ct1;
  }
  // Free p2p tasks
  struct ncclTaskP2p* pt = ncclIntruQueueHead(&plan->p2pTaskQueue);
  while (pt != nullptr) {
    struct ncclTaskP2p* pt1 = pt->next;
    ncclMemoryPoolFree(&comm->memPool_ncclTaskP2p, pt);
    pt = pt1;
  }
  // Free broadcast tasks
  struct ncclTaskBcast* bt = ncclIntruQueueHead(&plan->bcastTaskQueue);
  while (bt != nullptr) {
    struct ncclTaskBcast* bt1 = bt->next;
    ncclMemoryPoolFree(&comm->memPool_ncclTaskBcast, bt);
    bt = bt1;
  }
  // Free proxy ops
  struct ncclProxyOp* q = ncclIntruQueueHead(&plan->proxyOpQueue);
  while (q != nullptr) {
    struct ncclProxyOp* q1 = q->enqNext;
    if (q->ringAlgo && q->ringAlgo->decRefCount() == 0) delete q->ringAlgo;
    ncclMemoryPoolFree(&comm->memPool_ncclProxyOp, q);
    q = q1;
  }
  // Run other free callbacks
  ncclResult_t result = ncclSuccess;
  while (!ncclIntruQueueEmpty(&plan->cleanupQueue)) {
    struct ncclCommCallback* cb = ncclIntruQueueDequeue(&plan->cleanupQueue);
    ncclResult_t res1 = cb->fn(comm, cb); // Expect to reclaim memory of cb
    if (res1 != ncclSuccess) result = res1;
  }
  NCCLCHECK(result);
  // Free plan struct
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

NCCL_PARAM(LaunchOrderImplicit, "LAUNCH_ORDER_IMPLICIT", 0);

namespace {
  enum ncclImplicitOrder {
    ncclImplicitOrderNone,
    ncclImplicitOrderSerial,
    ncclImplicitOrderLaunch
  };
}

static ncclResult_t getImplicitOrder(enum ncclImplicitOrder *mode, bool capturing, int driver=-1) {
  if (ncclParamLaunchOrderImplicit()) {
    if (driver < 0) { NCCLCHECK(ncclCudaDriverVersion(&driver)); }
    if (capturing && driver < 12090) { *mode = ncclImplicitOrderSerial; return ncclSuccess; }
    *mode = 12030 <= std::min<int>(CUDART_VERSION, driver) ? ncclImplicitOrderLaunch : ncclImplicitOrderSerial;
    return ncclSuccess;
  }
  *mode = ncclImplicitOrderNone;
  return ncclSuccess;
}

ncclResult_t ncclLaunchPrepare(struct ncclComm* comm) {
  ncclResult_t result = ncclSuccess;
  struct ncclKernelPlanner* planner = &comm->planner;
  bool persistent = ncclCudaGraphValid(planner->capturingGraph);
  planner->persistent = persistent;
  // Operations from different plans will not be batched together. A new batch will be created for each new plan that is used to schedule the ops (see ncclAddWorkBatchToPlan).
  // For p2p ops, we further guarantee that ops from different epochs will not be batched together (to avoid hangs).
  // The p2pEpoch value is incremented in scheduleP2pTasksToPlan and its value is carried over from one plan to another (even if not strictly required)
  int nPlans = 0, p2pEpoch=0;

  if (planner->nTasksColl + planner->nTasksP2p + planner->nTasksBcast != 0 ||
      !ncclIntruQueueEmpty(&planner->collSymTaskQueue) ||
      !ncclIntruQueueEmpty(&planner->collCeTaskQueue) ||
      planner->nTasksRma != 0) {
    do {
      memset(&planner->wipPlan, 0, sizeof(planner->wipPlan));

      struct ncclKernelPlan* plan = ncclMemoryPoolAlloc<struct ncclKernelPlan>(&comm->memPool_ncclKernelPlan, &comm->memPermanent);
      plan->comm = comm;
      plan->reclaimer.fn = reclaimPlan;
      plan->persistent = persistent;
      // finishPlan() promotes ncclDevWorkStorageType[Fifo|Persistent]->Args if the work can fit.
      plan->workStorageType = persistent ? ncclDevWorkStorageTypePersistent
                                         : ncclDevWorkStorageTypeFifo;

      if (planner->nTasksRma != 0) {
        NCCLCHECKGOTO(scheduleRmaTasksToPlan(comm, plan), result, failure);
        if (plan->isRma && plan->rmaArgs != NULL && plan->rmaArgs->nRmaTasks > 0) {
          ncclIntruQueueEnqueue(&planner->planQueue, plan);
          nPlans += 1;
        }
      } else if (!ncclIntruQueueEmpty(&planner->collCeTaskQueue)) {
        struct ncclTaskColl* task = ncclIntruQueueHead(&planner->collCeTaskQueue);
        plan->isCeColl = true;
        plan->ceCollArgs = ncclMemoryStackAlloc<struct ncclCeCollArgs>(&comm->memScoped);
        plan->ceCollArgs->rootRank = task->root;
        plan->ceCollArgs->datatype = task->datatype;
        plan->ceCollArgs->nElts = task->count;
        plan->ceCollArgs->eltSize = ncclTypeSize(task->datatype);
        plan->ceCollArgs->sendBuff = (uint8_t*)task->sendbuff;
        plan->ceCollArgs->recvBuff = (uint8_t*)task->recvbuff;
        plan->ceCollArgs->func = task->func;
        plan->ceCollArgs->sendWin = task->sendWin;
        plan->ceCollArgs->recvWin = task->recvWin;
        plan->ceCollArgs->collApiEventHandle = task->collApiEventHandle;

        if (comm->rank == 0) {
          const char* nvlsSync = comm->nvlsSupport ? "; CE synchronization with NVLS" : "";
          INFO(NCCL_TUNING, "%s [Copy Engine]: %ld Bytes -> cudaMemcpy%s",
            ncclFuncToString(task->func), task->count * ncclTypeSize(task->datatype), nvlsSync);
        }

        ncclIntruQueueEnqueue(&planner->planQueue, plan);
        ncclIntruQueueDequeue(&planner->collCeTaskQueue);
        ncclMemoryPoolFree(&comm->memPool_ncclTaskColl, task);
        nPlans += 1;
      } else {
        if (!ncclIntruQueueEmpty(&planner->collSymTaskQueue)) {
          NCCLCHECKGOTO(ncclSymmetricTaskScheduler(comm, &planner->collSymTaskQueue, plan), result, failure);
        }
        else {
          struct ncclKernelPlanBudget budget;
          budget.inArgsBytes = comm->workArgsBytes - sizeof(struct ncclDevKernelArgs);
          // Non-persistent kernels fill up at most half of our fifo per kernel.
          budget.outArgsBytes = plan->persistent ? (1<<30) : comm->workFifoBytes/2;

          // Drain coll tasks first. This is essential since we partition tasks based
          // on the work budget and p2p work isn't collective. If we were to drain p2p
          // first, the place where we cut the kernel could vary by rank which would
          // cause the "shortest channel first" channel picker to have divergent results.
          if (planner->nTasksColl != 0) {
            NCCLCHECKGOTO(scheduleCollTasksToPlan(comm, plan, &budget), result, failure);
          }
          if (planner->nTasksColl == 0 && planner->nTasksBcast != 0) {
            NCCLCHECKGOTO(ncclScheduleBcastTasksToPlan(comm, plan, &budget), result, failure);
          }
          // And only drain p2p tasks once colls are depleted.
          if (planner->nTasksColl == 0 && planner->nTasksBcast == 0 && planner->nTasksP2p != 0) {
            NCCLCHECKGOTO(scheduleP2pTasksToPlan(comm, &p2pEpoch, plan, &budget), result, failure);
          }
        }

        finishPlan(comm, plan);
        if (plan->workBytes != 0) {
          ncclIntruQueueEnqueue(&planner->planQueue, plan);
          nPlans += 1;
        }
      }
    } while (planner->nTasksColl + planner->nTasksP2p + planner->nTasksBcast != 0 ||
             !ncclIntruQueueEmpty(&planner->collSymTaskQueue) ||
             !ncclIntruQueueEmpty(&planner->collCeTaskQueue) ||
             planner->nTasksRma != 0);

    struct ncclKernelPlan* planHead = ncclIntruQueueHead(&planner->planQueue);
    planner->unlaunchedPlansHead = planHead;

    if (nPlans == 0) return ncclSuccess;

    cudaStream_t launchStream = planner->streams->stream;
    cudaStream_t deviceStream, launchOrder;
    NCCLCHECKGOTO(ncclStrongStreamAcquire(planner->capturingGraph, &comm->sharedRes->deviceStream, /*concurrent=*/false, &deviceStream), result, failure);

    // userStream[0] waits on each userStream[i]...
    for (struct ncclCudaStreamList* l=planner->streams->next; l != nullptr; l = l->next) {
      CUDACHECKGOTO(cudaEventRecord(comm->sharedRes->scratchEvent, l->stream), result, failure);
      CUDACHECKGOTO(cudaStreamWaitEvent(launchStream, comm->sharedRes->scratchEvent, 0), result, failure);
    }
    // userStream[0] waits on deviceStream
    NCCLCHECKGOTO(ncclStreamWaitStream(launchStream, deviceStream, comm->sharedRes->scratchEvent), result, failure);

    bool capturing = ncclCudaGraphValid(planner->capturingGraph);
    enum ncclImplicitOrder implicitOrder;
    cudaError_t status = cudaSuccess;
    NCCLCHECKGOTO(getImplicitOrder(&implicitOrder, capturing), result, failure);

    if (implicitOrder != ncclImplicitOrderNone) {
      // userStream[0] waits on per-device (context) launchOrder. Concurrent strong stream access is
      // required if this is a graph capture, non-captured cannot be concurrent because that would violate
      // deterministic program order of launches.
      bool concurrent = capturing;
      NCCLCHECKGOTO(ncclStrongStreamAcquire(planner->capturingGraph, &comm->context->launchOrder, concurrent, &launchOrder), result, failure);
      NCCLCHECKGOTO(ncclStreamWaitStream(launchStream, launchOrder, comm->sharedRes->scratchEvent), result, failure);
    }

    if (!persistent && comm->sharedRes->persistentRefs) status = CUDACLEARERROR(cudaEventQuery(comm->sharedRes->hostStream.serialEvent));
    if (persistent || ncclCudaLaunchBlocking || status == cudaErrorNotReady) {
      // We have to launch host tasks to push proxy args. We are careful to only
      // do this if necessary since host tasks impose a high performance cost in CUDA.
      bool acquired = false;
      cudaStream_t hostStream;
      for (struct ncclKernelPlan* plan=planHead; plan != nullptr; plan = plan->next) {
        if (plan->hasProxyOps) {
          if (!acquired) {
            acquired = true;
            NCCLCHECKGOTO(ncclStrongStreamAcquire(planner->capturingGraph, &comm->sharedRes->hostStream, /*concurrent=*/false, &hostStream), result, failure);
          }
          plan->isHostCbEnq = true;
          CUDACHECKGOTO(cudaLaunchHostFunc(hostStream, hostStreamPlanCallback, plan), result, failure);
        }
      }
      if (acquired) {
        // Make to-be-launched kernels dependent on just-launched host stream tasks.
        NCCLCHECKGOTO(ncclStreamWaitStream(launchStream, hostStream, comm->sharedRes->scratchEvent), result, failure);
        NCCLCHECKGOTO(ncclStrongStreamRelease(planner->capturingGraph, &comm->sharedRes->hostStream, /*concurrent=*/false), result, failure);
      }
    }

    if (persistent) {
      comm->sharedRes->persistentRefs += nPlans;
      comm->localPersistentRefs += nPlans;
      NCCLCHECKGOTO(ncclCudaGraphAddDestructor(planner->capturingGraph, persistentDestructor, (void*)planHead), result, failure);
    }
  }
failure:
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
  ncclResult_t ret = ncclSuccess;
  struct ncclKernelPlanner* planner = &comm->planner;
  int nChannels = countOneBits(plan->channelMask);
  void* sym = plan->kernelFn;
  dim3 grid = {(unsigned)nChannels, 1, 1};
  dim3 block = {(unsigned)plan->threadPerBlock, 1, 1};
  int smem = ncclShmemDynamicSize(comm->cudaArch);
  cudaStream_t launchStream = planner->streams->stream;

  NCCLCHECK(ncclProfilerStartKernelLaunchEvent(plan, launchStream));

  void* extra[] = {
    CU_LAUNCH_PARAM_BUFFER_POINTER, plan->kernelArgs,
    CU_LAUNCH_PARAM_BUFFER_SIZE, &plan->kernelArgsSize,
    CU_LAUNCH_PARAM_END
  };

  int driverVersion;
  NCCLCHECKGOTO(ncclCudaDriverVersion(&driverVersion), ret, do_return);

  CUfunction fn;
  CUDACHECKGOTO(cudaGetFuncBySymbol(&fn, sym), ret, do_return);

  if (CUDART_VERSION >= 11080 && driverVersion >= 11080) {
  #if CUDART_VERSION >= 11080
    int compCap = comm->compCap;
    unsigned int clusterSize = (compCap >= 90) ? comm->config.cgaClusterSize : 0;

    CUlaunchConfig launchConfig = {0};
    CUlaunchAttribute launchAttrs[6] = {};
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
      launchAttrs[attrs].id = CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION;
      launchAttrs[attrs++].value.clusterDim = {clusterSize, 1, 1};
      launchAttrs[attrs].id = CU_LAUNCH_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE;
      launchAttrs[attrs++].value.clusterSchedulingPolicyPreference = CU_CLUSTER_SCHEDULING_POLICY_SPREAD;
    }
    #if CUDART_VERSION >= 12000
    if (compCap >= 90 && driverVersion >= 12000) {
      // Set the NCCL Mem Sync domain on CUDA 12.0 and later (sm90)
      launchAttrs[attrs].id = CU_LAUNCH_ATTRIBUTE_MEM_SYNC_DOMAIN;
      launchAttrs[attrs++].value.memSyncDomain = (CUlaunchMemSyncDomain) ncclParamMemSyncDomain();
    }
    #endif
    #if CUDART_VERSION >= 12030
    enum ncclImplicitOrder implicitOrder;
    NCCLCHECKGOTO(getImplicitOrder(&implicitOrder, plan->persistent, driverVersion), ret, do_return);
    if (implicitOrder == ncclImplicitOrderLaunch) {
      launchAttrs[attrs].id = CU_LAUNCH_ATTRIBUTE_LAUNCH_COMPLETION_EVENT;
      launchAttrs[attrs].value.launchCompletionEvent.event = comm->sharedRes->launchEvent;
      launchAttrs[attrs].value.launchCompletionEvent.flags = 0;
      attrs++;
    }
    if (plan->isSymColl && compCap >= 90 && driverVersion >= 12030) {
      launchAttrs[attrs].id = CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION;
      launchAttrs[attrs].value.programmaticStreamSerializationAllowed = 1;
      attrs++;
    }
    #endif
    #if CUDART_VERSION >= 13000
    if (compCap >= 100 && driverVersion >= 13000) {
      launchAttrs[attrs].id = CU_LAUNCH_ATTRIBUTE_NVLINK_UTIL_CENTRIC_SCHEDULING;
      launchAttrs[attrs].value.nvlinkUtilCentricScheduling = comm->config.nvlinkCentricSched;
      attrs++;
    }
    #endif
    launchConfig.gridDimX = grid.x;
    launchConfig.gridDimY = grid.y;
    launchConfig.gridDimZ = grid.z;
    launchConfig.blockDimX = block.x;
    launchConfig.blockDimY = block.y;
    launchConfig.blockDimZ = block.z;
    launchConfig.sharedMemBytes = smem;
    launchConfig.attrs = launchAttrs;
    launchConfig.numAttrs = attrs;
    launchConfig.hStream = launchStream;
    CUCHECKGOTO(cuLaunchKernelEx(&launchConfig, fn, nullptr, extra), ret, do_return);
  #endif
  } else {
    // Standard kernel launch
    CUCHECKGOTO(cuLaunchKernel(fn, grid.x, grid.y, grid.z, block.x, block.y, block.z, smem, launchStream, nullptr, extra), ret, do_return);
  }

do_return:
  NCCLCHECK(ncclProfilerStopKernelLaunchEvent(plan));
  return ret;
}

ncclResult_t ncclLaunchKernelAfter_NoCuda(struct ncclComm* comm, struct ncclKernelPlan* plan) {
  if (!plan->isHostCbEnq) {
    // we are not using the host stream for proxy ops and reclaimation submission, call
    // hostStreamPlanTask directly
    NCCLCHECK(hostStreamPlanTask(comm, plan));
  }
  return ncclSuccess;
}

namespace {
  struct KernelFinishCallback {
    struct ncclCommEventCallback base;
    uint32_t workFifoConsumed;
  };
  ncclResult_t KernelFinishCallback_fn(
      struct ncclComm* comm, struct ncclCommEventCallback* cb
    ) {
    struct KernelFinishCallback* me = (struct KernelFinishCallback*)cb;
    comm->workFifoConsumed = me->workFifoConsumed;
    CUDACHECK(cudaEventDestroy(me->base.event));
    free(me);
    return ncclSuccess;
  }
}

ncclResult_t ncclLaunchFinish(struct ncclComm* comm) {
  struct ncclKernelPlanner* planner = &comm->planner;
  if (!ncclIntruQueueEmpty(&planner->planQueue)) {
    // Reset queue to empty without destroying plans since those will be sent
    // back to us for reclaiming via callbackQueue.
    ncclIntruQueueConstruct(&planner->planQueue);

    cudaStream_t launchStream = planner->streams->stream; // First user stream gets launch
    cudaStream_t deviceStream, launchOrder;
    cudaEvent_t finishedEvent = comm->sharedRes->scratchEvent;
    CUDACHECK(cudaEventRecord(finishedEvent, launchStream));

    if (comm->workFifoProduced - comm->workFifoProducedLastRecorded > comm->workFifoBytes/8) {
      comm->workFifoProducedLastRecorded = comm->workFifoProduced;
      struct KernelFinishCallback* cb;
      NCCLCHECK(ncclCalloc(&cb, 1));
      cb->base.event = finishedEvent;
      cb->base.fn = KernelFinishCallback_fn;
      cb->workFifoConsumed = comm->workFifoProduced;
      ncclIntruQueueEnqueue(&comm->eventCallbackQueue, &cb->base);
      // We just stole scratchEvent so must create a new one.
      CUDACHECK(cudaEventCreateWithFlags(&comm->sharedRes->scratchEvent, cudaEventDisableTiming));
    }

    // deviceStream waits on userStream[0]
    NCCLCHECK(ncclStrongStreamAcquiredWorkStream(planner->capturingGraph, &comm->sharedRes->deviceStream, /*concurrent=*/false, &deviceStream));

    // We know that deviceStream is strictly behind the launchStream because launchStream
    // synced with it before kernel launch. This allows us to to see deviceStream waiting
    // on launchStream as a fast-forward. When building CUDA graphs fast forwards should
    // be handled specially so as not to create graphs with a blowup in the number of edges.
    // So we could do this:
    //   CUDACHECK(cudaStreamWaitEvent(deviceStream, finishedEvent, 0));
    // But instead we do:
    NCCLCHECK(ncclStreamAdvanceToEvent(planner->capturingGraph, deviceStream, finishedEvent));

    // Each userStream[i] waits on userStream[0]
    for (struct ncclCudaStreamList* l=planner->streams->next; l != nullptr; l = l->next) {
      CUDACHECK(cudaStreamWaitEvent(l->stream, finishedEvent, 0));
    }
    bool capturing = ncclCudaGraphValid(planner->capturingGraph);
    enum ncclImplicitOrder implicitOrder;
    NCCLCHECK(getImplicitOrder(&implicitOrder, capturing));
    if (implicitOrder != ncclImplicitOrderNone) {
      // As in ncclLaunchPrepare, strong stream can be non-concurrent when non-captured.
      bool concurrent = capturing;
      // Incorporate launch event into per-device (context) launch order.
      NCCLCHECK(ncclStrongStreamAcquiredWorkStream(planner->capturingGraph, &comm->context->launchOrder, concurrent, &launchOrder));
      // If we don't have launch events (requires CUDA 12.3) then just use completion event (serialize execution).
      CUDACHECK(cudaStreamWaitEvent(launchOrder, implicitOrder == ncclImplicitOrderLaunch ? comm->sharedRes->launchEvent : finishedEvent));
      // Release launchOrder as acquired in ncclLaunchPrepare()
      NCCLCHECK(ncclStrongStreamRelease(planner->capturingGraph, &comm->context->launchOrder, concurrent));
    }
    // Release deviceStream as acquired in ncclLaunchPrepare()
    NCCLCHECK(ncclStrongStreamRelease(planner->capturingGraph, &comm->sharedRes->deviceStream, /*concurrent=*/false));
  }
  return ncclSuccess;
}

/*****************************************************************************/
/* Enqueueing system : computation of kernel and proxy operations parameters */
/*****************************************************************************/

ncclResult_t ncclGetCollNetSupport(
    struct ncclComm* comm, struct ncclTaskColl* info, int* collNetSupport
  ) {
  // Translate ncclAvg and PreMulSum
  ncclRedOp_t netOp = info->opHost;
  if (info->opDev.op == ncclDevPreMulSum || info->opDev.op == ncclDevSumPostDiv) {
    netOp = ncclSum;
  }
  *collNetSupport = comm->config.collnetEnable;
  switch (info->func) {
  case ncclFuncAllReduce:
  case ncclFuncReduce:
  case ncclFuncReduceScatter:
    *collNetSupport &= comm->collNetSupportMatrix[netOp][info->datatype];
    break;
  default:
    break;
  }
  return ncclSuccess;
}

static void initCollCostTable(float** collCostTable) {
  float (*table)[NCCL_NUM_PROTOCOLS] = (float (*)[NCCL_NUM_PROTOCOLS])collCostTable;
  for (int a = 0; a < NCCL_NUM_ALGORITHMS; a++) {
    for (int p = 0; p < NCCL_NUM_PROTOCOLS; p++) {
      table[a][p] = NCCL_ALGO_PROTO_IGNORE;
    }
  }
}

// numPipeOps: number of pipelined ops. Can be greater than 1 in aggregation mode. Used to adjust latency.
static ncclResult_t updateCollCostTable(
    struct ncclComm* comm, struct ncclTaskColl* info, size_t nBytes,
    int collNetSupport, int nvlsSupport, int numPipeOps,
    float** collCostTable) {
  float (*table)[NCCL_NUM_PROTOCOLS] = (float (*)[NCCL_NUM_PROTOCOLS])collCostTable;

  if (comm->nRanks == 1) {
    table[NCCL_ALGO_RING][NCCL_PROTO_SIMPLE] = 0.0;
    return ncclSuccess;
  }

  for (int a=0; a<NCCL_NUM_ALGORITHMS; a++) {
    if ((a == NCCL_ALGO_COLLNET_DIRECT || a == NCCL_ALGO_COLLNET_CHAIN) && collNetSupport != 1) continue;
    // CollNetDirect is only supported for up to 8 local GPUs
    if (a == NCCL_ALGO_COLLNET_DIRECT && comm->maxLocalRanks > NCCL_MAX_DIRECT_ARITY+1) continue;
    // Disable CollNet Chain for more than 8 local GPUs
    if (a == NCCL_ALGO_COLLNET_CHAIN && comm->maxLocalRanks > NCCL_MAX_DIRECT_ARITY+1) continue;
    if ((a == NCCL_ALGO_NVLS || a == NCCL_ALGO_NVLS_TREE) && (!nvlsSupport || (info->func != ncclFuncAllReduce && comm->localRanks > NCCL_MAX_NVLS_ARITY))) continue;
    if (a == NCCL_ALGO_NVLS && collNetSupport != 1 && comm->nNodes > 1) continue;
    /* Tree reduceScatter doesn't support scaling yet */
    if (a == NCCL_ALGO_PAT && info->func == ncclFuncReduceScatter
        && (info->opDev.op == ncclDevPreMulSum || info->opDev.op == ncclDevSumPostDiv)) continue;
    for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
      NCCLCHECK(ncclTopoGetAlgoTime(comm, info->func, a, p, nBytes, numPipeOps, &table[a][p]));
      // Relegate fp8 reduction trees of sufficient depth that they incur precision loss
      // to be least preferred.
      if (info->datatype == ncclFloat8e4m3 || info->datatype == ncclFloat8e5m2) {
        if (a == NCCL_ALGO_RING && comm->nRanks > 8) {
          table[a][p] *= 1024.0; // Any factor large enough to act as a partition between lossy and non-lossy algos.
        }
      }
    }
  }

  return ncclSuccess;
}

static ncclResult_t topoGetAlgoInfo(
    struct ncclComm* comm, struct ncclTaskColl* info, size_t nBytes,
    float** collCostTable, ncclSimInfo_t* simInfo
  ) {
  float (*table)[NCCL_NUM_PROTOCOLS] = (float (*)[NCCL_NUM_PROTOCOLS])collCostTable;

  float minTime = FLT_MAX;
  int algorithm = info->algorithm = NCCL_ALGO_UNDEF;
  int protocol = info->protocol = NCCL_PROTO_UNDEF;
  for (int a=0; a<NCCL_NUM_ALGORITHMS; a++) {
    for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
      if (table[a][p] == NCCL_ALGO_PROTO_IGNORE) continue;
      if (table[a][p] >= 0.0 && table[a][p] < minTime) {
        algorithm = a;
        protocol = p;
        minTime = table[a][p];
      }
    }
  }

  info->algorithm = algorithm;
  info->protocol = protocol;
  float time = minTime;

  // Yes, we are first assigning and then testing if protocol is sane, but that's OK in this case.
  // coverity[check_after_sink]
  if (info->algorithm == NCCL_ALGO_UNDEF || info->protocol == NCCL_PROTO_UNDEF) {
    char ncclAlgoEnvStr[1024] = "";
    char ncclProtoEnvStr[1024] = "";
    const char* algoEnv = ncclGetEnv("NCCL_ALGO");
    if (algoEnv) {
      snprintf(ncclAlgoEnvStr, 1023, " NCCL_ALGO was set to %s.", algoEnv);
    }
    const char* protoEnv = ncclGetEnv("NCCL_PROTO");
    if (protoEnv) {
      snprintf(ncclProtoEnvStr, 1023, " NCCL_PROTO was set to %s.", protoEnv);
    }
    WARN("Error : no algorithm/protocol available for function %s with datatype %s.%s%s", ncclFuncToString(info->func), ncclDatatypeToString(info->datatype), ncclAlgoEnvStr, ncclProtoEnvStr);
    return (algoEnv || protoEnv) ? ncclInvalidUsage : ncclInternalError;
  }
  if (simInfo) simInfo->estimatedTime = time;
  TRACE(NCCL_COLL, "%ld Bytes -> Algo %d proto %d time %f", nBytes, info->algorithm, info->protocol, time);

  int nc = comm->nChannels;
  int nt = comm->maxThreads[info->algorithm][info->protocol];
  int threadThreshold = comm->threadThresholds[info->algorithm][info->protocol];
  if (info->algorithm == NCCL_ALGO_COLLNET_DIRECT) {
    // CollNet channel tuning
    int ncSwitch = 16;
    bool flag = true;
    while (ncSwitch >= 1 && flag) {
      while ((flag = nBytes < nc*nt*comm->channels[0].collnetDirect.nHeads*threadThreshold) && nc > ncSwitch) {
        if (nc == ncSwitch+ncSwitch/2) threadThreshold /= 2;
        nc--;
      }
      ncSwitch /= 2;
    }
  } else if (info->algorithm == NCCL_ALGO_NVLS || info->algorithm == NCCL_ALGO_NVLS_TREE) {
    // NVLS should not need more than 16 channels to get peak BW.
    if (comm->nNodes > 1 && info->algorithm == NCCL_ALGO_NVLS) {
      nc = std::min(comm->nvlsChannels, comm->nChannels);
    } else {
      nc = comm->nvlsChannels;
    }
  } else {
    // Ring/Tree channel tuning
    while (nBytes < nc * nt * threadThreshold) {
      if (nc >= 2) nc--;
      else break;
    }
  }

  if (info->algorithm != NCCL_ALGO_NVLS && info->algorithm != NCCL_ALGO_NVLS_TREE &&
    info->algorithm != NCCL_ALGO_COLLNET_DIRECT) {
    while (nBytes < nc * nt * threadThreshold) {
      if (nt % 128 == 0) nt /= 2;
      else break;
    }
  }
  if (info->protocol == NCCL_PROTO_SIMPLE) {
    if (info->algorithm == NCCL_ALGO_RING) nt += WARP_SIZE; // Extra warp for sync
    // More threads or sync warps needed due to split thread model
    if (info->algorithm == NCCL_ALGO_TREE) nt += 4*WARP_SIZE;
  }
  nt = nt/WARP_SIZE < 3 ? 3*WARP_SIZE : nt;
  if (info->algorithm == NCCL_ALGO_TREE) nt = NCCL_MAX_NTHREADS; // Tree now uses all threads always.
  if (info->algorithm == NCCL_ALGO_PAT) nt = NCCL_MAX_NTHREADS;
  info->nMaxChannels = nc;
  info->nWarps = nt/WARP_SIZE;
  return ncclSuccess;
}

// Use the default topo-based tuner if tuner plugin is not successful.
// Call the plugin first. Let it set algo+proto, and/or nChannels.
// Then, topoGetAlgoInfo will set algo/proto if not set, then nChannels and nThreads based on algo/proto.
// Finally, nChannels will be overriden by the plugin setting.
ncclResult_t ncclGetAlgoInfo(
    struct ncclComm* comm, struct ncclTaskColl* info,
    int collNetSupport, int nvlsSupport, int numPipeOps, ncclSimInfo_t* simInfo/* = NULL*/
  ) {
  size_t elementSize = ncclTypeSize(info->datatype);
  size_t nBytes = elementSize * ncclFuncMaxSendRecvCount(info->func, comm->nRanks, info->count);
  struct ncclReg* regSendBuf = NULL;
  struct ncclReg* regRecvBuf = NULL;
  int regBuff;
  bool isSendValid, isRecvValid;
  size_t sendbuffSize = elementSize * ncclFuncSendCount(info->func, comm->nRanks, info->count);
  size_t recvbuffSize = elementSize * ncclFuncRecvCount(info->func, comm->nRanks, info->count);
  info->algorithm = NCCL_ALGO_UNDEF;
  info->protocol = NCCL_PROTO_UNDEF;
  int nMaxChannels = 0;
  float collCostTable[NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS];
  initCollCostTable((float **)collCostTable);
  NCCLCHECK(updateCollCostTable(comm, info, nBytes, collNetSupport, nvlsSupport, numPipeOps, (float **)collCostTable));
  if (comm->tuner != NULL) {
    NCCLCHECK(ncclRegFind(comm, info->sendbuff, sendbuffSize, &regSendBuf));
    NCCLCHECK(ncclRegFind(comm, info->recvbuff, recvbuffSize, &regRecvBuf));
    NCCLCHECK(ncclRegLocalIsValid(regSendBuf, &isSendValid));
    NCCLCHECK(ncclRegLocalIsValid(regRecvBuf, &isRecvValid));
    regBuff = (regSendBuf && regRecvBuf && isSendValid && isRecvValid) || (ncclCudaGraphValid(comm->planner.capturingGraph) && ncclParamGraphRegister());
    NCCLCHECK(comm->tuner->getCollInfo(
          comm->tunerContext, info->func, nBytes,
          numPipeOps, (float **)collCostTable, NCCL_NUM_ALGORITHMS, NCCL_NUM_PROTOCOLS,
          regBuff, &nMaxChannels));
    NCCLCHECK(topoGetAlgoInfo(comm, info, nBytes, (float **)collCostTable, simInfo));
  } else {
    NCCLCHECK(topoGetAlgoInfo(comm, info, nBytes, (float **)collCostTable, simInfo));
    // NCCL_CTA_POLICY_EFFICIENCY requires user (non-symmetric) buffer registration (currently unsupported with MNNVL)
    if ((comm->config.CTAPolicy & NCCL_CTA_POLICY_EFFICIENCY) && ncclGetEnv("NCCL_ALGO") == NULL && ncclGetEnv("NCCL_PROTO") == NULL && !comm->MNNVL) {
      // make algorithm selection based on buffer registration
      // there can be other specialized policies for algorithms and protocols pickup in the future
      NCCLCHECK(ncclRegFind(comm, info->sendbuff, sendbuffSize, &regSendBuf));
      NCCLCHECK(ncclRegFind(comm, info->recvbuff, recvbuffSize, &regRecvBuf));
      NCCLCHECK(ncclRegLocalIsValid(regSendBuf, &isSendValid));
      NCCLCHECK(ncclRegLocalIsValid(regRecvBuf, &isRecvValid));
      regBuff = (regSendBuf && regRecvBuf && isSendValid && isRecvValid) || (ncclCudaGraphValid(comm->planner.capturingGraph) && ncclParamGraphRegister());
      if (regBuff && (info->func == ncclFuncAllGather || info->func == ncclFuncReduceScatter)) {
        if ((comm->nNodes > 1 && collNetSupport && nvlsSupport) || (comm->nNodes == 1 && nvlsSupport)) {
          int recChannels;
          NCCLCHECK(ncclNvlsRegResourcesQuery(comm, info, &recChannels));
          if (recChannels <= info->nMaxChannels) {
            info->algorithm = NCCL_ALGO_NVLS;
            info->protocol = NCCL_PROTO_SIMPLE;
            info->nMaxChannels = recChannels;
            info->nWarps = comm->maxThreads[info->algorithm][info->protocol] / WARP_SIZE;
          }
        }
      }
    }
  }

  info->nMaxChannels = nMaxChannels == 0 ? info->nMaxChannels : nMaxChannels;
  return ncclSuccess;
}

NCCL_PARAM(NvlsTreeMaxChunkSize, "NVLSTREE_MAX_CHUNKSIZE", -2);

static ncclResult_t calcCollChunking(
    struct ncclComm* comm, struct ncclTaskColl* info, int nChannels, size_t nBytes,
    /*outputs*/uint32_t* outChunkSize, uint32_t* outDirectFlags, struct ncclProxyOp* proxyOp
  ) {
  ncclPattern_t pattern;
  size_t grainSize = ncclProtoGrainSize(info->protocol);

  switch (info->func) {
  case ncclFuncBroadcast:
    pattern = info->algorithm == NCCL_ALGO_TREE ? ncclPatternTreeDown : ncclPatternPipelineFrom;
    break;
  case ncclFuncReduce:
    pattern = info->algorithm == NCCL_ALGO_TREE ? ncclPatternTreeUp : ncclPatternPipelineTo;
    break;
  case ncclFuncReduceScatter:
    pattern =
      info->algorithm == NCCL_ALGO_PAT ? ncclPatternPatUp :
      info->algorithm == NCCL_ALGO_NVLS ? ncclPatternNvls :
      info->algorithm == NCCL_ALGO_COLLNET_DIRECT ? ncclPatternCollnetDirect :
      ncclPatternRing;
    break;
  case ncclFuncAllGather:
    pattern =
      info->algorithm == NCCL_ALGO_PAT ? ncclPatternPatDown :
      info->algorithm == NCCL_ALGO_NVLS ? ncclPatternNvls :
      info->algorithm == NCCL_ALGO_COLLNET_DIRECT ? ncclPatternCollnetDirect :
      ncclPatternRing;
    break;
  case ncclFuncAllReduce:
    pattern =
      info->algorithm == NCCL_ALGO_NVLS ? ncclPatternNvls :
      info->algorithm == NCCL_ALGO_NVLS_TREE ? ncclPatternNvlsTree :
      info->algorithm == NCCL_ALGO_COLLNET_DIRECT ? ncclPatternCollnetDirect :
      info->algorithm == NCCL_ALGO_COLLNET_CHAIN ? ncclPatternCollnetChain :
      info->algorithm == NCCL_ALGO_TREE ? ncclPatternTreeUpDown :
      ncclPatternRingTwice;
    break;
  default:
    WARN("Unknown pattern for collective %d algorithm %d", info->func, info->algorithm);
    return ncclInternalError;
  }

  int nstepsPerLoop, nchunksPerLoop;
  size_t loopOffset = 0;
  int stepSize   = comm->buffSizes[info->protocol]/NCCL_STEPS;
  int chunkSteps = (info->protocol == NCCL_PROTO_SIMPLE && info->algorithm == NCCL_ALGO_RING) ? info->chunkSteps : 1;
  int sliceSteps = (info->protocol == NCCL_PROTO_SIMPLE && info->algorithm == NCCL_ALGO_RING) ? info->sliceSteps : 1;
  int chunkSize = stepSize*chunkSteps;
  if (info->protocol == NCCL_PROTO_LL) chunkSize /= 2;
  if (info->protocol == NCCL_PROTO_LL128) chunkSize = (chunkSize / NCCL_LL128_LINEELEMS) * NCCL_LL128_DATAELEMS;

  if (info->algorithm == NCCL_ALGO_COLLNET_DIRECT) {
    // Optimize chunkSize / nSteps
    while (nBytes / (nChannels * comm->channels[0].collnetDirect.nHeads * chunkSize) < comm->channels[0].collnetDirect.depth * 64 && chunkSize > 131072) chunkSize /= 2;
    while (nBytes / (nChannels * comm->channels[0].collnetDirect.nHeads * chunkSize) < comm->channels[0].collnetDirect.depth * 8 && chunkSize > 65536) chunkSize /= 2;
    while (nBytes / (nChannels * comm->channels[0].collnetDirect.nHeads * chunkSize) < comm->channels[0].collnetDirect.depth * 8 && chunkSize > 32768) chunkSize /= 2;
  } else if (info->algorithm == NCCL_ALGO_COLLNET_CHAIN) {
    stepSize = comm->buffSizes[NCCL_PROTO_SIMPLE] / NCCL_STEPS;
    chunkSize = std::min(256 * 1024, stepSize * chunkSteps);
    while (nBytes / (nChannels * chunkSize) < comm->channels[0].collnetChain.depth * 64 && chunkSize > 131072) chunkSize /= 2;
    while (nBytes / (nChannels * chunkSize) < comm->channels[0].collnetChain.depth * 8 && chunkSize > 65536) chunkSize /= 2;
    while (nBytes / (nChannels * chunkSize) < comm->channels[0].collnetChain.depth && chunkSize > 32768) chunkSize /= 2;
  } else if (info->algorithm == NCCL_ALGO_NVLS) {
    if ((info->regBufType & NCCL_NVLS_REG_BUFFER) && (info->func == ncclFuncAllGather || info->func == ncclFuncReduceScatter)) {
      chunkSize = comm->buffSizes[NCCL_PROTO_SIMPLE] / NCCL_STEPS;
    } else {
      int maxChunkSize = comm->nvlsChunkSize;
      if (comm->nNodes > 1 && comm->bandwidths[ncclFuncAllReduce][NCCL_ALGO_NVLS][NCCL_PROTO_SIMPLE] < 150) maxChunkSize = 32768;
      if (chunkSize > maxChunkSize) chunkSize = maxChunkSize;
      // Use uint64_t so that concurrentOps*chunkSize*X does not overflow.
      // However, nChannels * comm->channels[0].nvls.nHeads should easily fit in 32 bits.
      // coverity[overflow_before_widen]
      uint64_t concurrentOps = nChannels * comm->channels[0].nvls.nHeads;
      if ((nBytes < (64 * (concurrentOps * chunkSize))) && (chunkSize > 65536)) chunkSize = 65536;
      if ((nBytes < (8 * (concurrentOps * chunkSize))) && (chunkSize > 32768)) chunkSize = 32768;
      if ((nBytes < (2 * (concurrentOps * chunkSize))) && (chunkSize > 16384)) chunkSize = 16384;
    }
  } else if (info->algorithm == NCCL_ALGO_NVLS_TREE) {
    // Use uint64_t so that concurrentOps*chunkSize*X does not overflow.
    // However, nChannels * comm->channels[0].nvls.nHeads should easily fit in 32 bits.
    // coverity[overflow_before_widen]
    uint64_t concurrentOps = nChannels * comm->channels[0].nvls.nHeads;
    chunkSize = comm->nvlsChunkSize;
    int maxChunkSize = (int)ncclParamNvlsTreeMaxChunkSize();
    if (maxChunkSize == -2) maxChunkSize = comm->nNodes >= 4 ? 65536 : chunkSize;
    chunkSize = std::min(chunkSize, maxChunkSize);
    if ((nBytes < (32 * (concurrentOps * chunkSize))) && (chunkSize > 262144)) chunkSize = 262144;
    if ((nBytes < (16 * (concurrentOps * chunkSize))) && (chunkSize > 131072)) chunkSize = 131072;
    if ((nBytes < (4 * (concurrentOps * chunkSize))) && (chunkSize > 65536)) chunkSize = 65536;
    if ((nBytes < (1 * (concurrentOps * chunkSize))) && (chunkSize > 32768)) chunkSize = 32768;
  } else if (info->algorithm == NCCL_ALGO_TREE && info->protocol == NCCL_PROTO_LL128) {
    int nNodes = comm->nNodes;
    float ppn = comm->nRanks / (float)nNodes;
    float nstepsLL128 = 1+log2i(nNodes) + 0.1*ppn;
    // Yes, we are OK with the division on the left side of the < operand being integer.
    // coverity[integer_division]
    while (nBytes / (nChannels*chunkSize) < nstepsLL128*64/ppn && chunkSize > 131072) chunkSize /= 2;
    // coverity[integer_division]
    while (nBytes / (nChannels*chunkSize) < nstepsLL128*16/ppn && chunkSize > 32768) chunkSize /= 2;
  } else if (info->func == ncclFuncAllGather && info->algorithm == NCCL_ALGO_PAT) {
    while (chunkSize*nChannels*32 > nBytes && chunkSize > 65536) chunkSize /= 2;
  } else if (info->func == ncclFuncReduceScatter && info->algorithm == NCCL_ALGO_PAT) {
    while (chunkSize*nChannels*16 > nBytes && chunkSize > 65536) chunkSize /= 2;
  }

  // Compute directFlags of work struct.
  if (info->algorithm == NCCL_ALGO_COLLNET_DIRECT) {
    *outDirectFlags = NCCL_P2P_WRITE;
  } else {
    *outDirectFlags = 0;
  }

  // Compute nSteps for proxies
  chunkSize = chunkSize / grainSize * grainSize; // align chunkSize to multiple grainSize
  switch (pattern) {
  case ncclPatternTreeUp:
  case ncclPatternTreeDown:
  case ncclPatternTreeUpDown:
  case ncclPatternPatUp:
  case ncclPatternPatDown:
  case ncclPatternPipelineFrom:
  case ncclPatternPipelineTo:
  case ncclPatternCollnetChain:
    nstepsPerLoop = nchunksPerLoop = 1;
    break;
  case ncclPatternNvls:
    nstepsPerLoop = 1; nchunksPerLoop = comm->channels[0].nvls.nHeads;
    loopOffset = nChannels * chunkSize * comm->channels[0].nvls.headRank;
    break;
  case ncclPatternCollnetDirect:
    nstepsPerLoop = 1; nchunksPerLoop = comm->channels[0].collnetDirect.nHeads;
    loopOffset = nChannels * chunkSize * comm->channels[0].collnetDirect.headRank;
    break;
  case ncclPatternRing:
    nstepsPerLoop = comm->nRanks-1; nchunksPerLoop = comm->nRanks;
    break;
  case ncclPatternRingTwice:
    nstepsPerLoop = 2*(comm->nRanks-1); nchunksPerLoop = comm->nRanks;
    break;
  case ncclPatternNvlsTree:
    nstepsPerLoop = 1; nchunksPerLoop = comm->channels[0].nvls.nHeads;
    break;
  default:
    WARN("Unknown pattern %d", pattern);
    return ncclInternalError;
  }

  // Compute nSteps for proxies
  size_t loopSize = size_t(nChannels)*nchunksPerLoop*chunkSize;
  int nLoops = (int)DIVUP(nBytes, loopSize);
  memset(proxyOp, 0, sizeof(*proxyOp));
  proxyOp->nsteps = nstepsPerLoop * nLoops * chunkSteps;
  proxyOp->sliceSteps = sliceSteps;
  proxyOp->chunkSteps = chunkSteps;
  proxyOp->chunkSize = chunkSize;
  proxyOp->sliceSize = chunkSize / chunkSteps * sliceSteps;
  proxyOp->loopSize = loopSize;
  proxyOp->loopOffset = loopOffset;
  proxyOp->protocol = info->protocol;
  proxyOp->dtype = info->datatype;
  proxyOp->algorithm = info->algorithm;
  if (info->opDev.op == ncclDevPreMulSum || info->opDev.op == ncclDevSumPostDiv) {
    proxyOp->redOp = ncclSum; // Network sees avg as sum
  } else {
    proxyOp->redOp = info->opHost;
  }
  proxyOp->pattern = pattern;
  proxyOp->coll = info->func;
  proxyOp->collAPI = info->func;
  proxyOp->root = info->root;
  proxyOp->isOneRPN = comm->isOneRPN;
  // This is used by P2P to reduce the receive buffer size. We don't use it in collectives
  // because some protocols need to transmit more than the total size, plus they sometimes
  // round up
  proxyOp->nbytes = stepSize*sliceSteps;

  if (info->regBufType & NCCL_NET_REG_BUFFER) {
    proxyOp->reg = 1;
    if (info->algorithm == NCCL_ALGO_COLLNET_DIRECT || info->algorithm == NCCL_ALGO_NVLS || info->algorithm == NCCL_ALGO_COLLNET_CHAIN) {
      if (proxyOp->isOneRPN) {
        proxyOp->nsteps = 1;
        proxyOp->loopOffset = 0;
        proxyOp->sendbuff = (uint8_t*)info->sendbuff;
        proxyOp->sendMhandle = info->sendMhandle;
      } else {
        if (info->func == ncclFuncAllGather || info->func == ncclFuncReduceScatter) {
          proxyOp->nbytes = nBytes / nchunksPerLoop;
          proxyOp->loopSize = proxyOp->loopSize / nchunksPerLoop;
          proxyOp->loopOffset = 0;
          if (info->func == ncclFuncAllGather) {
            proxyOp->sendbuff = (uint8_t*)info->sendbuff;
            proxyOp->sendMhandle = info->sendMhandle;
          }
        } else {
          proxyOp->sendbuff = (uint8_t*)info->recvbuff;
          proxyOp->sendMhandle = info->recvMhandle;
        }
      }
    } else if (info->algorithm == NCCL_ALGO_RING) {
      if (proxyOp->isOneRPN && info->func == ncclFuncAllGather) {
        proxyOp->chunkSize = NCCL_MAX_NET_SIZE;
        proxyOp->sliceSize = NCCL_MAX_NET_SIZE;
        proxyOp->chunkSteps = 1;
        proxyOp->sliceSteps = 1;
        proxyOp->loopSize = size_t(nChannels) * nchunksPerLoop * proxyOp->chunkSize;
        proxyOp->nsteps = DIVUP(nBytes, proxyOp->loopSize) * nstepsPerLoop;
        proxyOp->loopOffset = 0;
      }
    } else {
      WARN("Net registration invalid algorithm %s", ncclAlgoToString(info->algorithm));
      return ncclInternalError;
    }

    proxyOp->recvMhandle = info->recvMhandle;
    proxyOp->recvbuff = (uint8_t*)info->recvbuff;
    proxyOp->nbytes = nBytes;
  } else {
    proxyOp->reg = 0;
  }

  if (pattern == ncclPatternCollnetDirect || pattern == ncclPatternNvls) {
    proxyOp->specifics.collnetDirect.nNodes = comm->nNodes;
    proxyOp->specifics.collnetDirect.node = comm->node;
    if (info->func == ncclFuncAllGather || info->func == ncclFuncReduceScatter) {
      proxyOp->specifics.collnetDirect.sizePerRank = info->count*ncclTypeSize(info->datatype);
    }
  }

  if (pattern == ncclPatternPatUp || pattern == ncclPatternPatDown) {
    proxyOp->nbytes = DIVUP(nBytes, nChannels);
  }

  // Set peer count hints used by network plugin
  switch (proxyOp->pattern) {
  case ncclPatternRing:
  case ncclPatternRingTwice:
  case ncclPatternPipelineFrom:
  case ncclPatternPipelineTo:
  case ncclPatternPatUp:
  case ncclPatternPatDown:
    proxyOp->nPeers = 1;
    break;
  case ncclPatternTreeUp:
  case ncclPatternTreeDown:
  case ncclPatternTreeUpDown:
  case ncclPatternNvlsTree:
    proxyOp->nPeers = (NCCL_MAX_TREE_ARITY - 1) * 2;
    break;
  case ncclPatternCollnetChain:
  case ncclPatternCollnetDirect:
  case ncclPatternNvls:
  case ncclPatternProfiler:
    // Peer count hints unused
    break;
  case ncclPatternSend:
  case ncclPatternRecv:
  default:
    WARN("Unknown pattern %d", pattern);
    return ncclInternalError;
  }

  *outChunkSize = proxyOp->chunkSize;
  return ncclSuccess;
}

static ncclResult_t hostToDevRedOp(
    ncclDevRedOpFull *opFull, ncclRedOp_t op, ncclDataType_t datatype, ncclComm *comm
  ) {
  union {
    int8_t   i8; uint8_t   u8;
    int32_t i32; uint32_t u32;
    int64_t i64; uint64_t u64;
    __half f16; float f32; double f64;
    #if defined(__CUDA_BF16_TYPES_EXIST__)
      __nv_bfloat16 bf16;
    #endif
    #if defined(__CUDA_FP8_TYPES_EXIST__)
      __nv_fp8_storage_t f8;
    #endif
    void *ptr;
  };
  u64 = 0;
  opFull->scalarArgIsPtr = false;
  opFull->proxyOp = op;

  int nbits = 8*ncclTypeSize(datatype);
  if (nbits <= 0) return ncclInvalidArgument;
  uint64_t allBits = uint64_t(-1)>>(64-nbits);
  uint64_t signBit = allBits^(allBits>>1);
  bool datatype_signed = false;

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
      datatype_signed = true;
      // no break, we want to fall through...
    case ncclUint8: case ncclUint32: case ncclUint64:
      opFull->op = ncclDevSumPostDiv;
      u64 = comm->nRanks<<1 | datatype_signed;
      break;
    #if defined(__CUDA_FP8_TYPES_EXIST__)
    case ncclFloat8e4m3:
      opFull->op = ncclDevPreMulSum;
      f8 = __nv_cvt_float_to_fp8(float(1.0/comm->nRanks), __NV_SATFINITE, __NV_E4M3);
      break;
    case ncclFloat8e5m2:
      opFull->op = ncclDevPreMulSum;
      f8 = __nv_cvt_float_to_fp8(float(1.0/comm->nRanks), __NV_SATFINITE, __NV_E5M2);
      break;
    #endif
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

static ncclResult_t ncclPlannerSetCapturingGraph(struct ncclComm* comm, struct ncclInfo* info) {
  struct ncclKernelPlanner *planner = &comm->planner;
  if (info->stream != planner->streamRecent || planner->streams == nullptr) {
    planner->streamRecent = info->stream;
    struct ncclCudaStreamList* l = planner->streams;
    while (true) {
      if (l == nullptr) { // Got to the end, this must be a new stream.
        struct ncclCudaGraph graph;
        NCCLCHECK(ncclCudaGetCapturingGraph(&graph, info->stream, comm->config.graphUsageMode));
        if (planner->streams != nullptr && !ncclCudaGraphSame(planner->capturingGraph, graph)) {
          WARN("Streams given to a communicator within a NCCL group must either be all uncaptured or all captured by the same graph.");
          return ncclInvalidUsage;
        }
        planner->capturingGraph = graph; // C++ struct assignment
        // Add stream to list
        l = ncclMemoryStackAlloc<struct ncclCudaStreamList>(&comm->memScoped);
        l->stream = info->stream;
        l->next = planner->streams;
        planner->streams = l;
        break;
      }
      if (l->stream == info->stream)
        break; // Already seen stream.
      l = l->next;
    }
  }
  return ncclSuccess;
}

static ncclResult_t p2pTaskAppend(
    struct ncclComm* comm,
    struct ncclInfo* info,
    ncclFunc_t coll,
    ncclFunc_t collAPI,
    void* buff,
    size_t count,
    ncclDataType_t datatype,
    int peer, bool allowUB) {
  struct ncclKernelPlanner *planner = &comm->planner;

  // Determine peer and basic parameters.
  ssize_t nBytes = count*ncclTypeSize(datatype);
  bool isSendNotRecv = coll == ncclFuncSend;

  // Must be in thread local group before tasks can be alloc'd in `comm->memScoped`.
  ncclGroupCommJoin(comm, ncclGroupTaskTypeCollective);
  info->coll = coll;
  // Set capturing graph. Called here so that profiler can emit a group API event with this information
  NCCLCHECK(ncclPlannerSetCapturingGraph(comm, info));
  bool isGraphCaptured = ncclCudaGraphValid(planner->capturingGraph);
  NCCLCHECK(ncclProfilerStartGroupApiEvent(info, isGraphCaptured));
  NCCLCHECK(ncclProfilerRecordGroupApiEventState(ncclProfilerGroupStartApiStop));

  NCCLCHECK(ncclProfilerStartP2pApiEvent(info, isGraphCaptured));

  struct ncclTaskP2p* p2p = ncclMemoryPoolAlloc<struct ncclTaskP2p>(&comm->memPool_ncclTaskP2p, &comm->memPermanent);
  p2p->func = coll;
  p2p->collAPI = collAPI;
  p2p->buff = buff;
  p2p->count = count;
  p2p->datatype = datatype;
  p2p->root = peer;
  p2p->bytes = nBytes;
  p2p->allowUB = allowUB;
  p2p->eActivationMask = ncclProfilerApiState.eActivationMask;
  p2p->groupApiEventHandle = ncclProfilerApiState.groupApiEventHandle;
  p2p->p2pApiEventHandle = ncclProfilerApiState.p2pApiEventHandle;
  ncclIntruQueueEnqueue(
    isSendNotRecv ? &planner->peers[peer].sendQueue : &planner->peers[peer].recvQueue,
    p2p);
  planner->nTasksP2p += 1;
  if (isSendNotRecv)
    planner->nTasksP2pSend += 1;
  else
    planner->nTasksP2pRecv += 1;

  // Mark channels that need pre-connect
  if (comm->rank != peer) {
    if (!(isSendNotRecv ? planner->peers[peer].sendSeen : planner->peers[peer].recvSeen)) {
      // planner->peers[peer].send/recvSeen is private to each comm, so we need to set it anyway.
      (isSendNotRecv ? planner->peers[peer].sendSeen : planner->peers[peer].recvSeen) = true;
      int round = 0;
      while (peer != (isSendNotRecv ? comm->p2pSchedule[round].sendRank
                                    : comm->p2pSchedule[round].recvRank)) {
        round += 1;
      }
      uint8_t base = ncclP2pChannelBaseForRound(comm, round);
      for (int c=0; c < comm->p2pnChannelsPerPeer; c++) {
        int channelId = ncclP2pChannelForPart(comm->p2pnChannels, base, c);
        if (isSendNotRecv) {
          if (comm->channels[channelId].peers[peer]->send[1].hasSeen == 0) { // P2P uses only 1 connector
            // the send/recv connector is shared among split shared comms. We need to set hasSeen to
            // 1 in order to avoid duplicate connection setup if user group sendrecv ops with split
            // shared comms together.
            comm->channels[channelId].peers[peer]->send[1].hasSeen = 1;
            comm->channels[channelId].peers[peer]->send[1].p2pOnly = 1;
            comm->connectSend[peer] |= (1UL<<channelId);
            ncclGroupCommPreconnect(comm);
          }
        } else {
          if (comm->channels[channelId].peers[peer]->recv[1].hasSeen == 0) { // P2P uses only 1 connector
            comm->channels[channelId].peers[peer]->recv[1].hasSeen = 1;
            comm->channels[channelId].peers[peer]->recv[1].p2pOnly = 1;
            comm->connectRecv[peer] |= (1UL<<channelId);
            ncclGroupCommPreconnect(comm);
          }
        }
      }
    }
  }
  ncclProfilerStopP2pApiEvent();
  return ncclSuccess;
}

static ncclResult_t collTaskAppend(
    struct ncclComm* comm,
    struct ncclInfo* info,
    struct ncclDevRedOpFull opDev) {
  struct ncclKernelPlanner *planner = &comm->planner;

  // Must be in thread local group before tasks can be alloc'd in `comm->memScoped`.
  ncclGroupCommJoin(info->comm, ncclGroupTaskTypeCollective);
  // Set capturing graph. Called here so that profiler can emit a group API event with this information
  NCCLCHECK(ncclPlannerSetCapturingGraph(comm, info));

  bool isGraphCaptured = ncclCudaGraphValid(planner->capturingGraph);
  NCCLCHECK(ncclProfilerStartGroupApiEvent(info, isGraphCaptured));
  NCCLCHECK(ncclProfilerRecordGroupApiEventState(ncclProfilerGroupStartApiStop));
  NCCLCHECK(ncclProfilerStartCollApiEvent(info, isGraphCaptured));

  if (info->coll == ncclFuncBroadcast && ncclParamAllgathervEnable() && !comm->ccEnable) {
    // Must be in thread local group before tasks can be alloc'd in `comm->memScoped`.
    struct ncclTaskBcast* t = ncclMemoryPoolAlloc<struct ncclTaskBcast>(&comm->memPool_ncclTaskBcast, &comm->memPermanent);
    t->func = ncclFuncAllGatherV;
    t->sendbuff = info->sendbuff;
    t->recvbuff = info->recvbuff;
    t->count = info->count * ncclTypeSize(info->datatype);
    t->datatype = ncclInt8;
    t->root = info->root;

    // update bcast min/max peer
    planner->bcast_info.minBcastPeer = std::min(planner->bcast_info.minBcastPeer, info->root);
    planner->bcast_info.maxBcastPeer = std::max(planner->bcast_info.maxBcastPeer, info->root);
    if (ncclIntruQueueEmpty(&planner->peers[info->root].bcastQueue)) {
      planner->bcast_info.BcastPeers += 1;
    }

    // enqueue to peer's bcast queue instead of collSorter
    ncclIntruQueueEnqueue(&planner->peers[info->root].bcastQueue, t);
    planner->nTasksBcast += 1;
  }
  else {
  struct ncclTaskColl* t = ncclMemoryPoolAlloc<struct ncclTaskColl>(&comm->memPool_ncclTaskColl, &comm->memPermanent);
  t->func = info->coll;
  t->sendbuff = info->sendbuff;
  t->recvbuff = info->recvbuff;
  t->count = info->count;
  t->root = info->root;
  t->datatype = info->datatype;
  size_t elementSize = ncclTypeSize(t->datatype);
  if (t->func == ncclFuncAllGather || t->func == ncclFuncBroadcast) {
    t->count *= elementSize;
    t->datatype = ncclInt8;
    elementSize = 1;
  }
  t->trafficBytes = t->count*elementSize*ncclFuncTrafficPerByte(t->func, comm->nRanks);
  t->opHost = info->op;
  t->opDev = opDev; // C++ struct assignment
  t->chunkSteps = info->chunkSteps;
  t->sliceSteps = info->sliceSteps;
  t->eActivationMask = ncclProfilerApiState.eActivationMask;
  t->groupApiEventHandle = ncclProfilerApiState.groupApiEventHandle;
  t->collApiEventHandle = ncclProfilerApiState.collApiEventHandle;

  planner->nTasksColl += 1;
  ncclTaskCollSorterInsert(&planner->collSorter, t, t->trafficBytes);
  }
  ncclProfilerStopCollApiEvent();
  return ncclSuccess;
}

static ncclResult_t ceCollTaskAppend(
    struct ncclComm* comm,
    struct ncclInfo* info,
    struct ncclDevrWindow* sendWin,
    struct ncclDevrWindow* recvWin,
    struct ncclDevRedOpFull opDev) {
  struct ncclKernelPlanner *planner = &comm->planner;

  // Check if CE needs initialization
  if (comm->ceColl.baseUCSymReadyPtr == NULL && ncclIntruQueueEmpty(&comm->ceInitTaskQueue)) {
    struct ncclCeInitTask* ceTask;
    NCCLCHECK(ncclCalloc(&ceTask, 1));
    ceTask->comm = comm;
    ncclIntruQueueEnqueue(&comm->ceInitTaskQueue, ceTask);
    ncclGroupCommJoin(comm, ncclGroupTaskTypeSymRegister);
  }

  // Must be in thread local group before tasks can be alloc'd in `comm->memScoped`.
  ncclGroupCommJoin(info->comm, ncclGroupTaskTypeCollective);
  // Set capturing graph. Called here so that profiler can emit a group API event with this information
  NCCLCHECK(ncclPlannerSetCapturingGraph(comm, info));
  bool isGraphCaptured = ncclCudaGraphValid(planner->capturingGraph);
  NCCLCHECK(ncclProfilerStartGroupApiEvent(info, isGraphCaptured));
  NCCLCHECK(ncclProfilerRecordGroupApiEventState(ncclProfilerGroupStartApiStop));
  NCCLCHECK(ncclProfilerStartCollApiEvent(info, isGraphCaptured));

  struct ncclTaskColl* t = ncclMemoryPoolAlloc<struct ncclTaskColl>(&comm->memPool_ncclTaskColl, &comm->memPermanent);

  t->func = info->coll;
  t->sendbuff = info->sendbuff;
  t->recvbuff = info->recvbuff;
  t->count = info->count;
  t->root = info->root;
  t->datatype = info->datatype;
  size_t elementSize = ncclTypeSize(t->datatype);
  if (t->func == ncclFuncAllGather || t->func == ncclFuncBroadcast) {
    t->count *= elementSize;
    t->datatype = ncclInt8;
    elementSize = 1;
  }
  t->trafficBytes = t->count*elementSize*ncclFuncTrafficPerByte(t->func, comm->nRanks);
  t->opHost = info->op;
  t->opDev = opDev; // C++ struct assignment
  t->chunkSteps = info->chunkSteps;
  t->sliceSteps = info->sliceSteps;
  t->eActivationMask = COMPILER_ATOMIC_LOAD(&ncclProfilerEventMask, std::memory_order_relaxed);
  t->groupApiEventHandle = ncclProfilerApiState.groupApiEventHandle;
  t->collApiEventHandle = ncclProfilerApiState.collApiEventHandle;
  t->sendWin = sendWin;
  t->recvWin = recvWin;

  ncclIntruQueueEnqueue(&planner->collCeTaskQueue, t);

  ncclProfilerStopCollApiEvent();
  return ncclSuccess;
}

static ncclResult_t rmaTaskAppend(
  struct ncclComm* comm,
  struct ncclInfo* info) {
  struct ncclKernelPlanner *planner = &comm->planner;

  void const* srcBuff = info->sendbuff;

  if (!comm->symmetricSupport){
    WARN("One sided RMA: symmetric registration is not supported in this communicator.");
    return ncclInvalidArgument;
  }

  if (!comm->rmaProxySupport && comm->nNodes > 1) {
    WARN("One sided RMA: RMA proxy is not supported in this communicator.");
    return ncclInvalidArgument;
  }

  // Check if context is valid (must be 0 for now)
  if (info->ctx != 0) {
    WARN("Context %d is invalid (must be 0)", info->ctx);
    return ncclInvalidArgument;
  }

  // Check if signal index is valid (must be 0 for now)
  if (info->sigIdx != 0) {
    WARN("Signal index %d is invalid (must be 0)", info->sigIdx);
    return ncclInvalidArgument;
  }

  // Check if flags is valid
  if (info->flags != 0) {
    WARN("Flags %u is invalid (must be 0)", info->flags);
    return ncclInvalidArgument;
  }

  // Initialize window pointers - only needed for Put and Signal
  struct ncclDevrWindow* peerWinHost = NULL;
  struct ncclDevrWindow* srcWinHost = NULL;
  size_t srcWinOffset = 0;

  if (info->coll == ncclFuncPutSignal) {
    // Validate peer window with detailed debugging
    if (info->peerWin == NULL) {
      WARN("ncclPutSignal: peerWin is NULL");
      return ncclInvalidArgument;
    }

    struct ncclWindow_vidmem* peerWinDevHost = NULL;
    NCCLCHECK(ncclShadowPoolToHost(&comm->devrState.shadows, info->peerWin, &peerWinDevHost));
    peerWinHost = (struct ncclDevrWindow*)peerWinDevHost->winHost;

    // Validate source buffer and window
    if (srcBuff == NULL) {
      WARN("ncclPutSignal: srcBuff is NULL");
      return ncclInvalidArgument;
    }
    NCCLCHECK(ncclDevrFindWindow(comm, srcBuff, &srcWinHost));
    if (srcWinHost == NULL || !(srcWinHost->winFlags & NCCL_WIN_COLL_SYMMETRIC)) {
      WARN("ncclPutSignal: srcWinHost is not in a valid symmetric window");
      return ncclInvalidArgument;
    }
    srcWinOffset = (char*)srcBuff - (char*)srcWinHost->userPtr;
  }
  else if (info->coll == ncclFuncSignal) {
    // Check if count is valid
    if (info->count != 0) {
      WARN("ncclSignal: count must be 0");
      return ncclInvalidArgument;
    }
  }
  else if (info->coll == ncclFuncWaitSignal) {
    // Check if signalDescs is valid
    if (info->signalDescs == NULL || info->nDesc == 0) {
      WARN("ncclWaitSignal: invalid arguments");
      return ncclInvalidArgument;
    }
    // Validate each descriptor
    for (int i = 0; i < info->nDesc; i++) {
      if (info->signalDescs[i].opCnt <= 0) {
        WARN("ncclWaitSignal: descriptor %d has invalid opCnt %d", i, info->signalDescs[i].opCnt);
        return ncclInvalidArgument;
      }
      if (info->signalDescs[i].sigIdx != 0) {
        WARN("ncclWaitSignal: descriptor %d has invalid sigIdx %d (must be 0)", i, info->signalDescs[i].sigIdx);
        return ncclInvalidArgument;
      }
      if (info->signalDescs[i].ctx != 0) {
        WARN("ncclWaitSignal: descriptor %d has invalid context %d (must be 0)",
             i, info->signalDescs[i].ctx);
        return ncclInvalidArgument;
      }
    }
  }

  // Check if RMA CE needs initialization
  if (!comm->rmaState.rmaCeState.initialized && ncclIntruQueueEmpty(&comm->rmaCeInitTaskQueue)) {
    struct ncclRmaCeInitTask* ceTask;
    NCCLCHECK(ncclCalloc(&ceTask, 1));
    ceTask->comm = comm;
    ncclIntruQueueEnqueue(&comm->rmaCeInitTaskQueue, ceTask);
    ncclGroupCommJoin(comm, ncclGroupTaskTypeSymRegister);
  }

  // Must be in thread local group before tasks can be alloc'd in `comm->memScoped`.
  ncclGroupCommJoin(info->comm, ncclGroupTaskTypeCollective);
  NCCLCHECK(ncclPlannerSetCapturingGraph(comm, info));


  // Handle WaitSignal separately
  if (info->coll == ncclFuncWaitSignal) {
    struct ncclTaskRma* t = ncclMemoryPoolAlloc<struct ncclTaskRma>(&comm->memPool_ncclTaskRma, &comm->memPermanent);

    t->func = ncclFuncWaitSignal;
    t->ctx = 0;
    t->count = 0;
    t->bytes = 0;
    t->srcBuff = NULL;
    t->srcWinOffset = 0;
    t->srcWinHost = NULL;
    t->peer = 0;
    t->peerWinOffset = 0;
    t->peerWinHost = NULL;
    t->signalMode = NCCL_SIGNAL;

    // Convert descriptors to peers and nsignals arrays
    t->npeers = info->nDesc;
    t->peers = ncclMemoryStackAlloc<int>(&comm->memScoped, info->nDesc);
    t->nsignals = ncclMemoryStackAlloc<int>(&comm->memScoped, info->nDesc);

    for (int i = 0; i < info->nDesc; i++) {
      t->peers[i] = info->signalDescs[i].peer;
      t->nsignals[i] = info->signalDescs[i].opCnt;
    }

    t->eActivationMask = __atomic_load_n(&ncclProfilerEventMask, __ATOMIC_RELAXED);
    planner->nTasksRma++;
    ncclIntruQueueEnqueue(&planner->rmaTaskQueues[t->ctx], t);

  } else if (info->coll == ncclFuncPutSignal || info->coll == ncclFuncSignal) {

    // Calculate total bytes for the operation
    size_t totalBytes = info->count * ncclTypeSize(info->datatype);

    // Define 1GB chunk size for splitting large put operations
    const size_t chunkSize = 1ULL << 30; // 1GB = 1073741824 bytes

    // Determine if we need to split the operation
    int numChunks = 1;
    if (info->coll == ncclFuncPutSignal && totalBytes > chunkSize) {
      numChunks = (totalBytes + chunkSize - 1) / chunkSize;
    }

    // Create tasks for each chunk
    for (int chunkIdx = 0; chunkIdx < numChunks; chunkIdx++) {
      struct ncclTaskRma* t = ncclMemoryPoolAlloc<struct ncclTaskRma>(&comm->memPool_ncclTaskRma, &comm->memPermanent);

      // Calculate chunk-specific size and offsets
      size_t chunkBytes = (chunkIdx == numChunks - 1)
                          ? (totalBytes - chunkIdx * chunkSize)
                          : chunkSize;

      size_t chunkOffset = chunkIdx * chunkSize;

      t->func = info->coll;
      t->srcBuff = (const char*)srcBuff + chunkOffset;
      t->srcWinOffset = srcWinOffset + chunkOffset;
      t->srcWinHost = srcWinHost;
      t->count = chunkBytes / ncclTypeSize(info->datatype);
      t->datatype = info->datatype;
      t->bytes = chunkBytes;
      t->ctx = info->ctx;
      t->peer = info->root;
      t->peerWinOffset = info->peerWinOffset + chunkOffset;
      t->peerWinHost = peerWinHost;

      // Signal handling: only the last chunk gets the signal
      bool isLastChunk = (chunkIdx == numChunks - 1);
      if (isLastChunk) {
        t->signalMode = NCCL_SIGNAL;
      } else {
        // Earlier chunks: no signal
        t->signalMode = NCCL_SIGNAL_NONE;
      }
      t->peers = NULL;
      t->nsignals = NULL;
      t->npeers = 0;

      t->eActivationMask = __atomic_load_n(&ncclProfilerEventMask, __ATOMIC_RELAXED);

      planner->nTasksRma++;
      // Enqueue the task into the appropriate context queue
      ncclIntruQueueEnqueue(&planner->rmaTaskQueues[t->ctx], t);
    }
  }

  return ncclSuccess;
}

// Converts `info` to a task and adds it to `comm->planner`. The exception is with
// single rank communicators, collectives are issued as `ncclMemcpyAsync`s and
// thus don't need a task.
static ncclResult_t taskAppend(struct ncclComm* comm, struct ncclInfo* info) {
  ncclFunc_t collAPI = info->coll;

  if (info->coll == ncclFuncSend || info->coll == ncclFuncRecv) {
    NCCLCHECK(p2pTaskAppend(comm, info, info->coll, collAPI, (void*)info->recvbuff, info->count, info->datatype, info->root, true));
  } else if (info->coll == ncclFuncPutSignal || info->coll == ncclFuncSignal || info->coll == ncclFuncWaitSignal) {
    NCCLCHECK(rmaTaskAppend(comm, info));
  } else {
    // Empty collectives can be discarded.
    if (info->count == 0) return ncclSuccess;

    if (info->datatype == ncclFloat8e4m3 || info->datatype == ncclFloat8e5m2) {
      if (comm->minCompCap < 90 && info->coll != ncclFuncAllGather && info->coll != ncclFuncBroadcast && info->coll != ncclFuncAlltoAll && info->coll != ncclFuncScatter && info->coll != ncclFuncGather) {
        WARN("FP8 reduction support begins with sm90 capable devices.");
        return ncclInvalidArgument;
      }
    }

    // Copy reduction op state from op handle into info struct here since the
    // op handle may be destroyed before ncclGroupEnd().
    struct ncclDevRedOpFull opDev;
    NCCLCHECK(hostToDevRedOp(&opDev, info->op, info->datatype, comm));

    if (comm->nRanks == 1) {
      NCCLCHECK(ncclLaunchOneRank(info->recvbuff, info->sendbuff, info->count, opDev, info->datatype, info->stream));
      return ncclSuccess;
    } else {
      struct ncclDevrWindow* sendWin;
      struct ncclDevrWindow* recvWin;
      ncclDevrFindWindow(comm, info->sendbuff, &sendWin);
      ncclDevrFindWindow(comm, info->recvbuff, &recvWin);
      // Append CE collective task if CE is supported and requested by user
      ncclSymRegType_t winRegType;
      NCCLCHECK(ncclGetSymRegType(sendWin, recvWin, &winRegType));
      bool ceAvailable = ncclCeAvailable(comm, info->coll, info->op, info->datatype, winRegType);
      if ((comm->config.CTAPolicy & NCCL_CTA_POLICY_ZERO) && ceAvailable) {
        NCCLCHECK(ceCollTaskAppend(comm, info, sendWin, recvWin, opDev));
      }
      // Append kernel-based collective
      else {
        // currently legacy sendrecv needs src and dst buffers to be registered
        // we cannot allow UB if alltoall/scatter/gather fallback to legacy sendrecv
        // when src or dst buffers are not registered
        struct ncclReg* sendReg = NULL;
        struct ncclReg* recvReg = NULL;
        bool allowUB = false;
        bool captured = false;
        struct ncclCudaGraph graph;
        // For cuda graph checking
        NCCLCHECK(ncclCudaGetCapturingGraph(&graph, info->stream, comm->config.graphUsageMode));
        captured = ncclCudaGraphValid(graph);
        if (info->coll == ncclFuncAlltoAll) {
          NCCLCHECK(ncclRegFind(comm, info->sendbuff, comm->nRanks * info->count * ncclTypeSize(info->datatype), &sendReg));
          NCCLCHECK(ncclRegFind(comm, info->recvbuff, comm->nRanks * info->count * ncclTypeSize(info->datatype), &recvReg));
          allowUB = captured || (sendReg != NULL && recvReg != NULL);
          for (int r=0; r<comm->nRanks; r++) {
            NCCLCHECK(p2pTaskAppend(comm, info, ncclFuncSend, collAPI, (void*)((char*)info->sendbuff+r*info->count*ncclTypeSize(info->datatype)), info->count, info->datatype, r, allowUB));
            NCCLCHECK(p2pTaskAppend(comm, info, ncclFuncRecv, collAPI, (void*)((char*)info->recvbuff+r*info->count*ncclTypeSize(info->datatype)), info->count, info->datatype, r, allowUB));
          }
        } else if (info->coll == ncclFuncGather){
          size_t offset = 0;
          allowUB = captured;
          NCCLCHECK(p2pTaskAppend(comm, info, ncclFuncSend, collAPI, (void*)info->sendbuff, info->count, info->datatype, info->root, allowUB));
          if (comm->rank == info->root) {
            for (int r=0; r<comm->nRanks; r++) {
              void* buff = (void*)((char*)info->recvbuff + offset);
              NCCLCHECK(p2pTaskAppend(comm, info, ncclFuncRecv, collAPI, buff, info->count, info->datatype, r, allowUB));
              offset += info->count * ncclTypeSize(info->datatype);
            }
          }
        } else if (info->coll == ncclFuncScatter) {
          size_t offset = 0;
          allowUB = captured;
          if (comm->rank == info->root) {
            for (int r = 0; r < comm->nRanks; r++) {
              void* buff = (void*)((char*)info->sendbuff + offset);
              NCCLCHECK(p2pTaskAppend(comm, info, ncclFuncSend, collAPI, buff, info->count, info->datatype, r, allowUB));
              offset += info->count * ncclTypeSize(info->datatype);
            }
          }
          NCCLCHECK(p2pTaskAppend(comm, info, ncclFuncRecv, collAPI, (void*)info->recvbuff, info->count, info->datatype, info->root, allowUB));
        } else if (ceAvailable && comm->symmetricSupport && info->coll == ncclFuncAllGather && info->count > ncclParamSymCeThreshold() && comm->minCompCap >= 100 && comm->isAllDirectNvlink) {
          // Use CE for Allgather on Blackwell with size > 8MB
          NCCLCHECK(ceCollTaskAppend(comm, info, sendWin, recvWin, opDev));
        } else {
          NCCLCHECK(collTaskAppend(comm, info, opDev));
        }
      }
    }
  }

  return ncclSuccess;
}

ncclResult_t ncclEnqueueCheck(struct ncclInfo* info) {
  // Early-out on invalid or revoked communicator
  ncclResult_t ret = CommCheck(info->comm, info->opName, "comm");
  if (ret != ncclSuccess) return ncclGroupErrCheck(ret);
  if (info->comm->revokedFlag) {
    WARN("%s: communicator was revoked", info->opName);
    return ncclGroupErrCheck(ncclInvalidUsage);
  }
  // Profiler - If a group API event has already started, update the profilerGroupDepth so that the depth
  // updates correctly for implicit ncclGroupStartInternal and ncclGroupEndInternal calls
  if (ncclProfilerApiState.profilerGroupDepth > 0) {
    ncclProfilerApiState.profilerGroupDepth++;
  }
  NCCLCHECK(ncclGroupStartInternal());
  ret = ncclSuccess;
  int devOld = -1;
  // Check whether communicator is ready to communicate
  NCCLCHECKGOTO(ncclCommEnsureReady(info->comm), ret, fail);

  if (info->comm->checkPointers) {
    CUDACHECKGOTO(cudaGetDevice(&devOld), ret, fail);
    CUDACHECKGOTO(cudaSetDevice(info->comm->cudaDev), ret, fail);
  }
  NCCLCHECKGOTO(ArgsCheck(info), ret, fail);

  INFO(NCCL_COLL,"%s: opCount %lx sendbuff %p recvbuff %p count %zu datatype %d op %d root %d comm %p [nranks=%d] stream %p",
        info->opName, info->comm->opCount, info->sendbuff, info->recvbuff, info->count,
        info->datatype, info->op, info->root, info->comm, info->comm->nRanks, info->stream);
  TRACE_CALL("nccl%s(%" PRIx64 ",%" PRIx64 ",%zu,%d,%d,%d,%p,%p)", info->opName, reinterpret_cast<int64_t>(info->sendbuff), reinterpret_cast<int64_t>(info->recvbuff), info->count, info->datatype, info->op, info->root, info->comm, info->stream);

  NCCLCHECKGOTO(taskAppend(info->comm, info), ret, fail);

exit:
  if (devOld != -1) CUDACHECK(cudaSetDevice(devOld));
  ncclGroupErrCheck(ret);
  NCCLCHECK(ncclGroupEndInternal());
  /* if depth is 1, ncclGroupEndInternal() will trigger group ops. The state can change
   * so we have to check state here. */
  if (info->comm && !info->comm->config.blocking) { NCCLCHECK(ncclCommGetAsyncError(info->comm, &ret)); }
  return ret;
fail:
  if (info->comm && !info->comm->config.blocking) (void) ncclCommSetAsyncError(info->comm, ret);
  goto exit;
}

NCCL_API(ncclResult_t, ncclRedOpCreatePreMulSum, ncclRedOp_t *op, void *scalar, ncclDataType_t datatype, ncclScalarResidence_t residence, ncclComm_t comm);
ncclResult_t ncclRedOpCreatePreMulSum(ncclRedOp_t *op, void *scalar, ncclDataType_t datatype, ncclScalarResidence_t residence, ncclComm_t comm) {
  NCCLCHECK(CommCheck(comm, "ncclRedOpCreatePreMulSum", "comm"));
  /* join init thread before creating PreMulSum op. */
  NCCLCHECK(ncclCommEnsureReady(comm));

  if (comm->userRedOpFreeHead == comm->userRedOpCapacity) {
    // double capacity and resize
    int cap = 2*comm->userRedOpCapacity;
    if (cap < 4) cap = 4;
    ncclUserRedOp *ops = new ncclUserRedOp[cap];
    if (comm->userRedOpCapacity > 0)
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
    int size = ncclTypeSize(datatype);
    if (size < 1) return ncclInternalError;
    user->opFull.scalarArgIsPtr = false;
    std::memcpy(&user->opFull.scalarArg, scalar, size);
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
  // int(ncclMaxRedOp) < int(op) will always be false due to the sizes of
  // the datatypes involved, and that's by design.  We keep the check though
  // just as a reminder.
  // coverity[result_independent_of_operands]
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
