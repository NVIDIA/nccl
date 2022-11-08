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

#include <cstring> // std::memcpy
#include <cinttypes> // PRIx64
#include <float.h>

////////////////////////////////////////////////////////////////////////////////

static ncclResult_t startDevWorkBatch(
  struct ncclComm* comm, int* devWorkBudget, int channelId,
  ncclDevWorkType type, int funcIndex
);
template<typename DevWork>
static ncclResult_t enqueueDevWorkToBatch(
  struct ncclComm* comm, int* devWorkBudget, int channelId, DevWork const* work
);
static ncclResult_t finishDevWorkBatch(struct ncclComm* comm, int channelId);

static ncclResult_t addProxyWorkIfNeeded(struct ncclComm* comm, int channelId, struct ncclProxyWork* proxyWork);
static uint64_t leastLoadedChannels(struct ncclComm* comm, int nCandidates, int nChoose);

static ncclResult_t computeCollDevWork(struct ncclInfo* info, /*out*/int* workFuncIndex, /*out*/struct ncclDevWorkColl* work);
static ncclResult_t computeCollProxyWork(struct ncclInfo* info, int channelId, /*out*/struct ncclProxyWork* proxyWork);

////////////////////////////////////////////////////////////////////////////////

static void* const ncclKernelGeneric = (void*)NCCL_KERN_NAME(SendRecv, RING, SIMPLE, Sum, int8_t);

struct ncclKernelMatch {
  void* kernelFn;
  bool specialized;
};

// Only generate inline kernels for LL
#define NCCL_FUNC5(func, algo, devredop, dtype, specialized) \
  /*LL    */{(void*)NCCL_KERN_NAME(func, algo, LL, devredop, dtype), true && specialized}, \
  /*LL128 */{(void*)NCCL_KERN_NAME(func, algo, LL, devredop, dtype), false && specialized}, \
  /*SIMPLE*/{(void*)NCCL_KERN_NAME(func, algo, LL, devredop, dtype), false && specialized}

#define NCCL_FUNC4(func, devredop, type, specialized) \
  NCCL_FUNC5(func, TREE,           devredop, type, specialized), \
  NCCL_FUNC5(func, RING,           devredop, type, specialized), \
  NCCL_FUNC5(func, COLLNET_DIRECT, devredop, type, specialized), \
  NCCL_FUNC5(func, COLLNET_CHAIN,  devredop, type, specialized), \
  NCCL_FUNC5(func, NVLS,           devredop, type, specialized), \
  NCCL_FUNC5(func, NVLS_TREE,      devredop, type, specialized)

#ifdef __CUDA_BF16_TYPES_EXIST__
  #define HAVE_BFLOAT16 1
#else
  #define HAVE_BFLOAT16 0
#endif

// Must be consistent with ncclDataType_t
#define NCCL_FUNCS3(func, devredop, reduction, specialized) \
  NCCL_FUNC4(func, devredop, MACRO_IF(reduction, int8_t, int8_t), specialized), \
  NCCL_FUNC4(func, devredop, MACRO_IF(reduction, uint8_t, int8_t), specialized), \
  NCCL_FUNC4(func, devredop, MACRO_IF(reduction, int32_t, int8_t), specialized), \
  NCCL_FUNC4(func, devredop, MACRO_IF(reduction, uint32_t, int8_t), specialized), \
  NCCL_FUNC4(func, devredop, MACRO_IF(reduction, int64_t, int8_t), specialized), \
  NCCL_FUNC4(func, devredop, MACRO_IF(reduction, uint64_t, int8_t), specialized), \
  NCCL_FUNC4(func, devredop, MACRO_IF(reduction, half, int8_t), specialized), \
  NCCL_FUNC4(func, devredop, MACRO_IF(reduction, float, int8_t), specialized), \
  NCCL_FUNC4(func, devredop, MACRO_IF(reduction, double, int8_t), specialized) \
  MACRO_IF(HAVE_BFLOAT16, \
    SINGLE_ARG(, NCCL_FUNC4(func, devredop, MACRO_IF(reduction, __nv_bfloat16, int8_t), specialized)), \
    /*nothing*/ \
  )

// Must be consistent with ncclDevRedOp_t -- but we only generate kernel for sums.
#define NCCL_FUNCS2(func, reduction) \
  NCCL_FUNCS3(func, Sum, reduction, /*specialized=*/1), /*Sum*/ \
  NCCL_FUNCS3(func, Sum, reduction, /*specialized=*/0), /*Prod*/ \
  NCCL_FUNCS3(func, Sum, reduction, /*specialized=*/0), /*Max*/ \
  NCCL_FUNCS3(func, Sum, reduction, /*specialized=*/0), /*Min*/ \
  NCCL_FUNCS3(func, Sum, reduction, /*specialized=*/0), /*PreMulSum*/ \
  NCCL_FUNCS3(func, Sum, reduction, /*specialized=*/0)  /*SumPostDiv*/

// Must be consistent with the ncclFuncSet enum
static const ncclKernelMatch ncclKerns[1+ncclNumTypes+NCCL_NUM_FUNCTIONS*ncclNumDevRedOps*ncclNumTypes*NCCL_NUM_ALGORITHMS*NCCL_NUM_PROTOCOLS] = {
  {(void*)NCCL_KERN_NAME(SendRecv, RING, SIMPLE, Sum, int8_t), true},
  // We don't bake special kernels for the one-rank reductions
  {/*int8*/(void*)NCCL_KERN_NAME(SendRecv, RING, SIMPLE, Sum, int8_t), false},
  {/*uint8*/(void*)NCCL_KERN_NAME(SendRecv, RING, SIMPLE, Sum, int8_t), false},
  {/*int32*/(void*)NCCL_KERN_NAME(SendRecv, RING, SIMPLE, Sum, int8_t), false},
  {/*uint32*/(void*)NCCL_KERN_NAME(SendRecv, RING, SIMPLE, Sum, int8_t), false},
  {/*int64*/(void*)NCCL_KERN_NAME(SendRecv, RING, SIMPLE, Sum, int8_t), false},
  {/*uint64*/(void*)NCCL_KERN_NAME(SendRecv, RING, SIMPLE, Sum, int8_t), false},
  {/*half*/(void*)NCCL_KERN_NAME(SendRecv, RING, SIMPLE, Sum, int8_t), false},
  {/*float*/(void*)NCCL_KERN_NAME(SendRecv, RING, SIMPLE, Sum, int8_t), false},
  {/*double*/(void*)NCCL_KERN_NAME(SendRecv, RING, SIMPLE, Sum, int8_t), false},
  #if HAVE_BFLOAT16
    {/*bfloat16*/(void*)NCCL_KERN_NAME(SendRecv, RING, SIMPLE, Sum, int8_t), false},
  #endif
  NCCL_FUNCS2(Broadcast, /*reduction=*/0),
  NCCL_FUNCS2(Reduce, /*reduction=*/1),
  NCCL_FUNCS2(AllGather, /*reduction=*/0),
  NCCL_FUNCS2(ReduceScatter, /*reduction=*/1),
  NCCL_FUNCS2(AllReduce, /*reduction=*/1)
};

NCCL_PARAM(L1SharedMemoryCarveout, "L1_SHARED_MEMORY_CARVEOUT", 0);

// Returns maximum kernel stack size of all CUDA kernels
ncclResult_t ncclInitKernelsForDevice(int cudaArch, size_t* maxStackSize) {
  constexpr int KernelCount = sizeof(ncclKerns)/sizeof(ncclKerns[0]);
  ncclResult_t result = ncclSuccess;

  if (maxStackSize) *maxStackSize = 0;
  int carveout = ncclParamL1SharedMemoryCarveout();

  // Keep track if we already visited a function pointer.
  void* lru[2] = {nullptr, nullptr};
  for (int i=0; i < KernelCount; i++) {
    void* fn = ncclKerns[i].kernelFn;
    if (fn == lru[0] || fn == lru[1]) goto next_kernel;
    lru[1] = lru[0];
    lru[0] = fn;

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

// Start a new empty batch. Finishes any pre-existing work-in-progress batch.
static ncclResult_t startDevWorkBatch(
    struct ncclComm* comm, int* devWorkBudget, int channelId, ncclDevWorkType type, int funcIndex
  ) {
  struct ncclKernelPlanner::Channel* chan = &comm->planner.channels[channelId];
  *devWorkBudget += chan->devWorkFootprint; // subtract delta of chan->devWorkFootprint, part 1
  NCCLCHECK(finishDevWorkBatch(comm, channelId));
  chan->devWorkBatchWip = ncclMemoryStackAlloc<struct ncclKernelPlanWorkBatchList>(&comm->memScoped);
  struct ncclKernelPlanWorkBatchList* batch = chan->devWorkBatchWip;
  batch->header.funcIndex = funcIndex;
  batch->header.type = type;
  batch->header.nWorks = 0;
  batch->footprint = alignUp(sizeof(struct ncclDevWorkBatchHeader), 16);
  ncclIntruQueueConstruct(&batch->workQueue);
  chan->devWorkFootprint += batch->footprint;
  if (type == ncclDevWorkTypeP2p) {
    // Clear p2p connection hashtable.
    chan->p2p.nConns = 0;
    for (int ty=0; ty < int(sizeof(chan->p2p.nConnsOfType)/sizeof(chan->p2p.nConnsOfType[0])); ty++) {
      chan->p2p.nConnsOfType[ty] = 0;
    }
    for (int slot=0; slot < int(sizeof(chan->p2p.slotConnId)/sizeof(chan->p2p.slotConnId[0])); slot++) {
      chan->p2p.slotConnId[slot] = -1;
    }
  }
  *devWorkBudget -= chan->devWorkFootprint; // subtract delta of chan->devWorkFootprint, part 2
  return ncclSuccess;
}

// Add a work to the work-in-progress batch. This does no overflow checks and
// so assumes the batch can accommodate the work.
template<typename DevWork>
static ncclResult_t enqueueDevWorkToBatch(
    struct ncclComm* comm, int* devWorkBudget, int channelId, DevWork const* work
  ) {
  struct ncclKernelPlanner::Channel* chan = &comm->planner.channels[channelId];
  *devWorkBudget += chan->devWorkFootprint; // subtract delta of chan->devWorkFootprint, part 1
  struct ncclKernelPlanWorkBatchList* batch = chan->devWorkBatchWip;
  struct ncclKernelPlanWorkList* node = (struct ncclKernelPlanWorkList*)ncclMemoryStackAlloc(
    &comm->memScoped, sizeof(ncclKernelPlanWorkList)+sizeof(DevWork), alignof(ncclKernelPlanWorkList)
  );
  memcpy((DevWork*)(node+1), work, sizeof(DevWork));
  chan->devWorkFootprint -= batch->footprint; // add delta of batch->footprint, part 1
  batch->header.nWorks += 1;
  batch->footprint = ncclDevWorkBatchFootprint<DevWork>(batch->header.nWorks);
  chan->devWorkFootprint += batch->footprint; // add delta of batch->footprint, part 2
  ncclIntruQueueEnqueue(&batch->workQueue, node);
  *devWorkBudget -= chan->devWorkFootprint; // subtract delta of chan->devWorkFootprint, part 2
  return ncclSuccess;
}

// If the work-in-progress batch exists, do any required post processing and
// then add it to devWorkBatchQueue.
static ncclResult_t finishDevWorkBatch(struct ncclComm* comm, int channelId) {
  struct ncclKernelPlanner::Channel* chan = &comm->planner.channels[channelId];
  struct ncclKernelPlanWorkBatchList* batch = chan->devWorkBatchWip;
  if (batch == nullptr) return ncclSuccess;
  int nWorks = batch->header.nWorks;
  if (nWorks == 0) { // Empty batch. Drop it.
    chan->devWorkFootprint -= batch->footprint;
    chan->devWorkBatchWip = nullptr;
    return ncclSuccess;
  }
  if (batch->header.type == ncclDevWorkTypeP2p) {
    struct ncclKernelPlanWorkList* node;
    // Pass #1: Count fifo vs copy works
    int nSendWorks=0, nCopyWorks=0;
    node = ncclIntruQueueHead(&batch->workQueue);
    for (int w=0; w < nWorks; w++) {
      struct ncclDevWorkP2p* work = (struct ncclDevWorkP2p*)(node+1);
      if (work->p2pType == ncclDevWorkP2pTypeSend) nSendWorks++;
      if (work->p2pType == ncclDevWorkP2pTypeCopy) nCopyWorks++;
      node = node->next;
    }
    int nFifoWorks = nWorks - nCopyWorks;

    // Determine num warps per work
    constexpr int MaxWarps = NCCL_MAX_NTHREADS/WARP_SIZE;
    constexpr int CopyWeight=2, FifoWeight=1;
    int nCopyWarps =        (nCopyWorks*CopyWeight*MaxWarps)
                   / (nFifoWorks*FifoWeight + nCopyWorks*CopyWeight);
    if (nCopyWorks != 0 && nCopyWarps == 0) nCopyWarps = 1;
    if (nFifoWorks != 0 && nCopyWarps == MaxWarps) nCopyWarps -= 1;
    int nFifoWarps = MaxWarps - nCopyWarps;

    int nWarpPerFifoWork = nFifoWorks==0 ? 0 : nFifoWarps/nFifoWorks;
    int nWarpModFifoWork = nFifoWorks==0 ? 0 : nFifoWarps%nFifoWorks;
    int nWarpPerCopyWork = nCopyWorks==0 ? 0 : nCopyWarps/nCopyWorks;
    int extraWarpPerSend = (nSendWorks <= nWarpModFifoWork) ? 1 : 0;

    // Pass #2: Partition warps over works
    uint32_t lastWarpMask=0; // 1-bit for each last warp of a group
    int warp=0, group=0;
    node = ncclIntruQueueHead(&batch->workQueue);
    for (int w=0; w < nWorks; w++) {
      struct ncclDevWorkP2p* work = (struct ncclDevWorkP2p*)(node+1);
      int nWorkWarps = 0;
      switch (work->p2pType) {
      case ncclDevWorkP2pTypeRecv: nWorkWarps = nWarpPerFifoWork; break;
      case ncclDevWorkP2pTypeSend: nWorkWarps = nWarpPerFifoWork + extraWarpPerSend; break;
      case ncclDevWorkP2pTypeCopy: nWorkWarps = nWarpPerCopyWork; break;
      }
      lastWarpMask |= 1u<<(warp + nWorkWarps-1);
      work->group = 31; // Initialize as unassigned
      if (nWorkWarps > 1) { // Needs CUDA barrier id
        // Give out these group numbers first. Workers not needing a CUDA barrier id
        // go into higher groups in following pass.
        switch (work->p2pType) {
        case ncclDevWorkP2pTypeRecv:
          work->group = group;
          group += 1;
          break;
        case ncclDevWorkP2pTypeSend:
          work->group = group;
          group += (nWorkWarps*WARP_SIZE >= NCCL_SIMPLE_FENCE_WARP_NTHREADS) ? 2 : 1;
          break;
        }
      }
      warp += nWorkWarps;
      node = node->next;
    }
    batch->header.p2p.lastWarpMask = lastWarpMask;

    // Pass #3: Assign group ids to those that don't use a CUDA barrier id.
    node = ncclIntruQueueHead(&batch->workQueue);
    for (int w=0; w < nWorks; w++) {
      struct ncclDevWorkP2p* work = (struct ncclDevWorkP2p*)(node+1);
      if (work->group == /*unassigned*/31) {
        work->group = group;
        group += 1;
      }
      node = node->next;
    }

    chan->p2p.opCount++;
  }
  // Wip is complete, add to queue.
  ncclIntruQueueEnqueue(&chan->devWorkBatchQueue, batch);
  chan->devWorkBatchWip = nullptr;
  return ncclSuccess;
}

// Add a dev work to work-in-progress batch. Check for overflows and starts a
// new work-in-progress if necessary. For p2p's this will do all the extra
// constraint checks to ensure that redundant peers don't exist in the same batch.
template<ncclDevWorkType type, typename DevWork>
static ncclResult_t appendDevWork(
    struct ncclComm* comm, int* devWorkBudget, int channelId,
    int funcIndex, DevWork const* work, struct ncclProxyWork* proxyWork, bool fuseOk
  ) {
  struct ncclKernelPlanner::Channel* chan = &comm->planner.channels[channelId];

  constexpr int p2pSlotCount = 2*NCCL_MAX_P2P_CONNS_PER_WORK_BATCH;
  uint32_t p2pConnId = -1u;
  int p2pSlot = -1;
  struct ncclDevWorkP2p* p2pWork = nullptr;
  if (type == ncclDevWorkTypeP2p) {
    p2pWork = (struct ncclDevWorkP2p*)work;
    p2pConnId = (p2pWork->p2pType == ncclDevWorkP2pTypeCopy)
              ? comm->localRank
              : NCCL_MAX_LOCAL_RANKS + (p2pWork->fifo.peer<<1) + (int)p2pWork->p2pType;
    p2pSlot = p2pConnId % p2pSlotCount;
  }

  if (!fuseOk) goto new_batch;

  { struct ncclKernelPlanWorkBatchList* batch = chan->devWorkBatchWip;
    if (batch == nullptr) goto new_batch;
    if (funcIndex != batch->header.funcIndex) goto new_batch;

    size_t footprintNew = ncclDevWorkBatchFootprint<DevWork>(batch->header.nWorks+1);
    if (footprintNew > comm->maxDevWorkFootprint) goto new_batch;

    if (type == ncclDevWorkTypeP2p) {
      static_assert(!(NCCL_MAX_P2P_CONNS_PER_WORK_BATCH & (NCCL_MAX_P2P_CONNS_PER_WORK_BATCH-1)),
                    "NCCL_MAX_P2P_CONNS_PER_WORK_BATCH must be a pow2");
      int probe = 1; // quadratic probing index, hence the need for ispow2(nSlots)
      while (true) {
        if (chan->p2p.slotConnId[p2pSlot] == p2pConnId) {
          // Connection already exists.
          goto p2p_new_batch;
        }
        if (chan->p2p.slotConnId[p2pSlot] == ~0u) {
          // Empty slot implies new connection. Attempt to fit another.
          if (chan->p2p.nConns == NCCL_MAX_P2P_CONNS_PER_WORK_BATCH) goto p2p_new_batch;
          if (p2pWork->p2pType != ncclDevWorkP2pTypeCopy) {
            if (chan->p2p.nConnsOfType[(int)p2pWork->p2pType] == NCCL_MAX_P2P_FIFO_CONNS_PER_WORK_BATCH) {
              goto p2p_new_batch;
            }
          } else { // copy
            // The first of each two copies steals a spot from both send and recv.
            // This is because the p2p schedule requires that we can always process one
            // send and one recv concurrently. So we model this as N slots where each
            // slot can do either:
            //   * At most one send, and at most one recv, both concurrently.
            //   * At most two copies, concurrently.
            if (chan->p2p.nConnsOfType[(int)ncclDevWorkP2pTypeCopy]%2 == 0) {
              if (chan->p2p.nConnsOfType[(int)ncclDevWorkP2pTypeRecv] == NCCL_MAX_P2P_FIFO_CONNS_PER_WORK_BATCH ||
                  chan->p2p.nConnsOfType[(int)ncclDevWorkP2pTypeSend] == NCCL_MAX_P2P_FIFO_CONNS_PER_WORK_BATCH) {
                goto p2p_new_batch;
              }
              chan->p2p.nConnsOfType[(int)ncclDevWorkP2pTypeRecv]++;
              chan->p2p.nConnsOfType[(int)ncclDevWorkP2pTypeSend]++;
            }
          }
          goto enqueue_work;
        }
        // Didn't find matching or empty slot. Move to next slot via quadratic probing.
        p2pSlot = (p2pSlot + probe++) % p2pSlotCount;
      }
    p2p_new_batch:
      p2pSlot = p2pConnId%p2pSlotCount;
      goto new_batch;
    }
  }
  goto enqueue_work;

new_batch:
  NCCLCHECK(startDevWorkBatch(comm, devWorkBudget, channelId, type, funcIndex));

enqueue_work:
  NCCLCHECK(enqueueDevWorkToBatch(comm, devWorkBudget, channelId, work));
  if (type == ncclDevWorkTypeP2p) {
    // Add p2p connection to hashtable
    chan->p2p.nConns++;
    chan->p2p.nConnsOfType[(int)p2pWork->p2pType]++;
    chan->p2p.slotConnId[p2pSlot] = p2pConnId;
  }
  if (proxyWork != nullptr) {
    if (type == ncclDevWorkTypeP2p) {
      if (p2pWork->p2pType != ncclDevWorkP2pTypeCopy) { // Copies don't require proxy ops.
        proxyWork->opCount = uint64_t(chan->p2p.opCount)<<1 | /*p2p=*/1;
        NCCLCHECK(addProxyWorkIfNeeded(comm, channelId, proxyWork));
      }
    } else {
      NCCLCHECK(addProxyWorkIfNeeded(comm, channelId, proxyWork));
    }
  }
  return ncclSuccess;
}

// Add proxy work to plan's channel only if a proxy thread exists.
static ncclResult_t addProxyWorkIfNeeded(
    struct ncclComm* comm, int channelId, struct ncclProxyWork* proxyWork
  ) {
  bool needed = true;
  NCCLCHECK(ncclProxySaveWork(comm, channelId, proxyWork, &needed));
  if (needed) {
    struct ncclProxyWork* q = ncclMemoryPoolAlloc<struct ncclProxyWork>(&comm->memPool_ncclProxyWork, &comm->memPermanent);
    *q = *proxyWork; // C++ struct assignment
    ncclIntruQueueEnqueue(&comm->planner.channels[channelId].proxyWorkQueue, q);
  }
  return ncclSuccess;
}

// leastLoadedChannels: Returns mask of the `nChoose` least loaded channels from
// channels `[0, nCandidates)`.
static uint64_t leastLoadedChannels(struct ncclComm* comm, int nCandidates, int nChoose) {
  struct ncclKernelPlanner::Channel *chans = comm->planner.channels;
  uint64_t mask = ~uint64_t(0)>>(64-nChoose);
  if (nChoose == nCandidates) return mask;
  int maxChannelInMask = 0;
  size_t maxLoadInMask = chans[0].loadBytes;
  // Initialize such that the first `nChoose` channels are accounted for.
  for (int c=1; c < nChoose; c++) {
    if (maxLoadInMask < chans[c].loadBytes) {
      maxChannelInMask = c;
      maxLoadInMask = chans[c].loadBytes;
    }
  }
  // Sort in the rest of the channels. If a channel has less work than the max
  // member of mask, replace that member and compute the new max.
  for (int c=nChoose; c < nCandidates; c++) {
    if (chans[c].loadBytes < maxLoadInMask) {
      mask ^= uint64_t(1)<<maxChannelInMask; // Drop previous max channel
      mask ^= uint64_t(1)<<c; // Include new channel
      // Compute new max
      uint64_t maskAhead = mask; // Unvisited mask bits
      maxChannelInMask = bitffs(maskAhead)-1; // Find least 1 bit
      maskAhead &= maskAhead-1; // Drop least 1 bit
      maxLoadInMask = chans[maxChannelInMask].loadBytes;
      while (maskAhead != 0) {
        int i = bitffs(maskAhead)-1;
        maskAhead &= maskAhead-1;
        if (maxLoadInMask < chans[i].loadBytes) {
          maxChannelInMask = i;
          maxLoadInMask = chans[i].loadBytes;
        }
      }
    }
  }
  return mask;
}

// Put coll workElem & proxyWork in plan assuming nWorkBudget permits, so please
// ensure *nWorkBudget >= nBids upon entry.
static ncclResult_t addCollToPlan(
    struct ncclComm* comm, int* devWorkBudget, struct ncclInfo* info,
    int funcIndex, struct ncclDevWorkColl const* devWork_, int maxChannels,
    bool regBufUsed, void* regBufSend[], void* regBufRecv[]
  ) {
  struct ncclKernelPlanner::Channel *chans = comm->planner.channels;
  struct ncclDevWorkColl devWork = *devWork_; // C++ struct assignment
  int nParts = devWork.nChannels;
  uint64_t chanMask = leastLoadedChannels(comm, maxChannels, nParts);
  size_t bytes = info->nBytes/nParts;
  uint64_t proxyOpCount = uint64_t(comm->planner.collOpCount++)<<1 | /*coll=*/0;
  for (int part=0; part < nParts; part++) {
    int channelId = bitffs(chanMask)-1; // Find least 1 bit
    chanMask &= chanMask-1; // Drop least 1 bit
    chans[channelId].loadBytes += bytes; // Add our load to channel counter.

    devWork.bid = part;
    struct ncclProxyWork proxyWork;
    proxyWork.opCount = proxyOpCount;
    NCCLCHECK(computeCollProxyWork(info, channelId, &proxyWork));

    // Add devWork
    if (!regBufUsed) {
      NCCLCHECK(appendDevWork<ncclDevWorkTypeColl>(comm, devWorkBudget, channelId, funcIndex, &devWork, &proxyWork, /*fuseOk=*/true));
    } else {
      // Buffer registration in play which could only for CollNet at the moment.
      struct ncclChannel* channel = &comm->channels[channelId];
      struct ncclDevWorkCollReg devWorkReg;
      devWorkReg.coll = devWork; // C++ struct assignment
      for (int i=0; i < NCCL_MAX_DIRECT_ARITY; i++) {
        int peer = channel->collnetDirect.down[i];
        if (peer == -1) break;
        int j = comm->rankToLocalRank[peer]; // Get intra-node slot
        devWorkReg.dnInputs[i] = regBufSend[j]; // Input buffer of leaf peer
        devWorkReg.dnOutputs[i] = regBufRecv[j]; // Output buffer of leaf peer
      }
      for (int i=0; i < NCCL_MAX_DIRECT_ARITY; i++) {
        int peer = channel->collnetDirect.up[i];
        if (peer == -1) break;
        int j = comm->rankToLocalRank[peer];
        // Output buffer of root peer
        devWorkReg.upOutputs[i] = regBufRecv[j];
      }
      NCCLCHECK(appendDevWork<ncclDevWorkTypeCollReg>(comm, devWorkBudget, channelId, funcIndex, &devWorkReg, &proxyWork, /*fuseOk=*/true));
    }
  }
  return ncclSuccess;
}

NCCL_PARAM(ChunkSize, "CHUNK_SIZE", 0);
NCCL_PARAM(P2pLLThreshold, "P2P_LL_THRESHOLD", 16384);

// Put p2p op in plan assuming there is space in devWorkBudget
static ncclResult_t addP2pFifoToPlan(
    struct ncclComm* comm, int* devWorkBudget,
    int peerRank, bool isSendNotRecv, void* localBuf, size_t bytes, int chunk, bool fuseOk
  ) {
  int channelId;
  NCCLCHECK(ncclChannelCompute(comm, peerRank, chunk%comm->p2pnChannelsPerPeer, isSendNotRecv, &channelId));
  struct ncclChannel* channel = &comm->channels[channelId];
  struct ncclChannelPeer* peer = channel->peers[peerRank];
  struct ncclConnector* connector = isSendNotRecv ? &peer->send[1] : &peer->recv[1];
  struct ncclConnInfo* conn = &connector->conn;
  bool usesNetwork = connector->transportComm == (isSendNotRecv ? &netTransport.send : &netTransport.recv);
  int protocol = ((conn->buffs[NCCL_PROTO_LL] != nullptr) && bytes <= ncclParamP2pLLThreshold())
               ? NCCL_PROTO_LL : NCCL_PROTO_SIMPLE;

  int stepSize = comm->buffSizes[protocol]/NCCL_STEPS;
  if (protocol == NCCL_PROTO_SIMPLE) stepSize = comm->p2pChunkSize;

  int chunkSize = stepSize;
  if (usesNetwork && (peerRank != comm->rank)) {
    // Tune chunk size for the network
    if (bytes < stepSize) chunkSize /= 4;
    else if (bytes < 8*stepSize) chunkSize /= 2;
  }
  if (ncclParamChunkSize() != 0) chunkSize = ncclParamChunkSize();

  int chunkEffectiveSize = chunkSize;
  if (protocol == NCCL_PROTO_LL) chunkEffectiveSize /= 2;

  struct ncclDevWorkP2p devWork = {};
  devWork.p2pType = isSendNotRecv ? ncclDevWorkP2pTypeSend : ncclDevWorkP2pTypeRecv;
  devWork.protocol = protocol;
  devWork.bytes = bytes;
  devWork.fifo.peer = peerRank;
  devWork.fifo.localBuf = localBuf;
  devWork.fifo.chunkBytes = chunkEffectiveSize;

  struct ncclProxyWork proxyWork = {};
  proxyWork.pattern = isSendNotRecv ? ncclProxyWorkPatternP2pSend : ncclProxyWorkPatternP2pRecv;
  proxyWork.protocol = protocol;
  proxyWork.sliceSteps = 1;
  proxyWork.p2p.peer = peerRank;
  proxyWork.p2p.sliceBytes = stepSize;
  proxyWork.p2p.slices = std::max<int>(1, divUp(bytes, chunkEffectiveSize));

  NCCLCHECK(appendDevWork<ncclDevWorkTypeP2p>(comm, devWorkBudget, channelId, FUNC_INDEX_P2P, &devWork, &proxyWork, fuseOk));
  return ncclSuccess;
}

// Put p2p op in plan assuming there is space in devWorkBudget
static ncclResult_t addP2pCopyToPlan(
    struct ncclComm* comm, int* devWorkBudget,
    void* dstBuf, void* srcBuf, size_t bytes, int chunk
  ) {
  int peerRank = comm->rank; // peer==self
  int channelId;
  NCCLCHECK(ncclChannelCompute(comm, peerRank, chunk%comm->p2pnChannelsPerPeer, /*isSendNotRecv=*/false, &channelId));
  struct ncclDevWorkP2p devWork = {};
  devWork.p2pType = ncclDevWorkP2pTypeCopy;
  devWork.bytes = bytes;
  devWork.copy.dstBuf = dstBuf;
  devWork.copy.srcBuf = srcBuf;
  NCCLCHECK(appendDevWork<ncclDevWorkTypeP2p>(comm, devWorkBudget, channelId, FUNC_INDEX_P2P, &devWork, nullptr, /*fuseOk=*/true));
  return ncclSuccess;
}

static ncclResult_t reclaimPlan(struct ncclComm* comm, struct ncclCommCallback* me);

// Creates a new plan on `comm->planQueue` from contents of `comm->planner`.
static ncclResult_t finishPlan(struct ncclComm* comm) {
  struct ncclKernelPlan* plan = ncclMemoryPoolAlloc<struct ncclKernelPlan>(&comm->memPool_ncclKernelPlan, &comm->memPermanent);
  ncclIntruQueueEnqueue(&comm->planQueue, plan);
  plan->comm = comm;
  plan->reclaimer.fn = reclaimPlan;
  plan->args.comm = comm->devComm;
  plan->persistent = comm->planner.persistent;

  bool hasProxyWork = false;
  uint64_t channelMask = 0;
  for (int c=0; c < MAXCHANNELS; c++) {
    NCCLCHECK(finishDevWorkBatch(plan->comm, c));
    bool hasWork = !ncclIntruQueueEmpty(&comm->planner.channels[c].devWorkBatchQueue);
    channelMask |= uint64_t(hasWork ? 1 : 0)<<c;
    hasProxyWork |= !ncclIntruQueueEmpty(&comm->planner.channels[c].proxyWorkQueue);
    plan->channels[c].devWorkFootprint = comm->planner.channels[c].devWorkFootprint;
    plan->channels[c].p2pOpCount = comm->planner.channels[c].p2p.opCount;
    plan->channels[c].devWorkBatchQueue = comm->planner.channels[c].devWorkBatchQueue;
    plan->channels[c].proxyWorkQueue = comm->planner.channels[c].proxyWorkQueue;
  }
  plan->kernelFn = comm->planner.kernelFn;
  plan->threadPerBlock = std::max(comm->planner.threadPerBlock, NCCL_MIN_NTHREADS);
  plan->channelMask = channelMask;
  plan->hasProxyWork = hasProxyWork;
  plan->collOpCount = comm->planner.collOpCount;
  plan->ipcMemQueue = comm->planner.ipcMemQueue;
  return ncclSuccess;
}

static ncclResult_t registerIntraNodeBuffers(
    struct ncclComm* comm, struct ncclInfo* info, bool* outRegBufUsed,
    void* outRegBufSend[NCCL_MAX_LOCAL_RANKS],
    void* outRegBufRecv[NCCL_MAX_LOCAL_RANKS]
  ) {
  *outRegBufUsed = false;
  ncclResult_t result = ncclSuccess;

#if CUDART_VERSION >= 11030
  int localRank = comm->localRank;

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
      outRegBufSend[i] = nullptr;
      outRegBufRecv[i] = nullptr;
    } else {
      for (int sr=0; sr < 2; sr++) {
        // Get base address of mapping
        void* base;
        CUDACHECK(cudaIpcOpenMemHandle(&base, handles[i].ipc[sr], cudaIpcMemLazyEnablePeerAccess));
        // Get real buffer address by adding offset in the mapping
        (sr==0 ? outRegBufSend : outRegBufRecv)[i] = (char*)base + handles[i].offset[sr];
        // Enqueue reminder to close memory handle
        struct ncclPointerList* q = ncclMemoryPoolAlloc<struct ncclPointerList>(&comm->memPool_ncclPointerList, &comm->memPermanent);
        q->ptr = base;
        ncclIntruQueueEnqueue(&comm->planner.ipcMemQueue, q);
      }
    }
  }
  *outRegBufUsed = true;

fallback:
#endif
  return result;
}

NCCL_PARAM(GraphRegister, "GRAPH_REGISTER", 0);

static ncclResult_t getCollNetSupport(struct ncclInfo* info, int* collNetTypeSupport);
static ncclResult_t getAlgoInfo(struct ncclInfo* info, int collNetTypeSupport, int numPipeOps);

static ncclResult_t scheduleCollTasksToPlan(struct ncclComm* comm, int* devWorkBudget) {
  struct ncclTasks* tasks = &comm->tasks;
  int nRanks = comm->nRanks;

  size_t bytePerChannel[/*collNetSupport*/2];
  if (comm->channelSize > 0) {
    // Set by user
    bytePerChannel[/*collNetSupport=*/0] = comm->channelSize;
    bytePerChannel[/*collNetSupport=*/1] = comm->channelSize;
  } else {
    // Latency increases as scale increases
    // We would thus want to increase the chunk size to compensate for the lost efficiency
    bytePerChannel[/*collNetSupport=*/0] = NCCL_AGG_CHANNEL_SIZE * std::min(16, comm->nRanks);
    bytePerChannel[/*collNetSupport=*/1] = 256<<10; // Hand-tuned
  }

  for (int collNetSupport=0; collNetSupport < 2; collNetSupport++) {
    while (tasks->collLoadBytes < bytePerChannel[collNetSupport]*comm->nChannels &&
           bytePerChannel[collNetSupport] > NCCL_MIN_CHANNEL_SIZE) {
      // Reduce per-channel size so we utilize all channels.
      bytePerChannel[collNetSupport] /= 2;
    }
  }

  while (tasks->nTasksColl != 0) {
    struct ncclTaskColl* head = ncclIntruQueueHead(&tasks->collQueue);
    struct ncclInfo aggInfo = {};
    aggInfo.comm = comm;
    aggInfo.coll = head->func;
    aggInfo.datatype = head->datatype;
    aggInfo.opFull = head->op;
    aggInfo.op = (ncclRedOp_t)(int)head->op.op;
    aggInfo.count = head->count;
    int nAggChannels = 0;
    int nAggOps = 1;
    struct ncclTaskColl* aggEnd = head->next;
    int collNetSupport = 0;
    NCCLCHECK(getCollNetSupport(&aggInfo, &collNetSupport));

    // Find a range of ops that can be aggregated together.
    while (aggEnd != nullptr &&
           aggEnd->func == aggInfo.coll &&
           aggEnd->datatype == aggInfo.datatype &&
           aggEnd->op.op == aggInfo.opFull.op) {
      aggInfo.count += aggEnd->count;
      int nc = DIVUP(aggEnd->count*ncclTypeSize(aggInfo.datatype), bytePerChannel[collNetSupport]);
      nc = std::max(1, std::min(nc, comm->nChannels));
      nAggChannels += nc;
      nAggOps++;
      aggEnd = aggEnd->next;
    }

    if (nAggOps > 1) {
      NCCLCHECK(ncclInfoSetDerived(&aggInfo, comm->nRanks));
      aggInfo.nChannels = std::min(comm->nChannels, nAggChannels);
      int opPerChannel = DIVUP(nAggChannels, aggInfo.nChannels);
      NCCLCHECK(getAlgoInfo(&aggInfo, collNetSupport, opPerChannel));
    }

    while (head != aggEnd) {
      struct ncclInfo info = {};
      info.comm = comm;
      info.coll = head->func;
      info.sendbuff = head->sendbuff;
      info.recvbuff = head->recvbuff;
      info.count = head->count;
      info.root = head->root;
      info.datatype = head->datatype;
      info.opFull = head->op; // C++ struct assignment
      info.op = (ncclRedOp_t)(int)head->op.op;
      info.chunkSteps = head->chunkSteps;
      info.sliceSteps = head->sliceSteps;
      NCCLCHECK(ncclInfoSetDerived(&info, comm->nRanks));
      if (nAggOps > 1) {
        int maxChannels = aggInfo.algorithm == NCCL_ALGO_NVLS || aggInfo.algorithm == NCCL_ALGO_NVLS_TREE ? comm->nvlsChannels : comm->nChannels;
        info.nChannels = DIVUP(info.nBytes, bytePerChannel[collNetSupport]);
        info.nChannels = std::max(1, std::min(info.nChannels, maxChannels));
        info.algorithm = aggInfo.algorithm;
        info.protocol = aggInfo.protocol;
        info.nThreads = aggInfo.nThreads;
      }

      int workFuncIndex;
      struct ncclDevWorkColl workElem = {};
      NCCLCHECK(computeCollDevWork(&info, &workFuncIndex, &workElem));

      if (*devWorkBudget < info.nChannels*ncclDevWorkBatchFootprint<ncclDevWorkCollReg>(1)) {
        return ncclSuccess; // Ensure room for addCollToPlan()
      }

      bool regBufUsed = false;
      void* regBufSend[NCCL_MAX_LOCAL_RANKS];
      void* regBufRecv[NCCL_MAX_LOCAL_RANKS];
      if (comm->planner.persistent && ncclParamGraphRegister() &&
          info.algorithm == NCCL_ALGO_COLLNET_DIRECT &&   // limited to CollNetDirect for now
          comm->intraHighestTransportType == TRANSPORT_P2P && // only when all ranks can p2p each other
          comm->intraRanks < comm->localRanks) { // only with inter-process & intra-node peers
        NCCLCHECK(registerIntraNodeBuffers(comm, &info, &regBufUsed, regBufSend, regBufRecv));
      }

      int maxChannels = info.algorithm == NCCL_ALGO_NVLS || aggInfo.algorithm == NCCL_ALGO_NVLS_TREE ? comm->nvlsChannels : comm->nChannels;
      NCCLCHECK(addCollToPlan(comm, devWorkBudget, &info, workFuncIndex, &workElem,
                              maxChannels, regBufUsed, regBufSend, regBufRecv));
      tasks->nTasksColl -= 1;
      tasks->collLoadBytes -= info.nBytes;
      ncclIntruQueueDequeue(&tasks->collQueue);
      head = ncclIntruQueueHead(&tasks->collQueue);

      comm->planner.threadPerBlock = std::max(comm->planner.threadPerBlock, info.nThreads);
      if (!comm->planner.kernelSpecialized) {
        comm->planner.kernelFn = ncclKerns[workFuncIndex].kernelFn;
        comm->planner.kernelSpecialized = ncclKerns[workFuncIndex].specialized;
      }
    }
  }

  while (tasks->nTasksBcast != 0) {
    // Make a batch consisting of one bcast from each peer.
    size_t batchLoadBytes = 0;
    int batchTasks = 0;
    for (int peer=tasks->minBcastPeer; peer <= tasks->maxBcastPeer; peer++) {
      struct ncclTaskBcast* t = ncclIntruQueueHead(&tasks->peers[peer].bcastQueue);
      if (t != nullptr) {
        int footprint = ncclDevWorkBatchFootprint<ncclDevWorkBcast>(batchTasks+1);
        if (footprint > comm->maxDevWorkFootprint) break;
        batchTasks += 1;
        batchLoadBytes += t->bytes;
      }
    }
    // Number of channels, one part per channel.
    int nParts = divUp(batchLoadBytes, bytePerChannel[/*collNetSupport=*/0]);
    nParts = std::max(nParts, 1);
    nParts = std::min(nParts, comm->nChannels);
    // Ensure we can fit batch within our budget.
    if (*devWorkBudget < nParts*ncclDevWorkBatchFootprint<ncclDevWorkBcast>(batchTasks)) {
      break;
    }
    // Find best protocol.
    int proto = 0;
    float protoTime = FLT_MAX;
    for (int p=0; p < NCCL_NUM_PROTOCOLS; p++) {
      ncclFunc_t func = batchTasks==1 ? ncclFuncBroadcast : ncclFuncAllGather;
      float bw = comm->bandwidths[(int)func][NCCL_ALGO_RING][p];
      float lat = comm->latencies[(int)func][NCCL_ALGO_RING][p];
      float time = lat + float(batchLoadBytes)/(1000*bw);
      if (time < protoTime) { proto = p; protoTime = time; }
    }

    constexpr int SliceSteps = 1;
    int sliceWireBytes = SliceSteps*(comm->buffSizes[proto]/NCCL_STEPS);
    int sliceDataBytes = sliceWireBytes;
    if (proto == NCCL_PROTO_LL) sliceDataBytes = sliceWireBytes/2;
    if (proto == NCCL_PROTO_LL128) sliceDataBytes = (sliceWireBytes/NCCL_LL128_LINEELEMS)*NCCL_LL128_DATAELEMS;

    // Determine thread count per block.
    int threadPerBlock = comm->maxThreads[NCCL_ALGO_RING][proto];
    int threadThreshold = comm->threadThresholds[NCCL_ALGO_RING][proto];
    while (batchLoadBytes < threadPerBlock*threadThreshold) {
      threadPerBlock -= WARP_SIZE;
    }
    if (proto == NCCL_PROTO_SIMPLE) threadPerBlock += WARP_SIZE; // for threadfence_system()
    threadPerBlock = std::max(threadPerBlock, NCCL_MIN_NTHREADS);
    threadPerBlock = std::min(threadPerBlock, NCCL_MAX_NTHREADS);

    // Increase plan thread count if necessary.
    comm->planner.threadPerBlock = std::max(comm->planner.threadPerBlock, threadPerBlock);

    // Choose kernel for plan.
    int funcIndex = FUNC_INDEX_BCAST(proto);
    if (!comm->planner.kernelSpecialized) {
      comm->planner.kernelFn = ncclKerns[funcIndex].kernelFn;
      comm->planner.kernelSpecialized = ncclKerns[funcIndex].specialized;
    }

    // Compute opCount for proxy work.
    uint64_t proxyOpCount = uint64_t(comm->planner.collOpCount++)<<1 | /*coll=*/0;
    // Break each bcast into nParts, each part assigned to a channel.
    uint64_t chanMask = leastLoadedChannels(comm, comm->nChannels, nParts);
    size_t batchLoadPerPart = batchLoadBytes/nParts;
    for (int part=0; part < nParts; part++) {
      int channelId = bitffs(chanMask)-1; // Index of least 1 bit.
      chanMask &= chanMask-1; // Drop least 1 bit.
      // Add our load to channel counter.
      comm->planner.channels[channelId].loadBytes += batchLoadPerPart;
      // Sort tasks according to ring depth upstream from us.
      int nTasks = batchTasks;
      struct ncclTaskBcast** ringTasks = (struct ncclTaskBcast**)comm->ringTasks;
      for (int r=0; r < nRanks; r++) ringTasks[r] = nullptr;
      int minRingDepth = INT_MAX;
      int maxRingDepth = INT_MIN;
      for (int peer=tasks->minBcastPeer; nTasks != 0; peer++) {
        struct ncclTaskBcast* t = ncclIntruQueueHead(&tasks->peers[peer].bcastQueue);
        if (t != nullptr) {
          nTasks -= 1;
          int ringDepth = comm->channels[channelId].ring.rankToIndex[peer];
          // Need to flip from "downstream from us" to "upstream from us".
          ringDepth = (ringDepth == 0) ? 0 : nRanks-ringDepth;
          ringTasks[ringDepth] = t;
          minRingDepth = std::min(minRingDepth, ringDepth);
          maxRingDepth = std::max(maxRingDepth, ringDepth);
        }
      }
      // Start an empty dev work batch.
      NCCLCHECK(startDevWorkBatch(comm, devWorkBudget, channelId, ncclDevWorkTypeBcast, funcIndex));
      struct ncclDevWorkBatchHeader* batch = &comm->planner.channels[channelId].devWorkBatchWip->header;
      batch->bcast.sliceBytes = sliceDataBytes;
      int sendSlices=0, recvSlices=0;
      // Add each task to the batch in ring depth order.
      for (int ringDepth=minRingDepth; ringDepth <= maxRingDepth; ringDepth++) {
        struct ncclTaskBcast* t = ringTasks[ringDepth];
        if (t != nullptr) {
          struct ncclDevWorkBcast work;
          size_t partBytes = t->bytes/nParts;
          size_t lo = std::min<size_t>(alignUp((part+0)*partBytes, 256), t->bytes);
          size_t hi = std::min<size_t>(alignUp((part+1)*partBytes, 256), t->bytes);
          work.dstBuf = (char*)t->dstBuf + lo;
          work.bytes = hi-lo;
          work.ringDepth = ringDepth;
          if (work.bytes != 0) {
            int slices = divUp(work.bytes, sliceDataBytes);
            if (ringDepth != 0) recvSlices += slices;
            if (ringDepth != nRanks-1) sendSlices += slices;
            if (ringDepth == 0) batch->bcast.rootSrcBuf = (char*)t->srcBuf + lo;
            NCCLCHECK(enqueueDevWorkToBatch(comm, devWorkBudget, channelId, &work));
          }
        }
      }
      // Record a proxy work for this channel.
      if (sendSlices + recvSlices != 0) {
        struct ncclProxyWork proxyWork;
        proxyWork.opCount = proxyOpCount;
        proxyWork.protocol = proto;
        proxyWork.sliceSteps = SliceSteps;
        proxyWork.pattern = ncclProxyWorkPatternRing;
        proxyWork.ring.sendSlices = sendSlices;
        proxyWork.ring.recvSlices = recvSlices;
        NCCLCHECK(addProxyWorkIfNeeded(comm, channelId, &proxyWork));
      }
    }
    // Drop processed work from `comm->tasks`.
    tasks->nTasksBcast -= batchTasks;
    tasks->collLoadBytes -= batchLoadBytes;
    for (int peer=tasks->minBcastPeer; batchTasks != 0; peer++) {
      struct ncclTaskBcast* t = ncclIntruQueueTryDequeue(&tasks->peers[peer].bcastQueue);
      if (t != nullptr) batchTasks -= 1;
    }
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

static ncclResult_t scheduleP2pTasksToPlan(struct ncclComm* comm, int* devWorkBudget) {
  struct ncclTasks* tasks = &comm->tasks;
  int rank = comm->rank;
  int nRanks = comm->nRanks;
  struct ncclTasks::Peer* peers = tasks->peers;
  int const *sendOrder = tasks->p2pSendOrder;
  int const *recvOrder = tasks->p2pRecvOrder;

  comm->planner.threadPerBlock = std::max(comm->planner.threadPerBlock, NCCL_MAX_NTHREADS);
  if (!comm->planner.kernelSpecialized) {
    comm->planner.kernelFn = ncclKerns[FUNC_INDEX_P2P].kernelFn;
    comm->planner.kernelSpecialized = ncclKerns[FUNC_INDEX_P2P].specialized;
  }

  // Compute how much to split operations
  // Natural step size matching buffer steps.
  ssize_t stepSize = comm->p2pChunkSize;
  // Try to use all channels
  int nChannelsMax = comm->p2pnChannelsPerPeer;
  int nChannelsMin = nChannelsMax;
  // Try to use all channels, but one channel per operation.
  while (nChannelsMin*nRanks > comm->p2pnChannels && nChannelsMin > 1) nChannelsMin /= 2;
  // Avoid overloading channels with 8+ operations as we loose the sync warp, hence a bit of bandwidth.
  while (nChannelsMax*nRanks > comm->p2pnChannels*4 && nChannelsMax > 1) nChannelsMax /= 2;

  bool fuseOk;
  // We can perform 8 send/recv per round per CTA. Make sure we jump between fused blocks at node boundaries.
  while (tasks->nTasksP2p != 0) {
    for (int i=0; i < tasks->p2pOrderSteps; i++) {
      int sendPeer = sendOrder[i];
      int recvPeer = recvOrder[i];
      if ((i % NCCL_MAX_P2P_FIFO_CONNS_PER_WORK_BATCH) == 0) fuseOk = false;
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
        char* recvLocalBuf = recv ? (char*)recv->localBuf : nullptr;
        char* sendLocalBuf = send ? (char*)send->localBuf : nullptr;
        ssize_t recvBytes = recv ? recv->bytes : 0;
        ssize_t sendBytes = send ? send->bytes : 0;
        ssize_t minSize = stepSize/8;
        ssize_t maxSize = comm->nNodes > 1 ? stepSize : stepSize*32;
        ssize_t recvChunkBytesMax = calcP2pChunkSize(recvBytes, nChannelsMin, nChannelsMax, minSize, maxSize);
        ssize_t sendChunkBytesMax = calcP2pChunkSize(sendBytes, nChannelsMin, nChannelsMax, minSize, maxSize);
        // Zero size send/recv are syncs, encode here with -1.
        recvBytes = recv && recvBytes == 0 ? -1 : recvBytes;
        sendBytes = send && sendBytes == 0 ? -1 : sendBytes;
        // Advance to current chunk. Syncs will always have chunk=0 so no effect on the -1.
        if (recv) recvLocalBuf  += recv->chunk*recvChunkBytesMax;
        if (recv) recvBytes     -= recv->chunk*recvChunkBytesMax;
        if (send) sendLocalBuf  += send->chunk*sendChunkBytesMax;
        if (send) sendBytes     -= send->chunk*sendChunkBytesMax;

        do {
          ssize_t recvChunkBytes = std::min(recvBytes, recvChunkBytesMax); // -1 preserved
          ssize_t sendChunkBytes = std::min(sendBytes, sendChunkBytesMax);
          if (recvChunkBytes != 0) {
            if (recvChunkBytes == -1) recvChunkBytes = 0;
            if (*devWorkBudget < ncclDevWorkBatchFootprint<ncclDevWorkP2p>(1)) return ncclSuccess; // ensure room in budget
            if (recvPeer == rank) { // local copy
              NCCLCHECK(addP2pCopyToPlan(comm, devWorkBudget, recvLocalBuf, sendLocalBuf, recvChunkBytes, recv->chunk));
            } else {
              NCCLCHECK(addP2pFifoToPlan(comm, devWorkBudget, recvPeer, /*isSendNotRecv=*/false, recvLocalBuf, recvChunkBytes, recv->chunk, fuseOk));
            }
            fuseOk = true;
            recvLocalBuf += recvChunkBytes;
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
            if (*devWorkBudget < ncclDevWorkBatchFootprint<ncclDevWorkP2p>(1)) return ncclSuccess; // ensure room in budget
            if (sendPeer != rank) {
              NCCLCHECK(addP2pFifoToPlan(comm, devWorkBudget, sendPeer, /*isSendNotRecv=*/true, sendLocalBuf, sendChunkBytes, send->chunk,  /*fuseOk=*/true));
            }
            fuseOk = true;
            sendLocalBuf += sendChunkBytes;
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

// Spin until its safe to increase comm->workFifoProduced to desiredProduced.
static void waitWorkFifoAvailable(struct ncclComm* comm, uint32_t desiredProduced) {
  bool hasRoom = (desiredProduced - comm->devWorkFifoConsumedLbound) <= comm->devWorkFifoBytes;
  if (__builtin_expect(!hasRoom, false)) {
    while (1) {
      // We have to poll for notifications from device.
      uint32_t* consumedLive = comm->devWorkFifoConsumed;
      uint32_t consumed[MAXCHANNELS];
      for (int c=0; c < MAXCHANNELS; c++) {
        consumed[c] = __atomic_load_n(&consumedLive[c], __ATOMIC_RELAXED);
      }
      // Compiler-only fence to prevent fusion of loops to encourage dense loads.
      __atomic_signal_fence(__ATOMIC_SEQ_CST);

      uint32_t ubound = comm->devWorkFifoProduced;
      uint32_t consumedLb = ubound;
      for (int c=0; c < MAXCHANNELS; c++) {
        // consumedLb is min over all non-quiesced channels
        if (consumed[c] != comm->channels[c].devWorkFifoProduced) {
          if (ubound - consumedLb < ubound - consumed[c]) {
            consumedLb = consumed[c];
          }
        }
      }

      // Compiler only fence to prevent fusion of loops to encourage dense stores.
      __atomic_signal_fence(__ATOMIC_SEQ_CST);

      for (int c=0; c < MAXCHANNELS; c++) {
        // Advance counter on quiesced channels so they don't lag behind
        // too far where they could get lost in 32-bit wraparound.
        if (consumed[c] == comm->channels[c].devWorkFifoProduced) {
          comm->channels[c].devWorkFifoProduced = consumedLb;
          __atomic_store_n(&consumedLive[c], consumedLb, __ATOMIC_RELAXED);
        }
      }
      comm->devWorkFifoConsumedLbound = consumedLb;

      // See if that was enough.
      hasRoom = (desiredProduced - comm->devWorkFifoConsumedLbound) <= comm->devWorkFifoBytes;
      if (hasRoom) break;
      sched_yield();
    }
  }
}

// Byte copy a T into fifo where the bytes of T may wrap around the fifo boundary.
// * `cursor` is the monotonic position of the fifo writer which is advanced on return,
//   it only wraps around due to 32 bit overflow. It is assumed cursor is already
//   aligned to alignof(T).
// * `mask` is the size (power of 2) of the fifo minus 1.
template<typename T>
static void writeFifo(void* fifo, uint32_t* /*inout*/cursor, uint32_t mask, T const* src) {
  char* d = (char*)fifo;
  char const* s = (char const*)src;
  uint32_t cur = *cursor;
  *cursor += sizeof(T);
  for (int off=0; off < (int)sizeof(T); off += alignof(T)) {
    memcpy(
      __builtin_assume_aligned(d + ((cur + off) & mask), alignof(T)),
      __builtin_assume_aligned(s + off, alignof(T)),
      alignof(T)
    );
  }
}

static ncclResult_t uploadDevWork(struct ncclComm* comm, struct ncclKernelPlan* plan) {
  bool persistent = plan->persistent;
  size_t footprint = 0;
  for (int c=0; c < MAXCHANNELS; c++) {
    footprint += plan->channels[c].devWorkFootprint;
  }

  void* fifoBuf;
  uint32_t fifoCursor, fifoMask;
  if (!persistent) {
    fifoBuf = comm->devWorkFifoBuf;
    fifoCursor = comm->devWorkFifoProduced;
    fifoMask = comm->devWorkFifoBytes-1;
    waitWorkFifoAvailable(comm, fifoCursor + footprint);
    plan->args.devWorkBuf = comm->devWorkFifoBufDev;
    plan->args.devWorkCursor = fifoCursor;
    plan->args.devWorkFifoSizeLog2 = log2i(comm->devWorkFifoBytes);
    plan->args.inWorkFifo = true;
  } else {
    // Persistent kernels don't use a fifo, just a buffer big enough to hold everything.
    fifoBuf = ncclMemoryStackAlloc(&comm->memScoped, footprint, /*align=*/16);
    fifoCursor = 0;
    fifoMask = ~0u;
    plan->args.devWorkCursor = fifoCursor;
    plan->args.inWorkFifo = false;
  }

  // We use two passes over the channels when serializing the work:
  //   pass=0: put first work batch of each channel in fifo.
  //   pass=1: remaining batch of each channel.
  // We do this so each channel `c` in the kernel can find its first batch by
  // calculating:
  //   firstBatchCursor[c] = sum(firstBatchBytes[c1] for c1 < c)`.
  // Batches after the first are located using `nextCursor` pointers. When
  // building the first work of each channel we compute its `nextCursor` by:
  //   firstBatchNext[c] = sum(firstBatchBytes[c1] for all c1)
  //                     + sum(totalBatchBytes[c1] - firstBatchBytes[c1] for c1 < c)
  uint32_t firstBatchNext = fifoCursor; // `nextCursor` field for the first batch
  for (int c=0; c < MAXCHANNELS; c++) {
    struct ncclKernelPlanWorkBatchList* batch = ncclIntruQueueHead(&plan->channels[c].devWorkBatchQueue);
    plan->args.firstDevWorkBytes[c] = (batch ? batch->footprint : 0);
    firstBatchNext += (batch ? batch->footprint : 0);
  }
  for (int pass=0; pass < 2; pass++) {
    for (uint64_t mask=plan->channelMask; mask != 0; mask &= mask-1) {
      int channelId = bitffs(mask)-1;
      struct ncclKernelPlanWorkBatchList* batch = ncclIntruQueueHead(&plan->channels[channelId].devWorkBatchQueue);
      if (batch == nullptr) continue; // Skip channel if it has no work.

      int firstBatchBytes = batch->footprint;
      if (pass == 1) batch = batch->next; // Second pass skips first work.

      while (batch != nullptr) {
        struct ncclDevWorkBatchHeader header = batch->header; // C++ struct copy
        header.nextCursor = pass==0 ? firstBatchNext : (fifoCursor + batch->footprint);
        header.nextBytes = batch->next ? batch->next->footprint : 0;
        writeFifo(fifoBuf, &fifoCursor, fifoMask, &header);
        struct ncclKernelPlanWorkList* work = ncclIntruQueueHead(&batch->workQueue);
        while (work != nullptr) {
          switch (header.type) {
          case ncclDevWorkTypeP2p:
            writeFifo(fifoBuf, &fifoCursor, fifoMask, (struct ncclDevWorkP2p*)(work+1));
            break;
          case ncclDevWorkTypeColl:
            writeFifo(fifoBuf, &fifoCursor, fifoMask, (struct ncclDevWorkColl*)(work+1));
            break;
          case ncclDevWorkTypeCollReg:
            writeFifo(fifoBuf, &fifoCursor, fifoMask, (struct ncclDevWorkCollReg*)(work+1));
            break;
          case ncclDevWorkTypeBcast:
            writeFifo(fifoBuf, &fifoCursor, fifoMask, (struct ncclDevWorkBcast*)(work+1));
            break;
          }
          work = work->next;
        }
        batch = batch->next;
        fifoCursor = alignUp(fifoCursor, 16);
        if (pass == 0) break; // First pass processes only first work.
      }
      if (pass == 0) {
        // The next channel's first batch's `nextCursor` points just after all
        // of this channel's work.
        firstBatchNext += plan->channels[channelId].devWorkFootprint;
        // Don't double count bytes of first batch.
        firstBatchNext -= firstBatchBytes;
      }
      if (!persistent) comm->channels[channelId].devWorkFifoProduced = fifoCursor;
    }
  }

  if (!persistent) {
    comm->devWorkFifoProduced = fifoCursor;
    if (comm->devWorkFifoBufGdrHandle != nullptr) wc_store_fence();
  } else {
    char* buf;
    NCCLCHECK(ncclCudaMalloc(&buf, footprint));
    plan->args.devWorkBuf = (void*)buf;
    NCCLCHECK(ncclCudaMemcpy(buf, (char*)fifoBuf, footprint));
  }
  return ncclSuccess;
}

static ncclResult_t uploadProxyWork(struct ncclComm* comm, struct ncclKernelPlan* plan) {
  uint64_t collOpCount = comm->sharedRes->collOpCount;
  // Advance comm's collOpCount by number of colls in this plan.
  comm->sharedRes->collOpCount = collOpCount + plan->collOpCount;
  for (uint64_t m=plan->channelMask; m != 0; m &= m-1) {
    int channelId = bitffs(m)-1;
    struct ncclProxyWork* q = ncclIntruQueueHead(&plan->channels[channelId].proxyWorkQueue);
    uint64_t p2pOpCount = comm->sharedRes->channels[channelId].p2pOpCount;
    while (q != nullptr) {
      struct ncclProxyWork* qNext = q->next;
      // Ignoring the bottom tag bit, opCount's are zero-based within plan so
      // translate them to the tip of the comm's history.
      bool p2p = (q->opCount & 1);
      q->opCount = ((p2p ? p2pOpCount : collOpCount)<<1) + q->opCount;
      NCCLCHECK(ncclProxySaveWork(comm, channelId, q, nullptr));
      if (!plan->persistent) {
        // Non-persistent kernels have their memory reclaimed after upload.
        ncclMemoryPoolFree(&plan->memPool_ncclProxyWork, q);
      }
      q = qNext;
    }
    // Advance channel's p2pOpCount by number of p2p's in this plan channel.
    comm->sharedRes->channels[channelId].p2pOpCount += plan->channels[channelId].p2pOpCount;
  }
  return ncclSuccess;
}

static ncclResult_t hostStreamPlanTask(struct ncclComm* comm, struct ncclKernelPlan* plan) {
  NCCLCHECK(uploadProxyWork(comm, plan));
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
    NCCLCHECK(ncclCudaFree(plan->args.devWorkBuf));
    while (!ncclIntruQueueEmpty(&plan->ipcMemQueue)) {
      struct ncclPointerList* q = ncclIntruQueueDequeue(&plan->ipcMemQueue);
      CUDACHECKIGNORE(cudaIpcCloseMemHandle(q->ptr));
      ncclMemoryPoolFree(&comm->memPool_ncclPointerList, q);
    }
  }
  ncclMemoryPoolTakeAll(&comm->memPool_ncclProxyWork, &plan->memPool_ncclProxyWork);
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
  // work structs (see appendDevWork() variants all use scoped allocation).
  ncclMemoryStackPush(&comm->memScoped);

  if (tasks->nTasksColl + tasks->nTasksBcast + tasks->nTasksP2p != 0) {
    while (true) {
      memset(&comm->planner, 0, sizeof(comm->planner));
      comm->planner.persistent = persistent;

      // Non-persistent kernels fill up at most half of our fifo per kernel.
      int devWorkBudget = persistent ? INT_MAX : comm->devWorkFifoBytes/2;
      int nTasksOld = tasks->nTasksColl + tasks->nTasksBcast + tasks->nTasksP2p;

      // Drain coll tasks first. This is essential since we partition tasks based
      // on the work budget and p2p work isn't collective. If we were to drain p2p
      // first, the place where we cut the kernel could vary by rank which would
      // cause the least loaded channel picker to have divergent results.
      if (tasks->nTasksColl + tasks->nTasksBcast != 0) {
        NCCLCHECKGOTO(scheduleCollTasksToPlan(comm, &devWorkBudget), result, failure);
      }
      // And only drain p2p tasks once colls are depleted.
      if (tasks->nTasksColl + tasks->nTasksBcast == 0 && tasks->nTasksP2p != 0) {
        NCCLCHECKGOTO(scheduleP2pTasksToPlan(comm, &devWorkBudget), result, failure);
      }

      int nTasksNow = tasks->nTasksColl + tasks->nTasksBcast + tasks->nTasksP2p;
      if (nTasksNow == nTasksOld) {
        // We weren't able to fit any tasks into our budget which means now we're
        // stuck in an infinite loop. We defer this check until here, instead of
        // doing it in comm init, to permit testing with insanely shallow queues
        // for cases where that's expected to still work (e.g. few channels).
        WARN("'NCCL_WORK_FIFO_BYTES=%d' is too small. Minimum value is %d", comm->devWorkFifoBytes, 2*MAXCHANNELS*ncclMaxDevWorkBatchBytes(comm->cudaArch));
        result = ncclInvalidUsage;
        goto failure;
      }
      NCCLCHECKGOTO(finishPlan(comm), result, failure);
      nPlans += 1;
      if (0 == tasks->nTasksColl + tasks->nTasksBcast + tasks->nTasksP2p) break;
    }

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
        if (plan->hasProxyWork) {
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
    ncclMemoryStackPop(&comm->memScoped); // deallocate ncclDevWorkBatchHeader's
  }
  return result;
}

ncclResult_t ncclLaunchKernelBefore_NoUncapturedCuda(struct ncclComm* comm, struct ncclKernelPlan* plan) {
  // This code is called after we've checked in to the intra-process barrier
  // but before launching the kernel. We are not allowed to call CUDA unless the
  // kernel launch is captured.
  NCCLCHECK(uploadDevWork(comm, plan));
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
  dim3 grid = {(unsigned)bitpopcnt(plan->channelMask), 1, 1};
  dim3 block = {(unsigned)plan->threadPerBlock, 1, 1};
  size_t smem = ncclShmemDynamicSize(comm->cudaArch);
  void* args[1] = {&plan->args};

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
  cudaError_t err = cudaLaunchKernel(fn, grid, block, args, smem, launchStream);
  if (err != cudaSuccess) {
    WARN("cudaLaunchKernel(fn=%p, grid={%u,%u,%u}, block={%u,%u,%u}, smem=%llu, stream=%p) failed with %s: %s", fn, grid.x, grid.y, grid.z, block.x, block.y, block.z, (unsigned long long)smem, launchStream, cudaGetErrorName(err), cudaGetErrorString(err));
    return ncclUnhandledCudaError;
  }
  return ncclSuccess;
}

ncclResult_t ncclLaunchKernelAfter_NoCuda(struct ncclComm* comm, struct ncclKernelPlan* plan) {
  if (!(plan->persistent || comm->persistentRefs != 0 || ncclCudaLaunchBlocking)) {
    // If this isn't being captured and there aren't any CUDA graphs alive
    // then we don't need to do our proxyWork pushing on the host stream.
    NCCLCHECK(hostStreamPlanTask(comm, plan));
  }
  return ncclSuccess;
}

ncclResult_t ncclLaunchFinish(struct ncclComm* comm) {
  ncclResult_t result = ncclSuccess;
  struct ncclTasks* tasks = &comm->tasks;

  // Deallocate ncclDevWorkBatchHeader's. This frame exists so long as ncclLaunchPrepare
  // succeeded, and if ncclLaunchPrepare didn't succeed we wouldn't be here.
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

static inline ncclResult_t getCollNetSupport(struct ncclInfo* info, int* collNetTypeSupport) {
  if (info->comm->collNetSupport > 0) {
    // Translate ncclAvg and PreMulSum
    ncclRedOp_t netOp = info->op == ncclAvg || info->op >= ncclNumOps ? ncclSum : info->op;
    NCCLCHECK(collNetReduceSupport(info->comm, info->datatype, netOp, collNetTypeSupport));
  } else {
    *collNetTypeSupport = 0;
  }
  return ncclSuccess;
}

// numPipeOps: number of pipelined ops. Can be greater than 1 in aggregation mode. Used to adjust latency.
static ncclResult_t getAlgoInfo(struct ncclInfo* info, int collNetTypeSupport, int numPipeOps) {
  struct ncclComm* comm = info->comm;
  if (comm->nRanks == 1) {
    info->algorithm = NCCL_ALGO_RING;
    info->protocol = NCCL_PROTO_SIMPLE;
  }
  else {
    float minTime = 3600000000.0; // Hopefully no operation will take an hour to complete.
    // Find algorithm / protocol.
    info->algorithm = -1;
    info->protocol = -1;
    int nAlgos = NCCL_NUM_ALGORITHMS;
    for (int a=0; a<nAlgos; a++) {
      if ((a == NCCL_ALGO_COLLNET_DIRECT || a == NCCL_ALGO_COLLNET_CHAIN) && collNetTypeSupport != 1) continue;
      if (a == NCCL_ALGO_NVLS && !NCCL_NVLS_SUPPORTS(info->datatype, info->opFull.op)) continue;
      if (a == NCCL_ALGO_NVLS && collNetTypeSupport != 1 && comm->nNodes > 1) continue;
      if (a == NCCL_ALGO_NVLS_TREE && !NCCL_NVLS_SUPPORTS(info->datatype, info->opFull.op)) continue;

      for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
        float time;
        NCCLCHECK(ncclTopoGetAlgoTime(info, a, p, numPipeOps, &time));
        if (time >= 0 && time < minTime) {
          info->algorithm = a;
          info->protocol = p;
          minTime = time;
        }
      }
    }
    if (info->algorithm == -1 || info->protocol == -1) {
      WARN("Error : no algorithm/protocol available");
      return ncclInternalError;
    }
    //if (comm->rank == 0) INFO(NCCL_TUNING, "%ld Bytes -> Algo %d proto %d time %f", info->nBytes, info->algorithm, info->protocol, minTime);
    TRACE(NCCL_COLL, "%ld Bytes -> Algo %d proto %d time %f", info->nBytes, info->algorithm, info->protocol, minTime);
  }

  int nc = (info->nChannels > 0) ? info->nChannels : comm->nChannels;
  int nt = comm->maxThreads[info->algorithm][info->protocol];
  int threadThreshold = comm->threadThresholds[info->algorithm][info->protocol];
  if (info->algorithm == NCCL_ALGO_COLLNET_DIRECT) {
    // CollNet channel tuning
    int ncSwitch = 16;
    bool flag = true;
    while (ncSwitch >= 1 && flag) {
      while ((flag = info->nBytes < nc*nt*info->comm->channels[0].collnetDirect.nHeads*threadThreshold) && nc > ncSwitch) {
        if (nc == ncSwitch+ncSwitch/2) threadThreshold /= 2;
        nc--;
      }
      ncSwitch /= 2;
    }
  } else if (info->algorithm == NCCL_ALGO_NVLS || info->algorithm == NCCL_ALGO_NVLS_TREE) {
    // NVLS should not need more than 16 channels to get peak BW.
    nc = comm->nvlsChannels;
  } else {
    // Ring/Tree channel tuning
    while (info->nBytes < nc*nt*threadThreshold) {
      if (nc >= 2) nc--;
      else if ((nt % 128) == 0) nt/=2;
      else break;
    }
  }
  if (info->protocol == NCCL_PROTO_SIMPLE) {
    if (info->algorithm == NCCL_ALGO_RING) nt += WARP_SIZE; // Extra warp for sync
    // More threads or sync warps needed due to split thread model
    if (info->algorithm == NCCL_ALGO_TREE) nt += 4*WARP_SIZE;
  }
  nt = nt/WARP_SIZE < 3 ? 3*WARP_SIZE : nt;
  info->nChannels = nc;
  info->nThreads = nt;
  return ncclSuccess;
}

static ncclResult_t getPatternInfo(struct ncclInfo* info) {
  switch (info->coll) {
    case ncclFuncBroadcast:
      info->pattern = info->algorithm == NCCL_ALGO_TREE ? ncclPatternTreeDown : ncclPatternPipelineFrom; break;
    case ncclFuncReduce:
      info->pattern = info->algorithm == NCCL_ALGO_TREE ? ncclPatternTreeUp : ncclPatternPipelineTo; break;
    case ncclFuncReduceScatter:
    case ncclFuncAllGather:
      info->pattern =
        info->algorithm == NCCL_ALGO_NVLS ? ncclPatternNvls :
        ncclPatternRing; break;
    case ncclFuncAllReduce:
      info->pattern =
        info->algorithm == NCCL_ALGO_NVLS ? ncclPatternNvls :
        info->algorithm == NCCL_ALGO_NVLS_TREE ? ncclPatternNvlsTree :
        info->algorithm == NCCL_ALGO_COLLNET_DIRECT ? ncclPatternCollnetDirect :
        info->algorithm == NCCL_ALGO_COLLNET_CHAIN ? ncclPatternCollnetChain :
        info->algorithm == NCCL_ALGO_TREE ? ncclPatternTreeUpDown :
        ncclPatternRingTwice; break;
    default:
      WARN("Unknown pattern for collective %d algorithm %d", info->coll, info->algorithm);
      return ncclInternalError;
  }
  return ncclSuccess;
}

static ncclResult_t getLoopInfo(struct ncclInfo* info) {
  switch (info->pattern) {
    case ncclPatternNone: break;
    case ncclPatternTreeUp:
    case ncclPatternTreeDown:
    case ncclPatternTreeUpDown:
    case ncclPatternPipelineFrom:
    case ncclPatternPipelineTo:
    case ncclPatternCollnetChain:
      info->nstepsPerLoop = info->nchunksPerLoop = 1; break;
    case ncclPatternNvls:
      info->nstepsPerLoop = 1; info->nchunksPerLoop = info->comm->channels[0].nvls.nHeads; break;
    case ncclPatternCollnetDirect:
      info->nstepsPerLoop = 1; info->nchunksPerLoop = info->comm->channels[0].collnetDirect.nHeads; break;
    case ncclPatternRing:
      info->nstepsPerLoop = info->comm->nRanks-1; info->nchunksPerLoop = info->comm->nRanks; break;
    case ncclPatternRingTwice:
      info->nstepsPerLoop = 2*(info->comm->nRanks-1); info->nchunksPerLoop = info->comm->nRanks; break;
    case ncclPatternNvlsTree:
      info->nstepsPerLoop = 1; info->nchunksPerLoop = info->comm->channels[0].nvls.nHeads; break;
    default:
      WARN("Unknown pattern %d", info->pattern);
      return ncclInternalError;
  }
  return ncclSuccess;
}

static ncclResult_t computeCollDevWork(
    /*inout*/struct ncclInfo* info,
    /*out*/int* workFuncIndex, /*out*/struct ncclDevWorkColl* work
  ) {
  int collNetTypeSupport = 0;
  // Check whether algo and proto have been preset (as in aggregation case)
  // If so, skip the calculation
  if (info->nChannels == 0 || info->nThreads == 0) {
    NCCLCHECK(getCollNetSupport(info, &collNetTypeSupport));
    NCCLCHECK(getAlgoInfo(info, collNetTypeSupport, 1));
  }
  work->srcBuf = (void*)info->sendbuff;
  work->dstBuf = (void*)info->recvbuff;
  work->root = info->root;
  work->count = info->count;
  work->nChannels = info->nChannels;
  work->redOpArg = info->opFull.scalarArg;
  work->redOpArgIsPtr = info->opFull.scalarArgIsPtr;

  if (info->comm->nRanks == 1) {
    // one-rank reduce index
    *workFuncIndex = 1 + int(info->datatype);
    return ncclSuccess;
  }

  *workFuncIndex = FUNC_INDEX(info->coll, info->opFull.op, info->datatype, info->algorithm, info->protocol);

  // Set nstepsPerLoop and nchunksPerLoop
  NCCLCHECK(getPatternInfo(info));
  NCCLCHECK(getLoopInfo(info));

  int stepSize   = info->comm->buffSizes[info->protocol]/NCCL_STEPS;
  int chunkSteps = (info->protocol == NCCL_PROTO_SIMPLE && info->algorithm == NCCL_ALGO_RING) ? info->chunkSteps : 1;
  int sliceSteps = (info->protocol == NCCL_PROTO_SIMPLE && info->algorithm == NCCL_ALGO_RING) ? info->sliceSteps : 1;
  info->chunkSteps = chunkSteps;
  info->sliceSteps = sliceSteps;
  int chunkSize  = stepSize*chunkSteps;

  // Compute lastChunkSize
  if (info->algorithm == NCCL_ALGO_TREE && info->protocol == NCCL_PROTO_SIMPLE) {
    if (info->pattern == ncclPatternTreeUpDown) {
      // Optimize chunkSize / nSteps
      while (info->nBytes / (info->nChannels*chunkSize) < info->comm->channels[0].tree.depth*8 && chunkSize > 131072) chunkSize /= 2;
      while (info->nBytes / (info->nChannels*chunkSize) < info->comm->channels[0].tree.depth*4 && chunkSize > 65536) chunkSize /= 2;
      while (info->nBytes / (info->nChannels*chunkSize) < info->comm->channels[0].tree.depth && chunkSize > 32768) chunkSize /= 2;
    }
    // Use lastChunkSize as chunkSize
    work->lastChunkSize = chunkSize / ncclTypeSize(info->datatype);
  } else if (info->algorithm == NCCL_ALGO_COLLNET_DIRECT) {
    // Optimize chunkSize / nSteps
    while (info->nBytes / (info->nChannels*info->comm->channels[0].collnetDirect.nHeads*chunkSize) < info->comm->channels[0].collnetDirect.depth*64 && chunkSize > 131072) chunkSize /= 2;
    while (info->nBytes / (info->nChannels*info->comm->channels[0].collnetDirect.nHeads*chunkSize) < info->comm->channels[0].collnetDirect.depth*8 && chunkSize > 65536) chunkSize /= 2;
    while (info->nBytes / (info->nChannels*info->comm->channels[0].collnetDirect.nHeads*chunkSize) < info->comm->channels[0].collnetDirect.depth*8 && chunkSize > 32768) chunkSize /= 2;
    // Use lastChunkSize as chunkSize
    work->lastChunkSize = chunkSize / ncclTypeSize(info->datatype);
    // Set direct direction for broadcast-gather (read or write)
    work->direct = (info->nBytes / info->nChannels <= 1024*1024) ? NCCL_DIRECT_WRITE : NCCL_DIRECT_READ;
  } else if (info->algorithm == NCCL_ALGO_COLLNET_CHAIN) {
    stepSize   = info->comm->buffSizes[NCCL_PROTO_SIMPLE]/NCCL_STEPS;
    chunkSize  = std::min(256*1024, stepSize*chunkSteps);
    while (info->nBytes / (info->nChannels*chunkSize) < info->comm->channels[0].collnetChain.depth*64 && chunkSize > 131072) chunkSize /= 2;
    while (info->nBytes / (info->nChannels*chunkSize) < info->comm->channels[0].collnetChain.depth*8 && chunkSize > 65536) chunkSize /= 2;
    while (info->nBytes / (info->nChannels*chunkSize) < info->comm->channels[0].collnetChain.depth && chunkSize > 32768) chunkSize /= 2;
    work->lastChunkSize = chunkSize / ncclTypeSize(info->datatype);
  } else if (info->algorithm == NCCL_ALGO_NVLS) {
    int maxChunkSize = 131072;
    if (chunkSize > maxChunkSize) chunkSize = maxChunkSize;
    // Use uint64_t so that concurrentOps*chunkSize*X does not overflow
    uint64_t concurrentOps = info->nChannels*info->comm->channels[0].nvls.nHeads;
    if ((info->nBytes < (64 * (concurrentOps*chunkSize))) && (chunkSize > 65536)) chunkSize = 65536;
    if ((info->nBytes < (8 * (concurrentOps*chunkSize))) && (chunkSize > 32768)) chunkSize = 32768;
    if ((info->nBytes < (2 * (concurrentOps*chunkSize))) && (chunkSize > 16384)) chunkSize = 16384;
    work->lastChunkSize = chunkSize / ncclTypeSize(info->datatype);
  } else if (info->algorithm == NCCL_ALGO_NVLS_TREE) {
    // Use uint64_t so that concurrentOps*chunkSize*X does not overflow
    uint64_t concurrentOps = info->nChannels*info->comm->channels[0].nvls.nHeads;
    if ((info->nBytes < (32 * (concurrentOps*chunkSize))) && (chunkSize > 262144)) chunkSize = 262144;
    if ((info->nBytes < (16 * (concurrentOps*chunkSize))) && (chunkSize > 131072)) chunkSize = 131072;
    if ((info->nBytes < (4 * (concurrentOps*chunkSize))) && (chunkSize > 65536)) chunkSize = 65536;
    if ((info->nBytes < (1 * (concurrentOps*chunkSize))) && (chunkSize > 32768)) chunkSize = 32768;
    work->lastChunkSize = chunkSize / ncclTypeSize(info->datatype);
  } else if (info->protocol == NCCL_PROTO_LL) {
    const ssize_t sliceSize = stepSize*sizeof(uint64_t)/sizeof(union ncclLLFifoLine);
    const ssize_t loopSize = info->nChannels*info->nchunksPerLoop*(ssize_t)sliceSize;
    work->lastChunkSize = DIVUP((info->nBytes-(info->nBytes/loopSize)*loopSize), info->nChannels*info->nchunksPerLoop);
    ALIGN_SIZE(work->lastChunkSize, info->nThreads*sizeof(uint64_t));
    work->lastChunkSize /= ncclTypeSize(info->datatype);
  } else if (info->algorithm == NCCL_ALGO_TREE && info->protocol == NCCL_PROTO_LL128) {
    int nNodes = info->comm->nNodes;
    float ppn = info->comm->nRanks / (float)nNodes;
    float nstepsLL128 = 1+log2i(nNodes) + 0.1*ppn;
    while (info->nBytes / (info->nChannels*chunkSize) < nstepsLL128*64/ppn && chunkSize > 131072) chunkSize /= 2;
    while (info->nBytes / (info->nChannels*chunkSize) < nstepsLL128*16/ppn && chunkSize > 32768) chunkSize /= 2;
    // Use lastChunkSize as chunkSize
    work->lastChunkSize = chunkSize*NCCL_LL128_DATAELEMS/(NCCL_LL128_LINEELEMS*ncclTypeSize(info->datatype));
  }
  info->chunkSize = chunkSize;
  return ncclSuccess;
}

// Called after computeCollDevWork() on same `info`
static ncclResult_t computeCollProxyWork(
    struct ncclInfo* info, int channelId,
    /*out*/struct ncclProxyWork* proxyWork
  ) {
  if (info->pattern == ncclPatternNone) {
    proxyWork->pattern = ncclProxyWorkPatternNone;
    return ncclSuccess;
  }
  struct ncclComm* comm = info->comm;
  //int stepSize   = comm->buffSizes[info->protocol]/NCCL_STEPS;
  int chunkSteps = info->chunkSteps;
  int sliceSteps = info->sliceSteps;
  int chunkSize  = info->chunkSize;
  int chunkEffectiveSize = chunkSize;
  if (info->protocol == NCCL_PROTO_LL) chunkEffectiveSize /= 2;
  if (info->protocol == NCCL_PROTO_LL128) chunkEffectiveSize = (chunkSize / NCCL_LL128_LINEELEMS) * NCCL_LL128_DATAELEMS;

  int nLoops = (int)divUp(info->nBytes, (((size_t)(info->nChannels))*info->nchunksPerLoop*chunkEffectiveSize));
  int slices = nLoops*info->nstepsPerLoop*(chunkSteps/sliceSteps);
  proxyWork->sliceSteps = sliceSteps;
  proxyWork->protocol = info->protocol;
  switch (info->pattern) {
  case ncclPatternRing:
  case ncclPatternRingTwice:
    { proxyWork->pattern = ncclProxyWorkPatternRing;
      proxyWork->ring.sendSlices = slices;
      proxyWork->ring.recvSlices = slices;
    } break;
  case ncclPatternPipelineFrom:
  case ncclPatternPipelineTo:
    { proxyWork->pattern = ncclProxyWorkPatternRing;
      bool fromNotTo = (info->pattern == ncclPatternPipelineFrom);
      struct ncclRing* ring = &comm->channels[channelId].ring;
      bool isRoot = (comm->rank == info->root);
      bool isLeaf = ((fromNotTo ? ring->next : ring->prev) == info->root);
      proxyWork->ring.sendSlices = (fromNotTo ? !isLeaf : !isRoot) ? slices : 0;
      proxyWork->ring.recvSlices = (fromNotTo ? !isRoot : !isLeaf) ? slices : 0;
    } break;
  case ncclPatternTreeUp:
  case ncclPatternTreeDown:
  case ncclPatternTreeUpDown:
    { proxyWork->pattern = ncclProxyWorkPatternTree;
      proxyWork->tree.up = (info->pattern != ncclPatternTreeDown);
      proxyWork->tree.down = (info->pattern != ncclPatternTreeUp);
      proxyWork->tree.slices = slices;
    } break;
  case ncclPatternCollnetChain:
  case ncclPatternCollnetDirect:
    { proxyWork->pattern = ncclProxyWorkPatternCollnet;
      proxyWork->collnet.directNotChain = (info->pattern == ncclPatternCollnetDirect);
      proxyWork->collnet.slices = slices;
      proxyWork->collnet.dtype = info->datatype;
      proxyWork->collnet.redOp = info->op;
      // Network sees these as sum
      if (info->opFull.op == ncclDevPreMulSum) proxyWork->collnet.redOp = ncclSum;
      if (info->opFull.op == ncclDevSumPostDiv) proxyWork->collnet.redOp = ncclSum;
    } break;
  case ncclPatternNvls:
  case ncclPatternNvlsTree:
    { proxyWork->pattern = ncclProxyWorkPatternNvls;
      proxyWork->nvls.slices = slices;
      proxyWork->nvls.isTree = (info->pattern == ncclPatternNvlsTree);
    } break;
  default:
    return ncclInternalError;
  }
  return ncclSuccess;
}

static ncclResult_t hostToDevRedOp(
    ncclDevRedOpFull *opFull, ncclRedOp_t op, ncclDataType_t datatype, ncclComm *comm
  ) {
  union {
    int8_t i8;
    uint8_t u8;
    int32_t i32;
    uint32_t u32;
    int64_t i64;
    uint64_t u64;
    half f16;
    #if defined(__CUDA_BF16_TYPES_EXIST__)
      __nv_bfloat16 bf16;
    #endif
    float f32;
    double f64;
    void *ptr;
  };
  u64 = 0;
  opFull->scalarArgIsPtr = false;
  switch (int(op)) {
  case ncclSum:  opFull->op = ncclDevSum;  break;
  case ncclProd: opFull->op = ncclDevProd; break;
  case ncclMax:  opFull->op = ncclDevMax;  break;
  case ncclMin:  opFull->op = ncclDevMin;  break;
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

// Converts `info` to a task and adds it to `comm->tasks`. The exception is with
// single rank communicators, collectives are issued as `ncclMemcpyAsync`s and
// thus don't need a task.
static ncclResult_t taskAppend(struct ncclComm* comm, struct ncclInfo const* info) {
  ncclTasks *tasks = &comm->tasks;
  if (info->coll == ncclFuncSend || info->coll == ncclFuncRecv) {
    int peer = info->root;
    ssize_t nBytes = info->count*ncclTypeSize(info->datatype);
    bool isSendNotRecv = info->coll == ncclFuncSend;

    // Must be in thread local group before tasks can be alloc'd in `comm->memScoped`.
    ncclGroupCommJoin(comm);
    struct ncclTaskP2p* p2p = ncclMemoryStackAlloc<struct ncclTaskP2p>(&comm->memScoped);
    p2p->localBuf = (void*)info->recvbuff;
    p2p->bytes = nBytes;
    p2p->chunk = 0;
    ncclIntruQueueEnqueue(
      isSendNotRecv ? &tasks->peers[peer].sendQueue : &tasks->peers[peer].recvQueue,
      p2p);
    tasks->nTasksP2p += 1;
    if (comm->peerInfo[peer].isIntraProc) {
      tasks->p2pIntraPeerMask |= uint64_t(1)<<comm->peerInfo[peer].comm->intraRank;
    }
    // Mark channels that need pre-connect
    if (comm->rank != peer) {
      int channelBaseId;
      NCCLCHECK(ncclChannelComputeBase(comm, peer, isSendNotRecv, &channelBaseId));
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
    // Empty collectives are discarded entirely.
    if (info->count == 0) return ncclSuccess;

    // Copy reduction op state from op handle into info struct here since the
    // op handle may be destroyed before ncclGroupEnd().
    struct ncclDevRedOpFull opFull;
    NCCLCHECK(hostToDevRedOp(&opFull, info->op, info->datatype, comm));

    // User-defined reduction ops may need alter the data even for unitary reductions
    if (comm->nRanks == 1 && opFull.op < ncclDevPreMulSum) {
      if (info->sendbuff != info->recvbuff) {
        size_t bytes = info->count*ncclTypeSize(info->datatype);
        CUDACHECK(cudaMemcpyAsync(info->recvbuff, info->sendbuff, bytes, cudaMemcpyDeviceToDevice, info->stream));
      }
      return ncclSuccess;
    } else {
      // Must be in thread local group before tasks can be alloc'd in `comm->memScoped`.
      ncclGroupCommJoin(info->comm);
      if (info->coll == ncclFuncBroadcast) {
        struct ncclTaskBcast* t = ncclMemoryStackAlloc<struct ncclTaskBcast>(&comm->memScoped);
        t->srcBuf = (void*)info->sendbuff;
        t->dstBuf = (void*)info->recvbuff;
        t->bytes = info->count;
        ncclIntruQueueEnqueue(&tasks->peers[info->root].bcastQueue, t);
        tasks->collLoadBytes += t->bytes;
        tasks->nTasksBcast += 1;
        tasks->minBcastPeer = std::min(tasks->minBcastPeer, info->root);
        tasks->maxBcastPeer = std::max(tasks->maxBcastPeer, info->root);
      } else {
        struct ncclTaskColl* t = ncclMemoryStackAlloc<struct ncclTaskColl>(&comm->memScoped);
        t->func = info->coll;
        t->sendbuff = info->sendbuff;
        t->recvbuff = info->recvbuff;
        t->count = info->count;
        t->root = info->root;
        t->datatype = info->datatype;
        t->op = opFull; // C++ struct assignment
        t->chunkSteps = info->chunkSteps;
        t->sliceSteps = info->sliceSteps;
        ncclIntruQueueEnqueue(&tasks->collQueue, t);
        tasks->collLoadBytes += info->nBytes;
        tasks->nTasksColl += 1;
      }
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
