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
  NCCL_FUNC5(func, NVLS,           devredop, type, specialized)

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

static ncclResult_t computeColl(struct ncclInfo* info /* input */, int* workFuncIndex, struct ncclWorkElem* work, struct ncclProxyOp* proxyOp /* output */);

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

static void appendWorkElemColl(
    struct ncclComm* comm, struct ncclKernelPlan* plan, int channelId,
    int funcIndex, struct ncclWorkElem const *elem, int bid
  ) {
  struct ncclKernelPlan::Channel* chan = &plan->channels[channelId];
  struct ncclWorkList* q = ncclIntruQueueTail(&chan->workQueue);
  if (q && funcIndex == q->work.header.funcIndex
        && elem->nWarps == q->work.elems[0].nWarps
        && chan->nWorkElem < NCCL_MAX_WORK_ELEMENTS) {
    int e = chan->nWorkElem++;
    q->work.elems[e] = *elem; // C++ struct assignment
    q->work.elems[e].bid = bid;
    q->work.elems[e].isUsed = 1;
    return;
  }
  q = ncclMemoryStackAlloc<struct ncclWorkList>(&comm->memScoped);
  q->work.header.type = ncclWorkTypeColl;
  q->work.header.funcIndex = funcIndex;
  q->work.elems[0] = *elem; // C++ struct assignment
  q->work.elems[0].bid = bid;
  q->work.elems[0].isUsed = 1;
  chan->nWorkElem = 1;
  chan->nWork += 1;
  ncclIntruQueueEnqueue(&chan->workQueue, q);
}

static void appendWorkElemColl(
    struct ncclComm* comm, struct ncclKernelPlan* plan, int channelId,
    int funcIndex, struct ncclWorkElemReg const *elem, int bid
  ) {
  struct ncclKernelPlan::Channel* chan = &plan->channels[channelId];
  struct ncclWorkList* q = ncclIntruQueueTail(&chan->workQueue);
  if (q && funcIndex == q->work.header.funcIndex
        && elem->elem.nWarps == q->work.regElems[0].elem.nWarps
        && chan->nWorkElem < NCCL_MAX_WORK_ELEMENTS_REG) {
    int e = chan->nWorkElem++;
    q->work.regElems[e] = *elem; // C++ struct assignment
    q->work.regElems[e].elem.bid = bid;
    q->work.regElems[e].elem.isUsed = 1;
    return;
  }
  q = ncclMemoryStackAlloc<struct ncclWorkList>(&comm->memScoped);
  q->work.header.type = ncclWorkTypeRegColl;
  q->work.header.funcIndex = funcIndex;
  q->work.regElems[0] = *elem; // C++ struct assignment
  q->work.regElems[0].elem.bid = bid;
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
    struct ncclWorkElemP2p const *elem
  ) {
  constexpr int funcIndex = FUNC_INDEX_P2P;
  struct ncclKernelPlan::Channel* chan = &plan->channels[channelId];
  struct ncclWorkList* q = ncclIntruQueueTail(&chan->workQueue);
  if (q && funcIndex == q->work.header.funcIndex) {
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
  q->work.header.funcIndex = FUNC_INDEX_P2P;
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

// Put coll workelem & proxyOp in plan assuming nWorkBudget permits, so please
// ensure *nWorkBudget >= nBids upon entry.
static ncclResult_t addCollToPlan(
    struct ncclComm* comm, struct ncclKernelPlan* plan, int* nWorkBudget, int funcIndex,
    struct ncclWorkElem const* workElem, struct ncclProxyOp const* proxyOp,
    int nCollChannels, int nBid, size_t bytes, bool regBufUsed, void* regBufSend[], void* regBufRecv[]
  ) {
  struct ncclKernelPlan::Channel *chans = plan->channels;

  // Choose the `nBid` least loaded channels to do the work. This ensures
  // all bids go to different channels in case they need to synchronize.
  int least[/*nBid*/MAXCHANNELS];
  least[0] = 0;
  int maxIndexInLeast = 0;
  size_t maxBytesInLeast = chans[0].collBytes;
  // Initialize least[] such that the first nBid channels are accounted for.
  for (int b=1; b < nBid; b++) {
    least[b] = b;
    if (maxBytesInLeast < chans[b].collBytes) {
      maxIndexInLeast = b;
      maxBytesInLeast = chans[b].collBytes;
    }
  }
  // Sort in the rest of the channels. If a channel has less work than the max
  // member of least[], replace that member and compute the new max.
  for (int c=nBid; c < nCollChannels; c++) {
    if (chans[c].collBytes < maxBytesInLeast) {
      least[maxIndexInLeast] = c;
      maxBytesInLeast = chans[least[0]].collBytes;
      maxIndexInLeast = 0;
      for (int b=1; b < nBid; b++) {
        if (maxBytesInLeast < chans[least[b]].collBytes) {
          maxIndexInLeast = b;
          maxBytesInLeast = chans[least[b]].collBytes;
        }
      }
    }
  }

  uint64_t opCount = uint64_t(plan->collOpCount++)<<1 | 0;
  bytes /= nBid;
  for (int bid=0; bid < nBid; bid++) {
    int c = least[bid];
    chans[c].collBytes += bytes;

    // Add work elem
    *nWorkBudget += chans[c].nWork;
    if (!regBufUsed) {
      appendWorkElemColl(comm, plan, c, funcIndex, workElem, bid);
    } else {
      // Buffer registration in play which could only for CollNet at the moment.
      struct ncclChannel* channel = &comm->channels[c];
      struct ncclWorkElemReg workElemReg;
      workElemReg.elem = *workElem; // C++ struct assignment
      workElemReg.elem.regUsed = 1;
      for (int i=0; i < NCCL_MAX_DIRECT_ARITY; i++) {
        int peer = channel->collnetDirect.down[i];
        if (peer == -1) break;
        int j = comm->rankToLocalRank[peer]; // Get intra-node slot
        workElemReg.dnInputs[i] = regBufSend[j]; // Input buffer of leaf peer
        workElemReg.dnOutputs[i] = regBufRecv[j]; // Output buffer of leaf peer
      }
      for (int i=0; i < NCCL_MAX_DIRECT_ARITY; i++) {
        int peer = channel->collnetDirect.up[i];
        if (peer == -1) break;
        int j = comm->rankToLocalRank[peer];
        // Output buffer of root peer
        workElemReg.upOutputs[i] = regBufRecv[j];
      }
      appendWorkElemColl(comm, plan, c, funcIndex, &workElemReg, bid);
    }
    *nWorkBudget -= chans[c].nWork; // subtract delta of chans[c].nWork

    // Add proxy task. Empty collectives do not make it to the proxy thread
    // since they don't imply synchronization for the user like p2p.
    if (proxyOp->nsteps != 0) {
      struct ncclProxyOp tmp = *proxyOp; // C++ struct assignment
      tmp.channelId = c;
      tmp.opCount = opCount;
      NCCLCHECK(addProxyOpIfNeeded(comm, plan, &tmp));
    }
  }
  return ncclSuccess;
}

NCCL_PARAM(P2pLLThreshold, "P2P_LL_THRESHOLD", 16384);

// Put p2p op in plan assuming there is space in nWorkBudget, so you must
// ensure *nWorkBudget >= 1 upon entry.
static ncclResult_t addP2pToPlan(
    struct ncclComm* comm, struct ncclKernelPlan* plan, int* nWorkBudget,
    bool isSendNotRecv, int peer, int chunk, void *addr, size_t bytes
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
    &comm->channels[channelId].peers[peer].send[1].conn : &comm->channels[channelId].peers[peer].recv[1].conn;
  info.protocol = ((conn->buffs[NCCL_PROTO_LL] != nullptr) && bytes <= ncclParamP2pLLThreshold()) ? NCCL_PROTO_LL : NCCL_PROTO_SIMPLE;

  struct ncclProxyOp proxyOp = {};
  NCCLCHECK(ncclProxyComputeP2p(&info, &proxyOp));

  struct ncclWorkElemP2p elem = {0};
  elem.proto = info.protocol;
  elem.peer = peer;
  elem.nWarps = NCCL_MAX_NTHREADS/WARP_SIZE;
  elem.p2pType = isSendNotRecv ? ncclWorkP2pTypeSend : ncclWorkP2pTypeRecv;
  elem.buffLo32 = uint32_t(reinterpret_cast<uintptr_t>(addr));
  elem.buffHi32 = reinterpret_cast<uintptr_t>(addr)>>32;
  elem.countLo32 = uint32_t(bytes);
  elem.countHi32 = bytes>>32;
  elem.chunkSize = info.chunkSize; // computed by ncclProxyComputeP2p

  *nWorkBudget += plan->channels[channelId].nWork;
  appendWorkElemP2p(comm, plan, channelId, &elem);
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

static ncclResult_t registerIntraNodeBuffers(
    struct ncclComm* comm, struct ncclKernelPlan* plan, struct ncclInfo* info,
    bool* outRegBufUsed,
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
        ncclIntruQueueEnqueue(&plan->ipcMemQueue, q);
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

static ncclResult_t scheduleCollTasksToPlan(
    struct ncclComm* comm, struct ncclKernelPlan* plan, int* nWorkBudget
  ) {
  struct ncclTasks* tasks = &comm->tasks;

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
    while (tasks->collBytesTotal < bytePerChannel[collNetSupport]*comm->nChannels &&
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
        int maxChannels = aggInfo.algorithm == NCCL_ALGO_NVLS ? comm->nvlsChannels : comm->nChannels;
        info.nChannels = DIVUP(info.nBytes, bytePerChannel[collNetSupport]);
        info.nChannels = std::max(1, std::min(info.nChannels, maxChannels));
        info.algorithm = aggInfo.algorithm;
        info.protocol = aggInfo.protocol;
        info.nThreads = aggInfo.nThreads;
      }

      int workFuncIndex;
      struct ncclWorkElem workElem = {};
      struct ncclProxyOp proxyOp = {};
      NCCLCHECK(computeColl(&info, &workFuncIndex, &workElem, &proxyOp));

      if (*nWorkBudget < info.nChannels) return ncclSuccess; // Ensure room for addCollToPlan()

      bool regBufUsed = false;
      void* regBufSend[NCCL_MAX_LOCAL_RANKS];
      void* regBufRecv[NCCL_MAX_LOCAL_RANKS];
      if (plan->persistent && ncclParamGraphRegister() &&
          info.algorithm == NCCL_ALGO_COLLNET_DIRECT &&   // limited to CollNetDirect for now
          comm->intraHighestTransportType == TRANSPORT_P2P && // only when all ranks can p2p each other
          comm->intraRanks < comm->localRanks) { // only with inter-process & intra-node peers
        NCCLCHECK(registerIntraNodeBuffers(comm, plan, &info, &regBufUsed, regBufSend, regBufRecv));
      }

      int maxChannels = info.algorithm == NCCL_ALGO_NVLS ? comm->nvlsChannels : comm->nChannels;
      NCCLCHECK(addCollToPlan(comm, plan, nWorkBudget, workFuncIndex, &workElem, &proxyOp,
        maxChannels, info.nChannels, info.nBytes, regBufUsed, regBufSend, regBufRecv));
      tasks->nTasksColl -= 1;
      tasks->collBytesTotal -= info.nBytes;
      ncclIntruQueueDequeue(&tasks->collQueue);
      head = ncclIntruQueueHead(&tasks->collQueue);

      plan->threadPerBlock = std::max(plan->threadPerBlock, info.nThreads);
      if (!plan->kernelSpecialized) {
        plan->kernelFn = ncclKerns[workFuncIndex].kernelFn;
        plan->kernelSpecialized = ncclKerns[workFuncIndex].specialized;
      }
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
    plan->kernelFn = ncclKerns[FUNC_INDEX_P2P].kernelFn;
    plan->kernelSpecialized = ncclKerns[FUNC_INDEX_P2P].specialized;
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

  while (tasks->nTasksP2p != 0) {
    for (int i=0; i < nRanks; i++) {
      int sendPeer = sendOrder[i];
      int recvPeer = recvOrder[i];
      struct ncclTaskP2p* send = ncclIntruQueueHead(&peers[sendPeer].sendQueue);
      struct ncclTaskP2p* recv = ncclIntruQueueHead(&peers[recvPeer].recvQueue);
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
        ssize_t minSize = stepSize/8;
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
          ssize_t recvChunkBytes = std::min(recvBytes, recvChunkBytesMax); // -1 preserved
          ssize_t sendChunkBytes = std::min(sendBytes, sendChunkBytesMax);
          if (recvChunkBytes != 0) {
            if (recvChunkBytes == -1) recvChunkBytes = 0;
            if (*nWorkBudget < 1) return ncclSuccess; // ensure room in budget
            NCCLCHECK(addP2pToPlan(comm, plan, nWorkBudget, /*isSendNotRecv=*/false, recvPeer, recv->chunk, recvPtr, recvChunkBytes));
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
            NCCLCHECK(addP2pToPlan(comm, plan, nWorkBudget, /*isSendNotRecv=*/true, sendPeer, send->chunk, sendPtr, sendChunkBytes));
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
  uint64_t collOpCount = comm->collOpCount;
  // Advance comm's collOpCount by number of colls in this plan.
  comm->collOpCount = collOpCount + plan->collOpCount;
  for (int c=0; c < plan->channelUbound; c++) {
    struct ncclProxyOp* q = ncclIntruQueueHead(&plan->channels[c].proxyOpQueue);
    uint64_t p2pOpCount = comm->channels[c].p2pOpCount;
    uint64_t nextP2pOpCount = p2pOpCount;
    while (q != nullptr) {
      struct ncclProxyOp* qNext = q->enqNext;
      // Ignoring the bottom tag bit, opCount's are zero-based within plan so
      // translate them to the tip of the comm's history.
      if (q->opCount & 1) { // p2p
        // p2pOpCount is monotonic increasing within a plan's channel so just
        // remember last value to compute max.
        nextP2pOpCount = p2pOpCount + (q->opCount>>1);
        nextP2pOpCount += 1; // +1 to ensure next plan doesn't collide
        q->opCount = (p2pOpCount<<1) + q->opCount;
      } else { // coll
        q->opCount = (collOpCount<<1) + q->opCount;
      }
      NCCLCHECK(ncclProxySaveOp(comm, q, nullptr)); // May overwrite enqNext.
      if (!plan->persistent) {
        // Non-persistent kernels have their memory reclaimed after upload.
        ncclMemoryPoolFree(&plan->memPool_ncclProxyOp, q);
      }
      q = qNext;
    }
    // Advance channel's p2pOpCount by number of p2p's in this plan channel.
    comm->channels[c].p2pOpCount = nextP2pOpCount;
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
    while (!ncclIntruQueueEmpty(&plan->ipcMemQueue)) {
      struct ncclPointerList* q = ncclIntruQueueDequeue(&plan->ipcMemQueue);
      CUDACHECKIGNORE(cudaIpcCloseMemHandle(q->ptr));
      ncclMemoryPoolFree(&comm->memPool_ncclPointerList, q);
    }
  }
  ncclMemoryPoolTakeAll(&comm->memPool_ncclProxyOp, &plan->memPool_ncclProxyOp);
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
    NCCLCHECKGOTO(ncclStrongStreamAcquire(tasks->capturingGraph, &comm->deviceStream), result, failure);

    // Create dependency for device stream on user streams. First from extra user
    // streams to deviceStream. Then deviceStream to first user stream.
    for (struct ncclCudaStreamList* l=tasks->streams->next; l != nullptr; l = l->next) {
      NCCLCHECKGOTO(ncclStrongStreamWaitStream(tasks->capturingGraph, &comm->deviceStream, l->stream), result, failure);
    }
    NCCLCHECKGOTO(ncclStrongStreamWaitStream(tasks->capturingGraph, launchStream, &comm->deviceStream), result, failure);

    if (persistent || comm->persistentRefs != 0 || ncclCudaLaunchBlocking) {
      // We have to launch host tasks to push proxy args. We are careful to only
      // do this if necessary since host tasks impose a high performance cost in CUDA.
      bool acquired = false;
      for (struct ncclKernelPlan* plan=planHead; plan != nullptr; plan = plan->next) {
        if (plan->hasProxyOps) {
          if (!acquired) {
            acquired = true;
            NCCLCHECKGOTO(ncclStrongStreamAcquire(tasks->capturingGraph, &comm->hostStream), result, failure);
          }
          NCCLCHECKGOTO(ncclStrongStreamLaunchHost(tasks->capturingGraph, &comm->hostStream, hostStreamPlanCallback, plan), result, failure);
        }
      }
      if (acquired) {
        // Make to-be-launched kernels dependent on just-launched host stream tasks.
        NCCLCHECKGOTO(ncclStrongStreamWaitStream(tasks->capturingGraph, launchStream, &comm->hostStream), result, failure);
        NCCLCHECKGOTO(ncclStrongStreamRelease(tasks->capturingGraph, &comm->hostStream), result, failure);
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
    unsigned int clusterSize = (compCap == 90) ? comm->cgaClusterSize : 0;

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
    // If this isn't being captured and there aren't any CUDA graphs alive
    // then we don't need to do our proxyOp pushing on the host stream.
    NCCLCHECK(hostStreamPlanTask(comm, plan));
  }
  return ncclSuccess;
}

ncclResult_t ncclLaunchFinish(struct ncclComm* comm) {
  ncclResult_t result = ncclSuccess;
  struct ncclTasks* tasks = &comm->tasks;
  tasks->collBytesTotal = 0; // Just in case subtraction during scheduleCollTasksToPlan() doesn't get to 0

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
    NCCLCHECKGOTO(ncclStrongStreamWaitStream(tasks->capturingGraph, &comm->deviceStream, launchStream, /*b_subsumes_a=*/true), result, resume1);
  resume1:
    // Create dependency for other user streams (skip launch stream) on deviceStream.
    // Again, the user streams haven't been touched since deviceStream waited on them
    // so we can say they are subsumed by deviceStream.
    struct ncclCudaStreamList* sl = tasks->streams->next;
    tasks->streams = nullptr; // Reset comm->tasks.streams to empty.
    while (sl != nullptr) {
      NCCLCHECKGOTO(ncclStrongStreamWaitStream(tasks->capturingGraph, sl->stream, &comm->deviceStream, /*b_subsumes_a=*/true), result, resume2);
    resume2:
      sl = sl->next;
    }
    // Release device stream as acquired in ncclLaunchPrepare()
    NCCLCHECKGOTO(ncclStrongStreamRelease(tasks->capturingGraph, &comm->deviceStream), result, resume3);
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
  } else if (info->algorithm == NCCL_ALGO_NVLS) {
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
    nt += WARP_SIZE; // Extra warp for sync
    // More threads or sync warps needed due to split thread model
    if (info->algorithm == NCCL_ALGO_TREE) nt += 3*WARP_SIZE;
    if (info->algorithm == NCCL_ALGO_COLLNET_DIRECT) nt += 3*WARP_SIZE;
    if (info->algorithm == NCCL_ALGO_COLLNET_CHAIN) nt += 3*WARP_SIZE;
    if (info->algorithm == NCCL_ALGO_NVLS) nt = NCCL_MAX_NTHREADS;
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
      info->pattern = ncclPatternRing; break;
    case ncclFuncAllReduce:
      info->pattern =
        info->algorithm == NCCL_ALGO_NVLS ? ncclPatternNvls :
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
    case ncclPatternTreeUp:
    case ncclPatternTreeDown:
    case ncclPatternTreeUpDown:
    case ncclPatternPipelineFrom:
    case ncclPatternPipelineTo:
    case ncclPatternCollnetChain:
    case ncclPatternNvls:
      info->nstepsPerLoop = info-> nchunksPerLoop = 1; break;
    case ncclPatternCollnetDirect:
      info->nstepsPerLoop = 1; info->nchunksPerLoop = info->comm->channels[0].collnetDirect.nHeads; break;
    case ncclPatternRing:
      info->nstepsPerLoop = info->comm->nRanks-1; info->nchunksPerLoop = info->comm->nRanks; break;
    case ncclPatternRingTwice:
      info->nstepsPerLoop = 2*(info->comm->nRanks-1); info->nchunksPerLoop = info->comm->nRanks; break;
    default:
      WARN("Unknown pattern %d", info->pattern);
      return ncclInternalError;
  }
  return ncclSuccess;
}

static ncclResult_t computeColl(struct ncclInfo* info /* input */, int* workFuncIndex, struct ncclWorkElem* work, struct ncclProxyOp* proxyOp /* output */) {
  int collNetTypeSupport = 0;
  // Check whether algo and proto have been preset (as in aggregation case)
  // If so, skip the calculation
  if (info->nChannels > 0 && info->nThreads > 0) goto comp_next;
  NCCLCHECK(getCollNetSupport(info, &collNetTypeSupport));
  NCCLCHECK(getAlgoInfo(info, collNetTypeSupport, 1));

comp_next:
  // Set nstepsPerLoop and nchunksPerLoop
  NCCLCHECK(getPatternInfo(info));
  NCCLCHECK(getLoopInfo(info));

  work->sendbuff = info->sendbuff;
  work->recvbuff = info->recvbuff;
  work->root = info->root;
  work->count = info->count;
  work->nChannels = info->nChannels;
  work->nWarps = info->nThreads / WARP_SIZE;
  work->redOpArg = info->opFull.scalarArg;
  work->redOpArgIsPtr = info->opFull.scalarArgIsPtr;

  if (info->comm->nRanks == 1) {
    // one-rank reduce index
    *workFuncIndex = 1 + int(info->datatype);
    return ncclSuccess;
  }

  *workFuncIndex = FUNC_INDEX(info->coll, info->opFull.op, info->datatype, info->algorithm, info->protocol);

  int stepSize   = info->comm->buffSizes[info->protocol]/NCCL_STEPS;
  int chunkSteps = (info->protocol == NCCL_PROTO_SIMPLE && info->algorithm == NCCL_ALGO_RING) ? info->chunkSteps : 1;
  int sliceSteps = (info->protocol == NCCL_PROTO_SIMPLE && info->algorithm == NCCL_ALGO_RING) ? info->sliceSteps : 1;
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
    if (chunkSize > 131072) chunkSize = 131072;
    // Use uint64_t so that concurrentOps*chunkSize*X does not overflow
    uint64_t concurrentOps = info->nChannels*info->comm->channels[0].nvls.nHeads;
    if ((info->nBytes < (32 * (concurrentOps*chunkSize))) && (chunkSize > 65536)) chunkSize = 65536;
    if ((info->nBytes < (8 * (concurrentOps*chunkSize))) && (chunkSize > 32768)) chunkSize = 32768;
    if ((info->nBytes < (2 * (concurrentOps*chunkSize))) && (chunkSize > 16384)) chunkSize = 16384;
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

  // Compute nSteps for proxies
  int chunkEffectiveSize = chunkSize;
  if (info->protocol == NCCL_PROTO_LL) chunkEffectiveSize /= 2;
  if (info->protocol == NCCL_PROTO_LL128) chunkEffectiveSize = (chunkSize / NCCL_LL128_LINEELEMS) * NCCL_LL128_DATAELEMS;
  //if (info->comm->rank == 0) printf("Coll %d, size %ld -> %dx%d, chunkSize %d (algo %d proto%d)\n", info->coll, info->nBytes, info->nChannels, info->nThreads, chunkSize, info->algorithm, info->protocol);
  int nLoops = (int)(DIVUP(info->nBytes, (((size_t)(info->nChannels))*info->nchunksPerLoop*chunkEffectiveSize)));
  proxyOp->nsteps = info->nstepsPerLoop * nLoops * chunkSteps;
  proxyOp->sliceSteps = sliceSteps;
  proxyOp->chunkSteps = chunkSteps;
  proxyOp->chunkSize = chunkSize;
  proxyOp->protocol = info->protocol;
  proxyOp->dtype = info->datatype;
  proxyOp->redOp = (info->algorithm != NCCL_ALGO_COLLNET_DIRECT && info->algorithm != NCCL_ALGO_COLLNET_CHAIN) ? ncclNumOps : // Only set redOp when using CollNet
                     info->opFull.op==ncclDevPreMulSum || info->opFull.op==ncclDevSumPostDiv ? ncclSum : // Network sees avg as sum
                     info->op;
  proxyOp->pattern = info->pattern;
  proxyOp->root = info->root;
  // This is used by P2P to reduce the receive buffer size. We don't use it in collectives
  // because some protocols need to transmit more than the total size, plus they sometimes
  // round up
  proxyOp->nbytes = stepSize*proxyOp->sliceSteps;

  TRACE(NCCL_COLL,"opCount %lx slicesteps %d spl %d cpl %d nbytes %zi -> protocol %d nchannels %d nthreads %d, nloops %d nsteps %d chunksize %d comm %p",
      proxyOp->opCount, sliceSteps, info->nstepsPerLoop, info->nchunksPerLoop, info->nBytes, info->protocol, info->nChannels, info->nThreads,
      nLoops, proxyOp->nsteps, chunkSize, info->comm);
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
            if (comm->channels[channelId].peers[peer].send[1].connected == 0) { // P2P uses only 1 connector
              comm->connectSend[peer] |= (1UL<<channelId);
              ncclGroupCommPreconnect(comm);
            }
          } else {
            if (comm->channels[channelId].peers[peer].recv[1].connected == 0) { // P2P uses only 1 connector
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
      tasks->collBytesTotal += t->count*ncclTypeSize(t->datatype);
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
  if (info->comm && !info->comm->blocking) { NCCLCHECK(ncclCommGetAsyncError(info->comm, &ret)) };
  return ret;
fail:
  if (info->comm && !info->comm->blocking) (void) ncclCommSetAsyncError(info->comm, ret);
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
