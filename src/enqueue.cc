/*************************************************************************
 * Copyright (c) 2017-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "enqueue.h"
#include "argcheck.h"
#include "coll_net.h"

// Only generate inline kernels for LL
#define NCCL_FUNC5(func, algo, redop, dtype) \
  (void*)NCCL_KERN_NAME(func, algo, LL, redop, dtype), \
  (void*)NCCL_KERN_NAME(func, algo, LL, redop, dtype), \
  (void*)NCCL_KERN_NAME(func, algo, LL, redop, dtype)

#define NCCL_FUNC4(func, redop, type) \
  (void*)NCCL_FUNC5(func, TREE,    redop, type), \
  (void*)NCCL_FUNC5(func, RING,    redop, type), \
  (void*)NCCL_FUNC5(func, COLLNET, redop, type)

// Must be consistent with ncclDataType_t
#define NCCL_FUNCS3A(func, redop) \
  (void*)NCCL_FUNC4(func, redop, int8_t), \
  (void*)NCCL_FUNC4(func, redop, uint8_t), \
  (void*)NCCL_FUNC4(func, redop, int32_t), \
  (void*)NCCL_FUNC4(func, redop, uint32_t), \
  (void*)NCCL_FUNC4(func, redop, int64_t), \
  (void*)NCCL_FUNC4(func, redop, uint64_t), \
  (void*)NCCL_FUNC4(func, redop, half), \
  (void*)NCCL_FUNC4(func, redop, float), \
  (void*)NCCL_FUNC4(func, redop, double)
#define NCCL_FUNCS3B(func, redop) \
  (void*)NCCL_FUNC4(func, redop, int8_t), \
  (void*)NCCL_FUNC4(func, redop, int8_t), \
  (void*)NCCL_FUNC4(func, redop, int8_t), \
  (void*)NCCL_FUNC4(func, redop, int8_t), \
  (void*)NCCL_FUNC4(func, redop, int8_t), \
  (void*)NCCL_FUNC4(func, redop, int8_t), \
  (void*)NCCL_FUNC4(func, redop, int8_t), \
  (void*)NCCL_FUNC4(func, redop, int8_t), \
  (void*)NCCL_FUNC4(func, redop, int8_t)

// Must be consistent with ncclRedOp_t -- but we only generate kernel for sums.
#define NCCL_FUNCS2A(func) \
  NCCL_FUNCS3A(func, Sum), \
  NCCL_FUNCS3A(func, Sum), \
  NCCL_FUNCS3A(func, Sum), \
  NCCL_FUNCS3A(func, Sum)
#define NCCL_FUNCS2B(func) \
  NCCL_FUNCS3B(func, Sum), \
  NCCL_FUNCS3B(func, Sum), \
  NCCL_FUNCS3B(func, Sum), \
  NCCL_FUNCS3B(func, Sum)

// Must be consistent with the ncclFuncSet enum
static void* const ncclKerns[1+NCCL_NUM_FUNCTIONS*ncclNumOps*ncclNumTypes*NCCL_NUM_ALGORITHMS*NCCL_NUM_PROTOCOLS] = {
  (void*)NCCL_KERN_NAME(SendRecv, RING, SIMPLE, Sum, int8_t),
  NCCL_FUNCS2B(Broadcast),
  NCCL_FUNCS2A(Reduce),
  NCCL_FUNCS2B(AllGather),
  NCCL_FUNCS2A(ReduceScatter),
  NCCL_FUNCS2A(AllReduce)
};

/*****************************************************************************/
/*       Launch system : synchronization and CUDA kernel launch              */
/*****************************************************************************/

ncclResult_t ncclLaunchCooperativeKernelMultiDevice(struct cudaLaunchParams *paramsList, int* cudaDevs, int numDevices, int cgMode) {
#if CUDART_VERSION >= 9000
  if (cgMode & 0x01) {
    CUDACHECK(cudaLaunchCooperativeKernelMultiDevice(paramsList, numDevices,
            // These flags are to reduce the latency of using this API
            cudaCooperativeLaunchMultiDeviceNoPreSync|cudaCooperativeLaunchMultiDeviceNoPostSync));
    return ncclSuccess;
  }
#endif
  int savedDev;
  CUDACHECK(cudaGetDevice(&savedDev));
  for (int i = 0; i < numDevices; i++) {
    struct cudaLaunchParams* params = paramsList+i;
    CUDACHECK(cudaSetDevice(cudaDevs[i]));
    CUDACHECK(cudaLaunchKernel(params->func, params->gridDim, params->blockDim, params->args, params->sharedMem, params->stream));
  }
  CUDACHECK(cudaSetDevice(savedDev));
  return ncclSuccess;
}

static ncclResult_t getNextOp(struct ncclChannel* channel, struct ncclWork** work, struct ncclWorkElem* base) {
  if (channel->workCount == NCCL_MAX_OPS) {
    WARN("Too many aggregated operations on channel %d (%d max)", channel->id, NCCL_MAX_OPS);
    return ncclInvalidUsage;
  }
  int opIndex = channel->workFifoTail%NCCL_MAX_OPS;
  struct ncclWork* w = channel->workFifo+opIndex;
  struct ncclWorkElem* e = w->elems;
  volatile uint8_t* activePtr = (volatile uint8_t*)&e->active;
  while (activePtr[0] != 0) sched_yield();
  memset(w, 0, sizeof(struct ncclWork));
  // Initialize with work elem if provided
  if (base) memcpy(e, base, sizeof(struct ncclWorkElem));
  e->active = 1;
  e->index = opIndex;
  channel->workFifoTail++;
  channel->workCount++;
  if (work) *work = w;
  return ncclSuccess;
}

static ncclResult_t setupLaunch(struct ncclComm* comm, struct cudaLaunchParams* params) {
  // Only launch blocks where we have work to do.
  for (int c=0; c<comm->p2pnChannels; c++) {
    if (comm->channels[c].workCount) params->gridDim.x = c+1;
  }

  // Set active = 2 for the last operation and add a no-op on empty channels (p2p case).
  for (int c=0; c<params->gridDim.x; c++) {
    struct ncclChannel* channel = comm->channels+c;
    if (channel->workCount == 0) {
      struct ncclWork* w;
      NCCLCHECK(getNextOp(channel, &w, NULL));
      struct ncclWorkElem* e = w->elems;
      e->comm = comm->devComm;
      e->funcIndex = FUNC_INDEX_P2P;
      e->p2p.nThreads = 0;
    }
    channel->workFifo[(channel->workFifoTail-1)%NCCL_MAX_OPS].elems[0].active = 2;
  }

  // Find the first operation, choose the kernel accordingly and pass it
  // as the first argument.
  struct ncclChannel* c0 = comm->channels;
  struct ncclWork* work = c0->workFifo+((c0->workFifoTail-c0->workCount)%NCCL_MAX_OPS);
  struct ncclWorkElem* elem = work->elems;
  memcpy(&comm->args, elem, sizeof(struct ncclWorkElem));
  // As we inline the first coll directly, we can free it immediately.
  if (elem->funcIndex != FUNC_INDEX_P2P) elem->active = 0;

  params->func = ncclKerns[elem->funcIndex];
  return ncclSuccess;
}

ncclResult_t ncclCpuBarrierIn(struct ncclComm* comm, int* isLast) {
  volatile int* ptr = (volatile int*)(comm->intraBarrier+comm->intraPhase);
  int val = *ptr;
  bool done = false;
  while (done == false) {
    if (val >= comm->intraRanks) {
      WARN("Trying to launch too many work elements, max is %d", NCCL_MAX_OPS);
      return ncclInvalidUsage;
    }
    if (val+1 == comm->intraRanks) {
      // Reset the barrier.
      comm->intraBarrier[comm->intraPhase^1] = 0;
      *isLast = 1;
      return ncclSuccess;
    }
    done = __sync_bool_compare_and_swap(ptr, val, val+1);
    val++;
  }
  *isLast = 0;
  return ncclSuccess;
}

ncclResult_t ncclCpuBarrierLast(struct ncclComm* comm) {
  volatile int* ptr = (volatile int*)(comm->intraBarrier+comm->intraPhase);
  int val = *ptr;
  if (__sync_bool_compare_and_swap(ptr, val, val+1) != true) {
    WARN("Trying to launch too many work elements, max is %d", NCCL_MAX_OPS);
    return ncclInternalError;
  }
  return ncclSuccess;
}

ncclResult_t ncclCpuBarrierOut(struct ncclComm* comm) {
  volatile int* ptr = (volatile int*)(comm->intraBarrier+comm->intraPhase);
  while (*ptr < comm->intraRanks) pthread_yield();
  comm->intraPhase ^= 1;
  return ncclSuccess;
}

ncclResult_t ncclBarrierEnqueue(struct ncclComm* comm) {
  struct cudaLaunchParams* params = comm->myParams;
  if (params->gridDim.x == 0) return ncclSuccess;

  NCCLCHECK(setupLaunch(comm, params));

  // Use internal NCCL stream for CGMD/GROUP launch if required or if the user stream is NULL
  if (comm->launchMode == ncclComm::GROUP && (comm->groupCudaStream || comm->userStream == NULL)) {
    // Enqueue event in user stream
    CUDACHECK(cudaEventRecord(comm->doneEvent, comm->userStream));
    // Create dependency between user stream and internal NCCL stream
    CUDACHECK(cudaStreamWaitEvent(comm->groupStream, comm->doneEvent, 0));
    params->stream = comm->groupStream;
  } else {
    if (comm->userStream != params->stream) {
      // Stream changed from last call, create dependency against last NCCL kernel launch
      CUDACHECK(cudaStreamWaitEvent(comm->userStream, comm->doneEvent, 0));
    }
    params->stream = comm->userStream;
  }

  if (comm->launchMode == ncclComm::GROUP) {
    int isLast = 0;
    NCCLCHECK(ncclCpuBarrierIn(comm, &isLast));
    if (isLast) {
      // I'm the last. Launch all operations.
      NCCLCHECK(ncclLaunchCooperativeKernelMultiDevice(comm->intraParams, comm->intraCudaDevs, comm->intraRanks, *comm->intraCGMode));
      NCCLCHECK(ncclCpuBarrierLast(comm));
    }
  }
  return ncclSuccess;
}

ncclResult_t ncclBarrierEnqueueWait(ncclComm_t comm) {
  struct cudaLaunchParams *params = comm->myParams;
  if (params->gridDim.x == 0) return ncclSuccess;

  // We can't print the CG mode before the first barrier happened.
  if (comm->rank == 0 && *comm->intraCGMode & 0x10) {
    *comm->intraCGMode ^= 0x10;
    INFO(NCCL_INIT,"Launch mode %s%s%s",
        comm->launchMode == ncclComm::GROUP ? "Group" : "Parallel",
        *comm->intraCGMode ? "/CGMD" : "",
        (comm->launchMode == ncclComm::GROUP && comm->groupCudaStream) ? "/Stream" : "");
  }


  if (comm->launchMode == ncclComm::PARALLEL) {
    CUDACHECK(cudaLaunchKernel(params->func, params->gridDim, params->blockDim, params->args, params->sharedMem, params->stream));
  } else {
    NCCLCHECK(ncclCpuBarrierOut(comm));
  }

  // Start the network proxies as soon as the kernel has been launched. We can't
  // perform any CUDA call between the two or having a cudaFree between the CUDA
  // launch and the ncclProxyStart call could cause a deadlock.
  // Also, starting the proxies after the CUDA launch seems to be better for
  // performance (latency).
  uint64_t max = 0ULL;
  for (int r=0; r<params->gridDim.x; r++) {
    struct ncclChannel* channel = comm->channels+r;
    max = std::max(max, channel->workFifoTail);
    channel->workCount = 0;
  }
  for (int r=0; r<comm->p2pnChannels; r++) {
    struct ncclChannel* channel = comm->channels+r;
    channel->workFifoTail = max;
  }
  params->gridDim.x = params->blockDim.x = 0;
  comm->lastOpCount = max;
  NCCLCHECK(ncclProxyStart(comm));
  return ncclSuccess;
}

ncclResult_t ncclEnqueueEvents(ncclComm_t comm) {
  struct cudaLaunchParams *params = comm->myParams;
  // Enqueue event after NCCL kernel
  CUDACHECK(cudaEventRecord(comm->doneEvent, params->stream));
  // Use internal NCCL stream for CGMD/GROUP launch if required or if the user stream is NULL
  if (comm->launchMode == ncclComm::GROUP && (comm->groupCudaStream || comm->userStream == NULL)) {
    // Create dependency between NCCL internal stream and user stream
    CUDACHECK(cudaStreamWaitEvent(comm->userStream, comm->doneEvent, 0));
  }
  comm->userStreamSet = false;
  return ncclSuccess;
}

/*****************************************************************************/
/* Enqueueing system : computation of kernel and proxy operations parameters */
/*****************************************************************************/

static ncclResult_t getAlgoInfo(struct ncclInfo* info) {
  struct ncclComm* comm = info->comm;
  float minTime = 3600000000.0; // Hopefully no operation will take an hour to complete.
  // Find algorithm / protocol.
  info->algorithm = -1;
  info->protocol = -1;
  int nAlgos = NCCL_NUM_ALGORITHMS;
  // Check collNet support
  int collNetTypeSupport = 0;
  if (info->comm->collNetSupport)
    NCCLCHECK(collNetReduceSupport(info->datatype, info->op, &collNetTypeSupport));
  if (collNetTypeSupport != 1) nAlgos--;
  for (int a=0; a<nAlgos; a++) {
    for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
      float time;
      NCCLCHECK(ncclTopoGetAlgoTime(info, a, p, &time));
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

  int nc = (info->nChannels > 0) ? info->nChannels :
           (info->algorithm == NCCL_ALGO_COLLNET) ? comm->nChannels/2 : comm->nChannels; // CollNet uses one channel for up and one channel for down
  int nt = comm->maxThreads[info->algorithm][info->protocol];
  int threadThreshold = comm->threadThresholds[info->algorithm][info->protocol];
  while (info->nBytes < nc*nt*threadThreshold) {
    if (info->algorithm != NCCL_ALGO_COLLNET && nc >= 2) nc--;
    else if ((nt % 128) == 0) nt/=2;
    else break;
  }
  if (info->protocol == NCCL_PROTO_SIMPLE) nt += WARP_SIZE; // Extra warp for sync
  if (info->protocol == NCCL_PROTO_SIMPLE && info->algorithm == NCCL_ALGO_TREE) nt += WARP_SIZE;
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
      info->pattern = info->algorithm == NCCL_ALGO_COLLNET ? ncclPatternCollTreeUp : info->algorithm == NCCL_ALGO_TREE ? ncclPatternTreeUpDown : ncclPatternRingTwice; break;
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
    case ncclPatternCollTreeUp:
    case ncclPatternCollTreeDown:
      info->nstepsPerLoop = info-> nchunksPerLoop = 1; break;
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

static ncclResult_t computeColl(struct ncclInfo* info /* input */, struct ncclWorkElem* work, struct ncclProxyArgs* proxyArgs /* output */) {
  work->comm = info->comm->devComm;

  // Set nstepsPerLoop and nchunksPerLoop
  NCCLCHECK(getAlgoInfo(info));
  NCCLCHECK(getPatternInfo(info));
  NCCLCHECK(getLoopInfo(info));

  work->sendbuff = info->sendbuff;
  work->recvbuff = info->recvbuff;
  work->coll.root = info->root;
  work->coll.count = info->count;
  work->coll.nChannels = info->nChannels;
  work->nThreads = info->nThreads;

  work->funcIndex = FUNC_INDEX(info->coll, info->op, info->datatype, info->algorithm, info->protocol);

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
    work->coll.lastChunkSize = chunkSize / ncclTypeSize(info->datatype);
  } else if (info->algorithm == NCCL_ALGO_COLLNET && info->protocol == NCCL_PROTO_SIMPLE) {
    // Optimize chunkSize / nSteps
    while (info->nBytes / (info->nChannels*chunkSize) < info->comm->channels[0].collTree.depth*16 && chunkSize > 131072) chunkSize /= 2;
    while (info->nBytes / (info->nChannels*chunkSize) < info->comm->channels[0].collTree.depth*4 && chunkSize > 65536) chunkSize /= 2;
    while (info->nBytes / (info->nChannels*chunkSize) < info->comm->channels[0].collTree.depth && chunkSize > 32768) chunkSize /= 2;
    // Use lastChunkSize as chunkSize
    work->coll.lastChunkSize = chunkSize / ncclTypeSize(info->datatype);
  } else if (info->protocol == NCCL_PROTO_LL) {
    const ssize_t sliceSize = stepSize*sizeof(uint64_t)/sizeof(union ncclLLFifoLine);
    const ssize_t loopSize = info->nChannels*info->nchunksPerLoop*(ssize_t)sliceSize;
    work->coll.lastChunkSize = DIVUP((info->nBytes-(info->nBytes/loopSize)*loopSize), info->nChannels*info->nchunksPerLoop);
    ALIGN_SIZE(work->coll.lastChunkSize, info->nThreads*sizeof(uint64_t));
    work->coll.lastChunkSize /= ncclTypeSize(info->datatype);
  } else if (info->algorithm == NCCL_ALGO_TREE && info->protocol == NCCL_PROTO_LL128) {
    int nNodes = info->comm->nNodes;
    float ppn = info->comm->nRanks / (float)nNodes;
    float nstepsLL128 = 1+log2i(nNodes) + 0.1*ppn;
    while (info->nBytes / (info->nChannels*chunkSize) < nstepsLL128*64/ppn && chunkSize > 131072) chunkSize /= 2;
    while (info->nBytes / (info->nChannels*chunkSize) < nstepsLL128*16/ppn && chunkSize > 32768) chunkSize /= 2;
    // Use lastChunkSize as chunkSize
    work->coll.lastChunkSize = chunkSize*NCCL_LL128_DATAELEMS/(NCCL_LL128_LINEELEMS*ncclTypeSize(info->datatype));
  }

  // Compute nSteps for proxies
  int chunkEffectiveSize = chunkSize;
  if (info->protocol == NCCL_PROTO_LL) chunkEffectiveSize /= 2;
  if (info->protocol == NCCL_PROTO_LL128) chunkEffectiveSize = (chunkSize / NCCL_LL128_LINEELEMS) * NCCL_LL128_DATAELEMS;
  //if (info->comm->rank == 0) printf("Coll %d, size %ld -> %dx%d, chunkSize %d (algo %d proto%d)\n", info->coll, info->nBytes, info->nChannels, info->nThreads, chunkSize, info->algorithm, info->protocol);
  int nLoops = (int)(DIVUP(info->nBytes, (((size_t)(info->nChannels))*info->nchunksPerLoop*chunkEffectiveSize)));
  proxyArgs->nsteps = info->nstepsPerLoop * nLoops * chunkSteps;
  proxyArgs->sliceSteps = sliceSteps;
  proxyArgs->chunkSteps = chunkSteps;
  proxyArgs->protocol = info->protocol;
  proxyArgs->dtype = info->datatype;
  proxyArgs->redOp = info->op;
  // This is used by P2P to reduce the receive buffer size. We don't use it in collectives
  // because some protocols need to transmit more than the total size, plus they sometimes
  // round up
  proxyArgs->recvbytes = stepSize*proxyArgs->sliceSteps;

  TRACE(NCCL_NET,"opCount %lx slicesteps %d spl %d cpl %d nbytes %zi -> protocol %d nchannels %d nthreads %d, nloops %d nsteps %d comm %p",
      proxyArgs->opCount, proxyArgs->sliceSteps, info->nstepsPerLoop, info->nchunksPerLoop, info->nBytes, info->protocol, info->nChannels, info->nThreads,
      nLoops, proxyArgs->nsteps, info->comm);
  return ncclSuccess;
}

static ncclResult_t checkSetStream(struct ncclInfo* info) {
 if (info->comm->userStreamSet == false) {
    info->comm->userStream = info->stream;
    info->comm->userStreamSet = true;
  } else if (info->stream != info->comm->userStream) {
    WARN("Error : mixing different streams within a group call is not supported.");
    return ncclInvalidUsage;
  }
  return ncclSuccess;
}

ncclResult_t ncclSaveKernel(struct ncclInfo* info) {
  if (info->comm->nRanks == 1) {
    if (info->sendbuff != info->recvbuff)
      CUDACHECK(cudaMemcpyAsync(info->recvbuff, info->sendbuff, info->nBytes, cudaMemcpyDeviceToDevice, info->stream));
    return ncclSuccess;
  }

  struct ncclWorkElem work;
  struct ncclProxyArgs proxyArgs;
  memset(&proxyArgs, 0, sizeof(struct ncclProxyArgs));
  NCCLCHECK(computeColl(info, &work, &proxyArgs));

  info->comm->myParams->blockDim.x = std::max<unsigned>(info->comm->myParams->blockDim.x, info->nThreads);

  int nChannels = work.coll.nChannels;
  int nSubChannels = (info->pattern == ncclPatternCollTreeUp || info->pattern == ncclPatternCollTreeDown) ? 2 : 1;

  for (int bid=0; bid<nChannels*nSubChannels; bid++) {
    int channelId = info->comm->myParams->gridDim.x % info->comm->nChannels;
    struct ncclChannel* channel = info->comm->channels+channelId;

    // Proxy
    proxyArgs.channel = channel;
    // Adjust pattern for CollNet based on channel index
    if (nSubChannels == 2) {
      info->pattern = (channelId < info->comm->nChannels/nSubChannels) ? ncclPatternCollTreeUp : ncclPatternCollTreeDown;
    }

    if (proxyArgs.nsteps) NCCLCHECK(ncclProxySaveColl(&proxyArgs, info->pattern, info->root, info->comm->nRanks));

    info->comm->myParams->gridDim.x++;
    work.coll.bid = bid % nChannels;
    NCCLCHECK(getNextOp(channel, NULL, &work));
  }
  return ncclSuccess;
}

#define NCCL_MIN_CHANNEL_SIZE (NCCL_LL_THREAD_THRESHOLD*64)
#define NCCL_AGG_CHANNEL_SIZE (1LL << 21) /* 2 MiB, ideal per-channel size to fully utilize bandwidth */

ncclResult_t ncclSaveCommKernels(ncclComm_t comm) {
  if (comm->asyncOpCount == 0) {
    return ncclSuccess;
  } else if (comm->asyncOpCount == 1) {
    // No aggregation
    struct ncclInfo* info = comm->asyncOps;
    info->nChannels = 0;
    NCCLCHECK(ncclSaveKernel(info));
  } else {
    // Aggregation
    size_t channelSize = NCCL_AGG_CHANNEL_SIZE * comm->nRanks;  // scale channel size based on nranks as latency increases
    // Reduce the per-channel size if we cannot fully utilize the channels
    while (comm->asyncTotalSize < channelSize * comm->nChannels && channelSize > NCCL_MIN_CHANNEL_SIZE) channelSize /= 2;
    for (int c = 0; c < comm->asyncOpCount; c++) {
      struct ncclInfo* info = comm->asyncOps+c;
      info->nChannels = std::min((int)DIVUP(info->nBytes, channelSize), comm->nChannels); // assign number of channels
      NCCLCHECK(ncclSaveKernel(info));
    }
  }
  // Reset counters
  comm->asyncOpCount = 0;
  comm->asyncTotalSize = 0;
  return ncclSuccess;
}

static ncclResult_t ncclSaveAsyncColl(struct ncclInfo* info) {
  ncclComm_t comm = info->comm;
  if (comm->asyncOpCount >= NCCL_MAX_OPS) {
    WARN("Too many async operations in progress, max is %d", NCCL_MAX_OPS);
    return ncclInvalidUsage;
  }
  memcpy(comm->asyncOps+comm->asyncOpCount, info, sizeof(struct ncclInfo));
  comm->asyncOpCount++;
  comm->asyncTotalSize += info->nBytes;
  return ncclSuccess;
}

// Save p2p operations in comm->p2pSends and p2pRecvs. Operations will be posted to channels
// during ncclGroupEnd()
static ncclResult_t ncclSaveP2p(struct ncclInfo* info) {
  struct ncclComm* comm = info->comm;
  int peer = info->root;
  ssize_t nBytes = info->count*ncclTypeSize(info->datatype);
  if (info->opName[0] == 'S') { // Send
    if (peer != comm->rank) {
      int delta = (comm->nRanks - (comm->rank-peer)) % comm->nRanks;
      for (int c=0; c<comm->p2pnChannelsPerPeer; c++) {
        int channelId = (delta+comm->p2pChannels[c]) % comm->p2pnChannels;
        if (comm->channels[channelId].peers[peer].send.connected == 0) {
          comm->connectSend[peer] |= (1<<channelId);
          comm->connect = 1;
        }
      }
    }
    NCCLCHECK(enqueueP2pInfo(comm->p2pSends+info->root, (void*)info->sendbuff, nBytes));
    comm->p2pSendCount++;
  } else {
    if (peer != comm->rank) {
      int delta = (comm->nRanks + (comm->rank-peer)) % comm->nRanks;
      for (int c=0; c<comm->p2pnChannelsPerPeer; c++) {
        int channelId = (delta+comm->p2pChannels[c]) % comm->p2pnChannels;
        if (comm->channels[channelId].peers[peer].recv.connected == 0) {
          comm->connectRecv[peer] |= (1<<channelId);
          comm->connect = 1;
        }
      }
    }
    NCCLCHECK(enqueueP2pInfo(comm->p2pRecvs+info->root, info->recvbuff, nBytes));
    comm->p2pRecvCount++;
  }
  return ncclSuccess;
}

static int getSegment(struct ncclInfo* info, struct ncclWork* work) {
  for (int s=0; s<NCCL_MAX_WORK_ELEMENTS && work->elems[s].p2p.delta != info->delta; s++) {
    if (work->elems[s].p2p.nThreads == 0) return s;
  }
  return -1;
}

static ncclResult_t saveP2pOp(struct ncclInfo* info /* input */, struct ncclWork* work, int s) {
  struct ncclWorkElem* elem = work->elems+s;
  elem->comm = info->comm->devComm;
  elem->funcIndex = FUNC_INDEX_P2P;
  elem->nThreads = info->nThreads = NCCL_MAX_NTHREADS;
  elem->sendbuff = info->sendbuff;
  elem->recvbuff = info->recvbuff;
  elem->p2p.sendCount = info->sendbytes;
  elem->p2p.recvCount = info->recvbytes;
  elem->p2p.delta = info->delta;
  const int nsegments = s+1;
  int nThreads = 512;
  while (nsegments*nThreads > 512) nThreads /= 2;
  if (nThreads >= 128) nThreads += WARP_SIZE;
  for (int i=0; i<nsegments; i++) work->elems[i].p2p.nThreads = nThreads;
  return ncclSuccess;
}

ncclResult_t ncclSaveP2pKernel(struct ncclInfo* info) {
  int channelId = info->channelId;
  struct ncclChannel* channel = info->comm->channels+channelId;

  // Try to reuse last p2p operation if not full yet
  int opIndex = (channel->workFifoTail-1+NCCL_MAX_OPS)%NCCL_MAX_OPS;
  struct ncclWork* w = channel->workFifo+opIndex;
  int segment = -1;
  if (channel->workCount && w->elems[0].funcIndex == FUNC_INDEX_P2P && w->elems[NCCL_MAX_WORK_ELEMENTS-1].p2p.nThreads == 0) {
    // Try to pack more segments into a single operation
    segment = getSegment(info, w);
  }
  if (segment == -1) {
    NCCLCHECK(getNextOp(channel, &w, NULL));
    segment = 0;
  }

  NCCLCHECK(ncclProxySaveP2p(info, channel, segment));
  NCCLCHECK(saveP2pOp(info, w, segment));
  info->comm->myParams->gridDim.x = std::max<unsigned>(info->comm->myParams->gridDim.x, channelId+1);
  info->comm->myParams->blockDim.x = std::max<unsigned>(info->comm->myParams->blockDim.x, info->nThreads);

  return ncclSuccess;
}

ncclResult_t ncclEnqueueCheck(struct ncclInfo* info) {
  // Launch asynchronously if needed
  if (ncclAsyncMode()) {
    ncclResult_t ret = ncclSuccess;
    int savedDev = -1;
    // Check arguments
    NCCLCHECK(PtrCheck(info->comm, info->opName, "comm"));
    if (info->comm->checkPointers) {
      CUDACHECKGOTO(cudaGetDevice(&savedDev), ret, end);
      CUDACHECKGOTO(cudaSetDevice(info->comm->cudaDev), ret, end);
    }
    NCCLCHECKGOTO(ArgsCheck(info), ret, end);
    // Always register comm even in case of error to make sure ncclGroupEnd
    // cleans it up.
    NCCLCHECKGOTO(ncclAsyncColl(info->comm), ret, end);
    NCCLCHECKGOTO(checkSetStream(info), ret, end);

    INFO(NCCL_COLL,"%s: opCount %lx sendbuff %p recvbuff %p count %zi datatype %d op %d root %d comm %p [nranks=%d] stream %p",
        info->opName, info->comm->opCount, info->sendbuff, info->recvbuff, info->count,
        info->datatype, info->op, info->root, info->comm, info->comm->nRanks, info->stream);

    if (info->coll == ncclFuncSendRecv) { //p2p stored separately
      NCCLCHECKGOTO(ncclSaveP2p(info), ret, end);
    } else {
      NCCLCHECKGOTO(ncclSaveAsyncColl(info), ret, end);
    }
end:
    if (savedDev != -1) CUDACHECK(cudaSetDevice(savedDev));
    ncclAsyncErrCheck(ret);
    return ret;
  } else {
    NCCLCHECK(PtrCheck(info->comm, info->opName, "comm"));
    NCCLCHECK(ArgsCheck(info));
    NCCLCHECK(checkSetStream(info));

    INFO(NCCL_COLL,"%s: opCount %lx sendbuff %p recvbuff %p count %zi datatype %d op %d root %d comm %p [nranks=%d] stream %p",
        info->opName, info->comm->opCount, info->sendbuff, info->recvbuff, info->count,
        info->datatype, info->op, info->root, info->comm, info->comm->nRanks, info->stream);

    NCCLCHECK(ncclSaveKernel(info));
    NCCLCHECK(ncclBarrierEnqueue(info->comm));
    NCCLCHECK(ncclBarrierEnqueueWait(info->comm));
    NCCLCHECK(ncclEnqueueEvents(info->comm));
    return ncclSuccess;
  }
}
