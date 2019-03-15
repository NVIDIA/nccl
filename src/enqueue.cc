/*************************************************************************
 * Copyright (c) 2017-2019, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "enqueue.h"
#include "checks.h"
#include "param.h"

#include "collectives/collectives.h"

// Only generate inline kernels for LL
#define NCCL_FUNC5(coll, op, dtype) \
  (void*)NCCL_KERN_NAME(coll##LL, op, dtype), \
  (void*)NCCL_KERN_NAME(coll##LL, op, dtype)

#define NCCL_FUNC4(coll, op, dtype) \
  (void*)NCCL_FUNC5(coll##Ring, op, dtype), \
  (void*)NCCL_FUNC5(coll##Tree, op, dtype)

// Must be consistent with ncclDataType_t
#define NCCL_FUNCS3A(coll, op) \
  (void*)NCCL_FUNC4(coll, op,  i8), \
  (void*)NCCL_FUNC4(coll, op,  u8), \
  (void*)NCCL_FUNC4(coll, op, i32), \
  (void*)NCCL_FUNC4(coll, op, u32), \
  (void*)NCCL_FUNC4(coll, op, i64), \
  (void*)NCCL_FUNC4(coll, op, u64), \
  (void*)NCCL_FUNC4(coll, op, f16), \
  (void*)NCCL_FUNC4(coll, op, f32), \
  (void*)NCCL_FUNC4(coll, op, f64)
#define NCCL_FUNCS3B(coll, op) \
  (void*)NCCL_FUNC4(coll, op,  i8), \
  (void*)NCCL_FUNC4(coll, op,  i8), \
  (void*)NCCL_FUNC4(coll, op,  i8), \
  (void*)NCCL_FUNC4(coll, op,  i8), \
  (void*)NCCL_FUNC4(coll, op,  i8), \
  (void*)NCCL_FUNC4(coll, op,  i8), \
  (void*)NCCL_FUNC4(coll, op,  i8), \
  (void*)NCCL_FUNC4(coll, op,  i8), \
  (void*)NCCL_FUNC4(coll, op,  i8)

// Must be consistent with ncclRedOp_t -- but we only generate kernel for sums.
#define NCCL_FUNCS2A(coll) \
  NCCL_FUNCS3A(coll, sum), \
  NCCL_FUNCS3A(coll, sum), \
  NCCL_FUNCS3A(coll, sum), \
  NCCL_FUNCS3A(coll, sum)
#define NCCL_FUNCS2B(coll) \
  NCCL_FUNCS3B(coll, copy), \
  NCCL_FUNCS3B(coll, copy), \
  NCCL_FUNCS3B(coll, copy), \
  NCCL_FUNCS3B(coll, copy)

// Must be consistent with the ncclFuncSet enum
static void* const ncclKerns[ncclCollCount*ncclNumOps*ncclNumTypes*2*2] = {
  NCCL_FUNCS2B(ncclBroadcast),
  NCCL_FUNCS2A(ncclReduce),
  NCCL_FUNCS2B(ncclAllGather),
  NCCL_FUNCS2A(ncclReduceScatter),
  NCCL_FUNCS2A(ncclAllReduce)
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

ncclResult_t setupLaunch(struct ncclComm* comm, struct cudaLaunchParams* params) {
  params->gridDim.x = std::min<unsigned>(params->gridDim.x, comm->nChannels);

  // Set active = 2 for the last operation
  for (int r=0; r<params->gridDim.x; r++) {
    struct ncclChannel* channel = comm->channels+r;
    channel->collectives[(channel->collStart+channel->collCount-1)%NCCL_MAX_OPS].active = 2;
  }

  // Find the first operation, choose the kernel accordingly and pass it
  // as the first argument.
  struct ncclColl* coll = comm->channels[0].collectives+comm->channels[0].collStart;
  memcpy(&comm->args, coll, sizeof(struct ncclColl));
  // As we pass that coll directly, we can free it immediately.
  coll->active = 0;

  params->func = ncclKerns[coll->funcIndex];
  return ncclSuccess;
}

ncclResult_t ncclCpuBarrierIn(struct ncclComm* comm, int* isLast) {
  volatile int* ptr = (volatile int*)(comm->intraBarrier+comm->intraPhase);
  int val = *ptr;
  bool done = false;
  while (done == false) {
    if (val >= comm->intraRanks) {
      WARN("Trying to launch too many collectives");
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
    WARN("Trying to launch too many collectives");
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
  if (comm->nRanks == 1) return ncclSuccess;
  struct cudaLaunchParams* params = comm->myParams;

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

  int isLast = 0;
  NCCLCHECK(ncclCpuBarrierIn(comm, &isLast));

  if (isLast) {
    if (comm->launchMode == ncclComm::GROUP) {
      // I'm the last. Launch all operations.
      NCCLCHECK(ncclLaunchCooperativeKernelMultiDevice(comm->intraParams, comm->intraCudaDevs, comm->intraRanks, *comm->intraCGMode));
    }
    NCCLCHECK(ncclCpuBarrierLast(comm));
  }
  return ncclSuccess;
}

ncclResult_t ncclBarrierEnqueueWait(ncclComm_t comm) {
  if (comm->nRanks == 1) return ncclSuccess;
  // We can't print the CG mode before the first barrier happened.
  if (comm->rank == 0 && *comm->intraCGMode & 0x10) {
    *comm->intraCGMode ^= 0x10;
    INFO(NCCL_INIT,"Launch mode %s%s%s",
        comm->launchMode == ncclComm::GROUP ? "Group" : "Parallel",
        *comm->intraCGMode ? "/CGMD" : "",
        (comm->launchMode == ncclComm::GROUP && comm->groupCudaStream) ? "/Stream" : "");
  }

  NCCLCHECK(ncclCpuBarrierOut(comm));

  struct cudaLaunchParams *params = comm->myParams;
  if (comm->launchMode == ncclComm::PARALLEL) {
    CUDACHECK(cudaLaunchKernel(params->func, params->gridDim, params->blockDim, params->args, params->sharedMem, params->stream));
  }
  // Start the network proxies as soon as the kernel has been launched. We can't
  // perform any CUDA call between the two or having a cudaFree between the CUDA
  // launch and the transportStartProxy call could cause a deadlock.
  // Also, starting the proxies after the CUDA launch seems to be better for
  // performance (latency).
  for (int r=0; r<params->gridDim.x; r++) {
    struct ncclChannel* channel = comm->channels+r;
    channel->collStart = channel->collFifoTail;
    channel->collCount = 0;
  }
  params->gridDim.x = params->blockDim.x = 0;
  NCCLCHECK(transportStartProxy(comm));
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

static ncclResult_t getPatternInfo(struct ncclInfo* info) {
  if (info->coll == ncclCollBroadcast) info->pattern = ncclPatternPipelineFrom;
  else if (info->coll == ncclCollReduce) info->pattern = ncclPatternPipelineTo;
  else if (info->coll == ncclCollAllGather || info->coll == ncclCollReduceScatter) info->pattern = ncclPatternRing;
  else if (info->coll == ncclCollAllReduce) {
    if (info->nBytes <= info->comm->treeThreshold)
      info->pattern = ncclPatternTreeUpDown;
    else
      info->pattern = ncclPatternRingTwice;
  }
  else {
    WARN("Unknown collective %d", info->coll);
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
      info->nstepsPerLoop = info-> nchunksPerLoop = 1; break;
    case ncclPatternRing:
      info->nstepsPerLoop = info->comm->nRanks-1; info->nchunksPerLoop = info->comm->nRanks; break;
    case ncclPatternRingTwice:
      info->nstepsPerLoop = 2*(info->comm->nRanks-1); info->nchunksPerLoop = info->comm->nRanks; break;
    default:
      WARN("Unknown pattern %d\n", info->pattern);
      return ncclInternalError;
  }
  return ncclSuccess;
}

static void getKernelInfo(struct ncclInfo* info, uint8_t* nChannels, uint16_t* nThreads, int* llMode) {
  // Compute thresholds and limits that users can override
  ssize_t perThreadLLThreshold = std::min<ssize_t>(info->comm->threadThreshold, NCCL_LL_CHANNEL_THRESHOLD);
  int maxLLNthreads = std::min(NCCL_LL_MAX_NTHREADS, info->comm->nThreads);

  // First compute nThreads
  int nt = NCCL_LL_MIN_NTHREADS;
  while (DIVUP(info->nBytes, nt*info->nchunksPerLoop) > perThreadLLThreshold && nt*2 <= maxLLNthreads) nt *= 2;

  // Then compute nChannels
  int nc = DIVUP(info->nBytes, nt*info->nchunksPerLoop*perThreadLLThreshold);
  if (nc == 0) nc = 1;
  if (nc > info->comm->nChannels) nc = info->comm->nChannels;

  // Check if we have a fixed LL threshold, otherwise compute it.
  int perThreadThreshold = info->comm->threadThreshold;
  if (info->pattern >= ncclPatternTreeUp) perThreadThreshold *= 4;
  ssize_t llThreshold = info->comm->llThreshold >= 0 ?
    info->comm->llThreshold :
    nc*nt*info->nchunksPerLoop*perThreadThreshold;

  if (info->nBytes <= llThreshold) {
    *llMode = 1;
    *nChannels = nc;
    *nThreads = nt;
  } else {
    *llMode = 0;
    *nChannels = info->comm->nChannels;
    *nThreads = info->comm->nThreads+1;
  }
}

static ncclResult_t computeColl(struct ncclInfo* info /* input */, struct ncclColl* coll, struct ncclProxyArgs* proxyArgs /* output */) {
  // Set nstepsPerLoop and nchunksPerLoop
  NCCLCHECK(getPatternInfo(info));
  NCCLCHECK(getLoopInfo(info));

  coll->args.root = info->root;
  coll->args.N = info->count;
  coll->args.ThisInput = info->sendbuff;
  coll->args.ThisOutput = info->recvbuff;
  coll->args.comm = info->comm->devComm;
  coll->args.opCount = info->comm->opCount;

  // Compute llMode, nChannels, nThreads
  int llMode;
  getKernelInfo(info, &coll->args.nChannels, &coll->args.nThreads, &llMode);

  int treeMode = info->pattern >= ncclPatternTreeUp ? 1 : 0;
  coll->funcIndex = FUNC_INDEX(info->coll, info->op, info->datatype, llMode, treeMode);

  int stepSize   = ( llMode ? NCCL_LL_BUFF_SIZE : info->comm->channels[0].buffSize ) / NCCL_STEPS;
  int chunkSteps = (llMode|treeMode) ? 1 : info->chunkSteps;
  int sliceSteps = (llMode|treeMode) ? 1 : info->sliceSteps;
  int chunkSize  = stepSize*chunkSteps;

  // Compute lastChunkSize
  if (treeMode == 1 && llMode == 0) {
    if (info->pattern == ncclPatternTreeUpDown) {
      // Optimize chunkSize / nSteps
      while (info->nBytes / (coll->args.nChannels*chunkSize) < info->comm->channels[0].tree.depth*8 && chunkSize > 131072) chunkSize /= 2;
      while (info->nBytes / (coll->args.nChannels*chunkSize) < info->comm->channels[0].tree.depth*4 && chunkSize > 65536) chunkSize /= 2;
      while (info->nBytes / (coll->args.nChannels*chunkSize) < info->comm->channels[0].tree.depth && chunkSize > 32768) chunkSize /= 2;
    }
    // Use lastChunkSize as chunkSize
    coll->args.lastChunkSize = chunkSize / ncclTypeSize(info->datatype);
  } else if (llMode == 1) {
    int sliceSize = NCCL_LL_SLICE_LINES * sizeof(uint64_t);
    const ssize_t loopSize = coll->args.nChannels*info->nchunksPerLoop*(ssize_t)sliceSize;
    coll->args.lastChunkSize = DIVUP((info->nBytes-(info->nBytes/loopSize)*loopSize), coll->args.nChannels*info->nchunksPerLoop);
    ALIGN_SIZE(coll->args.lastChunkSize, coll->args.nThreads*sizeof(uint64_t));
    coll->args.lastChunkSize /= ncclTypeSize(info->datatype);
  }

  // Compute nSteps for proxies
  size_t nBytes  = llMode ? info->nBytes*2 : info->nBytes;

  int nLoops = (int)(DIVUP(nBytes, (((size_t)(coll->args.nChannels))*info->nchunksPerLoop*chunkSize)));
  proxyArgs->nsteps = info->nstepsPerLoop * nLoops * chunkSteps;
  proxyArgs->sliceSteps = sliceSteps;
  proxyArgs->chunkSteps = chunkSteps;
  proxyArgs->llMode = llMode;
  proxyArgs->opCount = info->comm->opCount;
  TRACE(NCCL_NET,"opCount %lx slicesteps %d spl %d cpl %d nbytes %zi -> llmode %d nchannels %d nthreads %d, nloops %d nsteps %d comm %p",
      coll->args.opCount, proxyArgs->sliceSteps, info->nstepsPerLoop, info->nchunksPerLoop, nBytes, llMode, coll->args.nChannels, coll->args.nThreads,
      nLoops, proxyArgs->nsteps, info->comm);
  return ncclSuccess;
}

static ncclResult_t saveKernel(struct ncclInfo* info) {
  if (info->comm->nRanks == 1) {
    if (info->sendbuff != info->recvbuff)
      CUDACHECK(cudaMemcpyAsync(info->recvbuff, info->sendbuff, info->nBytes, cudaMemcpyDeviceToDevice, info->stream));
    return ncclSuccess;
  }

  struct ncclColl coll;
  struct ncclProxyArgs proxyArgs;
  memset(&proxyArgs, 0, sizeof(struct ncclProxyArgs));
  NCCLCHECK(computeColl(info, &coll, &proxyArgs));

  info->comm->myParams->blockDim.x = std::max<unsigned>(info->comm->myParams->blockDim.x, coll.args.nThreads);
  if (info->comm->userStreamSet == false) {
    info->comm->userStream = info->stream;
    info->comm->userStreamSet = true;
  } else if (info->stream != info->comm->userStream) {
    WARN("Error : mixing different streams within a group call is not supported.");
    return ncclInvalidUsage;
  }
  for (int bid=0; bid<coll.args.nChannels; bid++) {
    struct ncclChannel* channel = info->comm->channels+(info->comm->myParams->gridDim.x % info->comm->nChannels);

    if (channel->collCount == NCCL_MAX_OPS) {
      WARN("Too many aggregated operations (%d max)", NCCL_MAX_OPS);
      return ncclInvalidUsage;
    }

    // Proxy
    proxyArgs.channel = channel;
    NCCLCHECK(transportSaveProxies(&proxyArgs, info->pattern, info->root, info->comm->nRanks));

    info->comm->myParams->gridDim.x++;

    int opIndex = channel->collFifoTail;
    struct ncclColl* c = channel->collectives+opIndex;
    volatile uint8_t* activePtr = (volatile uint8_t*)&c->active;
    while (activePtr[0] != 0) sched_yield();

    memcpy(c, &coll, sizeof(struct ncclColl));

    c->args.bid = bid;
    c->active = 1;
    opIndex = (opIndex+1)%NCCL_MAX_OPS;
    c->nextIndex = opIndex;
    channel->collFifoTail = opIndex;
    channel->collCount++;
  }
  /*if (llMode == 0)*/ info->comm->opCount++;
  return ncclSuccess;
}


ncclResult_t ncclEnqueueCheck(struct ncclInfo* info) {
  if (info->comm == NULL) return ncclInvalidArgument;

  INFO(NCCL_COLL,"%s: opCount %lx sendbuff %p recvbuff %p count %zi datatype %d op %d root %d comm %p [nranks=%d] stream %p",
       info->opName, info->comm->opCount, info->sendbuff, info->recvbuff, info->count,
       info->datatype, info->op, info->root, info->comm, info->comm->nRanks, info->stream);

  // Launch asynchronously if needed
  if (ncclAsyncMode()) {
    ncclResult_t ret = ncclSuccess;
    int savedDev = -1;
    if (info->comm->checkPointers) {
      CUDACHECKGOTO(cudaGetDevice(&savedDev), ret, end);
      CUDACHECKGOTO(cudaSetDevice(info->comm->cudaDev), ret, end);
    }
    // Check arguments
    NCCLCHECKGOTO(ArgsCheck(info), ret, end);
    // Always register comm even in case of error to make sure ncclGroupEnd
    // cleans it up.
    NCCLCHECKGOTO(ncclAsyncColl(info->comm), ret, end);
    NCCLCHECKGOTO(saveKernel(info), ret, end);
end:
    if (savedDev != -1) CUDACHECK(cudaSetDevice(savedDev));
    ncclAsyncErrCheck(ret);
    return ret;
  } else {
    NCCLCHECK(ArgsCheck(info));
    NCCLCHECK(saveKernel(info));
    NCCLCHECK(ncclBarrierEnqueue(info->comm));
    NCCLCHECK(ncclBarrierEnqueueWait(info->comm));
    NCCLCHECK(ncclEnqueueEvents(info->comm));
    return ncclSuccess;
  }
}
