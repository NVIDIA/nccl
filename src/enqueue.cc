/*************************************************************************
 * Copyright (c) 2017-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "enqueue.h"
#include "argcheck.h"
#include "coll_net.h"

// Only generate inline kernels for LL
#define NCCL_FUNC5(coll, op, dtype) \
  (void*)NCCL_KERN_NAME(coll##LL, op, dtype), \
  (void*)NCCL_KERN_NAME(coll##LL, op, dtype), \
  (void*)NCCL_KERN_NAME(coll##LL, op, dtype)

#define NCCL_FUNC4(coll, op, dtype) \
  (void*)NCCL_FUNC5(coll##Tree, op, dtype), \
  (void*)NCCL_FUNC5(coll##Ring, op, dtype), \
  (void*)NCCL_FUNC5(coll##CollNet, op, dtype)

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
static void* const ncclKerns[1+NCCL_NUM_FUNCTIONS*ncclNumOps*ncclNumTypes*NCCL_NUM_ALGORITHMS*NCCL_NUM_PROTOCOLS] = {
  (void*)NCCL_KERN_NAME(ncclSendRecv, copy, i8),
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
  // Only launch blocks where we have work to do.
  for (int c=0; c<comm->p2pnChannels; c++) {
    if (comm->channels[c].collCount) params->gridDim.x = c+1;
  }

  // Set active = 2 for the last operation and add a no-op on empty channels (p2p case).
  for (int c=0; c<params->gridDim.x; c++) {
    struct ncclChannel* channel = comm->channels+c;
    if (channel->collCount == 0) {
      int opIndex = channel->collFifoTail;
      struct ncclColl* c = channel->collectives+opIndex;
      volatile uint8_t* activePtr = (volatile uint8_t*)&c->active;
      while (activePtr[0] != 0) sched_yield();

      c->args.p2p.delta = -1; // no-op
      c->funcIndex = FUNC_INDEX_P2P;
      c->args.comm = comm->devComm;
      c->active = 1;
      opIndex = (opIndex+1)%NCCL_MAX_OPS;
      c->nextIndex = opIndex;
      channel->collFifoTail = opIndex;
      channel->collCount++;
    }
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
  for (int r=0; r<params->gridDim.x; r++) {
    struct ncclChannel* channel = comm->channels+r;
    channel->collStart = channel->collFifoTail;
    channel->collCount = 0;
  }
  params->gridDim.x = params->blockDim.x = 0;
  comm->lastOpCount = comm->opCount;
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

  int nc = (info->algorithm == NCCL_ALGO_COLLNET) ? comm->nChannels/2 : comm->nChannels; // CollNet uses one channel for up and one channel for down
  int nt = comm->maxThreads[info->algorithm][info->protocol];
  int threadThreshold = comm->threadThresholds[info->algorithm][info->protocol];
  while (info->nBytes < nc*nt*threadThreshold) {
    if (info->algorithm != NCCL_ALGO_COLLNET && nc >= 2) nc--;
    else if ((nt % 128) == 0) nt/=2;
    else break;
  }
  if (info->protocol == NCCL_PROTO_SIMPLE) nt += WARP_SIZE; // Extra warp for sync
  info->nChannels = nc;
  info->nThreads = nt;
  return ncclSuccess;
}

static ncclResult_t getPatternInfo(struct ncclInfo* info) {
  switch (info->coll) {
    case ncclCollBroadcast:
      info->pattern = info->algorithm == NCCL_ALGO_TREE ? ncclPatternTreeDown : ncclPatternPipelineFrom; break;
    case ncclCollReduce:
      info->pattern = info->algorithm == NCCL_ALGO_TREE ? ncclPatternTreeUp : ncclPatternPipelineTo; break;
    case ncclCollReduceScatter:
    case ncclCollAllGather:
      info->pattern = ncclPatternRing; break;
    case ncclCollAllReduce:
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
      WARN("Unknown pattern %d\n", info->pattern);
      return ncclInternalError;
  }
  return ncclSuccess;
}

static ncclResult_t computeColl(struct ncclInfo* info /* input */, struct ncclColl* coll, struct ncclProxyArgs* proxyArgs /* output */) {
  coll->args.sendbuff = info->sendbuff;
  coll->args.recvbuff = info->recvbuff;
  coll->args.comm = info->comm->devComm;
  coll->args.opCount = info->comm->opCount;

  if (info->coll == ncclCollSendRecv) {
    coll->args.p2p.sendCount = info->sendbytes;
    coll->args.p2p.recvCount = info->recvbytes;
    coll->args.p2p.delta = info->delta;
    coll->funcIndex = FUNC_INDEX_P2P;
    coll->args.p2p.nThreads = info->nThreads = info->comm->maxThreads[NCCL_ALGO_RING][NCCL_PROTO_SIMPLE]+2*WARP_SIZE;
    return ncclSuccess;
  }
  // Set nstepsPerLoop and nchunksPerLoop
  NCCLCHECK(getAlgoInfo(info));
  NCCLCHECK(getPatternInfo(info));
  NCCLCHECK(getLoopInfo(info));

  coll->args.coll.root = info->root;
  coll->args.coll.count = info->count;
  coll->args.coll.nChannels = info->nChannels;
  coll->args.coll.nThreads = info->nThreads;

  coll->funcIndex = FUNC_INDEX(info->coll, info->op, info->datatype, info->algorithm, info->protocol);

  int stepSize   = info->comm->buffSizes[info->protocol]/NCCL_STEPS;
  int chunkSteps = (info->protocol == NCCL_PROTO_SIMPLE && info->algorithm == NCCL_ALGO_RING) ? info->chunkSteps : 1;
  int sliceSteps = (info->protocol == NCCL_PROTO_SIMPLE && info->algorithm == NCCL_ALGO_RING) ? info->sliceSteps : 1;
  int chunkSize  = stepSize*chunkSteps;

  // Compute lastChunkSize
  if (info->algorithm == NCCL_ALGO_TREE && info->protocol == NCCL_PROTO_SIMPLE) {
    if (info->pattern == ncclPatternTreeUpDown) {
      // Optimize chunkSize / nSteps
      while (info->nBytes / (info->nChannels*chunkSize) < info->comm->channels[0].treeUp.depth*8 && chunkSize > 131072) chunkSize /= 2;
      while (info->nBytes / (info->nChannels*chunkSize) < info->comm->channels[0].treeUp.depth*4 && chunkSize > 65536) chunkSize /= 2;
      while (info->nBytes / (info->nChannels*chunkSize) < info->comm->channels[0].treeUp.depth && chunkSize > 32768) chunkSize /= 2;
    }
    // Use lastChunkSize as chunkSize
    coll->args.coll.lastChunkSize = chunkSize / ncclTypeSize(info->datatype);
  } else if (info->algorithm == NCCL_ALGO_COLLNET && info->protocol == NCCL_PROTO_SIMPLE) {
    // Optimize chunkSize / nSteps
    while (info->nBytes / (info->nChannels*chunkSize) < info->comm->channels[0].collTreeUp.depth*16 && chunkSize > 131072) chunkSize /= 2;
    while (info->nBytes / (info->nChannels*chunkSize) < info->comm->channels[0].collTreeUp.depth*4 && chunkSize > 65536) chunkSize /= 2;
    while (info->nBytes / (info->nChannels*chunkSize) < info->comm->channels[0].collTreeUp.depth && chunkSize > 32768) chunkSize /= 2;
    // Use lastChunkSize as chunkSize
    coll->args.coll.lastChunkSize = chunkSize / ncclTypeSize(info->datatype);
  } else if (info->protocol == NCCL_PROTO_LL) {
    const ssize_t sliceSize = stepSize*sizeof(uint64_t)/sizeof(union ncclLLFifoLine);
    const ssize_t loopSize = info->nChannels*info->nchunksPerLoop*(ssize_t)sliceSize;
    coll->args.coll.lastChunkSize = DIVUP((info->nBytes-(info->nBytes/loopSize)*loopSize), info->nChannels*info->nchunksPerLoop);
    ALIGN_SIZE(coll->args.coll.lastChunkSize, info->nThreads*sizeof(uint64_t));
    coll->args.coll.lastChunkSize /= ncclTypeSize(info->datatype);
  } else if (info->algorithm == NCCL_ALGO_TREE && info->protocol == NCCL_PROTO_LL128) {
    int nNodes = info->comm->nNodes;
    float ppn = info->comm->nRanks / (float)nNodes;
    float nstepsLL128 = 1+log2i(nNodes) + 0.1*ppn;
    while (info->nBytes / (info->nChannels*chunkSize) < nstepsLL128*64/ppn && chunkSize > 131072) chunkSize /= 2;
    while (info->nBytes / (info->nChannels*chunkSize) < nstepsLL128*16/ppn && chunkSize > 32768) chunkSize /= 2;
    // Use lastChunkSize as chunkSize
    coll->args.coll.lastChunkSize = chunkSize*NCCL_LL128_DATAELEMS/(NCCL_LL128_LINEELEMS*ncclTypeSize(info->datatype));
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
  proxyArgs->opCount = info->comm->opCount;
  proxyArgs->dtype = info->datatype;
  proxyArgs->redOp = info->op;
  TRACE(NCCL_NET,"opCount %lx slicesteps %d spl %d cpl %d nbytes %zi -> protocol %d nchannels %d nthreads %d, nloops %d nsteps %d comm %p",
      coll->args.opCount, proxyArgs->sliceSteps, info->nstepsPerLoop, info->nchunksPerLoop, info->nBytes, info->protocol, info->nChannels, info->nThreads,
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
  if (info->comm->nRanks == 1 && info->coll != ncclCollSendRecv) {
    if (info->sendbuff != info->recvbuff)
      CUDACHECK(cudaMemcpyAsync(info->recvbuff, info->sendbuff, info->nBytes, cudaMemcpyDeviceToDevice, info->stream));
    return ncclSuccess;
  }

  struct ncclColl coll;
  struct ncclProxyArgs proxyArgs;
  memset(&proxyArgs, 0, sizeof(struct ncclProxyArgs));
  NCCLCHECK(computeColl(info, &coll, &proxyArgs));

  info->comm->myParams->blockDim.x = std::max<unsigned>(info->comm->myParams->blockDim.x, info->nThreads);

  int nChannels = info->coll == ncclCollSendRecv ? 1 : coll.args.coll.nChannels;
  int nSubChannels = (info->pattern == ncclPatternCollTreeUp || info->pattern == ncclPatternCollTreeDown) ? 2 : 1;

  for (int bid=0; bid<nChannels*nSubChannels; bid++) {
    int channelId = (info->coll == ncclCollSendRecv) ? info->channelId :
      info->comm->myParams->gridDim.x % info->comm->nChannels;
    struct ncclChannel* channel = info->comm->channels+channelId;

    if (channel->collCount == NCCL_MAX_OPS) {
      WARN("Too many aggregated operations on channel %d (%d max)", channel->id, NCCL_MAX_OPS);
      return ncclInvalidUsage;
    }

    // Proxy
    proxyArgs.channel = channel;
    // Adjust pattern for CollNet based on channel index
    if (nSubChannels == 2) {
      info->pattern = (channelId < info->comm->nChannels/nSubChannels) ? ncclPatternCollTreeUp : ncclPatternCollTreeDown;
    }

    if (info->coll == ncclCollSendRecv) {
      info->comm->myParams->gridDim.x = std::max<unsigned>(info->comm->myParams->gridDim.x, channelId+1);
      NCCLCHECK(ncclProxySaveP2p(info, channel));
    } else {
      NCCLCHECK(ncclProxySaveColl(&proxyArgs, info->pattern, info->root, info->comm->nRanks));
    }
    info->comm->myParams->gridDim.x++;
    int opIndex = channel->collFifoTail;
    struct ncclColl* c = channel->collectives+opIndex;
    volatile uint8_t* activePtr = (volatile uint8_t*)&c->active;
    while (activePtr[0] != 0) sched_yield();

    memcpy(c, &coll, sizeof(struct ncclColl));
    if (info->coll != ncclCollSendRecv) c->args.coll.bid = bid % coll.args.coll.nChannels;

    c->active = 1;
    opIndex = (opIndex+1)%NCCL_MAX_OPS;
    c->nextIndex = opIndex;
    channel->collFifoTail = opIndex;
    channel->collCount++;
  }
  info->comm->opCount++;
  return ncclSuccess;
}

// Save p2p operations in comm->p2plist. Operations will be posted to channels
// during ncclGroupEnd()
ncclResult_t ncclSaveP2p(struct ncclInfo* info) {
  struct ncclComm* comm = info->comm;
  struct ncclP2Plist* p2plist = &comm->p2plist;
  int peer = info->root;
  p2plist->count++;
  ssize_t nBytes = info->count*ncclTypeSize(info->datatype);
  if (info->recvbuff == NULL) {
    if (peer != comm->rank) {
      int delta = (comm->nRanks - (comm->rank-peer)) % comm->nRanks;
      for (int c=0; c<comm->p2pnChannelsPerPeer; c++) {
        int channelId = (delta+comm->p2pChannels[c]) % comm->p2pnChannels;
        if (comm->channels[channelId].peers[peer].send.connected == 0) {
          p2plist->connect.send[channelId*comm->nRanks+p2plist->connect.nsend[channelId]++] = peer;
        }
      }
    }
    p2plist->peerlist[info->root].sendbytes = nBytes;
    p2plist->peerlist[info->root].sendbuff = info->sendbuff;
  } else {
    if (peer != comm->rank) {
      int delta = (comm->nRanks + (comm->rank-peer)) % comm->nRanks;
      for (int c=0; c<comm->p2pnChannelsPerPeer; c++) {
        int channelId = (delta+comm->p2pChannels[c]) % comm->p2pnChannels;
        if (comm->channels[channelId].peers[peer].recv.connected == 0) {
          p2plist->connect.recv[channelId*comm->nRanks+p2plist->connect.nrecv[channelId]++] = peer;
        }
      }
    }
    p2plist->peerlist[info->root].recvbytes = nBytes;
    p2plist->peerlist[info->root].recvbuff = info->recvbuff;
  }
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

    if (info->coll == ncclCollSendRecv) { //p2p stored separately
      NCCLCHECKGOTO(ncclSaveP2p(info), ret, end);
    } else {
      NCCLCHECKGOTO(ncclSaveKernel(info), ret, end);
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
