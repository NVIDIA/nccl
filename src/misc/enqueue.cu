/*************************************************************************
 * Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "enqueue.h"
#include "common_coll.h"
#include "param.h"

#include "collectives/collectives.h"

#define NCCL_FUNC4(coll, op, dtype) \
  (void*)NCCL_KERN_NAME(coll, op, dtype), \
  (void*)NCCL_KERN_NAME(coll##LL, op, dtype)

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

// Must be consistent with ncclRedOp_t
#define NCCL_FUNCS2A(coll) \
  NCCL_FUNCS3A(coll, sum ), \
  NCCL_FUNCS3A(coll, prod), \
  NCCL_FUNCS3A(coll, max ), \
  NCCL_FUNCS3A(coll, min )
#define NCCL_FUNCS2B(coll) \
  NCCL_FUNCS3B(coll, copy), \
  NCCL_FUNCS3B(coll, copy), \
  NCCL_FUNCS3B(coll, copy), \
  NCCL_FUNCS3B(coll, copy)

// Must be consistent with the ncclFuncSet enum
static void* const ncclKerns[ncclCollCount*ncclNumOps*ncclNumTypes*2] = {
  NCCL_FUNCS2B(ncclBroadcast),
  NCCL_FUNCS2A(ncclReduce),
  NCCL_FUNCS2B(ncclAllGather),
  NCCL_FUNCS2A(ncclReduceScatter),
  NCCL_FUNCS2A(ncclAllReduce)
};

ncclResult_t ncclLaunchCooperativeKernelMultiDevice(struct cudaLaunchParams *paramsList, int* cudaDevs, int numDevices, int cgMode) {
#if __CUDACC_VER_MAJOR__ >= 9
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
  params->gridDim.x = std::min((int) params->gridDim.x, comm->nRings);

  // Set active = 2 for the last operation
  for (int r=0; r<params->gridDim.x; r++) {
    struct ncclRing* ring = comm->rings+r;
    ring->collectives[(ring->collStart+ring->collCount-1)%NCCL_MAX_OPS].active = 2;
  }

  // Find the first operation, choose the kernel accordingly and pass it
  // as the first argument.
  struct ncclColl* coll = comm->rings[0].collectives+comm->rings[0].collStart;
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
  // launch and the transportStartProxies call could cause a deadlock.
  // Also, starting the proxies after the CUDA launch seems to be better for
  // performance (latency).
  for (int r=0; r<params->gridDim.x; r++) {
    struct ncclRing* ring = comm->rings+r;
    ring->collStart = ring->collFifoTail;
    ring->collCount = 0;
  }
  params->gridDim.x = params->blockDim.x = 0;
  NCCLCHECK(transportStartProxies(comm));
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

ncclResult_t ncclEnqueueCheck(ncclFunc_t func, const char* primName, const void* sendbuff,
    void* recvbuff, size_t count, ncclDataType_t type, ncclRedOp_t op, int root,
    ncclComm_t comm, cudaStream_t stream) {
  if (comm == NULL) return ncclInvalidArgument;
  // Launch asynchronously if needed
  if (ncclAsyncMode()) {
    ncclResult_t ret = ncclSuccess;
    int savedDev = -1;
    if (comm->checkPointers) {
      CUDACHECKGOTO(cudaGetDevice(&savedDev), ret, end);
      CUDACHECKGOTO(cudaSetDevice(comm->cudaDev), ret, end);
    }
    // Check arguments
    NCCLCHECKGOTO(ArgsCheck(sendbuff, recvbuff, count, type, op, root, comm, primName), ret, end);
    // Always register comm even in case of error to make sure ncclGroupEnd
    // cleans it up.
    NCCLCHECK(ncclAsyncColl(comm));
    NCCLCHECKGOTO(func(sendbuff, recvbuff, count, type, op, root, comm, stream), ret, end);
end:
    if (savedDev != -1) CUDACHECK(cudaSetDevice(savedDev));
    ncclAsyncErrCheck(ret);
    return ret;
  } else {
    NCCLCHECK(ArgsCheck(sendbuff, recvbuff, count, type, op, root, comm, primName));
    NCCLCHECK(func(sendbuff, recvbuff, count, type, op, root, comm, stream));
    NCCLCHECK(ncclBarrierEnqueue(comm));
    NCCLCHECK(ncclBarrierEnqueueWait(comm));
    NCCLCHECK(ncclEnqueueEvents(comm));
    return ncclSuccess;
  }
}
