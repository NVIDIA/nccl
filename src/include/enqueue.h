/*************************************************************************
 * Copyright (c) 2015-2021, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_ENQUEUE_H_
#define NCCL_ENQUEUE_H_

#include "comm.h"
#include "group.h"
#include "collectives.h"

#define NCCL_MIN_CHANNEL_SIZE (NCCL_LL_THREAD_THRESHOLD*64)
#define NCCL_AGG_CHANNEL_SIZE (1LL << 21) /* 2 MiB, ideal per-channel size to fully utilize bandwidth */

size_t ncclKernMaxLocalSize();
ncclResult_t ncclEnqueueCheck(struct ncclInfo* info);
ncclResult_t ncclCpuBarrierIn(struct ncclComm* comm, int* isLast);
ncclResult_t ncclCpuBarrierLast(struct ncclComm* comm);
ncclResult_t ncclCpuBarrierOut(struct ncclComm* comm);
ncclResult_t ncclLaunchBarrier(struct ncclComm* comm);
ncclResult_t ncclLaunchKernel(ncclComm_t comm);
ncclResult_t ncclRecordEvents(struct ncclComm* comm);
ncclResult_t ncclLaunchReset(ncclComm_t comm);
ncclResult_t ncclSetupP2pKernel(struct ncclInfo* info);
ncclResult_t ncclSetupAsyncKernels(struct ncclComm* comm);
template<int USING_CUDA_GRAPH>
void CUDART_CB ncclEnqueueHostSetup(void* arg);
ncclResult_t ncclGetCudaGraph(ncclComm_t comm, cudaGraph_t* graph);
ncclResult_t ncclCudaGraphHostSetup(ncclComm_t comm, cudaGraph_t graph);

// Enqueue information (for kernel and proxy) for each operation
struct ncclQueueElem {
  struct ncclWorkElem work;
  struct ncclProxyArgs proxyArgs;
};

typedef ncclRecyclableList<struct ncclQueueElem> ncclQueueElemList;

// Structure passed to CUDA graph
struct ncclQueueInfo {
  ncclComm_t comm;
  int maxChannels;    // Dynamic version of gridDim
  ncclResult_t ret;   // Return value of host setup call
  ncclQueueElemList* elemList;
};

static ncclResult_t ncclCreateQueueInfo(struct ncclQueueInfo** eqInfo, ncclComm_t comm) {
  NCCLCHECK(ncclCalloc(eqInfo, 1));
  (*eqInfo)->comm = comm;
  (*eqInfo)->elemList = new ncclQueueElemList();
  return ncclSuccess;
}

// Reset element queue
static ncclResult_t ncclResetQueueInfo(struct ncclQueueInfo* eqInfo) {
  if (eqInfo == NULL) return ncclInternalError;
  eqInfo->maxChannels = 0;
  eqInfo->ret = ncclSuccess;
  eqInfo->elemList->recycle();
  return ncclSuccess;
}

// Destroy enqueue info space
// used by both CUDA graph and non CUDA graph
static void ncclDestroyQueueInfo(void* ptr) {
  if (ptr == NULL) return;
  struct ncclQueueInfo* eqInfo = (struct ncclQueueInfo*)ptr;
  delete eqInfo->elemList;
  free(eqInfo);
}
#endif // End include guard
