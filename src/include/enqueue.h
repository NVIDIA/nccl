/*************************************************************************
 * Copyright (c) 2015-2022, NVIDIA CORPORATION. All rights reserved.
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
ncclResult_t ncclKernSetSharedMemoryCarveout(int carveOut);
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

struct ncclBuffRegInfo {
  void* sendbuffsBase[NCCL_MAX_LOCAL_RANKS];
  void* recvbuffsBase[NCCL_MAX_LOCAL_RANKS];
  void* sendbuffs[NCCL_MAX_LOCAL_RANKS];
  void* recvbuffs[NCCL_MAX_LOCAL_RANKS];
  int nBuffs;
};

// Enqueue information (for kernel and proxy) for each operation
struct ncclQueueElem {
  struct ncclWork work;
  struct ncclProxyOp proxyOp;
  struct ncclBuffRegInfo buffRegInfo;
};

typedef ncclRecyclableList<struct ncclQueueElem> ncclQueueElemList;

// Structure passed to CUDA graph
struct ncclQueueInfo {
  ncclComm_t comm;
  int maxChannels;    // Dynamic version of gridDim
  ncclResult_t ret;   // Return value of host setup call
  int nRegBuffs;
  ncclQueueElemList* elemList;
};

static ncclResult_t ncclCreateQueueInfo(struct ncclQueueInfo** eqInfo, ncclComm_t comm) {
  NCCLCHECK(ncclCalloc(eqInfo, 1));
  (*eqInfo)->comm = comm;
  (*eqInfo)->elemList = new ncclQueueElemList();
  (*eqInfo)->comm->nQueueInfoCreated++;
  return ncclSuccess;
}

// Reset element queue
static ncclResult_t ncclResetQueueInfo(struct ncclQueueInfo* eqInfo) {
  if (eqInfo == NULL) return ncclInternalError;
  eqInfo->maxChannels = 0;
  eqInfo->ret = ncclSuccess;
  eqInfo->nRegBuffs = 0;
  eqInfo->elemList->recycle();
  return ncclSuccess;
}

// Destroy enqueue info space
// used by both CUDA graph and non CUDA graph
static void ncclDestroyQueueInfo(void* ptr) {
  if (ptr == NULL) return;
  struct ncclQueueInfo* eqInfo = (struct ncclQueueInfo*)ptr;
  struct ncclComm* comm = eqInfo->comm;
  // Close IPC mem handles for registered buffers
  struct ncclQueueElem* eqElem = eqInfo->elemList->begin();
#if 0
  // Ideally, the deregistration should happen here
  // but currently the destroy function of CUDA objects does not allow CUDA API calls
  while (eqElem != NULL) {
    for (int i=0; i<eqElem->buffRegInfo.nBuffs; i++) {
      if (i == eqInfo->comm->localRank) continue;
      CUDACHECKIGNORE(cudaIpcCloseMemHandle(eqElem->buffRegInfo.sendbuffsBase[i]));
      CUDACHECKIGNORE(cudaIpcCloseMemHandle(eqElem->buffRegInfo.recvbuffsBase[i]));
    }
    eqElem = eqInfo->elemList->getNext();
  }
#else
  // Instead, we push these pointers to a pool owned by ncclComm
  // and asks a helper thread to close mem handles
  struct ncclGraphHelperResources* res = comm->graphHelperResources;
  int ipcTailOld = 0;
  if (res == NULL || (!comm->graphHelperThread) || eqInfo->nRegBuffs == 0) goto skip;

  pthread_mutex_lock(&res->threadLock);
  ipcTailOld = res->ipcTail;
  while (eqElem != NULL) {
    for (int i=0; i<eqElem->buffRegInfo.nBuffs; i++) {
      if (eqElem->buffRegInfo.sendbuffsBase[i] != NULL) {
        res->ipcBases[res->ipcTail] = eqElem->buffRegInfo.sendbuffsBase[i];
        res->ipcTail = (res->ipcTail+1)%NCCL_IPC_POOL_SIZE;
      }
      if (eqElem->buffRegInfo.recvbuffsBase[i] != NULL) {
        res->ipcBases[res->ipcTail] = eqElem->buffRegInfo.recvbuffsBase[i];
        res->ipcTail = (res->ipcTail+1)%NCCL_IPC_POOL_SIZE;
      }
    }
    eqElem = eqInfo->elemList->getNext();
  }
  if (res->ipcTail != ipcTailOld) {
    res->threadState = ThreadStart;
    TRACE(NCCL_COLL, "CUDA Graph destroy function signaling helper thread with %d IPC handles", res->ipcTail-ipcTailOld);
    pthread_cond_signal(&res->threadCond);
  }
  pthread_mutex_unlock(&res->threadLock);
#endif

skip:
  delete eqInfo->elemList;
  free(eqInfo);
  comm->nQueueInfoDestroyed++;
  return;
}
#endif // End include guard
