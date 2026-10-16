/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "comm.h"
#include "enqueue.h"
#include "enqueue/task_classify.h"
#include "register_inline.h"

static ncclResult_t enqueueP2pTask(
  struct ncclComm* comm, struct ncclIntruQueue<struct ncclTaskTuningInfo, &ncclTaskTuningInfo::next>* p2pTaskQueue,
  ncclFunc_t func, ncclFunc_t collAPI, void* buff, size_t count, ncclDataType_t datatype, int peer, cudaStream_t stream,
  cudaEvent_t launchCompletionEvent, bool allowUB, const struct ncclProfilerApiCtx& profiler) {
  struct ncclRawTask* raw = ncclMemoryPoolAlloc<struct ncclRawTask>(&comm->memPool_ncclRawTask, &comm->memPermanent);
  raw->kind = ncclTaskKindSendRecv;
  raw->profiler = profiler;
  raw->sendRecv.func = func;
  raw->sendRecv.collAPI = collAPI;
  raw->sendRecv.buff = buff;
  raw->sendRecv.count = count;
  raw->sendRecv.datatype = datatype;
  raw->sendRecv.peer = peer;
  raw->sendRecv.bytes = count * ncclTypeSize(datatype);
  raw->sendRecv.stream = stream;
  raw->sendRecv.launchCompletionEvent = launchCompletionEvent;
  raw->sendRecv.allowUB = allowUB;

  struct ncclTaskTuningInfo* pt = ncclMemoryStackAlloc<struct ncclTaskTuningInfo>(&comm->memScoped);
  pt->raw = raw;
  ncclIntruQueueEnqueue(p2pTaskQueue, pt);
  return ncclSuccess;
}

static ncclResult_t collToP2pTasks(
  struct ncclComm* comm, struct ncclTaskTuningInfo* tInfo,
  struct ncclIntruQueue<struct ncclTaskTuningInfo, &ncclTaskTuningInfo::next>* p2pTaskQueue) {
  struct ncclRawTaskColl* coll = &tInfo->raw->coll;
  ncclFunc_t collAPI = coll->func;
  size_t elemSize = ncclTypeSize(coll->datatype);
  cudaStream_t stream = coll->stream;
  cudaEvent_t launchCompletionEvent = ncclCollConfigGetLaunchCompletionEvent(&tInfo->raw->collConfig);
  // Send/recv needs both buffers registered, so user-buffer registration is allowed under graph
  // capture, or for AlltoAll when both of its full buffers are locally registered.
  bool allowUB = tInfo->tuningIn.captured;

  if (coll->func == ncclFuncAlltoAll) {
    if (!allowUB) {
      size_t bytes = comm->nRanks * coll->count * elemSize;
      struct ncclReg* sendReg;
      struct ncclReg* recvReg;
      bool sendLocalValid = false;
      bool recvLocalValid = false;
      NCCLCHECK(ncclRegFind(comm, coll->sendbuff, bytes, &sendReg));
      NCCLCHECK(ncclRegFind(comm, coll->recvbuff, bytes, &recvReg));
      NCCLCHECK(ncclRegLocalIsValid(sendReg, &sendLocalValid));
      NCCLCHECK(ncclRegLocalIsValid(recvReg, &recvLocalValid));
      allowUB = sendLocalValid && recvLocalValid;
    }
    for (int r = 0; r < comm->nRanks; r++) {
      void* sendBuff = (void*)((char*)coll->sendbuff + r * coll->count * elemSize);
      void* recvBuff = (void*)((char*)coll->recvbuff + r * coll->count * elemSize);
      NCCLCHECK(enqueueP2pTask(comm, p2pTaskQueue, ncclFuncSend, collAPI, sendBuff, coll->count, coll->datatype, r,
                               stream, launchCompletionEvent, allowUB, tInfo->raw->profiler));
      NCCLCHECK(enqueueP2pTask(comm, p2pTaskQueue, ncclFuncRecv, collAPI, recvBuff, coll->count, coll->datatype, r,
                               stream, launchCompletionEvent, allowUB, tInfo->raw->profiler));
    }
  } else if (coll->func == ncclFuncGather) {
    NCCLCHECK(enqueueP2pTask(comm, p2pTaskQueue, ncclFuncSend, collAPI, (void*)coll->sendbuff, coll->count,
                             coll->datatype, coll->root, stream, launchCompletionEvent, allowUB, tInfo->raw->profiler));
    if (comm->rank == coll->root) {
      size_t offset = 0;
      for (int r = 0; r < comm->nRanks; r++) {
        void* buff = (void*)((char*)coll->recvbuff + offset);
        NCCLCHECK(enqueueP2pTask(comm, p2pTaskQueue, ncclFuncRecv, collAPI, buff, coll->count, coll->datatype, r,
                                 stream, launchCompletionEvent, allowUB, tInfo->raw->profiler));
        offset += coll->count * elemSize;
      }
    }
  } else if (coll->func == ncclFuncScatter) {
    if (comm->rank == coll->root) {
      size_t offset = 0;
      for (int r = 0; r < comm->nRanks; r++) {
        void* buff = (void*)((char*)coll->sendbuff + offset);
        NCCLCHECK(enqueueP2pTask(comm, p2pTaskQueue, ncclFuncSend, collAPI, buff, coll->count, coll->datatype, r,
                                 stream, launchCompletionEvent, allowUB, tInfo->raw->profiler));
        offset += coll->count * elemSize;
      }
    }
    NCCLCHECK(enqueueP2pTask(comm, p2pTaskQueue, ncclFuncRecv, collAPI, coll->recvbuff, coll->count, coll->datatype,
                             coll->root, stream, launchCompletionEvent, allowUB, tInfo->raw->profiler));
  } else {
    return ncclInternalError;
  }

  ncclMemoryPoolFree(&comm->memPool_ncclRawTask, tInfo->raw);
  tInfo->raw = nullptr;
  return ncclSuccess;
}

static bool isSymKernel(struct ncclTaskTuningInfo* tInfo) {
  return tInfo->tuningOut.valid && tInfo->tuningOut.symKernelId >= 0 &&
         tInfo->tuningOut.symKernelId < ncclSymkKernelId_Count;
}

static bool isCeKernel(struct ncclTaskTuningInfo* tInfo) {
  return tInfo->tuningOut.valid && tInfo->tuningOut.ceMethodId >= 0 &&
         tInfo->tuningOut.ceMethodId < ncclCeMethodId_Count;
}

static bool isTaskAllGatherV(struct ncclTaskTuningInfo* tInfo) {
  return tInfo->raw->kind == ncclTaskKindAllGatherV;
}

ncclResult_t ncclTaskClassification(struct ncclComm* comm, struct ncclTaskTuningInfoQueue* tiq,
                                    struct ncclClassifiedTaskQueues* ctq) {
  struct ncclTaskTuningInfo* tInfo;
  struct ncclIntruQueue<struct ncclTaskTuningInfo, &ncclTaskTuningInfo::next>* outQueue;

  if (comm == nullptr || tiq == nullptr || ctq == nullptr) return ncclInvalidArgument;

  ncclClassifiedTaskQueuesConstruct(ctq);

  while (!ncclIntruQueueEmpty(&tiq->queue)) {
    tInfo = ncclIntruQueueDequeue(&tiq->queue);
    if (tInfo->raw == nullptr) return ncclInternalError;

    if (tInfo->raw->kind == ncclTaskKindColl &&
        (tInfo->raw->coll.func == ncclFuncAlltoAll || tInfo->raw->coll.func == ncclFuncScatter ||
         tInfo->raw->coll.func == ncclFuncGather) &&
        !isCeKernel(tInfo) && !isSymKernel(tInfo)) {
      NCCLCHECK(collToP2pTasks(comm, tInfo, &ctq->queues[ncclTaskQueueP2p]));
      continue;
    }

    if (tInfo->raw->kind == ncclTaskKindSendRecv) {
      outQueue = &ctq->queues[ncclTaskQueueP2p];
    } else if (tInfo->raw->kind == ncclTaskKindRma) {
      outQueue = &ctq->queues[ncclTaskQueueRma];
    } else if (isCeKernel(tInfo)) {
      outQueue = &ctq->queues[ncclTaskQueueCe];
    } else if (isSymKernel(tInfo)) {
      outQueue = &ctq->queues[ncclTaskQueueSym];
    } else if (isTaskAllGatherV(tInfo)) {
      outQueue = &ctq->queues[ncclTaskQueueAllGatherV];
    } else {
      outQueue = &ctq->queues[ncclTaskQueueLegacy];
    }
    ncclIntruQueueEnqueue(outQueue, tInfo);
  }
  return ncclSuccess;
}
