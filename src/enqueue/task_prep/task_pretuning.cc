/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "enqueue.h"
#include "ce_coll.h"
#include "checks.h"
#include "debug.h"
#include "dev_runtime.h"
#include "enqueue/task_pretuning.h"
#include "register.h"
#include "register_inline.h"
#include <algorithm>

int64_t ncclParamMinCTAs();
int64_t ncclParamMaxCTAs();

static ncclResult_t preTuningEnqueueRawTask(struct ncclComm* comm, struct ncclRawTask* raw,
                                            struct ncclTaskTuningInfoQueue* tiq);

static ncclResult_t fillCollTuningInput(struct ncclComm* comm, struct ncclRawTaskColl* raw, ncclTuningInput_t* in) {
  size_t elementSize;
  size_t sendbuffSize;
  size_t recvbuffSize;
  struct ncclDevrWindow* sendWin;
  struct ncclDevrWindow* recvWin;
  struct ncclReg* regSendBuf;
  struct ncclReg* regRecvBuf;
  bool isSendValid;
  bool isRecvValid;

  in->comm = comm;
  // disable ce temporarily
  in->tuningMask = NCCL_TUNING_MASK_SYM_KERNELS | NCCL_TUNING_MASK_GENERAL_KERNELS;
  in->func = raw->func;
  in->redOp = raw->opHost;
  in->devRedOp = raw->opDev.op;
  in->datatype = raw->datatype;
  in->count = raw->count;
  in->countMax = raw->count;
  in->nWorks = 1;
  in->numPipeOps = 1;
  elementSize = ncclTypeSize(in->datatype);
  if (in->func == ncclFuncAllGather || in->func == ncclFuncBroadcast) {
    in->count *= elementSize;
    in->datatype = ncclInt8;
    in->countMax = in->count;
    elementSize = 1;
  }
  in->nBytes = elementSize * ncclFuncMaxSendRecvCount(in->func, comm->nRanks, in->count);

  NCCLCHECK(ncclDevrFindWindow(comm, raw->sendbuff, &sendWin));
  NCCLCHECK(ncclDevrFindWindow(comm, raw->recvbuff, &recvWin));
  if (sendWin != nullptr || recvWin != nullptr) {
    NCCLCHECK(ncclGetSymRegType(sendWin, recvWin, &in->winRegType));
  }
  in->nvlsSupport =
    comm->nvlsSupport && (ncclNvlsSupported(raw->opDev.op, raw->datatype) || raw->func == ncclFuncAllGather);
  {
    size_t inputOff = sendWin ? (uintptr_t)raw->sendbuff - (uintptr_t)sendWin->userPtr : (uintptr_t)raw->sendbuff;
    size_t outputOff = recvWin ? (uintptr_t)raw->recvbuff - (uintptr_t)recvWin->userPtr : (uintptr_t)raw->recvbuff;
    in->symAligned16B = (uint32_t(inputOff - outputOff) % 16 == 0);
  }

  sendbuffSize = elementSize * ncclFuncSendCount(raw->func, comm->nRanks, raw->count);
  recvbuffSize = elementSize * ncclFuncRecvCount(raw->func, comm->nRanks, raw->count);
  NCCLCHECK(ncclRegFind(comm, raw->sendbuff, sendbuffSize, &regSendBuf));
  NCCLCHECK(ncclRegFind(comm, raw->recvbuff, recvbuffSize, &regRecvBuf));
  NCCLCHECK(ncclRegLocalIsValid(regSendBuf, &isSendValid));
  NCCLCHECK(ncclRegLocalIsValid(regRecvBuf, &isRecvValid));
  in->regBuff = (regSendBuf && regRecvBuf && isSendValid && isRecvValid) ||
                (ncclCudaGraphValid(comm->planner.capturingGraph) && ncclParamGraphRegister());
  NCCL_CONFIG_SET(in, minCTAs, ncclParamMinCTAs(), raw->collConfig.minCTAs, comm->config.minCTAs, 1, MAXCHANNELS);
  NCCL_CONFIG_SET(in, maxCTAs, ncclParamMaxCTAs(), std::min(raw->collConfig.maxCTAs, comm->config.maxCTAs),
                  comm->config.maxCTAs, 1, MAXCHANNELS);
  if (in->minCTAs > in->maxCTAs) in->minCTAs = 1;
  return ncclSuccess;
}

static ncclResult_t fillSendRecvTuningInput(struct ncclComm* comm, struct ncclRawTaskSendRecv* raw,
                                            ncclTuningInput_t* in) {
  struct ncclReg* regBuf;
  bool isValid;

  in->comm = comm;
  in->tuningMask = NCCL_TUNING_MASK_ALL;
  in->func = raw->func;
  in->datatype = raw->datatype;
  in->count = raw->count;
  in->countMax = raw->count;
  in->nWorks = 1;
  in->numPipeOps = 1;
  in->nBytes = raw->bytes;

  NCCLCHECK(ncclRegFind(comm, raw->buff, raw->bytes, &regBuf));
  NCCLCHECK(ncclRegLocalIsValid(regBuf, &isValid));
  in->regBuff = (regBuf && isValid) || (ncclCudaGraphValid(comm->planner.capturingGraph) && ncclParamGraphRegister());
  return ncclSuccess;
}

static ncclResult_t fillRmaTuningInput(struct ncclComm* comm, struct ncclRawTaskRma* raw, ncclTuningInput_t* in) {
  in->comm = comm;
  in->tuningMask = NCCL_TUNING_MASK_ALL;
  in->func = raw->func;
  in->nWorks = 1;
  in->numPipeOps = 1;

  if (raw->func == ncclFuncPutSignal) {
    in->datatype = raw->rmaOp.putSignal.datatype;
    in->count = raw->rmaOp.putSignal.count;
    in->countMax = raw->rmaOp.putSignal.count;
    in->nBytes = raw->rmaOp.putSignal.count * ncclTypeSize(raw->rmaOp.putSignal.datatype);
  } else if (raw->func == ncclFuncSignal || raw->func == ncclFuncWaitSignal) {
    in->datatype = ncclInt8;
    in->count = 0;
    in->countMax = 0;
    in->nBytes = 0;
  }
  return ncclSuccess;
}

static ncclResult_t fillAllGatherVTuningInput(struct ncclComm* comm, struct ncclRawTaskAllGatherV* raw,
                                              ncclTuningInput_t* in) {
  size_t elementSize = ncclTypeSize(raw->datatype);

  in->comm = comm;
  in->tuningMask = NCCL_TUNING_MASK_ALL;
  in->func = ncclFuncAllGatherV;
  in->datatype = raw->datatype;
  in->nWorks = 1;
  in->numPipeOps = 1;
  in->count = raw->maxCount;
  in->countMax = raw->maxCount;
  in->nBytes = raw->maxCount * elementSize;
  return ncclSuccess;
}

static ncclResult_t preTuningInitAllGatherVRaw(struct ncclComm* comm, struct ncclRawTask* raw) {
  struct ncclRawTaskAllGatherV* agv = &raw->allGatherV;
  int nRanks = comm->nRanks;

  raw->kind = ncclTaskKindAllGatherV;
  agv->func = ncclFuncAllGatherV;
  agv->nRanks = nRanks;
  agv->sendbuff = nullptr;
  agv->recvbuff = ncclMemoryStackAlloc<void*>(&comm->memScoped, nRanks);
  agv->counts = ncclMemoryStackAlloc<size_t>(&comm->memScoped, nRanks);
  agv->maxCount = 0;
  agv->datatype = ncclInt8;
  agv->stream = (cudaStream_t)-1; // invalid stream
  return ncclSuccess;
}

static bool preTuningAllGatherVRootUsed(struct ncclRawTaskAllGatherV* agv, int root) {
  return agv->counts[root] != 0;
}

static ncclResult_t preTuningAddBcastToAllGatherV(struct ncclComm* comm, struct ncclRawTaskAllGatherV* agv,
                                                  struct ncclRawTaskColl* bcast) {
  if (bcast->root < 0 || bcast->root >= agv->nRanks) return ncclInvalidArgument;

  agv->stream = bcast->stream;
  if (bcast->root == comm->rank) agv->sendbuff = bcast->sendbuff;
  agv->recvbuff[bcast->root] = bcast->recvbuff;
  size_t count = bcast->count * ncclTypeSize(bcast->datatype);
  agv->counts[bcast->root] = count;
  if (count > agv->maxCount) agv->maxCount = count;
  return ncclSuccess;
}

static bool preTuningRawTaskPtrEqual(struct ncclRawTask* a, struct ncclRawTask* b) {
  return a == b;
}

static bool preTuningBcastFallsBack(struct ncclRawTaskColl* bcast) {
  // it is temporary and will be improved in the future
  return true;
}

static bool preTuningBcastFitsAllGatherV(struct ncclRawTaskAllGatherV* agv, struct ncclRawTaskColl* bcast) {
  if (bcast->stream != agv->stream) return false;
  if (preTuningAllGatherVRootUsed(agv, bcast->root)) return false;
  return true;
}

static ncclResult_t preTuningRemoveBcastRawFromQueue(
  struct ncclComm* comm, struct ncclIntruQueue<struct ncclRawTask, &ncclRawTask::next>* bcastQueue,
  struct ncclRawTask* bcastRaw, bool freeRaw) {
  ncclIntruQueueDelete(bcastQueue, bcastRaw, preTuningRawTaskPtrEqual);
  if (freeRaw) ncclMemoryPoolFree(&comm->memPool_ncclRawTask, bcastRaw);
  return ncclSuccess;
}

static ncclResult_t preTuningMergeBcastQueue(struct ncclComm* comm,
                                             struct ncclIntruQueue<struct ncclRawTask, &ncclRawTask::next>* bcastQueue,
                                             struct ncclTaskTuningInfoQueue* tiq) {
  while (!ncclIntruQueueEmpty(bcastQueue)) {
    struct ncclRawTask* agvRaw = nullptr;

    for (struct ncclRawTask* bcastRaw = ncclIntruQueueHead(bcastQueue); bcastRaw != nullptr;) {
      struct ncclRawTask* next = bcastRaw->next;
      struct ncclRawTaskColl* bcast = &bcastRaw->coll;

      if (preTuningBcastFallsBack(bcast)) {
        NCCLCHECK(preTuningEnqueueRawTask(comm, bcastRaw, tiq));
        NCCLCHECK(preTuningRemoveBcastRawFromQueue(comm, bcastQueue, bcastRaw, /*freeRaw=*/false));
        bcastRaw = next;
        continue;
      }

      if (agvRaw != nullptr && !preTuningBcastFitsAllGatherV(&agvRaw->allGatherV, bcast)) {
        bcastRaw = next;
        continue;
      }

      if (agvRaw == nullptr) {
        agvRaw = ncclMemoryPoolAlloc<struct ncclRawTask>(&comm->memPool_ncclRawTask, &comm->memPermanent);
        NCCLCHECK(preTuningInitAllGatherVRaw(comm, agvRaw));
      }
      NCCLCHECK(preTuningAddBcastToAllGatherV(comm, &agvRaw->allGatherV, bcast));
      NCCLCHECK(preTuningRemoveBcastRawFromQueue(comm, bcastQueue, bcastRaw, /*freeRaw=*/true));
      bcastRaw = next;
    }

    if (agvRaw != nullptr) {
      NCCLCHECK(preTuningEnqueueRawTask(comm, agvRaw, tiq));
    }
  }
  return ncclSuccess;
}

static ncclResult_t preTuningEnqueueRawTask(struct ncclComm* comm, struct ncclRawTask* raw,
                                            struct ncclTaskTuningInfoQueue* tiq) {
  struct ncclTaskTuningInfo* pt = ncclMemoryStackAlloc<struct ncclTaskTuningInfo>(&comm->memScoped);
  pt->raw = raw;
  pt->tuningOut = NCCL_TUNING_RESULT_INIT;
  switch (raw->kind) {
  case ncclTaskKindColl:
    NCCLCHECK(fillCollTuningInput(comm, &raw->coll, &pt->tuningIn));
    break;
  case ncclTaskKindSendRecv:
    NCCLCHECK(fillSendRecvTuningInput(comm, &raw->sendRecv, &pt->tuningIn));
    break;
  case ncclTaskKindRma:
    NCCLCHECK(fillRmaTuningInput(comm, &raw->rma, &pt->tuningIn));
    break;
  case ncclTaskKindAllGatherV:
    NCCLCHECK(fillAllGatherVTuningInput(comm, &raw->allGatherV, &pt->tuningIn));
    break;
  default:
    return ncclInternalError;
  }
  ncclIntruQueueEnqueue(&tiq->queue, pt);
  return ncclSuccess;
}

ncclResult_t ncclTaskPreTuning(struct ncclComm* comm, struct ncclRawTaskQueue* rtq,
                               struct ncclTaskTuningInfoQueue* tiq) {
  struct ncclRawTask* raw;

  if (comm == nullptr || rtq == nullptr || tiq == nullptr) return ncclInvalidArgument;

  while (!ncclIntruQueueEmpty(&rtq->genericQueue)) {
    raw = ncclIntruQueueDequeue(&rtq->genericQueue);
    NCCLCHECK(preTuningEnqueueRawTask(comm, raw, tiq));
  }
  NCCLCHECK(preTuningMergeBcastQueue(comm, &rtq->bcastQueue, tiq));
  return ncclSuccess;
}
