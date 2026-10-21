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

static ncclResult_t prepareTaskTuningInfo(struct ncclComm* comm, struct ncclRawTask* raw,
                                          struct ncclTaskTuningInfoQueue* tiq);

static ncclResult_t fillCollTuningInput(struct ncclComm* comm, struct ncclRawTaskColl* raw,
                                        const ncclCollConfig_t* collConfig, ncclTuningInput_t* in) {
  size_t userElementSize;
  size_t sendbuffSize;
  size_t recvbuffSize;
  struct ncclDevrWindow* sendWin;
  struct ncclDevrWindow* recvWin;
  struct ncclReg* regSendBuf;
  struct ncclReg* regRecvBuf;
  bool isSendValid;
  bool isRecvValid;

  in->comm = comm;
  in->tuningMask = ncclCommDefaultAlgoMask(comm);
  in->algSelection = raw->algMask;
  in->forceAlgSelection = raw->algMask != 0 ? collConfig->forceAlgSelection : 0;
  // AllGather and Broadcast are counted in bytes from here on, matching the legacy enqueue
  // path: their device kernels are type agnostic and read the work count as a byte count.
  // Normalizing the raw task keeps every consumer downstream of pre-tuning in those units.
  if (raw->func == ncclFuncAllGather || raw->func == ncclFuncBroadcast) {
    raw->count *= ncclTypeSize(raw->datatype);
    raw->datatype = ncclInt8;
  }
  in->func = raw->func;
  in->redOp = raw->opHost;
  in->devRedOp = raw->opDev.op;
  in->datatype = raw->datatype;
  in->count = raw->count;
  in->countMax = raw->count;
  in->chunkSteps = ncclDefaultChunkStep(raw->func);
  in->sliceSteps = ncclDefaultSliceStep(raw->func);
  in->nWorks = 1;
  in->numPipeOps = 1;
  userElementSize = ncclTypeSize(raw->datatype);
  // Treat an AllGather as in-place when each rank sends from its own receive slice.
  in->inPlace = in->func == ncclFuncAllGather &&
                ncclAllGatherIsInPlace(raw->sendbuff, raw->recvbuff, comm->rank, raw->count * userElementSize);
  if (in->func == ncclFuncAllGather || in->func == ncclFuncBroadcast || in->func == ncclFuncAlltoAll) {
    in->count *= userElementSize;
    in->datatype = ncclInt8;
    in->countMax = in->count;
  }
  in->nBytes = ncclTypeSize(in->datatype) * ncclFuncMaxSendRecvCount(in->func, comm->nRanks, in->count);

  NCCLCHECK(ncclGetCollNetSupport(comm, raw, &in->collNetSupport));

  NCCLCHECK(ncclDevrFindWindow(comm, raw->sendbuff, &sendWin));
  NCCLCHECK(ncclDevrFindWindow(comm, raw->recvbuff, &recvWin));
  in->winRegType = ncclSymSendNonregRecvNonreg;
  if (sendWin != nullptr || recvWin != nullptr) {
    NCCLCHECK(ncclGetSymRegType(sendWin, recvWin, &in->winRegType));
  }
  // Symmetric kernels do not support windows that contain host-backed segments, nor
  // communicators that span NVLink cliques.
  if (comm->p2pCrossClique || ncclDevrWindowHasSysmemSegment(sendWin) || ncclDevrWindowHasSysmemSegment(recvWin)) {
    in->tuningMask &= ~NCCL_TUNING_MASK_SYM_KERNELS;
  }
  // The copy-engine models do not check availability themselves, so keep their entries out of the
  // selection when no copy-engine route can run this collective.
  if (!ncclCeAvailable(comm, raw->func, raw->opDev.op, raw->datatype, in->winRegType, sendWin, recvWin) &&
      !ncclHierCeAvailable(comm, raw->func, raw->opDev.op, raw->datatype, in->winRegType, sendWin, recvWin)) {
    in->tuningMask &= ~NCCL_TUNING_MASK_CE;
  }
  in->nvlsSupport = ncclNvlsTransportEnabled(comm) &&
                    (ncclNvlsSupported(raw->opDev.op, raw->datatype) || raw->func == ncclFuncAllGather);
  {
    size_t inputOff = sendWin ? (uintptr_t)raw->sendbuff - (uintptr_t)sendWin->userPtr : (uintptr_t)raw->sendbuff;
    size_t outputOff = recvWin ? (uintptr_t)raw->recvbuff - (uintptr_t)recvWin->userPtr : (uintptr_t)raw->recvbuff;
    in->symAligned16B = (uint32_t(inputOff - outputOff) % 16 == 0);
  }

  // Registration covers the user buffers, whose element type is unchanged by
  // the byte normalization above.
  sendbuffSize = userElementSize * ncclFuncSendCount(raw->func, comm->nRanks, raw->count);
  recvbuffSize = userElementSize * ncclFuncRecvCount(raw->func, comm->nRanks, raw->count);
  NCCLCHECK(ncclRegFind(comm, raw->sendbuff, sendbuffSize, &regSendBuf));
  NCCLCHECK(ncclRegFind(comm, raw->recvbuff, recvbuffSize, &regRecvBuf));
  NCCLCHECK(ncclRegLocalIsValid(regSendBuf, &isSendValid));
  NCCLCHECK(ncclRegLocalIsValid(regRecvBuf, &isRecvValid));
  in->captured = ncclCudaGraphValid(comm->planner.capturingGraph);
  in->regBuff = (regSendBuf && regRecvBuf && isSendValid && isRecvValid) || (in->captured && ncclParamGraphRegister());
  {
    bool ctaPolicyEnvOverridden = ncclGetEnvCtaPolicy() != NCCL_CONFIG_UNDEF_INT;
    NCCL_CONFIG_SET(in, minCTAs, ncclParamMinCTAs(), collConfig->minCTAs, comm->config.minCTAs, 1, MAXCHANNELS);
    NCCL_CONFIG_SET(in, maxCTAs, ncclParamMaxCTAs(), std::min(collConfig->maxCTAs, comm->config.maxCTAs),
                    comm->config.maxCTAs, 1, MAXCHANNELS);
    if (in->minCTAs > in->maxCTAs) in->minCTAs = 1;
    in->CTAPolicy =
      ncclCollConfigResolveCTAPolicy(collConfig->CTAPolicy, comm->config.CTAPolicy, ctaPolicyEnvOverridden);
  }
  return ncclSuccess;
}

static ncclResult_t fillSendRecvTuningInput(struct ncclComm* comm, struct ncclRawTaskSendRecv* raw,
                                            ncclTuningInput_t* in) {
  struct ncclReg* regBuf;
  bool isValid;

  in->comm = comm;
  in->tuningMask = ncclCommDefaultAlgoMask(comm);
  in->forceAlgSelection = 0;
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
  in->tuningMask = ncclCommDefaultAlgoMask(comm);
  in->forceAlgSelection = 0;
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
                                              const ncclCollConfig_t* collConfig, ncclTuningInput_t* in) {
  in->comm = comm;
  // Only general kernels run an AllGatherV.
  in->tuningMask = ncclCommDefaultAlgoMask(comm) & NCCL_TUNING_MASK_GENERAL_KERNELS;
  in->algSelection = raw->algMask;
  in->forceAlgSelection = raw->algMask != 0 ? collConfig->forceAlgSelection : 0;
  // AllGatherV uses same tuning as AllGather
  in->func = ncclFuncAllGather;
  in->datatype = ncclInt8;
  in->nWorks = 1;
  in->numPipeOps = 1;
  in->count = raw->maxCount;
  in->countMax = raw->maxCount;
  in->nBytes = raw->maxCount;
  // The AllGatherV scheduler sizes its chunks from a single step.
  in->chunkSteps = in->sliceSteps = 1;
  {
    bool ctaPolicyEnvOverridden = ncclGetEnvCtaPolicy() != NCCL_CONFIG_UNDEF_INT;
    NCCL_CONFIG_SET(in, minCTAs, ncclParamMinCTAs(), collConfig->minCTAs, comm->config.minCTAs, 1, MAXCHANNELS);
    NCCL_CONFIG_SET(in, maxCTAs, ncclParamMaxCTAs(), std::min(collConfig->maxCTAs, comm->config.maxCTAs),
                    comm->config.maxCTAs, 1, MAXCHANNELS);
    if (in->minCTAs > in->maxCTAs) in->minCTAs = 1;
    in->CTAPolicy =
      ncclCollConfigResolveCTAPolicy(collConfig->CTAPolicy, comm->config.CTAPolicy, ctaPolicyEnvOverridden);
  }
  return ncclSuccess;
}

static ncclResult_t initAllGatherVRaw(struct ncclRawTask* raw) {
  struct ncclRawTaskAllGatherV* agv = &raw->allGatherV;

  raw->kind = ncclTaskKindAllGatherV;
  raw->collConfig = NCCL_COLLCONFIG_INITIALIZER;
  raw->collConfig.size = 0;
  agv->maxCount = 0;
  agv->stream = (cudaStream_t)-1; // invalid stream
  agv->algMask = 0;
  NEW_NOTHROW(agv->bcastTasks, ncclRawTaskAllGatherVBcastMap);
  return ncclSuccess;
}

static ncclResult_t addBcastToAllGatherV(struct ncclComm* comm, struct ncclRawTask* agvRaw,
                                         struct ncclRawTask* bcastRaw) {
  struct ncclRawTaskAllGatherV* agv = &agvRaw->allGatherV;
  struct ncclRawTaskColl* bcast = &bcastRaw->coll;
  size_t count = bcast->count * ncclTypeSize(bcast->datatype);
  ncclRawTaskAllGatherVBcast contrib;

  if (agv->stream == (cudaStream_t)-1) {
    agv->stream = bcast->stream;
    agvRaw->profiler = bcastRaw->profiler;
  }
  // Simply use the first bcast's collConfig for now
  if (bcastRaw->collConfig.size != 0 && agvRaw->collConfig.size == 0) {
    agvRaw->collConfig = bcastRaw->collConfig;
    agv->algMask = bcast->algMask != 0 ? bcast->algMask : NCCL_TUNING_MASK_ALL;
  }
  if (count > agv->maxCount) agv->maxCount = count;
  contrib.sendbuff = (void*)bcast->sendbuff;
  contrib.recvbuff = bcast->recvbuff;
  contrib.count = count;
  contrib.profiler = bcastRaw->profiler;
  (*agv->bcastTasks)[bcast->root] = contrib;
  ncclMemoryPoolFree(&comm->memPool_ncclRawTask, bcastRaw);
  return ncclSuccess;
}

static ncclResult_t mergeBcastTasks(struct ncclComm* comm, ncclRawTaskBcastQueueMap* bcastQueue,
                                    struct ncclTaskTuningInfoQueue* tiq) {
  if (bcastQueue == nullptr) return ncclSuccess;

  // Each pass builds one AllGatherV by taking a single broadcast from every non-empty root
  // bucket. Repeat until every bucket is empty.
  while (!bcastQueue->empty()) {
    struct ncclRawTask* agvRaw =
      ncclMemoryPoolAlloc<struct ncclRawTask>(&comm->memPool_ncclRawTask, &comm->memPermanent);
    NCCLCHECK(initAllGatherVRaw(agvRaw));

    for (auto it = bcastQueue->begin(); it != bcastQueue->end();) {
      if (ncclIntruQueueEmpty(&it->second)) {
        it = bcastQueue->erase(it);
        continue;
      }
      struct ncclRawTask* bcastRaw = ncclIntruQueueDequeue(&it->second);
      NCCLCHECK(addBcastToAllGatherV(comm, agvRaw, bcastRaw));
      if (ncclIntruQueueEmpty(&it->second)) it = bcastQueue->erase(it);
      else ++it;
    }

    NCCLCHECK(prepareTaskTuningInfo(comm, agvRaw, tiq));
  }
  return ncclSuccess;
}

static ncclResult_t prepareTaskTuningInfo(struct ncclComm* comm, struct ncclRawTask* raw,
                                          struct ncclTaskTuningInfoQueue* tiq) {
  struct ncclTaskTuningInfo* pt = ncclMemoryStackAlloc<struct ncclTaskTuningInfo>(&comm->memScoped);
  pt->raw = raw;
  pt->tuningOut = NCCL_TUNING_RESULT_INIT;
  switch (raw->kind) {
  case ncclTaskKindColl:
    NCCLCHECK(fillCollTuningInput(comm, &raw->coll, &raw->collConfig, &pt->tuningIn));
    break;
  case ncclTaskKindSendRecv:
    NCCLCHECK(fillSendRecvTuningInput(comm, &raw->sendRecv, &pt->tuningIn));
    break;
  case ncclTaskKindRma:
    NCCLCHECK(fillRmaTuningInput(comm, &raw->rma, &pt->tuningIn));
    break;
  case ncclTaskKindAllGatherV:
    NCCLCHECK(fillAllGatherVTuningInput(comm, &raw->allGatherV, &raw->collConfig, &pt->tuningIn));
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
    NCCLCHECK(prepareTaskTuningInfo(comm, raw, tiq));
  }
  NCCLCHECK(mergeBcastTasks(comm, rtq->bcastQueue, tiq));
  return ncclSuccess;
}
