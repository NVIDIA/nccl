/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "comm.h"
#include "enqueue.h"
#include "bootstrap.h"
#include "group.h"

// Every rank takes part in a collective, so the exchange below can only pair up in a group that
// holds one; send/recv and put/signal groups may involve a subset of the ranks.
static bool rawTaskQueueHasCollective(struct ncclRawTaskQueue* rtq) {
  if (rtq->bcastQueue != nullptr && !rtq->bcastQueue->empty()) return true;
  for (struct ncclRawTask* task = ncclIntruQueueHead(&rtq->genericQueue); task != nullptr; task = task->next) {
    if (task->kind == ncclTaskKindColl || task->kind == ncclTaskKindAllGatherV) return true;
  }
  return false;
}

// One-time check at the communicator's first group end holding a collective. When one thread
// drives several ranks of the communicator in some process, every rank switches to the legacy
// enqueue path, whose group end does not spawn a thread per rank. The ranks exchange their answer
// so they all pick the same path; the switch takes effect from the next group on.
static ncclResult_t enqueuePathDecide(struct ncclComm* comm) {
  ncclResult_t ret = ncclSuccess;
  int* shared = nullptr;

  if (comm->enqueuePath != ncclEnqueuePathPending || !rawTaskQueueHasCollective(&comm->rawTaskQueue)) {
    return ncclSuccess;
  }
  // The group list keeps the ranks of one communicator that this thread drives consecutive, so
  // the first of them is followed by another one whenever there are several.
  struct ncclComm* next = comm->groupNext[ncclGroupTaskTypeRawTask];
  NCCLCHECK(ncclCalloc(&shared, comm->nRanks));
  shared[comm->rank] = next != nullptr && next != reinterpret_cast<struct ncclComm*>(NCCL_COMM_GROUP_INVALID) &&
                       next->intraComm0 == comm->intraComm0;
  NCCLCHECKGOTO(bootstrapAllGather(comm->bootstrap, shared, sizeof(int)), ret, exit);
  comm->enqueuePath = ncclEnqueuePathRearch;
  for (int r = 0; r < comm->nRanks; r++) {
    if (shared[r]) comm->enqueuePath = ncclEnqueuePathLegacy;
  }
  if (comm->enqueuePath == ncclEnqueuePathLegacy) {
    INFO(NCCL_INIT, "One thread drives several ranks of this communicator; using the legacy enqueue path");
  }
exit:
  free(shared);
  return ret;
}

ncclResult_t ncclTaskPrepare(struct ncclComm* comm, ncclSimInfo_t* simInfo) {
  struct ncclTaskTuningInfo* tInfo;
  struct ncclTaskTuningInfoQueue taskTuningInfoQueue;

  // A simulation only estimates tuner cost, so it does not decide the enqueue path.
  if (simInfo == nullptr) NCCLCHECK(enqueuePathDecide(comm));
  ncclIntruQueueConstruct(&taskTuningInfoQueue.queue);
  NCCLCHECK(ncclTaskPreTuning(comm, &comm->rawTaskQueue, &taskTuningInfoQueue));

  tInfo = ncclIntruQueueHead(&taskTuningInfoQueue.queue);
  for (; tInfo != nullptr; tInfo = tInfo->next) {
    NCCLCHECK(ncclTuningCompute(&tInfo->tuningIn, &tInfo->tuningOut));
  }

  NCCLCHECK(ncclTaskClassification(comm, &taskTuningInfoQueue, &comm->classifiedTaskQueues));

  NCCLCHECK(ncclTaskPostTuning(comm, &comm->classifiedTaskQueues, simInfo));

  return ncclSuccess;
}
