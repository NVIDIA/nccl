/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "comm.h"
#include "enqueue.h"

ncclResult_t ncclTaskPrepare(struct ncclComm* comm, ncclSimInfo_t* simInfo) {
  struct ncclTaskTuningInfo* tInfo;
  struct ncclTaskTuningInfoQueue taskTuningInfoQueue;

  ncclIntruQueueConstruct(&taskTuningInfoQueue.queue);
  NCCLCHECK(ncclTaskPreTuning(comm, &comm->rawTaskQueue, &taskTuningInfoQueue));

  tInfo = ncclIntruQueueHead(&taskTuningInfoQueue.queue);
  for (; tInfo != nullptr; tInfo = tInfo->next) {
    // need tuning to support all functions.
    if (tInfo->raw->kind == ncclTaskKindColl && tInfo->tuningIn.func != ncclFuncAlltoAll &&
        tInfo->tuningIn.func != ncclFuncScatter && tInfo->tuningIn.func != ncclFuncGather &&
        tInfo->tuningIn.func != ncclFuncAllGatherV) {
      NCCLCHECK(ncclTuningCompute(&tInfo->tuningIn, &tInfo->tuningOut));
    }
  }

  NCCLCHECK(ncclTaskClassification(comm, &taskTuningInfoQueue, &comm->classifiedTaskQueues));

  NCCLCHECK(ncclTaskPostTuning(comm, &comm->classifiedTaskQueues, simInfo));

  return ncclSuccess;
}
