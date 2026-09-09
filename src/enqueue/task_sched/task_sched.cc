/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "comm.h"
#include "group.h"
#include "enqueue/task_sched.h"

ncclResult_t ncclTaskSchedule(struct ncclComm* comm, struct ncclClassifiedTaskQueues* ctq) {
  struct ncclStrongStream* depStream;
  struct ncclStrongStream* launchOrderStream;
  (void)depStream;
  (void)launchOrderStream;

  if (comm == nullptr || ctq == nullptr) return ncclInvalidArgument;
  if (comm->sharedRes == nullptr || comm->context == nullptr) return ncclInternalError;

  /* Keep it for future when we implement the proper schedulers
    depStream = &comm->sharedRes->deviceStream;
    launchOrderStream = &comm->context->launchOrder;
    NCCLCHECK(ncclScheduleSymTasks(comm, &ctq->symTaskQueue, depStream, launchOrderStream));
    if (!ncclIntruQueueEmpty(&ctq->legacyTaskQueue)) {
      NCCLCHECK(ncclScheduleLegacyTasks(comm, &ctq->legacyTaskQueue, depStream, launchOrderStream));
    } else if (!ncclIntruQueueEmpty(&ctq->p2pTaskQueue)) {
      NCCLCHECK(ncclScheduleP2pTasks(comm, &ctq->p2pTaskQueue, depStream, launchOrderStream));
    } else if (!ncclIntruQueueEmpty(&ctq->allgathervTaskQueue)) {
      NCCLCHECK(ncclScheduleAllGatherVTasks(comm, &ctq->allgathervTaskQueue, depStream, launchOrderStream));
    }
    NCCLCHECK(ncclScheduleRmaTasks(comm, &ctq->rmaTaskQueue, depStream, launchOrderStream));
    NCCLCHECK(ncclScheduleCeTasks(comm, &ctq->ceTaskQueue, depStream, launchOrderStream));
  */
  NCCLCHECK(doLaunches(comm));
  return ncclSuccess;
}
