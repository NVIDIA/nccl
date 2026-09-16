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

  if (comm == nullptr || ctq == nullptr) return ncclInvalidArgument;
  if (comm->sharedRes == nullptr || comm->context == nullptr) return ncclInternalError;

  depStream = &comm->sharedRes->deviceStream;
  launchOrderStream = &comm->context->launchOrder;
  // Post-tuning has already drained the classified queues into the planner, so the planner's
  // task counts -- not the queues -- tell us whether the legacy launcher has work. RMA, sym and
  // CE tasks are owned by the schedulers below and are deliberately not counted here.
  if (comm->planner.nTasksColl + comm->planner.nTasksP2p + comm->planner.nTasksBcast != 0) {
    NCCLCHECK(doLaunches(comm));
  }
  NCCLCHECK(ncclScheduleSymTasks(comm, &ctq->symTaskQueue, depStream, launchOrderStream));
  NCCLCHECK(ncclScheduleRmaTasks(comm, &ctq->rmaTaskQueue, depStream, launchOrderStream));
  NCCLCHECK(ncclScheduleCeTasks(comm, &ctq->ceTaskQueue, depStream, launchOrderStream));
  return ncclSuccess;
}
