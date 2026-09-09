/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "group.h"
#include "debug.h"

ncclResult_t ncclMgmtTaskEnqueue(struct ncclAsyncJob* task, ncclResult_t (*func)(struct ncclAsyncJob*),
                                 void (*destructor)(void*), ncclComm_t comm) {
  ncclResult_t ret = ncclSuccess;

  task->destroyFlag = comm->destroyFlag;
  task->func = func;
  task->destructor = destructor;
  task->abortFlag = comm->abortFlag;
  task->abortFlagDev = comm->abortFlagDev;
  task->childAbortFlag = comm->childAbortFlag;
  task->childAbortFlagDev = comm->childAbortFlagDev;
  task->state = ncclGroupJobRunning;
  task->comm = comm;
  ncclGroupCommJoin(comm, ncclGroupTaskTypeMgmtTask);
  /* check if there are blocking and nonblocking comms at the same time in group. */
  if (comm->destroyFlag) {
    ncclGroupBlocking = 1;
  } else if (ncclGroupBlocking == -1) {
    /* first met communicator */
    ncclGroupBlocking = comm->config.blocking;
  } else if (ncclGroupBlocking != comm->config.blocking) {
    WARN("Blocking and nonblocking communicators are not allowed in the same group.");
    ret = ncclInvalidArgument;
  }
  if (ret == ncclSuccess) {
    ncclIntruQueueEnqueue(&comm->mgmtTaskQueue, task);
  } else {
    // the task hasn't run
    if (destructor) destructor(task);
  }

  return ret;
}
