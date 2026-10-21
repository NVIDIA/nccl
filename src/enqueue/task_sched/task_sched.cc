/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "comm.h"
#include "group.h"
#include "enqueue/task_sched.h"

static_assert(ncclTaskQueueSym == 0 && ncclTaskQueueRma == 1 && ncclTaskQueueCe == 2,
              "ncclTaskSchedLaunches is indexed by ncclTaskQueueType");

static struct ncclTaskSchedLaunch ncclTaskSchedLaunches[] = {
  {ncclSchedLaunchSymTasks},
  {ncclSchedLaunchRmaTasks},
  {ncclSchedLaunchCeTasks},
};

// Run callbacks queued by CUDA graph destructors. Destructors cannot call CUDA, so they
// only enqueue. fn may free args, and this frees the callback. Every queued callback
// runs; the first error is returned.
ncclResult_t ncclSchedPollCallbacks(struct ncclComm* comm) {
  ncclResult_t result = ncclSuccess;
  struct ncclSchedCallback* cb = ncclIntruQueueMpscDequeueAll(&comm->schedCallbackQueue, /*waitSome=*/false);
  while (cb != nullptr) {
    struct ncclSchedCallback* next = cb->next;
    ncclResult_t res1 = cb->fn(comm, cb->args); // may free args
    free(cb);
    if (res1 != ncclSuccess) result = res1;
    cb = next;
  }
  NCCLCHECK(result);
  return ncclSuccess;
}

ncclResult_t ncclTaskSchedLaunch(struct ncclComm* comm, struct ncclClassifiedTaskQueues* ctq) {
  struct ncclStrongStream* depStream;
  struct ncclStrongStream* launchOrderStream;

  if (comm == nullptr || ctq == nullptr) return ncclInvalidArgument;
  if (comm->sharedRes == nullptr || comm->context == nullptr) return ncclInternalError;

  NCCLCHECK(ncclSchedPollCallbacks(comm));

  depStream = &comm->sharedRes->deviceStream;
  launchOrderStream = &comm->context->launchOrder;
  // Post-tuning has already drained the classified queues into the planner, so the planner's
  // task counts -- not the queues -- tell us whether the legacy launcher has work. RMA, sym and
  // CE tasks are owned by the schedulers below and are deliberately not counted here. General
  // kernels and AlltoAll LL fall back to the legacy symmetric queue, which the task counts leave out.
  if (comm->planner.nTasksColl + comm->planner.nTasksP2p + comm->planner.nTasksBcast != 0 ||
      !ncclIntruQueueEmpty(&comm->planner.collSymTaskQueue)) {
    NCCLCHECK(doLaunches(comm));
  }
  for (int type = ncclTaskQueueSym; type <= ncclTaskQueueCe; type++) {
    NCCLCHECK(ncclTaskSchedLaunches[type].schedLaunch(comm, &ctq->queues[type], depStream, launchOrderStream));
  }
  return ncclSuccess;
}
