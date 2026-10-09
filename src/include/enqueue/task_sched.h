/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_ENQUEUE_TASK_SCHED_H_
#define NCCL_ENQUEUE_TASK_SCHED_H_

#include "task_classify.h"
#include "strongstream.h"

ncclResult_t ncclSchedPollCallbacks(struct ncclComm* comm);

typedef ncclResult_t (*ncclTaskSchedLaunchFn)(
  struct ncclComm* comm, struct ncclIntruQueue<struct ncclTaskTuningInfo, &ncclTaskTuningInfo::next>* queue,
  struct ncclStrongStream* depStream, struct ncclStrongStream* launchOrderStream);

// One scheduler for a classified queue. ncclTaskSchedLaunches in task_sched.cc is indexed by ncclTaskQueueType.
struct ncclTaskSchedLaunch {
  ncclTaskSchedLaunchFn schedLaunch;
};

ncclResult_t ncclSchedLaunchSymTasks(struct ncclComm* comm,
                                     struct ncclIntruQueue<struct ncclTaskTuningInfo, &ncclTaskTuningInfo::next>* queue,
                                     struct ncclStrongStream* depStream, struct ncclStrongStream* launchOrderStream);

ncclResult_t ncclSchedLaunchRmaTasks(struct ncclComm* comm,
                                     struct ncclIntruQueue<struct ncclTaskTuningInfo, &ncclTaskTuningInfo::next>* queue,
                                     struct ncclStrongStream* depStream, struct ncclStrongStream* launchOrderStream);

ncclResult_t ncclSchedLaunchCeTasks(struct ncclComm* comm,
                                    struct ncclIntruQueue<struct ncclTaskTuningInfo, &ncclTaskTuningInfo::next>* queue,
                                    struct ncclStrongStream* depStream, struct ncclStrongStream* launchOrderStream);

ncclResult_t ncclTaskSchedLaunch(struct ncclComm* comm, struct ncclClassifiedTaskQueues* ctq);

#endif // NCCL_ENQUEUE_TASK_SCHED_H_
