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

ncclResult_t ncclScheduleSymTasks(struct ncclComm* comm,
                                  struct ncclIntruQueue<struct ncclTaskTuningInfo, &ncclTaskTuningInfo::next>* queue,
                                  struct ncclStrongStream* depStream, struct ncclStrongStream* launchOrderStream);

ncclResult_t ncclScheduleLegacyTasks(struct ncclComm* comm,
                                     struct ncclIntruQueue<struct ncclTaskTuningInfo, &ncclTaskTuningInfo::next>* queue,
                                     struct ncclStrongStream* depStream, struct ncclStrongStream* launchOrderStream);

ncclResult_t ncclScheduleAllGatherVTasks(
  struct ncclComm* comm, struct ncclIntruQueue<struct ncclTaskTuningInfo, &ncclTaskTuningInfo::next>* queue,
  struct ncclStrongStream* depStream, struct ncclStrongStream* launchOrderStream);

ncclResult_t ncclScheduleP2pTasks(struct ncclComm* comm,
                                  struct ncclIntruQueue<struct ncclTaskTuningInfo, &ncclTaskTuningInfo::next>* queue,
                                  struct ncclStrongStream* depStream, struct ncclStrongStream* launchOrderStream);

ncclResult_t ncclScheduleRmaTasks(struct ncclComm* comm,
                                  struct ncclIntruQueue<struct ncclTaskTuningInfo, &ncclTaskTuningInfo::next>* queue,
                                  struct ncclStrongStream* depStream, struct ncclStrongStream* launchOrderStream);

ncclResult_t ncclScheduleCeTasks(struct ncclComm* comm,
                                 struct ncclIntruQueue<struct ncclTaskTuningInfo, &ncclTaskTuningInfo::next>* queue,
                                 struct ncclStrongStream* depStream, struct ncclStrongStream* launchOrderStream);

ncclResult_t ncclTaskSchedule(struct ncclComm* comm, struct ncclClassifiedTaskQueues* ctq);

#endif // NCCL_ENQUEUE_TASK_SCHED_H_
