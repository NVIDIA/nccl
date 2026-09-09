/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "group.h"
#include "enqueue/task_sched.h"

ncclResult_t ncclScheduleRmaTasks(struct ncclComm* comm,
                                  struct ncclIntruQueue<struct ncclTaskTuningInfo, &ncclTaskTuningInfo::next>* queue,
                                  struct ncclStrongStream* depStream, struct ncclStrongStream* launchOrderStream) {
  (void)queue;
  (void)depStream;
  (void)launchOrderStream;
  return doLaunches(comm);
}
