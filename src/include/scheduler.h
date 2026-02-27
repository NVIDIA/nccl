/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2015-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_SCHEDULER_H_
#define NCCL_SCHEDULER_H_

#include "nccl.h"
#include "comm.h"
#include "sym_kernels.h"
#include "enqueue.h"

ncclResult_t ncclMakeSymmetricTaskList(struct ncclComm* comm, struct ncclTaskColl* task, struct ncclIntruQueue<struct ncclTaskColl, &ncclTaskColl::next>* symTaskQueue, struct ncclTaskColl** remainTasksHead);
ncclResult_t ncclSymmetricTaskScheduler(struct ncclComm* comm, struct ncclIntruQueue<struct ncclTaskColl, &ncclTaskColl::next>* symTaskQueue, struct ncclKernelPlan* plan);

ncclResult_t ncclScheduleBcastTasksToPlan(struct ncclComm* comm, struct ncclKernelPlan* plan, struct ncclKernelPlanBudget* budget);

#endif // NCCL_SCHEDULER_H_
