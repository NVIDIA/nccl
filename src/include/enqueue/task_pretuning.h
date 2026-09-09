/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_ENQUEUE_TASK_PRETUNING_H_
#define NCCL_ENQUEUE_TASK_PRETUNING_H_

#include "raw_task.h"
#include "tuning.h"

struct ncclTaskTuningInfo {
  struct ncclTaskTuningInfo* next;
  struct ncclRawTask* raw;
  ncclTuningInput_t tuningIn;
  ncclTuningResult_t tuningOut;
};

struct ncclTaskTuningInfoQueue {
  struct ncclIntruQueue<struct ncclTaskTuningInfo, &ncclTaskTuningInfo::next> queue;
};

ncclResult_t ncclTaskPreTuning(struct ncclComm* comm, struct ncclRawTaskQueue* rtq,
                               struct ncclTaskTuningInfoQueue* tiq);

#endif // NCCL_ENQUEUE_TASK_PRETUNING_H_
