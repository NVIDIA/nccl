/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_ENQUEUE_TASK_CLASSIFY_H_
#define NCCL_ENQUEUE_TASK_CLASSIFY_H_

#include "task_pretuning.h"
#include "nccl_tuner.h"

struct ncclClassifiedTaskQueues {
  struct ncclIntruQueue<struct ncclTaskTuningInfo, &ncclTaskTuningInfo::next> symTaskQueue;
  struct ncclIntruQueue<struct ncclTaskTuningInfo, &ncclTaskTuningInfo::next> legacyTaskQueue;
  struct ncclIntruQueue<struct ncclTaskTuningInfo, &ncclTaskTuningInfo::next> allgathervTaskQueue;
  struct ncclIntruQueue<struct ncclTaskTuningInfo, &ncclTaskTuningInfo::next> p2pTaskQueue;
  struct ncclIntruQueue<struct ncclTaskTuningInfo, &ncclTaskTuningInfo::next> rmaTaskQueue;
  struct ncclIntruQueue<struct ncclTaskTuningInfo, &ncclTaskTuningInfo::next> ceTaskQueue;
};

ncclResult_t ncclTaskClassification(struct ncclComm* comm, struct ncclTaskTuningInfoQueue* tiq,
                                    struct ncclClassifiedTaskQueues* ctq);

#endif // NCCL_ENQUEUE_TASK_CLASSIFY_H_
