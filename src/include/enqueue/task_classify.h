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

// Index of ncclClassifiedTaskQueues::queues.
enum ncclTaskQueueType {
  ncclTaskQueueSym = 0,
  ncclTaskQueueRma,
  ncclTaskQueueCe,
  ncclTaskQueueLegacy,
  ncclTaskQueueP2p,
  ncclTaskQueueAllGatherV,
  ncclTaskQueueNum
};

struct ncclClassifiedTaskQueues {
  struct ncclIntruQueue<struct ncclTaskTuningInfo, &ncclTaskTuningInfo::next> queues[ncclTaskQueueNum];
};

inline void ncclClassifiedTaskQueuesConstruct(struct ncclClassifiedTaskQueues* ctq) {
  for (int type = 0; type < ncclTaskQueueNum; type++) {
    ncclIntruQueueConstruct(&ctq->queues[type]);
  }
}

ncclResult_t ncclTaskClassification(struct ncclComm* comm, struct ncclTaskTuningInfoQueue* tiq,
                                    struct ncclClassifiedTaskQueues* ctq);

#endif // NCCL_ENQUEUE_TASK_CLASSIFY_H_
