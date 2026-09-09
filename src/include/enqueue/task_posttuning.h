/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_ENQUEUE_TASK_POSTTUNING_H_
#define NCCL_ENQUEUE_TASK_POSTTUNING_H_

#include "task_classify.h"
#include "nccl.h"

ncclResult_t ncclTaskPostTuning(struct ncclComm* comm, struct ncclClassifiedTaskQueues* ctq, ncclSimInfo_t* simInfo);

#endif // NCCL_ENQUEUE_TASK_POSTTUNING_H_
