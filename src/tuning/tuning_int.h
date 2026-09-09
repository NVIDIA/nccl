/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2023, Meta Platforms, Inc. and affiliates.
 * SPDX-License-Identifier: Apache-2.0 and BSD-3
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_INT_TUNING_INT_H_
#define NCCL_INT_TUNING_INT_H_

#include "nccl.h"
#include "tuning.h"

struct ncclTuningResultListNode {
  struct ncclTuningResultListNode* next;
  struct ncclTuningResult_t result;
};

struct ncclTuningResultList_t {
  struct ncclTuningResultListNode* head;
};

void ncclTuningResultListFree(struct ncclTuningResultList_t* list);
ncclResult_t ncclTuningResultListPushFront(struct ncclTuningResultList_t* list, struct ncclTuningResult_t result);

ncclResult_t ncclTuningComputeAllTunings(struct ncclTuningInput_t* const input,
                                         struct ncclTuningResultList_t* const results);
ncclResult_t ncclTuningComputeTuning(int id, struct ncclTuningInput_t* const input,
                                     struct ncclTuningResult_t* const result);

ncclResult_t ncclTuningExpandId(int tuningId, int* algo, int* proto, int* symKernelId, int* ceMethodId);

ncclResult_t ncclTuningSetThreadThresholds(struct ncclComm* comm);
ncclResult_t ncclTuningGetChannels(struct ncclTuningInput_t* const input, struct ncclTuningResult_t* const result);

#endif
