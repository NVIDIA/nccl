/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef _NCCL_RMA_H_
#define _NCCL_RMA_H_

#include "nccl.h"
#include "nccl_common.h"
#include "rma/rma_ce.h"
#include "rma/rma_proxy.h"

#define NCCL_RMA_MAX_CONNECTIONS 4

struct ncclRmaArgs {
  ncclFunc_t func;
  int nRmaTasks;
  int nRmaTasksProxy;
  int nRmaTasksCe;
};

struct ncclRmaState {
  struct ncclRmaProxyState rmaProxyState;
  struct ncclRmaCeState rmaCeState;
};

// Helper functions for signal slot and offset calculations
static inline size_t ncclRmaSignalSlot(int nRanks, int sigIdx, int rank) {
  return (size_t)sigIdx * nRanks + rank;
}

static inline size_t ncclRmaSignalOffset(int nRanks, int sigIdx, int rank) {
  return ncclRmaSignalSlot(nRanks, sigIdx, rank) * sizeof(uint64_t);
}

bool ncclRmaProxyEnabled(struct ncclComm* comm);
bool ncclRmaInitialized(struct ncclComm* comm);

// Main RMA function declarations
ncclResult_t scheduleRmaTasksToPlan(struct ncclComm* comm, struct ncclKernelPlan* plan);
ncclResult_t ncclLaunchRma(struct ncclComm* comm, struct ncclKernelPlan* plan);
ncclResult_t ncclRmaWaitSignal(struct ncclComm* comm, struct ncclKernelPlan* plan, cudaStream_t stream);
ncclResult_t ncclRmaPut(struct ncclComm* comm, struct ncclKernelPlan* plan, cudaStream_t stream);
#endif
