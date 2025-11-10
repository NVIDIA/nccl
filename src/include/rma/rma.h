/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef _NCCL_RMA_H_
#define _NCCL_RMA_H_

#include "nccl.h"
#include "nccl_common.h"
#include "rma/rma_ce.h"
#include "rma/rma_proxy.h"

struct ncclRmaArgs{
  int ctx;
  ncclFunc_t func;
  int nRmaTasks;
  int nRmaTasksProxy;
  int nRmaTasksCe;
};

struct ncclRmaState {
  struct ncclRmaProxyState rmaProxyState;
  struct ncclRmaCeState rmaCeState;
};

// Main RMA function declarations
ncclResult_t scheduleRmaTasksToPlan(struct ncclComm* comm, struct ncclKernelPlan* plan);
ncclResult_t ncclLaunchRma(struct ncclComm* comm, struct ncclKernelPlan* plan);
ncclResult_t ncclRmaWaitSignal(struct ncclComm* comm, struct ncclKernelPlan* plan, cudaStream_t stream);
ncclResult_t ncclRmaPut(struct ncclComm* comm, struct ncclKernelPlan* plan, cudaStream_t stream);

#endif
