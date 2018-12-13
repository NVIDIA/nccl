/*************************************************************************
 * Copyright (c) 2015-2018, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_ENQUEUE_H_
#define NCCL_ENQUEUE_H_

#include "core.h"
#include "group.h"

ncclResult_t ncclEnqueueCheck(struct ncclInfo* info);
ncclResult_t ncclCpuBarrierIn(ncclComm_t comm, int* isLast);
ncclResult_t ncclCpuBarrierLast(ncclComm_t comm);
ncclResult_t ncclCpuBarrierOut(ncclComm_t comm);
ncclResult_t ncclBarrierEnqueue(ncclComm_t comm);
ncclResult_t ncclBarrierEnqueueWait(ncclComm_t comm);
ncclResult_t ncclEnqueueEvents(ncclComm_t comm);

#endif // End include guard
