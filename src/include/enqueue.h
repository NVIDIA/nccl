/*************************************************************************
 * Copyright (c) 2015-2019, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_ENQUEUE_H_
#define NCCL_ENQUEUE_H_

#include "core.h"
#include "group.h"

// Channels / LL tuning
#define NCCL_LL_CHANNEL_THRESHOLD 8 // Per thread size before we start increasing nrings
#define NCCL_THREAD_THRESHOLD 64  // Per thread size before we switch to non-LL
#define NCCL_THREAD_THRESHOLD_PREVOLTA 32 // Per thread size before we switch to non-LL for pre-Volta archs
#define NCCL_LL_MIN_NTHREADS 64

ncclResult_t ncclEnqueueCheck(struct ncclInfo* info);
ncclResult_t ncclCpuBarrierIn(ncclComm_t comm, int* isLast);
ncclResult_t ncclCpuBarrierLast(ncclComm_t comm);
ncclResult_t ncclCpuBarrierOut(ncclComm_t comm);
ncclResult_t ncclBarrierEnqueue(ncclComm_t comm);
ncclResult_t ncclBarrierEnqueueWait(ncclComm_t comm);
ncclResult_t ncclEnqueueEvents(ncclComm_t comm);

#endif // End include guard
