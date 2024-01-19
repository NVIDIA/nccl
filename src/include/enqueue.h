/*************************************************************************
 * Copyright (c) 2015-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_ENQUEUE_H_
#define NCCL_ENQUEUE_H_

#include "comm.h"
#include "group.h"
#include "collectives.h"
#include "utils.h"

#define NCCL_MIN_CHANNEL_SIZE (NCCL_LL_THREAD_THRESHOLD*64)
#define NCCL_AGG_CHANNEL_SIZE (1LL << 21) /* 2 MiB, ideal per-channel size to fully utilize bandwidth */

ncclResult_t ncclInitKernelsForDevice(int cudaArch, size_t* maxStackSize);
ncclResult_t ncclEnqueueCheck(struct ncclInfo* info);
ncclResult_t ncclLaunchPrepare(struct ncclComm* comm);
ncclResult_t ncclLaunchKernelBefore_NoUncapturedCuda(struct ncclComm* comm, struct ncclKernelPlan* plan);
ncclResult_t ncclLaunchKernel(struct ncclComm* comm, struct ncclKernelPlan* plan);
ncclResult_t ncclLaunchKernelAfter_NoCuda(struct ncclComm* comm, struct ncclKernelPlan* plan);
ncclResult_t ncclLaunchFinish(struct ncclComm* comm);

#endif // End include guard
