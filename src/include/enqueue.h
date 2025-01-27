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

#define NCCL_LL_ALIGNMENT_PER_THREAD sizeof(uint64_t)
#define NCCL_LL128_ALIGNMENT_PER_WARP 480
#define NCCL_SIMPLE_ALIGNMENT (WARP_SIZE * 8LL * 16LL)
#define NCCL_BYTES_ALIGNMENT 16

ncclResult_t ncclInitKernelsForDevice(int cudaArch, int maxSharedMem, size_t* maxStackSize);
ncclResult_t ncclEnqueueCheck(struct ncclInfo* info);
ncclResult_t ncclLaunchPrepare(struct ncclComm* comm);
ncclResult_t ncclLaunchKernelBefore_NoUncapturedCuda(struct ncclComm* comm, struct ncclKernelPlan* plan);
ncclResult_t ncclLaunchKernel(struct ncclComm* comm, struct ncclKernelPlan* plan);
ncclResult_t ncclLaunchKernelAfter_NoCuda(struct ncclComm* comm, struct ncclKernelPlan* plan);
ncclResult_t ncclLaunchFinish(struct ncclComm* comm);
ncclResult_t ncclPrepareTasks(struct ncclComm* comm, bool* algoNeedConnect, bool* needConnect, ncclSimInfo_t* simInfo);
ncclResult_t ncclTasksRegAndEnqueue(struct ncclComm* comm);

static inline size_t ncclFuncSendCount(ncclFunc_t func, int nRanks, size_t count) {
  return func == ncclFuncReduceScatter ? nRanks*count : count;
}
static inline size_t ncclFuncRecvCount(ncclFunc_t func, int nRanks, size_t count) {
  return func == ncclFuncAllGather ? nRanks*count : count;
}
static inline size_t ncclFuncMaxSendRecvCount(ncclFunc_t func, int nRanks, size_t count) {
  return func == ncclFuncAllGather || func == ncclFuncReduceScatter ? nRanks*count : count;
}

#endif // End include guard
