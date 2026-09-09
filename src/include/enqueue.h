/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2015-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_ENQUEUE_H_
#define NCCL_ENQUEUE_H_

#include "comm.h"
#include "group.h"
#include "collectives.h"
#include "utils.h"
#include "enqueue/raw_task.h"
#include "enqueue/task_pretuning.h"
#include "enqueue/task_classify.h"
#include "enqueue/task_posttuning.h"
#include "enqueue/task_sched.h"
#include "enqueue/mgmt_task_enq.h"

#define NCCL_LL_ALIGNMENT_PER_THREAD sizeof(uint64_t)
#define NCCL_LL128_ALIGNMENT_PER_WARP 480
#define NCCL_SIMPLE_ALIGNMENT (WARP_SIZE * 8LL * 16LL)
#define NCCL_BYTES_ALIGNMENT 16

int64_t ncclParamGraphStreamOrdering();
int64_t ncclParamEnqueueRearchEnable();
int64_t ncclParamAllgathervEnable();
int64_t ncclParamP2pLLThreshold();
int64_t ncclParamChunkSize();
int64_t ncclParamLaunchOrderImplicit();

ncclResult_t ncclGroupJobLaunch(struct ncclIntruQueue<struct ncclAsyncJob, &ncclAsyncJob::next>* asyncJobsMain,
                                volatile bool* groupAbortFlag);

ncclResult_t ncclTaskPreTuning(struct ncclComm* comm, struct ncclRawTaskQueue* rtq,
                               struct ncclTaskTuningInfoQueue* tiq);
ncclResult_t ncclTaskPrepare(struct ncclComm* comm, ncclSimInfo_t* simInfo);

ncclResult_t ncclInitKernelsForDevice(int cudaArch, int maxSharedMem, size_t* maxStackSize);
ncclResult_t ncclEnqueueCheck(struct ncclInfo* info);
ncclResult_t ncclPlannerSetCapturingGraph(struct ncclComm* comm, struct ncclInfo* info);
ncclResult_t ncclLaunchPrepare(struct ncclComm* comm);
ncclResult_t ncclLaunchKernelBefore_NoUncapturedCuda(struct ncclComm* comm, struct ncclKernelPlan* plan);
ncclResult_t ncclLaunchKernel(struct ncclComm* comm, struct ncclKernelPlan* plan);
ncclResult_t ncclLaunchKernelAfter_NoCuda(struct ncclComm* comm, struct ncclKernelPlan* plan);
ncclResult_t ncclLaunchFinish(struct ncclComm* comm);
ncclResult_t ncclPrepareTasks(struct ncclComm* comm, bool* algoNeedConnect, bool* needConnect, ncclSimInfo_t* simInfo);
ncclResult_t ncclTasksRegAndEnqueue(struct ncclComm* comm);

static inline size_t ncclFuncSendCount(ncclFunc_t func, int nRanks, size_t count) {
  return func == ncclFuncReduceScatter ? nRanks * count : count;
}
static inline size_t ncclFuncRecvCount(ncclFunc_t func, int nRanks, size_t count) {
  return func == ncclFuncAllGather ? nRanks * count : count;
}
static inline size_t ncclFuncMaxSendRecvCount(ncclFunc_t func, int nRanks, size_t count) {
  return func == ncclFuncAllGather || func == ncclFuncReduceScatter ? nRanks * count : count;
}

ncclResult_t ncclGetCollNetSupport(struct ncclComm* comm, struct ncclTaskColl* task, int* collNetSupport);
ncclResult_t ncclGetAlgoInfo(struct ncclComm* comm, struct ncclTaskColl* task, int collNetSupport, int nvlsSupport,
                             int numPipeOps, ncclSimInfo_t* simInfo = NULL);
bool ncclTestBudget(struct ncclKernelPlanBudget* budget, int nWorkBatches, ssize_t nWorkBytes);

void ncclAddWorkBatchToPlan(struct ncclComm* comm, struct ncclKernelPlan* plan, int channelId,
                            enum ncclDevWorkType workType, int devFuncId, uint32_t workOffset, int p2pEpoch = -1,
                            int p2pRound = -1, bool newBatch = false);

ncclResult_t ncclAddProxyOpIfNeeded(struct ncclComm* comm, struct ncclKernelPlan* plan, struct ncclProxyOp* op);

ncclResult_t ncclGetRegBuff(struct ncclComm* comm, struct ncclTaskColl* info, int* regBuff);

#endif // End include guard
