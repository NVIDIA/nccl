/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "comm.h"
#include "alloc.h"
#include "checks.h"
#include "enqueue/sched_profiler.h"
#include "enqueue/task_pretuning.h"

ncclResult_t ncclSchedProfilerLaunchCreate(struct ncclComm* comm, bool captured,
                                           const struct ncclTaskTuningInfo* firstTask, int nTasks,
                                           struct ncclSchedProfilerLaunch** launchOut) {
  struct ncclSchedProfilerLaunch* launch = nullptr;
  ncclResult_t ret = ncclSuccess;

  NCCLCHECKGOTO(ncclCalloc(&launch, 1), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&launch->tasks, nTasks), ret, fail);
  launch->comm = comm;
  launch->captured = captured;
  launch->eActivationMask = firstTask->raw->profiler.eActivationMask;
  launch->groupApiEventHandle = firstTask->raw->profiler.groupApiEventHandle;
  *launchOut = launch;
  return ncclSuccess;
fail:
  ncclSchedProfilerLaunchDestroy(launch);
  return ret;
}

void ncclSchedProfilerLaunchDestroy(struct ncclSchedProfilerLaunch* launch) {
  if (launch == nullptr) return;
  free(launch->tasks);
  free(launch);
}

void ncclSchedProfilerLaunchAddTask(struct ncclSchedProfilerLaunch* launch, const struct ncclTaskTuningInfo* tInfo) {
  if (launch == nullptr) return;
  const struct ncclRawTaskColl* raw = &tInfo->raw->coll;
  struct ncclSchedProfilerTask* t = &launch->tasks[launch->nTasks++];

  t->func = raw->func;
  t->sendbuff = raw->sendbuff;
  t->recvbuff = raw->recvbuff;
  t->count = raw->count;
  t->root = raw->root;
  t->datatype = raw->datatype;
  // AllGather and Broadcast are scheduled in bytes; report them that way, as the legacy
  // path does.
  if (raw->func == ncclFuncAllGather || raw->func == ncclFuncBroadcast) {
    t->count *= ncclTypeSize(raw->datatype);
    t->datatype = ncclInt8;
  }
  t->symKernelId = tInfo->tuningOut.symKernelId;
  t->algo = tInfo->tuningOut.algo;
  t->proto = tInfo->tuningOut.proto;
  t->nChannels = (uint8_t)tInfo->tuningOut.nChannels;
  t->nWarps = (uint8_t)tInfo->tuningOut.nWarps;
  t->api = tInfo->raw->profiler;
  t->eventHandle = {};
}

void CUDART_CB ncclSchedProfilerReplayCallback(void* launch_) {
  struct ncclSchedProfilerLaunch* launch = (struct ncclSchedProfilerLaunch*)launch_;
  ncclResult_t result = ncclProfilerSchedReplay(launch);
  if (result != ncclSuccess) {
    WARN("ncclSchedProfilerReplayCallback() failed : %s", ncclGetErrorString(result));
  }
}
