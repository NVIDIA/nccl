/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_ENQUEUE_SCHED_PROFILER_H_
#define NCCL_ENQUEUE_SCHED_PROFILER_H_

#include "comm.h"

struct ncclComm;
struct ncclTaskTuningInfo;
struct ncclSymkDevWorkArgs;

// The ncclProfileColl descriptor fields for one collective, plus the API-call linkage it is
// parented to and the handle of the event once started. Filled from an ncclTaskTuningInfo
// (API arguments from the raw task, algorithm and channel/warp counts from the tuning result).
struct ncclSchedProfilerTask {
  ncclFunc_t func;
  void const* sendbuff;
  void* recvbuff;
  size_t count;
  int root;
  ncclDataType_t datatype;
  int symKernelId;
  int algo;
  int proto;
  uint8_t nChannels;
  uint8_t nWarps;
  struct ncclProfilerApiCtx api; // mask, parent events and user tag of the API call
  struct ncclProfilerEventRef eventHandle;             // ncclProfileColl event for this task
};

// Profiler state for one device launch by a rearch scheduler: the tasks it covers and the
// launch-level events (ncclProfileKernelLaunch, ncclProfileGroup). Created only when profiling
// is requested; heap-allocated so a captured launch can replay its events after the group ends.
struct ncclSchedProfilerLaunch {
  struct ncclComm* comm;
  bool captured;
  int eActivationMask;        // of the tasks (all tasks in a group share one mask)
  struct ncclProfilerEventRef groupApiEventHandle;  // parent ncclProfileGroupApi event
  int nTasks;
  struct ncclSchedProfilerTask* tasks;
  // Sym kernels launched with profiler counters (ncclSymkDevWorkArgs::profilerMode set);
  // nullptr for every other launch.
  struct ncclSymkDevWorkArgs* symArgs;
  int symChannels;
  struct ncclProfilerEventRef kernelLaunchEventHandle; // ncclProfileKernelLaunch
  struct ncclProfilerEventRef groupEventHandle;        // ncclProfileGroup, parent of the task events
};

// Creates a launch record with room for nTasks tasks; the event mask and group API parent
// come from firstTask (every task in a group shares them).
ncclResult_t ncclSchedProfilerLaunchCreate(struct ncclComm* comm, bool captured,
                                           const struct ncclTaskTuningInfo* firstTask, int nTasks,
                                           struct ncclSchedProfilerLaunch** launchOut);
void ncclSchedProfilerLaunchDestroy(struct ncclSchedProfilerLaunch* launch);

// Appends tInfo's collective to the launch, copying what the Coll event needs out of the raw
// task and tuning result so the scheduler can free them afterwards. No-op on a null launch.
void ncclSchedProfilerLaunchAddTask(struct ncclSchedProfilerLaunch* launch, const struct ncclTaskTuningInfo* tInfo);

#endif // NCCL_ENQUEUE_SCHED_PROFILER_H_
