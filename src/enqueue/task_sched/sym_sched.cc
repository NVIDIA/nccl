/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "comm.h"
#include "alloc.h"
#include "checks.h"
#include "collectives.h"
#include "enqueue.h"
#include "strongstream.h"
#include "cudawrap.h"
#include "dev_runtime.h"
#include "param.h"
#include "sym_kernels.h"
#include "enqueue/task_sched.h"
#include "enqueue/task_pretuning.h"
#include "enqueue/sched_profiler.h"
#include "config/collconfig.h"
#include <algorithm>
#include <map>
#include <cuda_fp16.h>
#if defined(__CUDA_FP8_TYPES_EXIST__)
#include <cuda_fp8.h>
#endif

NCCL_PARAM(SymSchedMemSyncDomain, "MEM_SYNC_DOMAIN", cudaLaunchMemSyncDomainRemote);

enum ncclImplicitOrder {
  ncclImplicitOrderNone = 0,
  ncclImplicitOrderSerial,
  ncclImplicitOrderLaunch
};

struct ncclSymSchedKernelPlan {
  struct ncclSymSchedKernelPlan* next;
  int kernelDynSmem;
  void* kernelFn;
  struct ncclSymkDevWorkArgs* kernelSymArgs;
  size_t kernelArgsSize;
  int nChannels;
  int threadPerBlock;
  struct ncclSchedProfilerLaunch* profiler; // nullptr unless the launch is profiled
};

// kernelIndex selects the symmetric kernel. llGroupSeq is 0 when tasks share a launch.
// LL tasks cannot be grouped, so each one gets a unique llGroupSeq.
struct ncclSymSchedKernelTaskGroupKey {
  int kernelIndex;
  int llGroupSeq;
};

inline bool operator<(struct ncclSymSchedKernelTaskGroupKey a, struct ncclSymSchedKernelTaskGroupKey b) {
  if (a.kernelIndex != b.kernelIndex) return a.kernelIndex < b.kernelIndex;
  return a.llGroupSeq < b.llGroupSeq;
}

using ncclSymSchedTaskQueue = ncclIntruQueue<struct ncclTaskTuningInfo, &ncclTaskTuningInfo::next>;
using ncclSymSchedKernelTaskGroup = std::map<struct ncclSymSchedKernelTaskGroupKey, ncclSymSchedTaskQueue>;

struct ncclSymSchedStreamPlan {
  struct ncclSymSchedStreamPlan* next;
  cudaStream_t stream;
  bool captured;
  struct ncclCudaGraph graph;
  cudaEvent_t launchCompletionEvent;
  ncclSymSchedKernelTaskGroup kernelTaskGroup;
  struct ncclSymSchedKernelPlan* kernelPlans;
  int llGroupSeq;
};

struct ncclSymSchedPlan {
  struct ncclIntruQueue<struct ncclSymSchedStreamPlan, &ncclSymSchedStreamPlan::next> streamPlans;
};

struct symSchedKernelPlanFreeArg {
  struct ncclSymkDevWorkArgs* kernelSymArgs;
  struct ncclSchedProfilerLaunch* profiler;
};

static ncclDevRedOp_t symSchedRedOp(ncclRedOp_t redOp, ncclDevRedOp_t devRedOp) {
  if (redOp == ncclAvg) return ncclDevSumPostDiv;
  return devRedOp;
}

static struct ncclDevRedOpFull symSchedSymkOpDev(struct ncclComm* comm, struct ncclTaskTuningInfo* tInfo) {
  struct ncclRawTaskColl* raw = &tInfo->raw->coll;
  struct ncclDevRedOpFull opDev = raw->opDev;

  opDev.op = symSchedRedOp(raw->opHost, opDev.op);
  if (opDev.op != ncclDevSumPostDiv) return opDev;
  if (tInfo->tuningOut.symKernelId == (int)ncclSymkKernelId_ReduceScatter_LDMC) return opDev;

  union {
    __half f16;
    float f32;
    uint64_t u64;
  };
  u64 = 0;
  switch (raw->datatype) {
  case ncclFloat16:
#if defined(__CUDA_BF16_TYPES_EXIST__)
  case ncclBfloat16:
#endif
    f32 = float(1.0 / comm->nRanks);
    opDev.scalarArg = u64;
    return opDev;
#if defined(__CUDA_FP8_TYPES_EXIST__)
  case ncclFloat8e4m3:
  case ncclFloat8e5m2:
    f16 = __float2half(float(1.0 / comm->nRanks));
    opDev.scalarArg = u64;
    return opDev;
#endif
  default:
    break;
  }
  return opDev;
}

static ncclResult_t symSchedWinRegTypeFromTuningInfo(struct ncclComm* comm, struct ncclTaskTuningInfo* tInfo,
                                                     ncclSymRegType_t* winRegType) {
  struct ncclRawTaskColl* raw = &tInfo->raw->coll;
  struct ncclDevrWindow* sendWin = nullptr;
  struct ncclDevrWindow* recvWin = nullptr;

  if (tInfo->raw->kind != ncclTaskKindColl) return ncclInternalError;

  NCCLCHECK(ncclDevrFindWindow(comm, raw->sendbuff, &sendWin));
  NCCLCHECK(ncclDevrFindWindow(comm, raw->recvbuff, &recvWin));
  NCCLCHECK(ncclGetSymRegType(sendWin, recvWin, winRegType));
  return ncclSuccess;
}

static ncclResult_t symSchedMakeDevWorkFromTuningInfo(struct ncclComm* comm, struct ncclTaskTuningInfo* tInfo,
                                                      struct ncclSymkDevWork* outDevWork) {
  struct ncclRawTaskColl* raw = &tInfo->raw->coll;
  struct ncclDevrWindow* sendWin = nullptr;
  struct ncclDevrWindow* recvWin = nullptr;
  struct ncclDevRedOpFull opDev;

  if (tInfo->raw->kind != ncclTaskKindColl) return ncclInternalError;

  NCCLCHECK(ncclDevrFindWindow(comm, raw->sendbuff, &sendWin));
  NCCLCHECK(ncclDevrFindWindow(comm, raw->recvbuff, &recvWin));
  opDev = symSchedSymkOpDev(comm, tInfo);

  outDevWork->rootRank = raw->root;
  outDevWork->redOpArg = opDev.scalarArg;
  outDevWork->nElts = raw->count;
  outDevWork->inputWin = sendWin ? sendWin->vidmem : nullptr;
  outDevWork->inputOff = sendWin ? (uint8_t*)raw->sendbuff - (uint8_t*)sendWin->userPtr : (size_t)raw->sendbuff;
  outDevWork->outputWin = recvWin ? recvWin->vidmem : nullptr;
  outDevWork->outputOff = recvWin ? (uint8_t*)raw->recvbuff - (uint8_t*)recvWin->userPtr : (size_t)raw->recvbuff;
  outDevWork->sChannelId = 0xffff;
  outDevWork->nChannels = 0;
  return ncclSuccess;
}

static void symSchedFreeTuningInfoRaw(struct ncclComm* comm, struct ncclTaskTuningInfo* taskInfo) {
  if (taskInfo->raw != nullptr) {
    ncclMemoryPoolFree(&comm->memPool_ncclRawTask, taskInfo->raw);
    taskInfo->raw = nullptr;
  }
}

static cudaStream_t symSchedTuningInfoStream(struct ncclTaskTuningInfo* tInfo) {
  return tInfo->raw->coll.stream;
}

static ncclResult_t symSchedStreamCaptured(struct ncclComm* comm, cudaStream_t stream, bool* capturedOut,
                                           struct ncclCudaGraph* graphOut) {
  NCCLCHECK(ncclCudaGetCapturingGraph(graphOut, stream, comm->config.graphUsageMode));
  *capturedOut = ncclCudaGraphValid(*graphOut);
  return ncclSuccess;
}

static struct ncclSymSchedStreamPlan* symSchedFindStreamPlan(struct ncclSymSchedPlan* symPlan, cudaStream_t stream,
                                                             bool captured) {
  for (struct ncclSymSchedStreamPlan* streamPlan = ncclIntruQueueHead(&symPlan->streamPlans); streamPlan != nullptr;
       streamPlan = streamPlan->next) {
    if (streamPlan->stream == stream && streamPlan->captured == captured) return streamPlan;
  }
  return nullptr;
}

// Captured stream plans stay ahead of non-captured ones.
static void symSchedEnqueueStreamPlan(struct ncclSymSchedPlan* symPlan, struct ncclSymSchedStreamPlan* streamPlan) {
  if (streamPlan->captured) {
    ncclIntruQueueEnqueueFront(&symPlan->streamPlans, streamPlan);
  } else {
    ncclIntruQueueEnqueue(&symPlan->streamPlans, streamPlan);
  }
}

static ncclResult_t symSchedGetStreamPlan(struct ncclSymSchedPlan* symPlan, cudaStream_t stream, bool captured,
                                          const struct ncclCudaGraph& graph,
                                          struct ncclSymSchedStreamPlan** streamPlanOut) {
  struct ncclSymSchedStreamPlan* streamPlan = symSchedFindStreamPlan(symPlan, stream, captured);
  if (streamPlan != nullptr) {
    *streamPlanOut = streamPlan;
    return ncclSuccess;
  }

  // Scoped memory does not construct the kernel-task map, so the plan is heap allocated.
  NEW_NOTHROW(streamPlan, ncclSymSchedStreamPlan);
  streamPlan->stream = stream;
  streamPlan->captured = captured;
  streamPlan->graph = graph;
  streamPlan->launchCompletionEvent = nullptr;
  streamPlan->kernelPlans = nullptr;
  streamPlan->llGroupSeq = 0;
  streamPlan->next = nullptr;
  symSchedEnqueueStreamPlan(symPlan, streamPlan);
  *streamPlanOut = streamPlan;
  return ncclSuccess;
}

static ncclResult_t symSchedEnqueueKernelTask(struct ncclComm* comm, struct ncclSymSchedStreamPlan* streamPlan,
                                              struct ncclTaskTuningInfo* taskInfo) {
  int symKernelId = taskInfo->tuningOut.symKernelId;
  cudaEvent_t launchCompletionEvent = ncclCollConfigGetLaunchCompletionEvent(&taskInfo->raw->collConfig);
  if (launchCompletionEvent != nullptr) streamPlan->launchCompletionEvent = launchCompletionEvent;

  struct ncclRawTaskColl* raw = &taskInfo->raw->coll;
  struct ncclDevRedOpFull opDev = symSchedSymkOpDev(comm, taskInfo);
  struct ncclSymSchedKernelTaskGroupKey key = {};
  key.kernelIndex = ncclSymkGetKernelIndex((ncclSymkKernelId)symKernelId, opDev.op, raw->datatype);
  // LL kernels do not support grouping, so each LL task gets its own group.
  bool isLLKernel = ((1 << symKernelId) & ncclSymkLLKernelMask()) != 0;
  key.llGroupSeq = isLLKernel ? ++streamPlan->llGroupSeq : 0;

  auto inserted = streamPlan->kernelTaskGroup.emplace(key, ncclSymSchedTaskQueue{});
  ncclIntruQueueEnqueue(&inserted.first->second, taskInfo);
  return ncclSuccess;
}

static int symSchedGroupMaxChannels(struct ncclComm* comm, ncclSymSchedTaskQueue* tasks) {
  int maxChannels = 0;
  for (struct ncclTaskTuningInfo* task = ncclIntruQueueHead(tasks); task != nullptr; task = task->next) {
    maxChannels += task->tuningOut.nChannels;
    if (maxChannels > comm->nChannels) return comm->nChannels;
  }
  return maxChannels;
}

static int symSchedCountKernelBatch(struct ncclComm* comm, ncclSymSchedTaskQueue* tasks, int maxChannels) {
  int nElems = tasks->nElems;
  if (nElems == 0) return 0;

  size_t workArgsBytes = comm->workArgsBytes;

  auto fits = [&](int nWorks) {
    return ncclSymkDevWorkArgsSize<ncclSymkDevWorkArgs>(maxChannels, nWorks, ncclProfilerPluginLoaded()) <= workArgsBytes;
  };

  if (fits(nElems)) return nElems;

  int lo = 1;
  int hi = nElems - 1;
  int best = 1;
  while (lo <= hi) {
    int mid = lo + (hi - lo) / 2;
    if (fits(mid)) {
      best = mid;
      lo = mid + 1;
    } else {
      hi = mid - 1;
    }
  }
  return best;
}

static ncclResult_t symSchedAppendKernelPlan(struct ncclSymSchedStreamPlan* streamPlan,
                                             struct ncclSymSchedKernelPlan* kernelPlan) {
  struct ncclSymSchedKernelPlan** tail = &streamPlan->kernelPlans;
  while (*tail != nullptr) tail = &(*tail)->next;
  *tail = kernelPlan;
  kernelPlan->next = nullptr;
  return ncclSuccess;
}

static ncclResult_t symSchedBuildKernelPlan(struct ncclComm* comm, struct ncclSymSchedStreamPlan* streamPlan,
                                            ncclSymSchedTaskQueue* tasks, int nWorks, int maxChannels) {
  ncclResult_t ret = ncclSuccess;
  struct ncclTaskTuningInfo* headTask = ncclIntruQueueHead(tasks);
  struct ncclRawTaskColl* headRaw = &headTask->raw->coll;
  struct ncclDevRedOpFull headOpDev = symSchedSymkOpDev(comm, headTask);
  ncclDataType_t datatype = headRaw->datatype;
  ssize_t totalCount = 0;
  ssize_t logCount = 0;
  ssize_t remainCell = 0;
  ssize_t cellPerChannel = 0;
  int workIndex = 0;
  int maxnWarps = 0;
  int totalChannels = 0;
  size_t cellCount = NCCL_SYM_KERNEL_CELL_SIZE / ncclTypeSize(datatype);
  int curChannel = 0;
  int curChannelWork = 0;
  int nMaxChannels = maxChannels;
  struct ncclSymkDevWork* workBufPtr = nullptr;
  struct ncclSymkChannelWorkRange* workRangePtr = nullptr;
  struct ncclSymkDevWorkArgs* argsBuf = nullptr;
  struct ncclSymSchedKernelPlan* kernelPlan = nullptr;
  struct ncclTaskTuningInfo* taskInfo = nullptr;
  ncclSymkKernelId kernelId = (ncclSymkKernelId)headTask->tuningOut.symKernelId;
  struct ncclSchedProfilerLaunch* profiler = nullptr;
  // Profiling requested = plugin loaded and the call's mask has ncclProfileKernelCh.
  // Device-side instrumentation is eager-only: the args buffer is snapshotted at launch,
  // so the mirrored counters cannot advance across graph replays. Under capture the clean
  // kernel runs and only the host-side group/coll events are replayed.
  int eActivationMask = headTask->raw->profiler.eActivationMask;
  bool profilingRequested = ncclProfilerPluginLoaded() && (eActivationMask & ncclProfileKernelCh);
  bool profilerEnabled = profilingRequested && !streamPlan->captured;

  {
    struct ncclTaskTuningInfo* task = headTask;
    for (int workCount = 0; workCount < nWorks; workCount++, task = task->next) {
      totalCount += alignUp(task->raw->coll.count, cellCount);
      logCount += task->raw->coll.count;
      totalChannels += task->tuningOut.nChannels;
      if (totalChannels > nMaxChannels) totalChannels = nMaxChannels;
      if (task->tuningOut.nWarps > maxnWarps) maxnWarps = task->tuningOut.nWarps;
    }
  }

  kernelPlan = ncclMemoryStackAlloc<struct ncclSymSchedKernelPlan>(&comm->memScoped);

  {
    int kernelIndex = ncclSymkGetKernelIndex(kernelId, headOpDev.op, datatype);
    int maxDynamicSmem = ncclSymkKernelMaxDynamicSmem[kernelIndex];
    kernelPlan->threadPerBlock = maxnWarps * WARP_SIZE;
    kernelPlan->kernelFn = (profilerEnabled && ncclSymkKernelListProfile[kernelIndex] != nullptr) ?
                             ncclSymkKernelListProfile[kernelIndex] :
                             ncclSymkKernelList[kernelIndex];
    kernelPlan->kernelDynSmem = (1 & ncclSymkDynamicSmemKernelMask() >> (int)kernelId) ? maxDynamicSmem : 0;
    kernelPlan->kernelArgsSize = ncclSymkDevWorkArgsSize<ncclSymkDevWorkArgs>(totalChannels, nWorks, profilerEnabled);
    NCCLCHECKGOTO(ncclCalloc((char**)&argsBuf, kernelPlan->kernelArgsSize), ret, fail);
    kernelPlan->kernelSymArgs = argsBuf;
    kernelPlan->profiler = nullptr;

    argsBuf->nMaxChannels = totalChannels;
    argsBuf->maxDynamicSmem = maxDynamicSmem;
    // KernelCh-only launches leave the phase bit clear, so the profile variant skips the
    // phase stamps and their fence at runtime.
    argsBuf->profilerMode =
      profilerEnabled ? ncclProfilerDeviceMode(eActivationMask) : (uint8_t)ncclDevProfilerModeNone;

    if (profilingRequested) {
      NCCLCHECKGOTO(ncclSchedProfilerLaunchCreate(comm, streamPlan->captured, headTask, nWorks, &profiler), ret, fail);
      profiler->symArgs = argsBuf;
      kernelPlan->profiler = profiler;
    }
    workRangePtr = ncclSymkGetWorkRange(argsBuf);
    workBufPtr = ncclSymkGetWorks(argsBuf, totalChannels);
  }

  remainCell = cellPerChannel = DIVUP(DIVUP(totalCount, totalChannels), cellCount);

  while (workIndex < nWorks) {
    struct ncclSymkDevWork devWork = {};
    size_t count = 0;
    size_t cellLeft = 0;
    size_t taskCell = 0;

    taskInfo = ncclIntruQueueDequeue(tasks);
    count = taskInfo->raw->coll.count;
    NCCLCHECKGOTO(symSchedMakeDevWorkFromTuningInfo(comm, taskInfo, &devWork), ret, fail);
    ncclSchedProfilerLaunchAddTask(profiler, taskInfo);
    symSchedFreeTuningInfoRaw(comm, taskInfo);
    taskInfo = nullptr;

    cellLeft = taskCell = DIVUP(count, cellCount);
    for (; curChannel < totalChannels;) {
      workRangePtr[curChannel].workHi = workIndex;
      if (curChannelWork == 0) {
        if (devWork.nChannels == 0) {
          devWork.sChannelId = curChannel;
          devWork.nChannels = 1;
        } else if (cellLeft <= remainCell) {
          assert(devWork.nChannels > 0);
          if ((remainCell - cellLeft) * NCCL_SYM_KERNEL_CELL_SIZE <= (1 << 10) || workIndex + 1 == nWorks) {
            devWork.nChannels++;
          }
        } else {
          devWork.nChannels++;
        }
      } else {
        assert(cellLeft == taskCell);
        if (taskCell <= remainCell) {
          devWork.sChannelId = curChannel;
          devWork.nChannels = 1;
        }
      }
      if (cellLeft < remainCell) {
        workRangePtr[curChannel].fracHi = uint16_t(0x10000UL - 1);
        remainCell -= cellLeft;
        curChannelWork++;
        break;
      } else if (cellLeft == remainCell) {
        workRangePtr[curChannel].fracHi = uint16_t(0x10000UL - 1);
        remainCell = cellPerChannel;
        curChannel++;
        curChannelWork = 0;
        break;
      } else {
        cellLeft -= remainCell;
        workRangePtr[curChannel].fracHi = uint16_t(DIVUP(0x10000L * (taskCell - cellLeft), taskCell) - 1);
        remainCell = cellPerChannel;
        curChannel++;
        curChannelWork = 0;
      }
    }
    memcpy(workBufPtr + workIndex, &devWork, sizeof(struct ncclSymkDevWork));
    workIndex++;
  }
  if (remainCell < cellPerChannel) curChannel++;

  memcpy(&argsBuf->kcomm, &comm->symkState.kcomm, sizeof(comm->symkState.kcomm));
  kernelPlan->nChannels = curChannel;
  if (profiler != nullptr) profiler->symChannels = curChannel;

  if (comm->rank == 0) {
    INFO(NCCL_TUNING, "%s [Symmetric]: %ld Bytes -> Kernel %s nchannels %d nthreads %d nWorks %d",
         ncclFuncToString(headRaw->func), logCount * ncclTypeSize(datatype), ncclSymkKernelIdToString(kernelId),
         curChannel, kernelPlan->threadPerBlock, nWorks);
  }

  NCCLCHECKGOTO(symSchedAppendKernelPlan(streamPlan, kernelPlan), ret, fail);

exit:
  return ret;
fail:
  if (taskInfo != nullptr) symSchedFreeTuningInfoRaw(comm, taskInfo);
  if (argsBuf != nullptr) free(argsBuf);
  ncclSchedProfilerLaunchDestroy(profiler);
  goto exit;
}

static ncclResult_t symSchedBuildKernelTaskGroupPlans(struct ncclComm* comm, struct ncclSymSchedStreamPlan* streamPlan,
                                                      ncclSymSchedTaskQueue* tasks) {
  int maxChannels = symSchedGroupMaxChannels(comm, tasks);
  while (!ncclIntruQueueEmpty(tasks)) {
    int nWorks = symSchedCountKernelBatch(comm, tasks, maxChannels);
    if (nWorks == 0) return ncclInternalError;
    NCCLCHECK(symSchedBuildKernelPlan(comm, streamPlan, tasks, nWorks, maxChannels));
  }
  return ncclSuccess;
}

static ncclResult_t symSchedBuildStreamKernelPlans(struct ncclComm* comm, struct ncclSymSchedStreamPlan* streamPlan) {
  for (auto& group : streamPlan->kernelTaskGroup) {
    NCCLCHECK(symSchedBuildKernelTaskGroupPlans(comm, streamPlan, &group.second));
  }
  return ncclSuccess;
}

static ncclResult_t symSchedBuildKernelPlans(struct ncclComm* comm, struct ncclSymSchedPlan* symPlan) {
  for (struct ncclSymSchedStreamPlan* streamPlan = ncclIntruQueueHead(&symPlan->streamPlans); streamPlan != nullptr;
       streamPlan = streamPlan->next) {
    NCCLCHECK(symSchedBuildStreamKernelPlans(comm, streamPlan));
  }
  return ncclSuccess;
}

static ncclResult_t symSchedGetImplicitOrder(enum ncclImplicitOrder* mode, bool capturing) {
  if (ncclParamLaunchOrderImplicit()) {
    int driver = 0;
    NCCLCHECK(ncclCudaDriverVersion(&driver));
    if (capturing && driver < 12090) {
      *mode = ncclImplicitOrderSerial;
      return ncclSuccess;
    }
    *mode = 12030 <= std::min<int>(CUDART_VERSION, driver) ? ncclImplicitOrderLaunch : ncclImplicitOrderSerial;
    return ncclSuccess;
  }
  *mode = ncclImplicitOrderNone;
  return ncclSuccess;
}

static ncclResult_t launchSymKernelPlan(struct ncclComm* comm, struct ncclSymSchedKernelPlan* kernelPlan,
                                        cudaStream_t launchStream, cudaEvent_t userLaunchEvent,
                                        cudaEvent_t launchEventOut) {
  ncclResult_t ret = ncclSuccess;
  void* sym = kernelPlan->kernelFn;
  dim3 grid = {(unsigned)kernelPlan->nChannels, 1, 1};
  dim3 block = {(unsigned)kernelPlan->threadPerBlock, 1, 1};
  int smem = kernelPlan->kernelDynSmem;
  int driverVersion = 0;
  bool userKernelEvent = userLaunchEvent != nullptr;
  bool userKernelEventArmed = false;
  bool relayUserLaunchCompletionEvent = false;
  cudaStream_t relayStream = nullptr;

  void* extra[] = {CU_LAUNCH_PARAM_BUFFER_POINTER, kernelPlan->kernelSymArgs, CU_LAUNCH_PARAM_BUFFER_SIZE,
                   &kernelPlan->kernelArgsSize, CU_LAUNCH_PARAM_END};

  CUfunction fn = nullptr;
  NCCLCHECKGOTO(ncclCudaDriverVersion(&driverVersion), ret, do_return);
  CUDACHECKGOTO(cudaGetFuncBySymbol(&fn, sym), ret, do_return);

  if (CUDART_VERSION >= 11080 && driverVersion >= 11080) {
#if CUDART_VERSION >= 11080
    int compCap = comm->compCap;
    unsigned int clusterSize = (compCap >= 90) ? comm->config.cgaClusterSize : 0;

    CUlaunchConfig launchConfig = {0};
    CUlaunchAttribute launchAttrs[6] = {};
    int attrs = 0;
    if (clusterSize) {
      if (grid.x % clusterSize) clusterSize = 1;
      launchAttrs[attrs].id = CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION;
      launchAttrs[attrs++].value.clusterDim = {clusterSize, 1, 1};
      launchAttrs[attrs].id = CU_LAUNCH_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE;
      launchAttrs[attrs++].value.clusterSchedulingPolicyPreference = CU_CLUSTER_SCHEDULING_POLICY_SPREAD;
    }
#if CUDART_VERSION >= 12000
    if (compCap >= 90 && driverVersion >= 12000) {
      launchAttrs[attrs].id = CU_LAUNCH_ATTRIBUTE_MEM_SYNC_DOMAIN;
      launchAttrs[attrs++].value.memSyncDomain = (CUlaunchMemSyncDomain)ncclParamSymSchedMemSyncDomain();
    }
#endif
#if CUDART_VERSION >= 12030
    // A null launchEventOut means this kernel does not record the launch-order event.
    if (launchEventOut != nullptr) {
      launchAttrs[attrs].id = CU_LAUNCH_ATTRIBUTE_LAUNCH_COMPLETION_EVENT;
      launchAttrs[attrs].value.launchCompletionEvent.event = launchEventOut;
      launchAttrs[attrs].value.launchCompletionEvent.flags = 0;
      attrs++;
      // The launch-order event occupies the completion attribute, so relay the user's event.
      if (userKernelEvent) {
        NCCLCHECKGOTO(ncclUncapturedStreamPoolAcquire(&comm->sharedRes->uncapturedStreamPool, &relayStream), ret,
                      do_return);
        relayUserLaunchCompletionEvent = true;
        userKernelEventArmed = true;
      }
    } else if (userKernelEvent && driverVersion >= 12030) {
      launchAttrs[attrs].id = CU_LAUNCH_ATTRIBUTE_LAUNCH_COMPLETION_EVENT;
      launchAttrs[attrs].value.launchCompletionEvent.event = userLaunchEvent;
      launchAttrs[attrs].value.launchCompletionEvent.flags = 0;
      attrs++;
      userKernelEventArmed = true;
    }
    if (compCap >= 90 && driverVersion >= 12030) {
      launchAttrs[attrs].id = CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION;
      launchAttrs[attrs].value.programmaticStreamSerializationAllowed = 1;
      attrs++;
    }
#endif
#if CUDART_VERSION >= 13000
    if (compCap >= 100 && driverVersion >= 13000) {
      launchAttrs[attrs].id = CU_LAUNCH_ATTRIBUTE_NVLINK_UTIL_CENTRIC_SCHEDULING;
      launchAttrs[attrs].value.nvlinkUtilCentricScheduling = comm->config.nvlinkCentricSched;
      attrs++;
    }
#endif
    launchConfig.gridDimX = grid.x;
    launchConfig.gridDimY = grid.y;
    launchConfig.gridDimZ = grid.z;
    launchConfig.blockDimX = block.x;
    launchConfig.blockDimY = block.y;
    launchConfig.blockDimZ = block.z;
    launchConfig.sharedMemBytes = smem;
    launchConfig.attrs = launchAttrs;
    launchConfig.numAttrs = attrs;
    launchConfig.hStream = launchStream;
    INFO(
      NCCL_COLL,
      "launchSymKernelPlan grid=(%u,%u,%u) block=(%u,%u,%u) smem=%d launchAttrs=%p numAttrs=%d launchStream=%p fn=%p",
      grid.x, grid.y, grid.z, block.x, block.y, block.z, smem, (void*)launchAttrs, attrs, (void*)launchStream,
      (void*)fn);
    if (userKernelEvent && !userKernelEventArmed) {
      WARN("CUDA launch-completion events require CUDA 12.3 or newer; recording the user event before launch");
      CUDACHECKGOTO(cudaEventRecord(userLaunchEvent, launchStream), ret, do_return);
    }
    CUCHECKGOTO(cuLaunchKernelEx(&launchConfig, fn, nullptr, extra), ret, do_return);
    if (relayUserLaunchCompletionEvent) {
      CUDACHECKGOTO(cudaStreamWaitEvent(relayStream, launchEventOut, 0), ret, do_return);
      CUDACHECKGOTO(cudaEventRecord(userLaunchEvent, relayStream), ret, do_return);
    }
#endif
  } else {
    if (userKernelEvent) {
      WARN("CUDA launch-completion events require CUDA 12.3 or newer; recording the user event before launch");
      CUDACHECKGOTO(cudaEventRecord(userLaunchEvent, launchStream), ret, do_return);
    }
    CUCHECKGOTO(cuLaunchKernel(fn, grid.x, grid.y, grid.z, block.x, block.y, block.z, smem, launchStream, nullptr,
                               extra),
                ret, do_return);
  }

do_return:
  return ret;
}

static void CUDART_CB symSchedKernelPlanDestructor(void* arg_) {
  struct symSchedKernelPlanFreeArg* arg = (struct symSchedKernelPlanFreeArg*)arg_;
  free(arg->kernelSymArgs);
  ncclSchedProfilerLaunchDestroy(arg->profiler);
  free(arg);
}

static ncclResult_t scheduleSymTasks(struct ncclComm* comm,
                                     struct ncclIntruQueue<struct ncclTaskTuningInfo, &ncclTaskTuningInfo::next>* queue,
                                     struct ncclSymSchedPlan* symPlanOut) {
  ncclIntruQueueConstruct(&symPlanOut->streamPlans);

  while (!ncclIntruQueueEmpty(queue)) {
    struct ncclTaskTuningInfo* taskInfo = ncclIntruQueueDequeue(queue);
    cudaStream_t stream = symSchedTuningInfoStream(taskInfo);
    struct ncclSymSchedStreamPlan* streamPlan = nullptr;
    bool captured = false;
    struct ncclCudaGraph graph = {};

    NCCLCHECK(symSchedStreamCaptured(comm, stream, &captured, &graph));
    NCCLCHECK(symSchedGetStreamPlan(symPlanOut, stream, captured, graph, &streamPlan));
    NCCLCHECK(symSchedEnqueueKernelTask(comm, streamPlan, taskInfo));
  }

  NCCLCHECK(symSchedBuildKernelPlans(comm, symPlanOut));

  return ncclSuccess;
}

static ncclResult_t symSchedPlanFree(struct ncclSymSchedPlan* symPlan) {
  ncclResult_t ret = ncclSuccess;
  struct symSchedKernelPlanFreeArg* freeArg = nullptr;
  struct ncclSymSchedStreamPlan* streamPlan = ncclIntruQueueHead(&symPlan->streamPlans);

  while (streamPlan != nullptr) {
    struct ncclSymSchedStreamPlan* next = streamPlan->next;
    for (struct ncclSymSchedKernelPlan* kernelPlan = streamPlan->kernelPlans; kernelPlan != nullptr;
         kernelPlan = kernelPlan->next) {
      freeArg = nullptr;
      NCCLCHECKGOTO(ncclCalloc(&freeArg, 1), ret, sym_free_graph_arg);
      freeArg->kernelSymArgs = kernelPlan->kernelSymArgs;
      freeArg->profiler = kernelPlan->profiler;
      if (streamPlan->captured) {
        NCCLCHECKGOTO(ncclCudaGraphAddDestructor(streamPlan->graph, symSchedKernelPlanDestructor, freeArg), ret,
                      sym_free_graph_arg);
      } else {
        symSchedKernelPlanDestructor(freeArg);
      }
      kernelPlan->kernelSymArgs = nullptr;
      kernelPlan->profiler = nullptr;
      freeArg = nullptr;
    }
    delete streamPlan;
    streamPlan = next;
  }

exit:
  ncclIntruQueueConstruct(&symPlan->streamPlans);
  return ret;
sym_free_graph_arg:
  free(freeArg);
  while (streamPlan != nullptr) {
    struct ncclSymSchedStreamPlan* next = streamPlan->next;
    delete streamPlan;
    streamPlan = next;
  }
  goto exit;
}

ncclResult_t ncclSchedLaunchSymTasks(struct ncclComm* comm,
                                     struct ncclIntruQueue<struct ncclTaskTuningInfo, &ncclTaskTuningInfo::next>* queue,
                                     struct ncclStrongStream* depStream, struct ncclStrongStream* launchOrderStream) {
  if (ncclIntruQueueEmpty(queue)) return ncclSuccess;

  assert(comm->workArgsBytes >= ncclSymkDevWorkArgsSize<ncclSymkDevWorkArgs>(MAXCHANNELS, 1, ncclProfilerPluginLoaded()));

  struct ncclSymSchedPlan symPlan = {};
  struct ncclSymSchedKernelPlan* kernelPlan = nullptr;
  ncclResult_t ret = ncclSuccess;

  NCCLCHECKGOTO(scheduleSymTasks(comm, queue, &symPlan), ret, fail);

  for (struct ncclSymSchedStreamPlan* streamPlan = ncclIntruQueueHead(&symPlan.streamPlans); streamPlan != nullptr;
       streamPlan = streamPlan->next) {
    cudaStream_t launchStream = streamPlan->stream;
    cudaEvent_t finishedEvent = comm->sharedRes->scratchEvent;
    cudaStream_t acqDepStream;
    cudaStream_t acqLaunchOrderStream;
    cudaEvent_t ncclLaunchEvent = nullptr;
    enum ncclImplicitOrder implicitOrder;

    NCCLCHECKGOTO(symSchedGetImplicitOrder(&implicitOrder, streamPlan->captured), ret, fail);

    if (depStream) {
      NCCLCHECKGOTO(ncclStrongStreamAcquire(streamPlan->graph, depStream, false, &acqDepStream), ret, fail);
      NCCLCHECKGOTO(ncclStreamWaitStream(launchStream, acqDepStream, finishedEvent), ret, fail);
    }

    if (launchOrderStream && implicitOrder != ncclImplicitOrderNone) {
      NCCLCHECKGOTO(ncclStrongStreamAcquire(streamPlan->graph, launchOrderStream, false, &acqLaunchOrderStream), ret,
                    fail);
      NCCLCHECKGOTO(ncclStreamWaitStream(launchStream, acqLaunchOrderStream, finishedEvent), ret, fail);
    }

    for (kernelPlan = streamPlan->kernelPlans; kernelPlan != nullptr; kernelPlan = kernelPlan->next) {
      bool lastKernel = kernelPlan->next == nullptr;
      // Only the last kernel records the stream's user launch-completion event.
      cudaEvent_t userLaunchEvent = lastKernel ? streamPlan->launchCompletionEvent : nullptr;
      // Serial order records its event after the loop. Launch order arms it on the last kernel.
      ncclLaunchEvent = implicitOrder == ncclImplicitOrderLaunch && lastKernel ? comm->sharedRes->launchEvent : nullptr;
      NCCLCHECKGOTO(ncclProfilerSchedPreLaunch(kernelPlan->profiler, launchStream), ret, fail);
      NCCLCHECKGOTO(launchSymKernelPlan(comm, kernelPlan, launchStream, userLaunchEvent, ncclLaunchEvent), ret,
                    sym_post_launch);
      NCCLCHECKGOTO(ncclProfilerSchedPostLaunch(kernelPlan->profiler), ret, fail);
      // A captured launch emits its group/coll events on every replay.
      if (kernelPlan->profiler != nullptr && streamPlan->captured) {
        CUDACHECKGOTO(cudaLaunchHostFunc(launchStream, ncclSchedProfilerReplayCallback, kernelPlan->profiler), ret,
                      fail);
      }
    }

    if (ncclLaunchEvent == nullptr && streamPlan->kernelPlans != nullptr && implicitOrder == ncclImplicitOrderSerial) {
      CUDACHECKGOTO(cudaEventRecord(comm->sharedRes->scratchEvent, launchStream), ret, fail);
      ncclLaunchEvent = comm->sharedRes->scratchEvent;
    }

    if (depStream) {
      CUDACHECKGOTO(cudaEventRecord(finishedEvent, launchStream), ret, fail);
      NCCLCHECKGOTO(ncclStreamAdvanceToEvent(streamPlan->graph, acqDepStream, finishedEvent), ret, fail);
      NCCLCHECKGOTO(ncclStrongStreamRelease(streamPlan->graph, depStream, false), ret, fail);
    }
    if (launchOrderStream && implicitOrder != ncclImplicitOrderNone) {
      NCCLCHECKGOTO(ncclStreamAdvanceToEvent(streamPlan->graph, acqLaunchOrderStream, ncclLaunchEvent), ret, fail);
      NCCLCHECKGOTO(ncclStrongStreamRelease(streamPlan->graph, launchOrderStream, false), ret, fail);
    }
  }

exit:
  NCCLCHECKIGNORE(symSchedPlanFree(&symPlan), ret);
  return ret;
sym_post_launch:
  (void)ncclProfilerSchedPostLaunch(kernelPlan->profiler);
  goto fail;
fail:
  goto exit;
}
