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
#include <algorithm>
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

struct ncclSymSchedKernelTaskQueue {
  struct ncclSymSchedKernelTaskQueue* next;
  int symKernelId;
  int maxChannels;
  bool captured;
  struct ncclIntruQueue<struct ncclTaskTuningInfo, &ncclTaskTuningInfo::next> taskQueue;
  struct ncclSymSchedKernelPlan* kernelPlans;
};

struct ncclSymSchedStreamPlan {
  struct ncclSymSchedStreamPlan* next;
  cudaStream_t stream;
  bool captured;
  struct ncclCudaGraph graph;
  struct ncclSymSchedKernelTaskQueue* kernelTaskQueues;
};

struct ncclSymSchedPlan {
  struct ncclSymSchedStreamPlan* noncapturedStreamPlans;
  struct ncclSymSchedStreamPlan* capturedStreamPlans;
};

struct symSchedKernelPlanFreeArg {
  struct ncclSymkDevWorkArgs* kernelSymArgs;
  struct ncclSchedProfilerLaunch* profiler;
};

static ncclDevRedOp_t symSchedRedOp(ncclRedOp_t redOp, ncclDevRedOp_t devRedOp) {
  if (redOp == ncclAvg) return ncclDevSumPostDiv;
  return devRedOp;
}

static struct ncclDevRedOpFull symSchedSymkOpDev(
    struct ncclComm* comm, struct ncclTaskTuningInfo* tInfo
  ) {
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

static ncclResult_t symSchedWinRegTypeFromTuningInfo(
    struct ncclComm* comm, struct ncclTaskTuningInfo* tInfo, ncclSymRegType_t* winRegType
  ) {
  struct ncclRawTaskColl* raw = &tInfo->raw->coll;
  struct ncclDevrWindow* sendWin = nullptr;
  struct ncclDevrWindow* recvWin = nullptr;

  if (tInfo->raw->kind != ncclTaskKindColl) return ncclInternalError;

  NCCLCHECK(ncclDevrFindWindow(comm, raw->sendbuff, &sendWin));
  NCCLCHECK(ncclDevrFindWindow(comm, raw->recvbuff, &recvWin));
  NCCLCHECK(ncclGetSymRegType(sendWin, recvWin, winRegType));
  return ncclSuccess;
}

static ncclResult_t symSchedMakeDevWorkFromTuningInfo(
    struct ncclComm* comm, struct ncclTaskTuningInfo* tInfo, struct ncclSymkDevWork* outDevWork
  ) {
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

static ncclResult_t symSchedStreamCaptured(
    struct ncclComm* comm, cudaStream_t stream,
    bool* capturedOut, struct ncclCudaGraph* graphOut
  ) {
  NCCLCHECK(ncclCudaGetCapturingGraph(graphOut, stream, comm->config.graphUsageMode));
  *capturedOut = ncclCudaGraphValid(*graphOut);
  return ncclSuccess;
}

static struct ncclSymSchedStreamPlan* symSchedFindStreamPlan(
    struct ncclSymSchedStreamPlan* head, cudaStream_t stream
  ) {
  for (struct ncclSymSchedStreamPlan* plan = head; plan != nullptr; plan = plan->next) {
    if (plan->stream == stream) return plan;
  }
  return nullptr;
}

static ncclResult_t symSchedGetStreamPlan(
    struct ncclComm* comm,
    struct ncclSymSchedStreamPlan** head,
    cudaStream_t stream,
    bool captured,
    const struct ncclCudaGraph& graph,
    struct ncclSymSchedStreamPlan** streamPlanOut
  ) {
  struct ncclSymSchedStreamPlan* streamPlan = symSchedFindStreamPlan(*head, stream);
  if (streamPlan != nullptr) {
    *streamPlanOut = streamPlan;
    return ncclSuccess;
  }

  streamPlan = ncclMemoryStackAlloc<struct ncclSymSchedStreamPlan>(&comm->memScoped);
  streamPlan->stream = stream;
  streamPlan->captured = captured;
  streamPlan->graph = graph;
  streamPlan->kernelTaskQueues = nullptr;
  streamPlan->next = *head;
  *head = streamPlan;
  *streamPlanOut = streamPlan;
  return ncclSuccess;
}

static ncclResult_t symSchedEnqueueKernelTask(
    struct ncclComm* comm,
    struct ncclSymSchedStreamPlan* streamPlan,
    struct ncclTaskTuningInfo* taskInfo
  ) {
  int symKernelId = taskInfo->tuningOut.symKernelId;
  // Symmetric selections never populate tuningOut.maxChannels (ncclTuningGetChannels only runs
  // for algo/proto picks), so the channel budget comes from nChannels like the legacy scheduler.
  int maxChannels = taskInfo->tuningOut.nChannels;
  struct ncclSymSchedKernelTaskQueue* tail = nullptr;
  struct ncclSymSchedKernelTaskQueue* subQueue = streamPlan->kernelTaskQueues;
  // LL kernels do not support grouping, so each LL task gets its own subQueue.
  bool isLLKernel = ((1 << symKernelId) & ncclSymkLLKernelMask()) != 0;

  while (subQueue != nullptr) {
    if (!isLLKernel && subQueue->symKernelId == symKernelId) {
      ncclIntruQueueEnqueue(&subQueue->taskQueue, taskInfo);
      subQueue->maxChannels += maxChannels;
      if (subQueue->maxChannels > comm->nChannels) subQueue->maxChannels = comm->nChannels;
      return ncclSuccess;
    }
    tail = subQueue;
    subQueue = subQueue->next;
  }

  subQueue = ncclMemoryStackAlloc<struct ncclSymSchedKernelTaskQueue>(&comm->memScoped);
  subQueue->symKernelId = symKernelId;
  subQueue->maxChannels = maxChannels;
  subQueue->captured = streamPlan->captured;
  subQueue->kernelPlans = nullptr;
  subQueue->next = nullptr;
  ncclIntruQueueConstruct(&subQueue->taskQueue);
  ncclIntruQueueEnqueue(&subQueue->taskQueue, taskInfo);

  if (tail == nullptr) {
    streamPlan->kernelTaskQueues = subQueue;
  } else {
    tail->next = subQueue;
  }
  return ncclSuccess;
}

static int symSchedCountKernelBatch(
    struct ncclComm* comm, struct ncclSymSchedKernelTaskQueue* kernelTaskQueue
  ) {
  int nElems = kernelTaskQueue->taskQueue.nElems;
  if (nElems == 0) return 0;

  int maxChannels = kernelTaskQueue->maxChannels;
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

static ncclResult_t symSchedAppendKernelPlan(
    struct ncclSymSchedKernelTaskQueue* kernelTaskQueue, struct ncclSymSchedKernelPlan* kernelPlan
  ) {
  struct ncclSymSchedKernelPlan** tail = &kernelTaskQueue->kernelPlans;
  while (*tail != nullptr) tail = &(*tail)->next;
  *tail = kernelPlan;
  kernelPlan->next = nullptr;
  return ncclSuccess;
}

static ncclResult_t symSchedBuildKernelPlan(
    struct ncclComm* comm,
    struct ncclSymSchedKernelTaskQueue* kernelTaskQueue,
    int nWorks
  ) {
  ncclResult_t ret = ncclSuccess;
  struct ncclTaskTuningInfo* headTask = ncclIntruQueueHead(&kernelTaskQueue->taskQueue);
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
  int nMaxChannels = kernelTaskQueue->maxChannels;
  struct ncclSymkDevWork* workBufPtr = nullptr;
  struct ncclSymkChannelWorkRange* workRangePtr = nullptr;
  struct ncclSymkDevWorkArgs* argsBuf = nullptr;
  struct ncclSymSchedKernelPlan* kernelPlan = nullptr;
  struct ncclTaskTuningInfo* taskInfo = nullptr;
  ncclSymkKernelId kernelId = (ncclSymkKernelId)kernelTaskQueue->symKernelId;
  struct ncclSchedProfilerLaunch* profiler = nullptr;
  // Profiling requested = plugin loaded and the call's mask has ncclProfileKernelCh.
  // Device-side instrumentation is eager-only: the args buffer is snapshotted at launch,
  // so the mirrored counters cannot advance across graph replays. Under capture the clean
  // kernel runs and only the host-side group/coll events are replayed.
  int eActivationMask = headTask->raw->profiler.eActivationMask;
  bool profilingRequested = ncclProfilerPluginLoaded() && (eActivationMask & ncclProfileKernelCh);
  bool profilerEnabled = profilingRequested && !kernelTaskQueue->captured;

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
      NCCLCHECKGOTO(ncclSchedProfilerLaunchCreate(comm, kernelTaskQueue->captured, headTask, nWorks, &profiler), ret,
                    fail);
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

    taskInfo = ncclIntruQueueDequeue(&kernelTaskQueue->taskQueue);
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
          if ((remainCell - cellLeft) * NCCL_SYM_KERNEL_CELL_SIZE <= (1 << 10) ||
              workIndex + 1 == nWorks) {
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
         ncclFuncToString(headRaw->func),
         logCount * ncclTypeSize(datatype),
         ncclSymkKernelIdToString(kernelTaskQueue->symKernelId),
         curChannel, kernelPlan->threadPerBlock, nWorks);
  }

  NCCLCHECKGOTO(symSchedAppendKernelPlan(kernelTaskQueue, kernelPlan), ret, fail);

exit:
  return ret;
fail:
  if (taskInfo != nullptr) symSchedFreeTuningInfoRaw(comm, taskInfo);
  if (argsBuf != nullptr) free(argsBuf);
  ncclSchedProfilerLaunchDestroy(profiler);
  goto exit;
}

static ncclResult_t symSchedBuildKernelTaskQueuePlans(
    struct ncclComm* comm, struct ncclSymSchedKernelTaskQueue* kernelTaskQueue
  ) {
  kernelTaskQueue->kernelPlans = nullptr;
  while (!ncclIntruQueueEmpty(&kernelTaskQueue->taskQueue)) {
    int nWorks = symSchedCountKernelBatch(comm, kernelTaskQueue);
    if (nWorks == 0) return ncclInternalError;
    NCCLCHECK(symSchedBuildKernelPlan(comm, kernelTaskQueue, nWorks));
  }
  return ncclSuccess;
}

static ncclResult_t symSchedBuildStreamKernelPlans(
    struct ncclComm* comm, struct ncclSymSchedStreamPlan* streamPlan
  ) {
  for (struct ncclSymSchedKernelTaskQueue* kernelTaskQueue = streamPlan->kernelTaskQueues;
       kernelTaskQueue != nullptr; kernelTaskQueue = kernelTaskQueue->next) {
    NCCLCHECK(symSchedBuildKernelTaskQueuePlans(comm, kernelTaskQueue));
  }
  return ncclSuccess;
}

static ncclResult_t symSchedBuildKernelPlans(
    struct ncclComm* comm, struct ncclSymSchedPlan* symPlan
  ) {
  for (struct ncclSymSchedStreamPlan* streamPlan = symPlan->noncapturedStreamPlans;
       streamPlan != nullptr; streamPlan = streamPlan->next) {
    NCCLCHECK(symSchedBuildStreamKernelPlans(comm, streamPlan));
  }
  for (struct ncclSymSchedStreamPlan* streamPlan = symPlan->capturedStreamPlans;
       streamPlan != nullptr; streamPlan = streamPlan->next) {
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

static ncclResult_t symSchedLaunchKernel(
    struct ncclComm* comm,
    struct ncclSymSchedStreamPlan* streamPlan,
    struct ncclSymSchedKernelPlan* kernelPlan,
    enum ncclImplicitOrder implicitOrder,
    cudaEvent_t* launchEventOut
  ) {
  ncclResult_t ret = ncclSuccess;
  cudaStream_t launchStream = streamPlan->stream;
  void* sym = kernelPlan->kernelFn;
  dim3 grid = {(unsigned)kernelPlan->nChannels, 1, 1};
  dim3 block = {(unsigned)kernelPlan->threadPerBlock, 1, 1};
  int smem = kernelPlan->kernelDynSmem;
  int driverVersion = 0;

  if (launchEventOut) *launchEventOut = nullptr;

  // Nothing to unwind if this fails, so return directly (a goto here would cross initializers).
  NCCLCHECK(ncclProfilerSchedPreLaunch(kernelPlan->profiler, launchStream));

  void* extra[] = {
    CU_LAUNCH_PARAM_BUFFER_POINTER, kernelPlan->kernelSymArgs,
    CU_LAUNCH_PARAM_BUFFER_SIZE, &kernelPlan->kernelArgsSize,
    CU_LAUNCH_PARAM_END
  };

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
      launchAttrs[attrs++].value.memSyncDomain =
        (CUlaunchMemSyncDomain)ncclParamSymSchedMemSyncDomain();
    }
#endif
#if CUDART_VERSION >= 12030
    if (launchEventOut && implicitOrder == ncclImplicitOrderLaunch) {
      launchAttrs[attrs].id = CU_LAUNCH_ATTRIBUTE_LAUNCH_COMPLETION_EVENT;
      launchAttrs[attrs].value.launchCompletionEvent.event = comm->sharedRes->launchEvent;
      launchAttrs[attrs].value.launchCompletionEvent.flags = 0;
      attrs++;
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
    INFO(NCCL_COLL, "symSchedLaunchKernel grid=(%u,%u,%u) block=(%u,%u,%u) smem=%d launchAttrs=%p numAttrs=%d launchStream=%p fn=%p",
         grid.x, grid.y, grid.z, block.x, block.y, block.z, smem, (void*)launchAttrs, attrs, (void*)launchStream, (void*)fn);
    CUCHECKGOTO(cuLaunchKernelEx(&launchConfig, fn, nullptr, extra), ret, do_return);
#endif
  } else {
    CUCHECKGOTO(
      cuLaunchKernel(fn, grid.x, grid.y, grid.z, block.x, block.y, block.z, smem, launchStream, nullptr, extra),
      ret, do_return);
  }

  if (launchEventOut) {
    if (implicitOrder == ncclImplicitOrderLaunch) {
      *launchEventOut = comm->sharedRes->launchEvent;
    } else if (implicitOrder == ncclImplicitOrderSerial) {
      CUDACHECKGOTO(cudaEventRecord(comm->sharedRes->scratchEvent, launchStream), ret, do_return);
      *launchEventOut = comm->sharedRes->scratchEvent;
    }
  }

do_return:
  (void)ncclProfilerSchedPostLaunch(kernelPlan->profiler);
  return ret;
}

static void CUDART_CB symSchedKernelPlanDestructor(void* arg_) {
  struct symSchedKernelPlanFreeArg* arg = (struct symSchedKernelPlanFreeArg*)arg_;
  free(arg->kernelSymArgs);
  ncclSchedProfilerLaunchDestroy(arg->profiler);
  free(arg);
}

static ncclResult_t launchSymStreamPlan(
    struct ncclComm* comm,
    struct ncclSymSchedStreamPlan* streamPlans,
    struct ncclStrongStream* depStream,
    struct ncclStrongStream* launchOrderStream
  ) {
  ncclResult_t ret = ncclSuccess;

  for (struct ncclSymSchedStreamPlan* streamPlan = streamPlans; streamPlan != nullptr; streamPlan = streamPlan->next) {
    cudaStream_t launchStream = streamPlan->stream;
    cudaEvent_t finishedEvent = comm->sharedRes->scratchEvent;
    cudaStream_t acqDepStream;
    cudaStream_t acqLaunchOrderStream;
    cudaEvent_t launchEvent = nullptr;
    cudaEvent_t* launchEventOut = nullptr;
    enum ncclImplicitOrder implicitOrder;

    NCCLCHECKGOTO(symSchedGetImplicitOrder(&implicitOrder, streamPlan->captured), ret, fail);

    if (depStream) {
      NCCLCHECKGOTO(ncclStrongStreamAcquire(streamPlan->graph, depStream, false, &acqDepStream), ret, fail);
      NCCLCHECKGOTO(ncclStreamWaitStream(launchStream, acqDepStream, finishedEvent), ret, fail);
    }

    if (launchOrderStream && implicitOrder != ncclImplicitOrderNone) {
      NCCLCHECKGOTO(ncclStrongStreamAcquire(streamPlan->graph, launchOrderStream, false, &acqLaunchOrderStream), ret, fail);
      NCCLCHECKGOTO(ncclStreamWaitStream(launchStream, acqLaunchOrderStream, finishedEvent), ret, fail);
    }

    for (struct ncclSymSchedKernelTaskQueue* kernelTaskQueue = streamPlan->kernelTaskQueues;
         kernelTaskQueue != nullptr; kernelTaskQueue = kernelTaskQueue->next) {
      for (struct ncclSymSchedKernelPlan* kernelPlan = kernelTaskQueue->kernelPlans;
           kernelPlan != nullptr; kernelPlan = kernelPlan->next) {
        launchEventOut = implicitOrder != ncclImplicitOrderNone &&
          kernelPlan->next == nullptr && kernelTaskQueue->next == nullptr ? &launchEvent : nullptr;
        NCCLCHECKGOTO(symSchedLaunchKernel(comm, streamPlan, kernelPlan, implicitOrder, launchEventOut), ret, fail);
        // A captured launch emits its group/coll events on every replay.
        if (kernelPlan->profiler != nullptr && streamPlan->captured) {
          CUDACHECKGOTO(cudaLaunchHostFunc(launchStream, ncclSchedProfilerReplayCallback, kernelPlan->profiler), ret,
                        fail);
        }
      }
    }

    if (depStream) {
      CUDACHECKGOTO(cudaEventRecord(finishedEvent, launchStream), ret, fail);
      NCCLCHECKGOTO(ncclStreamAdvanceToEvent(streamPlan->graph, acqDepStream, finishedEvent), ret, fail);
      NCCLCHECKGOTO(ncclStrongStreamRelease(streamPlan->graph, depStream, false), ret, fail);
    }
    if (launchOrderStream && implicitOrder != ncclImplicitOrderNone) {
      NCCLCHECKGOTO(ncclStreamAdvanceToEvent(streamPlan->graph, acqLaunchOrderStream, launchEvent), ret, fail);
      NCCLCHECKGOTO(ncclStrongStreamRelease(streamPlan->graph, launchOrderStream, false), ret, fail);
    }

    if (streamPlan->captured) {
      for (struct ncclSymSchedKernelTaskQueue* kernelTaskQueue = streamPlan->kernelTaskQueues;
           kernelTaskQueue != nullptr; kernelTaskQueue = kernelTaskQueue->next) {
        for (struct ncclSymSchedKernelPlan* kernelPlan = kernelTaskQueue->kernelPlans;
             kernelPlan != nullptr; kernelPlan = kernelPlan->next) {
          struct symSchedKernelPlanFreeArg* freeArg = nullptr;
          NCCLCHECKGOTO(ncclCalloc(&freeArg, 1), ret, fail);
          freeArg->kernelSymArgs = kernelPlan->kernelSymArgs;
          freeArg->profiler = kernelPlan->profiler;
          ret = ncclCudaGraphAddDestructor(
              streamPlan->graph, symSchedKernelPlanDestructor, (void*)freeArg);
          if (ret != ncclSuccess) {
            free(freeArg);
            goto fail;
          }
          kernelPlan->kernelSymArgs = nullptr;
          kernelPlan->profiler = nullptr;
        }
      }
    } else {
      for (struct ncclSymSchedKernelTaskQueue* kernelTaskQueue = streamPlan->kernelTaskQueues;
           kernelTaskQueue != nullptr; kernelTaskQueue = kernelTaskQueue->next) {
        for (struct ncclSymSchedKernelPlan* kernelPlan = kernelTaskQueue->kernelPlans;
             kernelPlan != nullptr; kernelPlan = kernelPlan->next) {
          free(kernelPlan->kernelSymArgs);
          kernelPlan->kernelSymArgs = nullptr;
          ncclSchedProfilerLaunchDestroy(kernelPlan->profiler);
          kernelPlan->profiler = nullptr;
        }
      }
    }
  }

exit:
  return ret;
fail:
  goto exit;
}

static ncclResult_t launchSymTasks(
    struct ncclComm* comm,
    struct ncclSymSchedPlan* symPlan,
    struct ncclStrongStream* depStream,
    struct ncclStrongStream* launchOrderStream
  ) {
  ncclResult_t ret = ncclSuccess;

  NCCLCHECKGOTO(launchSymStreamPlan(comm, symPlan->noncapturedStreamPlans, depStream, launchOrderStream), ret, fail);
  NCCLCHECKGOTO(launchSymStreamPlan(comm, symPlan->capturedStreamPlans, depStream, launchOrderStream), ret, fail);

exit:
  return ret;
fail:
  goto exit;
}

static ncclResult_t scheduleSymTasks(
    struct ncclComm* comm,
    struct ncclIntruQueue<struct ncclTaskTuningInfo, &ncclTaskTuningInfo::next>* queue,
    struct ncclSymSchedPlan* symPlanOut
  ) {
  symPlanOut->noncapturedStreamPlans = nullptr;
  symPlanOut->capturedStreamPlans = nullptr;

  while (!ncclIntruQueueEmpty(queue)) {
    struct ncclTaskTuningInfo* taskInfo = ncclIntruQueueDequeue(queue);
    cudaStream_t stream = symSchedTuningInfoStream(taskInfo);
    struct ncclSymSchedStreamPlan* streamPlan = nullptr;
    struct ncclSymSchedStreamPlan** streamListHead = nullptr;
    bool captured = false;
    struct ncclCudaGraph graph = {};

    NCCLCHECK(symSchedStreamCaptured(comm, stream, &captured, &graph));
    streamListHead = captured ? &symPlanOut->capturedStreamPlans : &symPlanOut->noncapturedStreamPlans;
    NCCLCHECK(symSchedGetStreamPlan(comm, streamListHead, stream, captured, graph, &streamPlan));
    NCCLCHECK(symSchedEnqueueKernelTask(comm, streamPlan, taskInfo));
  }

  NCCLCHECK(symSchedBuildKernelPlans(comm, symPlanOut));

  return ncclSuccess;
}

ncclResult_t ncclScheduleSymTasks(
    struct ncclComm* comm,
    struct ncclIntruQueue<struct ncclTaskTuningInfo, &ncclTaskTuningInfo::next>* queue,
    struct ncclStrongStream* depStream,
    struct ncclStrongStream* launchOrderStream
  ) {
  if (ncclIntruQueueEmpty(queue)) return ncclSuccess;

  NCCLCHECK(ncclDevrInitOnce(comm));
  assert(comm->workArgsBytes >= ncclSymkDevWorkArgsSize<ncclSymkDevWorkArgs>(MAXCHANNELS, 1, ncclProfilerPluginLoaded()));

  struct ncclSymSchedPlan symPlan = {};
  NCCLCHECK(scheduleSymTasks(comm, queue, &symPlan));
  NCCLCHECK(launchSymTasks(comm, &symPlan, depStream, launchOrderStream));
  return ncclSuccess;
}
