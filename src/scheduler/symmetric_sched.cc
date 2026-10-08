/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2015-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_SYMMETRIC_SCHED_H_
#define NCCL_SYMMETRIC_SCHED_H_

#include "device.h"
#include "nccl.h"
#include "scheduler.h"
#include "tuning.h"
#include "enqueue.h"
#include "config/algorithm_registry.h"
#include "profiler.h"
#include <cuda_fp16.h>
#if defined(__CUDA_FP8_TYPES_EXIST__)
#include <cuda_fp8.h>
#endif

ncclDevRedOp_t symkRedOp(ncclRedOp_t redOp, ncclDevRedOp_t devRedOp) {
  if (redOp == ncclAvg) {
    return ncclDevSumPostDiv;
  }
  return devRedOp;
}

void convertSymTaskDevOp(struct ncclComm* comm, struct ncclTaskColl* task) {
  task->opDev.op = symkRedOp(task->opHost, task->opDev.op);
  if (task->opDev.op == ncclDevSumPostDiv) {
    // LDMC uses the same accumulator type as data type. Do not re-pack the scalar.
    if (task->devFuncId == (uint32_t)ncclSymkKernelId_ReduceScatter_LDMC) {
      return;
    }
    union {
      __half f16;
      float f32;
      uint64_t u64;
      void* ptr;
    };
    u64 = 0;
    switch (task->datatype) {
      // 16-bit floats use float accumulator
    case ncclFloat16:
#if defined(__CUDA_BF16_TYPES_EXIST__)
    case ncclBfloat16:
#endif
      f32 = float(1.0 / comm->nRanks);  // ncclDevSumPostDiv actually multiplies by the scalar, not divides.
      task->opDev.scalarArg = u64;
      return;
#if defined(__CUDA_FP8_TYPES_EXIST__)
    case ncclFloat8e4m3:
    case ncclFloat8e5m2:
      f16 = __float2half(float(1.0 / comm->nRanks));
      task->opDev.scalarArg = u64;
      return;
#endif
    default:
      break;
    }
  }
}

// Match device deep-tier gate: uint32_t(input.offset - output.offset) % 16 == 0.
static ncclResult_t symBatchAlignment(struct ncclTaskColl* headTask, bool* relativeAligned16B) {
  *relativeAligned16B = true;
  for (struct ncclTaskColl* t = headTask; t != nullptr; t = t->isSymLast ? nullptr : t->next) {
    size_t inputOff = t->sendWin ? (uintptr_t)t->sendbuff - (uintptr_t)t->sendWin->userPtr : (uintptr_t)t->sendbuff;
    size_t outputOff = t->recvWin ? (uintptr_t)t->recvbuff - (uintptr_t)t->recvWin->userPtr : (uintptr_t)t->recvbuff;
    *relativeAligned16B &= uint32_t(inputOff - outputOff) % 16 == 0;
    if (t->isSymLast) break;
  }
  return ncclSuccess;
}

static size_t symkMaxDevWorkArgsSize(int nChannels, int nWorks, bool profiler) {
  return std::max(ncclSymkDevWorkArgsSize<ncclSymkDevWorkArgs>(nChannels, nWorks, profiler),
                  ncclSymkDevWorkArgsSize<ncclGenkDevWorkArgs>(nChannels, nWorks, profiler));
}

static void setMinChunkPayloadBytes(struct ncclSymkDevWorkArgs* /*args*/, int /*minChunkPayloadBytes*/) {}

static void setMinChunkPayloadBytes(struct ncclGenkDevWorkArgs* args, int minChunkPayloadBytes) {
  args->minChunkPayloadBytes = minChunkPayloadBytes;
}

ncclResult_t ncclMakeSymmetricTaskList(struct ncclComm* comm, struct ncclTaskColl* task,
                                       struct ncclIntruQueue<struct ncclTaskColl, &ncclTaskColl::next>* symTaskQueue,
                                       struct ncclTaskColl** remainTasksHead, bool* needGenkDevComm) {
  ncclResult_t ret = ncclSuccess;
  int fnOpTySymCount = 0;
  struct ncclTaskColl* tasksSymByFnOpTy[ncclNumFuncs * ncclNumDevRedOps * ncclNumTypes * ncclNumSymRegTypes];
  int fnOpTySymIndices[ncclNumFuncs * ncclNumDevRedOps * ncclNumTypes * ncclNumSymRegTypes];
  struct ncclKernelPlanner* planner = &comm->planner;
  struct ncclTaskColl* remainTasksTail = nullptr;
  bool foundSymm = false;

  memset(tasksSymByFnOpTy, 0, sizeof(tasksSymByFnOpTy));
  *remainTasksHead = nullptr;
  if (task) {
    NCCLCHECK(ncclDevrInitOnce(comm));
  }
  while (task != nullptr) {
    int index;
    struct ncclTaskColl* next = task->next;
    ncclDevRedOp_t symkOp = symkRedOp(task->opHost, task->opDev.op);
    bool symAvailable = ncclSymkAvailable(comm, task->func, symkOp, task->datatype, task->count);
    // Env (NCCL_ALGO/PROTO/SYM_KERNEL) is a global override that wins over per-call
    // algSelection for any function it forced.
    uint64_t effAlgMask = comm->tuningContext.forced[task->func] ? 0 : task->algMask;
    bool cfgAllowsSymk = (effAlgMask == 0) || ((effAlgMask & NCCL_TUNING_MASK_SYM_KERNELS) != 0);

    if (symAvailable && cfgAllowsSymk) {
      NCCLCHECK(ncclDevrFindWindow(comm, task->sendbuff, &task->sendWin));
      NCCLCHECK(ncclDevrFindWindow(comm, task->recvbuff, &task->recvWin));
      NCCLCHECK(ncclGetSymRegType(task->sendWin, task->recvWin, &task->winRegType));

      index =
        (((int)task->func * ncclNumDevRedOps + symkOp) * ncclNumTypes + (int)task->datatype) * ncclNumSymRegTypes +
        (int)task->winRegType;
      if (tasksSymByFnOpTy[index] == nullptr) fnOpTySymIndices[fnOpTySymCount++] = index;
      task->next = tasksSymByFnOpTy[index];
      tasksSymByFnOpTy[index] = task;
      planner->nTasksColl--;
      foundSymm = true;
    } else {
      if (*remainTasksHead) {
        remainTasksTail->next = task;
        remainTasksTail = task;
      } else {
        *remainTasksHead = remainTasksTail = task;
      }
    }
    task = next;
  }
  if (remainTasksTail) remainTasksTail->next = nullptr;
  if (!foundSymm) return ret;

  // make sure kernel args space can hold at least a single work
  size_t const minArgsSize = symkMaxDevWorkArgsSize(MAXCHANNELS, 1, ncclProfilerPluginLoaded());
  if (comm->workArgsBytes < minArgsSize) {
    WARN("Symmetric kernel args size %u is smaller than minimum size %zu", comm->workArgsBytes, minArgsSize);
    return ncclInternalError;
  }

  // Determine symmetric tasks kernels
  for (int cursor = 0; cursor < fnOpTySymCount; cursor++) {
    struct ncclTaskColl* task = tasksSymByFnOpTy[fnOpTySymIndices[cursor]];
    while (task != NULL) {
      ncclSymkKernelId kernelId = ncclSymkKernelId_Count;
      int nChannels = MAXCHANNELS;
      int nWarps = 0;
      int nWorks = 0;
      size_t countTotal = 0, countMax = 0;
      struct ncclTaskColl* headTask = task;
      size_t cellCount = NCCL_SYM_KERNEL_CELL_SIZE / ncclTypeSize(headTask->datatype);
      ncclDevRedOp_t symkOp = symkRedOp(task->opHost, task->opDev.op);
      // For now we assume higher kernel id means a kernel for larger data size
      while (task != nullptr) {
        size_t count;
        nWorks++;
        count = alignUp(task->count, cellCount);
        countTotal += count;
        // Keep countMax as the true largest work count; TMA eligibility is per work, not per aligned batch.
        if (task->count > countMax) countMax = task->count;
        // A configured task forms its own singleton batch: end here if this task is
        // configured (it is the head), or if the next task is configured (so it starts
        // its own batch). This realizes the per-call caps exactly and never ignores a
        // non-head configured task.
        bool configBoundary = task->aggIsolate || (task->next != nullptr && task->next->aggIsolate);
        if (symkMaxDevWorkArgsSize(MAXCHANNELS, nWorks + 1, ncclProfilerPluginLoaded()) > comm->workArgsBytes ||
            task->next == nullptr || configBoundary) {
          task->isSymLast = 1;
          break;
        }
        task = task->next;
      }
      struct ncclTuningInput_t input;
      input.comm = comm;
      input.tuningMask = NCCL_TUNING_MASK_SYM_KERNELS;
      // Env (NCCL_ALGO/PROTO/SYM_KERNEL) is a global override that wins over per-call
      // algSelection for any function it forced.
      uint64_t effAlgMask = comm->tuningContext.forced[headTask->func] ? 0 : headTask->algMask;
      if (effAlgMask != 0) {
        uint64_t symkMask = effAlgMask & NCCL_TUNING_MASK_SYM_KERNELS;
        if (symkMask != 0) input.tuningMask = symkMask;
      }
      input.func = headTask->func;
      input.redOp = headTask->opHost;
      input.devRedOp = symkOp;
      input.datatype = headTask->datatype;
      input.nBytes = countTotal * ncclTypeSize(headTask->datatype);
      input.numPipeOps = divUp(nWorks, comm->nChannels);
      input.count = headTask->count;
      input.countMax = countMax;
      input.nWorks = nWorks;
      input.winRegType = headTask->winRegType;
      input.inPlace = nWorks == 1 && headTask->func == ncclFuncAllGather &&
                      ncclAllGatherIsInPlace(headTask->sendbuff, headTask->recvbuff, comm->rank,
                                             headTask->count * ncclTypeSize(headTask->datatype));
      NCCLCHECK(symBatchAlignment(headTask, &input.symAligned16B));
      input.chunkSteps = headTask->chunkSteps;
      input.sliceSteps = headTask->sliceSteps;
      input.minCTAs = headTask->minCTAs;
      input.maxCTAs = headTask->maxCTAs;
      input.CTAPolicy = headTask->CTAPolicy;
      // Symmetric kernels use comm->symkState.hasLsaMultimem for multicast capability.
      // input.nvlsSupport only gates NVLS/NVLS_TREE during general-kernel fallback tuning.
      input.nvlsSupport =
        ncclNvlsTransportEnabled(comm) &&
        (ncclNvlsSupported(headTask->opDev.op, headTask->datatype) || headTask->func == ncclFuncAllGather);
      NCCLCHECK(ncclGetCollNetSupport(comm, headTask, &input.collNetSupport));
      NCCLCHECK(ncclGetRegBuff(comm, headTask, &input.regBuff));
      struct ncclTuningResult_t bestTuning = NCCL_TUNING_RESULT_INIT;
      NOWARN(ncclTuningCompute(&input, &bestTuning), NCCL_TUNING);
      kernelId = (ncclSymkKernelId)bestTuning.symKernelId;
      nChannels = bestTuning.nChannels;
      nWarps = bestTuning.nWarps;
      task = headTask;
      // Hard-error only when the selection was symmetric-only (no general algorithm to fall
      // back to) and force is on; otherwise let the legacy fallback below run.
      if (kernelId == ncclSymkKernelId_Count && effAlgMask != 0 && (effAlgMask & NCCL_TUNING_MASK_SYM_KERNELS) != 0 &&
          (effAlgMask & NCCL_TUNING_MASK_GENERAL_KERNELS) == 0 && headTask->forceAlgSelection) {
        WARN("algSelection names only symmetric kernel(s) that are unavailable for %s",
             ncclFuncToString(headTask->func));
        return ncclInvalidArgument;
      }
      if (effAlgMask != 0 && kernelId != ncclSymkKernelId_Count) {
        INFO(NCCL_TUNING, "algSelection: %s picked within the selected set", ncclAlgNameForSymk(kernelId));
      }
      // Override needFallback when buffers are registered but VAs contain sysmem segments.
      // The below functions return false when the window is NULL, so this covers non-reg cases as well.
      if (kernelId == ncclSymkKernelId_Count || ncclDevrWindowHasSysmemSegment(headTask->sendWin) ||
          ncclDevrWindowHasSysmemSegment(headTask->recvWin)) {
        // cannot find appropriate symmetric kernel for the tasks
        // fallback to legacy kernels
        while (task != nullptr) {
          struct ncclTaskColl* next = task->next;
          int isSymLast = task->isSymLast;
          if (*remainTasksHead) {
            remainTasksTail->next = task;
            remainTasksTail = task;
          } else {
            *remainTasksHead = remainTasksTail = task;
          }
          planner->nTasksColl++;
          task = next;
          if (isSymLast) break;
        }
        continue;
      }

      // Specialized LL kernels don't require registered user windows but they still need an initialized devComm.
      ncclSymkKernelMask const symkLLKernelMask = ncclSymkLLKernelMask() & ~ncclGenkKernelMask();
      if (((1ull << kernelId) & symkLLKernelMask) && headTask->winRegType == ncclSymSendNonregRecvNonreg) {
        NCCLCHECK(ncclSymkInitOnce(comm));
      }
      if ((1ull << kernelId) & ncclGenkKernelMask()) {
        NCCLCHECK(ncclGenkInitStart(comm, needGenkDevComm));
      }

      // set all symmetric tasks to the same kernel
      while (task != nullptr) {
        struct ncclTaskColl* next = task->next;
        int isSymLast = task->isSymLast;
        task->devFuncId = (uint32_t)kernelId;
        task->nMaxChannels = nChannels;
        task->nWarps = nWarps;
        task->minChunkPayloadBytes = bestTuning.minChunkPayloadBytes;
        convertSymTaskDevOp(comm, task);
        ncclIntruQueueEnqueue(&planner->collSymTaskQueue, task);
        task = next;
        if (isSymLast) break;
      }
    }
  }

  return ret;
}

template <bool Genk, typename Args, typename KernelComm>
static ncclResult_t ncclSymmetricTaskSchedulerImpl(
  struct ncclComm* comm, struct ncclIntruQueue<struct ncclTaskColl, &ncclTaskColl::next>* symTaskQueue,
  struct ncclKernelPlan* plan, KernelComm const& kcomm) {
  struct ncclTaskColl* headTask = ncclIntruQueueHead(symTaskQueue);
  int devFuncId = headTask->devFuncId;
  struct ncclTaskColl* task = NULL;
  ssize_t totalCount = 0;  // aligned bytes
  ssize_t logCount = 0;
  ssize_t remainCell = 0;
  ssize_t cellPerChannel = 0;
  int workCount = 0, workIndex = 0;
  size_t cellCount = NCCL_SYM_KERNEL_CELL_SIZE / ncclTypeSize(headTask->datatype); // minimal cell size
  ncclResult_t ret = ncclSuccess;
  int curChannel = 0;
  int curChannelWork = 0;
  int nMaxChannels = headTask->nMaxChannels;
  struct ncclSymkDevWork* workBufPtr = NULL;
  struct ncclSymkChannelWorkRange* workRangePtr = NULL;
  const char* funcName = ncclFuncToString(headTask->func);
  const char* kernelName = ncclSymkKernelIdToString(headTask->devFuncId);
  Args* argsBuf = NULL;

  plan->isSymColl = true;
  plan->threadPerBlock = headTask->nWarps * WARP_SIZE;
  plan->launchCompletionEvent = headTask->launchCompletionEvent;
  plan->hasProxyOps = false;
  ncclSymkKernelId kernelId = (ncclSymkKernelId)headTask->devFuncId;
  int kernelIndex = Genk ? ncclGenkGetKernelIndex(kernelId, headTask->opDev.op, headTask->datatype) :
                           ncclSymkGetKernelIndex(kernelId, headTask->opDev.op, headTask->datatype);
  if (kernelIndex < 0) {
    WARN("No %s kernel %s for %s with red op %d datatype %d", Genk ? "Genk" : "Symk", kernelName, funcName,
         (int)headTask->opDev.op, (int)headTask->datatype);
    return ncclInternalError;
  }
  void** kernelList = Genk ? ncclGenkKernelList : ncclSymkKernelList;
  void** kernelListProfile = Genk ? ncclGenkKernelListProfile : ncclSymkKernelListProfile;
  int* kernelMaxDynamicSmem = Genk ? ncclGenkKernelMaxDynamicSmem : ncclSymkKernelMaxDynamicSmem;
  // Profiling requested = plugin loaded and mask has ncclProfileKernelCh. Set
  // hasProfilerOps (like non-sym plans) so the host callback fires the group/coll events.
  bool profilingRequested = ncclProfilerPluginLoaded() && (headTask->eActivationMask & ncclProfileKernelCh);
  plan->hasProfilerOps = profilingRequested;
  // Device-side instrumentation is eager-only: the args buffer is snapshotted at
  // cuLaunchKernel, so the mirrored counter can't advance across graph replays. Under
  // capture we launch the clean kernel and keep only host-side group/coll events.
  bool profilerEnabled = profilingRequested && !plan->persistent;
  plan->kernelFn = (profilerEnabled && kernelListProfile[kernelIndex] != nullptr) ? kernelListProfile[kernelIndex] :
                                                                                    kernelList[kernelIndex];
  int maxDynamicSmem = kernelMaxDynamicSmem[kernelIndex];
  plan->kernelDynSmem = (1 & ncclSymkDynamicSmemKernelMask() >> (int)kernelId) ? maxDynamicSmem : 0;
  task = headTask;
  while (task != nullptr && task->devFuncId == devFuncId) {
    workCount++;
    totalCount += alignUp(task->count, cellCount);
    logCount += task->count;
    // per-coll cgaClusterSize is applied to the plan. User should use consistent cgaClusterSize in a Group.
    if (task->cgaClusterSize != NCCL_CONFIG_UNDEF_INT) plan->cgaClusterSize = task->cgaClusterSize;
    if (task->isSymLast == 1) break;
    task = task->next;
  }

  plan->kernelArgsSize = ncclSymkDevWorkArgsSize<Args>(nMaxChannels, workCount, profilerEnabled);
  argsBuf = (Args*)calloc(1, plan->kernelArgsSize);

  argsBuf->nMaxChannels = nMaxChannels;
  argsBuf->maxDynamicSmem = maxDynamicSmem;
  // KernelCh-only launches leave the phase bit clear, so the profile variant skips the
  // phase stamps and their fence at runtime.
  argsBuf->profilerMode =
    profilerEnabled ? ncclProfilerDeviceMode(headTask->eActivationMask) : (uint8_t)ncclDevProfilerModeNone;
  setMinChunkPayloadBytes(argsBuf, headTask->minChunkPayloadBytes);

  remainCell = cellPerChannel = DIVUP(DIVUP(totalCount, nMaxChannels), cellCount);
  workRangePtr = ncclSymkGetWorkRange(argsBuf);
  workBufPtr = ncclSymkGetWorks(argsBuf, nMaxChannels);

  while (!ncclIntruQueueEmpty(symTaskQueue)) {
    struct ncclSymkDevWork devWork = {};
    size_t cellLeft = 0, taskCell = 0;
    uint8_t isSymLast = 0;

    if (ncclIntruQueueHead(symTaskQueue)->devFuncId != devFuncId) break; // scheduling is done

    task = ncclIntruQueueDequeue(symTaskQueue);
    isSymLast = task->isSymLast;

    NCCLCHECKGOTO(ncclSymkMakeDevWork(comm, task, &devWork), ret, fail);

    cellLeft = taskCell = DIVUP(task->count, cellCount);
    for (; curChannel < nMaxChannels;) {
      workRangePtr[curChannel].workHi = workIndex;
      if (curChannelWork == 0) {
        if (devWork.nChannels == 0) {
          devWork.sChannelId = curChannel;
          devWork.nChannels = 1;
        } else if (cellLeft <= remainCell) {
          // the last segment of the task
          // if the remaining cell is less than 1024 bytes, we can fuse the last channel
          if ((remainCell - cellLeft) * NCCL_SYM_KERNEL_CELL_SIZE <= (1 << 10) || ncclIntruQueueEmpty(symTaskQueue)) {
            devWork.nChannels++;
          }
        } else {
          // middle segment of the task
          devWork.nChannels++;
        }
      } else {
        if (cellLeft != taskCell) {
          WARN("Symmetric task cell count %zu does not match remaining cell count %zu", taskCell, cellLeft);
          ret = ncclInternalError;
          goto fail;
        }
        if (taskCell <= remainCell) {
          // the first segment of the task is fully scheduled onto the channel
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
        // cellLeft > remainCell; the task is partially scheduled onto the channel
        cellLeft -= remainCell;
        workRangePtr[curChannel].fracHi = uint16_t(DIVUP(0x10000L * (taskCell - cellLeft), taskCell) - 1);
        remainCell = cellPerChannel;
        curChannel++;
        curChannelWork = 0;
      }
    }
    memcpy(workBufPtr + workIndex, &devWork, sizeof(struct ncclSymkDevWork));
    workIndex++;

    // Profiler: preserve task for profiler event firing in hostStreamPlanTask
    plan->groupApiEventHandle = task->groupApiEventHandle;
    ncclIntruQueueEnqueue(&plan->collTaskQueue, task);

    if (isSymLast == 1) break;
    if (curChannel == nMaxChannels) {
      WARN("ncclSymmetricTaskScheduler ran out of channel space (nMaxChannels=%d, workCount=%d, workIndex=%d)",
           nMaxChannels, workCount, workIndex);
      goto fail;
    }
  }
  if (remainCell < cellPerChannel) curChannel++;
  // At this point, curChannel indexes the first _empty_ channel.

  memcpy(&argsBuf->kcomm, &kcomm, sizeof(kcomm));
  plan->workBytes = totalCount * ncclTypeSize(headTask->datatype);
  // curChannel == 0 is not expected here (the caller ensures symTaskQueue is
  // non-empty), but guard it anyway to avoid the undefined behavior of shifting
  // a 64-bit value by 64 bits (Coverity BAD_SHIFT).
  plan->channelMask = curChannel == 0 ? 0 : (uint64_t(-1) >> (64 - curChannel));
  plan->kernelSymArgs = (void*)argsBuf;
  plan->workStorageType = ncclDevWorkStorageTypeArgs;

  if (comm->rank == 0) {
    char cgaCSBuf[30];
    // Append the cgaClusterSize info only if the value is non-default (not 4).
    if (plan->cgaClusterSize != 4 && ncclDebugShouldLog(NCCL_LOG_INFO, NCCL_TUNING, ncclDebugMask)) {
      snprintf(cgaCSBuf, sizeof(cgaCSBuf), " cgaClusterSize %d", plan->cgaClusterSize);
    } else {
      cgaCSBuf[0] = '\0';
    }
    INFO(NCCL_TUNING, "%s [Symmetric]: %ld Bytes -> Kernel %s nchannels %d nthreads %d nWorks %d%s", funcName,
         logCount * ncclTypeSize(headTask->datatype), kernelName, curChannel, plan->threadPerBlock, workCount,
         cgaCSBuf);
  }

exit:
  return ret;
fail:
  goto exit;
}

ncclResult_t ncclSymmetricTaskScheduler(struct ncclComm* comm,
                                        struct ncclIntruQueue<struct ncclTaskColl, &ncclTaskColl::next>* symTaskQueue,
                                        struct ncclKernelPlan* plan) {
  struct ncclTaskColl* headTask = ncclIntruQueueHead(symTaskQueue);
  ncclSymkKernelId const kernelId = (ncclSymkKernelId)headTask->devFuncId;
  if ((1ull << kernelId) & ncclGenkKernelMask()) {
    NCCLCHECK(ncclGenkInitEnd(comm));
    return ncclSymmetricTaskSchedulerImpl<true, ncclGenkDevWorkArgs>(comm, symTaskQueue, plan,
                                                                     comm->symkState.genkComm);
  }
  return ncclSymmetricTaskSchedulerImpl<false, ncclSymkDevWorkArgs>(comm, symTaskQueue, plan, comm->symkState.kcomm);
}
#endif // NCCL_SYMMETRIC_SCHED_H_
