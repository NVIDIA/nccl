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
#include <cuda_fp16.h>
#if defined(__CUDA_FP8_TYPES_EXIST__)
#include <cuda_fp8.h>
#endif

extern int64_t ncclParamSingleProcMemRegEnable();

ncclDevRedOp_t symkRedOp(ncclRedOp_t redOp, ncclDevRedOp_t devRedOp) {
  if (redOp == ncclAvg) {
    return ncclDevSumPostDiv;
  }
  return devRedOp;
}

void convertCollTaskToSymmetricTask(struct ncclComm* comm, struct ncclTaskColl* task) {
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

ncclResult_t ncclMakeSymmetricTaskList(struct ncclComm* comm, struct ncclTaskColl* task,
                                       struct ncclIntruQueue<struct ncclTaskColl, &ncclTaskColl::next>* symTaskQueue,
                                       struct ncclTaskColl** remainTasksHead) {
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

    if (symAvailable) {
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
  if (!foundSymm) goto exit;

  // make sure kernel args space can hold at least a single work
  assert(comm->workArgsBytes >= ncclSymkDevWorkArgs::calcArgsSize(MAXCHANNELS, 1));

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
        if (count > countMax) countMax = count;
        if (ncclSymkDevWorkArgs::calcArgsSize(MAXCHANNELS, nWorks + 1) > comm->workArgsBytes || task->next == nullptr) {
          task->isSymLast = 1;
          break;
        }
        task = task->next;
      }
      struct ncclTuningInput_t input;
      input.comm = comm;
      input.tuningMask = NCCL_TUNING_MASK_SYM_KERNELS;
      input.func = headTask->func;
      input.redOp = headTask->opHost;
      input.devRedOp = symkOp;
      input.datatype = headTask->datatype;
      input.nBytes = countTotal * ncclTypeSize(headTask->datatype);
      input.numPipeOps = 0;
      input.count = headTask->count;
      input.countMax = countMax;
      input.nWorks = nWorks;
      input.winRegType = headTask->winRegType;
      NCCLCHECK(ncclGetCollNetSupport(comm, headTask, &input.collNetSupport));
      NCCLCHECK(ncclGetRegBuff(comm, headTask, &input.regBuff));
      struct ncclTuningResult_t bestTuning = NCCL_TUNING_RESULT_INIT;
      NCCLCHECK(ncclTuningCompute(&input, &bestTuning));
      kernelId = (ncclSymkKernelId)bestTuning.symKernelId;
      nChannels = bestTuning.nChannels;
      nWarps = bestTuning.nWarps;
      task = headTask;
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

      // initialize symmetric objects for LL kernels
      if (((1 << kernelId) & ncclSymkLLKernelMask()) && headTask->winRegType == ncclSymSendNonregRecvNonreg) {
        NCCLCHECK(ncclSymkInitOnce(comm));
      }

      // set all symmetric tasks to the same kernel
      while (task != nullptr) {
        struct ncclTaskColl* next = task->next;
        int isSymLast = task->isSymLast;
        task->devFuncId = (uint32_t)kernelId;
        task->nMaxChannels = nChannels;
        task->nWarps = nWarps;
        convertCollTaskToSymmetricTask(comm, task);
        ncclIntruQueueEnqueue(&planner->collSymTaskQueue, task);
        task = next;
        if (isSymLast) break;
      }
    }
  }

exit:
  return ret;
}

ncclResult_t ncclSymmetricTaskScheduler(struct ncclComm* comm,
                                        struct ncclIntruQueue<struct ncclTaskColl, &ncclTaskColl::next>* symTaskQueue,
                                        struct ncclKernelPlan* plan) {
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
  struct ncclSymkDevWorkArgs* argsBuf = NULL;

  plan->isSymColl = true;
  plan->threadPerBlock = headTask->nWarps * WARP_SIZE;
  plan->hasProxyOps = false;
  ncclSymkKernelId kernelId = (ncclSymkKernelId)headTask->devFuncId;
  int kernelIndex = ncclSymkGetKernelIndex(kernelId, headTask->opDev.op, headTask->datatype);
  plan->kernelFn = ncclSymkKernelList[kernelIndex];
  int maxDynamicSmem = ncclSymkKernelMaxDynamicSmem[kernelIndex];
  plan->kernelDynSmem = (1 & ncclSymkDynamicSmemKernelMask() >> (int)kernelId) ? maxDynamicSmem : 0;
  task = headTask;
  while (task != nullptr && task->devFuncId == devFuncId) {
    workCount++;
    totalCount += alignUp(task->count, cellCount);
    logCount += task->count;
    if (task->isSymLast == 1) break;
    task = task->next;
  }

  plan->kernelArgsSize = ncclSymkDevWorkArgs::calcArgsSize(nMaxChannels, workCount);
  argsBuf = (struct ncclSymkDevWorkArgs*)calloc(1, plan->kernelArgsSize);

  remainCell = cellPerChannel = DIVUP(DIVUP(totalCount, nMaxChannels), cellCount);
  workRangePtr = argsBuf->getWorkRange();
  workBufPtr = argsBuf->getWorks(nMaxChannels);
  argsBuf->nMaxChannels = nMaxChannels;
  argsBuf->maxDynamicSmem = maxDynamicSmem;

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
          assert(devWork.nChannels > 0);
          // if the remaining cell is less than 1024 bytes, we can fuse the last channel
          if ((remainCell - cellLeft) * NCCL_SYM_KERNEL_CELL_SIZE <= (1 << 10) || ncclIntruQueueEmpty(symTaskQueue)) {
            devWork.nChannels++;
          }
        } else {
          // middle segment of the task
          devWork.nChannels++;
        }
      } else {
        assert(cellLeft == taskCell);
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

    // Profiler
    plan->groupApiEventHandle = task->groupApiEventHandle;

    ncclMemoryPoolFree<struct ncclTaskColl>(&comm->memPool_ncclTaskColl, task);
    if (isSymLast == 1) break;
    if (curChannel == nMaxChannels) {
      WARN("ncclSymmetricTaskScheduler ran out of channel space (nMaxChannels=%d, workCount=%d, workIndex=%d)",
           nMaxChannels, workCount, workIndex);
      goto fail;
    }
  }
  if (remainCell < cellPerChannel) curChannel++;

  memcpy(&argsBuf->kcomm, &comm->symkState.kcomm, sizeof(comm->symkState.kcomm));
  plan->workBytes = totalCount * ncclTypeSize(headTask->datatype);
  plan->channelMask = uint64_t(-1) >> (64 - curChannel);
  plan->kernelSymArgs = (void*)argsBuf;
  plan->workStorageType = ncclDevWorkStorageTypeArgs;

  if (comm->rank == 0) {
    INFO(NCCL_TUNING, "%s [Symmetric]: %ld Bytes -> Kernel %s nchannels %d nthreads %d nWorks %d", funcName,
         logCount * ncclTypeSize(headTask->datatype), kernelName, curChannel, plan->threadPerBlock, workCount);
  }

exit:
  return ret;
fail:
  goto exit;
}
#endif // NCCL_SYMMETRIC_SCHED_H_
