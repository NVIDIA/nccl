/*************************************************************************
 * Copyright (c) 2015-2025, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_SYMMETRIC_SCHED_H_
#define NCCL_SYMMETRIC_SCHED_H_

#include "scheduler.h"

ncclResult_t ncclMakeSymmetricTaskList(struct ncclComm* comm, struct ncclTaskColl* task, struct ncclIntruQueue<struct ncclTaskColl, &ncclTaskColl::next>* symTaskQueue, struct ncclTaskColl** remainTasksHead) {
  ncclResult_t ret = ncclSuccess;
  int fnOpTySymCount = 0;
  struct ncclTaskColl* tasksSymByFnOpTy[ncclNumFuncs * ncclNumDevRedOps * ncclNumTypes];
  int fnOpTySymIndices[ncclNumFuncs * ncclNumDevRedOps * ncclNumTypes];
  struct ncclKernelPlanner* planner = &comm->planner;
  struct ncclTaskColl* remainTasksTail = nullptr;
  bool foundSymm = false;

  memset(tasksSymByFnOpTy, 0, sizeof(tasksSymByFnOpTy));
  *remainTasksHead = nullptr;
  while (task != nullptr) {
    int index = ((int)task->func*ncclNumDevRedOps + (int)task->opDev.op)*ncclNumTypes + (int)task->datatype;
    struct ncclTaskColl* next = task->next;
    NCCLCHECK(ncclDevrFindWindow(comm, task->sendbuff, &task->sendWin));
    NCCLCHECK(ncclDevrFindWindow(comm, task->recvbuff, &task->recvWin));
    bool symAvailable = ncclSymkAvailable(comm, task->func, task->opDev.op, task->datatype, task->count);

    if (task->sendWin && task->recvWin && (task->sendWin->winFlags & task->recvWin->winFlags & NCCL_WIN_COLL_SYMMETRIC) && symAvailable) {
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
      float estTimeUs = 1.e18;
      size_t countTotal = 0, countMax = 0;
      struct ncclTaskColl* headTask = task;
      size_t cellCount = NCCL_SYM_KERNEL_CELL_SIZE / ncclTypeSize(headTask->datatype);
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
      NCCLCHECK(ncclSymkPickKernel(comm, headTask->func, headTask->opDev.op, headTask->datatype,
                                   countTotal, countMax, nWorks,
                                   &estTimeUs, &kernelId, &nChannels, &nWarps));
      if (kernelId == ncclSymkKernelId_Count) {
        char const* name = ncclGetEnv("NCCL_SYM_KERNEL");
        WARN("Error: no symmetric kernel available for function %s.%s%s",
             ncclFuncToString(headTask->func), (name ? " NCCL_SYM_KERNEL was set to " : ""), (name ? name: ""));
        ret = (name ? ncclInvalidUsage : ncclInternalError);
        goto fail;
      }
      // set all symmetric tasks to the same kernel
      task = headTask;
      while (task != nullptr) {
        struct ncclTaskColl* next = task->next;
        int isSymLast = task->isSymLast;
        task->devFuncId = (uint32_t)kernelId;
        task->nMaxChannels = nChannels;
        task->nWarps = nWarps;
        ncclIntruQueueEnqueue(&planner->collSymTaskQueue, task);
        task = next;
        if (isSymLast) break;
      }
    }
  }

exit:
  return ret;
fail:
  goto exit;
}

ncclResult_t ncclSymmetricTaskScheduler(struct ncclComm* comm, struct ncclIntruQueue<struct ncclTaskColl, &ncclTaskColl::next>* symTaskQueue, struct ncclKernelPlan* plan) {
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
  plan->kernelFn = ncclSymkGetKernelPtr((ncclSymkKernelId)headTask->devFuncId, headTask->opDev.op, headTask->datatype);
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

  while (!ncclIntruQueueEmpty(symTaskQueue)) {
    struct ncclSymkDevWork devWork = {};
    size_t cellLeft = 0, taskCell = 0;
    uint8_t isSymLast = 0;

    if (ncclIntruQueueHead(symTaskQueue)->devFuncId != devFuncId) break; // scheduling is done

    task = ncclIntruQueueDequeue(symTaskQueue);
    isSymLast = task->isSymLast;

    NCCLCHECKGOTO(ncclSymkMakeDevWork(comm, task, &devWork), ret, fail);

    cellLeft = taskCell = DIVUP(task->count, cellCount);
    for (;curChannel < nMaxChannels;) {
      workRangePtr[curChannel].workHi = workIndex;
      if (curChannelWork == 0) {
        if (devWork.nChannels == 0) {
          devWork.sChannelId = curChannel;
          devWork.nChannels = 1;
        } else if (cellLeft <= remainCell) {
          // the last segment of the task
          assert(devWork.nChannels > 0);
          // if the remaining cell is less than 1024 bytes, we can fuse the last channel
          if ((remainCell - cellLeft) * NCCL_SYM_KERNEL_CELL_SIZE <= (1 << 10) || ncclIntruQueueEmpty(symTaskQueue)) devWork.nChannels++;
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
