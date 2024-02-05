/*************************************************************************
 * Copyright (c) 2015-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "group.h"
#include "debug.h"
#include "enqueue.h"
#include "transport.h"
#include "channel.h"
#include <assert.h>

__thread int ncclGroupDepth = 0; // depth of ncclGroupStart nesting
__thread ncclResult_t ncclGroupError = ncclSuccess;
__thread struct ncclComm* ncclGroupCommHead = nullptr;
__thread struct ncclComm* ncclGroupCommPreconnectHead = nullptr;
__thread struct ncclIntruQueue<struct ncclAsyncJob, &ncclAsyncJob::next> ncclAsyncJobs;
__thread struct ncclGroupJob *ncclGroupJobMainPtr = NULL;
__thread struct ncclGroupJob ncclGroupJobMain;
__thread int ncclGroupBlocking = -1; /* default mode */
__thread bool ncclGroupJobAbortFlag = false;

void* ncclAsyncJobMain(void* arg);

ncclResult_t ncclAsyncLaunch(
    struct ncclAsyncJob* job,
    ncclResult_t(*func)(struct ncclAsyncJob*),
    void(*undo)(struct ncclAsyncJob*),
    void(*destructor)(void*), ncclComm_t comm
  ) {
  ncclResult_t ret = ncclSuccess;

  if (ncclGroupDepth == 0) {
    ret = func(job);
    if (ret != ncclSuccess && undo) undo(job);
    if (destructor) destructor(job);
  } else {
    job->func = func;
    job->undo = undo;
    job->destructor = destructor;
    job->abortFlag = comm->abortFlag;
    job->childAbortFlag = comm->childAbortFlag;
    job->state = ncclGroupJobRunning;
    job->comm = comm;
    /* check if there are blocking and nonblocking comms at the same time in group. */
    if (ncclGroupBlocking == -1) {
      /* first met communicator */
      ncclGroupBlocking = comm->config.blocking;
    } else if (ncclGroupBlocking != comm->config.blocking) {
      WARN("Blocking and nonblocking communicators are not allowed in the same group.");
      ret = ncclInvalidArgument;
    }
    ncclIntruQueueEnqueue(&ncclAsyncJobs, job);
  }

  return ret;
}

void* ncclAsyncJobMain(void* arg) {
  struct ncclAsyncJob* job = (struct ncclAsyncJob*)arg;
  job->result = job->func(job);
  if (job->result != ncclSuccess) {
    INFO(NCCL_INIT,"%s:%d -> %d [Async thread]", __FILE__, __LINE__, job->result);
  }
  __atomic_store_n(&job->state, ncclGroupJobDone, __ATOMIC_RELEASE);
  return arg;
}

ncclResult_t ncclAsyncJobComplete(struct ncclAsyncJob* job) {
  ncclResult_t ret;
  SYSCHECK(pthread_join(job->thread, NULL), "pthread_join");
  if (job->result != ncclSuccess) {
    WARN("ncclAsyncJobComplete: job %p failed, job error %d", job, job->result);
  }
  ret = job->result;
  if (job->destructor) job->destructor((void*)job);
  return ret;
}

NCCL_API(ncclResult_t, ncclGroupStart);
ncclResult_t ncclGroupStart() {
  ncclResult_t ret = ncclSuccess;
  NVTX3_FUNC_RANGE_IN(nccl_domain);

  NCCLCHECK(ncclGroupStartInternal());
  TRACE_CALL("ncclGroupStart()");
  return ret;
}

NCCL_API(ncclResult_t, ncclGroupEnd);
ncclResult_t ncclGroupEnd() {
  ncclResult_t ret = ncclSuccess;
  NVTX3_FUNC_RANGE_IN(nccl_domain);
  NCCLCHECKGOTO(ncclGroupEndInternal(), ret, exit);
  TRACE_CALL("ncclGroupEnd()");
exit:
  return ret;
}

struct ncclPreconnectJob {
  struct ncclAsyncJob base;
  struct ncclComm* comm;
};
ncclResult_t ncclPreconnectFunc(struct ncclAsyncJob* job_) {
  struct ncclPreconnectJob* job = (struct ncclPreconnectJob*)job_;
  struct ncclComm* comm = job->comm;
  CUDACHECK(cudaSetDevice(comm->cudaDev));
  if (CPU_COUNT(&comm->cpuAffinity)) sched_setaffinity(0, sizeof(cpu_set_t), &comm->cpuAffinity);
  NCCLCHECK(ncclTransportP2pSetup(comm, NULL, 1));
  return ncclSuccess;
}

static ncclResult_t doLaunches(struct ncclComm* head) {
  ncclResult_t result = ncclSuccess;
  struct ncclComm* cliqueComm0 = head->intraComm0;
  struct ncclComm* cliqueHead = head;
  struct ncclComm* cliqueNextHead;
  bool useBarrier = ncclParamLaunchMode == ncclLaunchModeGroup;
  // This outer loop iterates over cliques of comms which are siblings of the
  // same global entity. We calculate a clique as all comms which have the same
  // `intraComm0` value.
  do {
    struct ncclComm* comm = cliqueHead;
    bool capturingYes = false, capturingNo = false;
    do {
      (ncclCudaGraphValid(comm->tasks.capturingGraph) ? capturingYes : capturingNo) = true;
      CUDACHECKGOTO(cudaSetDevice(comm->cudaDev), result, failure);
      NCCLCHECKGOTO(ncclLaunchPrepare(comm), result, failure);
      if (useBarrier) ncclCommIntraBarrierIn(comm, 1);
      comm = comm->groupNext;
    } while (comm != nullptr && comm->intraComm0 == cliqueComm0);
    cliqueNextHead = comm;

    if (capturingYes && capturingNo) {
      // We have entered barriers but are aborting without leaving them. Thus
      // these comms are permanently trashed. We need a good mechanism for
      // tracking and reporting that.
      WARN("Either none or all communicators in a ncclGroup() can be CUDA graph captured.");
      result = ncclInvalidUsage;
      goto failure;
    }

    while (true) { // Iterate rounds of launches for clique.
      bool moreRounds;
      comm = cliqueHead;
      do { // Iterate clique members.
        struct ncclComm* next = comm->groupNext;
        if (useBarrier) {
          // Barrier reduction result tells us if this was the final round.
          moreRounds = 0 != ncclCommIntraBarrierOut(comm);
        } else {
          moreRounds = comm->unlaunchedPlansHead != nullptr;
        }
        if (moreRounds) {
          // Pop next unlaunched kernel
          struct ncclKernelPlan* plan = comm->unlaunchedPlansHead;
          if (plan != nullptr) {
            comm->unlaunchedPlansHead = plan->next;
            CUDACHECKGOTO(cudaSetDevice(comm->cudaDev), result, failure);
            NCCLCHECKGOTO(ncclLaunchKernelBefore_NoUncapturedCuda(comm, plan), result, failure);
            NCCLCHECKGOTO(ncclLaunchKernel(comm, plan), result, failure);
          }
          // Barrier reduction input indicates if we require further rounds.
          if (useBarrier) ncclCommIntraBarrierIn(comm, comm->unlaunchedPlansHead != nullptr ? 1 : 0);
          if (plan != nullptr) {
            NCCLCHECKGOTO(ncclLaunchKernelAfter_NoCuda(comm, plan), result, failure);
          }
        } else { // Final round.
          CUDACHECKGOTO(cudaSetDevice(comm->cudaDev), result, failure);
          NCCLCHECKGOTO(ncclLaunchFinish(comm), result, failure);
        }
        comm = next;
      } while (comm != cliqueNextHead);
      if (!moreRounds) break;
    }
    cliqueHead = cliqueNextHead;
  } while (cliqueHead != nullptr);
failure:
  return result;
}

static inline void groupResetJobState(struct ncclGroupJob* job) {
  if (job) {
    if (job->groupBlockingPtr) *job->groupBlockingPtr = -1;
    if (job->abortFlagPtr) *job->abortFlagPtr = false;
    if (job->groupErrorPtr) *job->groupErrorPtr = ncclSuccess;
    if (job->groupCommHeadPtr) *job->groupCommHeadPtr = NULL;
    if (job->groupCommPreconnectHeadPtr) *job->groupCommPreconnectHeadPtr = NULL;
    memset(job, 0, sizeof(struct ncclGroupJob));
  }
  return;
}

static void groupCleanup(struct ncclComm** groupCommHeadPtr, struct ncclComm** groupCommPreconnectHeadPtr, struct ncclIntruQueue<struct ncclAsyncJob, &ncclAsyncJob::next>* asyncJobsPtr, ncclResult_t* groupErrorPtr, int* groupBlockingPtr, volatile bool* groupJobAbortFlagPtr, ncclResult_t error) {
  struct ncclComm* comm = *groupCommHeadPtr;

  /* reset all thread local variables */
  *groupCommHeadPtr = NULL;
  *groupCommPreconnectHeadPtr = NULL;
  *groupErrorPtr = ncclSuccess;
  *groupBlockingPtr = -1;
  *groupJobAbortFlagPtr = false;

  while (comm != nullptr) {
    struct ncclComm* next = comm->groupNext;
    (void) ncclGroupCommLeave(comm); // overwrites comm->groupNext
    // We don't know if preconnect succeeded or happened at all, so clear
    // the flags that let `taskAppend()` skip over checking if preconnect
    // is needed.
    comm->preconnectNext = reinterpret_cast<struct ncclComm*>(0x1);
    for (int i = 0; i < comm->nRanks; i++) {
      comm->tasks.peers[i].sendSeen = false;
      comm->tasks.peers[i].recvSeen = false;
      comm->connectSend[i] = 0UL;
      comm->connectRecv[i] = 0UL;
    }
    comm->unlaunchedPlansHead = nullptr;
    // Reclaim abandoned kernel plan memory. Note ncclWork structs were already
    // reclaimed by a `ncclMemoryStackPop(&comm->memScoped)` during `ncclGroupCommLeave()`.
    while (!ncclIntruQueueEmpty(&comm->planQueue)) {
      struct ncclKernelPlan* plan = ncclIntruQueueDequeue(&comm->planQueue);
      // Persistent plans will be reclaimed via the callbackQueue when the
      // graph drops its UserObject reference.
      if (!plan->persistent) {
        for (int c = 0; c < MAXCHANNELS; c++) {
          while (!ncclIntruQueueEmpty(&plan->channels[c].proxyOpQueue)) {
            struct ncclProxyOp* pxop = ncclIntruQueueDequeue(&plan->channels[c].proxyOpQueue);
            ncclMemoryPoolFree(&comm->memPool_ncclProxyOp, pxop);
          }
        }
        ncclMemoryPoolFree(&comm->memPool_ncclKernelPlan, plan);
      }
    }
    // Reset comm->tasks to empty.
    comm->tasks.nTasksColl = 0;
    comm->tasks.nTasksP2p = 0;
    comm->tasks.workBytesTotal = 0;
    comm->tasks.streams = nullptr;
    ncclIntruQueueConstruct(&comm->tasks.collQueue);
    for (int i = 0; i < comm->nRanks; i++) {
      ncclIntruQueueConstruct(&comm->tasks.peers[i].sendQueue);
      ncclIntruQueueConstruct(&comm->tasks.peers[i].recvQueue);
    }

    if (!comm->config.blocking)
      (void) ncclCommSetAsyncError(comm, error);
    comm = next;
  }

  /* reset everything */
  while (!ncclIntruQueueEmpty(asyncJobsPtr)) {
    struct ncclAsyncJob* job = ncclIntruQueueDequeue(asyncJobsPtr);
    if (job->comm && !job->comm->config.blocking)
      (void) ncclCommSetAsyncError(job->comm, error);
    if (job->undo) job->undo(job);
    if (job->destructor) job->destructor((void*)job);
  }

  return;
}

static ncclResult_t groupLaunch(struct ncclAsyncJob *job_) {
  int savedDev;
  ncclResult_t ret = ncclSuccess;
  bool jobsDone = false;
  bool errorJobAbortFlag = false;
  struct ncclGroupJob *gjob = (struct ncclGroupJob*) job_;
  struct ncclComm *groupCommHeadMain = *gjob->groupCommHeadPtr;
  struct ncclComm *groupCommPreconnectHeadMain = *gjob->groupCommPreconnectHeadPtr;
  struct ncclIntruQueue<struct ncclAsyncJob, &ncclAsyncJob::next> *asyncJobsMain = gjob->asyncJobsPtr;
  volatile bool *groupAbortFlag = gjob->abortFlagPtr;

  CUDACHECKGOTO(cudaGetDevice(&savedDev), ret, fail);

  if (groupCommPreconnectHeadMain != nullptr) {
    struct ncclComm* comm = groupCommPreconnectHeadMain;
    do {
      struct ncclPreconnectJob* job;
      NCCLCHECKGOTO(ncclCalloc(&job, 1), ret, fail);
      job->base.func = ncclPreconnectFunc;
      job->base.undo = nullptr;
      job->base.destructor = free;
      job->base.state = ncclGroupJobRunning;
      job->base.abortFlag = comm->abortFlag;
      job->comm = comm;
      ncclIntruQueueEnqueue(asyncJobsMain, &job->base);

      struct ncclComm* next = comm->preconnectNext;
      comm->preconnectNext = reinterpret_cast<struct ncclComm*>(0x1);
      comm = next;
    } while (comm != nullptr);
  }

  if (!ncclIntruQueueEmpty(asyncJobsMain)) {
    struct ncclAsyncJob* job = ncclIntruQueueHead(asyncJobsMain);
    do {
      SYSCHECKGOTO(pthread_create(&job->thread, nullptr, ncclAsyncJobMain, job), ret, fail);
      job = job->next;
    } while (job != nullptr);

    do {
      jobsDone = true;
      job = ncclIntruQueueHead(asyncJobsMain);
      do {
        ncclGroupJobState_t state = __atomic_load_n(&job->state, __ATOMIC_ACQUIRE);
        if (state == ncclGroupJobRunning) {
          jobsDone = false;
        } else if (state == ncclGroupJobDone) {
          if (pthread_join(job->thread, nullptr) != 0) {
            WARN("Error waiting for pthread_join : %s", strerror(errno));
            ret = ncclSystemError;
          }
          job->state = ncclGroupJobJoined;
          if (job->result != ncclSuccess && ret == ncclSuccess) {
            ret = job->result;
            errorJobAbortFlag = true;
          }
        } else {
          /* safety check */
          assert(state == ncclGroupJobJoined);
        }

        if (__atomic_load_n(groupAbortFlag, __ATOMIC_RELAXED) || errorJobAbortFlag == true) {
          __atomic_store_n(job->abortFlag, 1, __ATOMIC_RELAXED);
          if (job->childAbortFlag) __atomic_store_n(job->childAbortFlag, 1, __ATOMIC_RELAXED);
        }

        job = job->next;
      } while (job != nullptr);
      // Let preconnect threads progress.
      if (jobsDone == false) usleep(1);
    } while (jobsDone == false);

    if (ret != ncclSuccess) goto fail;
  }

  if (groupCommHeadMain != nullptr) {
    NCCLCHECKGOTO(doLaunches(groupCommHeadMain), ret, fail);
  }

  while (!ncclIntruQueueEmpty(asyncJobsMain)) {
    struct ncclAsyncJob* job = ncclIntruQueueDequeue(asyncJobsMain);
    if (job->comm && !job->comm->config.blocking)
      (void) ncclCommSetAsyncError(job->comm, ret);
    if (job->destructor) job->destructor((void*)job);
  }

  while (groupCommHeadMain != nullptr) {
    struct ncclComm* comm = groupCommHeadMain;
    struct ncclComm* next = comm->groupNext;
    (void) ncclGroupCommLeave(comm);
    if (!comm->config.blocking) {
      (void) ncclCommSetAsyncError(comm, ret);
    }
    groupCommHeadMain = next;
  }

  CUDACHECK(cudaSetDevice(savedDev));

exit:
  return ret;
fail:
  groupCleanup(gjob->groupCommHeadPtr, gjob->groupCommPreconnectHeadPtr, gjob->asyncJobsPtr, gjob->groupErrorPtr, gjob->groupBlockingPtr, gjob->abortFlagPtr, ret);
  goto exit;
}

ncclResult_t ncclGroupEndInternal() {
  ncclResult_t ret = ncclSuccess;

  if (ncclGroupDepth == 0) {
    WARN("ncclGroupEnd: not in a group call.");
    ret = ncclInvalidUsage;
    goto exit;
  }

  if ((--ncclGroupDepth) > 0) goto exit;

  if ((ret = ncclGroupError) != ncclSuccess) goto fail;

  if (ncclGroupCommHead != nullptr || !ncclIntruQueueEmpty(&ncclAsyncJobs) || ncclGroupCommPreconnectHead != nullptr) {
    ncclGroupJobMain.groupCommHeadPtr = &ncclGroupCommHead;
    ncclGroupJobMain.groupCommPreconnectHeadPtr = &ncclGroupCommPreconnectHead;
    ncclGroupJobMain.groupErrorPtr = &ncclGroupError;
    ncclGroupJobMain.asyncJobsPtr = &ncclAsyncJobs;
    ncclGroupJobMain.abortFlagPtr = &ncclGroupJobAbortFlag;
    ncclGroupJobMain.groupBlockingPtr = &ncclGroupBlocking;
    ncclGroupJobMain.initialized = true;
    ncclGroupJobMainPtr = &ncclGroupJobMain;
    /* make sure ncclGroupBlocking has been set. */
    assert(ncclGroupBlocking == 0 || ncclGroupBlocking == 1);
    if (ncclGroupBlocking == 0 && (ncclGroupCommPreconnectHead != nullptr || !ncclIntruQueueEmpty(&ncclAsyncJobs))) {
      /* nonblocking group */
      if (!ncclIntruQueueEmpty(&ncclAsyncJobs)) {
        ncclAsyncJob* job = ncclIntruQueueHead(&ncclAsyncJobs);
        do {
          NCCLCHECKGOTO(ncclCommSetAsyncError(job->comm, ncclInProgress), ret, fail);
          job->comm->groupJob = ncclGroupJobMainPtr;
          job = job->next;
        } while (job);
      }

      if (ncclGroupCommHead) {
        ncclComm_t comm = ncclGroupCommHead;
        do {
          NCCLCHECKGOTO(ncclCommSetAsyncError(comm, ncclInProgress), ret, fail);
          /* link group job to communicators. */
          comm->groupJob = ncclGroupJobMainPtr;
          comm = comm->groupNext;
        } while (comm);
      }

      ncclGroupJobMainPtr->base.func = groupLaunch;
      SYSCHECKGOTO(pthread_create(&ncclGroupJobMainPtr->base.thread, NULL, ncclAsyncJobMain, (void*)&ncclGroupJobMainPtr->base), ret, fail);
      ret = ncclInProgress;
    } else {
      /* blocking group */
      NCCLCHECKGOTO(groupLaunch(&ncclGroupJobMainPtr->base), ret, fail);
      groupResetJobState(ncclGroupJobMainPtr);
    }
  }

exit:
  return ret;
fail:
  groupCleanup(&ncclGroupCommHead, &ncclGroupCommPreconnectHead, &ncclAsyncJobs, &ncclGroupError, &ncclGroupBlocking, &ncclGroupJobAbortFlag, ret);
  goto exit;
}

ncclResult_t ncclGroupJobComplete(struct ncclGroupJob* groupJob) {
  ncclResult_t ret = ncclSuccess;
  if (groupJob && groupJob->initialized) {
    ret = ncclAsyncJobComplete(&groupJob->base);
    groupResetJobState(groupJob);
  }
  return ret;
}

ncclResult_t ncclGroupJobAbort(struct ncclGroupJob* groupJob) {
  if (groupJob && groupJob->initialized) {
    __atomic_store_n(groupJob->abortFlagPtr, true, __ATOMIC_RELAXED);
    NCCLCHECK(ncclGroupJobComplete(groupJob));
  }
  return ncclSuccess;
}
