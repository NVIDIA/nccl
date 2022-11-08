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
static ncclResult_t groupJobComplete(struct ncclGroupJob *job);
static uint64_t groupIntraSync(bool isCollective, ptrdiff_t localPeerMaskOffsetInComm, struct ncclComm* head, uint64_t subsetMask, uint64_t orArgMask);

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

  /* if previous group launch does not complete, don't launch this one. */
  if (ncclGroupJobMainPtr != NULL) {
    if (__atomic_load_n(&ncclGroupJobMainPtr->doneFlag, __ATOMIC_ACQUIRE) == false) {
      ret = ncclInvalidUsage;
      goto exit;
    } else {
      NCCLCHECKGOTO(groupJobComplete(ncclGroupJobMainPtr), ret, exit);
    }
  }
  NCCLCHECK(ncclGroupStartInternal());
  TRACE_CALL("ncclGroupStart()");

exit:
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

  NCCLCHECK(ncclStrongStreamAcquireUncaptured(&comm->sharedRes->hostStream));
  NCCLCHECK(ncclTransportP2pSetup(comm, NULL, 1, comm->sharedRes->hostStream.cudaStream));

  NCCLCHECK(ncclStrongStreamAcquireUncaptured(&comm->sharedRes->deviceStream));
  NCCLCHECK(ncclStrongStreamWaitStream(ncclCudaGraphNone(), &comm->sharedRes->deviceStream, &comm->sharedRes->hostStream));
  NCCLCHECK(ncclStrongStreamRelease(ncclCudaGraphNone(), &comm->sharedRes->deviceStream));

  NCCLCHECK(ncclStrongStreamRelease(ncclCudaGraphNone(), &comm->sharedRes->hostStream));
  NCCLCHECK(ncclStrongStreamSynchronize(&comm->sharedRes->hostStream));
  return ncclSuccess;
}

static ncclResult_t doLaunches(struct ncclComm* head) {
  ncclResult_t result = ncclSuccess;
  struct ncclComm* cliqueHead = head;
  struct ncclComm* cliqueNextHead;
  bool isCollective = (0 != head->tasks.nTasksColl + head->tasks.nTasksBcast);
  // This outer loop iterates over cliques of comms which are siblings of the
  // same global entity. We calculate a clique as all comms which have the same
  // `intraComm0` value.
  do {
    struct ncclComm* cliqueComm0 = cliqueHead->intraComm0;
    struct ncclComm* comm = cliqueHead;
    bool capturingYes = false, capturingNo = false;
    uint64_t cliqueMask = 0;
    uint64_t subsetMask, moreMask;
    do {
      (ncclCudaGraphValid(comm->tasks.capturingGraph) ? capturingYes : capturingNo) = true;
      cliqueMask |= uint64_t(1)<<comm->intraRank;
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

    (void)groupIntraSync(isCollective, offsetof(ncclComm, tasks.p2pIntraPeerMask), cliqueHead, cliqueMask, /*orArgMask=*/0);
    comm = cliqueHead;
    do {
      CUDACHECKGOTO(cudaSetDevice(comm->cudaDev), result, failure);
      NCCLCHECKGOTO(ncclLaunchPrepare(comm), result, failure);
      comm = comm->groupNext;
    } while (comm != cliqueNextHead);

    // Iterate rounds of launches for clique. `subsetMask` tracks which comms
    // (by local rank index) are either still launching or communicate with an
    // intra-proc peer which is still launching. `moreMask` is which comms have
    // a plan to launch on the next round.
    subsetMask = cliqueMask;
    moreMask = cliqueMask;
    while (true) {
      subsetMask = groupIntraSync(isCollective, offsetof(ncclComm, tasks.p2pIntraPeerMask), cliqueHead, subsetMask, moreMask);
      if (subsetMask == 0) break;

      comm = cliqueHead;
      moreMask = 0;
      do { // Iterate clique members.
        if (1 & subsetMask>>comm->intraRank) {
          // Pop next unlaunched kernel
          struct ncclKernelPlan* plan = comm->unlaunchedPlansHead;
          if (plan != nullptr) {
            comm->unlaunchedPlansHead = plan->next;
            CUDACHECKGOTO(cudaSetDevice(comm->cudaDev), result, failure);
            NCCLCHECKGOTO(ncclLaunchKernelBefore_NoUncapturedCuda(comm, plan), result, failure);
            NCCLCHECKGOTO(ncclLaunchKernel(comm, plan), result, failure);
          }
          // Barrier reduction input indicates if we require further rounds.
          moreMask |= uint64_t(comm->unlaunchedPlansHead != nullptr ? 1 : 0)<<comm->intraRank;
          if (plan != nullptr) {
            NCCLCHECKGOTO(ncclLaunchKernelAfter_NoCuda(comm, plan), result, failure);
          }
        }
        comm = comm->groupNext;
      } while (comm != cliqueNextHead);
    }

    comm = cliqueHead;
    do {
      CUDACHECKGOTO(cudaSetDevice(comm->cudaDev), result, failure);
      NCCLCHECKGOTO(ncclLaunchFinish(comm), result, failure);
      comm = comm->groupNext;
    } while (comm != cliqueNextHead);

    cliqueHead = cliqueNextHead;
  } while (cliqueHead != nullptr);
failure:
  return result;
}

static inline void groupResetJobState() {
  ncclGroupBlocking = -1;
  ncclGroupJobMainPtr = NULL;
  memset(&ncclGroupJobMain, 0, sizeof(struct ncclGroupJob));
  return;
}

static void groupCleanup(struct ncclComm** groupCommHeadPtr, struct ncclComm** groupCommPreconnectHeadPtr, struct ncclIntruQueue<struct ncclAsyncJob, &ncclAsyncJob::next>* asyncJobsPtr, ncclResult_t* groupErrorPtr, ncclResult_t error) {
  struct ncclComm* comm = *groupCommHeadPtr;

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
          while (!ncclIntruQueueEmpty(&plan->channels[c].proxyWorkQueue)) {
            struct ncclProxyWork* proxyWork = ncclIntruQueueDequeue(&plan->channels[c].proxyWorkQueue);
            ncclMemoryPoolFree(&comm->memPool_ncclProxyWork, proxyWork);
          }
        }
        ncclMemoryPoolFree(&comm->memPool_ncclKernelPlan, plan);
      }
    }
    // Reset comm->tasks to empty.
    comm->tasks.nTasksColl = 0;
    comm->tasks.nTasksP2p = 0;
    comm->tasks.nTasksBcast = 0;
    comm->tasks.p2pIntraPeerMask = 0;
    comm->tasks.minBcastPeer = INT_MAX;
    comm->tasks.maxBcastPeer = INT_MIN;
    comm->tasks.streams = nullptr;
    ncclIntruQueueConstruct(&comm->tasks.collQueue);
    comm->tasks.collLoadBytes = 0;
    for (int i = 0; i < comm->nRanks; i++) {
      ncclIntruQueueConstruct(&comm->tasks.peers[i].sendQueue);
      ncclIntruQueueConstruct(&comm->tasks.peers[i].recvQueue);
      ncclIntruQueueConstruct(&comm->tasks.peers[i].bcastQueue);
    }

    if (!comm->config.blocking)
      (void) ncclCommSetAsyncError(comm, error);
    comm = next;
  }

  /* reset everything */
  while (!ncclIntruQueueEmpty(asyncJobsPtr)) {
    struct ncclAsyncJob* job = ncclIntruQueueDequeue(asyncJobsPtr);
    *job->abortFlag = 1;
    if (job->comm && !job->comm->config.blocking)
      (void) ncclCommSetAsyncError(job->comm, error);
    if (job->undo) job->undo(job);
    if (job->destructor) job->destructor((void*)job);
  }

  *groupErrorPtr = ncclSuccess;
  *groupCommHeadPtr = nullptr;
  *groupCommPreconnectHeadPtr = nullptr;
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

        if (*groupAbortFlag == true || errorJobAbortFlag == true) {
          *job->abortFlag = 1;
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

  /* this atomic must happen before cleanup and setting state of communicators */
  __atomic_store_n(&gjob->doneFlag, true, __ATOMIC_RELEASE);

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

  *gjob->groupErrorPtr = ncclSuccess;
  *gjob->groupCommHeadPtr = nullptr;
  *gjob->groupCommPreconnectHeadPtr = nullptr;

  CUDACHECK(cudaSetDevice(savedDev));

exit:
  return ret;
fail:
  groupCleanup(gjob->groupCommHeadPtr, gjob->groupCommPreconnectHeadPtr, gjob->asyncJobsPtr, gjob->groupErrorPtr, ret);
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
    ncclGroupJobMain.doneFlag = false;
    ncclGroupJobMainPtr = &ncclGroupJobMain;
    /* make sure ncclGroupBlocking has been set. */
    assert(ncclGroupBlocking == 0 || ncclGroupBlocking == 1);
    if (ncclGroupBlocking == 0 && (ncclGroupCommPreconnectHead != nullptr || !ncclIntruQueueEmpty(&ncclAsyncJobs))) {
      /* nonblocking group */
      if (!ncclIntruQueueEmpty(&ncclAsyncJobs)) {
        ncclAsyncJob* job = ncclIntruQueueHead(&ncclAsyncJobs);
        do {
          NCCLCHECKGOTO(ncclCommSetAsyncError(job->comm, ncclInProgress), ret, fail);
          job = job->next;
        } while (job);
      }

      if (ncclGroupCommHead) {
        ncclComm_t comm = ncclGroupCommHead;
        do {
          NCCLCHECKGOTO(ncclCommSetAsyncError(comm, ncclInProgress), ret, fail);
          comm = comm->groupNext;
        } while (comm);
      }
      ncclGroupJobMainPtr->base.func = groupLaunch;
      SYSCHECKGOTO(pthread_create(&ncclGroupJobMainPtr->base.thread, NULL, ncclAsyncJobMain, (void*)&ncclGroupJobMainPtr->base), ret, fail);
      ret = ncclInProgress;
    } else {
      /* blocking group */
      NCCLCHECKGOTO(groupLaunch(&ncclGroupJobMainPtr->base), ret, fail);
      groupResetJobState();
    }
  }

exit:
  return ret;
fail:
  groupCleanup(&ncclGroupCommHead, &ncclGroupCommPreconnectHead, &ncclAsyncJobs, &ncclGroupError, ret);
  groupResetJobState();
  goto exit;
}

static ncclResult_t groupJobComplete(struct ncclGroupJob* job) {
  ncclResult_t ret = ncclSuccess;
  if (job) {
    ret = ncclAsyncJobComplete(&job->base);
    groupResetJobState();
  }
  return ret;
}

void ncclGroupJobAbort() {
  ncclGroupJobAbortFlag = true;
  (void) groupJobComplete(ncclGroupJobMainPtr);
  /* reset group abort flag */
  ncclGroupJobAbortFlag = false;
}

static inline bool groupIntraSync_Collective(struct ncclComm* head, int nComms, bool orArg) {
  int intraRanks = head->intraRanks;
  uint32_t phase = head->intraBarrierPhase;
  { struct ncclComm* comm = head;
    for (int n=nComms; n-- != 0;) {
      comm->intraBarrierPhase = phase+1;
      comm = comm->groupNext;
    }
  }
  phase &= 1;
  struct ncclComm* comm0 = head->intraComm0;
  uint64_t count = __atomic_add_fetch(&comm0->intraBarrierCounter, (uint64_t(orArg?1:0)<<32) + nComms, __ATOMIC_RELEASE);

  if (uint32_t(count) == intraRanks) {
    // Reset.
    __atomic_store_n(&comm0->intraBarrierCounter, 0, __ATOMIC_RELAXED);
    // Release everyone.
    count = count>>32;
    __atomic_store_n(&comm0->intraBarrierGate, (count<<32)|(phase^1), __ATOMIC_RELEASE);
  } else {
    uint64_t gate = __atomic_load_n(&comm0->intraBarrierGate, __ATOMIC_RELAXED);
    if ((gate & 1) == phase) {
      uint64_t t0 = clockNano();
      do {
        // Spin vigorously for first 5us.
        if (clockNano()-t0 >= 5*1000) sched_yield();
        gate = __atomic_load_n(&comm0->intraBarrierGate, __ATOMIC_RELAXED);
      } while ((gate & 1) == phase);
    }
    count = gate>>32;
  }
  __atomic_thread_fence(__ATOMIC_ACQUIRE);
  return count != 0;
}

static uint64_t groupIntraSync_NonCollective(
    ptrdiff_t localPeerMaskOffsetInComm, struct ncclComm* head,
    uint64_t subsetMask, uint64_t orArgMask
  ) {
  struct ncclComm* comm = head;
  for (uint64_t mask=subsetMask; mask != 0;) {
    if (1 & subsetMask>>comm->intraRank) {
      uint64_t selfMask = uint64_t(1)<<comm->intraRank;
      uint64_t peerMask = *(uint64_t*)((char*)comm + localPeerMaskOffsetInComm);
      peerMask &= ~selfMask;
      if (peerMask != 0) {
        uint64_t t0 = clockNano();
        do {
          int intraPeer = bitffs(peerMask)-1;
          peerMask &= peerMask-1;
          struct ncclComm* peerComm = comm->intraComms[intraPeer];
          // Wait for previous syncs with this peer to be consumed.
          while (0 != (selfMask & __atomic_load_n(&peerComm->intraSyncArrived, __ATOMIC_RELAXED))) {
            // Spin vigorously for first 5us.
            if (clockNano()-t0 >= 5*1000) sched_yield();
          }
          __atomic_thread_fence(__ATOMIC_ACQUIRE);
          // Send our sync.
          __atomic_fetch_or(&peerComm->intraSyncOrArg, selfMask & orArgMask, __ATOMIC_RELAXED);
          __atomic_fetch_or(&peerComm->intraSyncArrived, selfMask, __ATOMIC_RELEASE);
        } while (peerMask != 0);
      }
    }
    mask &= ~(uint64_t(1)<<comm->intraRank);
    comm = comm->groupNext;
  }

  uint64_t orResult = 0;
  comm = head;
  for (uint64_t mask=subsetMask; mask != 0;) {
    if (1 & subsetMask>>comm->intraRank) {
      uint64_t selfMask = uint64_t(1)<<comm->intraRank;
      uint64_t peerMask = *(uint64_t*)((char*)comm + localPeerMaskOffsetInComm);
      peerMask &= ~selfMask;
      if (peerMask != 0) {
        uint64_t t0 = clockNano();
        // Wait for syncs to arrive.
        while (peerMask != (peerMask & __atomic_load_n(&comm->intraSyncArrived, __ATOMIC_RELAXED))) {
          // Spin vigorously for first 5us.
          if (clockNano()-t0 >= 5*1000) sched_yield();
        }
        __atomic_thread_fence(__ATOMIC_ACQUIRE);
        // Consume sync in a way that resets those bits to zero.
        if (0 != (peerMask & __atomic_fetch_and(&comm->intraSyncOrArg, ~peerMask, __ATOMIC_RELAXED))) {
          orResult |= selfMask;
        }
        __atomic_fetch_and(&comm->intraSyncArrived, ~peerMask, __ATOMIC_RELEASE);
      }
    }
    mask &= ~(uint64_t(1)<<comm->intraRank);
    comm = comm->groupNext;
  }

  return orResult;
}

// Sync with intra-proc peer threads. `isCollective=true` indicates if all
// peers are syncing with each other (barrier), otherwise the mask of local peers
// with whom to 1:1 sync with is obtained from each comm struct via the
// `localPeerMaskOffsetInComm` byte offset. The input list of comms are those
// managed by this thread, it starts at `head` and has `nComms` members and is
// linked via `ncclComm::groupNext` field. The value returned is the OR-reduction
// of the `orArg` supplied by each thread synced with.
inline static uint64_t groupIntraSync(bool isCollective, ptrdiff_t localPeerMaskOffsetInComm, struct ncclComm* head, uint64_t subsetMask, uint64_t orArgMask) {
  int nComms = bitpopcnt(subsetMask);
  if (nComms == head->intraRanks) return orArgMask;
  if (isCollective) {
    return groupIntraSync_Collective(head, nComms, orArgMask != 0) ? subsetMask : 0;
  } else {
    return groupIntraSync_NonCollective(localPeerMaskOffsetInComm, head, subsetMask, orArgMask);
  }
}
