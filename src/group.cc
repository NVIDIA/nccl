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
#include "bootstrap.h"

#define GROUP_MAX_RECLAIM_STEPS 10

__thread int ncclGroupDepth = 0; // depth of ncclGroupStart nesting
__thread ncclResult_t ncclGroupError = ncclSuccess;
__thread struct ncclComm* ncclGroupCommHead[ncclGroupTaskTypeNum] = {nullptr};
__thread struct ncclComm* ncclGroupCommPreconnectHead = nullptr;
__thread struct ncclIntruQueue<struct ncclAsyncJob, &ncclAsyncJob::next> ncclAsyncJobs;
__thread int ncclGroupBlocking = -1; /* default mode */
void* ncclAsyncJobMain(void* arg);

ncclResult_t ncclAsyncLaunch(
    struct ncclAsyncJob* job,
    ncclResult_t(*func)(struct ncclAsyncJob*),
    void(*undo)(struct ncclAsyncJob*),
    void(*destructor)(void*), ncclComm_t comm
  ) {
  ncclResult_t ret = ncclSuccess;

  job->destroyFlag = comm->destroyFlag;
  if (ncclGroupDepth == 0) {
    ret = func(job);
    if (ret != ncclSuccess && undo) undo(job);
    if (destructor) destructor(job);
  } else {
    job->func = func;
    job->undo = undo;
    job->destructor = destructor;
    job->abortFlag = comm->abortFlag;
    job->abortFlagDev = comm->abortFlagDev;
    job->childAbortFlag = comm->childAbortFlag;
    job->childAbortFlagDev = comm->childAbortFlagDev;
    job->state = ncclGroupJobRunning;
    job->comm = comm;
    /* check if there are blocking and nonblocking comms at the same time in group. */
    if (comm->destroyFlag) {
      ncclGroupBlocking = 1;
    } else if (ncclGroupBlocking == -1) {
      /* first met communicator */
      ncclGroupBlocking = comm->config.blocking;
    } else if (ncclGroupBlocking != comm->config.blocking) {
      WARN("Blocking and nonblocking communicators are not allowed in the same group.");
      ret = ncclInvalidArgument;
    }
    if (ret == ncclSuccess) {
      ncclIntruQueueEnqueue(&ncclAsyncJobs, job);
    } else {
      // no need to undo, the job hasn't run
      if (destructor) destructor(job);
    }
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
  PTHREADCHECK(pthread_join(job->thread, NULL), "pthread_join");
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

NCCL_API(ncclResult_t, ncclGroupSimulateEnd, ncclSimInfo_t* simInfo);
ncclResult_t ncclGroupSimulateEnd(ncclSimInfo_t* simInfo) {
  ncclResult_t ret = ncclSuccess;
  NVTX3_FUNC_RANGE_IN(nccl_domain);
  NCCLCHECKGOTO(ncclGroupEndInternal(simInfo), ret, exit);
  TRACE_CALL("ncclGroupSimulateEnd()");
exit:
  return ret;
}

struct ncclPreconnectJob {
  struct ncclAsyncJob base;
  struct ncclComm* comm;
  bool* algoNeedConnect;
};

ncclResult_t ncclP2PPreconnectFunc(struct ncclAsyncJob* job_) {
  struct ncclPreconnectJob* job = (struct ncclPreconnectJob*)job_;
  struct ncclComm* comm = job->comm;
  CUDACHECK(cudaSetDevice(comm->cudaDev));
  if (CPU_COUNT(&comm->cpuAffinity)) sched_setaffinity(0, sizeof(cpu_set_t), &comm->cpuAffinity);
  NCCLCHECK(ncclTransportP2pSetup(comm, NULL, 1));
  return ncclSuccess;
}

ncclResult_t ncclCollPreconnectFunc(struct ncclAsyncJob* job_) {
  struct ncclPreconnectJob* job = (struct ncclPreconnectJob*)job_;
  struct ncclComm* comm = job->comm;
  ncclResult_t ret = ncclSuccess;

  CUDACHECK(cudaSetDevice(comm->cudaDev));
  if (CPU_COUNT(&comm->cpuAffinity)) sched_setaffinity(0, sizeof(cpu_set_t), &comm->cpuAffinity);
  for (int i = 0; i < NCCL_NUM_ALGORITHMS; ++i) {
    if (job->algoNeedConnect[i]) {
      switch (i) {
        case NCCL_ALGO_RING: {
          NCCLCHECKGOTO(ncclTransportRingConnect(comm), ret, fail);
          break;
        }
        case NCCL_ALGO_TREE: {
          NCCLCHECKGOTO(ncclTransportTreeConnect(comm), ret, fail);
          break;
        }
        case NCCL_ALGO_NVLS: {
          /* If we are using NVLS_TREE algo, we must mark NVLS algo to set up
           * NVLS intra-node buffer */
          NCCLCHECKGOTO(ncclNvlsBufferSetup(comm), ret, fail);
          break;
        }
        case NCCL_ALGO_NVLS_TREE: {
          NCCLCHECKGOTO(ncclNvlsTreeConnect(comm), ret, fail);
          break;
        }
        case NCCL_ALGO_COLLNET_CHAIN: {
          NCCLCHECKGOTO(ncclCollNetChainBufferSetup(comm), ret, fail);
          break;
        }
        case NCCL_ALGO_COLLNET_DIRECT: {
          NCCLCHECKGOTO(ncclCollNetDirectBufferSetup(comm), ret, fail);
          break;
        }
        case NCCL_ALGO_PAT: {
          NCCLCHECKGOTO(ncclTransportPatConnect(comm), ret, fail);
          break;
        }
        // Yes, it's a dead code.  That's fine...
        // coverity[dead_error_begin]
        default: {
          ret = ncclInternalError;
          goto fail;
        }
      }
    }
  }

exit:
  free(job->algoNeedConnect);
  return ret;
fail:
  goto exit;
}

struct ncclGroupSymmetricJob {
  struct ncclAsyncJob base;
  struct ncclComm* comm;
};

NCCL_PARAM(WinStride, "WIN_STRIDE", -1);

ncclResult_t ncclCommGroupRegisterSymmetric(struct ncclAsyncJob* job_) {
  struct ncclGroupSymmetricJob* job = (struct ncclGroupSymmetricJob*)job_;
  struct ncclComm* comm = job->comm;
  ncclResult_t ret = ncclSuccess;

  CUDACHECKGOTO(cudaSetDevice(comm->cudaDev), ret, fail);
  if (comm->baseStride == 0) {
    cudaStream_t hostStream;
    // first time to allocate symmetric VA space.
    // calling into this function means symmetric is supported.
    struct ncclSymDevBase* symBase = NULL;
    size_t size = ncclSymDevBase::size(comm->localRanks);
    if (ncclParamWinStride() != -1) {
      comm->baseStride = ncclParamWinStride();
    } else {
      size_t maxStride = 0;
      for (int r = 0; r < comm->nRanks; ++r)
        if (comm->peerInfo[r].totalGlobalMem > maxStride) maxStride = comm->peerInfo[r].totalGlobalMem;
      comm->baseStride = maxStride;
    }
    INFO(NCCL_INIT, "rank %d base stride %zuGB total VM %zuGB", comm->rank, comm->baseStride >> 30, (comm->baseStride * comm->localRanks) >> 30);
    NCCLCHECKGOTO(ncclIpcSymmetricInit(comm), ret, fail);
    NCCLCHECKGOTO(ncclNvlsSymmetricInit(comm), ret, fail);
    comm->symAllocHead = 0;

    // Allocate symmetric memory for NCCL internal usage
    NCCLCHECKGOTO(ncclCommSymmetricAllocInternal(comm, size, alignof(struct ncclSymDevBase), (void**)&symBase), ret, fail);
    assert((void*)symBase == (void*)(comm->baseUCSymPtr + comm->localRank * comm->baseStride));
    NCCLCHECKGOTO(ncclStrongStreamAcquire(ncclCudaGraphNone(), &comm->sharedRes->hostStream, /*concurrent=*/false, &hostStream), ret, fail);
    CUDACHECKGOTO(cudaMemsetAsync(symBase, 0, size, hostStream), ret, fail);
    CUDACHECKGOTO(cudaStreamSynchronize(hostStream), ret, fail);
    NCCLCHECKGOTO(ncclStrongStreamRelease(ncclCudaGraphNone(), &comm->sharedRes->hostStream, /*concurrent=*/false), ret, fail);

    comm->symDevComm.base = (struct ncclSymDevBase*)(comm->baseUCSymPtr + comm->localRank * comm->baseStride);
    comm->symDevComm.baseMc = (struct ncclSymDevBase*)comm->baseMCSymPtr;
    comm->symDevComm.nRanks = comm->localRanks;
    comm->symDevComm.nRanks_rcp32 = idivRcp32(comm->localRanks);
    comm->symDevComm.rank = comm->localRank;
    comm->symDevComm.stride4G = comm->baseStride >> 32;
  }

  while (!ncclIntruQueueEmpty(&comm->symRegTaskQueue)) {
    struct ncclSymRegTask* task = ncclIntruQueueDequeue(&comm->symRegTaskQueue);
    NCCLCHECKGOTO(ncclCommSymmetricRegisterInternal(comm, task->buff, task->baseSize, task->alignment, task->memHandle, task->regHandle), ret, fail);
    free(task);
  }

exit:
  return ret;
fail:
  goto exit;
}

static ncclResult_t doLaunches(struct ncclComm* head) {
  ncclResult_t result = ncclSuccess;
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
      (ncclCudaGraphValid(comm->planner.capturingGraph) ? capturingYes : capturingNo) = true;
      CUDACHECKGOTO(cudaSetDevice(comm->cudaDev), result, failure);
      NCCLCHECKGOTO(ncclLaunchPrepare(comm), result, failure);
      if (useBarrier) ncclCommIntraBarrierIn(comm, 1);
      comm = comm->groupNext[ncclGroupTaskTypeCollective];
    } while (comm != nullptr && comm->intraComm0 == cliqueHead->intraComm0);
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
      bool moreRounds = false;
      comm = cliqueHead;
      do { // Iterate clique members.
        struct ncclComm* next = comm->groupNext[ncclGroupTaskTypeCollective];
        if (useBarrier) {
          // Barrier reduction result tells us if this was the final round.
          moreRounds = 0 != ncclCommIntraBarrierOut(comm);
        } else {
          moreRounds |= comm->planner.unlaunchedPlansHead != nullptr;
        }
        if (moreRounds) {
          // Pop next unlaunched kernel
          struct ncclKernelPlan* plan = comm->planner.unlaunchedPlansHead;
          if (plan != nullptr) {
            comm->planner.unlaunchedPlansHead = plan->next;
            CUDACHECKGOTO(cudaSetDevice(comm->cudaDev), result, failure);
            NCCLCHECKGOTO(ncclLaunchKernelBefore_NoUncapturedCuda(comm, plan), result, failure);
            NCCLCHECKGOTO(ncclLaunchKernel(comm, plan), result, failure);
          }
          // Barrier reduction input indicates if we require further rounds.
          if (useBarrier) ncclCommIntraBarrierIn(comm, comm->planner.unlaunchedPlansHead != nullptr ? 1 : 0);
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

static inline void groupLocalResetJobState() {
  ncclGroupError = ncclSuccess;
  for (int type = 0; type < ncclGroupTaskTypeNum; ++type) ncclGroupCommHead[type] = NULL;
  ncclGroupCommPreconnectHead = NULL;
  ncclGroupBlocking = -1;
  ncclIntruQueueConstruct(&ncclAsyncJobs);
  return;
}

static void groupCleanup(struct ncclComm** groupCommHeadPtr, struct ncclIntruQueue<struct ncclAsyncJob, &ncclAsyncJob::next>* asyncJobsPtr, ncclResult_t error) {
  struct ncclComm* comm;
  for (int type = 0; type < ncclGroupTaskTypeNum; ++type) {
    comm = groupCommHeadPtr[type];
    // reset groupCommHeadPtr[type]
    groupCommHeadPtr[type] = nullptr;
    while (comm != nullptr) {
      struct ncclComm* next = comm->groupNext[type];
      (void)ncclGroupCommLeave(comm, type); // overwrites comm->groupNext
      // We don't know if preconnect succeeded or happened at all, so clear
      // the flags that let `taskAppend()` skip over checking if preconnect
      // is needed.
      if (type == ncclGroupTaskTypeCollective) {
        comm->preconnectNext = reinterpret_cast<struct ncclComm*>(0x1);
        for (int i = 0; i < comm->nRanks; i++) {
          comm->connectSend[i] = 0UL;
          comm->connectRecv[i] = 0UL;
        }
        // Reclaim abandoned kernel plan memory. Note ncclWork structs were already
        // reclaimed by a `ncclMemoryStackPop(&comm->memScoped)` during `ncclGroupCommLeave()`.
        while (!ncclIntruQueueEmpty(&comm->planner.planQueue)) {
          struct ncclKernelPlan* plan = ncclIntruQueueDequeue(&comm->planner.planQueue);
          // Persistent plans will be reclaimed via the callbackQueue when the
          // graph drops its UserObject reference.
          if (!plan->persistent) {
            while (!ncclIntruQueueEmpty(&plan->proxyOpQueue)) {
              struct ncclProxyOp* pxop = ncclIntruQueueDequeue(&plan->proxyOpQueue);
              ncclMemoryPoolFree(&comm->memPool_ncclProxyOp, pxop);
            }
            ncclMemoryPoolFree(&comm->memPool_ncclKernelPlan, plan);
          }
        }

        { // Reset comm->planner to empty.
          ncclKernelPlanner::Peer* tmp = comm->planner.peers;
          memset(&comm->planner, 0, sizeof(comm->planner));
          comm->planner.peers = tmp;
          if (comm->planner.peers != NULL) memset(comm->planner.peers, 0, comm->nRanks * sizeof(comm->planner.peers[0]));
        }
      }

      if (!comm->config.blocking)
        (void)ncclCommSetAsyncError(comm, error);
      comm = next;
    }
  }

  /* reset everything */
  while (!ncclIntruQueueEmpty(asyncJobsPtr)) {
    struct ncclAsyncJob* job = ncclIntruQueueDequeue(asyncJobsPtr);
    if (!job->destroyFlag && job->comm && !job->comm->config.blocking)
      (void) ncclCommSetAsyncError(job->comm, error);
    if (job->undo) job->undo(job);
    if (job->destructor) job->destructor((void*)job);
  }

  return;
}

static ncclResult_t asyncJobLaunch(struct ncclIntruQueue<struct ncclAsyncJob, &ncclAsyncJob::next> *asyncJobsMain, volatile bool *groupAbortFlag) {
  ncclResult_t ret = ncclSuccess;
  bool jobsDone = false;
  bool errorJobAbortFlag = false;

  if (!ncclIntruQueueEmpty(asyncJobsMain)) {
    struct ncclAsyncJob* job = ncclIntruQueueHead(asyncJobsMain);
    do {
      PTHREADCHECKGOTO(pthread_create(&job->thread, nullptr, ncclAsyncJobMain, job), "pthread_create", ret, fail);
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
          int err;
          if ((err = pthread_join(job->thread, nullptr)) != 0) {
            WARN("Error waiting for pthread_join: %s", strerror(err));
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

        if (!job->destroyFlag && (__atomic_load_n(groupAbortFlag, __ATOMIC_ACQUIRE) || errorJobAbortFlag == true)) {
          __atomic_store_n(job->abortFlag, 1, __ATOMIC_RELEASE);
          __atomic_store_n(job->abortFlagDev, 1, __ATOMIC_RELEASE);
          if (job->childAbortFlag) {
            __atomic_store_n(job->childAbortFlag, 1, __ATOMIC_RELEASE);
            __atomic_store_n(job->childAbortFlagDev, 1, __ATOMIC_RELEASE);
          }
        }

        job = job->next;
      } while (job != nullptr);
      // Let preconnect threads progress.
      if (jobsDone == false) usleep(1);
    } while (jobsDone == false);

    if (ret != ncclSuccess) goto fail;
  }

exit:
  return ret;
fail:
  goto exit;
}

static ncclResult_t groupLaunch(struct ncclAsyncJob *job_, ncclSimInfo_t* simInfo = NULL) {
  ncclResult_t ret = ncclSuccess;
  struct ncclGroupJob *gjob = (struct ncclGroupJob*) job_;
  struct ncclComm **groupCommHeadMain = gjob->groupCommHead;
  struct ncclComm *groupCommPreconnectHeadMain = gjob->groupCommPreconnectHead;
  struct ncclIntruQueue<struct ncclAsyncJob, &ncclAsyncJob::next> *asyncJobsMain = &gjob->asyncJobs;
  bool *groupAbortFlag = &gjob->abortFlag;

  if (!simInfo && groupCommPreconnectHeadMain != nullptr) {
    struct ncclComm* comm = groupCommPreconnectHeadMain;
    do {
      struct ncclPreconnectJob* job;
      NCCLCHECKGOTO(ncclCalloc(&job, 1), ret, fail);
      job->base.func = ncclP2PPreconnectFunc;
      job->base.undo = nullptr;
      job->base.destructor = free;
      job->base.state = ncclGroupJobRunning;
      job->base.abortFlag = comm->abortFlag;
      job->base.abortFlagDev = comm->abortFlagDev;
      job->comm = comm;
      ncclIntruQueueEnqueue(asyncJobsMain,  (struct ncclAsyncJob*)job);

      struct ncclComm* next = comm->preconnectNext;
      comm->preconnectNext = reinterpret_cast<struct ncclComm*>(0x1);
      comm = next;
    } while (comm != nullptr);
  }

  NCCLCHECKGOTO(asyncJobLaunch(asyncJobsMain, groupAbortFlag), ret, fail);

  // only loop through sym alloc and register tasks
  for (int type = ncclGroupTaskTypeSymRegister; type <= ncclGroupTaskTypeSymRegister; ++type) {
    if (groupCommHeadMain[type]) {
      struct ncclComm* cliqueHead = groupCommHeadMain[type];
      struct ncclComm* comm = NULL;
      struct ncclIntruQueue<struct ncclAsyncJob, &ncclAsyncJob::next> asyncSymJobs;
      ncclIntruQueueConstruct(&asyncSymJobs);
      do {
        comm = cliqueHead;
        do {
          struct ncclGroupSymmetricJob* job;
          NCCLCHECKGOTO(ncclCalloc(&job, 1), ret, fail);
          job->base.func = ncclCommGroupRegisterSymmetric;
          job->base.undo = nullptr;
          job->base.destructor = free;
          job->base.state = ncclGroupJobRunning;
          job->base.abortFlag = comm->abortFlag;
          job->base.abortFlagDev = comm->abortFlagDev;
          job->comm = comm;
          ncclIntruQueueEnqueue(&asyncSymJobs, (struct ncclAsyncJob*)job);
          comm = comm->groupNext[type];
        } while (comm != nullptr && comm->intraComm0 == cliqueHead->intraComm0);
        NCCLCHECKGOTO(asyncJobLaunch(&asyncSymJobs, groupAbortFlag), ret, fail);
        while (!ncclIntruQueueEmpty(&asyncSymJobs)) {
          struct ncclAsyncJob* job = ncclIntruQueueDequeue(&asyncSymJobs);
          if (job->destructor) job->destructor((void*)job);
        }
        cliqueHead = comm;
      } while (cliqueHead != nullptr);
    }
  }

  /* Connect channels at runtime if cumem is supported */
  if (groupCommHeadMain[ncclGroupTaskTypeCollective] != nullptr) {
    struct ncclComm* cliqueHead = groupCommHeadMain[ncclGroupTaskTypeCollective];
    struct ncclComm* comm = NULL;
    struct ncclIntruQueue<struct ncclAsyncJob, &ncclAsyncJob::next> asyncCollJobs;
    ncclIntruQueueConstruct(&asyncCollJobs);
    do {
      // We need to preconnect connections for collectives clique by clique to avoid
      // race condition for split shared comms which can connect the same connections
      // at the same time.
      comm = cliqueHead;
      do {
        bool needConnect = false;
        bool algoNeedConnect[NCCL_NUM_ALGORITHMS];
        memset(algoNeedConnect, 0, sizeof(bool) * NCCL_NUM_ALGORITHMS);

        CUDACHECKGOTO(cudaSetDevice(comm->cudaDev), ret, fail);
        NCCLCHECKGOTO(ncclPrepareTasks(comm, algoNeedConnect, &needConnect, simInfo), ret, fail);

        if (comm->cuMemSupport && needConnect) {
          struct ncclPreconnectJob* job;
          NCCLCHECKGOTO(ncclCalloc(&job, 1), ret, fail);
          job->base.func = ncclCollPreconnectFunc;
          job->base.undo = nullptr;
          job->base.destructor = free;
          job->base.state = ncclGroupJobRunning;
          job->base.abortFlag = comm->abortFlag;
          job->base.abortFlagDev = comm->abortFlagDev;
          job->comm = comm;
          NCCLCHECKGOTO(ncclCalloc(&job->algoNeedConnect, NCCL_NUM_ALGORITHMS), ret, fail);
          memcpy(job->algoNeedConnect, algoNeedConnect, sizeof(bool) * NCCL_NUM_ALGORITHMS);
          ncclIntruQueueEnqueue(&asyncCollJobs, &job->base);
        }
        comm = comm->groupNext[ncclGroupTaskTypeCollective];
      } while (comm != nullptr && comm->intraComm0 == cliqueHead->intraComm0);
      // connect
      NCCLCHECKGOTO(asyncJobLaunch(&asyncCollJobs, groupAbortFlag), ret, fail);
      while (!ncclIntruQueueEmpty(&asyncCollJobs)) {
        struct ncclAsyncJob* job = ncclIntruQueueDequeue(&asyncCollJobs);
        if (job->destructor) job->destructor((void*)job);
      }
      cliqueHead = comm;
    } while (cliqueHead != nullptr);

    // done with all buffer allocation, start registration and enqueue
    comm = groupCommHeadMain[ncclGroupTaskTypeCollective];
    do {
      CUDACHECKGOTO(cudaSetDevice(comm->cudaDev), ret, fail);
      NCCLCHECKGOTO(ncclTasksRegAndEnqueue(comm), ret, fail);
      comm = comm->groupNext[ncclGroupTaskTypeCollective];
    } while (comm);
  }

  if ((!simInfo) && (groupCommHeadMain[ncclGroupTaskTypeCollective] != nullptr)) {
    NCCLCHECKGOTO(doLaunches(groupCommHeadMain[ncclGroupTaskTypeCollective]), ret, fail);
  }

  while (!ncclIntruQueueEmpty(asyncJobsMain)) {
    struct ncclAsyncJob* job = ncclIntruQueueDequeue(asyncJobsMain);
    if (!job->destroyFlag && job->comm && !job->comm->config.blocking && groupCommHeadMain[ncclGroupTaskTypeCollective] == nullptr)
      (void) ncclCommSetAsyncError(job->comm, ret);
    if (job->destructor) job->destructor((void*)job);
  }

  for (int type = 0; type < ncclGroupTaskTypeNum; ++type) {
    while (groupCommHeadMain[type] != nullptr) {
      struct ncclComm* comm = groupCommHeadMain[type];
      struct ncclComm* next = comm->groupNext[type];
      // Poll for callbacks sent to us from other threads. Typically these free
      // resources from to our memory pools and UB
      if (comm->reclaimSteps == GROUP_MAX_RECLAIM_STEPS) {
        NCCLCHECKGOTO(ncclCommPollCallbacks(comm, /*waitSome=*/false), ret, fail);
        comm->reclaimSteps = 0;
      } else {
        comm->reclaimSteps++;
      }
      (void)ncclGroupCommLeave(comm, type);
      if (!comm->config.blocking) {
        (void)ncclCommSetAsyncError(comm, ret);
      }
      groupCommHeadMain[type] = next;
    }
  }

exit:
  return ret;
fail:
  groupCleanup(gjob->groupCommHead, &gjob->asyncJobs, ret);
  goto exit;
}

static ncclResult_t groupLaunchNonBlocking(struct ncclAsyncJob *job_) {
  return groupLaunch(job_ /* estimatedTime = NULL */);
}

ncclResult_t ncclGroupEndInternal(ncclSimInfo_t* simInfo) {
  ncclResult_t ret = ncclSuccess;
  ncclSimInfo_t internalSimInfo = NCCL_SIM_INFO_INITIALIZER;
  ncclSimInfo_t* internalSimInfoPtr = NULL;
  size_t realSize = 0;
  bool hasCommHead = false;
  ncclGroupJob* groupJob = NULL;

  internalSimInfo.magic = 0;

  if (ncclGroupDepth == 0) {
    WARN("ncclGroupEnd: not in a group call.");
    ret = ncclInvalidUsage;
    goto exit;
  }

  if ((--ncclGroupDepth) > 0) goto exit;

  if ((ret = ncclGroupError) != ncclSuccess) goto fail;

  if (simInfo) {
    memcpy((void*)&realSize, (void*)&simInfo->size, sizeof(size_t));
    realSize = realSize > sizeof(ncclSimInfo_t) ? sizeof(ncclSimInfo_t) : realSize;
    memcpy((void*)&internalSimInfo, (void*)simInfo, realSize);
    if (internalSimInfo.magic != 0x74685283) {
      WARN("ncclSimInfo_t argument not initialized via NCCL_SIM_INFO_INITIALIZER");
      ret = ncclInvalidArgument;
      goto fail;
    }
    internalSimInfoPtr = &internalSimInfo;
  }

  for (int type = 0; type < ncclGroupTaskTypeNum; ++type) {
    if (ncclGroupCommHead[type]) {
      hasCommHead = true;
      break;
    }
  }

  NCCLCHECKGOTO(ncclCalloc(&groupJob, 1), ret, fail);
  ncclIntruQueueConstruct(&groupJob->asyncJobs);
  groupJob->groupRefCount = 0;
  groupJob->nonBlockingInit = false;
  memcpy(groupJob->groupCommHead, ncclGroupCommHead, sizeof(ncclGroupCommHead));
  groupJob->groupCommPreconnectHead = ncclGroupCommPreconnectHead;
  groupJob->groupError = ncclSuccess;
  groupJob->abortFlag = false;
  groupJob->joined = false;
  ncclIntruQueueTransfer(&groupJob->asyncJobs, &ncclAsyncJobs);

  if (hasCommHead || !ncclIntruQueueEmpty(&groupJob->asyncJobs) || ncclGroupCommPreconnectHead != nullptr) {
    /* make sure ncclGroupBlocking has been set. */
    assert(ncclGroupBlocking == 0 || ncclGroupBlocking == 1);
    if (ncclGroupBlocking == 0) {
      /* nonblocking group */
      if (!ncclIntruQueueEmpty(&groupJob->asyncJobs)) {
        ncclAsyncJob* job = ncclIntruQueueHead(&groupJob->asyncJobs);
        do {
          NCCLCHECKGOTO(ncclCommSetAsyncError(job->comm, ncclInProgress), ret, fail);
          if (job->comm->groupJob == NULL) {
            job->comm->groupJob = groupJob;
            groupJob->groupRefCount++;
          }
          job = job->next;
        } while (job);
      }

      for (int type = 0; type < ncclGroupTaskTypeNum; ++type) {
        if (ncclGroupCommHead[type]) {
          ncclComm_t comm = ncclGroupCommHead[type];
          do {
            NCCLCHECKGOTO(ncclCommSetAsyncError(comm, ncclInProgress), ret, fail);
            /* link group job to communicators. */
            if (comm->groupJob == NULL) {
              comm->groupJob = groupJob;
              groupJob->groupRefCount++;
            }
            comm = comm->groupNext[type];
          } while (comm);
        }
      }

      groupJob->base.func = groupLaunchNonBlocking;
      PTHREADCHECKGOTO(pthread_create(&groupJob->base.thread, NULL, ncclAsyncJobMain, (void*)&groupJob->base), "pthread_create", ret, fail);
      groupJob->nonBlockingInit = true;
      ret = ncclInProgress;
    } else {
      /* blocking group */
      int savedDev;
      CUDACHECKGOTO(cudaGetDevice(&savedDev), ret, fail);
      NCCLCHECKGOTO(groupLaunch(&groupJob->base, internalSimInfoPtr), ret, fail);
      CUDACHECKGOTO(cudaSetDevice(savedDev), ret, fail);
      if (simInfo) memcpy((void*)simInfo, (void*)internalSimInfoPtr, realSize);
      free(groupJob);
    }
  }
  /* Reset the job state for the next group call. */
  groupLocalResetJobState();

exit:
  return ret;
fail:
  if (groupJob) {
    groupCleanup(groupJob->groupCommHead, &groupJob->asyncJobs, ret);
    free(groupJob);
  } else {
    groupCleanup(ncclGroupCommHead, &ncclAsyncJobs, ret);
  }
  groupLocalResetJobState();
  goto exit;
}

ncclResult_t ncclGroupJobComplete(struct ncclGroupJob* groupJob) {
  ncclResult_t ret = ncclSuccess;
  if (groupJob && groupJob->nonBlockingInit) {
    if (!__atomic_exchange_n(&groupJob->joined, true, __ATOMIC_ACQ_REL)) {
      ret = ncclAsyncJobComplete(&groupJob->base);
    }
    if (ncclAtomicRefCountDecrement(&groupJob->groupRefCount) == 0) {
      free(groupJob);
    }
  }
  return ret;
}

ncclResult_t ncclGroupJobAbort(struct ncclGroupJob* groupJob) {
  if (groupJob && groupJob->nonBlockingInit) {
    if (!__atomic_exchange_n(&groupJob->joined, true, __ATOMIC_ACQ_REL)) {
      __atomic_store_n(&groupJob->abortFlag, true, __ATOMIC_RELAXED);
      ncclAsyncJobComplete(&groupJob->base);
    }
    if (ncclAtomicRefCountDecrement(&groupJob->groupRefCount) == 0) {
      free(groupJob);
    }
  }
  return ncclSuccess;
}
