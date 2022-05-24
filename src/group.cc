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

__thread int ncclGroupDepth = 0; // depth of ncclGroupStart nesting
__thread ncclResult_t ncclGroupError = ncclSuccess;
__thread struct ncclComm* ncclGroupCommHead = nullptr;
__thread struct ncclComm* ncclGroupCommPreconnectHead = nullptr;
__thread struct ncclIntruQueue<struct ncclAsyncJob, &ncclAsyncJob::next> ncclAsyncJobs;

ncclResult_t ncclAsyncLaunch(
    struct ncclAsyncJob* job,
    ncclResult_t(*func)(struct ncclAsyncJob*),
    void(*undo)(struct ncclAsyncJob*),
    void(*destructor)(void*)
  ) {
  if (0 == ncclGroupDepth) {
    ncclResult_t res = func(job);
    if (res != ncclSuccess && undo) undo(job);
    if (destructor) destructor(job);
    return res;
  } else {
    job->func = func;
    job->undo = undo;
    job->destructor = destructor;
    ncclIntruQueueEnqueue(&ncclAsyncJobs, job);
    return ncclSuccess;
  }
}

void* ncclAsyncJobMain(void* arg) {
  struct ncclAsyncJob* job = (struct ncclAsyncJob*)arg;
  job->result = job->func(job);
  if (job->result != ncclSuccess) {
    INFO(NCCL_INIT,"%s:%d -> %d [Async thread]", __FILE__, __LINE__, job->result);
  }
  return arg;
}

NCCL_API(ncclResult_t, ncclGroupStart);
ncclResult_t ncclGroupStart() {
  NVTX3_FUNC_RANGE_IN(nccl_domain);
  NCCLCHECK(ncclGroupStartInternal());
  TRACE_CALL("ncclGroupStart()");
  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclGroupEnd);
ncclResult_t ncclGroupEnd() {
  NVTX3_FUNC_RANGE_IN(nccl_domain);
  NCCLCHECK(ncclGroupEndInternal());
  TRACE_CALL("ncclGroupEnd()");
  return ncclSuccess;
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

ncclResult_t ncclGroupEndInternal() {
  if (ncclGroupDepth == 0) {
    WARN("ncclGroupEnd: not in a group call.");
    return ncclInvalidUsage;
  }
  ncclGroupDepth--;
  if (ncclGroupDepth > 0) return ncclSuccess;

  int savedDev;
  CUDACHECK(cudaGetDevice(&savedDev));

  ncclResult_t ret = ncclGroupError;
  bool jobsDone = false;
  if (ret != ncclSuccess) goto failure;

  if (ncclGroupCommPreconnectHead != nullptr) {
    struct ncclComm* comm = ncclGroupCommPreconnectHead;
    do {
      struct ncclPreconnectJob* job;
      NCCLCHECK(ncclCalloc(&job, 1));
      job->base.func = ncclPreconnectFunc;
      job->base.undo = nullptr;
      job->base.destructor = free;
      job->comm = comm;
      ncclIntruQueueEnqueue(&ncclAsyncJobs, &job->base);

      struct ncclComm* next = comm->preconnectNext;
      comm->preconnectNext = reinterpret_cast<struct ncclComm*>(0x1);
      comm = next;
    } while (comm != nullptr);
  }

  if (!ncclIntruQueueEmpty(&ncclAsyncJobs)) {
    struct ncclAsyncJob* job = ncclIntruQueueHead(&ncclAsyncJobs);
    do {
      pthread_create(&job->thread, nullptr, ncclAsyncJobMain, job);
      job = job->next;
    } while (job != nullptr);

    job = ncclIntruQueueHead(&ncclAsyncJobs);
    do {
      int err = pthread_join(job->thread, nullptr);
      if (err != 0) {
        WARN("Error waiting for pthread_join : %s", strerror(errno));
        ret = ncclSystemError;
      }
      if (ret == ncclSuccess && job->result != ncclSuccess) ret = job->result;
      job = job->next;
    } while (job != nullptr);

    jobsDone = true;
    if (ret != ncclSuccess) goto failure;
  }

  if (ncclGroupCommHead != nullptr) {
    NCCLCHECKGOTO(doLaunches(ncclGroupCommHead), ret, failure);
    do {
      struct ncclComm* comm = ncclGroupCommHead;
      struct ncclComm* next = comm->groupNext;
      ncclGroupCommLeave(comm);
      ncclGroupCommHead = next;
    } while (ncclGroupCommHead != nullptr);
  }

  if (false) {
  failure:
    struct ncclComm* comm = ncclGroupCommHead;
    while (comm != nullptr) {
      struct ncclComm* next = comm->groupNext;
      ncclGroupCommLeave(comm); // overwrites comm->groupNext
      // We don't know if preconnect succeeded or happened at all, so clear
      // the flags that let `taskAppend()` skip over checking if preconnect
      // is needed.
      comm->preconnectNext = reinterpret_cast<struct ncclComm*>(0x1);
      for (int i=0; i < comm->nRanks; i++) {
        comm->tasks.peers[i].sendSeen = false;
        comm->tasks.peers[i].recvSeen = false;
        comm->connectSend[i] = 0;
        comm->connectRecv[i] = 0;
      }
      comm->unlaunchedPlansHead = nullptr;
      // Reclaim abandoned kernel plan memory. Note ncclWork structs were already
      // reclaimed by a `ncclMemoryStackPop(&comm->memScoped)` during `ncclGroupCommLeave()`.
      while (!ncclIntruQueueEmpty(&comm->planQueue)) {
        struct ncclKernelPlan* plan = ncclIntruQueueDequeue(&comm->planQueue);
        // Persistent plans will be reclaimed via the callbackQueue when the
        // graph drops its UserObject reference.
        if (!plan->persistent) {
          for (int c=0; c < MAXCHANNELS; c++) {
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
      comm->tasks.streams = nullptr;
      ncclIntruQueueConstruct(&comm->tasks.collQueue);
      comm->tasks.collBytesTotal = 0;
      for (int i=0; i < comm->nRanks; i++) {
        ncclIntruQueueConstruct(&comm->tasks.peers[i].sendQueue);
        ncclIntruQueueConstruct(&comm->tasks.peers[i].recvQueue);
      }
      comm = next;
    }
  }

  while (!ncclIntruQueueEmpty(&ncclAsyncJobs)) {
    struct ncclAsyncJob* job = ncclIntruQueueDequeue(&ncclAsyncJobs);
    if (ret != ncclSuccess && jobsDone && job->undo) job->undo(job);
    if (job->destructor) job->destructor((void*)job);
  }

  ncclGroupError = ncclSuccess;
  ncclGroupCommHead = nullptr;
  ncclGroupCommPreconnectHead = nullptr;
  CUDACHECK(cudaSetDevice(savedDev)); // do other clean-ups first before calling cudaSetDevice, because this call can fail too
  return ret;
}
