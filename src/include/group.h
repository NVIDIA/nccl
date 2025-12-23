/*************************************************************************
 * Copyright (c) 2015-2017, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_GROUP_H_
#define NCCL_GROUP_H_

#include "nccl.h"
#include "comm.h"
#include "allocator.h"
#include "register.h"
#include "utils.h"

#include <thread>

ncclResult_t ncclGroupErrCheck(ncclResult_t ret);
void ncclGroupCommJoin(struct ncclComm* comm, int type);
void ncclGroupCommPreconnect(struct ncclComm* comm);
ncclResult_t ncclGroupCommLeave(struct ncclComm* comm);
ncclResult_t ncclGroupJobAbort(struct ncclGroupJob* groupJob);
ncclResult_t ncclGroupJobComplete(struct ncclGroupJob *groupJob);

typedef ncclResult_t(*ncclInitFunc_t)(ncclComm_t* newcomm, int ndev, ncclUniqueId commId, int myrank, int cudaDev);

ncclResult_t ncclAsyncInit(ncclInitFunc_t func, ncclComm_t* newcomm, int ndev, ncclUniqueId commId, int myrank, int cudaDev);

typedef enum ncclGroupJobState {
  ncclGroupJobRunning = 0,
  ncclGroupJobDone    = 1,
  ncclGroupJobJoined  = 2,
} ncclGroupJobState_t;

struct ncclAsyncJob {
  struct ncclAsyncJob* next;
  std::thread thread;
  ncclResult_t result;
  ncclResult_t(*func)(struct ncclAsyncJob*);
  void(*undo)(struct ncclAsyncJob*);
  void(*destructor)(void*);
  ncclGroupJobState_t state;
  uint32_t* abortFlag; /* point to comm abortFlag */
  uint32_t* abortFlagDev; /* point to comm abortFlagDev */
  uint32_t* childAbortFlag; /* point to child abortFlag */
  uint32_t* childAbortFlagDev; /* point to child abortFlagDev */
  ncclComm_t comm;
  int destroyFlag;
  bool isThreadMain;

  ~ncclAsyncJob() {
    if (thread.joinable()) {
      (void)ncclThreadJoin(thread);
    }
  }
};

ncclResult_t ncclAsyncLaunch(
  struct ncclAsyncJob* job,
  ncclResult_t(*func)(struct ncclAsyncJob*),
  void(*undo)(struct ncclAsyncJob*),
  void(*destructor)(void*), ncclComm_t comm
);

struct ncclGroupJob {
  struct ncclAsyncJob base;
  int groupRefCount;
  bool nonBlockingInit;
  bool joined;
  struct ncclComm *groupCommHead[ncclGroupTaskTypeNum];
  struct ncclComm *groupCommPreconnectHead;
  ncclResult_t groupError;
  bool abortFlag;
  struct ncclIntruQueue<struct ncclAsyncJob, &ncclAsyncJob::next> asyncJobs;
};

ncclResult_t ncclGroupStartInternal();
ncclResult_t ncclGroupEndInternal(ncclSimInfo_t* simInfo = NULL);
ncclResult_t ncclAsyncJobComplete(struct ncclAsyncJob* job);

////////////////////////////////////////////////////////////////////////////////

extern thread_local int ncclGroupDepth; // depth of ncclGroupStart nesting
extern thread_local ncclResult_t ncclGroupError;
extern thread_local struct ncclComm* ncclGroupCommHead[ncclGroupTaskTypeNum];
extern thread_local struct ncclComm* ncclGroupCommPreconnectHead;
extern thread_local int ncclGroupBlocking;

inline ncclResult_t ncclGroupStartInternal() {
  ncclGroupDepth++;
  return ncclSuccess;
}

inline bool ncclGroupEnabled() {
  return ncclGroupDepth != 0;
}

inline ncclResult_t ncclGroupErrCheck(ncclResult_t ret) {
  if (ncclGroupDepth > 0) {
    if (ret != ncclSuccess && ret != ncclInProgress) ncclGroupError = ret;
  }
  return ret;
}

// Add comm to this thread's group
inline void ncclGroupCommJoin(struct ncclComm* comm, int type) {
  if (comm->groupNext[type] == reinterpret_cast<struct ncclComm*>(0x1)) {
    // Insert comm into ncclGroupCommHead adjacent to sibling comms. This preserves
    // the users program order yet insures siblings occur consecutively. This
    // is required by doLaunches() in "group.cc".
    struct ncclComm** pp = &ncclGroupCommHead[type];
    while (*pp != nullptr && comm->intraComm0 != (*pp)->intraComm0)
      pp = &(*pp)->groupNext[type];

    // didn't find its clique, we need to insert it with ascending order based on commHash
    if (*pp == nullptr) {
      pp = &ncclGroupCommHead[type];
      while (*pp != nullptr && (*pp)->commHash < comm->commHash) pp = &(*pp)->groupNext[type];
    }
    comm->groupNext[type] = *pp;
    *pp = comm;
    // Comms gets a new memory stack scope upon joining. Each task batched for
    // this comm is allocated there.
    ncclMemoryStackPush(&comm->memScoped);
    if (type == ncclGroupTaskTypeCollective) {
      // Initialize planner
      ncclKernelPlanner::Peer* tmp = comm->planner.peers;
      ncclIntruQueue<ncclTaskRma, &ncclTaskRma::next>* tmpRmaQueues = comm->planner.rmaTaskQueues;
      int numRmaCtx = comm->config.numRmaCtx;
      memset(&comm->planner, 0, sizeof(comm->planner));
      comm->planner.peers = tmp;
      comm->planner.bcast_info.minBcastPeer = INT_MAX;
      comm->planner.bcast_info.maxBcastPeer = INT_MIN;
      comm->planner.rmaTaskQueues = tmpRmaQueues;
      if (comm->planner.rmaTaskQueues != NULL) {
        for (int i = 0; i < numRmaCtx; i++) {
          ncclIntruQueueConstruct(&comm->planner.rmaTaskQueues[i]);
        }
      }
    }
  }
  ncclGroupBlocking = comm->config.blocking;
}

// Add comm to this thread's group needing preconnect
inline void ncclGroupCommPreconnect(struct ncclComm* comm) {
  if (comm->preconnectNext == reinterpret_cast<struct ncclComm*>(0x1)) {
    comm->preconnectNext = ncclGroupCommPreconnectHead;
    ncclGroupCommPreconnectHead = comm;
  }
}

// Comm has left group
inline ncclResult_t ncclGroupCommLeave(struct ncclComm* comm, int type) {
  comm->groupNext[type] = reinterpret_cast<struct ncclComm*>(0x1);
  ncclMemoryStackPop(&comm->memScoped);
  return ncclSuccess;
}

#endif
