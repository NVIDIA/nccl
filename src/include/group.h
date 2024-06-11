/*************************************************************************
 * Copyright (c) 2015-2017, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_GROUP_H_
#define NCCL_GROUP_H_

#include "nccl.h"
#include "comm.h"

ncclResult_t ncclGroupErrCheck(ncclResult_t ret);
void ncclGroupCommJoin(struct ncclComm* comm);
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
  pthread_t thread;
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
};

ncclResult_t ncclAsyncLaunch(
  struct ncclAsyncJob* job,
  ncclResult_t(*func)(struct ncclAsyncJob*),
  void(*undo)(struct ncclAsyncJob*),
  void(*destructor)(void*), ncclComm_t comm
);

struct ncclGroupJob {
  struct ncclAsyncJob base;
  struct ncclComm **groupCommHeadPtr;
  struct ncclComm **groupCommPreconnectHeadPtr;
  ncclResult_t *groupErrorPtr;
  bool *abortFlagPtr;
  int *groupBlockingPtr;
  struct ncclIntruQueue<struct ncclAsyncJob, &ncclAsyncJob::next> *asyncJobsPtr;
  bool initialized;
};

ncclResult_t ncclGroupStartInternal();
ncclResult_t ncclGroupEndInternal(ncclSimInfo_t* simInfo = NULL);
ncclResult_t ncclAsyncJobComplete(struct ncclAsyncJob* job);

////////////////////////////////////////////////////////////////////////////////

extern __thread int ncclGroupDepth; // depth of ncclGroupStart nesting
extern __thread ncclResult_t ncclGroupError;
extern __thread struct ncclComm* ncclGroupCommHead;
extern __thread struct ncclComm* ncclGroupCommPreconnectHead;
extern __thread int ncclGroupBlocking;
extern __thread struct ncclGroupJob *ncclGroupJobMainPtr;
extern __thread struct ncclGroupJob ncclGroupJobMain;

static inline void groupResetJobState() {
  ncclGroupBlocking = -1;
  ncclGroupJobMainPtr = NULL;
  memset(&ncclGroupJobMain, 0, sizeof(struct ncclGroupJob));
  return;
}

static inline ncclResult_t groupJobComplete(struct ncclGroupJob* job) {
  ncclResult_t ret = ncclSuccess;
  if (job) {
    ret = ncclAsyncJobComplete(&job->base);
    groupResetJobState();
  }
  return ret;
}

inline ncclResult_t ncclGroupStartInternal() {
  ncclGroupDepth++;
  return ncclSuccess;
}

inline ncclResult_t ncclGroupErrCheck(ncclResult_t ret) {
  if (ncclGroupDepth > 0) {
    if (ret != ncclSuccess && ret != ncclInProgress) ncclGroupError = ret;
  }
  return ret;
}

// Add comm to this thread's group
inline void ncclGroupCommJoin(struct ncclComm* comm) {
  if (comm->groupNext == reinterpret_cast<struct ncclComm*>(0x1)) {
    // Insert comm into ncclGroupCommHead adjacent to sibling comms. This preserves
    // the users program order yet insures siblings occur consecutively. This
    // is required by doLaunches() in "group.cc".
    struct ncclComm** pp = &ncclGroupCommHead;
    while (*pp != nullptr && comm->intraComm0 != (*pp)->intraComm0)
      pp = &(*pp)->groupNext;
    comm->groupNext = *pp;
    *pp = comm;
    // Comms gets a new memory stack scope upon joining. Each task batched for
    // this comm is allocated there.
    ncclMemoryStackPush(&comm->memScoped);
    // Initialize planner
    ncclKernelPlanner::Peer* tmp = comm->planner.peers;
    memset(&comm->planner, 0, sizeof(comm->planner));
    comm->planner.peers = tmp;
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
inline ncclResult_t ncclGroupCommLeave(struct ncclComm* comm) {
  comm->groupNext = reinterpret_cast<struct ncclComm*>(0x1);
  ncclMemoryStackPop(&comm->memScoped);
  return ncclSuccess;
}

#endif
