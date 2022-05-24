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
void ncclGroupCommLeave(struct ncclComm* comm);

typedef ncclResult_t(*ncclInitFunc_t)(ncclComm_t* newcomm, int ndev, ncclUniqueId commId, int myrank, int cudaDev);

ncclResult_t ncclAsyncInit(ncclInitFunc_t func, ncclComm_t* newcomm, int ndev, ncclUniqueId commId, int myrank, int cudaDev);

struct ncclAsyncJob {
  struct ncclAsyncJob* next;
  pthread_t thread;
  ncclResult_t result;
  ncclResult_t(*func)(struct ncclAsyncJob*);
  void(*undo)(struct ncclAsyncJob*);
  void(*destructor)(void*);
};

ncclResult_t ncclAsyncLaunch(
  struct ncclAsyncJob* job,
  ncclResult_t(*func)(struct ncclAsyncJob*),
  void(*undo)(struct ncclAsyncJob*),
  void(*destructor)(void*)
);

ncclResult_t ncclGroupStartInternal();
ncclResult_t ncclGroupEndInternal();

////////////////////////////////////////////////////////////////////////////////

extern __thread int ncclGroupDepth; // depth of ncclGroupStart nesting
extern __thread ncclResult_t ncclGroupError;
extern __thread struct ncclComm* ncclGroupCommHead;
extern __thread struct ncclComm* ncclGroupCommPreconnectHead;

inline ncclResult_t ncclGroupStartInternal() {
  ncclGroupDepth++;
  return ncclSuccess;
}

inline ncclResult_t ncclGroupErrCheck(ncclResult_t ret) {
  if (ncclGroupDepth > 0) {
    if (ncclGroupError == ncclSuccess || ret != ncclSuccess) ncclGroupError = ret;
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
  }
}

// Add comm to this thread's group needing preconnect
inline void ncclGroupCommPreconnect(struct ncclComm* comm) {
  if (comm->preconnectNext == reinterpret_cast<struct ncclComm*>(0x1)) {
    comm->preconnectNext = ncclGroupCommPreconnectHead;
    ncclGroupCommPreconnectHead = comm;
  }
}

// Comm has left group
inline void ncclGroupCommLeave(struct ncclComm* comm) {
  comm->groupNext = reinterpret_cast<struct ncclComm*>(0x1);
  ncclMemoryStackPop(&comm->memScoped);
}

#endif
