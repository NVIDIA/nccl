/*************************************************************************
 * Copyright (c) 2016-2018, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "core.h"
#include "utils.h"
#include "transport.h"
#include "param.h"
#include "shm.h"
#include <unistd.h>
#include <cuda_runtime.h>

struct shmInfo {
  int rank;
  int cudaDev;
  uint64_t hostHash;
  uint64_t pidHash;
};

struct shmSendConnectInfo {
  uint64_t pidHash;
  int id;
  int rank;
  int shmSize;
};

struct shmRecvConnectInfo {
  uint64_t pidHash;
  int id;
  int rank;
  int shmSize;
};

struct shmSendResources {
  int remShmSize;
  struct ncclRecvMem* remHostMem;
  struct ncclRecvMem* devRemHostMem;
  int shmSize;
  struct ncclSendMem* hostMem;
  struct ncclSendMem* devHostMem;
};

struct shmRecvResources {
  int remShmSize;
  struct ncclSendMem* remHostMem;
  struct ncclSendMem* devRemHostMem;
  int shmSize;
  struct ncclRecvMem* hostMem;
  struct ncclRecvMem* devHostMem;
};

/* Fill information necessary to exchange between ranks to choose whether or not
 * to use this transport */
ncclResult_t shmFillInfo(ncclTinfo_t* opaqueInfo, int rank) {
  struct shmInfo* info = (struct shmInfo*)opaqueInfo;
  static_assert(sizeof(struct shmInfo) <= sizeof(ncclTinfo_t), "shm Info too large");
  info->rank = rank;
  CUDACHECK(cudaGetDevice(&info->cudaDev));
  info->hostHash=getHostHash();
  info->pidHash=getPidHash();
  return ncclSuccess;
}

NCCL_PARAM(ShmDisable, "SHM_DISABLE", 0);

/* Determine if we can communicate with the peer */
ncclResult_t shmCanConnect(ncclTvalue_t* ret, ncclTinfo_t* myOpaqueInfo, ncclTinfo_t* peerOpaqueInfo) {
  struct shmInfo* myInfo = (struct shmInfo*)myOpaqueInfo;
  struct shmInfo* peerInfo = (struct shmInfo*)peerOpaqueInfo;
  *ret = ((ncclParamShmDisable() == 1) || (myInfo->hostHash != peerInfo->hostHash)) ? 0 : 1;
  return ncclSuccess;
}

static inline int groupFirst(int nranks, int* groups, int group, int rankToAvoid) {
  for (int rank = 0; rank<nranks; rank++) {
    if ((groups[rank] == group) && (rank != rankToAvoid)) return rank;
  }
  return -1;
}

static inline int groupLast(int nranks, int* groups, int group, int rankToAvoid) {
  for (int rank = nranks-1; rank>=0; rank--) {
    if ((groups[rank] == group) && (rank != rankToAvoid)) return rank;
  }
  return -1;
}

ncclResult_t shmGetRings(int nranks, int* groups, int* subgroups, ncclTvalue_t* values, int* nringsRet, int* prev, int* next, int minScore, int* nthreads) {
  if (*nringsRet == MAXRINGS) *nringsRet = 1;
  int nGroups = groups[nranks-1] + 1;
  int starts[nGroups];
  int ends[nGroups];
  for (int ring = 0; ring<*nringsRet; ring++) {
    int startGroup = -1, endGroup = -1;
    for (int group = 0; group<nGroups; group++) {
      int start = -1;
      int end = -1;
      int nranksInGroup = 0;
      for (int rank=0; rank<nranks; rank++) {
        if (groups[rank] != group) continue;
        nranksInGroup++;
        if (prev[ring*nranks+rank] != -1) {
          if (start != -1) {
            WARN("Multiple starts found in group");
          }
          start = rank;
          startGroup = group;
        }
        if (next[ring*nranks+rank] != -1) {
          if (end != -1) {
            WARN("Multiple ends found in group");
          }
          end = rank;
          endGroup = group;
        }
      }
      if (nranksInGroup == 1) {
        start = end = groupFirst(nranks, groups, group, -1);
      } else {
        if (start == -1)
          start = groupFirst(nranks, groups, group, end);
        if (end == -1)
          end = groupLast(nranks, groups, group, start);
      }
      if (start == -1 || end == -1) {
        *nringsRet = ring;
        return ncclSuccess;
      }
      starts[group] = start;
      ends[group] = end;
    }
    if (endGroup == -1 || startGroup == -1) {
      startGroup = 0;
      endGroup = nGroups-1;
      // Close the loop
      next[ring*nranks+ends[endGroup]] = starts[startGroup];
      prev[ring*nranks+starts[startGroup]] = ends[endGroup];
    }
    int group = startGroup;
    for (int i=0; i<nGroups-2; i++) {
      int nextGroup = (group+1)%nGroups;
      if (nextGroup == endGroup) nextGroup = (nextGroup+1)%nGroups;
      next[ring*nranks+ends[group]] = starts[nextGroup];
      prev[ring*nranks+starts[nextGroup]] = ends[group];
      group = nextGroup;
    }
    // Connect with the last
    next[ring*nranks+ends[group]] = starts[endGroup];
    prev[ring*nranks+starts[endGroup]] = ends[group];
  }
  return ncclSuccess;
}

#define MAX_SHM_NAME_LEN 1024

/* Create and return connect structures for this peer to connect to me */
ncclResult_t shmSendSetup(ncclTinfo_t* myOpaqueInfo, ncclTinfo_t* peerOpaqueInfo, struct ncclConnect* connectInfo, struct ncclRing* ring) {
  struct shmInfo* myInfo = (struct shmInfo*)myOpaqueInfo;
  struct shmInfo* peerInfo = (struct shmInfo*)peerOpaqueInfo;

  struct shmSendResources* resources;
  NCCLCHECK(ncclCalloc(&resources, 1));
  ring->send.transportResources = resources;

  struct shmRecvConnectInfo info;
  char shmName[MAX_SHM_NAME_LEN];
  sprintf(shmName, "nccl-shm-send-%lx-%d-%d", myInfo->pidHash, ring->id, myInfo->rank);
  info.shmSize = resources->shmSize = sizeof(struct ncclSendMem);
  TRACE(NCCL_SHM,"Open shmName %s shmSize %d", shmName, info.shmSize);
  NCCLCHECK(shmOpen(shmName, resources->shmSize, (void**)&resources->hostMem, (void**)&resources->devHostMem, 1));

  INFO(NCCL_INIT|NCCL_SHM,"Ring %02d : %d[%d] -> %d[%d] via direct shared memory", ring->id, myInfo->rank, myInfo->cudaDev, peerInfo->rank, peerInfo->cudaDev);
  info.id = ring->id; info.rank = myInfo->rank; info.pidHash = myInfo->pidHash;
  static_assert(sizeof(struct shmRecvConnectInfo) <= sizeof(struct ncclConnect), "shm Connect Recv Info is too big");
  memcpy(connectInfo, &info, sizeof(struct shmRecvConnectInfo));
  return ncclSuccess;
}

ncclResult_t shmRecvSetup(ncclTinfo_t* myOpaqueInfo, ncclTinfo_t* peerOpaqueInfo, struct ncclConnect* connectInfo, struct ncclRing* ring) {
  struct shmInfo* myInfo = (struct shmInfo*)myOpaqueInfo;
  struct shmRecvResources* resources;
  NCCLCHECK(ncclCalloc(&resources, 1));
  ring->recv.transportResources = resources;

  struct shmSendConnectInfo info;

  char shmName[MAX_SHM_NAME_LEN];
  sprintf(shmName, "nccl-shm-recv-%lx-%d-%d", myInfo->pidHash, ring->id, myInfo->rank);
  info.shmSize = resources->shmSize = offsetof(struct ncclRecvMem, buff)+ring->buffSize;
  TRACE(NCCL_SHM,"Open shmName %s shmSize %d", shmName, info.shmSize);
  NCCLCHECK(shmOpen(shmName, resources->shmSize, (void**)&resources->hostMem, (void**)&resources->devHostMem, 1));

  info.id = ring->id; info.rank = myInfo->rank; info.pidHash = myInfo->pidHash;
  static_assert(sizeof(struct shmRecvConnectInfo) <= sizeof(struct ncclConnect), "shm Connect Send Info is too big");
  memcpy(connectInfo, &info, sizeof(struct shmSendConnectInfo));
  return ncclSuccess;
}

/* Connect to this peer */
ncclResult_t shmSendConnect(struct ncclConnect* connectInfo, struct ncclConnector* send) {
  // Setup device pointers
  struct shmSendConnectInfo* info = (struct shmSendConnectInfo*)connectInfo;
  struct shmSendResources* resources = (struct shmSendResources*)send->transportResources;

  char shmName[MAX_SHM_NAME_LEN];
  sprintf(shmName, "nccl-shm-recv-%lx-%d-%d", info->pidHash, info->id, info->rank);
  resources->remShmSize = info->shmSize;
  TRACE(NCCL_SHM,"Open shmName %s shmSize %d", shmName, info->shmSize);
  NCCLCHECK(shmOpen(shmName, resources->remShmSize, (void**)&resources->remHostMem, (void**)&resources->devRemHostMem, 0));
  // Remove the file to ensure proper clean-up
  NCCLCHECK(shmUnlink(shmName));

  send->transportResources = resources;
  send->conn.buff = resources->devRemHostMem->buff;
  send->conn.llBuff = resources->devRemHostMem->llBuff;
  send->conn.tail = &resources->devRemHostMem->tail;
  send->conn.opCount = &resources->devRemHostMem->opCount;

  send->conn.head = &resources->devHostMem->head;
  send->conn.llHead = &resources->devHostMem->llHead;
  return ncclSuccess;
}

ncclResult_t shmRecvConnect(struct ncclConnect* connectInfo, struct ncclConnector* recv) {
  // Setup device pointers
  struct shmRecvResources* resources = (struct shmRecvResources*)recv->transportResources;
  struct shmRecvConnectInfo* info = (struct shmRecvConnectInfo*)connectInfo;

  char shmName[MAX_SHM_NAME_LEN];
  sprintf(shmName, "nccl-shm-send-%lx-%d-%d", info->pidHash, info->id, info->rank);
  resources->remShmSize = info->shmSize;
  TRACE(NCCL_SHM,"Open shmName %s shmSize %d", shmName, info->shmSize);
  NCCLCHECK(shmOpen(shmName, resources->remShmSize, (void**)&resources->remHostMem, (void**)&resources->devRemHostMem, 0));
  NCCLCHECK(shmUnlink(shmName));
  recv->conn.head = &resources->devRemHostMem->head;
  recv->conn.llHead = &resources->devRemHostMem->llHead;

  recv->conn.buff = resources->devHostMem->buff;
  recv->conn.llBuff = resources->devHostMem->llBuff;
  recv->conn.tail = &resources->devHostMem->tail;
  recv->conn.opCount = &resources->devHostMem->opCount;
  return ncclSuccess;
}

ncclResult_t shmSendFree(void* transportResources) {
  struct shmSendResources* resources = (struct shmSendResources*)transportResources;
  NCCLCHECK(shmClose(resources->hostMem, resources->devHostMem, resources->shmSize));
  NCCLCHECK(shmClose(resources->remHostMem, resources->devRemHostMem, resources->remShmSize));
  free(resources);
  return ncclSuccess;
}

ncclResult_t shmRecvFree(void* transportResources) {
  struct shmRecvResources* resources = (struct shmRecvResources*)transportResources;
  NCCLCHECK(shmClose(resources->hostMem, resources->devHostMem, resources->shmSize));
  NCCLCHECK(shmClose(resources->remHostMem, resources->devRemHostMem, resources->remShmSize));
  free(resources);
  return ncclSuccess;
}

struct ncclTransport shmTransport = {
  "SHM",
  shmFillInfo,
  shmCanConnect,
  shmGetRings,
  { shmSendSetup, shmSendConnect, shmSendFree, NULL },
  { shmRecvSetup, shmRecvConnect, shmRecvFree, NULL }
};
