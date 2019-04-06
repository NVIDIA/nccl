/*************************************************************************
 * Copyright (c) 2016-2019, NVIDIA CORPORATION. All rights reserved.
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

struct shmConnectInfo {
  uint64_t pidHash;
  int id;
  int sendRank;
  int recvRank;
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

NCCL_PARAM(ShmDisable, "SHM_DISABLE", 0);

/* Determine if we can communicate with the peer */
ncclResult_t shmCanConnect(ncclTvalue_t* ret, struct ncclPeerInfo* myInfo, struct ncclPeerInfo* peerInfo) {
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

#define MAXGROUPS 16

ncclResult_t shmGetRings(int nranks, int* groups, int* subgroups, ncclTvalue_t* values, int* nringsRet, int* prev, int* next, int minScore, int* nthreads) {
  if (*nringsRet == MAXCHANNELS) *nringsRet = 1;
  int nGroups = groups[nranks-1] + 1;
  int starts[MAXGROUPS];
  int ends[MAXGROUPS];
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
ncclResult_t shmSendSetup(struct ncclPeerInfo* myInfo, struct ncclPeerInfo* peerInfo, struct ncclConnect* connectInfo, struct ncclConnector* send, int buffSize, int channelId) {

  struct shmSendResources* resources;
  NCCLCHECK(ncclCalloc(&resources, 1));
  send->transportResources = resources;

  struct shmConnectInfo info;
  info.id = channelId;
  info.pidHash = myInfo->pidHash;
  info.sendRank = myInfo->rank;
  info.recvRank = peerInfo->rank;

  char shmName[MAX_SHM_NAME_LEN];
  sprintf(shmName, "nccl-shm-send-%lx-%d-%d-%d", info.pidHash, info.id, info.sendRank, info.recvRank);
  info.shmSize = resources->shmSize = sizeof(struct ncclSendMem);
  TRACE(NCCL_SHM,"Open shmName %s shmSize %d", shmName, info.shmSize);
  NCCLCHECK(shmOpen(shmName, resources->shmSize, (void**)&resources->hostMem, (void**)&resources->devHostMem, 1));

  INFO(NCCL_INIT|NCCL_SHM,"Ring %02d : %d[%d] -> %d[%d] via direct shared memory", channelId, myInfo->rank, myInfo->cudaDev, peerInfo->rank, peerInfo->cudaDev);
  static_assert(sizeof(struct shmConnectInfo) <= sizeof(struct ncclConnect), "shm Connect Recv Info is too big");
  memcpy(connectInfo, &info, sizeof(struct shmConnectInfo));
  return ncclSuccess;
}

ncclResult_t shmRecvSetup(struct ncclPeerInfo* myInfo, struct ncclPeerInfo* peerInfo, struct ncclConnect* connectInfo, struct ncclConnector* recv, int buffSize, int channelId) {
  struct shmRecvResources* resources;
  NCCLCHECK(ncclCalloc(&resources, 1));
  recv->transportResources = resources;

  struct shmConnectInfo info;
  info.id = channelId;
  info.pidHash = myInfo->pidHash;
  info.sendRank = peerInfo->rank;
  info.recvRank = myInfo->rank;

  char shmName[MAX_SHM_NAME_LEN];
  sprintf(shmName, "nccl-shm-recv-%lx-%d-%d-%d", info.pidHash, info.id, info.sendRank, info.recvRank);
  info.shmSize = resources->shmSize = offsetof(struct ncclRecvMem, buff)+buffSize;
  TRACE(NCCL_SHM,"Open shmName %s shmSize %d", shmName, info.shmSize);
  NCCLCHECK(shmOpen(shmName, resources->shmSize, (void**)&resources->hostMem, (void**)&resources->devHostMem, 1));

  static_assert(sizeof(struct shmConnectInfo) <= sizeof(struct ncclConnect), "shm Connect Send Info is too big");
  memcpy(connectInfo, &info, sizeof(struct shmConnectInfo));
  return ncclSuccess;
}

/* Connect to this peer */
ncclResult_t shmSendConnect(struct ncclConnect* connectInfo, struct ncclConnector* send) {
  // Setup device pointers
  struct shmConnectInfo* info = (struct shmConnectInfo*)connectInfo;
  struct shmSendResources* resources = (struct shmSendResources*)send->transportResources;

  char shmName[MAX_SHM_NAME_LEN];
  sprintf(shmName, "nccl-shm-recv-%lx-%d-%d-%d", info->pidHash, info->id, info->sendRank, info->recvRank);
  resources->remShmSize = info->shmSize;
  TRACE(NCCL_SHM,"Open shmName %s shmSize %d", shmName, info->shmSize);
  NCCLCHECK(shmOpen(shmName, resources->remShmSize, (void**)&resources->remHostMem, (void**)&resources->devRemHostMem, 0));
  // Remove the file to ensure proper clean-up
  NCCLCHECK(shmUnlink(shmName));

  send->transportResources = resources;
  send->conn.buff = resources->devRemHostMem->buff;
  send->conn.llBuff = resources->devRemHostMem->llBuff;
  send->conn.tail = &resources->devRemHostMem->tail;
  send->conn.opCountRem = &resources->devRemHostMem->opCount;

  send->conn.head = &resources->devHostMem->head;
  send->conn.opCountLoc = &resources->devHostMem->opCount;
  return ncclSuccess;
}

ncclResult_t shmRecvConnect(struct ncclConnect* connectInfo, struct ncclConnector* recv) {
  // Setup device pointers
  struct shmRecvResources* resources = (struct shmRecvResources*)recv->transportResources;
  struct shmConnectInfo* info = (struct shmConnectInfo*)connectInfo;

  char shmName[MAX_SHM_NAME_LEN];
  sprintf(shmName, "nccl-shm-send-%lx-%d-%d-%d", info->pidHash, info->id, info->sendRank, info->recvRank);
  resources->remShmSize = info->shmSize;
  TRACE(NCCL_SHM,"Open shmName %s shmSize %d", shmName, info->shmSize);
  NCCLCHECK(shmOpen(shmName, resources->remShmSize, (void**)&resources->remHostMem, (void**)&resources->devRemHostMem, 0));
  NCCLCHECK(shmUnlink(shmName));
  recv->conn.head = &resources->devRemHostMem->head;
  recv->conn.opCountRem = &resources->devRemHostMem->opCount;

  recv->conn.buff = resources->devHostMem->buff;
  recv->conn.llBuff = resources->devHostMem->llBuff;
  recv->conn.tail = &resources->devHostMem->tail;
  recv->conn.opCountLoc = &resources->devHostMem->opCount;
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
  shmCanConnect,
  shmGetRings,
  { shmSendSetup, shmSendConnect, shmSendFree, NULL },
  { shmRecvSetup, shmRecvConnect, shmRecvFree, NULL }
};
