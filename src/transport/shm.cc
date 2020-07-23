/*************************************************************************
 * Copyright (c) 2016-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "comm.h"
#include "shm.h"

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

/* Determine two peers can communicate with SHM */
ncclResult_t shmCanConnect(int* ret, struct ncclTopoSystem* topo, struct ncclTopoGraph* graph, struct ncclPeerInfo* info1, struct ncclPeerInfo* info2) {
  *ret = 0;

  if (ncclParamShmDisable() == 1) return ncclSuccess;

  // Same host?
  TRACE(NCCL_INIT|NCCL_SHM, "peer1 hostHash %lx peer2 hostHash %lx", info1->hostHash, info2->hostHash);
  if (info1->hostHash != info2->hostHash) return ncclSuccess;

  // Common /dev/shm (between containers) ?
  TRACE(NCCL_INIT|NCCL_SHM, "peer1 shmDev %lx peer2 shmDev %lx", info1->shmDev, info2->shmDev);
  if (info1->shmDev != info2->shmDev) return ncclSuccess;

  *ret = 1;

  return ncclSuccess;
}

#define MAX_SHM_NAME_LEN 1024

/* Create and return connect structures for this peer to connect to me */
ncclResult_t shmSendSetup(struct ncclTopoSystem* topo, struct ncclTopoGraph* graph, struct ncclPeerInfo* myInfo, struct ncclPeerInfo* peerInfo, struct ncclConnect* connectInfo, struct ncclConnector* send, int channelId) {

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

  INFO(NCCL_INIT|NCCL_SHM,"Channel %02d : %d[%lx] -> %d[%lx] via direct shared memory", channelId, myInfo->rank, myInfo->busId, peerInfo->rank, peerInfo->busId);
  static_assert(sizeof(struct shmConnectInfo) <= sizeof(struct ncclConnect), "shm Connect Recv Info is too big");
  memcpy(connectInfo, &info, sizeof(struct shmConnectInfo));
  return ncclSuccess;
}

ncclResult_t shmRecvSetup(struct ncclTopoSystem* topo, struct ncclTopoGraph* graph, struct ncclPeerInfo* myInfo, struct ncclPeerInfo* peerInfo, struct ncclConnect* connectInfo, struct ncclConnector* recv, int channelId) {
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
  int shmSize = offsetof(struct ncclRecvMem, buff);
  for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) shmSize += recv->comm->buffSizes[p];
  info.shmSize = resources->shmSize = shmSize;
  TRACE(NCCL_SHM,"Open shmName %s shmSize %d", shmName, info.shmSize);
  NCCLCHECK(shmOpen(shmName, resources->shmSize, (void**)&resources->hostMem, (void**)&resources->devHostMem, 1));

  static_assert(sizeof(struct shmConnectInfo) <= sizeof(struct ncclConnect), "shm Connect Send Info is too big");
  memcpy(connectInfo, &info, sizeof(struct shmConnectInfo));
  return ncclSuccess;
}

/* Connect to this peer */
ncclResult_t shmSendConnect(struct ncclConnect* connectInfo, int nranks, int rank, struct ncclConnector* send) {
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
  int offset = 0;
  for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
    send->conn.buffs[p] = resources->devRemHostMem->buff + offset;
    offset += send->comm->buffSizes[p];
  }
  send->conn.tail = &resources->devRemHostMem->tail;

  send->conn.head = &resources->devHostMem->head;
  return ncclSuccess;
}

ncclResult_t shmRecvConnect(struct ncclConnect* connectInfo, int nranks, int rank, struct ncclConnector* recv) {
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

  int offset = 0;
  for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
    recv->conn.buffs[p] = resources->devHostMem->buff + offset;
    offset += recv->comm->buffSizes[p];
  }
  recv->conn.tail = &resources->devHostMem->tail;
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
  { shmSendSetup, shmSendConnect, shmSendFree, NULL },
  { shmRecvSetup, shmRecvConnect, shmRecvFree, NULL }
};
