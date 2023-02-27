/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "comm.h"
#include "shm.h"

struct shmConnectInfo {
  char shmName[7];
  int shmSize;
};
static_assert(sizeof(shmConnectInfo) <= CONNECT_SIZE, "SHM Connect info is too large");

struct shmSendResources {
  int remShmSize;
  struct ncclRecvMem* remHostMem;
  struct ncclRecvMem* devRemHostMem;
  ncclShmHandle_t remHandle;
  int shmSize;
  struct ncclSendMem* hostMem;
  struct ncclSendMem* devHostMem;
  ncclShmHandle_t hostHandle;
};

struct shmRecvResources {
  int remShmSize;
  struct ncclSendMem* remHostMem;
  struct ncclSendMem* devRemHostMem;
  ncclShmHandle_t remHandle;
  int shmSize;
  struct ncclRecvMem* hostMem;
  struct ncclRecvMem* devHostMem;
  ncclShmHandle_t hostHandle;
};

#define SHM_SEND_SIDE 1
#define SHM_RECV_SIDE 2
NCCL_PARAM(ShmDisable, "SHM_DISABLE", 0);
NCCL_PARAM(ShmUseCudaMemcpy, "SHM_USE_CUDA_MEMCPY", 0);
NCCL_PARAM(ShmMemcpyMode, "SHM_MEMCPY_MODE", SHM_SEND_SIDE); // 1 is sender-side, 2 is receiver-side, 3 is both
static int useMemcpySend = 0;
static int useMemcpyRecv = 0;
NCCL_PARAM(ShmLocality, "SHM_LOCALITY", SHM_RECV_SIDE); // 1 is sender-size, 2 is receiver-size
static int shmLocality = 0;
static void initCeOperation();

/* Determine two peers can communicate with SHM */
static ncclResult_t shmCanConnect(int* ret, struct ncclTopoSystem* topo, struct ncclTopoGraph* graph, struct ncclPeerInfo* info1, struct ncclPeerInfo* info2) {
  *ret = 0;
  initCeOperation();

  if (ncclParamShmDisable() == 1) return ncclSuccess;

  int useNet = 0;
  NCCLCHECK(ncclTopoCheckNet(topo, info1->busId, info2->busId, &useNet));
  if (useNet) return ncclSuccess;

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
static ncclResult_t shmSendSetup(struct ncclComm* comm, struct ncclTopoGraph* graph, struct ncclPeerInfo* myInfo, struct ncclPeerInfo* peerInfo, struct ncclConnect* connectInfo, struct ncclConnector* send, int channelId, int connIndex) {
  struct shmSendResources* resources;
  NCCLCHECK(ncclCalloc(&resources, 1));
  send->transportResources = resources;

  static_assert(sizeof(struct shmConnectInfo) <= sizeof(struct ncclConnect), "shm Connect Info is too big");
  struct shmConnectInfo* info = (struct shmConnectInfo*)connectInfo;

  char shmPath[PATH_MAX];
  shmPath[0] = '\0';
  int shmSize = sizeof(struct ncclSendMem);
  if (shmLocality == SHM_SEND_SIDE) {
    for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) shmSize += send->comm->buffSizes[p];
  }
  info->shmSize = resources->shmSize = shmSize;
  NCCLCHECK(ncclShmOpen(shmPath, resources->shmSize, (void**)&resources->hostMem, (void**)&resources->devHostMem, 1, &resources->hostHandle));
  TRACE(NCCL_SHM,"Opened shmName %s shmSize %d", shmPath, info->shmSize);
  memcpy(info->shmName, shmPath+sizeof("/dev/shm/nccl-")-1, sizeof(info->shmName));

  INFO(NCCL_INIT|NCCL_SHM,"Channel %02d : %d[%lx] -> %d[%lx] via SHM/%s/%s", channelId, myInfo->rank, myInfo->busId, peerInfo->rank, peerInfo->busId, useMemcpySend?"CE":"direct", useMemcpyRecv?"CE":"direct");
  return ncclSuccess;
}

static ncclResult_t shmRecvSetup(struct ncclComm* comm, struct ncclTopoGraph* graph, struct ncclPeerInfo* myInfo, struct ncclPeerInfo* peerInfo, struct ncclConnect* connectInfo, struct ncclConnector* recv, int channelId, int connIndex) {
  struct shmRecvResources* resources;
  NCCLCHECK(ncclCalloc(&resources, 1));
  recv->transportResources = resources;

  static_assert(sizeof(struct shmConnectInfo) <= sizeof(struct ncclConnect), "shm Connect Info is too big");
  struct shmConnectInfo* info = (struct shmConnectInfo*)connectInfo;

  char shmPath[PATH_MAX];
  shmPath[0] = '\0';
  int shmSize = sizeof(struct ncclRecvMem);
  if (shmLocality == SHM_RECV_SIDE) {
    for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) shmSize += recv->comm->buffSizes[p];
  }
  info->shmSize = resources->shmSize = shmSize;
  NCCLCHECK(ncclShmOpen(shmPath, resources->shmSize, (void**)&resources->hostMem, (void**)&resources->devHostMem, 1, &resources->hostHandle));
  TRACE(NCCL_SHM,"Opened shmName %s shmSize %d", shmPath, info->shmSize);
  memcpy(info->shmName, shmPath+sizeof("/dev/shm/nccl-")-1, sizeof(info->shmName));

  return ncclSuccess;
}

struct shmProxyInfo {
  struct ncclRecvMem* ceRecvMem;
  char* devFifo;
  char* shmFifo;
  struct ncclSendMem* sendMem;
  struct ncclRecvMem* recvMem;

  // used by progress only
  uint64_t step;
  cudaStream_t stream;
  cudaEvent_t events[NCCL_STEPS];
};

/* Connect to this peer */
static ncclResult_t shmSendConnect(struct ncclComm* comm, struct ncclConnect* connectInfo, int nranks, int rank, struct ncclConnector* send) {
  // Setup device pointers
  struct shmConnectInfo* info = (struct shmConnectInfo*)connectInfo;
  struct shmSendResources* resources = (struct shmSendResources*)send->transportResources;

  char shmPath[PATH_MAX];
  sprintf(shmPath, "/dev/shm/nccl-%s", info->shmName);
  resources->remShmSize = info->shmSize;
  TRACE(NCCL_SHM,"Open shmName %s shmSize %d", shmPath, info->shmSize);
  NCCLCHECK(ncclShmOpen(shmPath, resources->remShmSize, (void**)&resources->remHostMem, (void**)&resources->devRemHostMem, -1, &resources->remHandle));

  char* buff = shmLocality == SHM_SEND_SIDE ? (char*)(resources->devHostMem+1) : (char*)(resources->devRemHostMem+1);
  for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
    send->conn.buffs[p] = buff;
    buff += send->comm->buffSizes[p];
  }
  send->conn.tail = &resources->devRemHostMem->tail;
  send->conn.head = &resources->devHostMem->head;

  if (useMemcpyRecv) {
    send->conn.sizesFifo = resources->devRemHostMem->sizesFifo;
  }
  if (useMemcpySend) {
    NCCLCHECK(ncclProxyConnect(comm, TRANSPORT_SHM, 1, comm->rank, &send->proxyConn));
    struct shmProxyInfo proxyInfo = { NULL, NULL, send->conn.buffs[NCCL_PROTO_SIMPLE], resources->hostMem, resources->remHostMem };
    NCCLCHECK(ncclProxyCallBlocking(&send->proxyConn, ncclProxyMsgConnect, &proxyInfo, sizeof(struct shmProxyInfo), &proxyInfo, sizeof(struct shmProxyInfo)));
    send->conn.buffs[NCCL_PROTO_SIMPLE] = proxyInfo.devFifo;
    send->conn.tail = &proxyInfo.ceRecvMem->tail;
    send->conn.sizesFifo = proxyInfo.ceRecvMem->sizesFifo;
  }
  return ncclSuccess;
}

static ncclResult_t shmRecvConnect(struct ncclComm* comm, struct ncclConnect* connectInfo, int nranks, int rank, struct ncclConnector* recv) {
  // Setup device pointers
  struct shmRecvResources* resources = (struct shmRecvResources*)recv->transportResources;
  struct shmConnectInfo* info = (struct shmConnectInfo*)connectInfo;

  char shmPath[PATH_MAX];
  sprintf(shmPath, "/dev/shm/nccl-%s", info->shmName);
  resources->remShmSize = info->shmSize;
  TRACE(NCCL_SHM,"Open shmName %s shmSize %d", shmPath, info->shmSize);
  NCCLCHECK(ncclShmOpen(shmPath, resources->remShmSize, (void**)&resources->remHostMem, (void**)&resources->devRemHostMem, -1, &resources->remHandle));

  char* buff = shmLocality == SHM_RECV_SIDE ? (char*)(resources->devHostMem+1) : (char*)(resources->devRemHostMem+1);
  for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
    recv->conn.buffs[p] = buff;
    buff += recv->comm->buffSizes[p];
  }
  recv->conn.head = &resources->devRemHostMem->head;
  recv->conn.tail = &resources->devHostMem->tail;

  if (useMemcpyRecv) {
    NCCLCHECK(ncclProxyConnect(comm, TRANSPORT_SHM, 0, comm->rank, &recv->proxyConn));
    struct shmProxyInfo proxyInfo = { NULL, NULL, recv->conn.buffs[NCCL_PROTO_SIMPLE], resources->remHostMem, resources->hostMem };
    NCCLCHECK(ncclProxyCallBlocking(&recv->proxyConn, ncclProxyMsgConnect, &proxyInfo, sizeof(struct shmProxyInfo), &proxyInfo, sizeof(struct shmProxyInfo)));
    recv->conn.buffs[NCCL_PROTO_SIMPLE] = proxyInfo.devFifo;
    recv->conn.tail = &proxyInfo.ceRecvMem->tail;
  }
  return ncclSuccess;
}

static ncclResult_t shmSendFree(struct ncclConnector* send) {
  struct shmRecvResources* resources = (struct shmRecvResources*)send->transportResources;
  if (resources) {
    NCCLCHECK(ncclShmClose(resources->hostHandle));
    NCCLCHECK(ncclShmClose(resources->remHandle));
    free(resources);
  }
  return ncclSuccess;
}

static ncclResult_t shmRecvFree(struct ncclConnector* recv) {
  struct shmRecvResources* resources = (struct shmRecvResources*)recv->transportResources;
  if (resources) {
    NCCLCHECK(ncclShmClose(resources->hostHandle));
    NCCLCHECK(ncclShmClose(resources->remHandle));
    free(resources);
  }
  return ncclSuccess;
}

static ncclResult_t shmSendProxyConnect(struct ncclProxyConnection* connection, struct ncclComm* comm, void* reqBuff, int reqSize, void* respBuff, int respSize, int* done) {
  struct shmProxyInfo* proxyInfo;
  NCCLCHECK(ncclCalloc(&proxyInfo, 1));
  if (reqSize != sizeof(struct shmProxyInfo)) return ncclInternalError;
  memcpy(proxyInfo, reqBuff, reqSize);
  NCCLCHECK(ncclCudaCalloc(&proxyInfo->devFifo, comm->buffSizes[NCCL_PROTO_SIMPLE]));
  NCCLCHECK(ncclCudaHostCalloc(&proxyInfo->ceRecvMem, 1));
  CUDACHECK(cudaStreamCreateWithFlags(&proxyInfo->stream, cudaStreamNonBlocking));
  for (int i=0; i<NCCL_STEPS; i++) {
    CUDACHECK(cudaEventCreate(proxyInfo->events+i));
  }
  connection->proxyAppendPtr = &connection->proxyAppend;
  connection->transportResources = proxyInfo;
  if (respSize != sizeof(struct shmProxyInfo)) return ncclInternalError;
  memcpy(respBuff, proxyInfo, respSize);
  return ncclSuccess;
}

static ncclResult_t shmRecvProxyConnect(struct ncclProxyConnection* connection, struct ncclComm* comm, void* reqBuff, int reqSize, void* respBuff, int respSize, int* done) {
  struct shmProxyInfo* proxyInfo;
  NCCLCHECK(ncclCalloc(&proxyInfo, 1));
  if (reqSize != sizeof(struct shmProxyInfo)) return ncclInternalError;
  memcpy(proxyInfo, reqBuff, reqSize);
  NCCLCHECK(ncclCudaCalloc(&proxyInfo->devFifo, comm->buffSizes[NCCL_PROTO_SIMPLE]));
  NCCLCHECK(ncclCudaHostCalloc(&proxyInfo->ceRecvMem, 1));
  CUDACHECK(cudaStreamCreateWithFlags(&proxyInfo->stream, cudaStreamNonBlocking));
  for (int i=0; i<NCCL_STEPS; i++) {
    CUDACHECK(cudaEventCreate(proxyInfo->events+i));
  }
  connection->proxyAppendPtr = &connection->proxyAppend;
  connection->transportResources = proxyInfo;
  if (respSize != sizeof(struct shmProxyInfo)) return ncclInternalError;
  memcpy(respBuff, proxyInfo, respSize);
  return ncclSuccess;
}

static ncclResult_t shmSendProxyFree(struct ncclProxyConnection* connection, struct ncclComm* comm) {
  struct shmProxyInfo* resources = (struct shmProxyInfo*)connection->transportResources;

  if (resources) {
    CUDACHECK(cudaStreamDestroy(resources->stream));
    CUDACHECK(cudaFree(resources->devFifo));
    NCCLCHECK(ncclCudaHostFree(resources->ceRecvMem));
    for (int i=0; i<NCCL_STEPS; i++) {
      CUDACHECK(cudaEventDestroy(resources->events[i]));
    }
    free(connection->transportResources);
  }
  return ncclSuccess;
}

static ncclResult_t shmRecvProxyFree(struct ncclProxyConnection* connection, struct ncclComm* comm) {
  struct shmProxyInfo* resources = (struct shmProxyInfo*)connection->transportResources;

  if (resources) {
    CUDACHECK(cudaStreamDestroy(resources->stream));
    CUDACHECK(cudaFree(resources->devFifo));
    NCCLCHECK(ncclCudaHostFree(resources->ceRecvMem));
    for (int i=0; i<NCCL_STEPS; i++) {
      CUDACHECK(cudaEventDestroy(resources->events[i]));
    }
    free(connection->transportResources);
  }
  return ncclSuccess;
}

static ncclResult_t shmSendProxyProgress(struct ncclComm* comm, struct ncclProxyArgs* args) {
  if (args->state == ncclProxyOpReady) {
    for (int s=0; s<args->nsubs; s++) {
      struct ncclProxySubArgs* sub = args->subs+s;
      struct shmProxyInfo* resources = (struct shmProxyInfo*) (sub->connection->transportResources);
      // Round to next multiple of sliceSteps
      sub->base = ROUNDUP(resources->step, args->chunkSteps);
      sub->posted = sub->transmitted = sub->done = 0;
    }
    args->state = ncclProxyOpProgress;
  }
  args->idle = 1;
  if (args->state == ncclProxyOpProgress) {
    int p = args->protocol;
    int stepSize = comm->buffSizes[p] / NCCL_STEPS;
    for (int s=0; s<args->nsubs; s++) {
      struct ncclProxySubArgs* sub = args->subs+s;
      struct shmProxyInfo* resources = (struct shmProxyInfo*) (sub->connection->transportResources);
      if (p != NCCL_PROTO_SIMPLE) { // Only Simple uses cudaMemcpy
          resources->step = sub->base + sub->nsteps;
          args->done++;
          continue;
      }
      if (sub->transmitted < sub->done + NCCL_STEPS && sub->transmitted < sub->nsteps) {
        int buffSlot = (sub->base+sub->transmitted)%NCCL_STEPS;
        volatile int* sizesFifo = resources->ceRecvMem->sizesFifo;
        volatile uint64_t* recvTail = &resources->ceRecvMem->tail;
        // Check GPU has sent everything
        if ((*recvTail > sub->base+sub->transmitted)) {
          int size = sizesFifo[buffSlot];
          CUDACHECK(cudaMemcpyAsync(resources->shmFifo+buffSlot*stepSize, resources->devFifo+buffSlot*stepSize, size, cudaMemcpyDeviceToHost, resources->stream));
          CUDACHECK(cudaEventRecord(resources->events[buffSlot], resources->stream));
          resources->recvMem->sizesFifo[buffSlot] = size;
          __sync_synchronize(); // make sure sizesFifo is visible
          sub->transmitted += args->sliceSteps;
        }
      }
      if (sub->done < sub->transmitted) {
        int buffSlot = (sub->base+sub->done)%NCCL_STEPS;
        cudaError_t res = cudaEventQuery(resources->events[buffSlot]);
        if (res != cudaErrorNotReady) CUDACHECK(res);
        if (res == cudaSuccess) {
          sub->done += args->sliceSteps;
          // Notify SHM
          resources->recvMem->tail = sub->base + sub->done;
        }
        if (sub->done == sub->nsteps) {
          resources->step = sub->base + sub->nsteps;
          args->done++;
        }
      }
    }
    if (args->done == args->nsubs) {
      args->state = ncclProxyOpNone;
    }
  }
  return ncclSuccess;
}

static ncclResult_t shmRecvProxyProgress(struct ncclComm* comm, struct ncclProxyArgs* args) {
  if (args->state == ncclProxyOpReady) {
    for (int s=0; s<args->nsubs; s++) {
      struct ncclProxySubArgs* sub = args->subs+s;
      struct shmProxyInfo* resources = (struct shmProxyInfo*) (sub->connection->transportResources);
      // Round to next multiple of sliceSteps
      sub->base = ROUNDUP(resources->step, args->chunkSteps);
      sub->posted = sub->transmitted = sub->done = 0;
    }
    args->state = ncclProxyOpProgress;
  }
  args->idle = 1;
  if (args->state == ncclProxyOpProgress) {
    int p = args->protocol;
    int stepSize = comm->buffSizes[p] / NCCL_STEPS;
    for (int s=0; s<args->nsubs; s++) {
      struct ncclProxySubArgs* sub = args->subs+s;
      struct shmProxyInfo* resources = (struct shmProxyInfo*) (sub->connection->transportResources);
      if (p != NCCL_PROTO_SIMPLE) { // Only Simple uses cudaMemcpy
          resources->step = sub->base + sub->nsteps;
          args->done++;
          continue;
      }
      if (sub->transmitted < sub->done + NCCL_STEPS && sub->transmitted < sub->nsteps) {
        int buffSlot = (sub->base+sub->transmitted)%NCCL_STEPS;
        volatile int* sizesFifo = resources->recvMem->sizesFifo;
        volatile uint64_t* recvTail = &resources->recvMem->tail;
        // Check data is ready in SHM
        if ((*recvTail > sub->base+sub->transmitted)) {
          int size = sizesFifo[buffSlot];
          CUDACHECK(cudaMemcpyAsync(resources->devFifo+buffSlot*stepSize, resources->shmFifo+buffSlot*stepSize, size, cudaMemcpyHostToDevice, resources->stream));
          CUDACHECK(cudaEventRecord(resources->events[buffSlot], resources->stream));
          sub->transmitted += args->sliceSteps;
        }
      }
      if (sub->done < sub->transmitted) {
        int buffSlot = (sub->base+sub->done)%NCCL_STEPS;
        cudaError_t res = cudaEventQuery(resources->events[buffSlot]);
        if (res != cudaErrorNotReady) CUDACHECK(res);
        if (res == cudaSuccess) {
          sub->done += args->sliceSteps;
          // Notify GPU
          resources->ceRecvMem->tail = sub->base + sub->done;
        }
        if (sub->done == sub->nsteps) {
          resources->step = sub->base + sub->nsteps;
          args->done++;
        }
      }
    }
    if (args->done == args->nsubs) {
      args->state = ncclProxyOpNone;
    }
  }
  return ncclSuccess;
}

struct ncclTransport shmTransport = {
  "SHM",
  shmCanConnect,
  { shmSendSetup, shmSendConnect, shmSendFree, NULL, NULL, NULL, NULL, NULL },
  { shmRecvSetup, shmRecvConnect, shmRecvFree, NULL, NULL, NULL, NULL, NULL }
};

static void initCeOperation() {
  static int init = 0;
  if (!init) {
    useMemcpySend = ncclParamShmUseCudaMemcpy() && (ncclParamShmMemcpyMode() & 1);
    useMemcpyRecv = ncclParamShmUseCudaMemcpy() && (ncclParamShmMemcpyMode() & 2);
    if (useMemcpySend) {
      shmTransport.send.proxyConnect = shmSendProxyConnect;
      shmTransport.send.proxyFree = shmSendProxyFree;
      shmTransport.send.proxyProgress = shmSendProxyProgress;
    }
    if (useMemcpyRecv) {
      shmTransport.recv.proxyConnect = shmRecvProxyConnect;
      shmTransport.recv.proxyFree = shmRecvProxyFree;
      shmTransport.recv.proxyProgress = shmRecvProxyProgress;
    }
    shmLocality = ncclParamShmLocality();
    if (shmLocality != SHM_SEND_SIDE && shmLocality != SHM_RECV_SIDE) {
      WARN("Ignoring SHM locality, must be 1 (sender side) or 2 (receiver side, default)");
      shmLocality = SHM_RECV_SIDE;
    }
    init = 1;
  }
}
