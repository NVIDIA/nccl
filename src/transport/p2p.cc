/*************************************************************************
 * Copyright (c) 2016-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "comm.h"
#include "graph.h"
#include "utils.h"
#include "bootstrap.h"

struct p2pConnectInfo {
  int rank;
  int read;
  void* directPtr;
  cudaIpcMemHandle_t devIpc;
};

struct p2pSendResources {
  struct ncclSendMem* devMem;
  void* ipcPtr;
  int remoteId;
  int memRank;
  void* bootstrap;
};

struct p2pRecvResources {
  struct ncclRecvMem* devMem;
  void* ipcPtr;
  int remoteId;
  int memRank;
  void* bootstrap;
};

#include <sys/types.h>

/* Convert a PCI busId string into a local cudaDev device index (cf. CUDA_VISIBLE_DEVICES) */
static int busIdToCudaDev(int64_t busId) {
  int ndev;
  if (cudaGetDeviceCount(&ndev) != cudaSuccess)
    return -1;
  for (int i = 0; i < ndev; i++) {
    char devBusIdStr[NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE];
    if (cudaDeviceGetPCIBusId(devBusIdStr, NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE, i) != cudaSuccess)
      return -1;
    int64_t devBusId;
    NCCLCHECK(busIdToInt64(devBusIdStr, &devBusId));
    if (busId == devBusId) return i;
  }
  // BusId was not found in our locally visible CUDA devices
  return -1;
}

/* Determine if two peers can communicate through p2p */
ncclResult_t p2pCanConnect(int* ret, struct ncclTopoSystem* topo, struct ncclTopoGraph* graph, struct ncclPeerInfo* info1, struct ncclPeerInfo* info2) {
  // Rule out different nodes
  if (info1->hostHash != info2->hostHash) {
    *ret = 0;
    return ncclSuccess;
  }

  // Check topology / p2p level.
  int intermediateRank;
  NCCLCHECK(ncclTopoCheckP2p(topo, info1->busId, info2->busId, ret, NULL, &intermediateRank));
  if (*ret == 0) return ncclSuccess;
  if (intermediateRank != -1) return ncclSuccess;

  // Convert the peer's busId into a local cudaDev index (cf. CUDA_VISIBLE_DEVICES)
  int cudaDev1 = busIdToCudaDev(info1->busId);
  int cudaDev2 = busIdToCudaDev(info2->busId);
  if (cudaDev1 == -1 || cudaDev2 == -1) {
#if CUDART_VERSION >= 10010
    // CUDA 10.1 and later can use P2P with invisible devices.
    return ncclSuccess;
#else
    // Peer's CUDA device is not visible in this process : we can't communicate with it.
    *ret = 0;
    return ncclSuccess;
#endif
  }

  // Check that CUDA can do P2P
  int p2p;
  if (cudaDeviceCanAccessPeer(&p2p, cudaDev1, cudaDev2) != cudaSuccess) {
    INFO(NCCL_INIT|NCCL_P2P,"peer query failed between dev %d(=%lx) and dev %d(=%lx)",
         cudaDev1, info1->busId, cudaDev2, info2->busId);
    *ret = 0;
    return ncclSuccess;
  }
  if (p2p == 0) {
    INFO(NCCL_INIT|NCCL_P2P,"Could not enable P2P between dev %d(=%lx) and dev %d(=%lx)",
         cudaDev1, info1->busId, cudaDev2, info2->busId);
    *ret = 0;
    return ncclSuccess;
  }
  return ncclSuccess;
}

#define TRACE_DUMP_IPC(DEVIPC)                                                             \
  do {                                                                                     \
    unsigned long *devIpc = (unsigned long *) (DEVIPC);                                    \
    TRACE(P2P,"IPC: %016lx %016lx %016lx %016lx", devIpc[0], devIpc[1], devIpc[2], devIpc[3]); \
    TRACE(P2P,"IPC: %016lx %016lx %016lx %016lx", devIpc[4], devIpc[5], devIpc[6], devIpc[7]); \
  } while (0)

// Setting this to non zero causes P2P to use Reads rather than Writes
NCCL_PARAM(P2pReadEnable, "P2P_READ_ENABLE", -2);

static ncclResult_t p2pGetInfo(struct ncclTopoSystem* topo, struct ncclPeerInfo* info1, struct ncclPeerInfo* info2, int* read, int* intermediateRank) {
  int p2p;
  // Queries the topology to see if the GPUs are Ampere and
  // connected via NVLink, if so we enable P2P Read by default
  NCCLCHECK(ncclTopoCheckP2p(topo, info1->busId, info2->busId, &p2p, read, intermediateRank));

  int readEnable = ncclParamP2pReadEnable();
  if (readEnable != -2) *read = readEnable;
  return ncclSuccess;
}

static ncclResult_t p2pMap(struct ncclPeerInfo* myInfo, struct ncclPeerInfo* peerInfo, struct p2pConnectInfo* p2pInfo, void** devMem, void** ipcPtr) {
  if (myInfo->pidHash == peerInfo->pidHash) {
    if (peerInfo->cudaDev != myInfo->cudaDev) {
      // Enable P2P access
      cudaError_t err = cudaDeviceEnablePeerAccess(peerInfo->cudaDev, 0);
      if (err == cudaErrorPeerAccessAlreadyEnabled) {
        cudaGetLastError();
      } else if (err != cudaSuccess) {
        WARN("failed to peer with device %d(=%lx): %d %s",
            peerInfo->cudaDev, peerInfo->busId, err, cudaGetErrorString(err));
        return ncclInternalError;
      }
    }
    *devMem = p2pInfo->directPtr;
    *ipcPtr = NULL;
  } else {
    CUDACHECK(cudaIpcOpenMemHandle(devMem, p2pInfo->devIpc, cudaIpcMemLazyEnablePeerAccess));
    *ipcPtr = *devMem;
  }
  return ncclSuccess;
}

/* Send: Create and return connect structures for this peer to connect to me */
ncclResult_t p2pSendSetup(struct ncclComm* comm, struct ncclTopoGraph* graph, struct ncclPeerInfo* myInfo, struct ncclPeerInfo* peerInfo,
    struct ncclConnect* connectInfo, struct ncclConnector* send, int channelId) {

  struct p2pSendResources* resources;
  NCCLCHECK(ncclCalloc(&resources, 1));
  send->transportResources = resources;
  int useRead, intermediateRank;
  NCCLCHECK(p2pGetInfo(comm->topo, myInfo, peerInfo, &useRead, &intermediateRank));
  int sendSize = sizeof(struct ncclSendMem);
  // For P2P Read the SIMPLE buffer is tagged on the end of the ncclSendMem structure
  if (useRead) sendSize += send->comm->buffSizes[NCCL_PROTO_SIMPLE];
  ALIGN_SIZE(sendSize, CUDA_IPC_MIN);

  struct p2pConnectInfo info;
  info.read = useRead;
  const char* useReadStr = info.read ? "/read" : "";

  resources->remoteId = -1;
  resources->bootstrap = comm->bootstrap;
  if (intermediateRank == -1) {
    NCCLCHECK(ncclCudaCalloc((char**)&info.directPtr, sendSize));
    info.rank = myInfo->rank;
    if (myInfo->pidHash == peerInfo->pidHash) {
      if (useRead == 0) send->conn.direct |= NCCL_DIRECT_GPU;
      INFO(NCCL_INIT|NCCL_P2P, "Channel %02d : %d[%lx] -> %d[%lx] via P2P/direct pointer%s",
          channelId, myInfo->rank, myInfo->busId, peerInfo->rank, peerInfo->busId, useReadStr);
    } else {
      CUDACHECK(cudaIpcGetMemHandle(&info.devIpc, info.directPtr));
      INFO(NCCL_INIT|NCCL_P2P,"Channel %02d : %d[%lx] -> %d[%lx] via P2P/IPC%s",
          channelId, myInfo->rank, myInfo->busId, peerInfo->rank, peerInfo->busId, useReadStr);
    }
  } else {
    NCCLCHECK(bootstrapRemAlloc(sendSize, intermediateRank, resources->bootstrap, &resources->remoteId, &info.devIpc, &info.directPtr));
    info.rank = intermediateRank;
    INFO(NCCL_INIT|NCCL_P2P, "Channel %02d : %d[%lx] -> %d[%lx] via P2P/indirect/%d[%lx]%s",
        channelId, myInfo->rank, myInfo->busId, peerInfo->rank, peerInfo->busId, intermediateRank,
	comm->peerInfo[intermediateRank].busId, useReadStr);
  }
  resources->memRank = info.rank;

  NCCLCHECK(p2pMap(myInfo, comm->peerInfo+info.rank, &info, (void**)&resources->devMem, &resources->ipcPtr));

  static_assert(sizeof(struct p2pConnectInfo) <= sizeof(struct ncclConnect), "p2p Connect Info is too big");
  memcpy(connectInfo, &info, sizeof(struct p2pConnectInfo));
  return ncclSuccess;
}

/* Create and return connect structures for this peer to connect to me */
ncclResult_t p2pRecvSetup(struct ncclComm* comm, struct ncclTopoGraph* graph, struct ncclPeerInfo* myInfo, struct ncclPeerInfo* peerInfo,
    struct ncclConnect* connectInfo, struct ncclConnector * recv, int channelId) {

  struct p2pRecvResources* resources;
  NCCLCHECK(ncclCalloc(&resources, 1));
  recv->transportResources = resources;
  int useRead, intermediateRank;
  NCCLCHECK(p2pGetInfo(comm->topo, myInfo, peerInfo, &useRead, &intermediateRank));
  int recvSize = offsetof(struct ncclRecvMem, buff);
  // For P2P Read the SIMPLE buffer is tagged on the end of the ncclSendMem structure
  for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) if (!(useRead && p == NCCL_PROTO_SIMPLE)) recvSize += recv->comm->buffSizes[p];
  ALIGN_SIZE(recvSize, CUDA_IPC_MIN);

  struct p2pConnectInfo info;
  info.read = useRead;

  resources->remoteId = -1;
  resources->bootstrap = comm->bootstrap;
  if (intermediateRank == -1) {
    NCCLCHECK(ncclCudaCalloc((char**)&info.directPtr, recvSize));
    info.rank = myInfo->rank;
    if (myInfo->pidHash == peerInfo->pidHash) {
      if (useRead == 0) recv->conn.direct |= NCCL_DIRECT_GPU;
    } else {
      CUDACHECK(cudaIpcGetMemHandle(&info.devIpc, info.directPtr));
    }
  } else {
    NCCLCHECK(bootstrapRemAlloc(recvSize, intermediateRank, resources->bootstrap, &resources->remoteId, &info.devIpc, &info.directPtr));
    info.rank = intermediateRank;
  }
  resources->memRank = info.rank;

  NCCLCHECK(p2pMap(myInfo, comm->peerInfo+info.rank, &info, (void**)&resources->devMem, &resources->ipcPtr));

  static_assert(sizeof(struct p2pConnectInfo) <= sizeof(struct ncclConnect), "p2p Connect Info is too big");
  memcpy(connectInfo, &info, sizeof(struct p2pConnectInfo));
  return ncclSuccess;
}

/* Connect/Send to this peer */
static ncclResult_t p2pSendConnect(struct ncclComm* comm, struct ncclConnect* connectInfo, int nranks, int rank, struct ncclConnector* send) {
  struct p2pSendResources* resources = (struct p2pSendResources*)send->transportResources;
  struct ncclRecvMem* remDevMem;
  struct p2pConnectInfo* info = (struct p2pConnectInfo*)connectInfo;

  NCCLCHECK(p2pMap(comm->peerInfo+rank, comm->peerInfo+info->rank, info, (void**)&remDevMem, &resources->ipcPtr));

  int offset = 0;
  for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
    if (info->read && p == NCCL_PROTO_SIMPLE) {
      /* For P2P Read the SIMPLE buffer is local (ncclSendMem) */
      send->conn.buffs[p] = resources->devMem->buff;
    } else {
      send->conn.buffs[p] = remDevMem->buff + offset;
      offset += send->comm->buffSizes[p];
    }
  }
  send->conn.tail = &remDevMem->tail;
  send->conn.head = &resources->devMem->head;
  send->conn.ptrExchange = &resources->devMem->ptrExchange;
  return ncclSuccess;
}

/* Connect/Recv from this peer */
ncclResult_t p2pRecvConnect(struct ncclComm* comm, struct ncclConnect* connectInfo, int nranks, int rank, struct ncclConnector* recv) {
  struct p2pRecvResources* resources = (struct p2pRecvResources*)recv->transportResources;
  struct ncclSendMem* remDevMem;
  struct p2pConnectInfo* info = (struct p2pConnectInfo*)connectInfo;

  NCCLCHECK(p2pMap(comm->peerInfo+rank, comm->peerInfo+info->rank, info, (void**)&remDevMem, &resources->ipcPtr));

  int offset = 0;
  for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
    if (info->read && p == NCCL_PROTO_SIMPLE) {
      /* For P2P Read the SIMPLE buffer is remote (ncclSendMem) */
      recv->conn.buffs[p] = remDevMem->buff;
    } else {
      recv->conn.buffs[p] = resources->devMem->buff + offset;
      offset += recv->comm->buffSizes[p];
    }
  }
  recv->conn.tail = &resources->devMem->tail;
  recv->conn.head = &remDevMem->head;
  recv->conn.ptrExchange = &remDevMem->ptrExchange;
  return ncclSuccess;
}

ncclResult_t p2pSendFree(void* resources) {
  struct p2pSendResources* sendRes = (struct p2pSendResources*)resources;
  if (sendRes->ipcPtr)
    CUDACHECK(cudaIpcCloseMemHandle(sendRes->ipcPtr));
  if (sendRes->remoteId != -1) {
    NCCLCHECK(bootstrapRemFree(sendRes->remoteId, sendRes->memRank, sendRes->bootstrap));
    sendRes->devMem = NULL;
  }
  CUDACHECK(cudaFree(sendRes->devMem));
  free(sendRes);
  return ncclSuccess;
}

ncclResult_t p2pRecvFree(void* resources) {
  struct p2pRecvResources* recvRes = (struct p2pRecvResources*)resources;
  if (recvRes->ipcPtr)
    CUDACHECK(cudaIpcCloseMemHandle(recvRes->ipcPtr));
  if (recvRes->remoteId != -1) {
    NCCLCHECK(bootstrapRemFree(recvRes->remoteId, recvRes->memRank, recvRes->bootstrap));
    recvRes->devMem = NULL;
  }
  CUDACHECK(cudaFree(recvRes->devMem));
  free(recvRes);
  return ncclSuccess;
}

struct ncclTransport p2pTransport = {
  "P2P",
  p2pCanConnect,
  { p2pSendSetup, p2pSendConnect, p2pSendFree, NULL },
  { p2pRecvSetup, p2pRecvConnect, p2pRecvFree, NULL }
};
