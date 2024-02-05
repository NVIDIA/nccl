/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "comm.h"
#include "graph.h"
#include "utils.h"
#include "shm.h"
#include "p2p.h"

enum p2pType { P2P_DIRECT, P2P_INTERMEDIATE, P2P_IPC, P2P_CUMEM };

struct ncclP2pBuff {
  void* directPtr;
  size_t size;
  ncclIpcDesc ipcDesc;
};

struct p2pConnectInfo {
  int rank;
  int read;
  struct ncclP2pBuff p2pBuff;
  // Used by CE memcpy
  char shmName[7];
  int shmSize;
};
static_assert(sizeof(struct p2pConnectInfo) <= CONNECT_SIZE, "p2pConnectInfo is too large");

struct p2pShm {
  struct ncclSendMem sendMem;
  struct ncclRecvMem recvMem;
};
struct p2pShmProxyInfo {
  // Shared memory between proxy and receiving GPU
  struct p2pShm* shm;
  struct p2pShm* devShm;
  char shmName[7];
  int shmSize;
  ncclShmHandle_t handle;

  // Intermediate step for sender
  struct ncclRecvMem* ceRecvMem;
  char* ceDevBuff;

  // Receiver buffer
  char* recvFifo;

  // Used by CE memcpy progress only
  uint64_t step;
  cudaStream_t stream;
  cudaEvent_t events[NCCL_STEPS];
};
static_assert(sizeof(p2pConnectInfo) <= CONNECT_SIZE, "P2P Connect info is too large");

struct p2pResources {
  enum p2pType type;
  union {
    struct ncclSendMem* sendDevMem;
    struct ncclRecvMem* recvDevMem;
  };
  void* sendMemIpc;
  void* recvMemIpc;
  // CE memcpy support
  struct p2pShmProxyInfo proxyInfo;
  struct p2pShm* shm;
  struct p2pShm* devShm;
  int shmSize;
  ncclShmHandle_t handle;
};

// cuMem API support
struct p2pCuMemProxyInfo {
  struct ncclP2pBuff p2pBuff;
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

// CE memcpy support
NCCL_PARAM(P2pUseCudaMemcpy, "P2P_USE_CUDA_MEMCPY", 0);
static int useMemcpy = 0;
static void initCeOperation();

/* Determine if two peers can communicate through p2p */
ncclResult_t p2pCanConnect(int* ret, struct ncclTopoSystem* topo, struct ncclTopoGraph* graph, struct ncclPeerInfo* info1, struct ncclPeerInfo* info2) {
  initCeOperation();

  // MNNVL support
  if (info1->hostHash != info2->hostHash) {
    NCCLCHECK(ncclTopoCheckMNNVL(topo, info1, info2, ret));
    if (*ret) return ncclSuccess;
  }

  // Rule out different nodes / isolated containers
  if (info1->hostHash != info2->hostHash || info1->shmDev != info2->shmDev) {
    *ret = 0;
    return ncclSuccess;
  }

  // Check topology / p2p level.
  int intermediateRank;
  NCCLCHECK(ncclTopoCheckP2p(topo, info1->busId, info2->busId, ret, NULL, &intermediateRank));
  if (*ret == 0) return ncclSuccess;
  if (intermediateRank != -1) {
    if (useMemcpy) *ret = 0;
    return ncclSuccess;
  }

  // Check if NET would work better
  int useNet = 0;
  NCCLCHECK(ncclTopoCheckNet(topo, info1->busId, info2->busId, &useNet));
  if (useNet) {
    *ret = 0;
    return ncclSuccess;
  }

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

  // This will always fail when using NCCL_CUMEM_ENABLE=1
  if (p2p != 0 && !ncclCuMemEnable()) {
    // Cached result of the legacyIPC detection
    static int legacyIPC = -1;
    if (legacyIPC >= 0) {
      *ret = legacyIPC;
      return ncclSuccess;
    }
    // Check that legacy IPC support is available (WSL WAR)
    char *dummy;
    cudaIpcMemHandle_t ipc;
    NCCLCHECK(ncclCudaMalloc(&dummy, CUDA_IPC_MIN));
    if (cudaIpcGetMemHandle(&ipc, dummy) != cudaSuccess) {
      INFO(NCCL_INIT|NCCL_P2P,"Legacy IPC not supported");
      *ret = 0;
    }
    NCCLCHECK(ncclCudaFree(dummy));
    legacyIPC = *ret;
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

// cuMem API support
ncclResult_t ncclP2pAllocateShareableBuffer(size_t size, ncclIpcDesc *ipcDesc, void **ptr) {
  if (ncclCuMemEnable()) {
#if CUDART_VERSION >= 11030
    CUmemAllocationHandleType type = ncclCuMemHandleType;

    // cuMem API support
    CUmemGenericAllocationHandle handle;
    NCCLCHECK(ncclCuMemAlloc(ptr, &handle, size));
    if (type == CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR) {
      // Return the native cuMem handle for later Export/Import via UDS
      memcpy(&ipcDesc->cuDesc.data, &handle, sizeof(handle));
    } else {
      CUCHECK(cuMemExportToShareableHandle(&ipcDesc->cuDesc, handle, type, 0));
    }
#else
    return ncclInternalError;
#endif
  } else {
    // Allocate a CUDA buffer and generate an IPC handle for it
    NCCLCHECK(ncclCudaCalloc((char **)ptr, size));
    cudaError_t res = cudaIpcGetMemHandle(&ipcDesc->devIpc, *ptr);
    if (res != cudaSuccess) {
      WARN("cudaIpcGetMemHandle failed : %s", cudaGetErrorString(res));
      ncclCudaFree(*ptr);
      CUDACHECK(res);
    }
  }
  INFO(NCCL_P2P|NCCL_ALLOC, "Allocated shareable buffer %p size %zi ipcDesc %p", *ptr, size, ipcDesc);

  return ncclSuccess;
}

ncclResult_t ncclP2pFreeShareableBuffer(ncclIpcDesc *ipcDesc) {
  return ncclSuccess;
}

ncclResult_t ncclP2pImportShareableBuffer(struct ncclComm *comm, int tpPeer, size_t size, ncclIpcDesc *ipcDesc, void **devMemPtr) {
  if (ncclCuMemEnable()) {
#if CUDART_VERSION >= 11030
    // cuMem API support
    CUdeviceptr dptr = 0;
    CUmemAllocationHandleType type = ncclCuMemHandleType;
    CUmemGenericAllocationHandle handle;
    ncclCuDesc *cuDesc = &ipcDesc->cuDesc;

    // Import and map the remote memory descriptor to the local GPU
    if (type == CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR) {
      // UDS fd support
      int fd = -1;
      // Send cuMem handle to remote for conversion to an fd
      NCCLCHECK(ncclProxyClientGetFdBlocking(comm, tpPeer, &cuDesc->data, &fd));
      INFO(NCCL_P2P, "UDS converted handle 0x%lx to fd %d on remote peer %d", *(uint64_t*)&cuDesc->data, fd, tpPeer);
      CUCHECK(cuMemImportFromShareableHandle(&handle, (void *)(uintptr_t)fd, type));
      (void) close(fd);
    } else {
      CUCHECK(cuMemImportFromShareableHandle(&handle, cuDesc, type));
    }
    CUCHECK(cuMemAddressReserve(&dptr, size, /* alignment */ 0, /* addr */ 0, /* flags */ 0));
    CUCHECK(cuMemMap(dptr, size, /* offset */ 0, handle, /* flags */ 0));

    TRACE(NCCL_P2P, "Imported shareable buffer size %zi handle 0x%llx dptr %p", size, handle, (void*)dptr);

    // Allow access by the local GPU
    CUmemAccessDesc accessDesc = {};
    accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    accessDesc.location.id = comm->cudaDev;
    accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    CUCHECK(cuMemSetAccess(dptr, size, &accessDesc, 1));
    TRACE(NCCL_P2P, "Set Access for %p size %zi on dev %d", (void*)dptr, size, accessDesc.location.id);

    *devMemPtr = (void *)dptr;
#else
    return ncclInternalError;
#endif
  } else {
    // Legacy CUDA IPC
    CUDACHECK(cudaIpcOpenMemHandle(devMemPtr, ipcDesc->devIpc, cudaIpcMemLazyEnablePeerAccess));
  }

  INFO(NCCL_P2P, "Imported shareable buffer device %d size %zi ptr %p", comm->cudaDev, size, *devMemPtr);

  return ncclSuccess;
}

// Setting this to non zero causes P2P to use Reads rather than Writes
NCCL_PARAM(P2pReadEnable, "P2P_READ_ENABLE", -2);
NCCL_PARAM(P2pDirectDisable, "P2P_DIRECT_DISABLE", 0);

#define P2P_SAME_PID(MYINFO, PEERINFO) ((MYINFO->hostHash == PEERINFO->hostHash) && (MYINFO->pidHash == PEERINFO->pidHash))

static ncclResult_t p2pGetInfo(struct ncclTopoSystem* topo, struct ncclPeerInfo* info1, struct ncclPeerInfo* info2, int* read, int* intermediateRank) {
  int p2p;
  // Queries the topology to see if the GPUs are Ampere and
  // connected via NVLink, if so we enable P2P Read by default
  NCCLCHECK(ncclTopoCheckP2p(topo, info1->busId, info2->busId, &p2p, read, intermediateRank));

  int readEnable = ncclParamP2pReadEnable();
  if (readEnable != -2) *read = readEnable;
  return ncclSuccess;
}

static ncclResult_t p2pMap(struct ncclComm *comm, struct ncclProxyConnector* proxyConn, struct ncclPeerInfo* myInfo, struct ncclPeerInfo* peerInfo, struct ncclP2pBuff* p2pBuff, void** devMem, void** ipcPtr) {
  if (P2P_SAME_PID(myInfo, peerInfo)) {
    if (peerInfo->cudaDev != myInfo->cudaDev) {
      // Same PID different GPUs, enable P2P access
      // Legacy CUDA IPC
      cudaError_t err = cudaDeviceEnablePeerAccess(peerInfo->cudaDev, 0);
      if (err == cudaErrorPeerAccessAlreadyEnabled) {
        cudaGetLastError();
      } else if (err != cudaSuccess) {
        WARN("failed to peer with device %d(=%lx): %d %s",
            peerInfo->cudaDev, peerInfo->busId, err, cudaGetErrorString(err));
        return ncclInternalError;
      }
#if CUDART_VERSION >= 11030
      // cuMem API support
      if (ncclCuMemEnable()) {
        // Allow direct access to the remote buffer from the local GPU
        CUmemAccessDesc accessDesc = {};
        accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        accessDesc.location.id = myInfo->cudaDev;
        accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        INFO(NCCL_P2P, "Set Access for buffer %p size %zi on dev %d", p2pBuff->directPtr, p2pBuff->size, peerInfo->cudaDev);
        CUCHECK(cuMemSetAccess((CUdeviceptr) p2pBuff->directPtr, p2pBuff->size, &accessDesc, 1));
      }
#endif
    }
    *devMem = p2pBuff->directPtr;
    *ipcPtr = NULL;
  } else {
    // Different PID
    NCCLCHECK(ncclP2pImportShareableBuffer(comm, comm->topParentRanks[peerInfo->rank], p2pBuff->size, &p2pBuff->ipcDesc, devMem));
    *ipcPtr = *devMem;
  }
  return ncclSuccess;
}

/* Send: Create and return connect structures for this peer to connect to me */
ncclResult_t p2pSendSetup(struct ncclComm* comm, struct ncclTopoGraph* graph, struct ncclPeerInfo* myInfo, struct ncclPeerInfo* peerInfo,
    struct ncclConnect* connectInfo, struct ncclConnector* send, int channelId, int connIndex) {
  struct p2pResources* resources;
  int tpProxyRank;
  NCCLCHECK(ncclCalloc(&resources, 1));
  send->transportResources = resources;
  int useRead, intermediateRank;
  NCCLCHECK(p2pGetInfo(comm->topo, myInfo, peerInfo, &useRead, &intermediateRank));
  if (useMemcpy) useRead = 0;

  static_assert(sizeof(struct p2pConnectInfo) <= sizeof(struct ncclConnect), "p2p Connect Info is too big");
  struct p2pConnectInfo* info = (struct p2pConnectInfo*)connectInfo;
  info->read = useRead;
  // For CollNet, use write for scatter-reduce (conn 1), read for broadcast-gather (conn 0)
  if (graph && connIndex == 1) info->read = 0;
  const char* useReadStr = info->read ? "/read" : "";

  int sendSize = sizeof(struct ncclSendMem);
  // For P2P Read the SIMPLE buffer is tagged on the end of the ncclSendMem structure
  if (info->read) sendSize += comm->buffSizes[NCCL_PROTO_SIMPLE];
  ALIGN_SIZE(sendSize, CUDA_IPC_MIN);

  if (intermediateRank == -1) {
    info->rank = myInfo->rank;
    if (P2P_SAME_PID(myInfo, peerInfo) && ncclParamP2pDirectDisable() == 0 && useMemcpy == 0) {
      resources->type = P2P_DIRECT;
      send->conn.flags |= info->read ? NCCL_DIRECT_READ : NCCL_DIRECT_WRITE;
      INFO(NCCL_INIT|NCCL_P2P, "Channel %02d/%01d : %d[%d] -> %d[%d] via P2P/direct pointer%s",
          channelId, connIndex, myInfo->rank, myInfo->nvmlDev, peerInfo->rank, peerInfo->nvmlDev, useReadStr);
    } else {
      // cuMem API support
      if (ncclCuMemEnable()) {
        resources->type = P2P_CUMEM;
        const char *MNNVL = comm->MNNVL ? "MNNVL" : "CUMEM";
        INFO(NCCL_INIT|NCCL_P2P,"Channel %02d/%01d : %d[%d] -> %d[%d] via P2P/%s%s%s",
             channelId, connIndex, myInfo->rank, myInfo->nvmlDev, peerInfo->rank, peerInfo->nvmlDev, MNNVL, useReadStr, useMemcpy ? "/CE" : "");;
      } else {
        // Legacy CUDA IPC
        resources->type = P2P_IPC;
        INFO(NCCL_INIT|NCCL_P2P,"Channel %02d/%01d : %d[%d] -> %d[%d] via P2P/IPC%s%s",
             channelId, connIndex, myInfo->rank, myInfo->nvmlDev, peerInfo->rank, peerInfo->nvmlDev, useReadStr, useMemcpy ? "/CE" : "");
      }
      send->conn.flags |= info->read ? NCCL_IPC_READ : NCCL_IPC_WRITE;
    }
  } else {
    resources->type = P2P_INTERMEDIATE;
    info->rank = intermediateRank;
    INFO(NCCL_INIT|NCCL_P2P, "Channel %02d/%01d : %d[%d] -> %d[%d] via P2P/indirect/%d[%d]%s",
        channelId, connIndex, myInfo->rank, myInfo->nvmlDev, peerInfo->rank, peerInfo->nvmlDev, intermediateRank,
	  comm->peerInfo[intermediateRank].nvmlDev, useReadStr);
  }

  tpProxyRank = comm->topParentRanks[info->rank];
  NCCLCHECK(ncclProxyConnect(comm, TRANSPORT_P2P, 1, tpProxyRank, &send->proxyConn));
  if (useMemcpy) {
    NCCLCHECK(ncclProxyCallBlocking(comm, &send->proxyConn, ncclProxyMsgSetup, NULL, 0, &resources->proxyInfo, sizeof(struct p2pShmProxyInfo)));
    info->shmSize = resources->proxyInfo.shmSize;
    memcpy(info->shmName, resources->proxyInfo.shmName, sizeof(info->shmName));
  } else {
    NCCLCHECK(ncclProxyCallBlocking(comm, &send->proxyConn, ncclProxyMsgSetup, &sendSize, sizeof(int), &info->p2pBuff, sizeof(struct ncclP2pBuff)));
    NCCLCHECK(p2pMap(comm, &send->proxyConn, myInfo, comm->peerInfo+info->rank, &info->p2pBuff, (void**)&resources->sendDevMem, &resources->sendMemIpc));
  }

  return ncclSuccess;
}

/* Create and return connect structures for this peer to connect to me */
ncclResult_t p2pRecvSetup(struct ncclComm* comm, struct ncclTopoGraph* graph, struct ncclPeerInfo* myInfo, struct ncclPeerInfo* peerInfo,
    struct ncclConnect* connectInfo, struct ncclConnector * recv, int channelId, int connIndex) {
  struct p2pResources* resources;
  int tpProxyRank;
  NCCLCHECK(ncclCalloc(&resources, 1));
  recv->transportResources = resources;
  int useRead, intermediateRank;
  NCCLCHECK(p2pGetInfo(comm->topo, myInfo, peerInfo, &useRead, &intermediateRank));

  static_assert(sizeof(struct p2pConnectInfo) <= sizeof(struct ncclConnect), "p2p Connect Info is too big");
  struct p2pConnectInfo* info = (struct p2pConnectInfo*)connectInfo;
  info->read = useRead;
  // For CollNet, use write for scatter-reduce (conn 1), read for broadcast-gather (conn 0)
  if (graph && connIndex == 1) info->read = 0;

  int recvSize = sizeof(struct ncclRecvMem);
  // For P2P Read the SIMPLE buffer is tagged on the end of the ncclSendMem structure
  for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) if (!(info->read && p == NCCL_PROTO_SIMPLE)) recvSize += comm->buffSizes[p];
  ALIGN_SIZE(recvSize, CUDA_IPC_MIN);

  if (intermediateRank == -1) {
    info->rank = myInfo->rank;
    if (P2P_SAME_PID(myInfo, peerInfo) && ncclParamP2pDirectDisable() == 0 && useMemcpy == 0) {
      resources->type = P2P_DIRECT;
      recv->conn.flags |= info->read ? NCCL_DIRECT_READ : NCCL_DIRECT_WRITE;
    } else {
      if (ncclCuMemEnable()) {
        // cuMem API support
        resources->type = P2P_CUMEM;
        TRACE(NCCL_INIT|NCCL_P2P,"Ring %02d : %d[%d] <- %d[%d] via P2P/CUMEM",
              channelId, myInfo->rank, myInfo->nvmlDev, peerInfo->rank, peerInfo->nvmlDev);
      } else {
        // Legacy CUDA IPC
        resources->type = P2P_IPC;
      }
      recv->conn.flags |= info->read ? NCCL_IPC_READ : NCCL_IPC_WRITE;
    }
  } else {
    resources->type = P2P_INTERMEDIATE;
    info->rank = intermediateRank;
  }

  tpProxyRank = comm->topParentRanks[info->rank];
  NCCLCHECK(ncclProxyConnect(comm, TRANSPORT_P2P, 0, tpProxyRank, &recv->proxyConn));
  NCCLCHECK(ncclProxyCallBlocking(comm, &recv->proxyConn, ncclProxyMsgSetup, &recvSize, sizeof(int), &info->p2pBuff, sizeof(struct ncclP2pBuff)));

  NCCLCHECK(p2pMap(comm, &recv->proxyConn, myInfo, comm->peerInfo+info->rank, &info->p2pBuff, (void**)&resources->recvDevMem, &resources->recvMemIpc));
  return ncclSuccess;
}

/* Connect/Send to this peer */
static ncclResult_t p2pSendConnect(struct ncclComm* comm, struct ncclConnect* connectInfo, int nranks, int rank, struct ncclConnector* send) {
  struct p2pResources* resources = (struct p2pResources*)send->transportResources;
  struct ncclRecvMem* remDevMem = NULL;
  struct p2pConnectInfo* info = (struct p2pConnectInfo*)connectInfo;

  NCCLCHECK(p2pMap(comm, &send->proxyConn, comm->peerInfo+rank, comm->peerInfo+info->rank, &info->p2pBuff, (void**)&remDevMem, &resources->recvMemIpc));

  char* buff = (char*)(remDevMem+1);
  for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
    if (info->read && p == NCCL_PROTO_SIMPLE) {
      /* For P2P Read the SIMPLE buffer is local (ncclSendMem) */
      if (resources->sendDevMem == NULL) return ncclInternalError; // We should not use read + memcpy
      send->conn.buffs[p] = (char*)(resources->sendDevMem+1);
    } else {
      send->conn.buffs[p] = buff;
      buff += comm->buffSizes[p];
    }
  }

  if (useMemcpy) {
    send->conn.tail = &resources->proxyInfo.ceRecvMem->tail;
    send->conn.connFifo = resources->proxyInfo.ceRecvMem->connFifo;
    send->conn.head = &resources->proxyInfo.devShm->sendMem.head;
    // Send SIMPLE buff to proxy, and replace it by local buffer
    NCCLCHECK(ncclProxyCallBlocking(comm, &send->proxyConn, ncclProxyMsgConnect, &send->conn.buffs[NCCL_PROTO_SIMPLE], sizeof(void*), NULL, 0));
    send->conn.buffs[NCCL_PROTO_SIMPLE] = resources->proxyInfo.ceDevBuff;
  } else {
    send->conn.tail = &remDevMem->tail;
    send->conn.head = &resources->sendDevMem->head;
    send->conn.ptrExchange = &resources->sendDevMem->ptrExchange;
    send->conn.redOpArgExchange = resources->sendDevMem->redOpArgExchange;
  }
  // We must assign the proxyConn's proxyProgress property for proper checking at enqueue-time
  send->proxyConn.proxyProgress = p2pTransport.send.proxyProgress;
  return ncclSuccess;
}

/* Connect/Recv from this peer */
ncclResult_t p2pRecvConnect(struct ncclComm* comm, struct ncclConnect* connectInfo, int nranks, int rank, struct ncclConnector* recv) {
  struct p2pResources* resources = (struct p2pResources*)recv->transportResources;
  struct p2pConnectInfo* info = (struct p2pConnectInfo*)connectInfo;

  struct ncclSendMem* remDevMem = NULL;

  if (useMemcpy) {
    char shmPath[PATH_MAX];
    sprintf(shmPath, "/dev/shm/nccl-%s", info->shmName);
    TRACE(NCCL_SHM,"Open shmName %s shmSize %d", shmPath, info->shmSize);
    resources->shmSize = info->shmSize;
    // Attach to peer's SHM segment
    NCCLCHECK(ncclShmOpen(shmPath, info->shmSize, (void**)&resources->shm, (void**)&resources->devShm, -1, &resources->handle));

    recv->conn.tail = &resources->devShm->recvMem.tail;
    recv->conn.head = &resources->devShm->sendMem.head;
  } else {
    NCCLCHECK(p2pMap(comm, &recv->proxyConn, comm->peerInfo+rank, comm->peerInfo+info->rank, &info->p2pBuff, (void**)&remDevMem, &resources->sendMemIpc));

    struct ncclRecvMem* devMem = resources->recvDevMem;
    recv->conn.tail = &devMem->tail;
    recv->conn.head = &remDevMem->head;
    recv->conn.ptrExchange = &remDevMem->ptrExchange;
    recv->conn.redOpArgExchange = remDevMem->redOpArgExchange;
  }

  char* buff = (char*)(resources->recvDevMem+1);
  for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
    if (info->read && p == NCCL_PROTO_SIMPLE) {
      if (remDevMem == NULL) return ncclInternalError; // We should not use read + memcpy
      /* For P2P Read the SIMPLE buffer is remote (ncclSendMem) */
      recv->conn.buffs[p] = (char*)(remDevMem+1);
    } else {
      recv->conn.buffs[p] = buff;
      buff += comm->buffSizes[p];
    }
  }
  return ncclSuccess;
}

ncclResult_t p2pSendFree(struct ncclConnector* send) {
  struct p2pResources* resources = (struct p2pResources*)send->transportResources;
  if (resources) {
    if (ncclCuMemEnable()) {
      // cuMem API support
      if (resources->sendMemIpc) NCCLCHECK(ncclCudaFree(resources->sendMemIpc));
      if (resources->recvMemIpc) NCCLCHECK(ncclCudaFree(resources->recvMemIpc));
    }
    else {
      if (resources->sendMemIpc) CUDACHECK(cudaIpcCloseMemHandle(resources->sendMemIpc));
      if (resources->recvMemIpc) CUDACHECK(cudaIpcCloseMemHandle(resources->recvMemIpc));
    }
    free(resources);
  }
  return ncclSuccess;
}

ncclResult_t p2pRecvFree(struct ncclConnector* recv) {
  struct p2pResources* resources = (struct p2pResources*)recv->transportResources;
  if (resources) {
    if (ncclCuMemEnable()) {
      // cuMem API support
      if (resources->sendMemIpc) NCCLCHECK(ncclCudaFree(resources->sendMemIpc));
      if (resources->recvMemIpc) NCCLCHECK(ncclCudaFree(resources->recvMemIpc));
    }
    else {
      if (resources->sendMemIpc) CUDACHECK(cudaIpcCloseMemHandle(resources->sendMemIpc));
      if (resources->recvMemIpc) CUDACHECK(cudaIpcCloseMemHandle(resources->recvMemIpc));
      if (useMemcpy) {
        NCCLCHECK(ncclShmClose(resources->handle));
      }
    }
    free(resources);
  }
  return ncclSuccess;
}

static ncclResult_t p2pSendProxySetup(struct ncclProxyConnection* connection, struct ncclProxyState* proxyState, void* reqBuff, int reqSize, void* respBuff, int respSize, int* done) {
  if (useMemcpy) {
    // CE memcpy support
    struct p2pShmProxyInfo* proxyInfo;
    NCCLCHECK(ncclCalloc(&proxyInfo, 1));
    connection->transportResources = proxyInfo;

    NCCLCHECK(ncclCudaCalloc(&proxyInfo->ceDevBuff, proxyState->buffSizes[NCCL_PROTO_SIMPLE]));

    char shmPath[PATH_MAX];
    shmPath[0] = '\0';
    proxyInfo->shmSize = sizeof(struct ncclSendMem) + sizeof(struct ncclRecvMem);
    // Create a SHM segment for the peer to attach to
    NCCLCHECK(ncclShmOpen(shmPath, proxyInfo->shmSize, (void**)&proxyInfo->shm, (void**)&proxyInfo->devShm, 1, &proxyInfo->handle));
    TRACE(NCCL_SHM,"Opened shmName %s shmSize %d", shmPath, proxyInfo->shmSize);
    memcpy(proxyInfo->shmName, shmPath+sizeof("/dev/shm/nccl-")-1, sizeof(proxyInfo->shmName));

    NCCLCHECK(ncclCudaHostCalloc(&proxyInfo->ceRecvMem, 1));

    if (respSize != sizeof(struct p2pShmProxyInfo)) return ncclInternalError;
    memcpy(respBuff, proxyInfo, sizeof(struct p2pShmProxyInfo));
  } else {
    if (reqSize != sizeof(int)) return ncclInternalError;
    int size = *((int*)reqBuff);
    if (respSize != sizeof(struct ncclP2pBuff)) return ncclInternalError;
    struct ncclP2pBuff* p2pBuff = (struct ncclP2pBuff*)respBuff;
    NCCLCHECK(ncclP2pAllocateShareableBuffer(size, &p2pBuff->ipcDesc, &p2pBuff->directPtr));
    p2pBuff->size = size;
    if (ncclCuMemEnable()) {
      // cuMem API support
      struct p2pCuMemProxyInfo* proxyInfo;
      NCCLCHECK(ncclCalloc(&proxyInfo, 1));
      memcpy(&proxyInfo->p2pBuff, p2pBuff, sizeof(*p2pBuff));
      connection->transportResources = proxyInfo;
    } else {
      connection->transportResources = p2pBuff->directPtr;
    }
  }
  *done = 1;
  return ncclSuccess;
}

static ncclResult_t p2pRecvProxySetup(struct ncclProxyConnection* connection, struct ncclProxyState* proxyState, void* reqBuff, int reqSize, void* respBuff, int respSize, int* done) {
  if (reqSize != sizeof(int)) return ncclInternalError;
  int size = *((int*)reqBuff);
  if (respSize != sizeof(struct ncclP2pBuff)) return ncclInternalError;
  struct ncclP2pBuff* p2pBuff = (struct ncclP2pBuff*)respBuff;
  NCCLCHECK(ncclP2pAllocateShareableBuffer(size, &p2pBuff->ipcDesc, &p2pBuff->directPtr));
  p2pBuff->size = size;
  if (ncclCuMemEnable()) {
    // cuMem API support
    struct p2pCuMemProxyInfo* proxyInfo;
    NCCLCHECK(ncclCalloc(&proxyInfo, 1));
    memcpy(&proxyInfo->p2pBuff, p2pBuff, sizeof(*p2pBuff));
    connection->transportResources = proxyInfo;
  } else {
    connection->transportResources = p2pBuff->directPtr;
  }
  *done = 1;
  return ncclSuccess;
}

static ncclResult_t p2pSendProxyConnect(struct ncclProxyConnection* connection, struct ncclProxyState* proxyState, void* reqBuff, int reqSize, void* respBuff, int respSize, int* done) {
  struct p2pShmProxyInfo* proxyInfo = (struct p2pShmProxyInfo*)connection->transportResources;

  if (reqSize != sizeof(void*)) return ncclInternalError;
  proxyInfo->recvFifo = *((char**)reqBuff);

  CUDACHECK(cudaStreamCreateWithFlags(&proxyInfo->stream, cudaStreamNonBlocking));
  for (int i=0; i<NCCL_STEPS; i++) {
    CUDACHECK(cudaEventCreate(proxyInfo->events+i));
  }
  connection->proxyAppendPtr = &connection->proxyAppend;
  return ncclSuccess;
}

static ncclResult_t p2pSendProxyFree(struct ncclProxyConnection* connection, struct ncclProxyState* proxyState) {
  // CE memcpy support
  if (useMemcpy) {
    struct p2pShmProxyInfo* proxyInfo = (struct p2pShmProxyInfo*)connection->transportResources;
    if (proxyInfo) {
      NCCLCHECK(ncclShmClose(proxyInfo->handle));
      NCCLCHECK(ncclCudaHostFree(proxyInfo->ceRecvMem));
      NCCLCHECK(ncclCudaFree(proxyInfo->ceDevBuff));
      CUDACHECK(cudaStreamDestroy(proxyInfo->stream));
      for (int i=0; i<NCCL_STEPS; i++) {
        CUDACHECK(cudaEventDestroy(proxyInfo->events[i]));
      }
      free(proxyInfo);
    }
  } else {
    if (ncclCuMemEnable()) {
      // cuMem API support
      struct p2pCuMemProxyInfo *proxyInfo = (struct p2pCuMemProxyInfo *) connection->transportResources;
      if (proxyInfo) {
        struct ncclP2pBuff *p2pBuff = &proxyInfo->p2pBuff;
        ncclP2pFreeShareableBuffer(&p2pBuff->ipcDesc);
        ncclCudaFree(p2pBuff->directPtr);
        free(proxyInfo);
      }
    } else {
      // Do not check return code as CUDA may have already shut down
      ncclCudaFree(connection->transportResources);
    }
  }
  return ncclSuccess;
}

static ncclResult_t p2pRecvProxyFree(struct ncclProxyConnection* connection, struct ncclProxyState* proxyState) {
  if (ncclCuMemEnable()) {
    struct p2pCuMemProxyInfo *proxyInfo = (struct p2pCuMemProxyInfo *) connection->transportResources;
    if (proxyInfo) {
      struct ncclP2pBuff *p2pBuff = &proxyInfo->p2pBuff;
      ncclP2pFreeShareableBuffer(&p2pBuff->ipcDesc);
      ncclCudaFree(p2pBuff->directPtr);
      free(proxyInfo);
    }
  } else {
    // Do not check return code as CUDA may have already shut down
    ncclCudaFree(connection->transportResources);
  }
  return ncclSuccess;
}

// CE memcpy support
static ncclResult_t p2pSendProxyProgress(struct ncclProxyState* proxyState, struct ncclProxyArgs* args) {
  if (args->state == ncclProxyOpReady) {
    for (int s=0; s<args->nsubs; s++) {
      struct ncclProxySubArgs* sub = args->subs+s;
      struct p2pShmProxyInfo* resources = (struct p2pShmProxyInfo*) (sub->connection->transportResources);
      // Round to next multiple of sliceSteps
      sub->base = ROUNDUP(resources->step, args->chunkSteps);
      sub->posted = sub->transmitted = sub->done = 0;
    }
    args->state = ncclProxyOpProgress;
  }
  args->idle = 1;
  if (args->state == ncclProxyOpProgress) {
    int p = args->protocol;
    int stepSize = proxyState->buffSizes[p] / NCCL_STEPS;
    for (int s=0; s<args->nsubs; s++) {
      struct ncclProxySubArgs* sub = args->subs+s;
      struct p2pShmProxyInfo* resources = (struct p2pShmProxyInfo*) (sub->connection->transportResources);
      if (p != NCCL_PROTO_SIMPLE) { // Only Simple uses cudaMemcpy
          resources->step = sub->base + sub->nsteps;
          args->done++;
          continue;
      }
      if (sub->transmitted < sub->done + NCCL_STEPS && sub->transmitted < sub->nsteps) {
        int buffSlot = (sub->base+sub->transmitted)%NCCL_STEPS;
        volatile struct ncclConnFifo* connFifo = resources->ceRecvMem->connFifo;
        volatile uint64_t* recvTail = &resources->ceRecvMem->tail;
        // Check GPU has sent everything
        if ((*recvTail > sub->base+sub->transmitted)) {
          int size = connFifo[buffSlot].size;
          CUDACHECK(cudaMemcpyAsync(resources->recvFifo+buffSlot*stepSize, resources->ceDevBuff+buffSlot*stepSize, size, cudaMemcpyDeviceToDevice, resources->stream));
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
          // Notify SHM
          resources->shm->recvMem.tail = sub->base + sub->done;
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

struct ncclTransport p2pTransport = {
  "P2P",
  p2pCanConnect,
  { p2pSendSetup, p2pSendConnect, p2pSendFree, NULL, p2pSendProxySetup, NULL, p2pSendProxyFree, NULL },
  { p2pRecvSetup, p2pRecvConnect, p2pRecvFree, NULL, p2pRecvProxySetup, NULL, p2pRecvProxyFree, NULL }
};

static void initCeOperation() {
  static int init = 0;
  if (!init) {
    useMemcpy = ncclParamP2pUseCudaMemcpy();
    if (useMemcpy) {
      p2pTransport.send.proxyConnect = p2pSendProxyConnect;
      p2pTransport.send.proxyProgress = p2pSendProxyProgress;
    }
    init = 1;
  }
}
