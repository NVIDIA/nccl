/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "comm.h"
#include "net.h"
#include "graph.h"
#include "proxy.h"
#include "collectives.h"
#include "gdrwrap.h"
#include "shm.h"
#include "p2p.h"
#include "profiler.h"

static_assert(sizeof(ncclNetHandle_t) <= CONNECT_SIZE, "NET Connect info is too large");

#define NCCL_NET_MAP_HOSTMEM 0
#define NCCL_NET_MAP_DEVMEM 1
#define NCCL_NET_MAP_SHARED_HOSTMEM 2
#define NCCL_NET_MAP_SHARED_DEVMEM 3
#define NCCL_NET_MAP_GDCMEM 4
#define NCCL_NET_MAP_MEMS 5

#define NCCL_NET_MAP_MASK_DEVMEM 0x40000000
#define NCCL_NET_MAP_MASK_SHARED 0x80000000
#define NCCL_NET_MAP_MASK_USED   0x20000000
#define NCCL_NET_MAP_MASK_OFFSET 0x1fffffff

#define NCCL_NET_MAP_OFFSET_BANK(mapStruct, offsetName) \
  ((mapStruct)->offsets.offsetName >> 30)

#define NCCL_NET_MAP_OFFSET_NULL(mapStruct, offsetName) \
  (((mapStruct)->offsets.offsetName >> 29) == 0)

#define NCCL_NET_MAP_GET_POINTER(mapStruct, cpuOrGpu, offsetName) \
  (NCCL_NET_MAP_OFFSET_NULL(mapStruct, offsetName) ? NULL : \
   (mapStruct)->mems[NCCL_NET_MAP_OFFSET_BANK(mapStruct, offsetName)].cpuOrGpu##Ptr + ((mapStruct)->offsets.offsetName & NCCL_NET_MAP_MASK_OFFSET))

#define NCCL_NET_MAP_DEV_MEM(mapStruct, offsetName) \
  (((mapStruct)->offsets.offsetName & NCCL_NET_MAP_MASK_DEVMEM) != 0)

#define NCCL_NET_MAP_ADD_POINTER(mapStruct, shared, dev, memSize, offsetName) do { \
    int bank = NCCL_NET_MAP_MASK_USED + (dev)*NCCL_NET_MAP_MASK_DEVMEM + (shared)*NCCL_NET_MAP_MASK_SHARED; \
    if ((shared) == 0) { \
      if (dev) { \
        (mapStruct)->offsets.offsetName = bank + (mapStruct)->mems[NCCL_NET_MAP_DEVMEM].size; \
        (mapStruct)->mems[NCCL_NET_MAP_DEVMEM].size += memSize; \
      } else { \
        (mapStruct)->offsets.offsetName = bank + (mapStruct)->mems[NCCL_NET_MAP_HOSTMEM].size; \
        (mapStruct)->mems[NCCL_NET_MAP_HOSTMEM].size += memSize; \
      } \
    } else { \
      (mapStruct)->offsets.offsetName = bank; \
    } \
} while (0);

struct connectMapMem{
  char* gpuPtr;
  char* cpuPtr;
  int size;
  ncclIpcDesc ipcDesc;
  char shmPath[PATH_MAX];
  ncclShmHandle_t attachHandle;
  ncclShmHandle_t createHandle;
};

struct connectMap {
  int sameProcess;
  int shared;
  int cudaDev;
  // First 3 bits of offsets determine the mem bank. 001 is host mem, 011 is dev mem, 101 is shared host mem and 111 is shared dev mem.
  struct connectMapMem mems[NCCL_NET_MAP_MEMS];
  // Offsets. 3 MSBs indicate mem bank, 111 indicates NULL.
  struct {
    uint32_t sendMem;
    uint32_t recvMem;
    uint32_t buffs[NCCL_NUM_PROTOCOLS];
  } offsets;
};

struct sendNetResources {
  struct connectMap map;
  void* netSendComm;
  struct ncclSendMem* sendMem;
  struct ncclRecvMem* recvMem;

  int tpRank;
  int tpLocalRank;
  int tpRemoteRank;
  int netDev;
  int useGdr;
  int useDmaBuf;
  int maxRecvs;
  uint64_t* gdcSync;
  void* gdrDesc;
  int shared;
  int channelId;
  int connIndex;
  char* buffers[NCCL_NUM_PROTOCOLS];
  int buffSizes[NCCL_NUM_PROTOCOLS];
  void* mhandles[NCCL_NUM_PROTOCOLS];
  uint64_t step;
  uint64_t llLastCleaning;
  int netDeviceVersion;
  ncclNetDeviceType netDeviceType;
  ncclNetDeviceHandle_t* netDeviceHandle;
};

struct recvNetResources {
  struct connectMap map;
  void* netListenComm;
  void* netRecvComm;
  struct ncclSendMem* sendMem;
  struct ncclRecvMem* recvMem;

  int tpRank;
  int tpLocalRank;
  int tpRemoteRank;
  int tpRemoteProxyRank;
  int netDev;
  int useGdr;
  int useDmaBuf;
  int needFlush;
  int maxRecvs;
  uint64_t* gdcSync;
  uint64_t* gdcFlush;
  void* gdrDesc;
  int shared;
  int channelId;
  int connIndex;
  char* buffers[NCCL_NUM_PROTOCOLS];
  int buffSizes[NCCL_NUM_PROTOCOLS];
  void* mhandles[NCCL_NUM_PROTOCOLS];
  uint64_t step;
  uint64_t llLastCleaning;
  int netDeviceVersion;
  ncclNetDeviceType netDeviceType;
  ncclNetDeviceHandle_t* netDeviceHandle;
};

/* Determine if two peers can communicate with NET */
static ncclResult_t canConnect(int* ret, struct ncclTopoSystem* topo, struct ncclTopoGraph* graph, struct ncclPeerInfo* info1, struct ncclPeerInfo* info2) {
  *ret = 1;
  if (info1->hostHash == info2->hostHash) {
    // If on the same host, check intra-node net is not disabled.
    NCCLCHECK(ncclTopoCheckNet(topo, info1->busId, info2->busId, ret));
  }
  return ncclSuccess;
}

NCCL_PARAM(NetSharedBuffers, "NET_SHARED_BUFFERS", -2);
NCCL_PARAM(NetSharedComms, "NET_SHARED_COMMS", 1);

struct setupReq {
  int tpRank;
  int tpLocalRank;
  int tpRemoteRank;
  int shared;
  int netDev;
  int useGdr;
  int needFlush;
  int channelId;
  int connIndex;
};

// Forward declaration
static ncclResult_t sendProxyProgress(struct ncclProxyState* proxyState, struct ncclProxyArgs* args);

/* Determine if we will use this transport for this peer and return connect
 * information for this peer */
static ncclResult_t sendSetup(struct ncclComm* comm, struct ncclTopoGraph* graph, struct ncclPeerInfo* myInfo, struct ncclPeerInfo* peerInfo, struct ncclConnect* connectInfo, struct ncclConnector* send, int channelId, int connIndex) {
  struct setupReq req = { 0 };
  int tpProxyRank;

  send->conn.shared = req.shared = graph ? 0 : ncclParamNetSharedBuffers() != -2 ? ncclParamNetSharedBuffers() : 1;
  req.channelId = channelId;
  req.connIndex = connIndex;

  int proxyRank;
  NCCLCHECK(ncclTopoGetNetDev(comm, myInfo->rank, graph, channelId, peerInfo->rank, &req.netDev, &proxyRank));
  NCCLCHECK(ncclTopoCheckGdr(comm->topo, myInfo->busId, req.netDev, 1, &req.useGdr));
  send->conn.flags |= req.useGdr ? NCCL_DIRECT_NIC : 0;

  tpProxyRank = comm->topParentRanks[proxyRank];
  NCCLCHECK(ncclProxyConnect(comm, TRANSPORT_NET, 1, tpProxyRank, &send->proxyConn));
  req.tpLocalRank = comm->topParentLocalRanks[comm->localRank];
  req.tpRank = comm->topParentRanks[myInfo->rank];
  req.tpRemoteRank = comm->topParentRanks[peerInfo->rank];
  NCCLCHECK(ncclProxyCallBlocking(comm, &send->proxyConn, ncclProxyMsgSetup, &req, sizeof(req), NULL, 0));

  if (proxyRank == myInfo->rank) {
    INFO(NCCL_INIT|NCCL_NET,"Channel %02d/%d : %d[%d] -> %d[%d] [send] via NET/%s/%d%s%s", channelId, connIndex, myInfo->rank, myInfo->nvmlDev, peerInfo->rank, peerInfo->nvmlDev, comm->ncclNet->name, req.netDev,
        req.useGdr ? "/GDRDMA" : "", req.shared ? "/Shared" : "");
  } else {
    INFO(NCCL_INIT|NCCL_NET,"Channel %02d/%d : %d[%d] -> %d[%d] [send] via NET/%s/%d(%d)%s%s", channelId, connIndex, myInfo->rank, myInfo->nvmlDev, peerInfo->rank, peerInfo->nvmlDev, comm->ncclNet->name, req.netDev,
        proxyRank, req.useGdr ? "/GDRDMA" : "", req.shared ? "/Shared" : "");
  }
  *((int*)connectInfo) = tpProxyRank;
  return ncclSuccess;
}

// GDRCOPY support: TAIL_ENABLE When enabled locates the RX proxy tail in CUDA memory
NCCL_PARAM(GdrCopySyncEnable, "GDRCOPY_SYNC_ENABLE", 1);
// GDRCOPY support: FLUSH_ENABLE When enabled uses a PCI-E read to flush GDRDMA buffers
NCCL_PARAM(GdrCopyFlushEnable, "GDRCOPY_FLUSH_ENABLE", 0);

/* Setup recv connector */
static ncclResult_t recvSetup(struct ncclComm* comm, struct ncclTopoGraph* graph, struct ncclPeerInfo* myInfo, struct ncclPeerInfo* peerInfo, struct ncclConnect* connectInfo, struct ncclConnector* recv, int channelId, int connIndex) {
  struct setupReq req = { 0 };

  recv->conn.shared = req.shared = graph ? 0 : ncclParamNetSharedBuffers() != -2 ? ncclParamNetSharedBuffers() : 1;
  req.channelId = channelId;
  req.connIndex = connIndex;

  // Use myInfo->rank as the receiver uses its own NIC
  int proxyRank, tpProxyRank;
  NCCLCHECK(ncclTopoGetNetDev(comm, myInfo->rank, graph, channelId, myInfo->rank, &req.netDev, &proxyRank));
  NCCLCHECK(ncclTopoCheckGdr(comm->topo, myInfo->busId, req.netDev, 0, &req.useGdr));

  // Determine whether we need to flush the GDR buffer on recv or not
  if (req.useGdr) NCCLCHECK(ncclTopoNeedFlush(comm->topo, myInfo->busId, &req.needFlush));

  // We don't support PXN on receive yet
  tpProxyRank = comm->topParentRanks[myInfo->rank];
  NCCLCHECK(ncclProxyConnect(comm, TRANSPORT_NET, 0, tpProxyRank, &recv->proxyConn));

  req.tpLocalRank = comm->topParentLocalRanks[comm->localRank];
  req.tpRank = comm->topParentRanks[myInfo->rank];
  req.tpRemoteRank = comm->topParentRanks[peerInfo->rank];
  NCCLCHECK(ncclProxyCallBlocking(comm, &recv->proxyConn, ncclProxyMsgSetup, &req, sizeof(req), connectInfo, sizeof(ncclNetHandle_t)));
  INFO(NCCL_INIT|NCCL_NET,"Channel %02d/%d : %d[%d] -> %d[%d] [receive] via NET/%s/%d%s%s", channelId, connIndex, peerInfo->rank, peerInfo->nvmlDev, myInfo->rank, myInfo->nvmlDev, comm->ncclNet->name, req.netDev,
      req.useGdr ? "/GDRDMA" : "", req.shared ? "/Shared" : "");
  return ncclSuccess;
}

static ncclResult_t netMapShm(struct connectMapMem* mem) {
  NCCLCHECK(ncclShmOpen(mem->shmPath, mem->size, (void**)&mem->cpuPtr, (void**)&mem->gpuPtr, -1, &mem->attachHandle));
  return ncclSuccess;
}
static ncclResult_t netCreateShm(struct connectMapMem* mem) {
  mem->shmPath[0] = '\0'; // Let ncclShmOpen create a tmp file
  NCCLCHECK(ncclShmOpen(mem->shmPath, mem->size, (void**)&mem->cpuPtr, NULL, 1, &mem->createHandle));
  return ncclSuccess;
}

static ncclResult_t netDumpMap(struct connectMap* map) {
  printf("Dump map same process %d shared %d\n", map->sameProcess, map->shared);
  struct connectMapMem *mem = map->mems+NCCL_NET_MAP_HOSTMEM;
  printf("Mem 0: Host mem %s (%x B) CPU %p GPU %p\n", mem->shmPath, mem->size, mem->cpuPtr, mem->gpuPtr);
  mem = map->mems+NCCL_NET_MAP_DEVMEM;
  printf("Mem 1: Vid  mem (%x B) CPU %p GPU %p\n", mem->size, mem->cpuPtr, mem->gpuPtr);
  mem = map->mems+NCCL_NET_MAP_SHARED_HOSTMEM;
  printf("Mem 2: Shared Host mem %s (%x B) CPU %p GPU %p\n", mem->shmPath, mem->size, mem->cpuPtr, mem->gpuPtr);
  mem = map->mems+NCCL_NET_MAP_SHARED_DEVMEM;
  printf("Mem 3: Shared Vid mem (%x B) CPU %p GPU %p\n", mem->size, mem->cpuPtr, mem->gpuPtr);
  printf("SendMem -> Used %d Bank %d Offset %x, cpu %p gpu %p\n",
      map->offsets.sendMem & NCCL_NET_MAP_MASK_USED ? 1 : 0,
      NCCL_NET_MAP_OFFSET_BANK(map, sendMem), map->offsets.sendMem & NCCL_NET_MAP_MASK_OFFSET,
      NCCL_NET_MAP_GET_POINTER(map, cpu, sendMem), NCCL_NET_MAP_GET_POINTER(map, gpu, sendMem));
  printf("RecvMem -> Used %d Bank %d Offset %x, cpu %p gpu %p\n",
      map->offsets.recvMem & NCCL_NET_MAP_MASK_USED ? 1 : 0,
      NCCL_NET_MAP_OFFSET_BANK(map, recvMem), map->offsets.recvMem & NCCL_NET_MAP_MASK_OFFSET,
      NCCL_NET_MAP_GET_POINTER(map, cpu, recvMem), NCCL_NET_MAP_GET_POINTER(map, gpu, recvMem));
  for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
    printf("Proto %d -> Used %d Bank %d Offset %x, cpu %p, gpu %p\n", p,
        map->offsets.buffs[p] & NCCL_NET_MAP_MASK_USED ? 1 : 0,
        NCCL_NET_MAP_OFFSET_BANK(map, buffs[p]), map->offsets.buffs[p] & NCCL_NET_MAP_MASK_OFFSET,
        NCCL_NET_MAP_GET_POINTER(map, cpu, buffs[p]), NCCL_NET_MAP_GET_POINTER(map, gpu, buffs[p]));
  }
  printf("End of dump\n");
  return ncclSuccess;
}

struct netSendConnectArgs {
  ncclNetHandle_t handle;
};

struct netRecvConnectArgs {
  int proxyRank;
};

static ncclResult_t sendConnect(struct ncclComm* comm, struct ncclConnect* connectInfo, int nranks, int rank, struct ncclConnector* send) {
  struct connectMap* map = (connectMap*) send->transportResources;

  void* opId;

  // map isn't allocated thus this op hasn't been submitted yet
  if (!map) {
    // Setup device pointers
    NCCLCHECK(ncclCalloc(&map, 1));
    send->transportResources = map;
    opId = send;
    INFO(NCCL_PROXY, "sendConnect ncclProxyCallAsync opId=%p", opId);
    netSendConnectArgs args = {0};
    memcpy(&args.handle, connectInfo, sizeof(ncclNetHandle_t));
    NCCLCHECK(ncclProxyCallAsync(comm, &send->proxyConn, ncclProxyMsgConnect, &args, sizeof(netSendConnectArgs), sizeof(struct connectMap), opId));
  } else {
    opId =  send;
  }

  ncclResult_t ret;
  NCCLCHECK(ret = ncclPollProxyResponse(comm, &send->proxyConn, map, opId));
  if (ret == ncclInProgress) {
    return ret;
  }
  INFO(NCCL_PROXY, "sendConnect ncclPollProxyResponse opId=%p", opId);

  if (map->sameProcess && !ncclCuMemEnable()) {
    if (map->cudaDev != comm->cudaDev) {
      // Enable P2P access for Legacy IPC
      cudaError_t err = cudaDeviceEnablePeerAccess(map->cudaDev, 0);
      if (err == cudaErrorPeerAccessAlreadyEnabled) {
        cudaGetLastError();
      } else if (err != cudaSuccess) {
        WARN("failed to peer with device %d: %d %s", map->cudaDev, err, cudaGetErrorString(err));
        return ncclInternalError;
      }
    }
  } else if (!(map->sameProcess && map->cudaDev == comm->cudaDev)) {
    if (!map->sameProcess) NCCLCHECK(netMapShm(map->mems+NCCL_NET_MAP_HOSTMEM));
    if (map->mems[NCCL_NET_MAP_DEVMEM].size) {
      NCCLCHECK(ncclP2pImportShareableBuffer(comm, send->proxyConn.tpRank,
                                             map->mems[NCCL_NET_MAP_DEVMEM].size,
                                             &map->mems[NCCL_NET_MAP_DEVMEM].ipcDesc,
                                             (void**)&map->mems[NCCL_NET_MAP_DEVMEM].gpuPtr));
      map->mems[NCCL_NET_MAP_DEVMEM].cpuPtr = NULL;
    }
    if (map->mems[NCCL_NET_MAP_SHARED_DEVMEM].size) {
      void** sharedDevMemPtr = comm->proxyState->sharedDevMems + send->proxyConn.tpLocalRank;
      if (*sharedDevMemPtr == NULL) {
        NCCLCHECK(ncclP2pImportShareableBuffer(comm, send->proxyConn.tpRank,
                                               map->mems[NCCL_NET_MAP_SHARED_DEVMEM].size,
                                               &map->mems[NCCL_NET_MAP_SHARED_DEVMEM].ipcDesc,
                                               sharedDevMemPtr));
      }
      map->mems[NCCL_NET_MAP_SHARED_DEVMEM].gpuPtr = (char*)(*sharedDevMemPtr);
      map->mems[NCCL_NET_MAP_SHARED_DEVMEM].cpuPtr = NULL;
    }
  }
  //NCCLCHECK(netDumpMap(map));

  struct ncclSendMem *sendMem = (struct ncclSendMem*) NCCL_NET_MAP_GET_POINTER(map, gpu, sendMem);
  void* gdcMem = map->mems[NCCL_NET_MAP_GDCMEM].gpuPtr;
  send->conn.head = gdcMem ? (uint64_t*)gdcMem : &sendMem->head;

  struct ncclRecvMem *recvMem = (struct ncclRecvMem*) NCCL_NET_MAP_GET_POINTER(map, gpu, recvMem);
  send->conn.tail = &recvMem->tail;
  send->conn.connFifo = recvMem->connFifo;
  // Only fuse P2P buffers, continue to allocate dedicated buffers for ring/tree
  for (int i=0; i<NCCL_STEPS; i++) {
    send->conn.connFifo[i].offset = -1;
    recvMem->connFifo[i].mode = map->shared ? NCCL_MODE_OFFSET : NCCL_MODE_NORMAL;
  }

  for (int p=0; p<NCCL_NUM_PROTOCOLS; p++)
    send->conn.buffs[p] = NCCL_NET_MAP_GET_POINTER(map, gpu, buffs[p]);

  if (send->proxyConn.sameProcess) {
    if (send->proxyConn.connection->netDeviceHandle) {
      send->conn.netDeviceHandle = *send->proxyConn.connection->netDeviceHandle;

      for (int p=0; p<NCCL_NUM_PROTOCOLS; p++)
        send->conn.mhandles[p] = send->proxyConn.connection->mhandles[p];
    }

    if (send->proxyConn.connection->needsProxyProgress) {
      send->proxyConn.proxyProgress = sendProxyProgress;
    } else {
      send->proxyConn.proxyProgress = NULL;
    }
  } else {
    send->proxyConn.proxyProgress = sendProxyProgress;
  }

  return ncclSuccess;
}

// Forward declare
static ncclResult_t recvProxyProgress(struct ncclProxyState* proxyState, struct ncclProxyArgs* args);

/* Connect to this peer */
static ncclResult_t recvConnect(struct ncclComm* comm, struct ncclConnect* connectInfo, int nranks, int rank, struct ncclConnector* recv) {
  struct connectMap* map = (connectMap*) recv->transportResources;
  void* opId;
  if (!map) {
    NCCLCHECK(ncclCalloc(&map, 1));
    recv->transportResources = map;
    // Use recv connector as unique identifier
    opId = recv;
    INFO(NCCL_PROXY, "recvConnect ncclProxyCallAsync opId=%p &recv->proxyConn=%p connectInfo=%p",
       opId, &recv->proxyConn, connectInfo);
    netRecvConnectArgs args = {0};
    args.proxyRank = *((int*)connectInfo);
    NCCLCHECK(ncclProxyCallAsync(comm, &recv->proxyConn, ncclProxyMsgConnect, &args, sizeof(netRecvConnectArgs), sizeof(struct connectMap), opId));
  } else {
    opId = recv;
  }

  ncclResult_t ret;
  NCCLCHECK(ret = ncclPollProxyResponse(comm, &recv->proxyConn, map, opId));
  if (ret == ncclInProgress) {
    return ret;
  }
  INFO(NCCL_PROXY, "recvConnect ncclPollProxyResponse opId=%p", opId);
  //NCCLCHECK(netDumpMap(map));

  struct ncclSendMem *sendMem = (struct ncclSendMem*) NCCL_NET_MAP_GET_POINTER(map, gpu, sendMem);
  recv->conn.head = &sendMem->head;

  struct ncclRecvMem *recvMem = (struct ncclRecvMem*) NCCL_NET_MAP_GET_POINTER(map, gpu, recvMem);
  void* gdcMem = map->mems[NCCL_NET_MAP_GDCMEM].gpuPtr;
  recv->conn.tail = gdcMem ? (uint64_t*)gdcMem : &recvMem->tail;
  recv->conn.connFifo = recvMem->connFifo;
  // Only fuse P2P buffers, continue to allocate dedicated buffers for ring/tree
  for (int i=0; i<NCCL_STEPS; i++) {
    recvMem->connFifo[i].mode = map->shared ? NCCL_MODE_OFFSET : NCCL_MODE_NORMAL;
  }

  for (int p=0; p<NCCL_NUM_PROTOCOLS; p++)
    recv->conn.buffs[p] = NCCL_NET_MAP_GET_POINTER(map, gpu, buffs[p]);

  if (recv->proxyConn.sameProcess) {
    if (recv->proxyConn.connection->netDeviceHandle) {
      recv->conn.netDeviceHandle = *recv->proxyConn.connection->netDeviceHandle;

      for (int p=0; p<NCCL_NUM_PROTOCOLS; p++)
        recv->conn.mhandles[p] = recv->proxyConn.connection->mhandles[p];
    }

    if (recv->proxyConn.connection->needsProxyProgress) {
      recv->proxyConn.proxyProgress = recvProxyProgress;
    } else {
      recv->proxyConn.proxyProgress = NULL;
    }
  } else {
    recv->proxyConn.proxyProgress = recvProxyProgress;
  }

  return ncclSuccess;
}

static ncclResult_t sendFree(struct ncclConnector* send) {
  struct connectMap* map = (struct connectMap*)(send->transportResources);
  if (map) {
    int cudaDev;
    CUDACHECK(cudaGetDevice(&cudaDev));
    if (map->sameProcess && map->cudaDev == cudaDev) {
      // Our own GPU, so it wasn't mapped in
      free(map);
      return ncclSuccess;
    }
    if (!map->sameProcess || ncclCuMemEnable()) {
      if (!map->sameProcess) NCCLCHECK(ncclShmClose(map->mems[NCCL_NET_MAP_HOSTMEM].attachHandle));
      if (map->mems[NCCL_NET_MAP_DEVMEM].size) {
        if (ncclCuMemEnable()) {
          // cuMem API support
          NCCLCHECK(ncclP2pFreeShareableBuffer(&map->mems[NCCL_NET_MAP_DEVMEM].ipcDesc));
          NCCLCHECK(ncclCuMemFree(map->mems[NCCL_NET_MAP_DEVMEM].gpuPtr));
        } else {
          // Legacy CUDA IPC support
          CUDACHECK(cudaIpcCloseMemHandle(map->mems[NCCL_NET_MAP_DEVMEM].gpuPtr));
        }
      }
    }
    free(map);
  }

  return ncclSuccess;
}

static ncclResult_t recvFree(struct ncclConnector* recv) {
  if (recv->transportResources) free(recv->transportResources);
  return ncclSuccess;
}

#define NCCL_SHARED_STEPS 16
static ncclResult_t sharedNetBuffersInit(struct ncclProxyState* proxyState, int cuda, int tpLocalRank, int type, int sameProcess,
    int nChannels, char** gpuPtr, char** cpuPtr, int* size, ncclIpcDesc *ipcDesc) {
  if (cuda == 0 && sameProcess == 0) {
      WARN("PXN should not use host buffers for data");
      return ncclInternalError;
  }
  struct ncclProxyProgressState* progressState = &proxyState->progressState;
  if (progressState->localPeers == NULL) {
    NCCLCHECK(ncclCalloc(&progressState->localPeers, proxyState->tpLocalnRanks));
  }
  struct ncclProxyPeer** localPeers = progressState->localPeers;
  if (localPeers[tpLocalRank] == NULL) {
    NCCLCHECK(ncclCalloc(localPeers + tpLocalRank, 1));
  }
  struct ncclProxyPeer* peer = localPeers[tpLocalRank];
  struct ncclProxySharedP2p* state = type == 0 ? &peer->send : &peer->recv;
  state->refcount++;
  if (state->size == 0) {
    state->size = nChannels * NCCL_SHARED_STEPS * proxyState->p2pChunkSize;
  }

  if (size) *size = state->size;

  if (cuda && state->cudaBuff == NULL) {
    if (sameProcess == 0 || ncclCuMemEnable()) {
      NCCLCHECK(ncclP2pAllocateShareableBuffer(state->size, &state->ipcDesc, (void**)&state->cudaBuff));
    } else {
      NCCLCHECK(ncclCudaCalloc(&state->cudaBuff, state->size));
    }
  }
  if (!cuda && state->hostBuff == NULL) {
    NCCLCHECK(ncclCudaHostCalloc(&state->hostBuff, state->size));
  }
  if (cpuPtr) *cpuPtr = cuda ? state->cudaBuff : state->hostBuff;
  if (gpuPtr) *gpuPtr = sameProcess ? *cpuPtr : NULL;
  if (ipcDesc) memcpy(ipcDesc, &state->ipcDesc, sizeof(state->ipcDesc));
  return ncclSuccess;
}

static ncclResult_t sharedBuffersGet(struct ncclProxyState* proxyState, int channel, int slot, int* offset, int* size) {
  // Use different pools for different channels and also separate send/recv.
  int globalSlot = (channel*NCCL_SHARED_STEPS)+slot;
  *offset = proxyState->p2pChunkSize * globalSlot;
  if (size) *size = proxyState->p2pChunkSize;
  return ncclSuccess;
}

static ncclResult_t sharedNetBuffersDestroy(struct ncclProxyState* proxyState, int tpLocalRank, int type, struct ncclProxyConnection* connection) {
  if (proxyState->progressState.localPeers == NULL) NCCLCHECK(ncclInternalError);
  struct ncclProxyPeer* peer = proxyState->progressState.localPeers[tpLocalRank];
  if (peer == NULL) NCCLCHECK(ncclInternalError;)
  struct ncclProxySharedP2p* state = type == 0 ? &peer->send : &peer->recv;
  if (state->size == 0) NCCLCHECK(ncclInternalError);
  if (ncclAtomicRefCountDecrement(&state->refcount) == 0) {
    if (state->cudaBuff) {
      if (!connection->sameProcess || ncclCuMemEnable()) {
        NCCLCHECK(ncclP2pFreeShareableBuffer(&state->ipcDesc));
      }
      NCCLCHECK(ncclCudaFree(state->cudaBuff));
    }
    if (state->hostBuff) NCCLCHECK(ncclCudaHostFree(state->hostBuff));
  }

  if (peer->send.refcount || peer->recv.refcount) return ncclSuccess;

  free(peer);
  proxyState->progressState.localPeers[tpLocalRank] = NULL;
  for (int r = 0; r < proxyState->tpLocalnRanks; r++) {
    if (proxyState->progressState.localPeers[r]) return ncclSuccess;
  }
  // All peers are freed, free array
  free(proxyState->progressState.localPeers);
  proxyState->progressState.localPeers = NULL;
  return ncclSuccess;
}

static ncclResult_t proxySharedInit(struct ncclProxyConnection* connection, struct ncclProxyState* proxyState, int nChannels) {
  NCCLCHECK(sharedNetBuffersInit(proxyState, 1, connection->tpLocalRank, 0, connection->sameProcess, nChannels, NULL, NULL, NULL, NULL));
  return ncclSuccess;
}

static ncclResult_t sendProxySetup(struct ncclProxyConnection* connection, struct ncclProxyState* proxyState, void* reqBuff, int reqSize, void* respBuff, int respSize, int* done) {
  struct setupReq* req = (struct setupReq*) reqBuff;
  if (reqSize != sizeof(struct setupReq)) return ncclInternalError;

  struct sendNetResources* resources;
  NCCLCHECK(ncclCalloc(&resources, 1));
  connection->transportResources = resources;

  resources->tpRank = req->tpRank;
  resources->tpLocalRank = req->tpLocalRank;
  resources->tpRemoteRank = req->tpRemoteRank;
  resources->netDev = req->netDev;
  resources->shared = connection->shared = req->shared;
  resources->useGdr = req->useGdr;
  resources->channelId = req->channelId;
  resources->connIndex = req->connIndex;
  ncclNetProperties_t props;
  NCCLCHECK(proxyState->ncclNet->getProperties(req->netDev, &props));
  /* DMA-BUF support */
  resources->useDmaBuf = resources->useGdr && proxyState->dmaBufSupport && (props.ptrSupport & NCCL_PTR_DMABUF);
  resources->maxRecvs = props.maxRecvs;
  resources->netDeviceVersion = props.netDeviceVersion;
  resources->netDeviceType = props.netDeviceType;

  resources->netDeviceVersion = props.netDeviceVersion;
  resources->netDeviceType = props.netDeviceType;

  // We don't return any data
  if (respSize != 0) return ncclInternalError;
  *done = 1;
  return ncclSuccess;
}

static ncclResult_t recvProxySetup(struct ncclProxyConnection* connection, struct ncclProxyState* proxyState, void* reqBuff, int reqSize, void* respBuff, int respSize, int* done) {
  struct setupReq* req = (struct setupReq*) reqBuff;
  if (reqSize != sizeof(struct setupReq)) return ncclInternalError;

  struct recvNetResources* resources;
  NCCLCHECK(ncclCalloc(&resources, 1));
  connection->transportResources = resources;

  resources->tpRank = req->tpRank;
  resources->tpLocalRank = req->tpLocalRank;
  resources->tpRemoteRank = req->tpRemoteRank;
  resources->netDev = req->netDev;
  resources->shared = connection->shared = req->shared;
  resources->useGdr = req->useGdr;
  resources->needFlush = req->needFlush;
  resources->channelId = req->channelId;
  resources->connIndex = req->connIndex;
  ncclNetProperties_t props;
  NCCLCHECK(proxyState->ncclNet->getProperties(req->netDev, &props));
  /* DMA-BUF support */
  resources->useDmaBuf = resources->useGdr && proxyState->dmaBufSupport && (props.ptrSupport & NCCL_PTR_DMABUF);
  resources->maxRecvs = props.maxRecvs;
  resources->netDeviceVersion = props.netDeviceVersion;
  resources->netDeviceType = props.netDeviceType;

  if (respSize != sizeof(ncclNetHandle_t)) return ncclInternalError;
  NCCLCHECK(proxyState->ncclNet->listen(req->netDev, respBuff, &resources->netListenComm));
  *done = 1;

  return ncclSuccess;
}

// This function embeds plugin-specific rules given the current versions
static ncclResult_t ncclNetGetDeviceHandle(ncclNetDeviceType type, int version, bool isRecv, ncclNetDeviceHandle_t** handle) {
  bool needsDeviceHandle  = false;

  if (type == NCCL_NET_DEVICE_UNPACK) {
    if (version == NCCL_NET_DEVICE_UNPACK_VERSION && isRecv) {
      needsDeviceHandle  = true;
    }
  }

  // Don't re-alloc netDeviceHandles
  if (needsDeviceHandle && (*handle == NULL)) {
    NCCLCHECK(ncclCalloc(handle, 1));
    (*handle)->netDeviceType = type;
    (*handle)->netDeviceVersion = version;
  } else if (!needsDeviceHandle) {
    *handle = NULL;
  }

  return ncclSuccess;
}

static ncclResult_t sendProxyConnect(struct ncclProxyConnection* connection, struct ncclProxyState* proxyState, void* reqBuff, int reqSize, void* respBuff, int respSize, int* done) {
  struct sendNetResources* resources = (struct sendNetResources*)(connection->transportResources);
  if (reqSize != sizeof(netSendConnectArgs)) return ncclInternalError;
  ncclResult_t ret = ncclSuccess;
  netSendConnectArgs* req = (netSendConnectArgs*) reqBuff;
  NCCLCHECK(ncclNetGetDeviceHandle(resources->netDeviceType, resources->netDeviceVersion, false /*isRecv*/, &resources->netDeviceHandle));
  if (resources->shared) {
    // Shared buffers
    struct ncclProxyProgressState* progressState = &proxyState->progressState;
    if (progressState->localPeers == NULL) {
      NCCLCHECK(ncclCalloc(&progressState->localPeers, proxyState->tpLocalnRanks));
    }
    struct ncclProxyPeer** localPeers = progressState->localPeers;
    if (localPeers[resources->tpLocalRank] == NULL) {
      NCCLCHECK(ncclCalloc(localPeers + resources->tpLocalRank, 1));
    }
    connection->proxyAppendPtr = localPeers[resources->tpLocalRank]->send.proxyAppend + resources->channelId;

    if (resources->maxRecvs > 1 && ncclParamNetSharedComms()) {
      // Connect or reuse connection for a netdev/remote rank.
      if (progressState->netComms[resources->netDev] == NULL) {
        NCCLCHECK(ncclCalloc(progressState->netComms + resources->netDev, proxyState->tpnRanks));
      }
      struct ncclSharedNetComms* comms = progressState->netComms[resources->netDev] + resources->tpRemoteRank;
      if (comms->sendComm[resources->channelId] == NULL) ret = proxyState->ncclNet->connect(resources->netDev, req->handle, comms->sendComm + resources->channelId, &resources->netDeviceHandle);
      resources->netSendComm = comms->sendComm[resources->channelId];
      if (comms->sendComm[resources->channelId]) comms->sendRefCount[resources->channelId]++;
    } else {
      ret = proxyState->ncclNet->connect(resources->netDev, req->handle, &resources->netSendComm, &resources->netDeviceHandle);
    }
  } else {
    // Connect to remote peer
    ret = proxyState->ncclNet->connect(resources->netDev, req->handle, &resources->netSendComm, &resources->netDeviceHandle);
    connection->proxyAppendPtr = &connection->proxyAppend;
  }

  NCCLCHECK(ret);
  if (resources->netSendComm == NULL) {
    *done = 0;
    return ncclInProgress;
  }
  *done = 1;

  if (resources->netDeviceHandle) {
    connection->netDeviceHandle = resources->netDeviceHandle;
    connection->needsProxyProgress = connection->netDeviceHandle->needsProxyProgress;
  } else {
    connection->needsProxyProgress = 1;
  }

  // Create structures
  struct connectMap* map = &resources->map;
  map->sameProcess = connection->sameProcess;
  map->shared = resources->shared;
  CUDACHECK(cudaGetDevice(&map->cudaDev));

  if (resources->shared == 0) { // Only allocate dedicated buffers for ring/tree, not for p2p
    for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
      NCCL_NET_MAP_ADD_POINTER(map, 0, p!= NCCL_PROTO_LL && resources->useGdr, proxyState->buffSizes[p], buffs[p]);
      resources->buffSizes[p] = proxyState->buffSizes[p];
    }
  } else {
    // Get shared buffers
    int bank = resources->useGdr ? NCCL_NET_MAP_SHARED_DEVMEM : NCCL_NET_MAP_SHARED_HOSTMEM;
    struct connectMapMem* mapMem = map->mems+bank;
    NCCLCHECK(sharedNetBuffersInit(
          proxyState, resources->useGdr, resources->tpLocalRank, 0, map->sameProcess, proxyState->p2pnChannels,
          &mapMem->gpuPtr, &mapMem->cpuPtr, &mapMem->size, &mapMem->ipcDesc));
    resources->buffSizes[NCCL_PROTO_SIMPLE] = mapMem->size;

    if (proxyState->allocP2pNetLLBuffers) {
      NCCL_NET_MAP_ADD_POINTER(map, 0, 0 /*p == NCCL_PROTO_LL*/, proxyState->buffSizes[NCCL_PROTO_LL], buffs[NCCL_PROTO_LL]);
      resources->buffSizes[NCCL_PROTO_LL] = proxyState->buffSizes[NCCL_PROTO_LL];
    }

    NCCL_NET_MAP_ADD_POINTER(map, 1, resources->useGdr, mapMem->size, buffs[NCCL_PROTO_SIMPLE]);
  }

  NCCL_NET_MAP_ADD_POINTER(map, 0, 0, sizeof(struct ncclSendMem), sendMem);
  NCCL_NET_MAP_ADD_POINTER(map, 0, 0, sizeof(struct ncclRecvMem), recvMem);

  if (map->mems[NCCL_NET_MAP_DEVMEM].size) {
    if (resources->shared == 0) {
      if (!map->sameProcess || ncclCuMemEnable()) {
        ALIGN_SIZE(map->mems[NCCL_NET_MAP_DEVMEM].size, CUDA_IPC_MIN);
        NCCLCHECK(ncclP2pAllocateShareableBuffer(map->mems[NCCL_NET_MAP_DEVMEM].size, &map->mems[NCCL_NET_MAP_DEVMEM].ipcDesc,
                                                 (void**)&map->mems[NCCL_NET_MAP_DEVMEM].gpuPtr));
      } else {
        NCCLCHECK(ncclCudaCalloc(&map->mems[NCCL_NET_MAP_DEVMEM].gpuPtr, map->mems[NCCL_NET_MAP_DEVMEM].size));
      }
      map->mems[NCCL_NET_MAP_DEVMEM].cpuPtr = map->mems[NCCL_NET_MAP_DEVMEM].gpuPtr;
    }
  }
  if (map->sameProcess) {
    NCCLCHECK(ncclCudaHostCalloc(&map->mems[NCCL_NET_MAP_HOSTMEM].cpuPtr, map->mems[NCCL_NET_MAP_HOSTMEM].size));
    map->mems[NCCL_NET_MAP_HOSTMEM].gpuPtr = map->mems[NCCL_NET_MAP_HOSTMEM].cpuPtr;
  } else {
    NCCLCHECK(netCreateShm(map->mems+NCCL_NET_MAP_HOSTMEM));
  }
  if (ncclGdrCopy && map->sameProcess && ncclParamGdrCopySyncEnable()) {
    uint64_t *cpuPtr, *gpuPtr;
    NCCLCHECK(ncclGdrCudaCalloc(&cpuPtr, &gpuPtr, 1, &resources->gdrDesc));

    resources->gdcSync = cpuPtr;
    struct connectMapMem* gdcMem = map->mems+NCCL_NET_MAP_GDCMEM;
    gdcMem->cpuPtr = (char*)cpuPtr;
    gdcMem->gpuPtr = (char*)gpuPtr;
    gdcMem->size = sizeof(uint64_t); // sendMem->head
  }

  resources->sendMem = (struct ncclSendMem*) NCCL_NET_MAP_GET_POINTER(map, cpu, sendMem);
  resources->recvMem = (struct ncclRecvMem*) NCCL_NET_MAP_GET_POINTER(map, cpu, recvMem);

  // Don't give credits yet in shared mode.
  (resources->gdcSync ? *resources->gdcSync : resources->sendMem->head) =
    (map->shared ? -NCCL_STEPS : 0);
  for (int i=0; i<NCCL_STEPS; i++) resources->recvMem->connFifo[i].size = -1;

  for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
    resources->buffers[p] = NCCL_NET_MAP_GET_POINTER(map, cpu, buffs[p]);
    if (resources->buffers[p]) {
#if CUDA_VERSION >= 11070
      /* DMA-BUF support */
      int type = NCCL_NET_MAP_DEV_MEM(map, buffs[p]) ? NCCL_PTR_CUDA : NCCL_PTR_HOST;
      if (type == NCCL_PTR_CUDA && resources->useDmaBuf) {
        int dmabuf_fd;
        CUCHECK(cuMemGetHandleForAddressRange((void *)&dmabuf_fd, (CUdeviceptr)resources->buffers[p], resources->buffSizes[p], CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, 0));
        NCCLCHECK(proxyState->ncclNet->regMrDmaBuf(resources->netSendComm, resources->buffers[p], resources->buffSizes[p], type, 0ULL, dmabuf_fd, &resources->mhandles[p]));
        (void)close(dmabuf_fd);
      } else // FALL-THROUGH to nv_peermem GDR path
#endif
      {
        NCCLCHECK(proxyState->ncclNet->regMr(resources->netSendComm, resources->buffers[p], resources->buffSizes[p], NCCL_NET_MAP_DEV_MEM(map, buffs[p]) ? NCCL_PTR_CUDA : NCCL_PTR_HOST, &resources->mhandles[p]));
      }

      // Copy the mhandle dptr, if implemented
      if (resources->netDeviceHandle && proxyState->ncclNet->getDeviceMr)
        NCCLCHECK(proxyState->ncclNet->getDeviceMr(resources->netSendComm, resources->mhandles[p], &connection->mhandles[p]));
    }
  }

  //NCCLCHECK(netDumpMap(map));
  if (respSize != sizeof(struct connectMap)) return ncclInternalError;
  memcpy(respBuff, map, sizeof(struct connectMap));
  return ncclSuccess;
}

static ncclResult_t recvProxyConnect(struct ncclProxyConnection* connection, struct ncclProxyState* proxyState, void* reqBuff, int reqSize, void* respBuff, int respSize, int* done) {
  if (reqSize != sizeof(netRecvConnectArgs)) return ncclInternalError;
  struct recvNetResources* resources = (struct recvNetResources*)(connection->transportResources);
  netRecvConnectArgs* req = (netRecvConnectArgs*) reqBuff;
  resources->tpRemoteProxyRank = req->proxyRank;
  ncclResult_t ret = ncclSuccess;

  NCCLCHECK(ncclNetGetDeviceHandle(resources->netDeviceType, resources->netDeviceVersion, true /*isRecv*/, &resources->netDeviceHandle));
  // Finish connection establishment from remote peer
  if (resources->shared) {
    // Shared buffers
    struct ncclProxyProgressState* progressState = &proxyState->progressState;
    if (progressState->localPeers == NULL) {
      NCCLCHECK(ncclCalloc(&progressState->localPeers, proxyState->tpLocalnRanks));
    }
    struct ncclProxyPeer** localPeers = progressState->localPeers;
    if (localPeers[resources->tpLocalRank] == NULL) {
      NCCLCHECK(ncclCalloc(localPeers + resources->tpLocalRank, 1));
    }
    connection->proxyAppendPtr = localPeers[resources->tpLocalRank]->recv.proxyAppend + resources->channelId;

    if (resources->maxRecvs > 1 && ncclParamNetSharedComms()) {
      // Connect or reuse connection for a netdev/remote rank.
      if (progressState->netComms[resources->netDev] == NULL) {
        NCCLCHECK(ncclCalloc(progressState->netComms + resources->netDev, proxyState->tpnRanks));
      }
      struct ncclSharedNetComms* comms = progressState->netComms[resources->netDev] + resources->tpRemoteProxyRank;
      if (comms->recvComm[resources->channelId] == NULL) ret = proxyState->ncclNet->accept(resources->netListenComm, comms->recvComm+resources->channelId, &resources->netDeviceHandle);
      resources->netRecvComm = comms->recvComm[resources->channelId];
      if (comms->recvComm[resources->channelId]) comms->recvRefCount[resources->channelId]++;
    } else {
      ret = proxyState->ncclNet->accept(resources->netListenComm, &resources->netRecvComm, &resources->netDeviceHandle);
    }
  } else {
    // Connect to remote peer
    ret = proxyState->ncclNet->accept(resources->netListenComm, &resources->netRecvComm, &resources->netDeviceHandle);
    connection->proxyAppendPtr = &connection->proxyAppend;
  }

  NCCLCHECK(ret);
  if (resources->netRecvComm == NULL) {
    *done = 0;
    return ncclInProgress;
  }
  *done = 1;

  if (resources->netDeviceHandle) {
    connection->netDeviceHandle = resources->netDeviceHandle;
    connection->needsProxyProgress = connection->netDeviceHandle->needsProxyProgress;
  } else {
    connection->needsProxyProgress = 1;
  }

  NCCLCHECK(proxyState->ncclNet->closeListen(resources->netListenComm));

  // Create structures
  struct connectMap* map = &resources->map;
  map->sameProcess = connection->sameProcess;
  if (map->sameProcess == 0) return ncclInternalError; // We don't support remote proxy for recv
  map->shared = resources->shared;

  if (resources->shared == 0) { // Only allocate dedicated buffers for ring/tree, not for p2p
    for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
      NCCL_NET_MAP_ADD_POINTER(map, 0, resources->useGdr, proxyState->buffSizes[p], buffs[p]);
      resources->buffSizes[p] = proxyState->buffSizes[p];
    }
  } else {
    // Get shared buffers
    int bank = resources->useGdr ? NCCL_NET_MAP_SHARED_DEVMEM : NCCL_NET_MAP_SHARED_HOSTMEM;
    struct connectMapMem* mapMem = map->mems+bank;
    NCCLCHECK(sharedNetBuffersInit(
          proxyState, resources->useGdr, resources->tpLocalRank, 1, 1, proxyState->p2pnChannels,
          &mapMem->gpuPtr, &mapMem->cpuPtr, &mapMem->size, NULL));
    resources->buffSizes[NCCL_PROTO_SIMPLE] = mapMem->size;
    NCCL_NET_MAP_ADD_POINTER(map, 1, resources->useGdr, mapMem->size, buffs[NCCL_PROTO_SIMPLE]);
  }

  NCCL_NET_MAP_ADD_POINTER(map, 0, 0, sizeof(struct ncclSendMem), sendMem);
  NCCL_NET_MAP_ADD_POINTER(map, 0, 0, sizeof(struct ncclRecvMem), recvMem);

  if (proxyState->allocP2pNetLLBuffers) {
    NCCL_NET_MAP_ADD_POINTER(map, 0, 0 /*resources->useGdr*/, proxyState->buffSizes[NCCL_PROTO_LL], buffs[NCCL_PROTO_LL]);
    resources->buffSizes[NCCL_PROTO_LL] = proxyState->buffSizes[NCCL_PROTO_LL];
  }

  if (map->mems[NCCL_NET_MAP_DEVMEM].size) {
    if (resources->shared == 0) {
      if (ncclCuMemEnable()) {
        NCCLCHECK(ncclP2pAllocateShareableBuffer(map->mems[NCCL_NET_MAP_DEVMEM].size, &map->mems[NCCL_NET_MAP_DEVMEM].ipcDesc,
                                                 (void**)&map->mems[NCCL_NET_MAP_DEVMEM].gpuPtr));
      } else {
        NCCLCHECK(ncclCudaCalloc(&map->mems[NCCL_NET_MAP_DEVMEM].gpuPtr, map->mems[NCCL_NET_MAP_DEVMEM].size));
      }
      map->mems[NCCL_NET_MAP_DEVMEM].cpuPtr = map->mems[NCCL_NET_MAP_DEVMEM].gpuPtr;
    }
  }
  NCCLCHECK(ncclCudaHostCalloc(&map->mems[NCCL_NET_MAP_HOSTMEM].cpuPtr, map->mems[NCCL_NET_MAP_HOSTMEM].size));
  map->mems[NCCL_NET_MAP_HOSTMEM].gpuPtr = map->mems[NCCL_NET_MAP_HOSTMEM].cpuPtr;
  if (ncclGdrCopy && map->sameProcess) {
    uint64_t *cpuPtr, *gpuPtr;
    NCCLCHECK(ncclGdrCudaCalloc(&cpuPtr, &gpuPtr, 2, &resources->gdrDesc));

    if (ncclParamGdrCopySyncEnable()) {
      resources->gdcSync = cpuPtr;
      struct connectMapMem* gdcMem = map->mems+NCCL_NET_MAP_GDCMEM;
      gdcMem->cpuPtr = (char*)cpuPtr;
      gdcMem->gpuPtr = (char*)gpuPtr;
      gdcMem->size = sizeof(uint64_t);
    }
    if (ncclParamGdrCopyFlushEnable()) resources->gdcFlush = cpuPtr + 1;
  }

  resources->sendMem = (struct ncclSendMem*) NCCL_NET_MAP_GET_POINTER(map, cpu, sendMem);
  resources->recvMem = (struct ncclRecvMem*) NCCL_NET_MAP_GET_POINTER(map, cpu, recvMem);
  for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
    resources->buffers[p] = NCCL_NET_MAP_GET_POINTER(map, cpu, buffs[p]);
    if (resources->buffers[p]) {
#if CUDA_VERSION >= 11070
      /* DMA-BUF support */
      int type = NCCL_NET_MAP_DEV_MEM(map, buffs[p]) ? NCCL_PTR_CUDA : NCCL_PTR_HOST;
      if (type == NCCL_PTR_CUDA && resources->useDmaBuf) {
        int dmabuf_fd;
        CUCHECK(cuMemGetHandleForAddressRange((void *)&dmabuf_fd, (CUdeviceptr)resources->buffers[p], resources->buffSizes[p], CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, 0));
        NCCLCHECK(proxyState->ncclNet->regMrDmaBuf(resources->netRecvComm, resources->buffers[p], resources->buffSizes[p], type, 0ULL, dmabuf_fd, &resources->mhandles[p]));
        (void)close(dmabuf_fd);
      } else // FALL-THROUGH to nv_peermem GDR path
#endif
      {
        NCCLCHECK(proxyState->ncclNet->regMr(resources->netRecvComm, resources->buffers[p], resources->buffSizes[p], NCCL_NET_MAP_DEV_MEM(map, buffs[p]) ? NCCL_PTR_CUDA : NCCL_PTR_HOST, &resources->mhandles[p]));
      }

      // Copy the mhandle dptr
      if (resources->netDeviceType != NCCL_NET_DEVICE_HOST && proxyState->ncclNet->getDeviceMr)
        NCCLCHECK(proxyState->ncclNet->getDeviceMr(resources->netRecvComm, resources->mhandles[p], &connection->mhandles[p]));
    }
  }

  //NCCLCHECK(netDumpMap(map));
  if (respSize != sizeof(struct connectMap)) return ncclInternalError;
  memcpy(respBuff, map, sizeof(struct connectMap));
  return ncclSuccess;
}

static ncclResult_t sendProxyFree(struct ncclProxyConnection* connection, struct ncclProxyState* proxyState) {
  struct sendNetResources* resources = (struct sendNetResources*)(connection->transportResources);
  if (connection->state == connSharedInitialized) { // NVB Preconnect
    NCCLCHECK(sharedNetBuffersDestroy(proxyState, connection->tpLocalRank, 0, connection));
    return ncclSuccess;
  }

  if (connection->state == connConnected) {
    for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
      if (resources->buffers[p]) {
        NCCLCHECK(proxyState->ncclNet->deregMr(resources->netSendComm, resources->mhandles[p]));
      }
    }
    struct connectMapMem* mems = resources->map.mems;
    if (resources->map.sameProcess) {
      NCCLCHECK(ncclCudaHostFree(mems[NCCL_NET_MAP_HOSTMEM].cpuPtr));
    } else {
      NCCLCHECK(ncclShmClose(mems[NCCL_NET_MAP_HOSTMEM].createHandle));
    }
    NCCLCHECK(ncclCudaFree(mems[NCCL_NET_MAP_DEVMEM].cpuPtr));
    if (!resources->map.sameProcess || ncclCuMemEnable()) {
      // cuMem API support
      if (mems[NCCL_NET_MAP_DEVMEM].size) {
        NCCLCHECK(ncclP2pFreeShareableBuffer(&mems[NCCL_NET_MAP_DEVMEM].ipcDesc));
      }
    }
    if (mems[NCCL_NET_MAP_GDCMEM].cpuPtr) NCCLCHECK(ncclGdrCudaFree(resources->gdrDesc));
    if (resources->shared) {
      NCCLCHECK(sharedNetBuffersDestroy(proxyState, resources->tpLocalRank, 0, connection));
      if (resources->maxRecvs > 1 && ncclParamNetSharedComms()) {
        struct ncclSharedNetComms* comms = proxyState->progressState.netComms[resources->netDev]+resources->tpRemoteRank;
        comms->sendRefCount[resources->channelId]--;
        if (comms->sendRefCount[resources->channelId] == 0) NCCLCHECK(proxyState->ncclNet->closeSend(comms->sendComm[resources->channelId]));
      } else {
        NCCLCHECK(proxyState->ncclNet->closeSend(resources->netSendComm));
      }
    } else {
      NCCLCHECK(proxyState->ncclNet->closeSend(resources->netSendComm));
    }
  }

  if (resources) free(resources);
  return ncclSuccess;
}

static ncclResult_t recvProxyFree(struct ncclProxyConnection* connection, struct ncclProxyState* proxyState) {
  struct recvNetResources* resources = (struct recvNetResources*)(connection->transportResources);
  if (connection->state == connSharedInitialized) { // NVB Preconnect
    NCCLCHECK(sharedNetBuffersDestroy(proxyState, connection->tpLocalRank, 1, connection));
    return ncclSuccess;
  }

  if (connection->state == connConnected) {
    for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
      if (resources->buffers[p]) {
        NCCLCHECK(proxyState->ncclNet->deregMr(resources->netRecvComm, resources->mhandles[p]));
      }
    }
    struct connectMapMem* mems = resources->map.mems;
    NCCLCHECK(ncclCudaHostFree(mems[NCCL_NET_MAP_HOSTMEM].cpuPtr));
    NCCLCHECK(ncclCudaFree(mems[NCCL_NET_MAP_DEVMEM].cpuPtr));
    if (!resources->map.sameProcess || ncclCuMemEnable()) {
      // cuMem API support
      if (mems[NCCL_NET_MAP_DEVMEM].size) {
        NCCLCHECK(ncclP2pFreeShareableBuffer(&mems[NCCL_NET_MAP_DEVMEM].ipcDesc));
      }
    }
    if (mems[NCCL_NET_MAP_GDCMEM].cpuPtr) NCCLCHECK(ncclGdrCudaFree(resources->gdrDesc));
    if (resources->shared) {
      NCCLCHECK(sharedNetBuffersDestroy(proxyState, resources->tpLocalRank, 1, connection));
      if (resources->maxRecvs > 1 && ncclParamNetSharedComms()) {
        struct ncclSharedNetComms* comms = proxyState->progressState.netComms[resources->netDev] + resources->tpRemoteProxyRank;
        comms->recvRefCount[resources->channelId]--;
        if (comms->recvRefCount[resources->channelId] == 0) NCCLCHECK(proxyState->ncclNet->closeRecv(comms->recvComm[resources->channelId]));
      } else {
        NCCLCHECK(proxyState->ncclNet->closeRecv(resources->netRecvComm));
      }
    } else {
      NCCLCHECK(proxyState->ncclNet->closeRecv(resources->netRecvComm));
    }
  }

  if (resources) free(resources);
  return ncclSuccess;
}

static_assert(NCCL_STEPS <= NCCL_NET_MAX_REQUESTS, "Not enough net requests to cover for steps");
#define MAX_NET_SIZE (1024*1024*1024L) // Rather than send INT_MAX which is 2G-1, send a power of two.

static ncclResult_t sendProxyProgress(struct ncclProxyState* proxyState, struct ncclProxyArgs* args) {
  if (args->state == ncclProxyOpReady) {
    for (int s=0; s<args->nsubs; s++) {
      struct ncclProxySubArgs* sub = args->subs+s;
      struct sendNetResources* resources = (struct sendNetResources*) (sub->connection->transportResources);
      // Round to next multiple of sliceSteps
      sub->base = ROUNDUP(resources->step, args->chunkSteps);
      // Set step base for next op
      resources->step = sub->base + sub->nsteps;
      sub->posted = sub->transmitted = sub->done = 0;
      for (uint64_t step=0; step<sub->nsteps; step++) ncclProfilingRecord(args, s, step, ncclProxyProfileBegin);
      if (sub->reg && sub->nbytes > 0) {
        NCCLCHECK(proxyState->ncclNet->regMr(resources->netSendComm, sub->buffer, sub->nbytes, NCCL_PTR_CUDA, &sub->mhandle));
      } else {
        sub->mhandle = resources->mhandles[args->protocol];
      }
    }
    args->state = ncclProxyOpProgress;
  }
  args->idle = 1;
  if (args->state == ncclProxyOpProgress) {
    int p = args->protocol;
    int maxDepth = std::min(NCCL_STEPS, NCCL_SHARED_STEPS/args->nsubs);
    for (int s=0; s<args->nsubs; s++) {
      struct ncclProxySubArgs* sub = args->subs+s;
      if (sub->done == sub->nsteps) continue;
      struct sendNetResources* resources = (struct sendNetResources*) (sub->connection->transportResources);
      volatile struct ncclConnFifo* connFifo = (volatile struct ncclConnFifo*)resources->recvMem->connFifo;
      int stepSize = resources->buffSizes[p] / NCCL_STEPS;
      char* localBuff = NCCL_NET_MAP_GET_POINTER(&resources->map, cpu, buffs[p]);
      // Post buffers to the GPU
      if (sub->posted < sub->nsteps && sub->posted < sub->done + maxDepth) {
        int buffSlot = (sub->base+sub->posted)%NCCL_STEPS;
        if (resources->shared) {
          if (!sub->reg) {
            int sharedBuffSlot = sub->posted%maxDepth;
            int offset;
            NCCLCHECK(sharedBuffersGet(proxyState, sub->channelId, sharedBuffSlot*args->nsubs+s, &offset, NULL));
            resources->recvMem->connFifo[buffSlot].offset = offset;
            __sync_synchronize();
          }
          volatile uint64_t* sendHead = resources->gdcSync ? resources->gdcSync : &resources->sendMem->head;
          sub->posted += args->sliceSteps;
          // Only post one credit for registered buffer
          if (sub->reg == 0 || sub->posted == args->sliceSteps) *sendHead = sub->base + sub->posted - NCCL_STEPS;
          if (resources->gdcSync) wc_store_fence(); // Flush out WC write
        } else sub->posted += args->sliceSteps;
        for (uint64_t step=sub->posted-args->sliceSteps; step<sub->posted; step++) {
          ncclProfilingRecord(args, s, step, ncclProxyProfileSendGPUWait);
        }
        args->idle = 0;
        continue;
      }
      // Check whether we received data from the GPU and send it to the network
      if (sub->transmitted < sub->posted && sub->transmitted < sub->done + NCCL_STEPS) {
        int buffSlot = (sub->base+sub->transmitted)%NCCL_STEPS;
        volatile uint64_t* recvTail = &resources->recvMem->tail;
        uint64_t tail = sub->base + (sub->reg ? 0 : sub->transmitted);
        if ((sub->reg || connFifo[buffSlot].size != -1) && ((*recvTail > tail) || p == NCCL_PROTO_LL)) {
          // We have something to receive, let's check if it's completely ready.
          int size = sub->reg ? std::min(MAX_NET_SIZE, sub->nbytes) : connFifo[buffSlot].size;
          bool shared = (p == NCCL_PROTO_SIMPLE) && resources->shared;
          char* buff = shared ? localBuff+connFifo[buffSlot].offset : localBuff+buffSlot*stepSize;
          int ready = 1;
          if (p == NCCL_PROTO_LL128) {
            ready = resources->useGdr;
            if (!ready) {
              // When data is in sysmem, we need to wait until all flags are correct since the GPU only
              // called threadfence()
              uint64_t flag = sub->base+sub->transmitted+1;
              int nFifoLines = DIVUP(connFifo[buffSlot].size, sizeof(uint64_t)*NCCL_LL128_LINEELEMS);
              volatile uint64_t* lines = (volatile uint64_t*)buff;
              ready = 1;
              for (int i=0; i<nFifoLines; i++) {
                if (lines[i*NCCL_LL128_LINEELEMS+NCCL_LL128_DATAELEMS] != flag) { ready = 0; break; }
              }
            }
          } else if (p == NCCL_PROTO_LL) {
            uint32_t flag = NCCL_LL_FLAG(sub->base+sub->transmitted+1);
            int nFifoLines = DIVUP(size, sizeof(union ncclLLFifoLine));
            union ncclLLFifoLine* lines = (union ncclLLFifoLine*)buff;
            for (int i=0; i<nFifoLines; i++) {
              volatile uint32_t *f1 = &lines[i].flag1;
              volatile uint32_t *f2 = &lines[i].flag2;
              if (f1[0] != flag || f2[0] != flag) { ready = 0; break; }
            }
          } else if (p == NCCL_PROTO_SIMPLE && resources->shared) {
            buff = sub->reg ? (char*)sub->buffer : localBuff+resources->recvMem->connFifo[buffSlot].offset;
          }
          if (ready) {
            // Data is ready, try to send.
            NCCLCHECK(proxyState->ncclNet->isend(resources->netSendComm, buff, size, resources->tpRank, sub->mhandle, sub->requests+buffSlot));
            if (sub->requests[buffSlot] != NULL) {
              TRACE(NCCL_NET, "sendProxy [%ld/%d] Isend posted, req %p, size %d, proto %d, myRank %d, channelId %d", sub->transmitted, buffSlot, sub->requests[buffSlot], size, p, proxyState->tpRank, sub->channelId);
              sub->transmitted += args->sliceSteps;
              for (uint64_t step=sub->transmitted-args->sliceSteps; step<sub->transmitted; step++) ncclProfilingRecord(args, s, step, ncclProxyProfileSendWait);
              args->idle = 0;
              continue;
            }
          }
        }
      }
      // Check whether the network has completed some send operations.
      if (sub->done < sub->transmitted) {
        int done;
        int size;
        int buffSlot = (sub->base+sub->done)%NCCL_STEPS;
        NCCLCHECK(proxyState->ncclNet->test(sub->requests[buffSlot], &done, &size));
        if (done) {
          if (sub->reg) {
            if (size < sub->nbytes) {
              sub->buffer = ((char*)sub->buffer)+size;
              sub->nbytes -= size;
              // Do one more step (at least)
              sub->nsteps++;
            } else {
              // Signal the GPU the send is complete and it can return.
              connFifo[sub->base%NCCL_STEPS].size = -1;
            }
          }
          // Make sure size is reset to -1 before we update the head.
          if (sub->reg == 0) connFifo[buffSlot].size = -1;
          __sync_synchronize();
          TRACE(NCCL_NET, "sendProxy [%ld/%d] request %p done", sub->done, buffSlot, sub->requests[buffSlot]);
          sub->done += args->sliceSteps;
          for (uint64_t step=sub->done-args->sliceSteps; step<sub->done; step++) ncclProfilingRecord(args, s, step, ncclProxyProfileEnd);

          if (resources->shared == 0) {
            volatile uint64_t* sendHead = resources->gdcSync ? resources->gdcSync : &resources->sendMem->head;
            if (sub->reg) {
              // We may have added more net steps, but reg operations only have a single step w.r.t. the GPU.
              if (sub->done == sub->nsteps) *sendHead = sub->base + args->sliceSteps;
            } else {
              *sendHead = sub->base + sub->done;
            }
            if (resources->gdcSync) wc_store_fence(); // Flush out WC write
          }
          args->idle = 0;
          if (sub->done == sub->nsteps) {
            if (sub->reg && sub->nbytes > 0) {
              NCCLCHECK(proxyState->ncclNet->deregMr(resources->netSendComm, sub->mhandle));
            }
            args->done++;
          }
        }
      }
    }
    if (args->done == args->nsubs) {
      args->state = ncclProxyOpNone;
    }
  }
  return ncclSuccess;
}

static ncclResult_t recvProxyProgress(struct ncclProxyState* proxyState, struct ncclProxyArgs* args) {
  if (args->state == ncclProxyOpReady) {
    // Initialize subs and group them by same recvComm.
    void* recvComm;
    int groupSize = 0;
    int maxRecvs = 1;
    for (int s=0; s<args->nsubs; s++) {
      struct ncclProxySubArgs* sub = args->subs+s;
      if (groupSize == maxRecvs) {
        groupSize = 0;
      } else if (s>0) { // Find next sub with the same recvComm
        int next;
        for (next=s; next<args->nsubs; next++) {
          struct recvNetResources* nextRes = (struct recvNetResources*) (args->subs[next].connection->transportResources);
          if (nextRes->netRecvComm == recvComm) break;
        }
        if (next == args->nsubs) { // Not found
          groupSize = 0;
        } else if (s != next) { // We found a sub later with the same recvComm ; swap subs
          struct ncclProxySubArgs temp;
          memcpy(&temp, sub, sizeof(struct ncclProxySubArgs));
          memcpy(sub, args->subs+next, sizeof(struct ncclProxySubArgs));
          memcpy(args->subs+next, &temp, sizeof(struct ncclProxySubArgs));
        }
      }
      groupSize++;
      struct recvNetResources* resources = (struct recvNetResources*) (sub->connection->transportResources);
      maxRecvs = resources->maxRecvs;
      recvComm = resources->netRecvComm;
      // Round to next multiple of sliceSteps
      sub->base = ROUNDUP(resources->step, args->chunkSteps);
      // Set step base for next op
      resources->step = sub->base + sub->nsteps;
      sub->posted = sub->received = sub->transmitted = sub->done = 0;
      for (int i=0; i<groupSize; i++) sub[-i].groupSize = groupSize;
      for (uint64_t step=0; step<sub->nsteps; step++) ncclProfilingRecord(args, s, step, ncclProxyProfileBegin);
      if (sub->reg && sub->nbytes > 0) {
        // Register buffer
        NCCLCHECK(proxyState->ncclNet->regMr(resources->netRecvComm, sub->buffer, sub->nbytes, NCCL_PTR_CUDA, &sub->mhandle));
      } else {
        sub->mhandle = resources->mhandles[args->protocol];
      }
    }
    args->state = ncclProxyOpProgress;
  }
  args->idle = 1;
  if (args->state == ncclProxyOpProgress) {
    int p = args->protocol;
    int maxDepth = std::min(NCCL_STEPS, NCCL_SHARED_STEPS/args->nsubs);
    for (int s=0; s<args->nsubs; s+=args->subs[s].groupSize) {
      struct ncclProxySubArgs* subGroup = args->subs+s;
      int subCount = 0;
      void* ptrs[NCCL_PROXY_MAX_SUBS];
      int sizes[NCCL_PROXY_MAX_SUBS];
      int tags[NCCL_PROXY_MAX_SUBS];
      void* mhandles[NCCL_PROXY_MAX_SUBS];
      for (int i=0; i<subGroup->groupSize; i++) {
        struct ncclProxySubArgs* sub = subGroup + i;
        if (sub->posted < sub->nsteps) {
          if (sub->posted >= sub->done + maxDepth) { subCount = 0; break; }
          struct recvNetResources* resources = (struct recvNetResources*) (sub->connection->transportResources);
          if (sub->reg) maxDepth = 1;
          int stepSize = resources->buffSizes[p] / NCCL_STEPS;
          char* localBuff = NCCL_NET_MAP_GET_POINTER(&resources->map, cpu, buffs[p]);
          int buffSlot = (sub->base+sub->posted)%NCCL_STEPS;
          volatile struct ncclConnFifo* connFifo = (volatile struct ncclConnFifo*)resources->recvMem->connFifo;
          if (p == NCCL_PROTO_SIMPLE && resources->shared) {
            if (sub->reg) {
              // Wait until CUDA kernel has started before we access the user buffer directly.
              if (connFifo[sub->base%NCCL_STEPS].size == -1) continue;
              ptrs[subCount] = sub->buffer;
              sizes[subCount] = std::min(MAX_NET_SIZE, sub->nbytes);
            } else {
              int sharedBuffSlot = sub->posted%maxDepth;
              int offset;
              NCCLCHECK(sharedBuffersGet(proxyState, sub->channelId, sharedBuffSlot*args->nsubs+s+i, &offset, sizes+subCount));
              connFifo[buffSlot].offset = offset;
              ptrs[subCount] = localBuff+offset;
            }
          } else {
            ptrs[subCount] = localBuff+buffSlot*stepSize;
            sizes[subCount] = stepSize*args->sliceSteps;
          }
          sizes[subCount] = stepSize*args->sliceSteps;
          if (sub->nbytes < sizes[subCount]) sizes[subCount] = sub->nbytes;
          tags[subCount] = resources->tpRemoteRank;
          mhandles[subCount] = sub->mhandle;
          subCount++;
        }
      }
      if (subCount) {
        uint64_t step = subGroup->posted;
        struct recvNetResources* resources = (struct recvNetResources*) (subGroup->connection->transportResources);
        void** requestPtr = subGroup->requests+(step%NCCL_STEPS);
        NCCLCHECK(proxyState->ncclNet->irecv(resources->netRecvComm, subCount, ptrs, sizes, tags, mhandles, requestPtr));
        if (*requestPtr) {
          subGroup->recvRequestsCache[step%NCCL_STEPS] = *requestPtr;
          subGroup->recvRequestsSubCount = subCount;
          for (int i=0; i<subGroup->groupSize; i++) {
            struct ncclProxySubArgs* sub = subGroup+i;
            sub->posted += args->sliceSteps;
            for (uint64_t step=sub->posted-args->sliceSteps; step<sub->posted; step++) ncclProfilingRecord(args, s+i, step, ncclProxyProfileRecvWait);
          }
          args->idle = 0;
        }
      }
    }
    if (args->idle == 0) return ncclSuccess;

    for (int s=0; s<args->nsubs; s+=args->subs[s].groupSize) {
      struct ncclProxySubArgs* subGroup = args->subs+s;
      if (subGroup->posted > subGroup->received) {
        uint64_t step = subGroup->received;
        int done;
        void* ptrs[NCCL_PROXY_MAX_SUBS];
        int sizes[NCCL_PROXY_MAX_SUBS];
        void* mhandles[NCCL_PROXY_MAX_SUBS];
        for (int i=0; i<NCCL_PROXY_MAX_SUBS; i++) sizes[i] = 0;
        NCCLCHECK(proxyState->ncclNet->test(subGroup->requests[step%NCCL_STEPS], &done, sizes));
        if (done) {
          int needFlush = 0;
          int totalSize = 0;
          int subIndex = 0;
          for (int i=0; i<NCCL_PROXY_MAX_SUBS; i++) totalSize += sizes[i];
          for (int i=0; i<subGroup->groupSize; i++) {
            struct ncclProxySubArgs* sub = subGroup + i;
            if (sub->received < sub->nsteps) {
              int size = sizes[subIndex++];
              if (sub->reg) {
                if (size < sub->nbytes) {
                  sub->buffer = ((char*)sub->buffer) + size;
                  sub->nbytes -= size;
                  // Do one more step (at least)
                  sub->nsteps++;
                } else {
                  // Reset connFifo size indicating the GPU was ready to receive.
                  // There is a __sync_synchronize() later to ensure it is reset before it is set again by the GPU.
                  struct recvNetResources* resources = (struct recvNetResources*) (sub->connection->transportResources);
                  volatile struct ncclConnFifo* connFifo = (volatile struct ncclConnFifo*)resources->recvMem->connFifo;
                  connFifo[sub->base%NCCL_STEPS].size = -1;
                }
              }
            }
            sub->received += args->sliceSteps;
            for (uint64_t step=sub->received-args->sliceSteps; step<sub->received; step++) ncclProfilingRecord(args, s+i, step, ncclProxyProfileRecvFlushWait);
            if (step < sub->nsteps) {
              struct recvNetResources* resources = (struct recvNetResources*) (sub->connection->transportResources);
              if (resources->useGdr) needFlush |= resources->needFlush;
            }
          }
          subGroup->requests[step%NCCL_STEPS] = NULL;
          if (totalSize > 0 && p == NCCL_PROTO_SIMPLE && needFlush) {
            // GDRCOPY support
            struct recvNetResources* resources = (struct recvNetResources*) (subGroup->connection->transportResources);
            if (resources->gdcFlush) {
#if defined (__x86_64__)
              // Force a PCI-E read from GPU memory
              asm volatile ("mov (%0), %%eax" :: "l"(resources->gdcFlush) : "%eax");
#else
              WARN("NET: GDR Flush only supported on x86_64");
              return ncclInternalError;
#endif
            } else {
              int subCount = 0;
              for (int i=0; i<subGroup->groupSize; i++) {
                struct ncclProxySubArgs* sub = subGroup + i;
                if (step < sub->nsteps) {
                  struct recvNetResources* resources = (struct recvNetResources*) (sub->connection->transportResources);
                  int stepSize = resources->buffSizes[p] / NCCL_STEPS;
                  char* localBuff = NCCL_NET_MAP_GET_POINTER(&resources->map, cpu, buffs[p]);
                  int buffSlot = (sub->base+sub->received-args->sliceSteps)%NCCL_STEPS;
                  ptrs[subCount] = resources->shared ?
                    (sub->reg ? sub->buffer : localBuff+resources->recvMem->connFifo[buffSlot].offset) :
                    localBuff+buffSlot*stepSize;
                  mhandles[subCount] = sub->mhandle;
                  subCount++;
                }
              }
              struct recvNetResources* resources = (struct recvNetResources*) (subGroup->connection->transportResources);
              NCCLCHECK(proxyState->ncclNet->iflush(resources->netRecvComm, subCount, ptrs, sizes, mhandles, subGroup->requests+(step%NCCL_STEPS)));
            }
          }
          args->idle = 0;
        }
      }
    }
    if (args->idle == 0) return ncclSuccess;

    for (int s=0; s<args->nsubs; s+=args->subs[s].groupSize) {
      struct ncclProxySubArgs* subGroup = args->subs+s;
      if (subGroup->received > subGroup->transmitted) {
        uint64_t step = subGroup->transmitted;
        int done = 1;
        void* request = subGroup->requests[step%NCCL_STEPS];
        if (request) NCCLCHECK(proxyState->ncclNet->test(request, &done, NULL));
        if (done) {
          for (int i=0; i<subGroup->groupSize; i++) {
            struct ncclProxySubArgs* sub = subGroup + i;

            sub->transmitted += args->sliceSteps;
            for (uint64_t step=sub->transmitted-args->sliceSteps; step<sub->transmitted; step++) ncclProfilingRecord(args, s+i, step, ncclProxyProfileRecvGPUWait);
            if (step < sub->nsteps) {
              __sync_synchronize();
              struct recvNetResources* resources = (struct recvNetResources*) (sub->connection->transportResources);
              volatile uint64_t* recvTail = resources->gdcSync ? resources->gdcSync : &resources->recvMem->tail;
              if (sub->reg) {
                // We may have added more net steps, but reg operations only have a single step w.r.t. the GPU.
                if (sub->transmitted == sub->nsteps) *recvTail = sub->base + args->sliceSteps;
              } else
                *recvTail = sub->base + sub->transmitted;
              if (resources->gdcSync) wc_store_fence(); // Flush out WC write
            }
          }
          args->idle = 0;
        }
      }
    }
    if (args->idle == 0) return ncclSuccess;

    for (int s=0; s<args->nsubs; s+=args->subs[s].groupSize) {
      struct ncclProxySubArgs* subGroup = args->subs+s;
      for (int i=0; i<subGroup->groupSize; i++) {
        struct ncclProxySubArgs* sub = subGroup + i;
        if (sub->done == sub->nsteps) continue;
        if (sub->transmitted > sub->done) {
          struct recvNetResources* resources = (struct recvNetResources*) (sub->connection->transportResources);
          volatile uint64_t* sendHead = &resources->sendMem->head;
          uint64_t done = sub->reg ? sub->base + sub->nsteps : *sendHead;
          while (done > sub->base + sub->done &&
              // LL and LL128 can acknowledge 0-bytes send before they even happen. Don't go past what we transmitted.
              sub->transmitted > sub->done) {
            if (subGroup->recvRequestsCache[sub->done%NCCL_STEPS]) {
              // the multirecv requests are only cached in the first sub.
              if (proxyState->ncclNet->irecvConsumed)
                NCCLCHECK(proxyState->ncclNet->irecvConsumed(resources->netRecvComm, subGroup->recvRequestsSubCount, subGroup->recvRequestsCache[sub->done%NCCL_STEPS]));
              subGroup->recvRequestsCache[sub->done%NCCL_STEPS] = NULL;
            }
            sub->done += args->sliceSteps;
            for (uint64_t step=sub->done-args->sliceSteps; step<sub->done; step++) ncclProfilingRecord(args, s+i, step, ncclProxyProfileEnd);
            args->idle = 0;
            if (sub->done == sub->nsteps) {
              struct recvNetResources* resources = (struct recvNetResources*) (sub->connection->transportResources);
              if (sub->reg && sub->nbytes > 0) {
                NCCLCHECK(proxyState->ncclNet->deregMr(resources->netRecvComm, sub->mhandle));
              }
              args->done++;
              break;
            }
          }
        }
      }
    }
    if (args->done == args->nsubs) {
      args->state = ncclProxyOpNone;
    }
  }
  return ncclSuccess;
}

struct ncclTransport netTransport = {
  "NET",
  canConnect,
  { sendSetup, sendConnect, sendFree, proxySharedInit, sendProxySetup, sendProxyConnect, sendProxyFree, sendProxyProgress },
  { recvSetup, recvConnect, recvFree, proxySharedInit, recvProxySetup, recvProxyConnect, recvProxyFree, recvProxyProgress }
};
