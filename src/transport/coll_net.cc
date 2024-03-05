/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "comm.h"
#include "coll_net.h"
#include "graph.h"
#include "proxy.h"
#include "gdrwrap.h"

int64_t ncclParamGdrCopySyncEnable();
int64_t ncclParamGdrCopyFlushEnable();

struct collNetRecvConnectInfo {
  int rank;
  int nranks;
  collNetHandle_t collNetHandle;
};

struct collNetSendConnectInfo {
  void* mhandles[NCCL_NUM_PROTOCOLS];
  void* reqFifo;
};

#define COLLNET_GROUP_NSUBS 8
#define COLLNET_MAX_GROUPS (NCCL_PROXY_MAX_SUBS/COLLNET_GROUP_NSUBS)

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
};

struct connectMap {
  int shared;
  // First 3 bits of offsets determine the mem bank. 001 is host mem, 011 is dev mem, 101 is shared host mem and 111 is shared dev mem.
  struct connectMapMem mems[NCCL_NET_MAP_MEMS];
  // Offsets. 3 MSBs indicate mem bank, 111 indicates NULL.
  struct {
    uint32_t sendMem;
    uint32_t recvMem;
    uint32_t buffs[NCCL_NUM_PROTOCOLS];
  } offsets;
};

struct reqSlot {
  bool turnIsSendNotRecv;
  int size;
};

struct sendResources {
  struct connectMap map;
  void* collNetComm;
  struct ncclSendMem* sendMem;
  struct ncclRecvMem* recvMem;

  int rank;
  int nranks;
  int netDev;
  int useGdr;
  int useDmaBuf;
  uint64_t* gdcSync;
  void* gdrDesc;
  void* sendMhandles[NCCL_NUM_PROTOCOLS];
  void* recvMhandles[NCCL_NUM_PROTOCOLS];
  uint64_t step;
  struct reqSlot (*reqFifo)[NCCL_STEPS];
  int collNetRank;
};

struct recvResources {
  struct connectMap map;
  void* collNetComm;
  struct ncclSendMem* sendMem;
  struct ncclRecvMem* recvMem;

  int rank;
  int nranks;
  int netDev;
  int useGdr;
  int useDmaBuf;
  int needFlush;
  uint64_t* gdcSync;
  uint64_t* gdcFlush;
  void* gdrDesc;
  void* mhandles[NCCL_NUM_PROTOCOLS];
  uint64_t step;
  struct reqSlot reqFifo[COLLNET_MAX_GROUPS][NCCL_STEPS];
  int collNetRank;
};

static ncclResult_t canConnect(int* ret, struct ncclTopoSystem* topo, struct ncclTopoGraph* graph, struct ncclPeerInfo* info1, struct ncclPeerInfo* info2) {
  // This transport cannot be used for p2p
  *ret = 0;
  return ncclSuccess;
}

struct setupReq {
  int netDev;
  int useGdr;
  int needFlush;
  struct ncclCollNetSharedRes* collNet;
};


/* Setup send connector, and return connect information for others in the coll
 * communicator to connect to me */
static ncclResult_t sendSetup(struct ncclComm* comm, struct ncclTopoGraph* graph, struct ncclPeerInfo* myInfo, struct ncclPeerInfo* peerInfo, struct ncclConnect* connectInfo, struct ncclConnector* send, int channelId, int connIndex) {
  struct setupReq req = { 0 };

  int proxyRank, tpProxyRank;
  NCCLCHECK(ncclTopoGetNetDev(comm, myInfo->rank, graph, channelId, -1, &req.netDev, &proxyRank));
  NCCLCHECK(ncclTopoCheckGdr(comm->topo, myInfo->busId, req.netDev, 1, &req.useGdr));
  send->conn.flags |= req.useGdr ? NCCL_DIRECT_NIC : 0;

  send->proxyConn.tpLocalRank = comm->topParentLocalRanks[comm->localRank];
  tpProxyRank = comm->topParentRanks[myInfo->rank];
  NCCLCHECK(ncclProxyConnect(comm, TRANSPORT_COLLNET, 1, tpProxyRank, &send->proxyConn));
  ncclAtomicRefCountIncrement(&comm->collNetSharedRes->refCount);
  req.collNet = comm->collNetSharedRes;
  NCCLCHECK(ncclProxyCallBlocking(comm, &send->proxyConn, ncclProxyMsgSetup, &req, sizeof(req), NULL, 0));

  INFO(NCCL_INIT|NCCL_NET,"CollNet %02d/%1d : %d [send] via COLLNET/%s/%d%s", channelId, connIndex, myInfo->rank, collNetName(comm), req.netDev,
      req.useGdr ? "/GDRDMA" : "");
  return ncclSuccess;
}

static ncclResult_t recvSetup(struct ncclComm* comm, struct ncclTopoGraph* graph, struct ncclPeerInfo* myInfo, struct ncclPeerInfo* peerInfo, struct ncclConnect* connectInfo, struct ncclConnector* recv, int channelId, int connIndex) {
  struct setupReq req = { 0 };

  int proxyRank, tpProxyRank;
  NCCLCHECK(ncclTopoGetNetDev(comm, myInfo->rank, graph, channelId, -1, &req.netDev, &proxyRank));
  NCCLCHECK(ncclTopoCheckGdr(comm->topo, myInfo->busId, req.netDev, 0, &req.useGdr));
  recv->conn.flags |= req.useGdr ? NCCL_DIRECT_NIC : 0;
  // Determine whether we need to flush the GDR buffer on recv or not
  if (req.useGdr) NCCLCHECK(ncclTopoNeedFlush(comm->topo, myInfo->busId, &req.needFlush));

  recv->proxyConn.tpLocalRank = comm->topParentLocalRanks[comm->localRank];
  tpProxyRank = comm->topParentRanks[myInfo->rank];
  NCCLCHECK(ncclProxyConnect(comm, TRANSPORT_COLLNET, 0, tpProxyRank, &recv->proxyConn));
  struct collNetRecvConnectInfo* info = (struct collNetRecvConnectInfo*) connectInfo;
  ncclAtomicRefCountIncrement(&comm->collNetSharedRes->refCount);
  req.collNet = comm->collNetSharedRes;
  NCCLCHECK(ncclProxyCallBlocking(comm, &recv->proxyConn, ncclProxyMsgSetup, &req, sizeof(req), &info->collNetHandle, sizeof(collNetHandle_t)));

  INFO(NCCL_INIT|NCCL_NET,"CollNet %02d/%1d : %d [receive] via COLLNET/%s/%d%s", channelId, connIndex, myInfo->rank, collNetName(comm), req.netDev,
      req.useGdr ? "/GDRDMA" : "");
  return ncclSuccess;
}

static ncclResult_t collNetDumpMap(struct connectMap* map) {
  printf("Dump map\n");
  struct connectMapMem *mem = map->mems+NCCL_NET_MAP_HOSTMEM;
  printf("Mem 0: Host mem (%x B) CPU %p GPU %p\n", mem->size, mem->cpuPtr, mem->gpuPtr);
  mem = map->mems+NCCL_NET_MAP_DEVMEM;
  printf("Mem 1: Vid  mem CPU (%x B) %p GPU %p\n", mem->size, mem->cpuPtr, mem->gpuPtr);
  mem = map->mems+NCCL_NET_MAP_SHARED_HOSTMEM;
  printf("Mem 2: Shared Host mem (%x B) CPU %p GPU %p\n", mem->size, mem->cpuPtr, mem->gpuPtr);
  mem = map->mems+NCCL_NET_MAP_SHARED_DEVMEM;
  printf("Mem 3: Shared Vid  (%x B) mem CPU %p GPU %p\n", mem->size, mem->cpuPtr, mem->gpuPtr);
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

struct collNetConnectArgs {
  int rank;
  int nranks;
  struct ncclConnect* connectInfos;
};

static ncclResult_t sendProxyProgress(struct ncclProxyState* proxyState, struct ncclProxyArgs* args);

static ncclResult_t sendConnect(struct ncclComm* comm, struct ncclConnect* connectInfos, int nranks, int rank, struct ncclConnector* send) {
  // We're on the same process as the proxy. We can pass a pointer to a struct.
  struct collNetConnectArgs args = { rank, nranks, connectInfos };
  struct connectMap* map;
  NCCLCHECK(ncclProxyCallBlocking(comm, &send->proxyConn, ncclProxyMsgConnect, &args, sizeof(struct collNetConnectArgs), &map, sizeof(struct connectMap*)));

  // If collnet connect failed, propagate error to fallback on regular p2p
  if (map == NULL) return ncclSystemError;

  //NCCLCHECK(collNetDumpMap(map));

  struct ncclSendMem *sendMem = (struct ncclSendMem*) NCCL_NET_MAP_GET_POINTER(map, gpu, sendMem);
  void* gdcMem = map->mems[NCCL_NET_MAP_GDCMEM].gpuPtr;
  send->conn.head = gdcMem ? (uint64_t*)gdcMem : &sendMem->head;

  struct ncclRecvMem *recvMem = (struct ncclRecvMem*) NCCL_NET_MAP_GET_POINTER(map, gpu, recvMem);
  send->conn.tail = &recvMem->tail;
  send->conn.connFifo = recvMem->connFifo;
  for (int i=0; i<NCCL_STEPS; i++) {
    send->conn.connFifo[i].size = -1;
    send->conn.connFifo[i].mode = NCCL_MODE_OFFSET;
  }

  for (int p=0; p<NCCL_NUM_PROTOCOLS; p++)
    send->conn.buffs[p] = NCCL_NET_MAP_GET_POINTER(map, gpu, buffs[p]);

  send->proxyConn.proxyProgress = sendProxyProgress;

  return ncclSuccess;
}

static ncclResult_t recvProxyProgress(struct ncclProxyState* proxyState, struct ncclProxyArgs* args);

static ncclResult_t recvConnect(struct ncclComm* comm, struct ncclConnect* connectInfos, int nranks, int rank, struct ncclConnector* recv) {
  // We're on the same process as the proxy. We can pass a pointer to a struct.
  struct collNetConnectArgs args = { rank, nranks, connectInfos };
  struct connectMap* map;
  NCCLCHECK(ncclProxyCallBlocking(comm, &recv->proxyConn, ncclProxyMsgConnect, &args, sizeof(struct collNetConnectArgs), &map, sizeof(struct connectMap*)));

  // If collnet connect failed, propagate error to fallback on regular p2p
  if (map == NULL) return ncclSystemError;

  //NCCLCHECK(collNetDumpMap(map));

  struct ncclSendMem *sendMem = (struct ncclSendMem*) NCCL_NET_MAP_GET_POINTER(map, gpu, sendMem);
  recv->conn.head = &sendMem->head;

  struct ncclRecvMem *recvMem = (struct ncclRecvMem*) NCCL_NET_MAP_GET_POINTER(map, gpu, recvMem);
  void* gdcMem = map->mems[NCCL_NET_MAP_GDCMEM].gpuPtr;
  recv->conn.tail = gdcMem ? (uint64_t*)gdcMem : &recvMem->tail;
  recv->conn.connFifo = recvMem->connFifo;
  for (int i=0; i<NCCL_STEPS; i++) {
    recv->conn.connFifo[i].mode = NCCL_MODE_OFFSET;
  }

  for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
    recv->conn.buffs[p] = NCCL_NET_MAP_GET_POINTER(map, gpu, buffs[p]);
  }

  recv->proxyConn.proxyProgress = recvProxyProgress;

  return ncclSuccess;
}

static ncclResult_t sendFree(struct ncclConnector* send) {
  return ncclSuccess;
}

static ncclResult_t recvFree(struct ncclConnector* recv) {
  return ncclSuccess;
}

static ncclResult_t sendProxySetup(struct ncclProxyConnection* connection, struct ncclProxyState* proxyState, void* reqBuff, int reqSize, void* respBuff, int respSize, int* done) {
  struct setupReq* req = (struct setupReq*)reqBuff;
  if (reqSize != sizeof(struct setupReq)) return ncclInternalError;

  struct sendResources* resources;
  NCCLCHECK(ncclCalloc(&resources, 1));
  connection->transportResources = resources;
  connection->shared = 1;

  resources->netDev = req->netDev;
  resources->useGdr = req->useGdr;
  ncclNetProperties_t props;
  NCCLCHECK(proxyState->ncclCollNet->getProperties(req->netDev, &props));
  connection->collNet = req->collNet;
  /* DMA-BUF support */
  resources->useDmaBuf = resources->useGdr && proxyState->dmaBufSupport && (props.ptrSupport & NCCL_PTR_DMABUF);
  return ncclSuccess;
}

struct sharedResources {
  void* collNetListenComms[MAXCHANNELS];
  void* collNetComms[MAXCHANNELS];
  int commRefCount[NCCL_MAX_NETDEVS];
};

static ncclResult_t sharedListen(struct ncclProxyState* proxyState, int netDev, struct ncclCollNetSharedRes* collNet, void* collNetHandle) {
  struct sharedResources* resources = (struct sharedResources*)collNet->resources;
  if (resources == NULL) {
    NCCLCHECK(ncclCalloc(&resources, 1));
    collNet->resources = resources;
  }
  if (resources->collNetComms[netDev] == NULL)
    NCCLCHECK(proxyState->ncclCollNet->listen(netDev, collNetHandle, resources->collNetListenComms + netDev));
  return ncclSuccess;
}

static ncclResult_t sharedConnect(struct ncclProxyState* proxyState, int netDev, struct ncclConnect* connectInfos, int nranks, int rank, struct ncclCollNetSharedRes* collNet, void** collNetComm) {
  struct sharedResources* resources = (struct sharedResources*)collNet->resources;
  if (resources->collNetComms[netDev] == NULL) {
    // Connect to coll comm
    collNetHandle_t** handlePtrs = NULL;
    NCCLCHECK(ncclCalloc(&handlePtrs, nranks));
    for (int i = 0; i < nranks; i++) {
      struct collNetRecvConnectInfo* info = (struct collNetRecvConnectInfo*)(connectInfos+i);
      handlePtrs[i] = &(info->collNetHandle);
    }
    ncclResult_t ret = proxyState->ncclCollNet->connect((void**)handlePtrs, nranks, rank,
          resources->collNetListenComms[netDev],
          resources->collNetComms+netDev);
    free(handlePtrs);
    if (ret == ncclSuccess) {
      // Close listen comm
      NCCLCHECK(proxyState->ncclCollNet->closeListen(resources->collNetListenComms[netDev]));
    } else {
      resources->collNetListenComms[netDev] = NULL;
    }
  }
  *collNetComm = resources->collNetComms[netDev];
  if (*collNetComm) resources->commRefCount[netDev]++;
  return ncclSuccess;
}

static ncclResult_t sharedFree(struct ncclProxyState* proxyState, struct ncclCollNetSharedRes* collNet, int netDev) {
  struct sharedResources* resources = (struct sharedResources*)collNet->resources;
  resources->commRefCount[netDev]--;
  if (resources->commRefCount[netDev] == 0) {
    NCCLCHECK(proxyState->ncclCollNet->closeColl(resources->collNetComms[netDev]));
  }
  for (int n=0; n<NCCL_MAX_NETDEVS; n++) if (resources->commRefCount[n]) return ncclSuccess;
  collNet->resources = NULL;
  free(resources);
  return ncclSuccess;
}

static ncclResult_t sharedBuffersInit(struct ncclCollNetSharedRes* collNet, int cuda, char** gpuPtr, char** cpuPtr, int* size) {
  if (collNet->size == 0) {
    collNet->size = 2 * collNet->nChannels * collNet->buffSize;
  }

  *size = collNet->size;

  if (cuda && collNet->cudaBuff == NULL) {
    NCCLCHECK(ncclCudaCalloc(&collNet->cudaBuff, *size));
    cudaMemset(collNet->cudaBuff, 0x33, *size/2);
    cudaMemset((char*)collNet->cudaBuff + *size/2, 0x66, *size/2);
  }
  if (!cuda && collNet->hostBuff == NULL) {
    NCCLCHECK(ncclCudaHostCalloc(&collNet->hostBuff, *size));
  }
  *gpuPtr = *cpuPtr = cuda ? collNet->cudaBuff : collNet->hostBuff;
  return ncclSuccess;
}

static ncclResult_t sharedBuffersGet(struct ncclCollNetSharedRes* collNet, int type, int slot, int channel, int* offset) {
  // Use different pools for different channels and also separate send/recv.
  int slotSize = collNet->buffSize / NCCL_STEPS;
  int globalSlot = (type * NCCL_STEPS + slot) * collNet->nChannels + channel;
  *offset = slotSize * globalSlot;
  return ncclSuccess;
}

static ncclResult_t sharedBuffersDestroy(struct ncclCollNetSharedRes* collNet) {
  if (collNet->size == 0) return ncclSuccess;
  NCCLCHECK(ncclCudaFree(collNet->cudaBuff));
  NCCLCHECK(ncclCudaHostFree(collNet->hostBuff));
  // This will be called multiple times, with multiple channels and send/recv. Make sure we only do it once.
  collNet->size = 0;
  return ncclSuccess;
}

static ncclResult_t recvProxySetup(struct ncclProxyConnection* connection, struct ncclProxyState* proxyState, void* reqBuff, int reqSize, void* respBuff, int respSize, int* done) {
  struct setupReq* req = (struct setupReq*)reqBuff;
  if (reqSize != sizeof (struct setupReq)) return ncclInternalError;

  struct recvResources* resources;
  NCCLCHECK(ncclCalloc(&resources, 1));
  connection->transportResources = resources;
  connection->shared = 1;

  resources->netDev = req->netDev;
  resources->useGdr = req->useGdr;
  resources->needFlush = req->needFlush;
  ncclNetProperties_t props;
  NCCLCHECK(proxyState->ncclCollNet->getProperties(req->netDev, &props));
  connection->collNet = req->collNet;
  /* DMA-BUF support */
  resources->useDmaBuf = resources->useGdr && proxyState->dmaBufSupport && (props.ptrSupport & NCCL_PTR_DMABUF);

  collNetHandle_t* netHandle = (collNetHandle_t*) respBuff;
  if (respSize != sizeof(collNetHandle_t)) return ncclInternalError;

  NCCLCHECK(sharedListen(proxyState, req->netDev, req->collNet, netHandle));
  return ncclSuccess;
}

static ncclResult_t sendProxyConnect(struct ncclProxyConnection* connection, struct ncclProxyState* proxyState, void* reqBuff, int reqSize, void* respBuff, int respSize, int* done) {
  if (reqSize != sizeof(struct collNetConnectArgs)) { WARN("sendProxyConnect: reqSize is %d != %ld", reqSize, sizeof(struct collNetConnectArgs)); return ncclInternalError; }
  struct collNetConnectArgs* args = (struct collNetConnectArgs*)reqBuff;
  struct collNetSendConnectInfo* info = (struct collNetSendConnectInfo*)(args->connectInfos+args->rank);

  struct sendResources* resources = (struct sendResources*)(connection->transportResources);

  // Get info from recv side
  resources->collNetRank = args->rank;
  resources->reqFifo = (struct reqSlot (*)[NCCL_STEPS])(info->reqFifo);

  for (int p=0; p<NCCL_NUM_PROTOCOLS; p++)
    resources->recvMhandles[p] = info->mhandles[p];

  NCCLCHECK(sharedConnect(proxyState, resources->netDev, args->connectInfos, args->nranks, args->rank, connection->collNet, &resources->collNetComm));

  // Collnet connect is allowed to fail. Gracefully handle that case by returning NULL to the caller.
  if (respSize != sizeof(struct connectMap*)) { WARN("sendProxyConnect: respSize is %d != %ld", respSize, sizeof(void*)); return ncclInternalError; }
  if (resources->collNetComm == NULL) {
    *((struct connectMap**)respBuff) = NULL;
    return ncclSuccess;
  }
  connection->proxyAppendPtr = connection->collNet->proxyAppend + 2 * resources->netDev;

  struct connectMap* map = &resources->map;

  NCCL_NET_MAP_ADD_POINTER(map, 0, 0, sizeof(struct ncclSendMem), sendMem);
  NCCL_NET_MAP_ADD_POINTER(map, 0, 0, sizeof(struct ncclRecvMem), recvMem);

  NCCLCHECK(ncclCudaHostCalloc(&map->mems[NCCL_NET_MAP_HOSTMEM].cpuPtr, map->mems[NCCL_NET_MAP_HOSTMEM].size));
  map->mems[NCCL_NET_MAP_HOSTMEM].gpuPtr = map->mems[NCCL_NET_MAP_HOSTMEM].cpuPtr;
  if (ncclGdrCopy && ncclParamGdrCopySyncEnable()) {
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
  (resources->gdcSync ? *resources->gdcSync : resources->sendMem->head) = -NCCL_STEPS;

  // Allocate & Register shared buffers for the Simple protocol
  int bank = resources->useGdr ? NCCL_NET_MAP_SHARED_DEVMEM : NCCL_NET_MAP_SHARED_HOSTMEM;
  struct connectMapMem* mapMem = map->mems+bank;
  NCCLCHECK(sharedBuffersInit(connection->collNet, resources->useGdr, &mapMem->gpuPtr, &mapMem->cpuPtr, &mapMem->size));
  NCCL_NET_MAP_ADD_POINTER(map, 1, resources->useGdr, mapMem->size, buffs[NCCL_PROTO_SIMPLE]);

#if CUDA_VERSION >= 11070
  /* DMA-BUF support */
  if (resources->useGdr && resources->useDmaBuf) {
    int dmabuf_fd;
    CUCHECK(cuMemGetHandleForAddressRange((void *)&dmabuf_fd, (CUdeviceptr)mapMem->cpuPtr, mapMem->size, CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, 0));
    NCCLCHECK(proxyState->ncclCollNet->regMrDmaBuf(resources->collNetComm, mapMem->cpuPtr, mapMem->size,
                                                  NCCL_PTR_CUDA, 0ULL, dmabuf_fd,
                                                  &resources->sendMhandles[NCCL_PROTO_SIMPLE]));
    (void)close(dmabuf_fd);
  } else // FALL-THROUGH to nv_peermem GDR path
#endif
  {
    NCCLCHECK(proxyState->ncclCollNet->regMr(resources->collNetComm, mapMem->cpuPtr, mapMem->size,
                                            resources->useGdr ? NCCL_PTR_CUDA : NCCL_PTR_HOST,
                                            &resources->sendMhandles[NCCL_PROTO_SIMPLE]));
  }

  *((struct connectMap**)respBuff) = &resources->map;
  return ncclSuccess;
}

static ncclResult_t recvProxyConnect(struct ncclProxyConnection* connection, struct ncclProxyState* proxyState, void* reqBuff, int reqSize, void* respBuff, int respSize, int* done) {
  if (reqSize != sizeof(struct collNetConnectArgs)) { WARN("recvProxyConnect: reqSize is %d != %ld", reqSize, sizeof(struct collNetConnectArgs)); return ncclInternalError; }
  struct collNetConnectArgs* args = (struct collNetConnectArgs*)reqBuff;

  struct recvResources* resources = (struct recvResources*)(connection->transportResources);
  struct collNetSendConnectInfo* info = (struct collNetSendConnectInfo*)(args->connectInfos+args->rank);
  resources->collNetRank = args->rank;

  NCCLCHECK(sharedConnect(proxyState, resources->netDev, args->connectInfos, args->nranks, args->rank, connection->collNet, &resources->collNetComm));

  // Collnet connect is allowed to fail. Gracefully handle that case by returning NULL to the caller.
  if (respSize != sizeof(struct connectMap*)) { WARN("sendProxyConnect: respSize is %d != %ld", respSize, sizeof(void*)); return ncclInternalError; }
  if (resources->collNetComm == NULL) {
    *((struct connectMap**)respBuff) = NULL;
    return ncclSuccess;
  }
  connection->proxyAppendPtr = connection->collNet->proxyAppend + 2 * resources->netDev + 1;

  struct connectMap* map = &resources->map;

  NCCL_NET_MAP_ADD_POINTER(map, 0, 0, sizeof(struct ncclSendMem), sendMem);
  NCCL_NET_MAP_ADD_POINTER(map, 0, 0, sizeof(struct ncclRecvMem), recvMem);

  NCCLCHECK(ncclCudaHostCalloc(&map->mems[NCCL_NET_MAP_HOSTMEM].cpuPtr, map->mems[NCCL_NET_MAP_HOSTMEM].size));
  map->mems[NCCL_NET_MAP_HOSTMEM].gpuPtr = map->mems[NCCL_NET_MAP_HOSTMEM].cpuPtr;
  if (ncclGdrCopy) {
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

  // Allocate & Register shared buffers for the Simple protocol
  int bank = resources->useGdr ? NCCL_NET_MAP_SHARED_DEVMEM : NCCL_NET_MAP_SHARED_HOSTMEM;
  struct connectMapMem* mapMem = map->mems+bank;
  NCCLCHECK(sharedBuffersInit(connection->collNet, resources->useGdr, &mapMem->gpuPtr, &mapMem->cpuPtr, &mapMem->size));
  NCCL_NET_MAP_ADD_POINTER(map, 1, resources->useGdr, mapMem->size, buffs[NCCL_PROTO_SIMPLE]);

#if CUDA_VERSION >= 11070
  /* DMA-BUF support */
  if (resources->useGdr && resources->useDmaBuf) {
    int dmabuf_fd;
    CUCHECK(cuMemGetHandleForAddressRange((void *)&dmabuf_fd, (CUdeviceptr)mapMem->cpuPtr, mapMem->size, CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, 0));
    NCCLCHECK(proxyState->ncclCollNet->regMrDmaBuf(resources->collNetComm, mapMem->cpuPtr, mapMem->size,
                                                  NCCL_PTR_CUDA, 0ULL, dmabuf_fd,
                                                  &resources->mhandles[NCCL_PROTO_SIMPLE]));
    (void)close(dmabuf_fd);
  } else // FALL-THROUGH to nv_peermem GDR path
#endif
  {
    NCCLCHECK(proxyState->ncclCollNet->regMr(resources->collNetComm, mapMem->cpuPtr, mapMem->size,
                                            resources->useGdr ? NCCL_PTR_CUDA : NCCL_PTR_HOST,
                                            &resources->mhandles[NCCL_PROTO_SIMPLE]));
  }

  // Pass info to send side
  info->reqFifo = resources->reqFifo;
  for (int p=0; p<NCCL_NUM_PROTOCOLS; p++)
    info->mhandles[p] = resources->mhandles[p];

  if (respSize != sizeof(struct connectMap*)) { WARN("recvProxyConnect: respSize is %d != %ld", respSize, sizeof(void*)); return ncclInternalError; }
  *((struct connectMap**)respBuff) = &resources->map;
  return ncclSuccess;
}

static ncclResult_t sendProxyFree(struct ncclProxyConnection* connection, struct ncclProxyState* proxyState) {
  struct sendResources* resources = (struct sendResources*)(connection->transportResources);

  if (resources) {
    for (int p = 0; p < NCCL_NUM_PROTOCOLS; p++) {
      if (resources->sendMhandles[p]) {
        NCCLCHECK(proxyState->ncclCollNet->deregMr(resources->collNetComm, resources->sendMhandles[p]));
      }
    }
    struct connectMapMem* mems = resources->map.mems;
    NCCLCHECK(ncclCudaHostFree(mems[NCCL_NET_MAP_HOSTMEM].cpuPtr));
    NCCLCHECK(ncclCudaFree(mems[NCCL_NET_MAP_DEVMEM].cpuPtr));
    if (mems[NCCL_NET_MAP_GDCMEM].cpuPtr) NCCLCHECK(ncclGdrCudaFree(resources->gdrDesc));
    NCCLCHECK(sharedBuffersDestroy(connection->collNet));
    NCCLCHECK(sharedFree(proxyState, connection->collNet, resources->netDev));
    if (ncclAtomicRefCountDecrement(&connection->collNet->refCount) == 0) free(connection->collNet);
    free(connection->transportResources);
  }
  return ncclSuccess;
}

static ncclResult_t recvProxyFree(struct ncclProxyConnection* connection, struct ncclProxyState* proxyState) {
  struct recvResources* resources = (struct recvResources*)(connection->transportResources);

  if (resources) {
    for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
      if (resources->mhandles[p]) {
        NCCLCHECK(proxyState->ncclCollNet->deregMr(resources->collNetComm, resources->mhandles[p]));
      }
    }
    struct connectMapMem* mems = resources->map.mems;
    NCCLCHECK(ncclCudaHostFree(mems[NCCL_NET_MAP_HOSTMEM].cpuPtr));
    NCCLCHECK(ncclCudaFree(mems[NCCL_NET_MAP_DEVMEM].cpuPtr));
    if (mems[NCCL_NET_MAP_GDCMEM].cpuPtr) NCCLCHECK(ncclGdrCudaFree(resources->gdrDesc));
    NCCLCHECK(sharedBuffersDestroy(connection->collNet));
    NCCLCHECK(sharedFree(proxyState, connection->collNet, resources->netDev));
    if (ncclAtomicRefCountDecrement(&connection->collNet->refCount) == 0) free(connection->collNet);
    free(connection->transportResources);
  }
  return ncclSuccess;
}

static size_t calcAlgoOffset(struct ncclProxyArgs* args, int isAllNotOne, int sub, uint64_t step) {
  int chunkSize = args->chunkSize;
  int nNodes = args->specifics.collnetDirect.nNodes;
  int node = args->specifics.collnetDirect.node;
  size_t sizePerRank = args->specifics.collnetDirect.sizePerRank;
  size_t offset = (step*(args->nsubs) + sub)*chunkSize;
  if (isAllNotOne) {
    offset = std::min<size_t>(offset, nNodes*sizePerRank);
  } else {
    offset = std::max<size_t>(offset, (node+0)*sizePerRank);
    offset = std::min<size_t>(offset, (node+1)*sizePerRank);
  }
  return offset;
}

static int calcRegionOffset(
    struct ncclProxyArgs* args, int isRecvNotSend, int sub, uint64_t step,
    int side // 0=begin, 1=end
  ) {
  struct ncclCollNetSharedRes* collNet = args->subs[0].connection->collNet;
  int slotSize = collNet->buffSize/NCCL_STEPS;
  int chunkSize = args->chunkSize;
  int base = isRecvNotSend*NCCL_STEPS + (step%NCCL_STEPS);
  base *= collNet->nChannels*slotSize;
  if (args->coll == ncclFuncAllReduce) {
    return base + (sub+side)*chunkSize;
  } else {
    int isAllNotOne = isRecvNotSend ^ (args->coll == ncclFuncReduceScatter);
    int sub0 = sub - (sub%COLLNET_GROUP_NSUBS);
    size_t off = sub0*slotSize;
    off += calcAlgoOffset(args, isAllNotOne, sub+side, step)
         - calcAlgoOffset(args, isAllNotOne, sub0, step);
    return base + off;
  }
}

#define LAST_OF_GROUP(args, s) \
  ((s)%COLLNET_GROUP_NSUBS == COLLNET_GROUP_NSUBS-1 || (s) == (args)->nsubs-1)

static constexpr int calcStepsPerGroup(int nGroups) {
  //return NCCL_STEPS/nGroups;
  return NCCL_STEPS;
}

static ncclResult_t sendProxyProgress(struct ncclProxyState* proxyState, struct ncclProxyArgs* args) {
  if (args->state == ncclProxyOpReady) {
    for (int s=0; s<args->nsubs; s++) {
      struct ncclProxySubArgs* sub = args->subs+s;
      struct sendResources* resources = (struct sendResources*) (sub->connection->transportResources);
      // Round to next multiple of sliceSteps
      sub->base = ROUNDUP(resources->step, args->chunkSteps);
      sub->posted = sub->received = sub->transmitted = sub->done = 0;
      resources->step = sub->base + sub->nsteps;
    }
    args->state = ncclProxyOpProgress;
  }
  args->idle = 1;
  if (args->state == ncclProxyOpProgress) {
    int p = NCCL_PROTO_SIMPLE;
    int nGroups = DIVUP(args->nsubs, COLLNET_GROUP_NSUBS);
    for (int s=0; s<args->nsubs; s++) {
      struct ncclProxySubArgs* sub = args->subs+s;
      struct sendResources* resources = (struct sendResources*) (sub->connection->transportResources);
      void* sendMhandle = resources->sendMhandles[p];
      void* recvMhandle = resources->recvMhandles[p];
      char* region = NCCL_NET_MAP_GET_POINTER(&resources->map, gpu, buffs[p]);
      auto reqFifo = resources->reqFifo;
      int group = s/COLLNET_GROUP_NSUBS;
      int groupStart = s - (s%COLLNET_GROUP_NSUBS);

      if (sub->posted < sub->nsteps && sub->posted < sub->done + NCCL_STEPS) {
        int buffSlot = (sub->base+sub->posted)%NCCL_STEPS;
        resources->recvMem->connFifo[buffSlot].offset = calcRegionOffset(args, 0, s, sub->posted, 0);
        __sync_synchronize();
        volatile uint64_t* sendHead = resources->gdcSync ? resources->gdcSync : &resources->sendMem->head;
        TRACE(NCCL_NET, "sendProxy [%ld/%d/%d] posted offset %d @ %p signal %ld->%ld", long(sub->posted), group, buffSlot, resources->recvMem->connFifo[buffSlot].offset, &resources->recvMem->connFifo[buffSlot].offset, long(*sendHead), long(sub->base + sub->posted + args->sliceSteps - NCCL_STEPS));
        sub->posted += args->sliceSteps;
        *sendHead = sub->base + sub->posted - NCCL_STEPS;
        if (resources->gdcSync) wc_store_fence(); // Flush out WC write
      }
      if (sub->received < sub->posted && sub->received < sub->done + calcStepsPerGroup(nGroups)) {
        int buffSlot = (sub->base+sub->received)%NCCL_STEPS;
        volatile struct ncclConnFifo* connFifo = (volatile struct ncclConnFifo*)resources->recvMem->connFifo;
        volatile uint64_t* recvTail = &resources->recvMem->tail;
        if (connFifo[buffSlot].size != -1 && ((*recvTail > (sub->base+sub->received)))) {
          if (args->coll != ncclFuncAllReduce) {
            int sendBeg = calcRegionOffset(args, 0, s, sub->received, 0);
            int sendEnd = calcRegionOffset(args, 0, s, sub->received, 1);
            if (sendEnd-sendBeg != connFifo[buffSlot].size) {
              WARN("CollNet sizes: want=%d got=%ld", sendEnd-sendBeg, connFifo[buffSlot].size);
              return ncclInternalError;
            }
          }
          connFifo[buffSlot].size = -1;
          sub->received += args->sliceSteps;
          args->idle = 0;
        }
      }
      // Enforce collective ordering of collnet ops.
      bool ordered = s==0 ? args->subs[args->nsubs-1].transmitted == sub->transmitted
                          : sub->transmitted < (sub-1)->transmitted;
      if (ordered && (sub->transmitted < sub->received)) {
        if (LAST_OF_GROUP(args, s)) {
          int buffSlot = (sub->base+sub->transmitted)%NCCL_STEPS;
          if (!reqFifo[group][buffSlot].turnIsSendNotRecv) continue;

          ssize_t sizePerRank = 0;
          size_t allBeg = calcAlgoOffset(args, 1, groupStart, sub->transmitted);
          size_t allEnd = calcAlgoOffset(args, 1, s+1, sub->transmitted);
          int sendBeg = calcRegionOffset(args, 0, groupStart, sub->transmitted, 0);
          int sendEnd = calcRegionOffset(args, 0, s, sub->transmitted, 1);
          int recvBeg = calcRegionOffset(args, 1, groupStart, sub->transmitted, 0);
          int recvEnd = calcRegionOffset(args, 1, s, sub->transmitted, 1);
          reqFifo[group][buffSlot].size = recvEnd - recvBeg;
          size_t eltSize = ncclTypeSize((ncclDataType_t)args->dtype);

          if (sendBeg==sendEnd && recvBeg==recvEnd) {
            sub->requests[buffSlot] = nullptr; // trivally finished request
          } else {
            if (args->coll == ncclFuncAllReduce) {
              int count = (sendEnd-sendBeg)/eltSize;
              NCCLCHECK(proxyState->ncclCollNet->iallreduce(resources->collNetComm, region+sendBeg, region+recvBeg, count, (ncclDataType_t)args->dtype, (ncclRedOp_t)args->redOp, sendMhandle, recvMhandle, sub->requests+buffSlot));
            } else {
              sizePerRank = args->specifics.collnetDirect.sizePerRank;
              if (args->coll == ncclFuncAllGather) {
                ncclNetSGE_v8_t recvParts;
                recvParts.mhandle = recvMhandle;
                recvParts.address = region + recvBeg;
                recvParts.size = allEnd - allBeg;
                NCCLCHECK(proxyState->ncclCollNet->iallgather(
                  resources->collNetComm, region+sendBeg, 1, &recvParts,
                  sizePerRank, allBeg, allEnd-allBeg,
                  sendMhandle, sub->requests+buffSlot));
              } else {
                ncclNetSGE_v8_t sendParts;
                sendParts.mhandle = sendMhandle;
                sendParts.address = region + sendBeg;
                sendParts.size = allEnd - allBeg;
                NCCLCHECK(proxyState->ncclCollNet->ireducescatter(
                  resources->collNetComm, 1, &sendParts, region+recvBeg,
                  sizePerRank, allBeg, allEnd-allBeg,
                  (ncclDataType_t)args->dtype, (ncclRedOp_t)args->redOp,
                  recvMhandle, sub->requests+buffSlot));
              }
            }
            if (sub->requests[buffSlot] == nullptr) continue;

            if (args->coll == ncclFuncAllReduce) {
              TRACE(NCCL_NET, "sendProxy [%ld/%d/%d] Iallreduce posted, size %d req %p", (long)sub->transmitted, group, buffSlot, int(sendEnd-sendBeg), sub->requests[buffSlot]);
            } else if (args->coll == ncclFuncAllGather) {
              TRACE(NCCL_NET, "sendProxy [%ld/%d/%d] Iallgather posted sendSize=%ld recvOffset=%ld recvSize=%ld request=%p", (long)sub->transmitted, group, buffSlot, long(sizePerRank), long(allBeg), long(allEnd-allBeg), sub->requests[buffSlot]);
            } else {
              TRACE(NCCL_NET, "sendProxy [%ld/%d/%d] Ireducescatter posted sendOffset=%ld sendSize=%ld recvSize=%ld request=%p", (long)sub->transmitted, group, buffSlot, long(allBeg), long(allEnd-allBeg), long(sizePerRank), sub->requests[buffSlot]);
            }
          }
        }
        sub->transmitted += args->sliceSteps;
        args->idle = 0;
        continue;
      }
      // Check whether the network has completed some send operations.
      if (LAST_OF_GROUP(args, s) && sub->done < sub->transmitted) {
        int done, size;
        int buffSlot = (sub->base+sub->done)%NCCL_STEPS;
        done = 1;
        if (sub->requests[buffSlot]) NCCLCHECK(proxyState->ncclCollNet->test((void*)(sub->requests[buffSlot]), &done, &size));
        if (done) {
          TRACE(NCCL_NET, "sendProxy [%ld/%d/%d] request %p done, size %d", (long)sub->done, group, buffSlot, sub->requests[buffSlot], size);
          sub->requests[buffSlot] = nullptr;
          reqFifo[group][buffSlot].turnIsSendNotRecv = false; // Notify recvProxy
          for (int i=groupStart; i<=s; i++) args->subs[i].done += args->sliceSteps;
          args->idle = 0;
          int allDone = 1;
          for (int i=0; i<args->nsubs; i++) {
            if (args->subs[i].done < args->subs[i].nsteps) { allDone = 0; break; }
          }
          if (allDone) {
            args->state = ncclProxyOpNone;
            TRACE(NCCL_NET, "sendProxy [%ld/%d] stopped", (long)sub->done, s);
          }
        }
      }
    }
  }
  return ncclSuccess;
}

static ncclResult_t recvProxyProgress(struct ncclProxyState* proxyState, struct ncclProxyArgs* args) {
  if (args->state == ncclProxyOpReady) {
    for (int s=0; s<args->nsubs; s++) {
      struct ncclProxySubArgs* sub = args->subs+s;
      struct recvResources* resources = (struct recvResources*) (sub->connection->transportResources);
      // Round to next multiple of sliceSteps
      sub->base = ROUNDUP(resources->step, args->chunkSteps);
      sub->posted = sub->received = sub->flushed = sub->transmitted = sub->done = 0;
      resources->step = sub->base + sub->nsteps;
      memset(sub->requests, 0, sizeof(sub->requests));
    }
    args->state = ncclProxyOpProgress;
  }
  args->idle = 1;
  if (args->state == ncclProxyOpProgress) {
    int p = NCCL_PROTO_SIMPLE;
    int nGroups = DIVUP(args->nsubs, COLLNET_GROUP_NSUBS);
    for (int s=0; s<args->nsubs; s++) {
      int group = s/COLLNET_GROUP_NSUBS;
      int groupStart = s - (s%COLLNET_GROUP_NSUBS);
      struct ncclProxySubArgs* sub = args->subs+s;
      struct recvResources* resources = (struct recvResources*) (sub->connection->transportResources);
      void* mhandle = resources->mhandles[p];
      auto reqFifo = resources->reqFifo;
      char* region = NCCL_NET_MAP_GET_POINTER(&resources->map, cpu, buffs[p]);

      // Enforce sync between operations of the same group.
      if (LAST_OF_GROUP(args, s) && (sub->posted < sub->done + calcStepsPerGroup(nGroups)) && (sub->posted < sub->nsteps)) {
        int buffSlot = (sub->base+sub->posted)%NCCL_STEPS;
        reqFifo[group][buffSlot].turnIsSendNotRecv = true;
        TRACE(NCCL_NET, "recvProxy [%ld/%d/%d] posted buffer", (long)sub->posted, group, buffSlot);
        sub->posted += args->sliceSteps;
        args->idle = 0;
        continue;
      }
      if (LAST_OF_GROUP(args, s) && (sub->received < sub->posted)) {
        int buffSlot = (sub->base+sub->received)%NCCL_STEPS;
        if (!reqFifo[group][buffSlot].turnIsSendNotRecv) { // Buffer is cleared : coll is complete
          int recvBeg = calcRegionOffset(args, 1, groupStart, sub->received, 0);
          int recvEnd = calcRegionOffset(args, 1, s, sub->received, 1);
          int totalSize = recvEnd - recvBeg;
          TRACE(NCCL_NET, "recvProxy [%ld/%d/%d] received, size %d chunkSize=%d", (long)sub->received, group, buffSlot, totalSize, args->chunkSize);
          sub->received += args->sliceSteps;
          if (reqFifo[group][buffSlot].size > 0 && resources->useGdr && resources->needFlush) {
            // GDRCOPY support
            if (resources->gdcFlush) {
#if defined (__x86_64__)
              // Force a PCI-E read from GPU memory
              asm volatile ("mov (%0), %%eax" :: "l"(resources->gdcFlush) : "%eax");
#else
              WARN("NET: GDR Flush only supported on x86_64");
              return ncclInternalError;
#endif
            } else {
              NCCLCHECK(proxyState->ncclCollNet->iflush(resources->collNetComm, region+recvBeg, totalSize, mhandle, sub->requests+buffSlot));
            }
          }
          args->idle = 0;
          continue;
        }
      }
      if (LAST_OF_GROUP(args, s) && (sub->flushed < sub->received)) {
        // Progress flush operations
        int buffSlot = (sub->base + sub->flushed)%NCCL_STEPS;
        int done = 1;
        if (sub->requests[buffSlot]) NCCLCHECK(proxyState->ncclCollNet->test(sub->requests[buffSlot], &done, NULL));
        if (done) {
          sub->requests[buffSlot] = nullptr;
          TRACE(NCCL_NET, "recvProxy [%ld/%d/%d] flushed", (long)sub->flushed, group, buffSlot);
          for (int i=group*COLLNET_GROUP_NSUBS; i<=s; i++) args->subs[i].flushed += args->sliceSteps;
          args->idle = 0;
          //continue;
        }
      }
      if (sub->transmitted < sub->flushed) {
        int buffSlot = (sub->base + sub->transmitted)%NCCL_STEPS;
        volatile struct ncclConnFifo* connFifo = (volatile struct ncclConnFifo*)resources->recvMem->connFifo;
        connFifo[buffSlot].offset = calcRegionOffset(args, 1, s, sub->transmitted, 0);
        __sync_synchronize();
        volatile uint64_t* recvTail = resources->gdcSync ? resources->gdcSync : &resources->recvMem->tail;
        *recvTail = sub->base + sub->flushed;
        if (resources->gdcSync) wc_store_fence(); // Flush out WC write
        sub->transmitted += args->sliceSteps;
        args->idle = 0;
        continue;
      }
      // Enforce sync here to make sure the last sub doesn't increase "done" before all others in the group have
      // reached the same point, otherwise we would start posting buffers to the send proxy before we're done
      // processing all the shared buffer.
      bool groupSync = s==0 ? args->subs[args->nsubs-1].done == sub->done
                            : (sub-1)->done > sub->done;
      volatile uint64_t* sendHead = &resources->sendMem->head;
      if (groupSync && sub->done < sub->transmitted && (sub->base+sub->done) < *sendHead) {
        sub->done += args->sliceSteps;
        args->idle = 0;
        if (sub->done == sub->nsteps && s == args->nsubs-1) {
          args->state = ncclProxyOpNone;
          TRACE(NCCL_NET, "recvProxy [%ld/%d] stopped", (long)sub->done, s);
        }
      }
    }
  }
  return ncclSuccess;
}

struct ncclTransport collNetTransport = {
  "COL",
  canConnect,
  { sendSetup, sendConnect, sendFree, NULL, sendProxySetup, sendProxyConnect, sendProxyFree, sendProxyProgress },
  { recvSetup, recvConnect, recvFree, NULL, recvProxySetup, recvProxyConnect, recvProxyFree, recvProxyProgress }
};
