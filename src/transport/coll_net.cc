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
  volatile void* recvBuff;
  volatile int size;
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
  uint64_t* gdcSync;
  uint64_t* gdcFlush;
  void* gdrDesc;
  void* mhandles[NCCL_NUM_PROTOCOLS];
  uint64_t step;
  struct reqSlot reqFifo[COLLNET_MAX_GROUPS][NCCL_STEPS];
  int collNetRank;
};

/* Determine if we can communicate with the peer */
static ncclResult_t canConnect(int* ret, struct ncclTopoSystem* topo, struct ncclTopoGraph* graph, struct ncclPeerInfo* info1, struct ncclPeerInfo* info2) {
  *ret = 1;
  return ncclSuccess;
}

struct setupReq {
  int netDev;
  int useGdr;
};


/* Setup send connector, and return connect information for others in the coll
 * communicator to connect to me */
static ncclResult_t sendSetup(struct ncclComm* comm, struct ncclTopoGraph* graph, struct ncclPeerInfo* myInfo, struct ncclPeerInfo* peerInfo, struct ncclConnect* connectInfo, struct ncclConnector* send, int channelId, int connIndex) {
  struct setupReq req;

  int proxyRank;
  NCCLCHECK(ncclTopoGetNetDev(comm, myInfo->rank, graph, channelId, -1, &req.netDev, &proxyRank));
  NCCLCHECK(ncclTopoCheckGdr(comm->topo, myInfo->busId, req.netDev, 1, &req.useGdr));
  send->conn.direct |= req.useGdr ? NCCL_DIRECT_NIC : 0;

  NCCLCHECK(ncclTopoGetLocalRank(comm->topo, myInfo->rank, &send->proxyConn.localRank));
  NCCLCHECK(ncclProxyConnect(comm, TRANSPORT_COLLNET, 1, myInfo->rank, &send->proxyConn));
  NCCLCHECK(ncclProxyCall(&send->proxyConn, ncclProxyMsgSetup, &req, sizeof(req), NULL, 0));

  INFO(NCCL_INIT|NCCL_NET,"CollNet %02d : %d [send] via COLLNET/%s/%d%s", channelId, myInfo->rank, collNetName(), req.netDev,
      req.useGdr ? "/GDRDMA" : "");
  return ncclSuccess;
}

static ncclResult_t recvSetup(struct ncclComm* comm, struct ncclTopoGraph* graph, struct ncclPeerInfo* myInfo, struct ncclPeerInfo* peerInfo, struct ncclConnect* connectInfo, struct ncclConnector* recv, int channelId, int connIndex) {
  struct setupReq req;

  int proxyRank;
  NCCLCHECK(ncclTopoGetNetDev(comm, myInfo->rank, graph, channelId, -1, &req.netDev, &proxyRank));
  NCCLCHECK(ncclTopoCheckGdr(comm->topo, myInfo->busId, req.netDev, 0, &req.useGdr));
  recv->conn.direct |= req.useGdr ? NCCL_DIRECT_NIC : 0;

  NCCLCHECK(ncclTopoGetLocalRank(comm->topo, myInfo->rank, &recv->proxyConn.localRank));
  NCCLCHECK(ncclProxyConnect(comm, TRANSPORT_COLLNET, 0, myInfo->rank, &recv->proxyConn));
  struct collNetRecvConnectInfo* info = (struct collNetRecvConnectInfo*) connectInfo;
  NCCLCHECK(ncclProxyCall(&recv->proxyConn, ncclProxyMsgSetup, &req, sizeof(req), &info->collNetHandle, sizeof(collNetHandle_t)));

  INFO(NCCL_INIT|NCCL_NET,"CollNet %02d : %d [receive] via COLLNET/%s/%d%s", channelId, myInfo->rank, collNetName(), req.netDev,
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

static ncclResult_t sendConnect(struct ncclComm* comm, struct ncclConnect* connectInfos, int nranks, int rank, struct ncclConnector* send) {
  // We're on the same process as the proxy. We can pass a pointer to a struct.
  struct collNetConnectArgs args = { rank, nranks, connectInfos };
  struct connectMap* map;
  NCCLCHECK(ncclProxyCall(&send->proxyConn, ncclProxyMsgConnect, &args, sizeof(struct collNetConnectArgs), &map, sizeof(struct connectMap*)));

  // If collnet connect failed, propagate error to fallback on regular p2p
  if (map == NULL) return ncclSystemError;

  //NCCLCHECK(collNetDumpMap(map));

  struct ncclSendMem *sendMem = (struct ncclSendMem*) NCCL_NET_MAP_GET_POINTER(map, gpu, sendMem);
  void* gdcMem = map->mems[NCCL_NET_MAP_GDCMEM].gpuPtr;
  send->conn.head = gdcMem ? (uint64_t*)gdcMem : &sendMem->head;

  struct ncclRecvMem *recvMem = (struct ncclRecvMem*) NCCL_NET_MAP_GET_POINTER(map, gpu, recvMem);
  send->conn.tail = &recvMem->tail;
  send->conn.sizesFifo = recvMem->sizesFifo;
  for (int i=0; i<NCCL_STEPS; i++) send->conn.sizesFifo[i] = -1;
  send->conn.offsFifo = recvMem->offsFifo;

  for (int p=0; p<NCCL_NUM_PROTOCOLS; p++)
    send->conn.buffs[p] = NCCL_NET_MAP_GET_POINTER(map, gpu, buffs[p]);
  return ncclSuccess;
}

static ncclResult_t recvConnect(struct ncclComm* comm, struct ncclConnect* connectInfos, int nranks, int rank, struct ncclConnector* recv) {
  // We're on the same process as the proxy. We can pass a pointer to a struct.
  struct collNetConnectArgs args = { rank, nranks, connectInfos };
  struct connectMap* map;
  NCCLCHECK(ncclProxyCall(&recv->proxyConn, ncclProxyMsgConnect, &args, sizeof(struct collNetConnectArgs), &map, sizeof(struct connectMap*)));

  // If collnet connect failed, propagate error to fallback on regular p2p
  if (map == NULL) return ncclSystemError;

  //NCCLCHECK(collNetDumpMap(map));

  struct ncclSendMem *sendMem = (struct ncclSendMem*) NCCL_NET_MAP_GET_POINTER(map, gpu, sendMem);
  recv->conn.head = &sendMem->head;

  struct ncclRecvMem *recvMem = (struct ncclRecvMem*) NCCL_NET_MAP_GET_POINTER(map, gpu, recvMem);
  void* gdcMem = map->mems[NCCL_NET_MAP_GDCMEM].gpuPtr;
  recv->conn.tail = gdcMem ? (uint64_t*)gdcMem : &recvMem->tail;
  recv->conn.offsFifo = recvMem->offsFifo;

  for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
    recv->conn.buffs[p] = NCCL_NET_MAP_GET_POINTER(map, gpu, buffs[p]);
  }
  return ncclSuccess;
}

static ncclResult_t sendFree(struct ncclConnector* send) {
  return ncclSuccess;
}

static ncclResult_t recvFree(struct ncclConnector* recv) {
  return ncclSuccess;
}

static ncclResult_t sendProxySetup(struct ncclProxyConnection* connection, struct ncclComm* comm, void* reqBuff, int reqSize, void* respBuff, int respSize, int* done) {
  struct setupReq* req = (struct setupReq*)reqBuff;
  if (reqSize != sizeof(struct setupReq)) return ncclInternalError;

  struct sendResources* resources;
  NCCLCHECK(ncclCalloc(&resources, 1));
  connection->transportResources = resources;
  connection->shared = 1;

  resources->netDev = req->netDev;
  resources->useGdr = req->useGdr;
  return ncclSuccess;
}

struct sharedResources {
  void* collNetListenComms[MAXCHANNELS];
  void* collNetComms[MAXCHANNELS];
  int commRefCount[NCCL_MAX_NETDEVS];
};

ncclResult_t sharedListen(struct ncclComm* comm, int netDev, void* collNetHandle) {
  struct sharedResources* resources = (struct sharedResources*)comm->proxyState.progressState.collNet.resources;
  if (resources == NULL) {
    NCCLCHECK(ncclCalloc(&resources, 1));
    comm->proxyState.progressState.collNet.resources = resources;
  }
  if (resources->collNetComms[netDev] == NULL)
    NCCLCHECK(collNetListen(netDev, collNetHandle, resources->collNetListenComms+netDev));
  return ncclSuccess;
}

static ncclResult_t sharedConnect(struct ncclComm* comm, int netDev, struct ncclConnect* connectInfos, int nranks, int rank, void** collNetComm) {
  struct sharedResources* resources = (struct sharedResources*)comm->proxyState.progressState.collNet.resources;
  if (resources->collNetComms[netDev] == NULL) {
    // Connect to coll comm
    collNetHandle_t** handlePtrs = NULL;
    NCCLCHECK(ncclCalloc(&handlePtrs, nranks));
    for (int i = 0; i < nranks; i++) {
      struct collNetRecvConnectInfo* info = (struct collNetRecvConnectInfo*)(connectInfos+i);
      handlePtrs[i] = &(info->collNetHandle);
    }
    ncclResult_t ret = collNetConnect((void**)handlePtrs, nranks, rank,
          resources->collNetListenComms[netDev],
          resources->collNetComms+netDev);
    free(handlePtrs);
    if (ret == ncclSuccess) {
      // Close listen comm
      NCCLCHECK(collNetCloseListen(resources->collNetListenComms[netDev]));
    } else {
      resources->collNetListenComms[netDev] = NULL;
    }
  }
  *collNetComm = resources->collNetComms[netDev];
  if (*collNetComm) resources->commRefCount[netDev]++;
  return ncclSuccess;
}

static ncclResult_t sharedFree(struct ncclComm* comm, int netDev) {
  struct sharedResources* resources = (struct sharedResources*)comm->proxyState.progressState.collNet.resources;
  resources->commRefCount[netDev]--;
  if (resources->commRefCount[netDev] == 0) {
    NCCLCHECK(collNetCloseColl(resources->collNetComms[netDev]));
  }
  for (int n=0; n<NCCL_MAX_NETDEVS; n++) if (resources->commRefCount[n]) return ncclSuccess;
  comm->proxyState.progressState.collNet.resources = NULL;
  free(resources);
  return ncclSuccess;
}

static ncclResult_t sharedBuffersInit(struct ncclComm* comm, int cuda, char** gpuPtr, char** cpuPtr, int* size) {
  struct ncclProxySharedCollNet* state = &comm->proxyState.progressState.collNet;
  if (state->size == 0) {
    state->size = 2*comm->nChannels*comm->buffSizes[NCCL_PROTO_SIMPLE];
  }

  *size = state->size;

  if (cuda && state->cudaBuff == NULL) {
    NCCLCHECK(ncclCudaCalloc(&state->cudaBuff, *size));
  }
  if (!cuda && state->hostBuff == NULL) {
    NCCLCHECK(ncclCudaHostCalloc(&state->hostBuff, *size));
  }
  *gpuPtr = *cpuPtr = cuda ? state->cudaBuff : state->hostBuff;
  return ncclSuccess;
}

static ncclResult_t sharedBuffersGet(struct ncclComm* comm, int type, int slot, int channel, int* offset) {
  // Use different pools for different channels and also separate send/recv.
  int slotSize = comm->buffSizes[NCCL_PROTO_SIMPLE]/NCCL_STEPS;
  int globalSlot = (type*NCCL_STEPS+slot)*comm->nChannels+channel;
  *offset = slotSize * globalSlot;
  return ncclSuccess;
}

static ncclResult_t sharedBuffersDestroy(struct ncclComm* comm) {
  struct ncclProxySharedCollNet* state = &comm->proxyState.progressState.collNet;
  if (state->size == 0) return ncclSuccess;
  CUDACHECK(cudaFree(state->cudaBuff));
  NCCLCHECK(ncclCudaHostFree(state->hostBuff));
  // This will be called multiple times, with multiple channels and send/recv. Make sure we only do it once.
  state->size = 0;
  return ncclSuccess;
}

static ncclResult_t recvProxySetup(struct ncclProxyConnection* connection, struct ncclComm* comm, void* reqBuff, int reqSize, void* respBuff, int respSize, int* done) {
  struct setupReq* req = (struct setupReq*)reqBuff;
  if (reqSize != sizeof (struct setupReq)) return ncclInternalError;

  struct recvResources* resources;
  NCCLCHECK(ncclCalloc(&resources, 1));
  connection->transportResources = resources;
  connection->shared = 1;

  resources->netDev = req->netDev;
  resources->useGdr = req->useGdr;

  collNetHandle_t* netHandle = (collNetHandle_t*) respBuff;
  if (respSize != sizeof(collNetHandle_t)) return ncclInternalError;

  NCCLCHECK(sharedListen(comm, req->netDev, netHandle));
  return ncclSuccess;
}

static ncclResult_t sendProxyConnect(struct ncclProxyConnection* connection, struct ncclComm* comm, void* reqBuff, int reqSize, void* respBuff, int respSize, int* done) {
  if (reqSize != sizeof(struct collNetConnectArgs)) { WARN("sendProxyConnect: reqSize is %d != %ld\n", reqSize, sizeof(struct collNetConnectArgs)); return ncclInternalError; }
  struct collNetConnectArgs* args = (struct collNetConnectArgs*)reqBuff;
  struct collNetSendConnectInfo* info = (struct collNetSendConnectInfo*)(args->connectInfos+args->rank);

  struct sendResources* resources = (struct sendResources*)(connection->transportResources);

  // Get info from recv side
  resources->collNetRank = args->rank;
  resources->reqFifo = (struct reqSlot (*)[NCCL_STEPS])(info->reqFifo);

  for (int p=0; p<NCCL_NUM_PROTOCOLS; p++)
    resources->recvMhandles[p] = info->mhandles[p];

  NCCLCHECK(sharedConnect(comm, resources->netDev, args->connectInfos, args->nranks, args->rank, &resources->collNetComm));

  // Collnet connect is allowed to fail. Gracefully handle that case by returning NULL to the caller.
  if (respSize != sizeof(struct connectMap*)) { WARN("sendProxyConnect: respSize is %d != %ld\n", respSize, sizeof(void*)); return ncclInternalError; }
  if (resources->collNetComm == NULL) {
    *((struct connectMap**)respBuff) = NULL;
    return ncclSuccess;
  }
  connection->proxyAppendPtr = comm->proxyState.progressState.collNet.proxyAppend+2*resources->netDev;

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
  resources->sendMem->head = -NCCL_STEPS;

  // Allocate & Register shared buffers for the Simple protocol
  int bank = resources->useGdr ? NCCL_NET_MAP_SHARED_DEVMEM : NCCL_NET_MAP_SHARED_HOSTMEM;
  struct connectMapMem* mapMem = map->mems+bank;
  NCCLCHECK(sharedBuffersInit(comm, resources->useGdr, &mapMem->gpuPtr, &mapMem->cpuPtr, &mapMem->size));
  NCCL_NET_MAP_ADD_POINTER(map, 1, resources->useGdr, mapMem->size, buffs[NCCL_PROTO_SIMPLE]);

  NCCLCHECK(collNetRegMr(resources->collNetComm, mapMem->cpuPtr, mapMem->size,
        resources->useGdr ? NCCL_PTR_CUDA : NCCL_PTR_HOST,
        &resources->sendMhandles[NCCL_PROTO_SIMPLE]));

  *((struct connectMap**)respBuff) = &resources->map;
  return ncclSuccess;
}

static ncclResult_t recvProxyConnect(struct ncclProxyConnection* connection, struct ncclComm* comm, void* reqBuff, int reqSize, void* respBuff, int respSize, int* done) {
  if (reqSize != sizeof(struct collNetConnectArgs)) { WARN("recvProxyConnect: reqSize is %d != %ld\n", reqSize, sizeof(struct collNetConnectArgs)); return ncclInternalError; }
  struct collNetConnectArgs* args = (struct collNetConnectArgs*)reqBuff;

  struct recvResources* resources = (struct recvResources*)(connection->transportResources);
  struct collNetSendConnectInfo* info = (struct collNetSendConnectInfo*)(args->connectInfos+args->rank);
  resources->collNetRank = args->rank;

  NCCLCHECK(sharedConnect(comm, resources->netDev, args->connectInfos, args->nranks, args->rank, &resources->collNetComm));

  // Collnet connect is allowed to fail. Gracefully handle that case by returning NULL to the caller.
  if (respSize != sizeof(struct connectMap*)) { WARN("sendProxyConnect: respSize is %d != %ld\n", respSize, sizeof(void*)); return ncclInternalError; }
  if (resources->collNetComm == NULL) {
    *((struct connectMap**)respBuff) = NULL;
    return ncclSuccess;
  }
  connection->proxyAppendPtr = comm->proxyState.progressState.collNet.proxyAppend+2*resources->netDev+1;

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
  NCCLCHECK(sharedBuffersInit(comm, resources->useGdr, &mapMem->gpuPtr, &mapMem->cpuPtr, &mapMem->size));
  NCCL_NET_MAP_ADD_POINTER(map, 1, resources->useGdr, mapMem->size, buffs[NCCL_PROTO_SIMPLE]);

  NCCLCHECK(collNetRegMr(resources->collNetComm, mapMem->cpuPtr, mapMem->size,
        resources->useGdr ? NCCL_PTR_CUDA : NCCL_PTR_HOST,
        &resources->mhandles[NCCL_PROTO_SIMPLE]));

  // Pass info to send side
  info->reqFifo = resources->reqFifo;
  for (int p=0; p<NCCL_NUM_PROTOCOLS; p++)
    info->mhandles[p] = resources->mhandles[p];

  if (respSize != sizeof(struct connectMap*)) { WARN("recvProxyConnect: respSize is %d != %ld\n", respSize, sizeof(void*)); return ncclInternalError; }
  *((struct connectMap**)respBuff) = &resources->map;
  return ncclSuccess;
}

static ncclResult_t sendProxyFree(struct ncclProxyConnection* connection, struct ncclComm* comm) {
  struct sendResources* resources = (struct sendResources*)(connection->transportResources);
  for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
    if (resources->sendMhandles[p]) {
      NCCLCHECK(collNetDeregMr(resources->collNetComm, resources->sendMhandles[p]));
    }
  }
  struct connectMapMem* mems = resources->map.mems;
  NCCLCHECK(ncclCudaHostFree(mems[NCCL_NET_MAP_HOSTMEM].cpuPtr));
  CUDACHECK(cudaFree(mems[NCCL_NET_MAP_DEVMEM].cpuPtr));
  if (mems[NCCL_NET_MAP_GDCMEM].cpuPtr) NCCLCHECK(ncclGdrCudaFree(resources->gdrDesc));
  NCCLCHECK(sharedBuffersDestroy(comm));
  NCCLCHECK(sharedFree(comm, resources->netDev));
  free(connection->transportResources);
  return ncclSuccess;
}

static ncclResult_t recvProxyFree(struct ncclProxyConnection* connection, struct ncclComm* comm) {
  struct recvResources* resources = (struct recvResources*)(connection->transportResources);
  for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
    if (resources->mhandles[p]) {
      NCCLCHECK(collNetDeregMr(resources->collNetComm, resources->mhandles[p]));
    }
  }
  struct connectMapMem* mems = resources->map.mems;
  NCCLCHECK(ncclCudaHostFree(mems[NCCL_NET_MAP_HOSTMEM].cpuPtr));
  CUDACHECK(cudaFree(mems[NCCL_NET_MAP_DEVMEM].cpuPtr));
  if (mems[NCCL_NET_MAP_GDCMEM].cpuPtr) NCCLCHECK(ncclGdrCudaFree(resources->gdrDesc));
  NCCLCHECK(sharedBuffersDestroy(comm));
  NCCLCHECK(sharedFree(comm, resources->netDev));
  free(connection->transportResources);
  return ncclSuccess;
}


#define LAST_OF_GROUP(s) \
  (s % COLLNET_GROUP_NSUBS == COLLNET_GROUP_NSUBS-1 || s == args->nsubs-1)

static ncclResult_t sendProxyProgress(struct ncclComm* comm, struct ncclProxyArgs* args) {
  if (args->protocol != NCCL_PROTO_SIMPLE) {
    WARN("CollNet does not support LL/LL128");
    return ncclInternalError;
  }
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
    int p = args->protocol;
    int nGroups = DIVUP(args->nsubs, COLLNET_GROUP_NSUBS);
    int perGroupSteps = NCCL_STEPS / nGroups;
    for (int s=0; s<args->nsubs; s++) {
      struct ncclProxySubArgs* sub = args->subs+s;
      struct sendResources* resources = (struct sendResources*) (sub->connection->transportResources);
      void* sendMhandle = resources->sendMhandles[p];
      void* recvMhandle = resources->recvMhandles[p];
      auto reqFifo = resources->reqFifo;
      if (sub->posted < sub->nsteps && sub->posted < sub->done + NCCL_STEPS) {
        int buffSlot = (sub->base+sub->posted)%NCCL_STEPS;
        int sharedBuffSlot = sub->posted%NCCL_STEPS;
        int offset;
        NCCLCHECK(sharedBuffersGet(comm, 0, sharedBuffSlot, 0, &offset));
        resources->recvMem->offsFifo[buffSlot] = offset + s*args->chunkSize;
        __sync_synchronize();
        volatile uint64_t* sendHead = resources->gdcSync ? resources->gdcSync : &resources->sendMem->head;
        sub->posted += args->sliceSteps;
        *sendHead = sub->base + sub->posted - NCCL_STEPS;
        if (resources->gdcSync) wc_store_fence(); // Flush out WC write
      }
      // Enforce sync between operations of the same group.
      bool groupSync = (((s == 0) && ((sub+args->nsubs-1)->received == sub->received)) || (s && (sub-1)->received > sub->received));
      if (groupSync && sub->received < sub->posted && sub->received < sub->done + perGroupSteps) {
        int buffSlot = (sub->base+sub->received)%NCCL_STEPS;
        int sharedBuffSlot = sub->received%NCCL_STEPS;
        volatile int* sizesFifo = resources->recvMem->sizesFifo;
        volatile uint64_t* recvTail = &resources->recvMem->tail;
        char* localBuff = NCCL_NET_MAP_GET_POINTER(&resources->map, gpu, buffs[p]);
        if (sizesFifo[buffSlot] != -1 && ((*recvTail > (sub->base+sub->received)))) {
          // We have something to receive, let's check whether data is ready.
          int ready = 1;
          if (s == 0) {
            int offset;
            NCCLCHECK(sharedBuffersGet(comm, 0, sharedBuffSlot, 0, &offset));
            args->sharedBuff[sharedBuffSlot] = localBuff + offset;
            args->sharedSize[sharedBuffSlot] = args->chunkSize;
          }
          if (ready) {
            sizesFifo[buffSlot] = -1;
            sub->received += args->sliceSteps;
            args->idle = 0;
            //continue;
          }
        }
      }
      if (LAST_OF_GROUP(s) && (sub->transmitted < sub->received)) {
        int group = s / COLLNET_GROUP_NSUBS;
        int buffSlot = (sub->base+sub->transmitted)%NCCL_STEPS;
        int sharedBuffSlot = sub->transmitted%NCCL_STEPS;
        if (reqFifo[group][buffSlot].recvBuff != NULL) {
          int totalSize = (s-group*COLLNET_GROUP_NSUBS+1) * args->sharedSize[sharedBuffSlot];
          int count = totalSize / ncclTypeSize(args->dtype);
          reqFifo[group][buffSlot].size = args->sharedSize[sharedBuffSlot];
          char* sendAddress = (char*)args->sharedBuff[sharedBuffSlot] + group*COLLNET_GROUP_NSUBS*args->sharedSize[sharedBuffSlot];
          NCCLCHECK(collNetIallreduce(resources->collNetComm, sendAddress, (void*)(reqFifo[group][buffSlot].recvBuff), count, args->dtype, args->redOp, sendMhandle, recvMhandle, sub->requests+buffSlot));
          if (sub->requests[buffSlot] == NULL) continue;

          TRACE(NCCL_NET, "sendProxy [%d/%d/%d] Iallreduce posted, size %d req %p", sub->transmitted, group, buffSlot, totalSize, sub->requests[buffSlot]);
          // Make sure size is reset to zero before we update the head.
          __sync_synchronize();
          sub->transmitted += args->sliceSteps;
          args->idle = 0;
          continue;
        }
      }
      // Check whether the network has completed some send operations.
      if (LAST_OF_GROUP(s) && sub->done < sub->transmitted) {
        int done, size;
        int group = s / COLLNET_GROUP_NSUBS;
        int buffSlot = (sub->base+sub->done)%NCCL_STEPS;
        NCCLCHECK(collNetTest((void*)(sub->requests[buffSlot]), &done, &size));
        if (done) {
          TRACE(NCCL_NET, "sendProxy [%d/%d/%d] request %p done, size %d", sub->done, group, buffSlot, sub->requests[buffSlot], size);
          // Make sure size is updated before we set recvBuff to NULL (from the view of recv proxy, concerning the flush)
          // (reordered store after store is possible on POWER, though not on x86)
          __sync_synchronize();
          reqFifo[group][buffSlot].recvBuff = NULL; // Notify recvProxy
          for (int i=group*COLLNET_GROUP_NSUBS; i<=s; i++) args->subs[i].done += args->sliceSteps;
          args->idle = 0;
          int allDone = 1;
          for (int i=0; i<args->nsubs; i++) {
            if (args->subs[i].done < args->subs[i].nsteps) { allDone = 0; break; }
          }
          if (allDone) {
            args->state = ncclProxyOpNone;
            TRACE(NCCL_NET, "sendProxy [%d/%d] stopped", sub->done, s);
          }
        }
      }
    }
  }
  return ncclSuccess;
}

static ncclResult_t recvProxyProgress(struct ncclComm* comm, struct ncclProxyArgs* args) {
  if (args->protocol != NCCL_PROTO_SIMPLE) {
    WARN("CollNet does not support LL/LL128");
    return ncclInternalError;
  }
  if (args->state == ncclProxyOpReady) {
    for (int s=0; s<args->nsubs; s++) {
      struct ncclProxySubArgs* sub = args->subs+s;
      struct recvResources* resources = (struct recvResources*) (sub->connection->transportResources);
      // Round to next multiple of sliceSteps
      sub->base = ROUNDUP(resources->step, args->chunkSteps);
      sub->posted = sub->received = sub->flushed = sub->transmitted = sub->done = 0;
      resources->step = sub->base + sub->nsteps;
    }
    args->state = ncclProxyOpProgress;
  }
  args->idle = 1;
  if (args->state == ncclProxyOpProgress) {
    int p = args->protocol;
    int nGroups = DIVUP(args->nsubs, COLLNET_GROUP_NSUBS);
    int perGroupSteps = NCCL_STEPS / nGroups;
    for (int s=0; s<args->nsubs; s++) {
      struct ncclProxySubArgs* sub = args->subs+s;
      struct recvResources* resources = (struct recvResources*) (sub->connection->transportResources);
      void* mhandle = resources->mhandles[p];
      auto reqFifo = resources->reqFifo;
      char* localBuff = NCCL_NET_MAP_GET_POINTER(&resources->map, cpu, buffs[p]);

      // Enforce sync between operations of the same group.
      if (LAST_OF_GROUP(s) && (sub->posted < sub->done + perGroupSteps) && (sub->posted < sub->nsteps)) {
        int group = s / COLLNET_GROUP_NSUBS;
        int buffSlot = (sub->base+sub->posted)%NCCL_STEPS;
        int sharedBuffSlot = sub->posted%NCCL_STEPS;
        int startChannel = group*COLLNET_GROUP_NSUBS;
        int offset;
        NCCLCHECK(sharedBuffersGet(comm, 1, sharedBuffSlot, startChannel, &offset));
        reqFifo[group][buffSlot].recvBuff = localBuff + offset;
        TRACE(NCCL_NET, "recvProxy [%d/%d/%d] posted buffer %p", sub->posted, group, buffSlot, reqFifo[group][buffSlot].recvBuff);
        sub->posted += args->sliceSteps;
        args->idle = 0;
        continue;
      }
      if (LAST_OF_GROUP(s) && (sub->posted > sub->received)) {
        int group = s / COLLNET_GROUP_NSUBS;
        int buffSlot = (sub->base+sub->received)%NCCL_STEPS;
        int sharedBuffSlot = sub->received%NCCL_STEPS;
        if (reqFifo[group][buffSlot].recvBuff == NULL) { // Buffer is cleared : coll is complete
          args->sharedSize[sharedBuffSlot] = reqFifo[group][buffSlot].size;
          int totalSize = args->sharedSize[sharedBuffSlot]*(s-group*COLLNET_GROUP_NSUBS+1);
          TRACE(NCCL_NET, "recvProxy [%d/%d/%d] received, size %d", sub->received, group, buffSlot, totalSize);
          sub->received += args->sliceSteps;
          sub->requests[buffSlot] = NULL;
          if (reqFifo[group][buffSlot].size > 0 && resources->useGdr) {
            // GDRCOPY support
            if (resources->gdcFlush) {
#if defined (__x86_64__)
              // Force a PCI-E read from GPU memory
              asm volatile ("mov (%0), %%eax" :: "l"(resources->gdcFlush) : "%eax");
#else
              WARN("NET: GDR Flush only supported on x86_64");
              return ncclInternalError;
#endif
              sub->requests[buffSlot] = NULL;
            } else {
              int startChannel = group*COLLNET_GROUP_NSUBS;
              int offset;
              NCCLCHECK(sharedBuffersGet(comm, 1, sharedBuffSlot, startChannel, &offset));
              NCCLCHECK(collNetIflush(resources->collNetComm, localBuff + offset, totalSize, mhandle, sub->requests+buffSlot));
            }
          } else {
            for (int i=group*COLLNET_GROUP_NSUBS; i<=s; i++) args->subs[i].flushed += args->sliceSteps;
          }
          args->idle = 0;
          continue;
        }
      }
      if (LAST_OF_GROUP(s) && (sub->received > sub->flushed)) {
        // Progress flush operations
        int group = s / COLLNET_GROUP_NSUBS;
        int buffSlot = (sub->base + sub->flushed)%NCCL_STEPS;
        int done = 1;
        if (sub->requests[buffSlot]) NCCLCHECK(collNetTest(sub->requests[buffSlot], &done, NULL));
        if (done) {
          TRACE(NCCL_NET, "recvProxy [%d/%d/%d] flushed", sub->flushed, group, buffSlot);
          for (int i=group*COLLNET_GROUP_NSUBS; i<=s; i++) args->subs[i].flushed += args->sliceSteps;
          args->idle = 0;
          //continue;
        }
      }
      if (sub->flushed > sub->transmitted) {
        int group = s / COLLNET_GROUP_NSUBS;
        int buffSlot = (sub->base + sub->transmitted)%NCCL_STEPS;
        int sharedBuffSlot = sub->transmitted%NCCL_STEPS;
        int startChannel = group*COLLNET_GROUP_NSUBS;
        int offset;
        NCCLCHECK(sharedBuffersGet(comm, 1, sharedBuffSlot, startChannel, &offset));
        volatile int* offsFifo = (volatile int*)resources->recvMem->offsFifo;
        offsFifo[buffSlot] = offset + (s%COLLNET_GROUP_NSUBS)*args->chunkSize;
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
      bool groupSync = (((s == 0) && ((sub+args->nsubs-1)->done == sub->done)) || (s && (sub-1)->done > sub->done));
      volatile uint64_t* sendHead = &resources->sendMem->head;
      if (groupSync && sub->done < sub->transmitted && (sub->base+sub->done) < *sendHead) {
        sub->done += args->sliceSteps;
        args->idle = 0;
        if (sub->done == sub->nsteps && s == args->nsubs-1) {
          args->state = ncclProxyOpNone;
          TRACE(NCCL_NET, "recvProxy [%d/%d] stopped", sub->done, s);
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
