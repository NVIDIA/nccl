/*************************************************************************
 * Copyright (c) 2016-2021, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "comm.h"
#include "coll_net.h"
#include "graph.h"

#define COLLNET_GROUP_NSUBS 8
#define COLLNET_MAX_GROUPS (NCCL_PROXY_MAX_SUBS/COLLNET_GROUP_NSUBS)

struct collNetRecvConnectInfo {
  collNetHandle_t collNetHandle;
};

struct collNetSendConnectInfo {
  void* mhandles[NCCL_NUM_PROTOCOLS];
  void* reqFifo;
};

struct reqSlot {
  volatile void* recvBuff;
  volatile int size;
};

struct collNetSendResources {
  struct ncclComm* comm;
  void* collNetComm;
  struct ncclSendMem* sendMem;
  struct ncclRecvMem* recvMem;
  int netDev;
  int useGdr;
  void* sendMhandles[NCCL_NUM_PROTOCOLS];
  void* recvMhandles[NCCL_NUM_PROTOCOLS];
  struct ncclRecvMem* devRecvMem;
  uint64_t step;
  uint64_t llLastCleaning;
  struct reqSlot (*reqFifo)[NCCL_STEPS];
  int collNetRank;
};

struct collNetRecvResources {
  struct ncclComm* comm;
  void* collNetComm;
  struct ncclSendMem* sendMem;
  struct ncclRecvMem* recvMem;
  int netDev;
  int useGdr;
  void* mhandles[NCCL_NUM_PROTOCOLS];
  struct ncclRecvMem* devRecvMem;
  uint64_t step;
  uint64_t llLastCleaning;
  struct reqSlot reqFifo[COLLNET_MAX_GROUPS][NCCL_STEPS];
  int collNetRank;
};

struct collNetSharedResources {
  void* collNetListenComms[MAXCHANNELS];
  void* collNetComms[MAXCHANNELS];
  int collNetCommRefCount[MAXCHANNELS];
};

/* Determine if we can communicate with the peer */
ncclResult_t collNetCanConnect(int* ret, struct ncclTopoSystem* topo, struct ncclTopoGraph* graph, struct ncclPeerInfo* info1, struct ncclPeerInfo* info2) {
  *ret = 1;
  return ncclSuccess;
}

ncclResult_t collNetSharedListen(struct ncclComm* comm, int netDev, void* collNetHandle) {
  struct collNetSharedResources* resources = (struct collNetSharedResources*)comm->proxyState.sharedBuffs.collNetResources;
  if (resources == NULL) {
    NCCLCHECK(ncclCalloc(&resources, 1));
    comm->proxyState.sharedBuffs.collNetResources = resources;
  }
  if (resources->collNetComms[netDev] == NULL)
    NCCLCHECK(collNetListen(netDev, collNetHandle, resources->collNetListenComms+netDev));
  return ncclSuccess;
}

/* Setup send connector, and return connect information for others in the coll communicator to connect to me */
ncclResult_t collNetSendSetup(struct ncclComm* comm, struct ncclTopoGraph* graph, struct ncclPeerInfo* myInfo, struct ncclPeerInfo* peerInfo, struct ncclConnect* connectInfo, struct ncclConnector* send, int channelId, int connIndex) {
  struct collNetSendResources* resources;
  NCCLCHECK(ncclCalloc(&resources, 1));
  send->transportResources = resources;
  send->conn.shared = 1;
  resources->comm = comm;

  NCCLCHECK(ncclTopoGetNetDev(comm->topo, myInfo->rank, graph, channelId, 0, &resources->netDev));
  NCCLCHECK(ncclTopoCheckGdr(comm->topo, myInfo->busId, resources->netDev, 1, &resources->useGdr));

  send->proxyAppendPtr = comm->proxyState.sharedBuffs.proxyAppendCollNet+2*resources->netDev+1;

  NCCLCHECK(ncclCudaHostCalloc(&resources->sendMem, 1));

  int recvSize = offsetof(struct ncclRecvMem, buff);
  // Simple uses shared buffers and we don't support LL128
  recvSize += send->comm->buffSizes[NCCL_PROTO_LL];

  if (resources->useGdr) {
    NCCLCHECK(ncclCudaCalloc((char**)(&resources->devRecvMem), recvSize));
  }
  NCCLCHECK(ncclCudaHostCalloc((char**)&resources->recvMem, recvSize));

  INFO(NCCL_INIT|NCCL_NET,"CollNet %02d : %d [send] via COLLNET/%s/%d%s", channelId, myInfo->rank, collNetName(), resources->netDev,
      resources->useGdr ? "/GDRDMA" : "");
  return ncclSuccess;
}

/* Setup recv connector */
ncclResult_t collNetRecvSetup(struct ncclComm* comm, struct ncclTopoGraph* graph, struct ncclPeerInfo* myInfo, struct ncclPeerInfo* peerInfo, struct ncclConnect* connectInfo, struct ncclConnector* recv, int channelId, int connIndex) {
  struct collNetRecvResources* resources;
  NCCLCHECK(ncclCalloc(&resources, 1));
  recv->transportResources = resources;
  recv->conn.shared = 1;
  resources->comm = comm;

  NCCLCHECK(ncclTopoGetNetDev(comm->topo, myInfo->rank, graph, channelId, 0, &resources->netDev));
  NCCLCHECK(ncclTopoCheckGdr(comm->topo, myInfo->busId, resources->netDev, 0, &resources->useGdr));

  recv->proxyAppendPtr = comm->proxyState.sharedBuffs.proxyAppendCollNet+2*resources->netDev;

  NCCLCHECK(ncclCudaHostCalloc(&resources->sendMem, 1));

  int recvSize = offsetof(struct ncclRecvMem, buff);
  // Simple uses shared buffers and we don't support LL128
  recvSize += recv->comm->buffSizes[NCCL_PROTO_LL];

  if (resources->useGdr) {
    NCCLCHECK(ncclCudaCalloc((char**)(&resources->devRecvMem), recvSize));
  }
  NCCLCHECK(ncclCudaHostCalloc((char**)&resources->recvMem, recvSize));

  INFO(NCCL_INIT|NCCL_NET,"CollNet %02d : %d [receive] via COLLNET/%s/%d%s", channelId, myInfo->rank, collNetName(), resources->netDev,
      resources->useGdr ? "/GDRDMA" : "");
  struct collNetRecvConnectInfo* info = (struct collNetRecvConnectInfo*) connectInfo;

  NCCLCHECK(collNetSharedListen(comm, resources->netDev, &info->collNetHandle));
  return ncclSuccess;
}

ncclResult_t collNetSharedConnect(struct ncclComm* comm, int netDev, struct ncclConnect* connectInfos, int nranks, int rank, void** collNetComm) {
  struct collNetSharedResources* resources = (struct collNetSharedResources*)comm->proxyState.sharedBuffs.collNetResources;
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
    NCCLCHECK(ret);
    // Close listen comm
    NCCLCHECK(collNetCloseListen(resources->collNetListenComms[netDev]));
  }
  *collNetComm = resources->collNetComms[netDev];
  resources->collNetCommRefCount[netDev]++;
  return ncclSuccess;
}

ncclResult_t collNetSendConnect(struct ncclComm* comm, struct ncclConnect* connectInfos, int nranks, int rank, struct ncclConnector* send) {
  // Setup device pointers
  struct collNetSendResources* resources = (struct collNetSendResources*)send->transportResources;
  struct collNetSendConnectInfo* info = (struct collNetSendConnectInfo*)(connectInfos+rank);

  // Intermediate buffering on GPU for GPU Direct RDMA, but LL buffer is always on host
  send->conn.buffs[NCCL_PROTO_LL] = resources->recvMem->buff;
  send->conn.buffs[NCCL_PROTO_LL128] = send->conn.buffs[NCCL_PROTO_SIMPLE] = NULL;
  send->conn.direct |= resources->useGdr ? NCCL_DIRECT_NIC : 0;

  // Head/Tail/Opcount/Fifos are always on host
  send->conn.tail = &resources->recvMem->tail;
  send->conn.sizesFifo = resources->recvMem->sizesFifo;
  send->conn.ptrsFifo = resources->recvMem->ptrsFifo;
  send->conn.head = &resources->sendMem->head;
  resources->sendMem->head = -NCCL_STEPS; // Don't give any credit yet when sharing buffers
  for (int i=0; i<NCCL_STEPS; i++) send->conn.sizesFifo[i] = -1;

  // Get info from recv side
  resources->collNetRank = rank;
  resources->reqFifo = (struct reqSlot (*)[NCCL_STEPS])(info->reqFifo);

  for (int p=0; p<NCCL_NUM_PROTOCOLS; p++)
    resources->recvMhandles[p] = info->mhandles[p];

  NCCLCHECK(collNetSharedConnect(comm, resources->netDev, connectInfos, nranks, rank, &resources->collNetComm));

  int size;
  char* ptr;
  // Allocate & Register shared buffers for the Simple protocol
  NCCLCHECK(ncclProxySharedBuffersInit(send->comm, resources->useGdr, &size, &ptr));
  NCCLCHECK(collNetRegMr(resources->collNetComm, ptr, size,
        resources->useGdr ? NCCL_PTR_CUDA : NCCL_PTR_HOST,
        &resources->sendMhandles[NCCL_PROTO_SIMPLE]));

  // Allocate & Register shared buffers for the LL protocol
  NCCLCHECK(ncclProxySharedBuffersInit(send->comm, 0, &size, &ptr));
  NCCLCHECK(collNetRegMr(resources->collNetComm, ptr, size,
        NCCL_PTR_HOST,
        &resources->sendMhandles[NCCL_PROTO_LL]));
  return ncclSuccess;
}

ncclResult_t collNetRecvConnect(struct ncclComm* comm, struct ncclConnect* connectInfos, int nranks, int rank, struct ncclConnector* recv) {
  // Setup device pointers
  struct collNetRecvResources* resources = (struct collNetRecvResources*)recv->transportResources;
  struct collNetSendConnectInfo* info = (struct collNetSendConnectInfo*)(connectInfos+rank);
  resources->collNetRank = rank;

  // Intermediate buffering on GPU for GPU Direct RDMA
  struct ncclRecvMem* recvMem = resources->useGdr ? resources->devRecvMem : resources->recvMem;
  int offset = 0;
  for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
    recv->conn.buffs[p] = (p == NCCL_PROTO_LL ? resources->recvMem->buff : recvMem->buff) + offset;
    offset += recv->comm->buffSizes[p];
  }
  recv->conn.direct |= resources->useGdr ? NCCL_DIRECT_NIC : 0;

  // Head/Tail/Opcount are always on host
  recv->conn.tail = &resources->recvMem->tail;
  recv->conn.ptrsFifo = resources->recvMem->ptrsFifo;
  recv->conn.head = &resources->sendMem->head;

  NCCLCHECK(collNetSharedConnect(comm, resources->netDev, connectInfos, nranks, rank, &resources->collNetComm));

  int size;
  char* ptr;

  // Allocate & Register shared buffers for the Simple protocol
  NCCLCHECK(ncclProxySharedBuffersInit(recv->comm, resources->useGdr, &size, &ptr));
  NCCLCHECK(collNetRegMr(resources->collNetComm, ptr, size,
        resources->useGdr ? NCCL_PTR_CUDA : NCCL_PTR_HOST,
        &resources->mhandles[NCCL_PROTO_SIMPLE]));

  // Allocate & Register shared buffers for the LL protocol
  NCCLCHECK(ncclProxySharedBuffersInit(recv->comm, 0, &size, &ptr));
  NCCLCHECK(collNetRegMr(resources->collNetComm, ptr, size,
        NCCL_PTR_HOST,
        &resources->mhandles[NCCL_PROTO_LL]));

  // Pass info to send side
  info->reqFifo = resources->reqFifo;
  for (int p=0; p<NCCL_NUM_PROTOCOLS; p++)
    info->mhandles[p] = resources->mhandles[p];

  return ncclSuccess;
}

ncclResult_t collNetSharedFree(struct ncclComm* comm, int netDev) {
  struct collNetSharedResources* resources = (struct collNetSharedResources*)comm->proxyState.sharedBuffs.collNetResources;
  resources->collNetCommRefCount[netDev]--;
  if (resources->collNetCommRefCount[netDev] == 0) {
    NCCLCHECK(collNetCloseColl(resources->collNetComms[netDev]));
  }
  for (int c=0; c<MAXCHANNELS; c++) if (resources->collNetCommRefCount[c]) return ncclSuccess;
  comm->proxyState.sharedBuffs.collNetResources = NULL;
  free(resources);
  return ncclSuccess;
}

ncclResult_t collNetSendFree(void* sendTransportResources) {
  struct collNetSendResources* resources = (struct collNetSendResources*)sendTransportResources;
  NCCLCHECK(ncclCudaHostFree(resources->sendMem));
  NCCLCHECK(ncclCudaHostFree(resources->recvMem));
  if (resources->collNetComm) {
    NCCLCHECK(collNetDeregMr(resources->collNetComm, resources->sendMhandles[NCCL_PROTO_LL]));
    NCCLCHECK(collNetDeregMr(resources->collNetComm, resources->sendMhandles[NCCL_PROTO_SIMPLE]));
  }
  if (resources->useGdr) CUDACHECK(cudaFree(resources->devRecvMem));

  NCCLCHECK(collNetSharedFree(resources->comm, resources->netDev));
  free(resources);
  return ncclSuccess;
}

ncclResult_t collNetRecvFree(void* recvTransportResources) {
  struct collNetRecvResources* resources = (struct collNetRecvResources*)recvTransportResources;
  NCCLCHECK(ncclCudaHostFree(resources->sendMem));
  NCCLCHECK(ncclCudaHostFree(resources->recvMem));
  if (resources->collNetComm) {
    NCCLCHECK(collNetDeregMr(resources->collNetComm, resources->mhandles[NCCL_PROTO_LL]));
    NCCLCHECK(collNetDeregMr(resources->collNetComm, resources->mhandles[NCCL_PROTO_SIMPLE]));
  }
  if (resources->useGdr) CUDACHECK(cudaFree(resources->devRecvMem));

  NCCLCHECK(collNetSharedFree(resources->comm, resources->netDev));
  free(resources);
  return ncclSuccess;
}

#define LAST_OF_GROUP(s) \
  (s % COLLNET_GROUP_NSUBS == COLLNET_GROUP_NSUBS-1 || s == args->nsubs-1)

ncclResult_t collNetSendProxy(struct ncclProxyArgs* args) {
  if (args->protocol == NCCL_PROTO_LL128) {
    WARN("CollNet does not support LL128");
    return ncclInternalError;
  }
  if (args->state == ncclProxyOpReady) {
    for (int s=0; s<args->nsubs; s++) {
      struct ncclProxySubArgs* sub = args->subs+s;
      struct collNetSendResources* resources = (struct collNetSendResources*) (sub->connector->transportResources);
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
      struct collNetSendResources* resources = (struct collNetSendResources*) (sub->connector->transportResources);
      void* sendMhandle = resources->sendMhandles[p];
      void* recvMhandle = resources->recvMhandles[p];
      int stepSize = sub->connector->comm->buffSizes[p] / NCCL_STEPS;
      auto reqFifo = resources->reqFifo;
      if (sub->posted < sub->nsteps && sub->posted < sub->done + NCCL_STEPS) {
        int buffSlot = (sub->base+sub->posted)%NCCL_STEPS;
        if (p == NCCL_PROTO_SIMPLE) {
          char* ptr;
          int sharedBuffSlot = sub->posted%NCCL_STEPS;
          NCCLCHECK(ncclProxySharedBuffersGetCollNet(sub->connector->comm, resources->useGdr, 0, sharedBuffSlot, 0, &ptr));
          resources->recvMem->ptrsFifo[buffSlot] = ptr + s*args->chunkSize;
          __sync_synchronize();
        }
        volatile uint64_t* sendHead = &resources->sendMem->head;
        sub->posted += args->sliceSteps;
        *sendHead = sub->base + sub->posted - NCCL_STEPS;
      }
      // Enforce sync between operations of the same group.
      bool groupSync = (((s == 0) && ((sub+args->nsubs-1)->received == sub->received)) || (s && (sub-1)->received > sub->received));
      if (groupSync && sub->received < sub->posted && sub->received < sub->done + perGroupSteps) {
        int buffSlot = (sub->base+sub->received)%NCCL_STEPS;
        int sharedBuffSlot = sub->received%NCCL_STEPS;
        volatile int* sizesFifo = resources->recvMem->sizesFifo;
        volatile uint64_t* recvTail = &resources->recvMem->tail;
        if (sizesFifo[buffSlot] != -1 && ((*recvTail > (sub->base+sub->received)) || p == NCCL_PROTO_LL)) {
          // We have something to receive, let's check whether data is ready.
          int size = sizesFifo[buffSlot];
          int ready = 1;
          if (s == 0) {
            NCCLCHECK(ncclProxySharedBuffersGetCollNet(sub->connector->comm, p == NCCL_PROTO_SIMPLE ? resources->useGdr : 0, 0, sharedBuffSlot, 0, &args->sharedBuff[sharedBuffSlot]));
            args->sharedSize[sharedBuffSlot] = p == NCCL_PROTO_SIMPLE ? args->chunkSize : size/2;
          }
          if (p == NCCL_PROTO_LL) {
            char* localBuff = sub->connector->conn.buffs[p];
            uint32_t flag = NCCL_LL_FLAG(sub->base + sub->received + 1);
            int nFifoLines = size / sizeof(union ncclLLFifoLine);
            union ncclLLFifoLine* lines = (union ncclLLFifoLine*)(localBuff+buffSlot*stepSize);
            // Pack data into the shared buffer
            uint32_t* sendBuff = (uint32_t*)(args->sharedBuff[sharedBuffSlot]+args->sharedSize[sharedBuffSlot]*s);
            for (int i=0; i<nFifoLines; i++) {
              volatile uint32_t *f1 = &lines[i].flag1;
              volatile uint32_t *d1 = &lines[i].data1;
              volatile uint32_t *f2 = &lines[i].flag2;
              volatile uint32_t *d2 = &lines[i].data2;
              if (f1[0] != flag || f2[0] != flag) { ready = 0; break; }
              sendBuff[2*i] = d1[0];
              sendBuff[2*i+1] = d2[0];
            }
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

ncclResult_t collNetRecvProxy(struct ncclProxyArgs* args) {
  if (args->protocol == NCCL_PROTO_LL128) {
    WARN("CollNet does not support LL128");
    return ncclInternalError;
  }
  if (args->state == ncclProxyOpReady) {
    for (int s=0; s<args->nsubs; s++) {
      struct ncclProxySubArgs* sub = args->subs+s;
      struct collNetRecvResources* resources = (struct collNetRecvResources*) (sub->connector->transportResources);
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
      struct collNetRecvResources* resources = (struct collNetRecvResources*) (sub->connector->transportResources);
      void* mhandle = resources->mhandles[p];
      int stepSize = sub->connector->comm->buffSizes[p] / NCCL_STEPS;
      auto reqFifo = resources->reqFifo;
      // Enforce sync between operations of the same group.
      if (LAST_OF_GROUP(s) && (sub->posted < sub->done + perGroupSteps) && (sub->posted < sub->nsteps)) {
        int group = s / COLLNET_GROUP_NSUBS;
        int buffSlot = (sub->base+sub->posted)%NCCL_STEPS;
        char* ptr;
        int sharedBuffSlot = sub->posted%NCCL_STEPS;
        int startChannel = group*COLLNET_GROUP_NSUBS;
        NCCLCHECK(ncclProxySharedBuffersGetCollNet(sub->connector->comm, p == NCCL_PROTO_SIMPLE ? resources->useGdr : 0, 1, sharedBuffSlot, startChannel, &ptr));
        reqFifo[group][buffSlot].recvBuff = ptr;
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
          if (reqFifo[group][buffSlot].size > 0 && p == NCCL_PROTO_SIMPLE && resources->useGdr) {
            int startChannel = group*COLLNET_GROUP_NSUBS;
            char* groupRecvAddress;
            NCCLCHECK(ncclProxySharedBuffersGetCollNet(sub->connector->comm, 1, 1, sharedBuffSlot, startChannel, &groupRecvAddress));
            NCCLCHECK(collNetIflush(resources->collNetComm, groupRecvAddress, totalSize, mhandle, sub->requests+buffSlot));
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
        char* groupRecvAddress;
        NCCLCHECK(ncclProxySharedBuffersGetCollNet(sub->connector->comm, 1, 1, sharedBuffSlot, startChannel, &groupRecvAddress));
        char* ptr = groupRecvAddress + (s%COLLNET_GROUP_NSUBS)*args->sharedSize[sharedBuffSlot];
        if (p == NCCL_PROTO_SIMPLE) {
          volatile void** ptrsFifo = (volatile void**)resources->recvMem->ptrsFifo;
          ptrsFifo[buffSlot] = ptr;
          __sync_synchronize();
          resources->recvMem->tail = sub->base + sub->flushed;
        }
        if (p == NCCL_PROTO_LL) { // ll
          // re-attach flag
          char* localBuff = sub->connector->conn.buffs[p];
          uint32_t flag = NCCL_LL_FLAG(sub->base + sub->transmitted + 1);
          union ncclLLFifoLine* lines = (union ncclLLFifoLine*)(localBuff+buffSlot*stepSize);
          uint32_t* recvData = (uint32_t*)ptr;
          int nFifoLines = DIVUP(args->sharedSize[sharedBuffSlot], 2*sizeof(uint32_t));
          for (int i=0; i<nFifoLines; i++) {
            lines[i].v[0] = ((uint64_t)flag << 32) + recvData[2*i];
            lines[i].v[1] = ((uint64_t)flag << 32) + recvData[2*i+1];
          }
        }
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
  collNetCanConnect,
  { collNetSendSetup, collNetSendConnect, collNetSendFree, collNetSendProxy },
  { collNetRecvSetup, collNetRecvConnect, collNetRecvFree, collNetRecvProxy }
};
