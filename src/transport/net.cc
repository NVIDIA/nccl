/*************************************************************************
 * Copyright (c) 2016-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "comm.h"
#include "net.h"
#include "graph.h"
#include "collectives.h"

struct netConnectInfo {
  ncclNetHandle_t netHandle;
};

#define LOC_HOSTMEM 0
#define LOC_DEVMEM  1
#define LOC_COUNT   2

struct netSendResources {
  void* netSendComm;
  struct ncclSendMem* sendMem;
  struct ncclRecvMem* recvMem;
  int netDev;
  int useGdr;
  int shared;
  char* buffers[LOC_COUNT];
  int buffSizes[LOC_COUNT];
  void* mhandles[LOC_COUNT];
  void** mhandlesProto[NCCL_NUM_PROTOCOLS];
  uint64_t step;
  uint64_t llLastCleaning;
};

struct netRecvResources {
  void* netListenComm;
  void* netRecvComm;
  struct ncclSendMem* sendMem;
  struct ncclRecvMem* recvMem;
  int netDev;
  int useGdr;
  int shared;
  char* buffers[LOC_COUNT];
  int buffSizes[LOC_COUNT];
  void* mhandles[LOC_COUNT];
  void** mhandlesProto[NCCL_NUM_PROTOCOLS];
  uint64_t step;
  uint64_t llLastCleaning;
};

/* Determine if two peers can communicate with NET */
ncclResult_t netCanConnect(int* ret, struct ncclTopoSystem* topo, struct ncclTopoGraph* graph, struct ncclPeerInfo* info1, struct ncclPeerInfo* info2) {
  *ret = 1;
  return ncclSuccess;
}

NCCL_PARAM(NetSharedBuffers, "NET_SHARED_BUFFERS", -2);

/* Determine if we will use this transport for this peer and return connect
 * information for this peer */
ncclResult_t netSendSetup(struct ncclComm* comm, struct ncclTopoGraph* graph, struct ncclPeerInfo* myInfo, struct ncclPeerInfo* peerInfo, struct ncclConnect* connectInfo, struct ncclConnector* send, int channelId) {
  struct netSendResources* resources;
  NCCLCHECK(ncclCalloc(&resources, 1));
  send->transportResources = resources;
  send->conn.shared = resources->shared = ncclParamNetSharedBuffers() != -2 ? ncclParamNetSharedBuffers() : graph ? 0 : 1;

  NCCLCHECK(ncclTopoGetNetDev(comm->topo, myInfo->rank, graph, channelId, &resources->netDev));
  NCCLCHECK(ncclTopoCheckGdr(comm->topo, myInfo->busId, resources->netDev, 1, &resources->useGdr));

  NCCLCHECK(ncclCudaHostCalloc(&resources->sendMem, 1));
  NCCLCHECK(ncclCudaHostCalloc(&resources->recvMem, 1));

  send->conn.direct |= resources->useGdr ? NCCL_DIRECT_NIC : 0;
  send->conn.tail = &resources->recvMem->tail;
  send->conn.sizesFifo = resources->recvMem->sizesFifo;
  // Only fuse P2P buffers, continue to allocate dedicated buffers for ring/tree
  send->conn.ptrsFifo = resources->shared ? resources->recvMem->ptrsFifo : NULL;
  send->conn.head = &resources->sendMem->head;
  resources->sendMem->head = resources->shared ? -NCCL_STEPS : 0; // Don't give any credit yet when sharing buffers
  for (int i=0; i<NCCL_STEPS; i++) send->conn.sizesFifo[i] = -1;

  if (resources->shared == 0) {
    int protoLoc[NCCL_NUM_PROTOCOLS];
    for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
      protoLoc[p] = p != NCCL_PROTO_LL && resources->useGdr ? LOC_DEVMEM : LOC_HOSTMEM;
    }
    int buffSizes[NCCL_NUM_PROTOCOLS];
    for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
      buffSizes[p] = send->comm->buffSizes[p];
      resources->buffSizes[protoLoc[p]] += buffSizes[p];
    }

    if (resources->buffSizes[LOC_DEVMEM]) {
      NCCLCHECK(ncclCudaCalloc(resources->buffers+LOC_DEVMEM, resources->buffSizes[LOC_DEVMEM]));
    }
    if (resources->buffSizes[LOC_HOSTMEM]) {
      NCCLCHECK(ncclCudaHostCalloc(resources->buffers+LOC_HOSTMEM, resources->buffSizes[LOC_HOSTMEM]));
    }

    int offsets[LOC_COUNT];
    offsets[LOC_HOSTMEM] = offsets[LOC_DEVMEM] = 0;
    for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
      resources->mhandlesProto[p] = resources->mhandles+protoLoc[p];
      send->conn.buffs[p] = resources->buffers[protoLoc[p]] + offsets[protoLoc[p]];
      offsets[protoLoc[p]] += buffSizes[p];
    }
  }

  INFO(NCCL_INIT|NCCL_NET,"Channel %02d : %d[%lx] -> %d[%lx] [send] via NET/%s/%d%s%s", channelId, myInfo->rank, myInfo->busId, peerInfo->rank, peerInfo->busId, ncclNetName(), resources->netDev,
      resources->useGdr ? "/GDRDMA" : "", resources->shared ? "/Shared" : "");
  return ncclSuccess;
}

ncclResult_t netRecvSetup(struct ncclComm* comm, struct ncclTopoGraph* graph, struct ncclPeerInfo* myInfo, struct ncclPeerInfo* peerInfo, struct ncclConnect* connectInfo, struct ncclConnector* recv, int channelId) {
  struct netRecvResources* resources;
  NCCLCHECK(ncclCalloc(&resources, 1));
  recv->transportResources = resources;
  recv->conn.shared = resources->shared = ncclParamNetSharedBuffers() != -2 ? ncclParamNetSharedBuffers() : graph ? 0 : 1;

  NCCLCHECK(ncclTopoGetNetDev(comm->topo, myInfo->rank, graph, channelId, &resources->netDev));
  NCCLCHECK(ncclTopoCheckGdr(comm->topo, myInfo->busId, resources->netDev, 0, &resources->useGdr));

  NCCLCHECK(ncclCudaHostCalloc(&resources->sendMem, 1));
  NCCLCHECK(ncclCudaHostCalloc(&resources->recvMem, 1));

  recv->conn.direct |= resources->useGdr ? NCCL_DIRECT_NIC : 0;
  recv->conn.tail = &resources->recvMem->tail;
  // Only fuse P2P buffers, continue to allocate dedicated buffers for ring/tree
  recv->conn.ptrsFifo = resources->shared ? resources->recvMem->ptrsFifo : NULL;
  recv->conn.head = &resources->sendMem->head;

  if (resources->shared == 0) { // Only allocate dedicated buffers for ring/tree not for p2p
    int protoLoc[NCCL_NUM_PROTOCOLS];
    for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
      protoLoc[p] = resources->useGdr ? LOC_DEVMEM : LOC_HOSTMEM;
    }

    int buffSizes[NCCL_NUM_PROTOCOLS];
    for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
      buffSizes[p] = recv->comm->buffSizes[p];
      resources->buffSizes[protoLoc[p]] += buffSizes[p];
    }

    if (resources->buffSizes[LOC_DEVMEM]) {
      NCCLCHECK(ncclCudaCalloc(resources->buffers+LOC_DEVMEM, resources->buffSizes[LOC_DEVMEM]));
    }
    if (resources->buffSizes[LOC_HOSTMEM]) {
      NCCLCHECK(ncclCudaHostCalloc(resources->buffers+LOC_HOSTMEM, resources->buffSizes[LOC_HOSTMEM]));
    }

    int offsets[LOC_COUNT];
    offsets[LOC_HOSTMEM] = offsets[LOC_DEVMEM] = 0;
    for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
      resources->mhandlesProto[p] = resources->mhandles+protoLoc[p];
      recv->conn.buffs[p] = resources->buffers[protoLoc[p]] + offsets[protoLoc[p]];
      offsets[protoLoc[p]] += buffSizes[p];
    }
  }

  INFO(NCCL_INIT|NCCL_NET,"Channel %02d : %d[%lx] -> %d[%lx] [receive] via NET/%s/%d%s%s", channelId, peerInfo->rank, peerInfo->busId, myInfo->rank, myInfo->busId, ncclNetName(), resources->netDev,
      resources->useGdr ? "/GDRDMA" : "", resources->shared ? "/Shared" : "");
  struct netConnectInfo* info = (struct netConnectInfo*) connectInfo;
  NCCLCHECK(ncclNetListen(resources->netDev, &info->netHandle, &resources->netListenComm));

  return ncclSuccess;
}

ncclResult_t netSendConnect(struct ncclComm* comm, struct ncclConnect* connectInfo, int nranks, int rank, struct ncclConnector* send) {
  // Setup device pointers
  struct netSendResources* resources = (struct netSendResources*)send->transportResources;
  struct netConnectInfo* info = (struct netConnectInfo*)connectInfo;

  // Connect to remote peer
  NCCLCHECK(ncclNetConnect(resources->netDev, info->netHandle, &resources->netSendComm));

  if (resources->shared) {
    // Get shared buffers
    int loc = resources->useGdr ? LOC_DEVMEM : LOC_HOSTMEM;
    NCCLCHECK(ncclProxySharedBuffersInit(send->comm, resources->useGdr, resources->buffSizes+loc, resources->buffers+loc));
    resources->mhandlesProto[NCCL_PROTO_SIMPLE] = resources->mhandles+loc;
  }

  if (resources->buffSizes[LOC_DEVMEM]) {
    NCCLCHECK(ncclNetRegMr(resources->netSendComm, resources->buffers[LOC_DEVMEM], resources->buffSizes[LOC_DEVMEM], NCCL_PTR_CUDA, &resources->mhandles[LOC_DEVMEM]));
  }
  if (resources->buffSizes[LOC_HOSTMEM]) {
    NCCLCHECK(ncclNetRegMr(resources->netSendComm, resources->buffers[LOC_HOSTMEM], resources->buffSizes[LOC_HOSTMEM], NCCL_PTR_HOST, &resources->mhandles[LOC_HOSTMEM]));
  }
  return ncclSuccess;
}

/* Connect to this peer */
ncclResult_t netRecvConnect(struct ncclComm* comm, struct ncclConnect* connectInfo, int nranks, int rank, struct ncclConnector* recv) {
  // Setup device pointers
  struct netRecvResources* resources = (struct netRecvResources*)recv->transportResources;

  // Finish connection establishment from remote peer
  NCCLCHECK(ncclNetAccept(resources->netListenComm, &resources->netRecvComm));
  NCCLCHECK(ncclNetCloseListen(resources->netListenComm));

  if (resources->shared) {
    // Get shared buffers
    int loc = resources->useGdr ? LOC_DEVMEM : LOC_HOSTMEM;
    NCCLCHECK(ncclProxySharedBuffersInit(recv->comm, resources->useGdr, resources->buffSizes+loc, resources->buffers+loc));
    resources->mhandlesProto[NCCL_PROTO_SIMPLE] = resources->mhandles+loc;
  }

  if (resources->buffSizes[LOC_DEVMEM]) {
    NCCLCHECK(ncclNetRegMr(resources->netRecvComm, resources->buffers[LOC_DEVMEM], resources->buffSizes[LOC_DEVMEM], NCCL_PTR_CUDA, &resources->mhandles[LOC_DEVMEM]));
  }
  if (resources->buffSizes[LOC_HOSTMEM]) {
    NCCLCHECK(ncclNetRegMr(resources->netRecvComm, resources->buffers[LOC_HOSTMEM], resources->buffSizes[LOC_HOSTMEM], NCCL_PTR_HOST, &resources->mhandles[LOC_HOSTMEM]));
  }
  return ncclSuccess;
}

ncclResult_t netSendFree(void* transportResources) {
  struct netSendResources* resources = (struct netSendResources*)transportResources;
  NCCLCHECK(ncclCudaHostFree(resources->sendMem));
  NCCLCHECK(ncclCudaHostFree(resources->recvMem));
  for (int l=0; l<LOC_COUNT; l++) {
    if (resources->buffers[l])
      NCCLCHECK(ncclNetDeregMr(resources->netSendComm, resources->mhandles[l]));
  }
  if (resources->shared == 0) {
    NCCLCHECK(ncclCudaHostFree(resources->buffers[LOC_HOSTMEM]));
    CUDACHECK(cudaFree(resources->buffers[LOC_DEVMEM]));
  }
  NCCLCHECK(ncclNetCloseSend(resources->netSendComm));
  free(resources);
  return ncclSuccess;
}

ncclResult_t netRecvFree(void* transportResources) {
  struct netRecvResources* resources = (struct netRecvResources*)transportResources;
  NCCLCHECK(ncclCudaHostFree(resources->sendMem));
  NCCLCHECK(ncclCudaHostFree(resources->recvMem));
  for (int l=0; l<LOC_COUNT; l++) {
    if (resources->buffers[l])
      NCCLCHECK(ncclNetDeregMr(resources->netRecvComm, resources->mhandles[l]));
  }
  if (resources->shared == 0) {
    NCCLCHECK(ncclCudaHostFree(resources->buffers[LOC_HOSTMEM]));
    CUDACHECK(cudaFree(resources->buffers[LOC_DEVMEM]));
  }
  NCCLCHECK(ncclNetCloseRecv(resources->netRecvComm));
  free(resources);
  return ncclSuccess;
}

static_assert(NCCL_STEPS <= NCCL_NET_MAX_REQUESTS, "Not enough net requests to cover for steps");

ncclResult_t netSendProxy(struct ncclProxyArgs* args) {
  struct netSendResources* resources = (struct netSendResources*) (args->connector->transportResources);
  if (args->state == ncclProxyOpReady) {
    // Round to next multiple of sliceSteps
    resources->step = ROUNDUP(resources->step, args->chunkSteps);
    args->posted = args->transmitted = args->done = resources->step;
    args->end = resources->step + args->nsteps;
    args->state = ncclProxyOpProgress;
  }
  args->idle = 1;
  if (args->state == ncclProxyOpProgress) {
    int p = args->protocol;
    int stepSize = args->connector->comm->buffSizes[p] / NCCL_STEPS;
    char* localBuff = args->connector->conn.buffs[p];
    void* mhandle = *(resources->mhandlesProto[p]);
    int buffSize = stepSize*args->sliceSteps;
    if (resources->shared) buffSize /= SENDRECV_SLICEFACTOR;
    if (args->sendbytes < buffSize) buffSize = args->sendbytes;
    // Post buffers to the GPU
    if (args->posted < args->end && args->posted < args->done + NCCL_STEPS) {
      if (resources->shared) {
        char* ptr;
        NCCLCHECK(ncclProxySharedBuffersAlloc(args->connector->comm, resources->useGdr, 0, args->channel->id, buffSize, &ptr));
        if (ptr == NULL) return ncclInternalError;
        resources->recvMem->ptrsFifo[args->posted%NCCL_STEPS] = ptr;
        __sync_synchronize();
        volatile uint64_t* sendHead = &resources->sendMem->head;
        args->posted += args->sliceSteps;
        *sendHead = args->posted - NCCL_STEPS;
      } else args->posted += args->sliceSteps;
      args->idle = 0;
      return ncclSuccess;
    }
    // Check whether we received data from the GPU and send it to the network
    int buffSlot = args->transmitted%NCCL_STEPS;
    if (args->transmitted < args->posted && args->transmitted < args->done + NCCL_STEPS) {
      volatile int* sizesFifo = resources->recvMem->sizesFifo;
      volatile uint64_t* recvTail = &resources->recvMem->tail;
      if (sizesFifo[buffSlot] != -1 && (*recvTail > args->transmitted || args->protocol == NCCL_PROTO_LL)) {
        // We have something to receive, let's check if it's completely ready.
        int size = sizesFifo[buffSlot];
        char* buff = resources->shared ? (char*)resources->recvMem->ptrsFifo[buffSlot] : localBuff+buffSlot*stepSize;
        int ready = 1;
        if (args->protocol == NCCL_PROTO_LL128) {
          int ready = resources->useGdr;
          if (!ready) {
            // When data is in sysmem, we need to wait until all flags are correct since the GPU only
            // called threadfence()
            uint64_t flag = args->transmitted + 1;
            int nFifoLines = DIVUP(sizesFifo[buffSlot], sizeof(uint64_t)*NCCL_LL128_LINEELEMS);
            volatile uint64_t* lines = (volatile uint64_t*)buff;
            ready = 1;
            for (int i=0; i<nFifoLines; i++) {
              if (lines[i*NCCL_LL128_LINEELEMS+NCCL_LL128_DATAELEMS] != flag) { ready = 0; break; }
            }
          }
        } else if (args->protocol == NCCL_PROTO_LL) {
          uint32_t flag = NCCL_LL_FLAG(args->transmitted + 1);
          int nFifoLines = DIVUP(size, sizeof(union ncclLLFifoLine));
          union ncclLLFifoLine* lines = (union ncclLLFifoLine*)buff;
          for (int i=0; i<nFifoLines; i++) {
            volatile uint32_t *f1 = &lines[i].flag1;
            volatile uint32_t *f2 = &lines[i].flag2;
            if (f1[0] != flag || f2[0] != flag) { ready = 0; break; }
          }
        }
        if (ready) {
          // Data is ready, try to send.
          NCCLCHECK(ncclNetIsend(resources->netSendComm, buff, size, mhandle, args->requests+buffSlot));
          if (args->requests[buffSlot] != NULL) {
            TRACE(NCCL_NET, "sendProxy [%d/%d] Isend (LL) posted, req %p", args->transmitted, buffSlot, args->requests[buffSlot]);
            sizesFifo[buffSlot] = -1;
            // Make sure size is reset to zero before we update the head.
            __sync_synchronize();
            args->transmitted += args->sliceSteps;
            args->idle = 0;
            return ncclSuccess;
          }
        }
      }
    }
    // Check whether the network has completed some send operations.
    if (args->done < args->transmitted) {
      int done;
      int buffSlot = args->done%NCCL_STEPS;
      NCCLCHECK(ncclNetTest(args->requests[buffSlot], &done, NULL));
      if (done) {
        TRACE(NCCL_NET, "sendProxy [%d/%d] request %p done, size %d", args->done, buffSlot, args->requests[buffSlot]);
        if (resources->shared) {
          char* ptr = (char*)resources->recvMem->ptrsFifo[args->done%NCCL_STEPS];
          NCCLCHECK(ncclProxySharedBuffersFree(args->connector->comm, resources->useGdr, 0, args->channel->id, buffSize, ptr));
        }
        args->done += args->sliceSteps;

        if (resources->shared == 0) {
          resources->sendMem->head = args->done;
        }
        args->idle = 0;
        if (args->done == args->end) {
          resources->step = args->end;
          args->state = ncclProxyOpNone;
        }
        return ncclSuccess;
      }
    }
  }
  return ncclSuccess;
}

ncclResult_t netRecvProxy(struct ncclProxyArgs* args) {
  struct netRecvResources* resources = (struct netRecvResources*) (args->connector->transportResources);
  if (args->state == ncclProxyOpReady) {
    // Round to next multiple of sliceSteps
    resources->step = ROUNDUP(resources->step, args->chunkSteps);
    args->posted = args->received = args->transmitted = args->done = resources->step;
    args->end = resources->step + args->nsteps;
    args->state = ncclProxyOpProgress;
  }
  args->idle = 1;
  if (args->state == ncclProxyOpProgress) {
    int p = args->protocol;
    int stepSize = args->connector->comm->buffSizes[p] / NCCL_STEPS;
    char* localBuff = args->connector->conn.buffs[p];
    void* mhandle = *(resources->mhandlesProto[p]);
    int buffSize = stepSize*args->sliceSteps;
    if (resources->shared) buffSize /= SENDRECV_SLICEFACTOR;
    if (args->recvbytes < buffSize) buffSize = args->recvbytes;
    if ((args->posted < args->done + NCCL_STEPS) && (args->posted < args->end)) {
      int buffSlot = args->posted%NCCL_STEPS;
      char* ptr;
      if (resources->shared) {
        NCCLCHECK(ncclProxySharedBuffersAlloc(args->connector->comm, resources->useGdr, 1, args->channel->id, buffSize, &ptr));
        if (ptr == NULL) return ncclInternalError;
        volatile void** ptrsFifo = (volatile void**)resources->recvMem->ptrsFifo;
        ptrsFifo[buffSlot] = ptr;
      } else {
        ptr = localBuff+buffSlot*stepSize;
      }
      NCCLCHECK(ncclNetIrecv(resources->netRecvComm, ptr, buffSize, mhandle, args->requests+buffSlot));
      if (args->requests[buffSlot] != NULL) {
        TRACE(NCCL_NET, "recvProxy [%d/%d] posted recv request %p", args->posted, buffSlot, args->requests[buffSlot]);
        args->posted += args->sliceSteps;
        args->idle = 0;
        return ncclSuccess;
      } else if (resources->shared) {
        NCCLCHECK(ncclProxySharedBuffersFree(args->connector->comm, resources->useGdr, 1, args->channel->id, buffSize, ptr));
      }
    }
    if (args->posted > args->received) {
      int buffSlot = args->received%NCCL_STEPS;
      int done, size;
      NCCLCHECK(ncclNetTest(args->requests[buffSlot], &done, &size));
      if (done) {
        args->received += args->sliceSteps;
        if (size > 0 && args->protocol == NCCL_PROTO_SIMPLE && resources->useGdr) {
          // Don't pass data to the GPU yet, flush first.
          volatile void** ptrsFifo = (volatile void**)resources->recvMem->ptrsFifo;
          char* ptr = resources->shared ? (char*)(ptrsFifo[buffSlot]) : localBuff+buffSlot*stepSize;
          NCCLCHECK(ncclNetIflush(resources->netRecvComm, ptr, size, mhandle, args->requests+buffSlot));
        } else {
          args->requests[buffSlot] = NULL;
        }
        args->idle = 0;
        return ncclSuccess;
      }
    }
    if (args->received > args->transmitted) {
      // Progress flush operations
      int buffSlot = args->transmitted%NCCL_STEPS;
      int done = 1;
      if (args->requests[buffSlot]) NCCLCHECK(ncclNetTest(args->requests[buffSlot], &done, NULL));
      if (done) {
        args->transmitted += args->sliceSteps;
        __sync_synchronize();
        resources->recvMem->tail = args->transmitted;
        args->idle = 0;
        return ncclSuccess;
      }
    }
    if (args->transmitted > args->done) {
      volatile uint64_t* sendHead = &resources->sendMem->head;
      uint64_t done = *sendHead;
      while (done > args->done &&
          // LL and LL128 can acknowledge 0-bytes send before they even happen. Don't go past what we transmitted.
          args->transmitted > args->done) {
        if (resources->shared) {
          char* ptr = (char*)resources->recvMem->ptrsFifo[args->done%NCCL_STEPS];
          NCCLCHECK(ncclProxySharedBuffersFree(args->connector->comm, resources->useGdr, 1, args->channel->id, buffSize, ptr));
        }
        args->done += args->sliceSteps;
        args->idle = 0;
        if (args->done == args->end) {
          resources->step = args->end;
          args->state = ncclProxyOpNone;
        }
      }
    }
  }
  return ncclSuccess;
}

struct ncclTransport netTransport = {
  "NET",
  netCanConnect,
  { netSendSetup, netSendConnect, netSendFree, netSendProxy },
  { netRecvSetup, netRecvConnect, netRecvFree, netRecvProxy }
};
