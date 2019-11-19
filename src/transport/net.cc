/*************************************************************************
 * Copyright (c) 2016-2019, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "comm.h"
#include "net.h"
#include "graph.h"

struct netConnectInfo {
  ncclNetHandle_t netHandle;
};

struct netSendResources {
  void* netSendComm;
  struct ncclSendMem* hostSendMem;
  struct ncclRecvMem* hostRecvMem;
  struct ncclSendMem* devHostSendMem;
  struct ncclRecvMem* devHostRecvMem;
  int netDev;
  int useGdr;
  int buffSize;
  void* mhandle;
  void* llMhandle;
  void* ll128Mhandle;
  struct ncclRecvMem* devRecvMem;
  uint64_t step;
  uint64_t llLastCleaning;
};

struct netRecvResources {
  void* netListenComm;
  void* netRecvComm;
  struct ncclSendMem* hostSendMem;
  struct ncclRecvMem* hostRecvMem;
  struct ncclSendMem* devHostSendMem;
  struct ncclRecvMem* devHostRecvMem;
  int netDev;
  int useGdr;
  int buffSize;
  void* mhandle;
  void* llMhandle;
  void* ll128Mhandle;
  struct ncclRecvMem* devRecvMem;
  uint64_t step;
  uint64_t llLastCleaning;
};

/* Determine if two peers can communicate with NET */
ncclResult_t netCanConnect(int* ret, struct ncclTopoSystem* topo, struct ncclTopoGraph* graph, struct ncclPeerInfo* info1, struct ncclPeerInfo* info2) {
  *ret = 1;
  return ncclSuccess;
}

NCCL_PARAM(NetGdrRead, "NET_GDR_READ", -2);
NCCL_PARAM(NetGdrLevel, "NET_GDR_LEVEL", PATH_PHB);

static ncclResult_t netGetGdrSupport(struct ncclTopoSystem* topo, int64_t busId, int netDev, int read, int* useGdr) {
  *useGdr = 0;

  if (read) { // For reads (sends) only enable under certain conditions
    int gdrReadParam = ncclParamNetGdrRead();
    if (gdrReadParam == 0) return ncclSuccess;
    if (gdrReadParam < 0) {
       int nvlink;
       NCCLCHECK(ncclTopoHasNvlink(topo, busId, &nvlink));
       if (!nvlink) return ncclSuccess;
    }
  }

  // Check if we are close enough that it makes sense to enable GDR
  int netGdrLevel = ncclParamNetGdrLevel();
  int distance;
  NCCLCHECK(ncclTopoNetDistance(topo, busId, netDev, &distance));
  if (distance >= netGdrLevel) {
    INFO(NCCL_NET,"NET/%s : GPU Direct RDMA Disabled for GPU %lx / HCA %d (distance %d >= %d)", ncclNetName(), busId, netDev, distance, netGdrLevel);
    return ncclSuccess;
  }

  // Finally, check if the NIC supports it
  int flags;
  NCCLCHECK(ncclNetPtrSupport(netDev, &flags));
  if ((flags & NCCL_PTR_CUDA) == 0) return ncclSuccess;
  *useGdr = 1;
  INFO(NCCL_NET,"NET/%s : GPU Direct RDMA Enabled for GPU %lx / HCA %d (distance %d < %d), read %d", ncclNetName(), busId, netDev, distance, netGdrLevel, read);
  return ncclSuccess;
}

/* Determine if we will use this transport for this peer and return connect
 * information for this peer */
ncclResult_t netSendSetup(struct ncclTopoSystem* topo, struct ncclTopoGraph* graph, struct ncclPeerInfo* myInfo, struct ncclPeerInfo* peerInfo, struct ncclConnect* connectInfo, struct ncclConnector* send, int buffSize, int channelId) {
  struct netSendResources* resources;
  NCCLCHECK(ncclCalloc(&resources, 1));
  send->transportResources = resources;

  NCCLCHECK(ncclTopoGetNetDev(graph, 1, channelId, &resources->netDev));
  NCCLCHECK(netGetGdrSupport(topo, myInfo->busId, resources->netDev, 1, &resources->useGdr));

  int sendSize = sizeof(struct ncclSendMem);
  NCCLCHECK(ncclCudaHostAlloc((void**)&resources->hostSendMem, (void**)&resources->devHostSendMem, sendSize));

  int recvSize = offsetof(struct ncclRecvMem, buff)+buffSize;
  if (resources->useGdr) {
    NCCLCHECK(ncclCudaCalloc((char**)(&resources->devRecvMem), recvSize));
  }
  NCCLCHECK(ncclCudaHostAlloc((void**)&resources->hostRecvMem, (void**)&resources->devHostRecvMem, recvSize));
  resources->buffSize = buffSize;

  INFO(NCCL_INIT|NCCL_NET,"Ring %02d : %d[%lx] -> %d[%lx] [send] via NET/%s/%d%s", channelId, myInfo->rank, myInfo->busId, peerInfo->rank, peerInfo->busId, ncclNetName(), resources->netDev,
      resources->useGdr ? "/GDRDMA" : "");
  return ncclSuccess;
}

ncclResult_t netRecvSetup(struct ncclTopoSystem* topo, struct ncclTopoGraph* graph, struct ncclPeerInfo* myInfo, struct ncclPeerInfo* peerInfo, struct ncclConnect* connectInfo, struct ncclConnector* recv, int buffSize, int channelId) {
  struct netRecvResources* resources;
  NCCLCHECK(ncclCalloc(&resources, 1));
  recv->transportResources = resources;

  NCCLCHECK(ncclTopoGetNetDev(graph, 0, channelId, &resources->netDev));
  NCCLCHECK(netGetGdrSupport(topo, myInfo->busId, resources->netDev, 0, &resources->useGdr));

  int sendSize = sizeof(struct ncclSendMem);
  NCCLCHECK(ncclCudaHostAlloc((void**)&resources->hostSendMem, (void**)&resources->devHostSendMem, sendSize));

  int recvSize = offsetof(struct ncclRecvMem, buff)+buffSize;
  if (resources->useGdr) {
    NCCLCHECK(ncclCudaCalloc((char**)(&resources->devRecvMem), recvSize));
  }
  NCCLCHECK(ncclCudaHostAlloc((void**)&resources->hostRecvMem, (void**)&resources->devHostRecvMem, recvSize));
  resources->buffSize = buffSize;

  INFO(NCCL_INIT|NCCL_NET,"Ring %02d : %d[%lx] -> %d[%lx] [receive] via NET/%s/%d%s", channelId, peerInfo->rank, peerInfo->busId, myInfo->rank, myInfo->busId, ncclNetName(), resources->netDev,
      resources->useGdr ? "/GDRDMA" : "");
  struct netConnectInfo* info = (struct netConnectInfo*) connectInfo;
  NCCLCHECK(ncclNetListen(resources->netDev, &info->netHandle, &resources->netListenComm));
  return ncclSuccess;
}

ncclResult_t netSendConnect(struct ncclConnect* connectInfo, struct ncclConnector* send) {
  // Setup device pointers
  struct netSendResources* resources = (struct netSendResources*)send->transportResources;

  // Intermediate buffering on GPU for GPU Direct RDMA, but LL buffer is always on host
  struct ncclRecvMem* recvMem = resources->useGdr ? resources->devRecvMem : resources->devHostRecvMem;
  send->conn.buff = recvMem->buff;
  send->conn.llBuff = resources->devHostRecvMem->llBuff;
  send->conn.ll128Buff = recvMem->ll128Buff;

  // Head/Tail/Opcount/Fifos are always on host
  send->conn.tail = &resources->devHostRecvMem->tail;
  send->conn.opCountRem = &resources->devHostRecvMem->opCount;
  send->conn.fifo = resources->devHostRecvMem->sizesFifo;
  send->conn.head = &resources->devHostSendMem->head;
  send->conn.opCountLoc = &resources->devHostSendMem->opCount;
  for (int i=0; i<NCCL_STEPS; i++) send->conn.fifo[i] = -1;

  // Connect to remote peer
  struct netConnectInfo* info = (struct netConnectInfo*)connectInfo;
  NCCLCHECK(ncclNetConnect(resources->netDev, info->netHandle, &resources->netSendComm));

  NCCLCHECK(ncclNetRegMr(resources->netSendComm, recvMem->buff, resources->buffSize,
        resources->useGdr ? NCCL_PTR_CUDA : NCCL_PTR_HOST, &resources->mhandle));
  NCCLCHECK(ncclNetRegMr(resources->netSendComm, resources->devHostRecvMem->llBuff,
        NCCL_LL_BUFF_SIZE, NCCL_PTR_HOST, &resources->llMhandle));
  NCCLCHECK(ncclNetRegMr(resources->netSendComm, recvMem->ll128Buff, NCCL_LL128_BUFF_SIZE,
        resources->useGdr ? NCCL_PTR_CUDA : NCCL_PTR_HOST, &resources->ll128Mhandle));

  return ncclSuccess;
}

/* Connect to this peer */
ncclResult_t netRecvConnect(struct ncclConnect* connectInfo, struct ncclConnector* recv) {
  // Setup device pointers
  struct netRecvResources* resources = (struct netRecvResources*)recv->transportResources;

  // Intermediate buffering on GPU for GPU Direct RDMA
  struct ncclRecvMem* recvMem = resources->useGdr ? resources->devRecvMem : resources->devHostRecvMem;
  recv->conn.buff = recvMem->buff;
  recv->conn.llBuff = recvMem->llBuff;
  recv->conn.ll128Buff = recvMem->ll128Buff;

  // Head/Tail/Opcount are always on host
  recv->conn.tail = &resources->devHostRecvMem->tail;
  recv->conn.opCountLoc = &resources->devHostRecvMem->opCount;
  recv->conn.head = &resources->devHostSendMem->head;
  recv->conn.opCountRem = &resources->devHostSendMem->opCount;

  // Finish connection establishment from remote peer
  NCCLCHECK(ncclNetAccept(resources->netListenComm, &resources->netRecvComm));
  NCCLCHECK(ncclNetCloseListen(resources->netListenComm));

  NCCLCHECK(ncclNetRegMr(resources->netRecvComm, recvMem->buff, resources->buffSize,
        resources->useGdr ? NCCL_PTR_CUDA : NCCL_PTR_HOST, &resources->mhandle));
  NCCLCHECK(ncclNetRegMr(resources->netRecvComm, recvMem->llBuff, NCCL_LL_BUFF_SIZE,
        resources->useGdr ? NCCL_PTR_CUDA : NCCL_PTR_HOST, &resources->llMhandle));
  NCCLCHECK(ncclNetRegMr(resources->netRecvComm, recvMem->ll128Buff, NCCL_LL128_BUFF_SIZE,
        resources->useGdr ? NCCL_PTR_CUDA : NCCL_PTR_HOST, &resources->ll128Mhandle));

  return ncclSuccess;
}

ncclResult_t netSendFree(void* transportResources) {
  struct netSendResources* resources = (struct netSendResources*)transportResources;
  NCCLCHECK(ncclCudaHostFree(resources->hostSendMem));
  NCCLCHECK(ncclNetDeregMr(resources->netSendComm, resources->mhandle));
  NCCLCHECK(ncclNetDeregMr(resources->netSendComm, resources->llMhandle));
  NCCLCHECK(ncclNetDeregMr(resources->netSendComm, resources->ll128Mhandle));
  NCCLCHECK(ncclCudaHostFree(resources->hostRecvMem));
  if (resources->useGdr)
    CUDACHECK(cudaFree(resources->devRecvMem));
  NCCLCHECK(ncclNetCloseSend(resources->netSendComm));
  free(resources);
  return ncclSuccess;
}

ncclResult_t netRecvFree(void* transportResources) {
  struct netRecvResources* resources = (struct netRecvResources*)transportResources;
  NCCLCHECK(ncclCudaHostFree(resources->hostSendMem));
  NCCLCHECK(ncclNetDeregMr(resources->netRecvComm, resources->mhandle));
  NCCLCHECK(ncclNetDeregMr(resources->netRecvComm, resources->llMhandle));
  NCCLCHECK(ncclNetDeregMr(resources->netRecvComm, resources->ll128Mhandle));
  NCCLCHECK(ncclCudaHostFree(resources->hostRecvMem));
  if (resources->useGdr)
    CUDACHECK(cudaFree(resources->devRecvMem));
  NCCLCHECK(ncclNetCloseRecv(resources->netRecvComm));
  free(resources);
  return ncclSuccess;
}

ncclResult_t netSendProxy(struct ncclProxyArgs* args) {
  struct netSendResources* resources = (struct netSendResources*) (args->connector->transportResources);
  if (args->state == ncclProxyOpReady) {
    // Update opCount
    resources->hostRecvMem->opCount = args->opCount;

    // Round to next multiple of sliceSteps
    resources->step = ROUNDUP(resources->step, args->chunkSteps);
    args->head = resources->step;
    args->tail = resources->step;
    args->end = args->head + args->nsteps;
    args->state = ncclProxyOpProgress;
  }
  if (args->state == ncclProxyOpProgress) {
    args->idle = 1;
    if (args->head < args->end) {
      if (args->tail < args->end && args->tail < args->head + NCCL_STEPS) {
        volatile int* sizesFifo = resources->hostRecvMem->sizesFifo;
        volatile uint64_t* recvTail = &resources->hostRecvMem->tail;
        if (args->protocol == NCCL_PROTO_LL128) {
          int stepSize = NCCL_LL128_BUFF_SIZE/NCCL_STEPS;
          if (args->tail < *recvTail) {
            int buffSlot = args->tail%NCCL_STEPS;
            if (sizesFifo[buffSlot] != -1) {
              struct ncclRecvMem* localMem = resources->useGdr ? resources->devRecvMem : resources->hostRecvMem;
              char* localBuff = (char*)localMem->ll128Buff;
              int ready = resources->useGdr;
              if (!ready) {
                // When data is in sysmem, we need to wait until all flags are correct since the GPU only
                // called threadfence()
                uint64_t flag = args->tail + 1;
                int nFifoLines = DIVUP(sizesFifo[buffSlot], sizeof(uint64_t)*NCCL_LL128_LINEELEMS);
                volatile uint64_t* lines = (volatile uint64_t*)(localBuff+buffSlot*stepSize);
                ready = 1;
                for (int i=0; i<nFifoLines; i++) {
                  if (lines[i*NCCL_LL128_LINEELEMS+NCCL_LL128_DATAELEMS] != flag) { ready = 0; break; }
                }
              }
              if (ready) {
                // Send through network
                NCCLCHECK(ncclNetIsend(resources->netSendComm, localBuff+buffSlot*stepSize, sizesFifo[buffSlot], resources->ll128Mhandle, args->requests+buffSlot));
                if (args->requests[buffSlot] != NULL) {
                  sizesFifo[buffSlot] = -1;
                  // Make sure size is reset to zero before we update the head.
                  __sync_synchronize();
                  args->tail += args->sliceSteps;
                  args->idle = 0;
                }
              }
            }
          }
        } else if (args->protocol == NCCL_PROTO_LL) {
          int buffSlot = args->tail%NCCL_STEPS;
          int size = sizesFifo[buffSlot];
          if (size != -1) {
            uint32_t flag = NCCL_LL_FLAG(args->tail + 1);
            int nFifoLines = DIVUP(size, sizeof(union ncclLLFifoLine));
            size = nFifoLines * sizeof(union ncclLLFifoLine);
            union ncclLLFifoLine* lines = resources->hostRecvMem->llBuff+buffSlot*NCCL_LL_SLICE_LINES;
            int ready = 1;
            for (int i=0; i<nFifoLines; i++) {
              volatile uint32_t *f1 = &lines[i].flag1;
              volatile uint32_t *f2 = &lines[i].flag2;
              if (f1[0] != flag || f2[0] != flag) { ready = 0; break; }
            }
            if (ready) {
              NCCLCHECK(ncclNetIsend(resources->netSendComm, lines, size, resources->llMhandle, args->requests+buffSlot));
              if (args->requests[buffSlot] != NULL) {
                sizesFifo[buffSlot] = -1;
                // Make sure size is reset to zero before we update the head.
                __sync_synchronize();
                args->tail += args->sliceSteps;
                args->idle = 0;
              }
            }
          }
        } else if (args->tail < *recvTail) {
          int stepSize = args->channel->buffSize/NCCL_STEPS;
          struct ncclRecvMem* localMem = resources->useGdr ? resources->devRecvMem : resources->hostRecvMem;
          // Send through network
          int buffSlot = args->tail%NCCL_STEPS;
          if (sizesFifo[buffSlot] != -1) {
            NCCLCHECK(ncclNetIsend(resources->netSendComm, localMem->buff+buffSlot*stepSize, sizesFifo[buffSlot], resources->mhandle, args->requests+buffSlot));
            if (args->requests[buffSlot] != NULL) {
              sizesFifo[buffSlot] = -1;
              // Make sure size is reset to zero before we update the head.
              __sync_synchronize();
              args->tail += args->sliceSteps;
              args->idle = 0;
            }
          }
        }
      }
      if (args->head < args->tail) {
        int done;
        int buffSlot = args->head%NCCL_STEPS;
        NCCLCHECK(ncclNetTest(args->requests[buffSlot], &done, NULL));
        if (done) {
          args->head += args->sliceSteps;
          resources->hostSendMem->head = args->head;
          args->idle = 0;
        }
      }
    }
    if (args->head == args->end) {
      resources->step = args->end;
      args->idle = 0;
      args->state = ncclProxyOpNone;
    }
  }
  return ncclSuccess;
}

ncclResult_t netRecvProxy(struct ncclProxyArgs* args) {
  struct netRecvResources* resources = (struct netRecvResources*) (args->connector->transportResources);
  if (args->state == ncclProxyOpReady) {
    // Update opCount
    resources->hostSendMem->opCount = args->opCount;

    // Round to next multiple of sliceSteps
    resources->step = ROUNDUP(resources->step, args->chunkSteps);
    args->head = resources->step;
    args->tail = resources->step;
    args->end = args->head + args->nsteps;
    args->state = ncclProxyOpProgress;
  }
  if (args->state == ncclProxyOpProgress) {
    args->idle = 1;
    int stepSize = ( args->protocol == NCCL_PROTO_LL ? NCCL_LL_BUFF_SIZE : args->protocol == NCCL_PROTO_LL128 ? NCCL_LL128_BUFF_SIZE : args->channel->buffSize ) / NCCL_STEPS;
    if (args->head < args->end) {
      struct ncclRecvMem* localMem = resources->useGdr ? resources->devRecvMem : resources->hostRecvMem;
      char* localBuff = args->protocol == NCCL_PROTO_LL ? (char*)localMem->llBuff : args->protocol == NCCL_PROTO_LL128 ? (char*)localMem->ll128Buff : localMem->buff;
      void* mhandle = args->protocol == NCCL_PROTO_LL ? resources->llMhandle : args->protocol == NCCL_PROTO_LL128 ? resources->ll128Mhandle : resources->mhandle;
      volatile uint64_t* sendHead = &resources->hostSendMem->head;
      if ((args->tail < args->head + NCCL_STEPS) && (args->tail < *sendHead + NCCL_STEPS) && (args->tail < args->end)) {
        int buffSlot = args->tail%NCCL_STEPS;
        int sliceSize = stepSize * args->sliceSteps;
        NCCLCHECK(ncclNetIrecv(resources->netRecvComm, localBuff+buffSlot*stepSize, sliceSize, mhandle, args->requests+buffSlot));
        if (args->requests[buffSlot] != NULL) {
          args->tail += args->sliceSteps;
          args->idle = 0;
        }
      }
      if (args->tail > args->head) {
        int buffSlot = args->head%NCCL_STEPS;
        int done, size;
        NCCLCHECK(ncclNetTest(args->requests[buffSlot], &done, &size));
        if (done) {
          args->head += args->sliceSteps;
          if (args->protocol == NCCL_PROTO_SIMPLE) {
            if (resources->useGdr) ncclNetFlush(resources->netRecvComm, localBuff+buffSlot*stepSize, size, mhandle);
            resources->hostRecvMem->tail = args->head;
          }
          args->idle = 0;
        }
      }
    }
    if (args->head == args->end) {
      resources->step = args->end;
      args->idle = 0;
      args->state = ncclProxyOpNone;
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
