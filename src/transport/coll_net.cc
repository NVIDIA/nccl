/*************************************************************************
 * Copyright (c) 2016-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "comm.h"
#include "coll_net.h"
#include "graph.h"
#include <assert.h>

struct collNetRecvConnectInfo {
  collNetHandle_t collNetHandle;
};

struct collNetSendConnectInfo {
  void* collNetComm;
  void* mhandles[NCCL_NUM_PROTOCOLS];
  struct reqSlot* reqFifo;
};

struct reqSlot {
  volatile void* recvBuff;
  volatile int size;
};

struct collNetSendResources {
  void* collNetSendComm;
  struct ncclSendMem* sendMem;
  struct ncclRecvMem* recvMem;
  uint32_t* llData;
  int netDev;
  int useGdr;
  void* sendMhandles[NCCL_NUM_PROTOCOLS];
  void* recvMhandles[NCCL_NUM_PROTOCOLS];
  struct ncclRecvMem* devRecvMem;
  uint64_t step;
  uint64_t llLastCleaning;
  struct reqSlot* reqFifo;
  int collNetRank;
};

struct collNetRecvResources {
  void* netListenComm;
  void* collNetRecvComm;
  struct ncclSendMem* sendMem;
  struct ncclRecvMem* recvMem;
  uint32_t* llData;
  int netDev;
  int useGdr;
  void* mhandles[NCCL_NUM_PROTOCOLS];
  struct ncclRecvMem* devRecvMem;
  uint64_t step;
  uint64_t llLastCleaning;
  struct reqSlot* reqFifo;
  int collNetRank;
};

/* Determine if we can communicate with the peer */
ncclResult_t collNetCanConnect(int* ret, struct ncclTopoSystem* topo, struct ncclTopoGraph* graph, struct ncclPeerInfo* info1, struct ncclPeerInfo* info2) {
  *ret = 1;
  return ncclSuccess;
}

/* Setup send connector, and return connect information for others in the coll communicator to connect to me */
ncclResult_t collNetSendSetup(struct ncclComm* comm, struct ncclTopoGraph* graph, struct ncclPeerInfo* myInfo, struct ncclPeerInfo* peerInfo, struct ncclConnect* connectInfo, struct ncclConnector* send, int channelId) {
  struct collNetSendResources* resources;
  NCCLCHECK(ncclCalloc(&resources, 1));
  send->transportResources = resources;

  NCCLCHECK(ncclTopoGetNetDev(comm->topo, myInfo->rank, graph, channelId, &resources->netDev));
  NCCLCHECK(ncclTopoCheckGdr(comm->topo, myInfo->busId, resources->netDev, 1, &resources->useGdr));

  NCCLCHECK(ncclCudaHostCalloc(&resources->sendMem, 1));

  int recvSize = offsetof(struct ncclRecvMem, buff);
  for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) recvSize += send->comm->buffSizes[p];

  if (resources->useGdr) {
    NCCLCHECK(ncclCudaCalloc((char**)(&resources->devRecvMem), recvSize));
  }
  NCCLCHECK(ncclCudaHostCalloc((char**)&resources->recvMem, recvSize));
  NCCLCHECK(ncclIbMalloc((void**)&(resources->llData), send->comm->buffSizes[NCCL_PROTO_LL]/2));

  INFO(NCCL_INIT|NCCL_NET,"Coll %02d : %d [send] via COLLNET/%s/%d%s", channelId, myInfo->rank, collNetName(), resources->netDev,
      resources->useGdr ? "/GDRDMA" : "");
  return ncclSuccess;
}

/* Setup recv connector */
ncclResult_t collNetRecvSetup(struct ncclComm* comm, struct ncclTopoGraph* graph, struct ncclPeerInfo* myInfo, struct ncclPeerInfo* peerInfo, struct ncclConnect* connectInfo, struct ncclConnector* recv, int channelId) {
  struct collNetRecvResources* resources;
  NCCLCHECK(ncclCalloc(&resources, 1));
  recv->transportResources = resources;

  NCCLCHECK(ncclTopoGetNetDev(comm->topo, myInfo->rank, graph, channelId, &resources->netDev));
  NCCLCHECK(ncclTopoCheckGdr(comm->topo, myInfo->busId, resources->netDev, 0, &resources->useGdr));

  NCCLCHECK(ncclCudaHostCalloc(&resources->sendMem, 1));

  int recvSize = offsetof(struct ncclRecvMem, buff);
  for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) recvSize += recv->comm->buffSizes[p];

  if (resources->useGdr) {
    NCCLCHECK(ncclCudaCalloc((char**)(&resources->devRecvMem), recvSize));
  }
  NCCLCHECK(ncclCudaHostCalloc((char**)&resources->recvMem, recvSize));

  NCCLCHECK(ncclIbMalloc((void**)&(resources->llData), recv->comm->buffSizes[NCCL_PROTO_LL]/2));

  INFO(NCCL_INIT|NCCL_NET,"Coll %02d : %d [receive] via COLLNET/%s/%d%s", channelId, myInfo->rank, collNetName(), resources->netDev,
      resources->useGdr ? "/GDRDMA" : "");
  struct collNetRecvConnectInfo* info = (struct collNetRecvConnectInfo*) connectInfo;
  NCCLCHECK(collNetListen(resources->netDev, &info->collNetHandle, &resources->netListenComm));
  return ncclSuccess;
}

ncclResult_t collNetSendConnect(struct ncclComm* comm, struct ncclConnect* connectInfos, int nranks, int rank, struct ncclConnector* send) {
  // Setup device pointers
  struct collNetSendResources* resources = (struct collNetSendResources*)send->transportResources;
  struct collNetSendConnectInfo* info = (struct collNetSendConnectInfo*)(connectInfos+rank);

  // Intermediate buffering on GPU for GPU Direct RDMA, but LL buffer is always on host
  struct ncclRecvMem* recvMem = resources->useGdr ? resources->devRecvMem : resources->recvMem;
  int offset = 0;
  for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
    send->conn.buffs[p] = (p == NCCL_PROTO_LL ? resources->recvMem->buff : recvMem->buff) + offset;
    offset += send->comm->buffSizes[p];
  }
  send->conn.direct |= resources->useGdr ? NCCL_DIRECT_NIC : 0;

  // Head/Tail/Opcount/Fifos are always on host
  send->conn.tail = &resources->recvMem->tail;
  send->conn.sizesFifo = resources->recvMem->sizesFifo;
  send->conn.head = &resources->sendMem->head;
  for (int i=0; i<NCCL_STEPS; i++) send->conn.sizesFifo[i] = -1;

  // Get info from recv side
  resources->collNetRank = rank;
  resources->reqFifo = info->reqFifo;
  resources->collNetSendComm = info->collNetComm;

  for (int p=0; p<NCCL_NUM_PROTOCOLS; p++)
    resources->recvMhandles[p] = info->mhandles[p];

  // Register buffers
  NCCLCHECK(collNetRegMr(resources->collNetSendComm, send->conn.buffs[NCCL_PROTO_SIMPLE], send->comm->buffSizes[NCCL_PROTO_SIMPLE],
        resources->useGdr ? NCCL_PTR_CUDA : NCCL_PTR_HOST, &resources->sendMhandles[NCCL_PROTO_SIMPLE]));
  NCCLCHECK(collNetRegMr(resources->collNetSendComm, resources->llData, send->comm->buffSizes[NCCL_PROTO_LL]/2,
        NCCL_PTR_HOST, &resources->sendMhandles[NCCL_PROTO_LL]));
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
  recv->conn.head = &resources->sendMem->head;

  // Connect to coll comm
  collNetHandle_t** handlePtrs = NULL;
  NCCLCHECK(ncclCalloc(&handlePtrs, nranks));
  for (int i = 0; i < nranks; i++) {
    struct collNetRecvConnectInfo* info = (struct collNetRecvConnectInfo*)(connectInfos+i);
    handlePtrs[i] = &(info->collNetHandle);
  }
  ncclResult_t res;
  NCCLCHECKGOTO(collNetConnect((void**)handlePtrs, nranks, rank, resources->netListenComm, &resources->collNetRecvComm), res, cleanup);

  // Register buffers
  NCCLCHECK(collNetRegMr(resources->collNetRecvComm, recv->conn.buffs[NCCL_PROTO_SIMPLE], recv->comm->buffSizes[NCCL_PROTO_SIMPLE],
        resources->useGdr ? NCCL_PTR_CUDA : NCCL_PTR_HOST, &resources->mhandles[NCCL_PROTO_SIMPLE]));
  NCCLCHECK(collNetRegMr(resources->collNetRecvComm, resources->llData, recv->comm->buffSizes[NCCL_PROTO_LL]/2,
        NCCL_PTR_HOST, &resources->mhandles[NCCL_PROTO_LL]));

  // Create shared info between send and recv proxies
  NCCLCHECK(ncclCalloc(&(resources->reqFifo), NCCL_STEPS));

  // Pass info to send side
  info->reqFifo = resources->reqFifo;
  info->collNetComm = resources->collNetRecvComm;
  for (int p=0; p<NCCL_NUM_PROTOCOLS; p++)
    info->mhandles[p] = resources->mhandles[p];

cleanup:
  if (handlePtrs != NULL) free(handlePtrs);
  // Close listen comm
  NCCLCHECK(collNetCloseListen(resources->netListenComm));

  return res;
}

ncclResult_t collNetSendFree(void* sendTransportResources) {
  struct collNetSendResources* resources = (struct collNetSendResources*)sendTransportResources;
  NCCLCHECK(ncclCudaHostFree(resources->sendMem));
  NCCLCHECK(ncclCudaHostFree(resources->recvMem));
  if (resources->collNetSendComm) {
    NCCLCHECK(collNetDeregMr(resources->collNetSendComm, resources->sendMhandles[NCCL_PROTO_LL]));
    NCCLCHECK(collNetDeregMr(resources->collNetSendComm, resources->sendMhandles[NCCL_PROTO_SIMPLE]));
  }
  if (resources->useGdr)
    CUDACHECK(cudaFree(resources->devRecvMem));
  free(resources->llData);
  free(resources);
  return ncclSuccess;
}

ncclResult_t collNetRecvFree(void* recvTransportResources) {
  struct collNetRecvResources* resources = (struct collNetRecvResources*)recvTransportResources;
  NCCLCHECK(ncclCudaHostFree(resources->sendMem));
  if (resources->collNetRecvComm) {
    NCCLCHECK(collNetDeregMr(resources->collNetRecvComm, resources->mhandles[NCCL_PROTO_LL]));
    NCCLCHECK(collNetDeregMr(resources->collNetRecvComm, resources->mhandles[NCCL_PROTO_SIMPLE]));
  }
  NCCLCHECK(ncclCudaHostFree(resources->recvMem));
  if (resources->useGdr)
    CUDACHECK(cudaFree(resources->devRecvMem));
  free(resources->llData);
  free(resources->reqFifo);

  // Make sure SendFree is called before RecvFree
  if (resources->collNetRecvComm) {
    NCCLCHECK(collNetCloseColl(resources->collNetRecvComm));
  }
  free(resources);
  return ncclSuccess;
}

ncclResult_t collNetSendProxy(struct ncclProxyArgs* args) {
  if (args->protocol == NCCL_PROTO_LL128) {
    WARN("CollNet does not support LL128");
    return ncclInternalError;
  }
  struct collNetSendResources* resources = (struct collNetSendResources*) (args->connector->transportResources);
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
    void* sendMhandle = resources->sendMhandles[p];
    void* recvMhandle = resources->recvMhandles[p];
    struct reqSlot* reqFifo = resources->reqFifo;
    int buffSlot = args->transmitted%NCCL_STEPS;
    if (args->transmitted < args->end && args->transmitted < args->done + NCCL_STEPS
        && reqFifo[buffSlot].recvBuff != NULL) {
      volatile int* sizesFifo = resources->recvMem->sizesFifo;
      volatile uint64_t* recvTail = &resources->recvMem->tail;
      if (sizesFifo[buffSlot] != -1 && (*recvTail > args->transmitted || args->protocol == NCCL_PROTO_LL)) {
        // We have something to receive, let's check if it's completely ready.
        int size = sizesFifo[buffSlot];
        char* buff = localBuff+buffSlot*stepSize;
        int ready = 1;
        if (args->protocol == NCCL_PROTO_LL) {
          uint32_t flag = NCCL_LL_FLAG(args->transmitted + 1);
          int nFifoLines = DIVUP(size, sizeof(union ncclLLFifoLine));
          union ncclLLFifoLine* lines = (union ncclLLFifoLine*)buff;
          // Pack data into another buffer
          int stepLines = stepSize / sizeof(union ncclLLFifoLine);
          uint32_t* sendBuff = resources->llData+buffSlot*2*stepLines;  // each line has two data elements
          buff = (char*)sendBuff;
          for (int i=0; i<nFifoLines; i++) {
            volatile uint32_t *f1 = &lines[i].flag1;
            volatile uint32_t *d1 = &lines[i].data1;
            volatile uint32_t *f2 = &lines[i].flag2;
            volatile uint32_t *d2 = &lines[i].data2;
            if (f1[0] != flag || f2[0] != flag) { ready = 0; break; }
            sendBuff[2*i] = d1[0];
            sendBuff[2*i+1] = d2[0];
          }
          size = nFifoLines*2*sizeof(uint32_t);
        }
        if (ready) {
          // Data is ready, try to send.
          int count = size/ncclTypeSize(args->dtype);
          NCCLCHECK(collNetIallreduce(resources->collNetSendComm, (void*) buff, (void*)(reqFifo[buffSlot].recvBuff), count, args->dtype, args->redOp, sendMhandle, recvMhandle, args->requests+buffSlot));
          if (args->requests[buffSlot] != NULL) {
            TRACE(NCCL_NET, "sendProxy [%d/%d] Iallreduce posted, req %p", args->transmitted, buffSlot, args->requests[buffSlot]);
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
      int done, size;
      int buffSlot = args->done%NCCL_STEPS;
      NCCLCHECK(collNetTest((void*)(args->requests[buffSlot]), &done, &size));
      if (done) {
        TRACE(NCCL_NET, "sendProxy [%d/%d] request %p done, size %d", args->done, buffSlot, args->requests[buffSlot], size);
        reqFifo[buffSlot].size = size;
        // Make sure size is updated before we set recvBuff to NULL (from the view of recv proxy, concerning the flush)
        // (reordered store after store is possible on POWER, though not on x86)
        __sync_synchronize();
        reqFifo[buffSlot].recvBuff = NULL; // Notify recvProxy
        args->done += args->sliceSteps;
        resources->sendMem->head = args->done;
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

ncclResult_t collNetRecvProxy(struct ncclProxyArgs* args) {
  if (args->protocol == NCCL_PROTO_LL128) {
    WARN("CollNet does not support LL128");
    return ncclInternalError;
  }
  struct collNetRecvResources* resources = (struct collNetRecvResources*) (args->connector->transportResources);
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
    void* mhandle = resources->mhandles[p];
    struct reqSlot* reqFifo = resources->reqFifo;
    if ((args->posted < args->done + NCCL_STEPS) && (args->posted < args->end)) {
      int buffSlot = args->posted%NCCL_STEPS;
      char* recvBuff = p == NCCL_PROTO_LL ? (char*)resources->llData : localBuff;
      int recvStepSize = p == NCCL_PROTO_LL ? stepSize/2 : stepSize;
      reqFifo[buffSlot].recvBuff = recvBuff+buffSlot*recvStepSize;
      TRACE(NCCL_NET, "recvProxy [%d/%d] posted buffer %p", args->posted, buffSlot, reqFifo[buffSlot].recvBuff);
      args->posted += args->sliceSteps;
      args->idle = 0;
      return ncclSuccess;
    }
    if (args->posted > args->received) {
      int buffSlot = args->received%NCCL_STEPS;
      if (reqFifo[buffSlot].recvBuff == NULL) { // Buffer is cleared : coll is complete
        TRACE(NCCL_NET, "recvProxy [%d/%d] done, size %d", args->received, buffSlot, reqFifo[buffSlot].size);
        if (args->protocol == NCCL_PROTO_LL) { // ll
          // re-attach flag
          uint32_t flag = NCCL_LL_FLAG(args->received + 1);
          int stepLines = stepSize / sizeof(union ncclLLFifoLine);
          union ncclLLFifoLine* lines = (union ncclLLFifoLine*)(localBuff+buffSlot*stepSize);
          uint32_t* recvData = resources->llData+buffSlot*2*stepLines;
          int nFifoLines = DIVUP(reqFifo[buffSlot].size, 2*sizeof(uint32_t));
          for (int i=0; i<nFifoLines; i++) {
            lines[i].v[0] = ((uint64_t)flag << 32) + recvData[2*i];
            lines[i].v[1] = ((uint64_t)flag << 32) + recvData[2*i+1];
          }
        }
        args->received += args->sliceSteps;
        if (reqFifo[buffSlot].size > 0 && args->protocol == NCCL_PROTO_SIMPLE && resources->useGdr) {
          NCCLCHECK(collNetIflush(resources->collNetRecvComm, localBuff+buffSlot*stepSize, reqFifo[buffSlot].size, mhandle, args->requests+buffSlot));
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
      if (args->requests[buffSlot]) NCCLCHECK(collNetTest(args->requests[buffSlot], &done, NULL));
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

struct ncclTransport collNetTransport = {
  "COL",
  collNetCanConnect,
  { collNetSendSetup, collNetSendConnect, collNetSendFree, collNetSendProxy },
  { collNetRecvSetup, collNetRecvConnect, collNetRecvFree, collNetRecvProxy }
};
