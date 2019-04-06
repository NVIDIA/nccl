/*************************************************************************
 * Copyright (c) 2016-2019, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "core.h"
#include "transport.h"
#include "nvmlwrap.h"
#include "net.h"
#include "param.h"
#include "topo.h"
#include <cuda_runtime.h>
#include <assert.h>

#define NET_MAX_IFS 16
#define NET_MAX_GPUS 32

// Cache GPU-NIC distances to avoid re-computing them
#define NET_TVALUE_UNKNOWN 0ULL
static ncclTvalue_t ncclNetTvalues[NET_MAX_GPUS] = { NET_TVALUE_UNKNOWN };
static int ncclNetNDev;

// We encode 3 bits of distance per interface into a ncclTvalue_t (64-bit)
#define NET_BITS_PER_IF 3
#define NET_BITS_PER_IF_MASK ((1<<NET_BITS_PER_IF)-1)
static_assert(sizeof(ncclTvalue_t)*8 >= NET_MAX_IFS*NET_BITS_PER_IF, "NET_MAX_IFS*NET_BITS_PER_IF must fit in a ncclTvalue_t");
static ncclTvalue_t getTvalue(short* distances, int ndev) {
  ncclTvalue_t tvalue = 0;
  for (int d=0; d<ndev; d++) {
    int score = 1 + PATH_SYS - distances[d];
    // Keep 3 bits of score info per dev
    tvalue |= ((score & NET_BITS_PER_IF_MASK)<<(NET_BITS_PER_IF*d));
  }
  return tvalue;
}
static int getScore(ncclTvalue_t tvalue, int dev) {
  return (tvalue >> (dev*NET_BITS_PER_IF)) & NET_BITS_PER_IF_MASK;
}

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
  struct ncclRecvMem* devRecvMem;
  uint64_t step;
  uint64_t llLastCleaning;
};

static ncclResult_t netDistance(int cudaDev, int dev, short* distance) {
  char* cudaPath = NULL;
  char* nicPath = NULL;
  ncclResult_t err;
  NCCLCHECK(getCudaPath(cudaDev, &cudaPath));
  err = ncclNetPciPath(dev, &nicPath);
  *distance = (err != ncclSuccess || nicPath == NULL || cudaPath == NULL) ? PATH_SYS : pciDistance(nicPath, cudaPath);
  if (nicPath) free(nicPath);
  if (cudaPath) free(cudaPath);
  return ncclSuccess;
}

static ncclResult_t netDevices(int* ndev, short** distances) {
  NCCLCHECK(ncclNetDevices(ndev));
  if (*ndev == 0) {
    WARN("Error : Network returned 0 device");
    return ncclSystemError;
  }
  if (*ndev > NET_MAX_IFS) *ndev = NET_MAX_IFS;

  *distances = (short*)malloc(*ndev*sizeof(short));
  if (*distances == NULL) return ncclSystemError;

  // Find distance with current GPU
  int cudaDev, nvmlDev;
  CUDACHECK(cudaGetDevice(&cudaDev));
  NCCLCHECK(getNvmlDevice(cudaDev, &nvmlDev))
  char line[1024];
  sprintf(line, "CUDA Dev %d[%d], %s NIC distance : ", cudaDev, nvmlDev, ncclNetName());
  for (int d=0; d<*ndev; d++) {
    NCCLCHECK(netDistance(cudaDev, d, *distances+d));
    sprintf(line+strlen(line), " %s", pathDists[(*distances)[d]]);
  }
  INFO(NCCL_INIT|NCCL_NET, "%s", line);
  return ncclSuccess;
}

/* Determine if we can communicate with the peer */
ncclResult_t netCanConnect(ncclTvalue_t* ret, struct ncclPeerInfo* myInfo, struct ncclPeerInfo* peerInfo) {
  int cudaDev;
  CUDACHECK(cudaGetDevice(&cudaDev));
  ret[0] = ncclNetTvalues[cudaDev];
  if (ret[0] == NET_TVALUE_UNKNOWN) {
    if (cudaDev >= NET_MAX_GPUS) {
      WARN("CUDA device %d >= MAX %d\n", cudaDev, NET_MAX_GPUS);
      return ncclInternalError;
    }
    int nDev;
    short* distances;
    NCCLCHECK(netDevices(&nDev, &distances));
    ncclNetTvalues[cudaDev] = ret[0] = getTvalue(distances, nDev);
    ncclNetNDev = nDev;
    free(distances);
  }
  return ncclSuccess;
}

static inline int groupBestStart(int nranks, int* groups, int group, ncclTvalue_t* values, int card, int minScore) {
  int bestRank = -1;
  int bestScore = 0;
  for (int rank=0; rank<nranks; rank++) {
    if (groups[rank] != group) continue;
    for (int i=0; i<nranks; i++) {
      ncclTvalue_t netValue = values[rank*nranks+i];
      if (netValue != 0) {
        ncclTvalue_t score = (netValue>>(NET_BITS_PER_IF*card)) & NET_BITS_PER_IF_MASK;
        if (score >= minScore && score > bestScore) {
          bestScore = score;
          bestRank = rank;
        }
        // All other values should be the same, stop here for this rank
        break;
      }
    }
  }
  return bestRank;
}
static inline int groupBestEnd(int nranks, int* groups, int group, int* subgroups, int startSubGroup, int startRank, ncclTvalue_t* values, int card, int minScore) {
  // For the last rank, we don't need the absolute best score, just to be within minScore.
  for (int rank=nranks-1; rank>=0; rank--) {
    if (groups[rank] != group) continue;
    if (startSubGroup != -1 && startSubGroup == subgroups[rank]) continue;
    if (startRank == rank) continue;
    for (int i=0; i<nranks; i++) {
      ncclTvalue_t netValue = values[rank*nranks+i];
      if (netValue != 0) {
        ncclTvalue_t score = (netValue>>(NET_BITS_PER_IF*card)) & NET_BITS_PER_IF_MASK;
        if (score >= minScore) {
          return rank;
        }
        // All other values should be the same, stop here for this rank
        break;
      }
    }
  }
  return -1;
}

ncclResult_t netGetRings(int nranks, int* groups, int* subgroups, ncclTvalue_t* values, int* nringsRet, int* prev, int* next, int minScore, int* nthreads) {
  int nGroups = groups[nranks-1] + 1;
  int *cardUsed, *starts, *ends;
  NCCLCHECK(ncclCalloc(&cardUsed, NET_MAX_IFS*nGroups));
  NCCLCHECK(ncclCalloc(&starts, nGroups));
  NCCLCHECK(ncclCalloc(&ends, nGroups));

  for (int ring = 0; ring<*nringsRet; ring++) {
    for (int group = 0; group<nGroups; group++) {
      int nranksInGroup = 0;
      int nsubGroups = 0;
      for (int rank=0; rank<nranks; rank++)
        if (groups[rank] == group) {
          nranksInGroup++;
          nsubGroups = std::max(subgroups[rank], nsubGroups);
        }
      starts[group] = ends[group] = -1;
      // Receive on the rank closest to the NIC
      for (int card=0; card<NET_MAX_IFS; card++) {
        if (cardUsed[group*NET_MAX_IFS+card] == 1) continue;
        int start = groupBestStart(nranks, groups, group, values, card, minScore);
        // Send from any rank, but best on a different subgroup and close to the NIC also.
        int end = (nranksInGroup == 1) ? start
            : groupBestEnd(nranks, groups, group, subgroups, nsubGroups ? subgroups[start] : -1, start, values, card, minScore);
        //printf("Ring %d, Minscore %d, Card %d, group %d, start = %d, end = %d\n", ring, minScore, card, group, start, end);
        if (start != -1 && end != -1) {
          cardUsed[group*NET_MAX_IFS+card] = 1;
          starts[group] = start;
          ends[group] = end;
          break;
        }
      }
      if (starts[group] == -1 || ends[group] == -1) {
        *nringsRet = ring;
        goto done;
      }
    }
    // Link groups together
    for (int group = 0; group<nGroups; group++) {
      int nextGroup = (group+1)%nGroups;
      next[ring*nranks+ends[group]] = starts[nextGroup];
      prev[ring*nranks+starts[nextGroup]] = ends[group];
    }
  }
done:
  free(cardUsed);
  free(starts);
  free(ends);
  return ncclSuccess;
}

int getDev(int cudaDev, int ringId) {
  ncclTvalue_t tvalues = ncclNetTvalues[cudaDev];

  int dev = 0;
  int maxScore = 0;
  for (int d=0; d<ncclNetNDev; d++) if (getScore(tvalues,d) > maxScore) maxScore = getScore(tvalues,d);
  int skip = ringId+1;
  while (skip) {
    for (int d=0; d<ncclNetNDev; d++) {
      if (getScore(tvalues, d) == maxScore) {
        skip--;
        if (skip == 0) { dev = d; goto end; }
      }
    }
  }
end:
  return dev;
}

NCCL_PARAM(NetGdrRead, "NET_GDR_READ", -2);
NCCL_PARAM(NetGdrLevel, "NET_GDR_LEVEL", PATH_PHB);

static ncclResult_t netGetGdrSupport(int dev, int read, int* useGdr) {
  *useGdr = 0;

  int cudaDev, nvmlDev;
  CUDACHECK(cudaGetDevice(&cudaDev));
  NCCLCHECK(getNvmlDevice(cudaDev, &nvmlDev))

  if (read) { // For reads (sends) only enable under certain conditions
    int gdrReadParam = ncclParamNetGdrRead();
    if (gdrReadParam == 0) return ncclSuccess;
    if (gdrReadParam < 0) {
       int nvlink;
       NCCLCHECK(ncclNvlinkGpu(&nvlink));
       if (!nvlink) return ncclSuccess;
    }
  }

  // Check if we are close enough that it makes sense to enable GDR
  int netGdrLevel = ncclParamNetGdrLevel();
  short distance;
  NCCLCHECK(netDistance(cudaDev, dev, &distance));
  if (distance >= netGdrLevel) {
    INFO(NCCL_NET,"NET/%s : GPU Direct RDMA Disabled for GPU %d[%d] / HCA %d (distance %d >= %d)", ncclNetName(), cudaDev, nvmlDev, dev, distance, netGdrLevel);
    return ncclSuccess;
  }

  // Finally, check if the NIC supports it
  int flags;
  NCCLCHECK(ncclNetPtrSupport(dev, &flags));
  if ((flags & NCCL_PTR_CUDA) == 0) return ncclSuccess;
  *useGdr = 1;
  INFO(NCCL_NET,"NET/%s : GPU Direct RDMA Enabled for GPU %d[%d] / HCA %d (distance %d < %d), read %d", ncclNetName(), cudaDev, nvmlDev, dev, distance, netGdrLevel, read);
  return ncclSuccess;
}

/* Determine if we will use this transport for this peer and return connect
 * information for this peer */
ncclResult_t netSendSetup(struct ncclPeerInfo* myInfo, struct ncclPeerInfo* peerInfo, struct ncclConnect* connectInfo, struct ncclConnector* send, int buffSize, int channelId) {
  struct netSendResources* resources;
  NCCLCHECK(ncclCalloc(&resources, 1));
  send->transportResources = resources;

  int cudaDev;
  CUDACHECK(cudaGetDevice(&cudaDev));
  resources->netDev = getDev(cudaDev, channelId);
  NCCLCHECK(netGetGdrSupport(resources->netDev, 1, &resources->useGdr));

  int sendSize = sizeof(struct ncclSendMem);
  NCCLCHECK(ncclCudaHostAlloc((void**)&resources->hostSendMem, (void**)&resources->devHostSendMem, sendSize));

  int recvSize = offsetof(struct ncclRecvMem, buff)+buffSize;
  if (resources->useGdr) {
    NCCLCHECK(ncclCudaCalloc((char**)(&resources->devRecvMem), recvSize));
  }
  NCCLCHECK(ncclCudaHostAlloc((void**)&resources->hostRecvMem, (void**)&resources->devHostRecvMem, recvSize));
  resources->buffSize = buffSize;

  INFO(NCCL_INIT|NCCL_NET,"Ring %02d : %d -> %d [send] via NET/%s/%d%s", channelId, myInfo->rank, peerInfo->rank, ncclNetName(), resources->netDev,
      resources->useGdr ? "/GDRDMA" : "");
  return ncclSuccess;
}

ncclResult_t netRecvSetup(struct ncclPeerInfo* myInfo, struct ncclPeerInfo* peerInfo, struct ncclConnect* connectInfo, struct ncclConnector* recv, int buffSize, int channelId) {
  struct netRecvResources* resources;
  NCCLCHECK(ncclCalloc(&resources, 1));
  recv->transportResources = resources;

  int cudaDev;
  CUDACHECK(cudaGetDevice(&cudaDev));
  resources->netDev = getDev(cudaDev, channelId);
  NCCLCHECK(netGetGdrSupport(resources->netDev, 0, &resources->useGdr));

  int sendSize = sizeof(struct ncclSendMem);
  NCCLCHECK(ncclCudaHostAlloc((void**)&resources->hostSendMem, (void**)&resources->devHostSendMem, sendSize));

  int recvSize = offsetof(struct ncclRecvMem, buff)+buffSize;
  if (resources->useGdr) {
    NCCLCHECK(ncclCudaCalloc((char**)(&resources->devRecvMem), recvSize));
  }
  NCCLCHECK(ncclCudaHostAlloc((void**)&resources->hostRecvMem, (void**)&resources->devHostRecvMem, recvSize));
  resources->buffSize = buffSize;

  INFO(NCCL_INIT|NCCL_NET,"Ring %02d : %d -> %d [receive] via NET/%s/%d%s", channelId, peerInfo->rank, myInfo->rank, ncclNetName(), resources->netDev,
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

  return ncclSuccess;
}

ncclResult_t netSendFree(void* transportResources) {
  struct netSendResources* resources = (struct netSendResources*)transportResources;
  NCCLCHECK(ncclCudaHostFree(resources->hostSendMem));
  NCCLCHECK(ncclNetDeregMr(resources->netSendComm, resources->mhandle));
  NCCLCHECK(ncclNetDeregMr(resources->netSendComm, resources->llMhandle));
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
        if (args->llMode) {
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
          struct ncclRecvMem* localMem = resources->useGdr ? resources->devRecvMem : resources->hostRecvMem;
          int stepSize = args->channel->buffSize/NCCL_STEPS;
          // Send through network
          int buffSlot = args->tail%NCCL_STEPS;
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
    int stepSize = ( args->llMode ? NCCL_LL_BUFF_SIZE : args->channel->buffSize ) / NCCL_STEPS;
    if (args->head < args->end) {
      struct ncclRecvMem* localMem = resources->useGdr ? resources->devRecvMem : resources->hostRecvMem;
      char* localBuff = args->llMode ? (char*)localMem->llBuff : localMem->buff;
      void* mhandle = args->llMode ? resources->llMhandle : resources->mhandle;
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
          if (args->llMode == 0) {
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
  netGetRings,
  { netSendSetup, netSendConnect, netSendFree, netSendProxy },
  { netRecvSetup, netRecvConnect, netRecvFree, netRecvProxy }
};
