/*************************************************************************
 * Copyright (c) 2016-2018, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "core.h"
#include "transport.h"
#include "nvmlwrap.h"
#include "net.h"
#include "param.h"
#include "nvlink.h"
#include <cuda_runtime.h>
#include <assert.h>

#define NET_MAX_IFS 16

// We encode 3 bits of distance per interface into a ncclTvalue_t (64-bit)
#define NET_BITS_PER_IF 3
#define NET_BITS_PER_IF_MASK ((1<<NET_BITS_PER_IF)-1)
static_assert(sizeof(ncclTvalue_t)*8 >= NET_MAX_IFS*NET_BITS_PER_IF, "NET_MAX_IFS*NET_BITS_PER_IF must fit in a ncclTvalue_t");
static ncclTvalue_t getTvalue(short* distances, int ndev) {
  ncclTvalue_t tvalue = 0;
  for (int d=0; d<ndev; d++) {
    int score = 1 + PATH_SOC - distances[d];
    // Keep 3 bits of score info per dev
    tvalue |= ((score & NET_BITS_PER_IF_MASK)<<(NET_BITS_PER_IF*d));
  }
  return tvalue;
}

struct netInfo {
  int rank;
  int ndev;
  ncclTvalue_t tValue;
  short distances[NET_MAX_IFS];
};

struct netConnectInfo {
  ncclNetHandle_t netHandle;
};

struct netSendResources {
  void* netSendComm;
  struct ncclSendMem* hostSendMem;
  struct ncclRecvMem* hostRecvMem;
  struct ncclSendMem* devHostSendMem;
  struct ncclRecvMem* devHostRecvMem;
  struct ncclSendMem* hostDevMem;
  int netDev;
  int useGdr;
  struct ncclRecvMem* devNetMem;
  uint64_t llStep;
  uint64_t llLastCleaning;
};

struct netRecvResources {
  void* netListenComm;
  void* netRecvComm;
  struct ncclSendMem* hostSendMem;
  struct ncclRecvMem* hostRecvMem;
  struct ncclSendMem* devHostSendMem;
  struct ncclRecvMem* devHostRecvMem;
  struct ncclRecvMem* hostDevMem;
  int netDev;
  int useGdr;
  uint64_t llStep;
  uint64_t llLastCleaning;
};

/* Fill information necessary to exchange between ranks to choose whether or not
 * to use this transport */
ncclResult_t netFillInfo(ncclTinfo_t* opaqueInfo, int rank) {
  struct netInfo* info = (struct netInfo*)opaqueInfo;
  static_assert(sizeof(struct netInfo) <= sizeof(ncclTinfo_t), "NET Info too large");
  info->rank = rank;
  NCCLCHECK(ncclNetDevices(&info->ndev));
  if (info->ndev == 0) {
    WARN("Error : Network returned 0 device");
    return ncclSystemError;
  }
  if (info->ndev > NET_MAX_IFS) info->ndev = NET_MAX_IFS;

  // Find distance with current GPU
  int cudaDev;
  cudaGetDevice(&cudaDev);
  char* cudaPath;
  NCCLCHECK(getCudaPath(cudaDev, &cudaPath));

  char line[1024];
  sprintf(line, "CUDA Dev %d, %s NIC distance : ", cudaDev, ncclNetName());
  for (int d=0; d<info->ndev; d++) {
    char* nicPath;
    ncclResult_t err = ncclNetPciPath(d, &nicPath);
    info->distances[d] = (err != ncclSuccess || nicPath == NULL || cudaPath == NULL) ? PATH_SOC : pciDistance(nicPath, cudaPath);
    sprintf(line+strlen(line), " %s", pathDists[info->distances[d]]);
    if (err == ncclSuccess) free(nicPath);
  }
  INFO(NCCL_INIT|NCCL_NET, "%s", line);
  free(cudaPath);
  return ncclSuccess;
}

/* Determine if we can communicate with the peer */
ncclResult_t netCanConnect(ncclTvalue_t* ret, ncclTinfo_t* myOpaqueInfo, ncclTinfo_t* peerOpaqueInfo) {
  struct netInfo* myInfo = (struct netInfo*)myOpaqueInfo;
  ret[0] = getTvalue(myInfo->distances, myInfo->ndev);
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
  int cardUsed[NET_MAX_IFS*nGroups];
  for (int c=0; c<NET_MAX_IFS*nGroups; c++) cardUsed[c] = 0;

  for (int ring = 0; ring<*nringsRet; ring++) {
    int starts[nGroups];
    int ends[nGroups];
    for (int group = 0; group<nGroups; group++) {
      int nranksInGroup = 0;
      int nsubGroups = 0;
      for (int rank=0; rank<nranks; rank++) if (groups[rank] == group) {
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
        return ncclSuccess;
      }
    }
    // Link groups together
    for (int group = 0; group<nGroups; group++) {
      int nextGroup = (group+1)%nGroups;
      next[ring*nranks+ends[group]] = starts[nextGroup];
      prev[ring*nranks+starts[nextGroup]] = ends[group];
    }
  }
  return ncclSuccess;
}

int getDev(int ringId, int nDev, short* distances) {
  int minDistance = PATH_SOC;
  for (int d=0; d<nDev; d++) if (distances[d] < minDistance) minDistance = distances[d];
  int skip = ringId+1;
  while (skip) {
    for (int d=0; d<nDev; d++) {
      if (distances[d] == minDistance) {
        skip--;
        if (skip == 0) return d;
      }
    }
  }
  return 0;
}

NCCL_PARAM(NetGdrRead, "NET_GDR_READ", -2);
NCCL_PARAM(NetGdrLevel, "NET_GDR_LEVEL", PATH_PHB);

static ncclResult_t netGetGdrSupport(int dev, int distance, int read, int* useGdr) {
  *useGdr = 0;

  int cudaDev;
  CUDACHECK(cudaGetDevice(&cudaDev));

  if (read) { // For reads (sends) only enable under certain conditions
    int gdrReadParam = ncclParamNetGdrRead();
    if (gdrReadParam == 0) return ncclSuccess;
    else if (gdrReadParam < 0) { // default : enable only on DGX2
      char busId[NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE];
      CUDACHECK(cudaDeviceGetPCIBusId(busId, NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE, cudaDev));
      int nvlinks = getNumNvlinks(busId);
      if (nvlinks < CONNECT_NVSWITCH || ncclCudaCompCap() < 7) return ncclSuccess;
    }
  }

  // Check if we are close enough that it makes sense to enable GDR
  int netGdrLevel = ncclParamNetGdrLevel();
  if (distance >= netGdrLevel) {
    INFO(NCCL_INIT|NCCL_NET,"NET/%s : GPU Direct RDMA Disabled for GPU %d / HCA %d (distance %d >= %d)", ncclNetName(), cudaDev, dev, distance, netGdrLevel);
    return ncclSuccess;
  }

  // Finally, check if the NIC supports it
  int flags;
  NCCLCHECK(ncclNetPtrSupport(dev, &flags));
  if (flags & NCCL_PTR_CUDA == 0) return ncclSuccess;
  *useGdr = 1;
  INFO(NCCL_INIT|NCCL_NET,"NET/%s : GPU Direct RDMA Enabled for GPU %d / HCA %d (distance %d >= %d), read %d", ncclNetName(), cudaDev, dev, distance, netGdrLevel, read);
  return ncclSuccess;
}

/* Determine if we will use this transport for this peer and return connect
 * information for this peer */
ncclResult_t netSendSetup(ncclTinfo_t* myOpaqueInfo, ncclTinfo_t* peerOpaqueInfo, struct ncclConnect* connectInfo, struct ncclRing* ring) {
  struct netSendResources* resources;
  NCCLCHECK(ncclCalloc(&resources, 1));
  ring->send.transportResources = resources;

  struct netInfo* myInfo = (struct netInfo*)myOpaqueInfo;
  resources->netDev = getDev(ring->id, myInfo->ndev, myInfo->distances);
  NCCLCHECK(netGetGdrSupport(resources->netDev, myInfo->distances[resources->netDev], 1, &resources->useGdr));

  int size = offsetof(struct ncclRecvMem, buff)+ring->buffSize;
  if (resources->useGdr) {
    NCCLCHECK(ncclCudaCalloc((char**)(&resources->devNetMem), size));
  }

  NCCLCHECK(ncclCudaHostAlloc((void**)&resources->hostRecvMem, (void**)&resources->devHostRecvMem, size));
  NCCLCHECK(ncclCudaHostAlloc((void**)&resources->hostSendMem, (void**)&resources->devHostSendMem, size));

  return ncclSuccess;
}

ncclResult_t netRecvSetup(ncclTinfo_t* myOpaqueInfo, ncclTinfo_t* peerOpaqueInfo, struct ncclConnect* connectInfo, struct ncclRing* ring) {
  struct netRecvResources* resources;
  NCCLCHECK(ncclCalloc(&resources, 1));
  ring->recv.transportResources = resources;

  struct netInfo* myInfo = (struct netInfo*)myOpaqueInfo;
  resources->netDev = getDev(ring->id, myInfo->ndev, myInfo->distances);
  NCCLCHECK(netGetGdrSupport(resources->netDev, myInfo->distances[resources->netDev], 0, &resources->useGdr));

  int sendSize = sizeof(struct ncclSendMem);
  NCCLCHECK(ncclCudaHostAlloc((void**)&resources->hostSendMem, (void**)&resources->devHostSendMem, sendSize));

  int recvSize = offsetof(struct ncclRecvMem, buff)+ring->buffSize;
  NCCLCHECK(ncclCudaHostAlloc((void**)&resources->hostRecvMem, (void**)&resources->devHostRecvMem, recvSize));

  struct netInfo* peerInfo = (struct netInfo*)peerOpaqueInfo;
  INFO(NCCL_INIT|NCCL_NET,"Ring %02d : %d -> %d via NET/%s/%d%s%s", ring->id, peerInfo->rank, myInfo->rank, ncclNetName(), resources->netDev,
      resources->useGdr ? "/GDRDMA" : "",
      (resources->hostDevMem != NULL) ? "/GDCopy" : "");
  struct netConnectInfo* info = (struct netConnectInfo*) connectInfo;
  NCCLCHECK(ncclNetListen(resources->netDev, &info->netHandle, &resources->netListenComm));
  return ncclSuccess;
}

ncclResult_t netSendConnect(struct ncclConnect* connectInfo, struct ncclConnector* send) {
  // Setup device pointers
  struct netSendResources* resources = (struct netSendResources*)send->transportResources;

  if (resources->useGdr) {
    send->conn.buff = resources->devNetMem->buff;
    // We don't use devMem for llMode because the CPU has to read the data
    send->conn.llBuff = resources->devHostRecvMem->llBuff;
  } else {
    send->conn.buff = resources->devHostRecvMem->buff;
    send->conn.llBuff = resources->devHostRecvMem->llBuff;
  }
  send->conn.tail = &resources->devHostRecvMem->tail;
  send->conn.opCount = &resources->devHostRecvMem->opCount;
  send->conn.fifo = resources->devHostRecvMem->sizesFifo;
  send->conn.llFifo = resources->devHostRecvMem->llSizesFifo;

  if (resources->hostDevMem == NULL) {
    send->conn.head = &resources->devHostSendMem->head;
    send->conn.llHead = &resources->devHostSendMem->llHead;
  }

  // Connect to remote peer
  struct netConnectInfo* info = (struct netConnectInfo*)connectInfo;
  NCCLCHECK(ncclNetConnect(resources->netDev, info->netHandle, &resources->netSendComm));
  return ncclSuccess;
}

/* Connect to this peer */
ncclResult_t netRecvConnect(struct ncclConnect* connectInfo, struct ncclConnector* recv) {
  // Setup device pointers
  struct netRecvResources* resources = (struct netRecvResources*)recv->transportResources;

  recv->conn.head = &resources->devHostSendMem->head;
  recv->conn.llHead = &resources->devHostSendMem->llHead;

  if (resources->useGdr == 0) {
    recv->conn.buff = resources->devHostRecvMem->buff;
    recv->conn.llBuff = resources->devHostRecvMem->llBuff;
  }

  if (resources->hostDevMem == NULL) {
    recv->conn.tail = &resources->devHostRecvMem->tail;
    recv->conn.opCount = &resources->devHostRecvMem->opCount;
  }

  // Finish connection establishment
  NCCLCHECK(ncclNetAccept(resources->netListenComm, &resources->netRecvComm));
  NCCLCHECK(ncclNetCloseListen(resources->netListenComm));

  return ncclSuccess;
}

ncclResult_t netSendFree(void* transportResources) {
  struct netSendResources* resources = (struct netSendResources*)transportResources;
  NCCLCHECK(ncclCudaHostFree(resources->hostSendMem));
  NCCLCHECK(ncclCudaHostFree(resources->hostRecvMem));
  if (resources->useGdr)
    CUDACHECK(cudaFree(resources->devNetMem));
  NCCLCHECK(ncclNetCloseSend(resources->netSendComm));
  free(resources);
  return ncclSuccess;
}

ncclResult_t netRecvFree(void* transportResources) {
  struct netRecvResources* resources = (struct netRecvResources*)transportResources;
  NCCLCHECK(ncclCudaHostFree(resources->hostSendMem));
  NCCLCHECK(ncclCudaHostFree(resources->hostRecvMem));
  NCCLCHECK(ncclNetCloseRecv(resources->netRecvComm));
  free(resources);
  return ncclSuccess;
}

ncclResult_t netSendProxy(struct ncclProxyArgs* args) {
  struct ncclRing* ring = args->ring;
  struct netSendResources* resources = (struct netSendResources*) (ring->send.transportResources);
  const int llMode = args->llMode;

  volatile uint64_t* prevTail = &resources->hostRecvMem->tail;
  struct ncclSendMem* prevMem = resources->hostDevMem ? resources->hostDevMem : resources->hostSendMem;
  uint64_t* prevHead = llMode ? &prevMem->llHead : &prevMem->head;
  struct ncclRecvMem* localMem = resources->useGdr ? resources->devNetMem : resources->hostRecvMem;
  char* localBuff = llMode ? resources->hostRecvMem->llBuff : localMem->buff;
  int ptrType = resources->useGdr ? NCCL_PTR_CUDA : NCCL_PTR_HOST;
  volatile int* sizesFifo = llMode ? resources->hostRecvMem->llSizesFifo : resources->hostRecvMem->sizesFifo;
  int buffSize = llMode ? NCCL_LL_BUFF_SIZE : ring->buffSize;
  int sliceSize = buffSize / args->substeps;

  assert(args->substeps <= SIZES_FIFO_SIZE);

  uint64_t head = llMode ? resources->llStep : 0ULL;
  uint64_t tail = llMode ? resources->llStep : 0ULL;
  uint64_t end = head + args->nsteps;

  int idle = 0;
  void* requests[args->substeps];

  if (!args->needProxy) goto nextColl;

  TRACE(NCCL_NET,"opCount %lx head %lx tail %lx end %lx nsteps %d llMode %d", args->opCount, head, tail, end, args->nsteps, llMode);
  TRACE(NCCL_NET,"opCount %lx buffSize %d sliceSize %d ptrType %d", args->opCount, buffSize, sliceSize, ptrType);

  // Update in case we skipped some collectives
  if (llMode == 0) resources->hostRecvMem->opCount = args->opCount;

  while (head < end) {
    idle++;
    if (llMode) {
      if (tail < end && tail < head + args->substeps) {
        int slot = tail%args->substeps;
        int size = sizesFifo[slot];
        if (size != 0) {
          if (size == -1) size = 0;
          uint32_t flag = tail + 1;
          int nFifoLines = DIVUP(size, sizeof(union ncclLLFifoLine));
          size = nFifoLines * sizeof(union ncclLLFifoLine);
          union ncclLLFifoLine* lines = (union ncclLLFifoLine*)(localBuff+slot*sliceSize);
          for (int i=0; i<nFifoLines; i++) {
            volatile uint32_t *f1 = &lines[i].flag1;
            volatile uint32_t *f2 = &lines[i].flag2;
            while (f1[0] != flag || f2[0] != flag);
          }
          NCCLCHECK(ncclNetIsend(resources->netSendComm, lines, size, ptrType, requests+slot));
          sizesFifo[slot] = size;
          tail++;
          idle = 0;
        }
      }
    } else while (tail < *prevTail) {
        // Send through network
        int slot = tail%args->substeps;
        NCCLCHECK(ncclNetIsend(resources->netSendComm, localBuff+slot*sliceSize, sizesFifo[slot], ptrType, requests+slot));
        tail++;
        idle = 0;
      }
    if (head < tail) {
      int done;
      int slot = head%args->substeps;
      NCCLCHECK(ncclNetTest(requests[slot], &done, NULL));
      if (done) {
        if (llMode) {
          sizesFifo[slot] = 0;
          // Make sure size is reset to zero before we update the head.
          __sync_synchronize();
        }
        head++;
        *prevHead = head;
        idle = 0;
      }
    }
    if (idle) transportProxyIdle(idle);
  }

  // Reset
  if (llMode == 0) *prevTail = 0;

nextColl:
  if (llMode) {
    resources->llStep += args->nsteps;
    // Don't forget to ack otherwise the GPU won't be able to push data.
    *prevHead = resources->llStep;
    if (resources->llStep > resources->llLastCleaning + NCCL_LL_CLEAN_FREQ) {
      memset(localBuff, 0, NCCL_LL_BUFF_SIZE);
      resources->llStep += NCCL_LL_CHUNKS;
      *prevHead = resources->llStep;
      resources->llLastCleaning = resources->llStep;
    }
  }
  return ncclSuccess;
}

ncclResult_t netRecvProxy(struct ncclProxyArgs* args) {
  struct ncclRing* ring = args->ring;
  struct netRecvResources* resources = (struct netRecvResources*) (ring->recv.transportResources);
  int llMode = args->llMode;

  volatile uint64_t* nextHead = llMode ? &resources->hostSendMem->llHead : &resources->hostSendMem->head;
  struct ncclRecvMem* localMem = resources->useGdr ? ring->devMemRecv : resources->hostRecvMem;
  char* localBuff = llMode ? localMem->llBuff : localMem->buff;
  char* nextBuff = (resources->useGdr == 0 && resources->hostDevMem) ? resources->hostDevMem->buff : NULL;
  int ptrType = resources->useGdr ? NCCL_PTR_CUDA : NCCL_PTR_HOST;
  uint64_t* nextTail = resources->hostDevMem ? &resources->hostDevMem->tail : &resources->hostRecvMem->tail;

  int buffSize = llMode ? NCCL_LL_BUFF_SIZE : ring->buffSize;
  int sliceSize = buffSize / args->substeps;

  uint64_t head = llMode ? resources->llStep : 0ULL;
  uint64_t tail = llMode ? resources->llStep : 0ULL;
  uint64_t end = head + args->nsteps;

  int idle = 0;
  void* requests[args->substeps];

  if (!args->needProxy) goto nextColl;

  TRACE(NCCL_NET,"opCount %lx head %lx tail %lx end %lx nsteps %d llMode %d", args->opCount, head, tail, end, args->nsteps, llMode);
  TRACE(NCCL_NET,"opCount %lx buffSize %d sliceSize %d ptrType %d", args->opCount, buffSize, sliceSize, ptrType);

  if (llMode == 0) {
    // Waiting for next opCount is only needed before writing nextTail.
    uint64_t* nextOpCount = resources->hostDevMem ? &resources->hostDevMem->opCount : &resources->hostRecvMem->opCount;
    transportProxyWait([=] { return *nextOpCount >= args->opCount; });
  }

  while (head < end) {
    idle++;
    if ((tail < head + args->substeps) && (tail < *nextHead + args->substeps) && (tail < end)) {
      int slot = tail%args->substeps;
      NCCLCHECK(ncclNetIrecv(resources->netRecvComm, localBuff+slot*sliceSize, sliceSize, ptrType, requests+slot));
      tail++;
      idle = 0;
    }
    if (tail > head) {
      int done;
      int slot = head%args->substeps;
      int size;
      NCCLCHECK(ncclNetTest(requests[slot], &done, &size));
      if (done) {
        if (nextBuff) memcpy(nextBuff+slot*sliceSize, localBuff+slot*sliceSize, size);
        head++;
        if (llMode == 0) {
          if (ptrType == NCCL_PTR_CUDA) ncclNetFlush(resources->netRecvComm, localBuff+slot*sliceSize, size);
          *nextTail = head;
        }
        idle = 0;
      }
    }
    if (idle) transportProxyIdle(idle);
  }

  // Wait for last ack and reset
  if (llMode == 0) {
    transportProxyWait([=] { return *nextHead == head; });
    *nextHead = 0;
  }

nextColl:
  if (llMode) {
    resources->llStep += args->nsteps;
    if (resources->llStep > resources->llLastCleaning + NCCL_LL_CLEAN_FREQ) {
      resources->llStep += NCCL_LL_CHUNKS;
      while (*nextHead < resources->llStep);
      resources->llLastCleaning = resources->llStep;
    }
  }
  return ncclSuccess;
}

struct ncclTransport netTransport = {
  "NET",
  netFillInfo,
  netCanConnect,
  netGetRings,
  { netSendSetup, netSendConnect, netSendFree, netSendProxy },
  { netRecvSetup, netRecvConnect, netRecvFree, netRecvProxy }
};
