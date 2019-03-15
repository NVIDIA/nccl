/*************************************************************************
 * Copyright (c) 2016-2019, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "core.h"
#include "utils.h"
#include "topo.h"
#include "transport.h"
#include "param.h"
#include <unistd.h>
#include <cuda_runtime.h>
#include <ctype.h>
#include "nvlink.h"

struct p2pConnectInfo {
  int direct;
  union {
    void* directPtr;
    cudaIpcMemHandle_t devIpc;
  };
};

struct p2pSendResources {
  struct ncclSendMem* devMem;
  void* ipcPtr;
};

struct p2pRecvResources {
  struct ncclRecvMem* devMem;
  void* ipcPtr;
};

#include <sys/types.h>

NCCL_PARAM(P2pLevel, "P2P_LEVEL", -2);
NCCL_PARAM(P2pDisable, "P2P_DISABLE", -2);

/* Convert a PCI busId string into a local cudaDev device index (cf. CUDA_VISIBLE_DEVICES) */
static int busIdToCudaDev(const char* busId) {
  int ndev;
  if (cudaGetDeviceCount(&ndev) != cudaSuccess)
    return -1;
  for (int i = 0; i < ndev; i++) {
    char devBusId[NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE];
    if (cudaDeviceGetPCIBusId(devBusId, NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE, i) != cudaSuccess)
      return -1;
    if (strcmp(busId, devBusId) == 0) {
      return i;
    }
  }
  // BusId was not found in our locally visible CUDA devices
  return -1;
}

/* Determine if we can communicate with the peer through p2p */
ncclResult_t p2pCanConnect(ncclTvalue_t* ret, struct ncclPeerInfo* myInfo, struct ncclPeerInfo* peerInfo) {
  // Do not use P2P across root complexes by default (provided CUDA permits it)
  int p2pLevel = PATH_NODE;
  if (ncclParamP2pDisable() == 1) p2pLevel = 0;
  if (ncclParamP2pLevel() != -2) p2pLevel = ncclParamP2pLevel();

  *ret = 0;

  if (p2pLevel == 0) return ncclSuccess;

  // Rule out different nodes
  if (myInfo->hostHash != peerInfo->hostHash) return ncclSuccess;

  // Convert the peer's busId into a local cudaDev index (cf. CUDA_VISIBLE_DEVICES)
  int peerCudaDev = busIdToCudaDev(peerInfo->busId);
  if (peerCudaDev == -1) {
    // Peer's CUDA device is not visible in this process
#if CUDART_VERSION >= 10010
    // But in CUDA 10.1 we can still communicate with 'invisible' devices
    TRACE(NCCL_INIT|NCCL_P2P, "Checking P2P connection between %d(%s) and %d(%s)", myInfo->nvmlDev, myInfo->busId, peerInfo->nvmlDev, peerInfo->busId);
    // Check for NVLink/NVswitch including P2P access
    int nvlinkp2p = getNvlinkGpu(myInfo->busId, peerInfo->busId);
    if (nvlinkp2p > 0) {
      *ret = nvlinkp2p;
      return ncclSuccess;
    }
#endif
    return ncclSuccess;
  }

  TRACE(NCCL_INIT|NCCL_P2P, "Checking P2P connection between [%d=%d] and [%d=%d]", myInfo->cudaDev, myInfo->nvmlDev, peerCudaDev, peerInfo->nvmlDev);

  // Do not detect topology if we're on the same GPU. Note this is not really supported.
  if (myInfo->cudaDev == peerCudaDev) {
    *ret = 1 + PATH_SYS;
    return ncclSuccess;
  }

  // See if CUDA can do P2P
  int p2p;
  if (cudaDeviceCanAccessPeer(&p2p, myInfo->cudaDev, peerCudaDev) != cudaSuccess) {
    INFO(NCCL_INIT|NCCL_P2P,"peer query failed between dev %d(=%d) and dev %d(=%d)",
         myInfo->cudaDev, myInfo->nvmlDev, peerCudaDev, peerInfo->nvmlDev);
    return ncclSuccess;
  }
  if (p2p == 0) return ncclSuccess;

  // Check for NVLink/NVswitch
  int nvlinkp2p = getNvlinkGpu(myInfo->busId, peerInfo->busId);
  if (nvlinkp2p > 0) {
    *ret = nvlinkp2p;
    return ncclSuccess;
  }

  // Finally compute the PCI distance and compare with the p2pLevel.
  char* myPath;
  char* peerPath;
  ncclResult_t err1 = getCudaPath(myInfo->cudaDev, &myPath);
  ncclResult_t err2 = getCudaPath(peerCudaDev, &peerPath);
  if (err1 == ncclSuccess && err2 == ncclSuccess) {
    int distance = pciDistance(myPath, peerPath);
    if (distance < p2pLevel) {
      *ret = 1 + PATH_SYS - distance;
    }
  }
  if (err1 == ncclSuccess) free(myPath);
  if (err2 == ncclSuccess) free(peerPath);
  return ncclSuccess;
}

#define MAXGPUS_NVLINKP2P 8 // 16 would take an almost infinite time anyway
#define MAXGPUS_PCI 64

static int computeRingsRec(ncclTvalue_t* matrix, int n, int *rings, int currentRing, int nRingsMax, int* inTheRing, int current, int remaining, int connect) {
  int nrings = 0;
  ncclTvalue_t* line = matrix+current*n;
  inTheRing[current] = 1;
  int currentStep = (currentRing+1)*n-remaining;
  rings[currentStep-1] = current;
  if (remaining == 0) {
    int looprank = rings[currentRing*n];
    if (line[looprank] > 0) {
      if (currentRing+1 == nRingsMax) {
        nrings = 1;
      } else {
        line[looprank]--;
        for (int i=0; i<n; i++) inTheRing[i] = 0;
        if (connect) {
          // First two slots are already set and we need to respect those constraints
          inTheRing[rings[currentStep]] = 1;
          nrings = 1 + computeRingsRec(matrix, n, rings, currentRing+1, nRingsMax, inTheRing, rings[currentStep+1], n-2, connect);
        } else {
          rings[(currentRing+1)*n] = 0;
          nrings = 1 + computeRingsRec(matrix, n, rings, currentRing+1, nRingsMax, inTheRing, 0, n-1, connect);
        }
        line[looprank]++;
        for (int i=0; i<n; i++) inTheRing[i] = 1;
      }
    }
  } else {
    int ringsSave[MAXCHANNELS*MAXGPUS_NVLINKP2P];
    int maxStep = 0;
    for (int i=0; i<n; i++) {
      if (inTheRing[i] == 0 && line[i] > 0) {
        line[i]--;
        int nr = computeRingsRec(matrix, n, rings, currentRing, nRingsMax, inTheRing, i, remaining-1, connect);
        if (nr > nrings) {
          nrings = nr;
          maxStep = (nr+currentRing)*n;
          ringsSave[currentStep] = i;
          // Save the rest of the rings
          for (int r=currentStep+1; r<maxStep; r++) {
            ringsSave[r] = rings[r];
          }
          if (nrings + currentRing == nRingsMax) {
            // We found an optimal solution. Let's stop there.
            break;
          }
        }
        line[i]++;
      }
    }
    for (int r=currentStep; r<maxStep; r++) {
      rings[r] = ringsSave[r];
    }
  }
  inTheRing[current] = 0;
  return nrings;
}

static inline int copyRings(int nranks, int* rings, int nrings, int newNrings) {
  if (nrings == 0) return 0;
  // Copy rings by dup times
  if (newNrings > MAXCHANNELS) {
    newNrings = MAXCHANNELS;
  }
  for (int r=nrings; r<newNrings; r++) {
    for (int i=0; i<nranks; i++) rings[r*nranks+i] = rings[(r%nrings)*nranks+i];
  }
  return newNrings;
}

int p2pComputeRingsNvLink(ncclTvalue_t* matrix, int nranks, int *rings, int nringsMax, int connect) {
  int* inTheRing = (int*)malloc(sizeof(int)*nranks);
  if (inTheRing == NULL) { WARN("malloc of %ld bytes failed", sizeof(int)*nranks); return 0; }
  for (int i=0; i<nranks; i++) inTheRing[i] = 0;
  int nrings;
  if (connect) {
    inTheRing[rings[0]] = 1;
    nrings = computeRingsRec(matrix, nranks, rings, 0, nringsMax, inTheRing, rings[1], nranks-2, connect);
  } else {
    rings[0] = 0;
    nrings = computeRingsRec(matrix, nranks, rings, 0, nringsMax, inTheRing, 0, nranks-1, connect);
  }
  free(inTheRing);
  return nrings;
}

static inline int findConnect(int nranks, int* ranks) {
  for (int i = 0; i<nranks; i++) {
    if (ranks[i] != -1) return i;
  }
  return -1;
}

int p2pComputeRingsNvLink(ncclTvalue_t* values, int nranks, int* rings, int nrings, int* prev, int* next, int oversubscribe, int* nthreads) {
  if (nrings == 0) return 0;
  if (nrings > MAXCHANNELS) {
    WARN("Max rings reached, limiting to %d", MAXCHANNELS);
    nrings = MAXCHANNELS;
  }
  // Find existing constraints / connections
  int connect = 0;
  for (int r=0; r<nrings; r++) {
    int start = findConnect(nranks, prev+r*nranks);
    int end = findConnect(nranks, next+r*nranks);
    if (start != -1 && end != -1) {
      rings[r*nranks] = end;
      rings[r*nranks+1] = start;
      connect = 1;
    }
  }

  // Compute rings
  ncclTvalue_t* matrix = (ncclTvalue_t*)malloc(sizeof(ncclTvalue_t)*nranks*nranks);
  if (matrix == NULL) { WARN("malloc of %ld bytes failed", sizeof(ncclTvalue_t)*nranks*nranks); return 0; }
  for (int i=0; i<nranks; i++) for (int j=0; j<nranks; j++)
      matrix[i*nranks+j] = oversubscribe ? values[i*nranks+j]/CONNECT_NVLINK*2 : values[i*nranks+j]/CONNECT_NVLINK ;

  int compNrings = p2pComputeRingsNvLink(matrix, nranks, rings, nrings, connect);

  free(matrix);

  if (oversubscribe || connect) return compNrings;

  if (compNrings && compNrings < nrings && nranks <= 4) {
    // Try to oversubscribe to get a better result
    int *rings2 = (int *)malloc(sizeof(int)*MAXCHANNELS*nranks);
    if (rings2 == NULL) { WARN("malloc of %ld bytes failed", sizeof(int)*MAXCHANNELS*nranks); return 0; }
    for (int i=0; i<MAXCHANNELS*nranks; i++) rings2[i] = -1;
    int nThreads = *nthreads;
    int compNrings2 = p2pComputeRingsNvLink(values, nranks, rings2, nrings, prev, next, 1, &nThreads);
    if (compNrings2 > compNrings*2) {
      // Oversubscription worked.
      for (int i=0; i<compNrings2*nranks; i++) rings[i] = rings2[i];
      compNrings = compNrings2;
    }
    free(rings2);
  }

  // Duplicate the rings for direct NVLink
  compNrings = copyRings(nranks, rings, compNrings, compNrings*2);

  return compNrings;
}

int p2pComputeRingsSeqConnect(ncclTvalue_t* values, int nranks, int* rings, int nringsStart, int* prev, int* next, int minScore, int* nthreads) {
  int nrings = nringsStart;
  int connect = 0;
  for (int r=0; r<nrings; r++) {
    int start = findConnect(nranks, prev+r*nranks);
    int end = findConnect(nranks, next+r*nranks);
    if (start != -1 && end != -1) {
      rings[r*nranks] = end;
      rings[r*nranks+1] = start;
      int cur = start;
      for (int i=2; i<nranks; i++) {
        int next = (cur+1) % nranks;
        while (next == end || next == start) next = (next+1) % nranks;
        if (values[cur*nranks+next] < minScore) {
          return 0;
        }
        rings[r*nranks+i] = next;
        cur = next;
      }
      connect = 1;
    } else {
      if (connect == 1 && r > 0) {
        WARN("Connecting rings but did not find start/end for ring %d. Disabling other rings.", r);
        return r;
      } else {
        return 0;
      }
    }
  }
  return nrings;
}

int p2pComputeRingsSeqNew(ncclTvalue_t* values, int nranks, int* rings, int nringsStart, int* prev, int* next, int minScore, int* nthreads) {
  for (int r=0; r<nringsStart; r++) {
    for (int i=0; i<nranks; i++) {
      rings[r*nranks+i] = i;
    }
  }
  return nringsStart;
}

static int findClosestPci(ncclTvalue_t* values, int* inRing, int rank, int end, int nranks, int minScore) {
  for (int score = PATH_SYS+1; score >= minScore; score--) {
    int best = -1;
    int worst_end_score = PATH_SYS+2; // find the closest to rank, farthest from end
    for (int n = 0; n < nranks; n++) {
      if (inRing[n]) continue;
      if (values[rank*nranks+n] == score) {
        if (end == -1) return n;
        if (values[end*nranks+n] < worst_end_score) {
          best = n;
          worst_end_score = values[end*nranks+n];
        }
      }
    }
    if (best != -1) return best;
  }
  return -1;
}

int p2pComputeRingsPci(ncclTvalue_t* values, int nranks, int* rings, int nrings, int* prev, int* next, int minScore) {
  int connect = 0;
  for (int r=0; r<nrings; r++) {
    int start = findConnect(nranks, prev+r*nranks);
    int end = findConnect(nranks, next+r*nranks);

    int inRing[MAXGPUS_PCI];
    for (int i=0; i<nranks; i++) inRing[i] = 0;

    if (start == -1 && end == -1) {
      if (connect == 1 && r > 0) {
        WARN("Connecting ring %d : did not find start/end. Disabling other rings.", r);
        return r;
      }
      end = 0;
      inRing[end] = 1;
      start = findClosestPci(values, inRing, end, -1, nranks, minScore);
      if (start == -1) return r;
    } else if (start == -1 || end == -1) {
      WARN("Connecting ring %d : inconsistent start/end. Disabling other rings.", r);
      return r;
    } else {
      connect = 1;
    }
    rings[r*nranks] = end;
    rings[r*nranks+1] = start;
    inRing[start] = inRing[end] = 1;
    int cur = start;
    for (int i=2; i<nranks; i++) {
      int next = findClosestPci(values, inRing, cur, end, nranks, minScore);
      if (next == -1) return r;

      inRing[next] = 1;
      rings[r*nranks+i] = next;
      cur = next;
    }
    // Check the loop is closing
    inRing[end] = 0;
    if (findClosestPci(values, inRing, cur, end, nranks, minScore) != end) return r;

    if (connect == 0) return 1;
  }
  return nrings;
}

ncclResult_t p2pGetRings(int nranks, int* groups, int* subgroups, ncclTvalue_t* values, int* nringsRet, int* prev, int* next, int minScore, int* nthreads) {
  if (*nringsRet == 0) return ncclSuccess;
  int *rings;
  NCCLCHECK(ncclCalloc(&rings, MAXCHANNELS*nranks));
  for (int i=0; i<MAXCHANNELS*nranks; i++) rings[i] = -1;
  int nrings = *nringsRet;

  // NVswitch
  int nvswitchLinks = 0;
  int directLinks = 0;
  for (int rank=0; rank<nranks; rank++) {
    for (int j=1; j<nranks; j++) {
      int i = (rank + j) % nranks;
      ncclTvalue_t links = values[rank*nranks+i]/CONNECT_NVSWITCH;
      if (j>1 && links != nvswitchLinks) {
        WARN("Internal error : NVswitch links mismatch");
        return ncclInternalError;
      }
      nvswitchLinks = links;
    }
  }
  if (nvswitchLinks) {
    // NVSwitch : Connect existing rings
    int nringsConnected = p2pComputeRingsSeqConnect(values, nranks, rings, nrings, prev, next, minScore, nthreads);
    if (nringsConnected > 0) {
      nrings = nringsConnected;
    } else {
      nrings = std::min(nrings, nvswitchLinks); // NVSwitch: Limit rings to number of NVLinks
      // Or create new ones
      nrings = p2pComputeRingsSeqNew(values, nranks, rings, nrings, prev, next, minScore, nthreads);
      // And duplicate them
      nrings = copyRings(nranks, rings, nrings, nrings*2);
    }
    goto end;
  }

  // point-to-point NVLink
  for (int rank=0; rank<nranks; rank++) {
    int links = 0;
    for (int i=0; i<nranks; i++) {
      ncclTvalue_t val = values[rank*nranks+i];
      if (val >= CONNECT_NVSWITCH) continue;
      links += val/CONNECT_NVLINK;
    }
    if (rank == 0) directLinks = links;
    else directLinks = std::min(directLinks, links);
  }
  if (directLinks > 0) {
    // NVLink : Connect rings or create new ones
    if (nranks > MAXGPUS_NVLINKP2P) {
      WARN("Recursive P2P computation cannot work for >8 GPUs");
      return ncclInternalError;
    }
    nrings = p2pComputeRingsNvLink(values, nranks, rings, nrings, prev, next, 0, nthreads);
    goto end;
  }

  // PCIe or QPI : Connect rings or create new ones
  nrings = p2pComputeRingsPci(values, nranks, rings, *nringsRet, prev, next, minScore);

end:
  *nringsRet = nrings;
  for (int ring = 0; ring<nrings; ring++) {
    for (int index=0; index<nranks; index++) {
      int prevIndex = (index - 1 + nranks) % nranks;
      int nextIndex = (index + 1) % nranks;
      int curRank = rings[ring*nranks+index];
      int prevRank = rings[ring*nranks+prevIndex];
      int nextRank = rings[ring*nranks+nextIndex];
      if (prev[ring*nranks+curRank] == -1) prev[ring*nranks+curRank] = prevRank;
      if (next[ring*nranks+curRank] == -1) next[ring*nranks+curRank] = nextRank;
    }
  }

  free(rings);
  return ncclSuccess;
}

#define TRACE_DUMP_IPC(DEVIPC)                                                             \
  do {                                                                                     \
    unsigned long *devIpc = (unsigned long *) (DEVIPC);                                    \
    TRACE(P2P,"IPC: %016lx %016lx %016lx %016lx", devIpc[0], devIpc[1], devIpc[2], devIpc[3]); \
    TRACE(P2P,"IPC: %016lx %016lx %016lx %016lx", devIpc[4], devIpc[5], devIpc[6], devIpc[7]); \
  } while (0)

/* Send: Create and return connect structures for this peer to connect to me */
ncclResult_t p2pSendSetup(struct ncclPeerInfo* myInfo, struct ncclPeerInfo* peerInfo,
    struct ncclConnect* connectInfo, struct ncclConnector* send, int buffSize, int channelId) {

  struct p2pSendResources* resources;
  NCCLCHECK(ncclCalloc(&resources, 1));
  send->transportResources = resources;
  const int sendSize = sizeof(struct ncclSendMem);
  NCCLCHECK(ncclCudaCalloc((char**)&resources->devMem, sendSize));

  struct p2pConnectInfo info;
  if (myInfo->pidHash == peerInfo->pidHash) {
    info.direct = 1;
    info.directPtr = resources->devMem;
    if (myInfo->cudaDev == peerInfo->cudaDev) {
      INFO(NCCL_INIT|NCCL_P2P,"Ring %02d : %d -> %d via P2P/common device", channelId, myInfo->rank, peerInfo->rank);
    } else {
      // Enable P2P access
      cudaError_t err = cudaDeviceEnablePeerAccess(peerInfo->cudaDev, 0);
      if (err == cudaErrorPeerAccessAlreadyEnabled) {
        cudaGetLastError();
      } else if (err != cudaSuccess) {
        WARN("failed to peer with device %d(=%d): %d %s",
             peerInfo->cudaDev, peerInfo->nvmlDev, err, cudaGetErrorString(err));
        return ncclInternalError;
      }
      INFO(NCCL_INIT|NCCL_P2P,"Ring %02d : %d[%d] -> %d[%d] via P2P/direct pointer",
          channelId, myInfo->rank, myInfo->nvmlDev, peerInfo->rank, peerInfo->nvmlDev);
    }
  } else {
    // Convert the peer's busId into a local cudaDev index (cf. CUDA_VISIBLE_DEVICES)
    int peerCudaDev = busIdToCudaDev(peerInfo->busId);
    info.direct = 0;
    // Map IPC and enable P2P access
    cudaError_t err = cudaIpcGetMemHandle(&info.devIpc, (void*)resources->devMem);
    if (err != cudaSuccess) {
      WARN("rank %d failed to get CUDA IPC handle to device %d(=%d) : %d %s",
           myInfo->rank, peerCudaDev, peerInfo->nvmlDev, err, cudaGetErrorString(err));
      return ncclInternalError;
    }
    INFO(NCCL_INIT|NCCL_P2P,"Ring %02d : %d[%d] -> %d[%d] via P2P/IPC",
        channelId, myInfo->rank, myInfo->nvmlDev, peerInfo->rank, peerInfo->nvmlDev);
    //TRACE_DUMP_IPC(&info.devIpc);
  }
  static_assert(sizeof(struct p2pConnectInfo) <= sizeof(struct ncclConnect), "p2p Connect Info is too big");
  memcpy(connectInfo, &info, sizeof(struct p2pConnectInfo));
  return ncclSuccess;
}

/* Create and return connect structures for this peer to connect to me */
ncclResult_t p2pRecvSetup(struct ncclPeerInfo* myInfo, struct ncclPeerInfo* peerInfo,
    struct ncclConnect* connectInfo, struct ncclConnector * recv, int buffSize, int channelId) {

  struct p2pRecvResources* resources;
  NCCLCHECK(ncclCalloc(&resources, 1));
  recv->transportResources = resources;
  const int recvSize = offsetof(struct ncclRecvMem, buff)+buffSize;
  NCCLCHECK(ncclCudaCalloc((char**)&resources->devMem, recvSize));

  struct p2pConnectInfo info;
  if (myInfo->pidHash == peerInfo->pidHash) {
    info.direct = 1;
    info.directPtr = resources->devMem;
    if (myInfo->cudaDev == peerInfo->cudaDev) {
      TRACE(NCCL_INIT|NCCL_P2P,"%d <- %d via P2P/common device", myInfo->rank, peerInfo->rank);
    } else {
      // Enable P2P access
      cudaError_t err = cudaDeviceEnablePeerAccess(peerInfo->cudaDev, 0);
      if (err == cudaErrorPeerAccessAlreadyEnabled) {
        cudaGetLastError();
      } else if (err != cudaSuccess) {
        WARN("failed to peer with device %d(=%d): %d %s",
             peerInfo->cudaDev, peerInfo->nvmlDev, err, cudaGetErrorString(err));
        return ncclInternalError;
      }
      TRACE(NCCL_INIT|NCCL_P2P,"Ring %02d : %d[%d] <- %d[%d] via P2P/direct pointer", channelId, myInfo->rank, myInfo->nvmlDev, peerInfo->rank, peerInfo->nvmlDev);
    }
  } else {
    // Convert the peer's busId into a local cudaDev index (cf. CUDA_VISIBLE_DEVICES)
    int peerCudaDev = busIdToCudaDev(peerInfo->busId);
    info.direct = 0;
    // Map IPC and enable P2P access
    cudaError_t err = cudaIpcGetMemHandle(&info.devIpc, (void*)resources->devMem);
    if (err != cudaSuccess) {
      WARN("rank %d failed to get CUDA IPC handle to device %d(=%d) : %d %s",
           myInfo->rank, peerCudaDev, peerInfo->nvmlDev, err, cudaGetErrorString(err));
      return ncclInternalError;
    }
    TRACE(NCCL_INIT|NCCL_P2P,"Ring %02d : %d[%d] <- %d[%d] via P2P/IPC", channelId, myInfo->rank, myInfo->nvmlDev, peerInfo->rank, peerInfo->nvmlDev);
    //TRACE_DUMP_IPC(&info.devIpc);
  }
  static_assert(sizeof(struct p2pConnectInfo) <= sizeof(struct ncclConnect), "p2p Connect Info is too big");
  memcpy(connectInfo, &info, sizeof(struct p2pConnectInfo));
  return ncclSuccess;
}

/* Connect/Send to this peer */
static ncclResult_t p2pSendConnect(struct ncclConnect* connectInfo, struct ncclConnector* send) {
  struct p2pSendResources* resources = (struct p2pSendResources*)send->transportResources;
  struct ncclRecvMem* remDevMem;
  struct p2pConnectInfo* info = (struct p2pConnectInfo*)connectInfo;
  if (info->direct) {
    remDevMem = (struct ncclRecvMem*)(info->directPtr);
    send->conn.direct = 1;
  } else {
    //TRACE_DUMP_IPC(&info->devIpc);
    cudaError_t err = cudaIpcOpenMemHandle(&resources->ipcPtr, info->devIpc, cudaIpcMemLazyEnablePeerAccess);
    remDevMem = (struct ncclRecvMem*)resources->ipcPtr;
    if (err != cudaSuccess) {
      WARN("failed to open CUDA IPC handle : %d %s",
          err, cudaGetErrorString(err));
      return ncclUnhandledCudaError;
    }
  }

  send->conn.buff = remDevMem->buff;
  send->conn.llBuff = remDevMem->llBuff;
  send->conn.tail = &remDevMem->tail;
  send->conn.opCountRem = &remDevMem->opCount;
  send->conn.head = &resources->devMem->head;
  send->conn.ptrExchange = &resources->devMem->ptrExchange;
  send->conn.opCountLoc = &resources->devMem->opCount;
  return ncclSuccess;
}

/* Connect/Recv from this peer */
ncclResult_t p2pRecvConnect(struct ncclConnect* connectInfo, struct ncclConnector* recv) {
  struct p2pRecvResources* resources = (struct p2pRecvResources*)recv->transportResources;
  struct ncclSendMem* remDevMem;
  struct p2pConnectInfo* info = (struct p2pConnectInfo*)connectInfo;
  if (info->direct) {
    remDevMem = (struct ncclSendMem*)(info->directPtr);
    recv->conn.direct = 1;
    recv->conn.ptrExchange = &remDevMem->ptrExchange;
  } else {
    //TRACE_DUMP_IPC(&info->devIpc);
    cudaError_t err = cudaIpcOpenMemHandle(&resources->ipcPtr, info->devIpc, cudaIpcMemLazyEnablePeerAccess);
    remDevMem = (struct ncclSendMem*)resources->ipcPtr;
    if (err != cudaSuccess) {
      WARN("failed to open CUDA IPC handle : %d %s",
          err, cudaGetErrorString(err));
      return ncclUnhandledCudaError;
    }
  }

  recv->conn.buff = resources->devMem->buff;
  recv->conn.llBuff = resources->devMem->llBuff;
  recv->conn.tail = &resources->devMem->tail;
  recv->conn.opCountLoc = &resources->devMem->opCount;
  recv->conn.head = &remDevMem->head;
  recv->conn.opCountRem = &remDevMem->opCount;
  return ncclSuccess;
}

ncclResult_t p2pSendFree(void* resources) {
  struct p2pSendResources* sendRes = (struct p2pSendResources*)resources;
  if (sendRes->ipcPtr)
    CUDACHECK(cudaIpcCloseMemHandle(sendRes->ipcPtr));
  CUDACHECK(cudaFree(sendRes->devMem));
  free(sendRes);
  return ncclSuccess;
}

ncclResult_t p2pRecvFree(void* resources) {
  struct p2pRecvResources* recvRes = (struct p2pRecvResources*)resources;
  if (recvRes->ipcPtr)
    CUDACHECK(cudaIpcCloseMemHandle(recvRes->ipcPtr));
  CUDACHECK(cudaFree(recvRes->devMem));
  free(recvRes);
  return ncclSuccess;
}

struct ncclTransport p2pTransport = {
  "P2P",
  p2pCanConnect,
  p2pGetRings,
  { p2pSendSetup, p2pSendConnect, p2pSendFree, NULL },
  { p2pRecvSetup, p2pRecvConnect, p2pRecvFree, NULL }
};
