/*************************************************************************
 * Copyright (c) 2015, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 ************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include "core.h"
#include "libwrap.h"
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sched.h>
#include <fcntl.h>
#include <unistd.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <string.h>
#include <errno.h>

DebugLevel ncclDebugLevel;

extern "C" DSOGLOBAL
ncclResult_t ncclGetUniqueId(ncclUniqueId* out) {
  pid_t pid = getpid();
  static int count = 0;
  int commId = __sync_fetch_and_add(&count, 1);
  int len = snprintf(out->internal, NCCL_UNIQUE_ID_BYTES, "nccl-%d-%d", pid, commId);
  if(strlen(out->internal) < len) {
    WARN("ncclUniqueId truncated");
    return ncclInternalError;
  }
  return ncclSuccess;
}


static ncclResult_t shmOpen(const char* shmname, size_t bytes, void** ptr) {
  int fd = shm_open(shmname, O_CREAT | O_RDWR, S_IRUSR | S_IWUSR);
  if (fd == -1) {
    WARN("shm_open failed to open %s", shmname);
    return ncclSystemError;
  }

  if (ftruncate(fd, bytes) == -1) {
    WARN("ftruncate failed to allocate %ld bytes", bytes);
    shm_unlink(shmname);
    close(fd);
    return ncclSystemError;
  }

  *ptr = mmap(NULL, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  if (*ptr == MAP_FAILED) {
    WARN("failure in mmap");
    shm_unlink(shmname);
    close(fd);
    return ncclSystemError;
  }

  close(fd);
  return ncclSuccess;
}

static ncclResult_t shmUnlink(const char* shmname) {
  if(shm_unlink(shmname) == -1) {
    WARN("smh_unlink failed");
    return ncclSystemError;
  } else {
    return ncclSuccess;
  }
}

static ncclResult_t shmUnmap(void* ptr, size_t bytes) {
  if(munmap(ptr, bytes) == -1) {
    WARN("munmap failed");
    return ncclSystemError;
  } else {
    return ncclSuccess;
  }
}


typedef struct {
  int rank;
  int ndev;
  int cudaDev;
  int ncclId;
  pid_t pid;
  ncclMem* hostptr;
  ncclMem* devptr;
  CUipcMemHandle devipc;
  size_t buffSize;
} RankEntry;

static int compRanks(const void* a, const void* b) {
  const RankEntry* A = (const RankEntry*)a;
  const RankEntry* B = (const RankEntry*)b;
  if (A->ncclId < B->ncclId) return -1;
  if (A->ncclId > B->ncclId) return  1;
  return 0;
}

static void orderRanks(RankEntry* ranks, int count) {
  qsort(ranks, count, sizeof(RankEntry), compRanks);
  for(int i=0; i<count; ++i)
    ranks[i].ncclId = i;
}


typedef struct {
  union {
    struct {
      volatile int bar;
    };
    char pad[16];
   };
   RankEntry ranks[1];
} RankGather;

static ncclResult_t initGather(RankGather** gather, ncclUniqueId commId, 
    int ndev, int rank, RankEntry myInfo) {
  size_t bytes = offsetof(RankGather, ranks) + ndev*sizeof(RankEntry);
  RankGather* tmp = NULL;
  int bar_tmp;

  ncclResult_t res = shmOpen(commId.internal, bytes, (void**)&tmp);
  if (res != ncclSuccess) {
    WARN("rank %d failed to open shm segment for gather", rank);
    return res;
  }

  tmp->ranks[rank] = myInfo;

  bar_tmp = tmp->bar - 1;
  bool swapped;
  do {
    bar_tmp += 1;
    if (bar_tmp == ndev-1) { // everyone is done
      ncclResult_t res = shmUnlink(commId.internal);
      if (res != ncclSuccess) {
        WARN("rank %d failed to unlink shm segment for gather", rank);
        shmUnmap(tmp, bytes);
        return res;
      }
      
      orderRanks(tmp->ranks, ndev);
    }
    swapped = __sync_bool_compare_and_swap(&tmp->bar, bar_tmp, bar_tmp+1);
  } while(!swapped);

  while (tmp->bar < ndev)
    sched_yield();
  __sync_synchronize();

  *gather = tmp;
  return ncclSuccess;
}

static ncclResult_t closeGather(RankGather* gather, int ndev) {
  int bar_tmp = gather->bar - 1;
  bool swapped;
  do {
    bar_tmp += 1;
    swapped = __sync_bool_compare_and_swap(&gather->bar, bar_tmp, bar_tmp+1);
  } while(!swapped);

  while (gather->bar != 2*ndev)
    sched_yield();
  __sync_synchronize();

  size_t bytes = offsetof(RankGather, ranks) + ndev*sizeof(RankEntry);
  ncclResult_t res = shmUnmap(gather, bytes);
  if (res != ncclSuccess) {
    WARN("failed to unmap %ld bytes of gather", bytes);
    return res;
  }

  return ncclSuccess;
}


static ncclResult_t allocDevMem(ncclMem** ptr, size_t buffSize) {
  size_t size = offsetof(struct ncclMem, buff) + buffSize;
  cudaError_t res = cudaMalloc((void**)ptr, size);
  if (res != cudaSuccess) {
    *ptr = NULL;
    WARN("failed to allocate %lu byte device buffer", size);
    return ncclCudaMallocFailed;
  }
  if (cudaMemset(*ptr, 0, size) != cudaSuccess) {
    WARN("failed to memset device buffer.");
    cudaFree(*ptr);
    *ptr = NULL;
    return ncclUnhandledCudaError;
  }
  return ncclSuccess;
}

static const int ShmMapped = 1;
static const int ShmLinked = 2;

static ncclResult_t allocHostMem(ncclMem** ptr, size_t buffSize) {
  size_t size = offsetof(struct ncclMem, buff) + buffSize;
  cudaError_t res = cudaMallocHost((void**)ptr, size);
  if (res != cudaSuccess) {
    *ptr = NULL;
    WARN("failed to allocate %lu byte host buffer", size);
    return ncclSystemError;
  }
  memset(*ptr, 0, size);
  return ncclSuccess;
}

static ncclResult_t openHostMemShm(const char* shmname, ncclMem** ptr, size_t buffSize) {
  size_t size = offsetof(struct ncclMem, buff) + buffSize;
  ncclResult_t res = shmOpen(shmname, size, (void**)ptr);
  if (res != ncclSuccess) {
    WARN("failed to allocate %lu byte shm buffer", size);
    *ptr = NULL;
    return res;
  }

  if(cudaHostRegister(*ptr, size, cudaHostRegisterMapped) != cudaSuccess) {
    WARN("failed to register host buffer");
    shmUnlink(shmname);
    shmUnmap(*ptr, size);
    *ptr = NULL;
    return ncclUnhandledCudaError;
  }
  return ncclSuccess;
}

static ncclResult_t populateRankInfo(RankEntry* info, int rank, ncclComm_t comm) {
  char busId[13];
  nvmlDevice_t nvmlHandle;
  cudaError_t res = cudaDeviceGetPCIBusId(busId, 13, comm->cudaDev);
  if (res == cudaErrorInvalidDevice) {
    WARN("rank %d attempted to access an invalid cuda device %d", rank, comm->cudaDev);
    return ncclInvalidDeviceIndex;
  } else if (res != cudaSuccess) {
    WARN("rank %d failed to get PCI Bus Id for device %d", rank, comm->cudaDev);
    return ncclUnhandledCudaError;
  }
  INFO("rank %d using device %d (%s)", rank, comm->cudaDev, busId);
    
  if (wrapNvmlDeviceGetHandleByPciBusId(busId, &nvmlHandle) != ncclSuccess) {
    WARN("rank %d failed to get nvml handle for device %s", rank, busId);
    return ncclUnhandledCudaError;
  }
  // Order by nvml index
  if (wrapNvmlDeviceGetIndex(nvmlHandle, (unsigned*)&info->ncclId) != ncclSuccess) {
    WARN("rank %d failed to get nvml device index for device %d", rank, comm->cudaDev);
    return ncclUnhandledCudaError;
  }

  info->rank = rank;
  info->ndev = comm->nDev;
  info->cudaDev = comm->cudaDev;
  info->pid = getpid();
  info->buffSize = comm->buffSize;
  info->hostptr = comm->hostMem;
  info->devptr = comm->devMem;
  if (wrapCuIpcGetMemHandle(&info->devipc, (CUdeviceptr)comm->devMem) != ncclSuccess) {
    WARN("rank %d failed to open CUDA IPC handle", rank);
    return ncclUnhandledCudaError;
  }

  return ncclSuccess;
}


static const int CLEANUP_NONE  = 0;
static const int CLEANUP_CUIPC = 1;
static const int CLEANUP_UNMAP = 2;

static ncclResult_t commClearMaps(ncclComm_t comm) {
  ncclResult_t res, retval = ncclSuccess;
  cudaError_t cures;

  for(int d=0; d<comm->nDev; ++d) {
    switch(comm->cleanup[d].type) {
      case CLEANUP_NONE:
        break;
      case CLEANUP_CUIPC:
        res = wrapCuIpcCloseMemHandle((CUdeviceptr)comm->cleanup[d].handle);
        if (res != ncclSuccess) {
          WARN("rank %d failed to close IPC handle to rank %d", 
            comm->userFromRing[comm->ncclId], comm->userFromRing[d]);
          retval = (retval == ncclSuccess) ? res : retval;
        }
        break;
      case CLEANUP_UNMAP:
        cures = cudaHostUnregister(comm->cleanup[d].handle);
        if (cures != cudaSuccess) {
          WARN("rank %d failed to unregister handle to rank %d",
            comm->userFromRing[comm->ncclId], comm->userFromRing[d]);
            retval = (retval == ncclSuccess) ? ncclUnhandledCudaError : retval;
        }
        res = shmUnmap(comm->cleanup[d].handle, offsetof(ncclMem, buff) + comm->buffSize);
        if (res != ncclSuccess) {
          WARN("rank %d failed to unmap handle to rank %d",
            comm->userFromRing[comm->ncclId], comm->userFromRing[d]);
            retval = (retval == ncclSuccess) ? res : retval;
        }
        break;
      default:
        WARN("Unknown cleanup type %d", comm->cleanup[d].type);
    }
    comm->cleanup[d].type = 0;
    comm->cleanup[d].handle = NULL;
  }

  memset(comm->userFromRing, 0, sizeof(int)*MAXPEERS);
  memset(comm->ringFromUser, 0, sizeof(int)*MAXPEERS);
  if (comm->devUserFromRing != NULL) {
    cudaError_t err = cudaMemset(comm->devUserFromRing, 0, sizeof(int)*MAXPEERS);
    if (err != cudaSuccess) {
      WARN("Faild to clear dev map: %s", cudaGetErrorString(err));
      retval = (retval == ncclSuccess) ? ncclUnhandledCudaError : retval;
    }
  }
  return retval;
}

static ncclResult_t commBuildMaps(ncclComm_t comm, ncclUniqueId* commId, int rank, RankEntry* ranks) {
  int ndev = comm->nDev;
  for(int i=0; i<ndev; ++i) {
    // Check for inconsistencies between ranks
    // If two ranks use the same rank, then one slot of
    // ranks[] will be left unset with zero ndev/buffSize.
    if (ranks[i].buffSize != comm->buffSize
        || ranks[i].ndev != comm->nDev) {
      commClearMaps(comm);
      return ncclRankMismatch;
    }

    // Create rank<->nccl maps
    int iRank = ranks[i].rank;
    comm->userFromRing[i] = iRank;
    comm->ringFromUser[iRank] = i;
  }

  if (cudaMemcpy(comm->devUserFromRing, comm->userFromRing, ndev*sizeof(int),
      cudaMemcpyHostToDevice) != cudaSuccess) {
    WARN("rank %d failed to copy maps to device", rank);
    commClearMaps(comm);
    return ncclUnhandledCudaError;
  }

  int myId = -1;
  for (int i=0; i<ndev; ++i) {
    if(ranks[i].rank == rank) {
      myId = i;
      break;
    }
  }

  if (myId == -1) {
    WARN("rank %d not found in communicator", rank);
    return ncclInvalidRank;
  }
  comm->ncclId = myId;
  
  int myDev = ranks[myId].cudaDev;
  pid_t myPid = ranks[myId].pid;
  comm->useRemoteRecv = 1; // Assume we directly write to result ptrs.

  for (int i=0; i<ndev; ++i) {
    int iRank = ranks[i].rank;
    int iDev = ranks[i].cudaDev;
    pid_t iPid = ranks[i].pid;
    int canpeer = 0;

    if (cudaDeviceCanAccessPeer(&canpeer, myDev, iDev) != cudaSuccess) {
      INFO("peer query failed between rank %d (dev %d) and rank %d (dev %d)",
        rank, myDev, iRank, iDev);
      canpeer = 0;
    }

    if (canpeer) {
      cudaError_t err;
      err = cudaDeviceEnablePeerAccess(iDev, 0);
      if (err == cudaErrorPeerAccessAlreadyEnabled) {
        cudaGetLastError();
      } else if (err != cudaSuccess) {
        INFO("peer access failed between rank %d (dev %d) and rank %d (dev %d)\n",
          rank, myDev, iRank, iDev);
        
        canpeer = 0;
      }
    }

    if (iPid == myPid && (canpeer || myDev == iDev)) {
      INFO("rank access %d -> %d via P2P device mem", rank, iRank);
      comm->local[i] = ranks[myId].devptr;
      comm->remote[i] = ranks[i].devptr;
      comm->cleanup[i].type = CLEANUP_NONE;
    } else if (iPid == myPid) {
      INFO("rank access %d -> %d via zero-copy host mem", rank, iRank);
      comm->useRemoteRecv = 0;
      if (cudaHostGetDevicePointer(comm->local+i, ranks[myId].hostptr, 0) != cudaSuccess) {
        WARN("rank %d failed to map zero copy buffer to device", rank);
        commClearMaps(comm);
        return ncclUnhandledCudaError;
      }
      if (cudaHostGetDevicePointer(comm->remote+i, ranks[i].hostptr, 0) != cudaSuccess) {
        WARN("rank %d failed to map %d's zero copy buffer to device", rank, iRank);
        commClearMaps(comm);
        return ncclUnhandledCudaError;
      }
      comm->cleanup[i].type = CLEANUP_NONE;
    } else if (canpeer || myDev == iDev) {
      INFO("rank access %d -> %d via Ipc P2P device mem", rank, iRank);
      comm->useRemoteRecv = 0;
      comm->local[i] = ranks[myId].devptr;
      if (wrapCuIpcOpenMemHandle((CUdeviceptr*)(&comm->remote[i]),
          ranks[i].devipc, CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS) != ncclSuccess) {
        WARN("rank %d failed to open Ipc handle to rank %d", rank, iRank);
        commClearMaps(comm);
        return ncclUnhandledCudaError;
      }
      comm->cleanup[i].type = CLEANUP_CUIPC;
      comm->cleanup[i].handle = comm->remote[i];
    } else {
      INFO("rank access %d -> %d via zero copy host shm", rank, iRank);
      comm->useRemoteRecv = 0;
      if (cudaHostGetDevicePointer(comm->local+i, ranks[myId].hostptr, 0) != cudaSuccess) {
        WARN("rank %d failed to obtain dev ptr to sysmem buffer", rank);
        commClearMaps(comm);
        return ncclUnhandledCudaError;
      }
      char rankname[1024];
      sprintf(rankname, "%s-%d", commId->internal, ranks[i].rank);
      if (openHostMemShm(rankname, (ncclMem**)&comm->cleanup[i].handle, ranks[i].buffSize)
          != ncclSuccess) {
        WARN("rank %d failed to open sysmem buffer of rank %d", rank, iRank);
        commClearMaps(comm);
        return ncclUnhandledCudaError;
      }
      if (cudaHostGetDevicePointer(comm->remote+i, comm->cleanup[i].handle, 0) != cudaSuccess) {
        WARN("rank %d failed to obtain dev ptr for rank %d", rank, iRank);
        commClearMaps(comm);
        return ncclUnhandledCudaError;
      }
      comm->cleanup[i].type = CLEANUP_UNMAP;
    }
  }
  INFO("PushToRecv algos are %s\n", (comm->useRemoteRecv) ? "enabled" : "disabled");
  return ncclSuccess;
}

static void initDebug() {
  const char* nccl_debug = getenv("NCCL_DEBUG");
  if (nccl_debug == NULL) {
    ncclDebugLevel = NONE;
  } else if (strcmp(nccl_debug, "WARN") == 0) {
    ncclDebugLevel = WARN;
  } else if (strcmp(nccl_debug, "INFO") == 0) {
    ncclDebugLevel = INFO;
    INFO("NCCL debug level set to INFO");
  } else if (strcmp(nccl_debug, "ABORT") == 0) {
    ncclDebugLevel = ABORT;
    INFO("NCCL debug level set to ABORT");
  }

}

static void commFree(ncclComm_t comm) {
  if (comm == NULL)
    return;

  for(int i=0; i<MAXQUEUE; ++i) {
    if (comm->events.isDone[i] != NULL)
      if (cudaEventDestroy(comm->events.isDone[i]) != cudaSuccess)
        INFO("failed to destroy cuda event %d", i);
  }

  ncclResult_t res = commClearMaps(comm);
  if (res != ncclSuccess)
    INFO("failed to cleanup comm maps");

  if (comm->devUserFromRing != NULL)
    if (cudaFree(comm->devUserFromRing) != cudaSuccess)
      INFO("commFree failed to free dev maps");

  if (comm->devMem != NULL && cudaFree(comm->devMem) != cudaSuccess)
    INFO("Failed to free devMap");

  if (comm->hostMem != NULL) {
    if (comm->hostMemState & ShmMapped) {
      if (cudaHostUnregister(comm->hostMem) != cudaSuccess)
        INFO("Failed to unregister hostMem");
      size_t size = offsetof(ncclMem, buff) + comm->buffSize;
      if (shmUnmap(comm->hostMem, size) != ncclSuccess)
        INFO("Failed to unmap hostMem");
      comm->hostMemState ^= ShmMapped;
    } else {
      cudaFreeHost(comm->hostMem);
    }
  }
  free(comm);
}

static ncclResult_t commAlloc(ncclComm_t* comret, int ndev, const ncclUniqueId* commId, int rank) {
  if (ndev < 1 || ndev > MAXPEERS) {
    WARN("requested device count (%d) exceeds maximum of %d", ndev, MAXPEERS);
    return ncclUnsupportedDeviceCount;
  }
  if (rank >= ndev || rank < 0) {
    WARN("rank %d exceeds ndev=%d", rank, ndev);
    return ncclInvalidRank;
  }

  struct ncclComm* comm = (struct ncclComm*)malloc(sizeof(struct ncclComm));
  if (comm == NULL) {
    WARN("comm allocation failed");
    return ncclSystemError;
  }
  memset(comm, 0, sizeof(struct ncclComm));

  comm->nDev = ndev;
  cudaGetDevice(&comm->cudaDev);

  const char* str = getenv("NCCL_BUFFSIZE");
  if (str != NULL) {
    errno = 0;
    comm->buffSize = strtol(str, NULL, 10);
    if (errno == ERANGE || comm->buffSize == 0) {
      INFO("rank %d invalid NCCL_BUFFSIZE: %s, using default %lu",
          rank, str, DEFAULT_BUFFER_SIZE_BYTES);
      comm->buffSize = DEFAULT_BUFFER_SIZE_BYTES;
    }
  } else {
    comm->buffSize = DEFAULT_BUFFER_SIZE_BYTES;
  }
  INFO("rank %d using buffSize = %lu", rank, comm->buffSize);


  ncclResult_t res;
  res = allocDevMem(&comm->devMem, comm->buffSize);
  if (res != ncclSuccess) {
    WARN("rank %d failed to allocate device buffer", rank);
    commFree(comm);
    return res;
  }
  if (cudaMalloc(&comm->devUserFromRing, MAXPEERS*sizeof(int)) != cudaSuccess) {
    WARN("rank %d failed to allocated device maps", rank);
    commFree(comm);
    return ncclCudaMallocFailed;
  }

  EventQueue* eq = &comm->events;
  for(int i=0; i<MAXQUEUE; ++i) {
    if (cudaEventCreateWithFlags(eq->isDone+i, cudaEventDisableTiming) != cudaSuccess) {
      WARN("rank %d failed to create nccl event %d", rank, i);
      commFree(comm);
      return ncclUnhandledCudaError;
    }
  }

  if(commId == NULL) {
    comm->hostMemState = 0;
    res = allocHostMem(&comm->hostMem, comm->buffSize);
  } else {
    char rankname[1024];
    sprintf(rankname, "%s-%d", commId->internal, rank);
    res = openHostMemShm(rankname, &comm->hostMem, comm->buffSize);
    if (res != ncclSuccess) {
      WARN("rank %d failed to allocate host buffer", rank);
      commFree(comm);
      return res;
    }
    comm->hostMemState = ShmMapped | ShmLinked;
  }

  *comret = comm;
  return ncclSuccess;
}

static ncclResult_t commUnlinkHostMem(ncclComm_t comm, ncclUniqueId commId, int rank) {
  char rankname[1024];
  sprintf(rankname, "%s-%d", commId.internal, rank);
  if (comm->hostMemState & ShmLinked)
    comm->hostMemState ^= ShmLinked;
  return shmUnlink(rankname);
}

extern "C" DSOGLOBAL
ncclResult_t ncclCommInitRank(ncclComm_t* newcomm, int ndev, ncclUniqueId commId, int myrank) {
  if (strlen(commId.internal) < 1 || 
      strlen(commId.internal) >= NCCL_UNIQUE_ID_BYTES) {
    WARN("rank %d invalid commId", myrank);
    return ncclInvalidArgument;
  }

  initDebug();
  ncclResult_t res;
  RankEntry myStuff;
  RankGather* gath = NULL;

  res = wrapSymbols();
  if (res != ncclSuccess) {
    WARN("NCCL failed to initialize client libs");
    return res;
  }

  res = wrapNvmlInit();
  if (res != ncclSuccess) {
    WARN("rank %d failed to initialize nvml", myrank);
    return res;
  }

  res = commAlloc(newcomm, ndev, &commId, myrank);
  if (res != ncclSuccess) {
    WARN("rank %d failed to allocate communicator", myrank);
    return res;
  }

  res = populateRankInfo(&myStuff, myrank, *newcomm);
  if (res != ncclSuccess) {
    WARN("rank %d failed to obtain rank info", myrank);
    goto cleanup;
  }

  res = initGather(&gath, commId, ndev, myrank, myStuff);
  if (res != ncclSuccess) {
    WARN("rank %d failed to gather rank info", myrank);
    goto cleanup;
  }

  res = commBuildMaps(*newcomm, &commId, myrank, gath->ranks);
  if (res != ncclSuccess) {
    WARN("rank %d failed to build comm maps", myrank);
    goto cleanup;
  }

  res = closeGather(gath, ndev); // includes a barrier
  gath = NULL;
  if (res != ncclSuccess) {
    WARN("rank %d failed to close gather", myrank);
    goto cleanup;
  }

  goto final;

  cleanup:
  if (gath != NULL)
    closeGather(gath, ndev);
  commFree(*newcomm);

  final:
  if ((*newcomm)->hostMemState & ShmLinked) {
    if (commUnlinkHostMem(*newcomm, commId, myrank) != ncclSuccess)
      INFO("rank %d failed to unlink host mem shm segment", myrank);
  }
  
  if (wrapNvmlShutdown() != ncclSuccess)
    INFO("rank %d did not shutdown nvml properly", myrank);
  return res;
}

extern "C" DSOGLOBAL
ncclResult_t ncclCommInitAll(ncclComm_t* comms, int ndev, int* devlist) {
  initDebug();

  ncclResult_t res;
  int savedDevice;
  RankEntry* ranks = NULL;
  int rank, cudaDev;
  ncclComm_t comm = NULL;
  char busId[13];
  nvmlDevice_t nvmlHandle;
  int affinity_set = 0;

  res = wrapSymbols();
  if (res != ncclSuccess) {
    WARN("NCCL failed to initialize client libs");
    return res;
  }

  cudaGetDevice(&savedDevice);
  ranks = (RankEntry*)malloc(ndev*sizeof(RankEntry));
  if (ranks == NULL) {
    WARN("NCCL allocation failed");
    return ncclSystemError;
  }
  memset(ranks, 0, ndev*sizeof(RankEntry));

  res = wrapNvmlInit();
  if (res != ncclSuccess) {
    WARN("nccl failed to initialize nvml");
    return res;
  }

  for(rank=0; rank<ndev; ++rank)
    comms[rank] = NULL;

  for (rank=0; rank<ndev; ++rank) {
    cudaDev = (devlist == NULL) ? rank : devlist[rank];
    if (cudaSetDevice(cudaDev) != cudaSuccess) {
      WARN("rank %d failed to set cuda device %d", rank, cudaDev);
      res = ncclInvalidDeviceIndex;
      goto cleanup;
    }

    // Set CPU affinity
    affinity_set = 0;
    if (cudaDeviceGetPCIBusId(busId, 13, cudaDev) != cudaSuccess) {
      INFO("rank %d failed to get PCI Bus Id for device %d", rank, cudaDev);
      goto skipaffinity;
    }
    if (wrapNvmlDeviceGetHandleByPciBusId(busId, &nvmlHandle) != ncclSuccess) {
      INFO("rank %d failed to get nvml handle for device %s", rank, busId);
      goto skipaffinity;
    }
    if (wrapNvmlDeviceSetCpuAffinity(nvmlHandle) != ncclSuccess) {
      INFO("rank %d failed to set affinity", rank);
      goto skipaffinity;
    }
    affinity_set = 1; 
    skipaffinity: 

    res = commAlloc(&comm, ndev, NULL, rank);
    if (res != ncclSuccess) {
      WARN("rank %d failed to allocate communicator", rank);
      goto cleanup;
    }
    comms[rank] = comm;

    if (affinity_set && wrapNvmlDeviceClearCpuAffinity(nvmlHandle) != ncclSuccess) {
      INFO("rank %d set but failed to clear cpu affinity", rank);
    }
    res = populateRankInfo(ranks+rank, rank, comm);
    if (res != ncclSuccess) {
      WARN("rank %d failed to obtain rank info", rank);
      goto cleanup;
    }
  }

  orderRanks(ranks, ndev);
  for(rank=0; rank<ndev; ++rank) {
    comm = comms[rank];
    cudaSetDevice(comm->cudaDev);
    res = commBuildMaps(comm, NULL, rank, ranks);
    if (res != ncclSuccess) {
      WARN("rank %d failed to build comm maps", rank);
      goto cleanup;
    }
  }

  free(ranks);
  ranks = NULL;
  res = ncclSuccess;
  goto final;

  cleanup:
  if (ranks != NULL)
    free(ranks);
  for(rank=0; rank<ndev; ++rank) {
    if(comms[rank] != NULL) {
      commFree(comms[rank]);
    }
  }

  final:
  if(wrapNvmlShutdown() != ncclSuccess)
    INFO("NCCL did not shutdown nvml properly");
  cudaSetDevice(savedDevice);
  return res;
}


extern "C" DSOGLOBAL
void ncclCommDestroy(ncclComm_t comm) {
  if (comm == NULL)
    return;

  int savedDevice;
  cudaGetDevice(&savedDevice);
  int commDevice = comm->cudaDev;

  if (savedDevice != commDevice) {
    CUDACHECK(cudaSetDevice(commDevice));
  }

  commFree(comm);

  if (savedDevice != commDevice)
    cudaSetDevice(savedDevice);
}

extern "C" DSOGLOBAL
const char* ncclGetErrorString(ncclResult_t code) {
  switch (code) {
  case ncclSuccess                : return "no error";
  case ncclUnhandledCudaError     : return "unhandled cuda error";
  case ncclSystemError            : return "system error";
  case ncclInternalError          : return "internal error";
  case ncclInvalidDevicePointer   : return "invalid device pointer";
  case ncclInvalidRank            : return "invalid rank";
  case ncclUnsupportedDeviceCount : return "unsupported device count";
  case ncclDeviceNotFound         : return "device not found";
  case ncclInvalidDeviceIndex     : return "invalid device index";
  case ncclLibWrapperNotSet       : return "lib wrapper not initialized";
  case ncclCudaMallocFailed       : return "cuda malloc failed";
  case ncclRankMismatch           : return "parameter mismatch between ranks";
  case ncclInvalidArgument        : return "invalid argument";
  case ncclInvalidType            : return "invalid data type";
  case ncclInvalidOperation       : return "invalid reduction operations";
  }
  return "unknown result code";
}

extern "C" DSOGLOBAL
ncclResult_t ncclCommCount(const ncclComm_t comm, int* count) {
  *count = comm->nDev;
  return ncclSuccess;
}

extern "C" DSOGLOBAL
ncclResult_t ncclCommCuDevice(const ncclComm_t comm, int* devid) {
  *devid = comm->cudaDev;
  return ncclSuccess;
}

extern "C" DSOGLOBAL
ncclResult_t ncclCommUserRank(const ncclComm_t comm, int* rank) {
  *rank = comm->userFromRing[comm->ncclId];
  return ncclSuccess;
}

