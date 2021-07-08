/*************************************************************************
 * Copyright (c) 2016-2021, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "nccl.h"
#include "core.h"
#include "utils.h"
#include "bootstrap.h"
#include "net.h"
#include "socket.h"
#include <unistd.h>
#include <sys/types.h>

/* Init functions */
static char bootstrapNetIfName[MAX_IF_NAME_SIZE+1];
static union socketAddress bootstrapNetIfAddr;
static int bootstrapNetInitDone = 0;
pthread_mutex_t bootstrapNetLock = PTHREAD_MUTEX_INITIALIZER;

ncclResult_t bootstrapNetInit() {
  if (bootstrapNetInitDone == 0) {
    pthread_mutex_lock(&bootstrapNetLock);
    if (bootstrapNetInitDone == 0) {
      char* env = getenv("NCCL_COMM_ID");
      if (env) {
        union socketAddress remoteAddr;
        if (GetSocketAddrFromString(&remoteAddr, env) != ncclSuccess) {
          WARN("Invalid NCCL_COMM_ID, please use format: <ipv4>:<port> or [<ipv6>]:<port> or <hostname>:<port>");
          return ncclInvalidArgument;
        }
        if (findInterfaceMatchSubnet(bootstrapNetIfName, &bootstrapNetIfAddr, &remoteAddr, MAX_IF_NAME_SIZE, 1) <= 0) {
          WARN("NET/Socket : No usable listening interface found");
          return ncclSystemError;
        }
      } else {
        int nIfs = findInterfaces(bootstrapNetIfName, &bootstrapNetIfAddr, MAX_IF_NAME_SIZE, 1);
        if (nIfs <= 0) {
          WARN("Bootstrap : no socket interface found");
          return ncclInternalError;
        }
      }
      char line[SOCKET_NAME_MAXLEN+MAX_IF_NAME_SIZE+2];
      sprintf(line, " %s:", bootstrapNetIfName);
      socketToString(&bootstrapNetIfAddr, line+strlen(line));
      INFO(NCCL_INIT, "Bootstrap : Using%s", line);
      bootstrapNetInitDone = 1;
    }
    pthread_mutex_unlock(&bootstrapNetLock);
  }
  return ncclSuccess;
}

/* Socket Interface Selection type */
enum bootstrapInterface_t { findSubnetIf = -1, dontCareIf = -2 };

static ncclResult_t bootstrapNetAccept(int listenFd, int* recvFd, union socketAddress *addr) {
  struct sockaddr *saddr = &addr->sa;
  socklen_t socklen = sizeof(union socketAddress);
  SYSCHECKVAL(accept(listenFd, saddr, &socklen), "accept", *recvFd);
  return ncclSuccess;
}

// Additional sync functions
static ncclResult_t bootstrapNetSend(int fd, union socketAddress *addr, void* data, int size) {
  NCCLCHECK(socketSend(fd, addr, &size, sizeof(int)));
  NCCLCHECK(socketSend(fd, addr, data, size));
  return ncclSuccess;
}
static ncclResult_t bootstrapNetRecv(int fd, union socketAddress *addr, void* data, int size) {
  int recvSize;
  NCCLCHECK(socketRecv(fd, addr, &recvSize, sizeof(int)));
  if (recvSize > size) {
    WARN("Message truncated : received %d bytes instead of %d", recvSize, size);
    return ncclInternalError;
  }
  NCCLCHECK(socketRecv(fd, addr, data, std::min(recvSize, size)));
  return ncclSuccess;
}

struct extInfo {
  int rank;
  int nranks;
  union socketAddress extAddressListenRoot;
  union socketAddress extAddressListen;
};

#include <sys/resource.h>

static ncclResult_t setFilesLimit() {
  struct rlimit filesLimit;
  SYSCHECK(getrlimit(RLIMIT_NOFILE, &filesLimit), "getrlimit");
  filesLimit.rlim_cur = filesLimit.rlim_max;
  SYSCHECK(setrlimit(RLIMIT_NOFILE, &filesLimit), "setrlimit");
  return ncclSuccess;
}

static void *bootstrapRoot(void* args) {
  int listenFd = (uint64_t)args;
  ncclResult_t res = ncclSuccess;
  int nranks = 0, c = 0;
  struct extInfo info;
  union socketAddress *rankAddresses = NULL;
  union socketAddress *rankAddressesRoot = NULL; // for initial rank <-> root information exchange
  union socketAddress *zero = NULL;
  NCCLCHECKGOTO(ncclCalloc(&zero, 1), res, out);
  setFilesLimit();

  TRACE(NCCL_INIT, "BEGIN");
  /* Receive addresses from all ranks */
  do {
    int tmpFd;
    union socketAddress addr;
    NCCLCHECKGOTO(bootstrapNetAccept(listenFd, &tmpFd, &addr), res, out);
    NCCLCHECKGOTO(bootstrapNetRecv(tmpFd, &addr, &info, sizeof(info)), res, out);
    close(tmpFd);

    if (c == 0) {
      nranks = info.nranks;
      NCCLCHECKGOTO(ncclCalloc(&rankAddresses, nranks), res, out);
      NCCLCHECKGOTO(ncclCalloc(&rankAddressesRoot, nranks), res, out);
    }

    if (nranks != info.nranks) {
      WARN("Bootstrap Root : mismatch in rank count from procs %d : %d", nranks, info.nranks);
      goto out;
    }

    if (memcmp(zero, &rankAddressesRoot[info.rank], sizeof(union socketAddress)) != 0) {
      WARN("Bootstrap Root : rank %d of %d ranks has already checked in", info.rank, nranks);
      goto out;
    }

    // Save the connection handle for that rank
    memcpy(rankAddressesRoot+info.rank, &info.extAddressListenRoot, sizeof(union socketAddress));
    memcpy(rankAddresses+info.rank, &info.extAddressListen, sizeof(union socketAddress));

    ++c;
    TRACE(NCCL_INIT, "Received connect from rank %d total %d/%d",  info.rank, c, nranks);
  } while (c < nranks);
  TRACE(NCCL_INIT, "COLLECTED ALL %d HANDLES", nranks);

  // Send the connect handle for the next rank in the AllGather ring
  for (int r=0; r<nranks; ++r) {
    int next = (r+1) % nranks;
    int tmpSendFd;
    NCCLCHECKGOTO(connectAddress(&tmpSendFd, rankAddressesRoot+r), res, out);
    NCCLCHECKGOTO(bootstrapNetSend(tmpSendFd, rankAddressesRoot+r, rankAddresses+next, sizeof(union socketAddress)), res, out);
    close(tmpSendFd);
  }
  TRACE(NCCL_INIT, "SENT OUT ALL %d HANDLES", nranks);

out:
  close(listenFd);
  if (rankAddresses) free(rankAddresses);
  if (rankAddressesRoot) free(rankAddressesRoot);
  if (zero) free(zero);

  TRACE(NCCL_INIT, "DONE");
  return NULL;
}

ncclResult_t bootstrapCreateRoot(ncclUniqueId* id, bool idFromEnv) {
  union socketAddress* connectAddr = (union socketAddress*) id;
  int listenFd;
  NCCLCHECK(createListenSocket(&listenFd, connectAddr));
  pthread_t thread;
  pthread_create(&thread, NULL, bootstrapRoot, (void*)(uint64_t)listenFd);
  return ncclSuccess;
}

ncclResult_t bootstrapGetUniqueId(ncclUniqueId* id) {
  static_assert(sizeof(union socketAddress) < sizeof(ncclUniqueId), "NetId does not fit inside ncclUniqueId");
  memset(id, 0, sizeof(ncclUniqueId));
  union socketAddress* connectAddr = (union socketAddress*) id;

  char* env = getenv("NCCL_COMM_ID");
  if (env) {
    INFO(NCCL_ENV, "NCCL_COMM_ID set by environment to %s", env);
    if (GetSocketAddrFromString(connectAddr, env) != ncclSuccess) {
      WARN("Invalid NCCL_COMM_ID, please use format: <ipv4>:<port> or [<ipv6>]:<port> or <hostname>:<port>");
      return ncclInvalidArgument;
    }
  } else {
    memcpy(id, &bootstrapNetIfAddr, sizeof(union socketAddress));
    NCCLCHECK(bootstrapCreateRoot(id, false));
  }

  return ncclSuccess;
}

struct unexConn {
  int peer;
  int tag;
  int fd;
  union socketAddress addr;
  struct unexConn* next;
};

// Remote allocator state
struct remAllocState {
  int cudaDev;
  int listenFd;
  int stop;
};

struct extState {
  int extListenFd;
  int extRingRecvFd;
  int extRingSendFd;
  union socketAddress extRingRecvAddr, extRingSendAddr;
  union socketAddress* peerCommAddresses;
  union socketAddress* peerAllocAddresses;
  struct unexConn* unexpectedConnections;
  int cudaDev;
  int rank;
  int nranks;

  // Intermediate memory allocation service
  struct remAllocState* allocState;
  pthread_t allocThread;
};

#define MAX_SEGMENTS 128

static ncclResult_t remoteAlloc(void** ptr, int fd, union socketAddress *addr) {
  size_t size;
  NCCLCHECK(socketRecv(fd, addr, &size, sizeof(size_t)));
  cudaIpcMemHandle_t devIpc;
  NCCLCHECK(ncclCudaCalloc((char**)ptr, size));
  cudaError_t res = cudaIpcGetMemHandle(&devIpc, *ptr);
  if (res != cudaSuccess) {
    WARN("[Rem Allocator] cudaIpcGetMemHandle failed : %s", cudaGetErrorString(res));
    cudaFree(*ptr);
    CUDACHECK(res);
  }
  // The CUDA IPC
  NCCLCHECK(socketSend(fd, addr, &devIpc, sizeof(cudaIpcMemHandle_t)));
  // And the direct pointer
  NCCLCHECK(socketSend(fd, addr, ptr, sizeof(void*)));
  return ncclSuccess;
}

#include <poll.h>

// Service thread to allocate memory for other GPUs, used as intermediate step.
void* ncclRemoteMemAllocationService(void* args) {
  struct remAllocState* state = (struct remAllocState *) args;
  if (cudaSetDevice(state->cudaDev) != cudaSuccess) {
    WARN("[Rem Allocator] Failed to set CUDA device %d", state->cudaDev);
  }

  // Prepare poll descriptor
  void* segments[MAX_SEGMENTS];
  struct pollfd pollfds[MAX_SEGMENTS+1];
  for (int s=0; s<MAX_SEGMENTS; s++) segments[s] = NULL;
  for (int s=0; s<MAX_SEGMENTS; s++) {
    pollfds[s].fd = -1;
    pollfds[s].events = POLLHUP;
  }
  pollfds[MAX_SEGMENTS].fd = state->listenFd;
  pollfds[MAX_SEGMENTS].events = POLLIN;

  int nbuffers = 0;
  while (state->stop == 0 || (state->stop == 1 && nbuffers > 0)) {
    if (int error = poll(pollfds, MAX_SEGMENTS+1, 100/*ms*/) < 0) {
      WARN("[Rem Allocator] Poll failed with error %d", error);
      return NULL;
    }
    if (pollfds[MAX_SEGMENTS].revents) {
      int s = 0;
      union socketAddress addr;
      while (segments[s] != NULL && s < MAX_SEGMENTS) s++;
      if (bootstrapNetAccept(pollfds[MAX_SEGMENTS].fd, &pollfds[s].fd, &addr) != ncclSuccess) {
        pollfds[s].fd = -1;
      } else {
        if (s == MAX_SEGMENTS || (remoteAlloc(segments+s, pollfds[s].fd, &addr) != ncclSuccess)) {
          WARN("[Rem Allocator] Allocation failed (segment %d, fd %d)", s, pollfds[s].fd);
          close(pollfds[s].fd);
          pollfds[s].fd = -1;
        } else {
          nbuffers++;
        }
      }
    }
    for (int s=0; s<MAX_SEGMENTS; s++) {
      if (pollfds[s].revents & POLLHUP) {
        if (cudaFree(segments[s]) != cudaSuccess) {
          WARN("[Rem Allocator] cudaFree %p failed", segments[s]);
        }
        segments[s] = NULL;
        close(pollfds[s].fd);
        pollfds[s].fd = -1;
        nbuffers--;
      }
    }
  }
  for (int s=0; s<MAX_SEGMENTS; s++) {
    if (segments[s]) cudaFree(segments[s]);
    close(pollfds[s].fd);
  }
  close(state->listenFd);
  free(state);
  return NULL;
}

ncclResult_t bootstrapRemAlloc(size_t size, int rank, void* commState, int* id, cudaIpcMemHandle_t* ipc, void** ptr) {
  struct extState* state = (struct extState*)commState;
  int fd;
  ncclResult_t res;
  *id = -1;
  union socketAddress *addr = state->peerAllocAddresses+rank;
  NCCLCHECK(connectAddress(&fd, addr));
  NCCLCHECKGOTO(socketSend(fd, addr, &size, sizeof(size_t)), res, end);
  NCCLCHECKGOTO(socketRecv(fd, addr, ipc, sizeof(cudaIpcMemHandle_t)), res, end);
  NCCLCHECKGOTO(socketRecv(fd, addr, ptr, sizeof(void*)), res, end);
  *id = fd;
end:
  return res;
}

ncclResult_t bootstrapRemFree(int id, int rank, void* commState) {
  SYSCHECK(close(id), "close");
  return ncclSuccess;
}

ncclResult_t bootstrapInit(ncclUniqueId * id, int rank, int nranks, void** commState) {
  struct extState* state;
  NCCLCHECK(ncclCalloc(&state, 1));
  state->rank = rank;
  state->nranks = nranks;
  *commState = state;

  TRACE(NCCL_INIT, "rank %d nranks %d", rank, nranks);

  struct extInfo info = { 0 };
  info.rank = rank;
  info.nranks = nranks;
  int tmpSendFd, tmpRecvFd;

  int extListenFdRoot;
  memcpy(&info.extAddressListen,     &bootstrapNetIfAddr, sizeof(union socketAddress));
  memcpy(&info.extAddressListenRoot, &bootstrapNetIfAddr, sizeof(union socketAddress));
  NCCLCHECK(createListenSocket(&state->extListenFd, &info.extAddressListen));
  NCCLCHECK(createListenSocket(&extListenFdRoot, &info.extAddressListenRoot));

  // stagger connection times to avoid an overload of the root
  if (nranks > 128) {
    long msec = rank;
    struct timespec tv;
    tv.tv_sec = msec / 1000;
    tv.tv_nsec = 1000000 * (msec % 1000);
    TRACE(NCCL_INIT, "rank %d delaying connection to root by %ld msec", rank, msec);
    (void) nanosleep(&tv, NULL);
  }

  // send info on my listening socket to root
  union socketAddress* rootAddr = (union socketAddress*)id;
  NCCLCHECK(connectAddress(&tmpSendFd, rootAddr));
  NCCLCHECK(bootstrapNetSend(tmpSendFd, rootAddr,  &info, sizeof(info)));
  close(tmpSendFd);

  // get info on my "next" rank in the bootstrap ring from root
  union socketAddress addr;
  NCCLCHECK(bootstrapNetAccept(extListenFdRoot, &tmpRecvFd, &addr));
  NCCLCHECK(bootstrapNetRecv(tmpRecvFd, &addr, &state->extRingSendAddr, sizeof(state->extRingSendAddr)));
  close(tmpRecvFd);
  close(extListenFdRoot);

  NCCLCHECK(connectAddress(&state->extRingSendFd, &state->extRingSendAddr));
  // Accept the connect request from the previous rank in the AllGather ring
  NCCLCHECK(bootstrapNetAccept(state->extListenFd, &state->extRingRecvFd, &state->extRingRecvAddr));

  // AllGather all listen handlers
  NCCLCHECK(ncclCalloc(&state->peerCommAddresses, nranks));
  memcpy(state->peerCommAddresses+rank, &info.extAddressListen, sizeof(union socketAddress));
  NCCLCHECK(bootstrapAllGather(state, state->peerCommAddresses, sizeof(union socketAddress)));

  // Create the memory allocation service
  NCCLCHECK(ncclCalloc(&state->peerAllocAddresses, nranks));
  memcpy(state->peerAllocAddresses+rank, &bootstrapNetIfAddr, sizeof(union socketAddress));
  NCCLCHECK(ncclCalloc(&state->allocState, 1));
  CUDACHECK(cudaGetDevice(&state->allocState->cudaDev));
  NCCLCHECK(createListenSocket(&state->allocState->listenFd, state->peerAllocAddresses+rank));
  pthread_create(&state->allocThread, NULL, ncclRemoteMemAllocationService, state->allocState);
  NCCLCHECK(bootstrapAllGather(state, state->peerAllocAddresses, sizeof(union socketAddress)));

  TRACE(NCCL_INIT, "rank %d nranks %d - DONE", rank, nranks);

  return ncclSuccess;
}

ncclResult_t bootstrapAllGather(void* commState, void* allData, int size) {
  struct extState* state = (struct extState*)commState;
  char* data = (char*)allData;
  int rank = state->rank;
  int nranks = state->nranks;

  TRACE(NCCL_INIT, "rank %d nranks %d size %d", rank, nranks, size);

  /* Simple ring based AllGather
   * At each step i receive data from (rank-i-1) from left
   * and send previous step's data from (rank-i) to right
   */
  for (int i=0; i<nranks-1; i++) {
    size_t rslice = (rank - i - 1 + nranks) % nranks;
    size_t sslice = (rank - i + nranks) % nranks;

    // Send slice to the right
    NCCLCHECK(bootstrapNetSend(state->extRingSendFd, &state->extRingSendAddr, data+sslice*size, size));
    // Recv slice from the left
    NCCLCHECK(bootstrapNetRecv(state->extRingRecvFd, &state->extRingRecvAddr, data+rslice*size, size));
  }

  TRACE(NCCL_INIT, "rank %d nranks %d size %d - DONE", rank, nranks, size);
  return ncclSuccess;
}

ncclResult_t bootstrapSend(void* commState, int peer, int tag, void* data, int size) {
  struct extState* state = (struct extState*)commState;
  int tmpSendFd;
  union socketAddress *addr = state->peerCommAddresses+peer;
  NCCLCHECK(connectAddress(&tmpSendFd, addr));
  NCCLCHECK(bootstrapNetSend(tmpSendFd, addr, &state->rank, sizeof(int)));
  NCCLCHECK(bootstrapNetSend(tmpSendFd, addr, &tag, sizeof(int)));
  NCCLCHECK(bootstrapNetSend(tmpSendFd, addr, data, size));
  close(tmpSendFd);
  return ncclSuccess;
}

ncclResult_t bootstrapBarrier(void* commState, int *ranks, int tag, int rank, int nranks) {
  if (nranks == 1) return ncclSuccess;
  TRACE(NCCL_INIT, "rank %d nranks %d tag %x - ENTER", rank, nranks, tag);

  /* Simple intra process barrier
   *
   * Based on the dissemination algorithm by Debra Hensgen, Raphael Finkel, and Udi Manbet,
   * "Two Algorithms for Barrier Synchronization," International Journal of Parallel Programming, 17(1):1-17, 1988"
   */
  int data[1];
  for (int mask=1; mask<nranks; mask<<=1) {
    int src = (rank - mask + nranks) % nranks;
    int dst = (rank + mask) % nranks;
    NCCLCHECK(bootstrapSend(commState, ranks[dst], tag, data, sizeof(data)));
    NCCLCHECK(bootstrapRecv(commState, ranks[src], tag, data, sizeof(data)));
  }

  TRACE(NCCL_INIT, "rank %d nranks %d tag %x - DONE", rank, nranks, tag);
  return ncclSuccess;
}

ncclResult_t unexpectedEnqueue(struct extState* state, int peer, int tag, int fd, union socketAddress *addr) {
  // New unex
  struct unexConn* unex;
  NCCLCHECK(ncclCalloc(&unex, 1));
  unex->peer = peer;
  unex->tag = tag;
  unex->fd = fd;
  unex->addr = *addr;

  // Enqueue
  struct unexConn* list = state->unexpectedConnections;
  if (list == NULL) {
    state->unexpectedConnections = unex;
    return ncclSuccess;
  }
  while (list->next) list = list->next;
  list->next = unex;
  return ncclSuccess;
}

int unexpectedDequeue(struct extState* state, int peer, int tag, union socketAddress *addr) {
  struct unexConn* elem = state->unexpectedConnections;
  struct unexConn* prev = NULL;
  while (elem) {
    if (elem->peer == peer && elem->tag == tag) {
      if (prev == NULL) {
        state->unexpectedConnections = elem->next;
      } else {
        prev->next = elem->next;
      }
      int fd = elem->fd;
      *addr = elem->addr;
      free(elem);
      return fd;
    }
    prev = elem;
    elem = elem->next;
  }
  return -1;
}

// We can't know who we'll receive from, so we need to receive everything at once
ncclResult_t bootstrapRecv(void* commState, int peer, int tag, void* data, int size) {
  struct extState* state = (struct extState*)commState;

  int tmpRecvFd;
  union socketAddress addr;

  // Search unexpected connections first
  if ((tmpRecvFd = unexpectedDequeue(state, peer, tag, &addr)) != -1) {
    NCCLCHECK(bootstrapNetRecv(tmpRecvFd, &addr, ((char*)data), size));
    close(tmpRecvFd);
    return ncclSuccess;
  }

  // Then look for new connections
  while (1) {
    union socketAddress addr;
    NCCLCHECK(bootstrapNetAccept(state->extListenFd, &tmpRecvFd, &addr));
    int newPeer, newTag;
    NCCLCHECK(bootstrapNetRecv(tmpRecvFd, &addr, &newPeer, sizeof(int)));
    NCCLCHECK(bootstrapNetRecv(tmpRecvFd, &addr, &newTag, sizeof(int)));
    if (newPeer == peer && newTag == tag) {
      NCCLCHECK(bootstrapNetRecv(tmpRecvFd, &addr, ((char*)data), size));
      close(tmpRecvFd);
      return ncclSuccess;
    }
    // Unexpected connection. Save for later.
    NCCLCHECK(unexpectedEnqueue(state, newPeer, newTag, tmpRecvFd, &addr));
  }
}

ncclResult_t bootstrapClose(void* commState) {
  struct extState* state = (struct extState*)commState;
  if (state->unexpectedConnections != NULL) {
    WARN("Unexpected connections are not empty");
    return ncclInternalError;
  }
  close(state->extListenFd);
  close(state->extRingSendFd);
  close(state->extRingRecvFd);

  state->allocState->stop = 1;

  // Join the allocThread so we catch resource leaks as being hung here
  // pthread_join(state->allocThread, nullptr);

  free(state->peerCommAddresses);
  free(state->peerAllocAddresses);
  free(state);

  return ncclSuccess;
}

ncclResult_t bootstrapAbort(void* commState) {
  struct extState* state = (struct extState*)commState;
  if (commState == NULL) return ncclSuccess;
  if (state->extListenFd) close(state->extListenFd);
  if (state->extRingSendFd) close(state->extRingSendFd);
  if (state->extRingRecvFd) close(state->extRingRecvFd);
  if (state->allocState) state->allocState->stop = 2;
  free(state->peerCommAddresses);
  free(state->peerAllocAddresses);
  free(state);
  return ncclSuccess;
}
