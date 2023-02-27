/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "nccl.h"
#include "core.h"
#include "utils.h"
#include "bootstrap.h"
#include "net.h"
#include <unistd.h>
#include <sys/types.h>
#include "proxy.h"

struct bootstrapRootArgs {
  struct ncclSocket* listenSock;
  uint64_t magic;
};

/* Init functions */
static char bootstrapNetIfName[MAX_IF_NAME_SIZE+1];
static union ncclSocketAddress bootstrapNetIfAddr;
static int bootstrapNetInitDone = 0;
pthread_mutex_t bootstrapNetLock = PTHREAD_MUTEX_INITIALIZER;

ncclResult_t bootstrapNetInit() {
  if (bootstrapNetInitDone == 0) {
    pthread_mutex_lock(&bootstrapNetLock);
    if (bootstrapNetInitDone == 0) {
      char* env = getenv("NCCL_COMM_ID");
      if (env) {
        union ncclSocketAddress remoteAddr;
        if (ncclSocketGetAddrFromString(&remoteAddr, env) != ncclSuccess) {
          WARN("Invalid NCCL_COMM_ID, please use format: <ipv4>:<port> or [<ipv6>]:<port> or <hostname>:<port>");
          return ncclInvalidArgument;
        }
        if (ncclFindInterfaceMatchSubnet(bootstrapNetIfName, &bootstrapNetIfAddr, &remoteAddr, MAX_IF_NAME_SIZE, 1) <= 0) {
          WARN("NET/Socket : No usable listening interface found");
          return ncclSystemError;
        }
      } else {
        int nIfs = ncclFindInterfaces(bootstrapNetIfName, &bootstrapNetIfAddr, MAX_IF_NAME_SIZE, 1);
        if (nIfs <= 0) {
          WARN("Bootstrap : no socket interface found");
          return ncclInternalError;
        }
      }
      char line[SOCKET_NAME_MAXLEN+MAX_IF_NAME_SIZE+2];
      sprintf(line, " %s:", bootstrapNetIfName);
      ncclSocketToString(&bootstrapNetIfAddr, line+strlen(line));
      INFO(NCCL_INIT, "Bootstrap : Using%s", line);
      bootstrapNetInitDone = 1;
    }
    pthread_mutex_unlock(&bootstrapNetLock);
  }
  return ncclSuccess;
}

/* Socket Interface Selection type */
enum bootstrapInterface_t { findSubnetIf = -1, dontCareIf = -2 };

// Additional sync functions
static ncclResult_t bootstrapNetSend(struct ncclSocket* sock, void* data, int size) {
  NCCLCHECK(ncclSocketSend(sock, &size, sizeof(int)));
  NCCLCHECK(ncclSocketSend(sock, data, size));
  return ncclSuccess;
}
static ncclResult_t bootstrapNetRecv(struct ncclSocket* sock, void* data, int size) {
  int recvSize;
  NCCLCHECK(ncclSocketRecv(sock, &recvSize, sizeof(int)));
  if (recvSize > size) {
    WARN("Message truncated : received %d bytes instead of %d", recvSize, size);
    return ncclInternalError;
  }
  NCCLCHECK(ncclSocketRecv(sock, data, std::min(recvSize, size)));
  return ncclSuccess;
}

struct extInfo {
  int rank;
  int nranks;
  union ncclSocketAddress extAddressListenRoot;
  union ncclSocketAddress extAddressListen;
};

#include <sys/resource.h>

static ncclResult_t setFilesLimit() {
  struct rlimit filesLimit;
  SYSCHECK(getrlimit(RLIMIT_NOFILE, &filesLimit), "getrlimit");
  filesLimit.rlim_cur = filesLimit.rlim_max;
  SYSCHECK(setrlimit(RLIMIT_NOFILE, &filesLimit), "setrlimit");
  return ncclSuccess;
}

static void *bootstrapRoot(void* rargs) {
  struct bootstrapRootArgs* args = (struct bootstrapRootArgs*)rargs;
  struct ncclSocket* listenSock = args->listenSock;
  uint64_t magic = args->magic;
  ncclResult_t res = ncclSuccess;
  int nranks = 0, c = 0;
  struct extInfo info;
  union ncclSocketAddress *rankAddresses = NULL;
  union ncclSocketAddress *rankAddressesRoot = NULL; // for initial rank <-> root information exchange
  union ncclSocketAddress *zero = NULL;
  NCCLCHECKGOTO(ncclCalloc(&zero, 1), res, out);
  setFilesLimit();

  TRACE(NCCL_INIT, "BEGIN");
  /* Receive addresses from all ranks */
  do {
    struct ncclSocket sock;
    NCCLCHECKGOTO(ncclSocketInit(&sock), res, out);
    NCCLCHECKGOTO(ncclSocketAccept(&sock, listenSock), res, out);
    NCCLCHECKGOTO(bootstrapNetRecv(&sock, &info, sizeof(info)), res, out);
    NCCLCHECKGOTO(ncclSocketClose(&sock), res, out);

    if (c == 0) {
      nranks = info.nranks;
      NCCLCHECKGOTO(ncclCalloc(&rankAddresses, nranks), res, out);
      NCCLCHECKGOTO(ncclCalloc(&rankAddressesRoot, nranks), res, out);
    }

    if (nranks != info.nranks) {
      WARN("Bootstrap Root : mismatch in rank count from procs %d : %d", nranks, info.nranks);
      goto out;
    }

    if (memcmp(zero, &rankAddressesRoot[info.rank], sizeof(union ncclSocketAddress)) != 0) {
      WARN("Bootstrap Root : rank %d of %d ranks has already checked in", info.rank, nranks);
      goto out;
    }

    // Save the connection handle for that rank
    memcpy(rankAddressesRoot+info.rank, &info.extAddressListenRoot, sizeof(union ncclSocketAddress));
    memcpy(rankAddresses+info.rank, &info.extAddressListen, sizeof(union ncclSocketAddress));

    ++c;
    TRACE(NCCL_INIT, "Received connect from rank %d total %d/%d",  info.rank, c, nranks);
  } while (c < nranks);
  TRACE(NCCL_INIT, "COLLECTED ALL %d HANDLES", nranks);

  // Send the connect handle for the next rank in the AllGather ring
  for (int r=0; r<nranks; ++r) {
    int next = (r+1) % nranks;
    struct ncclSocket sock;
    NCCLCHECKGOTO(ncclSocketInit(&sock, rankAddressesRoot+r, magic, ncclSocketTypeBootstrap), res, out);
    NCCLCHECKGOTO(ncclSocketConnect(&sock), res, out);
    NCCLCHECKGOTO(bootstrapNetSend(&sock, rankAddresses+next, sizeof(union ncclSocketAddress)), res, out);
    NCCLCHECKGOTO(ncclSocketClose(&sock), res, out);
  }
  TRACE(NCCL_INIT, "SENT OUT ALL %d HANDLES", nranks);

out:
  if (listenSock != NULL) {
    ncclSocketClose(listenSock);
    free(listenSock);
  }
  if (rankAddresses) free(rankAddresses);
  if (rankAddressesRoot) free(rankAddressesRoot);
  if (zero) free(zero);
  free(rargs);

  TRACE(NCCL_INIT, "DONE");
  return NULL;
}

ncclResult_t bootstrapCreateRoot(struct ncclBootstrapHandle* handle, bool idFromEnv) {
  struct ncclSocket* listenSock;
  struct bootstrapRootArgs* args;
  pthread_t thread;

  NCCLCHECK(ncclCalloc(&listenSock, 1));
  NCCLCHECK(ncclSocketInit(listenSock, &handle->addr, handle->magic, ncclSocketTypeBootstrap, NULL, 0));
  NCCLCHECK(ncclSocketListen(listenSock));
  NCCLCHECK(ncclSocketGetAddr(listenSock, &handle->addr));

  NCCLCHECK(ncclCalloc(&args, 1));
  args->listenSock = listenSock;
  args->magic = handle->magic;
  NEQCHECK(pthread_create(&thread, NULL, bootstrapRoot, (void*)args), 0);
  ncclSetThreadName(thread, "NCCL BootstrapR");
  NEQCHECK(pthread_detach(thread), 0); // will not be pthread_join()'d
  return ncclSuccess;
}

ncclResult_t bootstrapGetUniqueId(struct ncclBootstrapHandle* handle) {
  memset(handle, 0, sizeof(ncclBootstrapHandle));
  NCCLCHECK(getRandomData(&handle->magic, sizeof(handle->magic)));

  char* env = getenv("NCCL_COMM_ID");
  if (env) {
    INFO(NCCL_ENV, "NCCL_COMM_ID set by environment to %s", env);
    if (ncclSocketGetAddrFromString(&handle->addr, env) != ncclSuccess) {
      WARN("Invalid NCCL_COMM_ID, please use format: <ipv4>:<port> or [<ipv6>]:<port> or <hostname>:<port>");
      return ncclInvalidArgument;
    }
  } else {
    memcpy(&handle->addr, &bootstrapNetIfAddr, sizeof(union ncclSocketAddress));
    NCCLCHECK(bootstrapCreateRoot(handle, false));
  }

  return ncclSuccess;
}

struct unexConn {
  int peer;
  int tag;
  struct ncclSocket sock;
  struct unexConn* next;
};

struct bootstrapState {
  struct ncclSocket listenSock;
  struct ncclSocket ringRecvSocket;
  struct ncclSocket ringSendSocket;
  union ncclSocketAddress* peerCommAddresses;
  union ncclSocketAddress* peerProxyAddresses;
  struct unexConn* unexpectedConnections;
  int cudaDev;
  int rank;
  int nranks;
  uint64_t magic;
  volatile uint32_t *abortFlag;
};

ncclResult_t bootstrapInit(struct ncclBootstrapHandle* handle, struct ncclComm* comm) {
  int rank = comm->rank;
  int nranks = comm->nRanks;
  struct bootstrapState* state;
  struct ncclSocket* proxySocket;
  ncclSocketAddress nextAddr;
  struct ncclSocket sock, listenSockRoot;
  struct extInfo info = { 0 };

  NCCLCHECK(ncclCalloc(&state, 1));
  state->rank = rank;
  state->nranks = nranks;
  state->abortFlag = comm->abortFlag;
  comm->bootstrap = state;
  comm->magic = state->magic = handle->magic;

  TRACE(NCCL_INIT, "rank %d nranks %d", rank, nranks);

  info.rank = rank;
  info.nranks = nranks;
  // Create socket for other ranks to contact me
  NCCLCHECK(ncclSocketInit(&state->listenSock, &bootstrapNetIfAddr, comm->magic, ncclSocketTypeBootstrap, comm->abortFlag));
  NCCLCHECK(ncclSocketListen(&state->listenSock));
  NCCLCHECK(ncclSocketGetAddr(&state->listenSock, &info.extAddressListen));

  // Create socket for root to contact me
  NCCLCHECK(ncclSocketInit(&listenSockRoot, &bootstrapNetIfAddr, comm->magic, ncclSocketTypeBootstrap, comm->abortFlag));
  NCCLCHECK(ncclSocketListen(&listenSockRoot));
  NCCLCHECK(ncclSocketGetAddr(&listenSockRoot, &info.extAddressListenRoot));

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
  NCCLCHECK(ncclSocketInit(&sock, &handle->addr, comm->magic, ncclSocketTypeBootstrap, comm->abortFlag));
  NCCLCHECK(ncclSocketConnect(&sock));
  NCCLCHECK(bootstrapNetSend(&sock, &info, sizeof(info)));
  NCCLCHECK(ncclSocketClose(&sock));

  // get info on my "next" rank in the bootstrap ring from root
  NCCLCHECK(ncclSocketInit(&sock));
  NCCLCHECK(ncclSocketAccept(&sock, &listenSockRoot));
  NCCLCHECK(bootstrapNetRecv(&sock, &nextAddr, sizeof(union ncclSocketAddress)));
  NCCLCHECK(ncclSocketClose(&sock));
  NCCLCHECK(ncclSocketClose(&listenSockRoot));

  NCCLCHECK(ncclSocketInit(&state->ringSendSocket, &nextAddr, comm->magic, ncclSocketTypeBootstrap, comm->abortFlag));
  NCCLCHECK(ncclSocketConnect(&state->ringSendSocket));
  // Accept the connect request from the previous rank in the AllGather ring
  NCCLCHECK(ncclSocketInit(&state->ringRecvSocket));
  NCCLCHECK(ncclSocketAccept(&state->ringRecvSocket, &state->listenSock));

  // AllGather all listen handlers
  NCCLCHECK(ncclCalloc(&state->peerCommAddresses, nranks));
  NCCLCHECK(ncclSocketGetAddr(&state->listenSock, state->peerCommAddresses+rank));
  NCCLCHECK(bootstrapAllGather(state, state->peerCommAddresses, sizeof(union ncclSocketAddress)));

  // Create the service proxy
  NCCLCHECK(ncclCalloc(&state->peerProxyAddresses, nranks));

  // proxy is aborted through a message; don't set abortFlag
  NCCLCHECK(ncclCalloc(&proxySocket, 1));
  NCCLCHECK(ncclSocketInit(proxySocket, &bootstrapNetIfAddr, comm->magic, ncclSocketTypeProxy, comm->abortFlag));
  NCCLCHECK(ncclSocketListen(proxySocket));
  NCCLCHECK(ncclSocketGetAddr(proxySocket, state->peerProxyAddresses+rank));
  NCCLCHECK(bootstrapAllGather(state, state->peerProxyAddresses, sizeof(union ncclSocketAddress)));
  NCCLCHECK(ncclProxyInit(comm, proxySocket, state->peerProxyAddresses));

  TRACE(NCCL_INIT, "rank %d nranks %d - DONE", rank, nranks);

  return ncclSuccess;
}

ncclResult_t bootstrapAllGather(void* commState, void* allData, int size) {
  struct bootstrapState* state = (struct bootstrapState*)commState;
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
    NCCLCHECK(bootstrapNetSend(&state->ringSendSocket, data+sslice*size, size));
    // Recv slice from the left
    NCCLCHECK(bootstrapNetRecv(&state->ringRecvSocket, data+rslice*size, size));
  }

  TRACE(NCCL_INIT, "rank %d nranks %d size %d - DONE", rank, nranks, size);
  return ncclSuccess;
}

ncclResult_t bootstrapSend(void* commState, int peer, int tag, void* data, int size) {
  ncclResult_t ret = ncclSuccess;
  struct bootstrapState* state = (struct bootstrapState*)commState;
  struct ncclSocket sock;

  NCCLCHECKGOTO(ncclSocketInit(&sock, state->peerCommAddresses+peer, state->magic, ncclSocketTypeBootstrap, state->abortFlag), ret, fail);
  NCCLCHECKGOTO(ncclSocketConnect(&sock), ret, fail);
  NCCLCHECKGOTO(bootstrapNetSend(&sock, &state->rank, sizeof(int)), ret, fail);
  NCCLCHECKGOTO(bootstrapNetSend(&sock, &tag, sizeof(int)), ret, fail);
  NCCLCHECKGOTO(bootstrapNetSend(&sock, data, size), ret, fail);

exit:
  NCCLCHECK(ncclSocketClose(&sock));
  return ret;
fail:
  goto exit;
}

ncclResult_t bootstrapBarrier(void* commState, int *ranks, int rank, int nranks, int tag) {
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

ncclResult_t bootstrapIntraNodeAllGather(void* commState, int *ranks, int rank, int nranks, void* allData, int size) {
  if (nranks == 1) return ncclSuccess;
  char* data = (char*)allData;
  TRACE(NCCL_INIT, "rank %d nranks %d size %d - ENTER", rank, nranks, size);

  for (int i=1; i<nranks; i++) {
    int src = (rank - i + nranks) % nranks;
    int dst = (rank + i) % nranks;
    NCCLCHECK(bootstrapSend(commState, ranks[dst], /*tag=*/i, data+rank*size, size));
    NCCLCHECK(bootstrapRecv(commState, ranks[src], /*tag=*/i, data+src*size, size));
  }

  TRACE(NCCL_INIT, "rank %d nranks %d size %d - DONE", rank, nranks, size);
  return ncclSuccess;
}

// IntraNode in-place Broadcast
ncclResult_t bootstrapIntraNodeBroadcast(void* commState, int *ranks, int rank, int nranks, int root, void* bcastData, int size) {
  if (nranks == 1) return ncclSuccess;
  TRACE(NCCL_INIT, "rank %d nranks %d root %d size %d - ENTER", rank, nranks, root, size);

  if (rank == root) {
    for (int i=0; i<nranks; i++) {
      if (i != root) NCCLCHECK(bootstrapSend(commState, ranks[i], /*tag=*/ranks[i], bcastData, size));
    }
  }
  else {
    NCCLCHECK(bootstrapRecv(commState, ranks[root], /*tag=*/rank, bcastData, size));
  }

  TRACE(NCCL_INIT, "rank %d nranks %d root %d size %d - DONE", rank, nranks, root, size);
  return ncclSuccess;
}

ncclResult_t unexpectedEnqueue(struct bootstrapState* state, int peer, int tag, struct ncclSocket* sock) {
  // New unex
  struct unexConn* unex;
  NCCLCHECK(ncclCalloc(&unex, 1));
  unex->peer = peer;
  unex->tag = tag;
  memcpy(&unex->sock, sock, sizeof(struct ncclSocket));

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

ncclResult_t unexpectedDequeue(struct bootstrapState* state, int peer, int tag, struct ncclSocket* sock, int* found) {
  struct unexConn* elem = state->unexpectedConnections;
  struct unexConn* prev = NULL;
  *found = 0;
  while (elem) {
    if (elem->peer == peer && elem->tag == tag) {
      if (prev == NULL) {
        state->unexpectedConnections = elem->next;
      } else {
        prev->next = elem->next;
      }
      memcpy(sock, &elem->sock, sizeof(struct ncclSocket));
      free(elem);
      *found = 1;
      return ncclSuccess;
    }
    prev = elem;
    elem = elem->next;
  }
  return ncclSuccess;
}

static void unexpectedFree(struct bootstrapState* state) {
  struct unexConn* elem = state->unexpectedConnections;
  struct unexConn* prev = NULL;

  while (elem) {
    prev = elem;
    elem = elem->next;
    free(prev);
  }
  return;
}

// We can't know who we'll receive from, so we need to receive everything at once
ncclResult_t bootstrapRecv(void* commState, int peer, int tag, void* data, int size) {
  ncclResult_t ret = ncclSuccess;
  struct bootstrapState* state = (struct bootstrapState*)commState;
  struct ncclSocket sock;
  int newPeer, newTag;

  // Search unexpected connections first
  int found;
  NCCLCHECK(unexpectedDequeue(state, peer, tag, &sock, &found));
  if (found) {
    NCCLCHECKGOTO(bootstrapNetRecv(&sock, ((char*)data), size), ret, fail);
    goto exit;
  }

  // Then look for new connections
  while (1) {
    NCCLCHECKGOTO(ncclSocketInit(&sock), ret, fail);
    NCCLCHECKGOTO(ncclSocketAccept(&sock, &state->listenSock), ret, fail);
    NCCLCHECKGOTO(bootstrapNetRecv(&sock, &newPeer, sizeof(int)), ret, fail);
    NCCLCHECKGOTO(bootstrapNetRecv(&sock, &newTag, sizeof(int)), ret, fail);
    if (newPeer == peer && newTag == tag) {
      NCCLCHECKGOTO(bootstrapNetRecv(&sock, ((char*)data), size), ret, fail);
      goto exit;
    }
    // Unexpected connection. Save for later.
    NCCLCHECKGOTO(unexpectedEnqueue(state, newPeer, newTag, &sock), ret, fail);
  }
exit:
  NCCLCHECK(ncclSocketClose(&sock));
  return ret;
fail:
  goto exit;
}

ncclResult_t bootstrapClose(void* commState) {
  struct bootstrapState* state = (struct bootstrapState*)commState;
  if (state->unexpectedConnections != NULL) {
    unexpectedFree(state);
    if (*state->abortFlag == 0) {
      WARN("Unexpected connections are not empty");
      return ncclInternalError;
    }
  }

  NCCLCHECK(ncclSocketClose(&state->listenSock));
  NCCLCHECK(ncclSocketClose(&state->ringSendSocket));
  NCCLCHECK(ncclSocketClose(&state->ringRecvSocket));

  free(state->peerCommAddresses);
  free(state);

  return ncclSuccess;
}

ncclResult_t bootstrapAbort(void* commState) {
  struct bootstrapState* state = (struct bootstrapState*)commState;
  if (commState == NULL) return ncclSuccess;
  NCCLCHECK(ncclSocketClose(&state->listenSock));
  NCCLCHECK(ncclSocketClose(&state->ringSendSocket));
  NCCLCHECK(ncclSocketClose(&state->ringRecvSocket));
  free(state->peerCommAddresses);
  free(state->peerProxyAddresses);
  free(state);
  return ncclSuccess;
}
