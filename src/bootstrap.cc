/*************************************************************************
 * Copyright (c) 2016-2019, NVIDIA CORPORATION. All rights reserved.
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

// Always use sockets for bootstrap
struct bootstrapNetHandle {
  union socketAddress connectAddr;
};

struct bootstrapNetComm {
  int fd;
};

/* Init functions */
static char bootstrapNetIfNames[MAX_IF_NAME_SIZE*MAX_IFS];
static union socketAddress bootstrapNetIfAddrs[MAX_IFS];
static int bootstrapNetIfs = -1;
pthread_mutex_t bootstrapNetLock = PTHREAD_MUTEX_INITIALIZER;

ncclResult_t bootstrapNetInit() {
  if (bootstrapNetIfs == -1) {
    pthread_mutex_lock(&bootstrapNetLock);
    if (bootstrapNetIfs == -1) {
      bootstrapNetIfs = findInterfaces(bootstrapNetIfNames, bootstrapNetIfAddrs, MAX_IF_NAME_SIZE, MAX_IFS);
      if (bootstrapNetIfs <= 0) {
        WARN("Bootstrap : no socket interface found");
        return ncclInternalError;
      } else {
        char line[1024];
        char addrline[1024];
        line[0] = '\0';
        for (int i=0; i<bootstrapNetIfs; i++) {
          snprintf(line+strlen(line), 1023-strlen(line), " [%d]%s:%s", i, bootstrapNetIfNames+i*MAX_IF_NAME_SIZE,
              socketToString(&bootstrapNetIfAddrs[i].sa, addrline));
        }
        line[1023] = '\0';
        INFO(NCCL_INIT, "Bootstrap : Using%s", line);
      }
    }
    pthread_mutex_unlock(&bootstrapNetLock);
  }
  return ncclSuccess;
}

static ncclResult_t bootstrapNetNewComm(struct bootstrapNetComm** comm) {
  NCCLCHECK(ncclCalloc(comm, 1));
  (*comm)->fd = -1;
  return ncclSuccess;
}

static ncclResult_t bootstrapNetGetSocketAddr(int dev, union socketAddress* addr) {
  if (dev >= bootstrapNetIfs) return ncclInternalError;
  memcpy(addr, bootstrapNetIfAddrs+dev, sizeof(*addr));
  return ncclSuccess;
}

/* Socket Interface Selection type */
enum bootstrapInterface_t { findSubnetIf = -1, dontCareIf = -2 };

static ncclResult_t bootstrapNetListen(int dev, void* opaqueHandle, void** listenComm) {
  struct bootstrapNetHandle* handle = (struct bootstrapNetHandle*) opaqueHandle;
  static_assert(sizeof(struct bootstrapNetHandle) < NCCL_NET_HANDLE_MAXSIZE, "bootstrapNetHandle size too large");
  // if dev >= 0, listen based on dev
  if (dev >= 0) {
    NCCLCHECK(bootstrapNetGetSocketAddr(dev, &(handle->connectAddr)));
  } else if (dev == findSubnetIf) {
    // handle stores a remote address
    // need to find a local addr that is in the same network as the remote addr
    union socketAddress localAddr;
    char ifName[MAX_IF_NAME_SIZE];
    if (findInterfaceMatchSubnet(ifName, &localAddr, handle->connectAddr, MAX_IF_NAME_SIZE, 1) <= 0) {
      WARN("NET/Socket : No usable listening interface found");
      return ncclSystemError;
    }
    // pass the local address back
    memcpy(&handle->connectAddr, &localAddr, sizeof(handle->connectAddr));
  } // Otherwise, handle stores a local address
  struct bootstrapNetComm* comm;
  NCCLCHECK(bootstrapNetNewComm(&comm));
  NCCLCHECK(createListenSocket(&comm->fd, &handle->connectAddr));
  *listenComm = comm;
  return ncclSuccess;
}

static ncclResult_t bootstrapNetConnect(int dev, void* opaqueHandle, void** sendComm) {
  struct bootstrapNetComm* comm;
  NCCLCHECK(bootstrapNetNewComm(&comm));
  struct bootstrapNetHandle* handle = (struct bootstrapNetHandle*) opaqueHandle;
  NCCLCHECK(connectAddress(&comm->fd, &handle->connectAddr));
  *sendComm = comm;
  return ncclSuccess;
}

static ncclResult_t bootstrapNetAccept(void* listenComm, void** recvComm) {
  struct bootstrapNetComm* lComm = (struct bootstrapNetComm*)listenComm;
  struct bootstrapNetComm* rComm;
  NCCLCHECK(bootstrapNetNewComm(&rComm));
  struct sockaddr_in sockaddr;
  socklen_t socklen = sizeof(struct sockaddr_in);
  SYSCHECKVAL(accept(lComm->fd, (struct sockaddr*)&sockaddr, &socklen), "accept", rComm->fd);
  *recvComm = rComm;
  return ncclSuccess;
}

static ncclResult_t bootstrapNetClose(void* opaqueComm) {
  struct bootstrapNetComm* comm = (struct bootstrapNetComm*)opaqueComm;
  if (comm) {
    close(comm->fd);
    free(comm);
  }
  return ncclSuccess;
}

static ncclResult_t bootstrapNetCloseSend(void* sendComm) { NCCLCHECK(bootstrapNetClose(sendComm)); return ncclSuccess; }
static ncclResult_t bootstrapNetCloseRecv(void* recvComm) { NCCLCHECK(bootstrapNetClose(recvComm)); return ncclSuccess; }
static ncclResult_t bootstrapNetCloseListen(void* listenComm) { NCCLCHECK(bootstrapNetClose(listenComm)); return ncclSuccess; }

// Additional sync functions
static ncclResult_t bootstrapNetSend(void* sendComm, void* data, int size) {
  struct bootstrapNetComm* comm = (struct bootstrapNetComm*)sendComm;
  NCCLCHECK(socketSend(comm->fd, &size, sizeof(int)));
  NCCLCHECK(socketSend(comm->fd, data, size));
  return ncclSuccess;
}
static ncclResult_t bootstrapNetRecv(void* recvComm, void* data, int size) {
  struct bootstrapNetComm* comm = (struct bootstrapNetComm*)recvComm;
  int recvSize;
  NCCLCHECK(socketReceive(comm->fd, &recvSize, sizeof(int)));
  if (recvSize > size) {
    WARN("Message truncated : received %d bytes instead of %d\n", recvSize, size);
    return ncclInternalError;
  }
  NCCLCHECK(socketReceive(comm->fd, data, std::min(recvSize, size)));
  return ncclSuccess;
}

ncclResult_t bootstrapNetCreateHandle(void* opaqueHandle, const char* str) {
  struct bootstrapNetHandle* handle = (struct bootstrapNetHandle*) opaqueHandle;
  NCCLCHECK(GetSocketAddrFromString(&handle->connectAddr, str));
  return ncclSuccess;
}

struct extId {
  ncclNetHandle_t extHandleRoot;
  void* extListenComm;
  uint64_t hostHash;
  pid_t pid;
  int fd;
  pthread_t boostrapThread;
};

struct extInfo {
  int rank;
  int nranks;
  ncclNetHandle_t extHandleListenRoot;
  ncclNetHandle_t extHandleListen;
};

#include <sys/resource.h>

static ncclResult_t setFilesLimit() {
  struct rlimit filesLimit;
  SYSCHECK(getrlimit(RLIMIT_NOFILE, &filesLimit), "getrlimit");
  filesLimit.rlim_cur = filesLimit.rlim_max;
  SYSCHECK(setrlimit(RLIMIT_NOFILE, &filesLimit), "setrlimit");
  return ncclSuccess;
}

static void *bootstrapRoot(void* commId) {
  struct extInfo info;
  struct extId* id = (struct extId*)commId;
  ncclNetHandle_t *rankHandles = NULL;
  ncclNetHandle_t *rankHandlesRoot = NULL; // for initial rank <-> root information exchange
  ncclNetHandle_t zero = { 0 }; // for sanity checking
  void* tmpComm;
  ncclResult_t res;
  setFilesLimit();

  TRACE(NCCL_INIT, "BEGIN");
  /* Receive addresses from all ranks */
  int nranks = 0, c = 0;
  do {
    NCCLCHECKGOTO(bootstrapNetAccept(id->extListenComm, &tmpComm), res, out);
    NCCLCHECKGOTO(bootstrapNetRecv(tmpComm, &info, sizeof(info)), res, out);
    NCCLCHECKGOTO(bootstrapNetCloseRecv(tmpComm), res, out);

    if (c == 0) {
      nranks = info.nranks;
      NCCLCHECKGOTO(ncclCalloc(&rankHandles, nranks), res, out);
      NCCLCHECKGOTO(ncclCalloc(&rankHandlesRoot, nranks), res, out);
    }

    if (nranks != info.nranks) {
      WARN("Bootstrap Root : mismatch in rank count from procs %d : %d", nranks, info.nranks);
      goto out;
    }

    if (memcmp(&zero, &rankHandlesRoot[info.rank], sizeof(ncclNetHandle_t)) != 0) {
      WARN("Bootstrap Root : rank %d of %d ranks has already checked in", info.rank, nranks);
      goto out;
    }

    // Save the connection handle for that rank
    memcpy(rankHandlesRoot+info.rank, info.extHandleListenRoot, sizeof(ncclNetHandle_t));
    memcpy(rankHandles+info.rank, info.extHandleListen, sizeof(ncclNetHandle_t));

    ++c;
  } while (c < nranks);
  TRACE(NCCL_INIT, "COLLECTED HANDLES");

  // Send the connect handle for the next rank in the AllGather ring
  for (int r=0; r<nranks; ++r) {
    int next = (r+1) % nranks;
    void *tmpSendComm;
    NCCLCHECKGOTO(bootstrapNetConnect(0, rankHandlesRoot[r], &tmpSendComm), res, out);
    NCCLCHECKGOTO(bootstrapNetSend(tmpSendComm, rankHandles+next, sizeof(ncclNetHandle_t)), res, out);
    NCCLCHECKGOTO(bootstrapNetCloseSend(tmpSendComm), res, out);
  }
  TRACE(NCCL_INIT, "SENT OUT HANDLES");

out:
  bootstrapNetCloseListen(id->extListenComm);
  free(commId);
  if (rankHandles) free(rankHandles);
  if (rankHandlesRoot) free(rankHandlesRoot);

  TRACE(NCCL_INIT, "DONE");
  return NULL;
}

ncclResult_t bootstrapCreateRoot(ncclUniqueId* commId, bool idFromEnv) {
  struct extId* id = (struct extId*)commId;
  id->hostHash = getHostHash();
  NCCLCHECK(bootstrapNetListen(idFromEnv ? dontCareIf : 0, &id->extHandleRoot, &id->extListenComm));
  ncclUniqueId* threadIdCopy;
  NCCLCHECK(ncclCalloc(&threadIdCopy, 1));
  memcpy(threadIdCopy, id, sizeof(ncclUniqueId));
  pthread_create(&id->boostrapThread, NULL, bootstrapRoot, (void *)threadIdCopy);
  return ncclSuccess;
}

ncclResult_t bootstrapGetUniqueId(ncclUniqueId* out) {
  static_assert(sizeof(extId) < sizeof(ncclUniqueId), "NetId does not fit inside ncclUniqueId");
  extId* id = (extId*)out;

  char* env = getenv("NCCL_COMM_ID");
  if (env) {
    if (bootstrapNetCreateHandle(&id->extHandleRoot, env) != 0) {
      WARN("Invalid NCCL_COMM_ID, please use format: <ipv4>:<port> or [<ipv6>]:<port> or <hostname>:<port>");
      return ncclInvalidArgument;
    }
    id->pid = -1;
  } else {
    id->pid = getpid();
    NCCLCHECK(bootstrapCreateRoot(out, false));
  }

  return ncclSuccess;
}

struct unexConn {
  int peer;
  void* comm;
  struct unexConn* next;
};

struct extState {
  void* extBstrapListenComm;
  void* extBstrapRingRecvComm;
  void* extBstrapRingSendComm;
  ncclNetHandle_t* peerBstrapHandles;
  struct unexConn* unexpectedConnections;
  int rank;
  int nranks;
  int dev;
};

ncclResult_t bootstrapInit(ncclUniqueId* commId, int rank, int nranks, void** commState) {
  struct extId* id = (struct extId*)commId;
  bool idFromEnv = id->pid < 0;
  struct extState* state;
  NCCLCHECK(ncclCalloc(&state, 1));
  state->rank = rank;
  state->nranks = nranks;
  *commState = state;

  TRACE(NCCL_INIT, "rank %d nranks %d", rank, nranks);

  struct extInfo info = { 0 };
  info.rank = rank;
  info.nranks = nranks;
  void *tmpSendComm, *tmpRecvComm;
  // Pass the remote address to listen via info
  if (idFromEnv) {
    memcpy(&info.extHandleListen, &id->extHandleRoot, sizeof(ncclNetHandle_t));
    memcpy(&info.extHandleListenRoot, &id->extHandleRoot, sizeof(ncclNetHandle_t));
  }
  // listen will return the local address via info (specify interface type 'findSubnetIf')
  state->dev = idFromEnv ? findSubnetIf : 0;
  void* extBstrapListenCommRoot;
  NCCLCHECK(bootstrapNetListen(state->dev, &info.extHandleListen, &state->extBstrapListenComm));
  NCCLCHECK(bootstrapNetListen(state->dev, &info.extHandleListenRoot, &extBstrapListenCommRoot));

  // stagger connection times to avoid an overload of the root at very high rank counts
  if (nranks > 128) {
    long msec = rank;
    struct timespec tv;
    tv.tv_sec = msec / 1000;
    tv.tv_nsec = 1000000 * (msec % 1000);
    TRACE(NCCL_INIT, "rank %d delaying connection to root by %ld msec", rank, msec);
    (void) nanosleep(&tv, NULL);
  }

  // send info on my listening socket to root
  NCCLCHECK(bootstrapNetConnect(state->dev, id->extHandleRoot, &tmpSendComm));
  NCCLCHECK(bootstrapNetSend(tmpSendComm, &info, sizeof(info)));
  NCCLCHECK(bootstrapNetCloseSend(tmpSendComm));

  // get info on my "next" rank in the bootstrap ring from root
  ncclNetHandle_t extHandleNext;
  NCCLCHECK(bootstrapNetAccept(extBstrapListenCommRoot, &tmpRecvComm));
  NCCLCHECK(bootstrapNetRecv(tmpRecvComm, &extHandleNext, sizeof(extHandleNext)));
  NCCLCHECK(bootstrapNetCloseRecv(tmpRecvComm));
  NCCLCHECK(bootstrapNetCloseListen(extBstrapListenCommRoot));

  NCCLCHECK(bootstrapNetConnect(state->dev, extHandleNext, &state->extBstrapRingSendComm));
  // Accept the connect request from the previous rank in the AllGather ring
  NCCLCHECK(bootstrapNetAccept(state->extBstrapListenComm, &state->extBstrapRingRecvComm));

  // AllGather all listen handlers
  NCCLCHECK(ncclCalloc(&state->peerBstrapHandles, nranks));
  memcpy(state->peerBstrapHandles+rank, info.extHandleListen, sizeof(ncclNetHandle_t));
  NCCLCHECK(bootstrapAllGather(state, state->peerBstrapHandles, sizeof(ncclNetHandle_t)));

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
    NCCLCHECK(bootstrapNetSend(state->extBstrapRingSendComm, data+sslice*size, size));
    // Recv slice from the left
    NCCLCHECK(bootstrapNetRecv(state->extBstrapRingRecvComm, data+rslice*size, size));
  }

  TRACE(NCCL_INIT, "rank %d nranks %d size %d - DONE", rank, nranks, size);
  return ncclSuccess;
}

ncclResult_t bootstrapSend(void* commState, int peer, void* data, int size) {
  struct extState* state = (struct extState*)commState;
  void* tmpSendComm;
  NCCLCHECK(bootstrapNetConnect(state->dev, state->peerBstrapHandles[peer], &tmpSendComm));
  NCCLCHECK(bootstrapNetSend(tmpSendComm, &state->rank, sizeof(int)));
  NCCLCHECK(bootstrapNetSend(tmpSendComm, data, size));
  NCCLCHECK(bootstrapNetCloseSend(tmpSendComm));
  return ncclSuccess;
}

ncclResult_t unexpectedEnqueue(struct extState* state, int peer, void* comm) {
  // New unex
  struct unexConn* unex;
  NCCLCHECK(ncclCalloc(&unex, 1));
  unex->peer = peer;
  unex->comm = comm;

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

void* unexpectedDequeue(struct extState* state, int peer) {
  struct unexConn* elem = state->unexpectedConnections;
  struct unexConn* prev = NULL;
  while (elem) {
    if (elem->peer == peer) {
      if (prev == NULL) {
        state->unexpectedConnections = elem->next;
      } else {
        prev->next = elem->next;
      }
      void* comm = elem->comm;
      free(elem);
      return comm;
    }
    prev = elem;
    elem = elem->next;
  }
  return NULL;
}

// We can't know who we'll receive from, so we need to receive everything at once
ncclResult_t bootstrapRecv(void* commState, int peer, void* data, int size) {
  struct extState* state = (struct extState*)commState;

  void* tmpRecvComm;

  // Search unexpected connections first
  if ((tmpRecvComm = unexpectedDequeue(state, peer)) != NULL) {
    NCCLCHECK(bootstrapNetRecv(tmpRecvComm, ((char*)data), size));
    NCCLCHECK(bootstrapNetCloseRecv(tmpRecvComm));
    return ncclSuccess;
  }

  // Then look for new connections
  while (1) {
    NCCLCHECK(bootstrapNetAccept(state->extBstrapListenComm, &tmpRecvComm));
    int newPeer;
    NCCLCHECK(bootstrapNetRecv(tmpRecvComm, &newPeer, sizeof(int)));
    if (newPeer == peer) {
      NCCLCHECK(bootstrapNetRecv(tmpRecvComm, ((char*)data), size));
      NCCLCHECK(bootstrapNetCloseRecv(tmpRecvComm));
      return ncclSuccess;
    }
    // Unexpected connection. Save for later.
    NCCLCHECK(unexpectedEnqueue(state, newPeer, tmpRecvComm));
  }
}

ncclResult_t bootstrapClose(void* commState) {
  struct extState* state = (struct extState*)commState;
  if (state->unexpectedConnections != NULL) {
    WARN("Unexpected connections are not empty.\n");
    return ncclInternalError;
  }
  NCCLCHECK(bootstrapNetCloseListen(state->extBstrapListenComm));
  NCCLCHECK(bootstrapNetCloseSend(state->extBstrapRingSendComm));
  NCCLCHECK(bootstrapNetCloseRecv(state->extBstrapRingRecvComm));

  free(state->peerBstrapHandles);
  free(state);

  return ncclSuccess;
}
