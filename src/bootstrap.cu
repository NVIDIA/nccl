/*************************************************************************
 * Copyright (c) 2016-2018, NVIDIA CORPORATION. All rights reserved.
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

// Always use sockets for bootstrap
ncclNet_t* ncclBootstrapNet = &ncclNetSocket;

static ncclResult_t bootstrapListen(int dev, void* handle, void** listenComm) { NCCLCHECK(ncclBootstrapNet->listen(dev, handle, listenComm)); return ncclSuccess; }
static ncclResult_t bootstrapConnect(int dev, void* handle, void** sendComm) { NCCLCHECK(ncclBootstrapNet->connect(dev, handle, sendComm)); return ncclSuccess; }
static ncclResult_t bootstrapAccept(void* listenComm, void** recvComm) { NCCLCHECK(ncclBootstrapNet->accept(listenComm, recvComm)); return ncclSuccess; }
static ncclResult_t bootstrapTest(void* request, int* done, int* size) { NCCLCHECK(ncclBootstrapNet->test(request, done, size)); return ncclSuccess; }
static ncclResult_t bootstrapCloseSend(void* sendComm) { NCCLCHECK(ncclBootstrapNet->closeSend(sendComm)); return ncclSuccess; }
static ncclResult_t bootstrapCloseRecv(void* recvComm) { NCCLCHECK(ncclBootstrapNet->closeRecv(recvComm)); return ncclSuccess; }
static ncclResult_t bootstrapCloseListen(void* listenComm) { NCCLCHECK(ncclBootstrapNet->closeListen(listenComm)); return ncclSuccess; }

// Additional sync functions based on async + test for bootstrap, using host ptrs.
static ncclResult_t bootstrapSend(void* sendComm, void* data, int size) {
  void* request;
  NCCLCHECK(ncclBootstrapNet->isend(sendComm, data, size, NCCL_PTR_HOST, &request));
  int done = 0;
  while (!done) NCCLCHECK(bootstrapTest(request, &done, NULL));
  return ncclSuccess;
}
static ncclResult_t bootstrapRecv(void* recvComm, void* data, int size) {
  void* request;
  NCCLCHECK(ncclBootstrapNet->irecv(recvComm, data, size, NCCL_PTR_HOST, &request));
  int done = 0;
  while (!done) NCCLCHECK(bootstrapTest(request, &done, NULL));
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
  ncclNetHandle_t extHandleListenFromRoot;
  ncclNetHandle_t extHandleRing;
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
  ncclNetHandle_t *extHandleBstrap = NULL; // for initial rank <-> root information exchange
  ncclNetHandle_t *extHandleRing = NULL; // for bootstrap ring creation
  ncclNetHandle_t zero = { 0 }; // for sanity checking
  void* tmpComm;
  char* data = NULL;
  ncclResult_t res;
  setFilesLimit();

  /* Receive addresses from all ranks */
  int nranks = 0, c = 0;
  do {
    NCCLCHECKGOTO(bootstrapAccept(id->extListenComm, &tmpComm), res, out);
    NCCLCHECKGOTO(bootstrapRecv(tmpComm, &info, sizeof(info)), res, out);
    NCCLCHECKGOTO(bootstrapCloseRecv(tmpComm), res, out);

    if (c == 0) {
      extHandleBstrap = (ncclNetHandle_t *)calloc(info.nranks, sizeof(ncclNetHandle_t));
      extHandleRing = (ncclNetHandle_t *)calloc(info.nranks, sizeof(ncclNetHandle_t));
      if (extHandleBstrap == NULL || extHandleRing == NULL) {
        WARN("Bootstrap thread : failed to allocate memory");
        goto out;
      }
      nranks = info.nranks;
    }

    if (nranks != info.nranks) {
      WARN("Bootstrap Root : mismatch in rank count from procs %d : %d", nranks, info.nranks);
      goto out;
    }

    if (memcmp(&zero, &extHandleBstrap[info.rank], sizeof(ncclNetHandle_t)) != 0) {
      WARN("Bootstrap Root : rank %d of %d ranks has already checked in", info.rank, nranks);
      goto out;
    }

    // Save the connection handle for connecting back to the ranks
    memcpy(&extHandleBstrap[info.rank], info.extHandleListenFromRoot, sizeof(ncclNetHandle_t));
    // Save the connection handle for the AllGather ring
    memcpy(&extHandleRing[info.rank], info.extHandleRing, sizeof(ncclNetHandle_t));

    ++c;
  } while (c < nranks);

  // Send the connect handle for the next rank in the AllGather ring
  for (int r=0; r<nranks; ++r) {
    int next = (r+1) % nranks;
    void *tmpSendComm;
    NCCLCHECKGOTO(bootstrapConnect(0, extHandleBstrap[r], &tmpSendComm), res, out);
    NCCLCHECKGOTO(bootstrapSend(tmpSendComm, &extHandleRing[next], sizeof(ncclNetHandle_t)), res, out);
    NCCLCHECKGOTO(bootstrapCloseSend(tmpSendComm), res, out);
  }

out:
  bootstrapCloseListen(id->extListenComm);
  free(commId);
  if (data) free(data);
  return NULL;
}

ncclResult_t bootstrapCreateRoot(ncclUniqueId* commId, bool idFromEnv) {
  struct extId* id = (struct extId*)commId;
  id->hostHash = getHostHash();
  NCCLCHECK(bootstrapListen(idFromEnv ? dontCareIf : 0, &id->extHandleRoot, &id->extListenComm));
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
    if (ncclSocketCreateHandle(&id->extHandleRoot, env) != 0) {
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

struct extState {
  void* extBstrapRingRecvComm;
  void* extBstrapRingSendComm;
  ncclNetHandle_t extBstrapRootHandle;
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
  void* extBstrapRootListenComm; // comm on which we accept root's connections

  struct extInfo info = { 0 };
  info.rank = rank;
  info.nranks = nranks;
  void *tmpSendComm, *extBstrapRingListenComm, *tmpRecvComm;
  // Pass the remote address to listen via info
  if (idFromEnv) {
    memcpy(&info.extHandleListenFromRoot, &id->extHandleRoot, sizeof(ncclNetHandle_t));
    memcpy(&info.extHandleRing, &id->extHandleRoot, sizeof(ncclNetHandle_t));
  }
  // listen will return the local address via info (specify interface type 'findSubnetIf')
  state->dev = idFromEnv ? findSubnetIf : 0;
  NCCLCHECK(bootstrapListen(state->dev, &info.extHandleListenFromRoot, &extBstrapRootListenComm));
  NCCLCHECK(bootstrapListen(state->dev, &info.extHandleRing, &extBstrapRingListenComm)); // AllGather Ring

  memcpy(&state->extBstrapRootHandle, &id->extHandleRoot, sizeof(ncclNetHandle_t));
  // send info on my listening sockets to root
  NCCLCHECK(bootstrapConnect(state->dev, id->extHandleRoot, &tmpSendComm));
  NCCLCHECK(bootstrapSend(tmpSendComm, &info, sizeof(info)));
  NCCLCHECK(bootstrapCloseSend(tmpSendComm));

  // get info on my "next" rank in the bootstrap ring from root
  ncclNetHandle_t extHandleNext;
  NCCLCHECK(bootstrapAccept(extBstrapRootListenComm, &tmpRecvComm));
  NCCLCHECK(bootstrapRecv(tmpRecvComm, &extHandleNext, sizeof(extHandleNext)));
  NCCLCHECK(bootstrapCloseRecv(tmpRecvComm));

  NCCLCHECK(bootstrapConnect(state->dev, extHandleNext, &state->extBstrapRingSendComm));
  // Accept the connect request from the previous rank in the AllGather ring
  NCCLCHECK(bootstrapAccept(extBstrapRingListenComm, &state->extBstrapRingRecvComm));
  NCCLCHECK(bootstrapCloseListen(extBstrapRingListenComm));
  NCCLCHECK(bootstrapCloseListen(extBstrapRootListenComm));

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
    int rslice = (rank - i - 1 + nranks) % nranks;
    int sslice = (rank - i + nranks) % nranks;

    // Send slice to the right
    NCCLCHECK(bootstrapSend(state->extBstrapRingSendComm, data+sslice*size, size));
    // Recv slice from the left
    NCCLCHECK(bootstrapRecv(state->extBstrapRingRecvComm, data+rslice*size, size));
  }

  TRACE(NCCL_INIT, "rank %d nranks %d size %d - DONE", rank, nranks, size);
  return ncclSuccess;
}

ncclResult_t bootstrapClose(void* commState) {
  struct extState* state = (struct extState*)commState;

  NCCLCHECK(bootstrapCloseSend(state->extBstrapRingSendComm));
  NCCLCHECK(bootstrapCloseRecv(state->extBstrapRingRecvComm));

  free(state);

  return ncclSuccess;
}
