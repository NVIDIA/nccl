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
  ncclNetHandle_t extHandle;
  void* extListenComm;
  uint64_t hostHash;
  pid_t pid;
  int fd;
  pthread_t boostrapThread;
};

struct bootstrapOp {
  int op;
  int size;
};

struct extInfo {
  int rank;
  int nranks;
  ncclNetHandle_t extHandle;
};

enum {
  BOOTSTRAP_ALLGATHER = 1,
  BOOTSTRAP_RINGEXCHANGE,
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
  struct bootstrapOp bop;
  void **extSendComm = NULL;
  void **extRecvComm = NULL;
  int size, alloc_size = 0;
  char* data = NULL;
  ncclResult_t res;
  setFilesLimit();

  /* Receive addresses from all ranks */
  int nranks = 0, c = 0;
  do {
    void* tmpRecvComm;
    NCCLCHECKGOTO(bootstrapAccept(id->extListenComm, &tmpRecvComm), res, out);
    NCCLCHECKGOTO(bootstrapRecv(tmpRecvComm, &info, sizeof(info)), res, out);
    if (!c) {
      extSendComm = (void**)calloc(info.nranks, sizeof(void*));
      extRecvComm = (void**)calloc(info.nranks, sizeof(void*));
      if (extSendComm == NULL || extRecvComm == NULL) {
        WARN("Bootstrap thread : failed to allocate memory");
        goto out;
      }
      nranks = info.nranks;
    }

    if (nranks != info.nranks) {
      WARN("Bootstrap Root : mismatch in rank count from procs %d : %d", nranks, info.nranks);
      goto out;
    }

    extRecvComm[info.rank] = tmpRecvComm;
    NCCLCHECKGOTO(bootstrapConnect(0, info.extHandle, extSendComm+info.rank), res, out);
    c++;
  } while (c < nranks);

  do {
    NCCLCHECKGOTO(bootstrapRecv(extRecvComm[0], &bop, sizeof(struct bootstrapOp)), res, out);
    if (bop.size == -1) {
      break;
    } else {
      size = bop.size;
      if (size*nranks*2 > alloc_size) {
        if (data) free(data); data = NULL;
        NCCLCHECKGOTO(ncclCalloc(&data, size*nranks*2), res, out);
        alloc_size = size*nranks*2;
      }
    }

    if (bop.op == BOOTSTRAP_ALLGATHER) {
      for (int r=0; r<nranks; r++) {
        NCCLCHECKGOTO(bootstrapRecv(extRecvComm[r], data+size*r, size), res, out);
      }

      for (int r=0; r<nranks; r++) {
        NCCLCHECKGOTO(bootstrapSend(extSendComm[r], data, size*nranks), res, out);
      }
    } else if (bop.op == BOOTSTRAP_RINGEXCHANGE) {
      // Receive from all and build total table
      for (int r=0; r<nranks; r++) {
        NCCLCHECKGOTO(bootstrapRecv(extRecvComm[r], data+r*2*size, 2*size), res, out);
      }

      // Get prev/next request from everyone and answer.
      for (int r=0; r<nranks; r++) {
        int offset;
        NCCLCHECKGOTO(bootstrapRecv(extRecvComm[r], &offset, sizeof(int)), res, out);
        NCCLCHECKGOTO(bootstrapSend(extSendComm[r], data+offset, size), res, out);
        NCCLCHECKGOTO(bootstrapRecv(extRecvComm[r], &offset, sizeof(int)), res, out);
        NCCLCHECKGOTO(bootstrapSend(extSendComm[r], data+offset, size), res, out);
      }
    } else {
      WARN("Bootstrap Root : invalid op type received %d", bop.op);
      break;
    }
  } while (1);

out:
  bootstrapCloseListen(id->extListenComm);
  for (int r=0; r<nranks; r++) {
    if (extSendComm[r]) bootstrapCloseSend(extSendComm[r]);
    if (extRecvComm[r]) bootstrapCloseRecv(extRecvComm[r]);
  }
  free(commId);
  if (data) free(data);
  if (extSendComm) free(extSendComm);
  if (extRecvComm) free(extRecvComm);
  return NULL;
}

ncclResult_t bootstrapCreateRoot(ncclUniqueId* commId, bool idFromEnv) {
  struct extId* id = (struct extId*)commId;
  id->hostHash = getHostHash();
  NCCLCHECK(bootstrapListen(idFromEnv ? dontCareIf : 0, &id->extHandle, &id->extListenComm));
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
    if (ncclSocketCreateHandle(&id->extHandle, env) != 0) {
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
  void* extRecvComm;
  void* extSendComm;
  int rank;
  int nranks;
};

ncclResult_t bootstrapInit(ncclUniqueId* commId, int rank, int nranks, void** commState) {
  struct extId* id = (struct extId*)commId;
  bool idFromEnv = id->pid < 0;
  struct extState* state;
  NCCLCHECK(ncclCalloc(&state, 1));
  state->rank = rank;
  state->nranks = nranks;
  *commState = state;

  struct extInfo info;
  info.rank = rank;
  info.nranks = nranks;
  void* tmpListenComm;
  // Pass the remote address to listen via info
  if (idFromEnv) {
    memcpy(&info.extHandle, &id->extHandle, sizeof(ncclNetHandle_t));
  }
  // listen will return the local address via info ('findSubnetIf' indicates that the net device is unknown)
  int dev = idFromEnv ? findSubnetIf : 0;
  NCCLCHECK(bootstrapListen(dev, &info.extHandle, &tmpListenComm));
  NCCLCHECK(bootstrapConnect(dev, id->extHandle, &state->extSendComm));
  NCCLCHECK(bootstrapSend(state->extSendComm, &info, sizeof(info)));
  NCCLCHECK(bootstrapAccept(tmpListenComm, &state->extRecvComm));
  NCCLCHECK(bootstrapCloseListen(tmpListenComm));

  return ncclSuccess;
}

ncclResult_t bootstrapAllGather(void* commState, void* allData, int size) {
  struct extState* state = (struct extState*)commState;
  char* data = (char*)allData;
  struct bootstrapOp bop;

  bop.op = BOOTSTRAP_ALLGATHER;
  bop.size = size;

  if (!state->rank) {
    NCCLCHECK(bootstrapSend(state->extSendComm, &bop, sizeof(struct bootstrapOp)));
  }

  NCCLCHECK(bootstrapSend(state->extSendComm, data+state->rank*size, size));
  NCCLCHECK(bootstrapRecv(state->extRecvComm, data, size*state->nranks));

  return ncclSuccess;
}

ncclResult_t bootstrapRingExchange(void* commState, void* prevNextData, int prev, int next, int size) {
  struct extState* state = (struct extState*)commState;
  char* mydata = (char*)prevNextData;
  int prev_offset = prev*2*size+size, next_offset = next*2*size;

  struct bootstrapOp bop;
  bop.op = BOOTSTRAP_RINGEXCHANGE;
  bop.size = size;

  if (!state->rank) {
    NCCLCHECK(bootstrapSend(state->extSendComm, &bop, sizeof(struct bootstrapOp)));
  }

  // Send data to root
  NCCLCHECK(bootstrapSend(state->extSendComm, mydata, 2*size));

  // Receive prev and next data
  NCCLCHECK(bootstrapSend(state->extSendComm, &prev_offset, sizeof(int)));
  NCCLCHECK(bootstrapRecv(state->extRecvComm, mydata, size));
  NCCLCHECK(bootstrapSend(state->extSendComm, &next_offset, sizeof(int)));
  NCCLCHECK(bootstrapRecv(state->extRecvComm, mydata+size, size));


  return ncclSuccess;
}

ncclResult_t bootstrapClose(void* commState) {
  struct extState* state = (struct extState*)commState;
  struct bootstrapOp bop;
  bop.size = -1;

  if (!state->rank) {
    NCCLCHECK(bootstrapSend(state->extSendComm, &bop, sizeof(struct bootstrapOp)));
  }

  NCCLCHECK(bootstrapCloseSend(state->extSendComm));
  NCCLCHECK(bootstrapCloseRecv(state->extRecvComm));

  free(state);

  return ncclSuccess;
}
