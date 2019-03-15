/*************************************************************************
 * Copyright (c) 2016-2019, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "nccl.h"
#include "core.h"
#include "socket.h"
#include "net.h"

#include <assert.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <poll.h>
#include <limits.h>

/* Init functions */
static char ncclNetIfNames[MAX_IF_NAME_SIZE*MAX_IFS];
static union socketAddress ncclNetIfAddrs[MAX_IFS];
static int ncclNetIfs = -1;
pthread_mutex_t ncclSocketLock = PTHREAD_MUTEX_INITIALIZER;

ncclResult_t ncclSocketInit(ncclDebugLogger_t logFunction) {
  if (ncclNetIfs == -1) {
    pthread_mutex_lock(&ncclSocketLock);
    if (ncclNetIfs == -1) {
      ncclNetIfs = findInterfaces(ncclNetIfNames, ncclNetIfAddrs, MAX_IF_NAME_SIZE, MAX_IFS);
      if (ncclNetIfs <= 0) {
        WARN("NET/Socket : no interface found");
        return ncclInternalError;
      } else {
        char line[1024];
        char addrline[1024];
        line[0] = '\0';
        for (int i=0; i<ncclNetIfs; i++) {
          snprintf(line+strlen(line), 1023-strlen(line), " [%d]%s:%s", i, ncclNetIfNames+i*MAX_IF_NAME_SIZE,
              socketToString(&ncclNetIfAddrs[i].sa, addrline));
        }
        line[1023] = '\0';
        INFO(NCCL_INIT|NCCL_NET,"NET/Socket : Using%s", line);
      }
    }
    pthread_mutex_unlock(&ncclSocketLock);
  }
  return ncclSuccess;
}

ncclResult_t ncclSocketPtrSupport(int dev, int* supportedTypes) {
  *supportedTypes = NCCL_PTR_HOST;
  return ncclSuccess;
}

ncclResult_t ncclSocketDevices(int* ndev) {
  *ndev = ncclNetIfs;
  return ncclSuccess;
}

ncclResult_t ncclSocketPciPath(int dev, char** path) {
  char devicepath[PATH_MAX];
  snprintf(devicepath, PATH_MAX, "/sys/class/net/%s/device", ncclNetIfNames+dev*MAX_IF_NAME_SIZE);
  *path = realpath(devicepath, NULL);
  if (*path == NULL) {
    INFO(NCCL_NET|NCCL_INIT, "Could not find real path of %s", devicepath);
    return ncclSystemError;
  }
  return ncclSuccess;
}

static ncclResult_t GetSocketAddr(int dev, union socketAddress* addr) {
  if (dev >= ncclNetIfs) return ncclInternalError;
  memcpy(addr, ncclNetIfAddrs+dev, sizeof(*addr));
  return ncclSuccess;
}

/* Communication functions */

struct ncclSocketHandle {
  union socketAddress connectAddr;
};

struct ncclSocketRequest {
  int op;
  void* data;
  int size;
  int fd;
  int offset;
  int used;
};

struct ncclSocketReqs {
  struct ncclSocketRequest* requests;
};

struct ncclSocketComm {
  int fd;
  struct ncclSocketReqs reqs;
};

ncclResult_t ncclSocketNewComm(struct ncclSocketComm** comm) {
  NCCLCHECK(ncclCalloc(comm, 1));
  (*comm)->fd = -1;
  return ncclSuccess;
}

ncclResult_t ncclSocketCreateHandle(void* opaqueHandle, const char* str) {
  struct ncclSocketHandle* handle = (struct ncclSocketHandle*) opaqueHandle;
  NCCLCHECK(GetSocketAddrFromString(&(handle->connectAddr), str));
  return ncclSuccess;
}

ncclResult_t ncclSocketListen(int dev, void* opaqueHandle, void** listenComm) {
  struct ncclSocketHandle* handle = (struct ncclSocketHandle*) opaqueHandle;
  static_assert(sizeof(struct ncclSocketHandle) < NCCL_NET_HANDLE_MAXSIZE, "ncclSocketHandle size too large");
  // if dev >= 0, listen based on dev
  if (dev >= 0) {
    NCCLCHECK(GetSocketAddr(dev, &(handle->connectAddr)));
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
  struct ncclSocketComm* comm;
  NCCLCHECK(ncclSocketNewComm(&comm));
  NCCLCHECK(createListenSocket(&comm->fd, &handle->connectAddr));
  *listenComm = comm;
  return ncclSuccess;
}

ncclResult_t ncclSocketConnect(int dev, void* opaqueHandle, void** sendComm) {
  struct ncclSocketComm* comm;
  NCCLCHECK(ncclSocketNewComm(&comm));
  struct ncclSocketHandle* handle = (struct ncclSocketHandle*) opaqueHandle;
  NCCLCHECK(connectAddress(&comm->fd, &handle->connectAddr));
  *sendComm = comm;
  return ncclSuccess;
}

ncclResult_t ncclSocketAccept(void* listenComm, void** recvComm) {
  struct ncclSocketComm* lComm = (struct ncclSocketComm*)listenComm;
  struct ncclSocketComm* rComm;
  NCCLCHECK(ncclSocketNewComm(&rComm));
  struct sockaddr_in sockaddr;
  socklen_t socklen = sizeof(struct sockaddr_in);
  SYSCHECKVAL(accept(lComm->fd, (struct sockaddr*)&sockaddr, &socklen), "accept", rComm->fd);
  *recvComm = rComm;
  return ncclSuccess;
}

#define MAX_REQUESTS 128

ncclResult_t ncclSocketGetRequest(struct ncclSocketReqs* reqs, int op, void* data, int size, int fd, struct ncclSocketRequest** req) {
  if (reqs->requests == NULL) {
    NCCLCHECK(ncclCalloc(&reqs->requests, MAX_REQUESTS));
  }
  for (int i=0; i<MAX_REQUESTS; i++) {
    struct ncclSocketRequest* r = reqs->requests+i;
    if (r->used == 0) {
      r->op = op;
      r->data = data;
      r->size = size;
      r->fd = fd;
      r->offset = -1;
      r->used = 1;
      *req = r;
      return ncclSuccess;
    }
  }
  WARN("Socket : unable to allocate requests");
  return ncclInternalError;
}

ncclResult_t ncclSocketTest(void* request, int* done, int* size) {
  *done = 0;
  struct ncclSocketRequest *r = (struct ncclSocketRequest*)request;
  if (r == NULL) {
    WARN("NET/Socket : test called with NULL request");
    return ncclInternalError;
  }
  if (r->offset == -1) { /* try to send/recv size */
    int data = r->size;
    int offset = 0;
    NCCLCHECK(socketProgress(r->op, r->fd, &data, sizeof(int), &offset));

    if (offset == 0) return ncclSuccess; /* Not ready -- retry later */

    // Not sure we could ever receive less than 4 bytes, but just in case ...
    if (offset < sizeof(int)) NCCLCHECK(socketWait(r->op, r->fd, &data, sizeof(int), &offset));

    // Check size is less or equal to the size provided by the user
    if (r->op == NCCL_SOCKET_RECV && data > r->size) {
      WARN("NET/Socket : message truncated : receiving %d bytes instead of %d", data, r->size);
      return ncclInternalError;
    }
    r->size = data;
    r->offset = 0;
  }
  if (r->offset < r->size) {
    NCCLCHECK(socketProgress(r->op, r->fd, r->data, r->size, &r->offset));
  }
  if (r->offset == r->size) {
    if (size) *size = r->size;
    *done = 1;
    r->used = 0;
  }
  return ncclSuccess;
}

ncclResult_t ncclSocketRegMr(void* comm, void* data, int size, int type, void** mhandle) {
  return (type != NCCL_PTR_HOST) ? ncclInternalError : ncclSuccess;
}
ncclResult_t ncclSocketDeregMr(void* comm, void* mhandle) { return ncclSuccess; }

ncclResult_t ncclSocketIsend(void* sendComm, void* data, int size, void* mhandle, void** request) {
  struct ncclSocketComm* comm = (struct ncclSocketComm*)sendComm;
  NCCLCHECK(ncclSocketGetRequest(&comm->reqs, NCCL_SOCKET_SEND, data, size, comm->fd, (struct ncclSocketRequest**)request));
  return ncclSuccess;
}

ncclResult_t ncclSocketIrecv(void* recvComm, void* data, int size, void* mhandle, void** request) {
  struct ncclSocketComm* comm = (struct ncclSocketComm*)recvComm;
  NCCLCHECK(ncclSocketGetRequest(&comm->reqs, NCCL_SOCKET_RECV, data, size, comm->fd, (struct ncclSocketRequest**)request));
  return ncclSuccess;
}

ncclResult_t ncclSocketFlush(void* recvComm, void* data, int size, void* mhandle) {
  // We don't support CUDA pointers, so we don't need a flush operation
  return ncclInternalError;
}

ncclResult_t ncclSocketClose(void* opaqueComm) {
  struct ncclSocketComm* comm = (struct ncclSocketComm*)opaqueComm;
  if (comm) {
    free(comm->reqs.requests);
    close(comm->fd);
    free(comm);
  }
  return ncclSuccess;
}

ncclNet_t ncclNetSocket = {
  "Socket",
  ncclSocketInit,
  ncclSocketDevices,
  ncclSocketPciPath,
  ncclSocketPtrSupport,
  ncclSocketListen,
  ncclSocketConnect,
  ncclSocketAccept,
  ncclSocketRegMr,
  ncclSocketDeregMr,
  ncclSocketIsend,
  ncclSocketIrecv,
  ncclSocketFlush,
  ncclSocketTest,
  ncclSocketClose,
  ncclSocketClose,
  ncclSocketClose
};
