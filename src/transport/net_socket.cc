/*************************************************************************
 * Copyright (c) 2016-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "comm.h"
#include "core.h"
#include "socket.h"
#include "net.h"
#include "param.h"

#include <assert.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <poll.h>
#include <limits.h>
#include <fcntl.h>

/* Init functions */
static int ncclNetIfs = -1;
struct ncclSocketDev {
  union socketAddress addr;
  char devName[MAX_IF_NAME_SIZE];
  char* pciPath;
};
static struct ncclSocketDev ncclSocketDevs[MAX_IFS];

pthread_mutex_t ncclSocketLock = PTHREAD_MUTEX_INITIALIZER;

static ncclResult_t ncclSocketGetPciPath(char* devName, char** pciPath) {
  char devicePath[PATH_MAX];
  snprintf(devicePath, PATH_MAX, "/sys/class/net/%s/device", devName);
  // May return NULL if the file doesn't exist.
  *pciPath = realpath(devicePath, NULL);
  return ncclSuccess;
}

ncclResult_t ncclSocketInit(ncclDebugLogger_t logFunction) {
  if (ncclNetIfs == -1) {
    pthread_mutex_lock(&ncclSocketLock);
    if (ncclNetIfs == -1) {
      char names[MAX_IF_NAME_SIZE*MAX_IFS];
      union socketAddress addrs[MAX_IFS];
      ncclNetIfs = findInterfaces(names, addrs, MAX_IF_NAME_SIZE, MAX_IFS);
      if (ncclNetIfs <= 0) {
        WARN("NET/Socket : no interface found");
        return ncclInternalError;
      } else {
        char line[1024];
        char addrline[1024];
        line[0] = '\0';
        for (int i=0; i<ncclNetIfs; i++) {
          strcpy(ncclSocketDevs[i].devName, names+i*MAX_IF_NAME_SIZE);
          memcpy(&ncclSocketDevs[i].addr, addrs+i, sizeof(union socketAddress));
          NCCLCHECK(ncclSocketGetPciPath(ncclSocketDevs[i].devName, &ncclSocketDevs[i].pciPath));
          snprintf(line+strlen(line), 1023-strlen(line), " [%d]%s:%s", i, names+i*MAX_IF_NAME_SIZE,
              socketToString(&addrs[i].sa, addrline));
        }
        line[1023] = '\0';
        INFO(NCCL_INIT|NCCL_NET,"NET/Socket : Using%s", line);
      }
    }
    pthread_mutex_unlock(&ncclSocketLock);
  }
  return ncclSuccess;
}

ncclResult_t ncclSocketDevices(int* ndev) {
  *ndev = ncclNetIfs;
  return ncclSuccess;
}

static ncclResult_t ncclSocketGetSpeed(char* devName, int* speed) {
  *speed = 0;
  char speedPath[PATH_MAX];
  sprintf(speedPath, "/sys/class/net/%s/speed", devName);
  int fd = open(speedPath, O_RDONLY);
  if (fd != -1) {
    char speedStr[] = "        ";
    if (read(fd, speedStr, sizeof(speedStr)-1) > 0) {
      *speed = strtol(speedStr, NULL, 0);
    }
    close(fd);
  }
  if (*speed <= 0) {
    INFO(NCCL_NET, "Could not get speed from %s. Defaulting to 10 Gbps.", speedPath);
    *speed = 10000;
  }
  return ncclSuccess;
}

ncclResult_t ncclSocketGetProperties(int dev, ncclNetProperties_t* props) {
  props->name = ncclSocketDevs[dev].devName;
  props->pciPath = ncclSocketDevs[dev].pciPath;
  props->guid = dev;
  props->ptrSupport = NCCL_PTR_HOST;
  NCCLCHECK(ncclSocketGetSpeed(props->name, &props->speed));
  props->port = 0;
  props->maxComms = 65536;
  return ncclSuccess;
}

ncclResult_t GetSocketAddr(int dev, union socketAddress* addr) {
  if (dev >= ncclNetIfs) return ncclInternalError;
  memcpy(addr, &ncclSocketDevs[dev].addr, sizeof(*addr));
  return ncclSuccess;
}

/* Communication functions */

#define MAX_SOCKETS 64
#define MAX_THREADS 16
#define MAX_REQUESTS 128
#define MAX_QUEUE_LEN MAX_REQUESTS
#define MIN_CHUNKSIZE (64*1024)

NCCL_PARAM(SocketNsocksPerThread, "NSOCKS_PERTHREAD", -2);
NCCL_PARAM(SocketNthreads, "SOCKET_NTHREADS", -2);

struct ncclSocketHandle {
  union socketAddress connectAddr;
  int nSocks;
  int nThreads;
};

struct ncclSocketTask {
  int op;
  void* data;
  int size;
  int fd;
  int offset;
  int used;
  ncclResult_t result;
};

struct ncclSocketRequest {
  int op;
  void* data;
  int size;
  int ctrlFd;
  int offset;
  int used;
  struct ncclSocketComm* comm;
  struct ncclSocketTask* tasks[MAX_SOCKETS];
  int nSubs;
};

struct ncclSocketTaskQueue {
  int next;
  struct ncclSocketTask* tasks;
};

enum threadState {start, stop};

struct ncclSocketThreadResources {
  struct ncclSocketTaskQueue threadTaskQueue;
  enum threadState state;
  struct ncclSocketComm* comm;
  pthread_mutex_t threadLock;
  pthread_cond_t  threadCond;
};

struct ncclSocketListenComm {
  int fd;
  int nSocks;
  int nThreads;
};

struct ncclSocketComm {
  int ctrlFd;
  int fds[MAX_SOCKETS];
  int nSocks;
  int nThreads;
  int nextFd;
  struct ncclSocketRequest requests[MAX_REQUESTS];
  pthread_t helperThread[MAX_THREADS];
  struct ncclSocketThreadResources threadResources[MAX_THREADS];
};

void* persistentSocketThread(void *args_) {
  struct ncclSocketThreadResources* resource = (struct ncclSocketThreadResources*)args_;
  struct ncclSocketComm* comm = resource->comm;
  volatile enum threadState* state = &resource->state;
  struct ncclSocketTaskQueue* myQueue = &resource->threadTaskQueue;
  int nSocksPerThread = comm->nSocks / comm->nThreads;
  while (1) {
    int idle = 1;
    int mark = myQueue->next; // mark newest task seen
    for (int i=0; i<MAX_QUEUE_LEN; i+=nSocksPerThread) {
      int repeat;
      do {
        repeat = 0;
        for (int j=0; j<nSocksPerThread; j++) {
          struct ncclSocketTask* r = myQueue->tasks+i+j;
          if (r != NULL && r->used == 1 && r->offset < r->size) {
            r->result = socketProgress(r->op, r->fd, r->data, r->size, &r->offset);
            if (r->result != ncclSuccess) {
              WARN("NET/Socket : socket progress error");
              return NULL;
            }
            idle = 0;
            if (r->offset < r->size) repeat = 1;
          }
        }
      } while (repeat);
    }
    if (idle) {
      pthread_mutex_lock(&resource->threadLock);
      while (mark == myQueue->next && *state != stop) { // no new tasks, wait
        pthread_cond_wait(&resource->threadCond, &resource->threadLock);
      }
      pthread_mutex_unlock(&resource->threadLock);
    }
    if (*state == stop) return NULL;
  }
}

ncclResult_t ncclSocketGetNsockNthread(int dev, int* ns, int* nt) {
  int nSocksPerThread = ncclParamSocketNsocksPerThread();
  int nThreads = ncclParamSocketNthreads();
  if (nThreads > MAX_THREADS) {
    WARN("NET/Socket : NCCL_SOCKET_NTHREADS is greater than the maximum allowed, setting to %d", MAX_THREADS);
    nThreads = MAX_THREADS;
  }
  if (nThreads == -2 || nSocksPerThread == -2) {
    // Auto-detection
    int autoNt=0, autoNs=1; // By default, we only use the main thread and do not spawn extra threads
    char vendorPath[PATH_MAX];
    snprintf(vendorPath, PATH_MAX, "/sys/class/net/%s/device/vendor", ncclSocketDevs[dev].devName);
    char* rPath = realpath(vendorPath, NULL);
    int fd = open(rPath, O_RDONLY);
    free(rPath);
    if (fd == -1) {
      // Could not find device vendor. This is handled silently so
      // we don't want to print an INFO error.
      TRACE(NCCL_NET, "Open of %s failed : %s\n", vendorPath, strerror(errno));
      goto end;
    }
    char vendor[7];
    strncpy(vendor, "0x0000", 7);
    int len;
    SYSCHECKVAL(read(fd, vendor, 6), "read", len);
    SYSCHECK(close(fd), "close");
    if (strcmp(vendor, "0x1d0f") == 0) { // AWS
      autoNt = 2;
      autoNs = 8;
    } else if (strcmp(vendor, "0x1ae0") == 0) { // GCP
      autoNt = 4;
      autoNs = 1;
    }
end:
    if (nThreads == -2) nThreads = autoNt;
    if (nSocksPerThread == -2) nSocksPerThread = autoNs;
  }
  int nSocks = nSocksPerThread * nThreads;
  if (nSocks > MAX_SOCKETS) {
    nSocksPerThread = MAX_SOCKETS/nThreads;
    WARN("NET/Socket : the total number of sockets is greater than the maximum allowed, setting NCCL_NSOCKS_PERTHREAD to %d", nSocksPerThread);
    nSocks = nSocksPerThread * nThreads;
  }
  *ns = nSocks;
  *nt = nThreads;
  if (nSocks > 0) INFO(NCCL_INIT, "NET/Socket: Using %d threads and %d sockets per thread", nThreads, nSocksPerThread);
  return ncclSuccess;
}

ncclResult_t ncclSocketNewListenComm(struct ncclSocketListenComm** comm) {
  NCCLCHECK(ncclCalloc(comm, 1));
  (*comm)->fd = -1;
  return ncclSuccess;
}

ncclResult_t ncclSocketNewComm(struct ncclSocketComm** comm) {
  NCCLCHECK(ncclCalloc(comm, 1));
  (*comm)->ctrlFd = -1;
  for (int i=0; i < MAX_SOCKETS; i++) {
    (*comm)->fds[i] = -1;
  }
  (*comm)->nextFd = 0;
  return ncclSuccess;
}

ncclResult_t ncclSocketListen(int dev, void* opaqueHandle, void** listenComm) {
  if (dev < 0) { // data transfer socket is based on specified dev
    return ncclInternalError;
  }
  struct ncclSocketHandle* handle = (struct ncclSocketHandle*) opaqueHandle;
  static_assert(sizeof(struct ncclSocketHandle) < NCCL_NET_HANDLE_MAXSIZE, "ncclSocketHandle size too large");
  struct ncclSocketListenComm* comm;
  NCCLCHECK(ncclSocketNewListenComm(&comm));
  NCCLCHECK(GetSocketAddr(dev, &handle->connectAddr));
  NCCLCHECK(createListenSocket(&comm->fd, &handle->connectAddr));
  NCCLCHECK(ncclSocketGetNsockNthread(dev, &comm->nSocks, &comm->nThreads));
  handle->nSocks = comm->nSocks;
  handle->nThreads = comm->nThreads;
  *listenComm = comm;
  return ncclSuccess;
}

ncclResult_t ncclSocketConnect(int dev, void* opaqueHandle, void** sendComm) {
  if (dev < 0) { // data transfer socket is based on specified dev
    return ncclInternalError;
  }
  struct ncclSocketComm* comm;
  NCCLCHECK(ncclSocketNewComm(&comm));
  struct ncclSocketHandle* handle = (struct ncclSocketHandle*) opaqueHandle;
  comm->nSocks = handle->nSocks;
  comm->nThreads = handle->nThreads;
  for (int i=0; i<comm->nSocks+1; i++) {
    int tmpFd, offset=0;
    NCCLCHECK(connectAddress(&tmpFd, &handle->connectAddr));
    NCCLCHECK(socketWait(NCCL_SOCKET_SEND, tmpFd, &i, sizeof(int), &offset));
    if (i == comm->nSocks) comm->ctrlFd = tmpFd;
    else comm->fds[i] = tmpFd;
  }
  *sendComm = comm;
  return ncclSuccess;
}

ncclResult_t ncclSocketAccept(void* listenComm, void** recvComm) {
  struct ncclSocketListenComm* lComm = (struct ncclSocketListenComm*)listenComm;
  struct ncclSocketComm* rComm;
  NCCLCHECK(ncclSocketNewComm(&rComm));
  rComm->nSocks = lComm->nSocks;
  rComm->nThreads = lComm->nThreads;
  for (int i=0; i<rComm->nSocks+1; i++) {
    int tmpFd, sendSockIdx, offset=0;
    struct sockaddr_in sockaddr;
    socklen_t socklen = sizeof(struct sockaddr_in);
    SYSCHECKVAL(accept(lComm->fd, (struct sockaddr*)&sockaddr, &socklen), "accept", tmpFd);
    NCCLCHECK(socketWait(NCCL_SOCKET_RECV, tmpFd, &sendSockIdx, sizeof(int), &offset));
    if (sendSockIdx == rComm->nSocks) rComm->ctrlFd = tmpFd;
    else rComm->fds[sendSockIdx] = tmpFd;
  }
  *recvComm = rComm;
  return ncclSuccess;
}

ncclResult_t ncclSocketGetRequest(struct ncclSocketComm* comm, int op, void* data, int size, struct ncclSocketRequest** req) {
  for (int i=0; i<MAX_REQUESTS; i++) {
    struct ncclSocketRequest* r = comm->requests+i;
    if (r->used == 0) {
      r->op = op;
      r->data = data;
      r->size = size;
      r->ctrlFd = comm->ctrlFd;
      r->used = 1;
      r->comm = comm;
      r->nSubs = 0;
      *req = r;
      return ncclSuccess;
    }
  }
  WARN("NET/Socket : unable to allocate requests");
  return ncclInternalError;
}

ncclResult_t ncclSocketGetTask(struct ncclSocketComm* comm, int op, void* data, int size, struct ncclSocketTask** req) {
  int tid = comm->nextFd % comm->nThreads;
  struct ncclSocketThreadResources* res = comm->threadResources+tid;
  struct ncclSocketTaskQueue* queue = &res->threadTaskQueue;
  // create helper threads and prepare per-thread task queue
  if (queue->tasks == NULL) {
    NCCLCHECK(ncclCalloc(&queue->tasks, MAX_QUEUE_LEN));
    queue->next = 0;
    res->comm = comm;
    pthread_mutex_init(&res->threadLock, NULL);
    pthread_cond_init(&res->threadCond, NULL);
    pthread_create(comm->helperThread+tid, NULL, persistentSocketThread, res);
  }
  struct ncclSocketTask* r = queue->tasks+queue->next;
  if (r->used == 0) {
    r->op = op;
    r->data = data;
    r->size = size;
    r->fd = comm->fds[comm->nextFd];
    r->offset = 0;
    r->result = ncclSuccess;
    comm->nextFd = (comm->nextFd + 1) % comm->nSocks;
    r->used = 1;
    *req = r;
    pthread_mutex_lock(&res->threadLock);
    queue->next = (queue->next+1)%MAX_QUEUE_LEN;
    res->state = start;
    pthread_cond_signal(&res->threadCond);
    pthread_mutex_unlock(&res->threadLock);
    return ncclSuccess;
  }
  WARN("NET/Socket : unable to allocate subtasks");
  return ncclInternalError;
}

ncclResult_t ncclSocketTest(void* request, int* done, int* size) {
  *done = 0;
  struct ncclSocketRequest *r = (struct ncclSocketRequest*)request;
  if (r == NULL) {
    WARN("NET/Socket : test called with NULL request");
    return ncclInternalError;
  }
  if (r->used == 1) { /* try to send/recv size */
    int data = r->size;
    int offset = 0;
    NCCLCHECK(socketProgress(r->op, r->ctrlFd, &data, sizeof(int), &offset));

    if (offset == 0) return ncclSuccess; /* Not ready -- retry later */

    // Not sure we could ever receive less than 4 bytes, but just in case ...
    if (offset < sizeof(int)) NCCLCHECK(socketWait(r->op, r->ctrlFd, &data, sizeof(int), &offset));

    // Check size is less or equal to the size provided by the user
    if (r->op == NCCL_SOCKET_RECV && data > r->size) {
      WARN("NET/Socket : message truncated : receiving %d bytes instead of %d", data, r->size);
      return ncclInternalError;
    }
    r->size = data;
    r->offset = 0;
    r->used = 2; // done exchanging size
    // divide into subtasks
    int chunkOffset = 0, i = 0;
    if (r->comm->nSocks > 0) {
      int taskSize = std::max(MIN_CHUNKSIZE, DIVUP(r->size, r->comm->nSocks));
      while (chunkOffset < r->size) {
        int chunkSize = std::min(taskSize, r->size-chunkOffset);
        NCCLCHECK(ncclSocketGetTask(r->comm, r->op, (char*)(r->data)+chunkOffset, chunkSize, r->tasks+i++));
        chunkOffset += chunkSize;
      }
    }
    r->nSubs = i;
  }
  if (r->used == 2) { // already exchanged size
    if (r->nSubs > 0) {
      int nCompleted = 0;
      for (int i=0; i<r->nSubs; i++) {
        struct ncclSocketTask* sub = r->tasks[i];
        if (sub->result != ncclSuccess) return sub->result;
        if (sub->offset == sub->size) nCompleted++;
      }
      if (nCompleted == r->nSubs) {
        if (size) *size = r->size;
        *done = 1;
        r->used = 0;
        for (int i=0; i<r->nSubs; i++) {
          struct ncclSocketTask* sub = r->tasks[i];
          sub->used = 0;
        }
      }
    } else { // progress request using main thread
      if (r->offset < r->size) {
        NCCLCHECK(socketProgress(r->op, r->ctrlFd, r->data, r->size, &r->offset));
      }
      if (r->offset == r->size) {
        if (size) *size = r->size;
        *done = 1;
        r->used = 0;
      }
    }
  }
  return ncclSuccess;
}

ncclResult_t ncclSocketRegMr(void* comm, void* data, int size, int type, void** mhandle) {
  return (type != NCCL_PTR_HOST) ? ncclInternalError : ncclSuccess;
}
ncclResult_t ncclSocketDeregMr(void* comm, void* mhandle) { return ncclSuccess; }

ncclResult_t ncclSocketIsend(void* sendComm, void* data, int size, void* mhandle, void** request) {
  struct ncclSocketComm* comm = (struct ncclSocketComm*)sendComm;
  NCCLCHECK(ncclSocketGetRequest(comm, NCCL_SOCKET_SEND, data, size, (struct ncclSocketRequest**)request));
  return ncclSuccess;
}

ncclResult_t ncclSocketIrecv(void* recvComm, void* data, int size, void* mhandle, void** request) {
  struct ncclSocketComm* comm = (struct ncclSocketComm*)recvComm;
  NCCLCHECK(ncclSocketGetRequest(comm, NCCL_SOCKET_RECV, data, size, (struct ncclSocketRequest**)request));
  return ncclSuccess;
}

ncclResult_t ncclSocketFlush(void* recvComm, void* data, int size, void* mhandle) {
  // We don't support CUDA pointers, so we don't need a flush operation
  return ncclInternalError;
}

ncclResult_t ncclSocketCloseListen(void* opaqueComm) {
  struct ncclSocketListenComm* comm = (struct ncclSocketListenComm*)opaqueComm;
  if (comm) {
    if (comm->fd != -1) close(comm->fd);
    free(comm);
  }
  return ncclSuccess;
}

ncclResult_t ncclSocketClose(void* opaqueComm) {
  struct ncclSocketComm* comm = (struct ncclSocketComm*)opaqueComm;
  if (comm) {
    for (int i=0; i<comm->nThreads; i++) {
      struct ncclSocketThreadResources* res = comm->threadResources+i;
      if (comm->helperThread[i]) {
        pthread_mutex_lock(&res->threadLock);
        res->state = stop;
        pthread_cond_signal(&res->threadCond);
        pthread_mutex_unlock(&res->threadLock);
        pthread_join(comm->helperThread[i], NULL);
      }
      free(res->threadTaskQueue.tasks);
    }
    if (comm->ctrlFd != -1) close(comm->ctrlFd);
    for (int i=0; i<comm->nSocks; i++) {
      if (comm->fds[i] != -1) close(comm->fds[i]);
    }
    free(comm);
  }
  return ncclSuccess;
}

ncclNet_t ncclNetSocket = {
  "Socket",
  ncclSocketInit,
  ncclSocketDevices,
  ncclSocketGetProperties,
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
  ncclSocketCloseListen
};
