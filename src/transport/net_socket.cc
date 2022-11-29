/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "comm.h"
#include "core.h"
#include "socket.h"
#include "net.h"
#include "param.h"

#include <pthread.h>
#include <stdlib.h>
#include <poll.h>
#include <limits.h>
#include <fcntl.h>

/* Init functions */
static int ncclNetIfs = -1;
struct ncclNetSocketDev {
  union ncclSocketAddress addr;
  char devName[MAX_IF_NAME_SIZE];
  char* pciPath;
};
static struct ncclNetSocketDev ncclNetSocketDevs[MAX_IFS];

pthread_mutex_t ncclNetSocketLock = PTHREAD_MUTEX_INITIALIZER;

static ncclResult_t ncclNetSocketGetPciPath(char* devName, char** pciPath) {
  char devicePath[PATH_MAX];
  snprintf(devicePath, PATH_MAX, "/sys/class/net/%s/device", devName);
  // May return NULL if the file doesn't exist.
  *pciPath = realpath(devicePath, NULL);
  return ncclSuccess;
}

ncclResult_t ncclNetSocketInit(ncclDebugLogger_t logFunction) {
  if (ncclNetIfs == -1) {
    pthread_mutex_lock(&ncclNetSocketLock);
    if (ncclNetIfs == -1) {
      char names[MAX_IF_NAME_SIZE*MAX_IFS];
      union ncclSocketAddress addrs[MAX_IFS];
      ncclNetIfs = ncclFindInterfaces(names, addrs, MAX_IF_NAME_SIZE, MAX_IFS);
      if (ncclNetIfs <= 0) {
        WARN("NET/Socket : no interface found");
        return ncclInternalError;
      } else {
        #define MAX_LINE_LEN (2047)
        char line[MAX_LINE_LEN+1];
        char addrline[SOCKET_NAME_MAXLEN+1];
        line[0] = '\0';
        addrline[SOCKET_NAME_MAXLEN] = '\0';
        for (int i=0; i<ncclNetIfs; i++) {
          strcpy(ncclNetSocketDevs[i].devName, names+i*MAX_IF_NAME_SIZE);
          memcpy(&ncclNetSocketDevs[i].addr, addrs+i, sizeof(union ncclSocketAddress));
          NCCLCHECK(ncclNetSocketGetPciPath(ncclNetSocketDevs[i].devName, &ncclNetSocketDevs[i].pciPath));
          snprintf(line+strlen(line), MAX_LINE_LEN-strlen(line), " [%d]%s:%s", i, names+i*MAX_IF_NAME_SIZE,
              ncclSocketToString(&addrs[i], addrline));
        }
        line[MAX_LINE_LEN] = '\0';
        INFO(NCCL_INIT|NCCL_NET,"NET/Socket : Using%s", line);
      }
    }
    pthread_mutex_unlock(&ncclNetSocketLock);
  }
  return ncclSuccess;
}

ncclResult_t ncclNetSocketDevices(int* ndev) {
  *ndev = ncclNetIfs;
  return ncclSuccess;
}

static ncclResult_t ncclNetSocketGetSpeed(char* devName, int* speed) {
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

ncclResult_t ncclNetSocketGetProperties(int dev, ncclNetProperties_t* props) {
  props->name = ncclNetSocketDevs[dev].devName;
  props->pciPath = ncclNetSocketDevs[dev].pciPath;
  props->guid = dev;
  props->ptrSupport = NCCL_PTR_HOST;
  NCCLCHECK(ncclNetSocketGetSpeed(props->name, &props->speed));
  props->latency = 0; // Not set
  props->port = 0;
  props->maxComms = 65536;
  props->maxRecvs = 1;
  return ncclSuccess;
}

/* Communication functions */

#define MAX_SOCKETS 64
#define MAX_THREADS 16
#define MAX_REQUESTS NCCL_NET_MAX_REQUESTS
#define MIN_CHUNKSIZE (64*1024)

NCCL_PARAM(SocketNsocksPerThread, "NSOCKS_PERTHREAD", -2);
NCCL_PARAM(SocketNthreads, "SOCKET_NTHREADS", -2);

enum ncclNetSocketCommState {
  ncclNetSocketCommStateStart = 0,
  ncclNetSocketCommStateConnect = 1,
  ncclNetSocketCommStateAccept = 3,
  ncclNetSocketCommStateSend = 4,
  ncclNetSocketCommStateRecv = 5,
};

struct ncclNetSocketCommStage {
  enum ncclNetSocketCommState state;
  uint8_t iteration;
  struct ncclSocket* sock;
  struct ncclNetSocketComm* comm;
};

struct ncclNetSocketHandle {
  union ncclSocketAddress connectAddr;
  uint64_t magic; // random number to help debugging
  int nSocks;
  int nThreads;
  struct ncclNetSocketCommStage stage;
};

struct ncclNetSocketTask {
  int op;
  void* data;
  int size;
  struct ncclSocket* sock;
  int offset;
  int used;
  ncclResult_t result;
};

struct ncclNetSocketRequest {
  int op;
  void* data;
  int size;
  struct ncclSocket* ctrlSock;
  int offset;
  int used;
  struct ncclNetSocketComm* comm;
  struct ncclNetSocketTask* tasks[MAX_SOCKETS];
  int nSubs;
};

struct ncclNetSocketTaskQueue {
  int next;
  int len;
  struct ncclNetSocketTask* tasks;
};

struct ncclNetSocketThreadResources {
  struct ncclNetSocketTaskQueue threadTaskQueue;
  int stop;
  struct ncclNetSocketComm* comm;
  pthread_mutex_t threadLock;
  pthread_cond_t  threadCond;
};

struct ncclNetSocketListenComm {
  struct ncclSocket sock;
  struct ncclNetSocketCommStage stage;
  int nSocks;
  int nThreads;
  int dev;
};

struct ncclNetSocketComm {
  struct ncclSocket ctrlSock;
  struct ncclSocket socks[MAX_SOCKETS];
  int dev;
  int cudaDev;
  int nSocks;
  int nThreads;
  int nextSock;
  struct ncclNetSocketRequest requests[MAX_REQUESTS];
  pthread_t helperThread[MAX_THREADS];
  struct ncclNetSocketThreadResources threadResources[MAX_THREADS];
};

void* persistentSocketThread(void *args_) {
  struct ncclNetSocketThreadResources* resource = (struct ncclNetSocketThreadResources*)args_;
  struct ncclNetSocketComm* comm = resource->comm;
  struct ncclNetSocketTaskQueue* myQueue = &resource->threadTaskQueue;
  int nSocksPerThread = comm->nSocks / comm->nThreads;
  while (1) {
    int idle = 1;
    int mark = myQueue->next; // mark newest task seen
    for (int i=0; i<myQueue->len; i+=nSocksPerThread) {
      int repeat;
      do {
        repeat = 0;
        for (int j=0; j<nSocksPerThread; j++) {
          struct ncclNetSocketTask* r = myQueue->tasks+i+j;
          if (r != NULL && r->used == 1 && r->offset < r->size) {
            r->result = ncclSocketProgress(r->op, r->sock, r->data, r->size, &r->offset);
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
      while (mark == myQueue->next && resource->stop == 0) { // no new tasks, wait
        pthread_cond_wait(&resource->threadCond, &resource->threadLock);
      }
      pthread_mutex_unlock(&resource->threadLock);
    }
    if (resource->stop) return NULL;
  }
}

ncclResult_t ncclNetSocketGetNsockNthread(int dev, int* ns, int* nt) {
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
    snprintf(vendorPath, PATH_MAX, "/sys/class/net/%s/device/vendor", ncclNetSocketDevs[dev].devName);
    char* rPath = realpath(vendorPath, NULL);
    int fd = open(rPath, O_RDONLY);
    free(rPath);
    if (fd == -1) {
      // Could not find device vendor. This is handled silently so
      // we don't want to print an INFO error.
      TRACE(NCCL_NET, "Open of %s failed : %s", vendorPath, strerror(errno));
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

ncclResult_t ncclNetSocketListen(int dev, void* opaqueHandle, void** listenComm) {
  if (dev < 0 || dev >= ncclNetIfs) { // data transfer socket is based on specified dev
    return ncclInternalError;
  }
  struct ncclNetSocketHandle* handle = (struct ncclNetSocketHandle*) opaqueHandle;
  memset(handle, 0, sizeof(struct ncclNetSocketHandle));
  static_assert(sizeof(struct ncclNetSocketHandle) <= NCCL_NET_HANDLE_MAXSIZE, "ncclNetSocketHandle size too large");
  struct ncclNetSocketListenComm* comm;
  NCCLCHECK(ncclCalloc(&comm, 1));
  handle->magic = NCCL_SOCKET_MAGIC;
  NCCLCHECK(ncclSocketInit(&comm->sock, &ncclNetSocketDevs[dev].addr, handle->magic, ncclSocketTypeNetSocket, NULL, 1));
  NCCLCHECK(ncclSocketListen(&comm->sock));
  NCCLCHECK(ncclSocketGetAddr(&comm->sock, &handle->connectAddr));
  NCCLCHECK(ncclNetSocketGetNsockNthread(dev, &comm->nSocks, &comm->nThreads));
  handle->nSocks = comm->nSocks;
  handle->nThreads = comm->nThreads;
  comm->dev = dev;
  *listenComm = comm;
  return ncclSuccess;
}

ncclResult_t ncclNetSocketConnect(int dev, void* opaqueHandle, void** sendComm) {
  if (dev < 0 || dev >= ncclNetIfs) { // data transfer socket is based on specified dev
    return ncclInternalError;
  }

  int ready;
  struct ncclNetSocketHandle* handle = (struct ncclNetSocketHandle*) opaqueHandle;
  struct ncclNetSocketCommStage* stage = &handle->stage;
  struct ncclNetSocketComm* comm = stage->comm;
  uint8_t i = stage->iteration;
  struct ncclSocket* sock = stage->sock;
  *sendComm = NULL;

  if (stage->state == ncclNetSocketCommStateConnect) goto socket_connect_check;
  if (stage->state == ncclNetSocketCommStateSend) goto socket_send;

  NCCLCHECK(ncclCalloc(&comm, 1));
  stage->comm = comm;
  comm->nSocks = handle->nSocks;
  comm->nThreads = handle->nThreads;
  comm->dev = dev;
  CUDACHECK(cudaGetDevice(&comm->cudaDev));
  for (; i<comm->nSocks+1; i++) {
    sock = (i == comm->nSocks) ? &comm->ctrlSock : comm->socks+i;
    NCCLCHECK(ncclSocketInit(sock, &handle->connectAddr, handle->magic, ncclSocketTypeNetSocket, NULL, 1));

    stage->sock = sock;
    stage->state = ncclNetSocketCommStateConnect;
    stage->iteration = i;
    NCCLCHECK(ncclSocketConnect(sock));

socket_connect_check:
    NCCLCHECK(ncclSocketReady(sock, &ready));
    if (! ready) return ncclSuccess;
    stage->state = ncclNetSocketCommStateSend;

socket_send:
    int done = 0;
    NCCLCHECK(ncclSocketProgress(NCCL_SOCKET_SEND, sock, &i, sizeof(uint8_t), &done));
    if (done == 0) return ncclSuccess;
  }
  *sendComm = comm;
  return ncclSuccess;
}

ncclResult_t ncclNetSocketAccept(void* listenComm, void** recvComm) {
  struct ncclNetSocketListenComm* lComm = (struct ncclNetSocketListenComm*)listenComm;
  struct ncclNetSocketCommStage* stage = &lComm->stage;
  struct ncclNetSocketComm* rComm = stage->comm;
  uint8_t i = stage->iteration;
  struct ncclSocket* sock = stage->sock;
  int ready;

  *recvComm = NULL;
  if (stage->state == ncclNetSocketCommStateAccept) goto socket_accept_check;
  if (stage->state == ncclNetSocketCommStateRecv) goto socket_recv;

  NCCLCHECK(ncclCalloc(&rComm, 1));
  stage->comm = rComm;
  rComm->nSocks = lComm->nSocks;
  rComm->nThreads = lComm->nThreads;
  rComm->dev = lComm->dev;
  CUDACHECK(cudaGetDevice(&rComm->cudaDev));
  for (; i<rComm->nSocks+1; i++) {
    uint8_t sendSockIdx;

    NCCLCHECK(ncclCalloc(&sock, 1));
    NCCLCHECK(ncclSocketInit(sock));
    stage->sock = sock;
    stage->state = ncclNetSocketCommStateAccept;
    stage->iteration = i;
    NCCLCHECK(ncclSocketAccept(sock, &lComm->sock));

socket_accept_check:
    NCCLCHECK(ncclSocketReady(sock, &ready));
    if (!ready) return ncclSuccess;

    stage->state = ncclNetSocketCommStateRecv;
socket_recv:
    int done = 0;
    NCCLCHECK(ncclSocketProgress(NCCL_SOCKET_RECV, sock, &sendSockIdx, sizeof(uint8_t), &done));
    if (done == 0) return ncclSuccess;

    if (sendSockIdx == rComm->nSocks)
      memcpy(&rComm->ctrlSock, sock, sizeof(struct ncclSocket));
    else
      memcpy(rComm->socks+sendSockIdx, sock, sizeof(struct ncclSocket));
    free(sock);
  }
  *recvComm = rComm;

  /* reset lComm state */
  stage->state = ncclNetSocketCommStateStart;
  stage->iteration = 0;
  stage->sock = NULL;
  stage->comm = NULL;
  return ncclSuccess;
}

ncclResult_t ncclNetSocketGetRequest(struct ncclNetSocketComm* comm, int op, void* data, int size, struct ncclNetSocketRequest** req) {
  for (int i=0; i<MAX_REQUESTS; i++) {
    struct ncclNetSocketRequest* r = comm->requests+i;
    if (r->used == 0) {
      r->op = op;
      r->data = data;
      r->size = size;
      r->ctrlSock = &comm->ctrlSock;
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

ncclResult_t ncclNetSocketGetTask(struct ncclNetSocketComm* comm, int op, void* data, int size, struct ncclNetSocketTask** req) {
  int tid = comm->nextSock % comm->nThreads;
  struct ncclNetSocketThreadResources* res = comm->threadResources+tid;
  struct ncclNetSocketTaskQueue* queue = &res->threadTaskQueue;
  // create helper threads and prepare per-thread task queue
  if (queue->tasks == NULL) {
    // each request can be divided up to nSocks tasks, and
    // these tasks are distributed to nThreads threads,
    // we need to make sure each thread queue has enough slots for MAX_REQUESTS
    queue->len = MAX_REQUESTS * DIVUP(comm->nSocks, comm->nThreads);
    NCCLCHECK(ncclCalloc(&queue->tasks, queue->len));
    queue->next = 0;
    res->comm = comm;
    pthread_mutex_init(&res->threadLock, NULL);
    pthread_cond_init(&res->threadCond, NULL);
    pthread_create(comm->helperThread+tid, NULL, persistentSocketThread, res);
    ncclSetThreadName(comm->helperThread[tid], "NCCL Sock%c%1u%2u%2u", op == NCCL_SOCKET_SEND ? 'S' : 'R', comm->dev, tid, comm->cudaDev);
  }
  struct ncclNetSocketTask* r = queue->tasks+queue->next;
  if (r->used == 0) {
    r->op = op;
    r->data = data;
    r->size = size;
    r->sock = comm->socks + comm->nextSock;
    r->offset = 0;
    r->result = ncclSuccess;
    comm->nextSock = (comm->nextSock + 1) % comm->nSocks;
    r->used = 1;
    *req = r;
    pthread_mutex_lock(&res->threadLock);
    queue->next = (queue->next+1)%queue->len;
    pthread_cond_signal(&res->threadCond);
    pthread_mutex_unlock(&res->threadLock);
    return ncclSuccess;
  }
  WARN("NET/Socket : unable to allocate subtasks");
  return ncclInternalError;
}

ncclResult_t ncclNetSocketTest(void* request, int* done, int* size) {
  *done = 0;
  struct ncclNetSocketRequest *r = (struct ncclNetSocketRequest*)request;
  if (r == NULL) {
    WARN("NET/Socket : test called with NULL request");
    return ncclInternalError;
  }
  if (r->used == 1) { /* try to send/recv size */
    int data = r->size;
    int offset = 0;
    NCCLCHECK(ncclSocketProgress(r->op, r->ctrlSock, &data, sizeof(int), &offset));

    if (offset == 0) return ncclSuccess; /* Not ready -- retry later */

    // Not sure we could ever receive less than 4 bytes, but just in case ...
    if (offset < sizeof(int)) NCCLCHECK(ncclSocketWait(r->op, r->ctrlSock, &data, sizeof(int), &offset));

    // Check size is less or equal to the size provided by the user
    if (r->op == NCCL_SOCKET_RECV && data > r->size) {
      char line[SOCKET_NAME_MAXLEN+1];
      union ncclSocketAddress addr;
      ncclSocketGetAddr(r->ctrlSock, &addr);
      WARN("NET/Socket : peer %s message truncated : receiving %d bytes instead of %d. If you believe your socket network is in healthy state, \
          there may be a mismatch in collective sizes or environment settings (e.g. NCCL_PROTO, NCCL_ALGO) between ranks",
          ncclSocketToString(&addr, line), data, r->size);
      return ncclInvalidUsage;
    }
    r->size = data;
    r->offset = 0;
    r->used = 2; // done exchanging size
    // divide into subtasks
    int chunkOffset = 0, i = 0;
    if (r->comm->nSocks > 0) {
      // each request can be divided up to nSocks tasks
      int taskSize = std::max(MIN_CHUNKSIZE, DIVUP(r->size, r->comm->nSocks));
      while (chunkOffset < r->size) {
        int chunkSize = std::min(taskSize, r->size-chunkOffset);
        NCCLCHECK(ncclNetSocketGetTask(r->comm, r->op, (char*)(r->data)+chunkOffset, chunkSize, r->tasks+i++));
        chunkOffset += chunkSize;
      }
    }
    r->nSubs = i;
  }
  if (r->used == 2) { // already exchanged size
    if (r->nSubs > 0) {
      int nCompleted = 0;
      for (int i=0; i<r->nSubs; i++) {
        struct ncclNetSocketTask* sub = r->tasks[i];
        if (sub->result != ncclSuccess) return sub->result;
        if (sub->offset == sub->size) nCompleted++;
      }
      if (nCompleted == r->nSubs) {
        if (size) *size = r->size;
        *done = 1;
        r->used = 0;
        for (int i=0; i<r->nSubs; i++) {
          struct ncclNetSocketTask* sub = r->tasks[i];
          sub->used = 0;
        }
      }
    } else { // progress request using main thread
      if (r->offset < r->size) {
        NCCLCHECK(ncclSocketProgress(r->op, r->ctrlSock, r->data, r->size, &r->offset));
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

ncclResult_t ncclNetSocketRegMr(void* comm, void* data, int size, int type, void** mhandle) {
  return (type != NCCL_PTR_HOST) ? ncclInternalError : ncclSuccess;
}
ncclResult_t ncclNetSocketDeregMr(void* comm, void* mhandle) { return ncclSuccess; }

ncclResult_t ncclNetSocketIsend(void* sendComm, void* data, int size, int tag, void* mhandle, void** request) {
  struct ncclNetSocketComm* comm = (struct ncclNetSocketComm*)sendComm;
  NCCLCHECK(ncclNetSocketGetRequest(comm, NCCL_SOCKET_SEND, data, size, (struct ncclNetSocketRequest**)request));
  return ncclSuccess;
}

ncclResult_t ncclNetSocketIrecv(void* recvComm, int n, void** data, int* sizes, int* tags, void** mhandles, void** request) {
  struct ncclNetSocketComm* comm = (struct ncclNetSocketComm*)recvComm;
  if (n != 1) return ncclInternalError;
  NCCLCHECK(ncclNetSocketGetRequest(comm, NCCL_SOCKET_RECV, data[0], sizes[0], (struct ncclNetSocketRequest**)request));
  return ncclSuccess;
}

ncclResult_t ncclNetSocketIflush(void* recvComm, int n, void** data, int* sizes, void** mhandles, void** request) {
  // We don't support CUDA pointers, so we don't need a flush operation
  return ncclInternalError;
}

ncclResult_t ncclNetSocketCloseListen(void* opaqueComm) {
  struct ncclNetSocketListenComm* comm = (struct ncclNetSocketListenComm*)opaqueComm;
  if (comm) {
    int ready;
    NCCLCHECK(ncclSocketReady(&comm->sock, &ready));
    if (ready) NCCLCHECK(ncclSocketClose(&comm->sock));
    free(comm);
  }
  return ncclSuccess;
}

ncclResult_t ncclNetSocketClose(void* opaqueComm) {
  struct ncclNetSocketComm* comm = (struct ncclNetSocketComm*)opaqueComm;
  if (comm) {
    for (int i=0; i<comm->nThreads; i++) {
      struct ncclNetSocketThreadResources* res = comm->threadResources+i;
      if (comm->helperThread[i]) {
        pthread_mutex_lock(&res->threadLock);
        res->stop = 1;
        pthread_cond_signal(&res->threadCond);
        pthread_mutex_unlock(&res->threadLock);
        pthread_join(comm->helperThread[i], NULL);
      }
      free(res->threadTaskQueue.tasks);
    }
    int ready;
    NCCLCHECK(ncclSocketReady(&comm->ctrlSock, &ready));
    if (ready) NCCLCHECK(ncclSocketClose(&comm->ctrlSock));
    for (int i=0; i<comm->nSocks; i++) {
      NCCLCHECK(ncclSocketReady(&comm->socks[i], &ready));
      if (ready) NCCLCHECK(ncclSocketClose(&comm->socks[i]));
    }
    free(comm);
  }
  return ncclSuccess;
}

ncclNet_t ncclNetSocket = {
  "Socket",
  ncclNetSocketInit,
  ncclNetSocketDevices,
  ncclNetSocketGetProperties,
  ncclNetSocketListen,
  ncclNetSocketConnect,
  ncclNetSocketAccept,
  ncclNetSocketRegMr,
  NULL, // No DMA-BUF support
  ncclNetSocketDeregMr,
  ncclNetSocketIsend,
  ncclNetSocketIrecv,
  ncclNetSocketIflush,
  ncclNetSocketTest,
  ncclNetSocketClose,
  ncclNetSocketClose,
  ncclNetSocketCloseListen
};
