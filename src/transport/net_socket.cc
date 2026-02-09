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
#include "profiler/net_socket.h"
#include "os.h"

#include <pthread.h>
#include <stdlib.h>
#include <poll.h>
#include <limits.h>
#include <fcntl.h>
#include <mutex>
#include <condition_variable>

/* Init functions */
static int ncclNetIfs = -1;
struct ncclNetSocketDev {
  union ncclSocketAddress addr;
  char devName[MAX_IF_NAME_SIZE];
  char* pciPath;
};
static struct ncclNetSocketDev ncclNetSocketDevs[MAX_IFS];

static std::mutex ncclNetSocketMutex;

static ncclResult_t ncclNetSocketGetPciPath(char* devName, char** pciPath) {
  char devicePath[PATH_MAX];
  snprintf(devicePath, PATH_MAX, "/sys/class/net/%s/device", devName);
  // May return NULL if the file doesn't exist.
  *pciPath = realpath(devicePath, NULL);
  return ncclSuccess;
}

static ncclProfilerCallback_t ncclProfilerFunction;

// With ncclNet_v11_t the NCCL core initializes the network plugin per-communicator
// rather than once for all communicators. However, the internal plugin implementation
// still assumes the plugin is initialized only once across all communicators. The ref
// counter makes sure the plugin internally initializes only once. When per communicator
// context support is added to the plugin the ref counter can be removed.
static int netRefCount;

ncclResult_t ncclNetSocketInit(void** ctx, uint64_t commId, ncclNetCommConfig_t* config, ncclDebugLogger_t logFunction, ncclProfilerCallback_t profFunction) {
  if (netRefCount++) return ncclSuccess;
  ncclProfilerFunction = profFunction;
  if (ncclNetIfs == -1) {
    std::lock_guard<std::mutex> lock(ncclNetSocketMutex);
    if (ncclNetIfs == -1) {
      char names[MAX_IF_NAME_SIZE*MAX_IFS];
      union ncclSocketAddress addrs[MAX_IFS];
      NCCLCHECK(ncclFindInterfaces(names, addrs, MAX_IF_NAME_SIZE, MAX_IFS, &ncclNetIfs));
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
  }
  return ncclSuccess;
}

ncclResult_t ncclNetSocketDevices(int* ndev) {
  *ndev = ncclNetIfs;
  return ncclSuccess;
}

static ncclResult_t ncclNetSocketGetSpeed(char* devName, int* speed) {
  ncclResult_t ret = ncclSuccess;
  *speed = 0;
  char speedPath[PATH_MAX];
  snprintf(speedPath, sizeof(speedPath), "/sys/class/net/%s/speed", devName);
  int fd = -1;
  SYSCHECKSYNC(open(speedPath, O_RDONLY), "open", fd);
  if (fd != -1) {
    char speedStr[] = "        ";
    int n;
    // Allow this to silently fail
    n = read(fd, speedStr, sizeof(speedStr)-1);
    if (n > 0) {
      *speed = strtol(speedStr, NULL, 0);
    }
  }
  if (*speed <= 0) {
    INFO(NCCL_NET, "Could not get speed from %s. Defaulting to 10 Gbps.", speedPath);
    *speed = 10000;
  }
  if (fd != -1) SYSCHECK(close(fd), "close");
  return ret;
}

ncclResult_t ncclNetSocketGetProperties(int dev, ncclNetProperties_t* props) {
  props->name = ncclNetSocketDevs[dev].devName;
  props->pciPath = ncclNetSocketDevs[dev].pciPath;
  props->guid = dev;
  props->ptrSupport = NCCL_PTR_HOST;
  props->regIsGlobal = 0;
  props->forceFlush = 0;
  NCCLCHECK(ncclNetSocketGetSpeed(props->name, &props->speed));
  props->latency = 0; // Not set
  props->port = 0;
  props->maxComms = 65536;
  props->maxRecvs = 1;
  props->netDeviceType    = NCCL_NET_DEVICE_HOST;
  props->netDeviceVersion = NCCL_NET_DEVICE_INVALID_VERSION;
  props->maxP2pBytes = NCCL_MAX_NET_SIZE_BYTES;
  props->maxCollBytes = MAX_COLLNET_SIZE;
  props->maxMultiRequestSize = 1;
  return ncclSuccess;
}

/* Communication functions */

#define MAX_SOCKETS 64
#define MAX_THREADS 16
#define MAX_REQUESTS NCCL_NET_MAX_REQUESTS

NCCL_PARAM(SocketInlineSize, "SOCKET_INLINE", /*128 B=*/1 << 7);
NCCL_PARAM(SocketMinTaskSize, "SOCKET_MIN_TASKSIZE", /*64 kiB=*/1 << 16);
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

struct ncclProfilerInfo {
  void* eHandle;
  void* pHandle;
};

struct ncclNetSocketRequest {
  int op;
  void* data;
  int size;
  void* inlineData;
  struct ncclSocket* ctrlSock;
  int offset;
  int used;
  struct ncclNetSocketComm* comm;
  struct ncclNetSocketTask* tasks[MAX_SOCKETS];
  int nSubs;
  struct ncclProfilerInfo pInfo;
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
  struct ncclProfilerInfo* pInfo;
  std::mutex threadMutex;
  std::condition_variable threadCond;
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
  void* inlineData;
  struct ncclNetSocketRequest requests[MAX_REQUESTS];
  std::thread helperThread[MAX_THREADS];
  struct ncclNetSocketThreadResources threadResources[MAX_THREADS];
};

void* persistentSocketThread(void *args_) {
  struct ncclNetSocketThreadResources* resource = (struct ncclNetSocketThreadResources*)args_;
  struct ncclNetSocketComm* comm = resource->comm;
  struct ncclNetSocketTaskQueue* myQueue = &resource->threadTaskQueue;
  int nSocksPerThread = comm->nSocks / comm->nThreads;
#ifdef NCCL_ENABLE_NET_PROFILING
  void* eHandle[MAX_REQUESTS*MAX_SOCKETS] = { 0 };
#endif
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
#ifdef NCCL_ENABLE_NET_PROFILING
            if (!eHandle[i+j]) {
              ncclProfilerNetSockDescr_v1_t data;
              data.type = ncclProfileSocket;
              data.sock.fd = r->sock->socketDescriptor;
              data.sock.op = r->op;
              data.sock.length = r->size;
              ncclProfilerFunction(&eHandle[i+j], ncclProfilerNetEventStart, resource->pInfo->pHandle, NCCL_PROFILER_NET_TYPE_SOCK | 1, &data);
            }
#endif
            r->result = ncclSocketProgress(r->op, r->sock, r->data, r->size, &r->offset);
            if (r->result != ncclSuccess) {
#ifdef NCCL_ENABLE_NET_PROFILING
              ncclProfilerFunction(&eHandle[i+j], ncclProfilerNetEventStop, NULL, 0, NULL);
              eHandle[i+j] = NULL;
#endif
              WARN("NET/Socket : socket progress error");
              return NULL;
            }
            idle = 0;
            if (r->offset < r->size) repeat = 1;
#ifdef NCCL_ENABLE_NET_PROFILING
            if (repeat == 0) {
              ncclProfilerFunction(&eHandle[i+j], ncclProfilerNetEventStop, NULL, 0, NULL);
              eHandle[i+j] = NULL;
            }
#endif
          }
        }
      } while (repeat);
    }
    if (idle) {
      std::unique_lock<std::mutex> lock(resource->threadMutex);
      resource->threadCond.wait(lock, [&] { return mark != myQueue->next || resource->stop; });
    }
    if (resource->stop) return NULL;
  }
}

ncclResult_t ncclNetSocketGetNsockNthread(int dev, int* ns, int* nt) {
  ncclResult_t ret = ncclSuccess;
  int nSocksPerThread = ncclParamSocketNsocksPerThread();
  int nThreads = ncclParamSocketNthreads();
  if (nThreads > MAX_THREADS) {
    WARN("NET/Socket : NCCL_SOCKET_NTHREADS is greater than the maximum allowed, setting to %d", MAX_THREADS);
    nThreads = MAX_THREADS;
  }
  int fd = -1;
  int nSocks;
  if (nThreads == -2 || nSocksPerThread == -2) {
    // Auto-detection
    int autoNt=0, autoNs=1; // By default, we only use the main thread and do not spawn extra threads
    char vendorPath[PATH_MAX];
    snprintf(vendorPath, PATH_MAX, "/sys/class/net/%s/device/vendor", ncclNetSocketDevs[dev].devName);
    // Coverity is wrong.  NULL second argument to realpath() is OK by POSIX.1-2008.
    // coverity[alias_transfer:FALSE]
    char* rPath = realpath(vendorPath, NULL);
    fd = open(rPath, O_RDONLY);
    free(rPath);
    if (fd == -1) {
      // Could not find device vendor. This is handled silently so
      // we don't want to print an INFO error.
      TRACE(NCCL_NET, "Open of %s failed : %s", vendorPath, strerror(errno));
      goto end;
    }
    char vendor[7];
    strncpy(vendor, "0x0000", 7);
    SYSCHECKGOTO(read(fd, vendor, 6), "read", ret, fail);
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
  nSocks = nSocksPerThread * nThreads;
  if (nSocks > MAX_SOCKETS) {
    nSocksPerThread = MAX_SOCKETS/nThreads;
    WARN("NET/Socket : the total number of sockets is greater than the maximum allowed, setting NCCL_NSOCKS_PERTHREAD to %d", nSocksPerThread);
    nSocks = nSocksPerThread * nThreads;
  }
  *ns = nSocks;
  *nt = nThreads;
  if (nSocks > 0) INFO(NCCL_INIT, "NET/Socket: Using %d threads and %d sockets per thread", nThreads, nSocksPerThread);
exit:
  if (fd != -1) close(fd);
  return ret;
fail:
  goto exit;
}

ncclResult_t ncclNetSocketListen(void* ctx, int dev, void* opaqueHandle, void** listenComm) {
  if (dev < 0 || dev >= ncclNetIfs) { // data transfer socket is based on specified dev
    WARN("NET/Socket : ncclNetSocketListen dev=%d ncclNetIfs=%d", dev, ncclNetIfs);
    return ncclInternalError;
  }
  ncclResult_t ret = ncclSuccess;
  struct ncclNetSocketHandle* handle = (struct ncclNetSocketHandle*) opaqueHandle;
  memset(handle, 0, sizeof(struct ncclNetSocketHandle));
  static_assert(sizeof(struct ncclNetSocketHandle) <= NCCL_NET_HANDLE_MAXSIZE, "ncclNetSocketHandle size too large");
  struct ncclNetSocketListenComm* comm;
  NCCLCHECK(ncclCalloc(&comm, 1));
  handle->magic = NCCL_SOCKET_MAGIC;
  NCCLCHECKGOTO(ncclSocketInit(&comm->sock, &ncclNetSocketDevs[dev].addr, handle->magic, ncclSocketTypeNetSocket, NULL, 1), ret, fail);
  NCCLCHECKGOTO(ncclSocketListen(&comm->sock), ret, fail);
  NCCLCHECKGOTO(ncclSocketGetAddr(&comm->sock, &handle->connectAddr), ret, fail);
  NCCLCHECKGOTO(ncclNetSocketGetNsockNthread(dev, &comm->nSocks, &comm->nThreads), ret, fail);
  handle->nSocks = comm->nSocks;
  handle->nThreads = comm->nThreads;
  comm->dev = dev;
  *listenComm = comm;
exit:
  return ret;
fail:
  (void)ncclSocketClose(&comm->sock);
  free(comm);
  goto exit;
}

#define SOCKET_CTRL_SIZE (sizeof(int))
ncclResult_t ncclNetSocketConnect(void* ctx, int dev, void* opaqueHandle, void** sendComm, ncclNetDeviceHandle_t** /*sendDevComm*/) {
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

  comm = new ncclNetSocketComm();
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
  NCCLCHECK(ncclCalloc(&comm->inlineData, MAX_REQUESTS * (SOCKET_CTRL_SIZE + ncclParamSocketInlineSize())));
  *sendComm = comm;
  return ncclSuccess;
}

ncclResult_t ncclNetSocketAccept(void* listenComm, void** recvComm, ncclNetDeviceHandle_t** /*recvDevComm*/) {
  struct ncclNetSocketListenComm* lComm = (struct ncclNetSocketListenComm*)listenComm;
  struct ncclNetSocketCommStage* stage = &lComm->stage;
  struct ncclNetSocketComm* rComm = stage->comm;
  uint8_t i = stage->iteration;
  struct ncclSocket* sock = stage->sock;
  int ready;

  *recvComm = NULL;
  if (stage->state == ncclNetSocketCommStateAccept) goto socket_accept_check;
  if (stage->state == ncclNetSocketCommStateRecv) goto socket_recv;

  rComm = new ncclNetSocketComm();
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
      rComm->ctrlSock = *sock;
    else
      rComm->socks[sendSockIdx] = *sock;
    free(sock);
  }
  NCCLCHECK(ncclCalloc(&rComm->inlineData, MAX_REQUESTS * (SOCKET_CTRL_SIZE + ncclParamSocketInlineSize())));
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
      r->inlineData = (uint8_t*)comm->inlineData + i * (SOCKET_CTRL_SIZE + ncclParamSocketInlineSize());
      *req = r;
      return ncclSuccess;
    }
  }
  WARN("NET/Socket : unable to allocate requests");
  return ncclInternalError;
}

ncclResult_t ncclNetSocketGetTask(struct ncclNetSocketComm* comm, struct ncclProfilerInfo* pInfo, int op, void* data, int size, struct ncclNetSocketTask** req) {
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
#ifdef NCCL_ENABLE_NET_PROFILING
    res->pInfo = pInfo;
#endif
    comm->helperThread[tid] = std::thread(persistentSocketThread, res);
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
    std::lock_guard<std::mutex> lock(res->threadMutex);
    queue->next = (queue->next+1)%queue->len;
    res->threadCond.notify_one();
    return ncclSuccess;
  }
  WARN("NET/Socket : unable to allocate subtasks");
  return ncclInternalError;
}

// if the dataSize is smaller than the inline size, return the inline size; if not, return 0 to avoid the extra copy.
static int ncclNetSocketInlineSize(int dataSize) { return (dataSize <= ncclParamSocketInlineSize()) ? dataSize : 0; }

ncclResult_t ncclNetSocketTest(void* request, int* done, int* size) {
  *done = 0;
  struct ncclNetSocketRequest *r = (struct ncclNetSocketRequest*)request;
  if (r == NULL) {
    WARN("NET/Socket : test called with NULL request");
    return ncclInternalError;
  }
  if (r->used == 1) { /* try to send/recv size (+ inline data if any) */
    int msgSize;
    uint8_t* msg = (uint8_t*)r->inlineData;
    if (r->op == NCCL_SOCKET_SEND) {
      // sender side has the right data size, copy size info + inline data to the buffer
      int inlineSize = ncclNetSocketInlineSize(r->size);
      msgSize = inlineSize + SOCKET_CTRL_SIZE;
      memcpy(msg, &r->size, SOCKET_CTRL_SIZE);
      if (inlineSize > 0) memcpy(msg + SOCKET_CTRL_SIZE, r->data, inlineSize);
    } else {
      // receiver side doesn't have the right data size, wait for the sender to send it
      int sizeOffset = 0, senderSize = 0;
      while (sizeOffset < SOCKET_CTRL_SIZE) {
        NCCLCHECK(ncclSocketProgress(r->op, r->ctrlSock, msg, SOCKET_CTRL_SIZE, &sizeOffset));
        if (sizeOffset == 0) return ncclSuccess; /* not ready yet*/
      }
      memcpy(&senderSize, msg, SOCKET_CTRL_SIZE);
      if (senderSize > r->size) {
        char line[SOCKET_NAME_MAXLEN + 1];
        union ncclSocketAddress addr;
        NCCLCHECK(ncclSocketGetAddr(r->ctrlSock, &addr));
        WARN("NET/Socket : peer %s message truncated : receiving %d bytes instead of %d. If you believe your socket network is in a healthy state, "
             "there may be a mismatch in collective sizes or environment settings (e.g. NCCL_PROTO, NCCL_ALGO) between ranks",
             ncclSocketToString(&addr, line), senderSize, r->size);
        return ncclInvalidUsage;
      }
      // copy to the data buffer if we have received some inline data already
      int receivedInline = sizeOffset - SOCKET_CTRL_SIZE;
      if (receivedInline > 0) memcpy(r->data, msg + SOCKET_CTRL_SIZE, receivedInline);
      // from the actual size, extract the remaining inline size to be received and redirect the msg buffer to the user data
      r->size = senderSize;
      msgSize = ncclNetSocketInlineSize(r->size) - receivedInline;
      msg = (uint8_t*)r->data + receivedInline;
    }
    int offset = 0;
    while (offset < msgSize) {
      NCCLCHECK(ncclSocketProgress(r->op, r->ctrlSock, msg, msgSize, &offset));
      if (offset == 0) return ncclSuccess; /* not ready yet*/
    }
    // done exchanging sizes, r->size now contains the actual size
    r->used = 2;
    r->offset = ncclNetSocketInlineSize(r->size);
    int chunkOffset = r->offset, i = 0;
    if (r->comm->nSocks > 0) {
      // each request can be divided up to nSocks tasks, we use the size left to transfer
      int taskSize = std::max((int)ncclParamSocketMinTaskSize(), DIVUP(r->size - r->offset, r->comm->nSocks));
      while (chunkOffset < r->size) {
        int chunkSize = std::min(taskSize, r->size - chunkOffset);
        NCCLCHECK(ncclNetSocketGetTask(r->comm, &r->pInfo, r->op, (char*)(r->data) + chunkOffset, chunkSize, r->tasks + i++));
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
#ifdef NCCL_ENABLE_NET_PROFILING
      if (!r->pInfo.eHandle) {
        ncclProfilerNetSockDescr_v1_t data;
        data.type = ncclProfileSocket;
        data.sock.fd = r->ctrlSock->socketDescriptor;
        data.sock.op = r->op;
        data.sock.length = r->size;
        ncclProfilerFunction(&r->pInfo.eHandle, ncclProfilerNetEventStart, r->pInfo.pHandle, NCCL_PROFILER_NET_TYPE_SOCK | 1, &data);
      }
#endif
      if (r->offset < r->size) {
        NCCLCHECK(ncclSocketProgress(r->op, r->ctrlSock, r->data, r->size, &r->offset));
      }
      if (r->offset == r->size) {
        if (size) *size = r->size;
        *done = 1;
        r->used = 0;
#ifdef NCCL_ENABLE_NET_PROFILING
        ncclProfilerFunction(&r->pInfo.eHandle, ncclProfilerNetEventStop, NULL, 0, NULL);
        r->pInfo.eHandle = NULL;
#endif
      }
    }
  }
  return ncclSuccess;
}

ncclResult_t ncclNetSocketRegMr(void* comm, void* data, size_t size, int type, void** mhandle) {
  return (type != NCCL_PTR_HOST) ? ncclInternalError : ncclSuccess;
}
ncclResult_t ncclNetSocketDeregMr(void* comm, void* mhandle) { return ncclSuccess; }

ncclResult_t ncclNetSocketIsend(void* sendComm, void* data, size_t size, int tag, void* mhandle, void* phandle, void** request) {
  struct ncclNetSocketComm* comm = (struct ncclNetSocketComm*)sendComm;
  NCCLCHECK(ncclNetSocketGetRequest(comm, NCCL_SOCKET_SEND, data, (int) size, (struct ncclNetSocketRequest**)request));
#ifdef NCCL_ENABLE_NET_PROFILING
  // NCCL core profiler callback
  struct ncclNetSocketRequest* req = *(struct ncclNetSocketRequest **)request;
  req->pInfo.pHandle = phandle;
#endif
  return ncclSuccess;
}

ncclResult_t ncclNetSocketIrecv(void* recvComm, int n, void** data, size_t* sizes, int* tags, void** mhandles, void** phandles, void** request) {
  struct ncclNetSocketComm* comm = (struct ncclNetSocketComm*)recvComm;
  if (n != 1) return ncclInternalError;
  NCCLCHECK(ncclNetSocketGetRequest(comm, NCCL_SOCKET_RECV, data[0], (int)sizes[0], (struct ncclNetSocketRequest**)request));
#ifdef NCCL_ENABLE_NET_PROFILING
  // NCCL core profiler callback
  struct ncclNetSocketRequest* req = *(struct ncclNetSocketRequest **)request;
  if (phandles) req->pInfo.pHandle = phandles[0];
#endif
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
      if (comm->helperThread[i].joinable()) {
        {
          std::lock_guard<std::mutex> lock(res->threadMutex);
          res->stop = 1;
          res->threadCond.notify_one();
        }
        comm->helperThread[i].join();
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
    if(comm->inlineData) free(comm->inlineData);
    delete comm;
  }
  return ncclSuccess;
}

ncclResult_t ncclNetSocketFinalize(void* ctx) {
  netRefCount--;
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
  ncclNetSocketCloseListen,
  NULL /* getDeviceMr */,
  NULL /* irecvConsumed */,
  NULL /* mergeDevices */,
  ncclNetSocketFinalize,
  NULL /* setNetAttr */,
};
