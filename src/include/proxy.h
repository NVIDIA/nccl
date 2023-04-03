/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_PROXY_H_
#define NCCL_PROXY_H_

#include "devcomm.h"
#include "info.h"
#include "socket.h"
#include "ipcsocket.h"
#include <pthread.h>
#include "shm.h"
#include "p2p.h"

enum ncclProxyOpState { ncclProxyOpNone, ncclProxyOpReady, ncclProxyOpProgress };

struct ncclProxyArgs;
typedef ncclResult_t (*proxyProgressFunc_t)(struct ncclProxyState*, struct ncclProxyArgs*);

#define NCCL_PROXY_MAX_SUBS MAXCHANNELS
static_assert(NCCL_MAX_WORK_ELEMENTS <= MAXCHANNELS, "Not enough sub space for max work elements");

struct ncclProxyOp {
  struct ncclProxyConnection* connection;
  int channelId;
  int nsteps;
  ssize_t nbytes;
  int root;
  int next;

  uint64_t opCount;
  int sliceSteps;
  int chunkSteps;
  int chunkSize;
  uint8_t /*ncclDataType_t*/ dtype;
  uint8_t /*ncclDevRedOp_t*/ redOp;
  uint8_t /*ncclPattern_t*/ pattern;
  uint8_t protocol;

  union {
    uint64_t unused;
    // For use by enqueue.cc
    struct ncclProxyOp *enqNext;
  };
};
static_assert(sizeof(struct ncclProxyOp) == 64, "Keep ProxyOp aligned with cache lines for effective prefetch");

struct ncclProxySubArgs {
  struct ncclProxyConnection* connection;
  int channelId;
  int nsteps;
  ssize_t nbytes;
  int peer;

  int groupSize; // Number of consecutive sub operations sharing the same recvComm
  uint64_t base;
  uint64_t posted;
  uint64_t received;
  uint64_t flushed;
  uint64_t transmitted;
  uint64_t done;
  uint64_t end;
  void* requests[NCCL_STEPS];
  void* profilingEvents[NCCL_STEPS];
};

struct ncclProxyArgs {
  struct ncclProxySubArgs subs[NCCL_PROXY_MAX_SUBS];
  proxyProgressFunc_t progress;
  int nsubs;
  int done;
  uint64_t opCount;
  int sliceSteps;
  int chunkSteps;
  int chunkSize;
  uint8_t /*ncclDataType_t*/ dtype;
  uint8_t /*ncclDevRedOp_t*/ redOp;
  uint8_t /*ncclPattern_t*/ pattern;
  uint8_t protocol;
  int state;
  char* sharedBuff[NCCL_STEPS];
  int sharedSize[NCCL_STEPS];

  int idle;

  // Element linking
  struct ncclProxyArgs* next;
  struct ncclProxyArgs* nextPeer;
  struct ncclProxyArgs** proxyAppendPtr;
};
#define NCCL_MAX_NETDEVS 128

// ProxyOps are used to communicate between main thread and service thread
// Make sure we have enough to store two full rounds of operations on all channels.
// Otherwise we'd be unable to post half of them to free new elements.
#define MAX_OPS_PER_PEER (2*MAXCHANNELS*NCCL_MAX_WORK_ELEMENTS_P2P)
#define NCCL_MAX_LOCAL_RANKS 64
struct ncclProxyOpsPool {
  struct ncclProxyOp ops[MAX_OPS_PER_PEER*NCCL_MAX_LOCAL_RANKS];
  volatile int nextOps;
  volatile int nextOpsEnd;
  volatile int freeOps[NCCL_MAX_LOCAL_RANKS];
  pthread_mutex_t mutex;
  pthread_cond_t cond;
};

struct ncclProxyOps {
  ncclProxyOpsPool* pool;
  ncclShmHandle_t handle;
  int count;
  int freeOp;
  int nextOps;
  int nextOpsEnd;
};

struct ncclProxySharedP2p {
  int refcount;
  int size;
  char* cudaBuff;
  char* hostBuff;
  // CUDA IPC
  ncclIpcDesc ipcDesc;
  struct ncclProxyArgs* proxyAppend[MAXCHANNELS]; // Separate send and recv
};

struct ncclProxyPeer {
  struct ncclProxySharedP2p send;
  struct ncclProxySharedP2p recv;
};

struct ncclSharedNetComms {
  void* sendComm[MAXCHANNELS];
  void* recvComm[MAXCHANNELS];
  int sendRefCount[MAXCHANNELS];
  int recvRefCount[MAXCHANNELS];
};

struct ncclProxyPool;
struct ncclProxyProgressState {
  // Used by main threads to send work to progress thread
  struct ncclProxyOpsPool* opsPool;
  ncclShmHandle_t handle;
  char opsPoolShmSuffix[6];

  pthread_t thread;
  bool stop;
  struct ncclProxyPeer** localPeers;
  struct ncclSharedNetComms* netComms[NCCL_MAX_NETDEVS];
  struct ncclProxyArgs* active;
  struct ncclProxyArgs* pool;
  struct ncclProxyPool* pools;
  int nextOps;
};

// Expected proxy response fifo
struct ncclExpectedProxyResponse {
  void*    opId;
  int      respSize;
  bool     done;
  void*    respBuff;
  struct   ncclExpectedProxyResponse* next;
};

struct ncclProxyAsyncOp {
  int type;
  struct ncclProxyConnection* connection;
  int reqSize, respSize;
  char *reqBuff, *respBuff;
  void* opId;
  ncclProxyAsyncOp* next;
};

struct ncclProxyLocalPeer {
  struct ncclSocket sock;
  int tpRank;
  int tpLocalRank;
  ncclProxyAsyncOp* asyncOps;
  int asyncOpCounter;
};

struct ncclProxyState {
  int refCount;
  int tpRank;
  int tpnRanks;
  int tpLocalnRanks;
  int cudaDev;
  int p2pnChannels;
  int p2pChunkSize;
  int nChannels;
  int buffSizes[NCCL_NUM_PROTOCOLS];
  bool allocP2pNetLLBuffers;
  bool dmaBufSupport;
  ncclNet_t* ncclNet;
  ncclCollNet_t* ncclCollNet;
  volatile uint32_t* abortFlag;
  // Service thread
  pthread_t thread;
  struct ncclSocket* listenSock;
  int stop;
  CUcontext cudaCtx;

  // Used by main thread
  union ncclSocketAddress* peerAddresses;
  struct ncclSocket* peerSocks;
  struct ncclProxyOps* proxyOps;
  void** sharedDevMems;
  struct ncclIpcSocket peerIpcSock; // cuMEM API support (UDS)

  // Progress thread
  struct ncclProxyProgressState progressState;

  // Queue of expected responses from the proxy
  struct ncclExpectedProxyResponse* expectedResponses;
};

enum proxyConnectState {
  connUninitialized     = 0,
  connInitialized       = 1,
  connSharedInitialized = 2,
  connSetupDone         = 3,
  connConnected         = 4,
  numConnStates         = 5
};

struct ncclProxyConnection {
  int send, transport, shared;
  int tpLocalRank, sameProcess;
  struct ncclSocket* sock;
  struct ncclTransportComm* tcomm;
  struct ncclProxyArgs *proxyAppend;
  struct ncclProxyArgs **proxyAppendPtr;
  void* transportResources;
  proxyConnectState state;
  struct ncclCollNetSharedRes* collNet;
};

typedef ncclResult_t (*threadFunc_t)(struct ncclProxyArgs*);

enum proxyMode {
  proxyRing = 0,
  proxyFrom = 1,
  proxyTo = 2
};

ncclResult_t ncclProxySaveOp(struct ncclComm* comm, struct ncclProxyOp* proxyOp, bool *justInquire);
ncclResult_t ncclProxyComputeP2p(struct ncclInfo* info, struct ncclProxyOp* proxyOp);
ncclResult_t ncclProxyStart(struct ncclComm* comm);
ncclResult_t ncclProxyInit(struct ncclComm* comm, struct ncclSocket* sock, union ncclSocketAddress* peerAddresses);
ncclResult_t ncclProxyCreate(struct ncclComm* comm);
ncclResult_t ncclProxyConnect(struct ncclComm* comm, int transport, int send, int proxyRank, struct ncclProxyConnector* proxyConn);
enum ncclProxyMsgType {
  ncclProxyMsgInit = 1,
  ncclProxyMsgSharedInit = 2,
  ncclProxyMsgSetup = 3,
  ncclProxyMsgConnect = 4,
  ncclProxyMsgStart = 5,
  ncclProxyMsgClose = 6,
  ncclProxyMsgAbort = 7,
  ncclProxyMsgStop = 8,
  ncclProxyMsgConvertFd = 9, // cuMem API support (UDS)
};

// This function is called by a client of the proxy that needs to invoke any of the non-progress proxyOp types
// Call this function on the client, supplying a locally unique opId. Then, poll on the return value of
// ncclPollProxyResponse(), supplying the same opId to confirm the operation has completed
ncclResult_t ncclProxyCallAsync(struct ncclComm* comm, struct ncclProxyConnector* proxyConn, int type, void* reqBuff, int reqSize, int respSize, void* opId);

// This function will internally call ncclProxyCallAsync() and spin until ncclPollProxyResponse() confirms the result is received
ncclResult_t ncclProxyCallBlocking(struct ncclComm* comm, struct ncclProxyConnector* proxyConn, int type, void* reqBuff, int reqSize, void* respBuff, int respSize);
ncclResult_t ncclPollProxyResponse(struct ncclComm* comm, struct ncclProxyConnector* proxyConn, void* respBuff, void* opId);

ncclResult_t ncclProxyClientConvertFdBlocking(struct ncclComm* comm, struct ncclProxyConnector* proxyConn, int fd, int* convertedFd);

ncclResult_t ncclProxyStop(struct ncclComm* comm);
ncclResult_t ncclProxyShmUnlink(struct ncclComm* comm);
ncclResult_t ncclProxyDestroy(struct ncclComm* comm);
#endif
