/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_PROXY_H_
#define NCCL_PROXY_H_

#include "device.h"
#include "info.h"
#include "socket.h"
#include "ipcsocket.h"
#include "nccl_net.h"
#include <pthread.h>
#include "shmutils.h"
#include "p2p.h"

typedef enum : uint8_t {
  ncclPatternRing,
  ncclPatternRingTwice,
  ncclPatternPipelineFrom,
  ncclPatternPipelineTo,
  ncclPatternTreeUp,
  ncclPatternTreeDown,
  ncclPatternTreeUpDown,
  ncclPatternCollnetChain,
  ncclPatternCollnetDirect,
  ncclPatternNvls,
  ncclPatternNvlsTree,
  ncclPatternPatUp,
  ncclPatternPatDown,
  ncclPatternSend,
  ncclPatternRecv
} ncclPattern_t;

enum ncclProxyOpState { ncclProxyOpNone, ncclProxyOpReady, ncclProxyOpProgress };

struct ncclProxyArgs;
typedef ncclResult_t (*proxyProgressFunc_t)(struct ncclProxyState*, struct ncclProxyArgs*);

#define NCCL_PROXY_MAX_SUBS MAXCHANNELS
static_assert(2*NCCL_MAX_DEV_WORK_P2P_PER_BATCH <= MAXCHANNELS, "Not enough sub space for max work elements");

union ncclProxyOpSpecifics {
  struct {
    size_t sizePerRank;
    int nNodes, node;
  } collnetDirect;
};

struct ncclProxyOp {
  struct ncclProxyConnection* connection;
  ssize_t nbytes;
  uint64_t opCount;
  int root;
  int next;
  int nsteps;
  int chunkSize;
  uint8_t sliceSteps;
  uint8_t chunkSteps;
  uint8_t channelId;
  uint8_t /*ncclDataType_t*/ dtype;
  uint8_t /*ncclDevRedOp_t*/ redOp;
  uint8_t /*ncclFunc_t*/ coll;
  uint8_t /*ncclPattern_t*/ pattern;
  uint8_t protocol;
  uint8_t reg;
  // collnet buffer reg handles
  void* sendMhandle;
  void* recvMhandle;
  uint8_t* sendbuff;
  uint8_t* recvbuff;

  union ncclProxyOpSpecifics specifics;

  // Profiler plugin
  union {
    struct ncclTaskColl* coll;
    struct ncclTaskP2p* p2p;
  } task;

  int eActivationMask;
  void* taskEventHandle;
  int rank;
  int peer;
  pid_t pid;
  void* profilerContext;

  struct ncclProxyOp *enqNext;
};

struct ncclProxySubArgs {
  struct ncclProxyConnection* connection;
  int reg;
  // p2p mhandle
  void* mhandle;
  // collnet handles
  void* sendMhandle;
  void* recvMhandle;
  uint8_t* sendbuff;
  uint8_t* recvbuff;
  size_t offset;
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

  // Profiler plugin
  int eActivationMask;
  int rank;
  void* taskEventHandle;
  void* opEventHandle;
  void* stepEventHandles[NCCL_STEPS];
  size_t transSize;

  void* recvRequestsCache[NCCL_STEPS];
  int recvRequestsSubCount;
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
  size_t totalSendSize;
  size_t totalRecvSize;
  size_t sendSizePerRound;
  size_t recvSizePerRound;
  uint8_t /*ncclDataType_t*/ dtype;
  uint8_t /*ncclDevRedOp_t*/ redOp;
  uint8_t /*ncclPattern_t*/ pattern;
  uint8_t /*ncclFunc_t*/ coll;
  uint8_t protocol;
  int state;
  char* sharedBuff[NCCL_STEPS];
  int sharedSize[NCCL_STEPS];

  int idle;

  // Profiler plugin
  pid_t pid;
  void* profilerContext;

  // Element linking
  struct ncclProxyArgs* next;
  struct ncclProxyArgs* nextPeer;
  struct ncclProxyArgs** proxyAppendPtr;

  union ncclProxyOpSpecifics specifics;
};
#define NCCL_MAX_NETDEVS 128

// ProxyOps are used to communicate between main thread and service thread
// Make sure we have enough to store two full rounds of operations on all channels.
// Otherwise we'd be unable to post half of them to free new elements. Each
// p2p work contains a send and recv proxy op hence the 2x before it.
#define MAX_OPS_PER_PEER (2*MAXCHANNELS*2*NCCL_MAX_DEV_WORK_P2P_PER_BATCH)

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
  volatile int stop;
  struct ncclProxyPeer** localPeers;
  struct ncclSharedNetComms* netComms[NCCL_MAX_NETDEVS];
  struct ncclProxyArgs* active;
  struct ncclProxyArgs* pool;
  struct ncclProxyPool* pools;
  int nextOps;
};

// Expected proxy response fifo
struct ncclExpectedProxyResponse {
  void*                             opId;
  int                               respSize;
  bool                              done;
  void*                             respBuff;
  ncclResult_t                      res;
  struct ncclExpectedProxyResponse* next;
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

// Common response header for all proxyOps
// We pack this into a struct to reduce the number of blocking send and recv calls
struct ncclProxyRpcResponseHeader {
  void* opId;
  ncclResult_t res;
  int respSize;
};

// UDS support
struct ncclIpcHdr {
  int type;
  int rank;
  int reqSize;
  int respSize;
  void *opId;
  uint64_t data[16]; // 128-bytes
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
  uint32_t* abortFlag;
  bool directMode;
  // Service threads
  pthread_t thread;
  pthread_t threadUDS;
  struct ncclSocket* listenSock;
  struct ncclIpcSocket ipcSock;
  int stop;
  CUcontext cudaCtx;
  ncclResult_t asyncResult;

  // Used by main thread
  union ncclSocketAddress* peerAddresses;
  struct ncclSocket* peerSocks;
  struct ncclProxyOps* proxyOps;
  void** sharedDevMems;
  struct ncclIpcSocket peerIpcSock; // cuMEM API support (UDS)
  uint64_t *peerAddressesUDS; // cuMem API support (UDS)

  // Progress thread
  struct ncclProxyProgressState progressState;

  // Profiler plugin
  void* profilerContext;

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
  ncclNetDeviceHandle_t* netDeviceHandle;
  void* mhandles[NCCL_NUM_PROTOCOLS];
  proxyConnectState state;
  struct ncclCollNetSharedRes* collNet;
  int needsProxyProgress;
};

typedef ncclResult_t (*threadFunc_t)(struct ncclProxyArgs*);

enum proxyMode {
  proxyRing = 0,
  proxyFrom = 1,
  proxyTo = 2
};

ncclResult_t ncclProxySaveOp(struct ncclComm* comm, struct ncclProxyOp* proxyOp, bool *justInquire);
ncclResult_t ncclProxyStart(struct ncclComm* comm);
ncclResult_t ncclProxyInit(struct ncclComm* comm, struct ncclSocket* sock, union ncclSocketAddress* peerAddresses, uint64_t *peerAddressesUDS);
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
  ncclProxyMsgGetFd = 9, // cuMem API support (UDS)
  ncclProxyMsgQueryFd = 10,
  ncclProxyMsgRegister = 11,
  ncclProxyMsgDeregister = 12
};

// This function is called by a client of the proxy that needs to invoke any of the non-progress proxyOp types
// Call this function on the client, supplying a locally unique opId. Then, poll on the return value of
// ncclPollProxyResponse(), supplying the same opId to confirm the operation has completed
ncclResult_t ncclProxyCallAsync(struct ncclComm* comm, struct ncclProxyConnector* proxyConn, int type, void* reqBuff, int reqSize, int respSize, void* opId);

// This function will internally call ncclProxyCallAsync() and spin until ncclPollProxyResponse() confirms the result is received
ncclResult_t ncclProxyCallBlocking(struct ncclComm* comm, struct ncclProxyConnector* proxyConn, int type, void* reqBuff, int reqSize, void* respBuff, int respSize);
ncclResult_t ncclPollProxyResponse(struct ncclComm* comm, struct ncclProxyConnector* proxyConn, void* respBuff, void* opId);

// UDS support
ncclResult_t ncclProxyClientGetFdBlocking(struct ncclComm* comm, int rank, void *handle, int* convertedFd);
ncclResult_t ncclProxyClientQueryFdBlocking(struct ncclComm* comm, struct ncclProxyConnector* proxyConn, int localFd, int* rmtFd);

ncclResult_t ncclProxyStop(struct ncclComm* comm);
ncclResult_t ncclProxyShmUnlink(struct ncclComm* comm);
ncclResult_t ncclProxyDestroy(struct ncclComm* comm);
#endif
