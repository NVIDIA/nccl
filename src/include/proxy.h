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
#include <pthread.h>

enum ncclProxyOpState { ncclProxyOpNone, ncclProxyOpReady, ncclProxyOpProgress };

struct ncclProxyArgs;
typedef ncclResult_t (*proxyProgressFunc_t)(struct ncclComm*, struct ncclProxyArgs*);

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
  ncclDataType_t dtype;
  ncclRedOp_t redOp;
  ncclPattern_t pattern; // uint8_t
  uint8_t protocol;
  uint16_t pad;
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
  ncclDataType_t dtype;
  ncclRedOp_t redOp;
  ncclPattern_t pattern;
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
  cudaIpcMemHandle_t ipc;
  struct ncclProxyArgs* proxyAppend[MAXCHANNELS]; // Separate send and recv
};

struct ncclProxySharedCollNet {
  int size;
  char* cudaBuff;
  char* hostBuff;
  struct ncclProxyArgs* proxyAppend[2*NCCL_MAX_NETDEVS];
  void* resources;
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
  char opsPoolShmSuffix[6];

  pthread_t thread;
  bool stop;
  struct ncclProxyPeer** localPeers;
  struct ncclSharedNetComms* netComms[NCCL_MAX_NETDEVS];
  struct ncclProxySharedCollNet collNet;
  struct ncclProxyArgs* active;
  struct ncclProxyArgs* pool;
  struct ncclProxyPool* pools;
  int nextOps;
};

struct ncclProxyState {
  // Service thread
  pthread_t thread;
  struct ncclSocket* listenSock;
  int stop;

  // Used by main thread
  union ncclSocketAddress* peerAddresses;
  struct ncclSocket* peerSocks;
  struct ncclProxyOps* proxyOps;
  void** sharedDevMems;

  // Progress thread
  struct ncclProxyProgressState progressState;
};

struct ncclProxyConnection {
  int send, transport, shared;
  int localRank;
  struct ncclSocket* sock;
  struct ncclTransportComm* tcomm;
  struct ncclProxyArgs *proxyAppend;
  struct ncclProxyArgs **proxyAppendPtr;
  void* transportResources;
};

typedef ncclResult_t (*threadFunc_t)(struct ncclProxyArgs*);

enum proxyMode {
  proxyRing = 0,
  proxyFrom = 1,
  proxyTo = 2
};

ncclResult_t ncclProxySaveColl(struct ncclComm* comm, struct ncclProxyOp* proxyOp, int nranks);
ncclResult_t ncclProxyComputeP2p(struct ncclInfo* info, struct ncclProxyOp* proxyOp);
ncclResult_t ncclProxySaveP2p(struct ncclComm* comm, struct ncclProxyOp* proxyOp);
ncclResult_t ncclProxyStart(struct ncclComm* comm);
ncclResult_t ncclProxyInit(struct ncclComm* comm, struct ncclSocket* sock, union ncclSocketAddress* peerAddresses);
ncclResult_t ncclProxyCreate(struct ncclComm* comm);
ncclResult_t ncclProxyConnect(struct ncclComm* comm, int transport, int send, int rank, struct ncclProxyConnector* proxyConn);
enum ncclProxyMsgType {
  ncclProxyMsgInit = 1,
  ncclProxyMsgSharedInit = 2,
  ncclProxyMsgSetup = 3,
  ncclProxyMsgConnect = 4,
  ncclProxyMsgStart = 5,
  ncclProxyMsgClose = 6,
  ncclProxyMsgAbort = 7,
  ncclProxyMsgStop = 8
};

ncclResult_t ncclProxyCall(struct ncclProxyConnector* proxyConn, int type, void* reqBuff, int reqSize, void* respBuff, int respSize);
ncclResult_t ncclProxyDestroy(struct ncclComm* comm);
ncclResult_t ncclProxyShmUnlink(struct ncclComm* comm);
#endif
