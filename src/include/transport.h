/*************************************************************************
 * Copyright (c) 2016-2019, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_TRANSPORT_H_
#define NCCL_TRANSPORT_H_

#include "nccl.h"
#include "devcomm.h"
#include <stdint.h>
#include "nvmlwrap.h"

#define NTRANSPORTS 3

extern struct ncclTransport ncclTransports[];

// Forward declarations
struct ncclRing;
struct ncclConnector;
struct ncclComm;

struct ncclPeerInfo {
  int rank;
  int cudaDev;
  int nvmlDev;
  uint64_t hostHash;
  uint64_t pidHash;
  char busId[NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE];
};

// Used to hold the transport connection values
typedef int64_t ncclTvalue_t;

#define CONNECT_SIZE 128
struct ncclConnect {
  char data[CONNECT_SIZE];
};

enum ncclProxyOpState { ncclProxyOpNone, ncclProxyOpReady, ncclProxyOpProgress };

struct ncclProxyArgs;
typedef ncclResult_t (*proxyProgressFunc_t)(struct ncclProxyArgs*);

struct ncclProxyArgs {
  proxyProgressFunc_t progress;
  struct ncclChannel* channel;
  struct ncclConnector* connector;
  int sliceSteps;
  int chunkSteps;
  int nsteps;
  uint64_t opCount;
  int llMode;
  int state;   // add component before this line -- it is left out during initialization

  // Internal state
  uint64_t head;
  uint64_t tail;
  uint64_t end;
  void* requests[NCCL_STEPS];
  int idle;

  // Element linking
  pthread_mutex_t mutex;
  struct ncclProxyArgs* next;
  struct ncclProxyArgs* nextPeer;
};

struct ncclProxyPool;
struct ncclProxyState {
  pthread_cond_t cond;
  pthread_mutex_t mutex;
  bool stop;
  struct ncclProxyArgs* ops;
  struct ncclProxyArgs* pool;
  struct ncclProxyPool* pools;
};

struct ncclTransportComm {
  ncclResult_t (*setup)(struct ncclPeerInfo*, struct ncclPeerInfo*, struct ncclConnect*, struct ncclConnector*, int buffSize, int channelId);
  ncclResult_t (*connect)(struct ncclConnect*, struct ncclConnector*);
  ncclResult_t (*free)(void*);
  ncclResult_t (*proxy)(struct ncclProxyArgs*);
};

struct ncclTransport {
  const char name[4];
  ncclResult_t (*canConnect)(ncclTvalue_t*, struct ncclPeerInfo*, struct ncclPeerInfo*);
  ncclResult_t (*getRings)(int, int*, int*, ncclTvalue_t*, int*, int*, int*, int, int*);
  struct ncclTransportComm send;
  struct ncclTransportComm recv;
};

#include <pthread.h>

typedef ncclResult_t (*threadFunc_t)(struct ncclProxyArgs*);

enum proxyMode {
  proxyRing = 0,
  proxyFrom = 1,
  proxyTo = 2
};

ncclResult_t transportAllocateProxyArgs(struct ncclComm* comm, struct ncclProxyArgs** argsptr);
ncclResult_t transportSaveProxies(struct ncclProxyArgs* args, int pattern, int root, int nranks);
ncclResult_t transportStartProxy(struct ncclComm* comm);
ncclResult_t transportCreateProxy(struct ncclComm* comm);
ncclResult_t transportDestroyProxy(struct ncclComm* comm);

#include <unistd.h>

// Spin wait until func evaluates to true
template<typename FUNC>
inline void transportProxyWait(const FUNC& func) {
  while (!func()) {
    sched_yield();
  }
}

#endif
