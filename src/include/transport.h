/*************************************************************************
 * Copyright (c) 2016-2018, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_TRANSPORT_H_
#define NCCL_TRANSPORT_H_

#include "nccl.h"
#include <stdint.h>

#define NTRANSPORTS 3

extern struct ncclTransport ncclTransports[];

// Forward declarations
struct ncclRing;
struct ncclConnector;
struct ncclComm;

#define RANK_INFO_SIZE 64
typedef char ncclTinfo_t[RANK_INFO_SIZE];

struct ncclInfo {
  ncclTinfo_t tinfo[NTRANSPORTS];
};

// Used to hold the transport connection values
typedef int64_t ncclTvalue_t;

#define CONNECT_SIZE 128
struct ncclConnect {
  char data[CONNECT_SIZE];
};

struct ncclProxyArgs {
  struct ncclRing* ring;
  int substeps;
  int nsteps;
  uint64_t opCount;
  int llMode;
  bool needProxy;
  int active;   // add component before this line -- it is left out during initialization
};

struct ncclTransportComm {
  ncclResult_t (*setup)(ncclTinfo_t*, ncclTinfo_t*, struct ncclConnect*, struct ncclRing*);
  ncclResult_t (*connect)(struct ncclConnect*, struct ncclConnector*);
  ncclResult_t (*free)(void*);
  ncclResult_t (*proxy)(struct ncclProxyArgs*);
};

struct ncclTransport {
  const char name[4];
  ncclResult_t (*fillInfo)(ncclTinfo_t*, int);
  ncclResult_t (*canConnect)(ncclTvalue_t*, ncclTinfo_t*, ncclTinfo_t*);
  ncclResult_t (*getRings)(int, int*, int*, ncclTvalue_t*, int*, int*, int*, int, int*);
  struct ncclTransportComm send;
  struct ncclTransportComm recv;
};

#include <pthread.h>

typedef ncclResult_t (*threadFunc_t)(struct ncclProxyArgs*);

#define TRANSPORT_PROXY_FIFO_SIZE NCCL_MAX_OPS

struct transportProxyInfo {
  struct ncclComm* comm;
  pthread_t thread;
  threadFunc_t func;
  volatile int proxyReady;
  struct ncclProxyArgs argsFifo[TRANSPORT_PROXY_FIFO_SIZE];
  volatile uint64_t argsFifoHead;
  volatile uint64_t argsFifoTail;
  pthread_cond_t cond;
  pthread_mutex_t mutex;
};

ncclResult_t transportCreateProxy(int type, struct ncclRing* ring, struct ncclComm* comm);
ncclResult_t transportDestroyProxy(struct ncclConnector* connector);

enum proxyMode {
  proxyRing = 0,
  proxyFrom = 1,
  proxyTo = 2
};

static int proxyPatternRing = proxyRing;
static inline int proxyPatternFrom(int root) { return 1+root; }
static inline int proxyPatternTo(int root) { return -1-root; }
static inline enum proxyMode proxyPatternMode(int pattern) { return (pattern == 0) ? proxyRing : ((pattern > 0) ? proxyFrom : proxyTo); }
static inline int proxyPatternRoot(int pattern) { return (pattern > 0) ? pattern-1 : -pattern-1; }

ncclResult_t transportSaveProxies(int substeps, int subchunks, int nstepsPerRound, int nblocksPerRound, size_t size, int pattern, struct ncclComm* comm);
ncclResult_t transportStartProxies(struct ncclComm* comm);

#include <unistd.h>

// Spin wait until func evaluates to true
template<typename FUNC>
inline void transportProxyWait(const FUNC& func) {
  while (!func()) {
    sched_yield();
  }
}

inline void transportProxyIdle(int idle) {
  sched_yield();
}

#endif
