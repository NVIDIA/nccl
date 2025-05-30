/*************************************************************************
 * Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef EVENT_H_
#define EVENT_H_

#include <sys/types.h>
#include <stdint.h>
#include <unistd.h>
#include "profiler.h"

#define MAX_CHANNELS                     32
#define MAX_STEPS                        16
#define MAX_OPS                          16 // Up to 64K ranks for PAT
#define MAX_EVENTS_PER_REQ               (8)

struct proxyOp;
struct proxyStep;

struct netPlugin {
  uint8_t type;
  int pluginType;
  int pluginVer;
  uint8_t pluginEvent;
  union {
    struct {
      int device;
      int qpNum;
      int opcode;
      uint64_t wr_id;
      size_t length;
    } qp;
    struct {
      int fd;
      int op;
      size_t length;
    } sock;
  };
  double startTs;
  double stopTs;
  struct proxyStep* parent;
};

struct kernelCh {
  uint8_t type;
  uint8_t channelId;
  struct taskEventBase* parent;
  double startTs;
  double stopTs;
  uint64_t startGpuClk;
  uint64_t stopGpuClk;
};

#define PROXY_STEP_SEND_GPU_WAIT 0
#define PROXY_STEP_SEND_PEER_WAIT 1
#define PROXY_STEP_SEND_WAIT 2
#define PROXY_STEP_RECV_WAIT 0
#define PROXY_STEP_RECV_FLUSH_WAIT 1
#define PROXY_STEP_RECV_GPU_WAIT 2
#define PROXY_STEP_MAX_STATES 3

struct proxyStep {
  uint8_t type;                     // type of event: network transfer
  int state;
  int step;                         // network transfer id in given channel
  int isSend;                       // send/recv channel operation
  double timestamp[PROXY_STEP_MAX_STATES];
  double startTs;
  double stopTs;
  struct proxyOp* parent;
  struct netPlugin net[MAX_EVENTS_PER_REQ];
  int nNetEvents;
};

struct proxyOp {
  uint8_t type;                     // type of event: proxy operation
  uint8_t channelId;                // channel id for this proxy operation
  pid_t pid;
  int rank;
  int peer;                         // peer rank for this proxy operation
  int nSteps;                       // total number of network transfers for this proxy operation
  int chunkSize;                    // chunk size for this proxy operation
  int isSend;                       // send/recv channel operation
  size_t transSize;                 // transfer data size for this proxy operation
  double startTs;
  double progrTs;                   // In progress state transition
  double stopTs;
  int stepCount;                    // last processed network operation for this proxy operation
  struct proxyStep step[MAX_STEPS]; // array of network transfer events
  struct taskEventBase* parent;     // parent event p2p/collective
};

struct group;
struct context;

struct proxyCtrl {
  uint8_t type;
  struct context* ctx;              // profiler context
  double startTs;
  double stopTs;
  int state;
  int appended;                     // appended proxy operations
};

// task level event base structure
struct taskEventBase {
  uint8_t type;                     // event type: collective/p2p
  int rank;                         // rank of the operation in NCCL communicator
  const char* func;                 // ncclFunc*
  int refCount;                     // number of references for this operation
  struct group* parent;             // parent event group
  struct taskEventBase* next;       // next top level event in group
  double startTs;
  double stopTs;
};

struct collective {
  struct taskEventBase base;        // base structure for this event
  uint64_t seqNumber;               // sequence number for this collective in communicator
  void const* sendBuff;
  void* recvBuff;
  size_t count;
  int root;
  const char* datatype;
  uint8_t nChannels;
  const char* algo;
  const char* proto;
  int nWarps;
  struct proxyOp op[MAX_CHANNELS][2*MAX_OPS];
  int nProxyOps[MAX_CHANNELS];
  struct kernelCh kernel[MAX_CHANNELS];
};

struct p2p {
  struct taskEventBase base;        // base structure for this event
  uint8_t func;
  void const* buff;
  size_t count;
  const char* datatype;
  int peer;
  uint8_t nChannels;
  struct proxyOp op[MAX_CHANNELS];
  struct kernelCh kernel[MAX_CHANNELS];
};

struct group {
  uint8_t type;
  struct context* ctx;              // profiler context
  int groupId;
  int refCount;
  struct taskEventBase* eventHead;  // queue head for task events
  struct taskEventBase* eventTail;  // queue tail for task events
  double startTs;
  double stopTs;
  struct group* next;               // next group event in queue
};

// arrays for different event objects
struct context {
  const char* commName;
  uint64_t commHash;
  int nranks;
  int rank;

  int groupPoolSize;
  int groupPoolBase;
  int groupPoolIndex;
  struct group* groupPool;

  int collPoolSize;
  int collPoolBase;
  int collPoolIndex;
  struct collective* collPool;

  int p2pPoolSize;
  int p2pPoolBase;
  int p2pPoolIndex;
  struct p2p* p2pPool;

  int proxyCtrlPoolSize;
  int proxyCtrlPoolBase;
  int proxyCtrlPoolIndex;
  struct proxyCtrl* proxyCtrlPool;
};

int taskEventQueueEmpty(struct group* g);
void taskEventQueueEnqueue(struct group* g, struct taskEventBase* event);
struct taskEventBase* taskEventQueueHead(struct group* g);
struct taskEventBase* taskEventQueueDequeue(struct group* g);

#endif
