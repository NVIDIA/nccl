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

#define PROXY_OP_SEND_STATE_OFFSET       (ncclProfilerProxyOpSendPosted)
#define PROXY_OP_RECV_STATE_OFFSET       (ncclProfilerProxyOpRecvPosted)
#define PROXY_STEP_SEND_STATE_OFFSET     (ncclProfilerProxyStepSendGPUWait)
#define PROXY_STEP_RECV_STATE_OFFSET     (ncclProfilerProxyStepRecvWait)

#define NUM_PROXY_OP_SEND_STATES         (ncclProfilerProxyOpSendDone      - ncclProfilerProxyOpSendPosted    + 1)
#define NUM_PROXY_OP_RECV_STATES         (ncclProfilerProxyOpRecvDone      - ncclProfilerProxyOpRecvPosted    + 1)
#define NUM_PROXY_STEP_SEND_STATES       (ncclProfilerProxyStepSendWait    - ncclProfilerProxyStepSendGPUWait + 1)
#define NUM_PROXY_STEP_RECV_STATES       (ncclProfilerProxyStepRecvGPUWait - ncclProfilerProxyStepRecvWait    + 1)

#define PROXY_OP_SEND_STATE_IDX(state)   (state - PROXY_OP_SEND_STATE_OFFSET)
#define PROXY_OP_RECV_STATE_IDX(state)   (state - PROXY_OP_RECV_STATE_OFFSET)
#define PROXY_STEP_SEND_STATE_IDX(state) (state - PROXY_STEP_SEND_STATE_OFFSET)
#define PROXY_STEP_RECV_STATE_IDX(state) (state - PROXY_STEP_RECV_STATE_OFFSET)

#define MAX_PROXY_OP_STATES              ((NUM_PROXY_OP_SEND_STATES   > NUM_PROXY_OP_RECV_STATES  ) ? NUM_PROXY_OP_SEND_STATES   : NUM_PROXY_OP_RECV_STATES)
#define MAX_PROXY_STEP_STATES            ((NUM_PROXY_STEP_SEND_STATES > NUM_PROXY_STEP_RECV_STATES) ? NUM_PROXY_STEP_SEND_STATES : NUM_PROXY_STEP_RECV_STATES)

#define MAX_COMM_CLIQUES                 (32 * 8)

struct proxyOp;

struct proxyStep {
  uint8_t type;                     // type of event: network transfer
  int step;                         // network transfer id in given channel
  int isSend;                       // send/recv channel operation
  double timestamp[MAX_PROXY_STEP_STATES];
  double startTs;
  double stopTs;
  struct proxyOp* parent;
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
  struct {
    int steps;                      // completed steps for this proxy operation state
    double timestamp;
  } states[MAX_PROXY_OP_STATES];
  double startTs;
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
  const char* name;                 // FIXME: unused
  uint64_t commHash;                // communicator identifier
  uint8_t func;                     // ncclFunc*
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
  size_t trafficBytes;
  int root;
  uint8_t datatype;
  uint8_t nMaxChannels;
  uint8_t algo;
  uint8_t proto;
  int op;
  int nWarps;
  int isCollnet;
  int isNvls;
  struct proxyOp send[MAX_CHANNELS];// array of send proxy operation events
  struct proxyOp recv[MAX_CHANNELS];// array of recv proxy operation events
};

struct p2p {
  struct taskEventBase base;        // base structure for this event
  uint8_t func;
  void const* buff;
  size_t count;
  uint8_t datatype;
  int peer;
  struct proxyOp op;
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
