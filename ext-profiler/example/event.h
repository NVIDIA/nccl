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
#include <cstring>
#include "err.h"
#include "profiler.h"
#include "queue.h"
#include <cuda_runtime.h>

// CE timing modes
typedef enum {
  CE_TIMING_CPU = 0,
  CE_TIMING_GPU = 1
} CeTimingMode_t;

#define MAX_CHANNELS                     32
#define MAX_STEPS                        32
#define MAX_OPS                          16 // Up to 64K ranks for PAT
#define MAX_EVENTS_PER_REQ               (8)

struct proxyOp;
struct proxyStep;

struct netPlugin {
  uint64_t type;
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
  uint64_t type;                     // type of event: network transfer
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
  uint64_t type;                     // type of event: proxy operation
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
  uint64_t type;
  struct context* ctx;              // profiler context
  double startTs;
  double stopTs;
  int state;
  int appended;                     // appended proxy operations
};

// task level event base structure
struct taskEventBase {
  uint64_t type;                     // event type: collective/p2p
  int rank;                         // rank of the operation in NCCL communicator
  const char* func;                 // ncclFunc*
  int refCount;                     // number of references for this operation
  void* parent;                     // parent API event
  struct taskEventBase* next;       // next top level event
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
  uint64_t type;
  struct context* ctx;              // profiler context
  int groupId;
  int refCount;
  struct taskEventBase* eventHead;  // queue head for task events
  struct taskEventBase* eventTail;  // queue tail for task events
  double startTs;
  double stopTs;
  struct group* next;               // next group event in queue
};

struct collApi {
  uint64_t type;
  struct groupApi* parent;
  struct context* ctx;              // profiler context
  int collApiId;
  int refCount;
  cudaStream_t stream;
  const char* func;
  size_t count;
  const char* datatype;
  int root;
  bool graphCaptured;
  struct taskEventBase* eventHead;  // queue head for task events
  struct taskEventBase* eventTail;  // queue tail for task events
  double startTs;
  double stopTs;
  struct collApi* next;
};

struct p2pApi {
  uint64_t type;
  struct groupApi* parent;
  struct context* ctx;              // profiler context
  int p2pApiId;
  int refCount;
  const char* func;
  cudaStream_t stream;
  size_t count;
  const char* datatype;
  bool graphCaptured;
  struct taskEventBase* eventHead;  // queue head for task events
  struct taskEventBase* eventTail;  // queue tail for task events
  double startTs;
  double stopTs;
  struct p2pApi* next;
};

struct kernelLaunch {
  uint64_t type;
  struct groupApi* parent;
  cudaStream_t stream;
  int kernelLaunchId;
  double startTs;
  double stopTs;
  struct kernelLaunch* next;
};

// CE event structures
struct ceColl {
  struct taskEventBase base;  // Must be first for task event queue (uses base.next)
  struct collApi* parent;
  int ceCollId;
  uint64_t seqNumber;
  size_t count;
  const char* datatype;
  int root;
  const char* syncStrategy;
  cudaStream_t stream;
  uint64_t eventId;
  int timingMode;
  // Timing fields:
  // - cpuStartTime/cpuStopTime: Captured using CLOCK_MONOTONIC (via gettime()), units: microseconds (double)
  // - cpuDuration: Always CPU-measured time difference (cpuStopTime - cpuStartTime), units: microseconds (double)
  // - elapsedTime: Final reported timing, units: microseconds (uint64_t)
  //   * If timingMode==CE_TIMING_GPU: GPU-measured time from cudaEventElapsedTime (converted from ms to us)
  //   * If timingMode==CE_TIMING_CPU: Same as cpuDuration (CPU-measured)
  double cpuStartTime;
  double cpuStopTime;
  double cpuDuration;
  uint64_t elapsedTime;
  // Child events (CeSync and CeBatch)
  struct taskEventBase* eventHead;
  struct taskEventBase* eventTail;
  // Plugin-managed CUDA events for timing
  cudaEvent_t startEvent;
  cudaEvent_t stopEvent;
  bool startCompleted;
  bool stopCompleted;
  struct ceColl* pollerNext;  // For poller tracking list (separate from base.next)
};

struct ceSync {
  struct taskEventBase base;  // For parent CeColl's event queue
  struct ceColl* parent;
  int ceSyncId;
  bool isComplete;
  uint32_t seqNumber;
  int nRanks;
  cudaStream_t stream;
  uint64_t eventId;
  int timingMode;
  // Timing fields: See ceColl struct for detailed clock/unit documentation
  // - cpuStartTime/cpuStopTime: CLOCK_MONOTONIC, microseconds (double)
  // - cpuDuration: CPU-measured (cpuStopTime - cpuStartTime), microseconds (double)
  // - elapsedTime: GPU or CPU-measured depending on timingMode, microseconds (uint64_t)
  double cpuStartTime;
  double cpuStopTime;
  double cpuDuration;
  uint64_t elapsedTime;
  // Plugin-managed CUDA events for timing
  cudaEvent_t startEvent;
  cudaEvent_t stopEvent;
  bool startCompleted;
  bool stopCompleted;
  struct ceSync* pollerNext;  // For poller tracking list
};

struct ceBatch {
  struct taskEventBase base;  // For parent CeColl's event queue
  struct ceColl* parent;
  int ceBatchId;
  int numOps;
  size_t totalBytes;
  bool useIntraSync;
  cudaStream_t stream;
  uint64_t eventId;
  int timingMode;
  // Timing fields: See ceColl struct for detailed clock/unit documentation
  // - cpuStartTime/cpuStopTime: CLOCK_MONOTONIC, microseconds (double)
  // - cpuDuration: CPU-measured (cpuStopTime - cpuStartTime), microseconds (double)
  // - elapsedTime: GPU or CPU-measured depending on timingMode, microseconds (uint64_t)
  double cpuStartTime;
  double cpuStopTime;
  double cpuDuration;
  uint64_t elapsedTime;
  // Plugin-managed CUDA events for timing
  cudaEvent_t startEvent;
  cudaEvent_t stopEvent;
  bool startCompleted;
  bool stopCompleted;
  struct ceBatch* pollerNext;  // For poller tracking list
};

struct groupApi {
  uint64_t type;
  struct context* ctx;
  int groupApiId;
  int refCount;
  bool graphCaptured;
  int groupDepth;
  struct profilerQueue<struct p2pApi, &p2pApi::next> p2pApiEvents;
  struct profilerQueue<struct collApi, &collApi::next> collApiEvents;
  struct profilerQueue<struct kernelLaunch, &kernelLaunch::next> kernelLaunchEvents;
  double endOfncclGroupStartTs;
  double startOfncclGroupEndTs;
  double startTs;
  double stopTs;
  struct groupApi* next;
};

// CE event poller tracking
struct ceEventList {
  struct ceColl* ceCollHead;
  struct ceSync* ceSyncHead;
  struct ceBatch* ceBatchHead;
  pthread_mutex_t mutex;
};

// arrays for different event objects
struct context {
  const char* commName;
  uint64_t commHash;
  int nranks;
  int rank;

  // CE event tracking for poller
  struct ceEventList ceEvents;

  int groupApiPoolSize;
  int groupApiPoolBase;
  int groupApiPoolIndex;
  struct groupApi* groupApiPool;

  int collApiPoolSize;
  int collApiPoolBase;
  int collApiPoolIndex;
  struct collApi* collApiPool;

  int p2pApiPoolSize;
  int p2pApiPoolBase;
  int p2pApiPoolIndex;
  struct p2pApi* p2pApiPool;

  int kernelLaunchPoolSize;
  int kernelLaunchPoolBase;
  int kernelLaunchPoolIndex;
  struct kernelLaunch* kernelLaunchPool;

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

  // CE event pools
  int ceCollPoolSize;
  int ceCollPoolBase;
  int ceCollPoolIndex;
  struct ceColl* ceCollPool;

  int ceSyncPoolSize;
  int ceSyncPoolBase;
  int ceSyncPoolIndex;
  struct ceSync* ceSyncPool;

  int ceBatchPoolSize;
  int ceBatchPoolBase;
  int ceBatchPoolIndex;
  struct ceBatch* ceBatchPool;
};

template <typename T>
inline int taskEventQueueEmpty(T *obj) {
  return obj->eventHead == NULL;
}

template <typename T>
inline void taskEventQueueEnqueue(T* obj, struct taskEventBase* event) {
  event->next = NULL;
  if (obj->eventHead) obj->eventTail->next = event;
  else obj->eventHead = event;
  obj->eventTail = event;
}

template <typename T>
inline struct taskEventBase* taskEventQueueHead(T *obj) {
    return obj->eventHead;
}

template <typename T>
inline struct taskEventBase* taskEventQueueDequeue(T* obj) {
  struct taskEventBase* tmp = obj->eventHead;
  obj->eventHead = obj->eventHead->next;
  if (obj->eventHead == NULL) obj->eventTail = NULL;
  return tmp;
}

template <typename T>
inline void resetTaskEvents(T *obj, struct context* ctx) {
  while (!taskEventQueueEmpty(obj)) {
    struct taskEventBase* base = taskEventQueueDequeue(obj);
    if (base->type == ncclProfileColl) {
      struct collective* c = (struct collective *)base;
      // reset event proxyOps & proxySteps
      memset(c->nProxyOps, 0, sizeof(int)*MAX_CHANNELS);
      // release collective events in the group and return them to the collective pool
      __atomic_fetch_add(&ctx->collPoolBase, 1, __ATOMIC_RELAXED);
    } else if (base->type == ncclProfileP2p) {
      struct p2p* p = (struct p2p *)base;
      // reset event proxyOp and proxySteps
      memset(&p->op, 0, sizeof(struct proxyOp)*MAX_CHANNELS);
      // release p2p events in the group and return them to the p2p pool
      __atomic_fetch_add(&ctx->p2pPoolBase, 1, __ATOMIC_RELAXED);
    }
  }
}

#endif
