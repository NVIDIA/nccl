/*************************************************************************
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_PROFILER_H_
#define NCCL_PROFILER_H_

#include "nccl.h"
#include <stdint.h>

enum ncclProxyProfileState {
  ncclProxyProfileBegin = 0,

  ncclProxyProfileSendGPUWait = 1,
  ncclProxyProfileSendWait = 2,

  ncclProxyProfileRecvWait = 1,
  ncclProxyProfileRecvFlushWait = 2,
  ncclProxyProfileRecvGPUWait = 3,

  ncclProxyProfileEnd = 4,

  ncclProxyProfileSleep = 8,
  ncclProxyProfileWakeup = 9,

  ncclProxyProfileIdle = 16,
  ncclProxyProfileActive = 17,

  ncclProxyProfileAppend = 24,
  ncclProxyProfileAppendEnd = 25
};

typedef enum : uint8_t {
  ncclProxySend,
  ncclProxyRecv
} ncclProxyPattern_t;

typedef struct {
  uint64_t opCount;
  int peer;
  int step;
  uint16_t channel;
  ncclProxyPattern_t type; // send / recv
  uint8_t opIndex;
} ncclProxyProfileInfo_t;

typedef struct {
  // Name of the profiler
  const char* name;
  // Record an event
  ncclResult_t (*profilingRecord)(ncclProxyProfileInfo_t* info, int state, void** profileEvent);
  void (*profilingDump)();
} ncclProfiler_t;

#endif
