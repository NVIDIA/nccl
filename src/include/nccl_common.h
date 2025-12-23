/*************************************************************************
 * Copyright (c) 2017-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_DEBUG_H_
#define NCCL_DEBUG_H_

// Workaround for libstdc++ trying to force public visibility of std:: symbols.  We don't want to do that in libnccl.so.
#include <bits/c++config.h>
#undef _GLIBCXX_VISIBILITY
#define _GLIBCXX_VISIBILITY(V)

#include <cstdint>

typedef enum {
  NCCL_LOG_NONE = 0,
  NCCL_LOG_VERSION = 1,
  NCCL_LOG_WARN = 2,
  NCCL_LOG_INFO = 3,
  NCCL_LOG_ABORT = 4,
  NCCL_LOG_TRACE = 5
} ncclDebugLogLevel;

typedef enum {
  NCCL_INIT = 0x1,
  NCCL_COLL = 0x2,
  NCCL_P2P = 0x4,
  NCCL_SHM = 0x8,
  NCCL_NET = 0x10,
  NCCL_GRAPH = 0x20,
  NCCL_TUNING = 0x40,
  NCCL_ENV = 0x80,
  NCCL_ALLOC = 0x100,
  NCCL_CALL = 0x200,
  NCCL_PROXY = 0x400,
  NCCL_NVLS = 0x800,
  NCCL_BOOTSTRAP = 0x1000,
  NCCL_REG = 0x2000,
  NCCL_PROFILE = 0x4000,
  NCCL_RAS = 0x8000,
  NCCL_ALL = ~0
} ncclDebugLogSubSys;

typedef void (*ncclDebugLogger_t)(ncclDebugLogLevel level, unsigned long flags, const char *file, int line, const char *fmt, ...);

// NCCL core profiler callback for network defined events instrumentation
enum {
  ncclProfilerNetEventStart = 0,
  ncclProfilerNetEventStop,
  ncclProfilerNetEventUpdate,
  ncclProfilerNetEventUpdateAndStop,
};

typedef ncclResult_t (*ncclProfilerCallback_t)(void** eHandle, int type, void* pHandle, int64_t pluginId, void* extData);

#define NCCL_NUM_FUNCTIONS 5 // Send/Recv not included for now
typedef enum {
  ncclFuncBroadcast = 0,
  ncclFuncReduce = 1,
  ncclFuncAllGather = 2,
  ncclFuncReduceScatter = 3,
  ncclFuncAllReduce = 4,
  ncclFuncSendRecv = 5,
  ncclFuncSend = 6,
  ncclFuncRecv = 7,
  ncclFuncAlltoAll = 8,
  ncclFuncScatter = 9,
  ncclFuncGather = 10,
  ncclFuncAllGatherV = 11,
  ncclFuncPutSignal = 12,
  ncclFuncSignal = 13,
  ncclFuncWaitSignal = 14,
  ncclNumFuncs = 15
} ncclFunc_t;


#endif
