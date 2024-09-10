/*************************************************************************
 * Copyright (c) 2017-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_DEBUG_H_
#define NCCL_DEBUG_H_

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
  NCCL_ALL = ~0
} ncclDebugLogSubSys;

typedef void (*ncclDebugLogger_t)(ncclDebugLogLevel level, unsigned long flags, const char *file, int line, const char *fmt, ...);

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
  ncclNumFuncs = 8
} ncclFunc_t;

#define NCCL_NUM_ALGORITHMS 7 // Tree/Ring/CollNet*
#define NCCL_ALGO_UNDEF -1
#define NCCL_ALGO_TREE 0
#define NCCL_ALGO_RING 1
#define NCCL_ALGO_COLLNET_DIRECT 2
#define NCCL_ALGO_COLLNET_CHAIN 3
#define NCCL_ALGO_NVLS 4
#define NCCL_ALGO_NVLS_TREE 5
#define NCCL_ALGO_PAT 6

#define NCCL_NUM_PROTOCOLS 3 // Simple/LL/LL128
#define NCCL_PROTO_UNDEF -1
#define NCCL_PROTO_LL 0
#define NCCL_PROTO_LL128 1
#define NCCL_PROTO_SIMPLE 2

#define NCCL_ALGO_PROTO_IGNORE -1.0
#endif
