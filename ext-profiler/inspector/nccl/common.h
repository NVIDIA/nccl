/*************************************************************************
 * Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef COMMON_H_
#define COMMON_H_

/* typedef enum {NCCL_LOG_NONE=0, NCCL_LOG_VERSION=1, NCCL_LOG_WARN=2, NCCL_LOG_INFO=3, NCCL_LOG_ABORT=4, NCCL_LOG_TRACE=5} ncclDebugLogLevel; */
/* typedef enum {NCCL_INIT=1, NCCL_COLL=2, NCCL_P2P=4, NCCL_SHM=8, NCCL_NET=16, NCCL_GRAPH=32, NCCL_TUNING=64, NCCL_ENV=128, NCCL_ALLOC=256, NCCL_CALL=512, NCCL_PROXY=1024, NCCL_NVLS=2048, NCCL_BOOTSTRAP=4096, NCCL_REG=8192, NCCL_ALL=~0} ncclDebugLogSubSys; */

/* Data types */
typedef enum { ncclInt8       = 0, ncclChar       = 0,
               ncclUint8      = 1,
               ncclInt32      = 2, ncclInt        = 2,
               ncclUint32     = 3,
               ncclInt64      = 4,
               ncclUint64     = 5,
               ncclFloat16    = 6, ncclHalf       = 6,
               ncclFloat32    = 7, ncclFloat      = 7,
               ncclFloat64    = 8, ncclDouble     = 8,
               ncclBfloat16   = 9,
               ncclFloat8e4m3 = 10,
               ncclFloat8e5m2 = 11,
               ncclNumTypes   = 12
} ncclDataType_t;

typedef enum {
  NCCL_LOG_NONE = 0,
  NCCL_LOG_VERSION = 1,
  NCCL_LOG_WARN = 2,
  NCCL_LOG_INFO = 3,
  NCCL_LOG_ABORT = 4,
  NCCL_LOG_TRACE = 5
} ncclDebugLogLevel;

typedef enum { ncclSuccess                 =  0,
               ncclUnhandledCudaError      =  1,
               ncclSystemError             =  2,
               ncclInternalError           =  3,
               ncclInvalidArgument         =  4,
               ncclInvalidUsage            =  5,
               ncclRemoteError             =  6,
               ncclInProgress              =  7,
               ncclNumResults              =  8 } ncclResult_t;


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

#endif
