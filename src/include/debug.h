/*************************************************************************
 * Copyright (c) 2015-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_INT_DEBUG_H_
#define NCCL_INT_DEBUG_H_

#include "nccl.h"
#include "nccl_common.h"
#include <stdio.h>
#include <thread>
#include "compiler.h"

// Conform to pthread and NVTX standard
#define NCCL_THREAD_NAMELEN 16

extern int ncclDebugLevel;
extern uint64_t ncclDebugMask;
extern FILE *ncclDebugFile;

void ncclDebugLog(ncclDebugLogLevel level, unsigned long flags, const char *filefunc, int line, const char *fmt, ...) __attribute__ ((format (printf, 5, 6)));

// Let code temporarily downgrade WARN into INFO
extern thread_local int ncclDebugNoWarn;
extern char ncclLastError[];

#define VERSION(...) ncclDebugLog(NCCL_LOG_VERSION, NCCL_ALL, __FILE__, __LINE__, __VA_ARGS__)
#define WARN(...) ncclDebugLog(NCCL_LOG_WARN, NCCL_ALL, __FILE__, __LINE__, __VA_ARGS__)

#define NOWARN(EXPR, FLAGS) \
  do { \
    int oldNoWarn = ncclDebugNoWarn; \
    ncclDebugNoWarn = FLAGS; \
    (EXPR); \
    ncclDebugNoWarn = oldNoWarn; \
  } while(0)

#define INFO(FLAGS, ...) \
    do{ \
        int level = COMPILER_ATOMIC_LOAD(&ncclDebugLevel, std::memory_order_acquire); \
        if((level >= NCCL_LOG_INFO && ((unsigned long)(FLAGS) & ncclDebugMask)) || (level < 0)) \
            ncclDebugLog(NCCL_LOG_INFO, (unsigned long)(FLAGS), __func__, __LINE__, __VA_ARGS__); \
    } while(0)

#define TRACE_CALL(...) \
    do { \
        int level = COMPILER_ATOMIC_LOAD(&ncclDebugLevel, std::memory_order_acquire); \
        if((level >= NCCL_LOG_TRACE && (NCCL_CALL & ncclDebugMask)) || (level < 0)) { \
            ncclDebugLog(NCCL_LOG_TRACE, NCCL_CALL, __func__, __LINE__, __VA_ARGS__); \
        } \
    } while (0)

#ifdef ENABLE_TRACE
#define TRACE(FLAGS, ...) \
    do { \
        int level = COMPILER_ATOMIC_LOAD(&ncclDebugLevel, std::memory_order_acquire); \
        if ((level >= NCCL_LOG_TRACE && ((unsigned long)(FLAGS) & ncclDebugMask)) || (level < 0)) { \
            ncclDebugLog(NCCL_LOG_TRACE, (unsigned long)(FLAGS), __func__, __LINE__, __VA_ARGS__); \
        } \
    } while (0)
#else
#define TRACE(...)
#endif

void ncclSetThreadName(std::thread& thread, const char *fmt, ...);

void ncclResetDebugInit();

#endif
