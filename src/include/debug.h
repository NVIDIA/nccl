/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2015-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_INT_DEBUG_H_
#define NCCL_INT_DEBUG_H_

#include "nccl.h"
#include "nccl_common.h"
#include <stdio.h>
#include <thread>
#include "compiler.h"

#ifdef _WIN32
/* Suppress GCC __attribute__ on MSVC */
#ifndef __attribute__
#define __attribute__(x)
#endif
/* POSIX string/path compat */
#include <string.h>
#ifndef strcasecmp
#define strcasecmp _stricmp
#endif
#ifndef strncasecmp
#define strncasecmp _strnicmp
#endif
#ifndef PATH_MAX
#define PATH_MAX 4096
#endif
/* Thread-safe strtok: strtok_r -> strtok_s on Windows */
#ifndef strtok_r
#define strtok_r strtok_s
#endif
/* GCC __ATOMIC_* ordering constants as integers (matching GCC/CCCL conventions).
 * Use integers (not std::memory_order enum class values) so that CCCL's headers,
 * which check #ifndef __ATOMIC_RELAXED and use these as int switch-case labels,
 * work correctly under C++20 where std::memory_order is a scoped enum class. */
#ifndef __ATOMIC_RELAXED
#define __ATOMIC_RELAXED 0
#define __ATOMIC_CONSUME 1
#define __ATOMIC_ACQUIRE 2
#define __ATOMIC_RELEASE 3
#define __ATOMIC_ACQ_REL 4
#define __ATOMIC_SEQ_CST 5
#endif
/* GCC atomic builtins -> MSVC equivalents.
 * Defined as inline functions in msvc.h (NOT macros) so that CCCL's
 * msvc_to_builtins.h can declare its own template functions with the same names
 * without triggering macro expansion that breaks the declarations. */
#endif // _WIN32

// Conform to pthread and NVTX standard
#define NCCL_THREAD_NAMELEN 16

extern int ncclDebugLevel;
extern uint64_t ncclDebugMask;
extern FILE *ncclDebugFile;

#ifndef _WIN32
void ncclDebugLog(ncclDebugLogLevel level, unsigned long flags, const char *filefunc, int line, const char *fmt, ...) __attribute__ ((format (printf, 5, 6)));
#else
void ncclDebugLog(ncclDebugLogLevel level, unsigned long flags, const char *filefunc, int line, const char *fmt, ...);
#endif

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
