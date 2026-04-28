/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_CHECKS_H_
#define NCCL_CHECKS_H_

#include "debug.h"

// Check CUDA RT calls
#define CUDACHECK(cmd)                                                         \
  do {                                                                         \
    cudaError_t err = cmd;                                                     \
    if (err != cudaSuccess) {                                                  \
      WARN("Cuda failure '%s'", cudaGetErrorString(err));                      \
      (void)cudaGetLastError();                                                \
      return ncclUnhandledCudaError;                                           \
    }                                                                          \
  } while (false)

#define CUDACHECKGOTO(cmd, RES, label)                                         \
  do {                                                                         \
    cudaError_t err = cmd;                                                     \
    if (err != cudaSuccess) {                                                  \
      WARN("Cuda failure '%s'", cudaGetErrorString(err));                      \
      (void)cudaGetLastError();                                                \
      RES = ncclUnhandledCudaError;                                            \
      goto label;                                                              \
    }                                                                          \
  } while (false)

// Report failure but clear error and continue
#define CUDACHECKIGNORE(cmd)                                                   \
  do {                                                                         \
    cudaError_t err = cmd;                                                     \
    if (err != cudaSuccess) {                                                  \
      INFO(NCCL_ALL, "%s:%d Cuda failure '%s'", __FILE__, __LINE__,            \
           cudaGetErrorString(err));                                           \
      (void)cudaGetLastError();                                                \
    }                                                                          \
  } while (false)

// Use inline function to clear CUDA error inside expressions
static inline cudaError_t cuda_clear(cudaError_t err) {
  if (err != cudaSuccess)
    (void)cudaGetLastError();
  return err;
}

// Check if cudaSuccess & clear CUDA error
#define CUDASUCCESS(cmd) cuda_clear(cmd) == cudaSuccess
// Clear CUDA error, return CUDA return code
#define CUDACLEARERROR(cmd) cuda_clear(cmd)

#include <errno.h>
#include <string.h>
#include <stdio.h>

// Thread-safe strerror wrapper. Handles GNU vs POSIX strerror_r variants.
static inline const char* ncclStrerror(int errnum, char* buf, size_t buflen) {
#if (_POSIX_C_SOURCE >= 200112L) && !defined(_GNU_SOURCE)
  // POSIX variant: int strerror_r(int, char*, size_t)
  if (strerror_r(errnum, buf, buflen) != 0)
    snprintf(buf, buflen, "Unknown error %d", errnum);
  return buf;
#else
  // GNU variant: char* strerror_r(int, char*, size_t)
  return strerror_r(errnum, buf, buflen);
#endif
}

// Check system calls
#define SYSCHECK(statement, name) do { \
  int retval; \
  SYSCHECKSYNC((statement), name, retval); \
  if (retval == -1) { \
    char _errBuf[256]; \
    WARN("Call to " name " failed: %s", ncclStrerror(errno, _errBuf, sizeof(_errBuf))); \
    return ncclSystemError; \
  } \
} while (false)

#define SYSCHECKSYNC(statement, name, retval) do { \
  retval = (statement); \
  if (retval == -1 && (errno == EINTR || errno == EWOULDBLOCK || errno == EAGAIN)) { \
    char _errBuf[256]; \
    INFO(NCCL_ALL,"Call to " name " returned %s, retrying", ncclStrerror(errno, _errBuf, sizeof(_errBuf))); \
  } else { \
    break; \
  } \
} while(true)

#define SYSCHECKGOTO(statement, name, RES, label) do { \
  int retval; \
  SYSCHECKSYNC((statement), name, retval); \
  if (retval == -1) { \
    char _errBuf[256]; \
    WARN("Call to " name " failed: %s", ncclStrerror(errno, _errBuf, sizeof(_errBuf))); \
    RES = ncclSystemError; \
    goto label; \
  } \
} while (0)

// Pthread calls don't set errno and never return EINTR.
#define PTHREADCHECK(statement, name) do { \
  int retval = (statement); \
  if (retval != 0) { \
    char _errBuf[256]; \
    WARN("Call to " name " failed: %s", ncclStrerror(retval, _errBuf, sizeof(_errBuf))); \
    return ncclSystemError; \
  } \
} while (0)

#define PTHREADCHECKGOTO(statement, name, RES, label) do { \
  int retval = (statement); \
  if (retval != 0) { \
    char _errBuf[256]; \
    WARN("Call to " name " failed: %s", ncclStrerror(retval, _errBuf, sizeof(_errBuf))); \
    RES = ncclSystemError; \
    goto label; \
  } \
} while (0)

#define NEQCHECK(statement, value) do {   \
  if ((statement) != value) {             \
    /* Print the back trace*/             \
    char _errBuf[256]; \
    INFO(NCCL_ALL,"%s:%d -> %d (%s)", __FILE__, __LINE__, ncclSystemError, ncclStrerror(errno, _errBuf, sizeof(_errBuf)));    \
    return ncclSystemError;     \
  }                             \
} while (0)

#define NEQCHECKGOTO(statement, value, RES, label) do { \
  if ((statement) != value) { \
    /* Print the back trace*/ \
    char _errBuf[256]; \
    RES = ncclSystemError;    \
    INFO(NCCL_ALL,"%s:%d -> %d (%s)", __FILE__, __LINE__, RES, ncclStrerror(errno, _errBuf, sizeof(_errBuf)));    \
    goto label; \
  } \
} while (0)

#define EQCHECK(statement, value) do {    \
  if ((statement) == value) {             \
    /* Print the back trace*/             \
    char _errBuf[256]; \
    INFO(NCCL_ALL,"%s:%d -> %d (%s)", __FILE__, __LINE__, ncclSystemError, ncclStrerror(errno, _errBuf, sizeof(_errBuf)));    \
    return ncclSystemError;     \
  }                             \
} while (0)

#define EQCHECKGOTO(statement, value, RES, label) do { \
  if ((statement) == value) { \
    /* Print the back trace*/ \
    char _errBuf[256]; \
    RES = ncclSystemError;    \
    INFO(NCCL_ALL,"%s:%d -> %d (%s)", __FILE__, __LINE__, RES, ncclStrerror(errno, _errBuf, sizeof(_errBuf)));    \
    goto label; \
  } \
} while (0)

// Propagate errors up
#define NCCLCHECK(call) do { \
  ncclResult_t RES = call; \
  if (RES != ncclSuccess && RES != ncclInProgress) { \
    /* Print the back trace*/ \
    if (ncclDebugNoWarn == 0) INFO(NCCL_ALL,"%s:%d -> %d", __FILE__, __LINE__, RES);    \
    return RES; \
  } \
} while (0)

#define NCCLCHECKGOTO(call, RES, label) do { \
  RES = call; \
  if (RES != ncclSuccess && RES != ncclInProgress) { \
    /* Print the back trace*/ \
    if (ncclDebugNoWarn == 0) INFO(NCCL_ALL,"%s:%d -> %d", __FILE__, __LINE__, RES);    \
    goto label; \
  } \
} while (0)

// Report failure but continue - useful for cleanup paths where we want to
// attempt all cleanup steps. Preserves the first error in RES.
#define NCCLCHECKIGNORE(call, RES) do { \
  ncclResult_t TMPRES = call; \
  if (TMPRES != ncclSuccess && TMPRES != ncclInProgress) { \
    if (ncclDebugNoWarn == 0) INFO(NCCL_ALL,"%s:%d -> %d", __FILE__, __LINE__, TMPRES); \
    if (RES == ncclSuccess) RES = TMPRES; \
  } \
} while (0)

#define NCCLCHECKNOWARN(call, FLAGS) do { \
  ncclResult_t RES; \
  NOWARN(RES = call, FLAGS); \
  if (RES != ncclSuccess && RES != ncclInProgress) { \
    return RES; \
  } \
} while (0)

#define NCCLCHECKGOTONOWARN(call, RES, label, FLAGS) do { \
  NOWARN(RES = call, FLAGS); \
  if (RES != ncclSuccess && RES != ncclInProgress) { \
    goto label; \
  } \
} while (0)

#define NCCLWAIT(call, cond, abortFlagPtr) do {         \
  uint32_t* tmpAbortFlag = (abortFlagPtr);     \
  ncclResult_t RES = call;                \
  if (RES != ncclSuccess && RES != ncclInProgress) {               \
    if (ncclDebugNoWarn == 0) INFO(NCCL_ALL,"%s:%d -> %d", __FILE__, __LINE__, RES);    \
    return ncclInternalError;             \
  }                                       \
  if (COMPILER_ATOMIC_LOAD(tmpAbortFlag, std::memory_order_acquire)) NEQCHECK(*tmpAbortFlag, 0); \
} while (!(cond))

#define NCCLWAITGOTO(call, cond, abortFlagPtr, RES, label) do { \
  uint32_t* tmpAbortFlag = (abortFlagPtr);             \
  RES = call;                             \
  if (RES != ncclSuccess && RES != ncclInProgress) {               \
    if (ncclDebugNoWarn == 0) INFO(NCCL_ALL,"%s:%d -> %d", __FILE__, __LINE__, RES);    \
    goto label;                           \
  }                                       \
  if (COMPILER_ATOMIC_LOAD(tmpAbortFlag, std::memory_order_acquire)) NEQCHECKGOTO(*tmpAbortFlag, 0, RES, label); \
} while (!(cond))

#define NCCLCHECKTHREAD(a, args) do { \
  if (((args)->ret = (a)) != ncclSuccess && (args)->ret != ncclInProgress) { \
    INFO(NCCL_INIT,"%s:%d -> %d [Async thread]", __FILE__, __LINE__, (args)->ret); \
    return args; \
  } \
} while(0)

#define CUDACHECKTHREAD(a) do { \
  if ((a) != cudaSuccess) { \
    INFO(NCCL_INIT,"%s:%d -> %d [Async thread]", __FILE__, __LINE__, args->ret); \
    args->ret = ncclUnhandledCudaError; \
    return args; \
  } \
} while(0)

// Common thread creation implementation with error handling
#define STDTHREADCREATE_IMPL(var, func, error_action, ...) do { \
  try { \
    (var) = std::thread(func, __VA_ARGS__); \
  } catch (const std::exception& e) { \
    WARN("Thread creation failed: %s", e.what()); \
    error_action; \
  } \
} while(0)

#define STDTHREADCREATE(var, func, ...) \
  STDTHREADCREATE_IMPL(var, func, return ncclSystemError, __VA_ARGS__)

#define STDTHREADCREATE_GOTO(var, func, RES, label, ...) \
  STDTHREADCREATE_IMPL(var, func, do { RES = ncclSystemError; goto label; } while(0), __VA_ARGS__)

#define NEW_NOTHROW(var, x) do { \
  (var) = new (std::nothrow) x{}; \
  if (!(var)) { \
    WARN("Allocation failed at %s:%d", __FILE__, __LINE__); \
    return ncclSystemError; \
  } \
} while(0)

#define NEW_NOTHROW_GOTO(var, x, RES, label) do { \
  (var) = new (std::nothrow) x{}; \
  if (!(var)) { \
    WARN("Allocation failed at %s:%d", __FILE__, __LINE__); \
    RES = ncclSystemError; \
    goto label; \
  } \
} while(0)

#endif
