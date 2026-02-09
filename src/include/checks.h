/*************************************************************************
 * Copyright (c) 2019-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

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
// Check system calls
#define SYSCHECK(statement, name) do { \
  int retval; \
  SYSCHECKSYNC((statement), name, retval); \
  if (retval == -1) { \
    WARN("Call to " name " failed: %s", strerror(errno)); \
    return ncclSystemError; \
  } \
} while (false)

#define SYSCHECKSYNC(statement, name, retval) do { \
  retval = (statement); \
  if (retval == -1 && (errno == EINTR || errno == EWOULDBLOCK || errno == EAGAIN)) { \
    INFO(NCCL_ALL,"Call to " name " returned %s, retrying", strerror(errno)); \
  } else { \
    break; \
  } \
} while(true)

#define SYSCHECKGOTO(statement, name, RES, label) do { \
  int retval; \
  SYSCHECKSYNC((statement), name, retval); \
  if (retval == -1) { \
    WARN("Call to " name " failed: %s", strerror(errno)); \
    RES = ncclSystemError; \
    goto label; \
  } \
} while (0)

// Pthread calls don't set errno and never return EINTR.
#define PTHREADCHECK(statement, name) do { \
  int retval = (statement); \
  if (retval != 0) { \
    WARN("Call to " name " failed: %s", strerror(retval)); \
    return ncclSystemError; \
  } \
} while (0)

#define PTHREADCHECKGOTO(statement, name, RES, label) do { \
  int retval = (statement); \
  if (retval != 0) { \
    WARN("Call to " name " failed: %s", strerror(retval)); \
    RES = ncclSystemError; \
    goto label; \
  } \
} while (0)

#define NEQCHECK(statement, value) do {   \
  if ((statement) != value) {             \
    /* Print the back trace*/             \
    INFO(NCCL_ALL,"%s:%d -> %d (%s)", __FILE__, __LINE__, ncclSystemError, strerror(errno));    \
    return ncclSystemError;     \
  }                             \
} while (0)

#define NEQCHECKGOTO(statement, value, RES, label) do { \
  if ((statement) != value) { \
    /* Print the back trace*/ \
    RES = ncclSystemError;    \
    INFO(NCCL_ALL,"%s:%d -> %d (%s)", __FILE__, __LINE__, RES, strerror(errno));    \
    goto label; \
  } \
} while (0)

#define EQCHECK(statement, value) do {    \
  if ((statement) == value) {             \
    /* Print the back trace*/             \
    INFO(NCCL_ALL,"%s:%d -> %d (%s)", __FILE__, __LINE__, ncclSystemError, strerror(errno));    \
    return ncclSystemError;     \
  }                             \
} while (0)

#define EQCHECKGOTO(statement, value, RES, label) do { \
  if ((statement) == value) { \
    /* Print the back trace*/ \
    RES = ncclSystemError;    \
    INFO(NCCL_ALL,"%s:%d -> %d (%s)", __FILE__, __LINE__, RES, strerror(errno));    \
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
