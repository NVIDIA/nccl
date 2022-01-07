/*************************************************************************
 * Copyright (c) 2019-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_CHECKS_H_
#define NCCL_CHECKS_H_

#include "debug.h"

// Check CUDA calls
#define CUDACHECK(cmd) do {                                 \
    cudaError_t err = cmd;                                  \
    if( err != cudaSuccess ) {                              \
        WARN("Cuda failure '%s'", cudaGetErrorString(err)); \
        return ncclUnhandledCudaError;                      \
    }                                                       \
} while(false)

#define CUDACHECKGOTO(cmd, res, label) do {                 \
    cudaError_t err = cmd;                                  \
    if( err != cudaSuccess ) {                              \
        WARN("Cuda failure '%s'", cudaGetErrorString(err)); \
        res = ncclUnhandledCudaError;                       \
        goto label;                                         \
    }                                                       \
} while(false)

// Report failure but clear error and continue
#define CUDACHECKIGNORE(cmd) do {  \
    cudaError_t err = cmd;         \
    if( err != cudaSuccess ) {     \
        INFO(NCCL_ALL,"%s:%d Cuda failure '%s'", __FILE__, __LINE__, cudaGetErrorString(err)); \
        (void) cudaGetLastError(); \
    }                              \
} while(false)

#include <errno.h>
// Check system calls
#define SYSCHECK(call, name) do { \
  int retval; \
  SYSCHECKVAL(call, name, retval); \
} while (false)

#define SYSCHECKVAL(call, name, retval) do { \
  SYSCHECKSYNC(call, name, retval); \
  if (retval == -1) { \
    WARN("Call to " name " failed : %s", strerror(errno)); \
    return ncclSystemError; \
  } \
} while (false)

#define SYSCHECKSYNC(call, name, retval) do { \
  retval = call; \
  if (retval == -1 && (errno == EINTR || errno == EWOULDBLOCK || errno == EAGAIN)) { \
    INFO(NCCL_ALL,"Call to " name " returned %s, retrying", strerror(errno)); \
  } else { \
    break; \
  } \
} while(true)

#define SYSCHECKGOTO(statement, res, label) do { \
  if ((statement) == -1) {    \
    /* Print the back trace*/ \
    res = ncclSystemError;    \
    INFO(NCCL_ALL,"%s:%d -> %d", __FILE__, __LINE__, res);    \
    goto label; \
  } \
} while (0);

#define NEQCHECK(statement, value) do {   \
  if ((statement) != value) {             \
    /* Print the back trace*/             \
    INFO(NCCL_ALL,"%s:%d -> %d", __FILE__, __LINE__, ncclSystemError);    \
    return ncclSystemError;     \
  }                             \
} while (0);

#define NEQCHECKGOTO(statement, value, res, label) do { \
  if ((statement) != value) { \
    /* Print the back trace*/ \
    res = ncclSystemError;    \
    INFO(NCCL_ALL,"%s:%d -> %d", __FILE__, __LINE__, res);    \
    goto label; \
  } \
} while (0);

#define EQCHECK(statement, value) do {    \
  if ((statement) == value) {             \
    /* Print the back trace*/             \
    INFO(NCCL_ALL,"%s:%d -> %d", __FILE__, __LINE__, ncclSystemError);    \
    return ncclSystemError;     \
  }                             \
} while (0);

#define EQCHECKGOTO(statement, value, res, label) do { \
  if ((statement) == value) { \
    /* Print the back trace*/ \
    res = ncclSystemError;    \
    INFO(NCCL_ALL,"%s:%d -> %d", __FILE__, __LINE__, res);    \
    goto label; \
  } \
} while (0);

// Propagate errors up
#define NCCLCHECK(call) do { \
  ncclResult_t res = call; \
  if (res != ncclSuccess) { \
    /* Print the back trace*/ \
    if (ncclDebugNoWarn == 0) INFO(NCCL_ALL,"%s:%d -> %d", __FILE__, __LINE__, res);    \
    return res; \
  } \
} while (0);

#define NCCLCHECKGOTO(call, res, label) do { \
  res = call; \
  if (res != ncclSuccess) { \
    /* Print the back trace*/ \
    if (ncclDebugNoWarn == 0) INFO(NCCL_ALL,"%s:%d -> %d", __FILE__, __LINE__, res);    \
    goto label; \
  } \
} while (0);

#define NCCLWAIT(call, cond, abortFlagPtr) do {         \
  volatile uint32_t* tmpAbortFlag = (abortFlagPtr);     \
  ncclResult_t res = call;                \
  if (res != ncclSuccess) {               \
    if (ncclDebugNoWarn == 0) INFO(NCCL_ALL,"%s:%d -> %d", __FILE__, __LINE__, res);    \
    return ncclInternalError;             \
  }                                       \
  if (tmpAbortFlag) NEQCHECK(*tmpAbortFlag, 0); \
} while (!(cond));

#define NCCLWAITGOTO(call, cond, abortFlagPtr, res, label) do { \
  volatile uint32_t* tmpAbortFlag = (abortFlagPtr);             \
  res = call;                             \
  if (res != ncclSuccess) {               \
    if (ncclDebugNoWarn == 0) INFO(NCCL_ALL,"%s:%d -> %d", __FILE__, __LINE__, res);    \
    goto label;                           \
  }                                       \
  if (tmpAbortFlag) NEQCHECKGOTO(*tmpAbortFlag, 0, res, label); \
} while (!(cond));

#define NCCLCHECKTHREAD(a) do { \
  if ((args->ret = (a)) != ncclSuccess) { \
    INFO(NCCL_INIT,"%s:%d -> %d [Async thread]", __FILE__, __LINE__, args->ret); \
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

#endif
