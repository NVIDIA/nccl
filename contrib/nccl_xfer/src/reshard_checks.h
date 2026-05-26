/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information.
 ************************************************************************/

#ifndef NCCLXFER_RESHARD_CHECKS_H_
#define NCCLXFER_RESHARD_CHECKS_H_

#include <cstdio>
#include <cstdlib>

#include "cuda_runtime.h"
#include "nccl.h"

/*
 * Library-safe macros: return an error code instead of calling exit().
 * Use these in any function that returns ncclResult_t.
 */
#define NCCLXFER_CHECK(cmd)                                                                                   \
  do {                                                                                                        \
    ncclResult_t res = (cmd);                                                                                 \
    if (res != ncclSuccess) {                                                                                 \
      fprintf(stderr, "[nccl-reshard] NCCL error %s:%d '%s'\n", __FILE__, __LINE__, ncclGetErrorString(res)); \
      return res;                                                                                             \
    }                                                                                                         \
  } while (0)

#define NCCLXFER_CUDACHECK(cmd)                                                                               \
  do {                                                                                                        \
    cudaError_t err = (cmd);                                                                                  \
    if (err != cudaSuccess) {                                                                                 \
      fprintf(stderr, "[nccl-reshard] CUDA error %s:%d '%s'\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
      return ncclSystemError;                                                                                 \
    }                                                                                                         \
  } while (0)

/*
 * Warn-and-continue variants: log the error code/message but do NOT
 * return.  Use in teardown paths where we must keep iterating to free
 * other resources.
 */
#define NCCLXFER_CHECK_WARN(cmd)                                                                 \
  do {                                                                                           \
    ncclResult_t res = (cmd);                                                                    \
    if (res != ncclSuccess) {                                                                    \
      fprintf(stderr, "[nccl-reshard] NCCL error %s:%d '%s' (continuing)\n", __FILE__, __LINE__, \
              ncclGetErrorString(res));                                                          \
    }                                                                                            \
  } while (0)

#define NCCLXFER_CUDACHECK_WARN(cmd)                                                             \
  do {                                                                                           \
    cudaError_t err = (cmd);                                                                     \
    if (err != cudaSuccess) {                                                                    \
      fprintf(stderr, "[nccl-reshard] CUDA error %s:%d '%s' (continuing)\n", __FILE__, __LINE__, \
              cudaGetErrorString(err));                                                          \
    }                                                                                            \
  } while (0)

#endif /* NCCLXFER_RESHARD_CHECKS_H_ */
