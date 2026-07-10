/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef UBX_COMMON_H_
#define UBX_COMMON_H_

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define UBX_CHECK_CUDA(cmd)                                                          \
  do {                                                                                      \
    cudaError_t e = cmd;                                                                    \
    if (e != cudaSuccess) {                                                                 \
      printf("Failed: Cuda error %s:%d '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
      exit(EXIT_FAILURE);                                                                   \
    }                                                                                       \
  } while (0)

#define UBX_CHECK_CUDA_DRV(cmd)                                                          \
  do {                                                                                          \
    CUresult e = cmd;                                                                           \
    if (e != CUDA_SUCCESS) {                                                                    \
      const char *errStr;                                                                       \
      cuGetErrorString(e, &errStr);                                                             \
      printf("Failed: CUDA driver error %s:%d '%s'\n", __FILE__, __LINE__, errStr ? errStr : "unknown"); \
      exit(EXIT_FAILURE);                                                                       \
    }                                                                                           \
  } while (0)

#endif  // UBX_COMMON_H_
