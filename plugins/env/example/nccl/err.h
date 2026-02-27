/*
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
 */

#ifndef ERR_H_
#define ERR_H_

// NCCL error codes
#define ncclSuccess 0
#define ncclSystemError 1
#define ncclInternalError 2
#define ncclInvalidUsage 3
#define ncclInvalidArgument 4
#define ncclUnhandledCudaError 5

typedef int ncclResult_t;

#endif
