/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_SHM_H_
#define NCCL_SHM_H_

#include "nccl.h"

typedef void* ncclShmHandle_t;
ncclResult_t ncclShmOpen(char* shmPath, size_t shmSize, void** shmPtr, void** devShmPtr, int refcount, ncclShmHandle_t* handle);
ncclResult_t ncclShmClose(ncclShmHandle_t handle);
ncclResult_t ncclShmUnlink(ncclShmHandle_t handle);

#endif
