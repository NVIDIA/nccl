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

struct ncclShmemCollBuff {
  volatile size_t *cnt[2];
  volatile void *ptr[2];
  int round;
  size_t maxTypeSize;
};

ncclResult_t ncclShmemAllgather(struct ncclComm *comm, struct ncclShmemCollBuff *shmem, void *sendbuff, void *recvbuff, size_t typeSize);

#endif
