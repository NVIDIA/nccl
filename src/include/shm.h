/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_SHM_H_
#define NCCL_SHM_H_

#include "nccl.h"

ncclResult_t ncclShmOpen(char* shmPath, const int shmSize, void** shmPtr, void** devShmPtr, int create);
ncclResult_t ncclShmUnlink(const char* shmname);
ncclResult_t ncclShmClose(void* shmPtr, void* devShmPtr, const int shmSize);
#endif
