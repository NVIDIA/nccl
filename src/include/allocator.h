/*************************************************************************
 * Copyright (c) 2015-2025, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_ALLOCATOR_H_
#define NCCL_ALLOCATOR_H_

ncclResult_t ncclCommSymmetricAllocInternal(struct ncclComm* comm, size_t size, size_t alignment, void** symPtr);
ncclResult_t ncclCommSymmetricFreeInternal(struct ncclComm* comm, void* symPtr);

#endif
