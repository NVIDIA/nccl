/*************************************************************************
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_ARGCHECK_H_
#define NCCL_ARGCHECK_H_

#include "core.h"
#include "info.h"

ncclResult_t PtrCheck(void* ptr, const char* opname, const char* ptrname);
ncclResult_t ArgsCheck(struct ncclInfo* info);
ncclResult_t CudaPtrCheck(const void* pointer, struct ncclComm* comm, const char* ptrname, const char* opname);

#endif
