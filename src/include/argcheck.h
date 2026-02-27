/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_ARGCHECK_H_
#define NCCL_ARGCHECK_H_

#include "core.h"
#include "info.h"

struct ncclArgsInfo {
  struct ncclInfo info;
  struct ncclArgsInfo* next;
};

ncclResult_t PtrCheck(const void* ptr, const char* opname, const char* ptrname);
ncclResult_t CommCheck(struct ncclComm* ptr, const char* opname, const char* ptrname);
ncclResult_t ArgsCheck(struct ncclInfo* info);
ncclResult_t CudaPtrCheck(const void* pointer, struct ncclComm* comm, const char* ptrname, const char* opname);
ncclResult_t ncclArgsGlobalCheck(struct ncclArgsInfo* info);

#endif
