/*************************************************************************
 * Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "core.h"

ncclResult_t PtrCheck(void* ptr, const char* opname, const char* ptrname);
ncclResult_t ArgsCheck(struct ncclInfo* info);
