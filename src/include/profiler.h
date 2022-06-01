/*************************************************************************
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_INT_PROFILER_H_
#define NCCL_INT_PROFILER_H_

#include "nccl_profiler.h"
#include "proxy.h"

ncclResult_t ncclProfilerInit();

ncclResult_t ncclProfilingRecord(struct ncclProxyArgs* args, int sub, int step, int state);
void ncclProfilingDump();

#endif
