/*************************************************************************
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_PROFILER_H_
#define NCCL_PROFILER_H_

#include "proxy.h"

enum ncclProxyProfileState {
  ncclProxyProfileBegin = 0,

  ncclProxyProfileSendGPUWait = 1,
  ncclProxyProfileSendWait = 2,

  ncclProxyProfileRecvWait = 1,
  ncclProxyProfileRecvFlushWait = 2,
  ncclProxyProfileRecvGPUWait = 3,

  ncclProxyProfileEnd = 4,

  ncclProxyProfileSleep = 8,
  ncclProxyProfileWakeup = 9,

  ncclProxyProfileIdle = 16,
  ncclProxyProfileActive = 17,

  ncclProxyProfileAppend = 24,
  ncclProxyProfileAppendEnd = 25
};

ncclResult_t ncclProfilingRecord(struct ncclProxyArgs* args, int sub, int step, int state);
void ncclProfilingDump();

#endif
