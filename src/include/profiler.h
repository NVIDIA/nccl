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
  ncclProxyProfileRemFIFOWait = 2,
  ncclProxyProfileSendWait = 3,

  ncclProxyProfileRecvWait = 4,
  ncclProxyProfileRecvFlushWait = 5,
  ncclProxyProfileRecvGPUWait = 6,

  ncclProxyProfileEnd = 7,

  ncclProxyProfileSleep = 8,
  ncclProxyProfileWakeup = 9,

  ncclProxyProfileIdle = 16,
  ncclProxyProfileActive = 17,

  ncclProxyProfileAppend = 24,
  ncclProxyProfileAppendEnd = 25
};

ncclResult_t ncclProfilingRecord(struct ncclProxyArgs* args, int sub, int step, int state);
#ifdef ENABLE_FB_PROFILE_PROXY
ncclResult_t ncclProfilingRecordUpdate(struct ncclProxyArgs* args, int sub, int step, int peer, int chunkSize);
#else
#define ncclProfilingRecordUpdate(args, sub, step, peer, chunkSize) do {} while (0)
#endif
void ncclProfilingDump();

#endif
