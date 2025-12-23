/*************************************************************************
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef _NCCL_GIN_HOST_H_
#define _NCCL_GIN_HOST_H_

#include "allocator.h"
#include "nccl.h"
#include "nccl_net.h"
#include "nccl_device/gin/gin_device_host_common.h"
#include <thread>
#include <mutex>
#include <condition_variable>

struct ncclGinState {
  ncclGin_t* ncclGin;
  void* ginInstance;
  bool connected;
  ncclGinType_t ginType;
  int ginCommCount;
  void* ginComms[NCCL_GIN_MAX_CONTEXTS];
  void* ginCtx[NCCL_GIN_MAX_CONTEXTS];
  ncclNetDeviceHandle_t* ginDevHandles[NCCL_GIN_MAX_CONTEXTS];
  int needsProxyProgress;  // Whether we need to progress GIN operations with the proxy
  int ginProgress;         // GIN progress is enabled
  std::thread thread;
  std::mutex mutex;
  std::condition_variable cond;
  ncclResult_t asyncResult;

  int signalSpaceSize;
  int counterSpaceSize;
  ncclSpace signalSpace;
  ncclSpace counterSpace;
};

extern int64_t ncclParamGinType();

// Get the GIN type from comm
ncclResult_t getGinType(struct ncclComm* comm, ncclGinType_t* ginType);

// FIXME change to ncclGinState instead of ncclComm, no need to pass comm
ncclResult_t ncclGinConnectOnce(struct ncclComm* comm);
ncclResult_t ncclGinFinalize(struct ncclComm* comm);
ncclResult_t ncclGinRegister(struct ncclComm* comm, void* address, size_t size,
                             void* ginHostWins[NCCL_GIN_MAX_CONTEXTS],
                             ncclGinWindow_t ginDevWins[NCCL_GIN_MAX_CONTEXTS]);
ncclResult_t ncclGinDeregister(struct ncclComm* comm, void* ginHostWins[NCCL_GIN_MAX_CONTEXTS]);
ncclResult_t ncclGinAllocSignalsCounters(struct ncclComm* comm, int nSignals, uint32_t* outSignal0,
                                         int nCounters, uint32_t* outCounter0);
ncclResult_t ncclGinFreeSignalsCounters(struct ncclComm* comm, uint32_t signal0, int nSignals,
                                        uint32_t counter0, int nCounters);
ncclResult_t ncclGinQueryLastError(struct ncclGinState* ginState, bool* hasError);

#endif
