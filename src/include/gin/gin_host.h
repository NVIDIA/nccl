/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef _NCCL_GIN_HOST_H_
#define _NCCL_GIN_HOST_H_

#include "allocator.h"
#include "nccl.h"
#include "nccl_gin.h"
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
  int ginContextCount;
  void* ginComms[NCCL_GIN_MAX_CONNECTIONS];
  void* ginCtx[NCCL_GIN_MAX_CONNECTIONS];
  ncclNetDeviceHandle_t* ginDevHandles[NCCL_GIN_MAX_CONNECTIONS];
  int needsProxyProgress;  // Whether we need to progress GIN operations with the proxy
  int ginProgress;         // GIN progress is enabled
  std::thread thread;
  std::mutex mutex;
  std::condition_variable cond;
  ncclResult_t asyncResult;
  int ginVersion;

  int signalSpaceSize;
  int counterSpaceSize;
  ncclSpace signalSpace;
  ncclSpace counterSpace;
  int ctxFirstAvailable; // We allocate shared contexts starting from index 0.
  int ctxLastExclusive; // We allocate exclusive contexts starting from the highest index.
  int ginQueueDepth;
  ncclGinConnectionType_t ginConnectionType;
};

extern int64_t ncclParamGinType();

// Sets the local GIN type for comm. The GIN type that is set for comm is the
// GIN type supported by the call process itself, without taking into account
// (1) GIN support of other ranks, and (2) additional local constraints like
// cross-NIC
ncclResult_t setLocalGinType(struct ncclComm* comm);
// Get the GIN type from comm. ginType is set to the GIN type that can be used
// by the comm to communicate with other nodes.
ncclResult_t getGlobalGinType(struct ncclComm* comm, ncclGinType_t* ginType);
ncclResult_t getGlobalRailedGinType(struct ncclComm* comm, ncclGinType_t* ginType);

// FIXME change to ncclGinState instead of ncclComm, no need to pass comm
ncclResult_t ncclGinConnectOnce(struct ncclComm* comm, ncclGinConnectionType_t requestedConnectionType, int reqGinContextCount = 0, int reqGinQueueDepth = 0);
ncclResult_t ncclGinHostFinalize(struct ncclComm* comm);
ncclResult_t ncclGinRegister(struct ncclComm* comm, void* address, size_t size,
                             void* ginHostWins[NCCL_GIN_MAX_CONNECTIONS],
                             ncclGinWindow_t ginDevWins[NCCL_GIN_MAX_CONNECTIONS], int winFlags);
ncclResult_t ncclGinDeregister(struct ncclComm* comm, void* ginHostWins[NCCL_GIN_MAX_CONNECTIONS]);
ncclResult_t ncclGinAllocSignalsCounters(struct ncclComm* comm, int nSignals, uint32_t* outSignal0,
                                         int nCounters, uint32_t* outCounter0);
ncclResult_t ncclGinFreeSignalsCounters(struct ncclComm* comm, uint32_t signal0, int nSignals,
                                        uint32_t counter0, int nCounters);
ncclResult_t ncclGinQueryLastError(struct ncclGinState* ginState, bool* hasError);

#endif
