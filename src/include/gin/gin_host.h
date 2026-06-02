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
#include "os.h"
#include "nccl_device/gin/gin_device_host_common.h"
#include <thread>
#include <mutex>
#include <condition_variable>

#define NCCL_GIN_MAX_ACTIVE_BACKENDS 3
struct ncclGinStateDevComm {
  int contextCount;
  void* ginCtx[NCCL_GIN_MAX_CONNECTIONS];
  ncclNetDeviceHandle_t* devHandles[NCCL_GIN_MAX_CONNECTIONS];
  struct ncclGinStateDevComm* next;
};

struct ncclGinBackendState {
  ncclGinType_t ginType;      // GIN backend type.
  ncclGin_t* ncclGin;
  void* ginInstance;          // Plugin's per-comm opaque context.
  int pluginIndex;            // Index into pluginLibs[].
  int ginCommCount;
  void* ginComms[NCCL_GIN_MAX_CONNECTIONS];
  ncclNetProperties_t ginProps[NCCL_GIN_MAX_CONNECTIONS];
  int ginVersion;
  bool supportsStrongSignals;
  bool supportsVASignals;
};

struct ncclGinState {
  ncclAffinity cpuAffinity;
  bool connected;
  bool supported;              // True if any backend is loaded on this comm.
  int needsProxyProgress;      // Whether we need to progress GIN operations with the proxy
  int ginProgress;             // GIN progress is enabled
  std::thread thread;
  std::mutex mutex;
  std::condition_variable cond;
  ncclResult_t asyncResult;

  struct ncclGinStateDevComm* devComms;
  ncclGinConnectionType_t ginConnectionType;

  int numActiveBackends;
  struct ncclGinBackendState backends[NCCL_GIN_MAX_ACTIVE_BACKENDS];
};

extern int64_t ncclParamGinType();

// Get the GIN type from comm. ginType is set to the GIN type that can be used
// by the comm to communicate with other nodes.
ncclResult_t ncclGetGinType(struct ncclComm* comm, ncclGinType_t* ginType);
ncclResult_t ncclGetRailedGinType(struct ncclComm* comm, ncclGinType_t* ginType);

// FIXME change to ncclGinState instead of ncclComm, no need to pass comm
ncclResult_t ncclGinConnectOnce(struct ncclComm* comm);
ncclResult_t ncclGinHostFinalize(struct ncclComm* comm);
ncclResult_t ncclGinDevCommSetup(struct ncclComm* comm, struct ncclDevCommRequirements const* reqs,
                                 struct ncclDevComm* devComm);
ncclResult_t ncclGinDevCommFree(struct ncclComm* comm, struct ncclDevComm const* devComm);
ncclResult_t ncclGinRegister(struct ncclComm* comm, void* address, size_t size,
                             void* ginHostWins[NCCL_GIN_MAX_CONNECTIONS],
                             ncclGinWindow_t ginDevWins[NCCL_GIN_MAX_CONNECTIONS], int winFlags,
                             bool multiSegment = false, int memType = NCCL_PTR_CUDA);
ncclResult_t ncclGinDeregister(struct ncclComm* comm, void* ginHostWins[NCCL_GIN_MAX_CONNECTIONS]);

ncclResult_t ncclGinQueryLastError(struct ncclGinState* ginState, bool* hasError);

#endif
