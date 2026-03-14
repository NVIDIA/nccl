/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Windows-only stub: minimal GIN types and declarations so that no real GIN
 * headers (gin_host.h, nccl_gin.h, gin_v*.h) are included when building on Windows.
 *************************************************************************/

#ifndef _NCCL_GIN_HOST_WIN_STUB_H_
#define _NCCL_GIN_HOST_WIN_STUB_H_

#if !defined(NCCL_OS_WINDOWS)
#error "gin_host_win_stub.h is for Windows builds only"
#endif

#include "allocator.h"
#include "nccl.h"
#include "nccl_common.h"
#include "plugin/nccl_net.h"
#include <thread>
#include <mutex>
#include <condition_variable>

#define NCCL_GIN_MAX_CONNECTIONS 4

typedef void* ncclGinWindow_t;

/* Config type (same layout as ncclGinConfig_v13_t in gin_v13.h) */
typedef struct {
  int nSignals;
  int nCounters;
  int nContexts;
  int queueDepth;
  int trafficClass;
} ncclGinConfig_t;

/* Plugin struct (same layout as ncclGin_v13_t) so gin->name, gin->regMrSym, etc. compile. Not used at runtime on Windows.
 * When __CUDACC__ is defined we are in a .cu file: use a different struct tag (ncclGinHostPlugin) so the name "ncclGin"
 * is left for the device stub's type alias (ncclGin_BackendMask<...>), avoiding redefinition. */
#if defined(__CUDACC__)
struct ncclGinHostPlugin {
#else
struct ncclGin {
#endif
  const char* name;
  ncclResult_t (*init)(void** ctx, uint64_t commId, ncclDebugLogger_t logFunction);
  ncclResult_t (*devices)(int* ndev);
  ncclResult_t (*getProperties)(int dev, ncclNetProperties_t* props);
  ncclResult_t (*listen)(void* ctx, int dev, void* handle, void** listenComm);
  ncclResult_t (*connect)(void* ctx, void* handles[], int nranks, int rank, void* listenComm, void** collComm);
  ncclResult_t (*createContext)(void* collComm, ncclGinConfig_t* config, void** ginCtx, ncclNetDeviceHandle_t** devHandle);
  ncclResult_t (*regMrSym)(void* collComm, void* data, size_t size, int type, uint64_t mrFlags, void** mhandle, void** ginHandle);
  ncclResult_t (*regMrSymDmaBuf)(void* collComm, void* data, size_t size, int type, uint64_t offset, int fd, uint64_t mrFlags, void** mhandle, void** ginHandle);
  ncclResult_t (*deregMrSym)(void* collComm, void* mhandle);
  ncclResult_t (*destroyContext)(void* ginCtx);
  ncclResult_t (*closeColl)(void* collComm);
  ncclResult_t (*closeListen)(void* listenComm);
  ncclResult_t (*iput)(void* ginCtx, int context, uint64_t srcOff, void* srcMhandle, size_t size, uint64_t dstOff, void* dstMhandle, uint32_t rank, void** request);
  ncclResult_t (*iputSignal)(void* ginCtx, int context, uint64_t srcOff, void* srcMhandle, size_t size, uint64_t dstOff, void* dstMhandle, uint32_t rank, uint64_t signalOff, void* signalMhandle, uint64_t signalValue, uint32_t signalOp, void** request);
  ncclResult_t (*test)(void* collComm, void* request, int* done);
  ncclResult_t (*ginProgress)(void* ginCtx);
  ncclResult_t (*queryLastError)(void* ginCtx, bool* hasError);
  ncclResult_t (*finalize)(void* ctx);
};
#if defined(__CUDACC__)
typedef struct ncclGinHostPlugin ncclGin_t;
#else
typedef struct ncclGin ncclGin_t;
#endif

struct ncclGinStateDevComm {
  int contextCount;
  void* ginCtx[NCCL_GIN_MAX_CONNECTIONS];
  ncclNetDeviceHandle_t* devHandles[NCCL_GIN_MAX_CONNECTIONS];
  struct ncclGinStateDevComm* next;
};

struct ncclGinState {
  ncclGin_t* ncclGin;
  void* ginInstance;
  bool connected;
  ncclGinType_t ginType;
  int ginCommCount;
  void* ginComms[NCCL_GIN_MAX_CONNECTIONS];
  ncclNetProperties_t ginProps[NCCL_GIN_MAX_CONNECTIONS];
  int needsProxyProgress;
  int ginProgress;
  std::thread thread;
  std::mutex mutex;
  std::condition_variable cond;
  ncclResult_t asyncResult;
  int ginVersion;

  struct ncclGinStateDevComm* devComms;
  ncclGinConnectionType_t ginConnectionType;
};

struct ncclComm;
struct ncclDevCommRequirements;
struct ncclDevComm;

ncclResult_t setLocalGinType(struct ncclComm* comm);
ncclResult_t getGlobalGinType(struct ncclComm* comm, ncclGinType_t* ginType);
ncclResult_t getGlobalRailedGinType(struct ncclComm* comm, ncclGinType_t* ginType);
ncclResult_t ncclGinConnectOnce(struct ncclComm* comm);
ncclResult_t ncclGinHostFinalize(struct ncclComm* comm);
ncclResult_t ncclGinDevCommSetup(struct ncclComm* comm, struct ncclDevCommRequirements const* reqs, struct ncclDevComm* devComm);
ncclResult_t ncclGinDevCommFree(struct ncclComm* comm, struct ncclDevComm const* devComm);
ncclResult_t ncclGinRegister(struct ncclComm* comm, void* address, size_t size,
                             void* ginHostWins[NCCL_GIN_MAX_CONNECTIONS],
                             ncclGinWindow_t ginDevWins[NCCL_GIN_MAX_CONNECTIONS], int winFlags);
ncclResult_t ncclGinDeregister(struct ncclComm* comm, void* ginHostWins[NCCL_GIN_MAX_CONNECTIONS]);
ncclResult_t ncclGinQueryLastError(struct ncclGinState* ginState, bool* hasError);
ncclResult_t ncclGinGetDevCount(int ginPluginIndex, int* nPhysDev, int* nVirtDev);

/* Internal GIN API (from include/gin.h); stubbed on Windows */
ncclResult_t ncclGinInit(struct ncclComm* comm);
ncclResult_t ncclGinInitFromParent(struct ncclComm* comm, struct ncclComm* parent);
ncclResult_t ncclGinFinalize(struct ncclComm* comm);

#endif
