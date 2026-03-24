/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef _NCCL_RMA_PROXY_H_
#define _NCCL_RMA_PROXY_H_

#include "nccl.h"
#include "nccl_net.h"
#include "nccl_common.h"
#if defined(NCCL_OS_WINDOWS)
#include "gin/gin_host_win_stub.h"
#else
#include "gin/gin_host.h"
#endif
#include "alloc.h"
#include <thread>
#include <mutex>
#include <condition_variable>

struct ncclComm;
struct ncclRmaArgs;

struct ncclRmaSignal_t {
  void *signalMhandle;
  uint64_t offset;
  uint64_t val;
  uint32_t op;
};

typedef enum ncclRmaDescState_t {
  ncclRmaDescStateInit = 0,
  ncclRmaDescStateReady = 1,
  ncclRmaDescStateInProgress = 2,
} ncclRmaDescState_t;

typedef enum ncclRmaDescType_t {
  ncclRmaDescTypePutSignal = 0,
  ncclRmaDescTypeWaitSignal,
} ncclRmaDescType_t;

struct ncclRmaPutSignalDesc {
  // Network function descriptor
  uint64_t srcOff;
  void *srcHandle;
  uint64_t dstOff;
  void *dstHandle;
  size_t size;
  int targetRank;
  ncclRmaSignal_t signal;

  // Request handle for the network operation
  void* request;
};

struct ncclRmaWaitSignalDesc {
  int npeers;
  int* waitPeers;
  int* waitSignals;
  // Local flush in graph mode
  int needFlush;
};

struct ncclRmaProxyDesc {
  struct ncclRmaProxyDesc *next;
  ncclRmaDescType_t rmaDescType;
  ncclRmaDescState_t rmaDescState;

  union {
    struct ncclRmaPutSignalDesc putSignal;
    struct ncclRmaWaitSignalDesc waitSignal;
  };

  // Non graph mode, desc does not own the sequence allocations but points to the ctx's sequence allocations
  // Graph mode, desc owns the per-descriptor sequence allocations and this needs to be freed when the desc is destroyed
  uint64_t opSeq;
  uint64_t* readySeq;
  uint64_t* readySeqDev;
  void* readySeqGdrHandle;
  uint64_t* doneSeq;
  uint64_t* doneSeqDev;
  void* doneSeqGdrHandle;

  // Graph capture fields
  struct ncclKernelPlan* persistPlan; // Back reference to persistent plan during clean up
  bool persistDescValid; // Persistent descriptor is valid

};

struct ncclRmaProxyCtx {
  struct ncclComm *comm;

  // GIN context for the RMA proxy context
  void *ginCollComm;
  void *ginCtx;
  ncclNetDeviceHandle_t *devHandle;
  ncclNetProperties_t props;

  //---------Non-graph descriptor queues and synchronization---------

  // Lock-free circular buffer for pending Descs
  size_t queueSize;  // Power of 2 size for pending queue
  struct ncclRmaProxyDesc** circularBuffers;  // Lock-free circular buffer per peer
  uint32_t* pis;  // Producer Indices per peer
  uint32_t* cis;  // Consumer Indices per peer

  // Per-rank inProgressQueues: Descs with issued network operations waiting for completion
  struct ncclIntruQueue<struct ncclRmaProxyDesc, &ncclRmaProxyDesc::next>* inProgressQueues;

  // Per-rank sequence number and counters
  uint64_t* opSeqs;
  uint64_t* opSeqsDev;
  void* opSeqsGdrHandle;
  uint64_t* readySeqs;
  uint64_t* readySeqsDev;
  void* readySeqsGdrHandle;
  uint64_t* doneSeqs;
  uint64_t* doneSeqsDev;
  void* doneSeqsGdrHandle;

  // Signal memory layout and management
  // Each RMA context allocates a signal buffer with the following layout:
  // - Offsets [0 to nRanks*8-1]: per-rank distinct signals (8 bytes per rank)
  // - Offset [nRanks*8]: shared aggregate signal counter (8 bytes)
  // Total signal buffer size: (nRanks + 1) * 8 bytes
  CUmemGenericAllocationHandle signalsCumemhandle;
  void *signalsMhandle;
  void *signalsGinHandle;
  uint64_t *signalsDev;
  uint64_t* signalsHost; // Host buffer to track the expected values of the signals

  //---------Graph descriptor queues and synchronization---------

  // Per-rank persistent descriptor queue: Descs from all live graphs
  struct ncclIntruQueue<struct ncclRmaProxyDesc, &ncclRmaProxyDesc::next>* persistentQueues;

  // CPU-accessible signal is required as proxy needs to poll on the signal values
  void *cpuAccessSignalsGdrHandle;
  void *cpuAccessSignalsMhandle;
  void *cpuAccessSignalsGinHandle;
  uint64_t *cpuAccessSignals;
  uint64_t *cpuAccessSignalsDev;
  uint64_t* cpuAccessSignalsHost; // Host buffer to track the expected values of the signals

  // Local flush buffer
  CUmemGenericAllocationHandle flushBufCumemhandle;
  void *flushBufMhandle;
  void *flushBufGinHandle;
  uint64_t *flushBufDev;
};

struct ncclRmaProxyState {
  struct ncclComm *comm;
  ncclGin_t* ncclGin;
  void* ginInstance;
  bool connected;
  int ginType;

  // Physical GIN communicator contexts
  int ginCommCount;
  void* ginComms[NCCL_GIN_MAX_CONNECTIONS];
  ncclNetProperties_t props[NCCL_GIN_MAX_CONNECTIONS];

  // Virtual RMA proxy contexts
  int rmaProxyCtxCount;
  void** rmaProxyCtxs;
  ncclNetDeviceHandle_t** rmaProxyDevHandles;

  int needsProxyProgress;  // Whether we need to progress GIN operations with the proxy
  int ginProgress;         // GIN progress is enabled
  std::thread thread;
  std::mutex mutex;
  std::condition_variable cond;
  ncclResult_t asyncResult;
};

// Proxy-specific function declarations
ncclResult_t ncclRmaProxyPutLaunch(struct ncclComm* comm, struct ncclKernelPlan* plan, cudaStream_t stream);
ncclResult_t ncclRmaProxyWaitLaunch(struct ncclComm* comm, struct ncclKernelPlan* plan, cudaStream_t stream);
ncclResult_t ncclRmaProxyReclaimPlan(struct ncclComm* comm, struct ncclKernelPlan* plan);

// RMA Proxy lifecycle functions
ncclResult_t ncclRmaProxyConnectOnce(struct ncclComm* comm);
ncclResult_t ncclRmaProxyFinalize(struct ncclComm* comm);

// RMA Proxy context management
ncclResult_t ncclRmaProxyCreateContext(struct ncclComm *comm, void *collComm, ncclNetProperties_t props,
                                       void **outRmaProxyCtx, ncclNetDeviceHandle_t **outDevHandle);
ncclResult_t ncclRmaProxyDestroyContext(ncclGin_t* ginComm, void* rmaProxyCtx);
ncclResult_t ncclRmaProxyProgress(ncclGin_t* ncclGin, void* rmaProxyCtx);

// RMA Proxy memory registration
ncclResult_t ncclRmaProxyRegister(struct ncclComm* comm, void* address, size_t size,
                                  void* rmaHostWins[NCCL_GIN_MAX_CONNECTIONS],
                                  ncclGinWindow_t rmaDevWins[NCCL_GIN_MAX_CONNECTIONS]);
ncclResult_t ncclRmaProxyDeregister(struct ncclComm* comm, void* rmaHostWins[NCCL_GIN_MAX_CONNECTIONS]);

// Circular buffer helpers
bool ncclRmaProxyCircularBufFull(struct ncclRmaProxyCtx* ctx, int peer);
bool ncclRmaProxyCircularBufEmpty(struct ncclRmaProxyCtx* ctx, int peer);

// Descriptor destruction
ncclResult_t ncclRmaProxyDestroyDescNonPersistent(struct ncclRmaProxyDesc* desc);
ncclResult_t ncclRmaProxyDestroyDescPersistent(struct ncclComm* comm, struct ncclRmaProxyDesc* desc);

// Progress thread function
void* ncclRmaProxyProgressThread(struct ncclRmaProxyState* rmaProxyState_);
#endif
