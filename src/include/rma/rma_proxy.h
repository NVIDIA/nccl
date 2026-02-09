/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef _NCCL_RMA_PROXY_H_
#define _NCCL_RMA_PROXY_H_

#include "nccl.h"
#include "nccl_net.h"
#include "nccl_common.h"
#include "gin/gin_host.h"
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
  ncclRmaDescStatePending = 0,
  ncclRmaDescStateInProgress,
} ncclRmaDescState_t;

struct ncclRmaProxyDesc {
  struct ncclRmaProxyDesc *next;

  // Network function descriptor
  uint64_t srcOff;
  void *srcHandle;
  uint64_t dstOff;
  void *dstHandle;
  size_t size;
  int targetRank;
  ncclRmaSignal_t signal;

  // Sequence number for the network operation
  uint64_t seq;

  // State of the network function descriptor
  ncclRmaDescState_t rmaDescState;

  // Request handle for the network operation
  void * request;
};

struct ncclRmaProxyCtx {
  struct ncclComm *comm;

  // GIN context for the RMA proxy context
  void *ginCollComm;
  ncclNetDeviceHandle_t *devHandle;
  ncclNetProperties_t props;

  // Lock-free circular buffer for pending Descs
  size_t queueSize;  // Power of 2 size for pending queue
  struct ncclRmaProxyDesc** pendingQueues;  // Pre-allocated arrays per peer
  uint32_t* pis;  // Producer Indices per peer
  uint32_t* cis;  // Consumer Indices per peer

  // Per-rank rmaProxyInProgressQueues: Descs with issued network operations waiting for completion
  struct ncclIntruQueue<struct ncclRmaProxyDesc, &ncclRmaProxyDesc::next>* rmaProxyInProgressQueues;

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
};

struct ncclRmaProxyState {
  struct ncclComm *comm;
  ncclGin_t* ncclGin;
  void* ginInstance;
  bool connected;
  int ginType;

  // Physical GIN communicator contexts
  int ginCommCount;
  void* ginComms[NCCL_GIN_MAX_CONTEXTS];
  ncclNetProperties_t props[NCCL_GIN_MAX_CONTEXTS];

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
ncclResult_t ncclRmaPutProxy(struct ncclComm* comm, struct ncclKernelPlan* plan, cudaStream_t stream);
ncclResult_t ncclRmaWaitSignalProxy(struct ncclComm* comm, struct ncclKernelPlan* plan, cudaStream_t stream);

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
                                  void* rmaHostWins[NCCL_GIN_MAX_CONTEXTS],
                                  ncclGinWindow_t rmaDevWins[NCCL_GIN_MAX_CONTEXTS]);
ncclResult_t ncclRmaProxyDeregister(struct ncclComm* comm, void* rmaHostWins[NCCL_GIN_MAX_CONTEXTS]);

// Progress thread function
void* ncclRmaProxyProgressThread(struct ncclRmaProxyState* rmaProxyState_);

#endif
