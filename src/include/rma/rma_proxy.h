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
#include "nccl_rma.h"
#include "alloc.h"
#include <thread>
#include <mutex>
#include <condition_variable>

struct ncclComm;
struct ncclRmaArgs;
struct ncclKernelPlan;
struct ncclDevrWindow;

// Signal mode for put-signal operations.
typedef enum {
  NCCL_SIGNAL_NONE = 0,        // No signaling
  NCCL_SIGNAL = 1              // Default signal operation
} ncclSignalMode_t;

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
  ncclRmaDescTypePutSignalGroup,
} ncclRmaDescType_t;

struct ncclRmaPutSignalOp {
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

struct ncclRmaWaitSignalOp {
  int npeers;
  int* waitPeers;
  int* waitSignals;
  // Local flush in graph mode
  int needFlush;
};

struct ncclRmaPutSignalGroupOp {
  int nOps;
  struct ncclRmaPutSignalOp* ops;
  int nIssued;
  int nCompleted;
};

struct ncclRmaProxyDesc {
  struct ncclRmaProxyDesc *next;
  ncclRmaDescType_t rmaDescType;
  ncclRmaDescState_t rmaDescState;

  union {
    struct ncclRmaPutSignalOp putSignal;
    struct ncclRmaWaitSignalOp waitSignal;
    struct ncclRmaPutSignalGroupOp putSignalGroup;
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
  void *rmaCollComm;
  void *rmaCtx;
  //ncclNetDeviceHandle_t *devHandle;
  ncclNetProperties_t props;

  //---------Non-graph descriptor queues and synchronization---------

  // Lock-free circular buffer for pending Descs
  size_t queueSize;  // Power of 2 size for pending queue
  struct ncclRmaProxyDesc** circularBuffers;  // Lock-free circular buffer per peer
  uint32_t* pis;  // Producer Indices per peer
  uint32_t* cis;  // Consumer Indices per peer

  // Per-rank inProgressQueues: Descs with issued network operations waiting for completion
  struct ncclIntruQueue<struct ncclRmaProxyDesc, &ncclRmaProxyDesc::next>* inProgressQueues;

  // Per-target-rank request credits. Each target rank maps to one RMA send comm request pool.
  uint32_t maxInflightRequests;
  uint32_t* inflightRequests;

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
  uint64_t *signalsDev;
  uint64_t* signalsHost; // Host buffer to track the expected values of the signals

  //---------Graph descriptor queues and synchronization---------

  // Per-rank persistent descriptor queue: Descs from all live graphs
  struct ncclIntruQueue<struct ncclRmaProxyDesc, &ncclRmaProxyDesc::next>* persistentQueues;

  // CPU-accessible signal is required as proxy needs to poll on the signal values
  void *cpuAccessSignalsGdrHandle;
  void *cpuAccessSignalsMhandle;
  uint64_t *cpuAccessSignals;
  uint64_t *cpuAccessSignalsDev;
  uint64_t* cpuAccessSignalsHost; // Host buffer to track the expected values of the signals

  // Local flush buffer
  CUmemGenericAllocationHandle flushBufCumemhandle;
  void *flushBufMhandle;
  uint64_t *flushBufDev;
};

struct ncclRmaProxyState {
  struct ncclComm *comm;
  ncclRma_t* ncclRma;
  int rmaVersion;
  void* rmaInstance;
  bool connected;
  int rmaType;

  // Physical GIN communicator contexts
  int rmaCommCount;
  void* rmaComms[NCCL_GIN_MAX_CONNECTIONS];
  ncclNetProperties_t props[NCCL_GIN_MAX_CONNECTIONS];

  // Virtual RMA proxy contexts
  int rmaProxyCtxCount;
  void** rmaProxyCtxs;
  int rmaProgress;         // RMA progress is enabled
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
ncclResult_t ncclRmaProxyDestroyContext(ncclRma_t* rmaComm, void* rmaProxyCtx);
ncclResult_t ncclRmaProxyProgress(ncclRma_t* ncclRma, void* rmaProxyCtx);
void* ncclRmaProxyProgressThread(struct ncclRmaProxyState* rmaProxyState_);

// RMA Proxy memory registration
ncclResult_t ncclRmaProxyRegister(struct ncclComm* comm, void* address, size_t size,
                                  void* rmaHostWins[NCCL_GIN_MAX_CONNECTIONS]);
ncclResult_t ncclRmaProxyDeregister(struct ncclComm* comm, void* rmaHostWins[NCCL_GIN_MAX_CONNECTIONS]);

// Circular buffer helpers
bool ncclRmaProxyCircularBufFull(struct ncclRmaProxyCtx* ctx, int peer);
bool ncclRmaProxyCircularBufEmpty(struct ncclRmaProxyCtx* ctx, int peer);

// Returns true if the queue this descriptor would enqueue into is full.
bool ncclRmaProxyEnqueueFull(struct ncclRmaProxyCtx* ctx,
                             const struct ncclRmaProxyDesc* desc);

// ============================================================================
// Descriptor API: 4-step protocol
// ============================================================================
//   1. BuildDesc(...desc)          allocate desc, populate fields
//   2. {Put,PutGroup,Wait}Params   snapshot fields into stream-batch params
//   3. EnqueueDesc(ctx, &desc)     transfer ownership (queue or destroy);
//   4. ncclCuStreamBatchMemOp(...) issue memops on the user stream
//
// Step 2 must precede step 3: EnqueueDesc may free the desc (non-persistent
// wait), so any field read happens in step 2.
//
// ============================================================================

// ---- Descriptor builders ----

// Helper to build a single put-signal op (used by both single put and group
// put builders).
ncclResult_t ncclRmaProxyPutBuildOp(
    struct ncclComm* comm, struct ncclRmaProxyCtx* rmaProxyCtx,
    int ctx, bool persistent,
    struct ncclDevrWindow* srcWin, size_t srcOff,
    struct ncclDevrWindow* peerWin, size_t peerOff,
    size_t size, int peer,
    ncclSignalMode_t signalMode,
    struct ncclRmaPutSignalOp* op);

// Build a single put descriptor.
ncclResult_t ncclRmaProxyPutBuildDesc(
    struct ncclComm* comm, struct ncclRmaProxyCtx* rmaProxyCtx,
    struct ncclKernelPlan* plan,
    struct ncclDevrWindow* srcWinHost, size_t srcWinOffset,
    struct ncclDevrWindow* peerWinHost, size_t peerWinOffset,
    size_t size, int peer, int ctx,
    ncclSignalMode_t signalMode,
    struct ncclRmaProxyDesc* desc);

// Build a put-signal-group descriptor over an array of pre-filled ops.
// Takes ownership of *ops and nulls the caller's slot on success.
ncclResult_t ncclRmaProxyPutGroupBuildDesc(
    struct ncclComm* comm, struct ncclRmaProxyCtx* rmaProxyCtx,
    struct ncclKernelPlan* plan,
    int nOps, struct ncclRmaPutSignalOp** ops,
    int ctx,
    struct ncclRmaProxyDesc* desc);

// Build a wait-signal descriptor.
// Takes ownership of caller-allocated peers/nsignals arrays.
ncclResult_t ncclRmaProxyWaitBuildDesc(
    struct ncclComm* comm, struct ncclRmaProxyCtx* rmaProxyCtx,
    struct ncclKernelPlan* plan,
    int npeers, int** peers, int** nsignals,
    struct ncclRmaProxyDesc* desc);

// Stream-batch memop param builders for put descriptors.
int ncclRmaProxyPutStartNumOps(bool persistent);
ncclResult_t ncclRmaProxyPutStartParams(struct ncclRmaProxyDesc* desc,
                                        CUstreamBatchMemOpParams* params);
int ncclRmaProxyPutDoneNumOps(bool persistent);
ncclResult_t ncclRmaProxyPutDoneParams(struct ncclRmaProxyDesc* desc,
                                       CUstreamBatchMemOpParams* params);

// Stream-batch memop param builders for put-signal-group descriptors.
int ncclRmaProxyPutGroupStartNumOps(bool persistent);
ncclResult_t ncclRmaProxyPutGroupStartParams(struct ncclRmaProxyDesc* desc,
                                             CUstreamBatchMemOpParams* params);
int ncclRmaProxyPutGroupDoneNumOps(bool persistent);
ncclResult_t ncclRmaProxyPutGroupDoneParams(struct ncclRmaProxyDesc* desc,
                                            CUstreamBatchMemOpParams* params);

// Stream-batch memop param builder for a wait descriptor.
int ncclRmaProxyWaitNumStreamOps(const struct ncclRmaProxyDesc* desc);
ncclResult_t ncclRmaProxyWaitParams(struct ncclRmaProxyCtx* rmaProxyCtx,
                                    struct ncclRmaProxyDesc* desc,
                                    CUstreamBatchMemOpParams* params);

// Descriptor enqueue dispatcher.
ncclResult_t ncclRmaProxyEnqueueDesc(struct ncclRmaProxyCtx* rmaProxyCtx,
                                     struct ncclRmaProxyDesc** desc);

// Descriptor destruction. Takes desc** and nulls *desc after free.
ncclResult_t ncclRmaProxyDestroyDesc(struct ncclComm* comm, struct ncclRmaProxyDesc** desc);
#endif
