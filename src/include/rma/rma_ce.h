/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef _NCCL_RMA_CE_H_
#define _NCCL_RMA_CE_H_

#include "nccl.h"
#include "nccl_common.h"
#include "dev_runtime.h"

struct ncclComm;
struct ncclRmaArgs;

struct ncclRmaCeInitTask {
  struct ncclRmaCeInitTask *next;
  struct ncclComm* comm;
};

struct ncclRmaCeCtx {
  struct ncclComm *comm;

  // Per-rank sequence number for the signal operations
  uint64_t* signalOpSeqs;

  // Signal memory layout and management
  // Each RMA context allocates a signal buffer with the following layout:
  // - Offsets [0 to nRanks*8-1]: per-rank distinct signals (8 bytes per rank)
  // - Offset [nRanks*8]: shared aggregate signal counter (8 bytes)
  // Total signal buffer size: (nRanks + 1) * 8 bytes
  struct ncclDevrWindow* signalsWin;
  uint64_t *signalsDev;
  uint64_t* signalsHost; // Host buffer to track the expected values of the signals
};


struct ncclRmaCeState {
  bool initialized;
  int rmaCeCtxCount;
  void** rmaCeCtxs;
  cudaStream_t ceStream;
  cudaEvent_t ceEvent;
};

// CE-specific function declarations
ncclResult_t ncclRmaCeInit(struct ncclComm* comm);
ncclResult_t ncclRmaCeFinalize(struct ncclComm* comm);
ncclResult_t ncclRmaPutCe(struct ncclComm* comm, struct ncclKernelPlan* plan, cudaStream_t stream);
ncclResult_t ncclRmaWaitSignalCe(struct ncclComm* comm, struct ncclKernelPlan* plan, cudaStream_t stream);
#endif
