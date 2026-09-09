/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef _NCCL_RMA_CE_H_
#define _NCCL_RMA_CE_H_

#include "nccl.h"
#include "nccl_common.h"
#include "dev_runtime.h"

struct ncclComm;
struct ncclRmaArgs;

struct ncclRmaCeInitTask {
  struct ncclRmaCeInitTask* next;
  struct ncclComm* comm;
};

struct ncclRmaCeCtx {
  struct ncclComm* comm;

  // CE only targets intra-node (LSA) peers, so all CE signal state is indexed by LSA-local rank.
  // Host lsaSize * numRmaSig sequence numbers for non-graph signal operations.
  uint64_t* signalOpSeqs;
  // Host buffer lsaSize * numRmaSig to track the expected values of the non-graph signals
  uint64_t* signalsHost;
  // Device staging slots for non-graph signal values. Indexed by signal op
  // within the current CE batch chunk, with capacity comm->nRanks.
  uint64_t* signalOpSeqsDev;

  // Single symmetric window for all signal and ack memory.
  // signalSlots = lsaSize * numRmaSig
  // Slot within each region is sigIdx * lsaSize + lsaRank.
  // Layout (all uint64_t slots):
  //   [0 .. signalSlots-1]                 non-graph per-(sigIdx, rank) signals
  //   [signalSlots .. 2*signalSlots-1]     graph per-(sigIdx, rank) signals
  //   [2*signalSlots .. 3*signalSlots-1]   graph per-(sigIdx, rank) ack flags
  // Total: 3 * signalSlots * sizeof(uint64_t)
  struct ncclDevrWindow* signalsWin;
  uint64_t* signalsDev;       // non-graph per-(sigIdx, rank) signals
  uint64_t* graphSignalsDev;  // graph per-(sigIdx, rank) signals
  uint64_t* graphAckDev;      // graph per-(sigIdx, rank) ack flags
  size_t signalOffset;        // byte offset of non-graph signals
  size_t graphSignalOffset;   // byte offset of graph signals
  size_t graphAckOffset;      // byte offset of graph ack flags

  // Device-resident constants for graph-safe D2D signal/ack writes
  uint64_t* signalConstDev;
  uint64_t* signalConstOneDev;
  uint64_t* signalConstZeroDev;
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
ncclResult_t ncclRmaCePutLaunch(struct ncclComm* comm, struct ncclKernelPlan* plan, cudaStream_t stream);
ncclResult_t ncclRmaCeWaitLaunch(struct ncclComm* comm, struct ncclKernelPlan* plan, cudaStream_t stream);
#endif
