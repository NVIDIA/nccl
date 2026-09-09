/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "../p2p.h"
#include "checks.h"

static constexpr int kDiagP2pThreads = 128;

__global__ void diagP2pInitSlotsKernel(struct ncclDiagP2pSlot* slots, const int* slotRanks, int slotCount,
                                       int dstRank) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= slotCount) return;
  slots[idx].writeValue = 0;
  slots[idx].verifyValue = 0;
  slots[idx].readPattern = ncclDiagP2pReadPattern(dstRank, slotRanks[idx]);
}

__global__ void diagP2pRemoteWriteKernel(const struct ncclDiagP2pRemoteOp* ops, int opCount) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= opCount) return;
  const struct ncclDiagP2pRemoteOp op = ops[idx];
  op.remoteSlots[op.srcSlot].writeValue = ncclDiagP2pWritePattern(op.srcRank, op.dstRank);
}

__global__ void diagP2pVerifyWritesKernel(struct ncclDiagP2pSlot* slots, int slotCount) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= slotCount) return;
  slots[idx].verifyValue = slots[idx].writeValue;
}

__global__ void diagP2pRemoteReadKernel(const struct ncclDiagP2pRemoteOp* ops, int opCount, uint64_t* readback) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= opCount) return;
  const struct ncclDiagP2pRemoteOp op = ops[idx];
  readback[idx] = op.remoteSlots[op.srcSlot].readPattern;
}

static int diagP2pBlocks(int count) {
  return (count + kDiagP2pThreads - 1) / kDiagP2pThreads;
}

ncclResult_t ncclDiagP2pInitSlots(struct ncclDiagP2pSlot* slots, const int* slotRanks, int slotCount, int dstRank,
                                  cudaStream_t stream) {
  if (slotCount <= 0) return ncclSuccess;
  diagP2pInitSlotsKernel<<<diagP2pBlocks(slotCount), kDiagP2pThreads, 0, stream>>>(slots, slotRanks, slotCount,
                                                                                   dstRank);
  CUDACHECK(cudaGetLastError());
  return ncclSuccess;
}

ncclResult_t ncclDiagP2pRemoteWrite(const struct ncclDiagP2pRemoteOp* ops, int opCount, cudaStream_t stream) {
  if (opCount <= 0) return ncclSuccess;
  diagP2pRemoteWriteKernel<<<diagP2pBlocks(opCount), kDiagP2pThreads, 0, stream>>>(ops, opCount);
  CUDACHECK(cudaGetLastError());
  return ncclSuccess;
}

ncclResult_t ncclDiagP2pVerifyWrites(struct ncclDiagP2pSlot* slots, int slotCount, cudaStream_t stream) {
  if (slotCount <= 0) return ncclSuccess;
  diagP2pVerifyWritesKernel<<<diagP2pBlocks(slotCount), kDiagP2pThreads, 0, stream>>>(slots, slotCount);
  CUDACHECK(cudaGetLastError());
  return ncclSuccess;
}

ncclResult_t ncclDiagP2pRemoteRead(const struct ncclDiagP2pRemoteOp* ops, int opCount, uint64_t* readback,
                                   cudaStream_t stream) {
  if (opCount <= 0) return ncclSuccess;
  diagP2pRemoteReadKernel<<<diagP2pBlocks(opCount), kDiagP2pThreads, 0, stream>>>(ops, opCount, readback);
  CUDACHECK(cudaGetLastError());
  return ncclSuccess;
}
