/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_DIAGNOSTICS_P2P_H_
#define NCCL_DIAGNOSTICS_P2P_H_

#include <stdint.h>

#include "nccl.h"
#include <cuda_runtime.h>

struct ncclComm;

struct ncclDiagP2pSlot {
  uint64_t writeValue;
  uint64_t verifyValue;
  uint64_t readPattern;
};

struct ncclDiagP2pRemoteOp {
  struct ncclDiagP2pSlot* remoteSlots;
  int srcRank;
  int dstRank;
  int srcSlot;
};

#if defined(__CUDACC__)
#define NCCL_DIAG_P2P_HD __host__ __device__
#else
#define NCCL_DIAG_P2P_HD
#endif

// Reserve two high bits for the pattern type and keep two 31-bit payload fields. NCCL ranks are non-negative ints, so
// this covers the current possible NCCL rank value range.
static constexpr uint64_t kDiagP2pPatternMask = 0x7fffffffULL;
static constexpr uint64_t kDiagP2pWritePatternTag = 1ULL << 62;
static constexpr uint64_t kDiagP2pReadPatternTag = 2ULL << 62;

static inline NCCL_DIAG_P2P_HD uint64_t ncclDiagP2pWritePattern(int srcRank, int dstRank) {
  return kDiagP2pWritePatternTag | (((uint64_t)(uint32_t)srcRank & kDiagP2pPatternMask) << 31) |
         ((uint64_t)(uint32_t)dstRank & kDiagP2pPatternMask);
}

static inline NCCL_DIAG_P2P_HD uint64_t ncclDiagP2pReadPattern(int dstRank, int srcRank) {
  return kDiagP2pReadPatternTag | (((uint64_t)(uint32_t)dstRank & kDiagP2pPatternMask) << 31) |
         ((uint64_t)(uint32_t)srcRank & kDiagP2pPatternMask);
}

#undef NCCL_DIAG_P2P_HD

ncclResult_t ncclDiagP2pInitSlots(struct ncclDiagP2pSlot* slots, const int* slotRanks, int slotCount, int dstRank,
                                  cudaStream_t stream);
ncclResult_t ncclDiagP2pRemoteWrite(const struct ncclDiagP2pRemoteOp* ops, int opCount, cudaStream_t stream);
ncclResult_t ncclDiagP2pVerifyWrites(struct ncclDiagP2pSlot* slots, int slotCount, cudaStream_t stream);
ncclResult_t ncclDiagP2pRemoteRead(const struct ncclDiagP2pRemoteOp* ops, int opCount, uint64_t* readback,
                                   cudaStream_t stream);

ncclResult_t ncclDiagP2pRun(struct ncclComm* comm);

#endif // NCCL_DIAGNOSTICS_P2P_H_
