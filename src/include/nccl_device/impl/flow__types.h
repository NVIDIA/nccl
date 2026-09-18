/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more information
 *************************************************************************/

#ifndef NCCL_DEVICE_FLOW__TYPES_H_
#define NCCL_DEVICE_FLOW__TYPES_H_

#include "core__types.h"
#include "ll__types.h"
#include "ll128__types.h"

// Maximum number of simultaneous peers in either direction. A direction uses either a warp or half of a shared warp.
#define NCCL_FLOW_MAX_PEERS 4
// Maximum number of independently synchronized parts in one Flow processor.
#define NCCL_FLOW_MAX_PARTS 4
// Most protocols use separate send and receive signaling warps. LL128 shares
// one warp between the two directions to leave more threads for processing.
#define NCCL_FLOW_SIGNAL_WARPS_PER_PART 2
#define NCCL_FLOW_LL128_SIGNAL_WARPS_PER_PART 1

#if __cplusplus
// Persistent state shared by the protocols and send/receive directions of one Flow topology slot. step tracks the
// shared transport credits and signals. LL uses a separate 32-bit step for FIFO-slot selection and embedded flags so
// Simple operations cannot skip an LL cleanup epoch. signalValue tracks the last readiness or credit value published,
// allowing additive GIN signals and batched receive credits to resume across operations.
struct alignas(64) ncclFlowConnState {
  uint64_t signal[2];
  uint64_t step[2];
  uint64_t signalValue[2];
  uint32_t ginContextId_plus_1[2];
  uint32_t llStep[2];
};
static_assert(sizeof(ncclFlowConnState) == 64, "ncclFlowConnState must occupy one cache line");
#endif

#ifdef __CUDACC__

constexpr int ncclFlowTypeRecv = 0;
constexpr int ncclFlowTypeSend = 1;
constexpr int ncclFlowTypeNone = -1;

enum ncclFlowProtocol {
  ncclFlowProtocolSimple,
  ncclFlowProtocolLL,
  ncclFlowProtocolLL128,
  ncclFlowProtocolCount
};

constexpr NCCL_DEVICE_INLINE int ncclFlowGetSignalWarpsPerPart(ncclFlowProtocol protocol) {
  return protocol == ncclFlowProtocolLL128 ? NCCL_FLOW_LL128_SIGNAL_WARPS_PER_PART : NCCL_FLOW_SIGNAL_WARPS_PER_PART;
}

NCCL_DEVICE_INLINE int ncclFlowPartWorkerWarps(int nWorkerWarps, int nParts, int part, int const* partWorkerWarps) {
  return partWorkerWarps == nullptr ? (part + 1) * nWorkerWarps / nParts - part * nWorkerWarps / nParts :
                                      partWorkerWarps[part];
}

NCCL_DEVICE_INLINE int ncclFlowPartWorkerWarp0(int nWorkerWarps, int nParts, int part, int const* partWorkerWarps,
                                               int signalWarpsPerPart = NCCL_FLOW_SIGNAL_WARPS_PER_PART) {
  int warp0 = part * signalWarpsPerPart;
  if (partWorkerWarps == nullptr) return warp0 + part * nWorkerWarps / nParts;
  for (int p = 0; p < part; p++) warp0 += partWorkerWarps[p];
  return warp0;
}

struct ncclFlowPartShmem {
  void* sendBuf[NCCL_FLOW_MAX_PEERS];
  void* recvBuf[NCCL_FLOW_MAX_PEERS];
  uint64_t sendFlag[NCCL_FLOW_MAX_PEERS];
  uint64_t recvFlag[NCCL_FLOW_MAX_PEERS];
  bool sendUsesFifo[NCCL_FLOW_MAX_PEERS];
  bool recvUsesFifo[NCCL_FLOW_MAX_PEERS];
};

struct ncclFlowShmem {
  ncclFlowPartShmem parts[NCCL_FLOW_MAX_PARTS];
};

static_assert(2 * NCCL_FLOW_MAX_PARTS < 16, "ncclFlowProcessor needs two CUDA named barriers per part");

#endif

#endif // NCCL_DEVICE_FLOW__TYPES_H_
