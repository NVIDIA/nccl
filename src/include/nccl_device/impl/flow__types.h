/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more information
 *************************************************************************/

#ifndef NCCL_DEVICE_FLOW__TYPES_H_
#define NCCL_DEVICE_FLOW__TYPES_H_

#include "core__types.h"

// Maximum number of simultaneous peers in either direction. A signaling warp can service both sets concurrently.
#define NCCL_FLOW_MAX_PEERS 4
// Maximum number of independently synchronized parts in one Flow processor.
#define NCCL_FLOW_MAX_PARTS 4

#if __cplusplus
// Persistent state shared by the send and receive directions of one Flow topology slot. LSA uses signal directly;
// GIN uses an allocated signal pair and keeps signalValue so additive signal updates can resume across operations.
struct alignas(64) ncclFlowConnState {
  uint64_t signal[2];
  uint64_t step[2];
  uint64_t signalValue[2];
  uint32_t ginContextId_plus_1[2];
  uint32_t reserved[2];
};
static_assert(sizeof(ncclFlowConnState) == 64, "ncclFlowConnState must occupy one cache line");
#endif

#ifdef __CUDACC__

constexpr int ncclFlowTypeRecv = 0;
constexpr int ncclFlowTypeSend = 1;
constexpr int ncclFlowTypeNone = -1;

struct ncclFlowPartShmem {
  void* sendBuf[NCCL_FLOW_MAX_PEERS];
  void* recvBuf[NCCL_FLOW_MAX_PEERS];
};

struct ncclFlowShmem {
  ncclFlowPartShmem parts[NCCL_FLOW_MAX_PARTS];
};

static_assert(2 * NCCL_FLOW_MAX_PARTS < 16, "ncclFlowProcessor needs two CUDA named barriers per part");

#endif

#endif // NCCL_DEVICE_FLOW__TYPES_H_
