/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more information
 *************************************************************************/

#ifndef NCCL_DEVICE_GIN_FLOW_H_
#define NCCL_DEVICE_GIN_FLOW_H_

#include "gin.h"
#include "impl/flow__types.h"
#include "ptr.h"

#ifdef __CUDACC__
template <unsigned GinBackendMask = NCCL_GIN_BACKEND_MASK_ALL, ncclFlowProtocol Protocol = ncclFlowProtocolSimple>
struct ncclGinFlowConn {
  const int nSlots;
  const size_t slotSize;
  const int type;
  ncclGin_BackendMask<GinBackendMask> gin;
  ncclTeam team;
  int peer;
  ncclFlowConnState* state;
  ncclSymPtr<char> fifo;
  ncclSymPtr<char> peerFifo;
  ncclGinSignal_t localSignal;
  ncclGinSignal_t peerSignal;
  // Transport credits are shared across protocols; LL slots and flags follow their own wrapping sequence.
  uint64_t step;
  uint32_t llStep;
  uint64_t signalValue;
  NCCL_DEVICE_INLINE ncclGinFlowConn(ncclDevComm const& comm, int contextId, ncclTeam team, int peer,
                                     ncclSymPtr<char> recvFifo, ncclSymPtr<char> sendFifo,
                                     ncclSymPtr<ncclFlowConnState> state, ncclGinSignal_t signal0, size_t signalIndex,
                                     int nSlots, size_t slotSize, int type, bool waitRole, bool recvPost);

  NCCL_DEVICE_INLINE void close();
  NCCL_DEVICE_INLINE void* waitSend();
  NCCL_DEVICE_INLINE void postSend(size_t bytes);
  NCCL_DEVICE_INLINE void* waitRecv();
  NCCL_DEVICE_INLINE void postRecv();

  NCCL_DEVICE_INLINE uint64_t advanceStep();
  NCCL_DEVICE_INLINE void postSendSimple(size_t bytes, ncclSymPtr<char> source, ncclSymPtr<char> destination);
  NCCL_DEVICE_INLINE void signal(uint64_t value);
  NCCL_DEVICE_INLINE void* advanceFifoSlot();
  NCCL_DEVICE_INLINE uint64_t stepFlag() const;
};

#endif

#endif // NCCL_DEVICE_GIN_FLOW_H_
