/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more information
 *************************************************************************/

#ifndef NCCL_DEVICE_GIN_FLOW__FUNCS_H_
#define NCCL_DEVICE_GIN_FLOW__FUNCS_H_

#include "gin_flow__types.h"
#include "core__funcs.h"
#include "gin__funcs.h"
#include "ptr__funcs.h"

#ifdef __CUDACC__
template <unsigned GinBackendMask, ncclFlowProtocol Protocol>
NCCL_DEVICE_INLINE ncclGinFlowConn<GinBackendMask, Protocol>::ncclGinFlowConn(
  ncclDevComm const& comm, int contextId, ncclTeam team_, int peer_, ncclSymPtr<char> recvFifo,
  ncclSymPtr<char> sendFifo, ncclSymPtr<ncclFlowConnState> state_, ncclGinSignal_t signal0, size_t signalIndex,
  int nSlots_, size_t slotSize_, int type_, bool waitRole, bool recvPost)
  : nSlots(nSlots_), slotSize(slotSize_), type(type_), gin(comm, contextId), team(team_), peer(peer_),
    state(state_.localPtr()), fifo(), peerFifo(), step(0), llStep(0), signalValue(0) {
  if (type != ncclFlowTypeNone) {
    if (waitRole) {
      assert(state->ginContextId_plus_1[type] == 0 || state->ginContextId_plus_1[type] == gin.contextId + 1);
      state->ginContextId_plus_1[type] = gin.contextId + 1;
    }
    if (type == ncclFlowTypeSend) {
      fifo = sendFifo;
      peerFifo = recvFifo;
    } else {
      fifo = recvFifo;
    }

    localSignal = signal0 + 2 * signalIndex + type;
    peerSignal = signal0 + 2 * signalIndex + 1 - type;
    step = state->step[type];
    llStep = state->llStep[type];
    signalValue = state->signalValue[type];

    if (recvPost) postRecv();
  }
}

template <unsigned GinBackendMask, ncclFlowProtocol Protocol>
NCCL_DEVICE_INLINE void ncclGinFlowConn<GinBackendMask, Protocol>::close() {
  state->step[type] = step;
  state->llStep[type] = llStep;
  state->signalValue[type] = signalValue;
}

template <unsigned GinBackendMask, ncclFlowProtocol Protocol>
NCCL_DEVICE_INLINE void* ncclGinFlowConn<GinBackendMask, Protocol>::waitSend() {
  void* const ptr = advanceFifoSlot();
  gin.waitSignal(ncclCoopThread{}, localSignal, step, /*bits=*/64);
  return ptr;
}

template <unsigned GinBackendMask, ncclFlowProtocol Protocol>
NCCL_DEVICE_INLINE void ncclGinFlowConn<GinBackendMask, Protocol>::postSend(size_t bytes) {
  uint64_t const slotStep = Protocol == ncclFlowProtocolLL ? llStep - 1 : step - 1;
  int const slot = slotStep % nSlots;
  size_t const offset = (size_t)slot * slotSize;
  assert(bytes <= slotSize);
  if (Protocol == ncclFlowProtocolSimple) {
    postSendSimple(bytes, fifo + offset, peerFifo + offset);
  } else if (bytes != 0) {
    gin.put(team, peer, peerFifo + offset, fifo + offset, bytes);
  }
}

template <unsigned GinBackendMask, ncclFlowProtocol Protocol>
NCCL_DEVICE_INLINE void ncclGinFlowConn<GinBackendMask, Protocol>::postSendSimple(size_t bytes, ncclSymPtr<char> source,
                                                                                  ncclSymPtr<char> destination) {
  assert(Protocol == ncclFlowProtocolSimple);
  assert(bytes <= slotSize);
  uint64_t const delta = step - signalValue;
  if (bytes != 0) {
    gin.put(team, peer, destination, source, bytes, ncclGin_StrongSignalAdd{peerSignal, delta});
  } else {
    gin.signal(team, peer, ncclGin_StrongSignalAdd{peerSignal, delta});
  }
  signalValue = step;
}

template <unsigned GinBackendMask, ncclFlowProtocol Protocol>
NCCL_DEVICE_INLINE uint64_t ncclGinFlowConn<GinBackendMask, Protocol>::advanceStep() {
  uint64_t const fifoStep = Protocol == ncclFlowProtocolLL ? llStep++ : step;
  step++;
  return fifoStep;
}

template <unsigned GinBackendMask, ncclFlowProtocol Protocol>
NCCL_DEVICE_INLINE void ncclGinFlowConn<GinBackendMask, Protocol>::signal(uint64_t value) {
  if (value == signalValue) return;
  uint64_t const delta = value - signalValue;
  gin.signal(team, peer, ncclGin_StrongSignalAdd{peerSignal, delta});
  signalValue = value;
}

template <unsigned GinBackendMask, ncclFlowProtocol Protocol>
NCCL_DEVICE_INLINE void* ncclGinFlowConn<GinBackendMask, Protocol>::waitRecv() {
  void* const ptr = advanceFifoSlot();
  gin.waitSignal(ncclCoopThread{}, localSignal, step, /*bits=*/64);
  return ptr;
}

template <unsigned GinBackendMask, ncclFlowProtocol Protocol>
NCCL_DEVICE_INLINE void* ncclGinFlowConn<GinBackendMask, Protocol>::advanceFifoSlot() {
  uint64_t const fifoStep = advanceStep();
  return fifo.localPtr() + (fifoStep % nSlots) * slotSize;
}

template <unsigned GinBackendMask, ncclFlowProtocol Protocol>
NCCL_DEVICE_INLINE uint64_t ncclGinFlowConn<GinBackendMask, Protocol>::stepFlag() const {
  return Protocol == ncclFlowProtocolLL ? NCCL_LL_FLAG(llStep) : step;
}

template <unsigned GinBackendMask, ncclFlowProtocol Protocol>
NCCL_DEVICE_INLINE void ncclGinFlowConn<GinBackendMask, Protocol>::postRecv() {
  uint64_t const value = step + nSlots;
  uint64_t const batch = (nSlots + 1) / 2;
  if (signalValue == 0 || (value > signalValue && value - signalValue >= batch)) signal(value);
}

#endif

#endif // NCCL_DEVICE_GIN_FLOW__FUNCS_H_
