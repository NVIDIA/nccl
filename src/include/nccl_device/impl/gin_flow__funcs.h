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
template <unsigned GinBackendMask>
NCCL_DEVICE_INLINE ncclGinFlowConn<GinBackendMask>::ncclGinFlowConn(
  ncclDevComm const& comm, int contextId, ncclTeam team_, int peer_, ncclSymPtr<char> recvFifo,
  ncclSymPtr<char> sendFifo, ncclSymPtr<ncclFlowConnState> state_, ncclGinSignal_t signal0, size_t signalIndex,
  int nSlots_, size_t slotSize_, int type_, bool waitRole, bool recvPost)
  : nSlots(nSlots_), slotSize(slotSize_), type(type_), gin(comm, contextId), team(team_), peer(peer_),
    state(state_.localPtr()), fifo(), peerFifo(), step(0), signalValue(0) {
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
    signalValue = state->signalValue[type];

    if (recvPost) postRecv();
  }
}

template <unsigned GinBackendMask>
NCCL_DEVICE_INLINE void ncclGinFlowConn<GinBackendMask>::close() {
  state->step[type] = step;
  state->signalValue[type] = signalValue;
}

template <unsigned GinBackendMask>
NCCL_DEVICE_INLINE void* ncclGinFlowConn<GinBackendMask>::waitSend() {
  int const slot = (step++) % nSlots;
  gin.waitSignal(ncclCoopThread{}, localSignal, step, /*bits=*/64);
  return fifo.localPtr() + slot * slotSize;
}

template <unsigned GinBackendMask>
NCCL_DEVICE_INLINE void ncclGinFlowConn<GinBackendMask>::postSend(size_t bytes) {
  int const slot = (step - 1) % nSlots;
  size_t const offset = (size_t)slot * slotSize;
  assert(bytes <= slotSize);
  postSendSimple(bytes, fifo + offset, peerFifo + offset);
}

template <unsigned GinBackendMask>
NCCL_DEVICE_INLINE void ncclGinFlowConn<GinBackendMask>::postSendSimple(size_t bytes, ncclSymPtr<char> source,
                                                                        ncclSymPtr<char> destination) {
  assert(bytes <= slotSize);
  uint64_t const delta = step - signalValue;
  if (bytes != 0) {
    gin.put(team, peer, destination, source, bytes, ncclGin_StrongSignalAdd{peerSignal, delta});
  } else {
    gin.signal(team, peer, ncclGin_StrongSignalAdd{peerSignal, delta});
  }
  signalValue = step;
}

template <unsigned GinBackendMask>
NCCL_DEVICE_INLINE void ncclGinFlowConn<GinBackendMask>::advanceStep() {
  step++;
}

template <unsigned GinBackendMask>
NCCL_DEVICE_INLINE void ncclGinFlowConn<GinBackendMask>::signal(uint64_t value) {
  if (value == signalValue) return;
  uint64_t const delta = value - signalValue;
  gin.signal(team, peer, ncclGin_StrongSignalAdd{peerSignal, delta});
  signalValue = value;
}

template <unsigned GinBackendMask>
NCCL_DEVICE_INLINE void* ncclGinFlowConn<GinBackendMask>::waitRecv() {
  int const slot = (step++) % nSlots;
  gin.waitSignal(ncclCoopThread{}, localSignal, step, /*bits=*/64);
  return fifo.localPtr() + slot * slotSize;
}

template <unsigned GinBackendMask>
NCCL_DEVICE_INLINE void ncclGinFlowConn<GinBackendMask>::postRecv() {
  uint64_t const value = step + nSlots;
  uint64_t const batch = (nSlots + 1) / 2;
  if (signalValue == 0 || (value > signalValue && value - signalValue >= batch)) signal(value);
}

#endif

#endif // NCCL_DEVICE_GIN_FLOW__FUNCS_H_
