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
#include "../utility.h"

#ifdef __CUDACC__
template <unsigned GinBackendMask, ncclFlowProtocol Protocol>
NCCL_DEVICE_INLINE ncclGinFlowConn<GinBackendMask, Protocol>::ncclGinFlowConn(
  ncclDevComm const& comm, int contextId, ncclTeam team_, int peer_, ncclSymPtr<char> recvFifo,
  ncclSymPtr<char> sendFifo, ncclSymPtr<ncclFlowConnState> state_, ncclGinSignal_t signal0, size_t signalIndex,
  int nSlices_, size_t stepSize_, int type_, bool waitRole, bool recvPost)
  : nSlices(nSlices_), sliceSteps(ncclFlowFifoSlots / nSlices_), stepSize(stepSize_), type(type_), gin(comm, contextId),
    team(team_), peer(peer_), state(state_.localPtr()), fifo(), peerFifo(), step(0), llStep(0), signalValue(0),
    flushOnClose(false) {
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
    // Resume on a slice boundary if the preceding protocol used a different FIFO depth.
    uint64_t const savedStep = state->step[type];
    step = nccl::utility::alignUp(savedStep, (uint64_t)sliceSteps);
    llStep = state->llStep[type];
    signalValue = state->signalValue[type];

    // Publish skipped credits even when below the usual batch size.
    if (recvPost) postRecv(step != savedStep);
  }
}

template <unsigned GinBackendMask, ncclFlowProtocol Protocol>
NCCL_DEVICE_INLINE void ncclGinFlowConn<GinBackendMask, Protocol>::close() {
  if (flushOnClose) gin.flush(ncclCoopThread{});
  state->step[type] = step;
  state->llStep[type] = llStep;
  state->signalValue[type] = signalValue;
}

template <unsigned GinBackendMask, ncclFlowProtocol Protocol>
NCCL_DEVICE_INLINE void* ncclGinFlowConn<GinBackendMask, Protocol>::waitSend() {
  void* const ptr = advanceSlice();
  gin.waitSignal(ncclCoopThread{}, localSignal, step, /*bits=*/64);
  return ptr;
}

template <unsigned GinBackendMask, ncclFlowProtocol Protocol>
NCCL_DEVICE_INLINE void ncclGinFlowConn<GinBackendMask, Protocol>::postSend(size_t bytes) {
  size_t const offset = Protocol == ncclFlowProtocolLL ? ((llStep - 1) % nSlices) * sliceSteps * stepSize :
                                                         ((step - sliceSteps) % ncclFlowFifoSlots) * stepSize;
  assert(bytes <= stepSize * sliceSteps);
  if (Protocol == ncclFlowProtocolSimple) {
    postSendSimple(bytes, fifo + offset, peerFifo + offset);
  } else if (bytes != 0) {
    gin.put(team, peer, peerFifo + offset, fifo + offset, bytes);
  }
}

template <unsigned GinBackendMask, ncclFlowProtocol Protocol>
NCCL_DEVICE_INLINE void ncclGinFlowConn<GinBackendMask, Protocol>::postSendFrom(size_t bytes, ncclSymPtr<char> source) {
  size_t const offset = ((step - sliceSteps) % ncclFlowFifoSlots) * stepSize;
  if (bytes != 0) flushOnClose = true;
  postSendSimple(bytes, source, peerFifo + offset);
}

template <unsigned GinBackendMask, ncclFlowProtocol Protocol>
NCCL_DEVICE_INLINE void ncclGinFlowConn<GinBackendMask, Protocol>::postSendDirect(
  size_t bytes, ncclSymPtr<char> peerOutput, ncclSymPtr<char> source) {
  bool const externalSource = source.window != nullptr;
  if (!externalSource) {
    size_t const offset = ((step - sliceSteps) % ncclFlowFifoSlots) * stepSize;
    source = fifo + offset;
  }
  if (bytes != 0 && externalSource) flushOnClose = true;
  postSendSimple(bytes, source, peerOutput);
}

template <unsigned GinBackendMask, ncclFlowProtocol Protocol>
NCCL_DEVICE_INLINE void ncclGinFlowConn<GinBackendMask, Protocol>::postSendSimple(size_t bytes, ncclSymPtr<char> source,
                                                                                  ncclSymPtr<char> destination) {
  assert(Protocol == ncclFlowProtocolSimple);
  assert(bytes <= stepSize * sliceSteps);
  uint64_t const delta = step - signalValue;
  if (bytes != 0) {
    gin.put(team, peer, destination, source, bytes, ncclGin_StrongSignalAdd{peerSignal, delta});
  } else {
    gin.signal(team, peer, ncclGin_StrongSignalAdd{peerSignal, delta});
  }
  signalValue = step;
}

template <unsigned GinBackendMask, ncclFlowProtocol Protocol>
NCCL_DEVICE_INLINE void ncclGinFlowConn<GinBackendMask, Protocol>::advanceStep() {
  if (Protocol == ncclFlowProtocolLL) llStep++;
  step += sliceSteps;
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
  void* const ptr = advanceSlice();
  gin.waitSignal(ncclCoopThread{}, localSignal, step, /*bits=*/64);
  return ptr;
}

template <unsigned GinBackendMask, ncclFlowProtocol Protocol>
NCCL_DEVICE_INLINE void* ncclGinFlowConn<GinBackendMask, Protocol>::advanceSlice() {
  size_t const offset =
    Protocol == ncclFlowProtocolLL ? (llStep % nSlices) * sliceSteps * stepSize : (step % ncclFlowFifoSlots) * stepSize;
  advanceStep();
  return fifo.localPtr() + offset;
}

template <unsigned GinBackendMask, ncclFlowProtocol Protocol>
NCCL_DEVICE_INLINE uint64_t ncclGinFlowConn<GinBackendMask, Protocol>::stepFlag() const {
  return Protocol == ncclFlowProtocolLL ? NCCL_LL_FLAG(llStep) : step;
}

template <unsigned GinBackendMask, ncclFlowProtocol Protocol>
NCCL_DEVICE_INLINE void ncclGinFlowConn<GinBackendMask, Protocol>::postRecv(bool force) {
  uint64_t const value = step + ncclFlowFifoSlots;
  uint64_t const batch = sliceSteps * ((nSlices + 1) / 2);
  if (signalValue == 0 || (value > signalValue && (force || value - signalValue >= batch))) signal(value);
}

#endif

#endif // NCCL_DEVICE_GIN_FLOW__FUNCS_H_
