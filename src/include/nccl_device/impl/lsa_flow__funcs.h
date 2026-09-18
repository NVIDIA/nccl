/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_DEVICE_LSA_FLOW__FUNCS_H_
#define NCCL_DEVICE_LSA_FLOW__FUNCS_H_

#include "lsa_flow__types.h"
#include "core__funcs.h"
#include "flow__types.h"
#include "ptr__funcs.h"
#include "../utility.h"

#ifdef __CUDACC__
NCCL_DEVICE_INLINE uint64_t ncclLsaFlowLoadSignal(uint64_t* ptr) {
  uint64_t value;
  asm volatile("ld.volatile.global.u64 %0, [%1];" : "=l"(value) : "l"(__cvta_generic_to_global(ptr)) : "memory");
  return value;
}

NCCL_DEVICE_INLINE void ncclLsaFlowStoreSignal(uint64_t* ptr, uint64_t value) {
#if __CUDA_ARCH__ >= 700
  asm volatile("st.relaxed.sys.global.u64 [%0], %1;" ::"l"(__cvta_generic_to_global(ptr)), "l"(value) : "memory");
#else
  asm volatile("st.volatile.global.u64 [%0], %1;" ::"l"(__cvta_generic_to_global(ptr)), "l"(value) : "memory");
#endif
}

NCCL_DEVICE_INLINE void ncclLsaFlowFence() {
#if __CUDA_ARCH__ >= 700
  asm volatile("fence.acq_rel.sys;" ::: "memory");
#else
  asm volatile("membar.sys;" ::: "memory");
#endif
}

template <ncclFlowProtocol Protocol>
NCCL_DEVICE_INLINE ncclLsaFlowConn<Protocol>::ncclLsaFlowConn(ncclDevComm const& comm, ncclSymPtr<char> recvFifo,
                                                              ncclSymPtr<ncclFlowConnState> state_, int peer,
                                                              int nSlots, size_t slotSize, int type, bool recvPost)
  : nSlots(nSlots), slotSize(slotSize), type(type), abortFlag(comm.abortFlag), state(state_.localPtr()), step(0),
    llStep(0), signalValue(0), cache(0) {
  if (type != ncclFlowTypeNone) {
    fifo = type == ncclFlowTypeSend ? recvFifo.peerPtr(peer) : recvFifo.localPtr();

    // Recv -> Local tail, Send -> Local head
    localSignal = &state->signal[type];

    // Recv -> Remote head, Send -> Remote tail
    peerSignal = &state_.peerPtr(peer)->signal[1 - type];

    step = state->step[type];
    llStep = state->llStep[type];
    signalValue = state->signalValue[type];

    if (recvPost) postRecv();
  }
}

template <ncclFlowProtocol Protocol>
NCCL_DEVICE_INLINE void ncclLsaFlowConn<Protocol>::close() {
  state->step[type] = step;
  state->llStep[type] = llStep;
  state->signalValue[type] = signalValue;
}

// Wait for credit and return the next writable slot in the peer's shared buffer.
template <ncclFlowProtocol Protocol>
NCCL_DEVICE_INLINE void* ncclLsaFlowConn<Protocol>::waitSend() {
  void* const ptr = advanceFifoSlot();

  // Wait for credit to send
  uint32_t spins = 0;
  while (cache < step) {
    cache = ncclLsaFlowLoadSignal(localSignal);
    if (nccl::utility::testAbort(abortFlag, spins)) break;
  }
  return ptr;
}

// Simple publishes readiness through the tail counter. LL and LL128 carry readiness in their data flags, and their
// workers have already written directly into the peer's FIFO.
template <ncclFlowProtocol Protocol>
NCCL_DEVICE_INLINE void ncclLsaFlowConn<Protocol>::postSend(size_t /*bytes*/) {
  if (Protocol == ncclFlowProtocolSimple) {
    ncclLsaFlowFence();
    ncclLsaFlowStoreSignal(peerSignal, step);
  }
}

template <ncclFlowProtocol Protocol>
NCCL_DEVICE_INLINE uint64_t ncclLsaFlowConn<Protocol>::advanceStep() {
  uint64_t const fifoStep = Protocol == ncclFlowProtocolLL ? llStep++ : step;
  step++;
  return fifoStep;
}

// Return a pointer into the local shared buffer at the next recv slot; call postRecv() after reading.
template <ncclFlowProtocol Protocol>
NCCL_DEVICE_INLINE void* ncclLsaFlowConn<Protocol>::waitRecv() {
  void* const ptr = advanceFifoSlot();

  // Wait until this slot is ready.
  uint32_t spins = 0;
  while (cache < step) {
    cache = ncclLsaFlowLoadSignal(localSignal);
    if (nccl::utility::testAbort(abortFlag, spins)) break;
  }
  return ptr;
}

template <ncclFlowProtocol Protocol>
NCCL_DEVICE_INLINE void* ncclLsaFlowConn<Protocol>::advanceFifoSlot() {
  uint64_t const fifoStep = advanceStep();
  return fifo + (fifoStep % nSlots) * slotSize;
}

template <ncclFlowProtocol Protocol>
NCCL_DEVICE_INLINE uint64_t ncclLsaFlowConn<Protocol>::stepFlag() const {
  return Protocol == ncclFlowProtocolLL ? NCCL_LL_FLAG(llStep) : step;
}

// Publish accumulated receive progress after half of the FIFO has been consumed.
template <ncclFlowProtocol Protocol>
NCCL_DEVICE_INLINE void ncclLsaFlowConn<Protocol>::postRecv() {
  uint64_t const value = step + nSlots;
  uint64_t const batch = (nSlots + 1) / 2;
  if (signalValue == 0 || (value > signalValue && value - signalValue >= batch)) {
    ncclLsaFlowStoreSignal(peerSignal, value);
    signalValue = value;
  }
}

#endif
#endif // NCCL_DEVICE_LSA_FLOW__FUNCS_H_
