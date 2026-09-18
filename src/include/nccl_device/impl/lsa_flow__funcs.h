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

NCCL_DEVICE_INLINE ncclLsaFlowConn::ncclLsaFlowConn(ncclDevComm const& comm, ncclSymPtr<char> recvFifo,
                                                    ncclSymPtr<ncclFlowConnState> state_, int peer, int nSlots,
                                                    size_t slotSize, int type, bool recvPost)
  : nSlots(nSlots), slotSize(slotSize), type(type), abortFlag(comm.abortFlag), state(state_.localPtr()), step(0),
    signalValue(0), cache(0) {
  if (type != ncclFlowTypeNone) {
    fifo = type == ncclFlowTypeSend ? recvFifo.peerPtr(peer) : recvFifo.localPtr();

    // Recv -> Local tail, Send -> Local head
    localSignal = &state->signal[type];

    // Recv -> Remote head, Send -> Remote tail
    peerSignal = &state_.peerPtr(peer)->signal[1 - type];

    step = state->step[type];
    signalValue = state->signalValue[type];

    if (recvPost) postRecv();
  }
}

NCCL_DEVICE_INLINE void ncclLsaFlowConn::close() {
  state->step[type] = step;
  state->signalValue[type] = signalValue;
}

// Return a pointer into peer's shared buffer at the next send slot; call postSend() after writing.
NCCL_DEVICE_INLINE void* ncclLsaFlowConn::waitSend() {
  // Compute slot in shared buffer
  int slot = (step++) % nSlots;
  void* ptr = fifo + slot * slotSize;

  // Wait for credit to send
  uint32_t spins = 0;
  while (cache < step) {
    cache = ncclLsaFlowLoadSignal(localSignal);
    if (nccl::utility::testAbort(abortFlag, spins)) break;
  }
  return ptr;
}

// Increment the tail counter at slot [comm.rank] in peer's signal buffer, notifying peer that this rank has
// made data available.
NCCL_DEVICE_INLINE void ncclLsaFlowConn::postSend() {
  ncclLsaFlowFence();
  ncclLsaFlowStoreSignal(peerSignal, step);
}

NCCL_DEVICE_INLINE void ncclLsaFlowConn::advanceStep() {
  step++;
}

// Return a pointer into the local shared buffer at the next recv slot; call postRecv() after reading.
NCCL_DEVICE_INLINE void* ncclLsaFlowConn::waitRecv() {
  int const slot = (step++) % nSlots;
  void* const ptr = fifo + slot * slotSize;

  // Wait until this slot is ready.
  uint32_t spins = 0;
  while (cache < step) {
    cache = ncclLsaFlowLoadSignal(localSignal);
    if (nccl::utility::testAbort(abortFlag, spins)) break;
  }
  return ptr;
}

// Publish accumulated receive progress after half of the FIFO has been consumed.
NCCL_DEVICE_INLINE void ncclLsaFlowConn::postRecv() {
  uint64_t const value = step + nSlots;
  uint64_t const batch = (nSlots + 1) / 2;
  if (signalValue == 0 || (value > signalValue && value - signalValue >= batch)) {
    ncclLsaFlowStoreSignal(peerSignal, value);
    signalValue = value;
  }
}

#endif
#endif // NCCL_DEVICE_LSA_FLOW__FUNCS_H_
