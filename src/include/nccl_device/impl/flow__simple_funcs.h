/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more information
 *************************************************************************/

#ifndef NCCL_DEVICE_FLOW__SIMPLE_FUNCS_H_
#define NCCL_DEVICE_FLOW__SIMPLE_FUNCS_H_

#include "../flow.h"
#include "../utility.h"
#include <cassert>

#ifdef __CUDACC__
// Implemented by the symmetric kernels using their internal reduceCopy primitive.
template <typename T, typename RedOp, int MaxSrcs, int MaxDsts, typename SrcFn, typename DstFn>
NCCL_DEVICE_INLINE void ncclFlowReduceCopy(int thread, int nThreads, int nSrcs, SrcFn const& srcFn, int nDsts,
                                           DstFn const& dstFn, RedOp const& redOp, int nElts);

template <typename Backend, int SlotsPerProcess, int MaxPeers>
template <bool Send, bool DirectSend, bool Recv, bool DirectRecv, typename T, typename SrcLambda, typename DstLambda,
          typename RedOp>
NCCL_DEVICE_INLINE void ncclFlowBase<Backend, SlotsPerProcess, MaxPeers>::process(
  bool sendDirect, bool recvDirect, ncclSymPtr<T> input, ncclSymPtr<T> output, SrcLambda srcLambda, int nSrc,
  DstLambda dstLambda, int nDst, RedOp const& redOp, size_t nElts, ncclFlowProtocolTag<ncclFlowProtocolSimple>) {
  static_assert(sizeof(T) <= 16 && 16 % sizeof(T) == 0, "T must divide the 16-byte transfer alignment");
  Backend& backend = *static_cast<Backend*>(this);
  ncclFlowPartShmem& shmem = *processShmem;
  assert(slotSize % sizeof(T) == 0);
  size_t const slotElts = slotSize / sizeof(T);
  int const nProcessSlots = (int)nccl::utility::divUp(nElts, slotElts);
  assert(nProcessSlots <= SlotsPerProcess);
  bool const isWorker = role == RoleWorker || role == RoleSendWait || role == RoleRecvWait;
  int const processSendPeers = Send ? nSendPeers : 0;
  int const processRecvPeers = Recv ? nRecvPeers : 0;
  bool const sourceIsInput = Send && !Recv && nSrc == 1 && input.window != nullptr;
  bool const sourceIsOutput = sendDirect && recvDirect;
  assert(!sendDirect || DirectSend);
  assert(!recvDirect || DirectRecv);
  assert(0 <= nSrc && nSrc <= 1 + (DirectRecv ? MaxPeers : 0));
  assert(0 <= nDst && nDst <= 1 + (DirectSend ? MaxPeers : 0));

  for (int processSlot = 0; processSlot < nProcessSlots; processSlot++) {
    size_t const slotOffset = processSlot * slotElts;
    size_t const slotRemaining = slotOffset < nElts ? nElts - slotOffset : 0;
    size_t const slotCount = slotElts < slotRemaining ? slotElts : slotRemaining;
    ncclSymPtr<T> const slotInput = input + slotOffset;
    ncclSymPtr<T> const slotOutput = output + slotOffset;

    if (isWorker) {
      if (Send && role == RoleSendWait) {
        shmem.sendBuf[rolePeer] = backend.waitSend();
        shmem.sendUsesFifo[rolePeer] = backend.usesGin() ? !(sourceIsInput || sourceIsOutput) : !sendDirect;
      }
      if (Recv && role == RoleRecvWait) {
        shmem.recvBuf[rolePeer] = backend.waitRecv();
        shmem.recvUsesFifo[rolePeer] = !recvDirect;
      }
      workers.sync();

      ncclFlowPartShmem* const shmemPtr = &shmem;
      int sendFifoPeers = 0;
      int recvFifoPeers = 0;
      for (int peer = 0; peer < processSendPeers; peer++) sendFifoPeers += shmem.sendUsesFifo[peer];
      for (int peer = 0; peer < processRecvPeers; peer++) recvFifoPeers += shmem.recvUsesFifo[peer];
      auto slotSrc = [=] __device__(int i) -> T* {
        if (i < nSrc) return srcLambda(i) + slotOffset;
        int fifo = i - nSrc;
        for (int peer = 0; peer < processRecvPeers; peer++) {
          if (shmemPtr->recvUsesFifo[peer] && fifo-- == 0) return (T*)shmemPtr->recvBuf[peer];
        }
        return nullptr;
      };
      auto slotDst = [=] __device__(int i) -> T* {
        if (i < nDst) return dstLambda(i) + slotOffset;
        int fifo = i - nDst;
        for (int peer = 0; peer < processSendPeers; peer++) {
          if (shmemPtr->sendUsesFifo[peer] && fifo-- == 0) return (T*)shmemPtr->sendBuf[peer];
        }
        return nullptr;
      };
      int const nProcessSrcs = nSrc + recvFifoPeers;
      int const nProcessDsts = nDst + sendFifoPeers;
      assert(nProcessDsts == 0 || nProcessSrcs != 0);
      if (nProcessDsts != 0) {
        ncclFlowReduceCopy<T, RedOp, 1 + (Recv ? MaxPeers : 0), 1 + (Send ? MaxPeers : 0)>(
          workers.thread_rank(), workers.size(), nProcessSrcs, slotSrc, nProcessDsts, slotDst, redOp, (int)slotCount);
      }
    }
    partThreads.sync();

    if ((Send && role == RoleSendPost) || (Recv && role == RoleRecvPost)) {
      backend.advanceStep();
      if (role == RoleSendPost) {
        ncclSymPtr<T> const source = sourceIsInput ? slotInput : sourceIsOutput ? slotOutput : ncclSymPtr<T>{};
        backend.postSend(slotCount * sizeof(T), sendDirect, slotOutput, source);
      } else backend.postRecv();
    }
  }
}
#endif

#endif // NCCL_DEVICE_FLOW__SIMPLE_FUNCS_H_
