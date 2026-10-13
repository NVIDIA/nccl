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
template <typename Backend, int SlicesPerChunk, int MaxPeers>
template <bool Send, bool DirectSend, bool Recv, bool DirectRecv, typename T, typename SrcLambda, typename DstLambda,
          typename RedOp>
NCCL_DEVICE_INLINE void ncclFlowBase<Backend, SlicesPerChunk, MaxPeers>::process(
  bool sendDirect, bool recvDirect, ncclSymPtr<T> input, ncclSymPtr<T> output, SrcLambda srcLambda, int nSrc,
  DstLambda dstLambda, int nDst, RedOp const& redOp, size_t nElts, ncclFlowProtocolTag<ncclFlowProtocolSimple>) {
  static_assert(sizeof(T) <= 16 && 16 % sizeof(T) == 0, "T must divide the 16-byte transfer alignment");
  Backend& backend = *static_cast<Backend*>(this);
  ncclFlowPartShmem& shmem = *partShmem;
  int const sliceSteps = ncclFlowFifoSlots / nSlices;
  assert((stepSize * sliceSteps) % sizeof(T) == 0);
  size_t const sliceElts = stepSize * sliceSteps / sizeof(T);
  int const nChunkSlices = (int)nccl::utility::divUp(nElts, sliceElts);
  assert(nChunkSlices <= SlicesPerChunk);
  bool const isWorker = role == RoleWorker || role == RoleSendWait || role == RoleRecvWait;
  int const chunkSendPeers = Send ? nSendPeers : 0;
  int const chunkRecvPeers = Recv ? nRecvPeers : 0;
  bool const sourceIsInput = Send && !Recv && nSrc == 1 && input.window != nullptr;
  bool const sourceIsOutput = sendDirect && recvDirect;
  assert(!sendDirect || DirectSend);
  assert(!recvDirect || DirectRecv);
  assert(0 <= nSrc && nSrc <= 1 + (DirectRecv ? MaxPeers : 0));
  assert(0 <= nDst && nDst <= 1 + (DirectSend ? MaxPeers : 0));

  for (int chunkSlice = 0; chunkSlice < nChunkSlices; chunkSlice++) {
    size_t const sliceOffset = chunkSlice * sliceElts;
    size_t const sliceRemaining = sliceOffset < nElts ? nElts - sliceOffset : 0;
    size_t const sliceCount = sliceElts < sliceRemaining ? sliceElts : sliceRemaining;
    ncclSymPtr<T> const sliceInput = input + sliceOffset;
    ncclSymPtr<T> const sliceOutput = output + sliceOffset;

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
      for (int peer = 0; peer < chunkSendPeers; peer++) sendFifoPeers += shmem.sendUsesFifo[peer];
      for (int peer = 0; peer < chunkRecvPeers; peer++) recvFifoPeers += shmem.recvUsesFifo[peer];
      auto sliceSrc = [=] __device__(int i) -> T* {
        if (i < nSrc) return srcLambda(i) + sliceOffset;
        int fifo = i - nSrc;
        for (int peer = 0; peer < chunkRecvPeers; peer++) {
          if (shmemPtr->recvUsesFifo[peer] && fifo-- == 0) return (T*)shmemPtr->recvBuf[peer];
        }
        return nullptr;
      };
      auto sliceDst = [=] __device__(int i) -> T* {
        if (i < nDst) return dstLambda(i) + sliceOffset;
        int fifo = i - nDst;
        for (int peer = 0; peer < chunkSendPeers; peer++) {
          if (shmemPtr->sendUsesFifo[peer] && fifo-- == 0) return (T*)shmemPtr->sendBuf[peer];
        }
        return nullptr;
      };
      int const nSliceSrcs = nSrc + recvFifoPeers;
      int const nSliceDsts = nDst + sendFifoPeers;
      assert(nSliceDsts == 0 || nSliceSrcs != 0);
      if (nSliceDsts != 0) {
        backend.operations.template reduceCopy<T, RedOp, 1 + (Recv ? MaxPeers : 0), 1 + (Send ? MaxPeers : 0)>(
          workers.thread_rank(), workers.size(), nSliceSrcs, sliceSrc, nSliceDsts, sliceDst, redOp, (int)sliceCount);
      }
    }
    partThreads.sync();

    if ((Send && role == RoleSendPost) || (Recv && role == RoleRecvPost)) {
      backend.advanceStep();
      if (role == RoleSendPost) {
        ncclSymPtr<T> const source = sourceIsInput ? sliceInput : sourceIsOutput ? sliceOutput : ncclSymPtr<T>{};
        backend.postSend(sliceCount * sizeof(T), sendDirect, sliceOutput, source);
      } else backend.postRecv();
    }
  }
}
#endif

#endif // NCCL_DEVICE_FLOW__SIMPLE_FUNCS_H_
