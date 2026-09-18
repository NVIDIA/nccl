/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more information
 *************************************************************************/

#ifndef NCCL_DEVICE_FLOW__LL128_FUNCS_H_
#define NCCL_DEVICE_FLOW__LL128_FUNCS_H_

#include "../flow.h"
#include "../utility.h"
#include <cassert>

#ifdef __CUDACC__
// Implemented by the symmetric kernels using their internal packed reduction operators.
template <typename RedOp>
NCCL_DEVICE_INLINE uint64_t ncclFlowLL128Reduce(RedOp const& redOp, uint64_t a, uint64_t b);

NCCL_DEVICE_INLINE void ncclFlowLL128Load128(uint64_t const* src, uint64_t& lo, uint64_t& hi) {
  asm volatile("ld.volatile.global.v2.u64 {%0,%1}, [%2];" : "=l"(lo), "=l"(hi) : "l"(src) : "memory");
}

NCCL_DEVICE_INLINE void ncclFlowLL128Store128(uint64_t* dst, uint64_t lo, uint64_t hi) {
  asm volatile("st.volatile.global.v2.u64 [%0], {%1,%2};" ::"l"(dst), "l"(lo), "l"(hi) : "memory");
}

template <typename T>
NCCL_DEVICE_INLINE uint64_t ncclFlowLL128LoadDataWord(T* src, int nElts) {
  if (nElts <= 0) return 0;
  constexpr int EltPerWord = sizeof(uint64_t) / sizeof(T);
  ncclFlowLLDataLoader<T> loader;
  loader.begin(src, nElts < EltPerWord ? nElts : EltPerWord);
  return loader.finish();
}

template <typename T>
NCCL_DEVICE_INLINE void ncclFlowLL128LoadDataBegin(uint64_t (&data)[NCCL_LL128_SHMEM_ELEMS_PER_THREAD], T* src,
                                                   int nElts, int lane, bool flagThread) {
  constexpr int EltPerWord = sizeof(uint64_t) / sizeof(T);
  constexpr int EltPer128 = 2 * EltPerWord;
  NVCC_PRAGMA_UNROLL_AUTO
  for (int g = 0; g < NCCL_LL128_SHMEM_ELEMS_PER_THREAD / 2; g++) {
    int const ix = g * 32 - 4 * (g / 2) + lane - (g % 2) * (lane / 8);
    int const eltOffset = ix * EltPer128;
    int const remaining = nElts - eltOffset;
    if (!flagThread || g % 2 == 0) {
      if (remaining <= 0) {
        data[2 * g] = 0;
        data[2 * g + 1] = 0;
      } else if (remaining >= EltPer128 && reinterpret_cast<uintptr_t>(src + eltOffset) % 16 == 0) {
        ncclFlowLL128Load128((uint64_t*)(src + eltOffset), data[2 * g], data[2 * g + 1]);
      } else {
        data[2 * g] = ncclFlowLL128LoadDataWord(src + eltOffset, remaining);
        data[2 * g + 1] =
          remaining > EltPerWord ? ncclFlowLL128LoadDataWord(src + eltOffset + EltPerWord, remaining - EltPerWord) : 0;
      }
    } else {
      data[2 * g] = 0;
      data[2 * g + 1] = 0;
    }
  }
}

NCCL_DEVICE_INLINE void ncclFlowLL128LoadDataFinish(uint64_t (&data)[NCCL_LL128_SHMEM_ELEMS_PER_THREAD],
                                                    bool flagThread) {
  NVCC_PRAGMA_UNROLL_AUTO
  for (int g = 1; g < NCCL_LL128_SHMEM_ELEMS_PER_THREAD / 2; g += 2) {
    if (flagThread) data[2 * g] = data[2 * g - 1];
  }
}

template <typename T>
NCCL_DEVICE_INLINE void ncclFlowLL128StoreData(T* dst, uint64_t (&data)[NCCL_LL128_SHMEM_ELEMS_PER_THREAD], int nElts,
                                               int lane, bool flagThread) {
  constexpr int EltPerWord = sizeof(uint64_t) / sizeof(T);
  constexpr int EltPer128 = 2 * EltPerWord;
  NVCC_PRAGMA_UNROLL_AUTO
  for (int g = 1; g < NCCL_LL128_SHMEM_ELEMS_PER_THREAD / 2; g += 2) {
    if (flagThread) data[2 * g - 1] = data[2 * g];
  }
  NVCC_PRAGMA_UNROLL_AUTO
  for (int g = 0; g < NCCL_LL128_SHMEM_ELEMS_PER_THREAD / 2; g++) {
    int const ix = g * 32 - 4 * (g / 2) + lane - (g % 2) * (lane / 8);
    int const eltOffset = ix * EltPer128;
    int const remaining = nElts - eltOffset;
    if (!flagThread || g % 2 == 0) {
      if (remaining >= EltPer128 && reinterpret_cast<uintptr_t>(dst + eltOffset) % 16 == 0) {
        ncclFlowLL128Store128((uint64_t*)(dst + eltOffset), data[2 * g], data[2 * g + 1]);
      } else {
        int const loElts = remaining < EltPerWord ? remaining : EltPerWord;
        int const hiRemaining = remaining - EltPerWord;
        int const hiElts = hiRemaining < EltPerWord ? hiRemaining : EltPerWord;
        if (loElts > 0) ncclFlowLLStoreData(dst + eltOffset, data[2 * g], loElts);
        if (hiElts > 0) {
          ncclFlowLLStoreData(dst + eltOffset + EltPerWord, data[2 * g + 1], hiElts);
        }
      }
    }
  }
}

NCCL_DEVICE_INLINE void ncclFlowLL128LoadWire(uint64_t (&data)[NCCL_LL128_SHMEM_ELEMS_PER_THREAD], uint64_t* src,
                                              uint64_t flag, bool flagThread, uint32_t* abortFlag) {
  uint32_t spins = 0;
  bool needReload;
  do {
    needReload = false;
    NVCC_PRAGMA_UNROLL_AUTO
    for (int u = 0; u < NCCL_LL128_SHMEM_ELEMS_PER_THREAD; u += 2) {
      ncclFlowLL128Load128(src + u * 32, data[u], data[u + 1]);
      needReload |= flagThread && data[u + 1] != flag;
    }
    if (nccl::utility::testAbort(abortFlag, spins)) needReload = false;
  } while (__any_sync(0xffffffffu, needReload));

  NVCC_PRAGMA_UNROLL_AUTO
  for (int u = 0; u < NCCL_LL128_SHMEM_ELEMS_PER_THREAD; u += 2) {
    ncclFlowLL128Load128(src + u * 32, data[u], data[u + 1]);
  }
}

NCCL_DEVICE_INLINE void ncclFlowLL128StoreWire(uint64_t* dst, uint64_t const (&data)[NCCL_LL128_SHMEM_ELEMS_PER_THREAD],
                                               uint64_t flag, bool flagThread) {
  NVCC_PRAGMA_UNROLL_AUTO
  for (int u = 0; u < NCCL_LL128_SHMEM_ELEMS_PER_THREAD; u += 2) {
    ncclFlowLL128Store128(dst + u * 32, data[u], flagThread ? flag : data[u + 1]);
  }
}

template <typename Backend, int SlotsPerProcess, int MaxPeers>
template <bool Send, bool DirectSend, bool Recv, bool DirectRecv, typename T, typename SrcLambda, typename DstLambda,
          typename RedOp>
NCCL_DEVICE_INLINE void ncclFlowBase<Backend, SlotsPerProcess, MaxPeers>::process(
  bool sendDirect, bool recvDirect, ncclSymPtr<T> /*input*/, ncclSymPtr<T> /*output*/, SrcLambda srcLambda, int nSrc,
  DstLambda dstLambda, int nDst, RedOp const& redOp, size_t nElts, ncclFlowProtocolTag<ncclFlowProtocolLL128>) {
  static_assert(!DirectSend && !DirectRecv, "LL128 Flow does not support direct operations");
  assert(!sendDirect && !recvDirect);
  static_assert(SlotsPerProcess == 1, "LL128 Flow processes one FIFO slot per call");
  static_assert(sizeof(T) <= sizeof(uint64_t) && sizeof(uint64_t) % sizeof(T) == 0,
                "T must divide the eight-byte LL128 payload word");
  constexpr int WireWordsPerWarp = 32 * NCCL_LL128_SHMEM_ELEMS_PER_THREAD;
  constexpr int DataWordsPerWarp = WireWordsPerWarp * NCCL_LL128_DATAELEMS / NCCL_LL128_LINEELEMS;
  constexpr int DataEltsPerWarp = DataWordsPerWarp * sizeof(uint64_t) / sizeof(T);
  Backend& backend = *static_cast<Backend*>(this);
  ncclFlowPartShmem& shmem = *processShmem;
  int const nWorkerWarps = workers.size() / 32;
  int const nDataElts = (int)nElts;
  int const nWireSlices = (nDataElts + DataEltsPerWarp - 1) / DataEltsPerWarp;
  size_t const wireBytes = (size_t)nWireSlices * WireWordsPerWarp * sizeof(uint64_t);
  bool const isWorker = role == RoleWorker || role == RoleSendWait || role == RoleRecvWait;
  int const processSendPeers = Send ? nSendPeers : 0;
  int const processRecvPeers = Recv ? nRecvPeers : 0;
  bool const isPost = (Send && role == RoleSendPost) || (Recv && role == RoleRecvPost);
  assert((size_t)nDataElts == nElts);
  assert(workers.size() % 32 == 0);
  assert(slotSize % NCCL_LL128_LINESIZE == 0);
  assert(wireBytes <= slotSize);
  assert(0 <= nSrc && nSrc <= 1);
  assert(0 <= nDst && nDst <= 1);
  assert(0 < nSrc + processRecvPeers);
  assert(0 < nDst + processSendPeers);

  if (isWorker) {
    if (Send && role == RoleSendWait) {
      shmem.sendBuf[rolePeer] = backend.waitSend();
      shmem.sendFlag[rolePeer] = backend.stepFlag();
    }
    if (Recv && role == RoleRecvWait) {
      shmem.recvBuf[rolePeer] = backend.advanceFifoSlot();
      shmem.recvFlag[rolePeer] = backend.stepFlag();
    }
    workers.sync();

    int const workerThread = workers.thread_rank();
    int const warp = workerThread / 32;
    int const lane = workerThread % 32;
    bool const flagThread = lane % (NCCL_LL128_LINEELEMS / 2) == NCCL_LL128_LINEELEMS / 2 - 1;
    int eltOffset = DataEltsPerWarp * warp;
    int wireOffset = WireWordsPerWarp * warp + 2 * lane;
    for (; eltOffset < nDataElts;
         eltOffset += DataEltsPerWarp * nWorkerWarps, wireOffset += WireWordsPerWarp * nWorkerWarps) {
      int const remaining = nDataElts - eltOffset;
      int const sliceElts = remaining < DataEltsPerWarp ? remaining : DataEltsPerWarp;
      uint64_t data[NCCL_LL128_SHMEM_ELEMS_PER_THREAD] = {};
      uint64_t peerData[NCCL_LL128_SHMEM_ELEMS_PER_THREAD];
      bool dataValid = nSrc != 0;
      if (dataValid) ncclFlowLL128LoadDataBegin(data, srcLambda(0) + eltOffset, sliceElts, lane, flagThread);

      for (int peer = 0; peer < processRecvPeers; peer++) {
        ncclFlowLL128LoadWire(peerData, (uint64_t*)shmem.recvBuf[peer] + wireOffset, shmem.recvFlag[peer], flagThread,
                              abortFlag);
        if (peer == 0 && dataValid) ncclFlowLL128LoadDataFinish(data, flagThread);
        NVCC_PRAGMA_UNROLL_AUTO
        for (int u = 0; u < NCCL_LL128_SHMEM_ELEMS_PER_THREAD; u++) {
          data[u] = dataValid ? ncclFlowLL128Reduce(redOp, peerData[u], data[u]) : peerData[u];
        }
        dataValid = true;
      }
      if (processRecvPeers == 0 && dataValid) ncclFlowLL128LoadDataFinish(data, flagThread);

      for (int peer = 0; peer < processSendPeers; peer++) {
        ncclFlowLL128StoreWire((uint64_t*)shmem.sendBuf[peer] + wireOffset, data, shmem.sendFlag[peer], flagThread);
      }
      if (nDst != 0) ncclFlowLL128StoreData(dstLambda(0) + eltOffset, data, sliceElts, lane, flagThread);
    }
  } else if (isPost) {
    backend.advanceStep();
  }
  partThreads.sync();
  if (isPost) {
    if (role == RoleSendPost) backend.postSend(wireBytes);
    else backend.postRecv();
  }
}
#endif

#endif // NCCL_DEVICE_FLOW__LL128_FUNCS_H_
