/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more information
 *************************************************************************/

#ifndef NCCL_DEVICE_FLOW__LL_FUNCS_H_
#define NCCL_DEVICE_FLOW__LL_FUNCS_H_

#include "../flow.h"
#include "../utility.h"
#include <cassert>

#ifdef __CUDACC__
// Implemented by the symmetric kernels using their internal packed reduction operators.
template <typename RedOp>
NCCL_DEVICE_INLINE uint64_t ncclFlowLLReduce(RedOp const& redOp, uint64_t a, uint64_t b);

NCCL_DEVICE_INLINE uint64_t ncclFlowLLLoad(ncclLLFifoLine* src, uint32_t flag, uint32_t* abortFlag) {
  uint32_t data1, flag1, data2, flag2;
  uint32_t spins = 0;
  do {
    asm volatile("ld.volatile.global.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(data1), "=r"(flag1), "=r"(data2), "=r"(flag2)
                 : "l"(&src->i4)
                 : "memory");
    if (nccl::utility::testAbort(abortFlag, spins)) break;
  } while (flag1 != flag || flag2 != flag);
  return data1 + ((uint64_t)data2 << 32);
}

NCCL_DEVICE_INLINE void ncclFlowLLStore(ncclLLFifoLine* dst, uint64_t data, uint32_t flag) {
  asm volatile("st.volatile.global.v4.u32 [%0], {%1,%2,%3,%4};" ::"l"(&dst->i4), "r"((uint32_t)data), "r"(flag),
               "r"((uint32_t)(data >> 32)), "r"(flag)
               : "memory");
}

NCCL_DEVICE_INLINE bool ncclFlowLLNeedsCleanup(uint32_t flag, int nSlots) {
#ifdef TEST_LL_CLEANUP
  uint32_t const flagMask = NCCL_LL_FLAG_MAX - 1;
#else
  uint32_t const flagMask = ~uint32_t(0);
#endif
  // Modular distance from this flag to zero. The final nSlots operations visit and clean every FIFO slot once.
  uint32_t const stepsUntilWrap = (-flag) & flagMask;
  return stepsUntilWrap < (uint32_t)nSlots;
}

template <typename T>
NCCL_DEVICE_INLINE T ncclFlowLLLoadData(T* src) {
  union {
    T elt;
    uint16_t u2;
    uint32_t u4;
    uint64_t u8;
  } value;
  if (sizeof(T) == 1) asm volatile("ld.volatile.global.b8 %0,[%1];" : "=r"(value.u4) : "l"(src) : "memory");
  else if (sizeof(T) == 2) asm volatile("ld.volatile.global.b16 %0,[%1];" : "=h"(value.u2) : "l"(src) : "memory");
  else if (sizeof(T) == 4) asm volatile("ld.volatile.global.b32 %0,[%1];" : "=r"(value.u4) : "l"(src) : "memory");
  else asm volatile("ld.volatile.global.b64 %0,[%1];" : "=l"(value.u8) : "l"(src) : "memory");
  return value.elt;
}

template <typename T>
struct ncclFlowLLDataLoader {
  static constexpr int EltPerLine = sizeof(uint64_t) / sizeof(T);
  int misalign;
  union {
    uint32_t u4[sizeof(T) <= 2 ? 3 : 2];
    uint64_t u8;
    T elt[EltPerLine];
  };

  NCCL_DEVICE_INLINE void begin(T* src, int nElts) {
    if (sizeof(T) <= 2) {
      misalign = reinterpret_cast<uintptr_t>(src) % 4;
      uint32_t* ptr = reinterpret_cast<uint32_t*>(reinterpret_cast<uintptr_t>(src) & -uintptr_t(4));
      u4[0] = ncclFlowLLLoadData(ptr + 0);
      u4[1] = misalign + nElts * sizeof(T) > 4 ? ncclFlowLLLoadData(ptr + 1) : 0;
      u4[sizeof(T) <= 2 ? 2 : 0] = misalign + nElts * sizeof(T) > 8 ? ncclFlowLLLoadData(ptr + 2) : 0;
    } else {
      NVCC_PRAGMA_UNROLL_AUTO
      for (int i = 0; i < EltPerLine; i++) {
        if (i == 0 || i < nElts) elt[i] = ncclFlowLLLoadData(src + i);
      }
    }
  }

  NCCL_DEVICE_INLINE uint64_t finish() {
    if (sizeof(T) <= 2) {
      u4[0] = __funnelshift_r(u4[0], u4[1], 8 * misalign);
      u4[1] = __funnelshift_r(u4[1], u4[sizeof(T) <= 2 ? 2 : 0], 8 * misalign);
    }
    return u8;
  }
};

template <typename T>
NCCL_DEVICE_INLINE void ncclFlowLLStoreData(T* dst, uint64_t data, int nElts) {
  constexpr int EltPerLine = sizeof(uint64_t) / sizeof(T);
  union {
    uint64_t u8;
    T elt[EltPerLine];
  } value;
  value.u8 = data;
  NVCC_PRAGMA_UNROLL_AUTO
  for (int i = 0; i < EltPerLine; i++) {
    if (i == 0 || i < nElts) dst[i] = value.elt[i];
  }
}

template <typename Backend, int SlotsPerProcess, int MaxPeers>
template <bool Send, bool DirectSend, bool Recv, bool DirectRecv, typename T, typename SrcLambda, typename DstLambda,
          typename RedOp>
NCCL_DEVICE_INLINE void ncclFlowBase<Backend, SlotsPerProcess, MaxPeers>::process(
  bool sendDirect, bool recvDirect, ncclSymPtr<T> /*input*/, ncclSymPtr<T> /*output*/, SrcLambda srcLambda, int nSrc,
  DstLambda dstLambda, int nDst, RedOp const& redOp, size_t nElts, ncclFlowProtocolTag<ncclFlowProtocolLL>) {
  static_assert(!DirectSend && !DirectRecv, "LL Flow does not support direct operations");
  assert(!sendDirect && !recvDirect);
  static_assert(SlotsPerProcess == 1, "LL Flow processes one FIFO slot per call");
  static_assert(sizeof(T) <= sizeof(uint64_t) && sizeof(uint64_t) % sizeof(T) == 0,
                "T must divide the eight-byte LL payload");
  constexpr int EltPerLine = sizeof(uint64_t) / sizeof(T);
  Backend& backend = *static_cast<Backend*>(this);
  ncclFlowPartShmem& shmem = *processShmem;
  int const nLines = (int)nccl::utility::divUp(nElts, (size_t)EltPerLine);
  int const slotLines = (int)(slotSize / sizeof(ncclLLFifoLine));
  bool const isWorker = role == RoleWorker || role == RoleSendWait || role == RoleRecvWait;
  int const processSendPeers = Send ? nSendPeers : 0;
  int const processRecvPeers = Recv ? nRecvPeers : 0;
  bool const isPost = (Send && role == RoleSendPost) || (Recv && role == RoleRecvPost);
  assert(slotSize % sizeof(ncclLLFifoLine) == 0);
  assert(0 <= nLines && nLines <= slotLines);
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

    int line = workers.thread_rank();
    for (; line < nLines; line += workers.size()) {
      int const eltOffset = line * EltPerLine;
      int const remainingElts = (int)nElts - eltOffset;
      int const lineElts = EltPerLine < remainingElts ? EltPerLine : remainingElts;
      uint64_t data = 0;
      bool dataValid = false;
      if (nSrc != 0) {
        ncclFlowLLDataLoader<T> loader;
        loader.begin(srcLambda(0) + eltOffset, lineElts);
        data = loader.finish();
        dataValid = true;
      }
      for (int peer = 0; peer < processRecvPeers; peer++) {
        uint64_t const peerData =
          ncclFlowLLLoad((ncclLLFifoLine*)shmem.recvBuf[peer] + line, (uint32_t)shmem.recvFlag[peer], abortFlag);
        data = dataValid ? ncclFlowLLReduce(redOp, peerData, data) : peerData;
        dataValid = true;
      }
      for (int peer = 0; peer < processSendPeers; peer++) {
        ncclFlowLLStore((ncclLLFifoLine*)shmem.sendBuf[peer] + line, data, (uint32_t)shmem.sendFlag[peer]);
      }
      if (nDst != 0) ncclFlowLLStoreData(dstLambda(0) + eltOffset, data, lineElts);
    }

    for (int peer = 0; peer < processSendPeers; peer++) {
      if (ncclFlowLLNeedsCleanup((uint32_t)shmem.sendFlag[peer], nSlots)) {
        for (int cleanLine = line; cleanLine < slotLines; cleanLine += workers.size()) {
          ncclFlowLLStore((ncclLLFifoLine*)shmem.sendBuf[peer] + cleanLine, 0, (uint32_t)shmem.sendFlag[peer]);
        }
      }
    }
  } else if (isPost) {
    backend.advanceStep();
  }
  partThreads.sync();
  if (isPost) {
    if (role == RoleSendPost) {
      size_t const sendBytes = ncclFlowLLNeedsCleanup((uint32_t)backend.stepFlag(), nSlots) ?
                                 slotSize :
                                 (size_t)nLines * sizeof(ncclLLFifoLine);
      backend.postSend(sendBytes);
    } else {
      backend.postRecv();
    }
  }
}
#endif

#endif // NCCL_DEVICE_FLOW__LL_FUNCS_H_
