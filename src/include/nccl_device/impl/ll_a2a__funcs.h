/*************************************************************************
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef _NCCL_DEVICE_LL_A2A__FUNCS_H_
#define _NCCL_DEVICE_LL_A2A__FUNCS_H_
#include "ll_a2a__types.h"
#include "comm__types.h"
#include "../utility.h"

#if NCCL_CHECK_CUDACC
template<typename Coop>
NCCL_DEVICE_INLINE ncclLLA2ASession<Coop>::ncclLLA2ASession(
    Coop coop, ncclDevComm const& comm, ncclTeam team,
    ncclLLA2AHandle handle, uint32_t block, int maxElts,
    bool multimem, ncclMultimemHandle mmHandle
  ):
  ncclLLA2ASession_internal<Coop>{
    coop, comm, team, handle, (int)block, /*pitch=*/maxElts,
    multimem, mmHandle, /*epoch=*/0, /*slotsOffset=*/0
  } {
  uint4* line = (uint4*)ncclGetResourceBufferLocalPointer(comm, handle.bufHandle);
  line += block*(1 + 2*handle.nSlots);
  this->epoch = line->x + 2;
  this->slotsOffset = this->calcSlotOffset();
}
#endif

#if NCCL_CHECK_CUDACC
template<typename Coop>
NCCL_DEVICE_INLINE ncclLLA2ASession<Coop>::~ncclLLA2ASession() {
  uint4* line = (uint4*)ncclGetResourceBufferLocalPointer(this->comm, this->handle.bufHandle);
  line += this->block*(1 + 2*this->handle.nSlots);
  if (this->coop.thread_rank() == 0) line->x = this->epoch - 2;
  this->coop.sync();
}
#endif

#if NCCL_CHECK_CUDACC
template<typename Coop>
template<typename T>
NCCL_DEVICE_INLINE void ncclLLA2ASession<Coop>::send(int peer, int elt, T data) {
  using nccl::utility::divUp;
  union { T tmp; uint32_t u32[divUp(sizeof(T), 8)][2]; };
  tmp = data;
  uint4* buf = (uint4*)ncclGetResourceBufferPeerPointer(this->comm, this->handle.bufHandle, this->team, peer);
  buf += this->slotsOffset + elt;
  #pragma unroll
  for (int u=0; u < divUp(sizeof(T), 8); u++) {
    asm volatile("st.volatile.v4.u32 [%0],{%1,%3,%2,%3};" ::
      "l"(buf + u*this->pitch),
      "r"(u32[u][0]), "r"(u32[u][1]), "r"(this->epoch)
    );
  }
}
#endif

#if NCCL_CHECK_CUDACC
template<typename Coop>
template<typename T>
NCCL_DEVICE_INLINE void ncclLLA2ASession<Coop>::bcast(int elt, T data) {
  using nccl::utility::divUp;
  if (this->multimem) {
    union { T tmp; uint32_t u32[divUp(sizeof(T),8)][2]; };
    tmp = data;
    uint4* bufmc = (uint4*)ncclGetResourceBufferMultimemPointer(this->comm, this->handle.bufHandle, this->mmHandle);
    bufmc += this->slotsOffset + elt;
    #pragma unroll
    for (int u=0; u < divUp(sizeof(T), 8); u++) {
      asm volatile("st.volatile.v4.u32 [%0],{%1,%3,%2,%3};" ::
        "l"(bufmc + this->pitch*u),
        "r"(u32[u][0]), "r"(u32[u][1]), "r"(this->epoch)
      );
    }
  } else {
    union { T tmp; uint32_t u32[divUp(sizeof(T), 8)][2]; };
    tmp = data;
    int dr = 0;
    int r = this->team.rank;
    #pragma unroll 1
    for (; dr+8 <= this->team.nRanks; dr += 8) {
      #pragma unroll
      for (int ur=0; ur < 8; ur++) {
        uint4* buf = (uint4*)ncclGetResourceBufferPeerPointer(this->comm, this->handle.bufHandle, this->team, r);
        buf += this->slotsOffset + elt;
        #pragma unroll
        for (int u=0; u < divUp(sizeof(T),8); u++) {
          asm volatile("st.volatile.v4.u32 [%0],{%1,%3,%2,%3};" ::
            "l"(buf + u*this->pitch),
            "r"(u32[u][0]), "r"(u32[u][1]), "r"(this->epoch)
          );
        }
        r += 1;
        if (r == this->team.nRanks) r = 0;
      }
    }
    #pragma unroll
    for (int ur=0; ur < 8; ur++, dr++) {
      if (dr == this->team.nRanks) break;
      uint4* buf = (uint4*)ncclGetResourceBufferPeerPointer(this->comm, this->handle.bufHandle, this->team, r);
      buf += this->slotsOffset + elt;
      #pragma unroll
      for (int u=0; u < divUp(sizeof(T),8); u++) {
        asm volatile("st.volatile.v4.u32 [%0],{%1,%3,%2,%3};" ::
          "l"(buf + u*this->pitch),
          "r"(u32[u][0]), "r"(u32[u][1]), "r"(this->epoch)
        );
      }
      r += 1;
      if (r == this->team.nRanks) r = 0;
    }
  }
}
#endif

#if NCCL_CHECK_CUDACC
template<typename Coop>
template<typename T>
NCCL_DEVICE_INLINE T ncclLLA2ASession<Coop>::recv(int elt) {
  T ret[1];
  this->template recvUnrolled</*MinEltCount=*/1, /*MaxEltCount=*/1>(elt, 1, 0, ret);
  return ret[0];
}
#endif

#if NCCL_CHECK_CUDACC
template<typename Coop>
template<int MinEltCount, int MaxEltCount, typename T>
NCCL_DEVICE_INLINE void ncclLLA2ASession<Coop>::recvUnrolled(int eltStart, int eltCount, int eltStride, T(&elts)[MaxEltCount]) {
  using nccl::utility::divUp;
  uint4* buf = (uint4*)ncclGetResourceBufferLocalPointer(this->comm, this->handle.bufHandle);
  buf += this->slotsOffset + eltStart;

  uint4 tmp[MaxEltCount][divUp(sizeof(T), 8)];
  #pragma unroll 1
  while (true) {
    #pragma unroll
    for (int u=0; u < MaxEltCount; u++) {
      if (u < MinEltCount || u < eltCount) {
        #pragma unroll
        for (int v=0; v < divUp(sizeof(T), 8); v++) {
          asm volatile("ld.volatile.v4.u32 {%0,%1,%2,%3},[%4];"
            : "=r"(tmp[u][v].x), "=r"(tmp[u][v].y), "=r"(tmp[u][v].z), "=r"(tmp[u][v].w)
            : "l"(buf + u*eltStride + v*this->pitch));
        }
      }
    }
    bool okAll = true;
    #pragma unroll
    for (int u=0; u < MaxEltCount; u++) {
      #pragma unroll
      for (int v=0; v < divUp(sizeof(T), 8); v++) {
        if (u < MinEltCount || u < eltCount) {
          bool ok = tmp[u][v].y == this->epoch &&
                    tmp[u][v].w == this->epoch;
          okAll &= ok;
        }
      }
    }
    if (__builtin_expect(okAll, true)) break;
  }

  #pragma unroll
  for (int u=0; u < MaxEltCount; u++) {
    if (MinEltCount <= u && u == eltCount) break;
    union { T val; uint32_t u32[divUp(sizeof(T), 8)][2]; };
    #pragma unroll
    for (int v=0; v < divUp(sizeof(T), 8); v++) {
      u32[v][0] = tmp[u][v].x;
      u32[v][1] = tmp[u][v].z;
    }
    elts[u] = val;
  }
}
#endif

#if NCCL_CHECK_CUDACC
template<typename Coop>
template<int Unroll, typename Elt, typename EltToAcc, typename Reduce>
NCCL_DEVICE_INLINE auto ncclLLA2ASession<Coop>::recvReduce(
    int eltStart, int eltCount, int eltStride, EltToAcc eltToAcc, Reduce reduce
  ) -> decltype(eltToAcc(nccl::utility::declval<Elt>())) {
  using Acc = decltype(eltToAcc(nccl::utility::declval<Elt>()));
  Acc acc;
  int i = 0;
  #pragma unroll 1
  for (; i+Unroll <= eltCount; i += Unroll) {
    Elt got[Unroll];
    this->template recvUnrolled</*Min=*/Unroll>(eltStart + i*eltStride, Unroll, eltStride, got);
    Acc acc0 = eltToAcc(got[0]);
    acc = i==0 ? acc0 : reduce(acc, acc0);
    #pragma unroll
    for (int j=1; j < Unroll; j++) acc = reduce(acc, eltToAcc(got[j]));
  }
  if (i < eltCount) {
    Elt got[Unroll];
    this->template recvUnrolled</*Min=*/1>(eltStart + i*eltStride, eltCount-i, eltStride, got);
    Acc acc0 = eltToAcc(got[0]);
    acc = i==0 ? acc0 : reduce(acc, acc0);
    #pragma unroll
    for (int j=1; j < Unroll-1; j++) {
      if (i+j < eltCount) acc = reduce(acc, eltToAcc(got[j]));
    }
  }
  return acc;
}
#endif

#if NCCL_CHECK_CUDACC
template<typename Coop>
NCCL_DEVICE_INLINE void ncclLLA2ASession<Coop>::endEpoch(Coop) {
  if (__builtin_expect(this->epoch >= -2u, false)) {
    this->coop.sync();
    uint4* buf = (uint4*)ncclGetResourceBufferLocalPointer(this->comm, this->handle.bufHandle);
    buf += this->slotsOffset;
    #pragma unroll 4
    for (int i=this->coop.thread_rank(); i < this->handle.nSlots; i += this->coop.size()) {
      buf[i] = uint4{0, 0, 0, 0};
    }
  }
  this->coop.sync();
  this->epoch += (this->epoch == -1u) ? 3 : 1;
  this->slotsOffset = this->calcSlotOffset();
}
#endif

#endif // _NCCL_DEVICE_LL_A2A__FUNCS_H_
