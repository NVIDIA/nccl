/*************************************************************************
 * Copyright (c) 2015-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_COMMON_KERNEL_H_
#define NCCL_COMMON_KERNEL_H_

#include "devcomm.h"
#include "op128.h"
#include "reduce_kernel.h"
#include <cstdint>

#include <cuda_runtime.h>

// Define min for ssize_t
inline __device__ int min(int a, ssize_t b) { return (a < b) ? a : b; }

inline __device__ int loadInt(int* ptr) {
  int v;
  asm volatile("ld.volatile.global.u32 %0, [%1];"
      : "=r"(v) : "l"(ptr));
  return v;
}

template<typename RedFn, typename T, int Unroll, int BytePerPack,
         int MultimemSrcs, int MinSrcs, int MaxSrcs,
         int MultimemDsts, int MinDsts, int MaxDsts,
         int PreOpSrcs, typename IntBytes>
__device__ __forceinline__ void reduceCopyPacks(
    int nThreads, int &thread,
    uint64_t redArg, uint64_t *preOpArgs, bool postOp,
    int nSrcs, void **srcPtrs, int nDsts, void **dstPtrs,
    IntBytes &nBytesBehind, IntBytes &nBytesAhead
  ) {
  static_assert(IntBytes(-1)>>1 == IntBytes(-1), "IntBytes must be signed");
  if (BytePerPack == 0) __trap();

  // A hunk is the amount of contiguous data a warp consumes per loop iteration
  // assuming all threads partake.
  constexpr int BytePerHunk = Unroll*WARP_SIZE*BytePerPack;
  int nWarps = nThreads/WARP_SIZE;
  int warp = thread/WARP_SIZE;
  int lane = thread%WARP_SIZE;

  // This thread's initial position.
  IntBytes threadBytesBehind = nBytesBehind + (warp*BytePerHunk + lane*BytePerPack);
  IntBytes threadBytesAhead = nBytesAhead - (warp*BytePerHunk + lane*BytePerPack);
  // Number of hunks to be consumed over all warps.
  IntBytes nHunksAhead = nBytesAhead/(BytePerHunk + !BytePerHunk);
  // Advance collective position.
  nBytesBehind += nHunksAhead*BytePerHunk;
  nBytesAhead -= nHunksAhead*BytePerHunk;
  if (Unroll==1 && BytePerPack <= nBytesAhead) {
    // Only Unroll=1 can do partial hunks (where not all threads partake).
    nHunksAhead += 1;
    nBytesBehind += nBytesAhead - (nBytesAhead%(BytePerPack + !BytePerPack));
    nBytesAhead = nBytesAhead%(BytePerPack + !BytePerPack);
  }
  nHunksAhead -= warp;

  RedFn redFn(redArg);
  uintptr_t minSrcs[MinSrcs + !MinSrcs];
  uintptr_t minDsts[MinDsts + !MinDsts];
  #pragma unroll
  for (int s=0; s < MinSrcs; s++)
    minSrcs[s] = cvta_to_global(srcPtrs[s]) + threadBytesBehind;
  #pragma unroll
  for (int d=0; d < MinDsts; d++)
    minDsts[d] = cvta_to_global(dstPtrs[d]) + threadBytesBehind;

  // Whether partial hunks can be handled dictates loop termination condition.
  while (Unroll==1 ? (BytePerPack <= threadBytesAhead) : (0 < nHunksAhead)) {
    BytePack<BytePerPack> acc[Unroll];

    { RedFn preFn(0 < PreOpSrcs ? preOpArgs[0] : 0);
      #pragma unroll Unroll
      for (int u=0; u < Unroll; u++) {
        if (0 < MultimemSrcs) {
          // applyLoadMultimem uses relaxed semantics for same reason we use volatile below.
          acc[u] = applyLoadMultimem<RedFn, BytePerPack>(preFn, minSrcs[0]);
        } else {
          // Use volatile loads in case credits are polled for with volatile (instead of acquire).
          acc[u] = ld_volatile_global<BytePerPack>(minSrcs[0]);
        }
        minSrcs[0] += WARP_SIZE*BytePerPack;
        if (0 < PreOpSrcs) acc[u] = applyPreOp(preFn, acc[u]);
      }
    }

    #pragma unroll (MinSrcs-1 + !(MinSrcs-1))
    for (int s=1; s < MinSrcs; s++) {
      BytePack<BytePerPack> tmp[Unroll];
      RedFn preFn(s < PreOpSrcs ? preOpArgs[s] : 0);
      #pragma unroll Unroll
      for (int u=0; u < Unroll; u++) {
        if (s < MultimemSrcs) {
          // applyLoadMultimem uses relaxed semantics for same reason we use volatile below.
          tmp[u] = applyLoadMultimem<RedFn, BytePerPack>(preFn, minSrcs[s]);
        } else {
          // Use volatile loads in case credits are polled for with volatile (instead of acquire).
          tmp[u] = ld_volatile_global<BytePerPack>(minSrcs[s]);
        }
        minSrcs[s] += WARP_SIZE*BytePerPack;
      }
      #pragma unroll Unroll
      for (int u=0; u < Unroll; u++) {
        if (s < PreOpSrcs) tmp[u] = applyPreOp(preFn, tmp[u]);
        acc[u] = applyReduce(redFn, acc[u], tmp[u]);
      }
    }

    for (int s=MinSrcs; (MinSrcs < MaxSrcs) && (s < MaxSrcs) && (s < nSrcs); s++) {
      uintptr_t src = cvta_to_global(srcPtrs[s]) + threadBytesBehind;
      BytePack<BytePerPack> tmp[Unroll];
      RedFn preFn(s < PreOpSrcs ? preOpArgs[s] : 0);
      #pragma unroll Unroll
      for (int u=0; u < Unroll; u++) {
        // Use volatile loads in case credits are polled for with volatile (instead of acquire).
        tmp[u] = ld_volatile_global<BytePerPack>(src);
        src += WARP_SIZE*BytePerPack;
      }
      #pragma unroll Unroll
      for (int u=0; u < Unroll; u++) {
        if (s < PreOpSrcs) tmp[u] = applyPreOp(preFn, tmp[u]);
        acc[u] = applyReduce(redFn, acc[u], tmp[u]);
      }
    }

    if (postOp) {
      #pragma unroll Unroll
      for (int u=0; u < Unroll; u++)
        acc[u] = applyPostOp(redFn, acc[u]);
    }

    #pragma unroll (MinDsts + !MinDsts)
    for (int d=0; d < MinDsts; d++) {
      #pragma unroll Unroll
      for (int u=0; u < Unroll; u++) {
        if (d < MultimemDsts) {
          multimem_st_global(minDsts[d], acc[u]);
        } else {
          st_global<BytePerPack>(minDsts[d], acc[u]);
        }
        minDsts[d] += WARP_SIZE*BytePerPack;
      }
    }
    for (int d=MinDsts; (MinDsts < MaxDsts) && (d < MaxDsts) && (d < nDsts); d++) {
      uintptr_t dst = cvta_to_global(dstPtrs[d]) + threadBytesBehind;
      #pragma unroll Unroll
      for (int u=0; u < Unroll; u++) {
        st_global<BytePerPack>(dst, acc[u]);
        dst += WARP_SIZE*BytePerPack;
      }
    }

    nWarps = nThreads/WARP_SIZE;
    #pragma unroll
    for (int s=0; s < MinSrcs; s++) minSrcs[s] += (nWarps-1)*BytePerHunk;
    #pragma unroll
    for (int d=0; d < MinDsts; d++) minDsts[d] += (nWarps-1)*BytePerHunk;
    threadBytesBehind += nWarps*BytePerHunk;
    threadBytesAhead -= nWarps*BytePerHunk;
    nHunksAhead -= nWarps;
  }

  nWarps = nThreads/WARP_SIZE;
  warp = thread/WARP_SIZE;
  lane = thread%WARP_SIZE;
  // The last loop iteration could have been partial, i.e. not taken by all
  // threads. The threads that weren't included need an extra subtraction to
  // make the value warp uniform.
  if (Unroll==1 && nHunksAhead > 0) nHunksAhead -= nWarps;
  // Rotate warps so the warp which got the least work here will be warp 0.
  // This effectively assigns: warp = (warp-nHunks+nWarps)%nWarps;
  warp = -nHunksAhead;
  thread = warp*WARP_SIZE + lane;
}

template<int Unroll, typename RedFn, typename T,
         int MultimemSrcs, int MinSrcs, int MaxSrcs,
         int MultimemDsts, int MinDsts, int MaxDsts, int PreOpSrcs,
         typename IntBytes>
__device__ __forceinline__ void reduceCopyFull(
    int thread, int nThreads,
    uint64_t redArg, uint64_t *preOpArgs, bool postOp,
    int nSrcs, void **srcPtrs, int nDsts, void **dstPtrs,
    IntBytes nElts
  ) {
  static_assert(IntBytes(-1)>>1 == IntBytes(-1), "IntBytes must be signed");
  static_assert(MultimemSrcs <= MinSrcs && MultimemDsts <= MinDsts, "Multimem pointers cannot exceed respective Min values.");
  //int nWarps = nThreads/WARP_SIZE;
  //int warp = thread/WARP_SIZE;
  int lane = thread%WARP_SIZE;
  // If a multimem src is present then our biggest pack size is limited to what
  // is supported for this redfn/type.
  constexpr int BigPackSize = (MultimemSrcs == 0) ? 16 : LoadMultimem_BigPackSize<RedFn>::BigPackSize;

  IntBytes nBytesBehind = 0;
  IntBytes nBytesAhead = nElts*sizeof(T);

  #if __cpp_if_constexpr
  if constexpr (BigPackSize > sizeof(T)) {
  #else
  if (BigPackSize > sizeof(T)) {
  #endif
    // Check that all pointers are BigPackSize aligned.
    bool aligned = true;
    if (lane < nSrcs) aligned &= 0 == cvta_to_global(srcPtrs[lane]) % (BigPackSize + !BigPackSize);
    if (lane < nDsts) aligned &= 0 == cvta_to_global(dstPtrs[lane]) % (BigPackSize + !BigPackSize);
    aligned = __all_sync(~0u, aligned);
    if (aligned) {
      reduceCopyPacks<RedFn, T, Unroll, BigPackSize,
        MultimemSrcs, MinSrcs, MaxSrcs, MultimemDsts, MinDsts, MaxDsts, PreOpSrcs>
        (nThreads, /*&*/thread, redArg, preOpArgs, postOp,
         nSrcs, srcPtrs, nDsts, dstPtrs, /*&*/nBytesBehind, /*&*/nBytesAhead);
      if (nBytesAhead == 0) return;

      reduceCopyPacks<RedFn, T, /*Unroll=*/1, BigPackSize,
        MultimemSrcs, MinSrcs, MaxSrcs, MultimemDsts, MinDsts, MaxDsts, PreOpSrcs>
        (nThreads, /*&*/thread, redArg, preOpArgs, postOp,
         nSrcs, srcPtrs, nDsts, dstPtrs, /*&*/nBytesBehind, /*&*/nBytesAhead);
      if (nBytesAhead == 0) return;
      goto last_resort;
    }
  }

  reduceCopyPacks<RedFn, T, Unroll*(16/sizeof(T))/2, /*BytePerPack=*/sizeof(T),
    MultimemSrcs, MinSrcs, MaxSrcs, MultimemDsts, MinDsts, MaxDsts, PreOpSrcs>
    (nThreads, /*&*/thread, redArg, preOpArgs, postOp,
     nSrcs, srcPtrs, nDsts, dstPtrs, /*&*/nBytesBehind, /*&*/nBytesAhead);
  if (nBytesAhead == 0) return;

last_resort:
  reduceCopyPacks<RedFn, T, /*Unroll=*/1, /*BytePerPack=*/sizeof(T),
    MultimemSrcs, MinSrcs, MaxSrcs, MultimemDsts, MinDsts, MaxDsts, PreOpSrcs>
    (nThreads, /*&*/thread, redArg, preOpArgs, postOp,
     nSrcs, srcPtrs, nDsts, dstPtrs, /*&*/nBytesBehind, /*&*/nBytesAhead);
}

// Warp-uniform memory copy from shared address (not generic) to global address.
// EltSize is the guaranteed alignment of the addresses and sizes.
// Bytes must be a multiple of WARP_SIZE*16
template<int EltSize, int Bytes>
__device__ __forceinline__ void copySharedToGlobal_WarpUnrolled16(
    int lane, uintptr_t dstAddr, uint32_t srcAddr
  ) {
  int nFrontBytes = (16u-unsigned(dstAddr))%16u;
  int nMiddleBytes = (Bytes-nFrontBytes) & -16;
  int nBackBytes = (Bytes-nFrontBytes) % 16;

  { int backLane = WARP_SIZE-1 - lane;
    bool hasFront = lane*EltSize < nFrontBytes;
    bool hasBack = backLane*EltSize < nBackBytes;
    int offset = hasFront ? lane*EltSize : (Bytes - (backLane+1)*EltSize);
    if (hasFront | hasBack) {
      BytePack<EltSize> tmp = ld_shared<EltSize>(srcAddr+offset);
      st_global<EltSize>(dstAddr+offset, tmp);
    }
  }

  srcAddr += nFrontBytes;
  int srcMisalign = (EltSize < 4) ? srcAddr%4 : 0;
  srcAddr += -srcMisalign + lane*16;
  srcMisalign *= 8; // Promote bytes to bits for funnelshift
  dstAddr += nFrontBytes + lane*16;
  nMiddleBytes -= lane*16;
  #pragma unroll
  for (int u=0; u < Bytes/(WARP_SIZE*16); u++) {
    // Only the last iteration could be unnecessary.
    if (u+1 == Bytes/(WARP_SIZE*16) && nMiddleBytes <= 0) break;
    union {
      BytePack<16> b16;
      BytePack<4> b4[4];
    };
    BytePack<4> b4_4;
    b4[0] = ld_shared<4>(srcAddr + 0*4);
    b4[1] = ld_shared<4>(srcAddr + 1*4);
    b4[2] = ld_shared<4>(srcAddr + 2*4);
    b4[3] = ld_shared<4>(srcAddr + 3*4);

    if (srcMisalign != 0) {
      b4_4 = ld_shared<4>(srcAddr + 4*4);
      b4[0].native = __funnelshift_r(b4[0].native, b4[1].native, srcMisalign);
      b4[1].native = __funnelshift_r(b4[1].native, b4[2].native, srcMisalign);
      b4[2].native = __funnelshift_r(b4[2].native, b4[3].native, srcMisalign);
      b4[3].native = __funnelshift_r(b4[3].native, b4_4.native, srcMisalign);
    }

    st_global<16>(dstAddr, b16);

    srcAddr += WARP_SIZE*16;
    dstAddr += WARP_SIZE*16;
    nMiddleBytes -= WARP_SIZE*16;
  }
}

template<int EltSize, int Unroll, int NumDsts1, int MinDsts2, int MaxDsts2, int MaxDstAddrs, typename IntBytes>
__device__ __forceinline__ void copyOneToMany_SrcAligned16(
    int nWarps, int warp, int lane,
    int nDsts2, uintptr_t dstAddrs[/*MaxDstAddrs*/], void** dstPtrs, bool dsts2Aligned,
    uintptr_t& srcAddr, IntBytes& bytesBehind, IntBytes& bytesAhead, IntBytes& itersAhead,
    uint32_t/*const*/& scratchAddr
  ) {
  static_assert(IntBytes(-1)>>1 == IntBytes(-1), "IntBytes must be signed");
  constexpr int MaxDsts = NumDsts1 + MaxDsts2;
  srcAddr += lane*16; // Convert to thread specific
  while (Unroll*WARP_SIZE*16 <= bytesAhead) {
    BytePack<16> reg[Unroll];
    #pragma unroll Unroll
    for (int u=0; u < Unroll; u++) {
      reg[u] = ld_volatile_global<16>(srcAddr + u*WARP_SIZE*16);
    }

    if (dsts2Aligned) {
      #pragma unroll
      for (int d=0; d < MaxDstAddrs; d++) dstAddrs[d] += lane*16;
      #pragma unroll
      for (int d=NumDsts1; d < MaxDstAddrs; d++) {
        #pragma unroll Unroll
        for (int u=0; u < Unroll; u++) {
          st_global<16>(dstAddrs[d] + u*WARP_SIZE*16, reg[u]);
        }
      }
      #pragma unroll
      for (int d=0; d < MaxDstAddrs; d++) dstAddrs[d] -= lane*16;

      bytesBehind += lane*16;
      #pragma unroll 2
      for (int d=MaxDstAddrs; d < MaxDsts && d < NumDsts1+nDsts2; d++) {
        uintptr_t dstAddr = cvta_to_global(dstPtrs[d]) + bytesBehind;
        #pragma unroll Unroll
        for (int u=0; u < Unroll; u++) {
          st_global<16>(dstAddr + u*WARP_SIZE*16, reg[u]);
        }
      }
      bytesBehind -= lane*16;
    }

    if (NumDsts1!=0 || (MaxDsts2!=0 && !dsts2Aligned)) {
      __syncwarp();
      scratchAddr += lane*16;
      #pragma unroll Unroll
      for (int u=0; u < Unroll; u++) {
        st_shared<16>(scratchAddr + u*WARP_SIZE*16, reg[u]);
      }
      scratchAddr -= lane*16;
      __syncwarp();
      #pragma unroll 1
      for (int d=0; d < NumDsts1+(dsts2Aligned ? 0 : nDsts2); d++) {
        uintptr_t dstAddr = (MaxDsts==1) ? dstAddrs[0] : cvta_to_global(dstPtrs[d]) + bytesBehind;
        copySharedToGlobal_WarpUnrolled16<EltSize, /*Bytes=*/Unroll*WARP_SIZE*16>
          (lane, dstAddr, scratchAddr);
      }
    }

    itersAhead -= nWarps;
    bytesAhead  -= nWarps*(Unroll*WARP_SIZE*16);
    bytesBehind += nWarps*(Unroll*WARP_SIZE*16);
    srcAddr += nWarps*(Unroll*WARP_SIZE*16);
    #pragma unroll
    for (int d=0; d < MaxDstAddrs; d++) dstAddrs[d] += nWarps*(Unroll*WARP_SIZE*16);
  }
  srcAddr -= lane*16; // Convert to warp uniform
}

// Diverge: whether a warp is allowed to diverge, which guarantees processing of
// all elements. If !Diverge then only a multiple of Unroll*WARP_SIZE elements
// will be processed.
// If Diverge=true: sizes/ptrs/offsets are block uniform.
// If Diverge=false: ... are warp uniform
template<int EltSize, int Unroll, bool Diverge, int MinDsts, int MaxDsts, int MaxDstAddrs, typename IntBytes>
__device__ __forceinline__ void copyOneToMany_EltWise(
    int nWarps, int warp, int lane, int nDsts, uintptr_t dstAddrs[/*MaxDstAddrs*/], void** dstPtrs,
    uintptr_t& srcAddr, IntBytes& bytesBehind, IntBytes& bytesAhead, IntBytes& itersAhead
  ) {
  static_assert(IntBytes(-1)>>1 == IntBytes(-1), "IntBytes must be signed");

  IntBytes delta = bytesBehind;
  // Convert to warp uniform or thread specific depending on Diverge
  if (Diverge) bytesAhead -= (warp*Unroll*WARP_SIZE + lane)*EltSize;
  // Convert to thread specific
  bytesBehind += ((Diverge ? warp*Unroll*WARP_SIZE : 0) + lane)*EltSize;
  srcAddr     += ((Diverge ? warp*Unroll*WARP_SIZE : 0) + lane)*EltSize;
  #pragma unroll
  for (int d=0; d < MaxDstAddrs; d++) dstAddrs[d] += ((Diverge ? warp*Unroll*WARP_SIZE : 0) + lane)*EltSize;

  while ((Diverge ? EltSize : Unroll*WARP_SIZE*EltSize) <= bytesAhead) {
    BytePack<EltSize> reg[Unroll];
    #pragma unroll
    for (int u=0; u < Unroll; u++) {
      if (!Diverge || u==0 || (u*WARP_SIZE + 1)*EltSize <= bytesAhead) {
        reg[u] = ld_volatile_global<EltSize>(srcAddr + u*WARP_SIZE*EltSize);
      }
    }
    #pragma unroll
    for (int d=0; d < MaxDstAddrs; d++) {
      #pragma unroll
      for (int u=0; u < Unroll; u++) {
        if (d < MinDsts || d < nDsts) {
          if (!Diverge || u==0 || (u*WARP_SIZE + 1)*EltSize <= bytesAhead) {
            st_global<EltSize>(dstAddrs[d] + u*WARP_SIZE*EltSize, reg[u]);
          }
        }
      }
    }
    #pragma unroll (8/Unroll ? 8/Unroll : 1)
    for (int d=MaxDstAddrs; d < MaxDsts && d < nDsts; d++) {
      uintptr_t dstAddr = cvta_to_global(dstPtrs[d]) + bytesBehind;
      #pragma unroll
      for (int u=0; u < Unroll; u++) {
        if (!Diverge || u==0 || (u*WARP_SIZE + 1)*EltSize <= bytesAhead) {
          st_global<EltSize>(dstAddr + u*WARP_SIZE*EltSize, reg[u]);
        }
      }
    }
    itersAhead -= nWarps;
    bytesAhead  -= nWarps*(Unroll*WARP_SIZE*EltSize);
    bytesBehind += nWarps*(Unroll*WARP_SIZE*EltSize);
    srcAddr += nWarps*(Unroll*WARP_SIZE*EltSize);
    #pragma unroll
    for (int d=0; d < MaxDstAddrs; d++) dstAddrs[d] += nWarps*(Unroll*WARP_SIZE*EltSize);
  }
  // if  Diverge: restore to pre-loop block uniform
  // if !Diverge: convert to post-loop warp uniform
  delta = Diverge ? bytesBehind-delta : lane*EltSize;
  if (Diverge) bytesAhead += delta;
  bytesBehind -= delta;
  srcAddr -= delta;
  #pragma unroll
  for (int d=0; d < MaxDstAddrs; d++) dstAddrs[d] -= delta;

  if (Diverge) {
    // Advance block uniform to post-loop
    delta = bytesAhead-(bytesAhead%EltSize);
    bytesAhead -= delta;
    bytesBehind += delta;
    srcAddr += delta;
    #pragma unroll
    for (int d=0; d < MaxDstAddrs; d++) dstAddrs[d] += delta;
  }
}

// The number of dsts is split as NumDsts1+nDsts2. dstPtrs[0..NumDsts1) are
// never assumed to be 16B aligned at compile time. dstPtrs[NumDsts1...] are
// assumed 16B aligned if Dsts2Aligned=true. Dynamic detection of alignment specializes
// to one of three cases tested in order:
//  1) All dstPtrs[d] are 16B aligned.
//  2) All dstPtrs[d] are 16B aligned where NumDsts1<=d
//  3) None of dstPtrs[d] are 16B aligned.
template<int EltSize, int NumDsts1, int MinDsts2_, int MaxDsts2,
         bool Dsts2Aligned, bool SrcAligned, typename IntBytes>
__device__ __forceinline__ void copyOneToMany(
    int nWarps, int warp, int lane,
    int nDsts2, void** dstPtrs, void* srcPtr,
    IntBytes bytesAhead, uint32_t/*const*/& scratchAddr
  ) {
  static_assert(IntBytes(-1)>>1 == IntBytes(-1), "IntBytes must be signed");
  constexpr int MinDsts2 = MaxDsts2==0 ? 0 : MinDsts2_;
  constexpr int Unroll = ncclCollUnroll();
  constexpr int MinDsts = NumDsts1 + MinDsts2;
  constexpr int MaxDsts = NumDsts1 + MaxDsts2;
  constexpr int MaxDstAddrs = (MaxDsts <= 4) ? MaxDsts : 4;

  IntBytes bytesBehind = 0;
  uintptr_t srcAddr = cvta_to_global(srcPtr);
  uintptr_t dstAddrs[MaxDstAddrs ? MaxDstAddrs : 1];
  if (MinDsts2 == MaxDsts2) nDsts2 = MinDsts2;
  #pragma unroll
  for (int d=0; d < MaxDstAddrs; d++) {
    if (d < MinDsts || d < NumDsts1+nDsts2) dstAddrs[d] = cvta_to_global(dstPtrs[d]);
  }

  if (!SrcAligned && srcAddr%16 != 0) {
    int delta = min(IntBytes(16-(srcAddr%16)), bytesAhead);
    if (warp == 0) {
      if (lane*EltSize < delta) {
        BytePack<EltSize> tmp = ld_volatile_global<EltSize>(srcAddr + lane*EltSize);
        #pragma unroll (MaxDsts ? MaxDsts : 1)
        for (int d=0; d < MaxDsts; d++) {
          if (d < MinDsts || d < NumDsts1+nDsts2) {
            uintptr_t dstAddr = (d < MaxDstAddrs) ? dstAddrs[d] : cvta_to_global(dstPtrs[d]);
            st_global<EltSize>(dstAddr + lane*EltSize, tmp);
          }
        }
      }
      warp = nWarps;
    }
    warp -= 1;
    bytesAhead -= delta;
    bytesBehind += delta;
    srcAddr += delta;
    #pragma unroll
    for (int d=0; d < MaxDstAddrs; d++) dstAddrs[d] += delta;
  }

  bool dsts1Aligned = true, dsts2Aligned = true;
  #pragma unroll
  for (int d=0; d < NumDsts1; d++) {
    uintptr_t dstAddr = (d < MaxDstAddrs) ? dstAddrs[d] : cvta_to_global(dstPtrs[d]) + bytesBehind;
    dsts1Aligned &= (0 == dstAddr%16);
  }
  if (!(SrcAligned && Dsts2Aligned)) {
    #pragma unroll (MaxDsts ? MaxDsts : 1)
    for (int d=NumDsts1; d < MaxDsts; d++) {
      if (d < MinDsts || d < NumDsts1+nDsts2) {
        uintptr_t dstAddr = (d < MaxDstAddrs) ? dstAddrs[d] : cvta_to_global(dstPtrs[d]) + bytesBehind;
        dsts2Aligned &= (0 == dstAddr%16);
      }
    }
  }

  IntBytes itersAhead = bytesAhead/(Unroll*WARP_SIZE*16);
  // Convert coordinates to warp uniform
  itersAhead -= warp;
  bytesAhead -= warp*(Unroll*WARP_SIZE*16);
  bytesBehind += warp*(Unroll*WARP_SIZE*16);
  srcAddr += warp*(Unroll*WARP_SIZE*16);
  #pragma unroll
  for (int d=0; d < MaxDstAddrs; d++) dstAddrs[d] += warp*(Unroll*WARP_SIZE*16);

  // All of these cases use warp-uniform coordinates and process only full hunks
  // of Unroll*WARP_SIZE*16 bytes.
  if (dsts1Aligned && dsts2Aligned) {
    copyOneToMany_SrcAligned16
      <EltSize, Unroll, /*NumDsts1=*/0, /*MinDsts2=*/MinDsts, /*MaxDsts2=*/MaxDsts, MaxDstAddrs>
        (nWarps, warp, lane, NumDsts1+nDsts2, dstAddrs, dstPtrs, /*dsts2Aligned=*/true,
         srcAddr, bytesBehind, bytesAhead, itersAhead, scratchAddr);
  } else if (EltSize < 4) {
    copyOneToMany_SrcAligned16
      <EltSize, Unroll, /*NumDsts1=*/NumDsts1, /*MinDsts2=*/MinDsts2, /*MaxDsts2=*/MaxDsts2, MaxDstAddrs>
        (nWarps, warp, lane, nDsts2, dstAddrs, dstPtrs, dsts2Aligned,
         srcAddr, bytesBehind, bytesAhead, itersAhead, scratchAddr);
  } else {
    copyOneToMany_EltWise
      <EltSize, Unroll*16/EltSize, /*Diverge=*/false, MinDsts, MaxDsts, MaxDstAddrs>
        (nWarps, warp, lane, NumDsts1+nDsts2, dstAddrs, dstPtrs,
         srcAddr, bytesBehind, bytesAhead, itersAhead);
  }

  // Rotate warp index. If there is any work left only warp=0 will have bytesAhead>0.
  warp = -itersAhead;
  // Bring coordinates to block uniform so all warps match warp=0.
  bytesAhead  += warp*(Unroll*WARP_SIZE*16);
  bytesBehind -= warp*(Unroll*WARP_SIZE*16);
  srcAddr -= warp*(Unroll*WARP_SIZE*16);
  #pragma unroll
  for (int d=0; d < MaxDstAddrs; d++) dstAddrs[d] -= warp*(Unroll*WARP_SIZE*16);

  if (dsts1Aligned && dsts2Aligned) {
    // Since the all remaining code has Unroll=1 we can afford more registers
    // for dst pointers.
    uintptr_t allDstAddrs[MaxDsts ? MaxDsts : 1];
    #pragma unroll (MaxDsts ? MaxDsts : 1)
    for (int d=0; d < MaxDsts; d++) {
      if (d < MinDsts || d < NumDsts1+nDsts2) {
        allDstAddrs[d] = (d < MaxDstAddrs) ? dstAddrs[d] : cvta_to_global(dstPtrs[d]) + bytesBehind;
      }
    }
    copyOneToMany_EltWise</*EltSize=*/16, /*Unroll=*/1, /*Diverge=*/true, MinDsts, MaxDsts, MaxDsts>
      (nWarps, warp, lane, NumDsts1+nDsts2, allDstAddrs, dstPtrs, srcAddr,
       bytesBehind, bytesAhead, /*ignored*/itersAhead);

    // Use warp=0 to clean up remainder (bytesAhead is less than 16).
    if (warp*WARP_SIZE + lane*EltSize < bytesAhead) {
      BytePack<EltSize> reg = ld_volatile_global<EltSize>(srcAddr + lane*EltSize);
      #pragma unroll (MaxDsts ? MaxDsts : 1)
      for (int d=0; d < MaxDsts; d++) {
        if (d < MinDsts || d < NumDsts1+nDsts2) {
          st_global<EltSize>(allDstAddrs[d] + lane*EltSize, reg);
        }
      }
    }
  } else {
    copyOneToMany_EltWise<EltSize, /*Unroll=*/4, /*Diverge=*/true, MinDsts, MaxDsts, MaxDstAddrs>
      (nWarps, warp, lane, NumDsts1+nDsts2, dstAddrs, dstPtrs, srcAddr,
       bytesBehind, bytesAhead, /*ignored*/itersAhead);
  }
}

template<int EltSize, bool DstAligned, bool SrcAligned, typename IntBytes>
__device__ __forceinline__ void copyOneToOne(
    int nWarps, int warp, int lane,
    void* dstPtr, void* srcPtr, IntBytes bytesAhead, uint32_t scratchAddr
  ) {
  copyOneToMany<EltSize, /*NumDsts1=*/0, /*{Min,Max}Dsts2=*/1,1, DstAligned, SrcAligned>
    (nWarps, warp, lane, 1, &dstPtr, srcPtr, bytesAhead, scratchAddr);
}

template<int Unroll, typename RedFn, typename T,
         int MultimemSrcs, int MinSrcs, int MaxSrcs,
         int MultimemDsts, int MinDsts, int MaxDsts, int PreOpSrcs,
         typename IntBytes>
__device__ __forceinline__ void reduceCopy(
    int thread, int nThreads,
    uint64_t redArg, uint64_t *preOpArgs, bool postOp,
    int nSrcs, void **srcPtrs, int nDsts, void **dstPtrs,
    IntBytes nElts, uint32_t/*const*/& scratchAddr
  ) {
  if (MaxDsts > 0 && MaxSrcs > 0) {
    if (MultimemSrcs+MultimemDsts == 0 && MinSrcs==1 && MaxSrcs==1 &&
        Apply_PreOp<RedFn, 1>::IsIdentity && Apply_PostOp<RedFn, 1>::IsIdentity) {
      int nWarps = nThreads/WARP_SIZE;
      int warp = thread/WARP_SIZE;
      int lane = thread%WARP_SIZE;
      if (MinDsts > 0 || nDsts > 0) {
        copyOneToMany<sizeof(T), /*NumDsts1=*/1, /*MinDsts2=*/0, /*MaxDsts2=*/MaxDsts-1,
                      /*Dst2Aligned=*/false, /*SrcAligned=*/false>
          (nWarps, warp, lane, /*nDsts2=*/nDsts-1, dstPtrs, srcPtrs[0],
           IntBytes(nElts*sizeof(T)), scratchAddr);
      }
    } else {
      reduceCopyFull<Unroll, RedFn, T, MultimemSrcs,
                     MinSrcs, MaxSrcs, MultimemDsts, MinDsts, MaxDsts, PreOpSrcs>
        (thread, nThreads, redArg, preOpArgs, postOp, nSrcs, srcPtrs, nDsts, dstPtrs, nElts);
    }
  }
}
#endif // COMMON_KERNEL_H_
