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
         int MultimemDsts, int MinDsts, int MaxDsts, int PreOpSrcs,
         typename IntBytes>
__device__ __forceinline__ void reduceCopyPacks(
    int nThreads, int &thread,
    uint64_t redArg, uint64_t *preOpArgs, bool postOp,
    int nSrcs, void **srcPtrs, int nDsts, void **dstPtrs,
    IntBytes &nBytesBehind, IntBytes &nBytesAhead
  ) {
  static_assert(std::is_signed<IntBytes>::value, "IntBytes must be a signed integral type.");
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
__device__ __forceinline__ void reduceCopy(
    int thread, int nThreads,
    uint64_t redArg, uint64_t *preOpArgs, bool postOp,
    int nSrcs, void **srcPtrs, int nDsts, void **dstPtrs,
    IntBytes nElts
  ) {
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
    }
  }

  reduceCopyPacks<RedFn, T, Unroll*(16/sizeof(T))/2, /*BytePerPack=*/sizeof(T),
    MultimemSrcs, MinSrcs, MaxSrcs, MultimemDsts, MinDsts, MaxDsts, PreOpSrcs>
    (nThreads, /*&*/thread, redArg, preOpArgs, postOp,
     nSrcs, srcPtrs, nDsts, dstPtrs, /*&*/nBytesBehind, /*&*/nBytesAhead);
  if (nBytesAhead == 0) return;

  reduceCopyPacks<RedFn, T, /*Unroll=*/1, /*BytePerPack=*/sizeof(T),
    MultimemSrcs, MinSrcs, MaxSrcs, MultimemDsts, MinDsts, MaxDsts, PreOpSrcs>
    (nThreads, /*&*/thread, redArg, preOpArgs, postOp,
     nSrcs, srcPtrs, nDsts, dstPtrs, /*&*/nBytesBehind, /*&*/nBytesAhead);
}

// Warp-uniform memory copy from shared address (not generic) to global address.
// The number of bytes copied is `min(MaxBytes, nBytesAhead)`, a negative value
// is interpeted as zero. EltSize is the guaranteed alignment of the addresses and sizes.
template<int EltSize, int MaxBytes, typename IntBytes>
__device__ __forceinline__ void copyGlobalShared_WarpUnrolled(
    int lane, uintptr_t dstAddr, uint32_t srcAddr, IntBytes nBytesAhead
  ) {
  static_assert(std::is_signed<IntBytes>::value, "`IntBytes` must be a signed integral type.");
  int nBytes = min(nBytesAhead, (IntBytes)MaxBytes);
  int nFrontBytes = min(nBytes, (16 - int(dstAddr%16))%16);
  int nMiddleBytes = (nBytes-nFrontBytes) & -16;
  int nBackBytes = (nBytes-nFrontBytes) % 16;

  { int backLane = WARP_SIZE-1 - lane;
    bool hasFront = lane*EltSize < nFrontBytes;
    bool hasBack = backLane*EltSize < nBackBytes;
    int offset = hasFront ? lane*EltSize : (nBytes - (backLane+1)*EltSize);
    if (hasFront | hasBack) {
      BytePack<EltSize> tmp = ld_shared<EltSize>(srcAddr+offset);
      st_global<EltSize>(dstAddr+offset, tmp);
    }
  }

  srcAddr += nFrontBytes;
  int srcMisalign = EltSize < 4 ? (srcAddr%4) : 0;
  srcAddr += -srcMisalign + lane*16;
  dstAddr += nFrontBytes + lane*16;
  nMiddleBytes -= lane*16;
  #pragma unroll
  for (int u=0; u < divUp(MaxBytes, WARP_SIZE*16); u++) {
    if (nMiddleBytes <= 0) break;
    union {
      BytePack<16> b16;
      BytePack<4> b4[4];
    };
    BytePack<4> b4_4;
    b4[0] = ld_shared<4>(srcAddr + 0*4);
    b4[1] = ld_shared<4>(srcAddr + 1*4);
    b4[2] = ld_shared<4>(srcAddr + 2*4);
    b4[3] = ld_shared<4>(srcAddr + 3*4);

    if(srcMisalign != 0) {
      b4_4 = ld_shared<4>(srcAddr + 4*4);
      b4[0].native = __funnelshift_r(b4[0].native, b4[1].native, srcMisalign*8);
      b4[1].native = __funnelshift_r(b4[1].native, b4[2].native, srcMisalign*8);
      b4[2].native = __funnelshift_r(b4[2].native, b4[3].native, srcMisalign*8);
      b4[3].native = __funnelshift_r(b4[3].native, b4_4.native, srcMisalign*8);
    }

    st_global<16>(dstAddr, b16);

    srcAddr += WARP_SIZE*16;
    dstAddr += WARP_SIZE*16;
    nMiddleBytes -= WARP_SIZE*16;
  }
}

template<int Unroll, bool DstAligned, bool SrcAligned, bool Partial, typename IntBytes>
__device__ __forceinline__ void copyGlobalGlobal_WarpUnrolled16(
    int lane, uintptr_t dstAddr, uintptr_t srcAddr, IntBytes bytes, uint32_t scratchAddr
  ) {
  int srcMisalign = SrcAligned ? 0 : srcAddr%16;
  srcAddr -= srcMisalign;

  BytePack<16> reg[Unroll];
  int offset = 0;
  bytes -= lane*16;
  srcAddr += lane*16;
  #pragma unroll Unroll
  for (int u=0; u < Unroll; u++) {
    if (!Partial || offset < srcMisalign + bytes) {
      reg[u] = ld_volatile_global<16>(srcAddr + offset);
    }
    offset += WARP_SIZE*16;
  }

  if (SrcAligned && DstAligned) {
    offset = 0;
    dstAddr += lane*16;
    #pragma unroll Unroll
    for (int u=0; u < Unroll; u++) {
      if (!Partial || offset < bytes) {
        st_global<16>(dstAddr + offset, reg[u]);
      }
      offset += WARP_SIZE*16;
    }
  } else {
    __syncwarp();
    offset = 0;
    scratchAddr += lane*16;
    #pragma unroll Unroll
    for (int u=0; u < Unroll; u++) {
      if (!Partial || (offset < srcMisalign + bytes)) {
        st_shared<16>(scratchAddr + offset, reg[u]);
      }
      offset += WARP_SIZE*16;
    }
    __syncwarp();
    bytes += lane*16;
    srcAddr -= lane*16;
    scratchAddr -= lane*16;
    if (!SrcAligned) {
      // Ignore the beginning of the first pack corresponding to bytes overread
      // due to misalignment.
      bytes = (int)min(bytes, IntBytes(Unroll*WARP_SIZE*16 - srcMisalign));
    }
    copyGlobalShared_WarpUnrolled<1, /*MaxBytes=*/Unroll*WARP_SIZE*16>
      (lane, dstAddr, scratchAddr + srcMisalign, bytes);
  }
}

template<bool DstAligned, bool SrcAligned, typename IntBytes>
__device__ __forceinline__ void copyGlobalGlobal(
    int nWarps, int warp, int lane,
    uintptr_t dstAddr, uintptr_t srcAddr, IntBytes bytesAhead, uint32_t scratchAddr
  ) {
  constexpr int Unroll = ncclCollUnroll();
  constexpr int BytePerHunk = Unroll*WARP_SIZE*16;

  if (!SrcAligned && srcAddr%16 != 0) {
    if (warp == 0) {
      copyGlobalGlobal_WarpUnrolled16
        <Unroll, /*DstAligned=*/false, /*SrcAligned=*/false, /*Partial=*/true>
          (lane, dstAddr, srcAddr, bytesAhead, scratchAddr);
      warp = nWarps;
    }
    warp -= 1;
    int delta = BytePerHunk-(srcAddr%16);
    srcAddr += delta;
    dstAddr += delta;
    bytesAhead -= delta;
  }

  srcAddr += warp*BytePerHunk;
  dstAddr += warp*BytePerHunk;
  bytesAhead -= warp*BytePerHunk;

  if ((SrcAligned && DstAligned) || dstAddr%16 == 0) {
    while (BytePerHunk <= bytesAhead) {
      copyGlobalGlobal_WarpUnrolled16
        <Unroll, /*DstAligned=*/true, /*SrcAligned=*/true, /*Partial=*/false>
          (lane, dstAddr, srcAddr, bytesAhead, scratchAddr);
      srcAddr += nWarps*BytePerHunk;
      dstAddr += nWarps*BytePerHunk;
      bytesAhead -= nWarps*BytePerHunk;
    }
  } else {
    while (BytePerHunk <= bytesAhead) {
      copyGlobalGlobal_WarpUnrolled16
        <Unroll, /*DstAligned=*/false, /*SrcAligned=*/true, /*Partial=*/false>
          (lane, dstAddr, srcAddr, bytesAhead, scratchAddr);
      srcAddr += nWarps*BytePerHunk;
      dstAddr += nWarps*BytePerHunk;
      bytesAhead -= nWarps*BytePerHunk;
    }
  }

  if (0 < bytesAhead) {
    copyGlobalGlobal_WarpUnrolled16
      <Unroll, /*DstAligned=*/false, /*SrcAligned=*/false, /*Partial=*/true>
        (lane, dstAddr, srcAddr, bytesAhead, scratchAddr);
  }
}
#endif // COMMON_KERNEL_H_
