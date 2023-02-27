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
#include <cstdio>
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
         int MinSrcs, int MaxSrcs, int MinDsts, int MaxDsts, int PreOpSrcs,
         typename IntBytes>
__device__ __forceinline__ void reduceCopyPacks(
    int nThreads, int &thread,
    uint64_t redArg, uint64_t *preOpArgs, bool postOp,
    int nSrcs, void **srcPtrs, int nDsts, void **dstPtrs,
    IntBytes &nBytesBehind, IntBytes &nBytesAhead
  ) {
  static_assert(std::is_signed<IntBytes>::value, "IntBytes must be a signed integral type.");

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
  IntBytes nHunksAhead = nBytesAhead/BytePerHunk;
  // Advance collective position.
  nBytesBehind += nHunksAhead*BytePerHunk;
  nBytesAhead -= nHunksAhead*BytePerHunk;
  if (Unroll==1 && BytePerPack <= nBytesAhead) {
    // Only Unroll=1 can do partial hunks (where not all threads partake).
    nHunksAhead += 1;
    nBytesBehind += nBytesAhead - (nBytesAhead%BytePerPack);
    nBytesAhead = nBytesAhead%BytePerPack;
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

  // We dictate loop termination condition according to whether partial hunks
  // can be handled or not.
  while (Unroll==1 ? (BytePerPack <= threadBytesAhead) : (0 < nHunksAhead)) {
    BytePack<BytePerPack> acc[Unroll];

    { RedFn preFn(0 < PreOpSrcs ? preOpArgs[0] : 0);
      #pragma unroll Unroll
      for (int u=0; u < Unroll; u++) {
        // Use volatile loads in case credits are polled for with volatile (instead of acquire).
        acc[u] = ld_volatile_global<BytePerPack>(minSrcs[0]);
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
        // Use volatile loads in case credits are polled for with volatile (instead of acquire).
        tmp[u] = ld_volatile_global<BytePerPack>(minSrcs[s]);
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
        st_global<BytePerPack>(minDsts[d], acc[u]);
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
         int MinSrcs, int MaxSrcs, int MinDsts, int MaxDsts, int PreOpSrcs,
         typename IntBytes>
__device__ __forceinline__ void ReduceOrCopyMulti(
    int thread, int nThreads,
    uint64_t redArg, uint64_t *preOpArgs, bool postOp,
    int nSrcs, void **srcPtrs, int nDsts, void **dstPtrs,
    IntBytes nElts
  ) {
  //int nWarps = nThreads/WARP_SIZE;
  //int warp = thread/WARP_SIZE;
  int lane = thread%WARP_SIZE;

  // Check that all is 16B aligned. If not don't use 16B load/stores.
  int aligned = 1;
  if (lane < nSrcs) aligned &= 0 == cvta_to_global(srcPtrs[lane])%16;
  if (lane < nDsts) aligned &= 0 == cvta_to_global(dstPtrs[lane])%16;
  aligned = __all_sync(~0u, aligned);

  IntBytes nBytesBehind = 0;
  IntBytes nBytesAhead = nElts*sizeof(T);
  if (aligned) {
    reduceCopyPacks<RedFn, T, Unroll, /*BytePerPack=*/16,
      MinSrcs, MaxSrcs, MinDsts, MaxDsts, PreOpSrcs>
      (nThreads, /*&*/thread, redArg, preOpArgs, postOp,
       nSrcs, srcPtrs, nDsts, dstPtrs, /*&*/nBytesBehind, /*&*/nBytesAhead);
    if (nBytesAhead == 0) return;

    reduceCopyPacks<RedFn, T, /*Unroll=*/1, /*BytePerPack=*/16,
      MinSrcs, MaxSrcs, MinDsts, MaxDsts, PreOpSrcs>
      (nThreads, /*&*/thread, redArg, preOpArgs, postOp,
       nSrcs, srcPtrs, nDsts, dstPtrs, /*&*/nBytesBehind, /*&*/nBytesAhead);
    if (nBytesAhead == 0) return;
  }

  reduceCopyPacks<RedFn, T, Unroll*(16/sizeof(T))/2, /*BytePerPack=*/sizeof(T),
    MinSrcs, MaxSrcs, MinDsts, MaxDsts, PreOpSrcs>
    (nThreads, /*&*/thread, redArg, preOpArgs, postOp,
     nSrcs, srcPtrs, nDsts, dstPtrs, /*&*/nBytesBehind, /*&*/nBytesAhead);
  if (nBytesAhead == 0) return;

  reduceCopyPacks<RedFn, T, /*Unroll=*/1, /*BytePerPack=*/sizeof(T),
    MinSrcs, MaxSrcs, MinDsts, MaxDsts, PreOpSrcs>
    (nThreads, /*&*/thread, redArg, preOpArgs, postOp,
     nSrcs, srcPtrs, nDsts, dstPtrs, /*&*/nBytesBehind, /*&*/nBytesAhead);
}

// Copies from srcAddr to dstAddr using multimem load/store. The amount copied
// will be at most Unroll*BytePerPack*WARP_SIZE. If Partial=1, then the amount
// will be the min() of that and nBytesAhead. If srcAddr is not BytePerPack
// aligned then the amount copied will be less by (srcAddr%BytePerPack) since
// we begin loads at the first pack containing the first element.
template<typename RedFn, typename T, int Unroll, int BytePerPack,
         bool SrcAligned, // is srcAddr aligned to BytePerPack
         bool DstAligned, // are dstAddr and nBytesAhead both aligned to BytePerPack
         bool Partial, // is this a possibly partial hunk
         typename IntBytes>
__device__ __forceinline__ void copyMultimemMultimem_WarpUnrolled(
    int lane, RedFn redFn, bool postOp, uintptr_t srcAddr, uintptr_t dstAddr,
    IntBytes nBytesAhead, uint32_t scratchAddr
  ) {
  int srcMisalign = SrcAligned ? 0 : srcAddr%BytePerPack;
  srcAddr -= srcMisalign;

  BytePack<BytePerPack> reg[Unroll];
  int offset = lane*BytePerPack;
  #pragma unroll Unroll
  for (int u=0; u < Unroll; u++) {
    if (!Partial || (offset < srcMisalign + nBytesAhead)) {
      reg[u] = applyLoadMultimem(redFn, srcAddr+offset);
      if (postOp) reg[u] = applyPostOp(redFn, reg[u]);
    }
    offset += WARP_SIZE*BytePerPack;
  }

  if (SrcAligned && DstAligned) {
    offset = lane*BytePerPack;
    #pragma unroll Unroll
    for (int u=0; u < Unroll; u++) {
      if (!Partial || offset < nBytesAhead) {
        multimem_st_global<BytePerPack>(dstAddr+offset, reg[u]);
      }
      offset += WARP_SIZE*BytePerPack;
    }
  } else {
    __syncwarp();
    offset = lane*BytePerPack;
    #pragma unroll Unroll
    for (int u=0; u < Unroll; u++) {
      if (!Partial || (offset < srcMisalign + nBytesAhead)) {
        st_shared<BytePerPack>(scratchAddr+offset, reg[u]);
      }
      offset += WARP_SIZE*BytePerPack;
    }
    __syncwarp();
    if (!SrcAligned) {
      // Ignore the beginning of the first pack corresponding to bytes overread
      // due to misalignment.
      nBytesAhead = min(nBytesAhead, Unroll*WARP_SIZE*BytePerPack - srcMisalign);
    }
    copyGlobalShared_WarpUnrolled
      <sizeof(T), /*MaxBytes=*/Unroll*WARP_SIZE*BytePerPack, /*Multimem=*/1>
        (lane, dstAddr, scratchAddr+srcMisalign, nBytesAhead);
  }
}

// copyMultimemMultimem_IfEnabled has two overloads: the enabled case whose first arg
// has type `std::true_type` and the disabled case with first arg `std::false_type`.
// This is to guard the template instantiations of Apply_LoadMultimem on types/ops where
// they aren't supported. A nicer approach is to use C++17's "if constexpr".
template<typename RedFn, typename IntBytes>
__device__ __forceinline__ void copyMultimemMultimem_IfEnabled(
    std::false_type enabled/*=false*/,
    int thread, int nThreads, uint64_t redArg, bool postOp,
    void *srcPtr, void *dstPtr, IntBytes nElts, uint32_t warpScratchAddr
  ) {
  // nop
}

template<typename RedFn, typename IntBytes>
__device__ __forceinline__ void copyMultimemMultimem_IfEnabled(
    std::true_type enabled/*=true*/,
    int thread, int nThreads, uint64_t redArg, bool postOp,
    void *srcPtr, void *dstPtr, IntBytes nElts, uint32_t warpScratchAddr
  ) {
  static_assert(std::is_signed<IntBytes>::value, "IntBytes must be a signed integral type.");

  constexpr int BytePerPack = Apply_LoadMultimem<RedFn>::PackSize;
  using T = typename RedFn::EltType;
  constexpr int Unroll = ncclNvlsUnroll(BytePerPack);
  constexpr int BytePerHunk = Unroll*WARP_SIZE*BytePerPack;
  int nWarps = nThreads/WARP_SIZE;
  int warp = thread/WARP_SIZE;
  int lane = thread%WARP_SIZE;
  RedFn redFn(redArg);

  uintptr_t srcAddr = cvta_to_global(srcPtr);
  uintptr_t dstAddr = cvta_to_global(dstPtr);
  IntBytes warpBytesAhead = nElts*sizeof(T);
  bool partialHunkIsFront;

  // First handle misalignment of srcAddr.
  if ((BytePerPack != sizeof(T)) && (srcAddr%BytePerPack != 0)) {
    // If srcAddr isn't pack aligned then the first hunk processed will be short
    // the same number of bytes as srcAddr's misalignment.
    if (warp == 0) {
      partialHunkIsFront = true;
      goto PartialHunk; // "call" PartialHunk()
    PartialHunkFrontReturn:
      warp = nWarps;
    }
    warp -= 1; // Rotate warp numbers for load balancing
    int advanced = BytePerHunk-(srcAddr%BytePerPack); // since copyMultimemMultimem_WarpUnrolled shorts by the misalignment
    srcAddr += advanced; // srcAddr is now pack aligned
    dstAddr += advanced;
    warpBytesAhead -= advanced;
  }

  warpBytesAhead -= warp*BytePerHunk;
  srcAddr += warp*BytePerHunk;
  dstAddr += warp*BytePerHunk;
  // Now that srcAddr is pack aligned detect if dstAddr is pack aligned.
  if ((BytePerPack == sizeof(T)) || (dstAddr%BytePerPack == 0)) {
    while (BytePerHunk <= warpBytesAhead) {
      copyMultimemMultimem_WarpUnrolled
        <RedFn, T, Unroll, BytePerPack, /*SrcAligned=*/true, /*DstAligned=*/true, /*Partial=*/false>
          (lane, redFn, postOp, srcAddr, dstAddr, warpBytesAhead, warpScratchAddr);
      srcAddr += nWarps*BytePerHunk;
      dstAddr += nWarps*BytePerHunk;
      warpBytesAhead -= nWarps*BytePerHunk;
    }
  } else {
    while (BytePerHunk <= warpBytesAhead) {
      copyMultimemMultimem_WarpUnrolled
        <RedFn, T, Unroll, BytePerPack, /*SrcAligned=*/true, /*DstAligned=*/false, /*Partial=*/false>
          (lane, redFn, postOp, srcAddr, dstAddr, warpBytesAhead, warpScratchAddr);
      srcAddr += nWarps*BytePerHunk;
      dstAddr += nWarps*BytePerHunk;
      warpBytesAhead -= nWarps*BytePerHunk;
    }
  }

  if (0 < warpBytesAhead) {
    partialHunkIsFront = false;
    goto PartialHunk; // "call" PartialHunk()
  PartialHunkBackReturn:;
  }
  return;

PartialHunk:
  // We have to handle a partial hunk possibly at the front and back of the
  // buffer. We generate the code once here since its a lot of instructions,
  // and then simulate function calls with gotos.
  copyMultimemMultimem_WarpUnrolled
    <RedFn, T, Unroll, BytePerPack, /*SrcAligned=*/false, /*DstAligned=*/false, /*Partial=*/true>
      (lane, redFn, postOp, srcAddr, dstAddr, warpBytesAhead, warpScratchAddr);
  if (partialHunkIsFront) goto PartialHunkFrontReturn;
  goto PartialHunkBackReturn;
}

template<typename RedFn, typename IntBytes>
__device__ __forceinline__ void copyMultimemMultimem(
    int thread, int nThreads, uint64_t redArg, bool postOp,
    void *srcPtr, void *dstPtr, IntBytes nElts, uint32_t warpScratchAddr
  ) {
  constexpr bool Enabled = Apply_LoadMultimem<RedFn>::PackSize != 0;
  copyMultimemMultimem_IfEnabled<RedFn>(
    /*enabled=*/std::integral_constant<bool, Enabled>(),
    thread, nThreads, redArg, postOp, srcPtr, dstPtr, nElts, warpScratchAddr);
}
#endif // COMMON_KERNEL_H_
