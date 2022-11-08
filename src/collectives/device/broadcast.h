/*************************************************************************
 * Copyright (c) 2015-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "devcomm.h"
#include "collectives.h"
#include "primitives.h"

namespace {
  template<typename Proto>
  __device__ __forceinline__ void runBcasts() {
    int tid = threadIdx.x;
    int tn = blockDim.x;
    ncclRing* ring = &ncclShmem.channel.ring;
    ncclDevWorkBcast* works = (ncclDevWorkBcast*)(&ncclShmem.workBatch+1);
    Primitives<int8_t, FuncSum<int8_t>, FanSymmetric<1>, /*Direct=*/0, Proto, 0>
      prims(tid, tn, &ring->prev, &ring->next, nullptr, nullptr, /*redOpArg=*/0);
    int w = 0; // `works[]` index of the current broadcaster.
    int wPrev = -1; // Value of `w` for previous loop iteration.
    bool wPrevIsEmpty = false; // Local cache of `works[wPrev].bytes==0`

    while (true) {
      int nWorks = ncclShmem.workBatch.nWorks;
      int nRanks = ncclShmem.comm.nRanks;
      size_t bytes = works[w].bytes;
      int chunkBytes = (Proto::SlicePerChunk)*ncclShmem.workBatch.bcast.sliceBytes;
      int ringDepth = works[w].ringDepth; // How far down the ring am I from this broadcaster
      void* srcBuf = ringDepth==0 ? ncclShmem.workBatch.bcast.rootSrcBuf : nullptr;
      void* dstBuf = works[w].dstBuf;
      bool inPlace = srcBuf == dstBuf;
      prims.setDataPtrs(srcBuf, dstBuf);

      // The current broadcaster we're going to work on is element `w`. Here
      // we compute the next broadcaster `wNext` which has work remaining. We
      // are careful to compare against `wPrev` since `tid=0` could still be
      // updating `works[wPrev]` in the previous loop iteration.
      int wNext = (w+1 == nWorks) ? 0 : w+1;
      while ((wNext==wPrev) ? wPrevIsEmpty : (works[wNext].bytes == 0)) {
        wNext = (wNext+1 == nWorks) ? 0 : wNext+1;
      }
      // The gap is the ring distance between `w` and `wNext`. This is the
      // number of chunks we'll be expecting from broadcaster `w`. If `w` is the
      // only remaining broadcaster (thus `w==next`) then we'll process all
      // remaining chunks by setting ringGap=INT_MAX.
      int ringGap = INT_MAX;
      if (w != wNext) {
        ringGap = works[wNext].ringDepth - ringDepth;
        if (ringGap <= 0) ringGap += nRanks;
      }

      // The following loop does data movement for the current broadcaster. We
      // expect another chunk for every hop in the gap. An important side effect
      // of the data movement primitives is that they include a thread barrier.
      // We rely on this barrier when we write to shmem later on.
      size_t offset = 0;
      do {
        int delta = min(bytes, (size_t)chunkBytes);
        if (ringDepth == 0) { // This rank is broadcast root
          if (inPlace) {
            prims.send(offset, delta); // barrier
          } else {
            prims.copySend(offset, offset, delta); // barrier
          }
        } else if (ringDepth == nRanks-1) { // Downstream neighbor is broadcast root
          prims.recv(offset, delta); // barrier
        } else {
          prims.recvCopySend(offset, delta); // barrier
        }
        bytes -= delta;
        offset += delta;
        ringGap -= 1;
      } while (ringGap != 0 && bytes != 0);
      // ...a thread barrier has happened.

      // If the next broadcaster is the current one, then we just completed all
      // work since ringGap was INT_MAX.
      if (wNext == w) break;

      if (tid == 0) {
        // Since there was a thread barrier in the primitive operation above, we
        // know that these writes to shmem are safe since all threads have moved
        // on to reading from works[wNext], and we know wNext cannot be equal to w.
        works[w].bytes = bytes;
        works[w].dstBuf = (char*)works[w].dstBuf + offset;
        if (ringDepth == 0) {
          ncclShmem.workBatch.bcast.rootSrcBuf = (char*)ncclShmem.workBatch.bcast.rootSrcBuf + offset;
        }
      }
      // The `tid!=0` threads in the next loop iteration must avoid racing with
      // `tid==0` above since it is updating `works[wPrev]`. From the perspective
      // of the next iteration, the threads are interested in `works[wPrev].bytes==0`,
      // so we compute it in a local boolean here to be carried forward.
      wPrevIsEmpty = (bytes == 0);
      wPrev = w;
      w = wNext;
    }
  }
}

template<typename T, typename RedOp>
struct RunWorkBatch<ncclFuncBroadcast, T, RedOp, NCCL_ALGO_RING, NCCL_PROTO_SIMPLE> {
  __device__ __forceinline__ void run() {
    using Proto = ProtoSimple</*SlicePerChunk      =*/NCCL_STEPS/2,
                              /*StepPerSlice       =*/1,
                              /*Unroll             =*/ncclCollUnroll(),
                              /*Multimem{Srcs,Dsts}=*/0, 0,
                              /*MinimizeNumSlices  =*/true>;
    runBcasts<Proto>();
  }
};

template<typename T, typename RedOp>
struct RunWorkBatch<ncclFuncBroadcast, T, RedOp, NCCL_ALGO_RING, NCCL_PROTO_LL> {
  __device__ __forceinline__ void run() {
    runBcasts<ProtoLL>();
  }
};

template<typename T, typename RedOp>
struct RunWorkBatch<ncclFuncBroadcast, T, RedOp, NCCL_ALGO_RING, NCCL_PROTO_LL128> {
  __device__ __forceinline__ void run() {
    runBcasts<ProtoLL128>();
  }
};
