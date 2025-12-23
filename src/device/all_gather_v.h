/*************************************************************************
* Copyright (c) 2015-2022, NVIDIA CORPORATION. All rights reserved.
*
* See LICENSE.txt for license information
************************************************************************/

#include "device.h"
#include "collectives.h"
#include "primitives.h"
#include <stdio.h>

namespace {
  template<typename T, typename RedOp, typename Proto>
  __device__ __forceinline__ void setDataPtrsHelper(Primitives<T, RedOp, FanSymmetric<1>, 0, Proto, 0, 0>& prims,
                                                    void const* srcBuf, void* dstBuf, uint64_t redOpArg) {
    prims.setDataPtrs(srcBuf, dstBuf);
  }

  template<typename T, typename RedOp>
  __device__ __forceinline__ void setDataPtrsHelper(Primitives<T, RedOp, FanSymmetric<1>, 0, ProtoSimple<1,1>, 0, 0>& prims,
                                                    void const* srcBuf, void* dstBuf, uint64_t redOpArg) {
    prims.setDataPtrs(srcBuf, dstBuf, redOpArg, nullptr, 0, 0);
  }
}

template<typename T, typename RedOp, typename Proto>
__device__ __forceinline__ void runAllGatherV() {
  int tid = threadIdx.x;
  int tn = blockDim.x;
  ncclRing* ring = &ncclShmem.channel.ring;

  ncclDevWorkBcast *works = (ncclDevWorkBcast*)ncclShmem.workStorage;
  Primitives<int8_t, FuncSum<int8_t>, FanSymmetric<1>, /*Direct=*/0, Proto, 0>
    prims(tid, tn, &ring->prev, &ring->next, nullptr, nullptr, /*redOpArg=*/0);
  int w = 0;

  while (true) {
    int nWorks = ncclShmem.nWorks;
    int nRanks = ncclShmem.comm.nRanks;
    size_t bytes = works[w].bytes;
    int ringDepth = works[w].ringDepth;
    void* srcBuf = ringDepth==0 ? works[w].sendbuff : nullptr;
    void* dstBuf = works[w].recvbuff;
    bool inPlace = srcBuf == dstBuf;
	  size_t offset = works[w].bytes_done;
    setDataPtrsHelper(prims, (void const*)srcBuf, (void *)dstBuf, 0);

    __syncthreads();

    int wNext = (w+1 == nWorks) ? 0 : w+1;

    size_t chunkBytes = (size_t)works[w].chunkSize;
    size_t delta = min(bytes, chunkBytes);

    if (delta > 0) {
      if (ringDepth == 0) {
        if (inPlace) {
          prims.send(offset, delta);
        } else {
          prims.copySend(offset, offset, delta);
        }
      } else if (ringDepth == nRanks-1) {
        prims.recv(offset, delta);
      } else {
        prims.recvCopySend(offset, delta);
      }
    }

    if (tid == 0) {
      works[w].bytes -= delta;
      works[w].bytes_done += delta;
    }
    __syncthreads();

    int nr_done = 0;
    for (int i = 0; i < ncclShmem.nWorks; i++) {
      if (works[i].bytes == 0) {
        nr_done += 1;
      }
    }
    if (nr_done == ncclShmem.nWorks) {
      break;
    }
    w = wNext;
  }
}

// Specialized for broadcast
template<typename T, typename RedOp>
struct RunWorkBatch<ncclFuncAllGatherV, T, RedOp, NCCL_ALGO_RING, NCCL_PROTO_SIMPLE> {
  __device__ __forceinline__ void run() {
    using Proto = ProtoSimple<1,1>;
    runAllGatherV<T, RedOp, Proto>();
  }
};
template<typename T, typename RedOp>
struct RunWorkBatch<ncclFuncAllGatherV, T, RedOp, NCCL_ALGO_RING, NCCL_PROTO_LL> {
  __device__ __forceinline__ void run() {
    runAllGatherV<T, RedOp, ProtoLL>();
  }
};
template<typename T, typename RedOp>
struct RunWorkBatch<ncclFuncAllGatherV, T, RedOp, NCCL_ALGO_RING, NCCL_PROTO_LL128> {
  __device__ __forceinline__ void run() {
    runAllGatherV<T, RedOp, ProtoLL128>();
  }
};
