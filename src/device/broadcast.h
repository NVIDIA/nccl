/*************************************************************************
 * Copyright (c) 2015-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "device.h"
#include "collectives.h"
#include "primitives.h"

namespace {
  template<typename T, typename RedOp, typename Proto>
  __device__ __forceinline__ void runRing(int tid, int nthreads, struct ncclDevWorkColl* work) {
    ncclRing *ring = &ncclShmem.channel.ring;
    const int rank = ring->userRanks[0];
    const int nextRank = ring->userRanks[1];
    const int root = work->root;
    size_t chunkCount;
    size_t channelCount;
    size_t gridOffset;
    ncclCollCbdPart(work, ncclShmem.channelId, Proto::Id, sizeof(T), (size_t*)nullptr, &gridOffset, &channelCount, &chunkCount);
    size_t offset;
    int nelem;

    T *inputBuf = (T*)work->sendbuff;
    T *outputBuf = (T*)work->recvbuff;
    Primitives<T, RedOp, FanSymmetric<1>, 0, Proto, 0>
      prims(tid, nthreads, &ring->prev, &ring->next, inputBuf, outputBuf, work->redOpArg);

    for (size_t elemOffset = 0; elemOffset < channelCount; elemOffset += chunkCount) {
      offset = gridOffset + elemOffset;
      nelem = min(chunkCount, channelCount - elemOffset);

      if (rank == root) {
        if (inputBuf == outputBuf) {
          prims.send(offset, nelem);
        } else {
          prims.copySend(offset, offset, nelem);
        }
      } else if (nextRank == root) {
        prims.recv(offset, nelem);
      } else {
        prims.recvCopySend(offset, nelem);
      }
    }
  }
}

template<typename T, typename RedOp>
struct RunWorkColl<ncclFuncBroadcast, T, RedOp, NCCL_ALGO_RING, NCCL_PROTO_SIMPLE> {
  __device__ __forceinline__ void run(int tid, int nthreads, struct ncclDevWorkColl* work) {
    using Proto = ProtoSimple<BROADCAST_CHUNKSTEPS/BROADCAST_SLICESTEPS, BROADCAST_SLICESTEPS>;
    runRing<T, RedOp, Proto>(tid, nthreads, work);
  }
};

template<typename T, typename RedOp>
struct RunWorkColl<ncclFuncBroadcast, T, RedOp, NCCL_ALGO_RING, NCCL_PROTO_LL> {
  __device__ __forceinline__ void run(int tid, int nthreads, struct ncclDevWorkColl* work) {
    runRing<T, RedOp, ProtoLL>(tid, nthreads, work);
  }
};

template<typename T, typename RedOp>
struct RunWorkColl<ncclFuncBroadcast, T, RedOp, NCCL_ALGO_RING, NCCL_PROTO_LL128> {
  __device__ __forceinline__ void run(int tid, int nthreads, struct ncclDevWorkColl* work) {
    runRing<T, RedOp, ProtoLL128>(tid, nthreads, work);
  }
};
