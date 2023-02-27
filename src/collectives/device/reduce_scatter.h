/*************************************************************************
 * Copyright (c) 2015-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "devcomm.h"
#include "collectives.h"
#include "primitives.h"

namespace {
  template<typename T, typename RedOp, typename Proto>
  __device__ __forceinline__ void runRing(ncclWorkElem *args) {
    const int tid = threadIdx.x;
    const int nthreads = args->nWarps*WARP_SIZE;
    const int bid = args->bid;
    const int nChannels = args->nChannels;
    ncclRing *ring = &ncclShmem.channel.ring;
    int const *ringRanks = ring->userRanks;
    const ssize_t chunkSize = int(Proto::calcBytePerStep()/sizeof(T) * (Proto::Id == NCCL_PROTO_SIMPLE ? REDUCESCATTER_CHUNKSTEPS : 1));
    // We should not need the final /2 but it makes performance much, much smoother. Might be a bug somewhere.
    const ssize_t minChunkSizeLL128 = int(nthreads*(Proto::calcBytePerGrain()/sizeof(T))/2);
    const int nranks = ncclShmem.comm.nRanks;
    const ssize_t loopSize = nChannels*chunkSize;
    const ssize_t size = args->count;

    Primitives<T, RedOp, FanSymmetric<1>, 0, Proto, 0>
      prims(tid, nthreads, &ring->prev, &ring->next, args->sendbuff, args->recvbuff, args->redOpArg);

    for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
      ssize_t realChunkSize;
      if (Proto::Id == NCCL_PROTO_SIMPLE) {
        realChunkSize = min(chunkSize, divUp(size-gridOffset, nChannels));
        realChunkSize = roundUp(realChunkSize, (nthreads-WARP_SIZE)*sizeof(uint64_t)/sizeof(T));
      }
      else if (Proto::Id == NCCL_PROTO_LL)
        realChunkSize = size-gridOffset < loopSize ? args->lastChunkSize : chunkSize;
      else if (Proto::Id == NCCL_PROTO_LL128)
        realChunkSize = min(divUp(size-gridOffset, nChannels*minChunkSizeLL128)*minChunkSizeLL128, chunkSize);
      realChunkSize = int(realChunkSize);

      ssize_t chunkOffset = gridOffset + bid*int(realChunkSize);

      /////////////// begin ReduceScatter steps ///////////////
      ssize_t offset;
      int nelem = min(realChunkSize, size-chunkOffset);
      int rankDest;

      // step 0: push data to next GPU
      rankDest = ringRanks[nranks-1];
      offset = chunkOffset + rankDest * size;
      prims.send(offset, nelem);

      // k-2 steps: reduce and copy to next GPU
      for (int j=2; j<nranks; ++j) {
        rankDest = ringRanks[nranks-j];
        offset = chunkOffset + rankDest * size;
        prims.recvReduceSend(offset, nelem);
      }

      // step k-1: reduce this buffer and data, which will produce the final result
      rankDest = ringRanks[0];
      offset = chunkOffset + rankDest * size;
      prims.recvReduceCopy(offset, chunkOffset, nelem, /*postOp=*/true);
    }
  }
}

template<typename T, typename RedOp>
struct RunWorkElement<ncclFuncReduceScatter, T, RedOp, NCCL_ALGO_RING, NCCL_PROTO_SIMPLE> {
  __device__ __forceinline__ void run(ncclWorkElem *args) {
    using Proto = ProtoSimple<REDUCESCATTER_CHUNKSTEPS/REDUCESCATTER_SLICESTEPS, REDUCESCATTER_SLICESTEPS>;
    runRing<T, RedOp, Proto>(args);
  }
};

template<typename T, typename RedOp>
struct RunWorkElement<ncclFuncReduceScatter, T, RedOp, NCCL_ALGO_RING, NCCL_PROTO_LL> {
  __device__ __forceinline__ void run(ncclWorkElem *args) {
    runRing<T, RedOp, ProtoLL>(args);
  }
};

template<typename T, typename RedOp>
struct RunWorkElement<ncclFuncReduceScatter, T, RedOp, NCCL_ALGO_RING, NCCL_PROTO_LL128> {
  __device__ __forceinline__ void run(ncclWorkElem *args) {
    runRing<T, RedOp, ProtoLL128>(args);
  }
};

template<typename T, typename RedOp>
struct RunWorkElement<ncclFuncReduceScatter, T, RedOp, NCCL_ALGO_NVLS, NCCL_PROTO_SIMPLE> {
  __device__ __forceinline__ void run(ncclWorkElem *args) {
    const int tid = threadIdx.x;
    const int bid = args->bid;
    const int nChannels = args->nChannels;
    struct ncclNvls* nvls = &ncclShmem.channel.nvls;
    const ssize_t chunkSize = int(args->lastChunkSize);
    const ssize_t size = args->count;
    const ssize_t loopSize = nChannels*chunkSize;

    const int nThreadsScatter = 128 + WARP_SIZE;
    const int nThreadsReduce = 384;
    const int tidEndScatter = nThreadsScatter;
    const int tidEndReduce = tidEndScatter + nThreadsReduce;

    using Proto = ProtoSimple<1, 1>;

    if (tid < tidEndScatter) {
      // Scatter
      int group = (0*Proto::MaxGroupWidth) | (0<<16);
      Primitives<T, RedOp, FanAsymmetric<0, NCCL_MAX_NVLS_ARITY>, /*Direct=*/0, Proto, 0>
        prims(tid, nThreadsScatter, NULL, nvls->up, args->sendbuff, NULL, args->redOpArg, group, args);
      for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
        ssize_t offset = gridOffset + bid*chunkSize;
        int nelem = min(chunkSize, size-offset);
        prims.scatter(offset, nvls->nHeads*size, nelem, size, -1, 0);
      }
    } else if (tid < tidEndReduce) {
      int group = (3*Proto::MaxGroupWidth) | (1<<16);
      // Reduce through MC
      Primitives<T, RedOp, FanAsymmetric<1, 0>, /*Direct=*/0, Proto, 0>
        prims(tid-tidEndScatter, nThreadsReduce, &nvls->down, NULL, NULL, args->recvbuff, args->redOpArg, group, args);
      for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
        ssize_t offset = gridOffset + bid*chunkSize;
        int nelem = min(chunkSize, size-offset);
        prims.recv(offset, nelem);
      }
    }
  }
};
