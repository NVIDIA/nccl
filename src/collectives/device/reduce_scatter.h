/*************************************************************************
 * Copyright (c) 2015-2019, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "devcomm.h"
#include "primitives.h"
#include "collectives.h"

template<int UNROLL, class FUNC, typename T>
__device__ void ncclReduceScatterRingKernel(struct CollectiveArgs* args) {
  const int tid = threadIdx.x;
  const int nthreads = blockDim.x - 1;
  const int bid = args->bid;
  struct ncclDevComm* comm = args->comm;
  struct ncclChannel* channel = comm->channels+blockIdx.x;
  struct ncclRing* ring = &channel->ring;
  const ssize_t size = args->N;
  const int nranks = comm->nRanks;
  const int stepSize = channel->buffSize / (sizeof(T)*NCCL_STEPS);
  const int chunkSize = stepSize * ALLREDUCE_CHUNKSTEPS;
  const ssize_t loopSize = args->nChannels*(ssize_t)chunkSize;

  // Compute pointers
  const T * __restrict__ thisInput = (const T*)args->ThisInput;
  T * __restrict__ thisOutput = (T*)args->ThisOutput;

  ncclPrimitives<UNROLL, REDUCESCATTER_CHUNKSTEPS/REDUCESCATTER_SLICESTEPS, REDUCESCATTER_SLICESTEPS, T, 1, 1, FUNC>
    prims(tid, nthreads, &ring->prev, &ring->next, NULL, stepSize, channel, comm, args->opCount);

  for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
    int realChunkSize = min(chunkSize, DIVUP(size-gridOffset,args->nChannels));
    ALIGN_SIZE(realChunkSize, nthreads*sizeof(uint64_t)/sizeof(T));
    ssize_t chunkOffset = gridOffset + bid*realChunkSize;

    /////////////// begin ReduceScatter steps ///////////////
    ssize_t offset;
    int nelem = min(realChunkSize, size-chunkOffset);
    int rankDest;

    // step 0: push data to next GPU
    rankDest = ring->devUserRanks[nranks-1];
    offset = chunkOffset + rankDest * size;

    prims.send(thisInput+offset, nelem);

    // k-2 steps: reduce and copy to next GPU
    for (int j=2; j<nranks; ++j) {
      rankDest = ring->devUserRanks[nranks-j];
      offset = chunkOffset + rankDest * size;

      prims.recvReduceSend(thisInput+offset, nelem);
    }

    // step k-1: reduce this buffer and data, which will produce the final result
    rankDest = ring->devUserRanks[0];
    offset = chunkOffset + rankDest * size;

    prims.recvReduceCopy(thisInput+offset, thisOutput+chunkOffset, nelem);
  }
}

template<int UNROLL, class FUNC, typename T>
__device__ void ncclReduceScatterTreeKernel(struct CollectiveArgs* args) { }

template<int UNUSED, class FUNC, typename T>
__device__ void ncclReduceScatterRingLLKernel(struct CollectiveArgs* args) {
  const int tid = threadIdx.x;
  const int bid = args->bid;
  const int nthreads = args->nThreads;
  struct ncclDevComm* comm = args->comm;
  struct ncclChannel* channel = comm->channels+blockIdx.x;
  struct ncclRing* ring = &channel->ring;

  ncclLLPrimitives<T, FUNC, 1, 1> LLprims(tid, nthreads, &ring->prev, &ring->next, channel, comm, args->opCount);

  const ssize_t size = args->N;
  //const int rank = comm->rank;
  const int nranks = comm->nRanks;
  ssize_t chunkSize = NCCL_LL_SLICE_LINES * sizeof(uint64_t) / sizeof(T);
  const ssize_t loopSize = args->nChannels*chunkSize;

  // Compute pointers
  const T * __restrict__ thisInput = (const T*)args->ThisInput;
  T * __restrict__ thisOutput = (T*)args->ThisOutput;

  for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
    if (size-gridOffset < loopSize) {
      chunkSize = args->lastChunkSize;
    }
    ssize_t chunkOffset = gridOffset + bid*chunkSize;

    /////////////// begin ReduceScatter steps ///////////////
    ssize_t offset;
    int nelem = min(chunkSize, size-chunkOffset);
    int rankDest;

    // step 0: push data to next GPU
    rankDest = ring->devUserRanks[nranks-1];
    offset = chunkOffset + rankDest * size;

    LLprims.send(thisInput+offset, nelem);

    // k-2 steps: reduce and copy to next GPU
    for (int j=2; j<nranks; ++j) {
      rankDest = ring->devUserRanks[nranks-j];
      offset = chunkOffset + rankDest * size;

      LLprims.recvReduceSend(thisInput+offset, nelem);
    }

    // step k-1: reduce this buffer and data, which will produce the final
    // result that we store in this data
    rankDest = ring->devUserRanks[0];
    offset = chunkOffset + rankDest * size;

    LLprims.recvReduceCopy(thisInput+offset, thisOutput+chunkOffset, nelem);
  }
}

template<int UNUSED, class FUNC, typename T>
__device__ void ncclReduceScatterTreeLLKernel(struct CollectiveArgs* args) { }
