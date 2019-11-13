/*************************************************************************
 * Copyright (c) 2015-2019, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "devcomm.h"
#include "primitives.h"
#include "collectives.h"

template<int UNROLL, class FUNC, typename T>
__device__ void ncclAllGatherRingKernel(struct CollectiveArgs* args) {
  const int tid = threadIdx.x;
  const int nthreads = args->nThreads-WARP_SIZE;
  const int bid = args->bid;
  struct ncclDevComm* comm = args->comm;
  struct ncclChannel* channel = comm->channels+blockIdx.x;
  struct ncclRing* ring = &channel->ring;
  const ssize_t size = args->N;
  const int nranks = comm->nRanks;
  const int stepSize = channel->buffSize / (sizeof(T)*NCCL_STEPS);
  const int chunkSize = stepSize * ALLGATHER_CHUNKSTEPS;
  const ssize_t loopSize = args->nChannels*(ssize_t)chunkSize;

  // Compute pointers
  const T * __restrict__ thisInput = (const T*)args->ThisInput;
  T * __restrict__ thisOutput = (T*)args->ThisOutput;

  ncclPrimitives<UNROLL, ALLGATHER_CHUNKSTEPS/ALLGATHER_SLICESTEPS, ALLGATHER_SLICESTEPS, T, 1, 1, FUNC>
    prims(tid, args->nThreads, &ring->prev, &ring->next, thisOutput, stepSize, channel, comm, args->opCount);

  for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
    int realChunkSize = min(chunkSize, DIVUP(size-gridOffset,args->nChannels));
    ALIGN_SIZE(realChunkSize, nthreads*sizeof(uint64_t)/sizeof(T));
    ssize_t chunkOffset = gridOffset + bid*realChunkSize;

    /////////////// begin AllGather steps ///////////////
    ssize_t offset;
    int nelem = min(realChunkSize, size-chunkOffset);
    int rankDest;

    // step 0: push data to next GPU
    rankDest = ring->devUserRanks[0];
    offset = chunkOffset + rankDest * size;

    if (thisInput + chunkOffset == thisOutput + offset) { // In place
      prims.directSend(thisInput+chunkOffset, offset, nelem);
    } else {
      prims.directCopySend(thisInput+chunkOffset, thisOutput+offset, offset, nelem);
    }

    // k-2 steps: copy to next GPU
    for (int j=1; j<nranks-1; ++j) {
      rankDest = ring->devUserRanks[nranks-j];
      offset = chunkOffset + rankDest * size;

      prims.directRecvCopySend(thisOutput+offset, offset, nelem);
    }

    // Make final copy from buffer to dest.
    rankDest = ring->devUserRanks[1];
    offset = chunkOffset + rankDest * size;

    // Final wait/copy.
    prims.directRecv(thisOutput+offset, offset, nelem);
  }
}

template<int UNROLL, class FUNC, typename T>
__device__ void ncclAllGatherTreeKernel(struct CollectiveArgs* args) { }

template<int UNUSED, class FUNC, typename T>
__device__ void ncclAllGatherRingLLKernel(struct CollectiveArgs* args) {
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

    /////////////// begin AllGather steps ///////////////
    ssize_t offset;
    int nelem = min(chunkSize, size-chunkOffset);
    int rankDest;

    // step 0: push data to next GPU
    rankDest = ring->devUserRanks[0];
    offset = chunkOffset + rankDest * size;

    if (thisInput + chunkOffset == thisOutput + offset) { // In place
      LLprims.send(thisInput+chunkOffset, nelem);
    } else {
      LLprims.copySend(thisInput+chunkOffset, thisOutput+offset, nelem);
    }

    // k-2 steps: copy to next GPU
    for (int j=1; j<nranks-1; ++j) {
      rankDest = ring->devUserRanks[nranks-j];
      offset = chunkOffset + rankDest * size;

      LLprims.recvCopySend(thisOutput+offset, nelem);
    }

    // step k-1: final store
    rankDest = ring->devUserRanks[1];
    offset = chunkOffset + rankDest * size;

    LLprims.recv(thisOutput+offset, nelem);
  }
}

template<int UNUSED, class FUNC, typename T>
__device__ void ncclAllGatherTreeLLKernel(struct CollectiveArgs* args) { }

#include "prims_ll128.h"
template<int UNUSED, class FUNC, typename T>
__device__ void ncclAllGatherRingLL128Kernel(struct CollectiveArgs* args) {
  const int tid = threadIdx.x;
  const int bid = args->bid;
  const int nthreads = args->nThreads;
  struct ncclDevComm* comm = args->comm;
  struct ncclChannel* channel = comm->channels+blockIdx.x;
  struct ncclRing* ring = &channel->ring;

  ncclLL128Primitives<T, FUNC, 1, 1> LLprims(tid, nthreads, &ring->prev, &ring->next, channel, comm, args->opCount);

  const ssize_t size = args->N;
  //const int rank = comm->rank;
  const int nranks = comm->nRanks;
  ssize_t chunkSize = (NCCL_LL128_ELEMS_PER_THREAD*nthreads*NCCL_LL128_DATAELEMS*sizeof(uint64_t))/(NCCL_LL128_LINEELEMS*sizeof(T));
  // We should not need the final /2 but it makes performance much, much smoother. Might be a bug somewhere.
  const ssize_t minChunkSize = (NCCL_LL128_SHMEM_ELEMS_PER_THREAD*nthreads*NCCL_LL128_DATAELEMS*sizeof(uint64_t))/(NCCL_LL128_LINEELEMS*sizeof(T))/2;

  const ssize_t loopSize = args->nChannels*chunkSize;

  // Compute pointers
  const T * __restrict__ thisInput = (const T*)args->ThisInput;
  T * __restrict__ thisOutput = (T*)args->ThisOutput;

  for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
    chunkSize = min(DIVUP(size-gridOffset, args->nChannels*minChunkSize)*minChunkSize, chunkSize);

    ssize_t chunkOffset = gridOffset + bid*chunkSize;

    /////////////// begin AllGather steps ///////////////
    ssize_t offset;
    int nelem = min(chunkSize, size-chunkOffset);
    int rankDest;

    // step 0: push data to next GPU
    rankDest = ring->devUserRanks[0];
    offset = chunkOffset + rankDest * size;

    if (thisInput + chunkOffset == thisOutput + offset) { // In place
      LLprims.send(thisInput+chunkOffset, nelem);
    } else {
      LLprims.copySend(thisInput+chunkOffset, thisOutput+offset, nelem);
    }

    // k-2 steps: copy to next GPU
    for (int j=1; j<nranks-1; ++j) {
      rankDest = ring->devUserRanks[nranks-j];
      offset = chunkOffset + rankDest * size;

      LLprims.recvCopySend(thisOutput+offset, nelem);
    }

    // step k-1: final store
    rankDest = ring->devUserRanks[1];
    offset = chunkOffset + rankDest * size;

    LLprims.recv(thisOutput+offset, nelem);
  }
}

template<int UNUSED, class FUNC, typename T>
__device__ void ncclAllGatherTreeLL128Kernel(struct CollectiveArgs* args) { }
