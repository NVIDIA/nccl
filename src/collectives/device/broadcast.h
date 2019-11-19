/*************************************************************************
 * Copyright (c) 2015-2019, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "devcomm.h"
#include "primitives.h"
#include "collectives.h"

template<int UNROLL, class FUNC, typename T>
__device__ void ncclBroadcastRingKernel(struct CollectiveArgs* args) {
  const int tid = threadIdx.x;
  const int nthreads = args->nThreads-WARP_SIZE;
  const int bid = args->bid;
  struct ncclDevComm* comm = args->comm;
  struct ncclChannel* channel = comm->channels+blockIdx.x;
  struct ncclRing* ring = &channel->ring;
  const ssize_t size = args->N;
  const int stepSize = channel->buffSize / (sizeof(T)*NCCL_STEPS);
  const int chunkSize = stepSize * BROADCAST_CHUNKSTEPS;
  const ssize_t loopSize = args->nChannels*(ssize_t)chunkSize;
  const int rank = ring->devUserRanks[0];
  const int nextRank = ring->devUserRanks[1];
  const int root = args->root;

  // Compute pointers
  const T * __restrict__ thisInput = (const T*)args->ThisInput;
  T * __restrict__ thisOutput = (T*)args->ThisOutput;

  ncclPrimitives<UNROLL, BROADCAST_CHUNKSTEPS/BROADCAST_SLICESTEPS, BROADCAST_SLICESTEPS, T, 1, 1, FUNC>
    prims(tid, args->nThreads, &ring->prev, &ring->next, NULL, stepSize, channel, comm, args->opCount);

  for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
    int realChunkSize = min(chunkSize, DIVUP(size-gridOffset,args->nChannels));
    ALIGN_SIZE(realChunkSize, nthreads*sizeof(uint64_t)/sizeof(T));
    ssize_t offset = gridOffset + bid*realChunkSize;
    int nelem = min(realChunkSize, size-offset);

    if (rank == root) {
      if (thisInput == thisOutput) {
        prims.send(thisInput+offset, nelem);
      } else {
        prims.copySend(thisInput+offset, thisOutput+offset, nelem);
      }
    } else if (nextRank == root) {
      prims.recv(thisOutput+offset, nelem);
    } else {
      prims.recvCopySend(thisOutput+offset, nelem);
    }
  }
}

template<int UNROLL, class FUNC, typename T>
__device__ void ncclBroadcastTreeKernel(struct CollectiveArgs* args) { }

template<int UNUSED, class FUNC, typename T>
__device__ void ncclBroadcastRingLLKernel(struct CollectiveArgs* args) {
  const int tid = threadIdx.x;
  const int bid = args->bid;
  const int nthreads = args->nThreads;
  struct ncclDevComm* comm = args->comm;
  struct ncclChannel* channel = comm->channels+blockIdx.x;
  struct ncclRing* ring = &channel->ring;

  ncclLLPrimitives<T, FUNC, 1, 1> LLprims(tid, nthreads, &ring->prev, &ring->next, channel, comm, args->opCount);

  const ssize_t size = args->N;
  const int rank = ring->devUserRanks[0];
  const int nextRank = ring->devUserRanks[1];
  const int root = args->root;

  ssize_t chunkSize = NCCL_LL_SLICE_LINES * sizeof(uint64_t) / sizeof(T);
  const ssize_t loopSize = args->nChannels*chunkSize;

  // Compute pointers
  const T * __restrict__ thisInput = (const T*)args->ThisInput;
  T * __restrict__ thisOutput = (T*)args->ThisOutput;

  for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
    if (size-gridOffset < loopSize) {
      chunkSize = args->lastChunkSize;
    }
    ssize_t offset = gridOffset + bid*chunkSize;

    int nelem = min(chunkSize, size-offset);
    if (rank == root) {
      if (thisInput == thisOutput) {
        LLprims.send(thisInput+offset, nelem);
      } else {
        LLprims.copySend(thisInput + offset, thisOutput + offset, nelem);
      }
    } else if (nextRank == root) {
      LLprims.recv(thisOutput + offset, nelem);
    } else {
      LLprims.recvCopySend(thisOutput + offset, nelem);
    }
  }
}

template<int UNUSED, class FUNC, typename T>
__device__ void ncclBroadcastTreeLLKernel(struct CollectiveArgs* args) { }

#include "prims_ll128.h"
template<int UNUSED, class FUNC, typename T>
__device__ void ncclBroadcastRingLL128Kernel(struct CollectiveArgs* args) {
  const int tid = threadIdx.x;
  const int bid = args->bid;
  const int nthreads = args->nThreads;
  struct ncclDevComm* comm = args->comm;
  struct ncclChannel* channel = comm->channels+blockIdx.x;
  struct ncclRing* ring = &channel->ring;

  ncclLL128Primitives<T, FUNC, 1, 1> LLprims(tid, nthreads, &ring->prev, &ring->next, channel, comm, args->opCount);

  const ssize_t size = args->N;
  const int rank = ring->devUserRanks[0];
  const int nextRank = ring->devUserRanks[1];
  const int root = args->root;

  ssize_t chunkSize = (NCCL_LL128_ELEMS_PER_THREAD*nthreads*NCCL_LL128_DATAELEMS*sizeof(uint64_t))/(NCCL_LL128_LINEELEMS*sizeof(T));
  const ssize_t minChunkSize = (NCCL_LL128_SHMEM_ELEMS_PER_THREAD*nthreads*NCCL_LL128_DATAELEMS*sizeof(uint64_t))/(NCCL_LL128_LINEELEMS*sizeof(T));

  const ssize_t loopSize = args->nChannels*chunkSize;

  // Compute pointers
  const T * __restrict__ thisInput = (const T*)args->ThisInput;
  T * __restrict__ thisOutput = (T*)args->ThisOutput;

  for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
    chunkSize = min(DIVUP(size-gridOffset, args->nChannels*minChunkSize)*minChunkSize, chunkSize);
    ssize_t offset = gridOffset + bid*chunkSize;

    int nelem = min(chunkSize, size-offset);
    if (rank == root) {
      if (thisInput == thisOutput) {
        LLprims.send(thisInput+offset, nelem);
      } else {
        LLprims.copySend(thisInput + offset, thisOutput + offset, nelem);
      }
    } else if (nextRank == root) {
      LLprims.recv(thisOutput + offset, nelem);
    } else {
      LLprims.recvCopySend(thisOutput + offset, nelem);
    }
  }
}

template<int UNUSED, class FUNC, typename T>
__device__ void ncclBroadcastTreeLL128Kernel(struct CollectiveArgs* args) { }
