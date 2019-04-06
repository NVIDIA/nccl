/*************************************************************************
 * Copyright (c) 2015-2019, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "devcomm.h"
#include "primitives.h"
#include "collectives.h"

template<int UNROLL, class FUNC, typename T>
__device__ void ncclAllReduceRingKernel(struct CollectiveArgs* args) {
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

  ncclPrimitives<UNROLL, ALLREDUCE_CHUNKSTEPS/ALLREDUCE_SLICESTEPS, ALLREDUCE_SLICESTEPS, T, 1, 1, FUNC>
    prims(tid, nthreads, &ring->prev, &ring->next, thisOutput, stepSize, channel, comm, args->opCount);

  for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += nranks*loopSize) {
    int realChunkSize = min(chunkSize, DIVUP(size-gridOffset,nranks*args->nChannels));
    ALIGN_SIZE(realChunkSize, nthreads*sizeof(uint64_t)/sizeof(T));
    ssize_t chunkOffset = gridOffset + bid*nranks*realChunkSize;

    /////////////// begin AllReduce steps ///////////////
    ssize_t offset;
    int nelem;
    int slice;

    // step 0: push data to next GPU
    slice = ring->devUserRanks[nranks-1];
    offset = chunkOffset + slice * realChunkSize;
    nelem = min(realChunkSize, size-offset);

    prims.send(thisInput+offset, nelem);

    // k-2 steps: reduce and copy to next GPU
    for (int j=2; j<nranks; ++j) {
      slice = ring->devUserRanks[nranks-j];
      offset = chunkOffset + slice * realChunkSize;
      nelem = min(realChunkSize, size-offset);

      prims.recvReduceSend(thisInput+offset, nelem);
    }

    // step k-1: reduce this buffer and data, which will produce the final
    // result that we store in this data and push to the next GPU
    slice = ring->devUserRanks[0];
    offset = chunkOffset + slice * realChunkSize;
    nelem = min(realChunkSize, size-offset);

    prims.directRecvReduceCopySend(thisInput+offset, thisOutput+offset, offset, nelem);

    // k-2 steps: copy to next GPU
    for (int j=1; j<nranks-1; ++j) {
      slice = ring->devUserRanks[nranks-j];
      offset = chunkOffset + slice * realChunkSize;
      nelem = min(realChunkSize, size-offset);

      prims.directRecvCopySend(thisOutput+offset, offset, nelem);
    }

    // Make final copy from buffer to dest.
    slice = ring->devUserRanks[1];
    offset = chunkOffset + slice * realChunkSize;
    nelem = min(realChunkSize, size-offset);

    // Final wait/copy.
    prims.directRecv(thisOutput+offset, offset, nelem);
  }
}

template<int UNROLL, class FUNC, typename T>
__device__ void ncclAllReduceTreeKernel(struct CollectiveArgs* args) {
  const int tid = threadIdx.x;
  const int nthreads = blockDim.x - 1;
  const int bid = args->bid;
  struct ncclDevComm* comm = args->comm;
  struct ncclChannel* channel = comm->channels+blockIdx.x;
  struct ncclTree* tree = &channel->tree;
  const ssize_t size = args->N;
  const int stepSize = channel->buffSize / (sizeof(T)*NCCL_STEPS);
  const int chunkSize = args->lastChunkSize;
  const ssize_t loopSize = args->nChannels*chunkSize;

  // Compute pointers
  const T * __restrict__ thisInput = (const T*)args->ThisInput;
  T * __restrict__ thisOutput = (T*)args->ThisOutput;

  do {
    // Reduce : max number of recv is 3, max number of send is 1 (binary tree + local)
    ncclPrimitives<UNROLL, 1, 1, T, NCCL_MAX_TREE_ARITY, 1, FUNC> prims(tid, nthreads, tree->down, &tree->up, NULL, stepSize, channel, comm, args->opCount);
    for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
      // Up
      ssize_t offset = gridOffset + bid*chunkSize;
      int nelem = min(chunkSize, size-offset);
      if (tree->up == -1) {
        prims.recvReduceCopy(thisInput+offset, thisOutput+offset, nelem);
      } else if (tree->down[0] == -1) {
        prims.send(thisInput+offset, nelem);
      } else {
        prims.recvReduceSend(thisInput+offset, nelem);
      }
    }
  } while(0);

  do {
    // Broadcast : max number of recv is 1, max number of send is 3 (binary tree + local)
    ncclPrimitives<UNROLL, 1, 1, T, 1, NCCL_MAX_TREE_ARITY, FUNC> prims(tid, nthreads, &tree->up, tree->down, NULL, stepSize, channel, comm, args->opCount);
    for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
      // Down
      ssize_t offset = gridOffset + bid*chunkSize;
      int nelem = min(chunkSize, size-offset);
      if (tree->up == -1) {
        prims.send(thisOutput+offset, nelem);
      } else if (tree->down[0] == -1) {
        prims.recv(thisOutput+offset, nelem);
      } else {
        prims.recvCopySend(thisOutput+offset, nelem);
      }
    }
  } while(0);
}

template<int UNUSED, class FUNC, typename T>
__device__ void ncclAllReduceRingLLKernel(struct CollectiveArgs* args) {
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
  const ssize_t loopSize = args->nChannels*nranks*chunkSize;

  // Compute pointers
  const T * __restrict__ thisInput = (const T*)args->ThisInput;
  T * __restrict__ thisOutput = (T*)args->ThisOutput;

  for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
    if (size-gridOffset < loopSize) {
      chunkSize = args->lastChunkSize;
    }
    ssize_t chunkOffset = gridOffset + bid*nranks*chunkSize;

    /////////////// begin AllReduce steps ///////////////
    ssize_t offset;
    int nelem;
    int slice;

    // step 0: push data to next GPU
    slice = ring->devUserRanks[nranks-1];
    offset = chunkOffset + slice * chunkSize;
    nelem = min(chunkSize, size-offset);

    LLprims.send(thisInput+offset, nelem);

    // k-2 steps: reduce and copy to next GPU
    for (int j=2; j<nranks; ++j) {
      slice = ring->devUserRanks[nranks-j];
      offset = chunkOffset + slice * chunkSize;
      nelem = min(chunkSize, size-offset);

      LLprims.recvReduceSend(thisInput+offset, nelem);
    }

    // step k-1: reduce this buffer and data, which will produce the final
    // result that we store in this data and push to the next GPU
    slice = ring->devUserRanks[0];
    offset = chunkOffset + slice * chunkSize;
    nelem = min(chunkSize, size-offset);

    LLprims.recvReduceCopySend(thisInput+offset, thisOutput+offset, nelem);

    // k-2 steps: copy to next GPU
    for (int j=1; j<nranks-1; ++j) {
      slice = ring->devUserRanks[nranks-j];
      offset = chunkOffset + slice * chunkSize;
      nelem = min(chunkSize, size-offset);

      LLprims.recvCopySend(thisOutput+offset, nelem);
    }

    // Make final copy from buffer to dest.
    slice = ring->devUserRanks[1];
    offset = chunkOffset + slice * chunkSize;
    nelem = min(chunkSize, size-offset);

    // Here we need to copy from buffer to this output.
    LLprims.recv(thisOutput+offset, nelem);
  }
}

template<int UNUSED, class FUNC, typename T>
__device__ void ncclAllReduceTreeLLKernel(struct CollectiveArgs* args) {
  const int tid = threadIdx.x;
  const int nthreads = args->nThreads;
  const int bid = args->bid;
  struct ncclDevComm* comm = args->comm;
  struct ncclChannel* channel = comm->channels+blockIdx.x;
  struct ncclTree* tree = &channel->tree;
  const ssize_t size = args->N;
  ssize_t chunkSize = NCCL_LL_SLICE_LINES * sizeof(uint64_t) / sizeof(T);
  const ssize_t loopSize = args->nChannels*chunkSize;

  // Compute pointers
  const T * __restrict__ thisInput = (const T*)args->ThisInput;
  T * __restrict__ thisOutput = (T*)args->ThisOutput;

  do {
    // Reduce : max number of recv is 3, max number of send is 1 (binary tree + local)
    ncclLLPrimitives<T, FUNC, NCCL_MAX_TREE_ARITY, 1> LLprims(tid, nthreads, tree->down, &tree->up, channel, comm, args->opCount);
    for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
      // Up
      ssize_t offset = gridOffset + bid*chunkSize;
      int nelem = min(chunkSize, size-offset);
      if (tree->up == -1) {
        LLprims.recvReduceCopy(thisInput+offset, thisOutput+offset, nelem);
      } else if (tree->down[0] == -1) {
        LLprims.send(thisInput+offset, nelem);
      } else {
        LLprims.recvReduceSend(thisInput+offset, nelem);
      }
    }
  } while(0);

  do {
    // Broadcast : max number of recv is 1, max number of send is 3 (binary tree + local)
    ncclLLPrimitives<T, FUNC, 1, NCCL_MAX_TREE_ARITY> LLprims(tid, nthreads, &tree->up, tree->down, channel, comm, args->opCount);
    for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
      // Down
      ssize_t offset = gridOffset + bid*chunkSize;
      int nelem = min(chunkSize, size-offset);
      if (tree->up == -1) {
        LLprims.send(thisOutput+offset, nelem);
      } else if (tree->down[0] == -1) {
        LLprims.recv(thisOutput+offset, nelem);
      } else {
        LLprims.recvCopySend(thisOutput+offset, nelem);
      }
    }
  } while(0);
}
