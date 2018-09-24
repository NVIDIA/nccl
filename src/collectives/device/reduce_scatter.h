/*************************************************************************
 * Copyright (c) 2015-2018, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "core.h"
#include "primitives.h"
#include "collectives.h"

// Increase Step and poffset/noffset for buffer sync
#define NEXT_STEP \
  step++; \
  poffset = noffset; \
  noffset += sliceSize; \
  if (noffset == buffSize) noffset = 0;

template<int UNROLL, class FUNC, typename T>
__device__ void ncclReduceScatterKernel(struct CollectiveArgs* args) {
  const int tid = threadIdx.x;
  const int nthreads = blockDim.x - 1;
  const int bid = args->bid;
  struct ncclComm* comm = args->comm;
  struct ncclRing* ring = comm->rings+blockIdx.x;

  WaitFlag waitDoneFromNext(ring->send.conn.head, REDUCESCATTER_BUFCHUNKS*REDUCESCATTER_SUBSTEPS);
  WaitFlag waitReadyFromPrev(ring->recv.conn.tail, REDUCESCATTER_SUBSTEPS);
  PostFlag postDoneToPrev(ring->recv.conn.head, REDUCESCATTER_SUBSTEPS, NULL, 0);
  PostFlag postReadyToNext(ring->send.conn.tail, 0, ring->send.conn.fifo, REDUCESCATTER_BUFCHUNKS*REDUCESCATTER_SUBSTEPS);

  typedef Primitives<UNROLL, REDUCESCATTER_SUBSTEPS, T, FUNC> Prims;

  const ssize_t size = args->N;
  const int nranks = comm->nRanks;
  const int buffSize = ring->buffSize / sizeof(T);
  const int sliceSize = buffSize / REDUCESCATTER_BUFCHUNKS;
  const ssize_t loopSize = args->nRings*(ssize_t)sliceSize;

  if (tid == 0) {
    // Update in case we skipped some collectives
    *ring->recv.conn.opCount = args->opCount;
    // Wait for next to be ready
    WaitFlag waitOpCountNext(ring->send.conn.opCount, 0);
    waitOpCountNext.wait(args->opCount);
  }
  __syncthreads();

  uint64_t step = 0ULL;
  int poffset, noffset = 0;

  // Compute pointers
  const T * __restrict__ thisInput = (const T*)args->ThisInput;
  T * __restrict__ thisOutput = (T*)args->ThisOutput;
  T * __restrict__ prevInput = (T*)ring->recv.conn.buff;
  T * __restrict__ nextOutput = (T*)ring->send.conn.buff;

  for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
    int chunkSize = min(sliceSize, DIVUP(size-gridOffset,args->nRings));
    ALIGN_SIZE(chunkSize, nthreads*sizeof(uint64_t)/sizeof(T));
    ssize_t chunkOffset = gridOffset + bid*chunkSize;

    /////////////// begin ReduceScatter steps ///////////////
    ssize_t offset;
    int maxOffset = min(chunkSize, size-chunkOffset);
    int rankDest;

    // step 0: push data to next GPU
    rankDest = ring->devUserRanks[nranks-1];
    offset = chunkOffset + rankDest * size;

    Prims::Copy(tid, nthreads,
        thisInput  + offset,
        nextOutput + noffset,
        sliceSize, maxOffset,
        step,
        waitDoneFromNext,
        postReadyToNext);

    NEXT_STEP; // Increases step, poffset, noffset

    // k-2 steps: reduce and copy to next GPU
    for (int j=2; j<nranks; ++j) {
      rankDest = ring->devUserRanks[nranks-j];
      offset = chunkOffset + rankDest * size;

      Prims::Reduce(tid, nthreads,
          prevInput  + poffset,
          thisInput  + offset,
          nextOutput + noffset,
          sliceSize, maxOffset,
          step,
          waitDoneFromNext, waitReadyFromPrev,
          postReadyToNext, postDoneToPrev);

      NEXT_STEP;
    }

    // step k-1: reduce this buffer and data, which will produce the final
    // result that we store in this data and push to the next GPU
    rankDest = ring->devUserRanks[0];
    offset = chunkOffset + rankDest * size;

    Prims::Reduce(tid, nthreads,
        prevInput  + poffset,
        thisInput  + offset,
        thisOutput + chunkOffset,
        sliceSize, maxOffset,
        step,
        waitReadyFromPrev,
        postDoneToPrev);
  }

  if (tid == 0) {
    waitDoneFromNext.wait(REDUCESCATTER_SUBSTEPS*(step + REDUCESCATTER_BUFCHUNKS));
    *ring->send.conn.head = 0ULL;
    *ring->recv.conn.tail = 0ULL;
    __threadfence_system();
    *ring->recv.conn.opCount = args->opCount+1;
  }
}

#include "ll_kernel.h"

#define NEXT_STEP_LL \
  poffset = noffset; \
  pflag = nflag; \
  noffset += NCCL_LL_SLICE_LINES; \
  if (noffset == NCCL_LL_BUFF_LINES) { noffset = 0; } \
  nflag++; \
  step++;

template<int UNUSED, class FUNC, typename T>
__device__ void ncclReduceScatterLLKernel(struct CollectiveArgs* args) {
  const int tid = threadIdx.x;
  const int bid = args->bid;
  const int llNthreads = args->nThreads;
  struct ncclComm* comm = args->comm;
  struct ncclRing* ring = comm->rings+blockIdx.x;
  volatile uint64_t * recvHeadPtr = ring->recv.conn.llHead;
  volatile uint64_t * sendHeadPtr = ring->send.conn.llHead;
  volatile int * sizesFifo = ring->send.conn.llFifo;
  uint64_t sendHead = sendHeadPtr[0];

  typedef LLPrimitives<T, FUNC> LL;

  const ssize_t size = args->N;
  //const int rank = comm->rank;
  const int nranks = comm->nRanks;
  ssize_t chunkSize = NCCL_LL_SLICE_LINES * sizeof(uint64_t) / sizeof(T);
  const ssize_t loopSize = args->nRings*chunkSize;

  uint64_t step = ring->send.conn.llStep;
  uint32_t pflag, nflag = step + 1;
  int poffset, noffset = NCCL_LL_SLICE_LINES * STEP_TO_SLOT(step);

  // Compute pointers
  const T * __restrict__ thisInput = (const T*)args->ThisInput;
  T * __restrict__ thisOutput = (T*)args->ThisOutput;
  union ncclLLFifoLine * prevInput = (union ncclLLFifoLine *)ring->recv.conn.llBuff;
  union ncclLLFifoLine * nextOutput = (union ncclLLFifoLine *)ring->send.conn.llBuff;

  for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
    if (size-gridOffset < loopSize) {
      chunkSize = args->lastChunkSize;
    }
    ssize_t chunkOffset = gridOffset + bid*chunkSize;

    /////////////// begin ReduceScatter steps ///////////////
    ssize_t offset;
    int maxOffset = min(chunkSize, size-chunkOffset);
    int rankDest;

    // step 0: push data to next GPU
    rankDest = ring->devUserRanks[nranks-1];
    offset = chunkOffset + rankDest * size;

    WAIT_NEXT;
    LL::ReduceCopy(
        thisInput  + offset,
        nextOutput + noffset,
        maxOffset, nflag, llNthreads);
    POST_SIZE;

    NEXT_STEP_LL;

    // k-2 steps: reduce and copy to next GPU
    for (int j=2; j<nranks; ++j) {
      rankDest = ring->devUserRanks[nranks-j];
      offset = chunkOffset + rankDest * size;

      WAIT_NEXT;
      LL::ReduceCopy(
          thisInput  + offset,
          prevInput  + poffset,
          nextOutput + noffset,
          maxOffset, pflag, nflag, llNthreads);
      POST_SIZE;
      ACK_PREV;

      NEXT_STEP_LL;
    }

    // step k-1: reduce this buffer and data, which will produce the final
    // result that we store in this data
    rankDest = ring->devUserRanks[0];
    offset = chunkOffset + rankDest * size;

    LL::ReduceCopy(
        thisInput  + offset,
        prevInput  + poffset,
        thisOutput + chunkOffset,
        maxOffset, pflag, llNthreads);
    ACK_PREV;
  }

  FIFO_CLEANING_AND_SAVE_STEP(nflag);
}
