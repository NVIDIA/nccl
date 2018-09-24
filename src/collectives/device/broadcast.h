/*************************************************************************
 * Copyright (c) 2015-2018, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "core.h"
#include "primitives.h"
#include "collectives.h"

// Increase Step and boffset for buffer sync
#define NEXT_STEP \
  step++; \
  boffset += sliceSize; \
  if (boffset == buffSize) boffset = 0;

template<int UNROLL, class FUNC, typename T>
__device__ void ncclBroadcastKernel(struct CollectiveArgs* args) {
  const int tid = threadIdx.x;
  const int nthreads = blockDim.x - 1;
  const int bid = args->bid;
  __shared__ T* sharedNextOutput;
  struct ncclComm* comm = args->comm;
  struct ncclRing* ring = comm->rings+blockIdx.x;
  int prevdirect = ring->recv.conn.direct;
  int nextdirect = ring->send.conn.direct;

  WaitFlag waitDoneFromNext(ring->send.conn.head, (BROADCAST_BUFCHUNKS-1)*BROADCAST_SUBSTEPS);
  WaitFlag waitReadyFromPrev(ring->recv.conn.tail, 0);
  PostFlag postDoneToPrev(ring->recv.conn.head, 0, NULL, 0);
  PostFlag postReadyToNext(ring->send.conn.tail, 0, ring->send.conn.fifo, BROADCAST_BUFCHUNKS*BROADCAST_SUBSTEPS);

  typedef Primitives<UNROLL, BROADCAST_SUBSTEPS, T> Prims;

  const ssize_t size = args->N;
  const int buffSize = ring->buffSize / sizeof(T);
  const int sliceSize = buffSize / BROADCAST_BUFCHUNKS;
  const ssize_t loopSize = args->nRings*(ssize_t)sliceSize;
  const int rank = ring->devUserRanks[0];
  const int nextRank = ring->devUserRanks[1];
  const int root = args->root;

  if (tid == 0) {
    // Update in case we skipped some collectives
    *ring->recv.conn.opCount = args->opCount;
    if (nextRank != root) {
      // Wait for next to be ready
      WaitFlag waitOpCountNext(ring->send.conn.opCount, 0);
      waitOpCountNext.wait(args->opCount);
    }
    if (rank != root && prevdirect) {
      *ring->recv.conn.ptrExchange = args->ThisOutput;
    }
    if (nextRank != root && nextdirect) {
      void* volatile* ptr = &(ring->devMemSend->ptrExchange);
      while (*ptr == nullptr);
      sharedNextOutput = (T*)*ptr;
      *ptr = nullptr;
    }
  }
  __syncthreads();

  uint64_t step = 0ULL;
  int boffset = 0;

  // Compute pointers
  const T * __restrict__ thisInput = (const T*)args->ThisInput;
  T * __restrict__ thisOutput = (T*)args->ThisOutput;
  T * __restrict__ prevInput = (T*)ring->recv.conn.buff;
  T * __restrict__ nextOutput = (T*)ring->send.conn.buff;

  for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
    int chunkSize = min(sliceSize, DIVUP(size-gridOffset,args->nRings));
    ALIGN_SIZE(chunkSize, nthreads*sizeof(uint64_t)/sizeof(T));
    ssize_t offset = gridOffset + bid*chunkSize;
    int maxOffset = min(chunkSize, size-offset);

    if (rank == root) {
      if (thisInput == thisOutput) {
        Prims::Copy(tid, nthreads,
            thisInput  + offset,
            nextdirect ? (sharedNextOutput + offset) : (nextOutput + boffset),
            sliceSize, maxOffset,
            step,
            waitDoneFromNext,
            postReadyToNext);
      } else {
        Prims::DoubleCopy(tid, nthreads,
            thisInput  + offset,
            thisOutput + offset,
            nextdirect ? (sharedNextOutput + offset) : (nextOutput + boffset),
            sliceSize, maxOffset,
            step,
            waitDoneFromNext,
            postReadyToNext);
      }
    } else if (nextRank == root) {
      if (prevdirect) maxOffset = 0; // Only wait for signals
      Prims::Copy(tid, nthreads,
          prevInput  + boffset,
          thisOutput + offset,
          sliceSize, maxOffset,
          step,
          waitReadyFromPrev,
          postDoneToPrev);
    } else {
      if (prevdirect) {
        Prims::Copy(tid, nthreads,
            thisOutput + offset,
            nextdirect ? (sharedNextOutput + offset) : (nextOutput + boffset),
            sliceSize, maxOffset,
            step,
            waitDoneFromNext, waitReadyFromPrev,
            postReadyToNext, postDoneToPrev);
      } else {
        Prims::DoubleCopy(tid, nthreads,
            prevInput + boffset,
            thisOutput + offset,
            nextdirect ? (sharedNextOutput + offset) : (nextOutput + boffset),
            sliceSize, maxOffset,
            step,
            waitDoneFromNext, waitReadyFromPrev,
            postReadyToNext, postDoneToPrev);
      }
    }
    NEXT_STEP; // Increases step, boffset
  }

  if (tid == 0) {
    if (nextRank != root) {
      // Wait for next to have consumed data before resetting the flag
      waitDoneFromNext.wait(BROADCAST_SUBSTEPS*(step + BROADCAST_BUFCHUNKS - 1));
      *ring->send.conn.head = 0ULL;
    }
    *ring->recv.conn.tail = 0ULL;
    __threadfence_system();
    *ring->recv.conn.opCount = args->opCount+1;
  }
}

#include "ll_kernel.h"

#define NEXT_STEP_LL \
  boffset += NCCL_LL_SLICE_LINES; \
  if (boffset == NCCL_LL_BUFF_LINES) boffset = 0; \
  flag++; \
  step++;

template<int UNUSED, class FUNC, typename T>
__device__ void ncclBroadcastLLKernel(struct CollectiveArgs* args) {
  const int tid = threadIdx.x;
  const int bid = args->bid;
  const int llNthreads = args->nThreads;
  struct ncclComm* comm = args->comm;
  struct ncclRing* ring = comm->rings+blockIdx.x;
  volatile uint64_t * recvHeadPtr = ring->recv.conn.llHead;
  volatile uint64_t * sendHeadPtr = ring->send.conn.llHead;
  volatile int * sizesFifo = ring->send.conn.llFifo;
  uint64_t sendHead = sendHeadPtr[0];
  const int rank = comm->rank;
  const int nextRank = ring->devUserRanks[1];
  const int root = args->root;

  typedef LLPrimitives<T, FUNC> LL;

  const ssize_t size = args->N;
  ssize_t chunkSize = NCCL_LL_SLICE_LINES * sizeof(uint64_t) / sizeof(T);
  const ssize_t loopSize = args->nRings*chunkSize;

  uint64_t step = ring->send.conn.llStep;
  uint32_t flag = step + 1;
  int boffset = NCCL_LL_SLICE_LINES * STEP_TO_SLOT(step);

  // Compute pointers
  const T * __restrict__ thisInput = (const T*)args->ThisInput;
  T * __restrict__ thisOutput = (T*)args->ThisOutput;
  union ncclLLFifoLine * prevInput = (union ncclLLFifoLine *)ring->recv.conn.llBuff;
  union ncclLLFifoLine * nextOutput = (union ncclLLFifoLine *)ring->send.conn.llBuff;

  for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
    if (size-gridOffset < loopSize) {
      chunkSize = args->lastChunkSize;
    }
    ssize_t offset = gridOffset + bid*chunkSize;

    int maxOffset = min(chunkSize, size-offset);
    if (rank == root) {
      WAIT_NEXT;
      if (thisInput == thisOutput) {
        LL::ReduceCopy(
            thisInput + offset,
            nextOutput + boffset,
            maxOffset, flag, llNthreads);
      } else {
        LL::ReduceCopy(
            thisInput + offset,
            thisOutput + offset,
            nextOutput + boffset,
            maxOffset, flag, llNthreads);
      }
      POST_SIZE;
      NEXT_STEP_LL;
    } else if (nextRank == root) {
      LL::ReduceCopy(
          prevInput + boffset,
          thisOutput + offset,
          maxOffset, flag, llNthreads);
      NEXT_STEP_LL;
      ACK_PREV;
    } else {
      WAIT_NEXT;
      LL::ReduceCopy(
          prevInput + boffset,
          thisOutput + offset,
          nextOutput + boffset,
          maxOffset, flag, flag, llNthreads);
      POST_SIZE;
      NEXT_STEP_LL;
      ACK_PREV;
    }
  }

  // We need everyone to acknowledge data even if they didn't receive anything
  // so that the next collective can start right away.
  ACK_PREV;

  FIFO_CLEANING_AND_SAVE_STEP(flag);
}
