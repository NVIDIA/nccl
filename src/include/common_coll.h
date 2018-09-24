/*************************************************************************
 * Copyright (c) 2016-2018, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef COMMON_COLL_H_
#define COMMON_COLL_H_

#include "core.h"
#include "enqueue.h"
#include "collectives/collectives.h"

static ncclResult_t PointerCheck(const void* pointer, struct ncclComm* comm, const char* ptrname, const char* opname) {
  cudaPointerAttributes attr;
  cudaError_t err = cudaPointerGetAttributes(&attr, pointer);
  if (err != cudaSuccess || attr.devicePointer == NULL) {
    WARN("%s : %s is not a valid pointer", opname, ptrname);
    return ncclInvalidArgument;
  }
#if __CUDACC_VER_MAJOR__ >= 10
  if (attr.type == cudaMemoryTypeDevice && attr.device != comm->cudaDev) {
#else
  if (attr.memoryType == cudaMemoryTypeDevice && attr.device != comm->cudaDev) {
#endif
    WARN("%s : %s allocated on device %d mismatchs with NCCL device %d", opname, ptrname, attr.device, comm->cudaDev);
    return ncclInvalidArgument;
  }
  return ncclSuccess;
}

static ncclResult_t PtrCheck(void* ptr, const char* opname, const char* ptrname) {
  if (ptr == NULL) {
    WARN("%s : %s argument is NULL", opname, ptrname);
    return ncclInvalidArgument;
  }
  return ncclSuccess;
}

static ncclResult_t ArgsCheck(const void* sendbuff, const void* recvbuff, size_t count, ncclDataType_t type, ncclRedOp_t op, int root, struct ncclComm* comm, const char* opname) {
  NCCLCHECK(PtrCheck(comm, opname, "comm"));
  // First, the easy ones
  if (root < 0 || root >= comm->nRanks) {
    WARN("%s : invalid root %d (root should be in the 0..%d range)", opname, root, comm->nRanks);
    return ncclInvalidArgument;
  }
  if (type < 0 || type >= ncclNumTypes) {
    WARN("%s : invalid type %d", opname, type);
    return ncclInvalidArgument;
  }
  if (op < 0 || op >= ncclNumOps) {
    WARN("%s : invalid reduction operation %d", opname, op);
    return ncclInvalidArgument;
  }

  if (comm->checkPointers) {
    // Check CUDA device pointers
    if (strcmp(opname, "Broadcast") != 0 || comm->rank == root) {
      NCCLCHECK(PointerCheck(sendbuff, comm, "sendbuff", opname));
    }
    if (strcmp(opname, "Reduce") != 0 || comm->rank == root) {
      NCCLCHECK(PointerCheck(recvbuff, comm, "recvbuff", opname));
    }
  }
  return ncclSuccess;
}

static __inline__ int ncclTypeSize(ncclDataType_t type) {
  switch (type) {
    case ncclInt8:
    case ncclUint8:
      return 1;
    case ncclFloat16:
      return 2;
    case ncclInt32:
    case ncclUint32:
    case ncclFloat32:
      return 4;
    case ncclInt64:
    case ncclUint64:
    case ncclFloat64:
      return 8;
    default:
      return -1;
  }
}

// In : comm, nbytes ; Out : nrings, nthreads, ll
// - We start with the minimum number of threads possible (64) and see if the size fits in LL;
//   If not, we increase the number of threads by 2x, until we reach the max number of LL threads (256, or set by user via NCCL_NTHREADS, or platform non-LL default)
// - We use "maxRings" to limit the max number of rings we can use before reaching the max number of LL threads
//   This ensures we don't use a large number of rings with a small number of threads
// - We use the NCCL_LL_RING_THRESHOLD as the per-thread threshold before we reach the max number of threads
//   we use NCCL_THREAD_THRESHOLD when we reach the max
// - If by the max number of LL threads, the size still cannot fit in LL, then we use non-LL setting
// - We honor the NCCL_LL_THRESHOLD (total threshold) set by user too
static inline void ncclGetCollResource(ncclComm_t comm, size_t nbytes, int* nrings, int* nthreads, int* ll) {
  *ll = 0;
  int llEnforced = 0; /* see if the size falls in the NCCL_LL_THRESHOLD range set by user */
  if (comm->llThreshold >= 0) { /* user sets total LL threshold */
    if (nbytes > comm->llThreshold) { /* non-LL */
      *nthreads = comm->nThreads+1;
      *nrings = comm->nRings;
      return;
    } else {
      llEnforced = 1; /* user wants to use LL */
    }
  }
  int nt = NCCL_LL_MIN_NTHREADS; /* start with min number of LL threads */
  size_t nr;
  int ll_max_nthreads = std::min(NCCL_LL_MAX_NTHREADS, comm->nThreads); /* respect user's setting or platform's default setting */
  int maxRings = (comm->nRanks <= 4) ? 1 : ll_max_nthreads / NCCL_LL_MIN_NTHREADS;
  ssize_t threshold = std::min(comm->threadThreshold, (ssize_t)NCCL_LL_RING_THRESHOLD);
  while (nt < ll_max_nthreads && *ll == 0) {
    nr = DIVUP(nbytes, (NCCL_LL_RING_THRESHOLD*nt*comm->nRanks));
    if (nr <= maxRings) { /* avoid using few threads but many rings */
      nr = nr == 0 ? 1 : nr > comm->nRings ? comm->nRings : nr;
      *ll = nbytes > comm->nRanks*nr*nt*threshold ? 0 : 1;
    }
    if (*ll == 0) {
      nt = nt << 1;
    }
  }
  if (*ll == 1) {
    *nthreads = nt;
    *nrings = (int)nr;
    return; /* we can use smaller number of threads to make LL work, stop here */
  }
  nr = DIVUP(nbytes, (NCCL_LL_RING_THRESHOLD*ll_max_nthreads*comm->nRanks)); /* else we try the max number of LL threads */
  nr = nr == 0 ? 1 : nr > comm->nRings ? comm->nRings : nr;
  *ll = nbytes > comm->nRanks*nr*ll_max_nthreads*comm->threadThreshold ? llEnforced : 1;
  *nthreads = *ll ? ll_max_nthreads : comm->nThreads+1;
  *nrings = *ll ? (int)nr : comm->nRings;
}

static ncclResult_t saveKernel(int coll, const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t dtype, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream, size_t nbytes, int loopFactor) {
  int llMode, nBlocks, nThreads;
  ncclGetCollResource(comm, nbytes, &nBlocks, &nThreads, &llMode);
  comm->myParams->blockDim.x = std::max((int)comm->myParams->blockDim.x, nThreads);
  if (comm->userStreamSet == false) {
    comm->userStream = stream;
    comm->userStreamSet = true;
  } else if (stream != comm->userStream) {
    WARN("Error : mixing different streams within a group call is not supported.");
    return ncclInvalidUsage;
  }
  int lastChunkSize = 0;
  if (llMode == 1) {
    int sliceSize = NCCL_LL_SLICE_LINES * sizeof(uint64_t) / ncclTypeSize(dtype);
    const ssize_t loopSize = nBlocks*loopFactor*(ssize_t)sliceSize;
    lastChunkSize = DIVUP((count-count/loopSize*loopSize), nBlocks*loopFactor);
    ALIGN_SIZE(lastChunkSize, nThreads*sizeof(uint64_t)/ncclTypeSize(dtype));
  }
  for (int bid=0; bid<nBlocks; bid++) {
    struct ncclRing* ring = comm->rings+(comm->myParams->gridDim.x % comm->nRings);
    if (ring->collCount == NCCL_MAX_OPS) {
      WARN("Too many aggregated operations (%d max)", NCCL_MAX_OPS);
      return ncclInvalidUsage;
    }

    comm->myParams->gridDim.x++;

    int opIndex = ring->collFifoTail;
    struct ncclColl* c = ring->collectives+opIndex;
    volatile uint8_t* activePtr = (volatile uint8_t*)&c->active;
    while (activePtr[0] != 0) sched_yield();

    struct CollectiveArgs* args = &c->args;
    args->root = root;
    args->N = count;
    args->ThisInput = sendbuff;
    args->ThisOutput = recvbuff;
    args->comm = comm->devComm;
    args->opCount = comm->opCount;
    args->bid = bid;
    args->nRings = nBlocks;
    args->nThreads = nThreads;
    args->lastChunkSize = lastChunkSize;

    c->nThreads = nThreads;
    c->funcIndex = FUNC_INDEX(coll, op, dtype, llMode);
    c->active = 1;
    opIndex = (opIndex+1)%NCCL_MAX_OPS;
    c->nextIndex = opIndex;
    ring->collFifoTail = opIndex;
    ring->collCount++;
  }
  /*if (llMode == 0)*/ comm->opCount++;
  return ncclSuccess;
}

extern __global__ void ncclMultiOpKernel (struct ncclColl firstColl);

#endif
