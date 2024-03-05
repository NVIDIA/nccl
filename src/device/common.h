/*************************************************************************
 * Copyright (c) 2017-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_DEVICE_COMMON_H_
#define NCCL_DEVICE_COMMON_H_

#include "collectives.h"
#include "device.h"
#include "op128.h"
#include "network/unpack/unpack_defs.h"

#define COLL_UNROLL (ncclCollUnroll())

typedef void(*ncclDevFuncPtr_t)();
extern __device__ ncclDevFuncPtr_t const ncclDevFuncTable[];

struct ncclShmemGroup {
  ncclConnInfo *recvConns[NCCL_MAX_NVLS_ARITY];
  ncclConnInfo *sendConns[NCCL_MAX_NVLS_ARITY];
  void* srcs[NCCL_MAX_NVLS_ARITY+1];
  void* dsts[NCCL_MAX_NVLS_ARITY+1];
  union {
    unpackGroupShmem unpack;
  } devicePlugin;
  int32_t dstSizes[NCCL_MAX_NVLS_ARITY+1];
};

struct ncclShmemData {
  struct ncclShmemGroup groups[NCCL_MAX_GROUPS];
  uint64_t redOpArgs[NCCL_MAX_NVLS_ARITY+1];
  int channelId;
  int aborted;
  alignas(16) struct ncclDevComm comm;
  alignas(16) struct ncclDevChannel channel;
  alignas(16) struct ncclWork work;
  alignas(16) union {
    unpackShmem unpack;
  } devicePlugin;
};
static_assert(offsetof(struct ncclShmemData, work)%16 == 0, "shmem.work needs to be 16B aligned");

extern __shared__ ncclShmemData ncclShmem;
#if __CUDA_ARCH__ >= 700
  extern __shared__ ulong2 ncclShmemPerWarp[/*ncclShmemDynamicSize()/sizeof(ulong2)*/];
#else
  extern __shared__ ulong2 ncclShmemPerWarp[ncclShmemScratchWarpSize()*(NCCL_MAX_NTHREADS/WARP_SIZE)/sizeof(ulong2)];
#endif

__device__ inline void* ncclScratchForWarp(int warp) {
  return (char*)ncclShmemPerWarp + warp*ncclShmemScratchWarpSize();
}

__device__ inline bool barrierReduceAny(int bit) {
  uint32_t popc;
  asm ("{"
    ".reg .pred barr_pred;"
    "setp.eq.u32 barr_pred, %1, 1;"
    "bar.red.popc.u32 %0, 2, barr_pred;"
  "}" : "=r"(popc) : "r"(bit));
  return popc != 0;
}

// Copy 16-byte aligned data. You must call with at least `(bytes+15)/16` threads.
inline __device__ void copyToShmem16(int tid, void* dst, void const* src, int bytes) {
  int offset = 16*tid;
  if (offset < bytes) {
    uint64_t a=0, b=0;
    asm("ld.v2.u64 {%0,%1},[%2];" : "=l"(a),"=l"(b) : "l"((char const*)src + offset));
    asm volatile("st.v2.u64 [%0],{%1,%2};" :: "l"((char*)dst + offset), "l"(a), "l"(b));
  }
}

template<ncclFunc_t Fn, typename T, typename RedOp, int Algo, int Proto>
struct RunWorkElement {
  __device__ void run(ncclWorkElem*) {
    // Put NOT IMPLEMENTED behavior here.
  }
};

template<ncclFunc_t Fn, typename T, typename RedOp, int Algo, int Proto>
struct RunWork {
  // This __forceinline__ is necessary. The compiler was inserting a function call
  // here from the LL ncclKernel.
  __device__ __forceinline__ void run(ncclWork *w) {
    int wid = threadIdx.x / WARP_SIZE;
    ncclWorkElem* we = w->header.type == ncclWorkTypeRegColl ? &w->regElems[0].elem : &w->elems[0];
    int stride = w->header.type == ncclWorkTypeRegColl ? sizeof(ncclWorkElemReg) : sizeof(ncclWorkElem);
    #pragma unroll 1
    while ((char*)we + stride <= (char*)(w+1) && we->isUsed) {
      if (wid < we->nWarps) {
        RunWorkElement<Fn, T, RedOp, Algo, Proto>().run(we);
      }
      we = (ncclWorkElem*)((char*)we + stride);
    }
  }
};

static __device__ void ncclRedopPtrDeref(struct ncclWorkElem* we) {
  if (we->isUsed && we->redOpArgIsPtr) {
    /* redOpArg is a pointer to the scalar value, so we'll dereference it
     * here so that redOpArg holds the bits of the scalar going forward.
     * The tricky thing is we don't know its type T since that's encoded in
     * the funcIndex. Because it would be difficult to get sizeof(T) from
     * funcIndex, we'll cheat and just dereference the largest possible size
     * given the alignment of the pointer. We might be reading in more bytes
     * than we need but that's harmless.
     */
    if (we->redOpArg%2 != 0)
      we->redOpArg = *reinterpret_cast<uint8_t*>(we->redOpArg);
    else if (we->redOpArg%4 != 0)
      we->redOpArg = *reinterpret_cast<uint16_t*>(we->redOpArg);
    else if (we->redOpArg%8 != 0)
      we->redOpArg = *reinterpret_cast<uint32_t*>(we->redOpArg);
    else
      we->redOpArg = *reinterpret_cast<uint64_t*>(we->redOpArg);
  }
}

template<int SpecializedFnId, typename SpecializedRunWork>
__device__ void ncclKernelMain(struct ncclDevComm* comm, uint64_t channelMask, struct ncclWork* workHead) {
  int tid = threadIdx.x;

  // To map blockId to channelId, we need the n'th set bit of channelMask which
  // is the inverse of counting the number of set bits among the the first n.
  if (tid < WARP_SIZE) {
    int x = tid;
    if (channelMask & (1ull<<x)) {
      int y = __popcll(channelMask & ((1ull<<x)-1));
      if (blockIdx.x == y) ncclShmem.channelId = x;
    }
    if (32 < MAXCHANNELS) {
      x = 32 + tid;
      if (channelMask & (1ull<<x)) {
        int y = __popcll(channelMask & ((1ull<<x)-1));
        if (blockIdx.x == y) ncclShmem.channelId = x;
      }
    }
  }
  __syncthreads(); // publish ncclShmem.channelId
  int channelId = ncclShmem.channelId;
  /* set abort flag to 0 */
  if (tid == 0) ncclShmem.aborted = 0;

  if (true) {
    void *dst, *src;
    int bytes;
    // Use first 3 warps to load comm, channel, and work into ncclShmem
    switch (tid/WARP_SIZE) {
    case 0:
      dst = &ncclShmem.comm;
      src = comm;
      bytes = sizeof(ncclDevComm);
      static_assert(sizeof(ncclDevComm) <= 16*WARP_SIZE, "ncclDevComm cannot be loaded by a single warp in one insn.");
      break;
    case 1:
      // Get address of channel without incurring indirect load from ncclDevComm::channels
      dst = &ncclShmem.channel;
      src = &((ncclDevCommAndChannels*)comm)->channels[channelId];
      bytes = sizeof(ncclDevChannel);
      static_assert(sizeof(ncclDevChannel) <= 16*WARP_SIZE, "ncclDevChannel cannot be loaded by a single warp in one insn.");
      break;
    case 2:
      dst = &ncclShmem.work;
      src = workHead + blockIdx.x;
      bytes = sizeof(ncclWork);
      static_assert(sizeof(ncclWork) <= 16*WARP_SIZE, "ncclWork cannot be loaded by a single warp in one insn.");
      break;
    default:
      bytes = 0;
      break;
    }
    if (bytes) copyToShmem16(tid%WARP_SIZE, dst, src, bytes);
  }
  __syncthreads(); // publish ncclShmem

  while (true) {
    // Notify host that all fifo reads are complete.
    if (tid == 0 && ncclShmem.work.header.isLast && ncclShmem.work.header.inFifo) {
      *ncclShmem.channel.workFifoDone = ncclShmem.work.header.doneAcks;
    }

    __syncwarp();
    if (ncclShmem.work.header.type == ncclWorkTypeColl) {
      if (tid < NCCL_MAX_WORK_ELEMENTS) ncclRedopPtrDeref(&ncclShmem.work.elems[tid]);
    } else if (ncclShmem.work.header.type == ncclWorkTypeRegColl) {
      if (tid < NCCL_MAX_WORK_ELEMENTS_REG) ncclRedopPtrDeref(&ncclShmem.work.regElems[tid].elem);
    }
    __syncthreads();

    if (0 <= SpecializedFnId && ncclShmem.work.header.funcIndex == (unsigned)SpecializedFnId) {
      SpecializedRunWork().run(&ncclShmem.work);
    } else {
      ncclDevFuncTable[ncclShmem.work.header.funcIndex]();
    }

    int workIxNext = ncclShmem.work.header.workNext;
    __syncthreads();
    if (ncclShmem.work.header.isLast) break;

    copyToShmem16(tid, &ncclShmem.work, workHead + workIxNext, sizeof(ncclWork));

    { // Check whether the last operation was aborted and make sure all threads exit
      int aborted = tid == 0 ? *comm->abortFlag : 0;
      if (barrierReduceAny(aborted)) // publish ncclShmem.work
        break;
    }
  }
}

__global__ void ncclDevKernel_Generic(struct ncclDevComm* comm, uint64_t channelMask, struct ncclWork* workHead);
__device__ void ncclDevFunc_Nop();

#define DEFINE_ncclDevKernel(suffix, coll, redop, ty, algo, proto, specializedFnId) \
  __global__ void ncclDevKernel_##suffix(struct ncclDevComm* comm, uint64_t channelMask, struct ncclWork* workHead) { \
    ncclKernelMain<specializedFnId, RunWork<coll, ty, redop<ty>, algo, proto>>(comm, channelMask, workHead); \
  }

#define DEFINE_ncclDevFunc(suffix, coll, redop, ty, algo, proto) \
  __device__ void ncclDevFunc_##suffix() { \
    RunWork<coll, ty, redop<ty>, algo, proto>().run(&ncclShmem.work); \
  }

#endif
