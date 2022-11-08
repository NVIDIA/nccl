/*************************************************************************
 * Copyright (c) 2017-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_DEVICE_COMMON_H_
#define NCCL_DEVICE_COMMON_H_

// Set this to 1 to make NCCL_HOLLOW_*** have effect. Their default value of
// all 1's makes it so all nccl device code becomes a no-op.
#define NCCL_HOLLOW_ENABLED 0

// If you want to enable only a specific reduction+datatype combo then refer to
// NCCL_OP and NCCL_TYPE in the expression. Next to all reduce is an example
// that retains only sum of float.
#define NCCL_HOLLOW_ALL_GATHER     1
#define NCCL_HOLLOW_ALL_REDUCE     1 /*!(NCCL_OP==0 && NCCL_TYPE==4)*/
#define NCCL_HOLLOW_BROADCAST      1
#define NCCL_HOLLOW_REDUCE         1
#define NCCL_HOLLOW_REDUCE_SCATTER 1
#define NCCL_HOLLOW_SENDRECV       1

// Defined in <foo>.cu:
//#define NCCL_HOLLOW_THIS_TU NCCL_HOLLOW_<FOO>

#include "collectives.h"
#include "devcomm.h"
#include "op128.h"
#include "reduce_kernel.h"

#define COLL_UNROLL (ncclCollUnroll())
#define NCCL_MAX_DEV_ARITY (NCCL_MAX_TREE_ARITY-1)  // Using balanced tree instead of split tree

#define NCCL_SPINS_BEFORE_CHECK_ABORT 1000000

typedef void(*ncclKern_t)();
extern __device__ ncclKern_t ncclFuncs[];

struct ncclShmemGroup {
  ncclConnInfo *recvConns[NCCL_MAX_NVLS_ARITY];
  ncclConnInfo *sendConns[NCCL_MAX_NVLS_ARITY];
  void* srcs[NCCL_MAX_NVLS_ARITY+1];
  void* dsts[NCCL_MAX_NVLS_ARITY+1];
};

struct ncclShmemData {
  struct ncclShmemGroup groups[NCCL_MAX_GROUPS];
  uint64_t redOpArgs[NCCL_MAX_NVLS_ARITY+1];
  int aborted;
  alignas(16) struct ncclDevComm comm;
  alignas(16) struct ncclDevChannel channel;
  union {
    alignas(16) char devWorkBatchStorage[ncclMaxDevWorkBatchBytes()];
    struct ncclDevWorkBatchHeader workBatch;
  };
};
static_assert(offsetof(struct ncclShmemData, workBatch)%16 == 0, "shmem.workBatch needs to be 16B aligned");

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
  asm("{ .reg .pred p;"
      "  setp.eq.u32 p, %1, 1;"
      "  bar.red.popc.u32 %0, 2, p; }"
      : "=r"(popc)
      : "r"(bit)
      : "memory");
  return popc != 0;
}

// Copy 16-byte aligned data.
template<int MaxBytes>
__device__ __forceinline__ void copyFifoToShmem16(
    int tn, int tid, void *dstPtr,
    void *fifoBuf, uint32_t fifoHead, uint32_t fifoMask,
    int bytes
  ) {
  uint32_t dst = cvta_to_shared(dstPtr) + 16*tid;
  uintptr_t fifo = cvta_to_global(fifoBuf);
  uint32_t src = fifoHead + 16*tid;
  bytes -= 16*tid;
  if (0 < bytes) {
    constexpr int Unroll1 = (MaxBytes + WARP_SIZE*16-1)/(WARP_SIZE*16);
    constexpr int Unroll = 4 < Unroll1 ? 4 : Unroll1;
    do {
      uint64_t a[Unroll], b[Unroll];
      int bytes1 = bytes;
      #pragma unroll Unroll
      for (int u=0; u < Unroll; u++) {
        if (0 < bytes1) asm("ld.global.v2.u64 {%0,%1}, [%2];" : "=l"(a[u]), "=l"(b[u]) : "l"(fifo + (src & fifoMask)) : "memory");
        bytes1 -= 16*tn;
        src += 16*tn;
      }
      #pragma unroll Unroll
      for (int u=0; u < Unroll; u++) {
        if (0 < bytes) asm("st.shared.v2.u64 [%0], {%1,%2};" :: "r"(dst), "l"(a[u]), "l"(b[u]) : "memory");
        bytes -= 16*tn;
        dst += 16*tn;
      }
    } while (Unroll*WARP_SIZE*16 < MaxBytes && 0 < bytes);
  }
}

template<ncclFunc_t Fn, typename T, typename RedOp, int Algo, int Proto>
struct RunWork {
  __device__ void run(ncclDevWorkColl*) {
    // Put NOT IMPLEMENTED behavior here.
    printf("r=%d b=%d RunWork NOT IMPLEMENTED\n", ncclShmem.comm.rank, blockIdx.x);
  }
};

template<ncclFunc_t Fn, typename T, typename RedOp, int Algo, int Proto>
struct RunWorkBatch {
  // This __forceinline__ is necessary. The compiler was inserting a function call
  // here from the LL ncclKernel.
  __device__ __forceinline__ void run() {
    int tid = threadIdx.x;
    int tn = blockDim.x;
    if (RedOpArg<RedOp>::ArgUsed) {
      int nWorks = ncclShmem.workBatch.nWorks;
      for (int w=tid; w < nWorks; w += tn) {
        struct ncclDevWorkColl* work;
        if (ncclShmem.workBatch.type == ncclDevWorkTypeColl) {
          work = (struct ncclDevWorkColl*)(&ncclShmem.workBatch+1) + w;
        } else {
          work = &((struct ncclDevWorkCollReg*)(&ncclShmem.workBatch+1))[w].coll;
        }
        if (work->redOpArgIsPtr) {
          work->redOpArg = RedOpArg<RedOp>::loadArg(reinterpret_cast<void*>(work->redOpArg));
        }
      }
      __syncthreads();
    }

    #pragma unroll 1
    for (int w=0; w < ncclShmem.workBatch.nWorks; w++) {
      struct ncclDevWorkColl* work;
      if (ncclShmem.workBatch.type == ncclDevWorkTypeColl) {
        work = (struct ncclDevWorkColl*)(&ncclShmem.workBatch+1) + w;
      } else {
        work = &((struct ncclDevWorkCollReg*)(&ncclShmem.workBatch+1))[w].coll;
      }
      RunWork<Fn, T, RedOp, Algo, Proto>().run(work);
    }
  }
};

template<ncclFunc_t Fn, typename T, typename RedOp, int Algo, int Proto, int FnIndex>
__device__ void ncclKernel(struct ncclKernelArgs args) {
  int tid = threadIdx.x;
  int tn = blockDim.x;
  int channelId;
  uint32_t workCursor = args.devWorkCursor;
  uint32_t workBytes;
  int channelsBelow = 0;
  #pragma unroll
  for (int c=0; c < MAXCHANNELS; c++) {
    if (args.firstDevWorkBytes[c] != 0) {
      if (channelsBelow == blockIdx.x) {
        channelId = c;
        workBytes = args.firstDevWorkBytes[c];
        break;
      }
      channelsBelow += 1;
    }
    workCursor += args.firstDevWorkBytes[c];
  }
  uint32_t workMask = args.inWorkFifo ? ~0u>>(32-args.devWorkFifoSizeLog2) : ~0u;

  static_assert(NCCL_MIN_NTHREADS >= 3*WARP_SIZE, "Require NCCL_MIN_NTHREADS >= 4*WARP_SIZE");
  switch (tid/WARP_SIZE) {
  case 0: // load ncclShmem.comm
    { int subtn = WARP_SIZE;
      int subtid = tid;
      copyFifoToShmem16</*MaxBytes=*/sizeof(ncclDevComm)>(subtn, subtid, &ncclShmem.comm, &args.comm->comm, 0, ~0u, sizeof(ncclDevComm));
    } break;
  case 1: // load ncclShmem.channel
    { int subtn = WARP_SIZE;
      int subtid = tid - WARP_SIZE;
      copyFifoToShmem16</*MaxBytes=*/sizeof(ncclDevChannel)>(subtn, subtid, &ncclShmem.channel, &args.comm->channels[channelId], 0, ~0u, sizeof(ncclDevChannel));
    } break;
  default: // load ncclShmem.workBatch
    { int subtn = tn - 2*WARP_SIZE;
      int subtid = tid - 2*WARP_SIZE;
      copyFifoToShmem16</*MaxBytes=*/ncclMaxDevWorkBatchBytes()>(subtn, subtid, &ncclShmem.workBatch, args.devWorkBuf, workCursor, workMask, workBytes);
    } break;
  }

  if (tid == 0) ncclShmem.aborted = 0;
  __syncthreads(); // publish ncclShmem

  if (workBytes != 0) {
    uint32_t consumed = workCursor + workBytes;
    while (true) {
      if (tid == 0 && args.inWorkFifo) {
        // Notify host of fifo read progress.
        ncclShmem.comm.devWorkFifoConsumed[channelId] = consumed;
      }

      if (ncclShmem.workBatch.funcIndex == FnIndex) {
        #if !(NCCL_HOLLOW_ENABLED && NCCL_HOLLOW_THIS_TU)
          RunWorkBatch<Fn, T, RedOp, Algo, Proto>().run();
        #endif
      } else {
        ncclFuncs[ncclShmem.workBatch.funcIndex]();
      }

      workCursor = ncclShmem.workBatch.nextCursor;
      workBytes = ncclShmem.workBatch.nextBytes;
      __syncthreads();
      if (workBytes == 0) break;

      copyFifoToShmem16</*MaxBytes=*/ncclMaxDevWorkBatchBytes()>
        (tn, tid, &ncclShmem.workBatch, args.devWorkBuf, workCursor, workMask, workBytes);
      consumed = workCursor + workBytes;

      { // Check whether the last operation was aborted and make sure all threads exit
        int aborted = (tid == 0) ? *ncclShmem.comm.abortFlag : 0;
        aborted = barrierReduceAny(aborted); // publish ncclShmem.workBatch
        if (aborted) return;
      }
    }
  }
}

// Only generate kernels for SUM
#if NCCL_OP == 0
#if !(NCCL_HOLLOW_ENABLED && NCCL_HOLLOW_THIS_TU)
#define IMPL_COLL_KERN(func, algo, proto, devredop, type, fIndex) \
__global__ void NCCL_KERN_NAME(func, algo, proto, devredop, type)(struct ncclKernelArgs args) { \
  ncclKernel<ncclFunc##func, type, Func##devredop<type>, NCCL_ALGO_##algo, NCCL_PROTO_##proto, fIndex>(args); \
}
#else
#define IMPL_COLL_KERN(func, algo, proto, devredop, type, fIndex) \
__global__ void NCCL_KERN_NAME(func, algo, proto, devredop, type)(struct ncclKernelArgs args) {}
#endif
#else
#define IMPL_COLL_KERN(func, algo, proto, devredop, type, fInded)
#endif

// Examples :     AllReduce, RING, LL,    Sum,   uint8
#if !(NCCL_HOLLOW_ENABLED && NCCL_HOLLOW_THIS_TU)
#define IMPL_COLL_FUNC(func, algo, proto, devredop, type) \
__device__ void NCCL_FUNC_NAME(func, algo, proto, devredop, type)() { \
  RunWorkBatch<ncclFunc##func, type, Func##devredop<type>, NCCL_ALGO_##algo, NCCL_PROTO_##proto>().run(); \
}
#else
#define IMPL_COLL_FUNC(func, algo, proto, devredop, type) \
__device__ void NCCL_FUNC_NAME(func, algo, proto, devredop, type)() {}
#endif

// Only generate inline kernels for LL
#define IMPL_COLL4(func, algo, devredop, type, ncclType) \
  IMPL_COLL_FUNC(func, algo, LL,     devredop, type) \
  IMPL_COLL_FUNC(func, algo, LL128,  devredop, type) \
  IMPL_COLL_FUNC(func, algo, SIMPLE, devredop, type) \
  IMPL_COLL_KERN(func, algo, LL,     devredop, type, FUNC_INDEX(ncclFunc##func, ncclDev##devredop, ncclType, NCCL_ALGO_##algo, NCCL_PROTO_LL)) \

#define IMPL_COLL3(func, devredop, type, ncclType) \
  IMPL_COLL4(func, TREE,    devredop, type, ncclType) \
  IMPL_COLL4(func, RING,    devredop, type, ncclType) \
  IMPL_COLL4(func, COLLNET_DIRECT, devredop, type, ncclType) \
  IMPL_COLL4(func, COLLNET_CHAIN, devredop, type, ncclType) \
  IMPL_COLL4(func, NVLS, devredop, type, ncclType) \
  IMPL_COLL4(func, NVLS_TREE, devredop, type, ncclType)

#if NCCL_TYPE == 0
#define IMPL_COLL2(func, devredop) IMPL_COLL3(func, devredop, int8_t,   ncclInt8)
#elif NCCL_TYPE == 1
#define IMPL_COLL2(func, devredop) IMPL_COLL3(func, devredop, uint8_t,  ncclUint8)
#elif NCCL_TYPE == 2
#define IMPL_COLL2(func, devredop) IMPL_COLL3(func, devredop, int32_t,  ncclInt32)
#elif NCCL_TYPE == 3
#define IMPL_COLL2(func, devredop) IMPL_COLL3(func, devredop, uint32_t, ncclUint32)
#elif NCCL_TYPE == 4
#define IMPL_COLL2(func, devredop) IMPL_COLL3(func, devredop, int64_t,  ncclInt64)
#elif NCCL_TYPE == 5
#define IMPL_COLL2(func, devredop) IMPL_COLL3(func, devredop, uint64_t, ncclUint64)
#elif NCCL_TYPE == 6
#define IMPL_COLL2(func, devredop) IMPL_COLL3(func, devredop, half,     ncclFloat16)
#elif NCCL_TYPE == 7
#define IMPL_COLL2(func, devredop) IMPL_COLL3(func, devredop, float,    ncclFloat32)
#elif NCCL_TYPE == 8
#define IMPL_COLL2(func, devredop) IMPL_COLL3(func, devredop, double,   ncclFloat64)
#elif NCCL_TYPE == 9 && defined(__CUDA_BF16_TYPES_EXIST__)
#define IMPL_COLL2(func, devredop) IMPL_COLL3(func, devredop, __nv_bfloat16, ncclBfloat16)
#endif

// Reduction define all functions
#if NCCL_OP == 0
#define IMPL_COLL_R(func) IMPL_COLL2(func, Sum);
#elif NCCL_OP == 1
#define IMPL_COLL_R(func) IMPL_COLL2(func, Prod);
#elif NCCL_OP == 2
#define IMPL_COLL_R(func) IMPL_COLL2(func, Min);
#elif NCCL_OP == 3
#define IMPL_COLL_R(func) IMPL_COLL2(func, Max);
#elif NCCL_OP == 4
#define IMPL_COLL_R(func) IMPL_COLL2(func, PreMulSum);
#elif NCCL_OP == 5
  #if NCCL_TYPE < 6
    #define IMPL_COLL_R(func) IMPL_COLL2(func, SumPostDiv);
  #else
    #define IMPL_COLL_R(func) // skip SumPostDiv for floating point
  #endif
#endif

#if NCCL_OP == 0 && NCCL_TYPE == 0
// Copy primitives only define one function for copy
#define IMPL_COLL_C(func) IMPL_COLL3(func, Sum, int8_t, ncclInt8);

// Point-to-point primitives only have one function/kernel.
#define IMPL_COLL_P(func) \
  IMPL_COLL_FUNC(func, RING, SIMPLE, Sum, int8_t); \
  IMPL_COLL_KERN(func, RING, SIMPLE, Sum, int8_t, FUNC_INDEX_P2P);
#else
#define IMPL_COLL_C(func)
#define IMPL_COLL_P(func)
#endif

#define NCCL_NVLS_ENABLED (__CUDA_ARCH__ >= 900 && NCCL_NVLS_SUPPORTS(NCCL_TYPE, NCCL_OP))

#endif
