/*************************************************************************
 * Copyright (c) 2017-2021, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_DEVICE_COMMON_H_
#define NCCL_DEVICE_COMMON_H_

#include "collectives.h"
#include "devcomm.h"

#if __CUDA_ARCH__ >= 800
#define COLL_UNROLL 8
#else
#define COLL_UNROLL 4
#endif

#define NCCL_MAX_DEV_ARITY (NCCL_MAX_TREE_ARITY-1)  // Using balanced tree instead of split tree

__device__ inline bool barrierReduceAny(int bit) {
  uint32_t popc;
  asm ("{"
    ".reg .pred barr_pred;"
    "setp.eq.u32 barr_pred, %1, 1;"
    "bar.red.popc.u32 %0, 0, barr_pred;"
  "}" : "=r"(popc) : "r"(bit));
  return popc != 0;
}

template<typename T>
__device__ int copyToShmem(T *dst, T const *src, int turn=0) {
  static_assert(sizeof(uint64_t) <= alignof(T), "Uhoh");
  uint64_t *d = reinterpret_cast<uint64_t*>(dst);
  uint64_t const *s = reinterpret_cast<uint64_t const*>(src);
  int t = threadIdx.x - turn;
  if (t < 0) t += blockDim.x;
  int n = sizeof(T)/sizeof(uint64_t);

  int delta = (n + WARP_SIZE-1) & -WARP_SIZE; // round up to warp lane 0
  if (delta < blockDim.x) {
    turn += delta;
    if (turn >= blockDim.x) turn -= blockDim.x;
  }
  else
    turn = 0;

  n -= t;
  d += t;
  s += t;
  #pragma unroll
  for (int i=0; i < divUp(sizeof(T), WARP_SIZE*sizeof(uint64_t)); i++) {
    if (n > 0) {
      *d = *s;
      d += blockDim.x;
      s += blockDim.x;
      n -= blockDim.x;
    }
  }
  return turn;
}

template<ncclFunc_t Fn, typename T, typename RedOp, int Algo, int Proto>
struct RunWorkElement {
  __device__ void run(ncclWorkElem*) {
    // Put NOT IMPLEMENTED behavior here.
  }
};

template<ncclFunc_t Fn, typename T, typename RedOp, int Algo, int Proto>
struct RunWork {
  __device__ void run(ncclWork *w) {
    int tid = threadIdx.x;
    #pragma unroll 1
    for(int e=0; e < NCCL_MAX_WORK_ELEMENTS && w->elems[e].active != 0; e++) {
      if (tid < w->elems[e].nThreads)
        RunWorkElement<Fn, T, RedOp, Algo, Proto>().run(&w->elems[e]);
    }
  }
};

typedef void(*ncclKern_t)();
extern __device__ ncclKern_t ncclFuncs[];

struct ncclShmemGroup {
  ncclConnInfo *recvConns[NCCL_MAX_DIRECT_ARITY];
  ncclConnInfo *sendConns[NCCL_MAX_DIRECT_ARITY];
  void* srcs[NCCL_MAX_DIRECT_ARITY+1];
  void* dsts[NCCL_MAX_DIRECT_ARITY+1];
};

struct ncclShmemData {
  union {
    uint64_t ll128warp[NCCL_LL128_MAX_NTHREADS/WARP_SIZE][NCCL_LL128_SHMEM_ELEMS_PER_THREAD*WARP_SIZE];
    struct ncclShmemGroup groups[NCCL_MAX_GROUPS];
  };
  ncclDevComm comm;
  ncclChannel channel;
  ncclWork work;
};

extern __shared__ ncclShmemData ncclShmem;

template<ncclFunc_t Fn, typename T, typename RedOp, int Algo, int Proto, int FnIndex>
__device__ void ncclKernel(ncclWorkElem first)  {
  int tid = threadIdx.x;
  int bid = blockIdx.x;

  int turn = copyToShmem(&ncclShmem.comm, first.comm);
  // get address of channel without incurring indirect load from ncclDevCom::channels
  ncclChannel *channel = &((ncclDevCommAndChannels*)first.comm)->channels[bid];
  turn = copyToShmem(&ncclShmem.channel, channel, turn);

  // To optimize for latency, (only) the first operation is passed as argument.
  if (bid == 0 && first.active != 0) {
    turn = copyToShmem(&ncclShmem.work.elems[0], &first, turn);
    if (tid == 0) ncclShmem.work.elems[1].active = 0;
  }
  __syncthreads(); // publish ncclShmem

  ncclWork *workFifoHost = ncclShmem.channel.workFifo;
  ncclWork *workFifoDev = ncclShmem.channel.workFifoDev;
  int workFifoIx = ncclShmem.channel.index;

  if (bid == 0 && first.active != 0)
    goto SkipLoadWork;

  while (true) {
    copyToShmem(&ncclShmem.work, &workFifoDev[workFifoIx]); // turn no longer helps
    { // Check whether the last operation was aborted and make sure all threads exit
      int aborted = tid == 0 ? *ncclShmem.comm.abortFlag : 0;
      if (barrierReduceAny(aborted)) // publish ncclShmem.work
        break;
      if (tid == 0)
        workFifoHost[workFifoIx].elems[0].active = 0;
    }

  SkipLoadWork:
    workFifoIx = (workFifoIx + 1)%NCCL_MAX_OPS;
    if (tid == 0)
      channel->index = workFifoIx; // write back to real channel, not shmem shadow

    if (ncclShmem.work.elems[0].funcIndex == FnIndex)
      RunWork<Fn, T, RedOp, Algo, Proto>().run(&ncclShmem.work);
    else
      ncclFuncs[ncclShmem.work.elems[0].funcIndex]();

    if (ncclShmem.work.elems[0].active == 2)
      break;
    __syncthreads();
  }
}

// Only generate kernels for SUM
#if NCCL_OP == 0
#define IMPL_COLL_KERN(func, algo, proto, redop, type, fIndex) \
__global__ void NCCL_KERN_NAME(func, algo, proto, redop, type)(ncclWorkElem first) { \
  ncclKernel<ncclFunc##func, type, Func##redop<type>, NCCL_ALGO_##algo, NCCL_PROTO_##proto, fIndex>(first); \
}
#else
#define IMPL_COLL_KERN(func, algo, proto, redop, type, fInded)
#endif

// Examples :     AllReduce, RING, LL,    Sum,   uint8
#define IMPL_COLL_FUNC(func, algo, proto, redop, type) \
__device__ void NCCL_FUNC_NAME(func, algo, proto, redop, type)() { \
  RunWork<ncclFunc##func, type, Func##redop<type>, NCCL_ALGO_##algo, NCCL_PROTO_##proto>().run(&ncclShmem.work); \
}

// Only generate inline kernels for LL
#define IMPL_COLL4(func, algo, redop, type, ncclType) \
  IMPL_COLL_FUNC(func, algo, LL,     redop, type) \
  IMPL_COLL_FUNC(func, algo, LL128,  redop, type) \
  IMPL_COLL_FUNC(func, algo, SIMPLE, redop, type) \
  IMPL_COLL_KERN(func, algo, LL,     redop, type, FUNC_INDEX(ncclFunc##func, nccl##redop, ncclType, NCCL_ALGO_##algo, NCCL_PROTO_LL)) \

#define IMPL_COLL3(func, redop, type, ncclType) \
  IMPL_COLL4(func, TREE,    redop, type, ncclType) \
  IMPL_COLL4(func, RING,    redop, type, ncclType) \
  IMPL_COLL4(func, COLLNET, redop, type, ncclType)

#if NCCL_TYPE == 0
#define IMPL_COLL2(func, redop) IMPL_COLL3(func, redop, int8_t,   ncclInt8)
#elif NCCL_TYPE == 1
#define IMPL_COLL2(func, redop) IMPL_COLL3(func, redop, uint8_t,  ncclUint8)
#elif NCCL_TYPE == 2
#define IMPL_COLL2(func, redop) IMPL_COLL3(func, redop, int32_t,  ncclInt32)
#elif NCCL_TYPE == 3
#define IMPL_COLL2(func, redop) IMPL_COLL3(func, redop, uint32_t, ncclUint32)
#elif NCCL_TYPE == 4
#define IMPL_COLL2(func, redop) IMPL_COLL3(func, redop, int64_t,  ncclInt64)
#elif NCCL_TYPE == 5
#define IMPL_COLL2(func, redop) IMPL_COLL3(func, redop, uint64_t, ncclUint64)
#elif NCCL_TYPE == 6
#define IMPL_COLL2(func, redop) IMPL_COLL3(func, redop, half,     ncclFloat16)
#elif NCCL_TYPE == 7
#define IMPL_COLL2(func, redop) IMPL_COLL3(func, redop, float,    ncclFloat32)
#elif NCCL_TYPE == 8
#define IMPL_COLL2(func, redop) IMPL_COLL3(func, redop, double,   ncclFloat64)
#elif NCCL_TYPE == 9 && defined(__CUDA_BF16_TYPES_EXIST__)
#define IMPL_COLL2(func, redop) IMPL_COLL3(func, redop, __nv_bfloat16, ncclBfloat16)
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
#define IMPL_COLL_R(func) IMPL_COLL2(func, Avg);
#endif

#if NCCL_OP == 0 && NCCL_TYPE == 0
// Copy primitives only define one function for copy
#define IMPL_COLL_C(func) IMPL_COLL3(func, Sum, int8_t, ncclInt8);

// Point-to-point primitives only have one function/kernel.
#define IMPL_COLL_P(func) \
  IMPL_COLL_FUNC(func, RING, SIMPLE, Sum, int8_t); \
  IMPL_COLL_KERN(func, RING, SIMPLE, Sum, int8_t, 0);
#else
#define IMPL_COLL_C(func)
#define IMPL_COLL_P(func)
#endif

#endif
