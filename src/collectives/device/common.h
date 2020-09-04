/*************************************************************************
 * Copyright (c) 2017-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_DEVICE_COMMON_H_
#define NCCL_DEVICE_COMMON_H_

#include "collectives.h"
#include "devcomm.h"


#if __CUDA_ARCH__ >= 800
#define COLL_UNROLL 8
#define NCCL_MAX_DEV_ARITY (NCCL_MAX_TREE_ARITY-1)  // Using balanced tree instead of split tree
#else
#define COLL_UNROLL 4
#define NCCL_MAX_DEV_ARITY NCCL_MAX_TREE_ARITY
#endif

// Exit If Abort Barrier across CTA: make sure all threads exit consistently
// Each thread sets a predicate to true if abort == 1
// all CTA's threads enter the barrier and do a popc on their predicates being True
// If any of the thread's predicate was True, all the threads call exit()
static inline __device__ void exitIfAbortBarrier(int abort) {
  uint32_t popc;
  asm ("{");
  asm volatile ("   .reg .pred barr_pred;");
  asm volatile ("   setp.eq.u32 barr_pred,%0,1;" :: "r"(abort));
  asm volatile ("   bar.red.popc.u32 %0, 0, barr_pred;" : "=r"(popc));
  asm ("}");
  if (popc) { asm volatile ("exit;"); }
}

typedef void(*ncclKern_t)(struct ncclWorkElem* args);
extern __device__ ncclKern_t ncclFuncs[];

static __device__ void load_parallel(void* dst, void* src, size_t size, int tid) {
  int* d = (int*)dst;
  int* s = (int*)src;
  for (int o = tid; o < (size/sizeof(int)); o += blockDim.x) d[o] = s[o];
}
static __device__ void load_coll(struct ncclWork* localWork, struct ncclWork* hostWork, int tid, struct ncclDevComm* comm) {
  __syncthreads();
  load_parallel(localWork, hostWork, sizeof(struct ncclWork), tid);
  // Check whether the last operation was aborted and make sure all threads exit
  int abort = tid == 0 ? *(comm->abortFlag) : 0;
  exitIfAbortBarrier(abort);
  if (tid == 0) hostWork->elems[0].active = 0;
}

template <ncclFunc_t FUNCTION, int ALGO, int PROTO, class REDOP, typename T, int UNROLL>
class ncclFunction {
  public:
  __device__ void run(struct ncclWorkElem* args) {}
};

struct ncclShmemPtrs {
  void* srcs[NCCL_MAX_DEV_ARITY+1];
  void* dsts[NCCL_MAX_DEV_ARITY+1];
};

struct ncclShmemData {
  union {
    volatile uint64_t data[NCCL_LL128_SHMEM_SIZE];
    struct ncclShmemPtrs ptrs[NCCL_MAX_GROUPS];
  };
  struct ncclWork localWork;
};

extern __device__ struct ncclShmemData *ncclShmem;
template <ncclFunc_t FUNCTION, int ALGO, int PROTO, class REDOP, typename T, int UNROLL, int FINDEX>
__device__ void ncclKernel(struct ncclWorkElem first)  {
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  __shared__ struct ncclShmemData shmem;
  ncclShmem = &shmem;

  auto f = ncclFunction<FUNCTION, ALGO, PROTO, REDOP, T, UNROLL>();

  struct ncclDevComm* comm = first.comm;
  struct ncclChannel* channel = comm->channels+bid;
  struct ncclWorkElem* w = NULL;
  uint16_t index = first.index;

  /* To optimize for latency, (only) the first operation is passed as argument.*/
  if (bid == 0 && first.funcIndex != FUNC_INDEX_P2P) w = &first;

  while (1) {
    if (w == NULL) {
      w = shmem.localWork.elems;
      load_coll(&shmem.localWork, channel->workFifo+index, tid, comm);
    }
    if (tid < w->nThreads) {
      if (w->funcIndex == FINDEX) {
        f.run(w);
      } else {
        ncclFuncs[w->funcIndex](w);
      }
    }
    index = (index+1) % NCCL_MAX_OPS;
    if (w->active == 2) {
      return;
    }
    w = NULL;
  }
}

// Only generate kernels for SUM
#if NCCL_OP == 0
#define IMPL_COLL_KERN(func, algo, proto, redop, type, fIndex) \
__global__ void NCCL_KERN_NAME(func, algo, proto, redop, type)(struct ncclWorkElem first) { \
  ncclKernel<ncclFunc##func, NCCL_ALGO_##algo, NCCL_PROTO_##proto, Func##redop<type>, type, COLL_UNROLL, fIndex>(first); \
}
#else
#define IMPL_COLL_KERN(func, algo, proto, redop, type, fInded)
#endif

// Examples :     AllReduce, RING, LL,    Sum,   uint8
#define IMPL_COLL_FUNC(func, algo, proto, redop, type) \
__device__ void NCCL_FUNC_NAME(func, algo, proto, redop, type)(struct ncclWorkElem* args) { \
  auto f = ncclFunction<ncclFunc##func, NCCL_ALGO_##algo, NCCL_PROTO_##proto, Func##redop<type>, type, COLL_UNROLL>(); \
  f.run(args); \
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
