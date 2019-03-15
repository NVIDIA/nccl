/*************************************************************************
 * Copyright (c) 2017-2019, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_DEVICE_COMMON_H_
#define NCCL_DEVICE_COMMON_H_

#include "../collectives.h"
#include "devcomm.h"
#include "nccl.h"

// Exit If Abort Barrier across CTA: make sure all threads exit consistently
// Each thread sets a predicate to true if abort == 1
// all CTA's threads enter the barrier and do a popc on their predicates being True
// If any of the thread's predicate was True, all the threads call exit()
static inline __device__ void exitIfAbortBarrier(int abort) {
  uint32_t popc;
  asm ("{");
  asm volatile ("   .reg .pred barr_pred;");
  asm volatile ("   setp.eq.u32 barr_pred,%0,1;" :: "r"(abort));
  asm volatile ("   bar.red.popc.u32 %0, 13, barr_pred;" : "=r"(popc));
  asm ("}");
  if (popc) { asm volatile ("exit;"); }
}

typedef void(*ncclKern_t)(struct CollectiveArgs* args);
extern __device__ ncclKern_t ncclFuncs[];

static __device__ void load_parallel(void* dst, void* src, size_t size, int tid) {
  int* d = (int*)dst;
  int* s = (int*)src;
  // When aggregation is effective, if some threads have aborted inside the LL kernel,
  // make sure the rest of the threads abort as well
  exitIfAbortBarrier(0);
  for (int o = tid; o < (size/sizeof(int)); o += blockDim.x) d[o] = s[o];
  __syncthreads();
}
static __device__ void load_coll(struct ncclColl* localColl, struct ncclColl* hostColl, int tid) {
  load_parallel(localColl, hostColl, sizeof(struct ncclColl), tid);
  if (tid == 0) hostColl->active = 0;
}

/* Functions for aggregation case */
#define IMPL_COLL_FUNC(coll, op, ncclFunc, dtype, ctype) \
__device__ void NCCL_COLL_NAME(coll, op, dtype)(struct CollectiveArgs* args) { \
  coll##Kernel<COLL_UNROLL, ncclFunc<ctype>, ctype>(args); \
}

#if NCCL_OP == 0
/* Kernels with the first operation inlined */
#define IMPL_COLL_KERN(coll, op, ncclFunc, dtype, ctype, fIndex) \
__launch_bounds__(MAXTHREADS+WARP_SIZE, 1) \
__global__ void NCCL_KERN_NAME(coll, op, dtype)(struct ncclColl firstColl) { \
  int tid = threadIdx.x; \
  int bid = blockIdx.x; \
  __shared__ struct ncclColl localColl; \
 \
  struct ncclDevComm* comm = firstColl.args.comm; \
  struct ncclChannel* channel = comm->channels+bid; \
  struct ncclColl* c; \
  if (bid == 0) { \
    /* To optimize for latency, (only) the first operation is passed as argument.*/ \
    c = &firstColl; \
  } else { \
    c = &localColl; \
    load_coll(c, channel->devCollectives+channel->collFifoHead, tid); \
  } \
  while (1) { \
    if (tid < c->args.nThreads) { \
      if (c->funcIndex == fIndex) { \
        coll##Kernel<COLL_UNROLL, ncclFunc<ctype>, ctype>(&c->args); \
      } else { \
        ncclFuncs[c->funcIndex](&c->args); \
      } \
    } \
    int nextIndex = c->nextIndex; \
    if (tid == 0) channel->collFifoHead = nextIndex; \
 \
    if (c->active == 2) { \
      return; \
    } \
 \
    /* Load next collective operation*/ \
    c = &localColl; /* for bid 0 */ \
    load_coll(c, channel->devCollectives+nextIndex, tid); \
  } \
}
#else
#define IMPL_COLL_KERN(coll, op, ncclFunc, dtype, ctype, fIndex)
#endif

// Only generate inline kernels for LL
#define IMPL_COLL4(coll, op, ncclFunc, dtype, ctype, ncclColl, ncclOp, ncclType, al) \
  IMPL_COLL_FUNC(coll, op, ncclFunc, dtype, ctype) \
  IMPL_COLL_FUNC(coll##LL, op, ncclFunc, dtype, ctype) \
  IMPL_COLL_KERN(coll##LL, op, ncclFunc, dtype, ctype, FUNC_INDEX(ncclColl, ncclOp, ncclType, 1, al)) \

#define IMPL_COLL3(coll, op, ncclFunc, dtype, ctype, ncclColl, ncclOp, ncclType) \
  IMPL_COLL4(coll##Ring, op, ncclFunc, dtype, ctype, ncclColl, ncclOp, ncclType, 0) \
  IMPL_COLL4(coll##Tree, op, ncclFunc, dtype, ctype, ncclColl, ncclOp, ncclType, 1)

#if NCCL_TYPE == 0
#define IMPL_COLL2(coll, op, ncclFunc, ncclColl, ncclOp) \
  IMPL_COLL3(coll, op, ncclFunc, i8,  int8_t,   ncclColl, ncclOp, ncclInt8)
#elif NCCL_TYPE == 1
#define IMPL_COLL2(coll, op, ncclFunc, ncclColl, ncclOp) \
  IMPL_COLL3(coll, op, ncclFunc, u8,  uint8_t,  ncclColl, ncclOp, ncclUint8)
#elif NCCL_TYPE == 2
#define IMPL_COLL2(coll, op, ncclFunc, ncclColl, ncclOp) \
  IMPL_COLL3(coll, op, ncclFunc, i32, int32_t,  ncclColl, ncclOp, ncclInt32)
#elif NCCL_TYPE == 3
#define IMPL_COLL2(coll, op, ncclFunc, ncclColl, ncclOp) \
  IMPL_COLL3(coll, op, ncclFunc, u32, uint32_t, ncclColl, ncclOp, ncclUint32)
#elif NCCL_TYPE == 4
#define IMPL_COLL2(coll, op, ncclFunc, ncclColl, ncclOp) \
  IMPL_COLL3(coll, op, ncclFunc, i64, int64_t,  ncclColl, ncclOp, ncclInt64)
#elif NCCL_TYPE == 5
#define IMPL_COLL2(coll, op, ncclFunc, ncclColl, ncclOp) \
  IMPL_COLL3(coll, op, ncclFunc, u64, uint64_t, ncclColl, ncclOp, ncclUint64)
#elif NCCL_TYPE == 6
#define IMPL_COLL2(coll, op, ncclFunc, ncclColl, ncclOp) \
  IMPL_COLL3(coll, op, ncclFunc, f16, half,     ncclColl, ncclOp, ncclFloat16)
#elif NCCL_TYPE == 7
#define IMPL_COLL2(coll, op, ncclFunc, ncclColl, ncclOp) \
  IMPL_COLL3(coll, op, ncclFunc, f32, float,    ncclColl, ncclOp, ncclFloat32)
#elif NCCL_TYPE == 8
#define IMPL_COLL2(coll, op, ncclFunc, ncclColl, ncclOp) \
  IMPL_COLL3(coll, op, ncclFunc, f64, double,   ncclColl, ncclOp, ncclFloat64)
#endif

// Reduction define all functions
#if NCCL_OP == 0
#define IMPL_COLL_R(collf, colln) \
  IMPL_COLL2(collf, sum,  FuncSum,  colln, ncclSum);
#elif NCCL_OP == 1
#define IMPL_COLL_R(collf, colln) \
  IMPL_COLL2(collf, prod, FuncProd, colln, ncclProd);
#elif NCCL_OP == 2
#define IMPL_COLL_R(collf, colln) \
  IMPL_COLL2(collf, min,  FuncMin,  colln, ncclMin);
#elif NCCL_OP == 3
#define IMPL_COLL_R(collf, colln) \
  IMPL_COLL2(collf, max,  FuncMax,  colln, ncclMax);
#endif

// Copy primitives only define one
#if NCCL_OP == 0 && NCCL_TYPE == 0
#define IMPL_COLL_C(collf, colln) \
  IMPL_COLL3(collf, copy, FuncSum, i8, int8_t, colln, ncclSum, ncclInt8);
#else
#define IMPL_COLL_C(collf, colln)
#endif

#define COLL_UNROLL 4

#endif
