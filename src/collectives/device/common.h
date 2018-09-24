/*************************************************************************
 * Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_DEVICE_COMMON_H_
#define NCCL_DEVICE_COMMON_H_

#include "../collectives.h"
#include "core.h"
#include "nccl.h"

typedef void(*ncclKern_t)(struct CollectiveArgs* args);
extern __device__ ncclKern_t ncclFuncs[];

static __device__ void load_parallel(void* dst, void* src, size_t size, int tid) {
  int* d = (int*)dst;
  int* s = (int*)src;
  __syncthreads();
  for (int o = tid; o < (size/sizeof(int)); o += blockDim.x) d[o] = s[o];
  __syncthreads();
}
static __device__ void load_coll(struct ncclColl* localColl, struct ncclColl* hostColl, int tid) {
  load_parallel(localColl, hostColl, sizeof(struct ncclColl), tid);
  if (tid == 0) hostColl->active = 0;
}

/* Functions for aggregation case */
#define IMPL_COLL4(coll, op, ncclFunc, dtype, ctype) \
__device__ void NCCL_COLL_NAME(coll, op, dtype)(struct CollectiveArgs* args) { \
  coll##Kernel<UNROLL, ncclFunc<ctype>, ctype>(args); \
}
/* Kernels with the first operation inlined */
#define IMPL_COLL4K(coll, op, ncclFunc, dtype, ctype, fIndex) \
__launch_bounds__(MAXTHREADS+WARP_SIZE, 1) \
__global__ void NCCL_KERN_NAME(coll, op, dtype)(struct ncclColl firstColl) { \
  int tid = threadIdx.x; \
  int bid = blockIdx.x; \
  __shared__ struct ncclColl localColl; \
 \
  struct ncclComm* comm = firstColl.args.comm; \
  struct ncclRing* ring = comm->rings+bid; \
  struct ncclColl* c; \
  if (bid == 0) { \
    /* To optimize for latency, (only) the first operation is passed as argument.*/ \
    c = &firstColl; \
  } else { \
    c = &localColl; \
    load_coll(c, ring->devCollectives+ring->collFifoHead, tid); \
  } \
  while (1) { \
    if (tid < c->nThreads) { \
      if (c->funcIndex == fIndex) { \
        coll##Kernel<UNROLL, ncclFunc<ctype>, ctype>(&c->args); \
      } else { \
        ncclFuncs[c->funcIndex](&c->args); \
      } \
    } \
    int nextIndex = c->nextIndex; \
    if (tid == 0) ring->collFifoHead = nextIndex; \
 \
    if (c->active == 2) { \
      return; \
    } \
 \
    /* Load next collective operation*/ \
    c = &localColl; /* for bid 0 */ \
    load_coll(c, ring->devCollectives+nextIndex, tid); \
  } \
}

#define IMPL_COLL3(coll, op, ncclFunc, dtype, ctype, ncclColl, ncclOp, ncclType) \
  IMPL_COLL4(coll##LL, op, ncclFunc, dtype, ctype) \
  IMPL_COLL4K(coll##LL, op, ncclFunc, dtype, ctype, FUNC_INDEX(ncclColl, ncclOp, ncclType, 1)) \
  IMPL_COLL4(coll, op, ncclFunc, dtype, ctype) \
  IMPL_COLL4K(coll, op, ncclFunc, dtype, ctype, FUNC_INDEX(ncclColl, ncclOp, ncclType, 0)) \

#define IMPL_COLL2(coll, op, ncclFunc, ncclColl, ncclOp) \
  IMPL_COLL3(coll, op, ncclFunc, i8,  int8_t,   ncclColl, ncclOp, ncclInt8) \
  IMPL_COLL3(coll, op, ncclFunc, u8,  uint8_t,  ncclColl, ncclOp, ncclUint8) \
  IMPL_COLL3(coll, op, ncclFunc, i32, int32_t,  ncclColl, ncclOp, ncclInt32) \
  IMPL_COLL3(coll, op, ncclFunc, u32, uint32_t, ncclColl, ncclOp, ncclUint32) \
  IMPL_COLL3(coll, op, ncclFunc, i64, int64_t,  ncclColl, ncclOp, ncclInt64) \
  IMPL_COLL3(coll, op, ncclFunc, u64, uint64_t, ncclColl, ncclOp, ncclUint64) \
  IMPL_COLL3(coll, op, ncclFunc, f16, half,     ncclColl, ncclOp, ncclFloat16) \
  IMPL_COLL3(coll, op, ncclFunc, f32, float,    ncclColl, ncclOp, ncclFloat32) \
  IMPL_COLL3(coll, op, ncclFunc, f64, double,   ncclColl, ncclOp, ncclFloat64)

#endif
