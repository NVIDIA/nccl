/*************************************************************************
 * Copyright (c) 2017-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_COLLECTIVES_H_
#define NCCL_COLLECTIVES_H_

#define FUNC_INDEX_P2P 0
#define FUNC_INDEX(coll, redop, dtype, al, pr) (1+(((((coll)*ncclNumOps + (redop))*ncclNumTypes) + (dtype))*NCCL_NUM_ALGORITHMS+(al))*NCCL_NUM_PROTOCOLS+(pr))

#define NCCL_COLL_NAME(coll, op, dtype) \
  coll##_##op##_##dtype

#define NCCL_KERN_NAME(coll, op, dtype) \
  coll##Kernel_##op##_##dtype

/* Declare all collective operations */
#define DECL_COLL5(coll, op, dtype) \
  extern __device__ void NCCL_COLL_NAME(coll, op, dtype)(struct CollectiveArgs* args); \
  extern __global__ void NCCL_KERN_NAME(coll, op, dtype)(struct ncclColl c); \

#define DECL_COLL4(coll, op, dtype) \
  DECL_COLL5(coll, op, dtype) \
  DECL_COLL5(coll##LL, op, dtype) \
  DECL_COLL5(coll##LL128, op, dtype)

#define DECL_COLL3(coll, op, dtype) \
  DECL_COLL4(coll##Ring, op, dtype) \
  DECL_COLL4(coll##Tree, op, dtype) \
  DECL_COLL4(coll##CollNet, op, dtype)

#define DECL_COLL2(coll, op) \
  DECL_COLL3(coll, op, i8) \
  DECL_COLL3(coll, op, u8) \
  DECL_COLL3(coll, op, i32) \
  DECL_COLL3(coll, op, u32) \
  DECL_COLL3(coll, op, i64) \
  DECL_COLL3(coll, op, u64) \
  DECL_COLL3(coll, op, f16) \
  DECL_COLL3(coll, op, f32) \
  DECL_COLL3(coll, op, f64)

#define DECL_COLL(coll) \
  DECL_COLL2(coll, sum) \
  DECL_COLL2(coll, prod) \
  DECL_COLL2(coll, min) \
  DECL_COLL2(coll, max)

#define DECL_ALL_COLLS \
  DECL_COLL2(ncclBroadcast, copy) \
  DECL_COLL(ncclReduce) \
  DECL_COLL2(ncclAllGather, copy) \
  DECL_COLL(ncclReduceScatter) \
  DECL_COLL(ncclAllReduce) \
  DECL_COLL5(ncclSendRecv,copy,i8) \

DECL_ALL_COLLS

// CHUNKSIZE must be a multiple of SLICESIZE
#define ALLREDUCE_SLICESTEPS (NCCL_STEPS/4)
#define ALLREDUCE_CHUNKSTEPS (NCCL_STEPS/2)
#define ALLGATHER_SLICESTEPS (NCCL_STEPS/4)
#define ALLGATHER_CHUNKSTEPS (NCCL_STEPS/2)
#define REDUCESCATTER_SLICESTEPS (NCCL_STEPS/4)
#define REDUCESCATTER_CHUNKSTEPS (NCCL_STEPS/2)
#define BROADCAST_SLICESTEPS 1
#define BROADCAST_CHUNKSTEPS 1
#define REDUCE_SLICESTEPS 1
#define REDUCE_CHUNKSTEPS 1
#define SENDRECV_SLICEFACTOR 4

#endif
