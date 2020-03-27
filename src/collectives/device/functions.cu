/*************************************************************************
 * Copyright (c) 2015-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "devcomm.h"
#include "collectives.h"
#include "common.h"

__device__ volatile uint64_t* ncclShmem;

#define NCCL_FUNC5(coll, op, dtype) \
  NCCL_COLL_NAME(coll##LL, op, dtype), \
  NCCL_COLL_NAME(coll##LL128, op, dtype), \
  NCCL_COLL_NAME(coll, op, dtype)

#define NCCL_FUNC4(coll, op, dtype) \
  NCCL_FUNC5(coll##Tree, op, dtype), \
  NCCL_FUNC5(coll##Ring, op, dtype), \
  NCCL_FUNC5(coll##CollNet, op, dtype)

// Must be consistent with ncclDataType_t
#define NCCL_FUNCS3A(coll, op) \
  NCCL_FUNC4(coll, op,  i8), \
  NCCL_FUNC4(coll, op,  u8), \
  NCCL_FUNC4(coll, op, i32), \
  NCCL_FUNC4(coll, op, u32), \
  NCCL_FUNC4(coll, op, i64), \
  NCCL_FUNC4(coll, op, u64), \
  NCCL_FUNC4(coll, op, f16), \
  NCCL_FUNC4(coll, op, f32), \
  NCCL_FUNC4(coll, op, f64)
#define NCCL_FUNCS3B(coll, op) \
  NCCL_FUNC4(coll, op,  i8), \
  NCCL_FUNC4(coll, op,  i8), \
  NCCL_FUNC4(coll, op,  i8), \
  NCCL_FUNC4(coll, op,  i8), \
  NCCL_FUNC4(coll, op,  i8), \
  NCCL_FUNC4(coll, op,  i8), \
  NCCL_FUNC4(coll, op,  i8), \
  NCCL_FUNC4(coll, op,  i8), \
  NCCL_FUNC4(coll, op,  i8)

// Must be consistent with ncclRedOp_t
#define NCCL_FUNCS2A(coll) \
  NCCL_FUNCS3A(coll, sum ), \
  NCCL_FUNCS3A(coll, prod), \
  NCCL_FUNCS3A(coll, max ), \
  NCCL_FUNCS3A(coll, min )
#define NCCL_FUNCS2B(coll) \
  NCCL_FUNCS3B(coll, copy), \
  NCCL_FUNCS3B(coll, copy), \
  NCCL_FUNCS3B(coll, copy), \
  NCCL_FUNCS3B(coll, copy)

// Must be consistent with ncclFunc_t
#define NCCL_FUNCS() { \
  NCCL_COLL_NAME(ncclSendRecv, copy, i8),\
  NCCL_FUNCS2B(ncclBroadcast), \
  NCCL_FUNCS2A(ncclReduce), \
  NCCL_FUNCS2B(ncclAllGather), \
  NCCL_FUNCS2A(ncclReduceScatter), \
  NCCL_FUNCS2A(ncclAllReduce) }

// Must be consistent with the ncclFuncSet enum
__device__ ncclKern_t ncclFuncs[1+NCCL_NUM_FUNCTIONS*ncclNumOps*ncclNumTypes*NCCL_NUM_ALGORITHMS*NCCL_NUM_PROTOCOLS] = {
// Don't try to initialize the host shadow copy of this device-side global
// variable. There is no host pointer to a device-side function, which
// confuses clang. This will be fixed in the next clang release.
#if __CUDA_ARCH__
  NCCL_COLL_NAME(ncclSendRecv, copy, i8),
  NCCL_FUNCS2B(ncclBroadcast),
  NCCL_FUNCS2A(ncclReduce),
  NCCL_FUNCS2B(ncclAllGather),
  NCCL_FUNCS2A(ncclReduceScatter),
  NCCL_FUNCS2A(ncclAllReduce)
#endif
};

// Workaround for https://reviews.llvm.org/D55580
__device__ void ncclWorkaroundClangD55580() {}
