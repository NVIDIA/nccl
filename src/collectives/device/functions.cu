/*************************************************************************
 * Copyright (c) 2015-2021, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "devcomm.h"
#include "collectives.h"
#include "common.h"

__shared__ ncclShmemData ncclShmem;

#define NCCL_FUNC5(func, algo, redop, type) \
  NCCL_FUNC_NAME(func, algo, LL,     redop, type), \
  NCCL_FUNC_NAME(func, algo, LL128,  redop, type), \
  NCCL_FUNC_NAME(func, algo, SIMPLE, redop, type)

#define NCCL_FUNC4(func, redop, type) \
  NCCL_FUNC5(func, TREE,    redop, type), \
  NCCL_FUNC5(func, RING,    redop, type), \
  NCCL_FUNC5(func, COLLNET, redop, type)

#if defined(__CUDA_BF16_TYPES_EXIST__)
// Must be consistent with ncclDataType_t
#define NCCL_FUNCS3A(func, redop) \
  NCCL_FUNC4(func, redop, int8_t), \
  NCCL_FUNC4(func, redop, uint8_t), \
  NCCL_FUNC4(func, redop, int32_t), \
  NCCL_FUNC4(func, redop, uint32_t), \
  NCCL_FUNC4(func, redop, int64_t), \
  NCCL_FUNC4(func, redop, uint64_t), \
  NCCL_FUNC4(func, redop, half), \
  NCCL_FUNC4(func, redop, float), \
  NCCL_FUNC4(func, redop, double), \
  NCCL_FUNC4(func, redop, __nv_bfloat16)
#define NCCL_FUNCS3B(func, redop) \
  NCCL_FUNC4(func, redop, int8_t), \
  NCCL_FUNC4(func, redop, int8_t), \
  NCCL_FUNC4(func, redop, int8_t), \
  NCCL_FUNC4(func, redop, int8_t), \
  NCCL_FUNC4(func, redop, int8_t), \
  NCCL_FUNC4(func, redop, int8_t), \
  NCCL_FUNC4(func, redop, int8_t), \
  NCCL_FUNC4(func, redop, int8_t), \
  NCCL_FUNC4(func, redop, int8_t), \
  NCCL_FUNC4(func, redop, int8_t)
#else
// Must be consistent with ncclDataType_t
#define NCCL_FUNCS3A(func, redop) \
  NCCL_FUNC4(func, redop, int8_t), \
  NCCL_FUNC4(func, redop, uint8_t), \
  NCCL_FUNC4(func, redop, int32_t), \
  NCCL_FUNC4(func, redop, uint32_t), \
  NCCL_FUNC4(func, redop, int64_t), \
  NCCL_FUNC4(func, redop, uint64_t), \
  NCCL_FUNC4(func, redop, half), \
  NCCL_FUNC4(func, redop, float), \
  NCCL_FUNC4(func, redop, double)
#define NCCL_FUNCS3B(func, redop) \
  NCCL_FUNC4(func, redop, int8_t), \
  NCCL_FUNC4(func, redop, int8_t), \
  NCCL_FUNC4(func, redop, int8_t), \
  NCCL_FUNC4(func, redop, int8_t), \
  NCCL_FUNC4(func, redop, int8_t), \
  NCCL_FUNC4(func, redop, int8_t), \
  NCCL_FUNC4(func, redop, int8_t), \
  NCCL_FUNC4(func, redop, int8_t), \
  NCCL_FUNC4(func, redop, int8_t)
#endif

// Must be consistent with ncclRedOp_t
#define NCCL_FUNCS2A(func) \
  NCCL_FUNCS3A(func, Sum ), \
  NCCL_FUNCS3A(func, Prod), \
  NCCL_FUNCS3A(func, Max ), \
  NCCL_FUNCS3A(func, Min ), \
  NCCL_FUNCS3A(func, Avg)

#define NCCL_FUNCS2B(func) \
  NCCL_FUNCS3B(func, Sum), \
  NCCL_FUNCS3B(func, Sum), \
  NCCL_FUNCS3B(func, Sum), \
  NCCL_FUNCS3B(func, Sum), \
  NCCL_FUNCS3B(func, Sum)

// Must be consistent with ncclFunc_t
#define NCCL_FUNCS() { \
  NCCL_FUNC_NAME(SendRecv, RING, SIMPLE, Sum, int8_t),\
  NCCL_FUNCS2B(Broadcast), \
  NCCL_FUNCS2A(Reduce), \
  NCCL_FUNCS2B(AllGather), \
  NCCL_FUNCS2A(ReduceScatter), \
  NCCL_FUNCS2A(AllReduce) }

// Must be consistent with the ncclFuncSet enum
__device__ ncclKern_t ncclFuncs[1+NCCL_NUM_FUNCTIONS*ncclNumOps*ncclNumTypes*NCCL_NUM_ALGORITHMS*NCCL_NUM_PROTOCOLS] = {
// Don't try to initialize the host shadow copy of this device-side global
// variable. There is no host pointer to a device-side function, which
// confuses clang. This will be fixed in the next clang release.
#if __CUDA_ARCH__
  NCCL_FUNC_NAME(SendRecv, RING, SIMPLE, Sum, int8_t),
  NCCL_FUNCS2B(Broadcast),
  NCCL_FUNCS2A(Reduce),
  NCCL_FUNCS2B(AllGather),
  NCCL_FUNCS2A(ReduceScatter),
  NCCL_FUNCS2A(AllReduce)
#endif
};

// Workaround for https://reviews.llvm.org/D55580
__device__ void ncclWorkaroundClangD55580() {}
