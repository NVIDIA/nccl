/*************************************************************************
 * Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "alloc.h"
#include "collectives.h"
#include "common_kernel.h"
#include "common.h"
#include <cuda_runtime.h>

namespace {
  template<typename RedOp>
  __global__ __launch_bounds__(512, 1)
  void oneRankReduce(void* dst, void* src, size_t nElts, uint64_t redOpArg, bool redOpArgIsPtr) {
    using T = typename RedOp::EltType;
    int tid = threadIdx.x;
    int tn = blockDim.x;
    int bid = blockIdx.x;
    int bn = gridDim.x;

    // each block/channel gets a roughly equal segment of 16 byte packs
    constexpr int EltPerPack = 16/sizeof(T);
    intptr_t i0 = (bid+0)*alignUp(nElts/bn, EltPerPack);
    intptr_t i1 = (bid+1)*alignUp(nElts/bn, EltPerPack);
    i0 = min(i0, nElts);
    i1 = min(i1, nElts);
    src = (T*)src + i0;
    dst = (T*)dst + i0;

    if (redOpArgIsPtr) {
      if (redOpArg%2 != 0) {
        redOpArg = *reinterpret_cast<uint8_t*>(redOpArg);
      } else if (redOpArg%4 != 0) {
        redOpArg = *reinterpret_cast<uint16_t*>(redOpArg);
      } else if (redOpArg%8 != 0) {
        redOpArg = *reinterpret_cast<uint32_t*>(redOpArg);
      } else {
        redOpArg = *reinterpret_cast<uint64_t*>(redOpArg);
      }
    }
    reduceCopy<COLL_UNROLL, RedOp, T, 0,1,1, 0,1,1, /*PreOpSrcs=*/1>
      (tid, tn, redOpArg, &redOpArg, true, 1, &src, 1, &dst, i1-i0);
  }
}

ncclResult_t ncclLaunchOneRank(void* dst, void const* src, size_t nElts, struct ncclDevRedOpFull redOp, ncclDataType_t eltType, cudaStream_t stream) {
  size_t eltSize = ncclTypeSize(eltType);
  if (redOp.op != ncclDevPreMulSum) {
    if (dst != src) {
      NCCLCHECK(ncclCudaMemcpyAsync((char*)dst, (char*)src, nElts*eltSize, stream));
    }
    return ncclSuccess;
  }

  void const* kernel;
  switch (eltType) {
  case ncclInt8:     kernel = (void const*)&oneRankReduce<FuncPreMulSum<int8_t>>; break;
  case ncclUint8:    kernel = (void const*)&oneRankReduce<FuncPreMulSum<uint8_t>>; break;
  case ncclInt32:    kernel = (void const*)&oneRankReduce<FuncPreMulSum<int32_t>>; break;
  case ncclUint32:   kernel = (void const*)&oneRankReduce<FuncPreMulSum<uint32_t>>; break;
  case ncclInt64:    kernel = (void const*)&oneRankReduce<FuncPreMulSum<int64_t>>; break;
  case ncclUint64:   kernel = (void const*)&oneRankReduce<FuncPreMulSum<uint64_t>>; break;
  case ncclFloat16:  kernel = (void const*)&oneRankReduce<FuncPreMulSum<half>>; break;
  #if defined(__CUDA_BF16_TYPES_EXIST__)
  case ncclBfloat16: kernel = (void const*)&oneRankReduce<FuncPreMulSum<__nv_bfloat16>>; break;
  #endif
  case ncclFloat32:  kernel = (void const*)&oneRankReduce<FuncPreMulSum<float>>; break;
  case ncclFloat64:  kernel = (void const*)&oneRankReduce<FuncPreMulSum<double>>; break;
  default: return ncclInvalidArgument;
  }
  dim3 grid = {0, 1, 1};
  grid.x = std::min(32, (int)divUp(nElts*eltSize, 16<<10));
  dim3 block = {512, 1, 1};
  void* args[5] = {&dst, &src, &nElts, &redOp.scalarArg, &redOp.scalarArgIsPtr};
  CUDACHECK(cudaLaunchKernel(kernel, grid, block, args, 0, stream));
  return ncclSuccess;
}
