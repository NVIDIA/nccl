/*************************************************************************
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "devcomm.h"
#include "collectives.h"
#include "reduce_kernel.h"
#include "common.h"

namespace {
  template<typename T, typename RedOp>
  __device__ __forceinline__ void oneRankReduce() {
    ncclWork *w = &ncclShmem.work;
    int tid = threadIdx.x;
    int tn = blockDim.x;
    #pragma unroll 1
    for(int e=0; e < NCCL_MAX_WORK_ELEMENTS && w->elems[e].header.type != ncclWorkTypeUnused; e++) {
      ncclWorkElem *we = &w->elems[e];
      intptr_t eltN = we->count;
      int bid = we->bid;
      int bn = we->nChannels;
      T const *src = (T const*)we->sendbuff;
      T *dst = (T*)we->recvbuff;

      // each block/channel gets a roughly equal segment of 16 byte packs
      constexpr int EltPerPack = 16/sizeof(T);
      intptr_t packN = (eltN + EltPerPack-1) - (eltN + EltPerPack-1)%EltPerPack;
      intptr_t i0 = (bid+0)*(packN/bn) + (bid+0 < packN%bn ? bid+0 : packN%bn);
      intptr_t i1 = (bid+1)*(packN/bn) + (bid+1 < packN%bn ? bid+1 : packN%bn);
      i0 *= EltPerPack;
      i0 = i0 < eltN ? i0 : eltN;
      i1 *= EltPerPack;
      i1 = i1 < eltN ? i1 : eltN;
      src += i0;
      dst += i0;
      ReduceOrCopyMulti<COLL_UNROLL, RedOp, T, 1, 1, 1, 1, 1>
        (tid, tn, &(we->redOpArg), true, 1, &src, 1, &dst, i1-i0);
    }
  }
}

#define INSTANTIATE(devredop, type) \
  __device__ void NCCL_ONERANK_REDUCE_NAME(devredop, type)() { \
    oneRankReduce<type, Func##devredop<type>>(); \
  }

INSTANTIATE(PreMulSum, int8_t)
INSTANTIATE(PreMulSum, uint8_t)
INSTANTIATE(PreMulSum, int32_t)
INSTANTIATE(PreMulSum, uint32_t)
INSTANTIATE(PreMulSum, int64_t)
INSTANTIATE(PreMulSum, uint64_t)
INSTANTIATE(PreMulSum, half)
#if defined(__CUDA_BF16_TYPES_EXIST__)
INSTANTIATE(PreMulSum, __nv_bfloat16)
#endif
INSTANTIATE(PreMulSum, float)
INSTANTIATE(PreMulSum, double)
