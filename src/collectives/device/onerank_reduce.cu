/*************************************************************************
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "devcomm.h"
#include "collectives.h"
#include "common_kernel.h"
#include "common.h"

namespace {
  template<typename T, typename RedOp>
  __device__ __forceinline__ void oneRankReduce() {
    int tid = threadIdx.x;
    int tn = blockDim.x;
    int nWorks = ncclShmem.workBatch.nWorks;
    for (int w=tid; w < nWorks; w += tn) {
      struct ncclDevWorkColl *work = (ncclDevWorkColl*)(&ncclShmem.workBatch+1) + w;
      if (work->redOpArgIsPtr) {
        work->redOpArg = RedOpArg<RedOp>::loadArg(reinterpret_cast<void*>(work->redOpArg));
      }
    }
    __syncthreads();
    #pragma unroll 1
    for(int w=0; w < nWorks; w++) {
      struct ncclDevWorkColl *work = (ncclDevWorkColl*)(&ncclShmem.workBatch+1) + w;
      intptr_t eltN = work->count;
      int bid = work->bid;
      int bn = work->nChannels;
      T const *src = (T const*)work->sendbuff;
      T *dst = (T*)work->recvbuff;

      // each block/channel gets a roughly equal segment of 16 byte packs
      constexpr int EltPerPack = 16/sizeof(T);
      intptr_t i0 = (bid+0)*alignUp(eltN/bn, EltPerPack);
      intptr_t i1 = (bid+1)*alignUp(eltN/bn, EltPerPack);
      i0 = i0 < eltN ? i0 : eltN;
      i1 = i1 < eltN ? i1 : eltN;
      src += i0;
      dst += i0;
      void *vsrc = (void*)src;
      void *vdst = (void*)dst;
      reduceCopy<COLL_UNROLL, RedOp, T, 0,1,1, 0,1,1, /*PreOpSrcs=*/1>
        (tid, tn, work->redOpArg, &(work->redOpArg), true, 1, &vsrc, 1, &vdst, i1-i0);
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
