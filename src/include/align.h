/*************************************************************************
 * Copyright (c) 2015-2019, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_ALIGN_H_
#define NCCL_ALIGN_H_

#define DIVUP(x, y) \
    (((x)+(y)-1)/(y))

#define ROUNDUP(x, y) \
    (DIVUP((x), (y))*(y))

#define ALIGN_SIZE(size, align) \
  size = ((size + (align) - 1) / (align)) * (align);

#if !__CUDA_ARCH__
  #ifndef __host__
    #define __host__
  #endif
  #ifndef __device__
    #define __device__
  #endif
#endif

template<typename X, typename Y, typename Z = decltype(X()+Y())>
__host__ __device__ constexpr Z divUp(X x, Y y) {
  return (x+y-1)/y;
}

template<typename X, typename Y, typename Z = decltype(X()+Y())>
__host__ __device__ constexpr Z roundUp(X x, Y y) {
  return (x+y-1) - (x+y-1)%y;
}

// assumes second argument is a power of 2
template<typename X, typename Z = decltype(X()+int())>
__host__ __device__ constexpr Z alignUp(X x, int a) {
  return (x+a-1) & Z(-a);
}

#endif
