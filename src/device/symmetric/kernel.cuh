#ifndef NCCL_DEVICE_SYMMETRIC_KERNEL_H_
#define NCCL_DEVICE_SYMMETRIC_KERNEL_H_

#include "sym_kernels.h"

template<template<typename> typename Red, typename T>
__device__ __forceinline__ void ncclSymkRun_AllReduce_AGxLL_R(struct ncclSymkDevWorkArgs const* args);
template<template<typename> typename Red, typename T>
__device__ __forceinline__ void ncclSymkRun_AllReduce_AGxLLMC_R(struct ncclSymkDevWorkArgs const* args);

template<template<typename> typename Red, typename T>
__device__ __forceinline__ void ncclSymkRun_AllReduce_RSxLD_AGxST(struct ncclSymkDevWorkArgs const* args);
template<template<typename> typename Red, typename T>
__device__ __forceinline__ void ncclSymkRun_AllReduce_RSxLDMC_AGxSTMC(struct ncclSymkDevWorkArgs const* args);

__device__ __forceinline__ void ncclSymkRun_AllGather_LL(struct ncclSymkDevWorkArgs const* args);
__device__ __forceinline__ void ncclSymkRun_AllGather_LLMC(struct ncclSymkDevWorkArgs const* args);
__device__ __forceinline__ void ncclSymkRun_AllGather_ST(struct ncclSymkDevWorkArgs const* args);
__device__ __forceinline__ void ncclSymkRun_AllGather_STMC(struct ncclSymkDevWorkArgs const* args);

template<template<typename> typename Red, typename T>
__device__ __forceinline__ void ncclSymkRun_ReduceScatter_LL(struct ncclSymkDevWorkArgs const* args);
template<template<typename> typename Red, typename T>
__device__ __forceinline__ void ncclSymkRun_ReduceScatter_LD(struct ncclSymkDevWorkArgs const* args);
template<template<typename> typename Red, typename T>
__device__ __forceinline__ void ncclSymkRun_ReduceScatter_LDMC(struct ncclSymkDevWorkArgs const* args);

__device__ __forceinline__ void ncclSymkRun_AllGather_GinHier_MCRing(struct ncclSymkDevWorkArgs const* args);
#endif
