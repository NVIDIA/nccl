#ifndef NCCL_DEVICE_SYMMETRIC_KERNEL_H_
#define NCCL_DEVICE_SYMMETRIC_KERNEL_H_

#include "symmetric.h"

template<template<typename> typename Red, typename T>
__device__ __forceinline__ void ncclSymRun_AllReduce_AGxLL_R(struct ncclSymDevArgs const* args);
template<template<typename> typename Red, typename T>
__device__ __forceinline__ void ncclSymRun_AllReduce_AGxLLMC_R(struct ncclSymDevArgs const* args);

template<template<typename> typename Red, typename T>
__device__ __forceinline__ void ncclSymRun_AllReduce_RSxLD_AGxST(struct ncclSymDevArgs const* args);
template<template<typename> typename Red, typename T>
__device__ __forceinline__ void ncclSymRun_AllReduce_RSxLDMC_AGxSTMC(struct ncclSymDevArgs const* args);

__device__ __forceinline__ void ncclSymRun_AllGather_LL(struct ncclSymDevArgs const* args);
__device__ __forceinline__ void ncclSymRun_AllGather_LLMC(struct ncclSymDevArgs const* args);
__device__ __forceinline__ void ncclSymRun_AllGather_ST(struct ncclSymDevArgs const* args);
__device__ __forceinline__ void ncclSymRun_AllGather_STMC(struct ncclSymDevArgs const* args);

template<template<typename> typename Red, typename T>
__device__ __forceinline__ void ncclSymRun_ReduceScatter_LL(struct ncclSymDevArgs const* args);
template<template<typename> typename Red, typename T>
__device__ __forceinline__ void ncclSymRun_ReduceScatter_LD(struct ncclSymDevArgs const* args);
template<template<typename> typename Red, typename T>
__device__ __forceinline__ void ncclSymRun_ReduceScatter_LDMC(struct ncclSymDevArgs const* args);
#endif
