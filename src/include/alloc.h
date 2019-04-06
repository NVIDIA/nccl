/*************************************************************************
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_ALLOC_H_
#define NCCL_ALLOC_H_

#include "nccl.h"
#include "checks.h"
#include <sys/mman.h>

static inline ncclResult_t ncclCudaHostAlloc(void** ptr, void** devPtr, size_t size) {
  CUDACHECK(cudaHostAlloc(ptr, size, cudaHostAllocMapped));
  memset(*ptr, 0, size);
  *devPtr = *ptr;
  return ncclSuccess;
}

static inline ncclResult_t ncclCudaHostFree(void* ptr) {
  CUDACHECK(cudaFreeHost(ptr));
  return ncclSuccess;
}

template <typename T>
static ncclResult_t ncclCalloc(T** ptr, size_t nelem) {
  void* p = malloc(nelem*sizeof(T));
  if (p == NULL) {
    WARN("Failed to malloc %ld bytes", nelem*sizeof(T));
    return ncclSystemError;
  }
  memset(p, 0, nelem*sizeof(T));
  *ptr = (T*)p;
  return ncclSuccess;
}

template <typename T>
static ncclResult_t ncclCudaCalloc(T** ptr, size_t nelem) {
  CUDACHECK(cudaMalloc(ptr, nelem*sizeof(T)));
  CUDACHECK(cudaMemset(*ptr, 0, nelem*sizeof(T)));
  return ncclSuccess;
}

template <typename T>
static ncclResult_t ncclCudaMemcpy(T* dst, T* src, size_t nelem) {
  CUDACHECK(cudaMemcpy(dst, src, nelem*sizeof(T), cudaMemcpyDefault));
  return ncclSuccess;
}

#endif
