/*************************************************************************
 * Copyright (c) 2015-2025, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_ALLOCATOR_H_
#define NCCL_ALLOCATOR_H_

#include "nccl.h"
#include <stdint.h>
#include <cuda_runtime.h>

////////////////////////////////////////////////////////////////////////////////
// ncclSpace: Allocates contiguous segments of non-negative integers. Useful
// as a memory allocator when we can't put allocator state within the memory
// being allocated.

struct ncclSpace {
  int count;
  int capacity;
  int64_t* cuts;
};

void ncclSpaceConstruct(struct ncclSpace* a);
void ncclSpaceDestruct(struct ncclSpace* a);
ncclResult_t ncclSpaceAlloc(struct ncclSpace* a, int64_t spaceLimit, int64_t objSize, int objAlign, int64_t* outObjOffset);
ncclResult_t ncclSpaceFree(struct ncclSpace* a, int64_t objOffset, int64_t objSize);


////////////////////////////////////////////////////////////////////////////////
// ncclShadowPool: Allocates device-side objects, their host-side shadows, and
// maintains the device->host object address mapping.

struct ncclShadowObject;
struct ncclShadowPage;
struct ncclShadowPool {
  int count, hbits;
  struct ncclShadowObject** table;
  cudaMemPool_t memPool;
  struct ncclShadowPage* pages;
};

void ncclShadowPoolConstruct(struct ncclShadowPool*);
ncclResult_t ncclShadowPoolDestruct(struct ncclShadowPool*);
ncclResult_t ncclShadowPoolAlloc(struct ncclShadowPool*, size_t size, void** outDevObj, void** outHostObj, cudaStream_t stream);
ncclResult_t ncclShadowPoolFree(struct ncclShadowPool*, void* devObj, cudaStream_t stream);
ncclResult_t ncclShadowPoolToHost(struct ncclShadowPool*, void* devObj, void** outHostObj);

template<typename T>
static inline ncclResult_t ncclShadowPoolAlloc(struct ncclShadowPool* pool, T** outDevObj, T** outHostObj, cudaStream_t stream) {
  void* devObj;
  void* hostObj;
  ncclResult_t got = ncclShadowPoolAlloc(pool, sizeof(T), &devObj, &hostObj, stream);
  if (outDevObj) *outDevObj = (T*)devObj;
  if (outHostObj) *outHostObj = (T*)hostObj;
  return got;
}

template<typename T>
static inline ncclResult_t ncclShadowPoolToHost(struct ncclShadowPool* pool, T* devObj, T** hostObj) {
  return ncclShadowPoolToHost(pool, (void*)devObj, (void**)hostObj);
}

#endif
