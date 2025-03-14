#ifndef NCCL_REGISTER_INLINE_H_
#define NCCL_REGISTER_INLINE_H_

#include "comm.h"
#include "register.h"

static inline ncclResult_t ncclRegFind(struct ncclComm* comm, const void* data, size_t size, struct ncclReg** outReg) {
  struct ncclRegCache* cache = &comm->regCache;
  *outReg = NULL;
  for (int slot=0; /*true*/; slot++) {
    if (slot == cache->population) return ncclSuccess;
    struct ncclReg *reg = cache->slots[slot];
    if ((uintptr_t)data < reg->begAddr) return ncclSuccess;
    if ((uintptr_t)data + size <= reg->endAddr) {
      *outReg = reg;
      return ncclSuccess;
    }
  }
}

static inline ncclResult_t ncclRegFindSymmetric(struct ncclComm* comm, const void* data, size_t size, void** symPtr) {
  struct ncclReg* regRecord = NULL;
  size_t startAddr = (size_t)(comm->baseUCSymPtr + comm->baseStride * comm->localRank);

  *symPtr = NULL;
  if (startAddr <= (size_t)data && (size_t)data + size <= startAddr + comm->baseStride) {
    // it is a symmetric buffer, just return
    *symPtr = (void*)data;
  } else {
    NCCLCHECK(ncclRegFind(comm, data, size, &regRecord));
    if (regRecord && regRecord->baseSymPtr) {
      *symPtr = (void*)((uintptr_t)regRecord->baseSymPtr + (uintptr_t)data - (uintptr_t)regRecord->begAddr);
    }
  }
  return ncclSuccess;
}

#endif
