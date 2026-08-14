/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "rma_socket.h"

#include <string.h>

/* Memory layer: pointer-type checks, staging buffers, GDRCopy mapping,
 * receive-buffer copies, and symmetric MR registration. */

/* Generic helpers. */

bool ncclRmaSocketProxySupportedPtrType(int type) {
  return type == NCCL_PTR_HOST || type == NCCL_PTR_CUDA;
}

ncclResult_t ncclRmaSocketProxyAllocChunkBuffer(void** ptr, size_t size) {
  return ncclCudaHostCalloc((char**)ptr, size);
}

ncclResult_t ncclRmaSocketProxyFreeChunkBuffer(void* ptr) {
  return ptr ? ncclCudaHostFree(ptr) : ncclSuccess;
}

ncclResult_t ncclRmaSocketProxyFreeMr(struct ncclRmaSocketProxyMrHandle* handle) {
  ncclResult_t ret = ncclSuccess;
  if (handle == NULL) return ncclSuccess;
  if (handle->gdr.mapped && ncclGdrCopy) {
    NCCLCHECKIGNORE(ncclGdrCudaUnmapPointer(handle->gdr.mh, handle->gdr.map, handle->gdr.mapSize), ret);
  }
  free(handle);
  return ret;
}

ncclResult_t ncclRmaSocketProxyValidateRange(size_t objectSize, uint64_t offset, size_t size, const char* objectName) {
  if (offset > objectSize || size > objectSize - offset) {
    WARN("RMA/Socket : %s range out of bounds offset=%lu size=%zu objectSize=%zu", objectName, offset, size,
         objectSize);
    return ncclInvalidArgument;
  }
  return ncclSuccess;
}

ncclResult_t ncclRmaSocketProxyGdrPtr(struct ncclRmaSocketProxyMrHandle* handle, const void* ptr, size_t size,
                                      void** gdrPtr) {
  if (gdrPtr == NULL) return ncclInvalidArgument;
  *gdrPtr = NULL;
  if (handle == NULL || !handle->gdr.mapped || handle->gdr.barPtr == NULL) return ncclInternalError;
  uintptr_t base = (uintptr_t)handle->data;
  uintptr_t addr = (uintptr_t)ptr;
  if (addr < base) return ncclInvalidArgument;
  size_t offset = addr - base;
  if (offset > handle->size || size > handle->size - offset) return ncclInvalidArgument;
  *gdrPtr = (void*)((char*)handle->gdr.barPtr + offset);
  return ncclSuccess;
}

ncclResult_t ncclRmaSocketProxyCopy(void* dst, int dstType, struct ncclRmaSocketProxyMrHandle* dstHandle,
                                    const void* src, int srcType, struct ncclRmaSocketProxyMrHandle* srcHandle,
                                    size_t size) {
  if (size == 0) return ncclSuccess;
  if (dstType == NCCL_PTR_HOST && srcType == NCCL_PTR_HOST) {
    memcpy(dst, src, size);
    return ncclSuccess;
  }
  if (dstType == NCCL_PTR_HOST && srcType == NCCL_PTR_CUDA) {
    if (srcHandle == NULL) {
      WARN("RMA/Socket : GPU-to-CPU copy missing CUDA MR handle");
      return ncclInternalError;
    }
    void* gdrSrc = NULL;
    NCCLCHECK(ncclRmaSocketProxyGdrPtr(srcHandle, src, size, &gdrSrc));
    return wrap_gdr_copy_from_mapping(srcHandle->gdr.mh, dst, gdrSrc, size);
  }
  if (dstType == NCCL_PTR_CUDA && srcType == NCCL_PTR_HOST) {
    if (dstHandle == NULL) {
      WARN("RMA/Socket : CPU-to-GPU copy missing CUDA MR handle");
      return ncclInternalError;
    }
    void* gdrDst = NULL;
    NCCLCHECK(ncclRmaSocketProxyGdrPtr(dstHandle, dst, size, &gdrDst));
    return wrap_gdr_copy_to_mapping(dstHandle->gdr.mh, gdrDst, src, size);
  }
  WARN("RMA/Socket : unsupported copy direction srcType=%d dstType=%d", srcType, dstType);
  return ncclInvalidArgument;
}

/* plugin API functions */

ncclResult_t ncclRmaSocketProxyRegMrSym(void* collComm, void* data, size_t size, int type, uint64_t mrFlags,
                                        void** mhandle) {
  if (collComm == NULL || mhandle == NULL) {
    WARN("RMA/Socket : invalid regMrSym arguments collComm=%p mhandle=%p", collComm, mhandle);
    return ncclInvalidArgument;
  }
  if (!ncclRmaSocketProxySupportedPtrType(type)) {
    WARN("RMA/Socket : unsupported memory registration type=%d", type);
    return ncclInvalidArgument;
  }

  ncclResult_t ret = ncclSuccess;
  struct ncclRmaSocketProxyCollComm* comm = (struct ncclRmaSocketProxyCollComm*)collComm;
  struct ncclRmaSocketProxyMrHandle* handle = NULL;
  *mhandle = NULL;

  NCCLCHECKGOTO(ncclCalloc(&handle, 1), ret, fail);
  handle->data = data;
  handle->size = size;
  handle->type = type;
  /* Collective, ordered registration assigns the same collComm-local ID on every rank. */
  handle->mrId = comm->nextMrId;
  if (handle->type == NCCL_PTR_CUDA && handle->size > 0) {
    gdr_info_t info;
    NCCLCHECKGOTO(ncclGdrCudaMapPointer(handle->data, handle->size, &handle->gdr.mh, &handle->gdr.map,
                                        &handle->gdr.mapSize, &info),
                  ret, fail);
    handle->gdr.barPtr = (void*)((char*)handle->gdr.map + ((uintptr_t)handle->data - info.va));
    handle->gdr.mapped = 1;
  }

  comm->nextMrId++;
  handle->next = comm->mrHead;
  comm->mrHead = handle;
  *mhandle = handle;
  return ncclSuccess;
fail:
  NCCLCHECKIGNORE(ncclRmaSocketProxyFreeMr(handle), ret);
  return ret;
}

ncclResult_t ncclRmaSocketProxyDeregMrSym(void* collComm, void* mhandle) {
  if (collComm == NULL || mhandle == NULL) {
    WARN("RMA/Socket : invalid deregMrSym arguments collComm=%p mhandle=%p", collComm, mhandle);
    return ncclInvalidArgument;
  }

  struct ncclRmaSocketProxyCollComm* comm = (struct ncclRmaSocketProxyCollComm*)collComm;
  struct ncclRmaSocketProxyMrHandle* handle = (struct ncclRmaSocketProxyMrHandle*)mhandle;
  struct ncclRmaSocketProxyMrHandle* prev = NULL;
  struct ncclRmaSocketProxyMrHandle* cur = comm->mrHead;
  while (cur != NULL && cur != handle) {
    prev = cur;
    cur = cur->next;
  }

  if (cur == NULL) {
    WARN("RMA/Socket : deregMrSym called with unknown handle=%p", mhandle);
    return ncclInvalidArgument;
  }

  if (prev == NULL) {
    comm->mrHead = cur->next;
  } else {
    prev->next = cur->next;
  }
  NCCLCHECK(ncclRmaSocketProxyFreeMr(handle));
  return ncclSuccess;
}

ncclResult_t ncclRmaSocketProxyRegMrSymDmaBuf(void*, void*, size_t, int, uint64_t, int, uint64_t, void**) {
  // WARN("RMA/Socket : RegMrSymDmaBuf is not supported.");
  return ncclInvalidUsage;
}
