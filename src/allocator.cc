/*************************************************************************
 * Copyright (c) 2015-2025, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "comm.h"
#include "transport.h"
#include "group.h"

NCCL_API(ncclResult_t, ncclMemAlloc, void **ptr, size_t size);
ncclResult_t  ncclMemAlloc(void **ptr, size_t size) {
  NVTX3_FUNC_RANGE_IN(nccl_domain);
  ncclResult_t ret = ncclSuccess;

#if CUDART_VERSION >= 12010
  size_t memGran = 0;
  CUdevice currentDev;
  CUmemAllocationProp memprop = {};
  CUmemAccessDesc accessDesc = {};
  CUmemGenericAllocationHandle handle = (CUmemGenericAllocationHandle)-1;
  int cudaDev;
  int flag;
  int dcnt;

  if (ptr == NULL || size == 0) goto fallback;

  if (ncclCudaLibraryInit() != ncclSuccess) goto fallback;

  CUDACHECK(cudaGetDevice(&cudaDev));
  CUCHECK(cuDeviceGet(&currentDev, cudaDev));

  if (ncclCuMemEnable()) {
    size_t handleSize = size;
    int requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
    // Query device to see if FABRIC handle support is available
    flag = 0;
    (void) CUPFN(cuDeviceGetAttribute(&flag, CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED, currentDev));
    if (flag) requestedHandleTypes |= CU_MEM_HANDLE_TYPE_FABRIC;
    memprop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    memprop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    memprop.requestedHandleTypes = (CUmemAllocationHandleType) requestedHandleTypes;
    memprop.location.id = currentDev;
    // Query device to see if RDMA support is available
    flag = 0;
    CUCHECK(cuDeviceGetAttribute(&flag, CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED, currentDev));
    if (flag) memprop.allocFlags.gpuDirectRDMACapable = 1;
    CUCHECK(cuMemGetAllocationGranularity(&memGran, &memprop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));
    CUDACHECK(cudaGetDeviceCount(&dcnt));
    ALIGN_SIZE(handleSize, memGran);

    if (requestedHandleTypes & CU_MEM_HANDLE_TYPE_FABRIC) {
      /* First try cuMemCreate() with FABRIC handle support and then remove if it fails */
      CUresult err = CUPFN(cuMemCreate(&handle, handleSize, &memprop, 0));
      if (err == CUDA_ERROR_NOT_PERMITTED || err == CUDA_ERROR_NOT_SUPPORTED) {
        requestedHandleTypes &= ~CU_MEM_HANDLE_TYPE_FABRIC;
        memprop.requestedHandleTypes = (CUmemAllocationHandleType) requestedHandleTypes;
        /* Allocate the physical memory on the device */
        CUCHECK(cuMemCreate(&handle, handleSize, &memprop, 0));
      } else if (err != CUDA_SUCCESS) {
        // Catch and report any error from above
        CUCHECK(cuMemCreate(&handle, handleSize, &memprop, 0));
      }
    } else {
      /* Allocate the physical memory on the device */
      CUCHECK(cuMemCreate(&handle, handleSize, &memprop, 0));
    }
    /* Reserve a virtual address range */
    CUCHECK(cuMemAddressReserve((CUdeviceptr*)ptr, handleSize, memGran, 0, 0));
    /* Map the virtual address range to the physical allocation */
    CUCHECK(cuMemMap((CUdeviceptr)*ptr, handleSize, 0, handle, 0));
    /* Now allow RW access to the newly mapped memory */
    for (int i = 0; i < dcnt; ++i) {
      int p2p = 0;
      if (i == cudaDev || ((cudaDeviceCanAccessPeer(&p2p, i, cudaDev) == cudaSuccess) && p2p)) {
        accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        accessDesc.location.id = i;
        accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        CUCHECK(cuMemSetAccess((CUdeviceptr)*ptr, handleSize, &accessDesc, 1));
      }
      if (0 == p2p && i != cudaDev) INFO(NCCL_ALLOC, "P2P not supported between GPU%d and GPU%d", cudaDev, i);
    }
    goto exit;
  }

fallback:
#endif
  // Coverity is right to complain that we may pass a NULL ptr to cudaMalloc.  That's deliberate though:
  // we want CUDA to return an error to the caller.
  // coverity[var_deref_model]
  CUDACHECKGOTO(cudaMalloc(ptr, size), ret, fail);

exit:
  return ret;
fail:
  goto exit;
}

NCCL_API(ncclResult_t, ncclMemFree, void *ptr);
ncclResult_t  ncclMemFree(void *ptr) {
  NVTX3_FUNC_RANGE_IN(nccl_domain);
  ncclResult_t ret = ncclSuccess;
  int saveDevice;

  CUDACHECK(cudaGetDevice(&saveDevice));
#if CUDART_VERSION >= 12010
  CUdevice ptrDev = 0;

  if (ptr == NULL) goto fallback;
  if (ncclCudaLibraryInit() != ncclSuccess) goto fallback;

  CUCHECKGOTO(cuPointerGetAttribute((void*)&ptrDev, CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL, (CUdeviceptr)ptr), ret, fail);
  CUDACHECKGOTO(cudaSetDevice((int)ptrDev), ret, fail);
  if (ncclCuMemEnable()) {
    NCCLCHECKGOTO(ncclCuMemFree(ptr), ret, fail);
    goto exit;
  }

fallback:
#endif
  CUDACHECKGOTO(cudaFree(ptr), ret, fail);

exit:
  CUDACHECK(cudaSetDevice(saveDevice));
  return ret;
fail:
  goto exit;
}

// This is a collective function and should be called by all ranks in the communicator
ncclResult_t ncclCommSymmetricAllocInternal(struct ncclComm* comm, size_t size, size_t alignment, void** symPtr) {
  ncclResult_t ret = ncclSuccess;
  void* regSymAddr = NULL;
  size_t allocSize = size;
  size_t granularity;
  CUdevice cuDev;
  CUmemAllocationProp memprop = {};
  CUmemGenericAllocationHandle memHandle;
  int bit = 0, cnt = 0;

  // aligment must be power of 2 as an input
  while (bit < sizeof(size_t) * 8) {
    if (alignment & (1L << bit)) cnt++;
    if (cnt == 2) {
      WARN("rank %d alignment %ld is not power of 2", comm->rank, alignment);
      goto fail;
    }
    bit++;
  }
  // temporarily align the alignment to NCCL_REC_PAGE_SIZE
  ALIGN_SIZE(alignment, NCCL_REC_PAGE_SIZE);

  CUCHECKGOTO(cuDeviceGet(&cuDev, comm->cudaDev), ret, fail);
  memprop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  memprop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  memprop.requestedHandleTypes = ncclCuMemHandleType;
  memprop.location.id = cuDev;
  CUCHECKGOTO(cuMemGetAllocationGranularity(&granularity, &memprop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED), ret, fail);
  ALIGN_SIZE(allocSize, granularity);

  CUCHECKGOTO(cuMemCreate(&memHandle, allocSize, &memprop, 0), ret, fail);
  ALIGN_SIZE(comm->symAllocHead, alignment);
  NCCLCHECKGOTO(ncclIpcSymmetricMap(comm, comm->symAllocHead, allocSize, memHandle, &regSymAddr), ret, fail);
  NCCLCHECKGOTO(ncclNvlsSymmetricMap(comm, comm->symAllocHead, allocSize, regSymAddr), ret, fail);
  NCCLCHECKGOTO(bootstrapIntraNodeBarrier(comm->bootstrap, comm->localRankToRank, comm->localRank, comm->localRanks, comm->localRankToRank[0]), ret, fail);
  comm->symAllocHead += allocSize;
  *symPtr = regSymAddr;

exit:
  return ret;
fail:
  *symPtr = NULL;
  goto exit;
}

ncclResult_t ncclCommSymmetricFreeInternal(struct ncclComm* comm, void* symPtr) {
  CUmemGenericAllocationHandle handle;
  size_t size = 0;
  ncclResult_t ret = ncclSuccess;
  int saveDev = comm->cudaDev;
  CUDACHECKGOTO(cudaGetDevice(&saveDev), ret, fail);
  if (ncclCuMemEnable()) {
    CUDACHECKGOTO(cudaSetDevice(comm->cudaDev), ret, fail);
    CUCHECKGOTO(cuMemRetainAllocationHandle(&handle, symPtr), ret, fail);
    CUCHECKGOTO(cuMemRelease(handle), ret, fail);
    CUCHECKGOTO(cuMemGetAddressRange(NULL, &size, (CUdeviceptr)symPtr), ret, fail);
    NCCLCHECKGOTO(ncclNvlsSymmetricFree(comm, size, symPtr), ret, fail);
    NCCLCHECKGOTO(ncclIpcSymmetricFree(comm, size, symPtr), ret, fail);
    CUCHECKGOTO(cuMemRelease(handle), ret, fail);
  }
exit:
  CUDACHECK(cudaSetDevice(saveDev));
  return ret;
fail:
  goto exit;
}
