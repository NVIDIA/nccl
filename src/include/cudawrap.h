/*************************************************************************
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_CUDAWRAP_H_
#define NCCL_CUDAWRAP_H_

#include <cuda.h>
#include <cuda_runtime.h>
#include "checks.h"

// Is cuMem API usage enabled
extern int ncclCuMemEnable();

#if CUDART_VERSION >= 11030
#include <cudaTypedefs.h>

// Handle type used for cuMemCreate()
extern CUmemAllocationHandleType ncclCuMemHandleType;

#endif

#define CUPFN(symbol) pfn_##symbol

// Check CUDA PFN driver calls
#define CUCHECK(cmd) do {				      \
    CUresult err = pfn_##cmd;				      \
    if( err != CUDA_SUCCESS ) {				      \
      const char *errStr;				      \
      (void) pfn_cuGetErrorString(err, &errStr);	      \
      WARN("Cuda failure %d '%s'", err, errStr);	      \
      return ncclUnhandledCudaError;			      \
    }							      \
} while(false)

#define CUCHECKGOTO(cmd, res, label) do {		      \
    CUresult err = pfn_##cmd;				      \
    if( err != CUDA_SUCCESS ) {				      \
      const char *errStr;				      \
      (void) pfn_cuGetErrorString(err, &errStr);	      \
      WARN("Cuda failure %d '%s'", err, errStr);	      \
      res = ncclUnhandledCudaError;			      \
      goto label;					      \
    }							      \
} while(false)

// Report failure but clear error and continue
#define CUCHECKIGNORE(cmd) do {						\
    CUresult err = pfn_##cmd;						\
    if( err != CUDA_SUCCESS ) {						\
      const char *errStr;						\
      (void) pfn_cuGetErrorString(err, &errStr);			\
      INFO(NCCL_ALL,"%s:%d Cuda failure %d '%s'", __FILE__, __LINE__, err, errStr); \
    }									\
} while(false)

#define CUCHECKTHREAD(cmd, args) do {					\
    CUresult err = pfn_##cmd;						\
    if (err != CUDA_SUCCESS) {						\
      INFO(NCCL_INIT,"%s:%d -> %d [Async thread]", __FILE__, __LINE__, err); \
      args->ret = ncclUnhandledCudaError;				\
      return args;							\
    }									\
} while(0)

#define DECLARE_CUDA_PFN_EXTERN(symbol) extern PFN_##symbol pfn_##symbol

#if CUDART_VERSION >= 11030
/* CUDA Driver functions loaded with cuGetProcAddress for versioning */
DECLARE_CUDA_PFN_EXTERN(cuDeviceGet);
DECLARE_CUDA_PFN_EXTERN(cuDeviceGetAttribute);
DECLARE_CUDA_PFN_EXTERN(cuGetErrorString);
DECLARE_CUDA_PFN_EXTERN(cuGetErrorName);
DECLARE_CUDA_PFN_EXTERN(cuMemGetAddressRange);
DECLARE_CUDA_PFN_EXTERN(cuCtxCreate);
DECLARE_CUDA_PFN_EXTERN(cuCtxDestroy);
DECLARE_CUDA_PFN_EXTERN(cuCtxGetCurrent);
DECLARE_CUDA_PFN_EXTERN(cuCtxSetCurrent);
DECLARE_CUDA_PFN_EXTERN(cuCtxGetDevice);
DECLARE_CUDA_PFN_EXTERN(cuPointerGetAttribute);
// cuMem API support
DECLARE_CUDA_PFN_EXTERN(cuMemAddressReserve);
DECLARE_CUDA_PFN_EXTERN(cuMemAddressFree);
DECLARE_CUDA_PFN_EXTERN(cuMemCreate);
DECLARE_CUDA_PFN_EXTERN(cuMemGetAllocationGranularity);
DECLARE_CUDA_PFN_EXTERN(cuMemExportToShareableHandle);
DECLARE_CUDA_PFN_EXTERN(cuMemImportFromShareableHandle);
DECLARE_CUDA_PFN_EXTERN(cuMemMap);
DECLARE_CUDA_PFN_EXTERN(cuMemRelease);
DECLARE_CUDA_PFN_EXTERN(cuMemRetainAllocationHandle);
DECLARE_CUDA_PFN_EXTERN(cuMemSetAccess);
DECLARE_CUDA_PFN_EXTERN(cuMemUnmap);
#if CUDA_VERSION >= 11070
DECLARE_CUDA_PFN_EXTERN(cuMemGetHandleForAddressRange); // DMA-BUF support
#endif
#if CUDA_VERSION >= 12010
/* NVSwitch Multicast support */
DECLARE_CUDA_PFN_EXTERN(cuMulticastAddDevice);
DECLARE_CUDA_PFN_EXTERN(cuMulticastBindMem);
DECLARE_CUDA_PFN_EXTERN(cuMulticastBindAddr);
DECLARE_CUDA_PFN_EXTERN(cuMulticastCreate);
DECLARE_CUDA_PFN_EXTERN(cuMulticastGetGranularity);
DECLARE_CUDA_PFN_EXTERN(cuMulticastUnbind);
#endif
#endif

ncclResult_t ncclCudaLibraryInit(void);

extern int ncclCudaDriverVersionCache;
extern bool ncclCudaLaunchBlocking; // initialized by ncclCudaLibraryInit()

inline ncclResult_t ncclCudaDriverVersion(int* driver) {
  int version = __atomic_load_n(&ncclCudaDriverVersionCache, __ATOMIC_RELAXED);
  if (version == -1) {
    CUDACHECK(cudaDriverGetVersion(&version));
    __atomic_store_n(&ncclCudaDriverVersionCache, version, __ATOMIC_RELAXED);
  }
  *driver = version;
  return ncclSuccess;
}
#endif
