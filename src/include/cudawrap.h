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
#else
typedef CUresult (CUDAAPI *PFN_cuInit_v2000)(unsigned int Flags);
typedef CUresult (CUDAAPI *PFN_cuDriverGetVersion_v2020)(int *driverVersion);
typedef CUresult (CUDAAPI *PFN_cuGetProcAddress_v11030)(const char *symbol, void **pfn, int driverVersion, cuuint64_t flags);
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

#define DECLARE_CUDA_PFN_EXTERN(symbol,version) extern PFN_##symbol##_v##version pfn_##symbol

#if CUDART_VERSION >= 11030
/* CUDA Driver functions loaded with cuGetProcAddress for versioning */
DECLARE_CUDA_PFN_EXTERN(cuDeviceGet, 2000);
DECLARE_CUDA_PFN_EXTERN(cuDeviceGetAttribute, 2000);
DECLARE_CUDA_PFN_EXTERN(cuGetErrorString, 6000);
DECLARE_CUDA_PFN_EXTERN(cuGetErrorName, 6000);
DECLARE_CUDA_PFN_EXTERN(cuMemGetAddressRange, 3020);
DECLARE_CUDA_PFN_EXTERN(cuCtxCreate, 3020);
DECLARE_CUDA_PFN_EXTERN(cuCtxDestroy, 4000);
DECLARE_CUDA_PFN_EXTERN(cuCtxGetCurrent, 4000);
DECLARE_CUDA_PFN_EXTERN(cuCtxSetCurrent, 4000);
DECLARE_CUDA_PFN_EXTERN(cuCtxGetDevice, 2000);
DECLARE_CUDA_PFN_EXTERN(cuPointerGetAttribute, 4000);
// cuMem API support
DECLARE_CUDA_PFN_EXTERN(cuMemAddressReserve, 10020);
DECLARE_CUDA_PFN_EXTERN(cuMemAddressFree, 10020);
DECLARE_CUDA_PFN_EXTERN(cuMemCreate, 10020);
DECLARE_CUDA_PFN_EXTERN(cuMemGetAllocationGranularity, 10020);
DECLARE_CUDA_PFN_EXTERN(cuMemExportToShareableHandle, 10020);
DECLARE_CUDA_PFN_EXTERN(cuMemImportFromShareableHandle, 10020);
DECLARE_CUDA_PFN_EXTERN(cuMemMap, 10020);
DECLARE_CUDA_PFN_EXTERN(cuMemRelease, 10020);
DECLARE_CUDA_PFN_EXTERN(cuMemRetainAllocationHandle, 11000);
DECLARE_CUDA_PFN_EXTERN(cuMemSetAccess, 10020);
DECLARE_CUDA_PFN_EXTERN(cuMemUnmap, 10020);
#if CUDA_VERSION >= 11070
DECLARE_CUDA_PFN_EXTERN(cuMemGetHandleForAddressRange, 11070); // DMA-BUF support
#endif
#if CUDA_VERSION >= 12010
/* NVSwitch Multicast support */
DECLARE_CUDA_PFN_EXTERN(cuMulticastAddDevice, 12010);
DECLARE_CUDA_PFN_EXTERN(cuMulticastBindMem, 12010);
DECLARE_CUDA_PFN_EXTERN(cuMulticastBindAddr, 12010);
DECLARE_CUDA_PFN_EXTERN(cuMulticastCreate, 12010);
DECLARE_CUDA_PFN_EXTERN(cuMulticastGetGranularity, 12010);
DECLARE_CUDA_PFN_EXTERN(cuMulticastUnbind, 12010);
#endif
#endif

/* CUDA Driver functions loaded with dlsym() */
DECLARE_CUDA_PFN_EXTERN(cuInit, 2000);
DECLARE_CUDA_PFN_EXTERN(cuDriverGetVersion, 2020);
DECLARE_CUDA_PFN_EXTERN(cuGetProcAddress, 11030);


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
