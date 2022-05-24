/*************************************************************************
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_CUDAWRAP_H_
#define NCCL_CUDAWRAP_H_

#include <cuda.h>

#if CUDART_VERSION >= 11030
#include <cudaTypedefs.h>
#else
typedef CUresult (CUDAAPI *PFN_cuInit)(unsigned int Flags);
typedef CUresult (CUDAAPI *PFN_cuDriverGetVersion)(int *driverVersion);
typedef CUresult (CUDAAPI *PFN_cuGetProcAddress)(const char *symbol, void **pfn, int driverVersion, cuuint64_t flags);
#endif

#define CUPFN(symbol) pfn_##symbol

// Check CUDA PFN driver calls
#define CUCHECK(cmd) do {				      \
    CUresult err = pfn_##cmd;				      \
    if( err != CUDA_SUCCESS ) {				      \
      const char *errStr;				      \
      (void) pfn_cuGetErrorString(err, &errStr);	      \
      WARN("Cuda failure '%s'", errStr);		      \
      return ncclUnhandledCudaError;			      \
    }							      \
} while(false)

#define CUCHECKGOTO(cmd, res, label) do {		      \
    CUresult err = pfn_##cmd;				      \
    if( err != CUDA_SUCCESS ) {				      \
      const char *errStr;				      \
      (void) pfn_cuGetErrorString(err, &errStr);	      \
      WARN("Cuda failure '%s'", errStr);		      \
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
      INFO(NCCL_ALL,"%s:%d Cuda failure '%s'", __FILE__, __LINE__, errStr);	\
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
DECLARE_CUDA_PFN_EXTERN(cuCtxCreate_v3020);
DECLARE_CUDA_PFN_EXTERN(cuCtxDestroy);
DECLARE_CUDA_PFN_EXTERN(cuCtxSetCurrent);
#if CUDA_VERSION >= 11070
DECLARE_CUDA_PFN_EXTERN(cuMemGetHandleForAddressRange); // DMA-BUF support
#endif
#endif

/* CUDA Driver functions loaded with dlsym() */
DECLARE_CUDA_PFN_EXTERN(cuInit);
DECLARE_CUDA_PFN_EXTERN(cuDriverGetVersion);
DECLARE_CUDA_PFN_EXTERN(cuGetProcAddress);


ncclResult_t cudaLibraryInit(void);

#endif
