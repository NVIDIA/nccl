/*************************************************************************
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "nccl.h"
#include "debug.h"
#include "cudawrap.h"

#include <dlfcn.h>

#define DECLARE_CUDA_PFN(symbol,version) PFN_##symbol##_v##version pfn_##symbol = nullptr

#if CUDART_VERSION >= 11030
/* CUDA Driver functions loaded with cuGetProcAddress for versioning */
DECLARE_CUDA_PFN(cuDeviceGet, 2000);
DECLARE_CUDA_PFN(cuDeviceGetAttribute, 2000);
DECLARE_CUDA_PFN(cuGetErrorString, 6000);
DECLARE_CUDA_PFN(cuGetErrorName, 6000);
/* enqueue.cc */
DECLARE_CUDA_PFN(cuMemGetAddressRange, 3020);
/* proxy.cc */
DECLARE_CUDA_PFN(cuCtxCreate, 3020);
DECLARE_CUDA_PFN(cuCtxDestroy, 4000);
DECLARE_CUDA_PFN(cuCtxGetCurrent, 4000);
DECLARE_CUDA_PFN(cuCtxSetCurrent, 4000);
DECLARE_CUDA_PFN(cuCtxGetDevice, 2000);
/* cuMem API support */
DECLARE_CUDA_PFN(cuMemAddressReserve, 10020);
DECLARE_CUDA_PFN(cuMemAddressFree, 10020);
DECLARE_CUDA_PFN(cuMemCreate, 10020);
DECLARE_CUDA_PFN(cuMemGetAllocationGranularity, 10020);
DECLARE_CUDA_PFN(cuMemExportToShareableHandle, 10020);
DECLARE_CUDA_PFN(cuMemImportFromShareableHandle, 10020);
DECLARE_CUDA_PFN(cuMemMap, 10020);
DECLARE_CUDA_PFN(cuMemRelease, 10020);
DECLARE_CUDA_PFN(cuMemSetAccess, 10020);
DECLARE_CUDA_PFN(cuMemUnmap, 10020);
#if CUDA_VERSION >= 11070
/* transport/collNet.cc/net.cc*/
DECLARE_CUDA_PFN(cuMemGetHandleForAddressRange, 11070); // DMA-BUF support
#endif
#if CUDA_VERSION >= 12010
/* NVSwitch Multicast support */
DECLARE_CUDA_PFN(cuMulticastAddDevice, 12010);
DECLARE_CUDA_PFN(cuMulticastBindMem, 12010);
DECLARE_CUDA_PFN(cuMulticastBindAddr, 12010);
DECLARE_CUDA_PFN(cuMulticastCreate, 12010);
DECLARE_CUDA_PFN(cuMulticastGetGranularity, 12010);
DECLARE_CUDA_PFN(cuMulticastUnbind, 12010);
#endif
#endif

/* CUDA Driver functions loaded with dlsym() */
DECLARE_CUDA_PFN(cuInit, 2000);
DECLARE_CUDA_PFN(cuDriverGetVersion, 2020);
DECLARE_CUDA_PFN(cuGetProcAddress, 11030);

#define CUDA_DRIVER_MIN_VERSION 11030

static void *cudaLib;
int ncclCudaDriverVersionCache = -1;
bool ncclCudaLaunchBlocking = false;

#if CUDART_VERSION >= 11030
/*
  Load the CUDA symbols
 */
static ncclResult_t cudaPfnFuncLoader(void) {
  CUresult res;

#define LOAD_SYM(symbol, version, ignore) do {                           \
    res = pfn_cuGetProcAddress(#symbol, (void **) (&pfn_##symbol), version, 0); \
    if (res != 0) {                                                     \
      if (!ignore) {                                                    \
        WARN("Retrieve %s version %d failed with %d", #symbol, version, res); \
        return ncclSystemError; }                                       \
    } } while(0)

  LOAD_SYM(cuGetErrorString, 6000, 0);
  LOAD_SYM(cuGetErrorName, 6000, 0);
  LOAD_SYM(cuDeviceGet, 2000, 0);
  LOAD_SYM(cuDeviceGetAttribute, 2000, 0);
  LOAD_SYM(cuMemGetAddressRange, 3020, 1);
  LOAD_SYM(cuCtxCreate, 3020, 1);
  LOAD_SYM(cuCtxDestroy, 4000, 1);
  LOAD_SYM(cuCtxGetCurrent, 4000, 1);
  LOAD_SYM(cuCtxSetCurrent, 4000, 1);
  LOAD_SYM(cuCtxGetDevice, 2000, 1);
/* cuMem API support */
#if CUDA_VERSION >= 11030
  LOAD_SYM(cuMemAddressReserve, 10020, 1);
  LOAD_SYM(cuMemAddressFree, 10020, 1);
  LOAD_SYM(cuMemCreate, 10020, 1);
  LOAD_SYM(cuMemGetAllocationGranularity, 10020, 1);
  LOAD_SYM(cuMemExportToShareableHandle, 10020, 1);
  LOAD_SYM(cuMemImportFromShareableHandle, 10020, 1);
  LOAD_SYM(cuMemMap, 10020, 1);
  LOAD_SYM(cuMemRelease, 10020, 1);
  LOAD_SYM(cuMemSetAccess, 10020, 1);
  LOAD_SYM(cuMemUnmap, 10020, 1);
#endif
#if CUDA_VERSION >= 11070
  LOAD_SYM(cuMemGetHandleForAddressRange, 11070, 1); // DMA-BUF support
#endif
#if CUDA_VERSION >= 12010
/* NVSwitch Multicast support */
  LOAD_SYM(cuMulticastAddDevice, 12010, 1);
  LOAD_SYM(cuMulticastBindMem, 12010, 1);
  LOAD_SYM(cuMulticastBindAddr, 12010, 1);
  LOAD_SYM(cuMulticastCreate, 12010, 1);
  LOAD_SYM(cuMulticastGetGranularity, 12010, 1);
  LOAD_SYM(cuMulticastUnbind, 12010, 1);
#endif
  return ncclSuccess;
}
#endif

static pthread_once_t initOnceControl = PTHREAD_ONCE_INIT;
static ncclResult_t initResult;

static void initOnceFunc() {
  do {
    char* val = getenv("CUDA_LAUNCH_BLOCKING");
    ncclCudaLaunchBlocking = val!=nullptr && val[0]!=0 && !(val[0]=='0' && val[1]==0);
  } while (0);

  CUresult res;
  /*
   * Load CUDA driver library
   */
  char path[1024];
  char *ncclCudaPath = getenv("NCCL_CUDA_PATH");
  if (ncclCudaPath == NULL)
    snprintf(path, 1024, "%s", "libcuda.so");
  else
    snprintf(path, 1024, "%s%s", ncclCudaPath, "libcuda.so");

  (void) dlerror(); // Clear any previous errors
  cudaLib = dlopen(path, RTLD_LAZY);
  if (cudaLib == NULL) {
    WARN("Failed to find CUDA library %s (NCCL_CUDA_PATH='%s') : %s", path, ncclCudaPath ? ncclCudaPath : "", dlerror());
    goto error;
  }

  /*
   * Load initial CUDA functions
   */

  pfn_cuInit = (PFN_cuInit_v2000) dlsym(cudaLib, "cuInit");
  if (pfn_cuInit == NULL) {
    WARN("Failed to load CUDA missing symbol cuInit");
    goto error;
  }

  pfn_cuDriverGetVersion = (PFN_cuDriverGetVersion_v2020) dlsym(cudaLib, "cuDriverGetVersion");
  if (pfn_cuDriverGetVersion == NULL) {
    WARN("Failed to load CUDA missing symbol cuDriverGetVersion");
    goto error;
  }

  int driverVersion;
  res = pfn_cuDriverGetVersion(&driverVersion);
  if (res != 0) {
    WARN("cuDriverGetVersion failed with %d", res);
    goto error;
  }

  INFO(NCCL_INIT, "cudaDriverVersion %d", driverVersion);

  if (driverVersion < CUDA_DRIVER_MIN_VERSION) {
    // WARN("CUDA Driver version found is %d. Minimum requirement is %d", driverVersion, CUDA_DRIVER_MIN_VERSION);
    // Silently ignore version check mismatch for backwards compatibility
    goto error;
  }

  pfn_cuGetProcAddress = (PFN_cuGetProcAddress_v11030) dlsym(cudaLib, "cuGetProcAddress");
  if (pfn_cuGetProcAddress == NULL) {
    WARN("Failed to load CUDA missing symbol cuGetProcAddress");
    goto error;
  }

  /*
   * Required to initialize the CUDA Driver.
   * Multiple calls of cuInit() will return immediately
   * without making any relevant change
   */
  pfn_cuInit(0);

  #if CUDART_VERSION >= 11030
  if (cudaPfnFuncLoader()) {
    WARN("CUDA some PFN functions not found in the library");
    goto error;
  }
  #endif

  initResult = ncclSuccess;
  return;
error:
  initResult = ncclSystemError;
  return;
}

ncclResult_t ncclCudaLibraryInit() {
  pthread_once(&initOnceControl, initOnceFunc);
  return initResult;
}
