/*************************************************************************
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "alloc.h"
#include "nccl.h"
#include "debug.h"
#include "param.h"
#include "cudawrap.h"

// This env var (NCCL_CUMEM_ENABLE) toggles cuMem API usage
NCCL_PARAM(CuMemEnable, "CUMEM_ENABLE", -2);
NCCL_PARAM(CuMemHostEnable, "CUMEM_HOST_ENABLE", -1);
// Handle type used for cuMemCreate()
CUmemAllocationHandleType ncclCuMemHandleType = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;

static int ncclCuMemSupported = 0;

// Determine whether CUMEM & VMM RDMA is supported on this platform
int ncclIsCuMemSupported() {
#if CUDART_VERSION < 11030
  return 0;
#else
  CUdevice currentDev;
  int cudaDev;
  int cudaDriverVersion;
  int flag = 0;
  ncclResult_t ret = ncclSuccess;
  CUDACHECKGOTO(cudaDriverGetVersion(&cudaDriverVersion), ret, error);
  if (cudaDriverVersion < 12000) return 0;  // Need CUDA_VISIBLE_DEVICES support
  CUDACHECKGOTO(cudaGetDevice(&cudaDev), ret, error);
  if (CUPFN(cuMemCreate) == NULL) return 0;
  CUCHECKGOTO(cuDeviceGet(&currentDev, cudaDev), ret, error);
  // Query device to see if CUMEM VMM support is available
  CUCHECKGOTO(cuDeviceGetAttribute(&flag, CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, currentDev), ret, error);
  if (!flag) return 0;
error:
  return (ret == ncclSuccess);
#endif
}

int ncclCuMemEnable() {
  // NCCL_CUMEM_ENABLE=-2 means auto-detect CUMEM support
  int param = ncclParamCuMemEnable();
  return  param >= 0 ? param : (param == -2 && ncclCuMemSupported);
}

static int ncclCumemHostEnable = -1;
int ncclCuMemHostEnable() {
  if (ncclCumemHostEnable != -1)
    return ncclCumemHostEnable;
#if CUDART_VERSION < 12020
  ncclCumemHostEnable = 0;
  return ncclCumemHostEnable;
#else
  ncclResult_t ret = ncclSuccess;
  int cudaDriverVersion;
  int paramValue = -1;
  CUDACHECKGOTO(cudaDriverGetVersion(&cudaDriverVersion), ret, error);
  if (cudaDriverVersion < 12020) {
    ncclCumemHostEnable = 0;
  }
  else {
    paramValue = ncclParamCuMemHostEnable();
    if (paramValue != -1)
      ncclCumemHostEnable = paramValue;
    else
      ncclCumemHostEnable = (cudaDriverVersion >= 12060) ? 1 : 0;
    if (ncclCumemHostEnable) {
      // Verify that host allocations actually work.  Docker in particular is known to disable "get_mempolicy",
      // causing such allocations to fail (this can be fixed by invoking Docker with "--cap-add SYS_NICE").
      int cudaDev;
      CUdevice currentDev;
      int cpuNumaNodeId = -1;
      CUmemAllocationProp prop = {};
      size_t granularity = 0;
      size_t size;
      CUmemGenericAllocationHandle handle;
      CUDACHECK(cudaGetDevice(&cudaDev));
      CUCHECK(cuDeviceGet(&currentDev, cudaDev));
      CUCHECK(cuDeviceGetAttribute(&cpuNumaNodeId, CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID, currentDev));
      if (cpuNumaNodeId < 0) cpuNumaNodeId = 0;
      prop.location.type = CU_MEM_LOCATION_TYPE_HOST_NUMA;
      prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
      prop.requestedHandleTypes = ncclCuMemHandleType;
      prop.location.id = cpuNumaNodeId;
      CUCHECK(cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
      size = 1;
      ALIGN_SIZE(size, granularity);
      if (CUPFN(cuMemCreate(&handle, size, &prop, 0)) != CUDA_SUCCESS) {
        INFO(NCCL_INIT, "cuMem host allocations do not appear to be working; falling back to a /dev/shm/ based "
             "implementation. This could be due to the container runtime disabling NUMA support. "
             "To disable this warning, set NCCL_CUMEM_HOST_ENABLE=0");
        ncclCumemHostEnable = 0;
      } else {
        CUCHECK(cuMemRelease(handle));
      }
    }
  }
  return ncclCumemHostEnable;
error:
  return (ret == ncclSuccess);
#endif
}

#define DECLARE_CUDA_PFN(symbol) PFN_##symbol pfn_##symbol = nullptr

#if CUDART_VERSION >= 11030
/* CUDA Driver functions loaded with cuGetProcAddress for versioning */
DECLARE_CUDA_PFN(cuDeviceGet);
DECLARE_CUDA_PFN(cuDeviceGetAttribute);
DECLARE_CUDA_PFN(cuGetErrorString);
DECLARE_CUDA_PFN(cuGetErrorName);
/* enqueue.cc */
DECLARE_CUDA_PFN(cuMemGetAddressRange);
DECLARE_CUDA_PFN(cuLaunchKernel);
#if CUDA_VERSION >= 11080
DECLARE_CUDA_PFN(cuLaunchKernelEx);
#endif
/* proxy.cc */
DECLARE_CUDA_PFN(cuCtxCreate);
DECLARE_CUDA_PFN(cuCtxDestroy);
DECLARE_CUDA_PFN(cuCtxGetCurrent);
DECLARE_CUDA_PFN(cuCtxSetCurrent);
DECLARE_CUDA_PFN(cuCtxGetDevice);
/* cuMem API support */
DECLARE_CUDA_PFN(cuMemAddressReserve);
DECLARE_CUDA_PFN(cuMemAddressFree);
DECLARE_CUDA_PFN(cuMemCreate);
DECLARE_CUDA_PFN(cuMemGetAllocationGranularity);
DECLARE_CUDA_PFN(cuMemExportToShareableHandle);
DECLARE_CUDA_PFN(cuMemImportFromShareableHandle);
DECLARE_CUDA_PFN(cuMemMap);
DECLARE_CUDA_PFN(cuMemRelease);
DECLARE_CUDA_PFN(cuMemRetainAllocationHandle);
DECLARE_CUDA_PFN(cuMemSetAccess);
DECLARE_CUDA_PFN(cuMemUnmap);
DECLARE_CUDA_PFN(cuMemGetAllocationPropertiesFromHandle);
/* ncclMemAlloc/Free */
DECLARE_CUDA_PFN(cuPointerGetAttribute);
#if CUDA_VERSION >= 11070
/* transport/collNet.cc/net.cc*/
DECLARE_CUDA_PFN(cuMemGetHandleForAddressRange); // DMA-BUF support
#endif
#if CUDA_VERSION >= 12010
/* NVSwitch Multicast support */
DECLARE_CUDA_PFN(cuMulticastAddDevice);
DECLARE_CUDA_PFN(cuMulticastBindMem);
DECLARE_CUDA_PFN(cuMulticastBindAddr);
DECLARE_CUDA_PFN(cuMulticastCreate);
DECLARE_CUDA_PFN(cuMulticastGetGranularity);
DECLARE_CUDA_PFN(cuMulticastUnbind);
#endif
#endif

#define CUDA_DRIVER_MIN_VERSION 11030

int ncclCudaDriverVersionCache = -1;
bool ncclCudaLaunchBlocking = false;

#if CUDART_VERSION >= 11030

#if CUDART_VERSION >= 12000
#define LOAD_SYM(symbol, ignore) do {                                   \
    cudaDriverEntryPointQueryResult driverStatus = cudaDriverEntryPointSymbolNotFound; \
    res = cudaGetDriverEntryPoint(#symbol, (void **) (&pfn_##symbol), cudaEnableDefault, &driverStatus); \
    if (res != cudaSuccess || driverStatus != cudaDriverEntryPointSuccess) { \
      if (!ignore) {                                                    \
        WARN("Retrieve %s failed with %d status %d", #symbol, res, driverStatus); \
        return ncclSystemError; }                                       \
    } } while(0)
#else
#define LOAD_SYM(symbol, ignore) do {                                   \
    res = cudaGetDriverEntryPoint(#symbol, (void **) (&pfn_##symbol), cudaEnableDefault); \
    if (res != cudaSuccess) { \
      if (!ignore) {                                                    \
        WARN("Retrieve %s failed with %d", #symbol, res);               \
        return ncclSystemError; }                                       \
    } } while(0)
#endif

/*
  Load the CUDA symbols
 */
static ncclResult_t cudaPfnFuncLoader(void) {

  cudaError_t res;

  LOAD_SYM(cuGetErrorString, 0);
  LOAD_SYM(cuGetErrorName, 0);
  LOAD_SYM(cuDeviceGet, 0);
  LOAD_SYM(cuDeviceGetAttribute, 0);
  LOAD_SYM(cuMemGetAddressRange, 1);
  LOAD_SYM(cuCtxCreate, 1);
  LOAD_SYM(cuCtxDestroy, 1);
  LOAD_SYM(cuCtxGetCurrent, 1);
  LOAD_SYM(cuCtxSetCurrent, 1);
  LOAD_SYM(cuCtxGetDevice, 1);
  LOAD_SYM(cuLaunchKernel, 1);
#if CUDA_VERSION >= 11080
  LOAD_SYM(cuLaunchKernelEx, 1);
#endif
/* cuMem API support */
  LOAD_SYM(cuMemAddressReserve, 1);
  LOAD_SYM(cuMemAddressFree, 1);
  LOAD_SYM(cuMemCreate, 1);
  LOAD_SYM(cuMemGetAllocationGranularity, 1);
  LOAD_SYM(cuMemExportToShareableHandle, 1);
  LOAD_SYM(cuMemImportFromShareableHandle, 1);
  LOAD_SYM(cuMemMap, 1);
  LOAD_SYM(cuMemRelease, 1);
  LOAD_SYM(cuMemRetainAllocationHandle, 1);
  LOAD_SYM(cuMemSetAccess, 1);
  LOAD_SYM(cuMemUnmap, 1);
  LOAD_SYM(cuMemGetAllocationPropertiesFromHandle, 1);
/* ncclMemAlloc/Free */
  LOAD_SYM(cuPointerGetAttribute, 1);
#if CUDA_VERSION >= 11070
  LOAD_SYM(cuMemGetHandleForAddressRange, 1); // DMA-BUF support
#endif
#if CUDA_VERSION >= 12010
/* NVSwitch Multicast support */
  LOAD_SYM(cuMulticastAddDevice, 1);
  LOAD_SYM(cuMulticastBindMem, 1);
  LOAD_SYM(cuMulticastBindAddr, 1);
  LOAD_SYM(cuMulticastCreate, 1);
  LOAD_SYM(cuMulticastGetGranularity, 1);
  LOAD_SYM(cuMulticastUnbind, 1);
#endif
  return ncclSuccess;
}
#endif

static pthread_once_t initOnceControl = PTHREAD_ONCE_INIT;
static ncclResult_t initResult;

static void initOnceFunc() {
  do {
    const char* val = ncclGetEnv("CUDA_LAUNCH_BLOCKING");
    ncclCudaLaunchBlocking = val!=nullptr && val[0]!=0 && !(val[0]=='0' && val[1]==0);
  } while (0);

  ncclResult_t ret = ncclSuccess;
  int cudaDev;
  int driverVersion;
  CUDACHECKGOTO(cudaGetDevice(&cudaDev), ret, error); // Initialize the driver

  CUDACHECKGOTO(cudaDriverGetVersion(&driverVersion), ret, error);
  INFO(NCCL_INIT, "cudaDriverVersion %d", driverVersion);

  if (driverVersion < CUDA_DRIVER_MIN_VERSION) {
    // WARN("CUDA Driver version found is %d. Minimum requirement is %d", driverVersion, CUDA_DRIVER_MIN_VERSION);
    // Silently ignore version check mismatch for backwards compatibility
    goto error;
  }

  #if CUDART_VERSION >= 11030
  if (cudaPfnFuncLoader()) {
    WARN("CUDA some PFN functions not found in the library");
    goto error;
  }
  #endif

  // Determine whether we support the cuMem APIs or not
  ncclCuMemSupported = ncclIsCuMemSupported();

  /* To use cuMem* for host memory allocation, we need to create context on each visible device.
   * This is a workaround needed in CUDA 12.2 and CUDA 12.3 which is fixed in 12.4. */
  if (ncclCuMemSupported && ncclCuMemHostEnable() && 12020 <= driverVersion && driverVersion <= 12030) {
    int deviceCnt, saveDevice;
    cudaGetDevice(&saveDevice);
    cudaGetDeviceCount(&deviceCnt);
    for (int i = 0; i < deviceCnt; ++i) {
      cudaSetDevice(i);
      cudaFree(NULL);
    }
    cudaSetDevice(saveDevice);
  }
  initResult = ret;
  return;
error:
  initResult = ncclSystemError;
  return;
}

ncclResult_t ncclCudaLibraryInit() {
  pthread_once(&initOnceControl, initOnceFunc);
  return initResult;
}
