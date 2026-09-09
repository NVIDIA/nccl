/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_CUDAWRAP_H_
#define NCCL_CUDAWRAP_H_

#include <cuda.h>
#include <cuda_runtime.h>
#include "checks.h"
#include "compiler.h"

// Is cuMem API usage enabled
extern int ncclCuMemEnable();
extern int ncclCuMemHostEnable();

#if CUDART_VERSION >= 11030
#include <cudaTypedefs.h>

// Handle type used for cuMemCreate()
extern CUmemAllocationHandleType ncclCuMemHandleType;

#endif

#define CUPFN(symbol) pfn_##symbol

// Emit an actionable hint for a subset of well-known CUDA driver errors.
static inline void printCudaDriverErrorHint(CUresult err) {
  switch (err) {
  case CUDA_ERROR_TOO_MANY_PEERS:
    INFO(NCCL_ALL, "HINT: In many cases this error indicates that the GPU peer-mapping (BAR1 P2P) resources are "
                   "exhausted, which can happen on nodes with more than 8 GPUs.");
    INFO(NCCL_ALL, "HINT: To confirm, set NCCL_CUMEM_ENABLE=0 to fall back to the legacy allocator.");
    return;
  case CUDA_ERROR_NOT_READY:
    INFO(NCCL_ALL, "HINT: In many cases this error indicates that the IMEX (NVLink fabric) service is not running or "
                   "is misconfigured on MNNVL systems.");
    INFO(NCCL_ALL, "HINT: To confirm, run 'nvidia-imex-ctl -N' (which may require elevated privileges) "
                   "to check the IMEX channel and domain status.");
    return;
  default:
    break;
  }
}

// Check CUDA PFN driver calls
#define CUCHECK(cmd) \
  do { \
    CUresult err = pfn_##cmd; \
    if (err != CUDA_SUCCESS) { \
      const char* errStr; \
      (void)pfn_cuGetErrorString(err, &errStr); \
      WARN("Cuda failure %d '%s'", err, errStr); \
      printCudaDriverErrorHint(err); \
      return ncclUnhandledCudaError; \
    } \
  } while (false)

#define CUCALL(cmd) \
  do { \
    pfn_##cmd; \
  } while (false)

#define CUCHECKGOTO(cmd, res, label) \
  do { \
    CUresult err = pfn_##cmd; \
    if (err != CUDA_SUCCESS) { \
      const char* errStr; \
      (void)pfn_cuGetErrorString(err, &errStr); \
      WARN("Cuda failure %d '%s'", err, errStr); \
      printCudaDriverErrorHint(err); \
      res = ncclUnhandledCudaError; \
      goto label; \
    } \
  } while (false)

// Report failure but clear error and continue
#define CUCHECKIGNORE(cmd) \
  do { \
    CUresult err = pfn_##cmd; \
    if (err != CUDA_SUCCESS) { \
      const char* errStr; \
      (void)pfn_cuGetErrorString(err, &errStr); \
      INFO_LOC(NCCL_ALL, "Cuda failure %d '%s'", err, errStr); \
    } \
  } while (false)

#define CUCHECKTHREAD(cmd, args) \
  do { \
    CUresult err = pfn_##cmd; \
    if (err != CUDA_SUCCESS) { \
      INFO_LOC(NCCL_INIT, "-> %d [Async thread]", (int)(err)); \
      args->ret = ncclUnhandledCudaError; \
      return args; \
    } \
  } while (0)

#define DECLARE_CUDA_PFN_EXTERN(symbol, version) extern PFN_##symbol##_v##version pfn_##symbol

#if CUDART_VERSION >= 11030
/* CUDA Driver functions loaded with cuGetProcAddress for versioning */
DECLARE_CUDA_PFN_EXTERN(cuInit, 2000);
DECLARE_CUDA_PFN_EXTERN(cuDeviceGet, 2000);
DECLARE_CUDA_PFN_EXTERN(cuDeviceGetCount, 2000);
DECLARE_CUDA_PFN_EXTERN(cuDeviceGetAttribute, 2000);
DECLARE_CUDA_PFN_EXTERN(cuDeviceGetUuid, 9020);
DECLARE_CUDA_PFN_EXTERN(cuGetErrorString, 6000);
DECLARE_CUDA_PFN_EXTERN(cuGetErrorName, 6000);
DECLARE_CUDA_PFN_EXTERN(cuMemGetAddressRange, 3020);
DECLARE_CUDA_PFN_EXTERN(cuCtxCreate, 11040);
DECLARE_CUDA_PFN_EXTERN(cuCtxDestroy, 4000);
DECLARE_CUDA_PFN_EXTERN(cuCtxGetCurrent, 4000);
DECLARE_CUDA_PFN_EXTERN(cuCtxSetCurrent, 4000);
DECLARE_CUDA_PFN_EXTERN(cuCtxGetDevice, 2000);
DECLARE_CUDA_PFN_EXTERN(cuDevicePrimaryCtxRetain, 7000);
DECLARE_CUDA_PFN_EXTERN(cuDevicePrimaryCtxRelease, 11000);
DECLARE_CUDA_PFN_EXTERN(cuPointerGetAttribute, 4000);
DECLARE_CUDA_PFN_EXTERN(cuLaunchKernel, 4000);
#if CUDART_VERSION >= 11080
DECLARE_CUDA_PFN_EXTERN(cuLaunchKernelEx, 11060);
#endif
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
DECLARE_CUDA_PFN_EXTERN(cuMemGetAllocationPropertiesFromHandle, 10020);
#if CUDA_VERSION >= 11070
DECLARE_CUDA_PFN_EXTERN(cuMemGetHandleForAddressRange, 11070); // DMA-BUF support
#endif

#if CUDA_VERSION >= 13030
#define DECLARE_CUDA_TYPE_EXTERN_V13030(...)  /* provided by cudaTypedefs.h */
#define DECLARE_CUDA_PFN_EXTERN_V13030(sym, ...) DECLARE_CUDA_PFN_EXTERN(sym, 13030)
#else
#define DECLARE_CUDA_TYPE_EXTERN_V13030(...) __VA_ARGS__
#define DECLARE_CUDA_PFN_EXTERN_V13030(sym, ...) \
  static inline CUresult pfn_##sym(__VA_ARGS__) { \
    return CUDA_ERROR_NOT_SUPPORTED; \
  }
#endif

// CFT support requires cuda 13.3
// clang-format off: maintain hand-formatted code
DECLARE_CUDA_TYPE_EXTERN_V13030(
  typedef struct {
    int type;
    int flags;
    int ipcHandleTypes;
    size_t size;
    struct { int device; } unicast;
    struct { unsigned int numDevices; } multicast;
  } CUlogicalEndpointProp;
  typedef struct {} CUlogicalEndpointFabricHandle;
  enum {
    CU_LOGICAL_ENDPOINT_TYPE_UNICAST          = 0,
    CU_LOGICAL_ENDPOINT_TYPE_MULTICAST        = 0,
    CU_LOGICAL_ENDPOINT_FLAG_NONE             = 0,
    CU_LOGICAL_ENDPOINT_IPC_HANDLE_TYPE_FABRIC = 0,
  };
)
DECLARE_CUDA_PFN_EXTERN_V13030(cuLogicalEndpointIdReserve, uint32_t* id, uint32_t count);
DECLARE_CUDA_PFN_EXTERN_V13030(cuLogicalEndpointIdRelease, uint32_t id, uint32_t count);
DECLARE_CUDA_PFN_EXTERN_V13030(cuLogicalEndpointCreate,    uint32_t id, CUlogicalEndpointProp* prop);
DECLARE_CUDA_PFN_EXTERN_V13030(cuLogicalEndpointDestroy,   uint32_t id);
DECLARE_CUDA_PFN_EXTERN_V13030(cuLogicalEndpointAddDevice, uint32_t id, CUdevice dev);
DECLARE_CUDA_PFN_EXTERN_V13030(cuLogicalEndpointQuery,     uint32_t id, uint32_t count, int* status);
DECLARE_CUDA_PFN_EXTERN_V13030(cuLogicalEndpointGetLimits, cuuint64_t* bindAlignment, cuuint64_t* maxSize, CUlogicalEndpointProp* prop);
DECLARE_CUDA_PFN_EXTERN_V13030(cuLogicalEndpointExport,    CUlogicalEndpointFabricHandle* handle, uint32_t id, int ipcHandleType);
DECLARE_CUDA_PFN_EXTERN_V13030(cuLogicalEndpointImport,    uint32_t id, CUlogicalEndpointFabricHandle* handle, int ipcHandleType);
DECLARE_CUDA_PFN_EXTERN_V13030(cuLogicalEndpointBindAddr,  uint32_t id, CUdevice dev, size_t offset, void* addr, size_t size, int flags);
DECLARE_CUDA_PFN_EXTERN_V13030(cuLogicalEndpointUnbind,    uint32_t id, CUdevice dev, size_t offset, size_t size);
// clang-format on

#if CUDA_VERSION >= 12010
/* NVSwitch Multicast support */
DECLARE_CUDA_PFN_EXTERN(cuMulticastAddDevice, 12010);
DECLARE_CUDA_PFN_EXTERN(cuMulticastBindMem, 12010);
DECLARE_CUDA_PFN_EXTERN(cuMulticastBindAddr, 12010);
DECLARE_CUDA_PFN_EXTERN(cuMulticastCreate, 12010);
DECLARE_CUDA_PFN_EXTERN(cuMulticastGetGranularity, 12010);
DECLARE_CUDA_PFN_EXTERN(cuMulticastUnbind, 12010);
#endif
/* Stream-MemOp support */
DECLARE_CUDA_PFN_EXTERN(cuStreamBatchMemOp, 11070);
DECLARE_CUDA_PFN_EXTERN(cuStreamWaitValue32, 11070);
DECLARE_CUDA_PFN_EXTERN(cuStreamWaitValue64, 11070);
DECLARE_CUDA_PFN_EXTERN(cuStreamWriteValue32, 11070);
DECLARE_CUDA_PFN_EXTERN(cuStreamWriteValue64, 11070);
#endif

ncclResult_t ncclCudaLibraryInit(void);

extern int ncclCudaDriverVersionCache;
extern bool ncclCudaLaunchBlocking; // initialized by ncclCudaLibraryInit()

// Checks whether the given stream is the legacy null stream.
inline ncclResult_t ncclCudaStreamIsLegacyNull(cudaStream_t stream, bool* isLegacy) {
#if CUDART_VERSION >= 12000
  unsigned long long nullStreamId, legacyNullStreamId;
  CUDACHECK(cudaStreamGetId(NULL, &nullStreamId));
  CUDACHECK(cudaStreamGetId(cudaStreamLegacy, &legacyNullStreamId));
  *isLegacy = (stream == cudaStreamLegacy) || ((stream == NULL) && (nullStreamId == legacyNullStreamId));
#else
  *isLegacy = (stream == NULL) || (stream == cudaStreamLegacy);
#endif
  return ncclSuccess;
}

inline ncclResult_t ncclCudaDriverVersion(int* driver) {
  int version = COMPILER_ATOMIC_LOAD(&ncclCudaDriverVersionCache, std::memory_order_relaxed);
  if (version == -1) {
    CUDACHECK(cudaDriverGetVersion(&version));
    COMPILER_ATOMIC_STORE(&ncclCudaDriverVersionCache, version, std::memory_order_relaxed);
  }
  *driver = version;
  return ncclSuccess;
}

ncclResult_t ncclCuStreamBatchMemOp(cudaStream_t stream, unsigned int numOps, CUstreamBatchMemOpParams* batchParams);

#endif
