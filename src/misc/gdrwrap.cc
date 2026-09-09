/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "gdrwrap.h"
#include <mutex>

#include "cudawrap.h"
#include "core.h"
#include "os.h"
#include <errno.h>
#include <string.h>
#if defined(NCCL_OS_LINUX)
#include <sys/mman.h>
#include <unistd.h>
#endif

NCCL_PARAM(GdrCopyUseInternalDmaBuf, "GDRCOPY_USE_INTERNAL_DMABUF", 0);

enum ncclGdrUseInternalDmaBufMode {
  ncclGdrUseInternalDmaBufDisabled = 0,
  ncclGdrUseInternalDmaBufRequired = 1,
  ncclGdrUseInternalDmaBufOptional = 2
};

/* Function pointers assigned from dynamic library (os layer) */
static gdr_t (*gdr_internal_open)(void);
static int (*gdr_internal_close)(gdr_t g);
static int (*gdr_internal_pin_buffer)(gdr_t g, unsigned long addr, size_t size, uint64_t p2p_token, uint32_t va_space,
                                      gdr_mh_t* handle);
static int (*gdr_internal_pin_buffer_v2)(gdr_t g, unsigned long addr, size_t size, uint32_t flags, gdr_mh_t* handle);
static int (*gdr_internal_unpin_buffer)(gdr_t g, gdr_mh_t handle);
static int (*gdr_internal_get_info)(gdr_t g, gdr_mh_t handle, gdr_info_t* info);
static int (*gdr_internal_get_info_v2)(gdr_t g, gdr_mh_t handle, gdr_info_t* info);
static int (*gdr_internal_map)(gdr_t g, gdr_mh_t handle, void** va, size_t size);
static int (*gdr_internal_unmap)(gdr_t g, gdr_mh_t handle, void* va, size_t size);
static void (*gdr_internal_runtime_get_version)(int* major, int* minor);
static void (*gdr_internal_driver_get_version)(gdr_t g, int* major, int* minor);
static int (*gdr_internal_driver_get_version_checked)(gdr_t g, int* major, int* minor);
static int (*gdr_internal_get_attribute)(gdr_t g, gdr_attr_t attr, int* v);
static int (*gdr_internal_copy_to_mapping)(gdr_mh_t handle, void* map_d_ptr, const void* h_ptr, size_t size);
static int (*gdr_internal_copy_from_mapping)(gdr_mh_t handle, void* h_ptr, const void* map_d_ptr, size_t size);
static int gdrRuntimeMajor = 0;
static int gdrRuntimeMinor = 0;
static void* gdrhandle = NULL;

enum ncclGdrBackend {
  ncclGdrBackendNone = 0,
  ncclGdrBackendLibrary,
  ncclGdrBackendDmaBuf
};

struct ncclGdrOps {
  const char* name;
  ncclGdrBackend backend;
  gdr_t (*open)(void);
  int (*close)(gdr_t g);
  int (*pinBuffer)(gdr_t g, unsigned long addr, size_t size, uint64_t p2pToken, uint32_t vaSpace, gdr_mh_t* handle);
  int (*pinBufferV2)(gdr_t g, unsigned long addr, size_t size, uint32_t flags, gdr_mh_t* handle);
  int (*unpinBuffer)(gdr_t g, gdr_mh_t handle);
  int (*getInfo)(gdr_t g, gdr_mh_t handle, gdr_info_t* info);
  int (*map)(gdr_t g, gdr_mh_t handle, void** va, size_t size);
  int (*unmap)(gdr_t g, gdr_mh_t handle, void* va, size_t size);
  void (*runtimeGetVersion)(int* major, int* minor);
  void (*driverGetVersion)(gdr_t g, int* major, int* minor);
  int (*driverGetVersionChecked)(gdr_t g, int* major, int* minor);
  int (*getAttribute)(gdr_t g, gdr_attr_t attr, int* v);
  int (*copyToMapping)(gdr_mh_t handle, void* map_d_ptr, const void* h_ptr, size_t size);
  int (*copyFromMapping)(gdr_mh_t handle, void* h_ptr, const void* map_d_ptr, size_t size);
};

static ncclGdrOps gdrLibraryOps = {};
static const ncclGdrOps* gdrOps = NULL;

struct gdr {
  size_t pageSize;
  size_t pageMask;
};

struct ncclGdrDmaBufMapping {
  int fd;
  uint64_t va;
  size_t pageOffset;
  uint64_t mappedSize;
  uint32_t pageSize;
  unsigned mapped:1;
  unsigned wcMapping:1;
  gdr_mapping_type_t mappingType;
  void* cpuMappedVa;
  size_t cpuMappedLen;
};

static gdr_mh_t ncclGdrDmaBufHandleFromMapping(ncclGdrDmaBufMapping* mapping) {
  gdr_mh_t mh;
  mh.h = (unsigned long)mapping;
  return mh;
}

static ncclGdrDmaBufMapping* ncclGdrDmaBufMappingFromHandle(gdr_mh_t mh) {
  return (ncclGdrDmaBufMapping*)mh.h;
}

static const char* ncclGdrMappingTypeName(gdr_mapping_type_t type) {
  switch (type) {
  case GDR_MAPPING_TYPE_NONE:
    return "none";
  case GDR_MAPPING_TYPE_WC:
    return "wc";
  case GDR_MAPPING_TYPE_CACHING:
    return "caching";
  case GDR_MAPPING_TYPE_DEVICE:
    return "device";
  default:
    return "unknown";
  }
}

static int ncclGdrCudaError(CUresult status, const char* call) {
  if (status == CUDA_SUCCESS) return 0;
  const char* errStr = nullptr;
  if (CUPFN(cuGetErrorString) != nullptr) (void)CUPFN(cuGetErrorString(status, &errStr));
  WARN("%s failed: %d%s%s", call, (int)status, errStr ? " " : "", errStr ? errStr : "");
  return (int)status;
}

static int ncclGdrDmaBufSupported() {
#if defined(NCCL_OS_LINUX) && CUDA_VERSION >= 11070
  int driverVersion = 0;
  if (ncclCudaLibraryInit() != ncclSuccess) return 0;
  if (ncclCudaDriverVersion(&driverVersion) != ncclSuccess || driverVersion < 13030) return 0;
  if (CUPFN(cuInit) == nullptr || CUPFN(cuDeviceGet) == nullptr || CUPFN(cuDeviceGetCount) == nullptr ||
      CUPFN(cuDeviceGetAttribute) == nullptr || CUPFN(cuCtxGetCurrent) == nullptr ||
      CUPFN(cuCtxSetCurrent) == nullptr || CUPFN(cuDevicePrimaryCtxRetain) == nullptr ||
      CUPFN(cuDevicePrimaryCtxRelease) == nullptr || CUPFN(cuPointerGetAttribute) == nullptr ||
      CUPFN(cuMemGetHandleForAddressRange) == nullptr) {
    return 0;
  }

  if (CUPFN(cuInit(0)) != CUDA_SUCCESS) return 0;

  int deviceCount = 0;
  if (CUPFN(cuDeviceGetCount(&deviceCount)) != CUDA_SUCCESS) return 0;
  if (deviceCount == 0) return 0;
  for (int cudaDev = 0; cudaDev < deviceCount; ++cudaDev) {
    CUdevice dev;
    int supported = 0;
    if (CUPFN(cuDeviceGet(&dev, cudaDev)) != CUDA_SUCCESS) return 0;
    if (CUPFN(cuDeviceGetAttribute(&supported, CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED, dev)) != CUDA_SUCCESS) return 0;
    if (!supported) {
      INFO(NCCL_INIT, "NCCL internal DMA-BUF mmap backend disabled: CUDA device %d does not support DMA-BUF", cudaDev);
      return 0;
    }
  }
  return 1;
#else
  return 0;
#endif
}

static gdr_t ncclGdrDmaBufOpen() {
  gdr_t g = nullptr;
  if (ncclCalloc(&g, 1) != ncclSuccess) return nullptr;
  g->pageSize = ncclOsGetPageSize();
  g->pageMask = ~(g->pageSize - 1);
  return g;
}

static int ncclGdrDmaBufClose(gdr_t g) {
  free(g);
  return 0;
}

static int ncclGdrDmaBufPin(gdr_t g, unsigned long addr, size_t size, uint32_t flags, gdr_mh_t* handle) {
#if defined(NCCL_OS_LINUX) && CUDA_VERSION >= 11070
  if (handle == nullptr) return EINVAL;
  if (g == nullptr) return EINVAL;
  *handle = ncclGdrDmaBufHandleFromMapping(nullptr);

  unsigned long long dmabufFlags = 0;
  if (flags & GDR_PIN_FLAG_FORCE_PCIE) {
#if CUDA_VERSION >= 12080
    dmabufFlags |= CU_MEM_RANGE_FLAG_DMA_BUF_MAPPING_TYPE_PCIE;
#else
    return EINVAL;
#endif
  }

  uint64_t alignedPtr = addr & g->pageMask;
  size_t pageOffset = addr - alignedPtr;
  size_t alignedSize = (size + pageOffset + g->pageSize - 1) & g->pageMask;
  int deviceOrdinal = -1;
  CUdevice dev;
  int dmabufSupported = 0;
  int coherent = 0;
  int fd = -1;
  CUcontext ctx = nullptr;
  CUcontext prevCtx = nullptr;
  bool switchedCtx = false;
  bool retainedPrimaryCtx = false;
  int ret = 0;
  ncclGdrDmaBufMapping* mapping = nullptr;

  CUresult status =
    CUPFN(cuPointerGetAttribute(&deviceOrdinal, CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL, (CUdeviceptr)alignedPtr));
  if (status != CUDA_SUCCESS)
    return ncclGdrCudaError(status, "cuPointerGetAttribute(CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL)");

  status = CUPFN(cuDeviceGet(&dev, deviceOrdinal));
  if (status != CUDA_SUCCESS) return ncclGdrCudaError(status, "cuDeviceGet");

  status = CUPFN(cuDeviceGetAttribute(&dmabufSupported, CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED, dev));
  if (status != CUDA_SUCCESS)
    return ncclGdrCudaError(status, "cuDeviceGetAttribute(CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED)");
  if (!dmabufSupported) return ENOTSUP;

  status =
    CUPFN(cuDeviceGetAttribute(&coherent, CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES, dev));
  if (status != CUDA_SUCCESS) coherent = 0;

  status = CUPFN(cuCtxGetCurrent(&prevCtx));
  if (status != CUDA_SUCCESS) return ncclGdrCudaError(status, "cuCtxGetCurrent");
  if (prevCtx == nullptr) {
    status = CUPFN(cuDevicePrimaryCtxRetain(&ctx, dev));
    if (status != CUDA_SUCCESS) return ncclGdrCudaError(status, "cuDevicePrimaryCtxRetain");
    retainedPrimaryCtx = true;

    status = CUPFN(cuCtxSetCurrent(ctx));
    if (status != CUDA_SUCCESS) {
      ret = ncclGdrCudaError(status, "cuCtxSetCurrent");
      goto cleanup;
    }
    switchedCtx = true;
  }

  status = CUPFN(cuMemGetHandleForAddressRange((void*)&fd, (CUdeviceptr)alignedPtr, alignedSize,
                                               CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, dmabufFlags));
  if (status != CUDA_SUCCESS) {
    ret = ncclGdrCudaError(status, "cuMemGetHandleForAddressRange");
    goto cleanup;
  }

  if (ncclCalloc(&mapping, 1) != ncclSuccess) {
    ret = ENOMEM;
    goto cleanup;
  }
  mapping->fd = fd;
  mapping->va = addr;
  mapping->pageOffset = pageOffset;
  mapping->mappedSize = alignedSize;
  mapping->pageSize = g->pageSize;
  mapping->mapped = 0;
  mapping->wcMapping = 0;
  mapping->mappingType =
    (coherent && !(flags & GDR_PIN_FLAG_FORCE_PCIE)) ? GDR_MAPPING_TYPE_CACHING : GDR_MAPPING_TYPE_WC;
  mapping->cpuMappedVa = nullptr;
  mapping->cpuMappedLen = 0;
  *handle = ncclGdrDmaBufHandleFromMapping(mapping);
  TRACE(NCCL_INIT,
        "GDRCOPY internal DMA-BUF pin: devVa 0x%lx alignedVa 0x%lx size %zu alignedSize %zu pageOffset %zu pageSize "
        "%u fd %d flags 0x%x dmabufFlags 0x%llx cudaDev %d mappingType %s",
        addr, (unsigned long)alignedPtr, size, alignedSize, pageOffset, mapping->pageSize, fd, flags, dmabufFlags,
        deviceOrdinal, ncclGdrMappingTypeName(mapping->mappingType));
  fd = -1;

cleanup:
  if (ret != 0 && fd >= 0) (void)close(fd);
  if (switchedCtx) {
    CUresult cleanupStatus = CUPFN(cuCtxSetCurrent(prevCtx));
    if (cleanupStatus != CUDA_SUCCESS) (void)ncclGdrCudaError(cleanupStatus, "cuCtxSetCurrent(previous)");
  }
  if (retainedPrimaryCtx) {
    CUresult cleanupStatus = CUPFN(cuDevicePrimaryCtxRelease(dev));
    if (cleanupStatus != CUDA_SUCCESS) (void)ncclGdrCudaError(cleanupStatus, "cuDevicePrimaryCtxRelease");
  }
  return ret;
#else
  return ENOTSUP;
#endif
}

static int ncclGdrDmaBufUnpin(gdr_t, gdr_mh_t handle) {
#if defined(NCCL_OS_LINUX)
  ncclGdrDmaBufMapping* mapping = ncclGdrDmaBufMappingFromHandle(handle);
  if (mapping == nullptr) return EINVAL;
  if (mapping->mapped && mapping->cpuMappedVa != nullptr) {
    (void)munmap(mapping->cpuMappedVa, mapping->cpuMappedLen);
  }
  if (mapping->fd >= 0) (void)close(mapping->fd);
  free(mapping);
  return 0;
#else
  return ENOTSUP;
#endif
}

static int ncclGdrDmaBufGetInfo(gdr_t, gdr_mh_t handle, gdr_info_t* info) {
  ncclGdrDmaBufMapping* mapping = ncclGdrDmaBufMappingFromHandle(handle);
  if (mapping == nullptr || info == nullptr) return EINVAL;
  info->va = mapping->va;
  info->mapped_size = mapping->mappedSize - mapping->pageOffset;
  info->page_size = mapping->pageSize;
  info->tm_cycles = 0;
  info->cycles_per_ms = 0;
  info->mapped = mapping->mapped;
  info->wc_mapping = mapping->wcMapping;
  info->mapping_type = mapping->mapped ? mapping->mappingType : GDR_MAPPING_TYPE_NONE;
  return 0;
}

static int ncclGdrDmaBufMap(gdr_t g, gdr_mh_t handle, void** va, size_t size) {
#if defined(NCCL_OS_LINUX)
  if (g == nullptr || va == nullptr) return EINVAL;
  ncclGdrDmaBufMapping* mapping = ncclGdrDmaBufMappingFromHandle(handle);
  if (mapping == nullptr || mapping->fd < 0) return EINVAL;
  if (mapping->mapped) return EAGAIN;
  if (mapping->pageOffset > mapping->mappedSize || size > mapping->mappedSize - mapping->pageOffset) return EINVAL;

  size_t roundedSize = (size + mapping->pageOffset + g->pageSize - 1) & g->pageMask;
  void* map = mmap(nullptr, roundedSize, PROT_READ | PROT_WRITE, MAP_SHARED, mapping->fd, 0);
  if (map == MAP_FAILED) {
    int ret = errno;
    WARN("mmap DMA-BUF fd %d size %zu offset 0 failed: %d %s", mapping->fd, roundedSize, ret, strerror(ret));
    return ret;
  }

  mapping->mapped = 1;
  mapping->wcMapping = (mapping->mappingType == GDR_MAPPING_TYPE_WC);
  mapping->cpuMappedVa = map;
  mapping->cpuMappedLen = roundedSize;
  *va = (char*)map + mapping->pageOffset;
  return 0;
#else
  return ENOTSUP;
#endif
}

static int ncclGdrDmaBufUnmap(gdr_t g, gdr_mh_t handle, void* va, size_t size) {
#if defined(NCCL_OS_LINUX)
  if (g == nullptr || va == nullptr) return EINVAL;
  ncclGdrDmaBufMapping* mapping = ncclGdrDmaBufMappingFromHandle(handle);
  if (mapping == nullptr || !mapping->mapped) return EINVAL;
  if (va != (char*)mapping->cpuMappedVa + mapping->pageOffset) return EINVAL;
  if (mapping->pageOffset > mapping->mappedSize || size > mapping->mappedSize - mapping->pageOffset) return EINVAL;

  size_t roundedSize = (size + mapping->pageOffset + g->pageSize - 1) & g->pageMask;
  if (roundedSize != mapping->cpuMappedLen) return EINVAL;
  if (munmap(mapping->cpuMappedVa, roundedSize) != 0) {
    int ret = errno;
    WARN("munmap DMA-BUF mapping %p size %zu failed: %d %s", mapping->cpuMappedVa, roundedSize, ret, strerror(ret));
    return ret;
  }

  mapping->mapped = 0;
  mapping->wcMapping = 0;
  mapping->cpuMappedVa = nullptr;
  mapping->cpuMappedLen = 0;
  return 0;
#else
  return ENOTSUP;
#endif
}

static int ncclGdrDmaBufCopyToMapping(gdr_mh_t handle, void* map_d_ptr, const void* h_ptr, size_t size) {
  ncclGdrDmaBufMapping* mapping = ncclGdrDmaBufMappingFromHandle(handle);
  if (mapping == nullptr || !mapping->mapped) return EINVAL;
  if (size == 0) return 0;
  memcpy(map_d_ptr, h_ptr, size);
  if (mapping->mappingType == GDR_MAPPING_TYPE_WC) wc_store_fence();
  return 0;
}

static int ncclGdrDmaBufCopyFromMapping(gdr_mh_t handle, void* h_ptr, const void* map_d_ptr, size_t size) {
  ncclGdrDmaBufMapping* mapping = ncclGdrDmaBufMappingFromHandle(handle);
  if (mapping == nullptr || !mapping->mapped) return EINVAL;
  if (size == 0) return 0;
  memcpy(h_ptr, map_d_ptr, size);
  if (mapping->mappingType == GDR_MAPPING_TYPE_WC) wc_store_fence();
  return 0;
}

static int ncclGdrDmaBufGetAttribute(gdr_t, gdr_attr_t attr, int* v) {
  if (v == nullptr) return EINVAL;
  switch (attr) {
  case GDR_ATTR_USING_DMA_BUF_MMAP:
    *v = 1;
    return 0;
  default:
    return EINVAL;
  }
}

static int ncclGdrDmaBufPinBuffer(gdr_t g, unsigned long addr, size_t size, uint64_t, uint32_t, gdr_mh_t* handle) {
  return ncclGdrDmaBufPin(g, addr, size, GDR_PIN_FLAG_DEFAULT, handle);
}

static int ncclGdrDmaBufPinBufferV2(gdr_t g, unsigned long addr, size_t size, uint32_t flags, gdr_mh_t* handle) {
  return ncclGdrDmaBufPin(g, addr, size, flags, handle);
}

static const ncclGdrOps ncclGdrDmaBufOps = {"internal DMA-BUF",
                                            ncclGdrBackendDmaBuf,
                                            ncclGdrDmaBufOpen,
                                            ncclGdrDmaBufClose,
                                            ncclGdrDmaBufPinBuffer,
                                            ncclGdrDmaBufPinBufferV2,
                                            ncclGdrDmaBufUnpin,
                                            ncclGdrDmaBufGetInfo,
                                            ncclGdrDmaBufMap,
                                            ncclGdrDmaBufUnmap,
                                            NULL, // runtimeGetVersion
                                            NULL, // driverGetVersion
                                            NULL, // driverGetVersionChecked
                                            ncclGdrDmaBufGetAttribute,
                                            ncclGdrDmaBufCopyToMapping,
                                            ncclGdrDmaBufCopyFromMapping};

static void ncclGdrClearLibraryBackend() {
  gdr_internal_open = NULL;
  gdr_internal_close = NULL;
  gdr_internal_pin_buffer = NULL;
  gdr_internal_pin_buffer_v2 = NULL;
  gdr_internal_unpin_buffer = NULL;
  gdr_internal_get_info = NULL;
  gdr_internal_get_info_v2 = NULL;
  gdr_internal_map = NULL;
  gdr_internal_unmap = NULL;
  gdr_internal_runtime_get_version = NULL;
  gdr_internal_driver_get_version = NULL;
  gdr_internal_driver_get_version_checked = NULL;
  gdr_internal_get_attribute = NULL;
  gdr_internal_copy_to_mapping = NULL;
  gdr_internal_copy_from_mapping = NULL;
  gdrRuntimeMajor = 0;
  gdrRuntimeMinor = 0;
  gdrLibraryOps = {};
  if (gdrOps != &ncclGdrDmaBufOps) gdrOps = NULL;
}

static void ncclGdrCloseLibraryBackend() {
  if (gdrhandle != NULL) {
    ncclOsDlclose(gdrhandle);
    gdrhandle = NULL;
  }
  ncclGdrClearLibraryBackend();
}

static ncclResult_t ncclGdrUseInternalDmaBufBackend() {
  if (!ncclGdrDmaBufSupported()) {
    INFO(NCCL_INIT, "NCCL internal DMA-BUF mmap backend unavailable");
    return ncclInternalError;
  }
  ncclGdrClearLibraryBackend();
  gdrOps = &ncclGdrDmaBufOps;
  return ncclSuccess;
}

static void ncclGdrUseLibraryBackend() {
  gdrLibraryOps.name = "GDRCOPY";
  gdrLibraryOps.backend = ncclGdrBackendLibrary;
  gdrLibraryOps.open = gdr_internal_open;
  gdrLibraryOps.close = gdr_internal_close;
  gdrLibraryOps.pinBuffer = gdr_internal_pin_buffer;
  gdrLibraryOps.pinBufferV2 = gdr_internal_pin_buffer_v2;
  gdrLibraryOps.unpinBuffer = gdr_internal_unpin_buffer;
  gdrLibraryOps.getInfo = gdr_internal_get_info_v2 != NULL ? gdr_internal_get_info_v2 : gdr_internal_get_info;
  gdrLibraryOps.map = gdr_internal_map;
  gdrLibraryOps.unmap = gdr_internal_unmap;
  gdrLibraryOps.runtimeGetVersion = gdr_internal_runtime_get_version;
  gdrLibraryOps.driverGetVersion = gdr_internal_driver_get_version;
  gdrLibraryOps.driverGetVersionChecked = gdr_internal_driver_get_version_checked;
  gdrLibraryOps.getAttribute = gdr_internal_get_attribute;
  gdrLibraryOps.copyToMapping = gdr_internal_copy_to_mapping;
  gdrLibraryOps.copyFromMapping = gdr_internal_copy_from_mapping;
  gdrOps = &gdrLibraryOps;
}

// Used to make the GDR library calls thread safe
std::mutex& getGdrMutex() {
  static std::mutex gdrMutex;
  return gdrMutex;
}

#if defined(NCCL_OS_WINDOWS)
#define GDRAPI_LIBNAME "gdrapi.dll"
#else
#define GDRAPI_LIBNAME "libgdrapi.so"
#endif

#define LOAD_SYM(handle, symbol, funcptr) \
  do { \
    cast = (void**)&funcptr; \
    tmp = ncclOsDlsym(handle, symbol); \
    if (tmp == NULL) { \
      WARN("ncclOsDlsym failed on %s - %s", symbol, ncclOsDlerror()); \
      goto teardown; \
    } \
    *cast = tmp; \
  } while (0)

#define LOAD_SYM_OPTIONAL(handle, symbol, funcptr) \
  do { \
    cast = (void**)&funcptr; \
    tmp = ncclOsDlsym(handle, symbol); \
    if (tmp == NULL) { \
      INFO(NCCL_INIT, "ncclOsDlsym failed on %s, ignoring", symbol); \
    } \
    *cast = tmp; \
  } while (0)

static std::once_flag initOnceFlag;
static ncclResult_t initResult;

static void initOnceFunc(void) {
  void* tmp;
  void** cast;
  int64_t useInternalDmaBuf = ncclParamGdrCopyUseInternalDmaBuf();

  if (useInternalDmaBuf == ncclGdrUseInternalDmaBufRequired || useInternalDmaBuf == ncclGdrUseInternalDmaBufOptional) {
    INFO(NCCL_INIT, "NCCL_GDRCOPY_USE_INTERNAL_DMABUF=%lld set; skipping %s", (long long)useInternalDmaBuf,
         GDRAPI_LIBNAME);
    initResult = ncclGdrUseInternalDmaBufBackend();
    return;
  }
  if (useInternalDmaBuf != ncclGdrUseInternalDmaBufDisabled) {
    WARN("Invalid NCCL_GDRCOPY_USE_INTERNAL_DMABUF=%lld; expected 0, 1, or 2", (long long)useInternalDmaBuf);
  }

  gdrhandle = ncclOsDlopen(GDRAPI_LIBNAME, NCCL_OS_DL_NOW);
  if (!gdrhandle) {
    INFO(NCCL_INIT, "Failed to open %s - %s; falling back to NCCL internal DMA-BUF mmap backend", GDRAPI_LIBNAME,
         ncclOsDlerror());
    initResult = ncclGdrUseInternalDmaBufBackend();
    return;
  }

  /* Load the function pointers from the DL library image */
  LOAD_SYM(gdrhandle, "gdr_open", gdr_internal_open);
  LOAD_SYM(gdrhandle, "gdr_close", gdr_internal_close);
  LOAD_SYM(gdrhandle, "gdr_pin_buffer", gdr_internal_pin_buffer);
  LOAD_SYM_OPTIONAL(gdrhandle, "gdr_pin_buffer_v2", gdr_internal_pin_buffer_v2);
  LOAD_SYM(gdrhandle, "gdr_unpin_buffer", gdr_internal_unpin_buffer);
  LOAD_SYM(gdrhandle, "gdr_get_info", gdr_internal_get_info);
  LOAD_SYM_OPTIONAL(gdrhandle, "gdr_get_info_v2", gdr_internal_get_info_v2);
  LOAD_SYM(gdrhandle, "gdr_map", gdr_internal_map);
  LOAD_SYM(gdrhandle, "gdr_unmap", gdr_internal_unmap);
  LOAD_SYM(gdrhandle, "gdr_runtime_get_version", gdr_internal_runtime_get_version);
  LOAD_SYM(gdrhandle, "gdr_driver_get_version", gdr_internal_driver_get_version);
  gdr_internal_driver_get_version_checked = (int (*)(gdr_t, int*, int*))tmp;
  LOAD_SYM_OPTIONAL(gdrhandle, "gdr_get_attribute", gdr_internal_get_attribute);
  LOAD_SYM(gdrhandle, "gdr_copy_to_mapping", gdr_internal_copy_to_mapping);
  LOAD_SYM(gdrhandle, "gdr_copy_from_mapping", gdr_internal_copy_from_mapping);

  ncclGdrUseLibraryBackend();
  initResult = ncclSuccess;
  return;

teardown:
  ncclGdrCloseLibraryBackend();
  INFO(NCCL_INIT, "Failed to initialize %s symbols; falling back to NCCL internal DMA-BUF mmap backend",
       GDRAPI_LIBNAME);
  initResult = ncclGdrUseInternalDmaBufBackend();
  return;
}

ncclResult_t wrap_gdr_symbols(void) {
  std::call_once(initOnceFlag, initOnceFunc);
  return initResult;
}

bool ncclGdrIsInternalDmaBufBackend(void) {
  return gdrOps != NULL && gdrOps->backend == ncclGdrBackendDmaBuf;
}

bool ncclGdrInternalDmaBufRequired(void) {
  return ncclParamGdrCopyUseInternalDmaBuf() == ncclGdrUseInternalDmaBufRequired;
}

static const char* ncclGdrBackendName() {
  return gdrOps != NULL && gdrOps->name != NULL ? gdrOps->name : "GDRCOPY";
}

static ncclResult_t ncclGdrUnavailable(const char* op) {
  if (gdrOps == NULL) {
    WARN("GDRCOPY lib wrapper not initialized.");
  } else {
    INFO(NCCL_INIT, "%s not available for %s backend", op, ncclGdrBackendName());
  }
  return ncclInternalError;
}

gdr_t wrap_gdr_open(void) {
  if (gdrOps == NULL || gdrOps->open == NULL) {
    WARN("GDRCOPY lib wrapper not initialized.");
    return NULL;
  }
  gdr_t handle = gdrOps->open();
  if (handle != NULL || gdrOps != &gdrLibraryOps) return handle;

  INFO(NCCL_INIT, "%s gdr_open() failed; falling back to NCCL internal DMA-BUF mmap backend", GDRAPI_LIBNAME);
  ncclGdrCloseLibraryBackend();
  if (ncclGdrUseInternalDmaBufBackend() != ncclSuccess || gdrOps == NULL || gdrOps->open == NULL) return NULL;
  return gdrOps->open();
}

ncclResult_t wrap_gdr_close(gdr_t g) {
  if (gdrOps == NULL || gdrOps->close == NULL) return ncclGdrUnavailable("close");
  int ret = gdrOps->close(g);
  if (ret != 0) {
    WARN("gdr_close() failed: %d", ret);
    return ncclSystemError;
  }
  return ncclSuccess;
}

ncclResult_t wrap_gdr_pin_buffer(gdr_t g, unsigned long addr, size_t size, uint64_t p2p_token, uint32_t va_space,
                                 gdr_mh_t* handle) {
  if (gdrOps == NULL || gdrOps->pinBuffer == NULL) return ncclGdrUnavailable("pin_buffer");
  int ret;
  GDRLOCKCALL(gdrOps->pinBuffer(g, addr, size, p2p_token, va_space, handle), ret);
  if (ret != 0) {
    WARN("gdr_pin_buffer(addr %lx, size %zu) failed: %d", addr, size, ret);
    return ncclSystemError;
  }
  return ncclSuccess;
}

bool ncclGdrPinV2Available(void) {
  static std::once_flag onceFlag;
  static bool available = false;
  std::call_once(onceFlag, []() {
    if (wrap_gdr_symbols() != ncclSuccess) return;
    if (gdrOps == NULL || gdrOps->pinBufferV2 == NULL) return;
    if (ncclGdrIsInternalDmaBufBackend()) {
      available = true;
      return;
    }
    if (gdrOps->runtimeGetVersion == NULL) return;
    int major, minor;
    gdrOps->runtimeGetVersion(&major, &minor);
    available = (major > 2 || (major == 2 && minor >= 5));
  });
  return available;
}

ncclResult_t wrap_gdr_pin_buffer_v2(gdr_t g, unsigned long addr, size_t size, uint32_t flags, gdr_mh_t* handle) {
  if (!ncclGdrPinV2Available()) {
    WARN("gdr_pin_buffer_v2 not available; GDRCopy >= 2.5 required");
    return ncclInternalError;
  }
  if (gdrOps == NULL || gdrOps->pinBufferV2 == NULL) return ncclGdrUnavailable("pin_buffer_v2");
  int ret;
  GDRLOCKCALL(gdrOps->pinBufferV2(g, addr, size, flags, handle), ret);
  if (ret != 0) {
    WARN("gdr_pin_buffer_v2(addr %lx, size %zu, flags %u) failed: %d", addr, size, flags, ret);
    return ncclSystemError;
  }
  return ncclSuccess;
}

ncclResult_t wrap_gdr_unpin_buffer(gdr_t g, gdr_mh_t handle) {
  if (gdrOps == NULL || gdrOps->unpinBuffer == NULL) return ncclGdrUnavailable("unpin_buffer");
  int ret;
  GDRLOCKCALL(gdrOps->unpinBuffer(g, handle), ret);
  if (ret != 0) {
    WARN("gdr_unpin_buffer(handle %lx) failed: %d", handle.h, ret);
    return ncclSystemError;
  }
  return ncclSuccess;
}

ncclResult_t wrap_gdr_get_info(gdr_t g, gdr_mh_t handle, gdr_info_t* info) {
  if (gdrOps == NULL || gdrOps->getInfo == NULL) return ncclGdrUnavailable("get_info");
  int ret;
  GDRLOCKCALL(gdrOps->getInfo(g, handle, info), ret);
  if (ret != 0) {
    WARN("gdr_get_info(handle %lx) failed: %d", handle.h, ret);
    return ncclSystemError;
  }
  return ncclSuccess;
}

ncclResult_t wrap_gdr_map(gdr_t g, gdr_mh_t handle, void** va, size_t size) {
  if (gdrOps == NULL || gdrOps->map == NULL) return ncclGdrUnavailable("map");
  int ret;
  GDRLOCKCALL(gdrOps->map(g, handle, va, size), ret);
  if (ret != 0) {
    WARN("gdr_map(handle %lx, size %zu) failed: %d", handle.h, size, ret);
    return ncclSystemError;
  }
  return ncclSuccess;
}

ncclResult_t wrap_gdr_unmap(gdr_t g, gdr_mh_t handle, void* va, size_t size) {
  if (gdrOps == NULL || gdrOps->unmap == NULL) return ncclGdrUnavailable("unmap");
  int ret;
  GDRLOCKCALL(gdrOps->unmap(g, handle, va, size), ret);
  if (ret != 0) {
    WARN("gdr_unmap(handle %lx, va %p, size %zu) failed: %d", handle.h, va, size, ret);
    return ncclSystemError;
  }
  return ncclSuccess;
}

ncclResult_t wrap_gdr_runtime_get_version(int* major, int* minor) {
  if (gdrOps == NULL || gdrOps->runtimeGetVersion == NULL) return ncclGdrUnavailable("runtime_get_version");
  gdrOps->runtimeGetVersion(major, minor);
  if (gdrOps == &gdrLibraryOps) {
    gdrRuntimeMajor = *major;
    gdrRuntimeMinor = *minor;
  }
  return ncclSuccess;
}

ncclResult_t wrap_gdr_driver_get_version(gdr_t g, int* major, int* minor) {
  if (gdrOps == NULL) return ncclGdrUnavailable("driver_get_version");
  if (gdrRuntimeMajor == 0 && gdrOps == &gdrLibraryOps && gdrOps->runtimeGetVersion != NULL) {
    gdrOps->runtimeGetVersion(&gdrRuntimeMajor, &gdrRuntimeMinor);
  }
  if (gdrRuntimeMajor > 2 || (gdrRuntimeMajor == 2 && gdrRuntimeMinor >= 6)) {
    if (gdrOps->driverGetVersionChecked == NULL) return ncclGdrUnavailable("driver_get_version");
    int ret;
    GDRLOCKCALL(gdrOps->driverGetVersionChecked(g, major, minor), ret);
    if (ret != 0) {
      WARN("gdr_driver_get_version() failed: %d", ret);
      return ncclSystemError;
    }
  } else {
    if (gdrOps->driverGetVersion == NULL) return ncclGdrUnavailable("driver_get_version");
    std::lock_guard<std::mutex> lock(getGdrMutex());
    gdrOps->driverGetVersion(g, major, minor);
  }
  return ncclSuccess;
}

ncclResult_t wrap_gdr_get_attribute(gdr_t g, gdr_attr_t attr, int* v) {
  if (gdrOps == NULL) return ncclGdrUnavailable("get_attribute");
  if (gdrOps->getAttribute == NULL) {
    INFO(NCCL_INIT, "gdr_get_attribute() not available");
    return ncclInternalError;
  }
  int ret;
  GDRLOCKCALL(gdrOps->getAttribute(g, attr, v), ret);
  if (ret != 0) {
    WARN("gdr_get_attribute(attr %d) failed: %d", attr, ret);
    return ncclSystemError;
  }
  return ncclSuccess;
}

ncclResult_t wrap_gdr_is_dma_buf_mmap(gdr_t g, int* v) {
  return wrap_gdr_get_attribute(g, GDR_ATTR_USING_DMA_BUF_MMAP, v);
}

ncclResult_t wrap_gdr_copy_to_mapping(gdr_mh_t handle, void* map_d_ptr, const void* h_ptr, size_t size) {
  if (gdrOps == NULL || gdrOps->copyToMapping == NULL) return ncclGdrUnavailable("copy_to_mapping");
  int ret;
  GDRLOCKCALL(gdrOps->copyToMapping(handle, map_d_ptr, h_ptr, size), ret);
  if (ret != 0) {
    WARN("gdr_copy_to_mapping(handle %lx, map_d_ptr %p, h_ptr %p, size %zu) failed: %d", handle.h, map_d_ptr, h_ptr,
         size, ret);
    return ncclSystemError;
  }
  return ncclSuccess;
}

ncclResult_t wrap_gdr_copy_from_mapping(gdr_mh_t handle, void* h_ptr, const void* map_d_ptr, size_t size) {
  if (gdrOps == NULL || gdrOps->copyFromMapping == NULL) return ncclGdrUnavailable("copy_from_mapping");
  int ret;
  GDRLOCKCALL(gdrOps->copyFromMapping(handle, h_ptr, map_d_ptr, size), ret);
  if (ret != 0) {
    WARN("gdr_copy_from_mapping(handle %lx, h_ptr %p, map_d_ptr %p, size %zu) failed: %d", handle.h, h_ptr, map_d_ptr,
         size, ret);
    return ncclSystemError;
  }
  return ncclSuccess;
}
