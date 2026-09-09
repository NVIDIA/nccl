/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_GDRWRAP_H_
#define NCCL_GDRWRAP_H_

#include "nccl.h"
#include "alloc.h"
#include <stdint.h> // for standard [u]intX_t types
#include <stdio.h>
#include <stdlib.h>
#include <mutex>

// These can be used if the GDR library isn't thread safe
std::mutex& getGdrMutex();
#define GDRLOCKCALL(cmd, ret) do {                      \
    std::lock_guard<std::mutex> lock(getGdrMutex());   \
    ret = cmd;                                          \
} while(false)

#define GDRCHECK(cmd) do {                              \
    int e;                                              \
    /* GDRLOCKCALL(cmd, e); */                          \
    e = cmd;                                            \
    if( e != 0 ) {                                      \
      WARN("GDRCOPY failure %d", e);                    \
      return ncclSystemError;                           \
    }                                                   \
} while(false)

// This is required as the GDR memory is mapped WC
#if !defined(__NVCC__)
#if defined(__PPC__)
static inline void wc_store_fence(void) { asm volatile("sync" : : : "memory"); }
#elif defined(__x86_64__) || (defined(_MSC_VER) && (defined(_M_X64) || defined(_M_AMD64)))
#include <immintrin.h>
static inline void wc_store_fence(void) { _mm_sfence(); }
#elif defined(__aarch64__)
static inline void wc_store_fence(void) { asm volatile("dsb st" : : : "memory"); }
#endif
#endif

// Dynamically handle dependency on the GDR API library

/* Extracted from gdrapi.h (v2.6 May 2026) */

typedef enum gdr_pin_flags {
  GDR_PIN_FLAG_DEFAULT     = 0,
  GDR_PIN_FLAG_FORCE_PCIE  = 1
} gdr_pin_flags_t;

#define GPU_PAGE_SHIFT   16
#define GPU_PAGE_SIZE    (1UL << GPU_PAGE_SHIFT)
#define GPU_PAGE_OFFSET  (GPU_PAGE_SIZE-1)
#define GPU_PAGE_MASK    (~GPU_PAGE_OFFSET)

struct gdr;
typedef struct gdr *gdr_t;

typedef struct gdr_mh_s {
  unsigned long h;
} gdr_mh_t;

typedef enum gdr_mapping_type {
  GDR_MAPPING_TYPE_NONE     = 0,
  GDR_MAPPING_TYPE_WC       = 1,
  GDR_MAPPING_TYPE_CACHING  = 2,
  GDR_MAPPING_TYPE_DEVICE   = 3,
  GDR_MAPPING_TYPE_MAX
} gdr_mapping_type_t;

typedef enum gdr_attr {
  GDR_ATTR_USE_PERSISTENT_MAPPING       = 1,
  GDR_ATTR_SUPPORT_PIN_FLAG_FORCE_PCIE  = 2,
  GDR_ATTR_USING_DMA_BUF_MMAP           = 3,
  GDR_ATTR_MAX
} gdr_attr_t;

typedef struct gdr_info_v2 {
    uint64_t va;
    uint64_t mapped_size;
    uint32_t page_size;
    uint64_t tm_cycles;
    uint32_t cycles_per_ms;
    unsigned mapped:1;
    unsigned wc_mapping:1;
    gdr_mapping_type_t mapping_type;
} gdr_info_v2_t;
typedef gdr_info_v2_t gdr_info_t;
typedef gdr_info_t ncclGdrInfo_t;

/* End of gdrapi.h */

ncclResult_t wrap_gdr_symbols(void);

gdr_t wrap_gdr_open(void);
ncclResult_t wrap_gdr_close(gdr_t g);
ncclResult_t wrap_gdr_pin_buffer(gdr_t g, unsigned long addr, size_t size, uint64_t p2p_token, uint32_t va_space, gdr_mh_t *handle);
bool ncclGdrPinV2Available(void);
ncclResult_t wrap_gdr_pin_buffer_v2(gdr_t g, unsigned long addr, size_t size, uint32_t flags, gdr_mh_t *handle);
ncclResult_t wrap_gdr_unpin_buffer(gdr_t g, gdr_mh_t handle);
ncclResult_t wrap_gdr_get_info(gdr_t g, gdr_mh_t handle, ncclGdrInfo_t *info);
ncclResult_t wrap_gdr_map(gdr_t g, gdr_mh_t handle, void **va, size_t size);
ncclResult_t wrap_gdr_unmap(gdr_t g, gdr_mh_t handle, void *va, size_t size);
ncclResult_t wrap_gdr_runtime_get_version(int *major, int *minor);
ncclResult_t wrap_gdr_driver_get_version(gdr_t g, int *major, int *minor);
ncclResult_t wrap_gdr_get_attribute(gdr_t g, gdr_attr_t attr, int *v);
ncclResult_t wrap_gdr_is_dma_buf_mmap(gdr_t g, int *v);
ncclResult_t wrap_gdr_copy_to_mapping(gdr_mh_t handle, void *map_d_ptr, const void *h_ptr, size_t size);
ncclResult_t wrap_gdr_copy_from_mapping(gdr_mh_t handle, void *h_ptr, const void *map_d_ptr, size_t size);
bool ncclGdrIsInternalDmaBufBackend(void);
bool ncclGdrInternalDmaBufRequired(void);

// Global GDR driver handle; set once during NCCL init.
extern gdr_t ncclGdrCopy;

#include "alloc.h"

typedef struct gdr_mem_desc {
  void *gdrDevMem;
  void *gdrMap;
  size_t gdrOffset;
  size_t gdrMapSize;
  gdr_mh_t gdrMh;
} gdr_mem_desc_t;

static gdr_t ncclGdrInit() {
  int libMajor = 0, libMinor = 0, drvMajor = 0, drvMinor = 0;
  gdr_t handle = NULL;
  // Dynamically load the GDRAPI library symbols
  if (wrap_gdr_symbols() == ncclSuccess) {
    handle = wrap_gdr_open();
    if (handle == NULL && ncclGdrIsInternalDmaBufBackend()) {
      INFO(NCCL_INIT, "NCCL internal DMA-BUF mmap backend unavailable");
    }

    if (handle != NULL) {
      ncclResult_t res;

      if (ncclGdrIsInternalDmaBufBackend()) {
        INFO(NCCL_INIT, "GDRCOPY enabled using NCCL internal DMA-BUF mmap backend");
        return handle;
      }

      // Query the version of libgdrapi
      NCCLCHECKGOTO(wrap_gdr_runtime_get_version(&libMajor, &libMinor), res, error);

      // Only support GDRAPI 2.1 and later
      if (libMajor < 2 || (libMajor == 2 && libMinor < 1)) {
        goto error;
      }

      // GDRCopy 2.6 can use a DMA-BUF backend without /dev/gdrdrv, in which
      // case querying the gdrdrv driver version is not meaningful.
      int usingDmaBuf = 0;
      if ((libMajor > 2 || (libMajor == 2 && libMinor >= 6)) &&
          wrap_gdr_is_dma_buf_mmap(handle, &usingDmaBuf) == ncclSuccess && usingDmaBuf) {
        INFO(NCCL_INIT, "GDRCOPY enabled library %d.%d using DMA-BUF mmap backend", libMajor, libMinor);
      } else {
        NCCLCHECKGOTO(wrap_gdr_driver_get_version(handle, &drvMajor, &drvMinor), res, error);
        if (drvMajor < 2 || (drvMajor == 2 && drvMinor < 1)) goto error;
        INFO(NCCL_INIT, "GDRCOPY enabled library %d.%d driver %d.%d", libMajor, libMinor, drvMajor, drvMinor);
      }
    }
  }
  return handle;
error:
  if (handle != NULL) (void) wrap_gdr_close(handle);
  return NULL;
}

template <typename T>
static ncclResult_t ncclGdrCudaCalloc(T** ptr, T** devPtr, size_t nelem, void** gdrHandle, struct ncclMemManager* manager, uint32_t pinFlags = 0) {
  ncclGdrInfo_t info = {};
  size_t mapSize;
  gdr_mh_t mh;
  char *devMem;
  void *gdrMap;

  mapSize = ncclSizeOfT<T>()*nelem;

  // GDRCOPY Pinned buffer has to be a minimum of a GPU_PAGE_SIZE
  ALIGN_SIZE(mapSize, GPU_PAGE_SIZE);
  // GDRCOPY Pinned buffer has to be GPU_PAGE_SIZE aligned too
  NCCLCHECK(ncclCudaCalloc(&devMem, mapSize+GPU_PAGE_SIZE-1, manager));
  uint64_t alignedAddr = (((uint64_t) devMem) + GPU_PAGE_OFFSET) & GPU_PAGE_MASK;
  size_t align = alignedAddr - (uint64_t)devMem;

  if (ncclGdrPinV2Available() || pinFlags == GDR_PIN_FLAG_FORCE_PCIE) {
    // If pingFlags is set to FORCE_PCIE, we will error out if we can't honnor it.
    NCCLCHECK(wrap_gdr_pin_buffer_v2(ncclGdrCopy, alignedAddr, mapSize, pinFlags, &mh));
  } else {
    // TRACE(NCCL_INIT, "GDRCOPY: Pin buffer 0x%lx (%p) align %zu size %zu", alignedAddr, devMem, align, mapSize);
    NCCLCHECK(wrap_gdr_pin_buffer(ncclGdrCopy, alignedAddr, mapSize, 0, 0, &mh));
  }

  NCCLCHECK(wrap_gdr_map(ncclGdrCopy, mh, &gdrMap, mapSize));
  //TRACE(NCCL_INIT, "GDRCOPY : mapped %p (0x%lx) at %p", devMem, alignedAddr, gdrMap);

  NCCLCHECK(wrap_gdr_get_info(ncclGdrCopy, mh, &info));

  // Will offset ever be non zero ?
  ssize_t off = info.va - alignedAddr;

  gdr_mem_desc_t* md;
  NCCLCHECK(ncclCalloc(&md, 1));
  md->gdrDevMem = devMem;
  md->gdrMap = gdrMap;
  md->gdrMapSize = mapSize;
  md->gdrOffset = off+align;
  md->gdrMh = mh;
  *gdrHandle = md;

  *ptr = (T *)((char *)gdrMap+off);
  if (devPtr) *devPtr = (T *)(devMem+off+align);

  TRACE(NCCL_INIT, "GDRCOPY : allocated devMem %p gdrMap %p offset %lx mh %lx mapSize %zu at %p",
       md->gdrDevMem, md->gdrMap, md->gdrOffset, md->gdrMh.h, md->gdrMapSize, *ptr);

  return ncclSuccess;
}

template <typename T>
static ncclResult_t ncclGdrCudaCopy(void *gdrHandle, T* dst, T* src, size_t nelem) {
  gdr_mem_desc_t *md = (gdr_mem_desc_t*)gdrHandle;
  NCCLCHECK(wrap_gdr_copy_to_mapping(md->gdrMh, dst, src, nelem*ncclSizeOfT<T>()));
  return ncclSuccess;
}

static ncclResult_t ncclGdrCudaRead(void* gdrHandle, void* dst, const void* src, size_t size) {
  gdr_mem_desc_t *md = (gdr_mem_desc_t*)gdrHandle;
  return wrap_gdr_copy_from_mapping(md->gdrMh, dst, src, size);
}

static ncclResult_t ncclGdrCudaFree(void* gdrHandle, struct ncclMemManager* manager) {
  gdr_mem_desc_t *md = (gdr_mem_desc_t*)gdrHandle;
  NCCLCHECK(wrap_gdr_unmap(ncclGdrCopy, md->gdrMh, md->gdrMap, md->gdrMapSize));
  NCCLCHECK(wrap_gdr_unpin_buffer(ncclGdrCopy, md->gdrMh));
  NCCLCHECK(ncclCudaFree(md->gdrDevMem, manager));
  free(md);

  return ncclSuccess;
}

// Helper: Allocate memory accessible from CPU (either GDR or host memory)
template <typename T>
static ncclResult_t allocMemCPUAccessible(T **ptr, T **devPtr, size_t nelem, int host_flags,
                                          void **gdrHandle, struct ncclMemManager* manager, bool forceHost = false) {
  if (ncclGdrCopy && !forceHost) {
    NCCLCHECK(ncclGdrCudaCalloc(ptr, devPtr, nelem, gdrHandle, manager));
  } else {
    NCCLCHECK(ncclCuMemHostAlloc((void **)ptr, NULL, nelem * sizeof(T)));
    memset((void *)*ptr, 0, nelem * sizeof(T));
    *devPtr = *ptr;
    if (gdrHandle) *gdrHandle = NULL;  // Mark as host allocated by nulling GDR handle
  }
  return ncclSuccess;
}

// Helper: Free memory allocated by allocMemCPUAccessible
template <typename T>
static ncclResult_t freeMemCPUAccessible(T *ptr, void *gdrHandle, struct ncclMemManager* manager) {
  if (gdrHandle != NULL) {
    // If a GDR handle exists, it was GDR memory
    NCCLCHECK(ncclGdrCudaFree(gdrHandle, manager));
  } else {
    // Otherwise, it was host memory (or GDR was off)
    NCCLCHECK(ncclCuMemHostFree(ptr));
  }
  return ncclSuccess;
}

#endif // End include guard
