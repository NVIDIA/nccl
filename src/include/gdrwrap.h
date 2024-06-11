/*************************************************************************
 * Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_GDRWRAP_H_
#define NCCL_GDRWRAP_H_

#include "nccl.h"
#include "alloc.h"
#include <stdint.h> // for standard [u]intX_t types
#include <stdio.h>
#include <stdlib.h>

// These can be used if the GDR library isn't thread safe
#include <pthread.h>
extern pthread_mutex_t gdrLock;
#define GDRLOCK() pthread_mutex_lock(&gdrLock)
#define GDRUNLOCK() pthread_mutex_unlock(&gdrLock)
#define GDRLOCKCALL(cmd, ret) do {                      \
    GDRLOCK();                                          \
    ret = cmd;                                          \
    GDRUNLOCK();                                        \
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
static inline void wc_store_fence(void) { asm volatile("sync") ; }
#elif defined(__x86_64__)
#include <immintrin.h>
static inline void wc_store_fence(void) { _mm_sfence(); }
#elif defined(__aarch64__)
#ifdef __cplusplus
#include <atomic>
static inline void wc_store_fence(void) { std::atomic_thread_fence(std::memory_order_release); }
#else
#include <stdatomic.h>
static inline void wc_store_fence(void) { atomic_thread_fence(memory_order_release); }
#endif
#endif
#endif

//#define GDR_DIRECT 1
#ifdef GDR_DIRECT
// Call the GDR API library code directly rather than via
// dlopen() wrappers
#include <gdrapi.h>

static ncclResult_t wrap_gdr_symbols(void) { return ncclSuccess; }
static gdr_t wrap_gdr_open(void) { gdr_t g = gdr_open(); return g; }
static ncclResult_t wrap_gdr_close(gdr_t g) { GDRCHECK(gdr_close(g)); return ncclSuccess; }
static ncclResult_t wrap_gdr_pin_buffer(gdr_t g, unsigned long addr, size_t size, uint64_t p2p_token, uint32_t va_space, gdr_mh_t *handle) {
  GDRCHECK(gdr_pin_buffer(g, addr, size, p2p_token, va_space, handle));
  return ncclSuccess;
}
static ncclResult_t wrap_gdr_unpin_buffer(gdr_t g, gdr_mh_t handle) {
  GDRCHECK(gdr_unpin_buffer(g, handle));
  return ncclSuccess;
}
static ncclResult_t wrap_gdr_get_info(gdr_t g, gdr_mh_t handle, gdr_info_t *info) {
  GDRCHECK(gdr_get_info(g, handle, info));
  return ncclSuccess;
}
static ncclResult_t wrap_gdr_map(gdr_t g, gdr_mh_t handle, void **va, size_t size) {
  GDRCHECK(gdr_map(gdr_t g, gdr_mh_t handle, void **va, size_t size));
  return ncclSuccess;
}
static ncclResult_t wrap_gdr_unmap(gdr_t g, gdr_mh_t handle, void *va, size_t size) {
  GDRCHECK(gdr_unmap(gdr_t g, gdr_mh_t handle, void **va, size_t size));
  return ncclSuccess;
}
static void wrap_gdr_runtime_get_version(int *major, int *minor) {
  gdr_runtime_get_version(major, minor);
  return ncclSuccess;
}
static void wrap_gdr_driver_get_version(gdr_t g, int *major, int *minor) {
  gdr_driver_get_version(g, major, minor);
  return ncclSuccess;
}
static ncclResult_t wrap_gdr_copy_to_mapping(gdr_mh_t handle, void *map_d_ptr, const void *h_ptr, size_t size) {
  GDRCHECK(gdr_copy_to_mapping(handle, map_d_ptr, h_ptr, size));
  return ncclSuccess;
}
static ncclResult_t wrap_gdr_copy_from_mapping(gdr_mh_t handle, void *h_ptr, const void *map_d_ptr, size_t size) {
  GDRCHECK(gdr_copy_from_mapping(handle, h_ptr, map_d_ptr, size));
  return ncclSuccess;
}

#else
// Dynamically handle dependency the GDR API library

/* Extracted from gdrapi.h (v2.1 Nov 2020) */

#define GPU_PAGE_SHIFT   16
#define GPU_PAGE_SIZE    (1UL << GPU_PAGE_SHIFT)
#define GPU_PAGE_OFFSET  (GPU_PAGE_SIZE-1)
#define GPU_PAGE_MASK    (~GPU_PAGE_OFFSET)

struct gdr;
typedef struct gdr *gdr_t;

typedef struct gdr_mh_s {
  unsigned long h;
} gdr_mh_t;

struct gdr_info {
    uint64_t va;
    uint64_t mapped_size;
    uint32_t page_size;
    uint64_t tm_cycles;
    uint32_t cycles_per_ms;
    unsigned mapped:1;
    unsigned wc_mapping:1;
};
typedef struct gdr_info gdr_info_t;

/* End of gdrapi.h */

ncclResult_t wrap_gdr_symbols(void);

gdr_t wrap_gdr_open(void);
ncclResult_t wrap_gdr_close(gdr_t g);
ncclResult_t wrap_gdr_pin_buffer(gdr_t g, unsigned long addr, size_t size, uint64_t p2p_token, uint32_t va_space, gdr_mh_t *handle);
ncclResult_t wrap_gdr_unpin_buffer(gdr_t g, gdr_mh_t handle);
ncclResult_t wrap_gdr_get_info(gdr_t g, gdr_mh_t handle, gdr_info_t *info);
ncclResult_t wrap_gdr_map(gdr_t g, gdr_mh_t handle, void **va, size_t size);
ncclResult_t wrap_gdr_unmap(gdr_t g, gdr_mh_t handle, void *va, size_t size);
ncclResult_t wrap_gdr_runtime_get_version(int *major, int *minor);
ncclResult_t wrap_gdr_driver_get_version(gdr_t g, int *major, int *minor);
ncclResult_t wrap_gdr_copy_to_mapping(gdr_mh_t handle, void *map_d_ptr, const void *h_ptr, size_t size);
ncclResult_t wrap_gdr_copy_from_mapping(gdr_mh_t handle, void *h_ptr, const void *map_d_ptr, size_t size);

#endif // GDR_DIRECT

// Global GDR driver handle
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
  int libMajor, libMinor, drvMajor, drvMinor;
  gdr_t handle = NULL;
  // Dynamically load the GDRAPI library symbols
  if (wrap_gdr_symbols() == ncclSuccess) {
    handle = wrap_gdr_open();

    if (handle != NULL) {
      ncclResult_t res;

      // Query the version of libgdrapi
      NCCLCHECKGOTO(wrap_gdr_runtime_get_version(&libMajor, &libMinor), res, error);

      // Query the version of gdrdrv driver
      NCCLCHECKGOTO(wrap_gdr_driver_get_version(handle, &drvMajor, &drvMinor), res, error);

      // Only support GDRAPI 2.1 and later
      if (libMajor < 2 || (libMajor == 2 && libMinor < 1) || drvMajor < 2 || (drvMajor == 2 && drvMinor < 1)) {
        goto error;
      }
      else
        INFO(NCCL_INIT, "GDRCOPY enabled library %d.%d driver %d.%d", libMajor, libMinor, drvMajor, drvMinor);
    }
  }
  return handle;
error:
  if (handle != NULL) (void) wrap_gdr_close(handle);
  return NULL;
}

template <typename T>
static ncclResult_t ncclGdrCudaCalloc(T** ptr, T** devPtr, size_t nelem, void** gdrHandle) {
  gdr_info_t info;
  size_t mapSize;
  gdr_mh_t mh;
  char *devMem;
  void *gdrMap;

  mapSize = ncclSizeOfT<T>()*nelem;

  // GDRCOPY Pinned buffer has to be a minimum of a GPU_PAGE_SIZE
  ALIGN_SIZE(mapSize, GPU_PAGE_SIZE);
  // GDRCOPY Pinned buffer has to be GPU_PAGE_SIZE aligned too
  NCCLCHECK(ncclCudaCalloc(&devMem, mapSize+GPU_PAGE_SIZE-1));
  uint64_t alignedAddr = (((uint64_t) devMem) + GPU_PAGE_OFFSET) & GPU_PAGE_MASK;
  size_t align = alignedAddr - (uint64_t)devMem;

  //TRACE(NCCL_INIT, "GDRCOPY: Pin buffer 0x%lx (%p) align %zu size %zu", alignedAddr, devMem, align, mapSize);
  NCCLCHECK(wrap_gdr_pin_buffer(ncclGdrCopy, alignedAddr, mapSize, 0, 0, &mh));

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

static ncclResult_t ncclGdrCudaFree(void* gdrHandle) {
  gdr_mem_desc_t *md = (gdr_mem_desc_t*)gdrHandle;
  NCCLCHECK(wrap_gdr_unmap(ncclGdrCopy, md->gdrMh, md->gdrMap, md->gdrMapSize));
  NCCLCHECK(wrap_gdr_unpin_buffer(ncclGdrCopy, md->gdrMh));
  NCCLCHECK(ncclCudaFree(md->gdrDevMem));
  free(md);

  return ncclSuccess;
}

#endif // End include guard
