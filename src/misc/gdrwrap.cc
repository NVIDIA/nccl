/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "gdrwrap.h"
#include <mutex>

#ifndef GDR_DIRECT
#include "core.h"
#include "os.h"

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
  static void* gdrhandle = NULL;
  void* tmp;
  void** cast;

  gdrhandle = ncclOsDlopen(GDRAPI_LIBNAME, NCCL_OS_DL_NOW);
  if (!gdrhandle) {
    WARN("Failed to open %s - %s", GDRAPI_LIBNAME, ncclOsDlerror());
    goto teardown;
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

  initResult = ncclSuccess;
  return;

teardown:
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

  if (gdrhandle != NULL) ncclOsDlclose(gdrhandle);
  initResult = ncclSystemError;
  return;
}

ncclResult_t wrap_gdr_symbols(void) {
  std::call_once(initOnceFlag, initOnceFunc);
  return initResult;
}

gdr_t wrap_gdr_open(void) {
  if (gdr_internal_open == NULL) {
    WARN("GDRCOPY lib wrapper not initialized.");
    return NULL;
  }
  return gdr_internal_open();
}

ncclResult_t wrap_gdr_close(gdr_t g) {
  if (gdr_internal_close == NULL) {
    WARN("GDRCOPY lib wrapper not initialized.");
    return ncclInternalError;
  }
  int ret = gdr_internal_close(g);
  if (ret != 0) {
    WARN("gdr_close() failed: %d", ret);
    return ncclSystemError;
  }
  return ncclSuccess;
}

ncclResult_t wrap_gdr_pin_buffer(gdr_t g, unsigned long addr, size_t size, uint64_t p2p_token, uint32_t va_space,
                                 gdr_mh_t* handle) {
  if (gdr_internal_pin_buffer == NULL) {
    WARN("GDRCOPY lib wrapper not initialized.");
    return ncclInternalError;
  }
  int ret;
  GDRLOCKCALL(gdr_internal_pin_buffer(g, addr, size, p2p_token, va_space, handle), ret);
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
    if (gdr_internal_pin_buffer_v2 == NULL || gdr_internal_runtime_get_version == NULL) return;
    int major, minor;
    gdr_internal_runtime_get_version(&major, &minor);
    available = (major > 2 || (major == 2 && minor >= 5));
  });
  return available;
}

ncclResult_t wrap_gdr_pin_buffer_v2(gdr_t g, unsigned long addr, size_t size, uint32_t flags, gdr_mh_t* handle) {
  if (!ncclGdrPinV2Available()) {
    WARN("gdr_pin_buffer_v2 not available; GDRCopy >= 2.5 required");
    return ncclInternalError;
  }
  int ret;
  GDRLOCKCALL(gdr_internal_pin_buffer_v2(g, addr, size, flags, handle), ret);
  if (ret != 0) {
    WARN("gdr_pin_buffer_v2(addr %lx, size %zu, flags %u) failed: %d", addr, size, flags, ret);
    return ncclSystemError;
  }
  return ncclSuccess;
}

ncclResult_t wrap_gdr_unpin_buffer(gdr_t g, gdr_mh_t handle) {
  if (gdr_internal_unpin_buffer == NULL) {
    WARN("GDRCOPY lib wrapper not initialized.");
    return ncclInternalError;
  }
  int ret;
  GDRLOCKCALL(gdr_internal_unpin_buffer(g, handle), ret);
  if (ret != 0) {
    WARN("gdr_unpin_buffer(handle %lx) failed: %d", handle.h, ret);
    return ncclSystemError;
  }
  return ncclSuccess;
}

ncclResult_t wrap_gdr_get_info(gdr_t g, gdr_mh_t handle, gdr_info_t* info) {
  if (gdr_internal_get_info == NULL) {
    WARN("GDRCOPY lib wrapper not initialized.");
    return ncclInternalError;
  }
  int ret;
  int (*getInfo)(gdr_t, gdr_mh_t, gdr_info_t*) =
    gdr_internal_get_info_v2 != NULL ? gdr_internal_get_info_v2 : gdr_internal_get_info;
  GDRLOCKCALL(getInfo(g, handle, info), ret);
  if (ret != 0) {
    WARN("%s(handle %lx) failed: %d", gdr_internal_get_info_v2 != NULL ? "gdr_get_info_v2" : "gdr_get_info", handle.h,
         ret);
    return ncclSystemError;
  }
  return ncclSuccess;
}

ncclResult_t wrap_gdr_map(gdr_t g, gdr_mh_t handle, void** va, size_t size) {
  if (gdr_internal_map == NULL) {
    WARN("GDRCOPY lib wrapper not initialized.");
    return ncclInternalError;
  }
  int ret;
  GDRLOCKCALL(gdr_internal_map(g, handle, va, size), ret);
  if (ret != 0) {
    WARN("gdr_map(handle %lx, size %zu) failed: %d", handle.h, size, ret);
    return ncclSystemError;
  }
  return ncclSuccess;
}

ncclResult_t wrap_gdr_unmap(gdr_t g, gdr_mh_t handle, void* va, size_t size) {
  if (gdr_internal_unmap == NULL) {
    WARN("GDRCOPY lib wrapper not initialized.");
    return ncclInternalError;
  }
  int ret;
  GDRLOCKCALL(gdr_internal_unmap(g, handle, va, size), ret);
  if (ret != 0) {
    WARN("gdr_unmap(handle %lx, va %p, size %zu) failed: %d", handle.h, va, size, ret);
    return ncclSystemError;
  }
  return ncclSuccess;
}

ncclResult_t wrap_gdr_runtime_get_version(int* major, int* minor) {
  if (gdr_internal_runtime_get_version == NULL) {
    WARN("GDRCOPY lib wrapper not initialized.");
    return ncclInternalError;
  }
  gdr_internal_runtime_get_version(major, minor);
  gdrRuntimeMajor = *major;
  gdrRuntimeMinor = *minor;
  return ncclSuccess;
}

ncclResult_t wrap_gdr_driver_get_version(gdr_t g, int* major, int* minor) {
  if (gdr_internal_driver_get_version == NULL) {
    WARN("GDRCOPY lib wrapper not initialized.");
    return ncclInternalError;
  }
  if (gdrRuntimeMajor > 2 || (gdrRuntimeMajor == 2 && gdrRuntimeMinor >= 6)) {
    int ret;
    GDRLOCKCALL(gdr_internal_driver_get_version_checked(g, major, minor), ret);
    if (ret != 0) {
      WARN("gdr_driver_get_version() failed: %d", ret);
      return ncclSystemError;
    }
  } else {
    std::lock_guard<std::mutex> lock(getGdrMutex());
    gdr_internal_driver_get_version(g, major, minor);
  }
  return ncclSuccess;
}

ncclResult_t wrap_gdr_get_attribute(gdr_t g, gdr_attr_t attr, int* v) {
  if (gdr_internal_get_attribute == NULL) {
    INFO(NCCL_INIT, "gdr_get_attribute() not available");
    return ncclInternalError;
  }
  int ret;
  GDRLOCKCALL(gdr_internal_get_attribute(g, attr, v), ret);
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
  if (gdr_internal_copy_to_mapping == NULL) {
    WARN("GDRCOPY lib wrapper not initialized.");
    return ncclInternalError;
  }
  int ret;
  GDRLOCKCALL(gdr_internal_copy_to_mapping(handle, map_d_ptr, h_ptr, size), ret);
  if (ret != 0) {
    WARN("gdr_copy_to_mapping(handle %lx, map_d_ptr %p, h_ptr %p, size %zu) failed: %d", handle.h, map_d_ptr, h_ptr,
         size, ret);
    return ncclSystemError;
  }
  return ncclSuccess;
}

ncclResult_t wrap_gdr_copy_from_mapping(gdr_mh_t handle, void* h_ptr, const void* map_d_ptr, size_t size) {
  if (gdr_internal_copy_from_mapping == NULL) {
    WARN("GDRCOPY lib wrapper not initialized.");
    return ncclInternalError;
  }
  int ret;
  GDRLOCKCALL(gdr_internal_copy_from_mapping(handle, h_ptr, map_d_ptr, size), ret);
  if (ret != 0) {
    WARN("gdr_copy_from_mapping(handle %lx, h_ptr %p, map_d_ptr %p, size %zu) failed: %d", handle.h, h_ptr, map_d_ptr,
         size, ret);
    return ncclSystemError;
  }
  return ncclSuccess;
}

#endif /* !GDR_DIRECT */
