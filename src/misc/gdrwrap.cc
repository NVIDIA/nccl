/*************************************************************************
 * Copyright (c) 2020-2021, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "gdrwrap.h"

#ifndef GDR_DIRECT
#include "core.h"

/* Function pointers assigned from dlopen() */
static gdr_t (*gdr_internal_open)(void);
static int (*gdr_internal_close)(gdr_t g);
static int (*gdr_internal_pin_buffer)(gdr_t g, unsigned long addr, size_t size, uint64_t p2p_token, uint32_t va_space, gdr_mh_t *handle);
static int (*gdr_internal_unpin_buffer)(gdr_t g, gdr_mh_t handle);
static int (*gdr_internal_get_info)(gdr_t g, gdr_mh_t handle, gdr_info_t *info);
static int (*gdr_internal_map)(gdr_t g, gdr_mh_t handle, void **va, size_t size);
static int (*gdr_internal_unmap)(gdr_t g, gdr_mh_t handle, void *va, size_t size);
static void (*gdr_internal_runtime_get_version)(int *major, int *minor);
static void (*gdr_internal_driver_get_version)(gdr_t g, int *major, int *minor);
static int (*gdr_internal_copy_to_mapping)(gdr_mh_t handle, void *map_d_ptr, const void *h_ptr, size_t size);
static int (*gdr_internal_copy_from_mapping)(gdr_mh_t handle, void *h_ptr, const void *map_d_ptr, size_t size);


// Used to make the GDR library calls thread safe
pthread_mutex_t gdrLock = PTHREAD_MUTEX_INITIALIZER;

#define GDRAPI_LIBNAME "libgdrapi.so"

#define LOAD_SYM(handle, symbol, funcptr) do {         \
    cast = (void**)&funcptr;                             \
    tmp = dlsym(handle, symbol);                         \
    if (tmp == NULL) {                                   \
      WARN("dlsym failed on %s - %s", symbol, dlerror());\
      goto teardown;                                     \
    }                                                    \
    *cast = tmp;                                         \
  } while (0)

#define LOAD_SYM_OPTIONAL(handle, symbol, funcptr) do {\
    cast = (void**)&funcptr;                             \
    tmp = dlsym(handle, symbol);                         \
    if (tmp == NULL) {                                   \
      INFO(NCCL_INIT,"dlsym failed on %s, ignoring", symbol); \
    }                                                    \
    *cast = tmp;                                         \
  } while (0)

static pthread_once_t initOnceControl = PTHREAD_ONCE_INIT;
static ncclResult_t initResult;

static void initOnceFunc(void) {
  static void* gdrhandle = NULL;
  void* tmp;
  void** cast;

  gdrhandle=dlopen(GDRAPI_LIBNAME, RTLD_NOW);
  if (!gdrhandle) {
    WARN("Failed to open %s", GDRAPI_LIBNAME);
    goto teardown;
  }

  /* Load the function pointers from the DL library image */
  LOAD_SYM(gdrhandle, "gdr_open", gdr_internal_open);
  LOAD_SYM(gdrhandle, "gdr_close", gdr_internal_close);
  LOAD_SYM(gdrhandle, "gdr_pin_buffer", gdr_internal_pin_buffer);
  LOAD_SYM(gdrhandle, "gdr_unpin_buffer", gdr_internal_unpin_buffer);
  LOAD_SYM(gdrhandle, "gdr_get_info", gdr_internal_get_info);
  LOAD_SYM(gdrhandle, "gdr_map", gdr_internal_map);
  LOAD_SYM(gdrhandle, "gdr_unmap", gdr_internal_unmap);
  LOAD_SYM(gdrhandle, "gdr_runtime_get_version", gdr_internal_runtime_get_version);
  LOAD_SYM(gdrhandle, "gdr_driver_get_version", gdr_internal_driver_get_version);
  LOAD_SYM(gdrhandle, "gdr_copy_to_mapping", gdr_internal_copy_to_mapping);
  LOAD_SYM(gdrhandle, "gdr_copy_from_mapping", gdr_internal_copy_from_mapping);

  initResult = ncclSuccess;
  return;

teardown:
  gdr_internal_open = NULL;
  gdr_internal_close = NULL;
  gdr_internal_pin_buffer = NULL;
  gdr_internal_unpin_buffer = NULL;
  gdr_internal_get_info = NULL;
  gdr_internal_map = NULL;
  gdr_internal_unmap = NULL;
  gdr_internal_runtime_get_version = NULL;
  gdr_internal_driver_get_version = NULL;
  gdr_internal_copy_to_mapping = NULL;
  gdr_internal_copy_from_mapping = NULL;

  if (gdrhandle != NULL) dlclose(gdrhandle);
  initResult = ncclSystemError;
  return;
}


ncclResult_t wrap_gdr_symbols(void) {
  pthread_once(&initOnceControl, initOnceFunc);
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

ncclResult_t wrap_gdr_pin_buffer(gdr_t g, unsigned long addr, size_t size, uint64_t p2p_token, uint32_t va_space, gdr_mh_t *handle) {
  if (gdr_internal_pin_buffer == NULL) {
    WARN("GDRCOPY lib wrapper not initialized.");
    return ncclInternalError;
  }
  int ret;
  GDRLOCKCALL(gdr_internal_pin_buffer(g, addr, size, p2p_token, va_space, handle), ret);
  if (ret != 0) {
    WARN("gdr_pin_buffer(addr %lx, size %zi) failed: %d", addr, size, ret);
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

ncclResult_t wrap_gdr_get_info(gdr_t g, gdr_mh_t handle, gdr_info_t *info) {
  if (gdr_internal_get_info == NULL) {
    WARN("GDRCOPY lib wrapper not initialized.");
    return ncclInternalError;
  }
  int ret;
  GDRLOCKCALL(gdr_internal_get_info(g, handle, info), ret);
  if (ret != 0) {
    WARN("gdr_get_info(handle %lx) failed: %d", handle.h, ret);
    return ncclSystemError;
  }
  return ncclSuccess;
}

ncclResult_t wrap_gdr_map(gdr_t g, gdr_mh_t handle, void **va, size_t size) {
  if (gdr_internal_map == NULL) {
    WARN("GDRCOPY lib wrapper not initialized.");
    return ncclInternalError;
  }
  int ret;
  GDRLOCKCALL(gdr_internal_map(g, handle, va, size), ret);
  if (ret != 0) {
    WARN("gdr_map(handle %lx, size %zi) failed: %d", handle.h, size, ret);
    return ncclSystemError;
  }
  return ncclSuccess;
}

ncclResult_t wrap_gdr_unmap(gdr_t g, gdr_mh_t handle, void *va, size_t size) {
  if (gdr_internal_unmap == NULL) {
    WARN("GDRCOPY lib wrapper not initialized.");
    return ncclInternalError;
  }
  int ret;
  GDRLOCKCALL(gdr_internal_unmap(g, handle, va, size), ret);
  if (ret != 0) {
    WARN("gdr_unmap(handle %lx, va %p, size %zi) failed: %d", handle.h, va, size, ret);
    return ncclSystemError;
  }
  return ncclSuccess;
}

ncclResult_t wrap_gdr_runtime_get_version(int *major, int *minor) {
  if (gdr_internal_runtime_get_version == NULL) {
    WARN("GDRCOPY lib wrapper not initialized.");
    return ncclInternalError;
  }
  gdr_internal_runtime_get_version(major, minor);
  return ncclSuccess;
}

ncclResult_t wrap_gdr_driver_get_version(gdr_t g, int *major, int *minor) {
  if (gdr_internal_driver_get_version == NULL) {
    WARN("GDRCOPY lib wrapper not initialized.");
    return ncclInternalError;
  }
  gdr_internal_driver_get_version(g, major, minor);
  return ncclSuccess;
}

ncclResult_t wrap_gdr_copy_to_mapping(gdr_mh_t handle, void *map_d_ptr, const void *h_ptr, size_t size) {
  if (gdr_internal_copy_to_mapping == NULL) {
    WARN("GDRCOPY lib wrapper not initialized.");
    return ncclInternalError;
  }
  int ret;
  GDRLOCKCALL(gdr_internal_copy_to_mapping(handle, map_d_ptr, h_ptr, size), ret);
  if (ret != 0) {
    WARN("gdr_copy_to_mapping(handle %lx, map_d_ptr %p, h_ptr %p, size %zi) failed: %d", handle.h, map_d_ptr, h_ptr, size, ret);
    return ncclSystemError;
  }
  return ncclSuccess;
}

ncclResult_t wrap_gdr_copy_from_mapping(gdr_mh_t handle, void *h_ptr, const void *map_d_ptr, size_t size) {
  if (gdr_internal_copy_from_mapping == NULL) {
    WARN("GDRCOPY lib wrapper not initialized.");
    return ncclInternalError;
  }
  int ret;
  GDRLOCKCALL(gdr_internal_copy_from_mapping(handle, h_ptr, map_d_ptr, size), ret);
  if (ret != 0) {
    WARN("gdr_copy_from_mapping(handle %lx, h_ptr %p, map_d_ptr %p, size %zi) failed: %d", handle.h, h_ptr, map_d_ptr, size, ret);
    return ncclSystemError;
  }
  return ncclSuccess;
}

#endif /* !GDR_DIRECT */
