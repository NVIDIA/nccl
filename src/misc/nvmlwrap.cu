/*************************************************************************
 * Copyright (c) 2015-2018, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "nvmlwrap.h"

#ifndef NVML_DIRECT
#include <dlfcn.h>
#include "core.h"

static enum { nvmlUninitialized, nvmlInitializing, nvmlInitialized, nvmlError } nvmlState = nvmlUninitialized;

static nvmlReturn_t (*nvmlInternalInit)(void);
static nvmlReturn_t (*nvmlInternalShutdown)(void);
static nvmlReturn_t (*nvmlInternalDeviceGetHandleByPciBusId)(const char* pciBusId, nvmlDevice_t* device);
static nvmlReturn_t (*nvmlInternalDeviceGetIndex)(nvmlDevice_t device, unsigned* index);
static nvmlReturn_t (*nvmlInternalDeviceSetCpuAffinity)(nvmlDevice_t device);
static nvmlReturn_t (*nvmlInternalDeviceClearCpuAffinity)(nvmlDevice_t device);
static const char* (*nvmlInternalErrorString)(nvmlReturn_t r);
static nvmlReturn_t (*nvmlInternalDeviceGetNvLinkState)(nvmlDevice_t device, unsigned int link, nvmlEnableState_t *isActive);
static nvmlReturn_t (*nvmlInternalDeviceGetPciInfo)(nvmlDevice_t device, nvmlPciInfo_t* pci);
static nvmlReturn_t (*nvmlInternalDeviceGetNvLinkRemotePciInfo)(nvmlDevice_t device, unsigned int link, nvmlPciInfo_t *pci);
static nvmlReturn_t (*nvmlInternalDeviceGetNvLinkCapability)(nvmlDevice_t device, unsigned int link,
    nvmlNvLinkCapability_t capability, unsigned int *capResult);

ncclResult_t wrapNvmlSymbols(void) {
  if (nvmlState == nvmlInitialized)
    return ncclSuccess;
  if (nvmlState == nvmlError)
    return ncclSystemError;

  if (__sync_bool_compare_and_swap(&nvmlState, nvmlUninitialized, nvmlInitializing) == false) {
    // Another thread raced in front of us. Wait for it to be done.
    while (nvmlState == nvmlInitializing) pthread_yield();
    return (nvmlState == nvmlInitialized) ? ncclSuccess : ncclSystemError;
  }

  static void* nvmlhandle = NULL;
  void* tmp;
  void** cast;

  nvmlhandle=dlopen("libnvidia-ml.so.1", RTLD_NOW);
  if (!nvmlhandle) {
    WARN("Failed to open libnvidia-ml.so.1");
    goto teardown;
  }

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

  LOAD_SYM(nvmlhandle, "nvmlInit", nvmlInternalInit);
  LOAD_SYM(nvmlhandle, "nvmlShutdown", nvmlInternalShutdown);
  LOAD_SYM(nvmlhandle, "nvmlDeviceGetHandleByPciBusId", nvmlInternalDeviceGetHandleByPciBusId);
  LOAD_SYM(nvmlhandle, "nvmlDeviceGetIndex", nvmlInternalDeviceGetIndex);
  LOAD_SYM(nvmlhandle, "nvmlDeviceSetCpuAffinity", nvmlInternalDeviceSetCpuAffinity);
  LOAD_SYM(nvmlhandle, "nvmlDeviceClearCpuAffinity", nvmlInternalDeviceClearCpuAffinity);
  LOAD_SYM(nvmlhandle, "nvmlErrorString", nvmlInternalErrorString);
  LOAD_SYM(nvmlhandle, "nvmlDeviceGetPciInfo", nvmlInternalDeviceGetPciInfo);
  LOAD_SYM_OPTIONAL(nvmlhandle, "nvmlDeviceGetNvLinkState", nvmlInternalDeviceGetNvLinkState);
  LOAD_SYM_OPTIONAL(nvmlhandle, "nvmlDeviceGetNvLinkRemotePciInfo", nvmlInternalDeviceGetNvLinkRemotePciInfo);
  LOAD_SYM_OPTIONAL(nvmlhandle, "nvmlDeviceGetNvLinkCapability", nvmlInternalDeviceGetNvLinkCapability);

  nvmlState = nvmlInitialized;
  return ncclSuccess;

teardown:
  nvmlInternalInit = NULL;
  nvmlInternalShutdown = NULL;
  nvmlInternalDeviceGetHandleByPciBusId = NULL;
  nvmlInternalDeviceGetIndex = NULL;
  nvmlInternalDeviceSetCpuAffinity = NULL;
  nvmlInternalDeviceClearCpuAffinity = NULL;
  nvmlInternalDeviceGetPciInfo = NULL;
  nvmlInternalDeviceGetNvLinkState = NULL;
  nvmlInternalDeviceGetNvLinkRemotePciInfo = NULL;
  nvmlInternalDeviceGetNvLinkCapability = NULL;

  if (nvmlhandle != NULL) dlclose(nvmlhandle);
  nvmlState = nvmlError;
  return ncclSystemError;
}


ncclResult_t wrapNvmlInit(void) {
  if (nvmlInternalInit == NULL) {
    WARN("lib wrapper not initialized.");
    return ncclInternalError;
  }
  nvmlReturn_t ret = nvmlInternalInit();
  if (ret != NVML_SUCCESS) {
    WARN("nvmlInit() failed: %s",
        nvmlInternalErrorString(ret));
    return ncclSystemError;
  }
  return ncclSuccess;
}

ncclResult_t wrapNvmlShutdown(void) {
  if (nvmlInternalShutdown == NULL) {
    WARN("lib wrapper not initialized.");
    return ncclInternalError;
  }
  nvmlReturn_t ret = nvmlInternalShutdown();
  if (ret != NVML_SUCCESS) {
    WARN("nvmlShutdown() failed: %s ",
        nvmlInternalErrorString(ret));
    return ncclSystemError;
  }
  return ncclSuccess;
}

ncclResult_t wrapNvmlDeviceGetHandleByPciBusId(const char* pciBusId, nvmlDevice_t* device) {
  if (nvmlInternalDeviceGetHandleByPciBusId == NULL) {
    WARN("lib wrapper not initialized.");
    return ncclInternalError;
  }
  nvmlReturn_t ret = nvmlInternalDeviceGetHandleByPciBusId(pciBusId, device);
  if (ret != NVML_SUCCESS) {
    WARN("nvmlDeviceGetHandleByPciBusId() failed: %s ",
        nvmlInternalErrorString(ret));
    return ncclSystemError;
  }
  return ncclSuccess;
}

ncclResult_t wrapNvmlDeviceGetIndex(nvmlDevice_t device, unsigned* index) {
  if (nvmlInternalDeviceGetIndex == NULL) {
    WARN("lib wrapper not initialized.");
    return ncclInternalError;
  }
  nvmlReturn_t ret = nvmlInternalDeviceGetIndex(device, index);
  if (ret != NVML_SUCCESS) {
    WARN("nvmlDeviceGetIndex() failed: %s ",
        nvmlInternalErrorString(ret));
    return ncclSystemError;
  }
  return ncclSuccess;
}

ncclResult_t wrapNvmlDeviceSetCpuAffinity(nvmlDevice_t device) {
  if (nvmlInternalDeviceSetCpuAffinity == NULL) {
    WARN("lib wrapper not initialized.");
    return ncclInternalError;
  }
  // Workaround : it seems SetCpuAffinity is not thread safe.
  static pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;
  pthread_mutex_lock(&lock);
  nvmlReturn_t ret = nvmlInternalDeviceSetCpuAffinity(device);
  pthread_mutex_unlock(&lock);
  if (ret != NVML_SUCCESS) {
    WARN("nvmlDeviceSetCpuAffinity() failed: %s ",
        nvmlInternalErrorString(ret));
    return ncclSystemError;
  }
  return ncclSuccess;
}

ncclResult_t wrapNvmlDeviceClearCpuAffinity(nvmlDevice_t device) {
  if (nvmlInternalInit == NULL) {
    WARN("lib wrapper not initialized.");
    return ncclInternalError;
  }
  nvmlReturn_t ret = nvmlInternalDeviceClearCpuAffinity(device);
  if (ret != NVML_SUCCESS) {
    WARN("nvmlDeviceClearCpuAffinity() failed: %s ",
        nvmlInternalErrorString(ret));
    return ncclSystemError;
  }
  return ncclSuccess;
}

ncclResult_t wrapNvmlDeviceGetPciInfo(nvmlDevice_t device, nvmlPciInfo_t* pci) {
  if (nvmlInternalDeviceGetPciInfo == NULL) {
    WARN("lib wrapper not initialized.");
    return ncclInternalError;
  }
  nvmlReturn_t ret = nvmlInternalDeviceGetPciInfo(device, pci);
  if (ret != NVML_SUCCESS) {
    WARN("nvmlDeviceGetPciInfo() failed: %s ",
        nvmlInternalErrorString(ret));
    return ncclSystemError;
  }
  return ncclSuccess;
}

ncclResult_t wrapNvmlDeviceGetNvLinkState(nvmlDevice_t device, unsigned int link, nvmlEnableState_t *isActive) {
  if (nvmlInternalDeviceGetNvLinkState == NULL) {
    /* Do not warn, this symbol is optional. */
    return ncclInternalError;
  }
  nvmlReturn_t ret = nvmlInternalDeviceGetNvLinkState(device, link, isActive);
  if (ret != NVML_SUCCESS) {
    INFO(NCCL_INIT,"nvmlDeviceGetNvLinkState() failed: %s ",
        nvmlInternalErrorString(ret));
    return ncclSystemError;
  }
  return ncclSuccess;
}

ncclResult_t wrapNvmlDeviceGetNvLinkRemotePciInfo(nvmlDevice_t device, unsigned int link, nvmlPciInfo_t *pci) {
  if (nvmlInternalDeviceGetNvLinkRemotePciInfo == NULL) {
    /* Do not warn, this symbol is optional. */
    return ncclInternalError;
  }
  nvmlReturn_t ret = nvmlInternalDeviceGetNvLinkRemotePciInfo(device, link, pci);
  if (ret != NVML_SUCCESS) {
    if (ret != NVML_ERROR_NOT_SUPPORTED)
      INFO(NCCL_INIT,"nvmlDeviceGetNvLinkRemotePciInfo() failed: %s ",
          nvmlInternalErrorString(ret));
    return ncclSystemError;
  }
  return ncclSuccess;
}

ncclResult_t wrapNvmlDeviceGetNvLinkCapability(nvmlDevice_t device, unsigned int link,
    nvmlNvLinkCapability_t capability, unsigned int *capResult) {
  if (nvmlInternalDeviceGetNvLinkCapability == NULL) {
    /* Do not warn, this symbol is optional. */
    return ncclInternalError;
  }
  nvmlReturn_t ret = nvmlInternalDeviceGetNvLinkCapability(device, link, capability, capResult);
  if (ret != NVML_SUCCESS) {
    if (ret != NVML_ERROR_NOT_SUPPORTED)
      INFO(NCCL_INIT,"nvmlDeviceGetNvLinkCapability() failed: %s ",
          nvmlInternalErrorString(ret));
    return ncclSystemError;
  }
  return ncclSuccess;
}
#endif
