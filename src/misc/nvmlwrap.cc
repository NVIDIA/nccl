/*************************************************************************
 * Copyright (c) 2015-2019, NVIDIA CORPORATION. All rights reserved.
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
static nvmlReturn_t (*nvmlInternalDeviceGetHandleByIndex)(unsigned int index, nvmlDevice_t* device);
static const char* (*nvmlInternalErrorString)(nvmlReturn_t r);
static nvmlReturn_t (*nvmlInternalDeviceGetNvLinkState)(nvmlDevice_t device, unsigned int link, nvmlEnableState_t *isActive);
static nvmlReturn_t (*nvmlInternalDeviceGetPciInfo)(nvmlDevice_t device, nvmlPciInfo_t* pci);
static nvmlReturn_t (*nvmlInternalDeviceGetNvLinkRemotePciInfo)(nvmlDevice_t device, unsigned int link, nvmlPciInfo_t *pci);
static nvmlReturn_t (*nvmlInternalDeviceGetNvLinkCapability)(nvmlDevice_t device, unsigned int link,
    nvmlNvLinkCapability_t capability, unsigned int *capResult);
static nvmlReturn_t (*nvmlInternalDeviceGetMinorNumber)(nvmlDevice_t device, unsigned int* minorNumber);
static nvmlReturn_t (*nvmlInternalDeviceGetCudaComputeCapability)(nvmlDevice_t device, int* major, int* minor);

// Used to make the NVML library calls thread safe
pthread_mutex_t nvmlLock = PTHREAD_MUTEX_INITIALIZER;

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
  LOAD_SYM(nvmlhandle, "nvmlDeviceGetHandleByIndex", nvmlInternalDeviceGetHandleByIndex);
  LOAD_SYM(nvmlhandle, "nvmlErrorString", nvmlInternalErrorString);
  LOAD_SYM(nvmlhandle, "nvmlDeviceGetPciInfo", nvmlInternalDeviceGetPciInfo);
  LOAD_SYM(nvmlhandle, "nvmlDeviceGetMinorNumber", nvmlInternalDeviceGetMinorNumber);
  LOAD_SYM_OPTIONAL(nvmlhandle, "nvmlDeviceGetNvLinkState", nvmlInternalDeviceGetNvLinkState);
  LOAD_SYM_OPTIONAL(nvmlhandle, "nvmlDeviceGetNvLinkRemotePciInfo", nvmlInternalDeviceGetNvLinkRemotePciInfo);
  LOAD_SYM_OPTIONAL(nvmlhandle, "nvmlDeviceGetNvLinkCapability", nvmlInternalDeviceGetNvLinkCapability);
  LOAD_SYM(nvmlhandle, "nvmlDeviceGetCudaComputeCapability", nvmlInternalDeviceGetCudaComputeCapability);

  nvmlState = nvmlInitialized;
  return ncclSuccess;

teardown:
  nvmlInternalInit = NULL;
  nvmlInternalShutdown = NULL;
  nvmlInternalDeviceGetHandleByPciBusId = NULL;
  nvmlInternalDeviceGetIndex = NULL;
  nvmlInternalDeviceGetHandleByIndex = NULL;
  nvmlInternalDeviceGetPciInfo = NULL;
  nvmlInternalDeviceGetMinorNumber = NULL;
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
  nvmlReturn_t ret;
  NVMLLOCKCALL(nvmlInternalDeviceGetHandleByPciBusId(pciBusId, device), ret);
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
  nvmlReturn_t ret;
  NVMLLOCKCALL(nvmlInternalDeviceGetIndex(device, index), ret);
  if (ret != NVML_SUCCESS) {
    WARN("nvmlDeviceGetIndex() failed: %s ",
        nvmlInternalErrorString(ret));
    return ncclSystemError;
  }
  return ncclSuccess;
}

ncclResult_t wrapNvmlDeviceGetHandleByIndex(unsigned int index, nvmlDevice_t* device) {
  if (nvmlInternalDeviceGetHandleByIndex == NULL) {
    WARN("lib wrapper not initialized.");
    return ncclInternalError;
  }
  nvmlReturn_t ret;
  NVMLLOCKCALL(nvmlInternalDeviceGetHandleByIndex(index, device), ret);
  if (ret != NVML_SUCCESS) {
    WARN("nvmlDeviceGetHandleByIndex() failed: %s ",
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
  nvmlReturn_t ret;
  NVMLLOCKCALL(nvmlInternalDeviceGetPciInfo(device, pci), ret);
  if (ret != NVML_SUCCESS) {
    WARN("nvmlDeviceGetPciInfo() failed: %s ",
        nvmlInternalErrorString(ret));
    return ncclSystemError;
  }
  return ncclSuccess;
}

ncclResult_t wrapNvmlDeviceGetMinorNumber(nvmlDevice_t device, unsigned int* minorNumber) {
  if (nvmlInternalDeviceGetMinorNumber == NULL) {
    WARN("lib wrapper not initialized.");
    return ncclInternalError;
  }
  nvmlReturn_t ret;
  NVMLLOCKCALL(nvmlInternalDeviceGetMinorNumber(device, minorNumber), ret);
  if (ret != NVML_SUCCESS) {
    WARN("nvmlDeviceGetMinorNumber() failed: %s ",
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
  nvmlReturn_t ret;
  NVMLLOCKCALL(nvmlInternalDeviceGetNvLinkState(device, link, isActive), ret);
  if (ret != NVML_SUCCESS) {
    if (ret != NVML_ERROR_NOT_SUPPORTED)
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
  nvmlReturn_t ret;
  NVMLLOCKCALL(nvmlInternalDeviceGetNvLinkRemotePciInfo(device, link, pci), ret);
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
  nvmlReturn_t ret;
  NVMLLOCKCALL(nvmlInternalDeviceGetNvLinkCapability(device, link, capability, capResult), ret);
  if (ret != NVML_SUCCESS) {
    if (ret != NVML_ERROR_NOT_SUPPORTED)
      INFO(NCCL_INIT,"nvmlDeviceGetNvLinkCapability() failed: %s ",
          nvmlInternalErrorString(ret));
    return ncclSystemError;
  }
  return ncclSuccess;
}

ncclResult_t wrapNvmlDeviceGetCudaComputeCapability(nvmlDevice_t device, int* major, int* minor) {
  if (nvmlInternalDeviceGetNvLinkCapability == NULL) {
    WARN("lib wrapper not initialized.");
    return ncclInternalError;
  }
  nvmlReturn_t ret;
  NVMLLOCKCALL(nvmlInternalDeviceGetCudaComputeCapability(device, major, minor), ret);
  if (ret != NVML_SUCCESS) {
    WARN("nvmlDeviceGetCudaComputeCapability() failed: %s ",
        nvmlInternalErrorString(ret));
    return ncclSystemError;
  }
  return ncclSuccess;
}
#endif
