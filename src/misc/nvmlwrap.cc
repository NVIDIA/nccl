/*************************************************************************
 * Copyright (c) 2015-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "nvmlwrap.h"
#include "checks.h"
#include "debug.h"

#include <initializer_list>
#include <memory>
#include <mutex>

int ncclNvmlDeviceCount = 0;
ncclNvmlDeviceInfo ncclNvmlDevices[ncclNvmlMaxDevices];
ncclNvmlDevicePairInfo ncclNvmlDevicePairs[ncclNvmlMaxDevices][ncclNvmlMaxDevices];

#if NCCL_NVML_DIRECT
  #define NCCL_NVML_FN(name, rettype, arglist) constexpr rettype(*pfn_##name)arglist = name;
#else
  #include <dlfcn.h>
  #define NCCL_NVML_FN(name, rettype, arglist) rettype(*pfn_##name)arglist = nullptr;
#endif

namespace {
  NCCL_NVML_FN(nvmlInit, nvmlReturn_t, ())
  NCCL_NVML_FN(nvmlInit_v2, nvmlReturn_t, ())
  NCCL_NVML_FN(nvmlShutdown, nvmlReturn_t, ())
  NCCL_NVML_FN(nvmlDeviceGetCount, nvmlReturn_t, (unsigned int*))
  NCCL_NVML_FN(nvmlDeviceGetCount_v2, nvmlReturn_t, (unsigned int*))
  NCCL_NVML_FN(nvmlDeviceGetHandleByPciBusId, nvmlReturn_t, (const char* pciBusId, nvmlDevice_t* device))
  NCCL_NVML_FN(nvmlDeviceGetHandleByIndex, nvmlReturn_t, (unsigned int index, nvmlDevice_t *device))
  NCCL_NVML_FN(nvmlDeviceGetIndex, nvmlReturn_t, (nvmlDevice_t device, unsigned* index))
  NCCL_NVML_FN(nvmlErrorString, char const*, (nvmlReturn_t r))
  NCCL_NVML_FN(nvmlDeviceGetNvLinkState, nvmlReturn_t, (nvmlDevice_t device, unsigned int link, nvmlEnableState_t *isActive))
  NCCL_NVML_FN(nvmlDeviceGetNvLinkRemotePciInfo, nvmlReturn_t, (nvmlDevice_t device, unsigned int link, nvmlPciInfo_t *pci))
  NCCL_NVML_FN(nvmlDeviceGetNvLinkCapability, nvmlReturn_t, (nvmlDevice_t device, unsigned int link, nvmlNvLinkCapability_t capability, unsigned int *capResult))
  NCCL_NVML_FN(nvmlDeviceGetCudaComputeCapability, nvmlReturn_t, (nvmlDevice_t device, int* major, int* minor))
  NCCL_NVML_FN(nvmlDeviceGetP2PStatus, nvmlReturn_t, (nvmlDevice_t device1, nvmlDevice_t device2, nvmlGpuP2PCapsIndex_t p2pIndex, nvmlGpuP2PStatus_t* p2pStatus))
  NCCL_NVML_FN(nvmlDeviceGetFieldValues, nvmlReturn_t, (nvmlDevice_t device, int valuesCount, nvmlFieldValue_t *values))

  std::mutex lock; // NVML has had some thread safety bugs
  bool initialized = false;
  thread_local bool threadInitialized = false;
  ncclResult_t initResult;
}

ncclResult_t ncclNvmlEnsureInitialized() {
  // Optimization to avoid repeatedly grabbing the lock when we only want to
  // read from the global tables.
  if (threadInitialized) return initResult;
  threadInitialized = true;

  std::lock_guard<std::mutex> locked(lock);

  if (initialized) return initResult;
  initialized = true;

  #if !NCCL_NVML_DIRECT
  if (pfn_nvmlInit == nullptr) {
    void *libhandle = dlopen("libnvidia-ml.so.1", RTLD_NOW);
    if (libhandle == nullptr) {
      WARN("Failed to open libnvidia-ml.so.1");
      initResult = ncclSystemError;
      return initResult;
    }

    struct Symbol { void **ppfn; char const *name; };
    std::initializer_list<Symbol> symbols = {
      {(void**)&pfn_nvmlInit, "nvmlInit"},
      {(void**)&pfn_nvmlInit_v2, "nvmlInit_v2"},
      {(void**)&pfn_nvmlShutdown, "nvmlShutdown"},
      {(void**)&pfn_nvmlDeviceGetCount, "nvmlDeviceGetCount"},
      {(void**)&pfn_nvmlDeviceGetCount_v2, "nvmlDeviceGetCount_v2"},
      {(void**)&pfn_nvmlDeviceGetHandleByPciBusId, "nvmlDeviceGetHandleByPciBusId"},
      {(void**)&pfn_nvmlDeviceGetHandleByIndex, "nvmlDeviceGetHandleByIndex"},
      {(void**)&pfn_nvmlDeviceGetIndex, "nvmlDeviceGetIndex"},
      {(void**)&pfn_nvmlErrorString, "nvmlErrorString"},
      {(void**)&pfn_nvmlDeviceGetNvLinkState, "nvmlDeviceGetNvLinkState"},
      {(void**)&pfn_nvmlDeviceGetNvLinkRemotePciInfo, "nvmlDeviceGetNvLinkRemotePciInfo"},
      {(void**)&pfn_nvmlDeviceGetNvLinkCapability, "nvmlDeviceGetNvLinkCapability"},
      {(void**)&pfn_nvmlDeviceGetCudaComputeCapability, "nvmlDeviceGetCudaComputeCapability"},
      {(void**)&pfn_nvmlDeviceGetP2PStatus, "nvmlDeviceGetP2PStatus"},
      {(void**)&pfn_nvmlDeviceGetFieldValues, "nvmlDeviceGetFieldValues"}
    };
    for(Symbol sym: symbols) {
      *sym.ppfn = dlsym(libhandle, sym.name);
    }
  }
  #endif

  #if NCCL_NVML_DIRECT
    bool have_v2 = true;
  #else
    bool have_v2 = pfn_nvmlInit_v2 != nullptr; // if this compare is done in the NCCL_NVML_DIRECT=1 case then GCC warns about it never being null
  #endif
  nvmlReturn_t res1 = (have_v2 ? pfn_nvmlInit_v2 : pfn_nvmlInit)();
  if (res1 != NVML_SUCCESS) {
    WARN("nvmlInit%s() failed: %s", have_v2 ? "_v2" : "", pfn_nvmlErrorString(res1));
    initResult = ncclSystemError;
    return initResult;
  }

  unsigned int ndev;
  res1 = (have_v2 ? pfn_nvmlDeviceGetCount_v2 : pfn_nvmlDeviceGetCount)(&ndev);
  if (res1 != NVML_SUCCESS) {
    WARN("nvmlDeviceGetCount%s() failed: %s", have_v2 ? "_v2" :"", pfn_nvmlErrorString(res1));
    initResult = ncclSystemError;
    return initResult;
  }

  ncclNvmlDeviceCount = int(ndev);
  if (ncclNvmlMaxDevices < ncclNvmlDeviceCount) {
    WARN("nvmlDeviceGetCount() reported more devices (%d) than the internal maximum (ncclNvmlMaxDevices=%d)", ncclNvmlDeviceCount, ncclNvmlMaxDevices);
    initResult = ncclInternalError;
    return initResult;
  }

  for(int a=0; a < ncclNvmlDeviceCount; a++) {
    res1 = pfn_nvmlDeviceGetHandleByIndex(a, &ncclNvmlDevices[a].handle);
    if (res1 != NVML_SUCCESS) {
      WARN("nvmlDeviceGetHandleByIndex(%d) failed: %s", int(a), pfn_nvmlErrorString(res1));
      initResult = ncclSystemError;
      return initResult;
    }

    res1 = pfn_nvmlDeviceGetCudaComputeCapability(ncclNvmlDevices[a].handle, &ncclNvmlDevices[a].computeCapabilityMajor, &ncclNvmlDevices[a].computeCapabilityMinor);
    if (res1 != NVML_SUCCESS) {
      WARN("nvmlDeviceGetCudaComputeCapability(%d) failed: %s", int(a), pfn_nvmlErrorString(res1));
      initResult = ncclSystemError;
      return initResult;
    }
  }

  for(int a=0; a < ncclNvmlDeviceCount; a++) {
    for(int b=0; b < ncclNvmlDeviceCount; b++) {
      nvmlDevice_t da = ncclNvmlDevices[a].handle;
      nvmlDevice_t db = ncclNvmlDevices[b].handle;

      res1 = pfn_nvmlDeviceGetP2PStatus(da, db, NVML_P2P_CAPS_INDEX_READ, &ncclNvmlDevicePairs[a][b].p2pStatusRead);
      if (res1 != NVML_SUCCESS) {
        WARN("nvmlDeviceGetP2PStatus(%d,%d,NVML_P2P_CAPS_INDEX_READ) failed: %s", a, b, pfn_nvmlErrorString(res1));
        initResult = ncclSystemError;
        return initResult;
      }

      res1 = pfn_nvmlDeviceGetP2PStatus(da, db, NVML_P2P_CAPS_INDEX_WRITE, &ncclNvmlDevicePairs[a][b].p2pStatusWrite);
      if (res1 != NVML_SUCCESS) {
        WARN("nvmlDeviceGetP2PStatus(%d,%d,NVML_P2P_CAPS_INDEX_READ) failed: %s", a, b, pfn_nvmlErrorString(res1));
        initResult = ncclSystemError;
        return initResult;
      }
    }
  }

  initResult = ncclSuccess;
  return initResult;
}

#define NVMLCHECK(name, ...) do { \
  nvmlReturn_t e44241808 = pfn_##name(__VA_ARGS__); \
  if (e44241808 != NVML_SUCCESS) { \
    WARN(#name "() failed: %s", pfn_nvmlErrorString(e44241808)); \
    return ncclSystemError; \
  } \
} while(0)

#define NVMLTRY(name, ...) do { \
  if (!NCCL_NVML_DIRECT && pfn_##name == nullptr) \
    return ncclInternalError; /* missing symbol is not a warned error */ \
  nvmlReturn_t e44241808 = pfn_##name(__VA_ARGS__); \
  if (e44241808 != NVML_SUCCESS) { \
    if (e44241808 != NVML_ERROR_NOT_SUPPORTED) \
      INFO(NCCL_INIT, #name "() failed: %s", pfn_nvmlErrorString(e44241808)); \
    return ncclSystemError; \
  } \
} while(0)

ncclResult_t ncclNvmlDeviceGetHandleByPciBusId(const char* pciBusId, nvmlDevice_t* device) {
  NCCLCHECK(ncclNvmlEnsureInitialized());
  std::lock_guard<std::mutex> locked(lock);
  NVMLCHECK(nvmlDeviceGetHandleByPciBusId, pciBusId, device);
  return ncclSuccess;
}

ncclResult_t ncclNvmlDeviceGetHandleByIndex(unsigned int index, nvmlDevice_t *device) {
  NCCLCHECK(ncclNvmlEnsureInitialized());
  *device = ncclNvmlDevices[index].handle;
  return ncclSuccess;
}

ncclResult_t ncclNvmlDeviceGetIndex(nvmlDevice_t device, unsigned* index) {
  NCCLCHECK(ncclNvmlEnsureInitialized());
  for (int d=0; d < ncclNvmlDeviceCount; d++) {
    if (ncclNvmlDevices[d].handle == device) {
      *index = d;
      return ncclSuccess;
    }
  }
  return ncclInvalidArgument;
}

ncclResult_t ncclNvmlDeviceGetNvLinkState(nvmlDevice_t device, unsigned int link, nvmlEnableState_t *isActive) {
  NCCLCHECK(ncclNvmlEnsureInitialized());
  std::lock_guard<std::mutex> locked(lock);
  NVMLTRY(nvmlDeviceGetNvLinkState, device, link, isActive);
  return ncclSuccess;
}

ncclResult_t ncclNvmlDeviceGetNvLinkRemotePciInfo(nvmlDevice_t device, unsigned int link, nvmlPciInfo_t *pci) {
  NCCLCHECK(ncclNvmlEnsureInitialized());
  std::lock_guard<std::mutex> locked(lock);
  NVMLTRY(nvmlDeviceGetNvLinkRemotePciInfo, device, link, pci);
  return ncclSuccess;
}

ncclResult_t ncclNvmlDeviceGetNvLinkCapability(
    nvmlDevice_t device, unsigned int link, nvmlNvLinkCapability_t capability,
    unsigned int *capResult
  ) {
  NCCLCHECK(ncclNvmlEnsureInitialized());
  std::lock_guard<std::mutex> locked(lock);
  NVMLTRY(nvmlDeviceGetNvLinkCapability, device, link, capability, capResult);
  return ncclSuccess;
}

ncclResult_t ncclNvmlDeviceGetCudaComputeCapability(nvmlDevice_t device, int* major, int* minor) {
  NCCLCHECK(ncclNvmlEnsureInitialized());

  for(int d=0; d < ncclNvmlDeviceCount; d++) {
    if(device == ncclNvmlDevices[d].handle) {
      *major = ncclNvmlDevices[d].computeCapabilityMajor;
      *minor = ncclNvmlDevices[d].computeCapabilityMinor;
      return ncclSuccess;
    }
  }
  return ncclInvalidArgument;
}

ncclResult_t ncclNvmlDeviceGetP2PStatus(
    nvmlDevice_t device1, nvmlDevice_t device2, nvmlGpuP2PCapsIndex_t p2pIndex,
    nvmlGpuP2PStatus_t* p2pStatus
  ) {
  NCCLCHECK(ncclNvmlEnsureInitialized());

  if (p2pIndex == NVML_P2P_CAPS_INDEX_READ || p2pIndex == NVML_P2P_CAPS_INDEX_WRITE) {
    int a = -1, b = -1;
    for(int d=0; d < ncclNvmlDeviceCount; d++) {
      if(device1 == ncclNvmlDevices[d].handle) a = d;
      if(device2 == ncclNvmlDevices[d].handle) b = d;
    }
    if (a == -1 || b == -1) return ncclInvalidArgument;
    if (p2pIndex == NVML_P2P_CAPS_INDEX_READ)
      *p2pStatus = ncclNvmlDevicePairs[a][b].p2pStatusRead;
    else
      *p2pStatus = ncclNvmlDevicePairs[a][b].p2pStatusWrite;
  }
  else {
    std::lock_guard<std::mutex> locked(lock);
    NVMLCHECK(nvmlDeviceGetP2PStatus, device1, device2, p2pIndex, p2pStatus);
  }
  return ncclSuccess;
}

ncclResult_t ncclNvmlDeviceGetFieldValues(nvmlDevice_t device, int valuesCount, nvmlFieldValue_t *values) {
  NCCLCHECK(ncclNvmlEnsureInitialized());
  std::lock_guard<std::mutex> locked(lock);
  NVMLTRY(nvmlDeviceGetFieldValues, device, valuesCount, values);
  return ncclSuccess;
}
