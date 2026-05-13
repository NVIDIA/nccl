/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "gin.h"
#include "bootstrap.h"
#include "checks.h"
#include "plugin.h"
#include "nccl_gin.h"
#include "gin/gin_host_proxy.h"

#include <string.h>
#include <errno.h>
#include <mutex>

typedef ncclGin_t* getNcclGin_t(void* ginPluginLib);

extern getNcclGin_t getNcclGin_v13;
extern getNcclGin_t getNcclGin_v14;
NCCL_PARAM(GinPluginRefCount, "GIN_PLUGIN_REF_COUNT", 0);
#define NCCL_GIN_VERSION_COUNT 2
int ncclGinVersion[NCCL_GIN_VERSION_COUNT] = {14, 13};
getNcclGin_t* getNcclGin[NCCL_GIN_VERSION_COUNT] = {getNcclGin_v14, getNcclGin_v13};

#define NCCL_GIN_NUM_INTERNAL_PLUGINS 2

typedef enum ncclGinPluginState {
  ncclGinPluginStateDisabled        = -2,       // Plugin library failed to initialize
  ncclGinPluginStateLoadFailed      = -1,       // Plugin library failed to load
  ncclGinPluginStateLoadReady       = 0,        // Plugin library is ready to be loaded
  ncclGinPluginStateInitReady       = 1,        // Plugin library is loaded and ready to be initialized
  ncclGinPluginStateEnabled         = 2,        // Plugin library is loaded and initialized
} ncclGinPluginState_t;

#define MAX_STR_LEN 255
typedef struct ginPluginLib {
  char name[MAX_STR_LEN];                       // Name of the plugin library
  void* dlHandle;                               // Handle to the plugin library
  ncclGin_t* ncclGin;                           // Pointer to the plugin structure
  int version;                                  // Version of the plugin
  ncclGinPluginState_t state;                   // State of the plugin
  int refCount;                                 // Reference count
  int physDevs;                                 // Number of physical devices
} ginPluginLib_t;

static int pluginCount = 0;
static ginPluginLib_t pluginLibs[NCCL_GIN_MAX_PLUGINS] = { 0 };
static std::mutex pluginMutex;
static std::once_flag initPluginLibsOnceFlag;

static ncclResult_t ncclGinPluginUnload(ginPluginLib_t* pluginLib) {
  if (pluginLib->dlHandle && pluginLib->refCount == 0) {
    INFO(NCCL_DESTROY|NCCL_NET, "Unloading plugin %s", pluginLib->name);
    NCCLCHECK(ncclClosePluginLib(pluginLib->dlHandle, ncclPluginTypeGin));

    // Reset fields but preserve name, to be reused when reloading
    pluginLib->dlHandle = NULL;
    pluginLib->ncclGin = NULL;
    pluginLib->state = ncclGinPluginStateLoadReady;
    pluginLib->refCount = 0;
    pluginLib->physDevs = 0;
  }
  return ncclSuccess;
}

static ncclResult_t ncclGinPluginLoad(ginPluginLib_t* pluginLib) {
  // Open library. NET plugin dlHandle should be already set.
  if (pluginLib->dlHandle == NULL) {
    pluginLib->dlHandle = ncclOpenGinPluginLib(pluginLib->name);
    if (pluginLib->dlHandle == nullptr) goto fail;
  }

  // load gin
  for (int i = 0; i < NCCL_GIN_VERSION_COUNT; i++) {
    pluginLib->version = ncclGinVersion[i];
    pluginLib->ncclGin = getNcclGin[i](pluginLib->dlHandle);
    if (pluginLib->ncclGin) break;
  }

  if (pluginLib->ncclGin == nullptr) {
    pluginLib->state = ncclGinPluginStateLoadFailed;
  } else {
    pluginLib->state = ncclGinPluginStateInitReady;
  }

  INFO(NCCL_INIT|NCCL_NET, "Successfully loaded external gin plugin %s",
       (ncclPluginLibPaths[ncclPluginTypeGin] ? ncclPluginLibPaths[ncclPluginTypeGin] : pluginLib->name));
exit:
  return ncclSuccess;
fail:
  if (pluginLib->dlHandle) {
    NCCLCHECK(ncclClosePluginLib(pluginLib->dlHandle, ncclPluginTypeGin));
  }
  pluginLib->dlHandle = nullptr;
  pluginLib->state = ncclGinPluginStateLoadFailed;
  goto exit;
}

static ncclResult_t ncclGinPluginInit(struct ncclComm* comm, ginPluginLib_t* pluginLib) {
  int ndev;
  // Init must be called for each new comm to set the right context
  if (pluginLib->state >= ncclGinPluginStateInitReady && pluginLib->ncclGin) {
    if (!pluginLib->ncclGin->init || pluginLib->ncclGin->init(&comm->ginContext, comm->commHash, ncclDebugLog) != ncclSuccess) {
      pluginLib->state = ncclGinPluginStateDisabled;
    }
  }
  if (pluginLib->state == ncclGinPluginStateInitReady && pluginLib->ncclGin) {
    if (pluginLib->ncclGin->devices(&ndev) != ncclSuccess || ndev <= 0) {
      pluginLib->state = ncclGinPluginStateDisabled;
    } else {
      pluginLib->physDevs = ndev;
      pluginLib->state = ncclGinPluginStateEnabled;
    }
  }
  return ncclSuccess;
}

static ncclResult_t ncclGinPluginAssignToComm(struct ncclComm* comm, int pluginIndex, bool* isAssigned) {
  *isAssigned = false;

  if (pluginLibs[pluginIndex].state >= ncclGinPluginStateEnabled) {
    ncclGin_t* gin = pluginLibs[pluginIndex].ncclGin;
    ncclNetProperties_t props;
    NCCLCHECK(gin->getProperties(0, &props));

    int64_t ginType = ncclParamGinType();
    bool isExternal = pluginIndex < (pluginCount - NCCL_GIN_NUM_INTERNAL_PLUGINS);

    if (ginType != -1 && props.netDeviceType != ginType) {
      INFO(NCCL_INIT|NCCL_NET, "Skipping GIN plugin %s index %d type %d: NCCL_GIN_TYPE=%ld requested",
           gin->name, pluginIndex, props.netDeviceType, ginType);
      return ncclSuccess;
    }

    if (isExternal && props.netDeviceType == NCCL_NET_DEVICE_GIN_PROXY) {
      INFO(NCCL_INIT|NCCL_NET, "Skipping external GIN proxy plugin %s index %d; using NCCL GIN proxy over RMA backend",
           gin->name, pluginIndex);
      return ncclSuccess;
    }

    INFO(NCCL_INIT|NCCL_NET, "Assigned GIN plugin %s to comm", gin->name);
    comm->sharedRes->ginState.ncclGin = gin;
    comm->sharedRes->ginState.ginVersion = pluginLibs[pluginIndex].version;
    // NOTE: The following cast is valid because ncclGinType_t variant values
    // should match NCCL_NET_DEVICE_GIN_* values from `enum ncclNetDeviceType`.
    comm->sharedRes->ginState.ginType = static_cast<ncclGinType_t>(props.netDeviceType);
    comm->ginPluginIndex = pluginIndex;

    ncclGinProperties_t ginProperties;
    NCCLCHECK(gin->getGinProperties(&ginProperties));
    comm->sharedRes->ginState.supportsStrongSignals = ginProperties.supportsStrongSignals;
    comm->sharedRes->ginState.supportsVASignals = ginProperties.supportsVASignals;
  }
  pluginLibs[pluginIndex].refCount++;
  *isAssigned = true;
  return ncclSuccess;
}

static ncclResult_t ncclGinPluginDisableOtherExternal(int pluginIndex) {
  // Only if an external plugin is enabled, disable other external plugins
  if (pluginIndex >= (pluginCount - NCCL_GIN_NUM_INTERNAL_PLUGINS)) return ncclSuccess;
  char names[MAX_STR_LEN*(NCCL_GIN_MAX_PLUGINS - NCCL_GIN_NUM_INTERNAL_PLUGINS)] = { 0 };
  for (int i = 0; i < (pluginCount - NCCL_GIN_NUM_INTERNAL_PLUGINS); i++) {
    if (i != pluginIndex) {
      // Append all disabled plugin names to a string
      snprintf(names+strlen(names), sizeof(names)-strlen(names), (strlen(names) == 0) ? "%s" : ", %s", pluginLibs[i].name);
      pluginLibs[i].state = ncclGinPluginStateDisabled;
    }
  }
  if(strlen(names) > 0) {
    INFO(NCCL_INIT|NCCL_NET, "Disabling external plugins: %s", names);
  }
  return ncclSuccess;
}

static void initPluginLibsOnceFunc() {
  char* ginPluginName = nullptr;
  const char* defaultGinPlugin = "libnccl-gin.so";
  const char* envGinPlugin = nullptr;
  char* envGinPluginList = nullptr;
  char* savePtr = nullptr;
  int pluginCounter = 0;

  memset(pluginLibs, 0, NCCL_GIN_MAX_PLUGINS * sizeof(ginPluginLib_t));
  envGinPlugin = ncclGetEnv("NCCL_GIN_PLUGIN");
  if (envGinPlugin) {
    INFO(NCCL_ENV|NCCL_NET, "NCCL_GIN_PLUGIN set by environment to %s", envGinPlugin);
    if (strcasecmp(envGinPlugin, "none") == 0)
      envGinPlugin = "";
    envGinPluginList = strdup(envGinPlugin);
    // Iterate over list until the list is empty
    ginPluginName = strtok_r(envGinPluginList, ",", &savePtr);
    while(ginPluginName) {
      // So, we can have at most( NCCL_GIN_MAX_PLUGINS - (NCCL_GIN_NUM_INTERNAL_PLUGINS)) in the NCCL_GIN_PLUGIN list
      if (pluginCounter >= (NCCL_GIN_MAX_PLUGINS - (NCCL_GIN_NUM_INTERNAL_PLUGINS))) {
        INFO(NCCL_NET|NCCL_ENV,"NCCL_GIN_PLUGIN list contains more than %d plugins, ignoring the rest", (NCCL_GIN_MAX_PLUGINS - (NCCL_GIN_NUM_INTERNAL_PLUGINS + 1)));
        break;
      }
      // need to leave space for the name + "\n"
      if ((strlen(ginPluginName)+1) <= MAX_STR_LEN) {
        pluginLibs[pluginCounter].state = ncclGinPluginStateLoadReady;
        pluginLibs[pluginCounter].refCount = ncclParamGinPluginRefCount();
        strcpy(pluginLibs[pluginCounter].name, ginPluginName);
        pluginCounter++;
      } else {
        INFO(NCCL_NET|NCCL_ENV,"NCCL_GIN_PLUGIN list contains a plugin name %s longer than %d characters, ignoring it.", ginPluginName, MAX_STR_LEN);
      }
      ginPluginName = strtok_r(nullptr, ",", &savePtr);
    }
    if (envGinPluginList) free(envGinPluginList);
  } else {
    // Add default gin plugin
    pluginLibs[pluginCounter].state = ncclGinPluginStateLoadReady;
    pluginLibs[pluginCounter].refCount = ncclParamGinPluginRefCount();
    strcpy(pluginLibs[pluginCounter++].name, defaultGinPlugin);
  }

  // Also check if the NET plugin has GIN support
  if ((pluginLibs[pluginCounter].dlHandle = ncclGetNetPluginLib(ncclPluginTypeGin)) != NULL) {
    pluginLibs[pluginCounter].state = ncclGinPluginStateLoadReady;
    pluginCounter++;
  }

  // Add internal ib plugin
  pluginLibs[pluginCounter].ncclGin = &ncclGinIbGdaki;
  pluginLibs[pluginCounter].state = ncclGinPluginStateInitReady;
  pluginLibs[pluginCounter].version = ncclGinVersion[0];
  pluginCounter++;
  // Add gin proxy as fallback
  pluginLibs[pluginCounter].ncclGin = &ncclGinProxy;
  pluginLibs[pluginCounter].state = ncclGinPluginStateInitReady;
  pluginLibs[pluginCounter].version = ncclGinProxyVersion;
  pluginCounter++;
  pluginCount = pluginCounter;
}

static ncclResult_t ncclGinPluginFinalize(struct ncclComm* comm, int pluginIndex) {
  if (pluginLibs[pluginIndex].ncclGin && pluginLibs[pluginIndex].state == ncclGinPluginStateEnabled) NCCLCHECK(pluginLibs[pluginIndex].ncclGin->finalize(comm->ginContext));
  pluginLibs[pluginIndex].refCount--;
  if (pluginIndex < (pluginCount - NCCL_GIN_NUM_INTERNAL_PLUGINS)) {
    NCCLCHECK(ncclGinPluginUnload(&pluginLibs[pluginIndex]));
  }
  return ncclSuccess;
}

ncclResult_t ncclGinInit(struct ncclComm* comm) {
  if (comm->compCap < 70) {
    /* GIN only supported for Volta and later */
    INFO(NCCL_INIT, "Compute Capability (%d) is not sufficient to enable GIN.  Require Volta (70) or newer.",comm->compCap);
    return ncclSuccess;
  }

  bool initialized = false;
  std::call_once(initPluginLibsOnceFlag, initPluginLibsOnceFunc);
  std::lock_guard<std::mutex> lock(pluginMutex);
  for (int pluginIndex = 0; pluginIndex < pluginCount; pluginIndex++) {
    if (pluginIndex < (pluginCount - NCCL_GIN_NUM_INTERNAL_PLUGINS) && pluginLibs[pluginIndex].state == ncclGinPluginStateLoadReady) {
      NCCLCHECK(ncclGinPluginLoad(&pluginLibs[pluginIndex]));
    }
    if (pluginLibs[pluginIndex].state >= ncclGinPluginStateInitReady) {
      // plugin init must be done by all comms to setup the context, therefore we use ">="
      NCCLCHECK(ncclGinPluginInit(comm, &pluginLibs[pluginIndex]));
      if (pluginLibs[pluginIndex].state == ncclGinPluginStateEnabled) {
        bool isAssigned = false;
        NCCLCHECK(ncclGinPluginAssignToComm(comm, pluginIndex, &isAssigned));
        if (isAssigned) {
          // If one external plugin is assigned to a comm, then disable all other external plugins
          ncclGinPluginDisableOtherExternal(pluginIndex);
          initialized = true;
          break;
        } else {
          ncclGinPluginFinalize(comm, pluginIndex);
        }
      }
    }
  }
  if (!initialized) INFO(NCCL_INIT|NCCL_NET, "Failed to initialize any GIN plugin");
  return ncclSuccess;
}

ncclResult_t ncclGinInitFromParent(struct ncclComm* comm, struct ncclComm* parent) {
  comm->ginContext = parent->ginContext;
  comm->ginPluginIndex = parent->ginPluginIndex;
  return ncclSuccess;
}

ncclResult_t ncclGinFinalize(struct ncclComm* comm) {
  int pluginIndex = comm->ginPluginIndex;
  std::lock_guard<std::mutex> lock(pluginMutex);
  NCCLCHECK(ncclGinPluginFinalize(comm, pluginIndex));
  return ncclSuccess;
}

ncclResult_t ncclGinGetDevCount(int pluginIndex, int* nPhysDevs, int* nVirtDevs) {
  if (pluginLibs[pluginIndex].state != ncclGinPluginStateEnabled ||
     pluginLibs[pluginIndex].physDevs == 0) goto fail;
  // lock not needed as it's called within a lock already in ncclTopoGetSystem
  *nPhysDevs = pluginLibs[pluginIndex].physDevs;
  return ncclSuccess;
fail:
  WARN("trying to access the number of devices of an uninitialized ginPlugin[%d]", pluginIndex);
  return ncclInternalError;
}
