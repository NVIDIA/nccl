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

#include <string.h>
#include <errno.h>
#include <mutex>

typedef ncclGin_t* getNcclGin_t(void* ginPluginLib);

extern getNcclGin_t getNcclGin_v11;
extern getNcclGin_t getNcclGin_v12;
NCCL_PARAM(GinPluginRefCount, "GIN_PLUGIN_REF_COUNT", 0);
#define NCCL_GIN_VERSION_COUNT 2
int ncclGinVersion[NCCL_GIN_VERSION_COUNT] = {12, 11};
getNcclGin_t* getNcclGin[NCCL_GIN_VERSION_COUNT] = {getNcclGin_v12, getNcclGin_v11};

#define NCCL_GIN_NUM_INTERNAL_PLUGINS 1

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
  ncclGin_t* ncclGin;                           // Pointer to the ncclGin_t structure
  int ncclGinVersion;                           // Version of the nccl gin plugin
  ncclGinPluginState_t ncclGinPluginState;      // State of the nccl gin plugin
  ncclGin_t* ncclRma;                           // Pointer to the ncclGin_t structure for RMA
  ncclGinPluginState_t ncclRmaPluginState;      // State of the nccl gin rma plugin
  int ncclGinPluginRefCount;                    // Reference count for the nccl gin plugin
  int ginPhysDevs;                              // ncclGin - number of physical devices
} ginPluginLib_t;

static int pluginCount = 0;
static ginPluginLib_t ginPluginLibs[NCCL_GIN_MAX_PLUGINS] = { 0 };
static std::mutex ginPluginMutex;
static std::once_flag initPluginLibsOnceFlag;

static ncclResult_t ncclGinPluginUnload(ginPluginLib_t* pluginLib) {
  if (pluginLib->dlHandle && pluginLib->ncclGinPluginRefCount == 0) {
    INFO(NCCL_INIT|NCCL_NET, "Unloading plugin %s", pluginLib->name);
    NCCLCHECK(ncclClosePluginLib(pluginLib->dlHandle, ncclPluginTypeGin));
    // memset will reset the status to ncllGinPluginStateLoadReady
    memset(pluginLib, 0, sizeof(ginPluginLib_t));
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
    pluginLib->ncclGinVersion = ncclGinVersion[i];
    pluginLib->ncclGin = getNcclGin[i](pluginLib->dlHandle);
    if (pluginLib->ncclGin) break;
  }

  if (pluginLib->ncclGin == nullptr)
    pluginLib->ncclGinPluginState = ncclGinPluginStateLoadFailed;
  else
    pluginLib->ncclGinPluginState = ncclGinPluginStateInitReady;

  INFO(NCCL_INIT|NCCL_NET, "Successfully loaded external gin plugin %s",
       (ncclPluginLibPaths[ncclPluginTypeGin] ? ncclPluginLibPaths[ncclPluginTypeGin] : pluginLib->name));
exit:
  return ncclSuccess;
fail:
  if (pluginLib->dlHandle) {
    NCCLCHECK(ncclClosePluginLib(pluginLib->dlHandle, ncclPluginTypeGin));
  }
  pluginLib->dlHandle = nullptr;
  pluginLib->ncclGinPluginState = ncclGinPluginStateLoadFailed;
  goto exit;
}

static ncclResult_t ncclGinPluginInit(struct ncclComm* comm, ginPluginLib_t* pluginLib) {
  int ndev;
  // Init must be called for each new comm to set the right context
  if (pluginLib->ncclGinPluginState == ncclGinPluginStateInitReady && pluginLib->ncclGin) {
    if (pluginLib->ncclGin->init(&comm->ginContext, comm->commHash, ncclDebugLog) != ncclSuccess ||
        pluginLib->ncclGin->devices(&ndev) != ncclSuccess || ndev <= 0) {
      pluginLib->ncclGinPluginState = ncclGinPluginStateDisabled;
    } else {
      pluginLib->ginPhysDevs = ndev;
      pluginLib->ncclGinPluginState = ncclGinPluginStateEnabled;
    }
  }

  // Initialize RMA plugin
  if (pluginLib->ncclRmaPluginState == ncclGinPluginStateInitReady && pluginLib->ncclRma) {
    if (pluginLib->ncclRma->init(&comm->ginContext, comm->commHash, ncclDebugLog) != ncclSuccess ||
        pluginLib->ncclRma->devices(&ndev) != ncclSuccess || ndev <= 0) {
      pluginLib->ncclRmaPluginState = ncclGinPluginStateDisabled;
    } else {
      pluginLib->ncclRmaPluginState = ncclGinPluginStateEnabled;
    }
  }
  return ncclSuccess;
}

static ncclResult_t ncclGinPluginAssignToComm(struct ncclComm* comm, int pluginIndex, bool* isAssigned) {
  *isAssigned = false;

  if (ginPluginLibs[pluginIndex].ncclGinPluginState >= ncclGinPluginStateEnabled) {
    INFO(NCCL_INIT|NCCL_NET, "Assigned GIN plugin %s to comm", ginPluginLibs[pluginIndex].ncclGin->name);
    comm->sharedRes->ginState.ncclGin = ginPluginLibs[pluginIndex].ncclGin;
    comm->sharedRes->ginState.ginVersion = ginPluginLibs[pluginIndex].ncclGinVersion;
    comm->ginPluginIndex = pluginIndex;
    NCCLCHECK(setLocalGinType(comm));
  }
  if (ginPluginLibs[pluginIndex].ncclRmaPluginState >= ncclGinPluginStateEnabled) {
    INFO(NCCL_INIT|NCCL_NET, "Assigned RMA plugin %s to comm", ginPluginLibs[pluginIndex].ncclRma->name);
    comm->rmaState.rmaProxyState.ncclGin = ginPluginLibs[pluginIndex].ncclRma;
  }
  ginPluginLibs[pluginIndex].ncclGinPluginRefCount++;
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
      snprintf(names+strlen(names), sizeof(names)-strlen(names), (strlen(names) == 0) ? "%s" : ", %s", ginPluginLibs[i].name);
      ginPluginLibs[i].ncclGinPluginState = ncclGinPluginStateDisabled;
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

  memset(ginPluginLibs, 0, NCCL_GIN_MAX_PLUGINS * sizeof(ginPluginLib_t));
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
        ginPluginLibs[pluginCounter].ncclGinPluginState = ncclGinPluginStateLoadReady;
        ginPluginLibs[pluginCounter].ncclGinPluginRefCount = ncclParamGinPluginRefCount();
        strcpy(ginPluginLibs[pluginCounter].name, ginPluginName);
        pluginCounter++;
      } else {
        INFO(NCCL_NET|NCCL_ENV,"NCCL_GIN_PLUGIN list contains a plugin name %s longer than %d characters, ignoring it.", ginPluginName, MAX_STR_LEN);
      }
      ginPluginName = strtok_r(nullptr, ",", &savePtr);
    }
    if (envGinPluginList) free(envGinPluginList);
  } else {
    // Add default gin plugin
    ginPluginLibs[pluginCounter].ncclGinPluginState = ncclGinPluginStateLoadReady;
    ginPluginLibs[pluginCounter].ncclGinPluginRefCount = ncclParamGinPluginRefCount();
    strcpy(ginPluginLibs[pluginCounter++].name, defaultGinPlugin);
  }

  // Also check if the NET plugin has GIN support
  if ((ginPluginLibs[pluginCounter].dlHandle = ncclGetNetPluginLib(ncclPluginTypeGin)) != NULL) {
    ginPluginLibs[pluginCounter].ncclGinPluginState = ncclGinPluginStateLoadReady;
    pluginCounter++;
  }

  // Add internal ib plugin
  ginPluginLibs[pluginCounter].ncclGin = &ncclGinIb;
  ginPluginLibs[pluginCounter].ncclGinPluginState = ncclGinPluginStateInitReady;
  ginPluginLibs[pluginCounter].ncclRma = &ncclGinIbProxy;
  ginPluginLibs[pluginCounter].ncclRmaPluginState = ncclGinPluginStateInitReady;
  pluginCounter++;
  pluginCount = pluginCounter;
}

static ncclResult_t ncclGinPluginFinalize(struct ncclComm* comm, int pluginIndex) {
  if (ginPluginLibs[pluginIndex].ncclGin && ginPluginLibs[pluginIndex].ncclGinPluginState == ncclGinPluginStateEnabled) NCCLCHECK(ginPluginLibs[pluginIndex].ncclGin->finalize(comm->ginContext));
  ginPluginLibs[pluginIndex].ncclGinPluginRefCount--;
  if (pluginIndex < (pluginCount - NCCL_GIN_NUM_INTERNAL_PLUGINS)) {
    NCCLCHECK(ncclGinPluginUnload(&ginPluginLibs[pluginIndex]));
  }
  return ncclSuccess;
}

ncclResult_t ncclGinInit(struct ncclComm* comm) {
  bool ncclGinPluginInitialized = false;
  std::call_once(initPluginLibsOnceFlag, initPluginLibsOnceFunc);
  std::lock_guard<std::mutex> lock(ginPluginMutex);
  for (int pluginIndex = 0; pluginIndex < pluginCount; pluginIndex++) {
    if (pluginIndex < (pluginCount - NCCL_GIN_NUM_INTERNAL_PLUGINS) && ginPluginLibs[pluginIndex].ncclGinPluginState == ncclGinPluginStateLoadReady) {
      NCCLCHECK(ncclGinPluginLoad(&ginPluginLibs[pluginIndex]));
    }
    if (ginPluginLibs[pluginIndex].ncclGinPluginState >= ncclGinPluginStateInitReady) {
      // plugin init must be done by all comms to setup the context, therefore we use ">="
      NCCLCHECK(ncclGinPluginInit(comm, &ginPluginLibs[pluginIndex]));
      if (ginPluginLibs[pluginIndex].ncclGinPluginState == ncclGinPluginStateEnabled) {
        bool isAssigned = false;
        NCCLCHECK(ncclGinPluginAssignToComm(comm, pluginIndex, &isAssigned));
        if (isAssigned) {
          // If one external plugin is assigned to a comm, then disable all other external plugins
          ncclGinPluginDisableOtherExternal(pluginIndex);
          ncclGinPluginInitialized = true;
          break;
        } else {
          ncclGinPluginFinalize(comm, pluginIndex);
        }
      }
    }
  }
  if (!ncclGinPluginInitialized) INFO(NCCL_INIT|NCCL_NET, "Failed to initialize any GIN plugin");
  return ncclSuccess;
}

ncclResult_t ncclGinInitFromParent(struct ncclComm* comm, struct ncclComm* parent) {
  comm->ginContext = parent->ginContext;
  comm->ginPluginIndex = parent->ginPluginIndex;
  return ncclSuccess;
}

ncclResult_t ncclGinFinalize(struct ncclComm* comm) {
  int pluginIndex = comm->ginPluginIndex;
  std::lock_guard<std::mutex> lock(ginPluginMutex);
  NCCLCHECK(ncclGinPluginFinalize(comm, pluginIndex));
  return ncclSuccess;
}

ncclResult_t ncclGinGetDevCount(int ginPluginIndex, int* nPhysDevs, int* nVirtDevs) {
  if (ginPluginLibs[ginPluginIndex].ncclGinPluginState != ncclGinPluginStateEnabled ||
     ginPluginLibs[ginPluginIndex].ginPhysDevs == 0) goto fail;
  // lock not needed as it's called within a lock already in ncclTopoGetSystem
  *nPhysDevs = ginPluginLibs[ginPluginIndex].ginPhysDevs;
  return ncclSuccess;
fail:
  WARN("%s: trying to access the number of devices of an uninitialized ginPlugin[%d]", __func__, ginPluginIndex);
  return ncclInternalError;
}
