/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "rma.h"
#include "bootstrap.h"
#include "checks.h"
#include "plugin.h"
#include "nccl_rma.h"
#include "gin/gin_host_proxy.h"

#include <string.h>
#include <errno.h>
#include <mutex>

typedef ncclRma_t* getNcclRma_t(void* rmaPluginLib);

extern getNcclRma_t getNcclRma_v15;
extern getNcclRma_t getNcclRma_v14;
extern getNcclRma_t getNcclRma_v13;
NCCL_PARAM(RmaPluginRefCount, "RMA_PLUGIN_REF_COUNT", 0);
#define NCCL_RMA_VERSION_COUNT 3
int ncclRmaVersion[NCCL_RMA_VERSION_COUNT] = {15, 14, 13};
getNcclRma_t* getNcclRma[NCCL_RMA_VERSION_COUNT] = {getNcclRma_v15, getNcclRma_v14, getNcclRma_v13};

#define NCCL_RMA_NUM_RESERVED_PLUGINS 3
#define NCCL_RMA_NUM_INTERNAL_PLUGINS 1

typedef enum ncclRmaPluginState {
  ncclRmaPluginStateDisabled = -2,       // Plugin library failed to initialize
  ncclRmaPluginStateLoadFailed = -1,       // Plugin library failed to load
  ncclRmaPluginStateLoadReady = 0,        // Plugin library is ready to be loaded
  ncclRmaPluginStateInitReady = 1,        // Plugin library is loaded and ready to be initialized
  ncclRmaPluginStateEnabled = 2,        // Plugin library is loaded and initialized
} ncclRmaPluginState_t;

#define MAX_STR_LEN 255
typedef struct rmaPluginLib {
  char name[MAX_STR_LEN];                       // Name of the plugin library
  void* dlHandle;                               // Handle to the plugin library
  ncclRma_t* ncclRma;                           // Pointer to the plugin structure
  int version;                                  // Version of the plugin
  ncclRmaPluginState_t state;                   // State of the plugin
  int refCount;                                 // Reference count
  int physDevs;                                 // Number of physical devices
} rmaPluginLib_t;

static int pluginCount = 0;
static rmaPluginLib_t pluginLibs[NCCL_RMA_MAX_PLUGINS] = {0};
static std::mutex pluginMutex;
static std::once_flag initPluginLibsOnceFlag;

static ncclResult_t ncclRmaPluginUnload(rmaPluginLib_t* pluginLib) {
  if (pluginLib->dlHandle && pluginLib->refCount == 0) {
    INFO(NCCL_DESTROY | NCCL_NET, "RMA/Plugin: Unloading plugin %s", pluginLib->name);
    NCCLCHECK(ncclClosePluginLib(pluginLib->dlHandle, ncclPluginTypeRma));

    // Reset fields but preserve name, to be reused when reloading
    pluginLib->dlHandle = NULL;
    pluginLib->ncclRma = NULL;
    pluginLib->state = ncclRmaPluginStateLoadReady;
    pluginLib->refCount = 0;
    pluginLib->physDevs = 0;
  }
  return ncclSuccess;
}

static ncclResult_t ncclRmaPluginLoad(rmaPluginLib_t* pluginLib) {
  // Entries borrowed from GIN/NET may already have a handle;
  // after unload, reopen the same library by its saved name/path.
  if (pluginLib->dlHandle == NULL) {
    pluginLib->dlHandle = ncclOpenRmaPluginLib(pluginLib->name);
    if (pluginLib->dlHandle == nullptr) goto fail;
  }

  // load rma
  for (int i = 0; i < NCCL_RMA_VERSION_COUNT; i++) {
    pluginLib->version = ncclRmaVersion[i];
    pluginLib->ncclRma = getNcclRma[i](pluginLib->dlHandle);
    if (pluginLib->ncclRma) break;
  }

  if (pluginLib->ncclRma == nullptr) goto fail;

  pluginLib->state = ncclRmaPluginStateInitReady;
  INFO(NCCL_INIT | NCCL_NET, "RMA/Plugin: Successfully loaded external plugin %s",
       (ncclPluginLibPaths[ncclPluginTypeRma] ? ncclPluginLibPaths[ncclPluginTypeRma] : pluginLib->name));
exit:
  return ncclSuccess;
fail:
  INFO(NCCL_INIT | NCCL_NET, "RMA/Plugin: Failed to load external plugin %s, dlHandle: %p, ncclRma: %p",
       (ncclPluginLibPaths[ncclPluginTypeRma] ? ncclPluginLibPaths[ncclPluginTypeRma] : pluginLib->name),
       pluginLib->dlHandle, pluginLib->ncclRma);
  if (pluginLib->dlHandle) {
    NCCLCHECK(ncclClosePluginLib(pluginLib->dlHandle, ncclPluginTypeRma));
  }
  pluginLib->dlHandle = nullptr;
  pluginLib->ncclRma = nullptr;
  pluginLib->version = 0;
  pluginLib->state = ncclRmaPluginStateLoadFailed;
  goto exit;
}

static ncclResult_t ncclRmaPluginInit(struct ncclComm* comm, rmaPluginLib_t* pluginLib) {
  int ndev;
  // Init must be called for each new comm to set the right context
  if (pluginLib->state >= ncclRmaPluginStateInitReady && pluginLib->ncclRma) {
    if (pluginLib->ncclRma->init(&comm->rmaContext, comm->commHash, ncclDebugLog) != ncclSuccess) {
      pluginLib->state = ncclRmaPluginStateDisabled;
    }
  }
  if (pluginLib->state == ncclRmaPluginStateInitReady && pluginLib->ncclRma) {
    if (pluginLib->ncclRma->devices(&ndev) != ncclSuccess || ndev <= 0) {
      pluginLib->state = ncclRmaPluginStateDisabled;
    } else {
      pluginLib->physDevs = ndev;
      pluginLib->state = ncclRmaPluginStateEnabled;
    }
  }
  return ncclSuccess;
}

static ncclResult_t ncclRmaPluginAssignToComm(struct ncclComm* comm, int pluginIndex, bool* isAssigned) {
  *isAssigned = false;

  if (pluginLibs[pluginIndex].state >= ncclRmaPluginStateEnabled) {
    INFO(NCCL_INIT | NCCL_NET, "RMA/Plugin: Assigned plugin %s to comm", pluginLibs[pluginIndex].ncclRma->name);
    comm->rmaState.rmaProxyState.ncclRma = pluginLibs[pluginIndex].ncclRma;
    comm->rmaState.rmaProxyState.rmaVersion = pluginLibs[pluginIndex].version;
    comm->rmaPluginIndex = pluginIndex;
  }
  pluginLibs[pluginIndex].refCount++;
  *isAssigned = true;
  return ncclSuccess;
}

static ncclResult_t ncclRmaPluginDisableOtherExternal(int pluginIndex) {
  // Only if an external plugin is enabled, disable other external plugins
  if (pluginIndex >= (pluginCount - NCCL_RMA_NUM_INTERNAL_PLUGINS)) return ncclSuccess;
  char names[MAX_STR_LEN * (NCCL_RMA_MAX_PLUGINS - NCCL_RMA_NUM_INTERNAL_PLUGINS)] = {0};
  for (int i = 0; i < (pluginCount - NCCL_RMA_NUM_INTERNAL_PLUGINS); i++) {
    if (i != pluginIndex && pluginLibs[i].state >= ncclRmaPluginStateEnabled) {
      // Append all disabled plugin names to a string
      snprintf(names + strlen(names), sizeof(names) - strlen(names), (strlen(names) == 0) ? "%s" : ", %s",
               pluginLibs[i].name);
      pluginLibs[i].state = ncclRmaPluginStateDisabled;
    }
  }
  if (strlen(names) > 0) {
    INFO(NCCL_INIT | NCCL_NET, "RMA/Plugin: Disabling external plugins: %s", names);
  }
  return ncclSuccess;
}

static void initPluginLibsOnceFunc() {
  char* rmaPluginName = nullptr;
  const char* defaultRmaPlugin = "libnccl-rma.so";
  const char* envRmaPlugin = nullptr;
  char* envRmaPluginList = nullptr;
  char* savePtr = nullptr;
  int pluginCounter = 0;

  memset(pluginLibs, 0, NCCL_RMA_MAX_PLUGINS * sizeof(rmaPluginLib_t));
  envRmaPlugin = ncclGetEnv("NCCL_RMA_PLUGIN");
  if (envRmaPlugin) {
    INFO(NCCL_ENV | NCCL_NET, "NCCL_RMA_PLUGIN set by environment to %s", envRmaPlugin);
    if (strcasecmp(envRmaPlugin, "none") == 0) envRmaPlugin = "";
    envRmaPluginList = strdup(envRmaPlugin);
    // Iterate over list until the list is empty
    rmaPluginName = strtok_r(envRmaPluginList, ",", &savePtr);
    while (rmaPluginName) {
      // So, we can have at most( NCCL_RMA_MAX_PLUGINS - (NCCL_RMA_NUM_RESERVED_PLUGINS)) in the NCCL_RMA_PLUGIN list
      if (pluginCounter >= (NCCL_RMA_MAX_PLUGINS - NCCL_RMA_NUM_RESERVED_PLUGINS)) {
        INFO(NCCL_NET | NCCL_ENV, "NCCL_RMA_PLUGIN list contains more than %d plugins, ignoring the rest",
             (NCCL_RMA_MAX_PLUGINS - NCCL_RMA_NUM_RESERVED_PLUGINS));
        break;
      }
      // need to leave space for the name + "\n"
      if ((strlen(rmaPluginName) + 1) <= MAX_STR_LEN) {
        pluginLibs[pluginCounter].state = ncclRmaPluginStateLoadReady;
        pluginLibs[pluginCounter].refCount = ncclParamRmaPluginRefCount();
        strcpy(pluginLibs[pluginCounter].name, rmaPluginName);
        pluginCounter++;
      } else {
        INFO(NCCL_NET | NCCL_ENV,
             "NCCL_RMA_PLUGIN list contains a plugin name %s longer than %d characters, ignoring it.", rmaPluginName,
             MAX_STR_LEN);
      }
      rmaPluginName = strtok_r(nullptr, ",", &savePtr);
    }
    if (envRmaPluginList) free(envRmaPluginList);
  } else {
    // Add default rma plugin
    pluginLibs[pluginCounter].state = ncclRmaPluginStateLoadReady;
    pluginLibs[pluginCounter].refCount = ncclParamRmaPluginRefCount();
    strcpy(pluginLibs[pluginCounter++].name, defaultRmaPlugin);
  }

  // check if the GIN plugin has RMA support
  if ((pluginLibs[pluginCounter].dlHandle = ncclGetGinPluginLib(ncclPluginTypeRma)) != NULL) {
    pluginLibs[pluginCounter].state = ncclRmaPluginStateLoadReady;
    strcpy(pluginLibs[pluginCounter++].name, ncclGetPluginLibName(ncclPluginTypeRma));
  }
  // Also check if the NET plugin has RMA support
  if ((pluginLibs[pluginCounter].dlHandle = ncclGetNetPluginLib(ncclPluginTypeRma)) != NULL) {
    pluginLibs[pluginCounter].state = ncclRmaPluginStateLoadReady;
    strcpy(pluginLibs[pluginCounter++].name, ncclGetPluginLibName(ncclPluginTypeRma));
  }

  // Add internal ib plugin
  pluginLibs[pluginCounter].ncclRma = &ncclRmaIbProxy;
  pluginLibs[pluginCounter].state = ncclRmaPluginStateInitReady;
  pluginLibs[pluginCounter].version = ncclRmaVersion[0];
  pluginCounter++;
  pluginCount = pluginCounter;
}

static ncclResult_t ncclRmaPluginFinalize(struct ncclComm* comm, int pluginIndex) {
  if (pluginLibs[pluginIndex].ncclRma && pluginLibs[pluginIndex].state == ncclRmaPluginStateEnabled) {
    NCCLCHECK(pluginLibs[pluginIndex].ncclRma->finalize(comm->rmaContext));
  }
  pluginLibs[pluginIndex].refCount--;
  if (pluginIndex < (pluginCount - NCCL_RMA_NUM_INTERNAL_PLUGINS)) {
    NCCLCHECK(ncclRmaPluginUnload(&pluginLibs[pluginIndex]));
  }
  return ncclSuccess;
}

ncclResult_t ncclRmaInit(struct ncclComm* comm) {
  bool initialized = false;
  comm->rmaPluginIndex = -1;
  std::call_once(initPluginLibsOnceFlag, initPluginLibsOnceFunc);
  std::lock_guard<std::mutex> lock(pluginMutex);
  for (int pluginIndex = 0; pluginIndex < pluginCount; pluginIndex++) {
    if (pluginIndex < (pluginCount - NCCL_RMA_NUM_INTERNAL_PLUGINS) &&
        pluginLibs[pluginIndex].state == ncclRmaPluginStateLoadReady) {
      NCCLCHECK(ncclRmaPluginLoad(&pluginLibs[pluginIndex]));
    }
    if (pluginLibs[pluginIndex].state >= ncclRmaPluginStateInitReady) {
      // plugin init must be done by all comms to setup the context, therefore we use ">="
      NCCLCHECK(ncclRmaPluginInit(comm, &pluginLibs[pluginIndex]));
      if (pluginLibs[pluginIndex].state == ncclRmaPluginStateEnabled) {
        bool isAssigned = false;
        NCCLCHECK(ncclRmaPluginAssignToComm(comm, pluginIndex, &isAssigned));
        if (isAssigned) {
          // If one external plugin is assigned to a comm, then disable all other external plugins
          ncclRmaPluginDisableOtherExternal(pluginIndex);
          initialized = true;
          break;
        } else {
          ncclRmaPluginFinalize(comm, pluginIndex);
        }
      }
    }
  }
  if (initialized) {
    NCCLCHECK(ncclGinProxyInit(comm));
  }
  if (!initialized) INFO(NCCL_INIT | NCCL_NET, "RMA/Plugin: Failed to initialize any plugin");
  return ncclSuccess;
}

ncclResult_t ncclRmaInitFromParent(struct ncclComm* comm, struct ncclComm* parent) {
  comm->rmaContext = parent->rmaContext;
  comm->rmaPluginIndex = parent->rmaPluginIndex;
  return ncclSuccess;
}

ncclResult_t ncclRmaFinalize(struct ncclComm* comm) {
  int pluginIndex = comm->rmaPluginIndex;
  if (pluginIndex < 0) return ncclSuccess;
  std::lock_guard<std::mutex> lock(pluginMutex);
  if (pluginIndex >= pluginCount) return ncclSuccess;
  NCCLCHECK(ncclRmaPluginFinalize(comm, pluginIndex));
  return ncclSuccess;
}

ncclResult_t ncclRmaGetDevCount(int pluginIndex, int* nPhysDevs, int* nVirtDevs) {
  if (pluginLibs[pluginIndex].state != ncclRmaPluginStateEnabled || pluginLibs[pluginIndex].physDevs == 0) goto fail;
  // lock not needed as it's called within a lock already in ncclTopoGetSystem
  *nPhysDevs = pluginLibs[pluginIndex].physDevs;
  return ncclSuccess;
fail:
  WARN("trying to access the number of devices of an uninitialized rmaPlugin[%d]", pluginIndex);
  return ncclInternalError;
}
