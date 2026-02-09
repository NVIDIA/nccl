/*************************************************************************
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <errno.h>
#include <stdlib.h>
#include <mutex>
#include <atomic>

#include "checks.h"
#include "debug.h"
#include "env.h"
#include "param.h"
#include "plugin.h"

extern ncclEnv_t* getNcclEnv_v1(void* lib);

static void* envPluginLib = nullptr;
static ncclEnv_t* ncclEnvPlugin = nullptr;
extern ncclEnv_v1_t ncclIntEnv_v1;

#define EXT_ENV_PLUGIN 0
#define INT_ENV_PLUGIN 1
#define NUM_ENV_PLUGIN 2
static ncclEnv_t *ncclEnvPlugins[NUM_ENV_PLUGIN] = { nullptr, &ncclIntEnv_v1 };

enum {
  envPluginLoadFailed  = -1,
  envPluginLoadReady   =  0,
  envPluginLoadSuccess =  1,
};
static int envPluginStatus = envPluginLoadReady;

static ncclResult_t ncclEnvPluginLoad(void) {
  const char* envName;
  if (envPluginStatus != envPluginLoadReady) goto exit;

  if ((envName = std::getenv("NCCL_ENV_PLUGIN")) != nullptr) {
    INFO(NCCL_ENV, "NCCL_ENV_PLUGIN set by environment to %s", envName);
    if (strcasecmp(envName, "none") == 0) {
      goto fail;
    }
  }
  envPluginLib = ncclOpenEnvPluginLib(envName);
  if (nullptr == envPluginLib) {
    goto fail;
  } else if (ncclPluginLibPaths[ncclPluginTypeEnv]) {
    envName = ncclPluginLibPaths[ncclPluginTypeEnv];
  }

  ncclEnvPlugins[EXT_ENV_PLUGIN] = getNcclEnv_v1(envPluginLib);
  if (nullptr == ncclEnvPlugins[EXT_ENV_PLUGIN]) {
    INFO(NCCL_INIT, "External env plugin %s is unsupported", envName);
    goto fail;
  }
  INFO(NCCL_INIT, "Successfully loaded external env plugin %s", envName);

  envPluginStatus = envPluginLoadSuccess;

exit:
  return ncclSuccess;
fail:
  // Fallback to internal/default plugin
  if (envPluginLib) NCCLCHECK(ncclClosePluginLib(envPluginLib, ncclPluginTypeEnv));
  envPluginLib = nullptr;
  envPluginStatus = envPluginLoadFailed;
  goto exit;
}

static ncclResult_t ncclEnvPluginUnload(void) {
  if (ncclEnvPlugin) {
    INFO(NCCL_INIT, "ENV/Plugin: Closing env plugin %s", ncclEnvPlugin->name);
  }
  if (ncclEnvPlugins[EXT_ENV_PLUGIN]) {
    ncclEnvPlugin = ncclEnvPlugins[INT_ENV_PLUGIN];
    ncclEnvPlugins[EXT_ENV_PLUGIN] = nullptr;
  }
  NCCLCHECK(ncclClosePluginLib(envPluginLib, ncclPluginTypeEnv));
  return ncclSuccess;
}

void ncclEnvPluginFinalize(void);

static bool initialized;

ncclResult_t ncclEnvPluginInit(void) {
  initEnv();
  NCCLCHECK(ncclEnvPluginLoad());
  ncclEnvPlugin = (envPluginLoadSuccess == envPluginStatus) ? ncclEnvPlugins[EXT_ENV_PLUGIN] : ncclEnvPlugins[INT_ENV_PLUGIN];
  NCCLCHECK(ncclEnvPlugin->init(NCCL_MAJOR, NCCL_MINOR, NCCL_PATCH, NCCL_SUFFIX));
  atexit(ncclEnvPluginFinalize);
  COMPILER_ATOMIC_STORE(&initialized, true, std::memory_order_release);
  return ncclSuccess;
}

void ncclEnvPluginFinalize(void) {
  if (ncclEnvPlugin->finalize) {
    ncclEnvPlugin->finalize();
    ncclEnvPluginUnload();
  }
}

const char* ncclEnvPluginGetEnv(const char* name) {
  return ncclEnvPlugin->getEnv(name);
}

bool ncclEnvPluginInitialized(void) {
  return COMPILER_ATOMIC_LOAD(&initialized, std::memory_order_acquire);
}
