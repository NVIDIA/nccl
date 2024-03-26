/*************************************************************************
 * Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
 * Copyright (c) 2023, Meta Platforms, Inc. and affiliates.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <dlfcn.h>
#include <errno.h>
#include <stdlib.h>

#include "debug.h"
#include "nccl_tuner.h"

pthread_mutex_t tunerPluginLock = PTHREAD_MUTEX_INITIALIZER;
static int tunerPluginRefCount;
static void* tunerPluginLib = nullptr;
ncclTuner_t* tunerSymbol = nullptr;

static void* tryOpenDynamicLib(const char* name) {
  if (nullptr == name || strlen(name) == 0) {
    return nullptr;
  }
  void *handle = dlopen(name, RTLD_LAZY | RTLD_LOCAL);
  if (nullptr == handle) {
    if (ENOENT == errno) {
      INFO(NCCL_ENV|NCCL_TUNING, "TUNER/Plugin: No plugin found (%s)", name);
    }
  }
  return handle;
}

static void summarizeOpenTunerPluginLibErrors(char* pluginNames) {
  const char *separator = " ";
  int len = strlen(pluginNames);
  // remove tail separator
  pluginNames[len - 1] = '\0';

  // remove last plugin name
  while (len > 0 && pluginNames[--len] != *separator);
  if (len > 0) {
    pluginNames[len] = '\0';
  }

  // distinguish between one load attempt and multiple attempts
  if (strstr(pluginNames, separator)) {
    INFO(NCCL_ENV|NCCL_TUNING, "TUNER/Plugin: Most recent plugin load returned %d : %s. All attempts to load '%s' also failed.", errno, dlerror(), pluginNames);
  } else {
    INFO(NCCL_ENV|NCCL_TUNING, "TUNER/Plugin: Plugin load returned %d : %s : when loading %s", errno, dlerror(), pluginNames);
  }
}

static void* openTunerPluginLib(void) {
  void *pluginLib;

#define MAX_PLUGIN_LOAD 4

  int len;
  char tunerPluginLibNameTried[MAX_PLUGIN_LOAD * PATH_MAX] = { 0 };
  char *ptr = tunerPluginLibNameTried;
  char tunerPluginLibName[PATH_MAX];
  const char *envTunerPluginName = getenv("NCCL_TUNER_PLUGIN");
  if (envTunerPluginName && strlen(envTunerPluginName)) {
    INFO(NCCL_ENV|NCCL_TUNING, "TUNER/Plugin: NCCL_TUNER_PLUGIN set to %s", envTunerPluginName);
    snprintf(tunerPluginLibName, PATH_MAX, "%s", envTunerPluginName);
    pluginLib = tryOpenDynamicLib(tunerPluginLibName);
    if (pluginLib) {
      INFO(NCCL_ENV|NCCL_TUNING, "TUNER/Plugin: Plugin name set by env to %s", tunerPluginLibName);
      return pluginLib;
    }
    len = PATH_MAX - strlen(ptr);
    snprintf(ptr + strlen(ptr), len + 1, "%s ", tunerPluginLibName);

    snprintf(tunerPluginLibName, PATH_MAX, "libnccl-tuner-%s.so", envTunerPluginName);
    pluginLib = tryOpenDynamicLib(tunerPluginLibName);
    if (pluginLib) {
      INFO(NCCL_ENV|NCCL_TUNING, "TUNER/Plugin: Plugin name set by env to %s", tunerPluginLibName);
      return pluginLib;
    }
    len = PATH_MAX - strlen(ptr);
    snprintf(ptr + strlen(ptr), len + 1, "%s ", tunerPluginLibName);
  } else {
    snprintf(tunerPluginLibName, PATH_MAX, "libnccl-tuner.so");
    pluginLib = tryOpenDynamicLib(tunerPluginLibName);
    if (pluginLib) {
      return pluginLib;
    }
    len = PATH_MAX - strlen(ptr);
    snprintf(ptr + strlen(ptr), len + 1, "%s ", tunerPluginLibName);
  }

  const char *envNetPluginName = getenv("NCCL_NET_PLUGIN");
  if (envNetPluginName && strlen(envNetPluginName)) {
    // Users are allowed to pack tuner into the net plugin
    snprintf(tunerPluginLibName, PATH_MAX, "%s", envNetPluginName);
    pluginLib = tryOpenDynamicLib(tunerPluginLibName);
    if (pluginLib) {
      INFO(NCCL_ENV|NCCL_TUNING, "TUNER/Plugin: Plugin name set by env to %s", tunerPluginLibName);
      return pluginLib;
    }
    len = PATH_MAX - strlen(ptr);
    snprintf(ptr + strlen(ptr), len + 1, "%s ", tunerPluginLibName);

    snprintf(tunerPluginLibName, PATH_MAX, "libnccl-net-%s.so", envNetPluginName);
    pluginLib = tryOpenDynamicLib(tunerPluginLibName);
    if (pluginLib) {
      INFO(NCCL_ENV|NCCL_TUNING, "TUNER/Plugin: Plugin name set by env to %s", tunerPluginLibName);
      return pluginLib;
    }
    len = PATH_MAX - strlen(ptr);
    snprintf(ptr + strlen(ptr), len + 1, "%s ", tunerPluginLibName);
  } else {
    snprintf(tunerPluginLibName, PATH_MAX, "libnccl-net.so");
    pluginLib = tryOpenDynamicLib(tunerPluginLibName);
    if (pluginLib) {
      return pluginLib;
    }
    len = PATH_MAX - strlen(ptr);
    snprintf(ptr + strlen(ptr), len + 1, "%s ", tunerPluginLibName);
  }
  summarizeOpenTunerPluginLibErrors(ptr);

  tunerPluginLibName[0] = '\0';
  return nullptr;
}

enum {
  tunerPluginLoadFailed  = -1,
  tunerPluginLoadReady   =  0,
  tunerPluginLoadSuccess =  1,
};

ncclResult_t ncclTunerPluginLoad(ncclTuner_t** tuner) {
  // Initialize to nullptr by default if plugin tuner cannot be loaded.
  *tuner = nullptr;
  static int status = tunerPluginLoadReady;
  if (tunerPluginLoadFailed == status) {
    return ncclSuccess;
  }

  pthread_mutex_lock(&tunerPluginLock);
  if (tunerPluginLoadFailed == status) {
    goto exit;
  }

  if (tunerPluginLoadSuccess == status) {
    *tuner = tunerSymbol;
    ++tunerPluginRefCount;
    goto exit;
  }

  tunerPluginLib = openTunerPluginLib();
  if (nullptr == tunerPluginLib) {
    INFO(NCCL_ENV|NCCL_TUNING, "TUNER/Plugin: Using internal tuner plugin.");
    goto fail;
  }

  tunerSymbol = (ncclTuner_t*)dlsym(tunerPluginLib, NCCL_TUNER_PLUGIN_SYMBOL);
  if (tunerSymbol == nullptr) {
    INFO(NCCL_ENV|NCCL_TUNING, "TUNER/Plugin: Failed to find " NCCL_TUNER_PLUGIN_SYMBOL ", using internal tuner instead.");
    dlclose(tunerPluginLib);
    goto fail;
  }

  INFO(NCCL_ENV|NCCL_TUNING, "TUNER/Plugin: Using tuner plugin %s", tunerSymbol->name);
  *tuner = tunerSymbol;
  ++tunerPluginRefCount;
  status = tunerPluginLoadSuccess;

exit:
  pthread_mutex_unlock(&tunerPluginLock);
  return ncclSuccess;
fail:
  tunerPluginLib = nullptr;
  status = tunerPluginLoadFailed;
  goto exit;
}

ncclResult_t ncclTunerPluginUnload(ncclTuner_t** tuner) {
  if (*tuner == nullptr) return ncclSuccess;
  pthread_mutex_lock(&tunerPluginLock);
  if (0 == (--tunerPluginRefCount)) {
    INFO(NCCL_TUNING, "TUNER/Plugin: Closing tuner: '%s'", tunerSymbol->name);
    dlclose(tunerPluginLib);
    tunerPluginLib = nullptr;
    tunerSymbol = nullptr;
    *tuner = nullptr;
  }
  pthread_mutex_unlock(&tunerPluginLock);
  return ncclSuccess;
}
