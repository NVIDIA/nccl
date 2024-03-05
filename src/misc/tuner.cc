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
static int tunerPluginRefCount = -1;
static void* tunerPluginLib = nullptr;
ncclTuner_t* tunerSymbol = nullptr;

ncclResult_t ncclLoadTunerPlugin(ncclTuner_t** tuner) {
  // Initialize to nullptr by default if plugin tuner cannot be loaded.
  *tuner = nullptr;
  if (tunerPluginRefCount == -2) return ncclSuccess;

  pthread_mutex_lock(&tunerPluginLock);
  if (tunerPluginRefCount == -1) {
    tunerPluginRefCount = -2; // Default: no plugin, don't try again later

    const char* name = getenv("NCCL_TUNER_PLUGIN");
    if (name) {
      INFO(NCCL_TUNING, "NCCL_TUNER_PLUGIN set to %s", name);
      tunerPluginLib = dlopen(name, RTLD_LAZY | RTLD_LOCAL);
      if (tunerPluginLib == nullptr) {
        // dlopen does not guarantee to set errno, but dlerror only gives us a
        // string, so checking errno doesn't hurt to try to provide a better
        // error message
        if (errno == ENOENT) {
          INFO(NCCL_TUNING, "Tuner: no plugin found '%s', using default tuner instead.", name);
        } else {
          INFO(NCCL_TUNING, "Tuner: plugin load '%s' returned error (%d : %s), using default tuner instead.", name, errno, dlerror());
        }
      } else {
        tunerSymbol = (ncclTuner_t*)dlsym(tunerPluginLib, NCCL_TUNER_PLUGIN_SYMBOL);
        if (tunerSymbol == nullptr) {
          INFO(NCCL_TUNING, "Tuner: failed to find " NCCL_TUNER_PLUGIN_SYMBOL " in plugin (%s), using default tuner instead.", name);
          dlclose(tunerPluginLib);
          tunerPluginLib = nullptr;
        } else {
          INFO(NCCL_TUNING, "Opened tuner: '%s'", tunerSymbol->name);
          tunerPluginRefCount = 0;
        }
      }
    }
  }

  if (tunerPluginRefCount >= 0) {
    *tuner = tunerSymbol;
    INFO(NCCL_INIT, "Using tuner plugin: '%s'", tunerSymbol->name);
    tunerPluginRefCount++;
  }
  pthread_mutex_unlock(&tunerPluginLock);
  return ncclSuccess;
}

ncclResult_t ncclCloseTunerPlugin(ncclTuner_t** tuner) {
  if (*tuner == nullptr) return ncclSuccess;
  pthread_mutex_lock(&tunerPluginLock);
  if (--tunerPluginRefCount == 0) {
    if (tunerPluginLib == nullptr) {
      WARN("Tuner plugin refcount is 0, yet tunerPluginLib ptr is NULL\n");
    } else {
      INFO(NCCL_TUNING, "Closing tuner: '%s'", tunerSymbol->name);
      dlclose(tunerPluginLib);
    }
    tunerPluginLib = nullptr;
    tunerSymbol = nullptr;
    *tuner = nullptr;
    tunerPluginRefCount = -1;
  }
  pthread_mutex_unlock(&tunerPluginLock);
  return ncclSuccess;
}
