/*************************************************************************
 * Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
 * Copyright (c) 2023, Meta Platforms, Inc. and affiliates.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <errno.h>
#include <stdlib.h>

#include "checks.h"
#include "debug.h"
#include "tuner.h"
#include "plugin.h"

extern ncclTuner_t* getNcclTuner_v2(void* lib);
extern ncclTuner_t* getNcclTuner_v3(void* lib);
extern ncclTuner_t* getNcclTuner_v4(void* lib);

pthread_mutex_t tunerPluginLock = PTHREAD_MUTEX_INITIALIZER;
static int tunerPluginRefCount;
static void* tunerPluginLib = nullptr;
static ncclTuner_t* tunerSymbol = nullptr;

enum {
  tunerPluginLoadFailed  = -1,
  tunerPluginLoadReady   =  0,
  tunerPluginLoadSuccess =  1,
};

#define MAX_PLUGIN_LOAD 4

static int status = tunerPluginLoadReady;

ncclResult_t ncclTunerPluginLoad(struct ncclComm* comm) {
  // Initialize to nullptr by default if plugin tuner cannot be loaded.
  comm->tuner = nullptr;
  if (tunerPluginLoadFailed == status) {
    return ncclSuccess;
  }

  pthread_mutex_lock(&tunerPluginLock);
  if (tunerPluginLoadFailed == status) {
    goto exit;
  }

  if (tunerPluginLoadSuccess == status) {
    comm->tuner = tunerSymbol;
    ++tunerPluginRefCount;
    goto exit;
  }

  tunerPluginLib = ncclOpenTunerPluginLib(ncclGetEnv("NCCL_TUNER_PLUGIN"));
  if (nullptr == tunerPluginLib) {
    tunerPluginLib = ncclGetNetPluginLib();
    if (nullptr == tunerPluginLib) {
      goto fail;
    }
  }

  tunerSymbol = getNcclTuner_v4(tunerPluginLib);
  if (tunerSymbol == NULL) {
    tunerSymbol = getNcclTuner_v3(tunerPluginLib);
  }
  if (tunerSymbol == NULL) {
    tunerSymbol = getNcclTuner_v2(tunerPluginLib);
  }
  if (tunerSymbol == NULL) {
    goto fail;
  }

  comm->tuner = tunerSymbol;
  ++tunerPluginRefCount;
  status = tunerPluginLoadSuccess;
  comm->tunerPluginLoaded = 1;

exit:
  pthread_mutex_unlock(&tunerPluginLock);
  return ncclSuccess;
fail:
  tunerPluginLib = nullptr;
  status = tunerPluginLoadFailed;
  goto exit;
}

ncclResult_t ncclTunerPluginUnload(struct ncclComm* comm) {
  pthread_mutex_lock(&tunerPluginLock);
  if (comm->tunerPluginLoaded && 0 == (--tunerPluginRefCount)) {
    INFO(NCCL_TUNING, "TUNER/Plugin: Closing tuner: '%s'", tunerSymbol->name);
    NCCLCHECK(ncclClosePluginLib(tunerPluginLib));
    tunerPluginLib = nullptr;
    tunerSymbol = nullptr;
    comm->tuner = nullptr;
    status = tunerPluginLoadReady;
    comm->tunerPluginLoaded = 0;
  }
  pthread_mutex_unlock(&tunerPluginLock);
  return ncclSuccess;
}
