/*************************************************************************
 * Copyright (c) 2022-2023, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <dlfcn.h>

#include "debug.h"

#define MAX_STR_LEN 255

enum ncclPluginType {
  ncclPluginTypeNet,
  ncclPluginTypeTuner,
  ncclPluginTypeProfiler,
};

#define NUM_LIBS 3
static void *libHandles[NUM_LIBS];
static const char *pluginNames[NUM_LIBS] = { "NET", "TUNER", "PROFILER" };
static const char *pluginPrefix[NUM_LIBS] = { "libnccl-net", "libnccl-tuner", "libnccl-profiler" };
static const char *pluginFallback[NUM_LIBS] = { "Using internal net plugin.", "Using internal tuner plugin.", "" };
static unsigned long subsys[NUM_LIBS] = { NCCL_INIT|NCCL_NET, NCCL_INIT|NCCL_TUNING, NCCL_INIT };

static void* tryOpenLib(char* name, int* err, char* errStr) {
  *err = 0;
  if (nullptr == name || strlen(name) == 0) {
    return nullptr;
  }

  if (strncasecmp(name, "STATIC_PLUGIN", strlen(name)) == 0) {
    name = nullptr;
  }

  void *handle = dlopen(name, RTLD_NOW | RTLD_LOCAL);
  if (nullptr == handle) {
    strncpy(errStr, dlerror(), MAX_STR_LEN);
    errStr[MAX_STR_LEN] = '\0';
    // "handle" and "name" won't be NULL at the same time.
    // coverity[var_deref_model]
    if (strstr(errStr, name) && strstr(errStr, "No such file or directory")) {
      *err = ENOENT;
    }
  }
  return handle;
}

static void appendNameToList(char* nameList, int *nameListLen, char* name) {
  snprintf(nameList, *nameListLen, " %s", name);
  nameList += strlen(name) + 1;
  *nameListLen -= strlen(name) + 1;
}

static void* openPluginLib(enum ncclPluginType type, const char* libName) {
  int openErr, len = PATH_MAX;
  char libName_[MAX_STR_LEN] = { 0 };
  char openErrStr[MAX_STR_LEN + 1] = { 0 };
  char eNoEntNameList[PATH_MAX] = { 0 };

  if (libName && strlen(libName)) {
    snprintf(libName_, MAX_STR_LEN, "%s", libName);
    libHandles[type] = tryOpenLib(libName_, &openErr, openErrStr);
    if (libHandles[type]) {
      INFO(subsys[type], "%s/Plugin: Plugin name set by env to %s", pluginNames[type], libName_);
      return libHandles[type];
    }
    if (openErr == ENOENT) {
      appendNameToList(eNoEntNameList, &len, libName_);
    } else {
      INFO(subsys[type], "%s/Plugin: %s", pluginNames[type], openErrStr);
    }

    snprintf(libName_, MAX_STR_LEN, "%s-%s.so", pluginPrefix[type], libName);
    libHandles[type] = tryOpenLib(libName_, &openErr, openErrStr);
    if (libHandles[type]) {
      INFO(subsys[type], "%s/Plugin: Plugin name set by env to %s", pluginNames[type], libName_);
      return libHandles[type];
    }
    if (openErr == ENOENT) {
      appendNameToList(eNoEntNameList, &len, libName_);
    } else {
      INFO(subsys[type], "%s/Plugin: %s", pluginNames[type], openErrStr);
    }
  } else {
    snprintf(libName_, MAX_STR_LEN, "%s.so", pluginPrefix[type]);
    libHandles[type] = tryOpenLib(libName_, &openErr, openErrStr);
    if (libHandles[type]) {
      return libHandles[type];
    }
    if (openErr == ENOENT) {
      appendNameToList(eNoEntNameList, &len, libName_);
    } else {
      INFO(subsys[type], "%s/Plugin: %s", pluginNames[type], openErrStr);
    }
  }

  if (strlen(eNoEntNameList)) {
    INFO(subsys[type], "%s/Plugin: Could not find:%s. %s", pluginNames[type], eNoEntNameList, pluginFallback[type]);
  } else if (strlen(pluginFallback[type])) {
    INFO(subsys[type], "%s/Plugin: %s", pluginNames[type], pluginFallback[type]);
  }
  return nullptr;
}

void* ncclOpenNetPluginLib(const char* name) {
  return openPluginLib(ncclPluginTypeNet, name);
}

void* ncclOpenTunerPluginLib(const char* name) {
  return openPluginLib(ncclPluginTypeTuner, name);
}

void* ncclOpenProfilerPluginLib(const char* name) {
  return openPluginLib(ncclPluginTypeProfiler, name);
}

void* ncclGetNetPluginLib(void) {
  return libHandles[ncclPluginTypeNet];
}

ncclResult_t ncclClosePluginLib(void* handle) {
  for (int l=0; l<NUM_LIBS; l++) {
    if (libHandles[l] == handle) {
      libHandles[l] = nullptr;
      dlclose(handle);
      return ncclSuccess;
    }
  }
  return ncclInternalError;
}
