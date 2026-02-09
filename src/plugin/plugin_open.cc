/*************************************************************************
 * Copyright (c) 2022-2023, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <link.h>
#include <dlfcn.h>

#include "debug.h"
#include "plugin.h"

#define MAX_STR_LEN 255

#define NUM_LIBS 4
static char* libNames[NUM_LIBS];
char* ncclPluginLibPaths[NUM_LIBS];
static void *libHandles[NUM_LIBS];
static const char *pluginNames[NUM_LIBS] = { "NET", "TUNER", "PROFILER", "ENV" };
static const char *pluginPrefix[NUM_LIBS] = { "libnccl-net", "libnccl-tuner", "libnccl-profiler", "libnccl-env" };
static const char *pluginFallback[NUM_LIBS] = { "", "", "", "" };
static unsigned long subsys[NUM_LIBS] = { NCCL_INIT|NCCL_NET, NCCL_INIT|NCCL_TUNING, NCCL_INIT, NCCL_INIT|NCCL_ENV };

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

static void appendNameToList(char* nameList, int *leftChars, char* name) {
  snprintf(nameList + PATH_MAX - *leftChars, *leftChars, " %s", name);
  *leftChars -= strlen(name) + 1;
}

static char* getLibPath(void* handle) {
  struct link_map* lm;
  if (dlinfo(handle, RTLD_DI_LINKMAP, &lm) != 0)
    return nullptr;
  else
    return strdup(lm->l_name);
}

static void* openPluginLib(enum ncclPluginType type, const char* libName) {
  int openErr, len = PATH_MAX;
  char libName_[MAX_STR_LEN] = { 0 };
  char openErrStr[MAX_STR_LEN + 1] = { 0 };
  char eNoEntNameList[PATH_MAX] = { 0 };

  if (libName && strlen(libName)) {
    snprintf(libName_, MAX_STR_LEN, "%s", libName);
  } else {
    snprintf(libName_, MAX_STR_LEN, "%s.so", pluginPrefix[type]);
  }

  libHandles[type] = tryOpenLib(libName_, &openErr, openErrStr);
  if (libHandles[type]) {
    libNames[type] = strdup(libName_);
    ncclPluginLibPaths[type] = getLibPath(libHandles[type]);
    return libHandles[type];
  }
  if (openErr == ENOENT) {
    appendNameToList(eNoEntNameList, &len, libName_);
  } else {
    INFO(subsys[type], "%s/Plugin: %s: %s", pluginNames[type], libName_, openErrStr);
  }

  // libName can't be a relative or absolute path (start with '.' or contain any '/'). It can't be a library name either (start with 'lib' or end with '.so')
  if (libName && strlen(libName) && strchr(libName, '/') == nullptr &&
      (strncmp(libName, "lib", strlen("lib")) || strlen(libName) < strlen(".so") ||
       strncmp(libName + strlen(libName) - strlen(".so"), ".so", strlen(".so")))) {
    snprintf(libName_, MAX_STR_LEN, "%s-%s.so", pluginPrefix[type], libName);

    libHandles[type] = tryOpenLib(libName_, &openErr, openErrStr);
    if (libHandles[type]) {
      libNames[type] = strdup(libName_);
      ncclPluginLibPaths[type] = getLibPath(libHandles[type]);
      return libHandles[type];
    }
    if (openErr == ENOENT) {
      appendNameToList(eNoEntNameList, &len, libName_);
    } else {
      INFO(subsys[type], "%s/Plugin: %s: %s", pluginNames[type], libName_, openErrStr);
    }
  }

  if (strlen(eNoEntNameList)) {
    INFO(subsys[type], "%s/Plugin: Could not find:%s%s%s", pluginNames[type], eNoEntNameList,
         (strlen(pluginFallback[type]) > 0 ? ". " : ""), pluginFallback[type]);
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

void* ncclOpenEnvPluginLib(const char* name) {
  return openPluginLib(ncclPluginTypeEnv, name);
}

void* ncclGetNetPluginLib(enum ncclPluginType type) {
  if (libNames[ncclPluginTypeNet]) {
    // increment the reference counter of the net library
    libNames[type] = strdup(libNames[ncclPluginTypeNet]);
    ncclPluginLibPaths[type] = strdup(ncclPluginLibPaths[ncclPluginTypeNet]);
    libHandles[type] = dlopen(libNames[ncclPluginTypeNet], RTLD_NOW | RTLD_LOCAL);
  }
  return libHandles[type];
}

ncclResult_t ncclClosePluginLib(void* handle, enum ncclPluginType type) {
  if (handle && libHandles[type] == handle) {
    dlclose(handle);
    libHandles[type] = nullptr;
    free(ncclPluginLibPaths[type]);
    ncclPluginLibPaths[type] = nullptr;
    free(libNames[type]);
    libNames[type] = nullptr;
  }
  return ncclSuccess;
}
