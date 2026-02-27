/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "param.h"
#include "debug.h"
#include "env.h"

#include <algorithm>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <sys/types.h>
#include <unistd.h>
#include <pthread.h>
#include <mutex>
#include <pwd.h>
#include <unordered_set>
#include "os.h"

const char* userHomeDir() {
  struct passwd *pwUser = getpwuid(getuid());
  return pwUser == NULL ? NULL : pwUser->pw_dir;
}

void setEnvFile(const char* fileName) {
  FILE * file = fopen(fileName, "r");
  if (file == NULL) return;

  char *line = NULL;
  char envVar[1024];
  char envValue[1024];
  size_t n = 0;
  ssize_t read;
  while ((read = getline(&line, &n, file)) != -1) {
    if (line[0] == '#') continue;
    if (line[read-1] == '\n') line[read-1] = '\0';
    int s=0; // Env Var Size
    while (line[s] != '\0' && line[s] != '=') s++;
    if (line[s] == '\0') continue;
    strncpy(envVar, line, std::min(1023,s));
    envVar[std::min(1023,s)] = '\0';
    s++;
    strncpy(envValue, line+s, 1023);
    envValue[1023]='\0';
    ncclOsSetEnv(envVar, envValue);
    //printf("%s : %s->%s\n", fileName, envVar, envValue);
  }
  if (line) free(line);
  fclose(file);
}

static void initEnvFunc() {
  char confFilePath[1024];
  const char* userFile = std::getenv("NCCL_CONF_FILE");
  if (userFile && strlen(userFile) > 0) {
    snprintf(confFilePath, sizeof(confFilePath), "%s", userFile);
    setEnvFile(confFilePath);
  } else {
    const char* userDir = userHomeDir();
    if (userDir) {
      snprintf(confFilePath, sizeof(confFilePath), "%s/.nccl.conf", userDir);
      setEnvFile(confFilePath);
    }
  }
  snprintf(confFilePath, sizeof(confFilePath), "/etc/nccl.conf");
  setEnvFile(confFilePath);
}

void initEnv() {
  static std::once_flag once;
  std::call_once(once, initEnvFunc);
}

static std::unordered_set<std::string> noCacheSet;
static bool noCacheAll = false;

static void ncclGetEnvNoCacheOnce() {
  const char* envNoCache = ncclGetEnv("NCCL_NO_CACHE");
  if (envNoCache == NULL || strlen(envNoCache) == 0) return;

  char* copy = strdup(envNoCache);
  char* token = strtok(copy, ",");
  while (token != NULL) {
    if (strcmp(token, "ALL") == 0) {
      noCacheAll = true;
      break;
    } else {
      noCacheSet.insert(token);
    }
    token = strtok(NULL, ",");
  }
  free(copy);
}

static void ncclGetCachePolicy(char const* env, int8_t* noCache) {
  *noCache = (noCacheAll || noCacheSet.count(env) > 0) ? /*noCache*/ 1 : /*cache*/ 0;
  if (*noCache) INFO(NCCL_ENV, "Disabling caching for environment variable %s.", env);
}

int64_t ncclLoadParam(char const* env, int64_t deftVal, int64_t uninitialized, int64_t* cache, int8_t* noCache) {
  static std::once_flag once;
  std::call_once(once, ncclGetEnvNoCacheOnce);

  static std::mutex mutex;
  std::lock_guard<std::mutex> lock(mutex);

  // noCache is only load/stored within the mutex, no need for atomic
  if (*noCache == /*uninitialized*/ -1) ncclGetCachePolicy(env, noCache);

  if (COMPILER_ATOMIC_LOAD(cache, std::memory_order_relaxed) != uninitialized) return COMPILER_ATOMIC_LOAD(cache, std::memory_order_relaxed);

  // Read the environment variable
  const char* str = ncclGetEnv(env);
  int64_t value = deftVal;

  if (str && strlen(str) > 0) {
    errno = 0;
    value = strtoll(str, nullptr, 0);
    if (errno) {
      value = deftVal;
      INFO(NCCL_ALL, "Invalid value %s for %s, using default %lld.", str, env, (long long)deftVal);
    } else {
      INFO(NCCL_ENV, "%s set by environment to %lld.", env, (long long)value);
    }
  }

  if (*noCache == /*cache*/ 0) COMPILER_ATOMIC_STORE(cache, value, std::memory_order_relaxed);
  return value;
}

const char* ncclGetEnv(const char* name) {
  ncclInitEnv();
  return ncclEnvPluginGetEnv(name);
}
