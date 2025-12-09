/*************************************************************************
 * Copyright (c) 2019-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "param.h"
#include "debug.h"

#include <algorithm>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <unistd.h>
#include <pthread.h>
#include <pwd.h>
#include <mutex>
#include <folly/init/Init.h>
#include <folly/logging/Init.h>
#include <folly/logging/xlog.h>
#include <folly/Singleton.h>
#include <folly/synchronization/HazptrThreadPoolExecutor.h>

std::once_flag initOnceFlag;

// From meta/logger
extern void initLogger();
extern void ncclCvarInit();

//using namespace meta::comms::colltrace;

void initFolly() {
  // Adapted from folly/init/Init.cpp
  // We can't use folly::init directly because:
  // - we don't have gflags
  // - xlformers already initialized the signal handler

  // Move from the registration phase to the "you can actually instantiate
  // things now" phase.
  folly::SingletonVault::singleton()->registrationComplete();

  auto const follyLoggingEnv = std::getenv(folly::kLoggingEnvVarName);
  auto const follyLoggingEnvOr = follyLoggingEnv ? follyLoggingEnv : "";
  folly::initLoggingOrDie(follyLoggingEnvOr);

  // Set the default hazard pointer domain to use a thread pool executor
  // for asynchronous reclamation
  folly::enable_hazptr_thread_pool_executor();
}

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
    setenv(envVar, envValue, 0);
    //printf("%s : %s->%s\n", fileName, envVar, envValue);
  }
  if (line) free(line);
  fclose(file);
}

static void initEnvFunc() {
  char confFilePath[1024];
  const char* userFile = getenv("NCCL_CONF_FILE");
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
  std::call_once(initOnceFlag, [] {
    initFolly();
    initEnvFunc();
    ncclCvarInit();
    initLogger();
  });
}

void ncclLoadParam(char const* env, int64_t deftVal, int64_t uninitialized, int64_t* cache) {
  static pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
  pthread_mutex_lock(&mutex);
  if (__atomic_load_n(cache, __ATOMIC_RELAXED) == uninitialized) {
    const char* str = ncclGetEnv(env);
    int64_t value = deftVal;
    if (str && strlen(str) > 0) {
      errno = 0;
      value = strtoll(str, nullptr, 0);
      if (errno) {
        value = deftVal;
        INFO(NCCL_ALL,"Invalid value %s for %s, using default %lld.", str, env, (long long)deftVal);
      } else {
        INFO(NCCL_ENV,"%s set by environment to %lld.", env, (long long)value);
      }
    }
    __atomic_store_n(cache, value, __ATOMIC_RELAXED);
  }
  pthread_mutex_unlock(&mutex);
}

const char* ncclGetEnv(const char* name) {
  initEnv();
  return getenv(name);
}
