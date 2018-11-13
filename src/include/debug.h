/*************************************************************************
 * Copyright (c) 2015-2018, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_DEBUG_H_
#define NCCL_DEBUG_H_

#include <pthread.h>
#include <stdio.h>
#include <chrono>

#include <unistd.h>
#include <sys/syscall.h>
#include <limits.h>
#include <string.h>
#include "nccl.h"
#include "nccl_net.h"

#define gettid() (pid_t) syscall(SYS_gettid)

extern int ncclDebugLevel;
extern uint64_t ncclDebugMask;
extern pthread_mutex_t ncclDebugOutputLock;
extern FILE *ncclDebugFile;
extern ncclResult_t getHostName(char* hostname, int maxlen);

extern void ncclDebugLog(ncclDebugLogLevel level, unsigned long flags, const char *filefunc, int line, const char *fmt, ...);

#define WARN(...) ncclDebugLog(NCCL_LOG_WARN, NCCL_ALL, __FILE__, __LINE__, __VA_ARGS__)
#define INFO(FLAGS, ...) ncclDebugLog(NCCL_LOG_INFO, (FLAGS), __func__, __LINE__, __VA_ARGS__)

#ifdef ENABLE_TRACE
#define TRACE(FLAGS, ...) ncclDebugLog(NCCL_LOG_TRACE, (FLAGS), __func__, __LINE__, __VA_ARGS__)
extern std::chrono::high_resolution_clock::time_point ncclEpoch;
#else
#define TRACE(...)
#endif

#include <stdlib.h>

static inline void initDebug() {
  const char* nccl_debug = getenv("NCCL_DEBUG");
  if (nccl_debug == NULL) {
    ncclDebugLevel = NCCL_LOG_NONE;
  } else if (strcasecmp(nccl_debug, "VERSION") == 0) {
    ncclDebugLevel = NCCL_LOG_VERSION;
  } else if (strcasecmp(nccl_debug, "WARN") == 0) {
    ncclDebugLevel = NCCL_LOG_WARN;
  } else if (strcasecmp(nccl_debug, "INFO") == 0) {
    ncclDebugLevel = NCCL_LOG_INFO;
  } else if (strcasecmp(nccl_debug, "ABORT") == 0) {
    ncclDebugLevel = NCCL_LOG_ABORT;
  } else if (strcasecmp(nccl_debug, "TRACE") == 0) {
    ncclDebugLevel = NCCL_LOG_TRACE;
  }

  /* Parse the NCCL_DEBUG_SUBSYS env var
   * This can be a comma separated list such as INIT,COLL
   * or ^INIT,COLL etc
   */
  char* nccl_debug_subsys = getenv("NCCL_DEBUG_SUBSYS");
  if (nccl_debug_subsys != NULL) {
    char *subsys = strtok(nccl_debug_subsys, ",");
    while (subsys != NULL) {
      int invert = 0;
      uint64_t mask = 0;
      if (subsys[0] == '^') { invert = 1; subsys++; }
      if (strcasecmp(subsys, "INIT") == 0) {
        mask = NCCL_INIT;
      } else if (strcasecmp(subsys, "COLL") == 0) {
        mask = NCCL_COLL;
      } else if (strcasecmp(subsys, "P2P") == 0) {
        mask = NCCL_P2P;
      } else if (strcasecmp(subsys, "SHM") == 0) {
        mask = NCCL_SHM;
      } else if (strcasecmp(subsys, "NET") == 0) {
        mask = NCCL_NET;
      } else if (strcasecmp(subsys, "ALL") == 0) {
        mask = NCCL_ALL;
      }
      if (mask) {
        if (invert) ncclDebugMask &= ~mask; else ncclDebugMask |= mask;
      }
      subsys = strtok(NULL, ",");
    }
  }

  /* Parse and expand the NCCL_DEBUG_FILE path and
   * then create the debug file. But don't bother unless the
   * NCCL_DEBUG level is > VERSION
   */
  const char* nccl_debug_file = getenv("NCCL_DEBUG_FILE");
  if (ncclDebugLevel > NCCL_LOG_VERSION && nccl_debug_file != NULL) {
    int c = 0;
    char debug_fn[PATH_MAX+1] = "";
    char *dfn = debug_fn;
    while (nccl_debug_file[c] != '\0' && c < PATH_MAX) {
      if (nccl_debug_file[c++] != '%') {
        *dfn++ = nccl_debug_file[c-1];
        continue;
      }
      switch (nccl_debug_file[c++]) {
        case '%': // Double %
          *dfn++ = '%';
          break;
        case 'h': // %h = hostname
          char hostname[1024];
          getHostName(hostname, 1024);
          dfn += snprintf(dfn, PATH_MAX, "%s", hostname);
          break;
        case 'p': // %p = pid
          dfn += snprintf(dfn, PATH_MAX, "%d", getpid());
          break;
        default: // Echo everything we don't understand
          *dfn++ = '%';
          *dfn++ = nccl_debug_file[c-1];
          break;
      }
    }
    *dfn = '\0';
    if (debug_fn[0] != '\0') {
      FILE *file = fopen(debug_fn, "w");
      if (file != NULL) {
        INFO(NCCL_ALL,"DEBUG file is '%s'", debug_fn);
        ncclDebugFile = file;
      }
    }
  }
  pthread_mutex_init(&ncclDebugOutputLock, NULL);

#ifdef ENABLE_TRACE
  ncclEpoch = std::chrono::high_resolution_clock::now();
#endif
}

#endif
