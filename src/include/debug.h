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
#define gettid() (pid_t) syscall(SYS_gettid)

typedef enum {NONE=0, VERSION=1, WARN=2, INFO=3, ABORT=4, TRACE=5} DebugLevel;
typedef enum {INIT=1, COLL=2, P2P=4, SHM=8, NET=16, ALL=~0} SubSys;
extern DebugLevel ncclDebugLevel;
extern uint64_t ncclDebugMask;
extern pthread_mutex_t ncclDebugOutputLock;
extern FILE *ncclDebugFile;
extern ncclResult_t getHostName(char* hostname, int maxlen);

#define WARN(...) do {                                           \
  if (ncclDebugLevel >= WARN) {                                  \
    char hostname[1024];                                         \
    getHostName(hostname, 1024);                                 \
    int cudaDev;                                                 \
    cudaGetDevice(&cudaDev);                                     \
    pthread_mutex_lock(&ncclDebugOutputLock);                    \
    fprintf(ncclDebugFile,"\n%s:%d:%d [%d] %s:%d NCCL WARN ", hostname, getpid(), gettid(), cudaDev, __FILE__, __LINE__); \
    fprintf(ncclDebugFile,__VA_ARGS__);                          \
    fprintf(ncclDebugFile,"\n");                                 \
    fflush(ncclDebugFile);                                       \
    pthread_mutex_unlock(&ncclDebugOutputLock);                  \
    if (ncclDebugLevel == ABORT) { fprintf(stderr,"\n%s:%d:%d [%d] %s:%d NCCL ABORT\n", hostname, getpid(), gettid(), cudaDev, __FILE__, __LINE__); abort(); } \
  }                                                              \
} while(0)

#define INFO(FLAGS, ...) do {                                    \
  if (ncclDebugLevel >= INFO && ((FLAGS) & ncclDebugMask)) {     \
    char hostname[1024];                                         \
    getHostName(hostname, 1024);                                 \
    int cudaDev;                                                 \
    cudaGetDevice(&cudaDev);                                     \
    pthread_mutex_lock(&ncclDebugOutputLock);                    \
    fprintf(ncclDebugFile,"%s:%d:%d [%d] NCCL INFO ", hostname, getpid(), gettid(), cudaDev); \
    fprintf(ncclDebugFile,__VA_ARGS__);fprintf(ncclDebugFile,"\n"); \
    fflush(ncclDebugFile);                                       \
    pthread_mutex_unlock(&ncclDebugOutputLock);                  \
  }                                                              \
} while(0)

#ifdef ENABLE_TRACE
#define TRACE(FLAGS, ...) do {                                   \
  if (ncclDebugLevel == TRACE && ((FLAGS) & ncclDebugMask)) {    \
    char hostname[1024];                                         \
    getHostName(hostname, 1024);                                 \
    int cudaDev;                                                 \
    cudaGetDevice(&cudaDev);                                     \
    pthread_mutex_lock(&ncclDebugOutputLock);                    \
    auto delta = std::chrono::high_resolution_clock::now() - ncclEpoch; \
    double timestamp = std::chrono::duration_cast<std::chrono::duration<double>>(delta).count()*1000; \
    fprintf(ncclDebugFile,"%s:%d:%d [%d] %f %s:%d NCCL TRACE ", hostname, getpid(), gettid(), cudaDev, timestamp, __func__, __LINE__); \
    fprintf(ncclDebugFile,__VA_ARGS__);fprintf(ncclDebugFile,"\n"); \
    fflush(ncclDebugFile);                                       \
    pthread_mutex_unlock(&ncclDebugOutputLock);                  \
  }                                                              \
} while(0)

extern std::chrono::high_resolution_clock::time_point ncclEpoch;

#else
#define TRACE(...)
#endif

#include <stdlib.h>

static inline void initDebug() {
  const char* nccl_debug = getenv("NCCL_DEBUG");
  if (nccl_debug == NULL) {
    ncclDebugLevel = NONE;
  } else if (strcasecmp(nccl_debug, "VERSION") == 0) {
    ncclDebugLevel = VERSION;
  } else if (strcasecmp(nccl_debug, "WARN") == 0) {
    ncclDebugLevel = WARN;
  } else if (strcasecmp(nccl_debug, "INFO") == 0) {
    ncclDebugLevel = INFO;
  } else if (strcasecmp(nccl_debug, "ABORT") == 0) {
    ncclDebugLevel = ABORT;
  } else if (strcasecmp(nccl_debug, "TRACE") == 0) {
    ncclDebugLevel = TRACE;
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
        mask = INIT;
      } else if (strcasecmp(subsys, "COLL") == 0) {
        mask = COLL;
      } else if (strcasecmp(subsys, "P2P") == 0) {
        mask = P2P;
      } else if (strcasecmp(subsys, "SHM") == 0) {
        mask = SHM;
      } else if (strcasecmp(subsys, "NET") == 0) {
        mask = NET;
      } else if (strcasecmp(subsys, "ALL") == 0) {
        mask = ALL;
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
  if (ncclDebugLevel > VERSION && nccl_debug_file != NULL) {
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
        INFO(ALL,"DEBUG file is '%s'", debug_fn);
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
