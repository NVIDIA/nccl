/*************************************************************************
 * Copyright (c) 2016-2019, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "core.h"
#include "nccl_net.h"
#include <stdlib.h>
#include <stdarg.h>

int ncclDebugLevel = -1;
thread_local int ncclDebugNoWarn = 0;
uint64_t ncclDebugMask = NCCL_INIT; // Default debug sub-system mask is INIT
FILE *ncclDebugFile = stdout;
pthread_mutex_t ncclDebugLock = PTHREAD_MUTEX_INITIALIZER;

void ncclDebugInit() {
  pthread_mutex_lock(&ncclDebugLock);
  if (ncclDebugLevel != -1) return;
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
  char* ncclDebugSubsysEnv = getenv("NCCL_DEBUG_SUBSYS");
  if (ncclDebugSubsysEnv != NULL) {
    int invert = 0;
    if (ncclDebugSubsysEnv[0] == '^') { invert = 1; ncclDebugSubsysEnv++; }
    ncclDebugMask = invert ? ~0ULL : 0ULL;
    char *ncclDebugSubsys = strdup(ncclDebugSubsysEnv);
    char *subsys = strtok(ncclDebugSubsys, ",");
    while (subsys != NULL) {
      uint64_t mask = 0;
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
      } else if (strcasecmp(subsys, "GRAPH") == 0) {
        mask = NCCL_GRAPH;
      } else if (strcasecmp(subsys, "TUNING") == 0) {
        mask = NCCL_TUNING;
      } else if (strcasecmp(subsys, "ALL") == 0) {
        mask = NCCL_ALL;
      }
      if (mask) {
        if (invert) ncclDebugMask &= ~mask; else ncclDebugMask |= mask;
      }
      subsys = strtok(NULL, ",");
    }
    free(ncclDebugSubsys);
  }

  /* Parse and expand the NCCL_DEBUG_FILE path and
   * then create the debug file. But don't bother unless the
   * NCCL_DEBUG level is > VERSION
   */
  const char* ncclDebugFileEnv = getenv("NCCL_DEBUG_FILE");
  if (ncclDebugLevel > NCCL_LOG_VERSION && ncclDebugFileEnv != NULL) {
    int c = 0;
    char debugFn[PATH_MAX+1] = "";
    char *dfn = debugFn;
    while (ncclDebugFileEnv[c] != '\0' && c < PATH_MAX) {
      if (ncclDebugFileEnv[c++] != '%') {
        *dfn++ = ncclDebugFileEnv[c-1];
        continue;
      }
      switch (ncclDebugFileEnv[c++]) {
        case '%': // Double %
          *dfn++ = '%';
          break;
        case 'h': // %h = hostname
          char hostname[1024];
          getHostName(hostname, 1024, '.');
          dfn += snprintf(dfn, PATH_MAX, "%s", hostname);
          break;
        case 'p': // %p = pid
          dfn += snprintf(dfn, PATH_MAX, "%d", getpid());
          break;
        default: // Echo everything we don't understand
          *dfn++ = '%';
          *dfn++ = ncclDebugFileEnv[c-1];
          break;
      }
    }
    *dfn = '\0';
    if (debugFn[0] != '\0') {
      FILE *file = fopen(debugFn, "w");
      if (file != NULL) {
        INFO(NCCL_ALL,"DEBUG file is '%s'", debugFn);
        ncclDebugFile = file;
      }
    }
  }

#ifdef ENABLE_TRACE
  ncclEpoch = std::chrono::high_resolution_clock::now();
#endif
  pthread_mutex_unlock(&ncclDebugLock);
}

/* Common logging function used by the INFO, WARN and TRACE macros
 * Also exported to the dynamically loadable Net transport modules so
 * they can share the debugging mechanisms and output files
 */
void ncclDebugLog(ncclDebugLogLevel level, unsigned long flags, const char *filefunc, int line, const char *fmt, ...) {
  if (ncclDebugLevel == -1) ncclDebugInit();
  if (ncclDebugNoWarn == 1 && level == NCCL_LOG_WARN) level = NCCL_LOG_INFO;

  char hostname[1024];
  getHostName(hostname, 1024, '.');
  int cudaDev;
  cudaGetDevice(&cudaDev);

  char buffer[1024];
  size_t len = 0;
  pthread_mutex_lock(&ncclDebugLock);
  if (ncclDebugNoWarn && ncclDebugLevel == NCCL_LOG_WARN) printf("WARN -> INFO\n");
  if (level == NCCL_LOG_WARN && ncclDebugLevel >= NCCL_LOG_WARN)
    len = snprintf(buffer, sizeof(buffer),
                   "\n%s:%d:%d [%d] %s:%d NCCL WARN ", hostname, getpid(), gettid(), cudaDev, filefunc, line);
  else if (level == NCCL_LOG_INFO && ncclDebugLevel >= NCCL_LOG_INFO && (flags & ncclDebugMask))
    len = snprintf(buffer, sizeof(buffer),
                   "%s:%d:%d [%d] NCCL INFO ", hostname, getpid(), gettid(), cudaDev);
#ifdef ENABLE_TRACE
  else if (level == NCCL_LOG_TRACE && ncclDebugLevel >= NCCL_LOG_TRACE && (flags & ncclDebugMask)) {
    auto delta = std::chrono::high_resolution_clock::now() - ncclEpoch;
    double timestamp = std::chrono::duration_cast<std::chrono::duration<double>>(delta).count()*1000;
    len = snprintf(buffer, sizeof(buffer),
                   "%s:%d:%d [%d] %f %s:%d NCCL TRACE ", hostname, getpid(), gettid(), cudaDev, timestamp, filefunc, line);
  }
#endif
  if (len) {
    va_list vargs;
    va_start(vargs, fmt);
    (void) vsnprintf(buffer+len, sizeof(buffer)-len, fmt, vargs);
    va_end(vargs);
    fprintf(ncclDebugFile,"%s\n", buffer);
    fflush(ncclDebugFile);
  }
  pthread_mutex_unlock(&ncclDebugLock);

  // If ncclDebugLevel == NCCL_LOG_ABORT then WARN() will also call abort()
  if (level == NCCL_LOG_WARN && ncclDebugLevel == NCCL_LOG_ABORT) {
    fprintf(stderr,"\n%s:%d:%d [%d] %s:%d NCCL ABORT\n",
            hostname, getpid(), gettid(), cudaDev, filefunc, line);
    abort();
  }
}
