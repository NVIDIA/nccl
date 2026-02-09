/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "core.h"
#include "nccl_net.h"
#include <ctime>
#include <stdlib.h>
#include <stdarg.h>
#include <stdio.h>
#include <string.h>
#include <strings.h>
#include <sys/syscall.h>
#include <chrono>
#include "param.h"
#include <mutex>
#include "os.h"
#include "env.h"

#define NCCL_DEBUG_RESET_TRIGGERED (-2)

int ncclDebugLevel = -1;
static uint32_t ncclDebugTimestampLevels = 0;     // bitmaps of levels that have timestamps turned on
static char ncclDebugTimestampFormat[256];        // with space for subseconds
static int ncclDebugTimestampSubsecondsStart;     // index where the subseconds starts
static uint64_t ncclDebugTimestampMaxSubseconds;  // Max number of subseconds plus 1, used in duration ratio
static int ncclDebugTimestampSubsecondDigits;     // Number of digits to display
static int pid = -1;
static char hostname[1024];
thread_local int ncclDebugNoWarn = 0;
char ncclLastError[1024] = ""; // Global string for the last error in human readable form
uint64_t ncclDebugMask = 0;
FILE *ncclDebugFile = stdout;
static std::mutex ncclDebugMutex;
static std::chrono::steady_clock::time_point ncclEpoch;
static bool ncclWarnSetDebugInfo = false;

static thread_local int tid = -1;

typedef const char* (*ncclGetEnvFunc_t)(const char*);

static ncclResult_t getHostNameForLog(char* hostname, int maxlen, const char delim) {
  ncclResult_t ret = getHostName(hostname, maxlen, delim);
  if (ret != ncclSuccess) return ret;

  for (int i = 0; i < maxlen-1 && hostname[i]; ++i) {
    // Replace special characters in hostnames with dashes
    switch (hostname[i]) {
    case '%':
    case '/':
      hostname[i] = '-';
      break;
    default:
      break;
    }
  }
  return ncclSuccess;
}

// This function must be called with ncclDebugLock locked!
static void ncclDebugInit() {
  ncclGetEnvFunc_t getEnvFunc = ncclEnvPluginInitialized() ? ncclGetEnv : (ncclGetEnvFunc_t)std::getenv;
  const char* nccl_debug = getEnvFunc("NCCL_DEBUG");
  int tempNcclDebugLevel = -1;
  uint64_t tempNcclDebugMask = NCCL_INIT | NCCL_BOOTSTRAP | NCCL_ENV; // Default debug sub-system mask
  if (ncclDebugLevel == NCCL_DEBUG_RESET_TRIGGERED && ncclDebugFile != stdout) {
    // Finish the reset initiated via ncclResetDebugInit().
    fclose(ncclDebugFile);
    ncclDebugFile = stdout;
  }

  if (nccl_debug == NULL) {
    tempNcclDebugLevel = NCCL_LOG_NONE;
  } else if (strcasecmp(nccl_debug, "VERSION") == 0) {
    tempNcclDebugLevel = NCCL_LOG_VERSION;
  } else if (strcasecmp(nccl_debug, "WARN") == 0) {
    tempNcclDebugLevel = NCCL_LOG_WARN;
  } else if (strcasecmp(nccl_debug, "INFO") == 0) {
    tempNcclDebugLevel = NCCL_LOG_INFO;
  } else if (strcasecmp(nccl_debug, "ABORT") == 0) {
    tempNcclDebugLevel = NCCL_LOG_ABORT;
  } else if (strcasecmp(nccl_debug, "TRACE") == 0) {
    tempNcclDebugLevel = NCCL_LOG_TRACE;
  }

  /* Parse the NCCL_DEBUG_SUBSYS env var
   * This can be a comma separated list such as INIT,COLL
   * or ^INIT,COLL etc
   */
  const char* ncclDebugSubsysEnv = getEnvFunc("NCCL_DEBUG_SUBSYS");
  if (ncclDebugSubsysEnv != NULL) {
    int invert = 0;
    if (ncclDebugSubsysEnv[0] == '^') { invert = 1; ncclDebugSubsysEnv++; }
    tempNcclDebugMask = invert ? ~0ULL : 0ULL;
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
      } else if (strcasecmp(subsys, "ENV") == 0) {
        mask = NCCL_ENV;
      } else if (strcasecmp(subsys, "ALLOC") == 0) {
        mask = NCCL_ALLOC;
      } else if (strcasecmp(subsys, "CALL") == 0) {
        mask = NCCL_CALL;
      } else if (strcasecmp(subsys, "PROXY") == 0) {
        mask = NCCL_PROXY;
      } else if (strcasecmp(subsys, "NVLS") == 0) {
        mask = NCCL_NVLS;
      } else if (strcasecmp(subsys, "BOOTSTRAP") == 0) {
        mask = NCCL_BOOTSTRAP;
      } else if (strcasecmp(subsys, "REG") == 0) {
        mask = NCCL_REG;
      } else if (strcasecmp(subsys, "PROFILE") == 0) {
        mask = NCCL_PROFILE;
      } else if (strcasecmp(subsys, "RAS") == 0) {
        mask = NCCL_RAS;
      } else if (strcasecmp(subsys, "ALL") == 0) {
        mask = NCCL_ALL;
      }
      if (mask) {
        if (invert) tempNcclDebugMask &= ~mask; else tempNcclDebugMask |= mask;
      }
      subsys = strtok(NULL, ",");
    }
    free(ncclDebugSubsys);
  }

  const char* ncclWarnSetDebugInfoEnv = getEnvFunc("NCCL_WARN_ENABLE_DEBUG_INFO");
  if (ncclWarnSetDebugInfoEnv != NULL && strlen(ncclWarnSetDebugInfoEnv) > 0) {
    int64_t value;
    errno = 0;
    value = strtoll(ncclWarnSetDebugInfoEnv, NULL, 0);
    if (!errno)
      ncclWarnSetDebugInfo = value;
  }

  // Determine which debug levels will have timestamps.
  const char* timestamps = getEnvFunc("NCCL_DEBUG_TIMESTAMP_LEVELS");
  if (timestamps == nullptr) {
    ncclDebugTimestampLevels = (1<<NCCL_LOG_WARN);
  } else {
    int invert = 0;
    if (timestamps[0] == '^') { invert = 1; ++timestamps; }
    ncclDebugTimestampLevels = invert ? ~0U : 0U;
    char *timestampsDup = strdup(timestamps);
    char *level = strtok(timestampsDup, ",");
    while (level != NULL) {
      uint32_t mask = 0;
      if (strcasecmp(level, "ALL") == 0) {
        mask = ~0U;
      } else if (strcasecmp(level, "VERSION") == 0) {
        mask = (1<<NCCL_LOG_VERSION);
      } else if (strcasecmp(level, "WARN") == 0) {
        mask = (1<<NCCL_LOG_WARN);
      } else if (strcasecmp(level, "INFO") == 0) {
        mask = (1<<NCCL_LOG_INFO);
      } else if (strcasecmp(level, "ABORT") == 0) {
        mask = (1<<NCCL_LOG_ABORT);
      } else if (strcasecmp(level, "TRACE") == 0) {
        mask = (1<<NCCL_LOG_TRACE);
      } else {
        // Silently fail.
      }
      if (mask) {
        if (invert) ncclDebugTimestampLevels &= ~mask;
        else ncclDebugTimestampLevels |= mask;
      }
      level = strtok(NULL, ",");
    }
    free(timestampsDup);
  }

  // Store a copy of the timestamp format with space for the subseconds, if used.
  const char* tsFormat = getEnvFunc("NCCL_DEBUG_TIMESTAMP_FORMAT");
  if (tsFormat == nullptr) tsFormat = "[%F %T] ";
  ncclDebugTimestampSubsecondsStart = -1;
  // Find where the subseconds are in the format.
  for (int i=0; tsFormat[i] != '\0'; ++i) {
    if (tsFormat[i]=='%' && tsFormat[i+1]=='%') { // Next two chars are "%"
      // Skip the next character, too, and restart checking after that.
      ++i;
      continue;
    }
    if (tsFormat[i]=='%' &&                               // Found a percentage
        ('1' <= tsFormat[i+1] && tsFormat[i+1] <= '9') && // Next char is a digit between 1 and 9 inclusive
        tsFormat[i+2]=='f'                                // Two characters later is an "f"
        ) {
      constexpr int replaceLen = sizeof("%Xf") - 1;
      ncclDebugTimestampSubsecondDigits = tsFormat[i+1] - '0';
      if (ncclDebugTimestampSubsecondDigits + strlen(tsFormat) - replaceLen > sizeof(ncclDebugTimestampFormat) - 1) {
        // Won't fit; fall back on the default.
        break;
      }
      ncclDebugTimestampSubsecondsStart = i;
      ncclDebugTimestampMaxSubseconds = 1;

      memcpy(ncclDebugTimestampFormat, tsFormat, i);
      for (int j=0; j<ncclDebugTimestampSubsecondDigits; ++j) {
        ncclDebugTimestampFormat[i+j] = ' ';
        ncclDebugTimestampMaxSubseconds *= 10;
      }
      strcpy(ncclDebugTimestampFormat+i+ncclDebugTimestampSubsecondDigits, tsFormat+i+replaceLen);
      break;
    }
  }
  if (ncclDebugTimestampSubsecondsStart == -1) {
    if (strlen(tsFormat) < sizeof(ncclDebugTimestampFormat)) {
      strcpy(ncclDebugTimestampFormat, tsFormat);
    } else {
      strcpy(ncclDebugTimestampFormat, "[%F %T] ");
    }
  }

  // Replace underscore with spaces... it is hard to put spaces in command line parameters.
  for (int i=0; ncclDebugTimestampFormat[i] != '\0'; ++i) {
    if (ncclDebugTimestampFormat[i]=='_') ncclDebugTimestampFormat[i] = ' ';
  }

  // Cache pid and hostname
  getHostNameForLog(hostname, 1024, '.');
  pid = ncclOsGetpid();

  /* Parse and expand the NCCL_DEBUG_FILE path and
   * then create the debug file. But don't bother unless the
   * NCCL_DEBUG level is > VERSION
   */
  const char* ncclDebugFileEnv = getEnvFunc("NCCL_DEBUG_FILE");
  if (tempNcclDebugLevel > NCCL_LOG_VERSION && ncclDebugFileEnv != NULL) {
    int c = 0;
    char debugFn[PATH_MAX+1] = "";
    char *dfn = debugFn;
    while (ncclDebugFileEnv[c] != '\0' && (dfn - debugFn) < PATH_MAX) {
      if (ncclDebugFileEnv[c++] != '%') {
        *dfn++ = ncclDebugFileEnv[c-1];
        continue;
      }
      switch (ncclDebugFileEnv[c++]) {
        case '%': // Double %
          *dfn++ = '%';
          break;
        case 'h': // %h = hostname
          dfn += snprintf(dfn, PATH_MAX + 1 - (dfn - debugFn), "%s", hostname);
          break;
        case 'p': // %p = pid
          dfn += snprintf(dfn, PATH_MAX + 1 - (dfn - debugFn), "%d", pid);
          break;
        default: // Echo everything we don't understand
          *dfn++ = '%';
          if ((dfn - debugFn) < PATH_MAX) {
            *dfn++ = ncclDebugFileEnv[c-1];
          }
          break;
      }
      if ((dfn - debugFn) > PATH_MAX) {
        // snprintf wanted to overfill the buffer: set dfn to the end
        // of the buffer (for null char) and it will naturally exit
        // the loop.
        dfn = debugFn + PATH_MAX;
      }
    }
    *dfn = '\0';
    if (debugFn[0] != '\0') {
      FILE *file = fopen(debugFn, "w");
      if (file != nullptr) {
        setlinebuf(file); // disable block buffering
        ncclDebugFile = file;
      }
    }
  }

  ncclEpoch = std::chrono::steady_clock::now();
  ncclDebugMask = tempNcclDebugMask;
  COMPILER_ATOMIC_STORE(&ncclDebugLevel, tempNcclDebugLevel, std::memory_order_release);
}

/* Common logging function used by the INFO, WARN and TRACE macros
 * Also exported to the dynamically loadable Net transport modules so
 * they can share the debugging mechanisms and output files
 */
void ncclDebugLog(ncclDebugLogLevel level, unsigned long flags, const char *filefunc, int line, const char *fmt, ...) {
  int gotLevel = COMPILER_ATOMIC_LOAD(&ncclDebugLevel, std::memory_order_acquire);

  if (ncclDebugNoWarn != 0 && level == NCCL_LOG_WARN) { level = NCCL_LOG_INFO; flags = ncclDebugNoWarn; }

  // Save the last error (WARN) as a human readable string
  if (level == NCCL_LOG_WARN) {
    std::lock_guard<std::mutex> lock(ncclDebugMutex);
    va_list vargs;
    va_start(vargs, fmt);
    (void) vsnprintf(ncclLastError, sizeof(ncclLastError), fmt, vargs);
    va_end(vargs);
  }

  if (gotLevel >= 0 && (gotLevel < level || (flags & ncclDebugMask) == 0)) {
    return;
  }

  std::lock_guard<std::mutex> lock(ncclDebugMutex);
  if (ncclDebugLevel < 0)
    ncclDebugInit();
  if (ncclDebugLevel < level || ((flags & ncclDebugMask) == 0)) {
    return;
  }

  if (tid == -1) {
    tid = syscall(SYS_gettid);
  }

  char buffer[1024];
  size_t len = 0;

  // WARNs come with an extra newline at the beginning.
  if (level == NCCL_LOG_WARN) {
    buffer[len++] = '\n';
  };

  // Add the timestamp to the buffer if they are turned on for this level.
  if (ncclDebugTimestampLevels & (1<<level)) {
    if (ncclDebugTimestampFormat[0] != '\0') {
      struct timespec ts;
      clock_gettime(CLOCK_REALTIME, &ts);   // clock_gettime failure should never happen
      std::tm nowTm;
      localtime_r(&ts.tv_sec, &nowTm);

      // Add the subseconds portion if it is part of the format.
      char localTimestampFormat[sizeof(ncclDebugTimestampFormat)];
      const char* pformat = ncclDebugTimestampFormat;
      if (ncclDebugTimestampSubsecondsStart != -1) {
        pformat = localTimestampFormat;   // Need to use the local version which has subseconds
        memcpy(localTimestampFormat, ncclDebugTimestampFormat, ncclDebugTimestampSubsecondsStart);
        snprintf(localTimestampFormat + ncclDebugTimestampSubsecondsStart,
                 ncclDebugTimestampSubsecondDigits+1,
                 "%0*ld", ncclDebugTimestampSubsecondDigits,
                 ts.tv_nsec / (1000000000UL/ncclDebugTimestampMaxSubseconds));
        strcpy(    localTimestampFormat+ncclDebugTimestampSubsecondsStart+ncclDebugTimestampSubsecondDigits,
               ncclDebugTimestampFormat+ncclDebugTimestampSubsecondsStart+ncclDebugTimestampSubsecondDigits);
      }

      // Format the time. If it runs out of space, fall back on a simpler format.
      int adv = std::strftime(buffer+len, sizeof(buffer)-len, pformat, &nowTm);
      if (adv==0 && ncclDebugTimestampFormat[0] != '\0') {
        // Ran out of space. Fall back on the default. This should never fail.
        adv = std::strftime(buffer+len, sizeof(buffer)-len, "[%F %T] ", &nowTm);
      }
      len += adv;
    }
  }
  len = std::min(len, sizeof(buffer)-1);  // prevent overflows

  // Add hostname, pid and tid portion of the log line.
  if (level != NCCL_LOG_VERSION) {
    len += snprintf(buffer+len, sizeof(buffer)-len, "%s:%d:%d ", hostname, pid, tid);
    len = std::min(len, sizeof(buffer)-1);  // prevent overflows
  }

  int cudaDev = 0;
  if (!(level == NCCL_LOG_TRACE && flags == NCCL_CALL)) {
    (void)cudaGetDevice(&cudaDev);
  }

  // Add level specific formatting. The format string from the call site is incorporated into this prefix.
  if (level == NCCL_LOG_WARN) {
    len += snprintf(buffer+len, sizeof(buffer)-len, "[%d] %s:%d NCCL WARN %s\n", cudaDev, filefunc, line, fmt);
    if (ncclWarnSetDebugInfo) COMPILER_ATOMIC_STORE(&ncclDebugLevel, NCCL_LOG_INFO, std::memory_order_release);
  } else if (level == NCCL_LOG_INFO) {
    len += snprintf(buffer+len, sizeof(buffer)-len, "[%d] NCCL INFO %s\n", cudaDev, fmt);
  } else if (level == NCCL_LOG_TRACE && flags == NCCL_CALL) {
    len += snprintf(buffer+len, sizeof(buffer)-len, "NCCL CALL %s\n", fmt);
  } else if (level == NCCL_LOG_TRACE) {
    auto delta = std::chrono::steady_clock::now() - ncclEpoch;
    double timestamp = std::chrono::duration_cast<std::chrono::duration<double>>(delta).count()*1000;
    len += snprintf(buffer+len, sizeof(buffer)-len, "[%d] %f %s:%d NCCL TRACE %s\n", cudaDev, timestamp, filefunc, line, fmt);
  } else {
    len += snprintf(buffer+len, sizeof(buffer)-len, "%s\n", fmt);
  }

  // If the prefixed format string overflows, make sure it is still terminated with a newline.
  if (len > sizeof(buffer)-1) {
    // snprintf already placed a \0 at sizeof(buffer)-1
    buffer[sizeof(buffer)-2] = '\n';
  }

  // Add the message as given by the call site.
  // The call site's format string has been incorporated into `buffer` along with our prefix.
  va_list vargs;
  va_start(vargs, fmt);
  (void) vfprintf(ncclDebugFile, buffer, vargs);
  va_end(vargs);
}

// Non-deprecated version for internal use.
extern "C"
__attribute__ ((visibility("default")))
void ncclResetDebugInitInternal() {
  // Cleans up from a previous ncclDebugInit() and reruns.
  // Use this after changing NCCL_DEBUG and related parameters in the environment.
  std::lock_guard<std::mutex> lock(ncclDebugMutex);
  // Let ncclDebugInit() know to complete the reset.
  COMPILER_ATOMIC_STORE(&ncclDebugLevel, NCCL_DEBUG_RESET_TRIGGERED, std::memory_order_release);
}

// In place of: NCCL_API(void, ncclResetDebugInit);
__attribute__ ((visibility("default")))
__attribute__ ((alias("ncclResetDebugInit")))
void pncclResetDebugInit();
extern "C"
__attribute__ ((visibility("default")))
__attribute__ ((weak))
__attribute__ ((deprecated("ncclResetDebugInit is not supported as part of the NCCL API and will be removed in the future")))
void ncclResetDebugInit();


void ncclResetDebugInit() {
  // This is now deprecated as part of the NCCL API. It will be removed
  // from the API in the future. It is still available as an
  // exported symbol.
  ncclResetDebugInitInternal();
}

NCCL_PARAM(SetThreadName, "SET_THREAD_NAME", 0);

void ncclSetThreadName(std::thread& thread, const char *fmt, ...) {
  // pthread_setname_np is nonstandard GNU extension
  // needs the following feature test macro
#ifdef _GNU_SOURCE
  if (ncclParamSetThreadName() != 1) return;
  char threadName[NCCL_THREAD_NAMELEN];
  va_list vargs;
  va_start(vargs, fmt);
  vsnprintf(threadName, NCCL_THREAD_NAMELEN, fmt, vargs);
  va_end(vargs);
  pthread_setname_np(thread.native_handle(), threadName);
#endif
}
