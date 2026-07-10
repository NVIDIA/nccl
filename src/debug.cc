/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "core.h"
#include "nccl_net.h"
#include <ctime>
#include <stdlib.h>
#include <stdarg.h>
#include <stdio.h>
#include <string.h>
#include <chrono>
#include "param.h"
#include "param/param.h"
#include <mutex>
#include "os.h"
#include "utils.h"
#include "env.h"
#include <cinttypes>

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
FILE* ncclDebugFile = stdout;
static std::mutex ncclDebugMutex;
static std::chrono::steady_clock::time_point ncclEpoch;
static bool ncclWarnSetDebugInfo = false;

static thread_local int tid = -1;

// clang-format off
DEFINE_NCCL_PARAM(ncclParamDebugLevel, ncclDebugLogLevel, NCCL_DEBUG, NCCL_LOG_NONE,
                  NCCL_PARAM_FLAG_PUBLISHED | NCCL_PARAM_FLAG_NO_ENVPLUGIN_INIT,
                  ncclParamOneOf<ncclDebugLogLevel>(makeOptions(
                    makeOption("VERSION", NCCL_LOG_VERSION, "Prints the NCCL version info only"),
                    makeOption("WARN", NCCL_LOG_WARN, "Prints only messages indicating a fatal error."),
                    makeOption("INFO", NCCL_LOG_INFO, "Prints debug message"),
                    makeOption("ABORT", NCCL_LOG_ABORT, ""),
                    makeOption("TRACE", NCCL_LOG_TRACE, "Prints replayable trace info on all calls")
                  )), "Set debug output level, the option is inclusive for any level that is less verbose than the set value");

DEFINE_NCCL_PARAM(ncclParamDebugSubsys, uint64_t, NCCL_DEBUG_SUBSYS,
                  NCCL_INIT | NCCL_BOOTSTRAP | NCCL_ENV,
                  NCCL_PARAM_FLAG_PUBLISHED | NCCL_PARAM_FLAG_NO_ENVPLUGIN_INIT,
                  (ncclParamBitsetOf<ncclDebugLogSubSys, uint64_t>(makeOptions(
                    makeOption("INIT", NCCL_INIT, "NCCL and comm initialization (included in default)"),
                    makeOption("COLL", NCCL_COLL, "Collective operations"),
                    makeOption("P2P", NCCL_P2P, "Peer-to-peer transport"),
                    makeOption("SHM", NCCL_SHM, "Shared memory transport"),
                    makeOption("NET", NCCL_NET, "Network transport"),
                    makeOption("GRAPH", NCCL_GRAPH, "Graph search and topology"),
                    makeOption("TUNING", NCCL_TUNING, "Algorithm tuning"),
                    makeOption("ENV", NCCL_ENV, "Parameter settings by config file, EnvVar or EnvPlugins (included in default)"),
                    makeOption("ALLOC", NCCL_ALLOC, "Device memory allocation"),
                    makeOption("ALLOC_HOST", NCCL_ALLOC_HOST, "Host memory allocation"),
                    makeOption("CALL", NCCL_CALL, "API call tracing"),
                    makeOption("PROXY", NCCL_PROXY, "Proxy thread operations"),
                    makeOption("NVLS", NCCL_NVLS, "NVLink SHARP operations"),
                    makeOption("BOOTSTRAP", NCCL_BOOTSTRAP, "Bootstrap network (included in default)"),
                    makeOption("REG", NCCL_REG, "Buffer registration"),
                    makeOption("PROFILE", NCCL_PROFILE,   "Profiling"),
                    makeOption("RAS", NCCL_RAS, "Reliability, availability, serviceability"),
                    makeOption("DESTROY", NCCL_DESTROY, "Communicator destroy, abort, revoke, and plugin unload/close operations"),
                    makeOption("ALL", NCCL_ALL, "All categories")
                  ))), "Filter debug output by (comma-separated)");

DEFINE_NCCL_PARAM(ncclParamWarnEnableDebugInfo, bool, NCCL_WARN_ENABLE_DEBUG_INFO, false,
                  NCCL_PARAM_FLAG_NO_ENVPLUGIN_INIT, NCCL_PARAM_DEFAULT,
                  "If enabled, the debug level will be set to INFO after a WARN level debug message is logged.");
// clang-format on

DEFINE_NCCL_PARAM(ncclParamDebugTimestampLevel, uint32_t, NCCL_DEBUG_TIMESTAMP_LEVELS, (1u << NCCL_LOG_WARN),
                  NCCL_PARAM_FLAG_PUBLISHED | NCCL_PARAM_FLAG_NO_ENVPLUGIN_INIT,
                  ncclParamBitsetOf<uint32_t>(
                    makeOptions(makeOption("VERSION", (1u << NCCL_LOG_VERSION), "on NCCL version info message"),
                                makeOption("WARN", (1u << NCCL_LOG_WARN), "on Explicit error message"),
                                makeOption("INFO", (1u << NCCL_LOG_INFO), "on Debug message"),
                                makeOption("ABORT", (1u << NCCL_LOG_ABORT), ""),
                                makeOption("TRACE", (1u << NCCL_LOG_TRACE), "on Replayable trace message"),
                                makeOption("ALL",
                                           (1u << NCCL_LOG_VERSION | 1u << NCCL_LOG_WARN | 1u << NCCL_LOG_INFO |
                                            1u << NCCL_LOG_ABORT | 1u << NCCL_LOG_TRACE),
                                           "on All messages"))),
                  "Set which log lines get a timestamp depending upon the level of the log");

DEFINE_NCCL_PARAM(ncclParamDebugTsFormat, const char*, NCCL_DEBUG_TIMESTAMP_FORMAT, "[%F %T] ",
                  NCCL_PARAM_FLAG_PUBLISHED | NCCL_PARAM_FLAG_NO_ENVPLUGIN_INIT, NCCL_PARAM_DEFAULT,
                  "Set the format used when printing debug log messages");

DEFINE_NCCL_PARAM(ncclParamDebugFile, const char*, NCCL_DEBUG_FILE, nullptr,
                  NCCL_PARAM_FLAG_PUBLISHED | NCCL_PARAM_FLAG_NO_ENVPLUGIN_INIT, NCCL_PARAM_DEFAULT,
                  "Set the NCCL debug logging output to a file. The filename format can be set to "
                  "filename.%h.%p where %h is replaced with the hostname "
                  "and %p is replaced "
                  "with the process PID. This does not accept the ~ character as part of the path, "
                  "please convert to a relative or absolute path first.");

typedef const char* (*ncclGetEnvFunc_t)(const char*);

static ncclResult_t getHostNameForLog(char* hostname, int maxlen, const char delim) {
  ncclResult_t ret = getHostName(hostname, maxlen, delim);
  if (ret != ncclSuccess) return ret;

  for (int i = 0; i < maxlen - 1 && hostname[i]; ++i) {
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
  int tempNcclDebugLevel = -1;
  if (ncclDebugLevel == NCCL_DEBUG_RESET_TRIGGERED && ncclDebugFile != stdout) {
    // Finish the reset initiated via ncclResetDebugInit().
    fclose(ncclDebugFile);
    ncclDebugFile = stdout;
  }

  tempNcclDebugLevel = ncclParamDebugLevel();

  ncclWarnSetDebugInfo = ncclParamWarnEnableDebugInfo();

  // Determine which debug levels will have timestamps.
  ncclDebugTimestampLevels = ncclParamDebugTimestampLevel();

  // Store a copy of the timestamp format with space for the subseconds, if used.
  const char* tsFormat = ncclParamDebugTsFormat();
  ncclDebugTimestampSubsecondsStart = -1;
  // Find where the subseconds are in the format.
  for (int i = 0; tsFormat[i] != '\0'; ++i) {
    if (tsFormat[i] == '%' && tsFormat[i + 1] == '%') {
      // Next two chars are "%"
      // Skip the next character, too, and restart checking after that.
      ++i;
      continue;
    }
    if (tsFormat[i] == '%' &&                               // Found a percentage
        ('1' <= tsFormat[i + 1] && tsFormat[i + 1] <= '9') && // Next char is a digit between 1 and 9 inclusive
        tsFormat[i + 2] == 'f'                                // Two characters later is an "f"
    ) {
      constexpr int replaceLen = sizeof("%Xf") - 1;
      ncclDebugTimestampSubsecondDigits = tsFormat[i + 1] - '0';
      if (ncclDebugTimestampSubsecondDigits + strlen(tsFormat) - replaceLen > sizeof(ncclDebugTimestampFormat) - 1) {
        // Won't fit; fall back on the default.
        break;
      }
      ncclDebugTimestampSubsecondsStart = i;
      ncclDebugTimestampMaxSubseconds = 1;

      memcpy(ncclDebugTimestampFormat, tsFormat, i);
      for (int j = 0; j < ncclDebugTimestampSubsecondDigits; ++j) {
        ncclDebugTimestampFormat[i + j] = ' ';
        ncclDebugTimestampMaxSubseconds *= 10;
      }
      strcpy(ncclDebugTimestampFormat + i + ncclDebugTimestampSubsecondDigits, tsFormat + i + replaceLen);
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
  for (int i = 0; ncclDebugTimestampFormat[i] != '\0'; ++i) {
    if (ncclDebugTimestampFormat[i] == '_') ncclDebugTimestampFormat[i] = ' ';
  }

  // Cache pid and hostname
  getHostNameForLog(hostname, 1024, '.');
  pid = ncclOsGetPid();

  /* Parse and expand the NCCL_DEBUG_FILE path and
   * then create the debug file. But don't bother unless the
   * NCCL_DEBUG level is > VERSION
   */
  const char* ncclDebugFileEnv = ncclParamDebugFile();
  if (tempNcclDebugLevel > NCCL_LOG_VERSION && ncclDebugFileEnv != NULL) {
    int c = 0;
    char debugFn[PATH_MAX + 1] = "";
    char* dfn = debugFn;
    while (ncclDebugFileEnv[c] != '\0' && (dfn - debugFn) < PATH_MAX) {
      if (ncclDebugFileEnv[c++] != '%') {
        *dfn++ = ncclDebugFileEnv[c - 1];
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
          *dfn++ = ncclDebugFileEnv[c - 1];
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
      FILE* file = fopen(debugFn, "w");
      if (file != nullptr) {
#if defined(NCCL_OS_LINUX)
        setlinebuf(file); // disable block buffering
#elif defined(NCCL_OS_WINDOWS)
        setvbuf(file, NULL, _IOLBF, 0); // disable block buffering
#endif
        ncclDebugFile = file;
      }
    }
  }

  ncclEpoch = std::chrono::steady_clock::now();
  ncclDebugMask = ncclParamDebugSubsys();
  COMPILER_ATOMIC_STORE(&ncclDebugLevel, tempNcclDebugLevel, std::memory_order_release);
}

static void ncclDebugLogV(ncclDebugLogLevel level, unsigned long flags, const char* file, const char* func, int line,
                          const char* fmt, va_list vargs) {
  int gotLevel = COMPILER_ATOMIC_LOAD(&ncclDebugLevel, std::memory_order_acquire);

  if (ncclDebugNoWarn != 0 && level == NCCL_LOG_WARN) {
    level = NCCL_LOG_INFO;
    flags = ncclDebugNoWarn;
  }

  // Save the last error (WARN) as a human readable string
  if (level == NCCL_LOG_WARN) {
    std::lock_guard<std::mutex> lock(ncclDebugMutex);
    va_list vcopy;
    va_copy(vcopy, vargs);
    (void)vsnprintf(ncclLastError, sizeof(ncclLastError), fmt, vcopy);
    va_end(vcopy);
  }

  if (gotLevel >= 0 && (gotLevel < level || (flags & ncclDebugMask) == 0)) {
    return;
  }

  std::lock_guard<std::mutex> lock(ncclDebugMutex);
  if (ncclDebugLevel < 0) ncclDebugInit();
  if (ncclDebugLevel < level || ((flags & ncclDebugMask) == 0)) {
    return;
  }

  if (tid == -1) {
    tid = ncclOsGetTid();
  }

  char buffer[1024];
  size_t len = 0;

  // WARNs come with an extra newline at the beginning.
  if (level == NCCL_LOG_WARN) {
    buffer[len++] = '\n';
  }

  // Add the timestamp to the buffer if they are turned on for this level.
  if (ncclDebugTimestampLevels & (1 << level)) {
    if (ncclDebugTimestampFormat[0] != '\0') {
      struct timespec ts;
      clockRealtime(&ts);
      time_t nowTimeT = ts.tv_sec;
      long nowNs = ts.tv_nsec;
      std::tm nowTm;
      ncclOsLocaltime(&nowTimeT, &nowTm);

      // Add the subseconds portion if it is part of the format.
      char localTimestampFormat[sizeof(ncclDebugTimestampFormat)];
      const char* pformat = ncclDebugTimestampFormat;
      if (ncclDebugTimestampSubsecondsStart != -1) {
        pformat = localTimestampFormat;   // Need to use the local version which has subseconds
        memcpy(localTimestampFormat, ncclDebugTimestampFormat, ncclDebugTimestampSubsecondsStart);
        snprintf(localTimestampFormat + ncclDebugTimestampSubsecondsStart, ncclDebugTimestampSubsecondDigits + 1,
                 "%0*" PRIu64, ncclDebugTimestampSubsecondDigits,
                 (uint64_t)(nowNs / (1000000000L / ncclDebugTimestampMaxSubseconds)));
        strcpy(localTimestampFormat + ncclDebugTimestampSubsecondsStart + ncclDebugTimestampSubsecondDigits,
               ncclDebugTimestampFormat + ncclDebugTimestampSubsecondsStart + ncclDebugTimestampSubsecondDigits);
      }

      // Format the time. If it runs out of space, fall back on a simpler format.
      int adv = std::strftime(buffer + len, sizeof(buffer) - len, pformat, &nowTm);
      if (adv == 0 && ncclDebugTimestampFormat[0] != '\0') {
        // Ran out of space. Fall back on the default. This should never fail.
        adv = std::strftime(buffer + len, sizeof(buffer) - len, "[%F %T] ", &nowTm);
      }
      len += adv;
    }
  }
  len = std::min(len, sizeof(buffer) - 1);  // prevent overflows

  // Add hostname, pid and tid portion of the log line.
  if (level != NCCL_LOG_VERSION) {
    len += snprintf(buffer + len, sizeof(buffer) - len, "%s:%d:%d ", hostname, pid, tid);
    len = std::min(len, sizeof(buffer) - 1);  // prevent overflows
  }

  int cudaDev = 0;
  if (!(level == NCCL_LOG_TRACE && flags == NCCL_CALL)) {
    (void)cudaGetDevice(&cudaDev);
  }

  const char* fileStr = file ? file : "<unknown>";
  const char* funcStr = func ? func : "<unknown>";

  // Add level specific formatting. The format string from the call site is incorporated into this prefix.
  if (level == NCCL_LOG_WARN) {
    if (func && func[0]) {
      len += snprintf(buffer + len, sizeof(buffer) - len, "[%d] %s:%d (%s) NCCL WARN %s\n", cudaDev, fileStr, line,
                      funcStr, fmt);
    } else {
      len += snprintf(buffer + len, sizeof(buffer) - len, "[%d] %s:%d NCCL WARN %s\n", cudaDev, fileStr, line, fmt);
    }
    if (ncclWarnSetDebugInfo) {
      COMPILER_ATOMIC_STORE(&ncclDebugLevel, static_cast<int>(NCCL_LOG_INFO), std::memory_order_release);
    }
  } else if (level == NCCL_LOG_INFO) {
    len += snprintf(buffer + len, sizeof(buffer) - len, "[%d] NCCL INFO %s\n", cudaDev, fmt);
  } else if (level == NCCL_LOG_TRACE && flags == NCCL_CALL) {
    len += snprintf(buffer + len, sizeof(buffer) - len, "NCCL CALL %s\n", fmt);
  } else if (level == NCCL_LOG_TRACE) {
    auto delta = std::chrono::steady_clock::now() - ncclEpoch;
    double timestamp = std::chrono::duration_cast<std::chrono::duration<double>>(delta).count() * 1000;
    len += snprintf(buffer + len, sizeof(buffer) - len, "[%d] %f %s:%d NCCL TRACE %s\n", cudaDev, timestamp, funcStr,
                    line, fmt);
  } else {
    len += snprintf(buffer + len, sizeof(buffer) - len, "%s\n", fmt);
  }

  // If the prefixed format string overflows, make sure it is still terminated with a newline.
  if (len > sizeof(buffer) - 1) {
    // snprintf already placed a \0 at sizeof(buffer)-1
    buffer[sizeof(buffer) - 2] = '\n';
  }

  // Add the message as given by the call site.
  // The call site's format string has been incorporated into `buffer` along with our prefix.
  va_list vcopy;
  va_copy(vcopy, vargs);
  (void)vfprintf(ncclDebugFile, buffer, vcopy);
  va_end(vcopy);
}

// Internal only Common logging function used by the INFO, WARN and TRACE macros
void ncclDebugLogInternal(ncclDebugLogLevel level, unsigned long flags, const char* file, const char* func, int line,
                          const char* fmt, ...) {
  va_list vargs;
  va_start(vargs, fmt);
  ncclDebugLogV(level, flags, file, func, line, fmt, vargs);
  va_end(vargs);
}

/* Exported ABI logging function exported to the dynamically loadable Net
 * transport modules so they can share the debugging mechanisms and output files
 */
void ncclDebugLog(ncclDebugLogLevel level, unsigned long flags, const char* filefunc, int line, const char* fmt, ...) {
  va_list vargs;
  va_start(vargs, fmt);
  const char* file = nullptr;
  const char* func = nullptr;
  if (level == NCCL_LOG_WARN) {
    file = filefunc;
  } else if (level == NCCL_LOG_TRACE) {
    func = filefunc;
  }
  ncclDebugLogV(level, flags, file, func, line, fmt, vargs);
  va_end(vargs);
}

// Non-deprecated version for internal use.
extern "C"
#if !defined(NCCL_OS_WINDOWS)
  __attribute__((visibility("default")))
#endif
  void ncclResetDebugInitInternal() {
  // Cleans up from a previous ncclDebugInit() and reruns.
  // Use this after changing NCCL_DEBUG and related parameters in the environment.
  std::lock_guard<std::mutex> lock(ncclDebugMutex);
  // Let ncclDebugInit() know to complete the reset.
  COMPILER_ATOMIC_STORE(&ncclDebugLevel, static_cast<int>(NCCL_DEBUG_RESET_TRIGGERED), std::memory_order_release);
}

// In place of: NCCL_API(void, ncclResetDebugInit);
#ifdef pncclResetDebugInit
#undef pncclResetDebugInit
#endif
#if defined(NCCL_OS_LINUX)
__attribute__((visibility("default"))) __attribute__((alias("ncclResetDebugInit")))
#endif
void pncclResetDebugInit();
extern "C"
#if defined(__GNUC__) || defined(__clang__)
  __attribute__((visibility("default"))) __attribute__((weak)) __attribute__((
    deprecated("ncclResetDebugInit is not supported as part of the NCCL API and will be removed in the future")))
#endif
  void ncclResetDebugInit();

extern "C" void ncclResetDebugInit() {
  // This is now deprecated as part of the NCCL API. It will be removed
  // from the API in the future. It is still available as an
  // exported symbol.
  ncclResetDebugInitInternal();
}

DEFINE_NCCL_PARAM(ncclParamSetThreadName, bool, NCCL_SET_THREAD_NAME, false,
                  NCCL_PARAM_FLAG_PUBLISHED | NCCL_PARAM_FLAG_CACHED, NCCL_PARAM_DEFAULT,
                  "Allow NCCL to give meaningful names to NCCL CPU threads via pthread_setname_np");

void ncclSetThreadName(std::thread& thread, const char* fmt, ...) {
  // pthread_setname_np is nonstandard GNU extension
  // needs the following feature test macro
#ifdef _GNU_SOURCE
  if (ncclParamSetThreadName() == false) return;
  char threadName[NCCL_THREAD_NAMELEN];
  va_list vargs;
  va_start(vargs, fmt);
  vsnprintf(threadName, NCCL_THREAD_NAMELEN, fmt, vargs);
  va_end(vargs);
  pthread_setname_np(thread.native_handle(), threadName);
#endif
}
