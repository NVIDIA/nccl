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
#include <shared_mutex>
#include <atomic>
#include "os.h"
#include "utils.h"
#include "env.h"
#include <cinttypes>

uint32_t ncclDebugLevelMask = NCCL_DEBUG_LEVEL_MASK_UNINITIALIZED;
static uint32_t ncclDebugTimestampLevels = 0;     // bitmaps of levels that have timestamps turned on
static char ncclDebugTimestampFormat[256];        // with space for subseconds
static int ncclDebugTimestampSubsecondsStart;     // index where the subseconds starts
static uint64_t ncclDebugTimestampMaxSubseconds;  // Max number of subseconds plus 1, used in duration ratio
static int ncclDebugTimestampSubsecondDigits;     // Number of digits to display
static int pid = -1;
static char hostname[1024];
static bool hostnameCached = false;
thread_local int ncclDebugNoWarn = 0;
char ncclLastError[1024] = ""; // Global string for the last error in human readable form
uint64_t ncclDebugMask = 0;
FILE* ncclDebugFile = stdout;
static std::mutex ncclDebugMutex;
static std::chrono::steady_clock::time_point ncclEpoch;
// Published by ncclDebugInit() and read on the logging path without holding ncclDebugMutex, so that a
// record that does not escalate anything does not have to take the lock to find that out.
static std::atomic<bool> ncclWarnSetDebugInfo{false};
// Optional caller-supplied destination for log records, installed by ncclSetDebugLogSink().
//
// Guarded by a reader/writer lock rather than ncclDebugMutex, and the read side is held across the whole
// onRecord call. That is what makes removing a sink safe: the exclusive lock in ncclSetDebugLogSink()
// cannot be taken until every in-flight dispatch has returned, so a sink's finalize() -- and, for a
// plugin, the dlclose that follows it -- can never run while another thread is still inside it.
//
// ncclDebugSinkPresent is a lock-free fast path for the common case of no sink at all, so a build that
// never registers one pays one acquire load per record and nothing else.
// std::shared_timed_mutex rather than std::shared_mutex: NCCL targets C++14 for CUDA < 13, and
// std::shared_mutex is C++17. Same rule, and the same reason, as gin_host.h's devCommRwMutex.
static std::shared_timed_mutex ncclDebugSinkLock;
static std::atomic<bool> ncclDebugSinkPresent{false};
static const ncclLogSink_v1_t* ncclDebugSink = nullptr;  // guarded by ncclDebugSinkLock
static void* ncclDebugSinkCtx = nullptr;                 // guarded by ncclDebugSinkLock

static thread_local int tid = -1;

// Sets a flag for the duration of a sink callback, so that a sink which logs is not dispatched back into
// itself. Scoped rather than a bare assignment so the flag is cleared even if the callback unwinds.
namespace {
struct SinkDispatchGuard {
  bool& flag;
  explicit SinkDispatchGuard(bool& f) : flag(f) {
    flag = true;
  }
  ~SinkDispatchGuard() {
    flag = false;
  }
};
}  // namespace

// clang-format off
DEFINE_NCCL_PARAM(ncclParamDebugLevel, ncclDebugLogLevel, NCCL_DEBUG, NCCL_LOG_NONE,
                  NCCL_PARAM_FLAG_PUBLISHED | NCCL_PARAM_FLAG_NO_ENVPLUGIN_INIT,
                  ncclParamOneOf<ncclDebugLogLevel>(makeOptions(
                    makeOption("VERSION", NCCL_LOG_VERSION, "Prints NCCL version information only."),
                    makeOption("WARN", NCCL_LOG_WARN, "Prints error messages."),
                    makeOption("ATTN", NCCL_LOG_ATTN, "Prints error messages plus informational notices."),
                    makeOption("INFO", NCCL_LOG_INFO, "Prints debug information."),
                    makeOption("ABORT", NCCL_LOG_ABORT, ""),
                    makeOption("TRACE", NCCL_LOG_TRACE, "Prints replayable trace information on all calls.")
                  )), "Set the debug output level. Each level includes less-verbose levels.");

DEFINE_NCCL_PARAM(ncclParamDebugLevels, uint32_t, NCCL_DEBUG_LEVELS, 0,
                  NCCL_PARAM_FLAG_PUBLISHED | NCCL_PARAM_FLAG_NO_ENVPLUGIN_INIT,
                  ncclParamBitsetOf<uint32_t>(makeOptions(
                    makeOption("VERSION", (1u << NCCL_LOG_VERSION), "Prints NCCL version information only."),
                    makeOption("WARN", (1u << NCCL_LOG_WARN), "Prints error messages."),
                    makeOption("ATTN", (1u << NCCL_LOG_ATTN), "Prints error messages plus informational notices."),
                    makeOption("INFO", (1u << NCCL_LOG_INFO), "Prints debug information."),
                    makeOption("ABORT", (1u << NCCL_LOG_ABORT), ""),
                    makeOption("TRACE", (1u << NCCL_LOG_TRACE), "Prints replayable trace information on all calls."),
                    makeOption("ALL", (1u << NCCL_LOG_VERSION | 1u << NCCL_LOG_WARN | 1u << NCCL_LOG_ATTN |
                                      1u << NCCL_LOG_INFO | 1u << NCCL_LOG_ABORT | 1u << NCCL_LOG_TRACE),
                               "Prints all debug messages")), ',', true),
                  "Add comma-separated debug levels to NCCL_DEBUG.");

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

DEFINE_NCCL_PARAM(ncclParamDebugTimestampLevel, uint32_t, NCCL_DEBUG_TIMESTAMP_LEVELS,
                  (1u << NCCL_LOG_WARN) | (1u << NCCL_LOG_ATTN),
                  NCCL_PARAM_FLAG_PUBLISHED | NCCL_PARAM_FLAG_NO_ENVPLUGIN_INIT,
                  ncclParamBitsetOf<uint32_t>(makeOptions(
                    makeOption("VERSION", (1u << NCCL_LOG_VERSION), "NCCL version information"),
                    makeOption("WARN", (1u << NCCL_LOG_WARN), "Error messages"),
                    makeOption("ATTN", (1u << NCCL_LOG_ATTN), "Informational notices"),
                    makeOption("INFO", (1u << NCCL_LOG_INFO), "Debug messages"),
                    makeOption("ABORT", (1u << NCCL_LOG_ABORT), ""),
                    makeOption("TRACE", (1u << NCCL_LOG_TRACE), "Replayable trace messages"),
                    makeOption("ALL", (1u << NCCL_LOG_VERSION | 1u << NCCL_LOG_WARN | 1u << NCCL_LOG_ATTN |
                                      1u << NCCL_LOG_INFO | 1u << NCCL_LOG_ABORT | 1u << NCCL_LOG_TRACE),
                               "All messages")
                  )), "Set the log levels that include timestamps.");
// clang-format on

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

// Convert the legacy scalar setting to its logical inclusive level set.
static uint32_t ncclDebugLevelToMask(ncclDebugLogLevel level) {
  if (level < NCCL_LOG_VERSION) return 0;
  uint32_t mask = 1u << NCCL_LOG_VERSION;
  if (level >= NCCL_LOG_INFO) mask |= 1u << NCCL_LOG_ATTN;
  if (level == NCCL_LOG_ATTN) level = NCCL_LOG_WARN;
  if (level >= NCCL_LOG_WARN) mask |= 1u << NCCL_LOG_WARN;
  if (level >= NCCL_LOG_INFO) mask |= 1u << NCCL_LOG_INFO;
  if (level >= NCCL_LOG_ABORT) mask |= 1u << NCCL_LOG_ABORT;
  if (level >= NCCL_LOG_TRACE) mask |= 1u << NCCL_LOG_TRACE;
  return mask;
}

// This function must be called with ncclDebugLock locked!
static void ncclDebugInit() {
  uint32_t tempNcclDebugLevelMask = 0;
  if (COMPILER_ATOMIC_LOAD(&ncclDebugLevelMask, std::memory_order_relaxed) == NCCL_DEBUG_LEVEL_MASK_RESET_TRIGGERED &&
      ncclDebugFile != stdout) {
    // Finish the reset initiated via ncclResetDebugInit().
    fclose(ncclDebugFile);
    ncclDebugFile = stdout;
  }

  tempNcclDebugLevelMask = ncclDebugLevelToMask(ncclParamDebugLevel()) | ncclParamDebugLevels();

  ncclWarnSetDebugInfo.store(ncclParamWarnEnableDebugInfo(), std::memory_order_relaxed);

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

  // Re-read the pid on every init: it changes across fork(), and a launcher that forks and then calls
  // ncclResetDebugInit() must expand NCCL_DEBUG_FILE's %p to its own pid. Caching it would make the child
  // reopen the parent's file in "w" mode and truncate it.
  pid = ncclOsGetPid();
  // The hostname, by contrast, is written once. It cannot change, and a log sink is handed this buffer
  // without holding ncclDebugMutex, so rewriting it in place would race a deliberately lock-free reader.
  if (!hostnameCached) {
    getHostNameForLog(hostname, 1024, '.');
    hostnameCached = true;
  }

  /* Parse and expand the NCCL_DEBUG_FILE path and
   * then create the debug file. But don't bother unless the
   * effective debug levels include more than VERSION.
   */
  const char* ncclDebugFileEnv = ncclParamDebugFile();
  if ((tempNcclDebugLevelMask & ~(1u << NCCL_LOG_VERSION)) != 0 && ncclDebugFileEnv != NULL) {
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
  COMPILER_ATOMIC_STORE(&ncclDebugLevelMask, tempNcclDebugLevelMask, std::memory_order_release);
}

// Internal logging helper used by the INFO, WARN, ATTN and TRACE macros.
static void ncclDebugLogV(ncclDebugLogLevel level, unsigned long flags, const char* file, const char* func, int line,
                          const char* fmt, va_list vargs) {
  if (ncclDebugNoWarn != 0 && level == NCCL_LOG_WARN) {
    level = NCCL_LOG_INFO;
    flags = ncclDebugNoWarn;
  }

  // Save the last error (WARN) as a human readable string. ATTN does not set lastError.
  //
  if (level == NCCL_LOG_WARN) {
    std::lock_guard<std::mutex> lock(ncclDebugMutex);
    va_list vcopy;
    va_copy(vcopy, vargs);
    (void)vsnprintf(ncclLastError, sizeof(ncclLastError), fmt, vcopy);
    va_end(vcopy);
  }

  if (!ncclDebugShouldLog(level, flags, ncclDebugMask)) {
    return;
  }

  // Initialize the masks on the first record only. Double-checked so that steady-state logging does not
  // take ncclDebugMutex here at all; the acquire load below pairs with the release store at the end of
  // ncclDebugInit(), which also publishes ncclWarnSetDebugInfo, hostname, pid and the timestamp settings.
  uint32_t levelMask = COMPILER_ATOMIC_LOAD(&ncclDebugLevelMask, std::memory_order_acquire);
  if (levelMask == NCCL_DEBUG_LEVEL_MASK_UNINITIALIZED || levelMask == NCCL_DEBUG_LEVEL_MASK_RESET_TRIGGERED) {
    std::lock_guard<std::mutex> lock(ncclDebugMutex);
    levelMask = COMPILER_ATOMIC_LOAD(&ncclDebugLevelMask, std::memory_order_relaxed);
    if (levelMask == NCCL_DEBUG_LEVEL_MASK_UNINITIALIZED || levelMask == NCCL_DEBUG_LEVEL_MASK_RESET_TRIGGERED)
      ncclDebugInit();
  }
  if (!ncclDebugShouldLog(level, flags, ncclDebugMask)) {
    return;
  }

  // A WARN can turn on INFO for everything that follows. Hoisted out of the formatting branch
  // below, which a record delivered to a sink never reaches, so that the behavior is the same either way.
  // Held under ncclDebugMutex because this is a read-modify-write on a mask that ncclDebugInit() -- and
  // therefore ncclResetDebugInit() -- also writes under that mutex; unsynchronised, an escalation racing
  // a reset can swallow the reset's sentinel and the reset is silently lost.
  if (level == NCCL_LOG_WARN && ncclWarnSetDebugInfo.load(std::memory_order_relaxed)) {
    std::lock_guard<std::mutex> lock(ncclDebugMutex);
    uint32_t mask = COMPILER_ATOMIC_LOAD(&ncclDebugLevelMask, std::memory_order_relaxed);
    if (mask != NCCL_DEBUG_LEVEL_MASK_RESET_TRIGGERED && mask != NCCL_DEBUG_LEVEL_MASK_UNINITIALIZED) {
      COMPILER_ATOMIC_STORE(&ncclDebugLevelMask, mask | ncclDebugLevelToMask(NCCL_LOG_INFO), std::memory_order_release);
    }
  }

  // A sink takes ownership of the record: it receives the message plus the structured fields that NCCL
  // would otherwise flatten into the text of a log line, and NCCL does not write it to ncclDebugFile.
  //
  // The shared lock is held across onRecord so the sink cannot be finalized or unloaded underneath it. It
  // is a shared lock, so concurrent logging threads do not serialize on each other, and it is why a sink
  // must not log: re-entering here on the same thread can deadlock against a waiting registration.
  // inSinkDispatch guards against a sink that logs. Without it the record re-enters here, takes the
  // shared lock recursively -- undefined behaviour, and a deadlock outright on a writer-preferring rwlock
  // with a registration pending -- and recurses until the stack is exhausted. A re-entrant record falls
  // through to the file instead of being lost.
  static thread_local bool inSinkDispatch = false;
  if (!inSinkDispatch && ncclDebugSinkPresent.load(std::memory_order_acquire)) {
    std::shared_lock<std::shared_timed_mutex> sinkLock(ncclDebugSinkLock);
    if (ncclDebugSink != nullptr) {
      char message[sizeof(ncclLastError)];
      va_list vcopy;
      va_copy(vcopy, vargs);
      (void)vsnprintf(message, sizeof(message), fmt, vcopy);
      va_end(vcopy);

      if (tid == -1) tid = ncclOsGetTid();
      int sinkCudaDev = 0;
      if (!(level == NCCL_LOG_TRACE && flags == NCCL_CALL)) (void)cudaGetDevice(&sinkCudaDev);

      ncclDebugLogRecord_v1_t record;
      record.level = (int)level;
      record.subSys = flags;
      record.file = file;
      record.func = func;
      record.line = line;
      // Reserved for the result code recorded at an error origin. Nothing sets one yet, so it is always
      // ncclSuccess; the field is present from the first version of the record because a versioned struct
      // cannot gain one later without forcing a v2 on every sink built against v1.
      record.code = ncclSuccess;
      record.format = fmt;
      record.message = message;
      record.hostname = hostname;
      record.pid = pid;
      record.tid = tid;
      record.cudaDev = sinkCudaDev;
      {
        SinkDispatchGuard guard(inSinkDispatch);
        (void)ncclDebugSink->onRecord(ncclDebugSinkCtx, &record);
      }
      return;
    }
  }

  std::lock_guard<std::mutex> lock(ncclDebugMutex);

  if (tid == -1) {
    tid = ncclOsGetTid();
  }

  char buffer[1024];
  size_t len = 0;

  // WARN and ATTN messages come with an extra newline at the beginning.
  if (level == NCCL_LOG_WARN || level == NCCL_LOG_ATTN) {
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
  } else if (level == NCCL_LOG_ATTN) {
    if (func && func[0]) {
      len += snprintf(buffer + len, sizeof(buffer) - len, "[%d] %s:%d (%s) NCCL ATTN %s\n", cudaDev, fileStr, line,
                      funcStr, fmt);
    } else {
      len += snprintf(buffer + len, sizeof(buffer) - len, "[%d] %s:%d NCCL ATTN %s\n", cudaDev, fileStr, line, fmt);
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

// Internal only Common logging function used by the INFO, WARN, ATTN and TRACE macros
void ncclDebugLogInternal(ncclDebugLogLevel level, unsigned long flags, const char* file, const char* func, int line,
                          const char* fmt, ...) {
  va_list vargs;
  va_start(vargs, fmt);
  ncclDebugLogV(level, flags, file, func, line, fmt, vargs);
  va_end(vargs);
}

/* Routes log records to a caller-supplied sink instead of ncclDebugFile. Passing NULL restores the
 * default file output. See the contract on ncclLogSink_v1_t in nccl.h.
 */
NCCL_API(ncclResult_t, ncclSetDebugLogSink, const ncclLogSink_v1_t* sink);
ncclResult_t ncclSetDebugLogSink(const ncclLogSink_v1_t* sink) {
  if (sink != nullptr && sink->onRecord == nullptr) return ncclInvalidArgument;

  const ncclLogSink_v1_t* previous = nullptr;
  void* previousCtx = nullptr;

  {
    // Taking the lock exclusively waits for every in-flight onRecord to return, so once the swap
    // completes no thread can still be inside the outgoing sink.
    std::unique_lock<std::shared_timed_mutex> sinkLock(ncclDebugSinkLock);

    // There is one sink slot and more than one possible claimant -- an application registering directly,
    // and a log plugin loaded from NCCL_LOG_PLUGIN. Silently replacing the incumbent would run its
    // finalize() and redirect its records with no indication to either party, so installing over a live
    // sink is refused. Remove the current one with NULL first if replacement is intended.
    //
    // Tested before init() runs, so a refused registration really does change nothing: the rejected sink
    // is never initialized and never finalized.
    if (sink != nullptr && ncclDebugSink != nullptr) return ncclInvalidUsage;

    void* context = nullptr;
    if (sink != nullptr && sink->init != nullptr) {
      // Running init() under the exclusive lock is safe precisely because this point is only reached
      // with no sink installed: ncclDebugSinkPresent is false, so anything the sink logs while
      // initializing takes the lock-free path to ncclDebugFile rather than re-entering this lock.
      ncclResult_t res = sink->init(&context);
      if (res != ncclSuccess) return res;
    }

    previous = ncclDebugSink;
    previousCtx = ncclDebugSinkCtx;
    ncclDebugSink = sink;
    ncclDebugSinkCtx = context;
    ncclDebugSinkPresent.store(sink != nullptr, std::memory_order_release);
  }

  // The outgoing sink is detached and drained, so finalize() runs outside the lock, where it may log.
  if (previous != nullptr && previous->finalize != nullptr) (void)previous->finalize(previousCtx);
  return ncclSuccess;
}

/* Exported ABI logging function exported to the dynamically loadable Net
 * transport modules so they can share the debugging mechanisms and output files
 */
void ncclDebugLog(ncclDebugLogLevel level, unsigned long flags, const char* filefunc, int line, const char* fmt, ...) {
  va_list vargs;
  va_start(vargs, fmt);
  const char* file = nullptr;
  const char* func = nullptr;
  if (level == NCCL_LOG_WARN || level == NCCL_LOG_ATTN) {
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
  COMPILER_ATOMIC_STORE(&ncclDebugLevelMask, NCCL_DEBUG_LEVEL_MASK_RESET_TRIGGERED, std::memory_order_release);
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
