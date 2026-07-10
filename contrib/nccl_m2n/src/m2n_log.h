/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information.
 ************************************************************************/

#ifndef NCCL_M2N_LOG_H_
#define NCCL_M2N_LOG_H_

#include <cstdio>
#include <cstdlib>
#include <cstring>

/*
 * Structured logging for the NCCL M2N library.
 *
 * Tiers:
 *   FATAL — correctness errors that must never be silenced (printed
 *           unconditionally to stderr, then abort()).
 *
 *   Level-gated (printed to stdout when current level >= the tier):
 *     NONE  — silent
 *     WARN  — recoverable issues (cache eviction)
 *     INFO  — high-level entry/exit, algorithm selection
 *     DEBUG — per-call parameter dumps, mesh analysis
 *     TRACE — per-target / per-source detail, transfer plan dumps
 *
 * The runtime level is controlled by the NCCL_RESHARD_LOG_LEVEL env
 * var at init time.  Default level is WARN.
 */

enum ReshardLogLevel {
  RESHARD_LOG_NONE = 0,
  RESHARD_LOG_WARN = 1,
  RESHARD_LOG_INFO = 2,
  RESHARD_LOG_DEBUG = 3,
  RESHARD_LOG_TRACE = 4
};

/* Process-wide log level. */
inline ReshardLogLevel gReshardLogLevel = RESHARD_LOG_WARN;

inline ReshardLogLevel reshardGetLogLevel() {
  return gReshardLogLevel;
}
inline void reshardSetLogLevel(ReshardLogLevel level) {
  gReshardLogLevel = level;
}

static inline const char* reshardLogLevelStr(ReshardLogLevel level) {
  switch (level) {
  case RESHARD_LOG_WARN:
    return "WARN";
  case RESHARD_LOG_INFO:
    return "INFO";
  case RESHARD_LOG_DEBUG:
    return "DEBUG";
  case RESHARD_LOG_TRACE:
    return "TRACE";
  default:
    return "???";
  }
}

/* Reverse of reshardLogLevelStr.  Returns true and writes *out on a
   recognised tier name; returns false otherwise (so the caller can leave
   the existing level unchanged). */
static inline bool reshardLogLevelFromStr(const char* s, ReshardLogLevel* out) {
  if (s == nullptr || out == nullptr) return false;
  if (strcmp(s, "NONE") == 0) {
    *out = RESHARD_LOG_NONE;
    return true;
  }
  if (strcmp(s, "WARN") == 0) {
    *out = RESHARD_LOG_WARN;
    return true;
  }
  if (strcmp(s, "INFO") == 0) {
    *out = RESHARD_LOG_INFO;
    return true;
  }
  if (strcmp(s, "DEBUG") == 0) {
    *out = RESHARD_LOG_DEBUG;
    return true;
  }
  if (strcmp(s, "TRACE") == 0) {
    *out = RESHARD_LOG_TRACE;
    return true;
  }
  return false;
}

#define RESHARD_LOG(level, rank, fmt, ...)                                                          \
  do {                                                                                              \
    if (reshardGetLogLevel() >= (level)) {                                                          \
      printf("[RESHARD][%s][Rank %d] " fmt "\n", reshardLogLevelStr(level), (rank), ##__VA_ARGS__); \
      fflush(stdout);                                                                               \
    }                                                                                               \
  } while (0)

#define RESHARD_WARN(rank, fmt, ...) RESHARD_LOG(RESHARD_LOG_WARN, rank, fmt, ##__VA_ARGS__)
#define RESHARD_INFO(rank, fmt, ...) RESHARD_LOG(RESHARD_LOG_INFO, rank, fmt, ##__VA_ARGS__)
#define RESHARD_DEBUG(rank, fmt, ...) RESHARD_LOG(RESHARD_LOG_DEBUG, rank, fmt, ##__VA_ARGS__)
#define RESHARD_TRACE(rank, fmt, ...) RESHARD_LOG(RESHARD_LOG_TRACE, rank, fmt, ##__VA_ARGS__)

/* Unconditional stderr emit + abort() for correctness errors. */
template <typename... Args>
[[noreturn]] inline void reshardFatal(int rank, const char* file, int line, const char* fmt, Args... args) {
  fprintf(stderr, "[RESHARD][FATAL][Rank %d] %s:%d: ", rank, file, line);
  /* Format-string safety: callers route through RESHARD_FATAL, whose
   * reshardFatalFmtCheck shim wears `__attribute__((format(printf,4,5)))`
   * so gcc/clang verify `fmt` against the trailing args at the call site.
   * The branch on sizeof...(Args) silences -Wformat-security when the
   * caller passed no args (where fprintf would otherwise see a runtime
   * `fmt` with nothing to substitute). */
  if constexpr (sizeof...(Args) == 0) fputs(fmt, stderr);
  else fprintf(stderr, fmt, args...);
  fputc('\n', stderr);
  fflush(stdout);
  fflush(stderr);
  abort();
}

#if defined(__GNUC__) || defined(__clang__)
__attribute__((format(printf, 4, 5))) void reshardFatalFmtCheck(int, const char*, int, const char*, ...);
#define RESHARD_FATAL(rank, fmt, ...)                                                  \
  do {                                                                                 \
    if (false) ::reshardFatalFmtCheck((rank), __FILE__, __LINE__, fmt, ##__VA_ARGS__); \
    ::reshardFatal((rank), __FILE__, __LINE__, fmt, ##__VA_ARGS__);                    \
  } while (0)
#else
#define RESHARD_FATAL(rank, fmt, ...) ::reshardFatal((rank), __FILE__, __LINE__, fmt, ##__VA_ARGS__)
#endif

#endif /* NCCL_M2N_LOG_H_ */
