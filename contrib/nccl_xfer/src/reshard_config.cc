/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information.
 ************************************************************************/

/*
 * Library configuration sources, in increasing precedence:
 *   1. Built-in defaults (the inline initializers in reshard_internal.h).
 *   2. ncclXferReshardConfig_t passed to ncclXferReshardInit (optional, may be
 *      NULL).
 *   3. Environment variables (always win when set; honors the
 *      "env-overrides-everything" convention used elsewhere in NCCL).
 *
 * applyReshardConfig() and applyReshardEnv() are called from
 * ncclXferReshardInit in that order.
 */

#include <cstdlib>
#include <cstring>
#include <climits>

#include "reshard_internal.h"
#include "reshard_log.h"
#include "reshard_types.h"

namespace {

bool parseAlgorithmEnv(const char* s, ReshardAlgorithm* out) {
  if (s == nullptr || out == nullptr) return false;
  if (strcasecmp(s, "AUTO") == 0) {
    *out = RESHARD_ALGO_AUTO;
    return true;
  }
  if (strcasecmp(s, "RING") == 0) {
    *out = RESHARD_ALGO_RING;
    return true;
  }
  if (strcasecmp(s, "DIRECT") == 0) {
    *out = RESHARD_ALGO_DIRECT;
    return true;
  }
  return false;
}

bool parseLbModeEnv(const char* s, ReshardLoadBalanceMode* out) {
  if (s == nullptr || out == nullptr) return false;
  if (strcasecmp(s, "UNIFORM") == 0) {
    *out = RESHARD_LB_UNIFORM;
    return true;
  }
  if (strcasecmp(s, "NODE_AWARE") == 0) {
    *out = RESHARD_LB_NODE_AWARE;
    return true;
  }
  return false;
}

bool parsePositiveIntEnv(const char* s, int* out) {
  if (s == nullptr || out == nullptr) return false;
  char* end = nullptr;
  long n = strtol(s, &end, 10);
  if (end == s || n <= 0 || n > INT_MAX) return false;
  *out = (int)n;
  return true;
}

} // namespace

ncclResult_t applyReshardConfig(const ncclXferReshardConfig_t* config) {
  if (config == nullptr) return ncclSuccess;

  if (config->size != sizeof(ncclXferReshardConfig_t) || config->magic != NCCLXFER_RESHARD_API_MAGIC) {
    RESHARD_WARN(-1,
                 "ncclXferReshardInit: ignoring malformed ncclXferReshardConfig_t "
                 "(size=%zu, magic=0x%x). Use NCCLXFER_RESHARD_CONFIG_INITIALIZER.",
                 config->size, config->magic);
    return ncclInvalidArgument;
  }

  if (config->maxCta != NCCLXFER_RESHARD_CONFIG_UNDEF_INT) {
    if (config->maxCta <= 0)
      RESHARD_WARN(-1, "ncclXferReshardInit: ignoring config.maxCta=%d (must be > 0).", config->maxCta);
    else gReshardMaxCta = config->maxCta;
  }

  return ncclSuccess;
}

// `getenv` is the only POSIX path to read process env vars; there is no portable
// thread-safe alternative (`secure_getenv` is glibc-only). Library init runs
// once on the calling thread before ncclXferReshardInit returns, so concurrent
// env mutation by user code is the caller's problem — scope the
// concurrency-mt-unsafe suppression to just this function.
//
// NOLINTBEGIN(concurrency-mt-unsafe)
void applyReshardEnv() {
  ReshardLogLevel lvl;
  if (reshardLogLevelFromStr(getenv("NCCLXFER_RESHARD_LOG_LEVEL"), &lvl)) reshardSetLogLevel(lvl);

  ReshardAlgorithm algo;
  if (parseAlgorithmEnv(getenv("NCCLXFER_RESHARD_ALGORITHM"), &algo)) gReshardAlgorithm = algo;

  ReshardLoadBalanceMode lb;
  if (parseLbModeEnv(getenv("NCCLXFER_RESHARD_LB_MODE"), &lb)) gReshardLbMode = lb;

  int n;
  if (parsePositiveIntEnv(getenv("NCCLXFER_RESHARD_MAX_CTA"), &n)) {
    if (gReshardMaxCta > 0) RESHARD_INFO(-1, "Reshard config maxCta reset to NCCLXFER_RESHARD_MAX_CTA=%d.", n);
    gReshardMaxCta = n;
  }

  if (parsePositiveIntEnv(getenv("NCCLXFER_RESHARD_SRC_DOMAIN_SIZE"), &n)) gReshardSrcDomainSize = n;
  if (parsePositiveIntEnv(getenv("NCCLXFER_RESHARD_DST_DOMAIN_SIZE"), &n)) gReshardDstDomainSize = n;

  /* Cache chunk-size override so prepareReshardParams doesn't touch
   * getenv on the hot path. */
  if (const char* env = getenv("NCCLXFER_RESHARD_CHUNK_SIZE")) {
    char* end = nullptr;
    long long v = strtoll(env, &end, 10);
    if (end != env && v > 0) gReshardChunkSizeBytes = (size_t)v;
  }
}
// NOLINTEND(concurrency-mt-unsafe)
