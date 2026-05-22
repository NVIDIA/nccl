/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information.
 ************************************************************************/

#include <climits>
#include <cstdlib>

#include "nccl.h"
#include "nccl_xfer.h"
#include "reshard_types.h"
#include "reshard_log.h"
#include "reshard_internal.h"

static bool gReshardInitialized = false;

// NCCLXFER_RESHARD_STREAM_POOL_SIZE: max number of distinct (comm, dev)
// pairs the pool will hold (1:1 stream+event mapping).  Default 4.
// Values <= 0 disable the pool — default-stream callers then run on
// the user's default stream directly (legacy synchronizing behavior).
// Values above STREAM_POOL_MAX_SIZE are capped (with a warning).
// strtol-based parsing: unparseable values (e.g. "abc") read as 0 and
// likewise disable the pool.
static int parseStreamPoolSize(const char* sizeEnv) {
  char* end = nullptr;
  long n = strtol(sizeEnv, &end, 10);
  if (end == sizeEnv) return 0;
  if (n < INT_MIN) return INT_MIN;
  if (n > INT_MAX) return INT_MAX;
  return (int)n;
}

static void applyStreamPoolFromEnv() {
  // NOLINTNEXTLINE(concurrency-mt-unsafe) — init-time, single-thread on the caller
  const char* sizeEnv = getenv("NCCLXFER_RESHARD_STREAM_POOL_SIZE");
  if (sizeEnv == nullptr) return;

  int n = parseStreamPoolSize(sizeEnv);
  if (n <= 0) {
    RESHARD_WARN(-1,
                 "NCCLXFER_RESHARD_STREAM_POOL_SIZE='%s' (parsed as %d) <= 0; "
                 "stream "
                 "pool disabled — default-stream callers will run on the user's "
                 "default stream directly.",
                 sizeEnv, n);
    gReshardStreamPoolSize = 0;
  } else if (n > STREAM_POOL_MAX_SIZE) {
    RESHARD_WARN(-1,
                 "NCCLXFER_RESHARD_STREAM_POOL_SIZE='%s' (parsed as %d) exceeds the "
                 "library max %d; capping to %d.",
                 sizeEnv, n, STREAM_POOL_MAX_SIZE, STREAM_POOL_MAX_SIZE);
    gReshardStreamPoolSize = STREAM_POOL_MAX_SIZE;
  } else {
    gReshardStreamPoolSize = n;
  }
}

extern "C" {
ncclResult_t ncclXferReshardInit(ncclXferReshardConfig_t* config) {
  if (gReshardInitialized) return ncclSuccess;
  ncclResult_t r = applyReshardConfig(config);
  if (r != ncclSuccess) return r;
  applyReshardEnv();
  applyStreamPoolFromEnv();

  gReshardNumCtas = (gReshardMaxCta > 0 && gReshardMaxCta < DEFAULT_NUM_CTAS) ? gReshardMaxCta : DEFAULT_NUM_CTAS;

  gReshardInitialized = true;
  return ncclSuccess;
}

ncclResult_t ncclXferReshardFinalize() {
  if (!gReshardInitialized) return ncclSuccess;
  cacheFinalize();
  transposeBufferFinalize();
  gReshardInitialized = false;
  return ncclSuccess;
}

} // extern "C"
