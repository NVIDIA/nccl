/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information.
 ************************************************************************/

/*************************************************************************
 * basic_api_test_core.h — Shared test matrix + per-case execution.
 *
 * Mirrors an external pytest reference suite at the C level: each test
 * case from the pytest parametrize matrix becomes a TestCase entry.
 * The bootstrap layer (MPI vs single-process pthreads) is injected via
 * the TestEnv struct's function pointers so the same matrix runs from
 * both binaries.
 *
 * Mesh encoding follows the Python reshape:
 *   mesh dims[0] = N_axis0 (the first arg to reshape)
 *   mesh dims[1] = total / N_axis0
 *   placement decides whether axis 0 is shard or replicate
 *
 * shardIdx for a rank inside a mesh of shape [d0, d1] (startRank=s):
 *   PL_RS  (placement = {REPL, SHARD}):  shardIdx = (rank - s) % d1
 *   PL_SR  (placement = {SHARD, REPL}):  shardIdx = (rank - s) / d1
 *   PL_REPL                          :   shardIdx = 0
 ************************************************************************/

#ifndef TESTS_BASIC_API_TEST_CORE_H_
#define TESTS_BASIC_API_TEST_CORE_H_

#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cctype>
#include <ostream>
#include <string>
#include <vector>
#include <algorithm>

#include <cuda_runtime.h>
#include <nccl.h>

#include "nccl_xfer.h"
#include "test_helpers.h"

/* ======================================================================
 * Bootstrap-agnostic test environment.
 * ====================================================================*/

struct TestEnv {
  int rank;
  int worldSize;
  int device;
  ncclComm_t comm;
  cudaStream_t stream;
  void* buffer;
  size_t bufferBytes;
  bool verbose;

  void (*barrier)(TestEnv* env);
  int (*allreduceMinInt)(TestEnv* env, int local);
  bool (*isRank0Printer)(TestEnv* env);
  void* ctx;
};

/* ======================================================================
 * Test descriptor.
 * ====================================================================*/

enum PlacementKind {
  PL_RS = 0, /* {REPLICATE, SHARD(d)} — Python "rs" */
  PL_SR = 1, /* {SHARD(d), REPLICATE} — Python "sr" */
  PL_REPL = 2, /* {REPLICATE, REPLICATE} — full-replicate, 1D mesh */
};

struct TestCase {
  std::string group;
  std::string name; /* full case name; populated by builders */
  int ndims;
  size_t globalDims[3];

  int srcDim0, dstDim0; /* mesh axis-0 size; 0 = "1D mesh, axis-0 is 1" */
  int srcShardDim; /* tensor dim sharded; -1 if PL_REPL */
  int dstShardDim;
  PlacementKind srcPl, dstPl;

  size_t elementSize; /* 1, 2, 4 */

  int worldMin; /* skip if worldSize < this */
  int worldDivisor; /* skip if worldSize % this != 0 */

  int srcRatioNum, dstRatioNum; /* (0,0) ⇒ even split */
};

static inline void printTo(const TestCase& tc, std::ostream* os) {
  *os << tc.name;
}

struct BasicApiCliArgs {
  int requestedRanks = 0; /* 0 = use cudaGetDeviceCount() */
  bool listOnly = false;
  bool verbose = false;
  const char* filter = nullptr;
  const char* algorithm = "ring";
  const char* lbMode = "uniform";
  int maxWorld = 0; /* 0 = unrestricted */
  int minWorld = 0; /* 0 = unrestricted */
};

static void basicApiPrintUsage(const char* prog, const char* usageFmt, bool allowRankCount, bool allowAlgorithmAll) {
  printf("Usage: ");
  printf(usageFmt, prog);
  printf("\n\nOptions:\n");
  if (allowRankCount) {
    printf("  -N <ranks>                   Number of ranks/threads (<= device "
           "count)\n");
  }
  printf("  --list                       List test cases (with min world) and "
         "exit\n");
  printf("  --filter <substr>            Run only cases whose name contains "
         "<substr>\n");
  printf("  --max-world <N>              Skip cases whose minimum world > N\n");
  printf("                               (lets a CI script pre-filter by "
         "allocation)\n");
  printf("  --min-world <N>              Skip cases whose minimum world < N\n");
  printf("                               (combine with --max-world to bin by "
         "rank tier)\n");
  printf("  --algorithm ring|direct%s  Reshard algorithm (default: ring%s)\n", allowAlgorithmAll ? "|all" : "   ",
         allowAlgorithmAll ? "; 'all' registers one gtest case per algorithm" : "");
  printf("  --lb-mode  uniform|node      Load-balance mode (default: "
         "uniform)\n");
  printf("  --verbose                    Verbose / per-rank output\n");
  printf("  --help                       Print this help\n");
}

static bool basicApiConsumeGtestArg(int argc, char** argv, int* i) {
  const char* arg = argv[*i];
  if (strncmp(arg, "--gtest_", 8) != 0) return false;
  if (strchr(arg, '=') == nullptr && *i + 1 < argc && argv[*i + 1][0] != '-') ++(*i);
  return true;
}

static const char* basicApiRequireValue(int argc, char** argv, int* i) {
  if (*i + 1 >= argc) {
    fprintf(stderr, "Missing value for %s\n", argv[*i]);
    _Exit(2);
  }
  return argv[++(*i)];
}

static int basicApiParseIntArg(const char* value) {
  char* end = nullptr;
  long n = strtol(value, &end, 10);
  if (end == value) return 0;
  if (n < INT_MIN) return INT_MIN;
  if (n > INT_MAX) return INT_MAX;
  return (int)n;
}

static BasicApiCliArgs basicApiParseCli(int argc, char** argv, const char* usageFmt, bool allowRankCount,
                                        bool allowAlgorithmAll) {
  BasicApiCliArgs a;
  for (int i = 1; i < argc; i++) {
    const char* k = argv[i];
    if (basicApiConsumeGtestArg(argc, argv, &i)) continue;
    if (allowRankCount && strcmp(k, "-N") == 0) {
      a.requestedRanks = basicApiParseIntArg(basicApiRequireValue(argc, argv, &i));
    } else if (strcmp(k, "--list") == 0) {
      a.listOnly = true;
    } else if (strcmp(k, "--verbose") == 0) {
      a.verbose = true;
    } else if (strcmp(k, "--filter") == 0) {
      a.filter = basicApiRequireValue(argc, argv, &i);
    } else if (strcmp(k, "--max-world") == 0) {
      a.maxWorld = basicApiParseIntArg(basicApiRequireValue(argc, argv, &i));
    } else if (strcmp(k, "--min-world") == 0) {
      a.minWorld = basicApiParseIntArg(basicApiRequireValue(argc, argv, &i));
    } else if (strcmp(k, "--algorithm") == 0) {
      a.algorithm = basicApiRequireValue(argc, argv, &i);
      if (!allowAlgorithmAll && strcmp(a.algorithm, "all") == 0) {
        fprintf(stderr, "--algorithm all is supported by MPI only\n");
        _Exit(2);
      }
    } else if (strcmp(k, "--lb-mode") == 0) {
      a.lbMode = basicApiRequireValue(argc, argv, &i);
    } else if (strcmp(k, "--help") == 0) {
      basicApiPrintUsage(argv[0], usageFmt, allowRankCount, allowAlgorithmAll);
      _Exit(0);
    } else {
      fprintf(stderr, "Unknown argument: %s\n", k);
      basicApiPrintUsage(argv[0], usageFmt, allowRankCount, allowAlgorithmAll);
      _Exit(2);
    }
  }
  return a;
}

/* ======================================================================
 * Naming helpers.
 * ====================================================================*/

static const char* plName(PlacementKind p) {
  switch (p) {
  case PL_RS:
    return "rs";
  case PL_SR:
    return "sr";
  case PL_REPL:
    return "repl";
  }
  return "?";
}

static std::string formatGlobalDims(int ndims, const size_t gd[3]) {
  char buf[64];
  if (ndims == 1) snprintf(buf, sizeof(buf), "%zu", gd[0]);
  else if (ndims == 2) snprintf(buf, sizeof(buf), "%zux%zu", gd[0], gd[1]);
  else snprintf(buf, sizeof(buf), "%zux%zux%zu", gd[0], gd[1], gd[2]);
  return buf;
}

static std::string buildCaseName(const TestCase& tc) {
  char buf[256];
  std::string gd = formatGlobalDims(tc.ndims, tc.globalDims);

  if (tc.srcPl == PL_REPL && tc.dstPl == PL_REPL) {
    snprintf(buf, sizeof(buf), "%s[gd=%s,esz=%zu]", tc.group.c_str(), gd.c_str(), tc.elementSize);
  } else if (tc.srcDim0 == 0 && tc.dstDim0 == 0) {
    /* 1D mesh per side */
    snprintf(buf, sizeof(buf), "%s[gd=%s,sd=%d/%d,esz=%zu]", tc.group.c_str(), gd.c_str(), tc.srcShardDim,
             tc.dstShardDim, tc.elementSize);
  } else if (tc.srcRatioNum == 0 && tc.dstRatioNum == 0) {
    snprintf(buf, sizeof(buf), "%s[gd=%s,m=%dx%d_%s/%s,sd=%d/%d,esz=%zu]", tc.group.c_str(), gd.c_str(), tc.srcDim0,
             tc.dstDim0, plName(tc.srcPl), plName(tc.dstPl), tc.srcShardDim, tc.dstShardDim, tc.elementSize);
  } else {
    snprintf(buf, sizeof(buf), "%s[gd=%s,m=%dx%d_%s/%s,sd=%d/%d,esz=%zu,ratio=%d:%d]", tc.group.c_str(), gd.c_str(),
             tc.srcDim0, tc.dstDim0, plName(tc.srcPl), plName(tc.dstPl), tc.srcShardDim, tc.dstShardDim, tc.elementSize,
             tc.srcRatioNum, tc.dstRatioNum);
  }
  return buf;
}

/* ======================================================================
 * Builders — one per Python test method.
 * ====================================================================*/

static void emitFullReplication(std::vector<TestCase>& cases) {
  /*  test_basic_api_full_replication
   *  - 1D mesh per side, both Replicate()
   *  - Tensor (200, 200), dtypes fp32 / bf16 / uint8
   *  - Pytest skips world < 8; we only need world >= 4 (2 src + 2 dst)
   *    since the C kernel has no inherent 8-rank requirement.
   */
  const size_t esz_list[] = {
    4, /* fp32 (size table) */
    2, /* bf16 (size table) */
    1, /* uint8 (size table default for esz=1) */
  };
  for (size_t esz : esz_list) {
    TestCase tc{};
    tc.group = "full_replication";
    tc.ndims = 2;
    tc.globalDims[0] = 200;
    tc.globalDims[1] = 200;
    tc.srcDim0 = 0; /* 1D mesh */
    tc.dstDim0 = 0;
    tc.srcShardDim = -1;
    tc.dstShardDim = -1;
    tc.srcPl = PL_REPL;
    tc.dstPl = PL_REPL;
    tc.elementSize = esz;
    tc.worldMin = 4;
    tc.worldDivisor = 2;
    tc.name = buildCaseName(tc);
    cases.push_back(std::move(tc));
  }
}

static void emitFullSharding(std::vector<TestCase>& cases) {
  /*  test_basic_api_full_sharding
   *  - 1D mesh per side, both Shard(d)
   *  - sharding_dims ∈ {(0,0),(0,1),(1,0),(1,1)}
   *  - Tensor (200, 200), dtypes fp32 / bf16 / uint8
   *  - Skip if world < 8 or world % 2 != 0
   */
  const int sd_list[][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
  const size_t esz_list[] = {
    4, /* fp32 */
    2, /* bf16 */
    1, /* uint8 */
  };
  for (auto& sd : sd_list) {
    for (size_t esz : esz_list) {
      TestCase tc{};
      tc.group = "full_sharding";
      tc.ndims = 2;
      tc.globalDims[0] = 200;
      tc.globalDims[1] = 200;
      tc.srcDim0 = 0;
      tc.dstDim0 = 0;
      tc.srcShardDim = sd[0];
      tc.dstShardDim = sd[1];
      tc.srcPl = PL_RS; /* 1D mesh: dims={1,N}, placement={REPL, SHARD(d)} */
      tc.dstPl = PL_RS;
      tc.elementSize = esz;
      tc.worldMin = 4;
      tc.worldDivisor = 2;
      tc.name = buildCaseName(tc);
      cases.push_back(std::move(tc));
    }
  }
}

static void emit2dPlacementMatrix(std::vector<TestCase>& cases, const char* group, size_t global0, size_t global1,
                                  int nShardsSrc, int nShardsDst, int ratioNumSrc, int ratioNumDst, size_t esz) {
  /* Inner helper used by 2d_placement, uneven_ratio, and
     tensor_size_sensitivity. */
  const int sd_list[][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
  const PlacementKind pl_list[][2] = {
    {PL_SR, PL_SR},
    {PL_RS, PL_RS},
    {PL_SR, PL_RS},
    {PL_RS, PL_SR},
  };
  const int totalRatio = (ratioNumSrc + ratioNumDst);
  const bool even = (ratioNumSrc == 0 && ratioNumDst == 0);

  /* Pytest enforces worldSize >= 8; the C kernel only needs the mesh
   * shapes to divide cleanly, so we use 4 as the floor and let runtime
   * feasibility checks (n_shards divides total, global divides shards)
   * skip configs that don't fit at small world sizes. The divisor is the
   * src+dst ratio sum (pytest's `world % (a+b)` rule), or 2 for an even
   * split.
   */
  int worldMin = 4;
  int worldDivisor;
  if (even) worldDivisor = 2;
  else worldDivisor = totalRatio;

  for (auto& sd : sd_list) {
    for (auto& pl : pl_list) {
      TestCase tc{};
      tc.group = group;
      tc.ndims = 2;
      tc.globalDims[0] = global0;
      tc.globalDims[1] = global1;
      tc.srcDim0 = nShardsSrc;
      tc.dstDim0 = nShardsDst;
      tc.srcShardDim = sd[0];
      tc.dstShardDim = sd[1];
      tc.srcPl = pl[0];
      tc.dstPl = pl[1];
      tc.elementSize = esz;
      tc.worldMin = worldMin;
      tc.worldDivisor = worldDivisor;
      tc.srcRatioNum = ratioNumSrc;
      tc.dstRatioNum = ratioNumDst;
      tc.name = buildCaseName(tc);
      cases.push_back(std::move(tc));
    }
  }
}

static void emit2dPlacement(std::vector<TestCase>& cases) {
  /*  test_basic_api_2d_placement
   *  - ratio (1,1)
   *  - n_shards ∈ {(2,4),(4,2),(2,2)}
   *  - Tensor (200, 200), dtype bf16
   */
  const int ns_list[][2] = {{2, 4}, {4, 2}, {2, 2}};
  for (auto& ns : ns_list) {
    emit2dPlacementMatrix(cases, "2d_placement", 200, 200, ns[0], ns[1], 0, 0, /* even split */
                          2); /* bf16 */
  }
}

static void emitUnevenRatio(std::vector<TestCase>& cases) {
  /*  test_basic_api_uneven_ratio
   *  - ratio ∈ {(3,1),(1,3)}
   *  - n_shards (2,2)
   *  - Tensor (240, 240), dtype bf16
   */
  const int ratio_list[][2] = {{3, 1}, {1, 3}};
  for (auto& r : ratio_list) emit2dPlacementMatrix(cases, "uneven_ratio", 240, 240, 2, 2, r[0], r[1], 2);
}

static void emitTensorSizeSensitivity(std::vector<TestCase>& cases) {
  /*  test_basic_api_tensor_size_sensitivity
   *  - n_shards (4,4)
   *  - global_shape ∈ {(576,576),(3072,6144),(3072,3072)}
   *  - dtype bf16
   *  - Skip the (rs,rs) large-shape cases on world < 32 (pytest does too,
   *    but our runtime skips on worldSize < worldMin=8 already; the
   *    fine-grained large-shape skip is mirrored by checking buffer size).
   */
  const size_t shape_list[][2] = {
    {576, 576},
    {3072, 6144},
    {3072, 3072},
  };
  for (auto& s : shape_list) emit2dPlacementMatrix(cases, "tensor_size_sensitivity", s[0], s[1], 4, 4, 0, 0, 2);
}

static void emitNdTensors(std::vector<TestCase>& cases) {
  /*  test_nd_tensors
   *  - 1D mesh per side, Shard(d), single shard
   *  - global_shape ∈ {(64,128,128),(128,64,64)}  (3D only; 4D skipped)
   *  - sharding_dims ∈ {(0,0),(0,1),(0,2),(2,0),(1,0),(1,1),(1,2)}
   *    except the historical (64,128,128), sd=(0,1) case owned by
   *    cross_dim_regression.
   *  - dtype bf16
   *  - Pytest requires world >= 32; the C kernel works with 2 ranks per
   *    side and the chosen 3D shapes are even-divisible by small shard
   *    counts, so we lower the floor to 4 and let runtime feasibility
   *    handle the rest.
   */
  const int sd_list[][2] = {
    {0, 0}, {0, 1}, {0, 2}, {2, 0}, {1, 0}, {1, 1}, {1, 2},
  };
  const size_t shape_list[][3] = {
    {64, 128, 128},
    {128, 64, 64},
  };
  for (auto& shape : shape_list) {
    for (auto& sd : sd_list) {
      if (shape[0] == 64 && shape[1] == 128 && shape[2] == 128 && sd[0] == 0 && sd[1] == 1) continue;
      /* Skip combos that exceed tensor ndim (mirrors pytest skip). */
      if (sd[0] >= 3 || sd[1] >= 3) continue;
      TestCase tc{};
      tc.group = "nd_tensors";
      tc.ndims = 3;
      tc.globalDims[0] = shape[0];
      tc.globalDims[1] = shape[1];
      tc.globalDims[2] = shape[2];
      tc.srcDim0 = 0;
      tc.dstDim0 = 0;
      tc.srcShardDim = sd[0];
      tc.dstShardDim = sd[1];
      tc.srcPl = PL_RS;
      tc.dstPl = PL_RS;
      tc.elementSize = 2; /* bf16 */
      tc.worldMin = 4;
      tc.worldDivisor = 2;
      tc.name = buildCaseName(tc);
      cases.push_back(std::move(tc));
    }
  }
}

/* ======================================================================
 * 1D tensor variants — placement / sharding coverage with ndims=1 and
 * sharding_dim always 0 (only one tensor axis).
 * The pytest source does not exercise 1D; we add coverage here because
 * the lib advertises 1 ≤ ndims ≤ 3 and 1D shards hit kernel paths that
 * the 2D/3D matrix doesn't.
 * ====================================================================*/

static void emit1dFullSharding(std::vector<TestCase>& cases) {
  /* Only sd=(0,0) is meaningful for ndims=1. */
  const size_t esz_list[] = {4, 2, 1};
  for (size_t esz : esz_list) {
    TestCase tc{};
    tc.group = "1d_full_sharding";
    tc.ndims = 1;
    tc.globalDims[0] = 8192;
    tc.globalDims[1] = 1;
    tc.globalDims[2] = 1;
    tc.srcDim0 = 0;
    tc.dstDim0 = 0;
    tc.srcShardDim = 0;
    tc.dstShardDim = 0;
    tc.srcPl = PL_RS;
    tc.dstPl = PL_RS;
    tc.elementSize = esz;
    tc.worldMin = 4;
    tc.worldDivisor = 2;
    tc.name = buildCaseName(tc);
    cases.push_back(std::move(tc));
  }
}

/* Inner helper for the 2D-mesh-on-1D-tensor groups. Mirrors
 * emit2dPlacementMatrix but with ndims=1 and sd fixed to (0,0).
 */
static void emit1dPlacementMatrix(std::vector<TestCase>& cases, const char* group, size_t global0, int nShardsSrc,
                                  int nShardsDst, int ratioNumSrc, int ratioNumDst, size_t esz) {
  const PlacementKind pl_list[][2] = {
    {PL_SR, PL_SR},
    {PL_RS, PL_RS},
    {PL_SR, PL_RS},
    {PL_RS, PL_SR},
  };
  const int totalRatio = (ratioNumSrc + ratioNumDst);
  const bool even = (ratioNumSrc == 0 && ratioNumDst == 0);

  int worldMin = 4;
  int worldDivisor = even ? 2 : totalRatio;

  for (auto& pl : pl_list) {
    TestCase tc{};
    tc.group = group;
    tc.ndims = 1;
    tc.globalDims[0] = global0;
    tc.globalDims[1] = 1;
    tc.globalDims[2] = 1;
    tc.srcDim0 = nShardsSrc;
    tc.dstDim0 = nShardsDst;
    tc.srcShardDim = 0;
    tc.dstShardDim = 0;
    tc.srcPl = pl[0];
    tc.dstPl = pl[1];
    tc.elementSize = esz;
    tc.worldMin = worldMin;
    tc.worldDivisor = worldDivisor;
    tc.srcRatioNum = ratioNumSrc;
    tc.dstRatioNum = ratioNumDst;
    tc.name = buildCaseName(tc);
    cases.push_back(std::move(tc));
  }
}

static void emit1d2dPlacement(std::vector<TestCase>& cases) {
  const int ns_list[][2] = {{2, 4}, {4, 2}, {2, 2}};
  for (auto& ns : ns_list) emit1dPlacementMatrix(cases, "1d_2d_placement", 8192, ns[0], ns[1], 0, 0, 2); /* bf16 */
}

static void emit1dUnevenRatio(std::vector<TestCase>& cases) {
  const int ratio_list[][2] = {{3, 1}, {1, 3}};
  for (auto& r : ratio_list) emit1dPlacementMatrix(cases, "1d_uneven_ratio", 16384, 2, 2, r[0], r[1], 2);
}

static void emit1dTensorSizeSensitivity(std::vector<TestCase>& cases) {
  const size_t shape_list[] = {16384, 1048576, 4194304};
  for (size_t s : shape_list) emit1dPlacementMatrix(cases, "1d_tensor_size_sensitivity", s, 4, 4, 0, 0, 2);
}

/* ======================================================================
 * Cross-dim regression group — hand-picked shapes from historical bugs.
 *
 * Each case is a cross-dim layout that has previously broken either the
 * non-transpose RING / DIRECT path or the `shouldTransposeForCrossDim`-
 * gated transpose path.  Curated, intentionally small — meant to be
 * run quickly via `--filter cross_dim_regression` as a fast targeted
 * gate.  Not a substitute for the full 2d_placement / nd_tensors
 * matrices; complements them.
 *
 * Mapping:
 *   issue !4 — 3D, sd=0/1, dstShardDim != ndims-1 → transpose path
 *              skipped, non-transpose RING/DIRECT was broken.
 *   issue !5 — 2D, sd=0/1, mesh 2x4, rs/rs placement → 2D transpose
 *              path (enabled by bc0b99c) hangs.
 * ====================================================================*/

static void emitCrossDimRegression(std::vector<TestCase>& cases) {
  /* issue !5: 2D mesh 2x4, sd=0/1, four placement permutations.
   * worldMin = src_shards * dst_shards = 2 * 4 = 8 (even split → /2). */
  {
    const PlacementKind pl_list[][2] = {
      {PL_SR, PL_SR},
      {PL_RS, PL_RS},
      {PL_SR, PL_RS},
      {PL_RS, PL_SR},
    };
    for (auto& pl : pl_list) {
      TestCase tc{};
      tc.group = "cross_dim_regression";
      tc.ndims = 2;
      tc.globalDims[0] = 200;
      tc.globalDims[1] = 200;
      tc.srcDim0 = 2;
      tc.dstDim0 = 4;
      tc.srcShardDim = 0;
      tc.dstShardDim = 1;
      tc.srcPl = pl[0];
      tc.dstPl = pl[1];
      tc.elementSize = 2; /* bf16 */
      tc.worldMin = 8;
      tc.worldDivisor = 2;
      tc.name = buildCaseName(tc);
      cases.push_back(std::move(tc));
    }
  }

  /* issue !4: 3D, sd=0/1, dstShardDim is NOT innermost so the
   * transpose gate stays off.  1D mesh per side (dim0=0) with rs/rs
   * placement; each side 4 shards → worldMin = 8. */
  {
    TestCase tc{};
    tc.group = "cross_dim_regression";
    tc.ndims = 3;
    tc.globalDims[0] = 64;
    tc.globalDims[1] = 128;
    tc.globalDims[2] = 128;
    tc.srcDim0 = 0;
    tc.dstDim0 = 0;
    tc.srcShardDim = 0;
    tc.dstShardDim = 1;
    tc.srcPl = PL_RS;
    tc.dstPl = PL_RS;
    tc.elementSize = 2;
    tc.worldMin = 8;
    tc.worldDivisor = 2;
    tc.name = buildCaseName(tc);
    cases.push_back(std::move(tc));
  }
}

static std::vector<TestCase> buildAllTestCases() {
  std::vector<TestCase> cases;
  emitFullReplication(cases);
  emitFullSharding(cases);
  emit2dPlacement(cases);
  emitUnevenRatio(cases);
  emitTensorSizeSensitivity(cases);
  emitNdTensors(cases);
  /* 1D tensor groups (extends pytest matrix). */
  emit1dFullSharding(cases);
  emit1d2dPlacement(cases);
  emit1dUnevenRatio(cases);
  emit1dTensorSizeSensitivity(cases);
  /* Targeted regression coverage for historical cross-dim bugs. */
  emitCrossDimRegression(cases);
  return cases;
}

/* ======================================================================
 * Mesh / shard math (no NCCL calls).
 * ====================================================================*/

struct MeshLayout {
  int dims[2];
  int placement[2];
  int startRank;
  int shardCount; /* 1 if PL_REPL */
  int shardDim; /* -1 if PL_REPL */
};

static int shardCountForMesh(const TestCase& tc, bool isSrc, int dim0, int dim1) {
  PlacementKind pl = isSrc ? tc.srcPl : tc.dstPl;
  switch (pl) {
  case PL_REPL:
    return 1;
  case PL_RS:
    return dim1; /* axis 1 is shard */
  case PL_SR:
    return dim0; /* axis 0 is shard */
  }
  return 1;
}

static int shardIdxForRank(const MeshLayout& m, PlacementKind pl, int rank) {
  int local = rank - m.startRank;
  switch (pl) {
  case PL_REPL:
    return 0;
  case PL_RS:
    return local % m.dims[1];
  case PL_SR:
    return local / m.dims[1];
  }
  return 0;
}

static void buildMesh(MeshLayout* out, PlacementKind pl, int shardDim, int dim0, int dim1, int startRank) {
  out->dims[0] = dim0;
  out->dims[1] = dim1;
  out->startRank = startRank;
  out->shardDim = (pl == PL_REPL) ? -1 : shardDim;
  if (pl == PL_REPL) {
    /* Encode "full replication" as a 1-shard PL_RS layout: every
     * rank still owns the full tensor, shardCount = 1 keeps the
     * expected global-range math simple, and the kernel goes through
     * the well-tested sharded path. A {REPLICATE, REPLICATE} mesh
     * lands in a degenerate prepare branch that the test suite does
     * not currently exercise.
     */
    out->placement[0] = NCCLXFER_RESHARD_REPLICATE;
    out->placement[1] = NCCLXFER_RESHARD_SHARD(0);
    out->shardCount = 1;
  } else if (pl == PL_RS) {
    out->placement[0] = NCCLXFER_RESHARD_REPLICATE;
    out->placement[1] = NCCLXFER_RESHARD_SHARD(shardDim);
    out->shardCount = dim1;
  } else { /* PL_SR */
    out->placement[0] = NCCLXFER_RESHARD_SHARD(shardDim);
    out->placement[1] = NCCLXFER_RESHARD_REPLICATE;
    out->shardCount = dim0;
  }
}

/* ======================================================================
 * Per-case execution result.
 * ====================================================================*/

enum CaseStatus {
  CASE_PASS,
  CASE_FAIL,
  CASE_SKIP
};

struct CaseResult {
  CaseStatus status;
  const char* skipReason; /* set when status == CASE_SKIP */
  const char* failReason; /* set when status == CASE_FAIL */
};

static inline CaseResult makeSkip(const char* reason) {
  return CaseResult{CASE_SKIP, reason, nullptr};
}
static inline CaseResult makeFail(const char* reason) {
  return CaseResult{CASE_FAIL, nullptr, reason};
}
static inline CaseResult makePass() {
  return CaseResult{CASE_PASS, nullptr, nullptr};
}

/* ======================================================================
 * Buffer-size estimator.
 * ====================================================================*/

static size_t maxLocalBytes(const TestCase& tc, int worldSize) {
  /* Conservative upper bound: assume the tensor is unsharded along all
   * dims (i.e. local size == global size). Multiplied by elementSize.
   * This dominates the actual per-case need and saves us the trouble
   * of recomputing per (src/dst) and per shard split.
   */
  (void)worldSize;
  size_t total = tc.globalDims[0];
  for (int d = 1; d < tc.ndims; d++) total *= tc.globalDims[d];
  return total * tc.elementSize;
}

static size_t computeMaxBufferBytes(const std::vector<TestCase>& cases, int worldSize) {
  size_t mx = 4096; /* NCCL min alloc */
  for (auto& tc : cases) {
    size_t need = maxLocalBytes(tc, worldSize);
    if (need > mx) mx = need;
  }
  /* Round up to 4 KiB. */
  const size_t pg = 4096;
  return ((mx + pg - 1) / pg) * pg;
}

/* ======================================================================
 * Feasibility helpers — shared between runOneCase() and
 * computeMinWorldForCase().
 * ====================================================================*/

struct CaseShape {
  int srcTotal, dstTotal;
  int srcDim0, srcDim1;
  int dstDim0, dstDim1;
  int srcShardCount, dstShardCount;
};

/* Returns true if test case `tc` is feasible at worldSize W. On false,
 * `*skipReason` (if non-null) is set to a static string describing why.
 * On true, `*shape` (if non-null) is filled with the resolved layout.
 *
 * This is the single source of truth for "does this case run at world W?".
 * runOneCase() uses it for the feasibility phase; computeMinWorldForCase()
 * iterates over W until it returns true.
 */
static bool caseFeasibleAt(const TestCase& tc, int w, CaseShape* shape = nullptr, const char** skipReason = nullptr) {
  auto fail = [&](const char* r) {
    if (skipReason != nullptr) *skipReason = r;
    return false;
  };

  if (w < tc.worldMin) return fail("worldSize below minimum");
  if (tc.worldDivisor != 0 && (w % tc.worldDivisor) != 0) return fail("worldSize not divisible by required factor");

  int srcTotal, dstTotal;
  if (tc.srcRatioNum == 0 && tc.dstRatioNum == 0) {
    srcTotal = w / 2;
    dstTotal = w - srcTotal;
  } else {
    int totalRatio = tc.srcRatioNum + tc.dstRatioNum;
    srcTotal = w * tc.srcRatioNum / totalRatio;
    dstTotal = w - srcTotal;
  }
  if (srcTotal + dstTotal != w || srcTotal == 0 || dstTotal == 0) return fail("ratio yields empty side");

  int srcDim0, srcDim1, dstDim0, dstDim1;
  if (tc.srcPl == PL_REPL) {
    srcDim0 = srcTotal;
    srcDim1 = 1;
  } else {
    srcDim0 = (tc.srcDim0 == 0) ? 1 : tc.srcDim0;
    if (srcTotal % srcDim0 != 0) return fail("srcTotal not divisible by srcDim0");
    srcDim1 = srcTotal / srcDim0;
  }
  if (tc.dstPl == PL_REPL) {
    dstDim0 = dstTotal;
    dstDim1 = 1;
  } else {
    dstDim0 = (tc.dstDim0 == 0) ? 1 : tc.dstDim0;
    if (dstTotal % dstDim0 != 0) return fail("dstTotal not divisible by dstDim0");
    dstDim1 = dstTotal / dstDim0;
  }

  int srcShardCount = shardCountForMesh(tc, /*isSrc=*/true, srcDim0, srcDim1);
  int dstShardCount = shardCountForMesh(tc, /*isSrc=*/false, dstDim0, dstDim1);

  if (tc.srcShardDim >= 0 && tc.globalDims[tc.srcShardDim] % (size_t)srcShardCount != 0)
    return fail("global dim not divisible by src shard count");
  if (tc.dstShardDim >= 0 && tc.globalDims[tc.dstShardDim] % (size_t)dstShardCount != 0)
    return fail("global dim not divisible by dst shard count");

  if (shape != nullptr) {
    shape->srcTotal = srcTotal;
    shape->dstTotal = dstTotal;
    shape->srcDim0 = srcDim0;
    shape->srcDim1 = srcDim1;
    shape->dstDim0 = dstDim0;
    shape->dstDim1 = dstDim1;
    shape->srcShardCount = srcShardCount;
    shape->dstShardCount = dstShardCount;
  }
  return true;
}

/* Smallest world W (>= worldMin, <= bound) at which this case is
 * feasible. Returns -1 if no W up to `bound` works.
 */
static int computeMinWorldForCase(const TestCase& tc, int bound = 4096) {
  int divisor = tc.worldDivisor > 0 ? tc.worldDivisor : 1;
  int start = tc.worldMin;
  /* Round start up to the next multiple of divisor. */
  if (start % divisor != 0) start += divisor - (start % divisor);
  if (start < divisor) start = divisor;
  for (int w = start; w <= bound; w += divisor)
    if (caseFeasibleAt(tc, w)) return w;
  return -1;
}

static bool caseMatchesSelection(const TestCase& tc, const char* filter, int minWorld, int maxWorld) {
  if (filter != nullptr && filter[0] != '\0' && strstr(tc.name.c_str(), filter) == nullptr) return false;

  if (maxWorld > 0 || minWorld > 0) {
    int mw = computeMinWorldForCase(tc);
    if (mw < 0) return false;
    if (maxWorld > 0 && mw > maxWorld) return false;
    if (minWorld > 0 && mw < minWorld) return false;
  }
  return true;
}

static std::vector<TestCase> basicApiSelectCases(const std::vector<TestCase>& cases, const BasicApiCliArgs& cli) {
  std::vector<TestCase> selected;
  for (const TestCase& tc : cases)
    if (caseMatchesSelection(tc, cli.filter, cli.minWorld, cli.maxWorld)) selected.push_back(tc);
  return selected;
}

static std::string basicApiGtestCaseName(const std::string& caseName, size_t index, const char* prefix) {
  char indexBuf[48];
  if (prefix != nullptr) snprintf(indexBuf, sizeof(indexBuf), "%s_case%04zu_", prefix, index);
  else snprintf(indexBuf, sizeof(indexBuf), "case%04zu_", index);

  std::string out = indexBuf;
  for (unsigned char ch : caseName) out.push_back((std::isalnum(ch) != 0) ? (char)ch : '_');
  return out;
}

static void basicApiPrintCaseList(const std::vector<TestCase>& cases, const BasicApiCliArgs& cli, bool shouldPrint) {
  if (!shouldPrint) return;

  printf("# total_cases=%zu\n", cases.size());
  printf("# columns: idx minWorld name\n");
  for (size_t i = 0; i < cases.size(); ++i) {
    const TestCase& tc = cases[i];
    if (!caseMatchesSelection(tc, cli.filter, cli.minWorld, cli.maxWorld)) continue;
    int mw = computeMinWorldForCase(tc);
    if (mw > 0) printf("[%4zu] %4d %s\n", i, mw, tc.name.c_str());
    else printf("[%4zu]    - %s  // no feasible world\n", i, tc.name.c_str());
  }
}

static std::string basicApiCurrentGtestName() {
  const ::testing::TestInfo* info = ::testing::UnitTest::GetInstance()->current_test_info();
  if (info == nullptr) return "<unknown>";

  std::string name = info->test_case_name();
  name += ".";
  name += info->name();
  return name;
}

static void basicApiRecordFallbackSkip(int* skippedCases, const char* reason, bool shouldPrint) {
  if (!shouldPrint) return;

  const char* message = (reason != nullptr) ? reason : "skipped";
  ++(*skippedCases);
  ::testing::Test::RecordProperty("skipReason", message);
  printf("[  SKIPPED ] %s (%s)\n", basicApiCurrentGtestName().c_str(), message);
  fflush(stdout);
}

static void basicApiPrintFallbackSkipSummary(int skippedCases, bool shouldPrint) {
  if (!shouldPrint || skippedCases == 0) return;

  printf("[  SKIPPED ] %d basic_api case%s (vendored gtest reports "
         "skips as OK)\n",
         skippedCases, skippedCases == 1 ? "" : "s");
  fflush(stdout);
}

static bool basicApiRunAllAlgorithms(const BasicApiCliArgs& cli) {
  return strcmp(cli.algorithm, "all") == 0;
}

static const char* basicApiRequestedAlgorithmEnv(const BasicApiCliArgs& cli, bool shouldPrintUnknown) {
  if (strcmp(cli.algorithm, "direct") == 0) return "DIRECT";
  if (strcmp(cli.algorithm, "ring") == 0 || basicApiRunAllAlgorithms(cli)) return "RING";

  if (shouldPrintUnknown) fprintf(stderr, "Unknown algorithm '%s', defaulting to ring\n", cli.algorithm);
  return "RING";
}

static void basicApiConfigureReshardEnv(const BasicApiCliArgs& cli, const char* algorithmEnv) {
  testSetEnv("NCCLXFER_RESHARD_ALGORITHM", algorithmEnv);
  testSetEnv("NCCLXFER_RESHARD_LB_MODE", strcmp(cli.lbMode, "node") == 0 ? "NODE_AWARE" : "UNIFORM");
  if (cli.verbose) testSetEnv("NCCLXFER_RESHARD_LOG_LEVEL", "DEBUG");
}

static void basicApiPrintRuntimeSummary(const char* title, int worldSize, int deviceCount, const BasicApiCliArgs& cli,
                                        size_t bufferBytes, const char* countLabel, size_t count, bool shouldPrint) {
  if (!shouldPrint) return;

  printf("=== %s ===\n", title);
  printf("worldSize=%d, devices=%d, algo=%s, lb=%s\n", worldSize, deviceCount, cli.algorithm, cli.lbMode);
  printf("bufferBytes=%zu, %s=%zu\n", bufferBytes, countLabel, count);
  if (cli.filter != nullptr) printf("filter='%s'\n", cli.filter);
  if (cli.maxWorld > 0) printf("maxWorld=%d\n", cli.maxWorld);
  if (cli.minWorld > 0) printf("minWorld=%d\n", cli.minWorld);
  printf("\n");
  fflush(stdout);
}

/* ======================================================================
 * Per-case driver.
 *
 * Returns CaseResult; rank-aggregation happens in the bootstrap loop.
 * ====================================================================*/

static CaseResult runOneCase(const TestCase& tc, TestEnv* env) {
  /* ----- 1. feasibility check (single source of truth, see
   * caseFeasibleAt). On skip, also include the minimum world that
   * would let the case run so the user can plan a larger allocation.
   */
  CaseShape shape;
  const char* baseReason = nullptr;
  if (!caseFeasibleAt(tc, env->worldSize, &shape, &baseReason)) {
    static thread_local char buf[160];
    int minWorld = computeMinWorldForCase(tc);
    if (minWorld > 0) {
      snprintf(buf, sizeof(buf), "%s (needs world >= %d)", (baseReason != nullptr) ? baseReason : "infeasible",
               minWorld);
    } else {
      snprintf(buf, sizeof(buf), "%s (no feasible world)", (baseReason != nullptr) ? baseReason : "infeasible");
    }
    return makeSkip(buf);
  }
  int srcTotal = shape.srcTotal;
  int srcDim0 = shape.srcDim0, srcDim1 = shape.srcDim1;
  int dstDim0 = shape.dstDim0, dstDim1 = shape.dstDim1;
  int srcShardCount = shape.srcShardCount;
  int dstShardCount = shape.dstShardCount;

  /* ----- 2. build mesh layouts ----- */
  MeshLayout srcLayout, dstLayout;
  buildMesh(&srcLayout, tc.srcPl, tc.srcShardDim, srcDim0, srcDim1,
            /*startRank=*/0);
  buildMesh(&dstLayout, tc.dstPl, tc.dstShardDim, dstDim0, dstDim1,
            /*startRank=*/srcTotal);

  /* ----- 3. determine role and per-rank local dims (in elements) ----- */
  bool isSrc = (env->rank < srcTotal);
  bool isDst = !isSrc;

  size_t srcLocalDimsElems[3] = {tc.globalDims[0], tc.globalDims[1], tc.globalDims[2]};
  size_t dstLocalDimsElems[3] = {tc.globalDims[0], tc.globalDims[1], tc.globalDims[2]};
  if (tc.srcShardDim >= 0) srcLocalDimsElems[tc.srcShardDim] /= (size_t)srcShardCount;
  if (tc.dstShardDim >= 0) dstLocalDimsElems[tc.dstShardDim] /= (size_t)dstShardCount;

  /* Local *byte* dims used by the validator: multiply innermost dim by
   * elementSize (the validator works at byte granularity).
   */
  size_t srcLocalBytesDims[3] = {srcLocalDimsElems[0], srcLocalDimsElems[1], srcLocalDimsElems[2]};
  size_t dstLocalBytesDims[3] = {dstLocalDimsElems[0], dstLocalDimsElems[1], dstLocalDimsElems[2]};
  int innermost = tc.ndims - 1;
  srcLocalBytesDims[innermost] *= tc.elementSize;
  dstLocalBytesDims[innermost] *= tc.elementSize;

  size_t myBytes = isSrc ? srcLocalBytesDims[0] * srcLocalBytesDims[1] * (tc.ndims == 3 ? srcLocalBytesDims[2] : 1) :
                           dstLocalBytesDims[0] * dstLocalBytesDims[1] * (tc.ndims == 3 ? dstLocalBytesDims[2] : 1);
  if (myBytes > env->bufferBytes) return makeSkip("local buffer exceeds preallocated max");

  /* ----- 6. window registration ----- */
  ncclWindow_t window = nullptr;
  TEST_NCCLCHECK(ncclCommWindowRegister(env->comm, env->buffer, env->bufferBytes, &window, NCCL_WIN_COLL_SYMMETRIC));
  TEST_CUDACHECK(cudaMemsetAsync(env->buffer, 0xDE, env->bufferBytes, env->stream));

  /* ----- 7. init source data ----- */
  if (isSrc) {
    int srcShardIdx = shardIdxForRank(srcLayout, tc.srcPl, env->rank);
    int sd = (tc.srcShardDim >= 0) ? tc.srcShardDim : -1;
    int sc = (tc.srcShardDim >= 0) ? srcShardCount : 1;
    testInitSourceData((char*)env->buffer, srcLocalBytesDims, tc.ndims, sd, srcShardIdx, sc, env->stream);
  }
  TEST_CUDACHECK(cudaStreamSynchronize(env->stream));
  env->barrier(env);

  /* ----- 8. resharding call ----- */
  ncclXferReshardMesh_t srcMesh{};
  ncclXferReshardMesh_t dstMesh{};
  srcMesh.dims[0] = srcLayout.dims[0];
  srcMesh.dims[1] = srcLayout.dims[1];
  srcMesh.startRank = srcLayout.startRank;
  srcMesh.placement[0] = srcLayout.placement[0];
  srcMesh.placement[1] = srcLayout.placement[1];

  dstMesh.dims[0] = dstLayout.dims[0];
  dstMesh.dims[1] = dstLayout.dims[1];
  dstMesh.startRank = dstLayout.startRank;
  dstMesh.placement[0] = dstLayout.placement[0];
  dstMesh.placement[1] = dstLayout.placement[1];

  /* The harness works at byte granularity, so pass the dtype whose
   * size matches tc.elementSize (1 / 2 / 4 / 8). */
  static const ncclDataType_t dtype_for_size[] = {
    /* 0 */ ncclInt8,
    /* 1 */ ncclInt8,
    /* 2 */ ncclBfloat16,
    /* 3 */ ncclInt8, /* unreachable */
    /* 4 */ ncclFloat32,
    /* 5 */ ncclInt8,     ncclInt8, ncclInt8,
    /* 8 */ ncclFloat64,
  };
  ncclDataType_t dtype = (tc.elementSize <= 8) ? dtype_for_size[tc.elementSize] : ncclBfloat16;

  ncclXferDistTensor_t srcT = {};
  srcT.dataPtr = isSrc ? env->buffer : nullptr;
  srcT.ndims = tc.ndims;
  srcT.dtype = dtype;
  srcT.mesh = &srcMesh;
  if (isSrc)
    for (int d = 0; d < tc.ndims; d++) srcT.localShape[d] = srcLocalDimsElems[d];

  ncclXferDistTensor_t dstT = {};
  dstT.dataPtr = isDst ? env->buffer : nullptr;
  dstT.ndims = tc.ndims;
  dstT.dtype = dtype;
  dstT.mesh = &dstMesh;
  if (isDst)
    for (int d = 0; d < tc.ndims; d++) dstT.localShape[d] = dstLocalDimsElems[d];

  ncclResult_t r = ncclXferReshardWithWindow(env->comm, window, &srcT, &dstT, env->stream);

  if (r != ncclSuccess) {
    TEST_NCCLCHECK(ncclCommWindowDeregister(env->comm, window));
    return makeFail("ncclXferReshardWithWindow returned error");
  }

  TEST_CUDACHECK(cudaStreamSynchronize(env->stream));
  env->barrier(env);

  /* ----- 9. validate dest ----- */
  int localOk = 1;
  if (isDst) {
    int dstShardIdx = shardIdxForRank(dstLayout, tc.dstPl, env->rank);
    int sd = (tc.dstShardDim >= 0) ? tc.dstShardDim : -1;
    int sc = (tc.dstShardDim >= 0) ? dstShardCount : 1;
    bool ok = testValidateDestData((const char*)env->buffer, dstLocalBytesDims, tc.ndims, sd, dstShardIdx, sc,
                                   env->rank, env->stream, nullptr);
    localOk = ok ? 1 : 0;
  }
  int globalOk = env->allreduceMinInt(env, localOk);

  TEST_NCCLCHECK(ncclCommWindowDeregister(env->comm, window));

  if (env->verbose && env->isRank0Printer(env))
    printf("    [rank %d] localOk=%d, globalOk=%d, myBytes=%zu\n", env->rank, localOk, globalOk, myBytes);

  return (globalOk != 0) ? makePass() : makeFail("byte-pattern mismatch");
}

#endif /* TESTS_BASIC_API_TEST_CORE_H_ */
