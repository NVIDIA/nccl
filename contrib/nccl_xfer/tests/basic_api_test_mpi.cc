/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information.
 ************************************************************************/

/*************************************************************************
 * basic_api_test_mpi — gtest-parametrized multi-process MPI driver for
 * the basic_api matrix. Each TestCase is one gtest parameter (or one
 * TestCase × algorithm parameter when --algorithm all is used).
 *
 * Bootstrap: standard ncclGetUniqueId + MPI_Bcast + ncclCommInitRank.
 * Drives the shared core in basic_api_test_core.h.
 ************************************************************************/

#include <gtest/gtest.h>

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ostream>
#include <string>
#include <vector>

#include <mpi.h>
#include <cuda_runtime.h>
#include <nccl.h>

#include "nccl_xfer.h"
#include "basic_api_test_core.h"

namespace {

static MPI_Comm testMpiWorld() {
  return MPI_COMM_WORLD; // NOLINT(bugprone-casting-through-void)
}

static MPI_Datatype testMpiByte() {
  return MPI_BYTE; // NOLINT(bugprone-casting-through-void)
}

static MPI_Datatype testMpiInt() {
  return MPI_INT; // NOLINT(bugprone-casting-through-void)
}

static MPI_Op testMpiMin() {
  return MPI_MIN; // NOLINT(bugprone-casting-through-void)
}

#define MPICHECK(cmd)                                                  \
  do {                                                                 \
    int e = (cmd);                                                     \
    if (e != MPI_SUCCESS) {                                            \
      fprintf(stderr, "MPI error %s:%d: %d\n", __FILE__, __LINE__, e); \
      MPI_Abort(testMpiWorld(), 1);                                    \
    }                                                                  \
  } while (0)

/* ======================================================================
 * Bootstrap-specific TestEnv hooks.
 * ====================================================================*/

static void mpiBarrier(TestEnv* env) {
  (void)env;
  MPICHECK(MPI_Barrier(testMpiWorld()));
}

static int mpiAllreduceMinInt(TestEnv* env, int local) {
  (void)env;
  int g = local;
  MPICHECK(MPI_Allreduce(&local, &g, 1, testMpiInt(), testMpiMin(), testMpiWorld()));
  return g;
}

static bool mpiIsRank0Printer(TestEnv* env) {
  return env->rank == 0;
}

/* ======================================================================
 * gtest parameter registry state.
 * ====================================================================*/

struct MpiParam {
  TestCase tc;
  std::string algorithmEnv; /* RING or DIRECT */
};

/* gtest pretty-printer hook for MpiParam — found via ADL by
 * INSTANTIATE_TEST_SUITE_P's value-printer. The lookup is template-driven,
 * so plain `-Wunused-function` cannot see the use; mark it accordingly. */
[[maybe_unused]] static void printTo(const MpiParam& param, std::ostream* os) {
  *os << param.algorithmEnv << ":" << param.tc.name;
}

static BasicApiCliArgs gCli;
static std::vector<TestCase> gCases;
static int gWorldRank = 0;
static int gWorldSize = 0;
static int gNumDevices = 0;
static int gDevice = 0;
static ncclComm_t gComm = nullptr;
static cudaStream_t gStream = nullptr;
static void* gBuffer = nullptr;
static size_t gBufferBytes = 4096;
static std::string gActiveAlgorithm;
#if !defined(GTEST_SKIP)
static int gSkippedCases = 0;
#endif

static bool runAllAlgorithms() {
  return basicApiRunAllAlgorithms(gCli);
}

static const char* requestedAlgorithmEnv() {
  return basicApiRequestedAlgorithmEnv(gCli, gWorldRank == 0);
}

static std::vector<MpiParam> selectedParams() {
  std::vector<MpiParam> params;
  std::vector<TestCase> cases = basicApiSelectCases(gCases, gCli);

  if (runAllAlgorithms()) {
    const char* algos[] = {"RING", "DIRECT"};
    for (const char* algo : algos)
      for (const TestCase& tc : cases) params.push_back(MpiParam{tc, algo});
  } else {
    const char* algo = requestedAlgorithmEnv();
    for (const TestCase& tc : cases) params.push_back(MpiParam{tc, algo});
  }
  return params;
}

static std::string gtestCaseName(const ::testing::TestParamInfo<MpiParam>& info) {
  return basicApiGtestCaseName(info.param.tc.name, info.index, info.param.algorithmEnv.c_str());
}

#if !defined(GTEST_SKIP)
static void recordFallbackSkip(const char* reason) {
  basicApiRecordFallbackSkip(&gSkippedCases, reason, gWorldRank == 0);
}
#endif

static void activateAlgorithm(const std::string& algorithmEnv) {
  if (gActiveAlgorithm == algorithmEnv) return;

  basicApiConfigureReshardEnv(gCli, algorithmEnv.c_str());
  TEST_NCCLCHECK(ncclXferReshardFinalize());
  TEST_NCCLCHECK(ncclXferReshardInit(NULL));
  gActiveAlgorithm = algorithmEnv;
}

class BasicApiMpiTest : public ::testing::TestWithParam<MpiParam> {};

TEST_P(BasicApiMpiTest, Reshard) {
  const MpiParam& param = GetParam();
  SCOPED_TRACE(param.tc.name);
  SCOPED_TRACE(param.algorithmEnv);

  activateAlgorithm(param.algorithmEnv);

  TestEnv env{};
  env.rank = gWorldRank;
  env.worldSize = gWorldSize;
  env.device = gDevice;
  env.comm = gComm;
  env.stream = gStream;
  env.buffer = gBuffer;
  env.bufferBytes = gBufferBytes;
  env.verbose = gCli.verbose;
  env.barrier = mpiBarrier;
  env.allreduceMinInt = mpiAllreduceMinInt;
  env.isRank0Printer = mpiIsRank0Printer;
  env.ctx = nullptr;

  CaseResult res = runOneCase(param.tc, &env);

  if (res.status == CASE_SKIP) {
    env.barrier(&env);
#if defined(GTEST_SKIP)
    GTEST_SKIP() << ((res.skipReason != nullptr) ? res.skipReason : "skipped");
    return;
#else
    recordFallbackSkip(res.skipReason);
    return;
#endif
  }

  if (res.status == CASE_FAIL) ADD_FAILURE() << ((res.failReason != nullptr) ? res.failReason : "case failed");
  env.barrier(&env);
}

INSTANTIATE_TEST_CASE_P(Matrix, BasicApiMpiTest, ::testing::ValuesIn(selectedParams()), gtestCaseName);

static int initMpiRuntime() {
  TEST_CUDACHECK(cudaGetDeviceCount(&gNumDevices));
  gDevice = gWorldRank % (gNumDevices > 0 ? gNumDevices : 1);
  TEST_CUDACHECK(cudaSetDevice(gDevice));

  ncclUniqueId uid;
  if (gWorldRank == 0) TEST_NCCLCHECK(ncclGetUniqueId(&uid));
  MPICHECK(MPI_Bcast(&uid, sizeof(uid), testMpiByte(), 0, testMpiWorld()));

  TEST_NCCLCHECK(ncclCommInitRank(&gComm, gWorldSize, uid, gWorldRank));

  std::vector<TestCase> cases = basicApiSelectCases(gCases, gCli);
  gBufferBytes = computeMaxBufferBytes(cases, gWorldSize);

  const char* initialAlgorithm = requestedAlgorithmEnv();
  basicApiConfigureReshardEnv(gCli, initialAlgorithm);
  TEST_NCCLCHECK(ncclXferReshardInit(NULL));
  gActiveAlgorithm = initialAlgorithm;

  TEST_CUDACHECK(cudaStreamCreate(&gStream));
  TEST_NCCLCHECK(ncclMemAlloc(&gBuffer, gBufferBytes));

  if (gWorldRank == 0) {
    std::vector<MpiParam> params = selectedParams();
    basicApiPrintRuntimeSummary("basic_api_test_mpi (gtest)", gWorldSize, gNumDevices, gCli, gBufferBytes, "num_tests",
                                params.size(), true);
  }
  return 0;
}

static void shutdownMpiRuntime() {
  if (gBuffer != nullptr) TEST_NCCLCHECK(ncclMemFree(gBuffer));
  if (gStream != nullptr) TEST_CUDACHECK(cudaStreamDestroy(gStream));
  TEST_NCCLCHECK(ncclXferReshardFinalize());
  if (gComm != nullptr) TEST_NCCLCHECK(ncclCommDestroy(gComm));
}

} // namespace

int main(int argc, char** argv) {
  MPICHECK(MPI_Init(&argc, &argv));
  MPICHECK(MPI_Comm_rank(testMpiWorld(), &gWorldRank));
  MPICHECK(MPI_Comm_size(testMpiWorld(), &gWorldSize));

  gCli = basicApiParseCli(argc, argv, "mpirun -np <N> %s [options] [--gtest_* flags]", false, true);
  gCases = buildAllTestCases();

  if (gCli.listOnly) {
    basicApiPrintCaseList(gCases, gCli, gWorldRank == 0);
    MPICHECK(MPI_Finalize());
    return 0;
  }

  ::testing::InitGoogleTest(&argc, argv);
  if (gWorldRank != 0) {
    FILE* devnull = freopen("/dev/null", "w", stdout);
    if (devnull == nullptr) fprintf(stderr, "rank %d: failed to redirect stdout\n", gWorldRank);
  }

  if (::testing::GTEST_FLAG(list_tests)) {
    int rc = RUN_ALL_TESTS();
    MPICHECK(MPI_Finalize());
    return rc;
  }

  int initRc = initMpiRuntime();
  if (initRc != 0) {
    MPICHECK(MPI_Finalize());
    return initRc;
  }

  int rc = RUN_ALL_TESTS();
#if !defined(GTEST_SKIP)
  basicApiPrintFallbackSkipSummary(gSkippedCases, gWorldRank == 0);
#endif
  shutdownMpiRuntime();
  MPICHECK(MPI_Finalize());
  return rc;
}
