/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more information
 *************************************************************************/

#include <cuda.h>
#include <cuda_runtime.h>
#include <mpi.h>
#include <nccl.h>
#include <nccl_device.h>

#include "common.h"
#include "cumod_common.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

constexpr size_t Count = 256;
constexpr size_t Offset = 4;
constexpr size_t Guard = 4;
constexpr int Threads = 128;
constexpr float GuardValue = -1.0f;

enum ReduceCopyApi {
  LsaReduceSum,
  MultimemReduceSum,
  LsaCopy,
  MultimemCopy,
  LsaReduceSumCopy,
  MultimemReduceSumCopy,
  LocalReduceSumCopy,
};

struct ApiSpec {
  ReduceCopyApi api;
  char const* name;
};

constexpr ApiSpec ApiSpecs[] = {
    {LsaReduceSum, "ncclIrLsaReduceSum_F32"},
    {MultimemReduceSum, "ncclIrMultimemReduceSum_F32"},
    {LsaCopy, "ncclIrLsaCopy_F32"},
    {MultimemCopy, "ncclIrMultimemCopy_F32"},
    {LsaReduceSumCopy, "ncclIrLsaReduceSumCopy_F32"},
    {MultimemReduceSumCopy, "ncclIrMultimemReduceSumCopy_F32"},
    {LocalReduceSumCopy, "ncclIrLocalReduceSumCopy_F32"},
};

struct Buffers {
  float* src = nullptr;
  float* dst = nullptr;
  ncclWindow_t srcWindow = nullptr;
  ncclWindow_t dstWindow = nullptr;
  float* mcSrc = nullptr;
  float* mcDst = nullptr;
};

int localDevice() {
  MPI_Comm localComm;
  MPICHECK(MPI_Comm_split_type(
      MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &localComm));
  int device = 0;
  MPICHECK(MPI_Comm_rank(localComm, &device));
  MPICHECK(MPI_Comm_free(&localComm));
  return device;
}

float sourceValue(int teamRank, int lane, size_t index) {
  return float(32 * (teamRank + 1) + 4 * lane + int(index % 3));
}

float reducedValue(int teamSize, size_t index) {
  return float(16 * teamSize * (teamSize + 1) +
               teamSize * int(index % 3));
}

std::vector<float> sourceValues(int teamRank, size_t stride) {
  size_t elements = 2 * stride;
  std::vector<float> values(elements, GuardValue);
  for (int lane = 0; lane < 2; ++lane) {
    size_t base = lane * stride + Offset;
    for (size_t index = 0; index < Count; ++index) {
      values[base + index] = sourceValue(teamRank, lane, index);
    }
  }
  return values;
}

void initializeBuffers(
    Buffers const& buffers, std::vector<float> const& source) {
  std::vector<float> destination(source.size(), GuardValue);
  CUDACHECK(cudaMemcpy(
      buffers.src, source.data(), source.size() * sizeof(float),
      cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(
      buffers.dst, destination.data(), destination.size() * sizeof(float),
      cudaMemcpyHostToDevice));
}

float expectedValue(
    ReduceCopyApi api, int teamRank, int teamSize, int lane, size_t index) {
  bool leader = teamRank == 0;
  switch (api) {
    case LsaReduceSum:
    case MultimemReduceSum:
      return leader && lane == 0
          ? reducedValue(teamSize, index) : GuardValue;
    case LsaCopy:
    case MultimemCopy:
      return lane == 0 ? sourceValue(0, 0, index) : GuardValue;
    case LsaReduceSumCopy:
    case MultimemReduceSumCopy:
      return lane == 0 ? reducedValue(teamSize, index) : GuardValue;
    case LocalReduceSumCopy:
      return leader
          ? sourceValue(0, 0, index) + sourceValue(0, 1, index)
          : GuardValue;
  }
  return GuardValue;
}

std::vector<float> destinationValues(
    ReduceCopyApi api, int teamRank, int teamSize, size_t stride) {
  size_t elements = 2 * stride;
  std::vector<float> values(elements, GuardValue);
  for (int lane = 0; lane < 2; ++lane) {
    size_t begin = lane * stride + Offset;
    for (size_t index = 0; index < Count; ++index) {
      values[begin + index] =
          expectedValue(api, teamRank, teamSize, lane, index);
    }
  }
  return values;
}

bool checkBuffer(
    ApiSpec const& spec, char const* label, float* buffer,
    std::vector<float> const& expected, int rank) {
  std::vector<float> actual(expected.size());
  CUDACHECK(cudaMemcpy(
      actual.data(), buffer, actual.size() * sizeof(float),
      cudaMemcpyDeviceToHost));
  if (actual == expected) return true;

  auto mismatch = std::mismatch(
      actual.begin(), actual.end(), expected.begin());
  size_t index = size_t(mismatch.first - actual.begin());
  std::fprintf(stderr,
      "Rank %d: %s %s mismatch at %zu: expected %.9g, got %.9g\n",
      rank, spec.name, label, index, expected[index], actual[index]);
  return false;
}

void destroyBuffers(ncclComm_t comm, Buffers& buffers) {
  if (buffers.dstWindow != nullptr)
    NCCLCHECK(ncclCommWindowDeregister(comm, buffers.dstWindow));
  if (buffers.srcWindow != nullptr)
    NCCLCHECK(ncclCommWindowDeregister(comm, buffers.srcWindow));
  if (buffers.dst != nullptr) NCCLCHECK(ncclMemFree(buffers.dst));
  if (buffers.src != nullptr) NCCLCHECK(ncclMemFree(buffers.src));
  buffers = Buffers{};
}

int run() {
  int rank = 0;
  int nRanks = 0;
  MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
  MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &nRanks));
  if (nRanks < 2) {
    if (rank == 0)
      std::fprintf(stderr, "ReduceCopy IR test requires at least two ranks\n");
    return EXIT_FAILURE;
  }

  int device = localDevice();
  CUDACHECK(cudaSetDevice(device));
  CU_CHECK(cuInit(0));

  ncclUniqueId id{};
  if (rank == 0) NCCLCHECK(ncclGetUniqueId(&id));
  MPICHECK(MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));

  ncclComm_t comm = nullptr;
  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
  config.blocking = 1;
  NCCLCHECK(ncclCommInitRankConfig(&comm, nRanks, id, rank, &config));

  ncclCommProperties_t properties = NCCL_COMM_PROPERTIES_INITIALIZER;
  NCCLCHECK(ncclCommQueryProperties(comm, &properties));
  if (!properties.deviceApiSupport || !properties.multimemSupport) {
    if (rank == 0)
      std::printf("SKIP: ReduceCopy IR test requires device API and multimem support\n");
    NCCLCHECK(ncclCommFinalize(comm));
    NCCLCHECK(ncclCommDestroy(comm));
    return EXIT_SUCCESS;
  }

  ncclTeam team = ncclTeamLsa(comm);
  int minimumTeamSize = 0;
  MPICHECK(MPI_Allreduce(
      &team.nRanks, &minimumTeamSize, 1, MPI_INT, MPI_MIN,
      MPI_COMM_WORLD));
  if (minimumTeamSize < 2) {
    if (rank == 0)
      std::printf("SKIP: ReduceCopy IR test requires two ranks per LSA team\n");
    NCCLCHECK(ncclCommFinalize(comm));
    NCCLCHECK(ncclCommDestroy(comm));
    return EXIT_SUCCESS;
  }

  size_t stride = Offset + Count + Guard;
  size_t bytes = 2 * stride * sizeof(float);
  size_t dataOffset = Offset * sizeof(float);
  Buffers buffers;
  NCCLCHECK(ncclMemAlloc(reinterpret_cast<void**>(&buffers.src), bytes));
  NCCLCHECK(ncclMemAlloc(reinterpret_cast<void**>(&buffers.dst), bytes));
  NCCLCHECK(ncclCommWindowRegister(
      comm, buffers.src, bytes, &buffers.srcWindow,
      NCCL_WIN_COLL_SYMMETRIC));
  NCCLCHECK(ncclCommWindowRegister(
      comm, buffers.dst, bytes, &buffers.dstWindow,
      NCCL_WIN_COLL_SYMMETRIC));
  void* mcSrc;
  void* mcDst;
  NCCLCHECK(ncclGetLsaMultimemDevicePointer(
      buffers.srcWindow, dataOffset, &mcSrc));
  NCCLCHECK(ncclGetLsaMultimemDevicePointer(
      buffers.dstWindow, dataOffset, &mcDst));
  buffers.mcSrc = static_cast<float*>(mcSrc);
  buffers.mcDst = static_cast<float*>(mcDst);

  ncclDevComm dcomm;
  ncclDevCommRequirements requirements = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
  requirements.lsaBarrierCount = 1;
  requirements.lsaMultimem = true;
  NCCLCHECK(ncclDevCommCreate(comm, &requirements, &dcomm));

  CUstream stream = nullptr;
  CU_CHECK(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));
  CUmodule module = nullptr;
#ifdef USE_LTO_IR
  initCumoduleFromLtoir(&module, "devapi_reduce_copy.ltoir");
#else
  initCumodule(&module, "devapi_reduce_copy_ir.cubin");
#endif
  CUfunction kernel = nullptr;
  initTestCaseKernel(module, &kernel, "runReduceCopyF32");

  bool passed = true;
  float* src = buffers.src + Offset;
  float* dst = buffers.dst + Offset;
  std::vector<float> expectedSource = sourceValues(team.rank, stride);
  for (ApiSpec const& spec : ApiSpecs) {
    initializeBuffers(buffers, expectedSource);
    MPICHECK(MPI_Barrier(MPI_COMM_WORLD));

    int api = int(spec.api);
    size_t count = Count;
    void* params[] = {
        &dcomm, &api, &buffers.srcWindow, &dataOffset, &src, &buffers.mcSrc,
        &buffers.dstWindow, &dataOffset, &dst, &buffers.mcDst, &stride, &count};
    CU_CHECK(cuLaunchKernel(
        kernel, 1, 1, 1, Threads, 1, 1, 0, stream, params, nullptr));
    CU_CHECK(cuStreamSynchronize(stream));
    MPICHECK(MPI_Barrier(MPI_COMM_WORLD));

    std::vector<float> expectedDestination =
        destinationValues(spec.api, team.rank, team.nRanks, stride);
    bool localPassed = checkBuffer(
        spec, "destination", buffers.dst, expectedDestination, rank) &&
        checkBuffer(spec, "source", buffers.src, expectedSource, rank);
    int localResult = localPassed ? 1 : 0;
    int globalResult = 0;
    MPICHECK(MPI_Allreduce(
        &localResult, &globalResult, 1, MPI_INT, MPI_MIN,
        MPI_COMM_WORLD));
    if (rank == 0)
      std::printf("%s: %s\n", spec.name, globalResult ? "PASS" : "FAIL");
    passed = passed && globalResult;
  }

  finiCumodule(&module);
  CU_CHECK(cuStreamDestroy(stream));
  NCCLCHECK(ncclDevCommDestroy(comm, &dcomm));
  destroyBuffers(comm, buffers);
  NCCLCHECK(ncclCommFinalize(comm));
  NCCLCHECK(ncclCommDestroy(comm));
  return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}

} // namespace

int main(int argc, char** argv) {
  MPICHECK(MPI_Init(&argc, &argv));
  int result = run();
  int globalResult = 0;
  MPICHECK(MPI_Allreduce(
      &result, &globalResult, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD));
  MPICHECK(MPI_Finalize());
  return globalResult;
}
