/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

// Two-rank smoke for NIIN's proxy-all AMO sidecar. Unlike the direct-WQE
// smoke, this deliberately sends a self AMO as well as remote integral and
// float/double/half AMOs: ProxyAll must put all of those operations through
// the same CPU RMW service rather than mixing local CUDA or NIC atomics.

#include <mpi.h>

#include <cuda_runtime.h>
#include <nccl.h>

#include <nvshmem.h>

#include "niin/host.h"
#include "niin/gpunetio/proxy_atomics/host.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {

constexpr size_t kHeapBytes = 4096;
constexpr size_t kIntOffset = 0;
constexpr size_t kInt64Offset = 16;
constexpr size_t kFloatOffset = 64;
constexpr size_t kDoubleOffset = 72;
constexpr size_t kHalfOffset = 96;
constexpr size_t kSelfIntOffset = 128;
constexpr size_t kConcurrentOffset = 160;
constexpr int kConcurrentThreads = 32;

struct Results {
  uint32_t failures;
  uint32_t remoteFetchAdd;
  uint32_t remoteCompareSwap;
  uint32_t remoteSwap;
  uint32_t remoteFetch;
  uint32_t remoteFetchInc;
  uint32_t remoteFetchAnd;
  uint32_t remoteFetchOr;
  uint32_t remoteFetchXor;
  uint64_t remoteFetchAdd64;
  uint64_t remoteCompareSwap64;
  uint64_t remoteSwap64;
  uint64_t remoteFetch64;
  uint64_t remoteFetchInc64;
  uint64_t remoteFetchAnd64;
  uint64_t remoteFetchOr64;
  uint64_t remoteFetchXor64;
  uint32_t selfFetchAdd;
  float floatFetchAdd;
  float floatSwap;
  float floatFetch;
  double doubleFetchAdd;
  double doubleSwap;
  double doubleFetch;
  uint16_t halfFetchAdd;
};

[[noreturn]] void fail(int rank, const char* where, const char* detail) {
  std::fprintf(stderr, "NIIN proxy atomic smoke [rank %d]: %s: %s\n", rank, where, detail);
  MPI_Abort(MPI_COMM_WORLD, 1);
  std::abort();
}

void checkCuda(int rank, cudaError_t status, const char* where) {
  if (status != cudaSuccess) fail(rank, where, cudaGetErrorString(status));
}

void checkNccl(int rank, ncclResult_t status, const char* where) {
  if (status != ncclSuccess) fail(rank, where, ncclGetErrorString(status));
}

int selectCudaDevice(int rank) {
  int deviceCount = 0;
  checkCuda(rank, cudaGetDeviceCount(&deviceCount), "cudaGetDeviceCount");
  if (deviceCount <= 0) fail(rank, "cudaGetDeviceCount", "no visible CUDA device");

  MPI_Comm localComm = MPI_COMM_NULL;
  if (MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &localComm) != MPI_SUCCESS)
    fail(rank, "MPI_Comm_split_type", "could not discover the node-local rank");
  int localRank = 0;
  if (MPI_Comm_rank(localComm, &localRank) != MPI_SUCCESS)
    fail(rank, "MPI_Comm_rank", "could not read the node-local rank");
  MPI_Comm_free(&localComm);

  // A launcher may expose one GPU per process, while a local development
  // launch can expose every GPU on the node.
  return deviceCount == 1 ? 0 : localRank % deviceCount;
}

void finalizeNiinContext(int rank, ncclComm_t comm, niinContext* deviceContext) {
  checkNccl(rank, ncclGroupStart(), "ncclGroupStart(niinFinalize)");
  checkNccl(rank, niinFinalize(comm, deviceContext), "niinFinalize");
  checkNccl(rank, ncclGroupEnd(), "ncclGroupEnd(niinFinalize)");
  checkCuda(rank, cudaFree(deviceContext), "cudaFree(device context)");
}

__global__ void issueProxyAtomics(niinContext* context, Results* results) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  // Exercise the public NIIN/NVSHMEM spelling, not just the transport
  // adapter. ProxyAll must intercept this before the normal self/LSA/direct
  // branches so all AMOs share one target CPU RMW ordering domain.
  niin_g_ctx = context;
  const int selfPe = nvshmem_my_pe();
  const int targetPe = (selfPe + 1) % nvshmem_n_pes();
  char* heap = static_cast<char*>(niin_heap_base());
  uint32_t* intTarget = reinterpret_cast<uint32_t*>(heap + kIntOffset);
  uint64_t* int64Target = reinterpret_cast<uint64_t*>(heap + kInt64Offset);
  float* floatTarget = reinterpret_cast<float*>(heap + kFloatOffset);
  double* doubleTarget = reinterpret_cast<double*>(heap + kDoubleOffset);
  __half* halfTarget = reinterpret_cast<__half*>(heap + kHalfOffset);
  uint32_t* selfIntTarget = reinterpret_cast<uint32_t*>(heap + kSelfIntOffset);

  uint32_t previous32 = nvshmem_uint32_atomic_fetch_add(intTarget, UINT32_C(5), targetPe);
  results->remoteFetchAdd = previous32;
  nvshmem_uint32_atomic_add(intTarget, UINT32_C(5), targetPe);
  previous32 = nvshmem_uint32_atomic_compare_swap(intTarget, UINT32_C(20), UINT32_C(30), targetPe);
  results->remoteCompareSwap = previous32;
  previous32 = nvshmem_uint32_atomic_swap(intTarget, UINT32_C(40), targetPe);
  results->remoteSwap = previous32;
  previous32 = nvshmem_uint32_atomic_fetch(intTarget, targetPe);
  results->remoteFetch = previous32;
  nvshmem_uint32_atomic_set(intTarget, UINT32_C(50), targetPe);
  nvshmem_uint32_atomic_inc(intTarget, targetPe);
  previous32 = nvshmem_uint32_atomic_fetch_inc(intTarget, targetPe);
  results->remoteFetchInc = previous32;
  previous32 = nvshmem_uint32_atomic_fetch_and(intTarget, UINT32_C(0x3f), targetPe);
  results->remoteFetchAnd = previous32;
  nvshmem_uint32_atomic_and(intTarget, UINT32_C(0x0f), targetPe);
  previous32 = nvshmem_uint32_atomic_fetch_or(intTarget, UINT32_C(0x10), targetPe);
  results->remoteFetchOr = previous32;
  nvshmem_uint32_atomic_or(intTarget, UINT32_C(0x20), targetPe);
  previous32 = nvshmem_uint32_atomic_fetch_xor(intTarget, UINT32_C(0x3), targetPe);
  results->remoteFetchXor = previous32;
  nvshmem_uint32_atomic_xor(intTarget, UINT32_C(0x1), targetPe);

  uint64_t previous64 = nvshmem_uint64_atomic_fetch_add(int64Target, UINT64_C(5), targetPe);
  results->remoteFetchAdd64 = previous64;
  nvshmem_uint64_atomic_add(int64Target, UINT64_C(5), targetPe);
  previous64 =
      nvshmem_uint64_atomic_compare_swap(int64Target, UINT64_C(20), UINT64_C(30), targetPe);
  results->remoteCompareSwap64 = previous64;
  previous64 = nvshmem_uint64_atomic_swap(int64Target, UINT64_C(40), targetPe);
  results->remoteSwap64 = previous64;
  previous64 = nvshmem_uint64_atomic_fetch(int64Target, targetPe);
  results->remoteFetch64 = previous64;
  nvshmem_uint64_atomic_set(int64Target, UINT64_C(50), targetPe);
  nvshmem_uint64_atomic_inc(int64Target, targetPe);
  previous64 = nvshmem_uint64_atomic_fetch_inc(int64Target, targetPe);
  results->remoteFetchInc64 = previous64;
  previous64 = nvshmem_uint64_atomic_fetch_and(int64Target, UINT64_C(0x3f), targetPe);
  results->remoteFetchAnd64 = previous64;
  nvshmem_uint64_atomic_and(int64Target, UINT64_C(0x0f), targetPe);
  previous64 = nvshmem_uint64_atomic_fetch_or(int64Target, UINT64_C(0x10), targetPe);
  results->remoteFetchOr64 = previous64;
  nvshmem_uint64_atomic_or(int64Target, UINT64_C(0x20), targetPe);
  previous64 = nvshmem_uint64_atomic_fetch_xor(int64Target, UINT64_C(0x3), targetPe);
  results->remoteFetchXor64 = previous64;
  nvshmem_uint64_atomic_xor(int64Target, UINT64_C(0x1), targetPe);

  previous32 = nvshmem_uint32_atomic_fetch_add(selfIntTarget, UINT32_C(1), selfPe);
  results->selfFetchAdd = previous32;

  float previousFloat = nvshmemx_float_atomic_fetch_add(floatTarget, 1.5f, targetPe);
  results->floatFetchAdd = previousFloat;
  nvshmemx_float_atomic_add(floatTarget, 0.5f, targetPe);
  previousFloat = nvshmem_float_atomic_swap(floatTarget, 5.0f, targetPe);
  results->floatSwap = previousFloat;
  previousFloat = nvshmem_float_atomic_fetch(floatTarget, targetPe);
  results->floatFetch = previousFloat;
  nvshmem_float_atomic_set(floatTarget, 6.0f, targetPe);

  double previousDouble = nvshmemx_double_atomic_fetch_add(doubleTarget, 1.5, targetPe);
  results->doubleFetchAdd = previousDouble;
  nvshmemx_double_atomic_add(doubleTarget, 0.5, targetPe);
  previousDouble = nvshmem_double_atomic_swap(doubleTarget, 5.0, targetPe);
  results->doubleSwap = previousDouble;
  previousDouble = nvshmem_double_atomic_fetch(doubleTarget, targetPe);
  results->doubleFetch = previousDouble;
  nvshmem_double_atomic_set(doubleTarget, 6.0, targetPe);

  const __half halfOne = __float2half_rn(1.0f);
  __half previousHalf = nvshmemx_half_atomic_fetch_add(halfTarget, halfOne, targetPe);
  results->halfFetchAdd = __half_as_ushort(previousHalf);
  nvshmemx_half_atomic_add(halfTarget, __float2half_rn(0.5f), targetPe);

  results->failures = 0;
}

// The first proxy implementation intentionally has one source slot and one
// GPU lock per destination. Exercise that serialization explicitly: all 32
// callers must complete, while the target observes one linearized sum.
__global__ void issueConcurrentProxyAdds(niinContext* context) {
  if (threadIdx.x == 0) niin_g_ctx = context;
  __syncthreads();
  if (threadIdx.x >= kConcurrentThreads) return;
  const int selfPe = nvshmem_my_pe();
  const int targetPe = (selfPe + 1) % nvshmem_n_pes();
  auto* target = reinterpret_cast<uint32_t*>(static_cast<char*>(niin_heap_base()) + kConcurrentOffset);
  nvshmem_uint32_atomic_add(target, UINT32_C(1), targetPe);
}

__global__ void readTarget(const uint32_t* remoteInt, const uint64_t* remoteInt64,
                           const uint32_t* remoteConcurrent,
                           const float* remoteFloat,
                           const double* remoteDouble, const uint16_t* remoteHalf,
                           uint32_t* intValue, uint64_t* int64Value, uint32_t* concurrentValue,
                           float* floatValue, double* doubleValue, uint16_t* halfValue) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  __threadfence_system();
  *intValue = *remoteInt;
  *int64Value = *remoteInt64;
  *concurrentValue = *remoteConcurrent;
  *floatValue = *remoteFloat;
  *doubleValue = *remoteDouble;
  *halfValue = *remoteHalf;
}

bool equalFloat(float a, float b) { return std::fabs(a - b) < 1.0e-6f; }
bool equalDouble(double a, double b) { return std::fabs(a - b) < 1.0e-12; }

}  // namespace

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rank = -1;
  int nPes = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nPes);
  if (nPes != 2) fail(rank, "startup", "run with exactly two ranks");
  checkCuda(rank, cudaSetDevice(selectCudaDevice(rank)), "cudaSetDevice");

  ncclUniqueId id;
  if (rank == 0) checkNccl(rank, ncclGetUniqueId(&id), "ncclGetUniqueId");
  if (MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD) != MPI_SUCCESS)
    fail(rank, "MPI_Bcast", "unique-id exchange failed");
  ncclComm_t comm = nullptr;
  checkNccl(rank, ncclCommInitRank(&comm, nPes, id, rank), "ncclCommInitRank");

  void* heap = nullptr;
  checkNccl(rank, ncclMemAlloc(&heap, kHeapBytes), "ncclMemAlloc");
  std::array<unsigned char, kHeapBytes> initial = {};
  if (rank == 1) {
    const uint32_t intInitial = 10;
    const uint64_t int64Initial = 10;
    const uint32_t concurrentInitial = 0;
    const float floatInitial = 2.0f;
    const double doubleInitial = 2.0;
    const uint16_t halfInitial = UINT16_C(0x3c00);
    std::memcpy(initial.data() + kIntOffset, &intInitial, sizeof(intInitial));
    std::memcpy(initial.data() + kInt64Offset, &int64Initial, sizeof(int64Initial));
    std::memcpy(initial.data() + kConcurrentOffset, &concurrentInitial, sizeof(concurrentInitial));
    std::memcpy(initial.data() + kFloatOffset, &floatInitial, sizeof(floatInitial));
    std::memcpy(initial.data() + kDoubleOffset, &doubleInitial, sizeof(doubleInitial));
    std::memcpy(initial.data() + kHalfOffset, &halfInitial, sizeof(halfInitial));
  } else {
    const uint32_t selfInitial = 7;
    std::memcpy(initial.data() + kSelfIntOffset, &selfInitial, sizeof(selfInitial));
  }
  checkCuda(rank, cudaMemcpy(heap, initial.data(), initial.size(), cudaMemcpyHostToDevice), "heap initialization");

  // Exercise proxy routing through a normal committed NIIN context. Disable
  // GIN here because the proxy is intentionally independent of it.
  niinContext_host hostContext = {};
  checkNccl(rank, ncclGroupStart(), "ncclGroupStart(niinInit)");
  checkNccl(rank, niinInit(comm, heap, kHeapBytes, &hostContext, false), "niinInit");
  checkNccl(rank, ncclGroupEnd(), "ncclGroupEnd(niinInit)");
  niinContext* deviceContext = nullptr;
  checkCuda(rank, cudaMalloc(&deviceContext, sizeof(*deviceContext)), "cudaMalloc(device context)");
  checkNccl(rank, niinCommit(&hostContext, deviceContext), "niinCommit");

  niinGpunetioProxyAtomicOptions options = NIIN_GPUNETIO_PROXY_ATOMIC_OPTIONS_INITIALIZER;
  niinGpunetioProxyAtomicHostContext* proxy = nullptr;
  checkNccl(rank, niinGpunetioProxyAtomicInit(comm, heap, kHeapBytes, &options, &proxy),
            "niinGpunetioProxyAtomicInit");
  checkNccl(rank, niinGpunetioProxyAtomicBind(proxy, deviceContext), "niinGpunetioProxyAtomicBind");
  if (MPI_Barrier(MPI_COMM_WORLD) != MPI_SUCCESS) fail(rank, "MPI_Barrier", "before AMOs");

  Results* deviceResults = nullptr;
  if (rank == 0) {
    checkCuda(rank, cudaMalloc(&deviceResults, sizeof(*deviceResults)), "cudaMalloc(results)");
    issueProxyAtomics<<<1, 1>>>(deviceContext, deviceResults);
    checkCuda(rank, cudaGetLastError(), "issueProxyAtomics launch");
    checkCuda(rank, cudaDeviceSynchronize(), "issueProxyAtomics completion");
    issueConcurrentProxyAdds<<<1, kConcurrentThreads>>>(deviceContext);
    checkCuda(rank, cudaGetLastError(), "issueConcurrentProxyAdds launch");
    checkCuda(rank, cudaDeviceSynchronize(), "issueConcurrentProxyAdds completion");
  }
  if (MPI_Barrier(MPI_COMM_WORLD) != MPI_SUCCESS) fail(rank, "MPI_Barrier", "after AMOs");

  uint32_t finalInt = 0;
  uint64_t finalInt64 = 0;
  uint32_t finalConcurrent = 0;
  float finalFloat = 0.0f;
  double finalDouble = 0.0;
  uint16_t finalHalf = 0;
  if (rank == 1) {
    uint32_t* dInt = nullptr;
    uint64_t* dInt64 = nullptr;
    uint32_t* dConcurrent = nullptr;
    float* dFloat = nullptr;
    double* dDouble = nullptr;
    uint16_t* dHalf = nullptr;
    checkCuda(rank, cudaMalloc(&dInt, sizeof(*dInt)), "cudaMalloc(final int)");
    checkCuda(rank, cudaMalloc(&dInt64, sizeof(*dInt64)), "cudaMalloc(final int64)");
    checkCuda(rank, cudaMalloc(&dConcurrent, sizeof(*dConcurrent)), "cudaMalloc(final concurrent)");
    checkCuda(rank, cudaMalloc(&dFloat, sizeof(*dFloat)), "cudaMalloc(final float)");
    checkCuda(rank, cudaMalloc(&dDouble, sizeof(*dDouble)), "cudaMalloc(final double)");
    checkCuda(rank, cudaMalloc(&dHalf, sizeof(*dHalf)), "cudaMalloc(final half)");
    readTarget<<<1, 1>>>(static_cast<const uint32_t*>(heap) + kIntOffset / sizeof(uint32_t),
                          reinterpret_cast<const uint64_t*>(static_cast<const char*>(heap) + kInt64Offset),
                          reinterpret_cast<const uint32_t*>(static_cast<const char*>(heap) + kConcurrentOffset),
                          reinterpret_cast<const float*>(static_cast<const char*>(heap) + kFloatOffset),
                          reinterpret_cast<const double*>(static_cast<const char*>(heap) + kDoubleOffset),
                          reinterpret_cast<const uint16_t*>(static_cast<const char*>(heap) + kHalfOffset),
                          dInt, dInt64, dConcurrent, dFloat, dDouble, dHalf);
    checkCuda(rank, cudaGetLastError(), "readTarget launch");
    checkCuda(rank, cudaDeviceSynchronize(), "readTarget completion");
    checkCuda(rank, cudaMemcpy(&finalInt, dInt, sizeof(finalInt), cudaMemcpyDeviceToHost), "copy final int");
    checkCuda(rank, cudaMemcpy(&finalInt64, dInt64, sizeof(finalInt64), cudaMemcpyDeviceToHost),
              "copy final int64");
    checkCuda(rank, cudaMemcpy(&finalConcurrent, dConcurrent, sizeof(finalConcurrent), cudaMemcpyDeviceToHost),
              "copy final concurrent");
    checkCuda(rank, cudaMemcpy(&finalFloat, dFloat, sizeof(finalFloat), cudaMemcpyDeviceToHost), "copy final float");
    checkCuda(rank, cudaMemcpy(&finalDouble, dDouble, sizeof(finalDouble), cudaMemcpyDeviceToHost), "copy final double");
    checkCuda(rank, cudaMemcpy(&finalHalf, dHalf, sizeof(finalHalf), cudaMemcpyDeviceToHost), "copy final half");
    cudaFree(dHalf);
    cudaFree(dDouble);
    cudaFree(dFloat);
    cudaFree(dConcurrent);
    cudaFree(dInt64);
    cudaFree(dInt);
  }

  int localPass = 1;
  if (rank == 0) {
    Results actual = {};
    checkCuda(rank, cudaMemcpy(&actual, deviceResults, sizeof(actual), cudaMemcpyDeviceToHost), "copy results");
    localPass = actual.failures == 0 && actual.remoteFetchAdd == 10 && actual.remoteCompareSwap == 20 &&
                actual.remoteSwap == 30 && actual.remoteFetch == 40 && actual.remoteFetchInc == 51 &&
                actual.remoteFetchAnd == 52 && actual.remoteFetchOr == 4 && actual.remoteFetchXor == 52 &&
                actual.remoteFetchAdd64 == 10 && actual.remoteCompareSwap64 == 20 &&
                actual.remoteSwap64 == 30 && actual.remoteFetch64 == 40 && actual.remoteFetchInc64 == 51 &&
                actual.remoteFetchAnd64 == 52 && actual.remoteFetchOr64 == 4 &&
                actual.remoteFetchXor64 == 52 &&
                actual.selfFetchAdd == 7 && equalFloat(actual.floatFetchAdd, 2.0f) &&
                equalFloat(actual.floatSwap, 4.0f) && equalFloat(actual.floatFetch, 5.0f) &&
                equalDouble(actual.doubleFetchAdd, 2.0) && equalDouble(actual.doubleSwap, 4.0) &&
                equalDouble(actual.doubleFetch, 5.0) && actual.halfFetchAdd == UINT16_C(0x3c00);
    if (!localPass) std::fprintf(stderr, "NIIN proxy atomic smoke: source results mismatch\n");
  }
  if (rank == 1 && (finalInt != 54 || finalInt64 != 54 || finalConcurrent != kConcurrentThreads ||
                    !equalFloat(finalFloat, 6.0f) ||
                    !equalDouble(finalDouble, 6.0) || finalHalf != UINT16_C(0x4100))) {
    std::fprintf(stderr,
                 "NIIN proxy atomic smoke: targets int=%u int64=%llu concurrent=%u float=%f double=%f half=0x%04x\n",
                 finalInt, static_cast<unsigned long long>(finalInt64), finalConcurrent, finalFloat,
                 finalDouble, finalHalf);
    localPass = 0;
  }
  int globalPass = 0;
  if (MPI_Allreduce(&localPass, &globalPass, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD) != MPI_SUCCESS)
    fail(rank, "MPI_Allreduce", "result reduction failed");

  if (deviceResults != nullptr) cudaFree(deviceResults);
  if (MPI_Barrier(MPI_COMM_WORLD) != MPI_SUCCESS) fail(rank, "MPI_Barrier", "before finalization");
  checkNccl(rank, niinGpunetioProxyAtomicFinalize(proxy), "niinGpunetioProxyAtomicFinalize");
  finalizeNiinContext(rank, comm, deviceContext);
  checkNccl(rank, ncclMemFree(heap), "ncclMemFree");
  checkNccl(rank, ncclCommDestroy(comm), "ncclCommDestroy");
  MPI_Finalize();
  if (!globalPass) return 1;
  std::fprintf(stderr, "NIIN proxy atomic smoke: PASS\n");
  return 0;
}
