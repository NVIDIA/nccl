/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

// Two-rank, one-GPU-per-rank smoke test for NIIN's atomic-only GPUNetIO
// provider. It calls the direct adapter rather than the normal NIIN routing,
// so it exercises the private atomic QPs even when both ranks are local. A
// two-node run is still preferred when validating a routed fabric path.

#include <mpi.h>

#include <cuda_runtime.h>
#include <cuda/atomic>
#include <nccl.h>

#include "niin/context_abi.h"
#include "niin/gpunetio/atomics.h"
#include "niin/gpunetio/host.h"
#include "niin/host.h"

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {

constexpr int kConcurrentThreads = 32;

[[noreturn]] void fail(int rank, const char* where, const char* detail) {
  std::fprintf(stderr, "NIIN GPUNetIO atomic smoke [rank %d]: %s: %s\n", rank, where, detail);
  MPI_Abort(MPI_COMM_WORLD, 1);
  std::abort();
}

void checkCuda(int rank, cudaError_t status, const char* where) {
  if (status != cudaSuccess) fail(rank, where, cudaGetErrorString(status));
}

void checkNccl(int rank, ncclResult_t status, const char* where) {
  if (status != ncclSuccess) fail(rank, where, ncclGetErrorString(status));
}

void phase(int rank, const char* name) {
  std::fprintf(stderr, "NIIN GPUNetIO atomic smoke [rank %d]: %s\n", rank, name);
  std::fflush(stderr);
}

// The atomic provider bootstraps its own QPs with public NCCL all-gather. Do
// a same-shaped byte all-gather first so a launcher/NCCL failure is reported
// separately from GPUNetIO setup.
void verifyMetadataCollective(int rank, int nPes, ncclComm_t comm) {
  constexpr size_t kBytes = 64;
  std::array<uint8_t, kBytes> sendHost = {};
  std::array<uint8_t, kBytes * 2> recvHost = {};
  std::memset(sendHost.data(), rank + 1, sendHost.size());

  void* sendDevice = nullptr;
  void* recvDevice = nullptr;
  cudaStream_t stream = nullptr;
  checkCuda(rank, cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "metadata stream");
  checkCuda(rank, cudaMalloc(&sendDevice, kBytes), "metadata send allocation");
  checkCuda(rank, cudaMalloc(&recvDevice, kBytes * static_cast<size_t>(nPes)), "metadata receive allocation");
  checkCuda(rank, cudaMemcpyAsync(sendDevice, sendHost.data(), kBytes, cudaMemcpyHostToDevice, stream),
            "metadata send copy");
  checkNccl(rank, ncclAllGather(sendDevice, recvDevice, kBytes, ncclUint8, comm, stream),
            "NCCL metadata all-gather");
  checkCuda(rank, cudaMemcpyAsync(recvHost.data(), recvDevice, kBytes * static_cast<size_t>(nPes),
                                  cudaMemcpyDeviceToHost, stream),
            "metadata receive copy");
  checkCuda(rank, cudaStreamSynchronize(stream), "metadata all-gather completion");
  for (int pe = 0; pe < nPes; ++pe) {
    for (size_t byte = 0; byte < kBytes; ++byte) {
      if (recvHost[static_cast<size_t>(pe) * kBytes + byte] != static_cast<uint8_t>(pe + 1))
        fail(rank, "NCCL metadata all-gather", "incorrect gathered rank marker");
    }
  }
  cudaFree(recvDevice);
  cudaFree(sendDevice);
  cudaStreamDestroy(stream);
}

struct AtomicResults {
  uint32_t failures;
  uint32_t fetchAdd32;
  uint32_t compareSwap32;
  uint32_t fetchAfterCompareSwap32;
  uint32_t swap32;
  uint32_t fetch32;
  uint32_t fetchInc32;
  uint32_t fetchAnd32;
  uint32_t fetchOr32;
  uint32_t fetchXor32;
  uint64_t fetchAdd64;
  uint64_t compareSwap64;
  uint64_t fetchAfterCompareSwap64;
  uint64_t swap64;
  uint64_t fetch64;
  uint64_t fetchInc64;
  uint64_t fetchAnd64;
  uint64_t fetchOr64;
  uint64_t fetchXor64;
};

struct ConcurrentResults {
  uint32_t failures;
  uint32_t previous[kConcurrentThreads];
};

__global__ void issueAllAtomics(niinContext* context, int targetPe, AtomicResults* results) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  const niinGpunetioAtomicContext* atomics = context->gpunetioAtomicContext;
  uint32_t failures = 0;

#define NIIN_SMOKE_TRY(expression) \
  do {                            \
    if (!(expression)) ++failures; \
  } while (0)

  // A bad symmetric-pointer cast must be rejected before it reserves a WQE;
  // otherwise an invalid hardware AMO could poison this PE's dedicated QP.
  NIIN_SMOKE_TRY(!niin_gpunetio_atomic_fetch_add_try(atomics, size_t{1}, UINT32_C(1), targetPe,
                                                      &results->fetchAdd32));

  NIIN_SMOKE_TRY(niin_gpunetio_atomic_fetch_add_try(atomics, size_t{0}, UINT32_C(5), targetPe,
                                                     &results->fetchAdd32));
  NIIN_SMOKE_TRY(niin_gpunetio_atomic_add_try(atomics, size_t{0}, UINT32_C(5), targetPe));
  NIIN_SMOKE_TRY(niin_gpunetio_atomic_compare_swap_try(atomics, size_t{0}, UINT32_C(20), UINT32_C(30),
                                                        targetPe, &results->compareSwap32));
  NIIN_SMOKE_TRY(
      niin_gpunetio_atomic_fetch_try(atomics, size_t{0}, targetPe, &results->fetchAfterCompareSwap32));
  NIIN_SMOKE_TRY(niin_gpunetio_atomic_swap_try(atomics, size_t{0}, UINT32_C(40), targetPe,
                                                &results->swap32));
  NIIN_SMOKE_TRY(niin_gpunetio_atomic_fetch_try(atomics, size_t{0}, targetPe, &results->fetch32));
  NIIN_SMOKE_TRY(niin_gpunetio_atomic_set_try(atomics, size_t{0}, UINT32_C(50), targetPe));
  NIIN_SMOKE_TRY(niin_gpunetio_atomic_inc_try<uint32_t>(atomics, size_t{0}, targetPe));
  NIIN_SMOKE_TRY(
      niin_gpunetio_atomic_fetch_inc_try(atomics, size_t{0}, targetPe, &results->fetchInc32));
  NIIN_SMOKE_TRY(niin_gpunetio_atomic_fetch_and_try(atomics, size_t{0}, UINT32_C(0x3f), targetPe,
                                                     &results->fetchAnd32));
  NIIN_SMOKE_TRY(niin_gpunetio_atomic_and_try(atomics, size_t{0}, UINT32_C(0x0f), targetPe));
  NIIN_SMOKE_TRY(niin_gpunetio_atomic_fetch_or_try(atomics, size_t{0}, UINT32_C(0x10), targetPe,
                                                    &results->fetchOr32));
  NIIN_SMOKE_TRY(niin_gpunetio_atomic_or_try(atomics, size_t{0}, UINT32_C(0x20), targetPe));
  NIIN_SMOKE_TRY(niin_gpunetio_atomic_fetch_xor_try(atomics, size_t{0}, UINT32_C(0x3), targetPe,
                                                     &results->fetchXor32));
  NIIN_SMOKE_TRY(niin_gpunetio_atomic_xor_try(atomics, size_t{0}, UINT32_C(0x1), targetPe));

  NIIN_SMOKE_TRY(niin_gpunetio_atomic_fetch_add_try(atomics, size_t{8}, UINT64_C(5), targetPe,
                                                     &results->fetchAdd64));
  NIIN_SMOKE_TRY(niin_gpunetio_atomic_add_try(atomics, size_t{8}, UINT64_C(5), targetPe));
  NIIN_SMOKE_TRY(niin_gpunetio_atomic_compare_swap_try(atomics, size_t{8}, UINT64_C(20), UINT64_C(30),
                                                        targetPe, &results->compareSwap64));
  NIIN_SMOKE_TRY(
      niin_gpunetio_atomic_fetch_try(atomics, size_t{8}, targetPe, &results->fetchAfterCompareSwap64));
  NIIN_SMOKE_TRY(niin_gpunetio_atomic_swap_try(atomics, size_t{8}, UINT64_C(40), targetPe,
                                                &results->swap64));
  NIIN_SMOKE_TRY(niin_gpunetio_atomic_fetch_try(atomics, size_t{8}, targetPe, &results->fetch64));
  NIIN_SMOKE_TRY(niin_gpunetio_atomic_set_try(atomics, size_t{8}, UINT64_C(50), targetPe));
  NIIN_SMOKE_TRY(niin_gpunetio_atomic_inc_try<uint64_t>(atomics, size_t{8}, targetPe));
  NIIN_SMOKE_TRY(
      niin_gpunetio_atomic_fetch_inc_try(atomics, size_t{8}, targetPe, &results->fetchInc64));
  NIIN_SMOKE_TRY(niin_gpunetio_atomic_fetch_and_try(atomics, size_t{8}, UINT64_C(0x3f), targetPe,
                                                     &results->fetchAnd64));
  NIIN_SMOKE_TRY(niin_gpunetio_atomic_and_try(atomics, size_t{8}, UINT64_C(0x0f), targetPe));
  NIIN_SMOKE_TRY(niin_gpunetio_atomic_fetch_or_try(atomics, size_t{8}, UINT64_C(0x10), targetPe,
                                                    &results->fetchOr64));
  NIIN_SMOKE_TRY(niin_gpunetio_atomic_or_try(atomics, size_t{8}, UINT64_C(0x20), targetPe));
  NIIN_SMOKE_TRY(niin_gpunetio_atomic_fetch_xor_try(atomics, size_t{8}, UINT64_C(0x3), targetPe,
                                                     &results->fetchXor64));
  NIIN_SMOKE_TRY(niin_gpunetio_atomic_xor_try(atomics, size_t{8}, UINT64_C(0x1), targetPe));

#undef NIIN_SMOKE_TRY
  results->failures = failures;
}

// Every thread targets the same remote word. The provider has exactly one
// response slot and QP lock per destination, so this checks that contention
// serializes callers without reusing an in-flight result buffer.
__global__ void issueConcurrentFetchAdds(niinContext* context, int targetPe, ConcurrentResults* results) {
  const int thread = static_cast<int>(threadIdx.x);
  if (blockIdx.x != 0 || thread >= kConcurrentThreads) return;

  uint32_t previous = 0;
  const bool succeeded = niin_gpunetio_atomic_fetch_add_try(
      context->gpunetioAtomicContext, size_t{16}, UINT32_C(1), targetPe, &previous);
  results->previous[thread] = previous;
  if (!succeeded) atomicAdd(&results->failures, UINT32_C(1));
}

__global__ void readTargetValues(const uint32_t* word32, const uint64_t* word64, uint32_t* result32,
                                 uint64_t* result64) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  cuda::atomic_thread_fence(cuda::memory_order_acquire, cuda::thread_scope_system);
  *result32 = *word32;
  *result64 = *word64;
}

bool validateResults(const AtomicResults& actual) {
  return actual.failures == 0 && actual.fetchAdd32 == UINT32_C(10) && actual.compareSwap32 == UINT32_C(20) &&
         actual.fetchAfterCompareSwap32 == UINT32_C(30) && actual.swap32 == UINT32_C(30) &&
         actual.fetch32 == UINT32_C(40) &&
         actual.fetchInc32 == UINT32_C(51) && actual.fetchAnd32 == UINT32_C(52) &&
         actual.fetchOr32 == UINT32_C(4) && actual.fetchXor32 == UINT32_C(52) &&
         actual.fetchAdd64 == UINT64_C(10) && actual.compareSwap64 == UINT64_C(20) &&
         actual.fetchAfterCompareSwap64 == UINT64_C(30) && actual.swap64 == UINT64_C(30) &&
         actual.fetch64 == UINT64_C(40) &&
         actual.fetchInc64 == UINT64_C(51) && actual.fetchAnd64 == UINT64_C(52) &&
         actual.fetchOr64 == UINT64_C(4) && actual.fetchXor64 == UINT64_C(52);
}

bool validateConcurrentResults(const ConcurrentResults& actual) {
  if (actual.failures != 0) return false;
  std::array<bool, kConcurrentThreads> seen = {};
  for (int thread = 0; thread < kConcurrentThreads; ++thread) {
    const uint32_t previous = actual.previous[thread];
    if (previous >= static_cast<uint32_t>(kConcurrentThreads) || seen[previous]) return false;
    seen[previous] = true;
  }
  return true;
}

void printResults(const AtomicResults& actual) {
  std::fprintf(stderr,
               "NIIN GPUNetIO atomic smoke: failures=%u; 32b [fa=%u cs=%u fcs=%u sw=%u f=%u fi=%u "
               "and=%u or=%u xor=%u]; 64b [fa=%llu cs=%llu fcs=%llu sw=%llu f=%llu fi=%llu and=%llu or=%llu xor=%llu]\n",
               actual.failures, actual.fetchAdd32, actual.compareSwap32, actual.fetchAfterCompareSwap32,
               actual.swap32, actual.fetch32,
               actual.fetchInc32, actual.fetchAnd32, actual.fetchOr32, actual.fetchXor32,
               static_cast<unsigned long long>(actual.fetchAdd64),
               static_cast<unsigned long long>(actual.compareSwap64),
               static_cast<unsigned long long>(actual.fetchAfterCompareSwap64),
               static_cast<unsigned long long>(actual.swap64), static_cast<unsigned long long>(actual.fetch64),
               static_cast<unsigned long long>(actual.fetchInc64),
               static_cast<unsigned long long>(actual.fetchAnd64), static_cast<unsigned long long>(actual.fetchOr64),
               static_cast<unsigned long long>(actual.fetchXor64));
}

int selectCudaDevice(int rank) {
  int deviceCount = 0;
  checkCuda(rank, cudaGetDeviceCount(&deviceCount), "cudaGetDeviceCount");
  if (deviceCount <= 0) fail(rank, "cudaGetDeviceCount", "no visible CUDA device");

  MPI_Comm localComm = MPI_COMM_NULL;
  if (MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &localComm) != MPI_SUCCESS)
    fail(rank, "MPI_Comm_split_type", "could not discover the node-local rank");
  int localRank = 0;
  if (MPI_Comm_rank(localComm, &localRank) != MPI_SUCCESS) fail(rank, "MPI_Comm_rank", "could not read local rank");
  MPI_Comm_free(&localComm);

  // Launchers that set CUDA_VISIBLE_DEVICES to one GPU per process report one
  // visible device, while unrestricted launchers expose every local GPU.
  return deviceCount == 1 ? 0 : localRank % deviceCount;
}

void finalizeNiinContext(int rank, ncclComm_t comm, niinContext* deviceContext) {
  checkNccl(rank, ncclGroupStart(), "ncclGroupStart(niinFinalize)");
  checkNccl(rank, niinFinalize(comm, deviceContext), "niinFinalize");
  checkNccl(rank, ncclGroupEnd(), "ncclGroupEnd(niinFinalize)");
  checkCuda(rank, cudaFree(deviceContext), "cudaFree(device context)");
}

}  // namespace

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rank = -1;
  int nPes = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nPes);
  if (nPes != 2) fail(rank, "startup", "run with exactly two MPI ranks");

  checkCuda(rank, cudaSetDevice(selectCudaDevice(rank)), "cudaSetDevice");

  ncclUniqueId id;
  if (rank == 0) checkNccl(rank, ncclGetUniqueId(&id), "ncclGetUniqueId");
  if (MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD) != MPI_SUCCESS)
    fail(rank, "MPI_Bcast", "NCCL unique-id exchange failed");

  ncclComm_t comm = nullptr;
  checkNccl(rank, ncclCommInitRank(&comm, nPes, id, rank), "ncclCommInitRank");
  phase(rank, "NCCL communicator initialized");
  verifyMetadataCollective(rank, nPes, comm);
  phase(rank, "NCCL metadata all-gather completed");

  constexpr size_t kHeapBytes = 4096;
  void* heap = nullptr;
  checkNccl(rank, ncclMemAlloc(&heap, kHeapBytes), "ncclMemAlloc");
  checkCuda(rank, cudaMemset(heap, 0, kHeapBytes), "cudaMemset(heap)");

  auto* word32 = static_cast<uint32_t*>(heap);
  auto* word64 = reinterpret_cast<uint64_t*>(static_cast<char*>(heap) + 8);
  const uint32_t initial32 = UINT32_C(10);
  const uint64_t initial64 = UINT64_C(10);
  checkCuda(rank, cudaMemcpy(word32, &initial32, sizeof(initial32), cudaMemcpyHostToDevice), "init word32");
  checkCuda(rank, cudaMemcpy(word64, &initial64, sizeof(initial64), cudaMemcpyHostToDevice), "init word64");

  niinContext_host hostContext = {};
  checkNccl(rank, ncclGroupStart(), "ncclGroupStart(niinInit)");
  checkNccl(rank, niinInit(comm, heap, kHeapBytes, &hostContext), "niinInit");
  checkNccl(rank, ncclGroupEnd(), "ncclGroupEnd(niinInit)");
  niinContext* deviceContext = nullptr;
  checkCuda(rank, cudaMalloc(&deviceContext, sizeof(*deviceContext)), "cudaMalloc(device context)");
  checkNccl(rank, niinCommit(&hostContext, deviceContext), "niinCommit");
  phase(rank, "heap and device context initialized");

  niinGpunetioAtomicOptions options = NIIN_GPUNETIO_ATOMIC_OPTIONS_INITIALIZER;
  // AUTO is the normal coverage. `gpu` requires the validated GPU-SM
  // doorbell path. `cpu` is retained to verify that setup fails collectively
  // instead of allowing an unsupported CPU doorbell configuration to hang a
  // device AMO.
  const char* handler = std::getenv("NIIN_GPUNETIO_SMOKE_NIC_HANDLER");
  if (handler != nullptr && std::strcmp(handler, "gpu") == 0) {
    options.nicHandler = NIIN_GPUNETIO_ATOMIC_NIC_HANDLER_GPU_SM_DB;
  } else if (handler != nullptr && std::strcmp(handler, "cpu") == 0) {
    options.nicHandler = NIIN_GPUNETIO_ATOMIC_NIC_HANDLER_CPU_PROXY;
  } else if (handler != nullptr && std::strcmp(handler, "auto") != 0) {
    fail(rank, "NIIN_GPUNETIO_SMOKE_NIC_HANDLER", "expected auto, gpu, or cpu");
  }
  niinGpunetioAtomicHostContext* atomics = nullptr;
  phase(rank, "starting direct-provider bootstrap");
  const ncclResult_t atomicInit = niinGpunetioAtomicInit(comm, heap, kHeapBytes, &options, &atomics);
  if (handler != nullptr && std::strcmp(handler, "cpu") == 0) {
    const int localExpectedRejection =
        atomicInit == ncclInvalidUsage && atomics == nullptr ? 1 : 0;
    int allExpectedRejection = 0;
    if (MPI_Allreduce(&localExpectedRejection, &allExpectedRejection, 1, MPI_INT, MPI_MIN,
                      MPI_COMM_WORLD) != MPI_SUCCESS)
      fail(rank, "MPI_Allreduce", "CPU-doorbell rejection reduction failed");
    finalizeNiinContext(rank, comm, deviceContext);
    checkNccl(rank, ncclMemFree(heap), "ncclMemFree");
    checkNccl(rank, ncclCommDestroy(comm), "ncclCommDestroy");
    if (rank == 0)
      std::printf("NIIN GPUNetIO atomic smoke: %s (CPU doorbell rejection)\n",
                  allExpectedRejection ? "PASS" : "FAIL");
    MPI_Finalize();
    return allExpectedRejection ? 0 : 1;
  }
  checkNccl(rank, atomicInit, "niinGpunetioAtomicInit");
  phase(rank, "direct-provider bootstrap completed");
  checkNccl(rank, niinGpunetioAtomicBind(atomics, deviceContext), "niinGpunetioAtomicBind");
  phase(rank, "direct-provider bound");

  if (MPI_Barrier(MPI_COMM_WORLD) != MPI_SUCCESS) fail(rank, "MPI_Barrier", "before atomics");

  AtomicResults* deviceResults = nullptr;
  ConcurrentResults* deviceConcurrentResults = nullptr;
  if (rank == 0) {
    checkCuda(rank, cudaMalloc(&deviceResults, sizeof(*deviceResults)), "cudaMalloc(results)");
    checkCuda(rank, cudaMemset(deviceResults, 0, sizeof(*deviceResults)), "cudaMemset(results)");
    phase(rank, "issuing direct atomics");
    issueAllAtomics<<<1, 1>>>(deviceContext, 1, deviceResults);
    checkCuda(rank, cudaGetLastError(), "issueAllAtomics launch");
    checkCuda(rank, cudaDeviceSynchronize(), "issueAllAtomics completion");
    checkCuda(rank, cudaMalloc(&deviceConcurrentResults, sizeof(*deviceConcurrentResults)),
              "cudaMalloc(concurrent results)");
    checkCuda(rank, cudaMemset(deviceConcurrentResults, 0, sizeof(*deviceConcurrentResults)),
              "cudaMemset(concurrent results)");
    issueConcurrentFetchAdds<<<1, kConcurrentThreads>>>(deviceContext, 1, deviceConcurrentResults);
    checkCuda(rank, cudaGetLastError(), "issueConcurrentFetchAdds launch");
    checkCuda(rank, cudaDeviceSynchronize(), "issueConcurrentFetchAdds completion");
    phase(rank, "direct atomics completed");
  }

  if (MPI_Barrier(MPI_COMM_WORLD) != MPI_SUCCESS) fail(rank, "MPI_Barrier", "after atomics");

  uint32_t final32 = 0;
  uint32_t finalConcurrent32 = 0;
  uint64_t final64 = 0;
  uint32_t* deviceFinal32 = nullptr;
  uint64_t* deviceFinal64 = nullptr;
  if (rank == 1) {
    checkCuda(rank, cudaMalloc(&deviceFinal32, sizeof(*deviceFinal32)), "cudaMalloc(final32)");
    checkCuda(rank, cudaMalloc(&deviceFinal64, sizeof(*deviceFinal64)), "cudaMalloc(final64)");
    readTargetValues<<<1, 1>>>(word32, word64, deviceFinal32, deviceFinal64);
    checkCuda(rank, cudaGetLastError(), "readTargetValues launch");
    checkCuda(rank, cudaDeviceSynchronize(), "readTargetValues completion");
    checkCuda(rank, cudaMemcpy(&final32, deviceFinal32, sizeof(final32), cudaMemcpyDeviceToHost), "copy final32");
    checkCuda(rank, cudaMemcpy(&finalConcurrent32, static_cast<char*>(heap) + 16, sizeof(finalConcurrent32),
                               cudaMemcpyDeviceToHost),
              "copy final concurrent32");
    checkCuda(rank, cudaMemcpy(&final64, deviceFinal64, sizeof(final64), cudaMemcpyDeviceToHost), "copy final64");
  }

  int localPass = 1;
  if (rank == 0) {
    AtomicResults results = {};
    ConcurrentResults concurrentResults = {};
    checkCuda(rank, cudaMemcpy(&results, deviceResults, sizeof(results), cudaMemcpyDeviceToHost), "copy results");
    checkCuda(rank, cudaMemcpy(&concurrentResults, deviceConcurrentResults, sizeof(concurrentResults),
                               cudaMemcpyDeviceToHost),
              "copy concurrent results");
    localPass = validateResults(results) && validateConcurrentResults(concurrentResults) ? 1 : 0;
    if (!localPass) {
      std::fprintf(stderr, "NIIN GPUNetIO atomic smoke: fetch-result or contention mismatch\n");
      printResults(results);
    }
  }
  if (rank == 1 && (final32 != UINT32_C(54) || final64 != UINT64_C(54) ||
                    finalConcurrent32 != static_cast<uint32_t>(kConcurrentThreads))) {
    std::fprintf(stderr,
                 "NIIN GPUNetIO atomic smoke: target values were %u, %llu, and %u; expected 54, 54, and %d\n",
                 final32, static_cast<unsigned long long>(final64), finalConcurrent32, kConcurrentThreads);
    localPass = 0;
  }

  int globalPass = 0;
  if (MPI_Allreduce(&localPass, &globalPass, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD) != MPI_SUCCESS)
    fail(rank, "MPI_Allreduce", "result reduction failed");

  if (deviceResults != nullptr) cudaFree(deviceResults);
  if (deviceConcurrentResults != nullptr) cudaFree(deviceConcurrentResults);
  if (deviceFinal32 != nullptr) cudaFree(deviceFinal32);
  if (deviceFinal64 != nullptr) cudaFree(deviceFinal64);
  cudaDeviceSynchronize();
  checkNccl(rank, niinGpunetioAtomicFinalize(atomics), "niinGpunetioAtomicFinalize");
  finalizeNiinContext(rank, comm, deviceContext);
  checkNccl(rank, ncclMemFree(heap), "ncclMemFree");
  checkNccl(rank, ncclCommDestroy(comm), "ncclCommDestroy");

  if (rank == 0) std::printf("NIIN GPUNetIO atomic smoke: %s\n", globalPass ? "PASS" : "FAIL");
  MPI_Finalize();
  return globalPass ? 0 : 1;
}
