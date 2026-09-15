/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

// Focused smoke test for NIIN's public host collective APIs. It intentionally
// includes no compat shim, so the extended API declarations must come from the
// NIIN public headers.

#include "nvshmem.h"
#include "nvshmemx.h"

#include <mpi.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <thread>
#include <vector>

// Populate the device sources on the same stream as the host NCCL collectives.
// This catches a collective implementation that fails to honor prior work on
// its caller-provided stream.
__global__ void populate_collective_sources(int32_t* src32, int64_t* src64,
                                            unsigned char* alltoallSrc, int rank,
                                            int nRanks) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;

  constexpr size_t kReduceCount = 7;
  constexpr size_t kAlltoallBytes = 13;
  for (size_t i = 0; i < kReduceCount; ++i) {
    src32[i] = rank + 1 + static_cast<int>(i);
    src64[i] = static_cast<int64_t>(100 * (rank + 1) + static_cast<int>(i));
  }
  for (int peer = 0; peer < nRanks; ++peer) {
    for (size_t byte = 0; byte < kAlltoallBytes; ++byte) {
      alltoallSrc[static_cast<size_t>(peer) * kAlltoallBytes + byte] =
          static_cast<unsigned char>((31 * rank + 7 * peer + byte) % 127);
    }
  }
}

// Every thread of this 3D CTA calls both public block APIs. The remote put
// before barrier verifies the barrier's release/acquire visibility guarantee.
__global__ void exercise_block_collective_apis(int32_t* blockStatus,
                                               int* blockFailure, int rank,
                                               int nRanks) {
  nvshmemx_sync_all_block();

  if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
    nvshmem_int32_p(blockStatus, rank + 1, (rank + 1) % nRanks);
  }

  nvshmemx_barrier_all_block();

  if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
    const int expected = (rank + nRanks - 1) % nRanks + 1;
    if (*blockStatus != expected) atomicExch(blockFailure, 1);
  }
}

namespace {

using StreamCollective = void (*)(cudaStream_t);

bool check_cuda(cudaError_t result, const char* operation, int rank) {
  if (result == cudaSuccess) return true;
  std::fprintf(stderr, "PE %d: %s failed: %s\n", rank, operation,
               cudaGetErrorString(result));
  return false;
}

bool check_mpi(int result, const char* operation, int rank) {
  if (result == MPI_SUCCESS) return true;
  std::fprintf(stderr, "PE %d: %s failed\n", rank, operation);
  return false;
}

bool check_niin(int result, const char* operation, int rank) {
  if (result == 0) return true;
  std::fprintf(stderr, "PE %d: %s failed\n", rank, operation);
  return false;
}

bool should_exercise_block_collectives() {
  const char* setting = std::getenv("NIIN_TEST_BLOCK_COLLECTIVES");
  return setting != nullptr && std::strcmp(setting, "0") != 0;
}

// Best-effort check for an actual collective: an event after a real NCCL
// collective cannot complete before every PE enters it. The worker thread
// avoids a host-side NCCL preconnect stall from preventing its PE from
// releasing PE 0 to enter the matching collective.
bool verify_stream_collective(cudaStream_t stream, StreamCollective collective,
                              const char* name, int rank, int nRanks) {
  constexpr auto kEarlyCompletionWindow = std::chrono::milliseconds(100);
  constexpr auto kRootArrivalDelay = std::chrono::milliseconds(250);
  bool ok = true;
  cudaEvent_t completion = nullptr;

  if (rank == 0) {
    for (int peer = 1; peer < nRanks; ++peer) {
      int released = 0;
      ok &= check_mpi(MPI_Recv(&released, 1, MPI_INT, peer, 0, MPI_COMM_WORLD,
                               MPI_STATUS_IGNORE),
                      "wait for stream collective check", rank);
    }
    std::this_thread::sleep_for(kRootArrivalDelay);
    collective(stream);
  } else {
    ok &= check_cuda(cudaEventCreateWithFlags(&completion, cudaEventDisableTiming),
                     "cudaEventCreateWithFlags", rank);

    std::atomic<bool> workerStarted{false};
    std::atomic<bool> workerReturned{false};
    std::atomic<cudaError_t> eventRecordResult{cudaSuccess};
    std::thread launcher([&] {
      workerStarted.store(true, std::memory_order_release);
      collective(stream);
      if (completion != nullptr) {
        eventRecordResult.store(cudaEventRecord(completion, stream),
                                std::memory_order_release);
      }
      workerReturned.store(true, std::memory_order_release);
    });

    while (!workerStarted.load(std::memory_order_acquire)) {
      std::this_thread::yield();
    }

    int released = 1;
    ok &= check_mpi(MPI_Send(&released, 1, MPI_INT, 0, 0, MPI_COMM_WORLD),
                    "release stream collective", rank);

    if (completion != nullptr) {
      const auto deadline = std::chrono::steady_clock::now() + kEarlyCompletionWindow;
      while (std::chrono::steady_clock::now() < deadline) {
        if (!workerReturned.load(std::memory_order_acquire)) {
          std::this_thread::yield();
          continue;
        }
        const cudaError_t recordResult = eventRecordResult.load(std::memory_order_acquire);
        if (recordResult != cudaSuccess) {
          ok &= check_cuda(recordResult, "cudaEventRecord", rank);
          break;
        }
        const cudaError_t result = cudaEventQuery(completion);
        if (result == cudaErrorNotReady) {
          std::this_thread::yield();
          continue;
        }
        if (result == cudaSuccess) {
          std::fprintf(stderr,
                       "PE %d: %s completed before PE 0 entered the collective\n",
                       rank, name);
        } else {
          ok &= check_cuda(result, "cudaEventQuery", rank);
        }
        ok = false;
        break;
      }
    }
    launcher.join();
  }

  ok &= check_cuda(cudaStreamSynchronize(stream), name, rank);
  if (completion != nullptr) cudaEventDestroy(completion);

  int localFailure = ok ? 0 : 1;
  int anyFailure = 0;
  ok &= check_mpi(MPI_Allreduce(&localFailure, &anyFailure, 1, MPI_INT, MPI_MAX,
                                MPI_COMM_WORLD),
                  "combine stream collective result", rank);
  return ok && anyFailure == 0;
}

}  // namespace

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  int rank = 0;
  int nRanks = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nRanks);
  if (nRanks < 2) {
    if (rank == 0) std::fprintf(stderr, "host_collectives requires at least two PEs\n");
    MPI_Finalize();
    return 1;
  }

  MPI_Comm mpiComm = MPI_COMM_WORLD;
  nvshmemx_init_attr_t attr = NVSHMEMX_INIT_ATTR_INITIALIZER;
  attr.mpi_comm = &mpiComm;
  if (!check_niin(nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr),
                  "nvshmemx_init_attr", rank)) {
    // NCCL communicator initialization is collective. Do not strand a peer
    // that is still inside it if one PE sees a local initialization failure.
    MPI_Abort(MPI_COMM_WORLD, 1);
    return 1;
  }

  bool ok = true;
  cudaStream_t stream = nullptr;
  int* blockFailure = nullptr;
  ok &= check_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
                   "cudaStreamCreateWithFlags", rank);
  ok &= check_cuda(cudaMalloc(&blockFailure, sizeof(*blockFailure)), "cudaMalloc", rank);

  constexpr size_t kReduceCount = 7;
  constexpr size_t kAlltoallBytes = 13;
  int32_t* src32 = static_cast<int32_t*>(nvshmem_malloc(kReduceCount * sizeof(int32_t)));
  int32_t* sum32 = static_cast<int32_t*>(nvshmem_malloc(kReduceCount * sizeof(int32_t)));
  int32_t* min32 = static_cast<int32_t*>(nvshmem_malloc(kReduceCount * sizeof(int32_t)));
  int32_t* max32 = static_cast<int32_t*>(nvshmem_malloc(kReduceCount * sizeof(int32_t)));
  int64_t* src64 = static_cast<int64_t*>(nvshmem_malloc(kReduceCount * sizeof(int64_t)));
  int64_t* sum64 = static_cast<int64_t*>(nvshmem_malloc(kReduceCount * sizeof(int64_t)));
  int64_t* min64 = static_cast<int64_t*>(nvshmem_malloc(kReduceCount * sizeof(int64_t)));
  int64_t* max64 = static_cast<int64_t*>(nvshmem_malloc(kReduceCount * sizeof(int64_t)));
  unsigned char* alltoallSrc = static_cast<unsigned char*>(
      nvshmem_malloc(static_cast<size_t>(nRanks) * kAlltoallBytes));
  unsigned char* alltoallDst = static_cast<unsigned char*>(
      nvshmem_malloc(static_cast<size_t>(nRanks) * kAlltoallBytes));
  int32_t* blockStatus = static_cast<int32_t*>(nvshmem_malloc(sizeof(*blockStatus)));

  if (!blockFailure || !src32 || !sum32 || !min32 || !max32 || !src64 || !sum64 ||
      !min64 || !max64 || !alltoallSrc || !alltoallDst || !blockStatus) {
    std::fprintf(stderr, "PE %d: test allocation failed\n", rank);
    ok = false;
  }

  // A failed setup on one PE must keep every PE out of the NCCL collectives.
  // Otherwise the successful PEs could enter a world collective alone.
  int localSetupFailure = ok ? 0 : 1;
  int anySetupFailure = 0;
  MPI_Allreduce(&localSetupFailure, &anySetupFailure, 1, MPI_INT, MPI_MAX,
                MPI_COMM_WORLD);
  if (anySetupFailure) ok = false;

  if (ok) {
    // Keep sync and barrier in isolated phases: neither can be masked by a
    // later collective, and a no-op has a completion event that fires early.
    ok &= verify_stream_collective(stream, nvshmemx_sync_all_on_stream,
                                   "nvshmemx_sync_all_on_stream", rank, nRanks);
    ok &= verify_stream_collective(stream, nvshmemx_barrier_all_on_stream,
                                   "nvshmemx_barrier_all_on_stream", rank, nRanks);

    if (should_exercise_block_collectives()) {
      // Exercise both public block APIs with every thread of a 3D CTA. The
      // barrier phase also verifies a predecessor PE's remote put is visible.
      ok &= check_cuda(cudaMemsetAsync(blockStatus, 0, sizeof(*blockStatus), stream),
                       "clear block status", rank);
      ok &= check_cuda(cudaMemsetAsync(blockFailure, 0, sizeof(*blockFailure), stream),
                       "clear block failure", rank);
      ok &= check_cuda(cudaStreamSynchronize(stream), "synchronize block setup", rank);
      ok &= check_mpi(MPI_Barrier(MPI_COMM_WORLD), "synchronize block setup PEs", rank);
      exercise_block_collective_apis<<<dim3(1), dim3(4, 4, 2), 0, stream>>>(
          blockStatus, blockFailure, rank, nRanks);
      ok &= check_cuda(cudaGetLastError(), "launch block collectives", rank);
      int hBlockFailure = 0;
      ok &= check_cuda(cudaMemcpyAsync(&hBlockFailure, blockFailure, sizeof(hBlockFailure),
                                       cudaMemcpyDeviceToHost, stream),
                       "copy block collective result", rank);
      ok &= check_cuda(cudaStreamSynchronize(stream), "block collectives", rank);
      if (hBlockFailure != 0) {
        std::fprintf(stderr, "PE %d: block barrier did not make remote put visible\n", rank);
        ok = false;
      }
    } else if (rank == 0) {
      std::fprintf(stderr,
                   "host_collectives: block runtime test skipped; set "
                   "NIIN_TEST_BLOCK_COLLECTIVES=1 on a validated topology\n");
    }

    // The producer kernel, host NCCL collectives, and result copies all share
    // one stream. This verifies direct NCCL calls honor stream ordering.
    populate_collective_sources<<<1, 1, 0, stream>>>(src32, src64, alltoallSrc, rank,
                                                       nRanks);
    ok &= check_cuda(cudaGetLastError(), "launch collective source producer", rank);
    ok &= check_niin(nvshmemx_int32_sum_reduce_on_stream(
                         NVSHMEM_TEAM_WORLD, sum32, src32, kReduceCount, stream),
                     "int32 sum reduce", rank);
    ok &= check_niin(nvshmemx_int32_min_reduce_on_stream(
                         NVSHMEM_TEAM_WORLD, min32, src32, kReduceCount, stream),
                     "int32 min reduce", rank);
    ok &= check_niin(nvshmemx_int32_max_reduce_on_stream(
                         NVSHMEM_TEAM_WORLD, max32, src32, kReduceCount, stream),
                     "int32 max reduce", rank);
    ok &= check_niin(nvshmemx_int64_sum_reduce_on_stream(
                         NVSHMEM_TEAM_WORLD, sum64, src64, kReduceCount, stream),
                     "int64 sum reduce", rank);
    ok &= check_niin(nvshmemx_int64_min_reduce_on_stream(
                         NVSHMEM_TEAM_WORLD, min64, src64, kReduceCount, stream),
                     "int64 min reduce", rank);
    ok &= check_niin(nvshmemx_int64_max_reduce_on_stream(
                         NVSHMEM_TEAM_WORLD, max64, src64, kReduceCount, stream),
                     "int64 max reduce", rank);
    ok &= check_niin(nvshmemx_alltoallmem_on_stream(NVSHMEM_TEAM_WORLD, alltoallDst,
                                                     alltoallSrc, kAlltoallBytes, stream),
                     "alltoallmem", rank);

    std::vector<int32_t> hSum32(kReduceCount);
    std::vector<int32_t> hMin32(kReduceCount);
    std::vector<int32_t> hMax32(kReduceCount);
    std::vector<int64_t> hSum64(kReduceCount);
    std::vector<int64_t> hMin64(kReduceCount);
    std::vector<int64_t> hMax64(kReduceCount);
    std::vector<unsigned char> hAlltoallDst(static_cast<size_t>(nRanks) * kAlltoallBytes);
    ok &= check_cuda(cudaMemcpyAsync(hSum32.data(), sum32, kReduceCount * sizeof(int32_t),
                                     cudaMemcpyDeviceToHost, stream),
                     "copy int32 sum", rank);
    ok &= check_cuda(cudaMemcpyAsync(hMin32.data(), min32, kReduceCount * sizeof(int32_t),
                                     cudaMemcpyDeviceToHost, stream),
                     "copy int32 min", rank);
    ok &= check_cuda(cudaMemcpyAsync(hMax32.data(), max32, kReduceCount * sizeof(int32_t),
                                     cudaMemcpyDeviceToHost, stream),
                     "copy int32 max", rank);
    ok &= check_cuda(cudaMemcpyAsync(hSum64.data(), sum64, kReduceCount * sizeof(int64_t),
                                     cudaMemcpyDeviceToHost, stream),
                     "copy int64 sum", rank);
    ok &= check_cuda(cudaMemcpyAsync(hMin64.data(), min64, kReduceCount * sizeof(int64_t),
                                     cudaMemcpyDeviceToHost, stream),
                     "copy int64 min", rank);
    ok &= check_cuda(cudaMemcpyAsync(hMax64.data(), max64, kReduceCount * sizeof(int64_t),
                                     cudaMemcpyDeviceToHost, stream),
                     "copy int64 max", rank);
    ok &= check_cuda(cudaMemcpyAsync(hAlltoallDst.data(), alltoallDst,
                                     hAlltoallDst.size(), cudaMemcpyDeviceToHost, stream),
                     "copy alltoall result", rank);
    ok &= check_cuda(cudaStreamSynchronize(stream), "collective result copies", rank);

    for (size_t i = 0; i < kReduceCount; ++i) {
      const int32_t expectedSum32 =
          nRanks * (nRanks + 1) / 2 + nRanks * static_cast<int>(i);
      const int32_t expectedMin32 = 1 + static_cast<int>(i);
      const int32_t expectedMax32 = nRanks + static_cast<int>(i);
      const int64_t expectedSum64 = static_cast<int64_t>(100) * nRanks * (nRanks + 1) / 2 +
                                    nRanks * static_cast<int64_t>(i);
      const int64_t expectedMin64 = 100 + static_cast<int64_t>(i);
      const int64_t expectedMax64 = static_cast<int64_t>(100 * nRanks) + i;
      if (hSum32[i] != expectedSum32 || hMin32[i] != expectedMin32 ||
          hMax32[i] != expectedMax32 || hSum64[i] != expectedSum64 ||
          hMin64[i] != expectedMin64 || hMax64[i] != expectedMax64) {
        std::fprintf(stderr, "PE %d: reduction result mismatch at element %zu\n", rank, i);
        ok = false;
      }
    }
    for (int source = 0; source < nRanks; ++source) {
      for (size_t byte = 0; byte < kAlltoallBytes; ++byte) {
        const unsigned char expected =
            static_cast<unsigned char>((31 * source + 7 * rank + byte) % 127);
        if (hAlltoallDst[static_cast<size_t>(source) * kAlltoallBytes + byte] != expected) {
          std::fprintf(stderr, "PE %d: alltoall result mismatch from PE %d at byte %zu\n",
                       rank, source, byte);
          ok = false;
        }
      }
    }
  }

  nvshmem_free(blockStatus);
  nvshmem_free(alltoallDst);
  nvshmem_free(alltoallSrc);
  nvshmem_free(max64);
  nvshmem_free(min64);
  nvshmem_free(sum64);
  nvshmem_free(src64);
  nvshmem_free(max32);
  nvshmem_free(min32);
  nvshmem_free(sum32);
  nvshmem_free(src32);
  if (blockFailure != nullptr) cudaFree(blockFailure);
  if (stream != nullptr) cudaStreamDestroy(stream);
  nvshmem_finalize();

  int localFailure = ok ? 0 : 1;
  int globalFailure = 0;
  MPI_Allreduce(&localFailure, &globalFailure, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  MPI_Finalize();
  return globalFailure;
}
