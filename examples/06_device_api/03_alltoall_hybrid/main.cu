/*************************************************************************
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "cuda_runtime.h"
#include "nccl.h"
#include "nccl_device.h"
#include "utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <unistd.h>

/*
 * NCCL Device API Hybrid AlltoAll Example
 *
 * This example demonstrates NCCL's hybrid communication approach that combines
 * GPU-Initiated Networking (GIN) for remote peers with Load Store Access (LSA)
 * for local peers, optimizing AlltoAll collective operations.
 *
 * Learning Objectives:
 * - Understand hybrid communication optimization
 * - Learn when to use GIN vs LSA for different peer types
 * - Practice combining network and memory-based communication
 * - See performance optimization through intelligent peer selection
 *
 * Key Hybrid Concepts:
 * - **LSA (Load Store Access)**: Direct memory access for local peers
 * - **GIN (GPU-Initiated Networking)**: Network communication for remote peers
 * - **Peer classification**: Distinguishing between local and remote peers
 * - **Hybrid synchronization**: Combining LSA and GIN completion mechanisms
 * - **Performance optimization**: Using the fastest method for each peer type
 *
 * When to Use Hybrid:
 * - Multi-node environments with both local and remote peers
 * - Performance-critical applications requiring optimal communication
 * - Mixed communication patterns (intra-node + inter-node)
 * - Production workloads where efficiency matters
 *
 * Performance Benefits:
 * - LSA provides low-latency local communication
 * - GIN handles remote communication efficiently
 * - Reduced network traffic for local operations
 * - Optimal bandwidth utilization across communication types
 */

// Device API kernel launch configuration
// CTA count must match railGinBarrierCount for proper barrier synchronization
#define NCCL_DEVICE_CTA_COUNT 16
#define NCCL_DEVICE_THREADS_PER_CTA 512

// ==========================================================================
// Device Kernel Implementation
// ==========================================================================

// Hybrid AlltoAll kernel - optimizes by using LSA for local peers, GIN for remote
// This kernel demonstrates performance optimization using both communication methods
template <typename T>
__global__ void HybridAlltoAllKernel(ncclWindow_t sendwin, size_t sendoffset,
                                      ncclWindow_t recvwin, size_t recvoffset,
                                      size_t count, int root, struct ncclDevComm devComm) {
  int ginContext = 0;
  unsigned int signalIndex = 0;
  ncclGin gin { devComm, ginContext };
  uint64_t signalValue = gin.readSignal(signalIndex);

  // GIN barriers for cross-node synchronization
  ncclBarrierSession<ncclCoopCta> bar { ncclCoopCta(), ncclTeamTagWorld(), gin, blockIdx.x };
  bar.sync(ncclCoopCta(), cuda::memory_order_relaxed, ncclGinFenceLevel::Relaxed);

  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int nthreads = blockDim.x * gridDim.x;

  ncclTeam world = ncclTeamWorld(devComm);
  ncclTeam lsa = ncclTeamLsa(devComm);
  const int startLsa = world.rank - lsa.rank;
  const int lsaSize = lsa.nRanks;

  // Handle remote peers (i.e., non-LSA) using GIN for network communication
  const size_t size = count * sizeof(T);
  for (int r = tid; r < startLsa; r += nthreads) {
    gin.put(world, r,
        recvwin, recvoffset + world.rank * size,
        sendwin, sendoffset + r * size,
        size, ncclGin_SignalInc{signalIndex});
  }
  for (int r = startLsa + lsaSize + tid; r < world.nRanks; r += nthreads) {
    gin.put(world, r,
        recvwin, recvoffset + world.rank * size,
        sendwin, sendoffset + r * size,
        size, ncclGin_SignalInc{signalIndex});
  }

  // Handle local peers with LSA (Load Store Access) for optimal performance
  T* sendLocal = (T*)ncclGetLocalPointer(sendwin, sendoffset);
  for (size_t offset = tid; offset < count; offset += nthreads) {
    for (int lp = 0; lp < lsa.nRanks; lp++) {
      int wr = startLsa + lp;
      T* recvPtr = (T*)ncclGetLsaPointer(recvwin, recvoffset, lp);
      recvPtr[world.rank * count + offset] = sendLocal[wr * count + offset];
    }
  }

  // Wait for remote GIN operations to complete
  int numRemotePeers = world.nRanks - lsa.nRanks;
  gin.waitSignal(ncclCoopCta(), signalIndex, signalValue + numRemotePeers);
  gin.flush(ncclCoopCta());

  // Final synchronization barrier
  bar.sync(ncclCoopCta(), cuda::memory_order_release, ncclGinFenceLevel::Relaxed);
}

 // ==========================================================================
 // Host-Side Setup and Device API Initialization
 // ==========================================================================

void* hybridAlltoAll(int my_rank, int total_ranks, int local_device, int devices_per_rank) {
  ncclComm_t comm;
  ncclUniqueId nccl_unique_id;

  if (my_rank == 0) {
    printf("Starting Hybrid AlltoAll initialization\n");
  }

  // Standard NCCL communicator initialization
  if (my_rank == 0) {
    NCCLCHECK(ncclGetUniqueId(&nccl_unique_id));
  }

  // Distribute unique ID
  util_broadcast(0, my_rank, &nccl_unique_id);

  // Set device context for this rank
  CUDACHECK(cudaSetDevice(local_device));
  printf("  Rank %d using GPU device %d\n", my_rank, local_device);

  // ==========================================================================
  // STEP 2: Initialize NCCL Communicator and Allocate Memory
  // ==========================================================================

  // Initialize NCCL communicator
  NCCLCHECK(ncclCommInitRank(&comm, total_ranks, nccl_unique_id, my_rank));
  printf("  Rank %d initialized NCCL communicator for %d total ranks\n", my_rank, total_ranks);

  // Allocate memory for AlltoAll operation
  size_t count = 1024; // Elements per rank
  size_t total_elements = count * total_ranks;
  size_t size_bytes = total_elements * sizeof(float);

  float *h_sendbuff = (float*)malloc(size_bytes);
  float *h_recvbuff = (float*)malloc(size_bytes);
  void* d_sendbuff;
  void* d_recvbuff;
  ncclWindow_t send_win;
  ncclWindow_t recv_win;

  // Device API requires symmetric memory allocation
  NCCLCHECK(ncclMemAlloc(&d_sendbuff, size_bytes));
  NCCLCHECK(ncclMemAlloc(&d_recvbuff, size_bytes));

  // ==========================================================================
  // STEP 3: Register Memory Windows for Device-Side Access
  // ==========================================================================

  // Register symmetric windows for both LSA and GIN access
  NCCLCHECK(ncclCommWindowRegister(comm, d_sendbuff, size_bytes, &send_win, NCCL_WIN_COLL_SYMMETRIC));
  NCCLCHECK(ncclCommWindowRegister(comm, d_recvbuff, size_bytes, &recv_win, NCCL_WIN_COLL_SYMMETRIC));

  // Initialize data: each rank sends unique values to each destination
  for (size_t i = 0; i < total_elements; i++) {
    int dest_rank = i / count;
    int element_idx = i % count;
    h_sendbuff[i] = (float)(my_rank * 1000 + dest_rank * 100 + element_idx);
  }
  CUDACHECK(cudaMemcpy(d_sendbuff, h_sendbuff, size_bytes, cudaMemcpyHostToDevice));
  printf("  Rank %d initialized send data\n", my_rank);

  // ==========================================================================
  // STEP 4: Create Device Communicator with Hybrid Support
  // ==========================================================================

  // Create stream for kernel execution
  cudaStream_t stream;
  CUDACHECK(cudaStreamCreate(&stream));

  // Create device communicator with both LSA and GIN support
  ncclDevComm devComm;
  ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
  reqs.lsaBarrierCount = NCCL_DEVICE_CTA_COUNT;  // LSA barriers for local synchronization
  reqs.railGinBarrierCount = NCCL_DEVICE_CTA_COUNT;  // GIN barriers for network synchronization
  reqs.ginSignalCount = 1;  // GIN signals for completion detection
  NCCLCHECK(ncclDevCommCreate(comm, &reqs, &devComm));
  printf("  Rank %d created device communicator with hybrid support\n", my_rank);

  if (my_rank == 0) {
    printf("Starting Hybrid AlltoAll with %zu elements per rank (%zu total elements, %zu MB)\n",
           count, total_elements, size_bytes / (1024 * 1024));
    printf("Using LSA for local peers and GIN for remote peers\n");
  }

  // ==========================================================================
  // STEP 5: Execute Hybrid AlltoAll Kernel
  // ==========================================================================

  if (my_rank == 0) {
    printf("\n=== Executing Hybrid AlltoAll ===\n");
  }

  // Clear receive buffer
  CUDACHECK(cudaMemset(d_recvbuff, 0, size_bytes));

  // Launch hybrid AlltoAll kernel
  HybridAlltoAllKernel<float><<<NCCL_DEVICE_CTA_COUNT, NCCL_DEVICE_THREADS_PER_CTA, 0, stream>>>(
      send_win, 0, recv_win, 0, count, 0, devComm);

  // Wait for completion
  CUDACHECK(cudaStreamSynchronize(stream));
  printf("  Rank %d completed hybrid AlltoAll kernel\n", my_rank);

  // ==========================================================================
  // STEP 6: Verify Results
  // ==========================================================================

  // Verify hybrid results
  CUDACHECK(cudaMemcpy(h_recvbuff, d_recvbuff, size_bytes, cudaMemcpyDeviceToHost));
  bool hybrid_success = true;
  for (int src_rank = 0; src_rank < total_ranks; src_rank++) {
    for (size_t i = 0; i < count; i++) {
      size_t recv_idx = src_rank * count + i;
      float expected = (float)(src_rank * 1000 + my_rank * 100 + i);
      if (h_recvbuff[recv_idx] != expected) {
        hybrid_success = false;
        printf("  Rank %d: Hybrid mismatch at [%d][%zu]: got %.0f, expected %.0f\n",
               my_rank, src_rank, i, h_recvbuff[recv_idx], expected);
        break;
      }
    }
    if (!hybrid_success) break;
  }

  if (my_rank == 0) {
    printf("Hybrid AlltoAll result: %s\n", hybrid_success ? "PASSED" : "FAILED");
    if (hybrid_success) {
      printf("✓ All %zu elements correctly exchanged using hybrid communication\n", total_elements);
    }
  }

  // ==========================================================================
  // STEP 7: Cleanup Resources
  // ==========================================================================

    // Cleanup host memory
    free(h_sendbuff);
    free(h_recvbuff);

    // Device API specific cleanup
    NCCLCHECK(ncclDevCommDestroy(comm, &devComm));
    NCCLCHECK(ncclCommWindowDeregister(comm, send_win));
    NCCLCHECK(ncclCommWindowDeregister(comm, recv_win));
    NCCLCHECK(ncclMemFree(d_sendbuff));
    NCCLCHECK(ncclMemFree(d_recvbuff));

    // Standard NCCL cleanup
    NCCLCHECK(ncclCommFinalize(comm));
    NCCLCHECK(ncclCommDestroy(comm));
    CUDACHECK(cudaStreamDestroy(stream));

    return NULL;
}

int main(int argc, char* argv[]) {
  // Run example using the provided utility framework
  return run_example(argc, argv, hybridAlltoAll);
}
