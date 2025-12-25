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
 * NCCL Device API Pure GIN AlltoAll Example
 *
 * This example demonstrates NCCL's GPU-Initiated Networking (GIN) capabilities
 * for performing AlltoAll collective operations directly from GPU kernels using
 * only network-based communication.
 * GIN enables GPU kernels to initiate network communication without CPU
 * intervention, providing low-latency communication for distributed applications.
 *
 * Learning Objectives:
 * - Understand pure GIN (GPU-Initiated Networking) communication
 * - Learn how to use ncclGin for device-initiated network communication
 * - See pure GIN AlltoAll implementation for network-based communication
 * - Practice GIN barriers and signal-based synchronization
 *
 * Key GIN Concepts:
 * - ncclGin: Device-side networking object for kernel-initiated communication
 * - GIN contexts: Network communication channels for parallel operations
 * - GIN signals: Completion notifications for asynchronous operations
 * - GIN barriers: Network-based synchronization across ranks
 * - One-sided put operations: Direct remote memory writes over network
 *
 * When to Use Pure GIN:
 * - Communication between ranks that cannot use LSA (different nodes)
 * - Network-based collective operations in multi-node environments
 * - Scenarios where all communication must go through the network
 * - Testing network performance without local optimizations
 *
 * Performance Considerations:
 * - GIN provides network communication from GPU kernels
 * - All communication goes through the network (no local optimizations)
 * - Signal-based completion detection enables asynchronous operation
 * - Multiple GIN contexts can improve parallel communication performance
 */

// Device API kernel launch configuration
// CTA count must match railGinBarrierCount for proper barrier synchronization
 #define NCCL_DEVICE_CTA_COUNT 1
 #define NCCL_DEVICE_THREADS_PER_CTA 512

 // ==========================================================================
 // Device Kernel Implementations
 // ==========================================================================

// Pure GIN AlltoAll kernel - uses GIN for all peer communication
// This kernel demonstrates network-based AlltoAll using GPU-initiated networking
template <typename T>
__global__ void PureGinAlltoAllKernel(ncclWindow_t sendwin, size_t sendoffset,
                                      ncclWindow_t recvwin, size_t recvoffset,
                                      size_t count, int root, struct ncclDevComm devComm) {
  int ginContext = 0;
  unsigned int signalIndex = 0;
  ncclGin gin { devComm, ginContext };
  uint64_t signalValue = gin.readSignal(signalIndex);

  // GIN barriers enable coordination between GPU threads across different ranks over network
  ncclGinBarrierSession<ncclCoopCta> bar { ncclCoopCta(), gin, ncclTeamWorld(devComm),
                                           devComm.railGinBarrier, blockIdx.x };
  bar.sync(ncclCoopCta(), cuda::memory_order_relaxed, ncclGinFenceLevel::Relaxed);

  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int nthreads = blockDim.x * gridDim.x;

  // Send to all peers via GIN (GPU-initiated networking)
  const size_t size = count * sizeof(T);
  for (int r = tid; r < devComm.nRanks; r += nthreads) {
    gin.put(ncclTeamWorld(devComm), r,
        recvwin, recvoffset + devComm.rank * size,
        sendwin, sendoffset + r * size,
        size, ncclGin_SignalInc{signalIndex});
  }

  // Wait for all remote puts to complete using signal-based synchronization
  gin.waitSignal(ncclCoopCta(), signalIndex, signalValue + devComm.nRanks);
  gin.flush(ncclCoopCta());
}

 // ==========================================================================
 // Host-Side Setup and Device API Initialization
 // ==========================================================================

void* pureGinAlltoAll(int my_rank, int total_ranks, int local_device, int devices_per_rank) {
  ncclComm_t comm;
  ncclUniqueId nccl_unique_id;

  if (my_rank == 0) {
    printf("Starting Pure GIN AlltoAll initialization\n");
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

  // Register symmetric windows for GIN access
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
  // STEP 4: Create Device Communicator with GIN Support
  // ==========================================================================

  // Create stream for kernel execution
  cudaStream_t stream;
  CUDACHECK(cudaStreamCreate(&stream));

  // Create device communicator with GIN support
  ncclDevComm devComm;
  ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
  reqs.railGinBarrierCount = NCCL_DEVICE_CTA_COUNT;  // GIN barriers for network synchronization
  reqs.ginSignalCount = 1;  // GIN signals for completion detection
  NCCLCHECK(ncclDevCommCreate(comm, &reqs, &devComm));
  printf("  Rank %d created device communicator with GIN support\n", my_rank);

  if (my_rank == 0) {
    printf("Starting Pure GIN AlltoAll with %zu elements per rank (%zu total elements, %zu MB)\n",
            count, total_elements, size_bytes / (1024 * 1024));
  }

  // ==========================================================================
  // STEP 5: Execute Pure GIN AlltoAll Kernel
  // ==========================================================================

  if (my_rank == 0) {
    printf("\n=== Executing Pure GIN AlltoAll ===\n");
  }

    // Clear receive buffer
    CUDACHECK(cudaMemset(d_recvbuff, 0, size_bytes));

  // Launch pure GIN AlltoAll kernel
  PureGinAlltoAllKernel<float><<<NCCL_DEVICE_CTA_COUNT, NCCL_DEVICE_THREADS_PER_CTA, 0, stream>>>(
      send_win, 0, recv_win, 0, count, 0, devComm);

  // Wait for completion
  CUDACHECK(cudaStreamSynchronize(stream));
  printf("  Rank %d completed pure GIN AlltoAll kernel\n", my_rank);

  // ==========================================================================
  // STEP 6: Verify Results
  // ==========================================================================

  // Verify pure GIN results
  CUDACHECK(cudaMemcpy(h_recvbuff, d_recvbuff, size_bytes, cudaMemcpyDeviceToHost));
  bool gin_success = true;
  for (int src_rank = 0; src_rank < total_ranks; src_rank++) {
    for (size_t i = 0; i < count; i++) {
      size_t recv_idx = src_rank * count + i;
      float expected = (float)(src_rank * 1000 + my_rank * 100 + i);
      if (h_recvbuff[recv_idx] != expected) {
        gin_success = false;
        printf("  Rank %d: Pure GIN mismatch at [%d][%zu]: got %.0f, expected %.0f\n",
                my_rank, src_rank, i, h_recvbuff[recv_idx], expected);
        break;
      }
    }
    if (!gin_success) break;
  }

  if (my_rank == 0) {
    printf("Pure GIN AlltoAll result: %s\n", gin_success ? "PASSED" : "FAILED");
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
  return run_example(argc, argv, pureGinAlltoAll);
}
