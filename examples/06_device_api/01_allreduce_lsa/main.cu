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
 * NCCL Device API AllReduce Example
 *
 * This example demonstrates NCCL's Device API, which enables GPU kernels to
 * directly interact with NCCL without CPU intervention. This is particularly
 * powerful for applications that need to perform communication
 * from within CUDA kernels.
 *
 * Learning Objectives:
 * - Understand NCCL Device API vs Host API differences
 * - Learn how to register memory windows for device-side access
 * - See how GPU kernels can perform collective operations directly
 * - Practice LSA (Load Store Access) barrier synchronization
 *
 * Key Device API Concepts:
 * - ncclDevComm: Device-side communicator for kernel use
 * - ncclWindow_t: Memory windows that enable direct peer access
 * - LSA barriers: Synchronization primitives for device-side coordination
 * - ncclGetLsaPointer: Direct access to peer memory from device code
 *
 * When to Use Device API:
 * - Compute kernels that need immediate communication results
 * - Fusion of computation and communication in a single kernel
 * - Reduced host-device synchronization overhead
 * - Custom collective operations not available in standard NCCL
 *
 * Performance Considerations:
 * - Lower latency than host API for small operations
 * - Enables computation-communication overlap within kernels
 * - Requires careful synchronization and memory ordering
 * - LSA barriers add coordination overhead but enable correctness
 */

// Device API kernel launch configuration
// CTA count must match lsaBarrierCount for proper barrier synchronization
#define NCCL_DEVICE_CTA_COUNT 16
#define NCCL_DEVICE_THREADS_PER_CTA 512

// ==========================================================================
// Device Kernel Implementation
// ==========================================================================

// Device kernel that performs AllReduce sum operation
// This kernel demonstrates direct NCCL communication from GPU threads
__global__ void simpleAllReduceKernel(ncclWindow_t sendwin, size_t sendoffset,
                                      ncclWindow_t recvwin, size_t recvoffset,
                                      size_t count, int root, struct ncclDevComm devComm) {
  // LSA barriers enable coordination between GPU threads across different ranks
  // Barrier scope: CTA (all threads in this block participate)
  // Barrier index: blockIdx.x selects this CTA's dedicated barrier (one barrier per CTA)
  ncclLsaBarrierSession<ncclCoopCta> bar { ncclCoopCta(), devComm, ncclTeamLsa(devComm),
                                           devComm.lsaBarrier, blockIdx.x };
  bar.sync(ncclCoopCta(), cuda::memory_order_relaxed);

  const int rank = devComm.rank, nRanks = devComm.nRanks;

  // We are going to spread the workload accross all GPU ranks.
  // So calculate the global thread ID accross all ranks.
  // This maps global threads to data elements in the data to be reduced
  const int globalTid = threadIdx.x + blockDim.x * (rank + blockIdx.x * nRanks);
  const int globalNthreads = blockDim.x * gridDim.x * nRanks;

  // Grid stride loop over all elements with the globalThreads
  for (size_t offset = globalTid; offset < count; offset += globalNthreads) {
    float v = 0;
    // Access remote (and local [peer==rank]) memory and reduce locally
    for (int peer=0; peer<nRanks; peer++) {
      // Access peer memory directly using LSA (Load/Store Accessible) pointers
      float* sendPtr = (float*)ncclGetLsaPointer(sendwin, sendoffset, peer);
      v += sendPtr[offset];
    }
    // Write the result back to remote and local memory
    for (int peer=0; peer<nRanks; peer++) {
      float* recvPtr = (float*)ncclGetLsaPointer(recvwin, recvoffset, peer);
      recvPtr[offset] = v;
    }
  }
  // Release barrier ensures that we received data from everyone before we unblock the stream
  // and allow the next kernel(s) to process the data.
  // Critical for correctness in device-side collective operations
  bar.sync(ncclCoopCta(), cuda::memory_order_release);
}

// ==========================================================================
// Host-Side Setup and Device API Initialization
// ==========================================================================

// This function can be called inside an MPI rank or pthread thread. The
// initialization and broadcast are implemented in common/src/utils.cc for
// easier readability. For fully integrated examples using pthreads or MPI see
// examples. in 01_communicators.

void* allReduce(int my_rank, int total_ranks, int local_device, int devices_per_rank) {
  ncclComm_t comm;
  ncclUniqueId nccl_unique_id;

  if (my_rank == 0) {
    printf("Starting Device API AllReduce initialization\n");
  }


  // Standard NCCL communicator initialization (same as Host API)
  if (my_rank == 0) {
    NCCLCHECK(ncclGetUniqueId(&nccl_unique_id));
  }

  // Distribute unique ID in case of MPI.
  util_broadcast(0, my_rank, &nccl_unique_id);

  // Set device context for this rank
  CUDACHECK(cudaSetDevice(local_device));
  printf("  Rank %d using GPU device %d\n", my_rank, local_device);

  // ==========================================================================
  // STEP 2: Initialize NCCL Communicator and Allocate Memory
  // ==========================================================================

  // Initialize NCCL communicator (same as Host API)
  NCCLCHECK(ncclCommInitRank(&comm, total_ranks, nccl_unique_id, my_rank));
  printf("  Rank %d initialized NCCL communicator for %d total ranks\n", my_rank, total_ranks);

  // Allocate memory for AllReduce operation
  size_t count = 1024 * 1024; // 1M elements
  size_t size_bytes = count * sizeof(float);

  float *h_data = (float*)malloc(size_bytes);
  void* d_sendbuff;
  void* d_recvbuff;
  ncclWindow_t send_win;
  ncclWindow_t recv_win;

  // Device API requires allocation compatible with symmetric memory allocation
  // This ensures memory can be accessed directly by device kernels from all ranks
  NCCLCHECK(ncclMemAlloc(&d_sendbuff, size_bytes));
  NCCLCHECK(ncclMemAlloc(&d_recvbuff, size_bytes));

  // ==========================================================================
  // STEP 4: Register Memory Windows for Device-Side Access
  // ==========================================================================

  // Register symmetric windows for LSA access
  // Windows enable direct peer-to-peer access from device kernels
  NCCLCHECK(ncclCommWindowRegister(comm, d_sendbuff, size_bytes, &send_win, NCCL_WIN_COLL_SYMMETRIC));
  NCCLCHECK(ncclCommWindowRegister(comm, d_recvbuff, size_bytes, &recv_win, NCCL_WIN_COLL_SYMMETRIC));

  // Initialize data with rank-specific values for verification
  for (size_t i = 0; i < count; i++) {
    h_data[i] = (float)my_rank;
  }
  CUDACHECK(cudaMemcpy(d_sendbuff, h_data, size_bytes, cudaMemcpyHostToDevice));
  printf("  Rank %d initialized data with value %d\n", my_rank, my_rank);

  // ==========================================================================
  // STEP 5: Create Device Communicator and Configure LSA Barriers
  // ==========================================================================

  // Create stream for kernel execution
  cudaStream_t stream;
  CUDACHECK(cudaStreamCreate(&stream));

  // Create device communicator - this is the key Device API component
  // Requirements specify resources to allocate (e.g., one barrier per CTA)
  ncclDevComm devComm;
  ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
  reqs.lsaBarrierCount = NCCL_DEVICE_CTA_COUNT; // Must match kernel launch config
  NCCLCHECK(ncclDevCommCreate(comm, &reqs, &devComm));
  printf("  Rank %d created device communicator with %d LSA barriers\n", my_rank, NCCL_DEVICE_CTA_COUNT);

  if (my_rank == 0) {
    printf("Starting AllReduce with %zu elements (%zu MB) using Device API\n",
           count, size_bytes / (1024 * 1024));
    printf("Expected result: sum of ranks 0 to %d = %d per element\n",
           total_ranks - 1, (total_ranks * (total_ranks - 1)) / 2);
  }

  // ==========================================================================
  // STEP 6: Launch Device Kernel for AllReduce Operation
  // ==========================================================================

  // Launch device kernel to perform AllReduce
  // This kernel will directly access peer memory and perform collective operation
  simpleAllReduceKernel<<<NCCL_DEVICE_CTA_COUNT, NCCL_DEVICE_THREADS_PER_CTA, 0, stream>>>(
                                                                                           send_win, 0, recv_win, 0, count, 0, devComm);

  // Wait for completion - kernel performs AllReduce.
  CUDACHECK(cudaStreamSynchronize(stream));
  printf("  Rank %d completed AllReduce kernel execution\n", my_rank);

  // ==========================================================================
  // STEP 7: Verify Results and Cleanup Resources
  // ==========================================================================

  // Verify results by copying back and checking
  CUDACHECK(cudaMemcpy(h_data, d_recvbuff, size_bytes, cudaMemcpyDeviceToHost));
  float expected = (float)((total_ranks * (total_ranks - 1)) / 2);
  bool success = true;
  for (int i = 0; i < count; i++) {
    if (h_data[i] != expected) {
      success = false;
      break;
    }
  }

  if (my_rank == 0) {
    printf("AllReduce completed. Result verification: %s\n",
           success ? "PASSED" : "FAILED");
    if (success) {
      printf("All elements correctly sum to %.0f (ranks 0-%d)\n",
             expected, total_ranks - 1);
    }
  }

  // Cleanup resources in proper order
  free(h_data);

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
  return run_example(argc, argv, allReduce);
}
