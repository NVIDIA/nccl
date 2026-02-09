/*************************************************************************
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "cuda_runtime.h"
#include "nccl.h"
#include "utils.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <unistd.h>

/*
 * NCCL Symmetric Memory AllReduce Example
 *
 * This example demonstrates how to use NCCL's symmetric memory feature
 * for collective operations. Symmetric memory provides optimized performance
 * by leveraging consistent memory layouts across all participating ranks.
 *
 * Learning Objectives:
 * - Learn how to register symmetric memory windows with NCCL communicators
 * - See the proper lifecycle management of symmetric memory
 *
 */

/*
 * This function can be called inside an MPI rank or pthread thread. The
 * initialization and broadcast are implemented in common/src/utils.cc for
 * easier readability. For fully integrated examples using pthreads or MPI see
 * examples in 01_communicators.
 */
void *allReduce(int my_rank, int total_ranks, int local_device,
                int devices_per_rank) {

  // ========================================================================
  // STEP 1: Initialize NCCL Communicator and Setup
  // ========================================================================

  ncclUniqueId nccl_unique_id;
  if (my_rank == 0) {
    printf("Starting AllReduce example with %d ranks\n", total_ranks);
    NCCLCHECK(ncclGetUniqueId(&nccl_unique_id));
  }

  // Distribute unique ID.
  // This step ensures all ranks have the same unique ID for communicator
  // creation
  util_broadcast(0, my_rank, &nccl_unique_id);

  // Set device context for this rank
  // Each rank manages its assigned GPU device
  CUDACHECK(cudaSetDevice(local_device));

  // Initialize NCCL communicator
  // This creates the communication context for collective operations
  ncclComm_t comm;
  NCCLCHECK(ncclCommInitRank(&comm, total_ranks, nccl_unique_id, my_rank));
  printf("  Rank %d communicator initialized using device %d\n", my_rank,
         local_device);

  // ========================================================================
  // STEP 2: Allocate Memory Using NCCL Allocator
  // ========================================================================

  if (my_rank == 0) {
    printf("Symmetric Memory allocation\n");
  }
  // Allocate memory - using larger buffers to demonstrate symmetric memory
  // benefits
  size_t count = 1024 * 1024; // 1M elements
  size_t size_bytes = count * sizeof(float);

  printf("  Rank %d allocating %.2f MB per buffer\n", my_rank,
         (float)size_bytes / (1024 * 1024));

  float *h_data = (float *)malloc(size_bytes);

  // Allocate buffers using NCCL allocator
  // NCCL's allocator is compatible with symmetric memory layouts
  void *d_sendbuff;
  void *d_recvbuff;
  NCCLCHECK(ncclMemAlloc(&d_sendbuff, size_bytes));
  NCCLCHECK(ncclMemAlloc(&d_recvbuff, size_bytes));

  // ========================================================================
  // STEP 3: Register Symmetric Memory Windows
  // ========================================================================

  /* Passing NCCL_WIN_COLL_SYMMETRIC requires users to provide the symmetric
   * buffers among all ranks in collectives.
   * Every rank needs to call ncclCommWindowRegister to register its buffers.
   */

  // Register symmetric memory windows with NCCL
  ncclWindow_t send_win;
  ncclWindow_t recv_win;
  NCCLCHECK(ncclCommWindowRegister(comm, d_sendbuff, size_bytes, &send_win,
                                   NCCL_WIN_COLL_SYMMETRIC));
  NCCLCHECK(ncclCommWindowRegister(comm, d_recvbuff, size_bytes, &recv_win,
                                   NCCL_WIN_COLL_SYMMETRIC));

  // ========================================================================
  // STEP 4: Initialize Data and Prepare for Communication
  // ========================================================================

  // Initialize data - each rank contributes its rank value
  // This creates a simple test pattern for verification
  for (size_t i = 0; i < count; i++) {
    h_data[i] = (float)my_rank;
  }
  CUDACHECK(cudaMemcpy(d_sendbuff, h_data, size_bytes, cudaMemcpyHostToDevice));
  printf("  Rank %d data initialized (value: %d)\n", my_rank, my_rank);

  // Create stream for asynchronous operations
  // Streams allow overlapping computation and communication
  cudaStream_t stream;
  CUDACHECK(cudaStreamCreate(&stream));

  // ========================================================================
  // STEP 5: Perform AllReduce Operation
  // ========================================================================

  if (my_rank == 0) {
    printf("Starting AllReduce with %zu elements (%zu MB)\n", count,
           size_bytes / (1024 * 1024));
  }

  // Perform AllReduce operation
  // Since symmetric memory is registered, NCCL can apply optimized algorithms
  NCCLCHECK(ncclAllReduce(d_sendbuff, d_recvbuff, count, ncclFloat, ncclSum,
                          comm, stream));

  if (my_rank == 0) {
    printf("AllReduce completed successfully\n");
  }

  // ========================================================================
  // STEP 6: Verify Results and Validate Correctness
  // ========================================================================

  // Synchronize to ensure completion
  CUDACHECK(cudaStreamSynchronize(stream));

  // Verify results (optional - copy back and check)
  float *h_result = (float *)malloc(size_bytes);
  CUDACHECK(cudaMemcpy(h_result, d_recvbuff, size_bytes,
                       cudaMemcpyDeviceToHost));

  // Each element should be the sum of all ranks
  float expected_sum = (float)(total_ranks * (total_ranks - 1)) / 2;
  bool all_ok = true;
  if (my_rank == 0) {
    printf("Verification - Expected: %.1f, Got: %.1f\n", expected_sum,
           h_result[0]);

    for (size_t i = 1; i < count; i++) {
      if (fabsf(h_result[i] - expected_sum) > 0.001) {
        printf(" Results verification failed at index %zu: Expected %.1f, Got "
               "%.1f\n", i, expected_sum, h_result[i]);
        all_ok = false;
        break;
      }
    }

    if (all_ok) {
      printf("Results verified correctly\n");
    } else {
      printf("Results verification failed\n");
    }
  }

  // ========================================================================
  // STEP 7: Cleanup and Resource Management
  // ========================================================================

  // Important: Cleanup must happen in the correct order
  // 1. Free host memory
  // 2. Deregister symmetric memory windows
  // 3. Free device memory
  // 4. Destroy CUDA resources
  // 5. Finalize and destroy NCCL communicator

  free(h_data);
  free(h_result);

  // Deregister symmetric memory windows from communicator
  // This must happen before freeing the buffers or destroying the
  // communicator
  NCCLCHECK(ncclCommWindowDeregister(comm, send_win));
  NCCLCHECK(ncclCommWindowDeregister(comm, recv_win));
  printf("  Rank %d symmetric memory windows deregistered\n", my_rank);

  // Free device memory allocated by NCCL
  NCCLCHECK(ncclMemFree(d_sendbuff));
  NCCLCHECK(ncclMemFree(d_recvbuff));

  // Finalize and destroy NCCL communicator
  NCCLCHECK(ncclCommFinalize(comm));
  NCCLCHECK(ncclCommDestroy(comm));

  // Destroy CUDA stream
  CUDACHECK(cudaStreamDestroy(stream));

  if (my_rank == 0) {
    printf("All resources cleaned up successfully\n");
    printf("Example completed - demonstrated symmetric memory lifecycle\n");
  }

  return NULL;
}

int main(int argc, char *argv[]) {
  // Run example using the standard test framework
  // This handles MPI/pthread initialization, device assignment, and cleanup
  return run_example(argc, argv, allReduce);
}
