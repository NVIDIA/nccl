/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

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
 * NCCL Symmetric Memory AllGather Example with Copy Engine
 *
 * This example demonstrates how to use NCCL's symmetric memory feature
 * combined with copy engine (CTAPolicy=2) for zero SM usage during
 * collective operations.
 *
 * Learning Objectives:
 * - Learn how to register symmetric memory windows with NCCL communicators
 * - See how to enable copy engine for zero SM usage via ncclConfig CTAPolicy=2
 * - See the proper lifecycle management of symmetric memory
 *
 */

/*
 * This function can be called inside an MPI rank or pthread thread. The
 * initialization and broadcast are implemented in common/src/utils.cc for
 * easier readability. For fully integrated examples using pthreads or MPI see
 * examples in 01_communicators.
 */
void *allGather(int my_rank, int total_ranks, int local_device,
                int devices_per_rank) {

  // ========================================================================
  // STEP 1: Initialize NCCL Communicator with Copy Engine Config
  // ========================================================================

  ncclUniqueId nccl_unique_id;
  if (my_rank == 0) {
    printf("Starting AllGather example with %d ranks (Copy Engine enabled)\n", total_ranks);
    NCCLCHECK(ncclGetUniqueId(&nccl_unique_id));
  }

  // Distribute unique ID.
  // This step ensures all ranks have the same unique ID for communicator
  // creation
  util_broadcast(0, my_rank, &nccl_unique_id);

  // Set device context for this rank
  // Each rank manages its assigned GPU device
  CUDACHECK(cudaSetDevice(local_device));

  // Configure NCCL to use copy engine (CTAPolicy=2) for zero SM usage
  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
  config.CTAPolicy = 2;

  // Initialize NCCL communicator with config
  // This creates the communication context for collective operations
  ncclComm_t comm;
  NCCLCHECK(ncclCommInitRankConfig(&comm, total_ranks, nccl_unique_id, my_rank, &config));
  printf("  Rank %d communicator initialized using device %d (CTAPolicy=2)\n", my_rank, local_device);

  // ========================================================================
  // STEP 2: Allocate Memory Using NCCL Allocator
  // ========================================================================

  if (my_rank == 0) {
    printf("Symmetric Memory allocation\n");
  }
  // Allocate memory - using larger buffers to demonstrate symmetric memory
  // benefits
  size_t sendcount = 1024 * 1024; // 1M elements per rank
  size_t send_size_bytes = sendcount * sizeof(float);
  size_t recv_size_bytes = sendcount * total_ranks * sizeof(float);

  printf("  Rank %d allocating %.2f MB send buffer, %.2f MB recv buffer\n", my_rank,
         (float)send_size_bytes / (1024 * 1024),
         (float)recv_size_bytes / (1024 * 1024));

  float *h_data = (float *)malloc(send_size_bytes);

  // Allocate buffers using NCCL allocator
  // NCCL's allocator is compatible with symmetric memory layouts
  void *d_sendbuff;
  void *d_recvbuff;
  NCCLCHECK(ncclMemAlloc(&d_sendbuff, send_size_bytes));
  NCCLCHECK(ncclMemAlloc(&d_recvbuff, recv_size_bytes));

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
  NCCLCHECK(ncclCommWindowRegister(comm, d_sendbuff, send_size_bytes, &send_win,
                                   NCCL_WIN_COLL_SYMMETRIC));
  NCCLCHECK(ncclCommWindowRegister(comm, d_recvbuff, recv_size_bytes, &recv_win,
                                   NCCL_WIN_COLL_SYMMETRIC));

  // ========================================================================
  // STEP 4: Initialize Data and Prepare for Communication
  // ========================================================================

  // Initialize data - each rank contributes its rank value
  // This creates a simple test pattern for verification
  for (size_t i = 0; i < sendcount; i++) {
    h_data[i] = (float)my_rank;
  }
  CUDACHECK(cudaMemcpy(d_sendbuff, h_data, send_size_bytes, cudaMemcpyHostToDevice));
  printf("  Rank %d data initialized (value: %d)\n", my_rank, my_rank);

  // Create stream for asynchronous operations
  // Streams allow overlapping computation and communication
  cudaStream_t stream;
  CUDACHECK(cudaStreamCreate(&stream));

  // ========================================================================
  // STEP 5: Perform AllGather Operation
  // ========================================================================

  if (my_rank == 0) {
    printf("Starting AllGather with %zu elements per rank (%zu MB total)\n", sendcount,
           recv_size_bytes / (1024 * 1024));
  }

  // Perform AllGather operation
  // Since symmetric memory is registered and CTAPolicy=2, NCCL uses copy engine
  NCCLCHECK(ncclAllGather(d_sendbuff, d_recvbuff, sendcount, ncclFloat,
                          comm, stream));

  if (my_rank == 0) {
    printf("AllGather completed successfully\n");
  }

  // ========================================================================
  // STEP 6: Verify Results and Validate Correctness
  // ========================================================================

  // Synchronize to ensure completion
  CUDACHECK(cudaStreamSynchronize(stream));

  // Verify results (optional - copy back and check)
  float *h_result = (float *)malloc(recv_size_bytes);
  CUDACHECK(cudaMemcpy(h_result, d_recvbuff, recv_size_bytes,
                       cudaMemcpyDeviceToHost));

  // Each segment should contain the rank value that was sent to it
  // Segment r should contain value r for all elements
  bool all_ok = true;
  if (my_rank == 0) {
    for (int r = 0; r < total_ranks && all_ok; r++) {
      float expected = (float)r;
      size_t offset = r * sendcount;
      printf("Verification - Segment %d: Expected: %.1f, Got: %.1f\n", r, expected,
             h_result[offset]);

      for (size_t i = 0; i < sendcount; i++) {
        if (fabsf(h_result[offset + i] - expected) > 0.001) {
          printf(" Results verification failed at segment %d, index %zu: Expected %.1f, Got "
                 "%.1f\n", r, i, expected, h_result[offset + i]);
          all_ok = false;
          break;
        }
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
  // 6. Destroy CUDA stream

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
    printf("Example completed - demonstrated symmetric memory + copy engine allgather\n");
  }

  return NULL;
}

int main(int argc, char *argv[]) {
  // Run example using the standard test framework
  // This handles MPI/pthread initialization, device assignment, and cleanup
  return run_example(argc, argv, allGather);
}
