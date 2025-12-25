/*************************************************************************
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "cuda_runtime.h"
#include "mpi.h"
#include "nccl.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/**
 * NCCL Example: One Device per Process with MPI
 * =============================================
 *
 * LEARNING OBJECTIVE:
 * This example teaches the fundamental NCCL pattern: one GPU device per MPI
 * process. This is the most common deployment pattern for multi-GPU distributed
 * training.
 *
 * WHAT THIS CODE DEMONSTRATES:
 * - How to initialize NCCL communicators across multiple processes
 * - Proper GPU assignment in both single-node and multi-node environments
 * - Complete NCCL communicator lifecycle management
 * - Error handling best practices for production code
 *
 * STEP-BY-STEP PROCESS:
 * 1. MPI Setup: Initialize MPI and determine process layout
 * 2. GPU Assignment: Map each process to a local GPU device
 * 3. NCCL ID Sharing: Rank 0 creates unique ID, broadcasts to all processes
 * 4. Communicator Creation: Each process joins the NCCL communicator
 * 5. Verification: Query and verify communicator properties
 * 6. Clean Shutdown: Properly destroy all resources in correct order
 *
 * MULTI-NODE INTELLIGENCE:
 * - Automatically detects which processes are on the same physical node
 * - Assigns local GPU indices (0, 1, 2, 3...) to processes on each node
 * - Uses MPI_Comm_split_type with MPI_COMM_TYPE_SHARED for robust node
 * identification
 * - Leverages MPI's native shared memory detection for optimal performance
 *
 * USAGE EXAMPLES:
 *   Single node (4 GPUs): mpirun -np 4 ./one_device_per_process_mpi
 *
 * EXPECTED OUTPUT:
 *   Each process will report: MPI rank → NCCL rank → GPU device assignment
 *   Success message confirms all communicators were created properly
 */

// Enhanced error checking macro for NCCL operations
// Provides detailed error information including the failed operation

#define NCCLCHECK(cmd)                                                         \
  do {                                                                         \
    ncclResult_t res = cmd;                                                    \
    if (res != ncclSuccess) {                                                  \
      fprintf(stderr, "Failed, NCCL error %s:%d '%s'\n", __FILE__, __LINE__,   \
              ncclGetErrorString(res));                                        \
      fprintf(stderr, "Failed NCCL operation: %s\n", #cmd);                    \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

#define CUDACHECK(cmd)                                                         \
  do {                                                                         \
    cudaError_t err = cmd;                                                     \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "Failed: Cuda error %s:%d '%s'\n", __FILE__, __LINE__,   \
              cudaGetErrorString(err));                                        \
      fprintf(stderr, "Failed CUDA operation: %s\n", #cmd);                    \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

// =============================================================================
// LOCAL RANK UTILITY FUNCTION - For Multi-Node GPU Assignment
// =============================================================================

/**
 * Determine the local rank of this process on its physical node
 *
 * Algorithm:
 * 1. Split the communicator based on shared memory (i.e., nodes)
 * 2. Get the rank within the node communicator
 * 3. This rank becomes the local rank for GPU assignment
 *
 * @param comm The MPI communicator to use for determining local rank
 * @return Local rank (0, 1, 2...) for GPU assignment, or -1 on error
 */
int getLocalRank(MPI_Comm comm) {

  int world_size;
  MPI_Comm_size(comm, &world_size);

  int world_rank;
  MPI_Comm_rank(comm, &world_rank);

  // Split the communicator based on shared memory (i.e., nodes)
  MPI_Comm node_comm;
  MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, world_rank, MPI_INFO_NULL,
                      &node_comm);

  // Get the rank and size within the node communicator
  int node_rank, node_size;
  MPI_Comm_rank(node_comm, &node_rank);
  MPI_Comm_size(node_comm, &node_size);

  // Clean up the node communicator
  MPI_Comm_free(&node_comm);

  return node_rank;
}

// =============================================================================
// MAIN FUNCTION - NCCL Communicator Lifecycle Example
// =============================================================================

int main(int argc, char *argv[]) {
  // Variables for MPI, CUDA, and NCCL components
  int mpi_rank, mpi_size, local_rank;
  int num_gpus = 0;
  ncclComm_t comm = NULL;
  cudaStream_t stream = NULL;
  ncclUniqueId nccl_id;

  // =========================================================================
  // STEP 1: Initialize MPI and determine process layout
  // =========================================================================

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);


  if (mpi_rank == 0) {
    printf("Starting NCCL communicator lifecycle example with %d processes\n",
           mpi_size);
  }
  // Determine which local GPU this process should use
  local_rank = getLocalRank(MPI_COMM_WORLD);

  printf("  MPI initialized - Process %d of %d total processes\n", mpi_rank,
         mpi_size);

  // =========================================================================
  // STEP 2: Setup CUDA device for this process
  // =========================================================================

  // Check how many CUDA devices are available on this node
  CUDACHECK(cudaGetDeviceCount(&num_gpus));
  printf("  Found %d CUDA devices on this node\n", num_gpus);

  if (num_gpus == 0) {
    fprintf(stderr, "ERROR: No CUDA devices found on this node!\n");
    exit(EXIT_FAILURE);
  }

  if (local_rank >= num_gpus) {
    fprintf(stderr,
            "ERROR: Process %d needs GPU %d but only %d devices available\n",
            mpi_rank, local_rank, num_gpus);
    exit(EXIT_FAILURE);
  }

  // Assign this process to its designated GPU device
  CUDACHECK(cudaSetDevice(local_rank));

  // Create CUDA stream for GPU operations
  CUDACHECK(cudaStreamCreate(&stream));

  printf("  MPI rank %d assigned to CUDA device %d\n", mpi_rank,
         local_rank);

  // =========================================================================
  // STEP 3: Initialize NCCL communicator
  // =========================================================================

  // Generate NCCL unique ID (only rank 0 needs to do this)
  if (mpi_rank == 0) {
    NCCLCHECK(ncclGetUniqueId(&nccl_id));
    printf("Rank 0 generated NCCL unique ID for all processes\n");
  }

  // Share the unique ID with all processes using MPI broadcast
  MPI_Bcast(&nccl_id, NCCL_UNIQUE_ID_BYTES, MPI_CHAR, 0, MPI_COMM_WORLD);
  printf("INFO: Rank %d received NCCL unique ID\n", mpi_rank);

  // Create NCCL communicator for this process
  // This is where each process joins the distributed NCCL communicator
  NCCLCHECK(ncclCommInitRank(&comm, mpi_size, nccl_id, mpi_rank));
  printf("  Rank %d created NCCL communicator\n", mpi_rank);

  // =========================================================================
  // STEP 4: Verify communicator setup
  // =========================================================================

  // Query communicator properties to verify everything is set up correctly
  int comm_rank, comm_size, comm_device;
  NCCLCHECK(ncclCommUserRank(comm, &comm_rank));
  NCCLCHECK(ncclCommCount(comm, &comm_size));
  NCCLCHECK(ncclCommCuDevice(comm, &comm_device));

  printf("  MPI rank %d → NCCL rank %d/%d on GPU device %d\n", mpi_rank,
         comm_rank, comm_size, comm_device);

  // Give all processes a chance to finish their printf
  MPI_Barrier(MPI_COMM_WORLD);

  // =========================================================================
  // STEP 5: Clean shutdown and resource cleanup
  // =========================================================================

  if (mpi_rank == 0) {
    printf(
        "\nAll communicators initialized successfully! Beginning cleanup...\n");
  }

  // Synchronize CUDA stream to ensure all GPU work is complete
  if (stream != NULL) {
    CUDACHECK(cudaStreamSynchronize(stream));
  }

  // Destroy NCCL communicator FIRST (before CUDA resources)
  // This is important - NCCL cleanup should happen before CUDA cleanup
  if (comm != NULL) {
    NCCLCHECK(ncclCommFinalize(comm));
    NCCLCHECK(ncclCommDestroy(comm));
    printf("  Rank %d destroyed NCCL communicator\n", mpi_rank);
  }

  // Now destroy CUDA stream
  if (stream != NULL) {
    CUDACHECK(cudaStreamDestroy(stream));
  }

  if (mpi_rank == 0) {
    printf(
        "\nAll NCCL communicators created and cleaned up properly!\n");
    printf("This example demonstrated the complete NCCL communicator "
           "lifecycle.\n");
    printf("Next steps: Try running NCCL collective operations (AllReduce, "
           "etc.)\n");
  }

  MPI_Finalize();
  return 0;
}
