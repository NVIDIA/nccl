/*************************************************************************
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "cuda_runtime.h"
#include "nccl.h"
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

/*
 * NCCL Example: Multiple Devices Single Process
 * =============================================
 *
 * PURPOSE:
 * This example demonstrates how to initialize NCCL communicators for multiple
 * GPUs within a single process. This is the simplest NCCL setup and is ideal
 * for learning NCCL basics or for applications that want to use multiple GPUs
 * without the complexity of multi-process coordination.
 *
 * LEARNING OBJECTIVES:
 * - Learn how to use ncclCommInitAll() for simple multi-GPU setups
 * - See proper NCCL communicator lifecycle management
 * - Understand GPU device management in NCCL applications
 * - Learn proper resource cleanup patterns
 *
 * HOW IT WORKS:
 * 1. Detect all available CUDA devices
 * 2. Create communicators for all devices using ncclCommInitAll()
 * 3. Verify communicator properties (rank, size, device assignment)
 * 4. Clean up all resources properly
 *
 * KEY CONCEPTS:
 * - ncclCommInitAll(): Creates multiple communicators in a single call
 * - Single-process topology: All GPUs managed by one process
 * - Device management: Setting active CUDA device for operations
 * - Stream management: Each GPU gets its own CUDA stream
 *
 * WHEN TO USE THIS PATTERN:
 * - Learning NCCL fundamentals
 * - Single-node, multi-GPU applications
 * - Applications that don't need multi-node scaling
 * - Prototyping and testing NCCL functionality
 *
 * USAGE EXAMPLES:
 * ./multiple_devices_single_process               # Use all available GPUs
 *
 * EXPECTED OUTPUT:
 * - Detection of all available GPUs
 * - Successful communicator initialization
 * - Display of rank/size information for each GPU
 * - Clean resource cleanup confirmation
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
// MAIN FUNCTION - NCCL Communicator Lifecycle Example
// =============================================================================

int main(int argc, char *argv[]) {
  // Variables for managing multiple GPU communicators
  int num_gpus;                 // Number of available CUDA devices
  ncclComm_t *comms = NULL;     // Array of NCCL communicators (one per GPU)
  cudaStream_t *streams = NULL; // Array of CUDA streams (one per GPU)
  int *devices = NULL;          // Array of device IDs to use

  // Discover how many CUDA devices are available
  // This determines how many NCCL communicators we'll create
  CUDACHECK(cudaGetDeviceCount(&num_gpus));

  if (num_gpus == 0) {
    fprintf(stderr, "ERROR: No CUDA devices found on this system\n");
    fprintf(
        stderr,
        "Please ensure CUDA is properly installed and GPUs are available\n");
    return 1;
  }

  printf("Found %d CUDA device(s) available\n\n", num_gpus);

  // =========================================================================
  // STEP 1: Prepare Device Information and Memory Allocation
  // =========================================================================

  // Allocate arrays to hold our per-device resources
  // We need one communicator, stream, and device ID per GPU
  devices = (int *)malloc(num_gpus * sizeof(int));
  comms = (ncclComm_t *)malloc(num_gpus * sizeof(ncclComm_t));
  streams = (cudaStream_t *)malloc(num_gpus * sizeof(cudaStream_t));

  if (!devices || !comms || !streams) {
    fprintf(stderr, "ERROR: Failed to allocate memory for device arrays\n");
    return 1;
  }

  // Create device list and display device information
  // By default, we use all available devices (0, 1, 2, ...)
  printf("Available GPU devices:\n");
  for (int i = 0; i < num_gpus; i++) {
    devices[i] = i; // Use device i for communicator i

    // Query device properties for informational display
    cudaDeviceProp prop;
    CUDACHECK(cudaGetDeviceProperties(&prop, devices[i]));
    printf("  GPU %d: %s (CUDA Device %d)\n", i, prop.name, devices[i]);
    printf("    Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("    Memory: %.1f GB\n",
           prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
  }

  // Create a CUDA stream for each GPU
  // Each GPU needs its own stream for optimal performance
  for (int i = 0; i < num_gpus; i++) {
    // Set the active CUDA device before creating resources on it
    // This ensures the stream is created on the correct GPU
    CUDACHECK(cudaSetDevice(devices[i]));
    CUDACHECK(cudaStreamCreate(&streams[i]));
  }

  // =========================================================================
  // STEP 2 : Initialize NCCL Communicators
  // =========================================================================

  printf("Using ncclCommInitAll() to create all communicators "
         "simultaneously\n");

  // ncclCommInitAll() creates all communicators at once and handles the
  // coordination internally
  //
  // Parameters:
  // - comms: Array to store the created communicators
  // - num_gpus: Number of communicators to create
  // - devices: Array of CUDA device IDs to use
  //
  // After this call:
  // - comms[0] will be the communicator for devices[0] with rank 0
  // - comms[1] will be the communicator for devices[1] with rank 1
  // - ... and so on
  //
  // All communicators will have the same 'size' (total number of
  // participants)
  NCCLCHECK(ncclCommInitAll(comms, num_gpus, devices));
  printf("All %d NCCL communicators initialized successfully\n\n", num_gpus);

  // =========================================================================
  // STEP 3: Create CUDA Streams and Verify Communicator Properties
  // =========================================================================

  printf("Communicator Details:\n");

  bool sizes_match = true;
  for (int i = 0; i < num_gpus; i++) {

    // Query the communicator to verify it was set up correctly
    // These calls validate that NCCL properly assigned ranks and devices
    int rank, size, device;
    // Get this communicator's rank
    NCCLCHECK(ncclCommUserRank(comms[i], &rank));
    // Get total number of participants
    NCCLCHECK(ncclCommCount(comms[i], &size));
    // Get assigned CUDA device
    NCCLCHECK(ncclCommCuDevice(comms[i], &device));

    printf("  Communicator %d: Rank %d/%d on CUDA device %d", i, rank, size,
           device);

    // Verify the assignment is correct
    if (rank != i) {
      printf(" [WARNING: Expected rank %d]", i);
    }
    if (device != devices[i]) {
      printf(" [WARNING: Expected device %d]", devices[i]);
    }
    printf("\n");

    // Verify that all communicators have the expected size
    if (size != num_gpus) {
      printf("WARNING: Communicator %d has size %d, expected %d\n", i, size, num_gpus);
      sizes_match = false;
    }
  }
  if (sizes_match)
    printf("All communicators have the expected size of %d\n", num_gpus);

  printf("\n");

  // =========================================================================
  // STEP 4: Cleanup and Resource Management
  // =========================================================================

  // IMPORTANT: Proper cleanup is critical for NCCL applications
  // Resources must be cleaned up in the correct order to avoid issues

  // First, synchronize all streams to ensure no operations are in flight
  // This prevents destroying resources while they're still being used
  printf("Synchronizing all CUDA streams...\n");
  for (int i = 0; i < num_gpus; i++) {
    CUDACHECK(cudaSetDevice(devices[i]));
    CUDACHECK(cudaStreamSynchronize(streams[i]));
  }
  printf("All streams synchronized\n");

  // Next, destroy NCCL communicators first
  // This must be done before destroying CUDA resources they depend on
  printf("Destroying NCCL communicators...\n");
  for (int i = 0; i < num_gpus; i++) {
    NCCLCHECK(ncclCommFinalize(comms[i]));
    NCCLCHECK(ncclCommDestroy(comms[i]));
  }
  printf("All NCCL communicators destroyed\n");

  // Finally, destroy CUDA streams
  // This is safe now that the communicators are gone
  printf("Destroying CUDA streams...\n");
  for (int i = 0; i < num_gpus; i++) {
    CUDACHECK(cudaSetDevice(devices[i]));
    CUDACHECK(cudaStreamDestroy(streams[i]));
  }
  printf("All CUDA streams destroyed\n");

  // Free host memory allocations
  free(devices);
  free(comms);
  free(streams);

  printf("\n=============================================================\n");
  printf("SUCCESS: Multiple devices single process example completed!\n");
  printf("=============================================================\n\n");

  return 0;
}
