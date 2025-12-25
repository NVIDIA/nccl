/*************************************************************************
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "cuda_runtime.h"
#include "nccl.h"
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

/**
 * NCCL Pthread Example - One Device Per Thread (Simple Version)
 *
 * This example demonstrates the basic lifecycle of NCCL communicators in a
 * multi-threaded environment. Each pthread manages one GPU device and shows
 * how to properly create and destroy NCCL communicators.
 *
 * Key Learning Points:
 * - NCCL communicator creation and destruction within threads
 * - CUDA stream management per thread
 * - Proper resource cleanup order
 *
 * This is a minimal example focusing purely on communicator lifecycle
 * management without performing actual collective operations.
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

// Thread data structure to pass parameters
typedef struct {
  int thread_id;
  int num_gpus;
  ncclUniqueId commId;
  ncclComm_t *comms;
} threadData_t;

void *thread_worker(void *arg) {
  threadData_t *data = (threadData_t *)arg;
  int thread_id = data->thread_id;
  cudaStream_t stream;

  // =========================================================================
  // Set Device Context and Create Stream
  // =========================================================================
  // Each thread must set its device context before any CUDA operations
  CUDACHECK(cudaSetDevice(thread_id));
  CUDACHECK(cudaStreamCreate(&stream));

  printf("  Thread %d: Set device %d and created stream\n", thread_id,
         thread_id);

  // =========================================================================
  // Initialize NCCL Communicator
  // =========================================================================
  // Each thread creates its own communicator using the shared unique ID
  NCCLCHECK(ncclCommInitRank(&data->comms[thread_id], data->num_gpus, data->commId,
                             thread_id));

  printf("  Thread %d: NCCL communicator initialized\n", thread_id);

  if (thread_id == 0) {
    printf("All threads initialized - communicators ready\n");
  }

  // =========================================================================
  // Query Communicator Properties
  // =========================================================================
  // Verify the communicator was created correctly
  int comm_thread_id, comm_size;
  NCCLCHECK(ncclCommUserRank(data->comms[thread_id], &comm_thread_id));
  NCCLCHECK(ncclCommCount(data->comms[thread_id], &comm_size));

  printf("  Thread %d: Communicator thread_id %d of %d\n", thread_id,
         comm_thread_id, comm_size);

  // Synchronize CUDA stream to ensure all GPU work is complete
  if (stream != NULL) {
    CUDACHECK(cudaStreamSynchronize(stream));
  }

  // =========================================================================
  // Cleanup Resources (Proper Order)
  // =========================================================================
  // Destroy NCCL communicator FIRST (before CUDA resources)
  // This is important - NCCL cleanup should happen before CUDA cleanup
  if (data->comms[thread_id] != NULL) {
    NCCLCHECK(ncclCommFinalize(data->comms[thread_id]));
    NCCLCHECK(ncclCommDestroy(data->comms[thread_id]));
    printf("  Thread %d: Destroyed NCCL communicator\n", comm_thread_id);
  }

  // Now destroy CUDA stream
  if (stream != NULL) {
    CUDACHECK(cudaStreamDestroy(stream));
  }

  printf("  Thread %d: Resources cleaned up\n", thread_id);

  return NULL;
}

int main(int argc, char *argv[]) {
  int num_gpus;
  pthread_t *threads;
  threadData_t *threadData;
  ncclComm_t *comms;
  ncclUniqueId commId;

  // =========================================================================
  // STEP 1: Initialize Variables and Check GPU Availability
  // =========================================================================

  CUDACHECK(cudaGetDeviceCount(&num_gpus));
  const char *nThreadsEnv = getenv("NTHREADS");
  if (nThreadsEnv) {
    num_gpus = atoi(nThreadsEnv);
  }

  if (num_gpus < 1) {
    printf("No CUDA devices found\n");
    return EXIT_FAILURE;
  }

  printf("Using %d devices with pthreads\n", num_gpus);

  // =========================================================================
  // STEP 2: Allocate Memory and Prepare Data Structures
  // =========================================================================

  threads = (pthread_t *)malloc(num_gpus * sizeof(pthread_t));
  threadData = (threadData_t *)malloc(num_gpus * sizeof(threadData_t));
  comms = (ncclComm_t *)malloc(num_gpus * sizeof(ncclComm_t));

  // Generate unique ID for NCCL communicator initialization
  NCCLCHECK(ncclGetUniqueId(&commId));

  // =========================================================================
  // STEP 3: Create and Launch Pthread Threads
  // =========================================================================

  printf("Creating %d threads for NCCL communicators\n", num_gpus);

  for (int i = 0; i < num_gpus; i++) {
    threadData[i].thread_id = i;
    threadData[i].num_gpus = num_gpus;
    threadData[i].commId = commId;
    threadData[i].comms = comms;

    pthread_create(&threads[i], NULL, thread_worker, &threadData[i]);
  }

  // =========================================================================
  // STEP 4: Wait for Thread Completion
  // =========================================================================

  for (int i = 0; i < num_gpus; i++) {
    pthread_join(threads[i], NULL);
  }

  printf("All threads completed\n");

  // =========================================================================
  // STEP 5: Cleanup Resources
  // =========================================================================

  free(threads);
  free(threadData);
  free(comms);

  printf("Success\n");
  return 0;
}
