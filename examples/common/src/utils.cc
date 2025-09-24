/*************************************************************************
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "utils.h"
#include <unistd.h>

#ifndef MPI_SUPPORT
pthread_barrier_t barrier;
ncclUniqueId nccl_unique_id;
#endif

/**
 * Common context structure for both MPI and pthread examples
 *
 * This structure provides a unified interface for NCCL examples that can
 * run in either MPI mode (one process per device) or pthread mode
 * (one thread per device).
 */
typedef struct {
  // Common variables
  int total_ranks;      // Total number of MPI ranks or pthreads
  int devices_per_rank; // Number of devices per rank or thread
  int local_device;     // Node local rank or thread id (0 to total_ranks-1)
  int my_rank;          // Rank or thread id for NCCL
  ncclUniqueId nccl_id; // NCCL unique ID
  ncclComm_t comm;      // NCCL communicator (single for all modes)
  ncclUniqueId *nccl_unique_id; // NCCL unique ID pointer
  void *func;

  // pthread-specific variables
#ifndef MPI_SUPPORT
  pthread_t *threads; // Thread array
  int *thread_ranks;  // Thread rank array
#endif
} context_t;

/**
 * Initialize MPI or pthread backend
 *
 * Sets up the backend and populates common context variables.
 *
 * MPI Mode (compiled with MPI=1):
 *   - Initializes MPI (rank, size)
 *   - Calculates local rank based on splitting communicator by node multi-node
 * support
 *   - Generates and broadcasts NCCL unique ID
 *   - Sets device assignment based on local rank
 *
 * pthread Mode (default):
 *   - Gets thread count from NTHREADS environment or GPU count
 *   - Validates thread count against available GPUs
 *   - Generates NCCL unique ID for sharing across threads
 *   - Allocates thread management resources
 *
 * @param argc      Command line argument count
 * @param argv      Command line arguments
 * @param ctx       Output: Populated example context
 *
 * @return 0 on success, non-zero on error
 */
int initialize(int argc, char *argv[], context_t *ctx);

/**
 * Wrap function to call the example function in a thread
 *
 * Note: This function is needed since pthread only allows a single void*
 * argument.
 */
void *thread_wrapper(void *arg);

/**
 * Run ncclExample in parallel using MPI or pthreads
 *
 * Starts the execution of the given NCCL example function in parallel
 *
 * MPI Mode:
 *   - Starts the function on each rank
 *   - Checks the output and calls MPI_Barrier to synchronize
 *
 * pthread Mode:
 *   - Creates threads (one per device)
 *   - Each thread runs the given example function
 *   - Waits for all threads to complete
 *
 * @param ctx               Context with backend setup completed
 * @param ncclExample       Function pointer to example-specific NCCL setup
 *
 * @return 0 on success, non-zero on error
 *
 * Note: This function expects ncclExample() to be defined by the example.
 * The ncclExample function should have signature:
 * void* ncclExample(int, int, int, int)
 */
int run_parallel(context_t *ctx, void *(*ncclExample)(int, int, int, int));

/**
 * Clean up resources
 *
 * Properly cleans up all resources allocated during initialization.
 * Note: NCCL communicators are destroyed by ncclCommSetup function.
 *
 * MPI Mode:
 *   - Finalizes MPI
 *
 * pthread Mode:
 *   - Frees thread arrays
 *
 * @param ctx       Context to clean up
 */
void cleanup(context_t *ctx);

/**
 * Broadcast NCCL unique ID
 */
int util_broadcast(int root, int my_rank, ncclUniqueId *arg) {
#ifdef MPI_SUPPORT
  MPICHECK(
      MPI_Bcast(arg, sizeof(ncclUniqueId), MPI_BYTE, root, MPI_COMM_WORLD));
#else
  if (my_rank == root) {
    nccl_unique_id = *arg;
  }
  int barrier_err = pthread_barrier_wait(&barrier);
  if (barrier_err != 0 && barrier_err != PTHREAD_BARRIER_SERIAL_THREAD) {
    fprintf(stderr, "pthread_barrier_wait failed at %s:%d with error code %d\n",
            __FILE__, __LINE__, barrier_err);
    abort();
  }
  if (my_rank != root) {
    *arg = nccl_unique_id;
  }
#endif
  return 0;
}

/**
 * Initialize MPI or pthread backend
 */
int initialize(int argc, char *argv[], context_t *ctx) {
#ifdef MPI_SUPPORT
  // Initialize MPI
  MPICHECK(MPI_Init(&argc, &argv));
  MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &ctx->my_rank));
  MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &ctx->total_ranks));

  if (ctx->my_rank == 0) {
    printf("Number of processes: %d\n", ctx->total_ranks);
  }
  // Only for printing the output in order
  MPI_Barrier(MPI_COMM_WORLD);
  printf("MPI initialized: rank %d of %d\n", ctx->my_rank, ctx->total_ranks);

  // Split the communicator based on shared memory (i.e., nodes)
  MPI_Comm node_comm;
  MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, ctx->my_rank,
                      MPI_INFO_NULL, &node_comm);

  // Get the rank within the node communicator
  MPI_Comm_rank(node_comm, &ctx->local_device);

  // Clean up the node communicator
  MPI_Comm_free(&node_comm);

#else

  // Get number of devices (threads) from environment or default to available
  // GPUs
  int num_gpus = 0;
  CUDACHECK(cudaGetDeviceCount(&num_gpus));
  ctx->total_ranks = num_gpus; // Default to all available GPUs
  const char *nThreadsEnv = getenv("NTHREADS");
  if (nThreadsEnv) {
    ctx->total_ranks = atoi(nThreadsEnv);
  }

  printf("Creating %d threads for %d devices\n", ctx->total_ranks, num_gpus);

  if (ctx->total_ranks < 1) {
    printf("Invalid number of threads: %d\n", ctx->total_ranks);
    return 1;
  }

  // Check if we have enough GPUs
  if (ctx->total_ranks > num_gpus) {
    printf("Error: Requested %d threads but only %d GPUs available\n",
           ctx->total_ranks, num_gpus);
    printf("Please reduce NTHREADS to %d or fewer\n", num_gpus);
    return 1;
  }

  // Thread synchronization needed for unique ID sharing later on
  pthread_barrier_init(&barrier, NULL, ctx->total_ranks);

  // Generate NCCL unique ID (shared across all threads)
  NCCLCHECK(ncclGetUniqueId(&ctx->nccl_id));

  // Allocate thread resources
  ctx->threads = (pthread_t *)malloc(ctx->total_ranks * sizeof(pthread_t));
  ctx->thread_ranks = (int *)malloc(ctx->total_ranks * sizeof(int));
  if (ctx->threads == NULL || ctx->thread_ranks == NULL) {
    printf("Failed to allocate memory for threads\n");
    return 1;
  }
#endif

  return 0;
}

/**
 * Wrap function to call the example function in a thread
 *
 * Note: This function is needed since pthread only allows a single void*
 * argument.
 */
void *thread_wrapper(void *arg) {
  context_t *ctx = (context_t *)arg;
  void *(*example_func)(int, int, int, int) =
      (void *(*)(int, int, int, int))ctx->func;
  return example_func(ctx->my_rank, ctx->total_ranks, ctx->local_device,
                      ctx->devices_per_rank);
}

/**
 * Run ncclExample in parallel using MPI or pthreads
 */
int run_parallel(context_t *ctx, void *(*ncclExample)(int, int, int, int)) {
#ifdef MPI_SUPPORT
  if (ctx->my_rank == 0) {
    printf("NCCL Example: One Device per Process\n");
    printf("====================================\n");
  }

  if (ncclExample(ctx->my_rank, ctx->total_ranks, ctx->local_device,
                  ctx->devices_per_rank) != NULL)
    return 1;
  // Synchronize to ensure ordered output
  MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
#else
  printf("NCCL Example: One Device per Thread\n");
  printf("===================================\n");

  // Create separate context for each thread
  context_t *thread_contexts =
      (context_t *)malloc(ctx->total_ranks * sizeof(context_t));
  if (thread_contexts == NULL) {
    printf("Failed to allocate thread contexts\n");
    return 1;
  }
  ncclUniqueId *nccl_unique_id =
      (ncclUniqueId *)calloc(1, sizeof(ncclUniqueId));

  for (int i = 0; i < ctx->total_ranks; i++) {
    // Copy main context to thread context
    memcpy(&thread_contexts[i], ctx, sizeof(context_t));
    thread_contexts[i].threads = NULL;
    thread_contexts[i].thread_ranks = NULL;
    thread_contexts[i].my_rank = i; // Set NCCL rank to thread id
    thread_contexts[i].local_device = i;
    thread_contexts[i].total_ranks = ctx->total_ranks;
    thread_contexts[i].devices_per_rank = 1;
    thread_contexts[i].func = (void *)ncclExample;
    thread_contexts[i].nccl_unique_id = nccl_unique_id;
    ctx->thread_ranks[i] = i;
    pthread_create(&ctx->threads[i], NULL, thread_wrapper, &thread_contexts[i]);
  }

  // Wait for all threads to complete
  for (int i = 0; i < ctx->total_ranks; i++) {
    pthread_join(ctx->threads[i], NULL);
  }

  free(thread_contexts);
#endif

  return 0;
}

/**
 * Run the given NCCL example in parallel
 */
int run_example(int argc, char *argv[],
                void *(*ncclExample)(int, int, int, int)) {

  // 1. Allocate context
  context_t *ctx = (context_t *)calloc(1, sizeof(context_t));
  if (ctx == NULL) {
    printf("Failed to allocate memory for context\n");
    return 1;
  }

  // 2. Initialize backend (MPI or pthread)
  if (initialize(argc, argv, ctx) != 0) {
    printf("Failed to initialize backend\n");
    return 1;
  }

  // 3. Start the given example code in parallel
  if (run_parallel(ctx, ncclExample) != 0) {
    printf("Failed to execute NCCL operations\n");
    cleanup(ctx); // Cleanup on failure
    return 1;
  }

  // 3. Cleanup
  cleanup(ctx);

  // 4. Print common success message
#ifdef MPI_SUPPORT
  if (ctx->my_rank == 0) {
#endif
    printf("\nAll NCCL communicators finalized successfully!\n");
#ifdef MPI_SUPPORT
  }
#endif

  return 0;
}

/**
 * Clean up resources
 */
void cleanup(context_t *ctx) {
#ifdef MPI_SUPPORT
  // Free MPI resources
  MPICHECK(MPI_Finalize());
#else
  free(ctx->threads);
  free(ctx->thread_ranks);
  pthread_barrier_destroy(&barrier);
#endif
}
