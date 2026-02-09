/*************************************************************************
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef UTILS_H_
#define UTILS_H_

#include "cuda_runtime.h"
#include "nccl.h"
#include "nccl_utils.h"
#include <stdio.h>
#include <stdlib.h>

#ifdef MPI_SUPPORT
#include "mpi.h"
#include "mpi_utils.h"
#else
#include <pthread.h>
#include <unistd.h>
#endif

/**
 * Broadcast NCCL unique ID
 *
 * Broadcasts the NCCL unique ID from the root rank to all other ranks.
 * Uses MPI_Bcast in MPI mode and pthread barrier in pthread mode.
 *
 * @param root      Root rank that holds the NCCL unique ID
 * @param my_rank   Current rank or thread id
 * @param arg       Pointer to NCCL unique ID to broadcast
 *
 * @return 0 on success, non-zero on error
 */
int util_broadcast(int root, int my_rank, ncclUniqueId *arg);

/**
 * Run the given NCCL example in parallel
 *
 * This function performs the complete NCCL example lifecycle:
 * 1. Initialize backend (MPI or pthread)
 * 2. Execute NCCL communicator setup function
 * 3. Cleanup of all resources
 *
 * @param argc              Command line argument count
 * @param argv              Command line arguments
 * @param ncclExample       Function pointer to example-specific NCCL setup
 *
 * @return 0 on success, non-zero on error
 */
int run_example(int argc, char *argv[],
                void *(*ncclExample)(int, int, int, int));

#endif // UTILS_H_
