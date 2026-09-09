/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

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
 * NCCL Device API CFT Barrier Example
 *
 * This example demonstrates NCCL's Device API, which enables GPU kernels to
 * directly interact with NCCL without CPU intervention. This is particularly
 * powerful for applications that need to perform communication
 * from within CUDA kernels.
 *
 * Learning Objectives:
 *
 * Key Device API Concepts:
 * - ncclDevComm: Device-side communicator for kernel use
 * - CFT barriers: Synchronization primitives for device-side coordination
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
 */

// Device API kernel launch configuration
// CTA count must match cftBarrierCount for proper barrier synchronization
#define NCCL_DEVICE_CTA_COUNT 16
#define NCCL_DEVICE_THREADS_PER_CTA 512

// ==========================================================================
// Device Kernel Implementation
// ==========================================================================

// Device kernel that performs CFT barrier sync operation
// This kernel demonstrates direct NCCL synchronization from GPU threads
__global__ void simpleCftBarrierKernel(struct ncclDevComm comm) {
  // CFT barriers enable coordination between GPU threads across different ranks
  // Barrier scope: CTA (all threads in this block participate)
  // Barrier index: blockIdx.x selects this CTA's dedicated barrier (one barrier per CTA)
  ncclCftBarrierSession<ncclCoopCta> bar { ncclCoopCta(), comm, blockIdx.x };
  bar.sync(ncclCoopCta(), cuda::memory_order_relaxed, ncclMemProxyType::Fabric, ncclMemProxyType::Fabric);
}

__global__ void multimemCftBarrierKernel(struct ncclDevComm comm) {
  // Multimem CFT barriers enable coordination between GPU threads across different ranks
  // Barrier scope: CTA (all threads in this block participate)
  // Barrier index: blockIdx.x selects this CTA's dedicated barrier (one barrier per CTA)
  ncclCftBarrierSession<ncclCoopCta> bar { ncclCoopCta(), comm, blockIdx.x, /*multimem=*/true };
  bar.sync(ncclCoopCta(), cuda::memory_order_relaxed, ncclMemProxyType::Fabric, ncclMemProxyType::Fabric);
}

// ==========================================================================
// Host-Side Setup and Device API Initialization
// ==========================================================================

// This function can be called inside an MPI rank or pthread thread. The
// initialization and broadcast are implemented in common/src/utils.cc for
// easier readability. For fully integrated examples using pthreads or MPI see
// examples. in 01_communicators.

void* cftBarriers(int my_rank, int total_ranks, int local_device, int devices_per_rank) {
  ncclComm_t comm;
  ncclUniqueId nccl_unique_id;

  if (my_rank == 0) {
    printf("Starting Device API CFT Barrier initialization\n");
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
  // STEP 2: Initialize NCCL Communicator
  // ==========================================================================

  // Initialize NCCL communicator (same as Host API)
  NCCLCHECK(ncclCommInitRank(&comm, total_ranks, nccl_unique_id, my_rank));
  printf("  Rank %d initialized NCCL communicator for %d total ranks\n", my_rank, total_ranks);

  // Check for Device API support
  ncclCommProperties_t props = NCCL_COMM_PROPERTIES_INITIALIZER;
  NCCLCHECK(ncclCommQueryProperties(comm, &props));
  if (!props.deviceApiSupport) {
    printf("ERROR: rank %d communicator does not support Device API!\n", my_rank);
    NCCLCHECK(ncclCommFinalize(comm));
    NCCLCHECK(ncclCommDestroy(comm));
    return NULL;
  }

  // ==========================================================================
  // STEP 3: Create Device Communicator and Configure CFT Barriers
  // ==========================================================================

  // Create stream for kernel execution
  cudaStream_t stream;
  CUDACHECK(cudaStreamCreate(&stream));

  // Create device communicator - this is the key Device API component
  // Requirements specify resources to allocate (e.g., one barrier per CTA)
  ncclDevComm devComm;
  ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
  reqs.cftCaps = NCCL_CFT | NCCL_CFT_MULTIMEM;
  reqs.cftBarrierCount = NCCL_DEVICE_CTA_COUNT;
  NCCLCHECK(ncclDevCommCreate(comm, &reqs, &devComm));
  printf("  Rank %d created device communicator with %d CFT barriers\n", my_rank, NCCL_DEVICE_CTA_COUNT);

  // ==========================================================================
  // STEP 4: Launch Device Kernel for CFT Barrier Operation
  // ==========================================================================

  // Launch device kernel to perform CFT Barrier
  // This kernel will directly access peer memory and perform collective operation
  simpleCftBarrierKernel<<<NCCL_DEVICE_CTA_COUNT, NCCL_DEVICE_THREADS_PER_CTA, 0, stream>>>(devComm);
  multimemCftBarrierKernel<<<NCCL_DEVICE_CTA_COUNT, NCCL_DEVICE_THREADS_PER_CTA, 0, stream>>>(devComm);

  // Wait for both barrier kernels to complete.
  CUDACHECK(cudaStreamSynchronize(stream));
  printf("  Rank %d completed CFT Barrier kernel execution\n", my_rank);

  // ==========================================================================
  // STEP 5: Cleanup Resources
  // ==========================================================================

  // Device API specific cleanup
  NCCLCHECK(ncclDevCommDestroy(comm, &devComm));

  // Standard NCCL cleanup
  NCCLCHECK(ncclCommFinalize(comm));
  NCCLCHECK(ncclCommDestroy(comm));
  CUDACHECK(cudaStreamDestroy(stream));

  return NULL;
}

int main(int argc, char* argv[]) {
  // Run example using the provided utility framework
  return run_example(argc, argv, cftBarriers);
}
