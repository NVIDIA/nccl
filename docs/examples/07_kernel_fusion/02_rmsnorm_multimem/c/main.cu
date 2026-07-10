/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

/*
 * RMSNorm with Multimem
 *
 * This example demonstrates fused computation and communication for distributed
 * RMSNorm using NCCL's Load Store Accessible (LSA) mechanism, showcasing
 * Multimem capabilities. LSA enables direct peer-to-peer memory access
 * between GPUs on the same node via NVLink.
 *
 * The kernel performs three phases in a single launch:
 *   1. Reduce-Scatter: Gather and sum partial results from all GPUs
 *   2. RMS Normalization: Compute and apply RMSNorm to each token
 *   3. All-Gather: Broadcast normalized results back to all GPUs
 */
#include "utils.h"
#include "rmsnorm_utils.cuh"
#include "nccl_device.h"

// Kernel: Fused RMSNorm with Multimem (LSA) Communication
//==============================================================================
/*
 * Performs distributed RMSNorm using LSA for peer-to-peer communication.
 * This example serves as a Multimem showcase.
 *
 * Grid Configuration:
 *   - Grid dimension: tokens_per_gpu (one block per token)
 *   - Block dimension: threads_per_block (typically 256)
 *   - Shared memory: threads_per_block floats for block-level reduction
 *
 * Memory Layout:
 *   - window: Symmetric LSA window containing all tokens (tokens x hidden dimension)
 *   - Results are stored in-place using ncclGetLocalPointer()
 *
 * Synchronization:
 *   - Uses LSA barriers for fine-grained GPU synchronization
 *   - Each block has its own barrier instance (indexed by blockIdx.x)
 *
 * Parameters:
 *   window          - LSA window for peer memory access
 *   devComm         - NCCL device communicator
 *   sequence_length - Total number of tokens across all GPUs
 *   hidden_dim      - Dimensionality of each token
 *   eps             - Epsilon for numerical stability
 */
__global__ void RMSNormMultimem(ncclWindow_t window, ncclDevComm devComm,
                               const int sequence_length, const int hidden_dim, const float eps) {

  // Shared memory buffer for reduction
  extern __shared__ float reduction_buffer[];

  ncclCoopCta coop = ncclCoopCta();

  // Initialize LSA barrier session for this block
  ncclLsaBarrierSession<ncclCoopCta> bar {
    coop, devComm, ncclTeamTagLsa(), blockIdx.x, /*multimem=*/true
  };

  // Initial synchronization across all GPUs
  bar.sync(coop, cuda::memory_order_acquire);

  //----------------------------------------------------------------------------
  // Calculate offsets for this block's token
  //----------------------------------------------------------------------------
  const int rank = devComm.rank;
  const int token_idx = rank * gridDim.x + blockIdx.x;  // Global token index
  const int window_offset = token_idx * hidden_dim * sizeof(float);
  float* local_pointer = (float*)ncclGetLocalPointer(window, window_offset);
  float* multimem_pointer = reinterpret_cast<float*>(ncclGetLsaMultimemPointer(window, window_offset, devComm));

  //============================================================================
  // Phase 1: Reduce-Scatter via Multimem
  //============================================================================
  // Load and sum from all peers. Each rank only needs the reduced result for its
  // own tokens, so we store to local only (no multimemStore here).
  for (int i = threadIdx.x; i < hidden_dim; i += blockDim.x) {
    local_pointer[i] = multimemLoadSum(multimem_pointer + i);
  }

  coop.sync();

  //============================================================================
  // Phase 2: RMS Normalization
  //============================================================================
  // Normalize the reduced token data using block-level RMSNorm.
  // All threads in the block collaborate to:
  //   1. Compute RMS = sqrt(mean(x^2) + eps)
  //   2. Apply normalization: x_i = x_i / RMS
  //----------------------------------------------------------------------------
  blockRMSNorm(local_pointer, hidden_dim, eps, reduction_buffer, coop);

  //============================================================================
  // Phase 3: All-Gather via Multimem
  //============================================================================
  // Broadcast the normalized token back to all GPUs. Each thread writes a
  // subset of dimensions to all peer GPUs using Multimem.
  for (int i = threadIdx.x; i < hidden_dim; i += blockDim.x) {
    float val = local_pointer[i];  // Read once from local
    multimemStore(multimem_pointer + i, val);
  }

  // Final barrier with release semantics to ensure visibility
  bar.sync(coop, cuda::memory_order_release);
}

//==============================================================================
// Host Function: RMSNorm Example Entry Point
//==============================================================================
void *rms_norm(int my_rank, int total_ranks, int local_device, int devices_per_rank) {
  //----------------------------------------------------------------------------
  // Configuration Parameters
  //----------------------------------------------------------------------------
  const float eps = 1e-6f;                    // Epsilon for numerical stability
  const int hidden_size = 1024;               // Hidden dimension per token
  const int threads_per_block = 256;          // Threads per CUDA block
  const int sequence_length = 4096;           // Total tokens (all GPUs)

  // Derived parameters
  const size_t tensor_size = sequence_length * hidden_size;
  const size_t tensor_size_bytes = tensor_size * sizeof(float);
  const int tokens_per_gpu = sequence_length / total_ranks;

  //----------------------------------------------------------------------------
  // Validate Configuration
  //----------------------------------------------------------------------------
  if (sequence_length % total_ranks != 0) {
    if (my_rank == 0) {
      fprintf(stderr, "ERROR: sequence_length (%d) must be divisible by number of ranks (%d)\n",
              sequence_length, total_ranks);
      fprintf(stderr, "       Each rank must process an equal number of tokens.\n");
      fprintf(stderr, "       sequence_length %% total_ranks = %d (must be 0)\n",
              sequence_length % total_ranks);
    }
    return NULL;
  }

  //============================================================================
  // Step 1: Initialize NCCL Communicator
  //============================================================================
  ncclComm_t comm;
  ncclUniqueId nccl_id;

  // Rank 0 generates unique ID and broadcasts to all ranks
  if (my_rank == 0) {
    NCCLCHECK(ncclGetUniqueId(&nccl_id));
  }
  util_broadcast(0, my_rank, &nccl_id);

  // Set device and initialize communicator
  CUDACHECK(cudaSetDevice(local_device));

  // Check compute capability (Multimem requires SM 9.0+)
  cudaDeviceProp prop;
  CUDACHECK(cudaGetDeviceProperties(&prop, local_device));
  if (prop.major < 9) {
    fprintf(stderr, "ERROR: Multimem requires GPU compute capability 9.0 or higher.\n");
    fprintf(stderr, "       Rank %d GPU: %s has compute capability %d.%d\n",
            my_rank, prop.name, prop.major, prop.minor);
    return NULL;
  }

  NCCLCHECK(ncclCommInitRank(&comm, total_ranks, nccl_id, my_rank));

  // Check for Device API support
  ncclCommProperties_t props = NCCL_COMM_PROPERTIES_INITIALIZER;
  NCCLCHECK(ncclCommQueryProperties(comm, &props));
  if (!props.deviceApiSupport) {
    printf("ERROR: rank %d communicator does not support Device API!\n", my_rank);
    NCCLCHECK(ncclCommFinalize(comm));
    NCCLCHECK(ncclCommDestroy(comm));
    return NULL;
  }
  if (!props.multimemSupport) {
    printf("ERROR: rank %d communicator does not support Multimem!\n", my_rank);
    NCCLCHECK(ncclCommFinalize(comm));
    NCCLCHECK(ncclCommDestroy(comm));
    return NULL;
  }
  // Multimem example requires a single LSA team where all ranks can directly access each other
  if (props.nLsaTeams != 1) {
    printf("ERROR: rank %d communicator has %d LSA teams, expected 1 for Multimem example!\n",
           my_rank, props.nLsaTeams);
    NCCLCHECK(ncclCommFinalize(comm));
    NCCLCHECK(ncclCommDestroy(comm));
    return NULL;
  }

  //============================================================================
  // Step 2: Allocate Host Memory and Initialize Data
  //============================================================================
  float *h_data = (float*)malloc(tensor_size_bytes);
  float *h_data_validation = (float*)malloc(tensor_size_bytes);

  if (!h_data || !h_data_validation) {
    fprintf(stderr, "ERROR: Failed to allocate host memory\n");
    NCCLCHECK(ncclCommFinalize(comm));
    NCCLCHECK(ncclCommDestroy(comm));
    return NULL;
  }
  initialize_data(h_data, tensor_size, my_rank);

  //============================================================================
  // Step 3: Allocate Device Memory and Register LSA Window
  //============================================================================
  float *d_data;
  ncclWindow_t window;
  cudaStream_t stream;

  // Allocate and register symmetric window for LSA access
  NCCLCHECK(ncclMemAlloc((void **)&d_data, tensor_size_bytes));
  NCCLCHECK(ncclCommWindowRegister(comm, d_data, tensor_size_bytes, &window, NCCL_WIN_COLL_SYMMETRIC));

  // Copy input data to device and create stream
  CUDACHECK(cudaMemcpy(d_data, h_data, tensor_size_bytes, cudaMemcpyHostToDevice));
  CUDACHECK(cudaStreamCreate(&stream));

  //============================================================================
  // Step 4: Create Device Communicator with LSA Barriers
  //============================================================================
  ncclDevComm devComm;
  ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
  reqs.lsaBarrierCount = tokens_per_gpu;  // One barrier per token
  reqs.lsaMultimem = true; // Enable multimem support
  NCCLCHECK(ncclDevCommCreate(comm, &reqs, &devComm));

  if (my_rank == 0) {
    printf("\n========================================\n");
    printf("Starting RMSNorm Multimem Kernel\n");
    printf("========================================\n");
    printf("Configuration:\n");
    printf("  - Total ranks:      %d\n", total_ranks);
    printf("  - Sequence length:  %d tokens\n", sequence_length);
    printf("  - Hidden size:      %d\n", hidden_size);
    printf("  - Tokens per GPU:   %d\n", tokens_per_gpu);
    printf("  - Threads per block: %d\n", threads_per_block);
    printf("========================================\n\n");
  }

  //============================================================================
  // Step 5: Launch Fused RMSNorm Kernel
  //============================================================================
  // Calculate shared memory size: one float per thread in the block
  const size_t shared_mem_size = threads_per_block * sizeof(float);

  RMSNormMultimem<<<tokens_per_gpu, threads_per_block, shared_mem_size, stream>>>(
      window, devComm, sequence_length, hidden_size, eps);
  CUDACHECK(cudaGetLastError());
  CUDACHECK(cudaStreamSynchronize(stream));

  if (my_rank == 0) {
    printf("Kernel execution completed\n\n");
  }

  //============================================================================
  // Step 6: Verify Results Against CPU Reference
  //============================================================================
  // Generate expected results on CPU
  rms_norm_generate(h_data, tensor_size, sequence_length, hidden_size, total_ranks, eps);

  if (my_rank == 0) {
    printf("Verifying results...\n");
  }

  // Copy GPU results back to host
  CUDACHECK(cudaMemcpy(h_data_validation, d_data, tensor_size_bytes, cudaMemcpyDeviceToHost));

  // Compare against expected results
  bool success = verify_results(h_data_validation, h_data, tensor_size);

  //============================================================================
  // Step 7: Cleanup Resources
  //============================================================================
  NCCLCHECK(ncclDevCommDestroy(comm, &devComm));
  NCCLCHECK(ncclCommWindowDeregister(comm, window));
  NCCLCHECK(ncclMemFree(d_data));
  NCCLCHECK(ncclCommFinalize(comm));
  NCCLCHECK(ncclCommDestroy(comm));
  CUDACHECK(cudaStreamDestroy(stream));

  free(h_data);
  free(h_data_validation);

  //============================================================================
  // Step 8: Report Results
  //============================================================================
  if (success) {
    if (my_rank == 0) {
      printf("\n========================================\n");
      printf("SUCCESS: Example completed successfully!\n");
      printf("========================================\n\n");
    }
  } else {
    printf("FAILED: Incorrect results detected on rank %d\n", my_rank);
    return NULL;
  }

  return NULL;
}

//==============================================================================
// Main Entry Point
//==============================================================================
int main(int argc, char* argv[]) {
  return run_example(argc, argv, rms_norm);
}
