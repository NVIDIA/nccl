/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

/*
 * RMSNorm with GPU-Initiated Networking (GIN)
 *
 * This example demonstrates fused computation and communication for distributed
 * RMSNorm using NCCL's GPU-Initiated Networking (GIN) mechanism. GIN enables
 * GPU kernels to directly initiate network communication without CPU involvement,
 * supporting both intra-node and inter-node communication.
 *
 * The kernel performs three phases in a single launch:
 *   1. Reduce-Scatter: Gather and sum partial results using GIN PUT
 *   2. RMS Normalization: Compute and apply RMSNorm to each token
 *   3. All-Gather: Broadcast normalized results using GIN PUT
 */

#include "utils.h"
#include "rmsnorm_utils.cuh"
#include "nccl_device.h"

// Kernel: Fused RMSNorm with GIN Communication
//==============================================================================
/*
 * Performs distributed RMSNorm using GIN for network-based communication.
 *
 * Grid Configuration:
 *   - Grid dimension: tokens_per_gpu (one block per token)
 *   - Block dimension: threads_per_block (typically 256)
 *   - Shared memory: threads_per_block floats for block-level reduction
 *
 * Memory Layout:
 *   - window_send: Source window containing input data (tokens x hidden dimension)
 *   - window_recv: Receive window for gathering contributions (tokens x hidden dimension)
 *     The receive window is structured to hold contributions from all peers
 *     in a strided layout: [peer0_data, peer1_data, ..., peerN_data]
 *
 * Communication Pattern:
 *   - Phase 1: One thread block per token on this rank. Each block exchanges partial
 *     activations so every rank's receive window holds all peers' contributions for
 *     the tokens it owns, then reduces (sums) in place to the full activation.
 *   - Phase 3: Each GPU broadcasts normalized results back to all peers using
 *              PUT (gin.put) to update their window_send buffers
 *
 * Synchronization:
 *   - Each PUT (gin.put) uses ncclGin_WeakSignalInc on the peer, which adds one
 *     completion increment to that peer's local signal; waitSignal/readSignal on a rank therefore count inbound signaled PUTs
 *     from others. For outbound PUTs, gin.flush ensures this rank's pending GIN work has
 *     locally consumed its source buffers so those regions (e.g. slices of window_send /
 *     window_recv used as PUT sources) can be safely overwritten or reused.
 *   - Phase 1: gin.waitSignal waits until every peer has finished PUT of the partial
 *     contribution for token_idx (the token owned by this rank and block) into this rank's
 *     window_recv—one remote completion signal per peer before we reduce.
 *   - Phase 3: the second wait is the same idea for all-gather: every peer has finished
 *     PUT of the normalized slice this block needs from them into this rank's window_send.
 *   - ncclGinBarrierSession (world team): acquire at kernel entry; release before
 *     Phase 3 and after Phase 3 for collective phase ordering.
 *   - Signal values accumulate across phases (no reset between phases)
 *
 * Parameters:
 *   window_send     - Window for sending data (source and final destination)
 *   window_recv     - Window for receiving data (intermediate staging)
 *   devComm         - NCCL device communicator with GIN context
 *   sequence_length - Total number of tokens across all GPUs
 *   hidden_dim      - Dimensionality of each token
 *   eps             - Epsilon for numerical stability
 */
__global__ void RMSNormGIN(ncclWindow_t window_send, ncclWindow_t window_recv, ncclDevComm devComm,
                           const int sequence_length, const int hidden_dim, const float eps) {

  // Shared memory buffer for reduction
  extern __shared__ float reduction_buffer[];

  ncclCoopCta coop = ncclCoopCta();

  //----------------------------------------------------------------------------
  // Initialize GIN context and synchronization primitives
  //----------------------------------------------------------------------------
  // Use multiple GIN contexts to spread blocks across network communication channels.
  // This improves parallel communication performance by utilizing all available contexts.
  int ginContext = blockIdx.x % devComm.ginContextCount;
  unsigned int signalIndex = blockIdx.x;  // Each block uses its own signal
  ncclGin gin { devComm, ginContext };
  uint64_t signalValue = gin.readSignal(signalIndex);

  const int rank = devComm.rank;
  const int nRanks = devComm.nRanks;
  const int token_idx = rank * gridDim.x + blockIdx.x;  // Global token index

  ncclGinBarrierSession<ncclCoopCta> bar { coop, gin, ncclTeamTagWorld(), blockIdx.x };
  bar.sync(coop, cuda::memory_order_acquire, ncclGinFenceLevel::None);

  //============================================================================
  // Phase 1: Reduce-Scatter via GIN PUT
  //============================================================================
  // This block owns one token: token_idx = rank * gridDim.x + blockIdx.x.
  // (1) Exchange: for each peer, a PUT (gin.put) sends this rank's partial for the slice
  //     that peer needs; symmetrically we receive every peer's contribution for
  //     token_idx into window_recv (strided layout).
  // (2) waitSignal waits for inbound signaled PUT payloads before reduction.
  //     gin.flush ensures our Phase 1 PUT sources in window_send are safe to reuse.
  //
  // Receive layout (peer-major in each rank's window): for peer p = 0..nRanks-1,
  // the H*B floats for that peer are [token0][token1]...[token_{B-1}], each token
  // having H dims. Flat offset from buffer start: p*H*B + b*H + j for peer p,
  // local token index b, dimension j. This block (b = blockIdx.x) uses my_token_data
  // at float offset b*H; peer p contributes at j + p*H*B relative to that pointer.
  //
  // Threading: each thread issues PUTs (gin.put) to a subset of peers (stride blockDim.x).
  //
  // Synchronization: gin.waitSignal(signalValue + nRanks) waits until all peers have
  // finished PUT of their partial contribution for token_idx—the token this rank and
  // block own—into this rank's window_recv (each peer's PUT carries a completion increment on this
  // signalIndex, so the local counter rises by nRanks).
  //----------------------------------------------------------------------------

  size_t my_window_offset = (token_idx * hidden_dim) * sizeof(float);

  for (int peer = threadIdx.x; peer < nRanks; peer += blockDim.x) {
    const int peer_token_idx = peer * gridDim.x + blockIdx.x;
    size_t peer_window_offset = (peer_token_idx * hidden_dim) * sizeof(float);

    // PUT: send our token data to peer's receive window
    gin.put(ncclTeamWorld(devComm), peer, window_recv, my_window_offset,
            window_send, peer_window_offset, sizeof(float) * hidden_dim,
            ncclGin_WeakSignalInc{signalIndex});
  }

  // Wait until every peer has completed its signaled PUT delivering its contribution for
  // token_idx into this rank's window_recv (reduce-scatter gather for this block's token).
  gin.waitSignal(coop, signalIndex, signalValue + devComm.nRanks);
  // Flush completes our outbound Phase 1 PUTs locally so window_send regions used as
  // PUT sources can be safely reused by subsequent operations on this rank.
  gin.flush(coop);

  //----------------------------------------------------------------------------
  // Reduction: Sum contributions from all peers
  //----------------------------------------------------------------------------
  // After the reduce-scatter phase, the receive window contains contributions
  // from all peers in a strided layout. We need to sum across the peer dimension.
  //
  // Contiguous layout in window_recv: token0_peer0, token1_peer0, token2_peer0, ...,
  // token_{B-1}_peer0, then token0_peer1, token1_peer1, token2_peer1, ... (peer changes
  // after each block of B*H floats). Reduction uses peer 0's row as accumulator.
  //
  // Each thread reduces a subset of dimensions, summing across all peer contributions.

  float *my_token_data = (float*)ncclGetLocalPointer(window_recv, blockIdx.x * hidden_dim * sizeof(float));


  // Sum across all peer contributions for this dimension, using the first row of the
  // corresponding token as a buffer, therefore skip index 0 to avoid adding twice
  for (int peer = 1; peer < nRanks; peer++) {
    for (int j = threadIdx.x; j < hidden_dim; j += blockDim.x) {
        my_token_data[j] += my_token_data[j + peer * hidden_dim * gridDim.x];
    }
  }

  coop.sync();

  //============================================================================
  // Phase 2: RMS Normalization
  //============================================================================
  // Normalize the reduced token data using block-level RMSNorm.
  // All threads in the block collaborate to compute RMS and apply normalization.
  //----------------------------------------------------------------------------
  blockRMSNorm(my_token_data, hidden_dim, eps, reduction_buffer, coop);

  //============================================================================
  // Phase 3: All-Gather via GIN PUT
  //============================================================================
  // Broadcast the normalized token back to all GPUs. Each GPU needs the complete
  // normalized results for the next layer computation.
  //
  // Communication Pattern:
  //   Each thread writes a subset of peers (stride blockDim.x).
  //   The normalized data from window_recv is copied to the appropriate
  //   position in each peer's window_send buffer.
  //
  // Synchronization:
  //   bar.sync(release) before Phase 3 publishes Phase 2 stores and participates in
  //   the world-team barrier so no rank starts Phase 3 PUTs until every rank has finished
  //   normalization. Signal values accumulate across phases.
  //   After Phase 1 wait, the signal has risen by nRanks (one inbound PUT per peer for
  //   token_idx). After Phase 3 wait, by 2*nRanks total (second inbound round per peer).
  //----------------------------------------------------------------------------

  // Release: publish normalization writes before Phase 3 PUTs
  bar.sync(coop, cuda::memory_order_release, ncclGinFenceLevel::None);

  size_t final_token_offset = (token_idx * hidden_dim) * sizeof(float);
  my_window_offset = (blockIdx.x * hidden_dim) * sizeof(float);

  for (int peer = threadIdx.x; peer < nRanks; peer += blockDim.x) {
    // PUT: send normalized data to peer's send window
    gin.put(ncclTeamWorld(devComm), peer, window_send, final_token_offset,
            window_recv, my_window_offset, sizeof(float) * hidden_dim,
            ncclGin_WeakSignalInc{signalIndex});
  }

  // Wait until every peer has completed its signaled PUT for Phase 3 all-gather: each
  // sends its normalized token slice into this rank's window_send where we expect it.
  gin.waitSignal(coop, signalIndex, signalValue + 2 * devComm.nRanks);
  // Phase 3 PUT used window_recv as source; flush lets us treat that staging as consumed
  // before the kernel ends and the host reuses the allocation.
  gin.flush(coop);
  bar.sync(coop, cuda::memory_order_release, ncclGinFenceLevel::None);
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
  NCCLCHECK(ncclCommInitRank(&comm, total_ranks, nccl_id, my_rank));

  // Check for Device API and GIN support
  ncclCommProperties_t props = NCCL_COMM_PROPERTIES_INITIALIZER;
  NCCLCHECK(ncclCommQueryProperties(comm, &props));
  if (!props.deviceApiSupport) {
    printf("ERROR: rank %d communicator does not support Device API!\n", my_rank);
    NCCLCHECK(ncclCommFinalize(comm));
    NCCLCHECK(ncclCommDestroy(comm));
    return NULL;
  }
  if (props.ginType == NCCL_GIN_TYPE_NONE) {
    printf("ERROR: rank %d communicator does not support GIN!\n", my_rank);
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
  // Step 3: Allocate Device Memory and Register GIN Windows
  //============================================================================
  float *d_data_send, *d_data_recv;
  ncclWindow_t window_send, window_recv;
  cudaStream_t stream;

  // Allocate and register send window (source data and final results)
  NCCLCHECK(ncclMemAlloc((void **)&d_data_send, tensor_size_bytes));
  NCCLCHECK(ncclCommWindowRegister(comm, d_data_send, tensor_size_bytes, &window_send, NCCL_WIN_COLL_SYMMETRIC));

  // Allocate and register receive window (intermediate staging for reduction)
  NCCLCHECK(ncclMemAlloc((void **)&d_data_recv, tensor_size_bytes));
  NCCLCHECK(ncclCommWindowRegister(comm, d_data_recv, tensor_size_bytes, &window_recv, NCCL_WIN_COLL_SYMMETRIC));

  // Copy input data to device and create stream
  CUDACHECK(cudaMemcpy(d_data_send, h_data, tensor_size_bytes, cudaMemcpyHostToDevice));
  CUDACHECK(cudaStreamCreate(&stream));

  //============================================================================
  // Step 4: Create Device Communicator with GIN Resources
  //============================================================================
  ncclDevComm devComm;
  ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
  reqs.ginConnectionType = NCCL_GIN_CONNECTION_FULL;  // Enable full GIN connectivity
  reqs.worldGinBarrierCount  = tokens_per_gpu;         // One barrier per token
  reqs.ginSignalCount = tokens_per_gpu;       // One signal per token
  NCCLCHECK(ncclDevCommCreate(comm, &reqs, &devComm));

  if (my_rank == 0) {
    printf("\n========================================\n");
    printf("Starting RMSNorm GIN Kernel\n");
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

  RMSNormGIN<<<tokens_per_gpu, threads_per_block, shared_mem_size, stream>>>(
      window_send, window_recv, devComm, sequence_length, hidden_size, eps);
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
  CUDACHECK(cudaMemcpy(h_data_validation, d_data_send, tensor_size_bytes, cudaMemcpyDeviceToHost));

  // Compare against expected results
  bool success = verify_results(h_data_validation, h_data, tensor_size);

  //============================================================================
  // Step 7: Cleanup Resources
  //============================================================================
  NCCLCHECK(ncclDevCommDestroy(comm, &devComm));
  NCCLCHECK(ncclCommWindowDeregister(comm, window_send));
  NCCLCHECK(ncclCommWindowDeregister(comm, window_recv));
  NCCLCHECK(ncclMemFree(d_data_send));
  NCCLCHECK(ncclMemFree(d_data_recv));
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
