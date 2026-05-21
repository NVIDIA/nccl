/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

/*
 * RMSNorm with Hybrid LSA/GIN Communication
 *
 * This example demonstrates fused computation and communication for distributed
 * RMSNorm using a hybrid approach that combines Load Store Accessible (LSA) for
 * intra-node communication and GIN (GPU-Initiated Networking) for inter-node
 * communication. This pattern is ideal for multi-node systems where GPUs within
 * a node can use fast NVLink (LSA) while communicating across nodes via network (GIN).
 *
 * The kernel performs three phases in a single launch:
 *   1. Reduce-Scatter: Gather and sum partial results using hybrid LSA/GIN
 *   2. RMS Normalization: Compute and apply RMSNorm to each token
 *   3. All-Gather: Broadcast normalized results using hybrid LSA/GIN
 */

#include "utils.h"
#include "rmsnorm_utils.cuh"
#include "nccl_device.h"

// Kernel: Fused RMSNorm with Hybrid LSA/GIN Communication
//==============================================================================
/*
 * Performs distributed RMSNorm using hybrid LSA/GIN communication.
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
 *   - Phase 1: Each GPU sends its token data to all peers (hybrid LSA/GIN)
 *              Peers accumulate data in strided layout in window_recv
 *   - Phase 3: Each GPU broadcasts normalized results back to all peers
 *              (hybrid LSA/GIN) to update their window_send buffers
 *
 * Synchronization:
 *   - Signals apply only to remote peers: remote PUT (gin.put) uses ncclGin_WeakSignalInc to
 *     add one completion increment on the destination rank, so waitSignal counts inbound signaled PUTs from remote
 *     (non-LSA) peers only (numRemotePeers = nRanks - lsa.nRanks per round). LSA
 *     stores do not increment this counter—an explicit LSA sub-barrier orders
 *     the LSA path.
 *   - Phase 1: gin.waitSignal waits until every remote peer has finished PUT of its
 *     partial contribution for token_idx (this rank and block's token) into this
 *     rank's window_recv. Same-node peers use LSA stores into recv instead (no signal);
 *     bar.lsaBarrier().sync(acq_rel) after wait + flush publishes local LSA stores
 *     and acquires same-node peers' LSA stores before reduction.
 *   - Phase 3: second wait until every remote peer has finished signaled PUT for
 *     all-gather into this rank's window_send; LSA copies are fenced by barriers.
 *   - gin.flush completes outbound GIN operations locally so window_send / window_recv
 *     regions used as PUT sources can be safely reused (LSA paths unaffected).
 *   - World-team bar.sync (acquire at kernel entry; release before Phase 3;
 *     ncclGinFenceLevel::None) aligns cross-node phases. The Phase 1 reduction
 *     consumes data after remote GIN signal waits and the LSA sub-barrier.
 *   - Signal thresholds accumulate across phases (no reset); each phase adds
 *     numRemotePeers remote completion increments to this block's signalIndex.
 *
 * Parameters:
 *   window_send     - Window for sending data (source and final destination)
 *   window_recv     - Window for receiving data (intermediate staging)
 *   devComm         - NCCL device communicator with GIN context
 *   sequence_length - Total number of tokens across all GPUs
 *   hidden_dim      - Dimensionality of each token
 *   eps             - Epsilon for numerical stability
 */
__global__ void RMSNormHybrid(ncclWindow_t window_send, ncclWindow_t window_recv, ncclDevComm devComm,
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

  ncclTeam world = ncclTeamWorld(devComm);
  ncclTeam lsa = ncclTeamLsa(devComm);
  const int startLsa = world.rank - lsa.rank;
  const int lsaSize = lsa.nRanks;
  const int numRemotePeers = world.nRanks - lsa.nRanks;

  ncclBarrierSession<ncclCoopCta> bar { coop, ncclTeamTagWorld(), gin, blockIdx.x };
  bar.sync(coop, cuda::memory_order_acquire, ncclGinFenceLevel::None);

  //============================================================================
  // Phase 1: Reduce-Scatter via Hybrid LSA/GIN Communication
  //============================================================================
  // This block owns one token: token_idx = rank * gridDim.x + blockIdx.x.
  // Each block sends its token data to all peer GPUs using hybrid communication:
  //   - Remote peers (outside LSA team): PUT (gin.put) with a completion signal (signals only here)
  //   - Local peers (within LSA team): direct LSA stores into window_recv (no signal)
  //
  // Receive layout (peer-major in each rank's window): for global peer p, the H*B
  // floats are [token0][token1]...[token_{B-1}]. Offset p*H*B + b*H + j for peer p,
  // local token b, dim j. Block b uses base &recv[b*H]; peer p at j + p*H*B.
  //   H = hidden_dim, B = tokens_per_gpu (gridDim.x)
  //
  // Communication Pattern:
  //   Each thread initiates PUTs (gin.put) to remote peers (stride blockDim.x)
  //   and direct LSA writes to local peers. All peers send concurrently.
  //
  // Synchronization:
  //   - gin.waitSignal(signalValue + numRemotePeers): every remote peer has finished
  //     its signaled PUT delivering its partial for token_idx into this rank's
  //     window_recv (numRemotePeers increments only—LSA peers are not in this count).
  //   - gin.flush: complete outbound GIN PUTs locally so window_send slices used as
  //     Phase 1 PUT sources can be safely reused.
  //   - bar.lsaBarrier().sync(acq_rel): publish this rank's LSA stores and acquire
  //     same-node peers' LSA stores before reduction loads.
  //----------------------------------------------------------------------------

  size_t my_window_offset = (token_idx * hidden_dim) * sizeof(float);

  // Remote peers: PUT (peers before LSA team)
  for (int peer = threadIdx.x; peer < startLsa; peer += blockDim.x) {
    const int peer_token_idx = peer * gridDim.x + blockIdx.x;
    size_t peer_window_offset = (peer_token_idx * hidden_dim) * sizeof(float);

    gin.put(ncclTeamWorld(devComm), peer, window_recv, my_window_offset,
            window_send, peer_window_offset, sizeof(float) * hidden_dim,
            ncclGin_WeakSignalInc{signalIndex});
  }

  // Remote peers: PUT (peers after LSA team)
  for (int peer = startLsa + lsaSize + threadIdx.x; peer < nRanks; peer += blockDim.x) {
    const int peer_token_idx = peer * gridDim.x + blockIdx.x;
    size_t peer_window_offset = (peer_token_idx * hidden_dim) * sizeof(float);

    gin.put(ncclTeamWorld(devComm), peer, window_recv, my_window_offset,
            window_send, peer_window_offset, sizeof(float) * hidden_dim,
            ncclGin_WeakSignalInc{signalIndex});
  }

  // Send to local peers using LSA direct writes
  for (size_t offset = threadIdx.x; offset < hidden_dim; offset += blockDim.x) {
    for (int lp = 0; lp < lsa.nRanks; lp++) {
      const int peer_token_idx = (lp + startLsa) * gridDim.x + blockIdx.x;
      size_t peer_window_offset = (peer_token_idx * hidden_dim) * sizeof(float);
      float* sendPtr = (float*)ncclGetLocalPointer(window_send, peer_window_offset);
      float* recvPtr = (float*)ncclGetLsaPointer(window_recv, my_window_offset, lp);
      recvPtr[offset] = sendPtr[offset];
    }
  }

  // Remote peers only: wait until each has completed its signaled PUT with its
  // contribution for token_idx into this rank's window_recv (reduce-scatter GIN leg).
  gin.waitSignal(coop, signalIndex, signalValue + numRemotePeers);
  // Flush outbound GIN Phase 1 PUTs so window_send sources are safe to reuse.
  gin.flush(coop);

  // LSA sub-barrier: publish our same-node stores and acquire peer same-node stores.
  // Remote GIN contributions are covered by the waitSignal above.
  bar.lsaBarrier().sync(coop, cuda::memory_order_acq_rel);

  //----------------------------------------------------------------------------
  // Reduction: Sum contributions from all peers
  //----------------------------------------------------------------------------
  // After the reduce-scatter phase, the receive window contains contributions
  // from all peers in a strided layout. We need to sum across the peer dimension.
  //
  // Contiguous layout: token0_peer0, token1_peer0, ..., token_{B-1}_peer0, then
  // token0_peer1, token1_peer1, ... (same peer-major order as 03_rmsnorm_gin).
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
  // Phase 3: All-Gather via Hybrid LSA/GIN Communication
  //============================================================================
  // Broadcast the normalized token back to all GPUs using hybrid communication:
  //   - Remote peers: use PUT (gin.put)
  //   - Local peers: Use direct LSA memory writes
  //
  // Each GPU needs the complete normalized results for the next layer computation.
  //
  // Communication Pattern:
  //   Each thread writes to a subset of remote peers (stride blockDim.x) via GIN
  //   and to all local peers via LSA. The normalized data from window_recv is
  //   copied to the appropriate position in each peer's window_send buffer.
  //
  // Synchronization:
  //   bar.sync(release) before Phase 3 publishes Phase 2 stores and participates in
  //   the world-team barrier so no rank starts Phase 3 PUT/LSA until every rank has finished
  //   normalization.
  //   GIN signals (remote peers only): after Phase 1 wait the counter rose by
  //   numRemotePeers; Phase 3 wait adds another numRemotePeers (signalValue +
  //   2*numRemotePeers). LSA all-gather uses stores; bar.sync fences them.
  //----------------------------------------------------------------------------

  // Release: publish normalization writes before Phase 3 PUTs / LSA writes
  bar.sync(coop, cuda::memory_order_release, ncclGinFenceLevel::None);

  size_t final_token_offset = (token_idx * hidden_dim) * sizeof(float);
  my_window_offset = (blockIdx.x * hidden_dim) * sizeof(float);

  // Remote peers: PUT (peers before LSA team)
  for (int peer = threadIdx.x; peer < startLsa; peer += blockDim.x) {
    gin.put(ncclTeamWorld(devComm), peer, window_send, final_token_offset,
            window_recv, my_window_offset, sizeof(float) * hidden_dim,
            ncclGin_WeakSignalInc{signalIndex});
  }

  // Remote peers: PUT (peers after LSA team)
  for (int peer = startLsa + lsaSize + threadIdx.x; peer < nRanks; peer += blockDim.x) {
    gin.put(ncclTeamWorld(devComm), peer, window_send, final_token_offset,
            window_recv, my_window_offset, sizeof(float) * hidden_dim,
            ncclGin_WeakSignalInc{signalIndex});
  }

  // Send to local peers using LSA direct writes
  for (size_t offset = threadIdx.x; offset < hidden_dim; offset += blockDim.x) {
    for (int lp = 0; lp < lsa.nRanks; lp++) {
      float* sendPtr = (float*)ncclGetLsaPointer(window_send, final_token_offset, lp);
      sendPtr[offset] = my_token_data[offset];
    }
  }

  // Remote peers only: wait until each has completed signaled Phase 3 PUT into
  // this rank's window_send (all-gather network leg). LSA writes use barriers, not signals.
  gin.waitSignal(coop, signalIndex, signalValue + 2 * numRemotePeers);
  // Flush outbound GIN Phase 3 PUTs so window_recv staging used as source is consumed.
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
  reqs.barrierCount = tokens_per_gpu;   // Hybrid world barrier per block
  reqs.ginSignalCount = tokens_per_gpu; // One signal per token
  NCCLCHECK(ncclDevCommCreate(comm, &reqs, &devComm));

  if (my_rank == 0) {
    printf("\n========================================\n");
    printf("Starting RMSNorm Hybrid Kernel\n");
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

  RMSNormHybrid<<<tokens_per_gpu, threads_per_block, shared_mem_size, stream>>>(
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
