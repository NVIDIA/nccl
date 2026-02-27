/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 * See LICENSE.txt for more license information.
 */

#include <unistd.h>
#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <memory>
#include <new>
#include <optional>
#include <set>
#include <string>
#include <vector>
#include <nccl.h>
#include <nccl_device.h>
#include "include/common.hpp"
#include "include/nccl_ep.h"

// HT (High Throughput) includes
#include "device/hybridep_adapter.cuh"
#include "device/hybridep_configs.cuh"

// Forward declarations for HT functions
static ncclResult_t init_hybridep_intranode(ncclEpGroup_t ep_group, const ncclEpGroupConfig_t* config, cudaStream_t stream);
static ncclResult_t destroy_hybridep_intranode(ncclEpGroup_t ep_group);
static ncclResult_t init_hybridep_internode(ncclEpGroup_t ep_group, const ncclEpGroupConfig_t* config, cudaStream_t stream);
static ncclResult_t destroy_hybridep_internode(ncclEpGroup_t ep_group);

// Define NCCL_CHECK_RESULT macro for NCCL error checking
#ifndef NCCL_CHECK_RESULT
#define NCCL_CHECK_RESULT(cmd) do {                 \
  ncclResult_t res = cmd;                           \
  if (res != ncclSuccess) {                         \
    fprintf(stderr, "NCCL error %d at %s:%d\n",     \
            res, __FILE__, __LINE__);               \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)
#endif

// Constants for low-latency mode
#define NUM_GPUS_PER_NODE_LOW_LATENCY 8

// Helper function to convert ncclDataType_t to cudaDataType_t
static cudaDataType_t ncclDataTypeToCudaDataType(ncclDataType_t nccl_type) {
    switch (nccl_type) {
        case ncclFloat16:    return CUDA_R_16F;
        case ncclFloat32:    return CUDA_R_32F;
        case ncclFloat64:    return CUDA_R_64F;
        case ncclBfloat16:   return CUDA_R_16BF;
        case ncclInt8:       return CUDA_R_8I;
        case ncclInt32:      return CUDA_R_32I;
        case ncclInt64:      return CUDA_R_64I;
        case ncclUint8:      return CUDA_R_8U;
        case ncclUint32:     return CUDA_R_32U;
        case ncclUint64:     return CUDA_R_64U;
        default:
            assert(false && "Unsupported ncclDataType_t for conversion to cudaDataType_t");
            return CUDA_R_16BF; // Default fallback
    }
}

static size_t ncclTypeSize(ncclDataType_t nccl_type) {
    switch (nccl_type) {
        case ncclInt8:
        case ncclUint8:
        case ncclFloat8e4m3:
        case ncclFloat8e5m2:
            return 1;
        case ncclFloat16:
        case ncclBfloat16:
            return 2;
        case ncclInt32:
        case ncclUint32:
        case ncclFloat32:
            return 4;
        case ncclInt64:
        case ncclUint64:
        case ncclFloat64:
            return 8;
        default:
            assert(false && "Unsupported ncclDataType_t for size query");
            return 0;
    }
}

// Allgather on host memory using NCCL
// Each rank contributes element_size bytes at offset rank * element_size
static void ncclAllGatherHost(
    void* host_buffer,
    size_t element_size,
    int rank,
    int nRanks,
    ncclComm_t comm,
    cudaStream_t stream
) {
    const size_t total_size = element_size * nRanks;
    void* d_buffer;
    CUDA_CHECK(cudaMalloc(&d_buffer, total_size));
    CUDA_CHECK(cudaMemcpy(
        static_cast<uint8_t*>(d_buffer) + rank * element_size,
        static_cast<uint8_t*>(host_buffer) + rank * element_size,
        element_size, cudaMemcpyHostToDevice));
    NCCL_CHECK_RESULT(ncclAllGather(
        static_cast<uint8_t*>(d_buffer) + rank * element_size,
        d_buffer, element_size, ncclUint8, comm, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaMemcpy(host_buffer, d_buffer, total_size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_buffer));
}

// NCCL barrier using AllReduce
static ncclResult_t ncclBarrier(ncclComm_t comm, cudaStream_t stream) {
    int *nccl_barrier_var = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&nccl_barrier_var), sizeof(int)));
    CUDA_CHECK(cudaMemset(nccl_barrier_var, 0, sizeof(int)));
    NCCL_CHECK_RESULT(ncclAllReduce(nccl_barrier_var, nccl_barrier_var, 1, ncclInt, ncclSum, comm, stream));
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(nccl_barrier_var));
    return ncclSuccess;
}

// Opaque struct definitions
struct ncclEpGroup {
    ncclComm_t comm;

    int nRanks;               // Total number of ranks (from ncclCommCount)
    int rank;                 // This rank's ID
    int nNodes;               // Number of nodes

    void* ep_workspace;       // Device workspace for EP operations
    int cuda_device_id;        // CUDA device ID
    int rdma_ranks;           // RDMA ranks (nRanks / NUM_MAX_NVL_PEERS)
    int rdma_rank;            // RDMA rank (rank / NUM_MAX_NVL_PEERS)
    void* rdma_buffer;
    ncclEpGroupConfig_t config;         // Stored configuration

    struct {
        // Communicators and device communicators
        std::vector<ncclComm_t> nccl_comms;   // Host array of communicators
        ncclDevComm_t* dcomms = nullptr;      // Host array of device communicators
        ncclDevComm_t* d_dcomms = nullptr;    // Device array of device communicators
        int num_comms = 0;                     // Number of NCCL communicators
        int num_dcomms = 0;                    // Number of device comms (can be > num_comms)
        int qps_per_rank = 0;                  // Total QPs (connections) per rank
        int num_ctx_per_comm = 0;              // Number of contexts per communicator

        // GIN memory base pointer and windows array
        void* gin_base_ptr = nullptr;         // Base pointer for all GIN memory
        ncclWindow_t* d_nccl_windows = nullptr;  // Device array of windows (one per comm)
        unsigned signals_base = 0;            // Base signal ID for dispatch
        unsigned combine_signal_offset = 0;   // Signal offset for combine operations
        int num_total_signals = 0;            // Total number of signals

        // Used by kernels to calculate actual addresses for RDMA puts
        size_t rdma_inter_node_group_token_offset = 0;
        size_t rdma_inter_node_group_prob_offset = 0;
        size_t rdma_inter_node_group_scaling_factor_offset = 0;
        size_t rdma_intra_node_red_token_offset = 0;
        size_t combine_rdma_inter_node_group_token_offset = 0;
        size_t rdma_intra_node_red_prob_offset = 0;
        size_t combine_rdma_inter_node_group_prob_offset = 0;
        size_t token_staging_offset = 0;
        size_t dense_prob_offset = 0;
        size_t scaling_factor_staging_offset = 0;

        // Layout: [NUM_NODES-1][BATCH_SIZE * bytes_per_entry]
        // bytes_per_entry = hidden * sizeof(TOKEN_DATA_TYPE) + prob_size + sf_size
        size_t rdma_send_staging_offset = 0;
        size_t rdma_inter_node_group_packed_offset = 0;  // Packed receive buffer (token+prob+sf)
        int rdma_batch_size = 6;

        unsigned signals_tail_base = 0;         // Base signal ID for tail tracking (sender -> receiver)
        int num_max_rdma_chunked_send_tokens = 6;

    } gin_config;

    int num_local_experts;    // Number of local experts (num_experts / comm->nRanks)
    int hidden;               // Hidden size (token_size_bytes / ncclTypeSize(ncclBfloat16))
    unsigned int device_sm_count; // Number of SMs on the device
    unsigned int num_sms_ht; // Number of SMs to use for HT kernels

    // Custom allocator function pointers
    ncclEpAllocFn_t alloc_fn;
    ncclEpFreeFn_t free_fn;

    // NVL
    int local_nvl_rank;
    int nvl_rank_count;

    // NCCL device API
    size_t num_nccl_comms;
    std::vector<ncclComm_t> nccl_comms;
    ncclDevComm_t* nccl_dev_comms;
    ncclWindow_t* nccl_wins;
    int num_dispatch_signals;

    // HT buffers for intranode communication
    struct {
        // IPC-mapped buffer pointer arrays (fixed-size, indexed by local NVL rank)
        // Host arrays for population and cleanup
        void **dispatch_expert_output_token_buffer_ptrs;
        float **dispatch_expert_output_prob_buffer_ptrs;
        float **dispatch_expert_output_scaling_factor_buffer_ptrs;
        uint16_t **combine_expert_input_token_buffer_ptrs;
        float **combine_expert_input_prob_buffer_ptrs;

        // Local buffers (owned by this rank)
        void *expert_output_token;
        float *expert_output_prob;
        float *expert_output_scaling_factor;
        uint16_t *expert_input_token;
        float *expert_input_prob;

        // Sync flags (rank 0 allocates, others IPC-map)
        uint32_t *intra_node_write_completion_flags;
        uint32_t *combine_intra_node_write_completion_flags;
        uint64_t *expected_rdma_flag_value;
        uint32_t *expected_intra_node_flag_value;
        uint64_t *combine_expected_rdma_flag_value;
        uint32_t *combine_expected_intra_node_flag_value;

        // RDMA buffers (multi-node only)
        void *rdma_inter_node_group_token;
        float *rdma_inter_node_group_prob;
        float *rdma_inter_node_group_scaling_factor;
        uint64_t *rdma_inter_node_group_flags;
        uint16_t *rdma_intra_node_red_token;
        float *rdma_intra_node_red_prob;
        uint16_t *combine_rdma_inter_node_group_token;
        float *combine_rdma_inter_node_group_prob;
        uint64_t *combine_rdma_inter_node_group_flags;

        // Pre-registered dispatch buffers (group-level, allocated during Group Create)
        // These are pre-registered with GIN to avoid ~60ms registration overhead during dispatch
        void *token_staging_buffer;          // Pre-registered staging buffer for user tokens
        float *dense_prob_buffer;            // Pre-registered buffer for sparse→dense prob conversion
        float *scaling_factor_staging_buffer; // Pre-registered staging buffer for FP8 scaling factors

        // Config
        int num_nvl_ranks;
        bool initialized;
        bool internode_initialized;
    } ht_buffers;

    // Constructor to properly initialize all members
    ncclEpGroup() :
        comm(nullptr),
        nRanks(0),
        rank(0),
        nNodes(0),
        ep_workspace(nullptr),
        cuda_device_id(0),
        rdma_ranks(0),
        rdma_rank(0),
        rdma_buffer(nullptr),
        config{},
        num_local_experts(0),
        hidden(0),
        device_sm_count(0),
        num_sms_ht(0),
        alloc_fn(nullptr),
        free_fn(nullptr),
        local_nvl_rank(0),
        nvl_rank_count(0),
        num_nccl_comms(0),
        nccl_comms{},
        nccl_dev_comms(nullptr),
        nccl_wins(nullptr),
        num_dispatch_signals(0),
        ht_buffers{} {}
};

// Helper to AllGather IPC handles using NCCL (device-staged)
static void allGatherIpcHandles(
    void* local_handle,
    void* all_handles,
    size_t handle_size,
    int rank,
    int nRanks,
    ncclComm_t comm,
    cudaStream_t stream
) {
    const size_t total_size = handle_size * nRanks;
    void* d_buffer;
    CUDA_CHECK(cudaMalloc(&d_buffer, total_size));
    CUDA_CHECK(cudaMemcpy(
        static_cast<uint8_t*>(d_buffer) + rank * handle_size,
        local_handle,
        handle_size, cudaMemcpyHostToDevice));
    NCCL_CHECK_RESULT(ncclAllGather(
        static_cast<uint8_t*>(d_buffer) + rank * handle_size,
        d_buffer, handle_size, ncclUint8, comm, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaMemcpy(all_handles, d_buffer, total_size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_buffer));
}

// HT Intranode Initialization (adapted for public NCCL APIs)
static ncclResult_t init_hybridep_intranode(ncclEpGroup_t ep_group,
                                    const ncclEpGroupConfig_t* in_config,
                                    cudaStream_t stream)
{
    ncclComm_t comm = ep_group->comm;
    int nRanks = ep_group->nRanks;
    int nNodes = ep_group->nNodes;
    int rank = ep_group->rank;
    int n_ranks_per_node = nRanks / nNodes;
    int local_nvl_rank = rank % n_ranks_per_node;
    int hidden = ep_group->hidden;
    int max_tokens_per_rank = in_config->max_tokens_per_rank;
    int num_local_experts = ep_group->num_local_experts;

    // Set group-level NVL info
    ep_group->local_nvl_rank = local_nvl_rank;
    ep_group->nvl_rank_count = n_ranks_per_node;
    ep_group->ht_buffers.num_nvl_ranks = n_ranks_per_node;
    ep_group->ht_buffers.initialized = false;

    // Enable P2P access
    for (int i = 0; i < n_ranks_per_node; i++) {
        if (i == local_nvl_rank) continue;
        int can_p2p = 0;
        CUDA_CHECK(cudaDeviceCanAccessPeer(&can_p2p, local_nvl_rank, i));
        if (can_p2p) {
            cudaError_t err = cudaDeviceEnablePeerAccess(i, 0);
            if (err != cudaSuccess && err != cudaErrorPeerAccessAlreadyEnabled) {
                fprintf(stderr, "HT: Failed to enable P2P from GPU %d to GPU %d\n", local_nvl_rank, i);
            }
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // max_recv_tokens: worst-case tokens a rank can receive from all ranks in the job
    // experts_per_node: total experts across all ranks on this node (stride per token in prob buffer)
    const size_t max_recv_tokens = static_cast<size_t>(max_tokens_per_rank) * nRanks;
    const int experts_per_node = num_local_experts * n_ranks_per_node;

    // Allocate expert_output_token buffer (IPC requires cudaMalloc)
    size_t expert_output_token_sz = max_recv_tokens * hidden * sizeof(uint16_t);
    CUDA_CHECK(cudaMalloc(&ep_group->ht_buffers.expert_output_token, expert_output_token_sz));

    // Allocate buffer pointer arrays
    size_t num_buffer_ptrs = sizeof(void*) * n_ranks_per_node;
    CUDA_CHECK(cudaHostAlloc(&ep_group->ht_buffers.dispatch_expert_output_token_buffer_ptrs,
                             num_buffer_ptrs, cudaHostAllocMapped));

    // Get and exchange IPC handles for token buffer
    cudaIpcMemHandle_t local_token_handle;
    CUDA_CHECK(cudaIpcGetMemHandle(&local_token_handle, ep_group->ht_buffers.expert_output_token));

    std::unique_ptr<uint8_t[]> all_token_handles(new uint8_t[CUDA_IPC_HANDLE_SIZE * nRanks]);
    allGatherIpcHandles(&local_token_handle, all_token_handles.get(), CUDA_IPC_HANDLE_SIZE, rank, nRanks, comm, stream);

    // Open IPC handles for peer ranks
    // Note: all_token_handles contains handles from all global ranks
    // We need to map local NVL rank to global rank for correct indexing
    int node_id = rank / n_ranks_per_node;
    for (int i = 0; i < n_ranks_per_node; i++) {
        if (i == local_nvl_rank) {
            ep_group->ht_buffers.dispatch_expert_output_token_buffer_ptrs[i] =
                ep_group->ht_buffers.expert_output_token;
        } else {
            // Map local NVL rank i to global rank
            int peer_global_rank = node_id * n_ranks_per_node + i;
            cudaIpcMemHandle_t peer_handle;
            memcpy(&peer_handle, all_token_handles.get() + peer_global_rank * CUDA_IPC_HANDLE_SIZE, CUDA_IPC_HANDLE_SIZE);
            CUDA_CHECK(cudaIpcOpenMemHandle(
                &ep_group->ht_buffers.dispatch_expert_output_token_buffer_ptrs[i],
                peer_handle, cudaIpcMemLazyEnablePeerAccess));
        }
    }

    // Allocate expert_output_prob buffer
    CUDA_CHECK(cudaHostAlloc(&ep_group->ht_buffers.dispatch_expert_output_prob_buffer_ptrs,
                             sizeof(float*) * n_ranks_per_node, cudaHostAllocMapped));
    size_t expert_output_prob_sz = max_recv_tokens * experts_per_node * sizeof(float);
    CUDA_CHECK(cudaMalloc(&ep_group->ht_buffers.expert_output_prob, expert_output_prob_sz));

    cudaIpcMemHandle_t local_prob_handle;
    CUDA_CHECK(cudaIpcGetMemHandle(&local_prob_handle, ep_group->ht_buffers.expert_output_prob));

    std::unique_ptr<uint8_t[]> all_prob_handles(new uint8_t[CUDA_IPC_HANDLE_SIZE * nRanks]);
    allGatherIpcHandles(&local_prob_handle, all_prob_handles.get(), CUDA_IPC_HANDLE_SIZE, rank, nRanks, comm, stream);

    for (int i = 0; i < n_ranks_per_node; i++) {
        if (i == local_nvl_rank) {
            ep_group->ht_buffers.dispatch_expert_output_prob_buffer_ptrs[i] =
                ep_group->ht_buffers.expert_output_prob;
        } else {
            // Map local NVL rank i to global rank (node_id already computed above)
            int peer_global_rank = node_id * n_ranks_per_node + i;
            cudaIpcMemHandle_t peer_handle;
            memcpy(&peer_handle, all_prob_handles.get() + peer_global_rank * CUDA_IPC_HANDLE_SIZE, CUDA_IPC_HANDLE_SIZE);
            void* ptr;
            CUDA_CHECK(cudaIpcOpenMemHandle(&ptr, peer_handle, cudaIpcMemLazyEnablePeerAccess));
            ep_group->ht_buffers.dispatch_expert_output_prob_buffer_ptrs[i] = static_cast<float*>(ptr);
        }
    }

    // Allocate completion flags (rank 0 allocates, others IPC-map)
    CUDA_CHECK(ep_group->alloc_fn(reinterpret_cast<void**>(&ep_group->ht_buffers.expected_intra_node_flag_value), sizeof(uint32_t)));
    CUDA_CHECK(cudaMemsetAsync(ep_group->ht_buffers.expected_intra_node_flag_value, 0, sizeof(uint32_t), stream));
    CUDA_CHECK(ep_group->alloc_fn(reinterpret_cast<void**>(&ep_group->ht_buffers.combine_expected_intra_node_flag_value), sizeof(uint32_t)));
    CUDA_CHECK(cudaMemsetAsync(ep_group->ht_buffers.combine_expected_intra_node_flag_value, 0, sizeof(uint32_t), stream));

    // Allocate and exchange dispatch completion flags
    cudaIpcMemHandle_t dispatch_completion_handle;
    if (local_nvl_rank == 0) {
        CUDA_CHECK(cudaMalloc(&ep_group->ht_buffers.intra_node_write_completion_flags, sizeof(uint32_t)));
        CUDA_CHECK(cudaMemsetAsync(ep_group->ht_buffers.intra_node_write_completion_flags, 0, sizeof(uint32_t), stream));
        CUDA_CHECK(cudaIpcGetMemHandle(&dispatch_completion_handle, ep_group->ht_buffers.intra_node_write_completion_flags));
    }

    // Broadcast handle from rank 0 to all ranks using AllGather + extract rank 0's data
    std::unique_ptr<uint8_t[]> all_dispatch_handles(new uint8_t[CUDA_IPC_HANDLE_SIZE * nRanks]);
    allGatherIpcHandles(&dispatch_completion_handle, all_dispatch_handles.get(), CUDA_IPC_HANDLE_SIZE, rank, nRanks, comm, stream);

    if (local_nvl_rank != 0) {
        // Get node-local rank 0's global rank (not global rank 0)
        int node_local_rank0_global = node_id * n_ranks_per_node;
        cudaIpcMemHandle_t rank0_handle;
        memcpy(&rank0_handle, all_dispatch_handles.get() + node_local_rank0_global * CUDA_IPC_HANDLE_SIZE, CUDA_IPC_HANDLE_SIZE);
        void* ptr;
        CUDA_CHECK(cudaIpcOpenMemHandle(&ptr, rank0_handle, cudaIpcMemLazyEnablePeerAccess));
        ep_group->ht_buffers.intra_node_write_completion_flags = static_cast<uint32_t*>(ptr);
    }

    // Allocate and exchange combine completion flags (similar pattern)
    cudaIpcMemHandle_t combine_completion_handle;
    if (local_nvl_rank == 0) {
        CUDA_CHECK(cudaMalloc(&ep_group->ht_buffers.combine_intra_node_write_completion_flags, sizeof(uint32_t)));
        CUDA_CHECK(cudaMemsetAsync(ep_group->ht_buffers.combine_intra_node_write_completion_flags, 0, sizeof(uint32_t), stream));
        CUDA_CHECK(cudaIpcGetMemHandle(&combine_completion_handle, ep_group->ht_buffers.combine_intra_node_write_completion_flags));
    }

    std::unique_ptr<uint8_t[]> all_combine_handles(new uint8_t[CUDA_IPC_HANDLE_SIZE * nRanks]);
    allGatherIpcHandles(&combine_completion_handle, all_combine_handles.get(), CUDA_IPC_HANDLE_SIZE, rank, nRanks, comm, stream);

    if (local_nvl_rank != 0) {
        // Get node-local rank 0's global rank (not global rank 0)
        int node_local_rank0_global = node_id * n_ranks_per_node;
        cudaIpcMemHandle_t rank0_handle;
        memcpy(&rank0_handle, all_combine_handles.get() + node_local_rank0_global * CUDA_IPC_HANDLE_SIZE, CUDA_IPC_HANDLE_SIZE);
        void* ptr;
        CUDA_CHECK(cudaIpcOpenMemHandle(&ptr, rank0_handle, cudaIpcMemLazyEnablePeerAccess));
        ep_group->ht_buffers.combine_intra_node_write_completion_flags = static_cast<uint32_t*>(ptr);
    }

    // Allocate combine input buffers
    CUDA_CHECK(cudaHostAlloc(&ep_group->ht_buffers.combine_expert_input_token_buffer_ptrs,
                             sizeof(uint16_t*) * n_ranks_per_node, cudaHostAllocMapped));
    CUDA_CHECK(cudaHostAlloc(&ep_group->ht_buffers.combine_expert_input_prob_buffer_ptrs,
                             sizeof(float*) * n_ranks_per_node, cudaHostAllocMapped));

    size_t expert_input_token_sz = max_recv_tokens * hidden * sizeof(uint16_t);
    CUDA_CHECK(cudaMalloc(&ep_group->ht_buffers.expert_input_token, expert_input_token_sz));

    size_t expert_input_prob_sz = max_recv_tokens * experts_per_node * sizeof(float);
    CUDA_CHECK(cudaMalloc(&ep_group->ht_buffers.expert_input_prob, expert_input_prob_sz));

    // Exchange combine token buffer IPC handles
    cudaIpcMemHandle_t local_combine_token_handle;
    CUDA_CHECK(cudaIpcGetMemHandle(&local_combine_token_handle, ep_group->ht_buffers.expert_input_token));

    std::unique_ptr<uint8_t[]> all_combine_token_handles(new uint8_t[CUDA_IPC_HANDLE_SIZE * nRanks]);
    allGatherIpcHandles(&local_combine_token_handle, all_combine_token_handles.get(), CUDA_IPC_HANDLE_SIZE, rank, nRanks, comm, stream);

    for (int i = 0; i < n_ranks_per_node; i++) {
        if (i == local_nvl_rank) {
            ep_group->ht_buffers.combine_expert_input_token_buffer_ptrs[i] =
                ep_group->ht_buffers.expert_input_token;
        } else {
            // Map local NVL rank i to global rank (node_id already computed above)
            int peer_global_rank = node_id * n_ranks_per_node + i;
            cudaIpcMemHandle_t peer_handle;
            memcpy(&peer_handle, all_combine_token_handles.get() + peer_global_rank * CUDA_IPC_HANDLE_SIZE, CUDA_IPC_HANDLE_SIZE);
            void* ptr;
            CUDA_CHECK(cudaIpcOpenMemHandle(&ptr, peer_handle, cudaIpcMemLazyEnablePeerAccess));
            ep_group->ht_buffers.combine_expert_input_token_buffer_ptrs[i] = static_cast<uint16_t*>(ptr);
        }
    }

    // Exchange combine prob buffer IPC handles
    cudaIpcMemHandle_t local_combine_prob_handle;
    CUDA_CHECK(cudaIpcGetMemHandle(&local_combine_prob_handle, ep_group->ht_buffers.expert_input_prob));

    std::unique_ptr<uint8_t[]> all_combine_prob_handles(new uint8_t[CUDA_IPC_HANDLE_SIZE * nRanks]);
    allGatherIpcHandles(&local_combine_prob_handle, all_combine_prob_handles.get(), CUDA_IPC_HANDLE_SIZE, rank, nRanks, comm, stream);

    for (int i = 0; i < n_ranks_per_node; i++) {
        if (i == local_nvl_rank) {
            ep_group->ht_buffers.combine_expert_input_prob_buffer_ptrs[i] =
                ep_group->ht_buffers.expert_input_prob;
        } else {
            // Map local NVL rank i to global rank (node_id already computed above)
            int peer_global_rank = node_id * n_ranks_per_node + i;
            cudaIpcMemHandle_t peer_handle;
            memcpy(&peer_handle, all_combine_prob_handles.get() + peer_global_rank * CUDA_IPC_HANDLE_SIZE, CUDA_IPC_HANDLE_SIZE);
            void* ptr;
            CUDA_CHECK(cudaIpcOpenMemHandle(&ptr, peer_handle, cudaIpcMemLazyEnablePeerAccess));
            ep_group->ht_buffers.combine_expert_input_prob_buffer_ptrs[i] = static_cast<float*>(ptr);
        }
    }

    ep_group->ht_buffers.initialized = true;
    CUDA_CHECK(cudaDeviceSynchronize());

    return ncclSuccess;
}

// HT Intranode Cleanup
static ncclResult_t destroy_hybridep_intranode(ncclEpGroup_t ep_group) {
    if (!ep_group->ht_buffers.initialized) return ncclSuccess;

    int local_nvl_rank = ep_group->local_nvl_rank;
    int n_ranks_per_node = ep_group->ht_buffers.num_nvl_ranks;

    // Close IPC handles for peer ranks
    for (int i = 0; i < n_ranks_per_node; i++) {
        if (i != local_nvl_rank) {
            if (ep_group->ht_buffers.dispatch_expert_output_token_buffer_ptrs &&
                ep_group->ht_buffers.dispatch_expert_output_token_buffer_ptrs[i]) {
                cudaIpcCloseMemHandle(ep_group->ht_buffers.dispatch_expert_output_token_buffer_ptrs[i]);
            }
            if (ep_group->ht_buffers.dispatch_expert_output_prob_buffer_ptrs &&
                ep_group->ht_buffers.dispatch_expert_output_prob_buffer_ptrs[i]) {
                cudaIpcCloseMemHandle(ep_group->ht_buffers.dispatch_expert_output_prob_buffer_ptrs[i]);
            }
            if (ep_group->ht_buffers.combine_expert_input_token_buffer_ptrs &&
                ep_group->ht_buffers.combine_expert_input_token_buffer_ptrs[i]) {
                cudaIpcCloseMemHandle(ep_group->ht_buffers.combine_expert_input_token_buffer_ptrs[i]);
            }
            if (ep_group->ht_buffers.combine_expert_input_prob_buffer_ptrs &&
                ep_group->ht_buffers.combine_expert_input_prob_buffer_ptrs[i]) {
                cudaIpcCloseMemHandle(ep_group->ht_buffers.combine_expert_input_prob_buffer_ptrs[i]);
            }
        }
    }

    // Close completion flag IPC handles (non-rank-0)
    if (local_nvl_rank != 0) {
        if (ep_group->ht_buffers.intra_node_write_completion_flags) {
            cudaIpcCloseMemHandle(ep_group->ht_buffers.intra_node_write_completion_flags);
        }
        if (ep_group->ht_buffers.combine_intra_node_write_completion_flags) {
            cudaIpcCloseMemHandle(ep_group->ht_buffers.combine_intra_node_write_completion_flags);
        }
    }

    // Free local buffers
    if (ep_group->ht_buffers.expert_output_scaling_factor) {
        ep_group->free_fn(ep_group->ht_buffers.expert_output_scaling_factor);
    }
    if (ep_group->ht_buffers.expert_output_token) {
        cudaFree(ep_group->ht_buffers.expert_output_token);
    }
    if (ep_group->ht_buffers.expert_output_prob) {
        cudaFree(ep_group->ht_buffers.expert_output_prob);
    }
    if (ep_group->ht_buffers.expert_input_token) {
        cudaFree(ep_group->ht_buffers.expert_input_token);
    }
    if (ep_group->ht_buffers.expert_input_prob) {
        cudaFree(ep_group->ht_buffers.expert_input_prob);
    }
    if (ep_group->ht_buffers.expected_intra_node_flag_value) {
        ep_group->free_fn(ep_group->ht_buffers.expected_intra_node_flag_value);
    }
    if (ep_group->ht_buffers.combine_expected_intra_node_flag_value) {
        ep_group->free_fn(ep_group->ht_buffers.combine_expected_intra_node_flag_value);
    }

    // Free completion flags (rank 0 only)
    if (local_nvl_rank == 0) {
        if (ep_group->ht_buffers.intra_node_write_completion_flags) {
            cudaFree(ep_group->ht_buffers.intra_node_write_completion_flags);
        }
        if (ep_group->ht_buffers.combine_intra_node_write_completion_flags) {
            cudaFree(ep_group->ht_buffers.combine_intra_node_write_completion_flags);
        }
    }

    // Free host pointer arrays
    if (ep_group->ht_buffers.dispatch_expert_output_scaling_factor_buffer_ptrs) {
        cudaFreeHost(ep_group->ht_buffers.dispatch_expert_output_scaling_factor_buffer_ptrs);
    }
    if (ep_group->ht_buffers.dispatch_expert_output_token_buffer_ptrs) {
        cudaFreeHost(ep_group->ht_buffers.dispatch_expert_output_token_buffer_ptrs);
    }
    if (ep_group->ht_buffers.dispatch_expert_output_prob_buffer_ptrs) {
        cudaFreeHost(ep_group->ht_buffers.dispatch_expert_output_prob_buffer_ptrs);
    }
    if (ep_group->ht_buffers.combine_expert_input_token_buffer_ptrs) {
        cudaFreeHost(ep_group->ht_buffers.combine_expert_input_token_buffer_ptrs);
    }
    if (ep_group->ht_buffers.combine_expert_input_prob_buffer_ptrs) {
        cudaFreeHost(ep_group->ht_buffers.combine_expert_input_prob_buffer_ptrs);
    }

    ep_group->ht_buffers.initialized = false;
    return ncclSuccess;
}

// NCCLCHECK macro for public API error checking
// Undef any existing definition from internal NCCL headers to avoid using ncclDebugLog
#ifdef NCCLCHECK
#undef NCCLCHECK
#endif
#define NCCLCHECK(cmd) do {                                 \
    ncclResult_t res = cmd;                                 \
    if (res != ncclSuccess) {                               \
        fprintf(stderr, "NCCL error %s:%d '%s'\n",          \
                __FILE__, __LINE__, ncclGetErrorString(res));\
        return res;                                          \
    }                                                        \
} while(0)

// CUDACHECK_RET macro for CUDA calls in functions returning ncclResult_t
#ifndef CUDACHECK_RET
#define CUDACHECK_RET(cmd) do {                             \
    cudaError_t err = cmd;                                  \
    if (err != cudaSuccess) {                               \
        fprintf(stderr, "CUDA error %s:%d '%s'\n",          \
                __FILE__, __LINE__, cudaGetErrorString(err));\
        return ncclInternalError;                            \
    }                                                        \
} while(0)
#endif

// Constants for GIN configuration
static constexpr int HYBRIDEP_GIN_MAX_CONTEXTS = 32;
static constexpr int HYBRIDEP_GIN_CTXS_PER_COMM = 4;
static constexpr int MAX_BARRIER_SESSIONS = 32;

static ncclResult_t init_hybridep_internode(ncclEpGroup_t ep_group,
    const ncclEpGroupConfig_t* in_config,
    cudaStream_t stream)
{
    // Initialize using public NCCL APIs
    if (!in_config || !ep_group) {
        fprintf(stderr, "init_hybridep_internode: null config or ep_group\n");
        return ncclInvalidArgument;
    }

    ncclComm_t comm = ep_group->comm;
    int nNodes = ep_group->nNodes;
    int nRanks = ep_group->nRanks;
    int rank = ep_group->rank;
    int n_ranks_per_node = nRanks / nNodes;
    int local_nvl_rank = rank % n_ranks_per_node;

    // Set group-level NVL info
    ep_group->local_nvl_rank = local_nvl_rank;
    ep_group->nvl_rank_count = n_ranks_per_node;
    ep_group->ht_buffers.num_nvl_ranks = n_ranks_per_node;
    ep_group->ht_buffers.internode_initialized = false;

    if (nNodes <= 1) {
        // Single node - no internode initialization needed
        return ncclSuccess;
    }

    // =========================================================================
    // Initialize sub-communicators for inter-node RDMA
    // Each sub-comm connects GPUs with the same local_rank across nodes
    // =========================================================================

    // Create sub-communicators by color (local_rank)
    int local_rank = ep_group->local_nvl_rank;
    int node_rank = ep_group->rdma_rank;
    int color = local_rank;  // GPUs with same local_rank form a sub-communicator
    int comm_rank = node_rank;  // Position within sub-communicator
    int comm_nranks = nNodes;  // Number of ranks in each sub-communicator

    // gin-deepep style: Use 1 NCCL communicator with qps_per_rank GIN contexts
    int qps_per_rank = ep_group->config.num_qp_per_rank;
    if (qps_per_rank == 0) qps_per_rank = 24;  // Default like in src/nccl_ep.cc
    ep_group->gin_config.qps_per_rank = qps_per_rank;  // Store for kernel use
    ep_group->gin_config.num_comms = (qps_per_rank / NCCL_GIN_MAX_CONNECTIONS) +
                                      ((qps_per_rank % NCCL_GIN_MAX_CONNECTIONS) > 0 ? 1 : 0);
    ep_group->gin_config.num_ctx_per_comm = NCCL_GIN_MAX_CONNECTIONS;

    ep_group->gin_config.nccl_comms.resize(ep_group->gin_config.num_comms);

    // Generate unique IDs - need n_ranks_per_node * num_comms IDs total
    // Each local_rank group needs its own set of IDs
    size_t single_id_size = sizeof(ncclUniqueId);
    size_t total_ids = n_ranks_per_node * ep_group->gin_config.num_comms;
    std::vector<ncclUniqueId> all_unique_ids(total_ids);

    // Rank 0 generates all IDs and broadcasts
    if (rank == 0) {
        for (size_t i = 0; i < total_ids; ++i) {
            NCCLCHECK(ncclGetUniqueId(&all_unique_ids[i]));
        }
    }

    // Broadcast unique IDs using NCCL
    {
        size_t ids_size = total_ids * single_id_size;
        void* d_ids = nullptr;
        CUDACHECK_RET(cudaMalloc(&d_ids, ids_size));
        CUDACHECK_RET(cudaMemcpy(d_ids, all_unique_ids.data(), ids_size, cudaMemcpyHostToDevice));
        cudaStream_t bcast_stream;
        CUDACHECK_RET(cudaStreamCreate(&bcast_stream));
        NCCLCHECK(ncclBroadcast(d_ids, d_ids, ids_size, ncclInt8, 0, comm, bcast_stream));
        CUDACHECK_RET(cudaStreamSynchronize(bcast_stream));
        CUDACHECK_RET(cudaMemcpy(all_unique_ids.data(), d_ids, ids_size, cudaMemcpyDeviceToHost));
        CUDACHECK_RET(cudaFree(d_ids));
        CUDACHECK_RET(cudaStreamDestroy(bcast_stream));
    }

    // Create sub-communicators - each GPU joins comms for its local_rank group
    // IDs are organized by color (local_rank)
    for (int c = 0; c < ep_group->gin_config.num_comms; ++c) {
        // ID offset: color * num_comms + c
        size_t id_offset = (color * ep_group->gin_config.num_comms + c);
        ncclUniqueId id = all_unique_ids[id_offset];

        // Initialize sub-communicator with only nNodes ranks (not all nRanks)
        NCCLCHECK(ncclCommInitRank(&ep_group->gin_config.nccl_comms[c], comm_nranks, id, comm_rank));
    }

    // Calculate signal requirements
    // IMPORTANT: Must match the kernel's template MAX_SUPPORTED_TOKENS_PER_RANK (8192) in hybridep_adapter.cu
    // The kernel uses hardcoded 8192 for signal ID calculation, not the runtime max_tokens_per_rank

    int max_chunks_per_rank = (MAX_SUPPORTED_TOKENS_PER_RANK + HT_OF_NUM_TOKENS_PER_CHUNK - 1) / HT_OF_NUM_TOKENS_PER_CHUNK;
    int dispatch_signals = n_ranks_per_node * nNodes * max_chunks_per_rank;
    int combine_signals = n_ranks_per_node * nNodes * max_chunks_per_rank;
    // Streaming RDMA signals: per-chunk tail signals to avoid out-of-order block completion races
    int streaming_tail_signals = nNodes * nNodes * n_ranks_per_node * max_chunks_per_rank;
    int streaming_head_signals = nNodes * nNodes * n_ranks_per_node;  // Head tracking (receiver -> sender)
    ep_group->gin_config.num_total_signals = dispatch_signals + combine_signals +
                                               streaming_tail_signals + streaming_head_signals + MAX_BARRIER_SESSIONS;
    ep_group->gin_config.signals_base = 0;
    ep_group->gin_config.combine_signal_offset = dispatch_signals;
    // Streaming signal bases (after dispatch and combine signals)
    ep_group->gin_config.signals_tail_base = dispatch_signals + combine_signals;

    // Create device communicator with GIN requirements
    // 1 NCCL comm, 1 device comm, qps_per_rank GIN contexts
    ep_group->gin_config.num_dcomms = ep_group->gin_config.num_comms;
    ep_group->gin_config.dcomms = new ncclDevComm_t[ep_group->gin_config.num_dcomms];
    for (int i = 0; i < ep_group->gin_config.num_dcomms; ++i) {
        ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
        reqs.barrierCount = MAX_BARRIER_SESSIONS;
        reqs.ginSignalCount = ep_group->gin_config.num_total_signals;
        reqs.ginForceEnable = true;
        reqs.ginContextCount = NCCL_GIN_MAX_CONNECTIONS;  // All QPs in single device comm
        NCCLCHECK(ncclDevCommCreate(ep_group->gin_config.nccl_comms[i], &reqs, &ep_group->gin_config.dcomms[i]));
    }

    // Allocate device memory for dcomms and copy
    CUDACHECK_RET(cudaMalloc(reinterpret_cast<void**>(&ep_group->gin_config.d_dcomms),
                          sizeof(ncclDevComm_t) * ep_group->gin_config.num_dcomms));
    CUDACHECK_RET(cudaMemcpy(ep_group->gin_config.d_dcomms, ep_group->gin_config.dcomms,
                          sizeof(ncclDevComm_t) * ep_group->gin_config.num_dcomms, cudaMemcpyHostToDevice));


    // =========================================================================
    // Allocate a single large RDMA buffer and register once
    // =========================================================================

    // GIN windows require aligned addresses (4KB alignment for RDMA)
    constexpr size_t GIN_ALIGNMENT = 4096;
    auto align_size = [](size_t sz, size_t alignment) {
        return (sz + alignment - 1) & ~(alignment - 1);
    };

    // Calculate total size needed for all RDMA buffers (aligned)
    // max_recv_tokens_inter: worst-case tokens received from all remote nodes
    // Each remote node has n_ranks_per_node senders, each sending up to max_tokens_per_rank tokens
    const size_t max_recv_tokens_inter = static_cast<size_t>(ep_group->config.max_tokens_per_rank) * n_ranks_per_node * (nNodes - 1);
    const int experts_per_node = ep_group->num_local_experts * n_ranks_per_node;

    size_t rdma_inter_node_group_token_sz = align_size(max_recv_tokens_inter * ep_group->hidden * sizeof(uint16_t), GIN_ALIGNMENT);
    size_t rdma_inter_node_group_prob_sz = align_size(max_recv_tokens_inter * experts_per_node * sizeof(float), GIN_ALIGNMENT);
    size_t rdma_inter_node_group_scaling_factor_sz = align_size(max_recv_tokens_inter * (ep_group->hidden / 128) * sizeof(float), GIN_ALIGNMENT);
    size_t rdma_intra_node_red_token_sz = align_size(max_recv_tokens_inter * ep_group->hidden * sizeof(uint16_t), GIN_ALIGNMENT);
    size_t combine_rdma_inter_node_group_token_sz = rdma_intra_node_red_token_sz;
    size_t rdma_intra_node_red_prob_sz = align_size(max_recv_tokens_inter * experts_per_node * sizeof(float), GIN_ALIGNMENT);
    size_t combine_rdma_inter_node_group_prob_sz = rdma_intra_node_red_prob_sz;
    size_t flags_sz = align_size(static_cast<size_t>(nNodes) * sizeof(uint64_t), GIN_ALIGNMENT);
    size_t token_staging_sz = align_size(static_cast<size_t>(ep_group->config.max_tokens_per_rank) * ep_group->hidden * sizeof(uint16_t), GIN_ALIGNMENT);
    size_t dense_prob_sz = align_size(static_cast<size_t>(ep_group->config.max_tokens_per_rank) * ep_group->config.num_experts * sizeof(float), GIN_ALIGNMENT);
    size_t scaling_factor_staging_sz = align_size(static_cast<size_t>(ep_group->config.max_tokens_per_rank) * sizeof(float), GIN_ALIGNMENT);

    // Per-destination batched staging buffer for RDMA sends
    // For each destination, we stage up to BATCH_SIZE tokens before issuing one large RDMA put
    // bytes_per_entry = token_bytes + prob_bytes + sf_bytes (packed for single RDMA put)
    constexpr int RDMA_BATCH_SIZE = 6;
    size_t bytes_per_token_entry = ep_group->hidden * sizeof(uint16_t);  // token data
    size_t bytes_per_prob_entry = (ep_group->num_local_experts * n_ranks_per_node) * sizeof(float);  // prob data
    size_t bytes_per_sf_entry = (ep_group->hidden / 128) * sizeof(float);  // scaling factor (FP8)
    size_t bytes_per_entry = bytes_per_token_entry + bytes_per_prob_entry + bytes_per_sf_entry;
    // Per-destination staging: [NUM_NODES-1][max_tokens_per_rank * bytes_per_entry]
    // We need to stage all tokens that might go to one destination (worst case: all tokens go to same dest)
    size_t rdma_send_staging_sz = align_size(static_cast<size_t>(nNodes - 1) * ep_group->config.max_tokens_per_rank * bytes_per_entry, GIN_ALIGNMENT);

    // Packed receive buffer: [NUM_NODES-1][max_tokens_per_rank * bytes_per_entry]
    // Sparse indexed by token position - receiver reads token X from position X
    size_t rdma_recv_packed_sz = align_size(static_cast<size_t>(nNodes - 1) * ep_group->config.max_tokens_per_rank * bytes_per_entry, GIN_ALIGNMENT);

    // Total size (each region is already aligned)
    size_t total_gin_buffer_size = 0;
    total_gin_buffer_size += rdma_inter_node_group_token_sz;
    total_gin_buffer_size += rdma_inter_node_group_prob_sz;
    total_gin_buffer_size += rdma_inter_node_group_scaling_factor_sz;
    total_gin_buffer_size += rdma_intra_node_red_token_sz;
    total_gin_buffer_size += combine_rdma_inter_node_group_token_sz;
    total_gin_buffer_size += rdma_intra_node_red_prob_sz;
    total_gin_buffer_size += combine_rdma_inter_node_group_prob_sz;
    total_gin_buffer_size += flags_sz * 2;  // dispatch and combine flags
    total_gin_buffer_size += token_staging_sz;
    total_gin_buffer_size += dense_prob_sz;
    total_gin_buffer_size += scaling_factor_staging_sz;
    total_gin_buffer_size += rdma_send_staging_sz;
    total_gin_buffer_size += rdma_recv_packed_sz;

    // Allocate single RDMA-capable buffer using public API
    NCCLCHECK(ncclMemAlloc(&ep_group->gin_config.gin_base_ptr, total_gin_buffer_size));

    // =========================================================================
    // Partition the buffer into individual regions FIRST
    // =========================================================================
    uint8_t* ptr = reinterpret_cast<uint8_t*>(ep_group->gin_config.gin_base_ptr);
    size_t offset = 0;

    ep_group->ht_buffers.rdma_inter_node_group_token = reinterpret_cast<void*>(ptr + offset);
    offset += rdma_inter_node_group_token_sz;

    ep_group->ht_buffers.rdma_inter_node_group_prob = reinterpret_cast<float*>(ptr + offset);
    offset += rdma_inter_node_group_prob_sz;

    ep_group->ht_buffers.rdma_inter_node_group_scaling_factor = reinterpret_cast<float*>(ptr + offset);
    offset += rdma_inter_node_group_scaling_factor_sz;

    ep_group->ht_buffers.rdma_intra_node_red_token = reinterpret_cast<uint16_t*>(ptr + offset);
    offset += rdma_intra_node_red_token_sz;

    ep_group->ht_buffers.combine_rdma_inter_node_group_token = reinterpret_cast<uint16_t*>(ptr + offset);
    offset += combine_rdma_inter_node_group_token_sz;

    ep_group->ht_buffers.rdma_intra_node_red_prob = reinterpret_cast<float*>(ptr + offset);
    offset += rdma_intra_node_red_prob_sz;

    ep_group->ht_buffers.combine_rdma_inter_node_group_prob = reinterpret_cast<float*>(ptr + offset);
    offset += combine_rdma_inter_node_group_prob_sz;

    ep_group->ht_buffers.rdma_inter_node_group_flags = reinterpret_cast<uint64_t*>(ptr + offset);
    CUDACHECK_RET(cudaMemset(ep_group->ht_buffers.rdma_inter_node_group_flags, 0, flags_sz));
    offset += flags_sz;

    ep_group->ht_buffers.combine_rdma_inter_node_group_flags = reinterpret_cast<uint64_t*>(ptr + offset);
    CUDACHECK_RET(cudaMemset(ep_group->ht_buffers.combine_rdma_inter_node_group_flags, 0, flags_sz));
    offset += flags_sz;

    ep_group->ht_buffers.token_staging_buffer = reinterpret_cast<void*>(ptr + offset);
    offset += token_staging_sz;

    ep_group->ht_buffers.dense_prob_buffer = reinterpret_cast<float*>(ptr + offset);
    offset += dense_prob_sz;

    ep_group->ht_buffers.scaling_factor_staging_buffer = reinterpret_cast<float*>(ptr + offset);
    offset += scaling_factor_staging_sz;

    // Per-destination staging buffer for batched RDMA sends
    // (pointer not stored in ht_buffers, accessed via offset from gin_base_ptr in kernel)
    // Layout: [dest_node_idx][staged_token_idx][packed_entry]
    offset += rdma_send_staging_sz;

    // =========================================================================
    // Register ONE window for the entire gin_base_ptr buffer
    // All sub-buffers are accessed via offsets from gin_base_ptr
    // =========================================================================

    // Calculate offsets for each buffer (relative to gin_base_ptr)
    // These will be passed to kernels via mr_info
    size_t cur_offset = 0;
    ep_group->gin_config.rdma_inter_node_group_token_offset = cur_offset;
    cur_offset += rdma_inter_node_group_token_sz;

    ep_group->gin_config.rdma_inter_node_group_prob_offset = cur_offset;
    cur_offset += rdma_inter_node_group_prob_sz;

    ep_group->gin_config.rdma_inter_node_group_scaling_factor_offset = cur_offset;
    cur_offset += rdma_inter_node_group_scaling_factor_sz;

    ep_group->gin_config.rdma_intra_node_red_token_offset = cur_offset;
    cur_offset += rdma_intra_node_red_token_sz;

    ep_group->gin_config.combine_rdma_inter_node_group_token_offset = cur_offset;
    cur_offset += combine_rdma_inter_node_group_token_sz;

    ep_group->gin_config.rdma_intra_node_red_prob_offset = cur_offset;
    cur_offset += rdma_intra_node_red_prob_sz;

    ep_group->gin_config.combine_rdma_inter_node_group_prob_offset = cur_offset;
    cur_offset += combine_rdma_inter_node_group_prob_sz;

    // Skip flags (not used in RDMA puts)
    cur_offset += flags_sz * 2;  // dispatch and combine flags

    ep_group->gin_config.token_staging_offset = cur_offset;
    cur_offset += token_staging_sz;

    ep_group->gin_config.dense_prob_offset = cur_offset;
    cur_offset += dense_prob_sz;

    ep_group->gin_config.scaling_factor_staging_offset = cur_offset;
    cur_offset += scaling_factor_staging_sz;

    ep_group->gin_config.rdma_send_staging_offset = cur_offset;
    ep_group->gin_config.rdma_batch_size = RDMA_BATCH_SIZE;
    cur_offset += rdma_send_staging_sz;

    ep_group->gin_config.rdma_inter_node_group_packed_offset = cur_offset;
    cur_offset += rdma_recv_packed_sz;

    // Allocate device window array (6 comms × 4 ctx = 24 total windows)
    int total_windows = ep_group->gin_config.num_comms * NCCL_GIN_MAX_CONNECTIONS;
    CUDACHECK_RET(cudaMalloc(reinterpret_cast<void**>(&ep_group->gin_config.d_nccl_windows),
                          total_windows * sizeof(ncclWindow_t)));

    std::vector<ncclWindow_t> host_windows(total_windows);

    // Register ONE window for the entire buffer per comm
    // IMPORTANT: ncclCommWindowRegister expects an array of NCCL_GIN_MAX_CONNECTIONS windows
    // to fill (one per GIN context)
    for (int c = 0; c < total_windows; ++c) {
        ncclComm_t reg_comm = ep_group->gin_config.nccl_comms[c/NCCL_GIN_MAX_CONNECTIONS];

        // Register the ENTIRE gin_base_ptr buffer as one window
        // NCCL will fill the array with windows for each context
        NCCLCHECK(ncclCommWindowRegister(reg_comm,
            ep_group->gin_config.gin_base_ptr,
            total_gin_buffer_size,
            host_windows.data() + c, 0));

    }

    // Copy windows to device
    CUDACHECK_RET(cudaMemcpy(ep_group->gin_config.d_nccl_windows, host_windows.data(),
                          host_windows.size() * sizeof(ncclWindow_t), cudaMemcpyHostToDevice));

    // =========================================================================
    // Local-only buffers (using custom allocator)
    // =========================================================================
    CUDACHECK_RET(ep_group->alloc_fn(reinterpret_cast<void**>(&ep_group->ht_buffers.expected_rdma_flag_value), flags_sz));
    CUDACHECK_RET(cudaMemset(ep_group->ht_buffers.expected_rdma_flag_value, 0, flags_sz));

    CUDACHECK_RET(ep_group->alloc_fn(reinterpret_cast<void**>(&ep_group->ht_buffers.combine_expected_rdma_flag_value), flags_sz));
    CUDACHECK_RET(cudaMemset(ep_group->ht_buffers.combine_expected_rdma_flag_value, 0, flags_sz));

    ep_group->ht_buffers.internode_initialized = true;
    return ncclSuccess;
}

static ncclResult_t destroy_hybridep_internode(ncclEpGroup_t ep_group){
    if (!ep_group->ht_buffers.internode_initialized) return ncclSuccess;

    // =========================================================================
    // Cleanup using public NCCL APIs
    // =========================================================================

    // Destroy device communicators
    if (ep_group->gin_config.dcomms != nullptr) {
        for (int c = 0; c < ep_group->gin_config.num_comms; ++c) {
            if (c < static_cast<int>(ep_group->gin_config.nccl_comms.size()) && ep_group->gin_config.nccl_comms[c]) {
                ncclResult_t res = ncclDevCommDestroy(ep_group->gin_config.nccl_comms[c], &ep_group->gin_config.dcomms[c]);
                if (res != ncclSuccess) {
                    fprintf(stderr, "[HT GIN] Warning: Failed to destroy device comm %d: %s\n",
                            c, ncclGetErrorString(res));
                }
            }
        }
        delete[] ep_group->gin_config.dcomms;
        ep_group->gin_config.dcomms = nullptr;
    }
    // Free device memory for dcomms
    if (ep_group->gin_config.d_dcomms != nullptr) {
        cudaFree(ep_group->gin_config.d_dcomms);
        ep_group->gin_config.d_dcomms = nullptr;
    }

    // Deregister and free windows
    if (ep_group->gin_config.d_nccl_windows != nullptr) {
        // d_nccl_windows is device memory - copy to host before dereferencing
        int total_windows = ep_group->gin_config.num_comms * NCCL_GIN_MAX_CONNECTIONS;
        ncclWindow_t* host_windows = new ncclWindow_t[total_windows];
        cudaMemcpy(host_windows, ep_group->gin_config.d_nccl_windows,
                   total_windows * sizeof(ncclWindow_t), cudaMemcpyDeviceToHost);
        for (int c = 0; c < total_windows; ++c) {
            ncclComm_t reg_comm = ep_group->gin_config.nccl_comms[c/NCCL_GIN_MAX_CONNECTIONS];
            ncclCommWindowDeregister(reg_comm, host_windows[c]);
        }
        delete[] host_windows;
        cudaFree(ep_group->gin_config.d_nccl_windows);
        ep_group->gin_config.d_nccl_windows = nullptr;
    }

    // Free the single GIN buffer (contains all RDMA regions)
    if (ep_group->gin_config.gin_base_ptr != nullptr) {
        ncclResult_t res = ncclMemFree(ep_group->gin_config.gin_base_ptr);
        if (res != ncclSuccess) {
            fprintf(stderr, "[HT GIN] Warning: Failed to free GIN memory: %s\n",
                    ncclGetErrorString(res));
        }
        ep_group->gin_config.gin_base_ptr = nullptr;

        // Clear buffer pointers (they pointed into gin_base_ptr)
        ep_group->ht_buffers.rdma_inter_node_group_token = nullptr;
        ep_group->ht_buffers.rdma_inter_node_group_prob = nullptr;
        ep_group->ht_buffers.rdma_inter_node_group_scaling_factor = nullptr;
        ep_group->ht_buffers.rdma_intra_node_red_token = nullptr;
        ep_group->ht_buffers.combine_rdma_inter_node_group_token = nullptr;
        ep_group->ht_buffers.rdma_intra_node_red_prob = nullptr;
        ep_group->ht_buffers.combine_rdma_inter_node_group_prob = nullptr;
        ep_group->ht_buffers.rdma_inter_node_group_flags = nullptr;
        ep_group->ht_buffers.combine_rdma_inter_node_group_flags = nullptr;
        ep_group->ht_buffers.token_staging_buffer = nullptr;
        ep_group->ht_buffers.dense_prob_buffer = nullptr;
        ep_group->ht_buffers.scaling_factor_staging_buffer = nullptr;
    }

    // Clear communicators vector (we don't destroy them since they're from user)
    ep_group->gin_config.nccl_comms.clear();
    ep_group->gin_config.num_comms = 0;

    // Free regular sync flag buffers (allocated with custom allocator)
    if (ep_group->ht_buffers.expected_rdma_flag_value) {
        ep_group->free_fn(ep_group->ht_buffers.expected_rdma_flag_value);
        ep_group->ht_buffers.expected_rdma_flag_value = nullptr;
    }
    if (ep_group->ht_buffers.combine_expected_rdma_flag_value) {
        ep_group->free_fn(ep_group->ht_buffers.combine_expected_rdma_flag_value);
        ep_group->ht_buffers.combine_expected_rdma_flag_value = nullptr;
    }
    ep_group->ht_buffers.internode_initialized = false;
    return ncclSuccess;
}

ncclResult_t ncclEpCreateGroup(
    ncclEpGroup_t* out_ep_group,
    ncclComm_t comm,
    const ncclEpGroupConfig_t* in_config,
    cudaStream_t stream,
    ncclEpAllocFn_t alloc_fn,
    ncclEpFreeFn_t free_fn
) {
    // Parameter validation
    assert(out_ep_group != nullptr);
    assert(in_config != nullptr);
    int nRanks;
    assert(comm != nullptr && ncclCommCount(comm, &nRanks) == ncclSuccess && nRanks > 0);
    assert(in_config->version == 1 && "ncclEpCreateGroup: invalid config version (expected 1)");
    assert((in_config->algorithm == NCCL_EP_ALGO_LOW_LATENCY ||
            in_config->algorithm == NCCL_EP_ALGO_HIGH_THROUGHPUT) &&
           "ncclEpCreateGroup: invalid algorithm, supported: low_latency, high_throughput");

    bool low_latency_mode = (in_config->algorithm == NCCL_EP_ALGO_LOW_LATENCY);
    bool hybridep_mode = (in_config->algorithm == NCCL_EP_ALGO_HIGH_THROUGHPUT);
    assert(in_config->num_experts > 0 && "ncclEpCreateGroup: num_experts must be greater than 0");
    assert(in_config->token_size_bytes > 0 && "ncclEpCreateGroup: token_size_bytes must be greater than 0");
    assert(!(in_config->algorithm == NCCL_EP_ALGO_LOW_LATENCY && in_config->max_tokens_per_rank == 0) &&
            "ncclEpCreateGroup: max_tokens_per_rank must be greater than 0 for low latency mode");
    assert(!(in_config->algorithm == NCCL_EP_ALGO_HIGH_THROUGHPUT && in_config->max_tokens_per_rank == 0) &&
             "ncclEpCreateGroup: max_tokens_per_rank must be set for HT backend");
    assert(!(in_config->algorithm == NCCL_EP_ALGO_HIGH_THROUGHPUT &&
             in_config->max_tokens_per_rank > MAX_SUPPORTED_TOKENS_PER_RANK) &&
             "ncclEpCreateGroup: HT max_tokens_per_rank exceeds build-time MAX_SUPPORTED_TOKENS_PER_RANK");
    if (hybridep_mode) {
        assert((nRanks % NUM_MAX_NVL_PEERS) == 0 && "ncclEpCreateGroup: HT requires num_ranks divisible by 8");
    }

    // Allocate EP group structure
    void* raw_memory = malloc(sizeof(ncclEpGroup));
    assert(raw_memory != nullptr && "Failed to malloc for ncclEpGroup");
    *out_ep_group = new (raw_memory) ncclEpGroup();
    ncclEpGroup_t ep_group = *out_ep_group;

    // Store configuration
    ep_group->comm = comm;
    ep_group->config = *in_config;
    // C-style cast required: cudaMalloc/cudaFree are overloaded and reinterpret_cast
    // cannot disambiguate overloaded functions.
    ep_group->alloc_fn = alloc_fn ? alloc_fn : (ncclEpAllocFn_t)cudaMalloc;
    ep_group->free_fn = free_fn ? free_fn : (ncclEpFreeFn_t)cudaFree;

    NCCL_CHECK_RESULT(ncclCommCount(comm, &ep_group->nRanks));
    NCCL_CHECK_RESULT(ncclCommUserRank(comm, &ep_group->rank));
    NCCL_CHECK_RESULT(ncclCommCuDevice(comm, &ep_group->cuda_device_id));

    // Determine number of nodes by gathering hostnames and counting unique ones
    constexpr size_t HOSTNAME_LEN = 256;
    std::vector<char> all_hostnames(HOSTNAME_LEN * ep_group->nRanks, 0);
    gethostname(all_hostnames.data() + ep_group->rank * HOSTNAME_LEN, HOSTNAME_LEN);
    ncclAllGatherHost(all_hostnames.data(), HOSTNAME_LEN, ep_group->rank, ep_group->nRanks, comm, stream);
    std::set<std::string> unique_hosts;
    for (int i = 0; i < ep_group->nRanks; ++i) {
        unique_hosts.insert(std::string(all_hostnames.data() + i * HOSTNAME_LEN));
    }
    ep_group->nNodes = static_cast<int>(unique_hosts.size());

    ep_group->num_local_experts = ep_group->config.num_experts / ep_group->nRanks;
    ep_group->hidden = ep_group->config.token_size_bytes / ncclTypeSize(ncclBfloat16);

    // Apply default values for auto-configured fields (when set to NCCL_EP_AUTO)
    if (ep_group->config.num_channels == NCCL_EP_AUTO) {
        ep_group->config.num_channels = 10;
    }

    if (ep_group->config.rdma_buffer_size == NCCL_EP_AUTO && ep_group->config.algorithm == NCCL_EP_ALGO_LOW_LATENCY) {
        ep_group->config.rdma_buffer_size = nccl_ep::get_low_latency_rdma_size_hint(ep_group->config.max_tokens_per_rank, ep_group->hidden, ep_group->nRanks, ep_group->config.num_experts);
    }

    if (ep_group->config.num_qp_per_rank == NCCL_EP_AUTO) {
        ep_group->config.num_qp_per_rank = 24;
    }

    // Calculate and store RDMA rank
    ep_group->rdma_ranks = std::max(1, ep_group->nRanks / NUM_MAX_NVL_PEERS);
    ep_group->rdma_rank = ep_group->rank / NUM_MAX_NVL_PEERS;
    ep_group->rdma_buffer = nullptr;

    CUDA_CHECK(cudaSetDevice(ep_group->cuda_device_id));
    cudaDeviceProp device_prop = {};
    CUDA_CHECK(cudaGetDeviceProperties(&device_prop, ep_group->cuda_device_id));
    ep_group->device_sm_count = device_prop.multiProcessorCount;

    CUDA_CHECK(ep_group->alloc_fn(&ep_group->ep_workspace, NUM_WORKSPACE_BYTES));
    CUDA_CHECK(cudaMemsetAsync(ep_group->ep_workspace, 0, NUM_WORKSPACE_BYTES, stream));

    // Initialize HT intranode buffers (IPC handles, completion flags, etc.)
    if (hybridep_mode) {
        NCCL_CHECK_RESULT(init_hybridep_intranode(ep_group, in_config, stream));
        NCCL_CHECK_RESULT(init_hybridep_internode(ep_group, in_config, stream));
    }

    if (ep_group->config.rdma_buffer_size > 0 && low_latency_mode) {
        // Allocate RDMA buffer
        ncclBarrier(ep_group->comm, stream);
        NCCL_CHECK_RESULT(ncclMemAlloc(&ep_group->rdma_buffer, ep_group->config.rdma_buffer_size));

        // Clean buffer (mainly for low-latency mode)
        CUDA_CHECK(cudaMemset(ep_group->rdma_buffer, 0, ep_group->config.rdma_buffer_size));
        // NCCL related setup
        ep_group->num_nccl_comms = (ep_group->config.num_qp_per_rank / MAX_NCCL_GIN_CTX_PER_COMM) +
                                    (ep_group->config.num_qp_per_rank % MAX_NCCL_GIN_CTX_PER_COMM > 0 ? 1 : 0);

        // setup nccl_comms - creating duplicate with commSplit
        ep_group->nccl_comms.resize(ep_group->num_nccl_comms);
        for (int i = 0; i < ep_group->num_nccl_comms; ++i) {
            NCCL_CHECK_RESULT(ncclCommSplit(ep_group->comm, 0, ep_group->rank, &ep_group->nccl_comms[i], nullptr));
        }

        // Cleaning up any pending CUDA error
        CUDA_CHECK(cudaGetLastError());

        // Create device communicators
        ncclDevComm_t* nccl_dev_comms_host = new ncclDevComm_t[ep_group->num_nccl_comms];
        NCCL_CHECK_RESULT(ncclGroupStart());
        ep_group->num_dispatch_signals = ep_group->num_local_experts * ep_group->nRanks;
        int num_total_signals = ep_group->num_dispatch_signals;

        for (int i = 0; i < ep_group->num_nccl_comms; ++i) {
            nccl_dev_comms_host[i] = ncclDevComm_t{};
            int max_barrier_sessions = (NUM_GPUS_PER_NODE_LOW_LATENCY * MAX_NCCL_GIN_CTX_PER_COMM);

            ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
            reqs.barrierCount = max_barrier_sessions;
            reqs.ginSignalCount = num_total_signals + max_barrier_sessions;
            reqs.ginForceEnable = true;
            NCCL_CHECK_RESULT(ncclDevCommCreate(ep_group->nccl_comms[i], &reqs, &nccl_dev_comms_host[i]));
        }
        NCCL_CHECK_RESULT(ncclGroupEnd());

        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ep_group->nccl_dev_comms),
                                sizeof(ncclDevComm_t) *ep_group->num_nccl_comms));
        CUDA_CHECK(cudaMemcpy(ep_group->nccl_dev_comms, nccl_dev_comms_host,
                                sizeof(ncclDevComm_t) *ep_group->num_nccl_comms, cudaMemcpyHostToDevice));

        // Register RDMA buffer with NCCL windows
        ncclWindow_t* nccl_wins_host = new ncclWindow_t[ep_group->num_nccl_comms];
        for (int i = 0; i < ep_group->num_nccl_comms; ++i) {
            NCCL_CHECK_RESULT(ncclCommWindowRegister(ep_group->nccl_comms[i], ep_group->rdma_buffer,
                                                        ep_group->config.rdma_buffer_size, &nccl_wins_host[i], 0));
        }

        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ep_group->nccl_wins),
                                sizeof(ncclWindow_t) * ep_group->num_nccl_comms));
        CUDA_CHECK(cudaMemcpy(ep_group->nccl_wins, nccl_wins_host,
                                sizeof(ncclWindow_t) * ep_group->num_nccl_comms, cudaMemcpyHostToDevice));

        // Cleanup host memory for NCCL windows and devcomms
        delete[] nccl_wins_host;
        nccl_wins_host = nullptr;

        delete[] nccl_dev_comms_host;
        nccl_dev_comms_host = nullptr;

        ncclBarrier(ep_group->comm, stream);

        CUDA_CHECK(cudaDeviceSynchronize());
    }

    return ncclSuccess;
}

ncclResult_t ncclEpGroupDestroy(
    ncclEpGroup_t ep_group,
    cudaStream_t stream
) {
    if (ep_group == nullptr) {
        return ncclSuccess;
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    // Clean up HT intranode resources
    if (ep_group->config.algorithm == NCCL_EP_ALGO_HIGH_THROUGHPUT &&
        ep_group->ht_buffers.initialized) {
        destroy_hybridep_intranode(ep_group);
    }
    // Clean up HT internode resources (GIN deregistration must happen before ncclCommDestroy)
    if (ep_group->config.algorithm == NCCL_EP_ALGO_HIGH_THROUGHPUT &&
        ep_group->ht_buffers.internode_initialized) {
        destroy_hybridep_internode(ep_group);
    }
    // Clean up workspace memory
    if (ep_group->ep_workspace != nullptr) {
        CUDA_CHECK(ep_group->free_fn(ep_group->ep_workspace));
    }

    // Clean up RDMA resources
    if (ep_group->config.rdma_buffer_size > 0 && NCCL_EP_ALGO_LOW_LATENCY == ep_group->config.algorithm) {
        CUDA_CHECK(cudaDeviceSynchronize());
        NCCL_CHECK_RESULT(ncclBarrier(ep_group->comm, stream));

        // Deregister NCCL windows
        ncclWindow_t* nccl_wins_host = nullptr;
        CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&nccl_wins_host),
                                    sizeof(ncclWindow_t) * ep_group->num_nccl_comms));
        CUDA_CHECK(cudaMemcpy(nccl_wins_host, ep_group->nccl_wins,
                                sizeof(ncclWindow_t) * ep_group->num_nccl_comms, cudaMemcpyDeviceToHost));

        for (int i = 0; i < ep_group->num_nccl_comms; ++i) {
            NCCL_CHECK_RESULT(ncclCommWindowDeregister(ep_group->nccl_comms[i], nccl_wins_host[i]));
        }

        CUDA_CHECK(cudaFreeHost(nccl_wins_host));
        CUDA_CHECK(cudaFree(ep_group->nccl_wins));
        ep_group->nccl_wins = nullptr;

        // Free RDMA buffer
        if (ep_group->rdma_buffer) {
            NCCL_CHECK_RESULT(ncclMemFree(ep_group->rdma_buffer));
            ep_group->rdma_buffer = nullptr;
        }

        // Free NCCL device communicators
        ncclDevComm_t* nccl_dev_comms_host = nullptr;
        CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&nccl_dev_comms_host),
                                    sizeof(ncclDevComm_t) * ep_group->num_nccl_comms));
        CUDA_CHECK(cudaMemcpy(nccl_dev_comms_host, ep_group->nccl_dev_comms,
                                sizeof(ncclDevComm_t) * ep_group->num_nccl_comms, cudaMemcpyDeviceToHost));

        for (int i = 0; i < ep_group->num_nccl_comms; ++i) {
            NCCL_CHECK_RESULT(ncclDevCommDestroy(ep_group->nccl_comms[i], &nccl_dev_comms_host[i]));
        }

        CUDA_CHECK(cudaFreeHost(nccl_dev_comms_host));
        CUDA_CHECK(cudaFree(ep_group->nccl_dev_comms));
        ep_group->nccl_dev_comms = nullptr;

        // Finalize and destroy NCCL communicators
        for (auto& comm : ep_group->nccl_comms) {
            NCCL_CHECK_RESULT(ncclCommFinalize(comm));
            NCCL_CHECK_RESULT(ncclCommDestroy(comm));
        }
        ep_group->nccl_comms.clear();
    }
    // Invoke destructor explicitly (placement new was used)
    ep_group->~ncclEpGroup();

    // Free the group structure
    free(ep_group);

    return ncclSuccess;
}

ncclResult_t ncclEpTensorCreate(
    ncclEpGroup_t ep_group,
    ncclNDTensor_t* tensor,
    unsigned int ndim,
    ncclDataType_t datatype,
    ncclEpTensorTag_t tag,
    unsigned int size0,
    unsigned int size1,
    unsigned int size2,
    unsigned int size3,
    unsigned int size4
) {
    assert(ep_group != nullptr);
    assert(tensor != nullptr);
    assert(ndim > 0 && ndim <= 5);

    unsigned int sizes[] = {size0, size1, size2, size3, size4};

    tensor->ndim = ndim;
    tensor->datatype = datatype;
    tensor->tag = tag;
    tensor->flags = NCCL_EP_TENSOR_FLAG_NONE;

    tensor->sizes = new unsigned int[ndim];
    tensor->strides = new unsigned int[ndim];

    unsigned int total_size = 1;
    for (unsigned int i = 0; i < ndim; i++) {
        tensor->sizes[i] = sizes[i];
        tensor->strides[i] = 1;
        total_size *= sizes[i];
    }

    CUDA_CHECK(ep_group->alloc_fn(&tensor->data, total_size * ncclTypeSize(datatype)));
    return ncclSuccess;
}

ncclResult_t ncclEpTensorDestroy(
    ncclEpGroup_t ep_group,
    ncclNDTensor_t* tensor
) {
    if (tensor == nullptr) return ncclSuccess;

    if (tensor->data && ep_group) {
        CUDA_CHECK(ep_group->free_fn(tensor->data));
    }
    delete[] tensor->sizes;
    delete[] tensor->strides;

    tensor->data = nullptr;
    tensor->sizes = nullptr;
    tensor->strides = nullptr;

    return ncclSuccess;
}

struct ncclEpHandle {
    ncclEpGroup_t group;

    bool use_fp8;

    // tensor that is owned by the user, do not free this tensor!
    const ncclNDTensor_t *topk_idx;
    int num_tokens, num_topk;

    bool cached_mode;
    int num_scales;
    int hidden_int4;

    union {
        struct {
            // Both intranode and internode
            int* recv_counter;
            int* recv_counter_device;
            int* internal_recv_expert_counter_host = nullptr;
            int received_token_count = -1;
            ncclNDTensor_t rank_token_counts;
            ncclNDTensor_t expert_token_counts;
            ncclNDTensor_t token_rank_mask;
            ncclNDTensor_t global_channel_prefix;
            ncclNDTensor_t nvl_send_head;
            ncclNDTensor_t recv_global_channel_prefix;

            // Internode only
            int* rdma_recv_counter;
            int* rdma_recv_counter_device;
            int rdma_received_token_count = -1;
            std::optional<ncclNDTensor_t> rdma_rank_token_counts;
            ncclNDTensor_t rdma_channel_prefix;
            ncclNDTensor_t recv_rdma_rank_prefix;
            ncclNDTensor_t recv_global_rank_prefix;
            ncclNDTensor_t rdma_send_head;
            ncclNDTensor_t recv_source_metadata;
            ncclNDTensor_t recv_rdma_channel_prefix;

            // Intranode only
            ncclNDTensor_t inter_rank_token_offsets;
            ncclNDTensor_t recv_token_source_map;
        } ht;
        struct {
            // packed tensors for LL
            ncclNDTensor_t expert_recv_source_indices;
            ncclNDTensor_t expert_dispatch_layout;

            int buffer_idx = 0;

            std::function<void(unsigned int)> continue_fn;
            nccl_ep::LowLatencyLayout layout;
        } ll;
        struct {
             // =================================================================================
             // ROUTING MAPS - Input to preprocessing, derived from topk_idx
             // =================================================================================

             // Local routing map: dense boolean representation of which experts each token routes to.
             // Converted from sparse topk_idx format before allgather.
             // dtype: bool
             // allocated: [max_tokens_per_rank, num_experts] (populated for num_tokens <= max_tokens_per_rank)
             // lifetime: created in ncclEpCreateHandle, freed in ncclEpHandleDestroy
             // vs NCCL HT: both produce dense structures, but different granularity and communication:
             //   - HT: computes is_token_in_rank [num_tokens, num_ranks], exchanges counts only (~KBs)
             //   - HT: converts to per-expert [num_tokens, num_experts], allgathers full map (~MBs)
            bool* local_routing_map;

             // Global routing map: allgathered routing decisions from ALL ranks.
             // Contains complete routing information needed for preprocessing.
             // dtype: bool
             // allocated: [max_tokens_per_rank * num_ranks, num_experts]
             // allgathered: [num_tokens * num_ranks, num_experts] (num_tokens <= max_tokens_per_rank)
             // lifetime: valid after ncclAllGather in ncclEpCreateHandle
             // vs NCCL LL: LL exchanges ~KBs of counts; HT exchanges ~MBs of full map
            bool* global_routing_map;

             // =================================================================================
             // PREPROCESSING OUTPUTS - Computed once per iteration, used by dispatch & combine
             // =================================================================================

             // Sparse-to-dense map: maps each (token, source_rank) pair to its position in
             // the destination rank's expert buffer. Used by NVLink (intra-node) warps.
             // Value of -1 indicates token is not routed to that rank.
             // dtype: int32_t
             // layout: [num_nodes * max_tokens_per_rank, num_ranks_per_node]
             // usage: dispatch S2G warp group, combine G2S warp group
             // lifetime: valid after metadata_preprocessing, constant within iteration
             // vs NCCL HT: no direct equivalent.
             //   - HT: uses is_token_in_rank (local tokens only) + atomics to compute positions
             //   - HT: precomputes positions for ALL nodes' tokens, -1 sentinel for not routed
            int32_t* sparse_to_dense_map;

             // RDMA-to-attention map: boolean mask indicating which tokens this node needs
             // to RECEIVE from RDMA (inter-node). Indexed by [node_id, token_id].
             // Primarily used during combine to know which remote tokens to wait for.
             // dtype: bool
             // layout: [num_nodes, max_tokens_per_rank_padded_to_16]
             //        (padding to 16 required for TMA alignment)
             // usage: dispatch G2S warp (polling), combine inter-node G2S/reduction warps
             // lifetime: valid after metadata_preprocessing, constant within iteration
             // vs NCCL HT: inverse perspective of is_token_in_rank.
             //   - is_token_in_rank: outbound - "where do MY tokens go?" [my_token, dest_rank]
             //   - rdma_to_attn_map: inbound - "which remote tokens do I receive?" [src_node, token]
            bool* rdma_to_attn_map;

             // Attention-to-RDMA map: boolean mask indicating which local tokens need to be
             // SENT via RDMA (inter-node) to each remote node.
             // Only allocated when num_nodes > 1.
             // dtype: bool
             // layout: [max_tokens_per_rank, num_nodes - 1]
             // usage: dispatch N2N (RDMA) warp group
             // lifetime: valid after metadata_preprocessing, constant within iteration
             // vs NCCL HT: closest equivalent to is_token_in_rank for inter-node RDMA.
             //   - is_token_in_rank: per-rank granularity [num_tokens, num_ranks]
             //   - attn_to_rdma_map: per-node granularity [num_tokens, num_nodes-1] (RDMA only)
            bool* attn_to_rdma_map;

             // Local expert routing map: per-expert routing for tokens in this rank's buffer.
             // Used by subsequent expert MLP layers to route tokens to correct experts.
             // dtype: bool
             // layout: [max_recv_tokens, experts_per_rank]
             //        where max_recv_tokens = num_ranks * max_tokens_per_rank
             // usage: passed to expert computation layers (not used by dispatch/combine directly)
             // lifetime: valid after metadata_preprocessing, constant within iteration
             // vs NCCL HT: similar purpose to num_tokens_per_expert but more detailed
            bool* local_expert_routing_map;

             // Number of tokens routed to local experts (total across all local experts).
             // Each token counted once even if routed to multiple local experts.
             // dtype: int32_t
             // layout: [1]
             // usage: buffer sizing, iteration control
             // lifetime: valid after metadata_preprocessing
            int32_t* num_tokens_for_experts;

             // =================================================================================
             // CONVERSION BUFFERS - Pre-allocated to avoid dispatch/combine-time malloc
             // =================================================================================

             // Dense prob buffer: shared scratch for sparse↔dense conversions
             // dtype: float
             // layout: [max_tokens_per_rank, num_experts]
             // usage:
             //   - dispatch forward: sparse→dense input topk_weights conversion
             //   - combine backward: dense→sparse output prob conversion
             // lifetime: allocated at handle creation, freed at handle destroy
             // note: dispatch and combine are sequential, so one buffer suffices
            // For multi-node: points to group-level pre-registered buffer
            // For single-node: handle-owned buffer
            float* dense_prob_buffer;

            // Token staging buffer: pre-registered buffer to avoid GIN registration during dispatch
            // User tokens are copied here during dispatch, then this buffer is used for RDMA
            // dtype: uint16_t (bf16) or uint8_t (fp8)
            // layout: [max_tokens_per_rank, hidden]
            // usage: copy user tokens → use for inter-node RDMA
            // lifetime: group-owned (allocated in Group Create, freed in Group Destroy)
            void* token_staging_buffer;  // Pointer to group-level buffer (not handle-owned)

            // Scaling factor staging buffer: pre-registered buffer for FP8 scaling factors
            // User scaling factors are copied here during dispatch, then this buffer is used for RDMA
            // dtype: float
            // layout: [max_tokens_per_rank]
            // usage: copy user scaling factors → use for inter-node RDMA
            // lifetime: group-owned (allocated in Group Create, freed in Group Destroy)
            float* scaling_factor_staging_buffer;  // Pointer to group-level buffer (not handle-owned)

            // RDMA inter-node group flags: atomic completion flags for each remote node.
            // Remote ranks increment via RDMA atomic fetch-add to signal chunk completion.
            // Only allocated when num_nodes > 1.
            // dtype: uint64_t
            // layout: [num_nodes - 1]
            // usage: dispatch N2N warp (signaling), dispatch G2S warp (polling)
            // lifetime: reset to 0 at init, incremented by remote RDMA atomics
            //uint64_t* rdma_inter_node_group_flags;
        } hybridep;
    };

    ncclEpHandle()
        : group(nullptr),
          topk_idx(nullptr),
          num_tokens(0),
          num_topk(0),
          cached_mode(false),
          num_scales(0),
          hidden_int4(0) {
        // Zero the entire union (ht, ll, hybridep share memory)
         // Use max size to ensure all members are zeroed
         constexpr size_t union_size = std::max({sizeof(ht), sizeof(ll), sizeof(hybridep)});
         memset(static_cast<void*>(&ht), 0, union_size);
    }

    ~ncclEpHandle() {
    }
};

static bool tensor_is_contiguous(const ncclNDTensor_t* tensor) {
    for (int i = 0; i < tensor->ndim; i++)
        if (tensor->strides[i] != 1)
            return false;
    return true;
}

static const ncclNDTensor_t* find_tensor_by_tag(const ncclNDTensor_t* const* tensors, int num_tensors, ncclEpTensorTag_t tag) {
    for (int i = 0; i < num_tensors; i++) {
        if (tensors[i]->tag == tag)
            return tensors[i];
    }
    return nullptr;
}

static ncclNDTensor_t* find_tensor_by_tag(ncclNDTensor_t* const* tensors, int num_tensors, ncclEpTensorTag_t tag) {
    for (int i = 0; i < num_tensors; i++) {
        if (tensors[i]->tag == tag)
            return tensors[i];
    }
    return nullptr;
}

static bool is_internode_available(ncclEpGroup_t ep_group) {
    return ep_group->nNodes > 1;
}


static void tensor_free(ncclEpGroup_t group, ncclNDTensor_t* t) {
    if (t->data)
        group->free_fn(t->data);
    if (t->strides)
        delete[] t->strides;
    if (t->sizes)
        delete[] t->sizes;
}

ncclResult_t ncclEpCreateHandle(
    ncclEpHandle_t* out_handle,
    ncclEpGroup_t ep_group,
    const ncclNDTensor_t* topk_idx,
    ncclNDTensor_t* const* local_tensors,
    unsigned int num_local_tensors,
    const ncclEpHandleConfig_t* config,
    cudaStream_t stream,
    bool use_fp8
) {
    assert(topk_idx != nullptr);
    assert(ep_group != nullptr);
    assert(out_handle != nullptr);
    assert(config == nullptr);

    // Validate communicator
    int nRanks = 0;
    assert(ep_group->comm != nullptr);
    assert(ncclCommCount(ep_group->comm, &nRanks) == ncclSuccess);
    assert(nRanks > 0);

    // Validate tensor properties
    assert(topk_idx->ndim == 2);
    assert(topk_idx->datatype == ncclInt64);
    assert(topk_idx->tag == NCCL_EP_TENSOR_TAG_TOPK_IDX);
    assert(tensor_is_contiguous(topk_idx));

    // Validate expert configuration
    assert(ep_group->config.num_experts > 0);
    assert(ep_group->config.num_experts % ep_group->nRanks == 0);

    // Create and initialize handle
    *out_handle = new ncclEpHandle();
    ncclEpHandle_t handle = *out_handle;

    handle->group = ep_group;
    handle->use_fp8 = use_fp8;
    handle->topk_idx = topk_idx;

    // Extract tensor dimensions
    const int num_tokens = static_cast<int>(topk_idx->sizes[0]);
    const int num_topk = static_cast<int>(topk_idx->sizes[1]);
    handle->num_tokens = num_tokens;
    handle->num_topk = num_topk;

    if (ep_group->config.algorithm == NCCL_EP_ALGO_LOW_LATENCY) {
        // LL mode does not accept local tensors
        assert(num_local_tensors == 0 && "LL mode does not accept local tensors in ncclEpCreateHandle");

        // Allocate packed tensors
        // packed_recv_x is the input tensor in the dispatch
        ncclEpTensorCreate(ep_group, &handle->ll.expert_recv_source_indices, 2, ncclInt32, NCCL_EP_TENSOR_TAG_NONE, static_cast<unsigned int>(handle->group->num_local_experts), static_cast<unsigned int>(ep_group->nRanks * ep_group->config.max_tokens_per_rank));
        ncclEpTensorCreate(ep_group, &handle->ll.expert_dispatch_layout, 2, ncclInt64, NCCL_EP_TENSOR_TAG_NONE, static_cast<unsigned int>(handle->group->num_local_experts), static_cast<unsigned int>(ep_group->nRanks));

        assert((ep_group->config.max_tokens_per_rank * handle->group->num_local_experts) % 4 == 0 and "TMA requires the number of tokens to be multiple of 4");

        handle->ll.layout = nccl_ep::LowLatencyLayout(handle->group->rdma_buffer, handle->group->config.max_tokens_per_rank, handle->group->hidden, handle->group->nRanks, handle->group->config.num_experts);

        assert(handle->ll.layout.total_bytes <= handle->group->config.rdma_buffer_size);
    } else { // HT
        assert(ep_group->config.max_tokens_per_rank > 0 && "HT requires max_tokens_per_rank > 0");
        assert(handle->num_tokens <= static_cast<int>(ep_group->config.max_tokens_per_rank) && "Token count exceeds HT buffer capacity");

        // Optional: per-expert token counts output
        ncclNDTensor_t* recv_expert_counter = nullptr;
        if (num_local_tensors > 0) {
            recv_expert_counter = find_tensor_by_tag(local_tensors, num_local_tensors, NCCL_EP_TENSOR_TAG_RECV_EXPERT_COUNTER_HOST);
            if (recv_expert_counter == nullptr) {
                recv_expert_counter = find_tensor_by_tag(local_tensors, num_local_tensors, NCCL_EP_TENSOR_TAG_RECV_EXPERT_COUNTER_DEVICE);
            }
        }

        const int nRanks = ep_group->nRanks;
        const int num_experts = ep_group->config.num_experts;
        const int max_tokens = ep_group->config.max_tokens_per_rank;
        const int total_tokens = nRanks * max_tokens;

        //===== HT-SPECIFIC: routing map + preprocessing =====
        // Unlike HT which uses per-rank routing (is_token_in_rank) + count exchange (~KBs),
        // HT uses per-expert routing and allgathers the full routing map (~MBs).
        // This allows precomputing exact buffer positions instead of using atomics.
        // rdma_ranks = num_nodes (1 for single-node), num_nvl_ranks = ranks per node
        // These are set in init_hybridep_intranode for single-node or during group creation for multi-node

        // Allocate routing maps
        CUDA_CHECK(ep_group->alloc_fn(reinterpret_cast<void**>(&handle->hybridep.local_routing_map),
                             static_cast<size_t>(max_tokens) * num_experts * sizeof(bool)));
        CUDA_CHECK(cudaMemset(handle->hybridep.local_routing_map, 0,
                             static_cast<size_t>(max_tokens) * num_experts * sizeof(bool)));
        CUDA_CHECK(ep_group->alloc_fn(reinterpret_cast<void**>(&handle->hybridep.global_routing_map),
                             static_cast<size_t>(total_tokens) * num_experts * sizeof(bool)));

        // Allocate preprocessing output buffers
        const int n_ranks_per_node = ep_group->nvl_rank_count;
        const int nNodes = ep_group->nNodes;

        CUDA_CHECK(ep_group->alloc_fn(reinterpret_cast<void**>(&handle->hybridep.sparse_to_dense_map),
                             static_cast<size_t>(nNodes) * max_tokens * n_ranks_per_node * sizeof(int32_t)));
        // Initialize sparse_to_dense_map to -1 (sentinel value for "no token")
        // This is critical for multinode: any unwritten entries must be -1 to avoid
        // garbage values being interpreted as valid indices in TMA operations
        CUDA_CHECK(cudaMemset(handle->hybridep.sparse_to_dense_map, 0xFF,
                             static_cast<size_t>(nNodes) * max_tokens * n_ranks_per_node * sizeof(int32_t)));

        CUDA_CHECK(ep_group->alloc_fn(reinterpret_cast<void**>(&handle->hybridep.rdma_to_attn_map),
                             static_cast<size_t>(nNodes) * ((max_tokens + 15) / 16 * 16) * sizeof(bool)));
        // Initialize rdma_to_attn_map to false (no tokens from RDMA initially)
        CUDA_CHECK(cudaMemset(handle->hybridep.rdma_to_attn_map, 0,
                             static_cast<size_t>(nNodes) * ((max_tokens + 15) / 16 * 16) * sizeof(bool)));

        if (nNodes > 1) {
        CUDA_CHECK(ep_group->alloc_fn(reinterpret_cast<void**>(&handle->hybridep.attn_to_rdma_map),
                                static_cast<size_t>(max_tokens) * (nNodes - 1) * sizeof(bool)));
        // Initialize attn_to_rdma_map to false (no tokens to RDMA initially)
        CUDA_CHECK(cudaMemset(handle->hybridep.attn_to_rdma_map, 0,
                                static_cast<size_t>(max_tokens) * (nNodes - 1) * sizeof(bool)));
        } else {
            handle->hybridep.attn_to_rdma_map = nullptr;
        }

        const int max_recv_tokens = nRanks * max_tokens;
        const int experts_per_rank = ep_group->num_local_experts;
        CUDA_CHECK(ep_group->alloc_fn(reinterpret_cast<void**>(&handle->hybridep.local_expert_routing_map),
                             static_cast<size_t>(max_recv_tokens) * experts_per_rank * sizeof(bool)));
        // Initialize local_expert_routing_map to false
        CUDA_CHECK(cudaMemset(handle->hybridep.local_expert_routing_map, 0,
                             static_cast<size_t>(max_recv_tokens) * experts_per_rank * sizeof(bool)));
        CUDA_CHECK(ep_group->alloc_fn(reinterpret_cast<void**>(&handle->hybridep.num_tokens_for_experts), sizeof(int32_t)));
        CUDA_CHECK(cudaMemset(handle->hybridep.num_tokens_for_experts, 0, sizeof(int32_t)));

        // Allocate conversion buffer
        size_t dense_prob_size = static_cast<size_t>(handle->num_tokens) * num_experts * sizeof(float);
        if (is_internode_available(ep_group)) {
            // Use group-level pre-registered buffer (allocated in init_hybridep_internode)
            handle->hybridep.dense_prob_buffer = ep_group->ht_buffers.dense_prob_buffer;
        } else {
            // Single-node: allocate local buffer (no GIN needed)
            CUDA_CHECK(ep_group->alloc_fn(reinterpret_cast<void**>(&handle->hybridep.dense_prob_buffer), dense_prob_size));
        }
        // Staging buffers for multi-node
        // For multi-node: use group-level pre-registered buffers (allocated in Group Create)
        // For single-node: not needed (use user buffers directly)
        if (is_internode_available(ep_group)) {
            // Use group-level pre-registered buffers (allocated in init_hybridep_internode)
            handle->hybridep.token_staging_buffer = ep_group->ht_buffers.token_staging_buffer;
            handle->hybridep.scaling_factor_staging_buffer = ep_group->ht_buffers.scaling_factor_staging_buffer;
        } else {
            handle->hybridep.token_staging_buffer = nullptr;
            handle->hybridep.scaling_factor_staging_buffer = nullptr;
        }
        // ===== Step 1: Convert sparse topk_idx to dense local_routing_map =====
        // NCCL sparse format: topk_idx[token][k] = expert_id
        // HT dense format: routing_map[token][expert] = true/false
        nccl_ep::hybridep::convert_topk_to_routing_map(
            static_cast<const int64_t*>(topk_idx->data),
            handle->hybridep.local_routing_map,
            handle->num_tokens,
            handle->num_topk,
            ep_group->config.num_experts,
            stream);
        // ===== Step 2: Allgather routing maps =====
        // Gather local routing maps from all ranks into global routing map
        // local_routing_map [num_tokens × num_experts] -> global_routing_map [total_tokens × num_experts]
        // Note: This exchanges ~MBs of data (vs HT's ~KBs count exchange)
        NCCL_CHECK_RESULT(ncclAllGather(
            handle->hybridep.local_routing_map,
            handle->hybridep.global_routing_map,
            static_cast<size_t>(handle->num_tokens) * num_experts,
            ncclUint8,
            ep_group->comm,
            stream));

        // Sync before preprocessing (ncclAllGather is async)
        CUDA_CHECK(cudaStreamSynchronize(stream));

             // ===== Step 3: Run metadata_preprocessing =====
             // Computes exact buffer positions via parallel prefix-sum:
             //   - sparse_to_dense_map: token→rank→buffer_position mapping
             //   - rdma_to_attn_map: which tokens come from RDMA (inter-node)
             //   - attn_to_rdma_map: which tokens go to RDMA (inter-node)
             //   - local_expert_routing_map: per-expert routing for received tokens
             //   - num_tokens_for_experts: total tokens routed to local experts
             //   - per_expert_token_counts: tokens per expert (optional, written to user buffer)
        int32_t* per_expert_counts_device = nullptr;
        if (recv_expert_counter != nullptr) {
            if (recv_expert_counter != nullptr) {
                assert(recv_expert_counter->ndim == 1 && "recv_expert_counter must be 1D");
                assert(recv_expert_counter->datatype == ncclInt32 && "recv_expert_counter must be ncclInt32");
                assert(recv_expert_counter->sizes[0] >= static_cast<unsigned int>(ep_group->num_local_experts) &&
                       "recv_expert_counter size must be >= num_local_experts");
                assert(recv_expert_counter->data != nullptr && "recv_expert_counter data must not be null");
            }

            if (recv_expert_counter->tag == NCCL_EP_TENSOR_TAG_RECV_EXPERT_COUNTER_HOST) {
                void* recv_expert_counter_device = nullptr;
                CUDA_CHECK(cudaHostGetDevicePointer(&recv_expert_counter_device, recv_expert_counter->data, /*flags=*/0));
                per_expert_counts_device = static_cast<int32_t*>(recv_expert_counter_device);
            } else {
                per_expert_counts_device = static_cast<int32_t*>(recv_expert_counter->data);
            }
        }

        nccl_ep::hybridep::call_metadata_preprocessing(
            handle->hybridep.global_routing_map,
            handle->hybridep.sparse_to_dense_map,
            handle->hybridep.rdma_to_attn_map,
            handle->hybridep.attn_to_rdma_map,
            handle->hybridep.num_tokens_for_experts,
            handle->hybridep.local_expert_routing_map,
            per_expert_counts_device,
            ep_group->rdma_rank,
            ep_group->local_nvl_rank,
            handle->num_tokens,
            ep_group->hidden,
            nNodes,
            n_ranks_per_node,
            experts_per_rank,
            stream);
            }

            return ncclSuccess;
    }

ncclResult_t ncclEpHandleDestroy(
    ncclEpHandle_t handle
) {
    if (!handle)
        return ncclSuccess;

    if (handle->group->config.algorithm == NCCL_EP_ALGO_LOW_LATENCY) {
        tensor_free(handle->group, &handle->ll.expert_recv_source_indices);
        tensor_free(handle->group, &handle->ll.expert_dispatch_layout);
    } else if (handle->group->config.algorithm == NCCL_EP_ALGO_HIGH_THROUGHPUT) {
        // Free HT routing maps
        if (handle->hybridep.local_routing_map)
            handle->group->free_fn(handle->hybridep.local_routing_map);
        if (handle->hybridep.global_routing_map)
            handle->group->free_fn(handle->hybridep.global_routing_map);

        // Free preprocessing outputs
        if (handle->hybridep.sparse_to_dense_map)
            handle->group->free_fn(handle->hybridep.sparse_to_dense_map);
        if (handle->hybridep.rdma_to_attn_map)
            handle->group->free_fn(handle->hybridep.rdma_to_attn_map);
        if (handle->hybridep.attn_to_rdma_map)
            handle->group->free_fn(handle->hybridep.attn_to_rdma_map);
        if (handle->hybridep.local_expert_routing_map)
            handle->group->free_fn(handle->hybridep.local_expert_routing_map);
        if (handle->hybridep.num_tokens_for_experts)
            handle->group->free_fn(handle->hybridep.num_tokens_for_experts);

        // Free conversion buffer
        if (handle->hybridep.dense_prob_buffer &&
            handle->hybridep.dense_prob_buffer != handle->group->ht_buffers.dense_prob_buffer) {
            // Handle-owned buffer (single-node case), free with custom allocator
            handle->group->free_fn(handle->hybridep.dense_prob_buffer);
        }
    }

    delete handle;
    return ncclSuccess;
}

    // EP Operations

ncclResult_t ncclEpDispatch(
    ncclEpHandle_t handle,
    const ncclNDTensor_t* const* inputs,
    unsigned int num_inputs,
    ncclNDTensor_t* const* outputs,
    unsigned int num_outputs,
    ncclNDTensor_t* const* local_tensors,
    unsigned int num_local_tensors,
    unsigned int send_only,
    const ncclEpDispatchConfig_t* config,
    cudaStream_t stream
) {
        ncclEpGroup_t group = handle->group;
    if (group->config.algorithm == NCCL_EP_ALGO_LOW_LATENCY) {
        const ncclNDTensor_t* x = find_tensor_by_tag(
            inputs, num_inputs, NCCL_EP_TENSOR_TAG_TOKENS
        );
        assert(x != nullptr);
        assert(x->ndim == 2);
        assert(tensor_is_contiguous(x));
        assert(x->datatype == ncclBfloat16);
        assert(x->sizes[0] == handle->num_tokens);
        assert(x->sizes[0] <= group->config.max_tokens_per_rank);
        assert(x->sizes[1] % sizeof(int4) == 0);
        assert(x->sizes[1] % 128 == 0);
        assert(x->sizes[1] * ncclTypeSize(x->datatype) == group->config.token_size_bytes);

        // Find and validate output tensors
        const ncclNDTensor_t* recv_x = find_tensor_by_tag(
            outputs, num_outputs, NCCL_EP_TENSOR_TAG_TOKENS
        );
        const ncclNDTensor_t* scales = find_tensor_by_tag(
            outputs, num_outputs, NCCL_EP_TENSOR_TAG_SCALES
        );
        assert(recv_x != nullptr);
        assert(recv_x->ndim == 3);
        assert(tensor_is_contiguous(recv_x));
        assert(recv_x->sizes[0] == group->num_local_experts);
        assert(recv_x->sizes[1] == group->nRanks * group->config.max_tokens_per_rank);
        assert(recv_x->sizes[2] == group->hidden);

        if (scales != nullptr) {
            constexpr int scale_block_size = 128;
            assert(group->hidden % 512 == 0);
            assert(scales->ndim == 3);
            assert(tensor_is_contiguous(scales));
            assert(scales->datatype == ncclFloat32);
            assert(scales->sizes[0] == group->num_local_experts);
            assert(scales->sizes[1] == group->config.max_tokens_per_rank * group->nRanks);
            assert(scales->sizes[2] == group->hidden / scale_block_size);
        }

        const ncclNDTensor_t* recv_count = find_tensor_by_tag(
            local_tensors, num_local_tensors, NCCL_EP_TENSOR_TAG_RECV_EXPERT_COUNTER_DEVICE
        );

        assert(recv_count != nullptr);
        assert(recv_count->ndim == 1);
        assert(tensor_is_contiguous(recv_count));
        assert(recv_count->datatype == ncclInt32);
        assert(recv_count->sizes[0] == group->num_local_experts);

        const auto& buffer = handle->ll.layout.buffers[handle->ll.buffer_idx];
        auto& next_buffer = handle->ll.layout.buffers[handle->ll.buffer_idx ^= 1];
        const auto next_clean_meta = next_buffer.clean_meta();

        unsigned signal_base = group->num_dispatch_signals;
        auto dispatch_fn = [=](int phases) {
            // Prepare data pointers
            auto* recv_x_data = recv_x->data;
            auto* scales_data = scales ? scales->data : nullptr;
            auto* expert_recv_source_indices_data = static_cast<int*>(handle->ll.expert_recv_source_indices.data);
            auto* expert_dispatch_layout_data = static_cast<int64_t*>(handle->ll.expert_dispatch_layout.data);
            auto* recv_count_data = static_cast<int*>(recv_count->data);
            auto* x_data = x->data;
            auto* topk_idx_data = static_cast<int64_t*>(handle->topk_idx->data);

            const bool use_fp8 = (scales != nullptr);
            const bool round_scale = config->round_scales;
            const bool use_ue8m0 = false;

            nccl_ep::internode_ll::dispatch(
                x_data,
                topk_idx_data,
                recv_x_data,
                scales_data,
                expert_recv_source_indices_data,
                expert_dispatch_layout_data,
                recv_count_data,
                buffer.dispatch_rdma_send_buffer,
                buffer.dispatch_rdma_recv_data_buffer,
                buffer.dispatch_rdma_recv_count_buffer,
                (reinterpret_cast<uint64_t>(buffer.dispatch_rdma_send_buffer) - reinterpret_cast<uint64_t>(group->rdma_buffer)),
                (reinterpret_cast<uint64_t>(buffer.dispatch_rdma_recv_data_buffer) - reinterpret_cast<uint64_t>(group->rdma_buffer)),
                (reinterpret_cast<uint64_t>(buffer.dispatch_rdma_recv_count_buffer) - reinterpret_cast<uint64_t>(group->rdma_buffer)),
                next_clean_meta.first,
                next_clean_meta.second,
                nullptr, /*recv_send_sizes=*/
                nullptr, /*recv_send_offsets=*/
                handle->num_tokens,
                group->hidden,
                group->config.max_tokens_per_rank,
                handle->num_topk,
                group->config.num_experts,
                group->rank,
                group->nRanks,
                use_fp8,
                round_scale,
                use_ue8m0,
                phases,
                group->num_nccl_comms,
                group->nccl_dev_comms,
                group->nccl_wins,
                signal_base,
                group->ep_workspace,
                group->device_sm_count,
                stream
            );
        };

        // Execute dispatch with appropriate phase flags
        const int dispatch_phases = send_only ?
            LOW_LATENCY_SEND_PHASE :
            (LOW_LATENCY_SEND_PHASE | LOW_LATENCY_RECV_PHASE);
        dispatch_fn(dispatch_phases);

        if (send_only) {
            handle->ll.continue_fn = dispatch_fn;
        }
    } else { // HT

        bool is_single_node = !is_internode_available(group);

        assert(num_local_tensors == 0 && "HT dispatch does not accept local_tensors");

        const ncclNDTensor_t* x = find_tensor_by_tag(inputs, num_inputs, NCCL_EP_TENSOR_TAG_TOKENS);
        const ncclNDTensor_t* topk_idx = find_tensor_by_tag(inputs, num_inputs, NCCL_EP_TENSOR_TAG_TOPK_IDX);
        const ncclNDTensor_t* topk_weights = find_tensor_by_tag(inputs, num_inputs, NCCL_EP_TENSOR_TAG_TOPK_WEIGHTS);
        const ncclNDTensor_t* scales = find_tensor_by_tag(inputs, num_inputs, NCCL_EP_TENSOR_TAG_SCALES);

        assert(x != nullptr);
        assert(x->ndim == 2 && tensor_is_contiguous(x));
        assert(x->sizes[0] == handle->num_tokens);
        assert(x->sizes[0] <= group->config.max_tokens_per_rank);

        // For multi-node: copy user buffers to pre-registered staging buffers
        // The staging buffers were allocated and GIN-registered during Group Create
        // This avoids ~60ms GIN registration overhead on the dispatch hot path
        void* token_ptr = x->data;  // Default: use user buffer directly
        if (!is_single_node && handle->hybridep.token_staging_buffer != nullptr) {
            // Copy user tokens to pre-registered staging buffer (D2D copy is ~0.1ms vs ~30ms GIN registration)
            size_t token_size = x->sizes[0] * x->sizes[1] * ncclTypeSize(x->datatype);
            CUDA_CHECK(cudaMemcpyAsync(handle->hybridep.token_staging_buffer, x->data, token_size, cudaMemcpyDeviceToDevice, stream));
            token_ptr = handle->hybridep.token_staging_buffer;
        }
        // Detect FP8 mode based on datatype
        bool use_fp8 = (x->datatype == ncclFloat8e4m3 || x->datatype == ncclFloat8e5m2);

        // For FP8: copy user scaling factors to pre-registered staging buffer
        float* scales_ptr = use_fp8 ? static_cast<float*>(scales->data) : nullptr;  // Default: use user buffer directly
        if (use_fp8 && !is_single_node && handle->hybridep.scaling_factor_staging_buffer != nullptr) {
            // Copy user scaling factors to pre-registered staging buffer (D2D copy is ~0.1ms vs ~30ms GIN registration)
            size_t scales_size = x->sizes[0] * sizeof(float);  // One scale per token
            CUDA_CHECK(cudaMemcpyAsync(handle->hybridep.scaling_factor_staging_buffer, scales->data, scales_size, cudaMemcpyDeviceToDevice, stream));
            scales_ptr = handle->hybridep.scaling_factor_staging_buffer;
        }
        if (!use_fp8) {
            assert(x->datatype == ncclBfloat16);
        } else {
            assert(scales != nullptr && "FP8 tokens require scales input");
            assert(scales->ndim == 2 && tensor_is_contiguous(scales));
            assert(scales->datatype == ncclFloat32);
        }

        // Output tensors
        ncclNDTensor_t* recv_topk_weights = find_tensor_by_tag(outputs, num_outputs, NCCL_EP_TENSOR_TAG_TOPK_WEIGHTS);
        ncclNDTensor_t* recv_topk_idx = find_tensor_by_tag(outputs, num_outputs, NCCL_EP_TENSOR_TAG_TOPK_IDX);

        // Detect forward/backward mode
        bool forward_dispatch = (topk_idx != nullptr);

         // Validate topk inputs - match HT mode validation logic
        if (forward_dispatch) {
            assert(topk_weights != nullptr);
            assert(topk_idx->ndim == 2 && tensor_is_contiguous(topk_idx) && topk_idx->datatype == ncclInt64);
            assert(topk_weights->ndim == 2 && tensor_is_contiguous(topk_weights) && topk_weights->datatype == ncclFloat32);
            assert(topk_weights->sizes[0] == handle->num_tokens);
            assert(topk_weights->sizes[1] == handle->num_topk);
        } else {
            assert(topk_weights == nullptr);
            assert(recv_topk_weights == nullptr);
            assert(recv_topk_idx == nullptr);
        }

        /* ===== Convert sparse topk_weights → dense prob (sparse→dense format) ===== */
        // NCCL HT uses sparse format: topk_idx[token][k] = expert_id, topk_weights[token][k] = weight
        // HT uses dense format: prob[token][expert] = weight (0 if not routed)
        float* dense_prob = handle->hybridep.dense_prob_buffer;
        if (forward_dispatch) {
            size_t dense_prob_size = static_cast<size_t>(handle->num_tokens) * group->config.num_experts * sizeof(float);
            CUDA_CHECK(cudaMemsetAsync(dense_prob, 0, dense_prob_size, stream));

            nccl_ep::hybridep::sparse_to_dense_prob(
                static_cast<const int64_t*>(topk_idx->data),
                static_cast<const float*>(topk_weights->data),
                dense_prob,
                handle->num_tokens,
                handle->num_topk,
                group->config.num_experts,
                stream);
        }

        /* ===== Build DispatchParams ===== */
        // DispatchParams encapsulates all buffers and metadata needed by HT dispatch kernel:
        //   - Input buffers: attn_input_token, attn_input_prob, attn_input_scaling_factor
        //   - IPC output buffers: expert_output_token_ptrs, expert_output_prob_ptrs (per-rank pointers)
        //   - RDMA staging buffers: rdma_inter_node_group_* (for multi-node only)
        //   - Metadata: sparse_to_dense_map, rdma_to_attn_map, attn_to_rdma_map
        //   - Sync flags: expected_*_flag_value, intra_node_write_completion_flags
        nccl_ep::hybridep::DispatchParams params;
        params.hidden_dim = group->hidden;
        params.experts_per_rank = group->num_local_experts;
        params.num_ranks_per_node = group->nvl_rank_count;
        params.attn_input_token = token_ptr;
        params.attn_input_prob = forward_dispatch ? dense_prob : nullptr;
        params.attn_input_scaling_factor = use_fp8 ? static_cast<const float*>(scales_ptr) : nullptr;
        // Use HOST pointer arrays - these get copied into the kernel param struct for fast __grid_constant__ access
        params.expert_output_token_ptrs = group->ht_buffers.dispatch_expert_output_token_buffer_ptrs;
        params.expert_output_prob_ptrs = group->ht_buffers.dispatch_expert_output_prob_buffer_ptrs;
        params.expert_output_scaling_factor_ptrs = use_fp8 ? group->ht_buffers.dispatch_expert_output_scaling_factor_buffer_ptrs : nullptr;
        params.rdma_to_attn_map = handle->hybridep.rdma_to_attn_map;
        params.attn_to_rdma_map = handle->hybridep.attn_to_rdma_map;
        params.sparse_to_dense_map = handle->hybridep.sparse_to_dense_map;
        params.expected_rdma_flag_value = is_single_node ? nullptr : group->ht_buffers.expected_rdma_flag_value;
        params.rdma_inter_node_group_flags = is_single_node ? nullptr : group->ht_buffers.rdma_inter_node_group_flags;
        params.expected_intra_node_flag_value = group->ht_buffers.expected_intra_node_flag_value;
        params.intra_node_write_completion_flags = group->ht_buffers.intra_node_write_completion_flags;
        // Pass device communicators and windows
        params.dcomms = is_single_node ? nullptr : group->gin_config.d_dcomms;
        params.nccl_windows = is_single_node ? nullptr : group->gin_config.d_nccl_windows;
        params.num_gin_comms = is_single_node ? 0 : group->gin_config.num_comms;
        params.num_ctx_per_comm = is_single_node ? 0 : group->gin_config.num_ctx_per_comm;
        params.gin_base_ptr = is_single_node ? nullptr : group->gin_config.gin_base_ptr;
        params.signals_base = group->gin_config.signals_base;
        // Use offsets relative to gin_base_ptr
        // All buffers are part of one large registered window
        // Calculate bytes_per_entry for batched staging
        size_t bytes_per_token_entry = group->hidden * sizeof(uint16_t);  // token data
        size_t bytes_per_prob_entry = (group->num_local_experts * group->nvl_rank_count) * sizeof(float);  // prob data
        size_t bytes_per_sf_entry = (group->hidden / 128) * sizeof(float);  // scaling factor (FP8)
        size_t bytes_per_entry = bytes_per_token_entry + bytes_per_prob_entry + bytes_per_sf_entry;

        params.mr_info = {
            .attn_input_token_offset = is_single_node ? 0 : group->gin_config.token_staging_offset,
            .rdma_inter_node_group_token_offset = is_single_node ? 0 : group->gin_config.rdma_inter_node_group_token_offset,
            .attn_input_prob_offset = is_single_node ? 0 : group->gin_config.dense_prob_offset,
            .rdma_inter_node_group_prob_offset = is_single_node ? 0 : group->gin_config.rdma_inter_node_group_prob_offset,
            .attn_input_scaling_factor_offset = is_single_node ? 0 : group->gin_config.scaling_factor_staging_offset,
            .rdma_inter_node_group_scaling_factor_offset = is_single_node ? 0 : group->gin_config.rdma_inter_node_group_scaling_factor_offset,
            // Batched staging parameters
            .rdma_send_staging_offset = is_single_node ? 0 : group->gin_config.rdma_send_staging_offset,
            .rdma_inter_node_group_packed_offset = is_single_node ? 0 : group->gin_config.rdma_inter_node_group_packed_offset,
            .rdma_batch_size = is_single_node ? 0 : group->gin_config.rdma_batch_size,
            .bytes_per_entry = bytes_per_entry,
            .max_tokens_per_dest = static_cast<size_t>(group->config.max_tokens_per_rank),
            // Streaming signal parameters
            .signals_tail_base = is_single_node ? 0 : static_cast<unsigned>(group->gin_config.signals_tail_base),
            .num_max_rdma_chunked_send_tokens = is_single_node ? 0 : group->gin_config.num_max_rdma_chunked_send_tokens,
        };
        params.local_rank = group->local_nvl_rank;
        params.node_rank = group->rdma_rank;
        params.num_tokens_per_rank = handle->num_tokens;

        // Call dispatch kernel
        nccl_ep::hybridep::call_dispatch(
            params,
            group->config.max_tokens_per_rank,
            group->nNodes,
            use_fp8,
            forward_dispatch,
            stream
        );

        /* ===== Copy IPC staging → caller outputs ===== */
        // HT kernel writes to IPC-mapped buffers (dispatch_expert_output_*_buffer_ptrs)
        // Copy results to user-provided output tensors
        ncclNDTensor_t* recv_x = find_tensor_by_tag(outputs, num_outputs, NCCL_EP_TENSOR_TAG_TOKENS);
        if (recv_x != nullptr) {
            assert(recv_x->ndim == 2 && tensor_is_contiguous(recv_x));
            size_t copy_size = static_cast<size_t>(recv_x->sizes[0]) * recv_x->sizes[1] * ncclTypeSize(recv_x->datatype);

            CUDA_CHECK(cudaMemcpyAsync(recv_x->data,
                group->ht_buffers.dispatch_expert_output_token_buffer_ptrs[group->local_nvl_rank],
                copy_size,
                cudaMemcpyDeviceToDevice,
                stream));
        }

        /* ===== Convert dense output → sparse format ===== */
         if (forward_dispatch) {
             // Convert outputs - both recv_topk_weights and recv_topk_idx must be provided together
             if (recv_topk_weights != nullptr && recv_topk_idx != nullptr) {
                 // Validate formats for both outputs
            assert(recv_topk_weights->ndim == 2 && tensor_is_contiguous(recv_topk_weights));
            assert(recv_topk_weights->datatype == ncclFloat32);
            assert(recv_topk_idx->ndim == 2 && tensor_is_contiguous(recv_topk_idx));
            assert(recv_topk_idx->datatype == ncclInt64);
            assert(recv_topk_weights->sizes[0] == recv_topk_idx->sizes[0]);

            int num_recv_tokens = static_cast<int>(recv_topk_weights->sizes[0]);
            int experts_per_node = group->num_local_experts * group->nvl_rank_count;

            nccl_ep::hybridep::dense_to_sparse_prob(
                group->ht_buffers.dispatch_expert_output_prob_buffer_ptrs[group->local_nvl_rank],
                handle->hybridep.local_expert_routing_map,
                static_cast<float*>(recv_topk_weights->data),
                static_cast<int64_t*>(recv_topk_idx->data),
                num_recv_tokens,
                handle->num_topk,
                group->num_local_experts,
                experts_per_node,
                group->local_nvl_rank,
                stream);
             } else {
                 // Both outputs must be provided together or neither
                 assert(recv_topk_weights == nullptr && recv_topk_idx == nullptr);
             }
        }

        // Copy FP8 scales output
        if (use_fp8) {
            ncclNDTensor_t* recv_scales = find_tensor_by_tag(outputs, num_outputs, NCCL_EP_TENSOR_TAG_SCALES);
            if (recv_scales != nullptr) {
                assert(recv_scales->ndim == 2 && tensor_is_contiguous(recv_scales));
                size_t copy_size = static_cast<size_t>(recv_scales->sizes[0]) * recv_scales->sizes[1] * ncclTypeSize(recv_scales->datatype);

                CUDA_CHECK(cudaMemcpyAsync(recv_scales->data,
                    group->ht_buffers.dispatch_expert_output_scaling_factor_buffer_ptrs[group->local_nvl_rank],
                    copy_size,
                    cudaMemcpyDeviceToDevice,
                    stream));
            }
        }
    }
    return ncclSuccess;
}

ncclResult_t ncclEpCombine(
    ncclEpHandle_t handle,
    const ncclNDTensor_t* const* inputs,
    unsigned int num_inputs,
    ncclNDTensor_t* const* outputs,
    unsigned int num_outputs,
    ncclNDTensor_t* const* local_tensors,
    unsigned int num_local_tensors,
    unsigned int send_only,
    const ncclEpCombineConfig_t* config,
    cudaStream_t stream
) {
    if (handle->group->config.algorithm == NCCL_EP_ALGO_LOW_LATENCY) {
        // Find and validate input tensors
        const ncclNDTensor_t* x = find_tensor_by_tag(
            inputs, num_inputs, NCCL_EP_TENSOR_TAG_TOKENS
        );
            assert(x != nullptr);

        const ncclNDTensor_t* topk_idx = handle->topk_idx;
        const ncclNDTensor_t* src_info = &handle->ll.expert_recv_source_indices;
        const ncclNDTensor_t* layout_range = &handle->ll.expert_dispatch_layout;

        // Find and validate local tensors
        const ncclNDTensor_t* topk_weights = find_tensor_by_tag(
            local_tensors, num_local_tensors, NCCL_EP_TENSOR_TAG_TOPK_WEIGHTS
        );
            assert(topk_weights != nullptr);

        // Extract configuration values
        const int num_experts = handle->group->config.num_experts;
        const int num_ranks = handle->group->nRanks;
        const int num_max_dispatch_tokens_per_rank = handle->group->config.max_tokens_per_rank;

        // Validate input tensor x
        assert(x->ndim == 3);
        assert(tensor_is_contiguous(x));
        assert(x->datatype == ncclBfloat16);
            assert(x->sizes[0] == num_experts / num_ranks);
            assert(x->sizes[1] == num_ranks * num_max_dispatch_tokens_per_rank);
        assert(x->sizes[2] % sizeof(int4) == 0);
        assert(x->sizes[2] % 128 == 0);

            // Validate topk_idx tensor
        assert(topk_idx->ndim == 2);
        assert(tensor_is_contiguous(topk_idx));
        assert(topk_idx->datatype == ncclInt64);

            // Validate src_info tensor
        assert(src_info->ndim == 2);
        assert(tensor_is_contiguous(src_info));
        assert(src_info->datatype == ncclInt32);
        assert(x->sizes[0] == src_info->sizes[0]);

            // Validate topk_weights tensor
        assert(topk_weights->ndim == 2);
        assert(tensor_is_contiguous(topk_weights));
        assert(topk_weights->sizes[0] == topk_idx->sizes[0]);
        assert(topk_weights->sizes[1] == topk_idx->sizes[1]);
        assert(topk_weights->datatype == ncclFloat32);
        assert(topk_weights->sizes[0] <= num_max_dispatch_tokens_per_rank);

        // Extract dimensions
        const int hidden = static_cast<int>(x->sizes[2]);
        const int num_topk = static_cast<int>(topk_weights->sizes[1]);
        const int num_combined_tokens = static_cast<int>(topk_weights->sizes[0]);

        // Manage double-buffering
        const auto& buffer = handle->ll.layout.buffers[handle->ll.buffer_idx];
        auto& next_buffer = handle->ll.layout.buffers[handle->ll.buffer_idx ^= 1];
        const auto next_clean_meta = next_buffer.clean_meta();

        // Validate buffer layout
        assert(handle->ll.layout.total_bytes <= handle->group->config.rdma_buffer_size);

        // Find and validate output tensor
        const ncclNDTensor_t* out = find_tensor_by_tag(
            outputs, num_outputs, NCCL_EP_TENSOR_TAG_TOKENS
        );

        assert(out != nullptr);
        assert(out->ndim == 2);
        assert(tensor_is_contiguous(out));
        assert(out->sizes[0] == num_combined_tokens);
        assert(out->sizes[1] == hidden);
        assert(out->datatype == x->datatype);

        // Define combine lambda
        unsigned signal_base = handle->ll.buffer_idx * (handle->group->num_dispatch_signals / 2);
        auto combine_fn = [=](int phases) {
            // Prepare data pointers
            auto* out_data = out->data;
            auto* x_data = x->data;
            auto* topk_idx_data = static_cast<int64_t*>(topk_idx->data);
            auto* topk_weights_data = static_cast<float*>(topk_weights->data);
            auto* src_info_data = static_cast<int*>(src_info->data);
            auto* layout_range_data = static_cast<int64_t*>(layout_range->data);

            const bool use_fp8 = false;
            const bool zero_copy = false;

            nccl_ep::internode_ll::combine(
                x_data,
                src_info_data,
                layout_range_data,
                topk_idx_data,
                topk_weights_data,
                out_data,
                buffer.combine_rdma_send_buffer,
                buffer.combine_rdma_recv_data_buffer,
                buffer.combine_rdma_recv_flag_buffer,
                (reinterpret_cast<uint64_t>(buffer.combine_rdma_send_buffer) - reinterpret_cast<uint64_t>(handle->group->rdma_buffer)),
                (reinterpret_cast<uint64_t>(buffer.combine_rdma_recv_data_buffer) - reinterpret_cast<uint64_t>(handle->group->rdma_buffer)),
                (reinterpret_cast<uint64_t>(buffer.combine_rdma_recv_flag_buffer) - reinterpret_cast<uint64_t>(handle->group->rdma_buffer)),
                next_clean_meta.first,
                next_clean_meta.second,
                nullptr, /*recv_topk_idx=*/
                num_combined_tokens,
                hidden,
                num_max_dispatch_tokens_per_rank,
                num_topk,
                num_experts,
                handle->group->rank,
                handle->group->nRanks,
                use_fp8,
                phases,
                zero_copy,
                handle->group->num_nccl_comms,
                handle->group->nccl_dev_comms,
                handle->group->nccl_wins,
                signal_base,
                handle->group->ep_workspace,
                handle->group->device_sm_count,
                stream
            );
        };

        // Execute combine with appropriate phase flags
        const int combine_phases = send_only ?
            LOW_LATENCY_SEND_PHASE : (LOW_LATENCY_SEND_PHASE | LOW_LATENCY_RECV_PHASE);
        combine_fn(combine_phases);

        if (send_only) {
                handle->ll.continue_fn = combine_fn;
            }
    } else if (handle->group->config.algorithm == NCCL_EP_ALGO_HIGH_THROUGHPUT) {
        // ===== HT mode =====
        // Combine: gather expert outputs back to original token positions
        // Forward combine: just tokens (inference)
        // Backward combine: tokens + gradients (training, topk_weights provided)
        ncclEpGroup_t group = handle->group;
        bool is_single_node = !is_internode_available(group);
             //assert(is_single_node && "HT mode only supports single-node");

        /* ===== Inputs validation ===== */
        const ncclNDTensor_t* x = find_tensor_by_tag(inputs, num_inputs, NCCL_EP_TENSOR_TAG_TOKENS);
        assert(x != nullptr);
        assert(x->ndim == 2 && tensor_is_contiguous(x));
        assert(x->datatype == ncclBfloat16); // HT combine only supports BF16

        // Get dimensions from input tensor
        auto num_tokens = static_cast<int>(x->sizes[0]);
        auto hidden = static_cast<int>(x->sizes[1]);

        // Validate int4 alignment for TMA
        assert((hidden * ncclTypeSize(x->datatype)) % sizeof(int4) == 0);

        // Number of tokens to combine back (original token count from this rank)
        auto num_combined_tokens = handle->num_tokens;

        // Top-k checks (for backward mode)
        int num_topk = 0;
        const ncclNDTensor_t* topk_weights = find_tensor_by_tag(inputs, num_inputs, NCCL_EP_TENSOR_TAG_TOPK_WEIGHTS);
        ncclNDTensor_t* combined_topk_weights = find_tensor_by_tag(outputs, num_outputs, NCCL_EP_TENSOR_TAG_TOPK_WEIGHTS);

        // Determine if this is backward mode (topk_weights provided = backward combine)
        bool backward_combine = (topk_weights != nullptr);

        if (backward_combine) {
            num_topk = static_cast<int>(topk_weights->sizes[1]);

            assert(combined_topk_weights != nullptr);
            assert(combined_topk_weights->ndim == 2 && tensor_is_contiguous(combined_topk_weights));
            assert(combined_topk_weights->sizes[0] == num_combined_tokens);
            assert(combined_topk_weights->sizes[1] == num_topk);
            assert(combined_topk_weights->datatype == ncclFloat32);
        }

        // HT combine doesn't use local tensors, so hybridep should not use local tensors either
        assert(num_local_tensors == 0);

        /* ===== Output tensors ===== */
        ncclNDTensor_t* combined_x = find_tensor_by_tag(outputs, num_outputs, NCCL_EP_TENSOR_TAG_TOKENS);
        assert(combined_x != nullptr);
        assert(combined_x->ndim == 2 && tensor_is_contiguous(combined_x));
        assert(combined_x->sizes[0] == num_combined_tokens); // Output should match original token count
        assert(combined_x->sizes[1] == hidden);              // Should match input hidden dimension

        /* ===== Copy input to IPC staging buffers ===== */
        // Expert MLP output needs to be in IPC buffer so other ranks can read it
        size_t token_copy_size = static_cast<size_t>(num_tokens) * hidden * sizeof(uint16_t); // BF16 = uint16_t
        CUDA_CHECK(cudaMemcpyAsync(
            group->ht_buffers.expert_input_token,
            x->data,
            token_copy_size,
            cudaMemcpyDeviceToDevice,
            stream));

        /* ===== Convert sparse topk_weights to dense prob for backward combine ===== */
        // For backward combine, convert sparse input weights to dense format for HT kernel
        if (backward_combine) {
            int experts_per_node = group->num_local_experts * group->nvl_rank_count;
            size_t dense_prob_size = static_cast<size_t>(num_tokens) * experts_per_node * sizeof(float);

            // Zero-initialize the dense prob buffer before scattering
            CUDA_CHECK(cudaMemsetAsync(
                group->ht_buffers.combine_expert_input_prob_buffer_ptrs[group->local_nvl_rank],
                0, dense_prob_size, stream));

            // Convert sparse [num_tokens, topk] to dense [num_tokens, experts_per_node]
            // Uses local_expert_routing_map to determine expert positions (matches dispatch output order)
            nccl_ep::hybridep::sparse_to_dense_prob_combine(
                static_cast<const float*>(topk_weights->data),
                handle->hybridep.local_expert_routing_map,
                group->ht_buffers.combine_expert_input_prob_buffer_ptrs[group->local_nvl_rank],
                num_tokens,
                num_topk,
                group->num_local_experts, // experts_per_rank
                experts_per_node,
                group->local_nvl_rank,
                stream);
        }

        // Use pre-allocated dense prob buffer for backward combine
        float* dense_output_prob = handle->hybridep.dense_prob_buffer;
        if (backward_combine) {
            size_t dense_output_prob_size = static_cast<size_t>(num_combined_tokens) * group->config.num_experts * sizeof(float);
            CUDA_CHECK(cudaMemsetAsync(dense_output_prob, 0, dense_output_prob_size, stream));
        }

        /* ===== Build CombineParams ===== */
        // CombineParams encapsulates all buffers and metadata needed by HT combine kernel:
        //   - IPC input buffers: expert_input_token_ptrs, expert_input_prob_ptrs (per-rank pointers)
        //   - Output buffers: attn_output_token, attn_output_prob (user-provided)
        //   - RDMA buffers: rdma_intra_node_red_*, combine_rdma_inter_node_group_* (for multi-node)
        //   - Metadata: sparse_to_dense_map, rdma_to_attn_map, attn_to_rdma_map, local_expert_routing_map
        //   - Sync flags: combine_expected_*_flag_value, combine_intra_node_write_completion_flags
        nccl_ep::hybridep::CombineParams params;
        params.hidden_dim = group->hidden;
        params.experts_per_rank = group->num_local_experts;
        params.num_ranks_per_node = group->nvl_rank_count;
        // Use HOST pointer arrays - these get copied into the kernel param struct for fast __grid_constant__ access
        params.expert_input_token_ptrs = group->ht_buffers.combine_expert_input_token_buffer_ptrs;
        params.expert_input_prob_ptrs = backward_combine ? group->ht_buffers.combine_expert_input_prob_buffer_ptrs : nullptr;
        params.attn_output_token = combined_x->data;
        params.attn_output_prob = backward_combine ? dense_output_prob : nullptr;
        params.rdma_intra_node_red_token = is_single_node ? nullptr : group->ht_buffers.rdma_intra_node_red_token;
        params.rdma_intra_node_red_prob = (!is_single_node && backward_combine) ? group->ht_buffers.rdma_intra_node_red_prob : nullptr;
        params.combine_rdma_inter_node_group_token = is_single_node ? nullptr : group->ht_buffers.combine_rdma_inter_node_group_token;
        params.combine_rdma_inter_node_group_prob = (!is_single_node && backward_combine) ? group->ht_buffers.combine_rdma_inter_node_group_prob : nullptr;
        params.sparse_to_dense_map = handle->hybridep.sparse_to_dense_map;
        params.rdma_to_attn_map = handle->hybridep.rdma_to_attn_map;
        params.attn_to_rdma_map = handle->hybridep.attn_to_rdma_map;
        params.local_expert_routing_map = handle->hybridep.local_expert_routing_map;
        params.combine_expected_rdma_flag_value = is_single_node ? nullptr : group->ht_buffers.combine_expected_rdma_flag_value;
        params.combine_rdma_inter_node_group_flags = is_single_node ? nullptr : group->ht_buffers.combine_rdma_inter_node_group_flags;
        params.combine_expected_intra_node_flag_value = group->ht_buffers.combine_expected_intra_node_flag_value;
        params.combine_intra_node_write_completion_flags = group->ht_buffers.combine_intra_node_write_completion_flags;
        // Pass device communicators and windows
        params.dcomms = is_single_node ? nullptr : group->gin_config.d_dcomms;
        params.nccl_windows = is_single_node ? nullptr : group->gin_config.d_nccl_windows;
        params.num_gin_comms = is_single_node ? 0 : group->gin_config.num_comms;
        params.num_ctx_per_comm = is_single_node ? 0 : group->gin_config.num_ctx_per_comm;
        params.gin_base_ptr = is_single_node ? nullptr : group->gin_config.gin_base_ptr;
        params.signals_base = group->gin_config.signals_base;
        params.combine_signal_offset = group->gin_config.combine_signal_offset;
        // Use offsets relative to gin_base_ptr
        params.mr_info = {
            .rdma_intra_node_red_token_offset = is_single_node ? 0 : group->gin_config.rdma_intra_node_red_token_offset,
            .combine_rdma_inter_node_group_token_offset = is_single_node ? 0 : group->gin_config.combine_rdma_inter_node_group_token_offset,
            .rdma_intra_node_red_prob_offset = is_single_node ? 0 : group->gin_config.rdma_intra_node_red_prob_offset,
            .combine_rdma_inter_node_group_prob_offset = is_single_node ? 0 : group->gin_config.combine_rdma_inter_node_group_prob_offset,
        };
        params.local_rank = group->local_nvl_rank;
        params.node_rank = group->rdma_rank;
        params.num_tokens_per_rank = num_combined_tokens;
        params.num_recv_tokens = num_tokens;

        /* ===== Call combine kernel ===== */
        nccl_ep::hybridep::call_combine(
            params,
            group->config.max_tokens_per_rank, // max_tokens_per_rank
            group->nNodes, // num_nodes (RDMA domain size)
            backward_combine, // backward mode flag
            stream
        );

        /* ===== Convert dense output prob to sparse format ===== */
        // For backward combine, convert kernel's dense output to sparse format
        // HT outputs dense [num_tokens, num_experts], NCCL expects sparse [num_tokens, topk]
        if (backward_combine && combined_topk_weights != nullptr) {
            nccl_ep::hybridep::dense_to_sparse_prob_combine(
                dense_output_prob,
                handle->hybridep.local_routing_map,
                static_cast<float*>(combined_topk_weights->data),
                nullptr,  // No need to output topk_idx for combine
                num_combined_tokens,
                num_topk,
                group->config.num_experts,
                stream);
        }

        handle->cached_mode = true;
    }

    return ncclSuccess;
}

ncclResult_t ncclEpComplete(
        ncclEpHandle_t handle,
    const ncclEpCompleteConfig_t* config,
    cudaStream_t stream
) {
    if (handle->group->config.algorithm == NCCL_EP_ALGO_LOW_LATENCY) {
        if (handle->ll.continue_fn) {
                handle->ll.continue_fn(LOW_LATENCY_RECV_PHASE);
                handle->ll.continue_fn = nullptr;
        }
    } else if (handle->group->config.algorithm == NCCL_EP_ALGO_HIGH_THROUGHPUT) {
        // HT mode - no continue needed (synchronous)
        }
        return ncclSuccess;
    }

ncclResult_t ncclEpHandleGetNumRecvTokens(
        ncclEpHandle_t handle,
    unsigned int* num_recv_tokens
) {
    if (handle->group->config.algorithm == NCCL_EP_ALGO_HIGH_THROUGHPUT) {
        // HT does not support dynamic token count, requires max_tokens_per_rank to be set
        // Max receive count = nRanks * max_tokens_per_rank (worst case: all ranks route to one).
        *num_recv_tokens = static_cast<unsigned int>(handle->group->nRanks) *
                           static_cast<unsigned int>(handle->group->config.max_tokens_per_rank);
    } else { // LL
        return ncclInvalidUsage;
    }
    return ncclSuccess;
}
