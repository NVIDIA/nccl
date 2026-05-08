/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 * See LICENSE.txt for more license information.
 */
// Throughput and validation methodology aligned with DeepEP (https://github.com/deepseek-ai/DeepEP).

#include <getopt.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <algorithm>
#include <cmath>
#include <functional>
#include <map>
#include <numeric>
#include <random>
#include <set>
#include <string>
#include <vector>
#include <mpi.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <cupti.h>
#include <nvtx3/nvToolsExt.h>
#include <nccl.h>
#include <nccl_device.h>
#include "nccl_ep.h"


#define MPICHECK(cmd) do {                          \
  int e = cmd;                                      \
  if( e != MPI_SUCCESS ) {                          \
    printf("Failed: MPI error %s:%d '%d'\n",        \
        __FILE__,__LINE__, e);   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

#define CUDACHECK(cmd) do {                         \
  cudaError_t e = cmd;                              \
  if( e != cudaSuccess ) {                          \
    printf("Failed: Cuda error %s:%d '%s'\n",             \
        __FILE__,__LINE__,cudaGetErrorString(e));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

#define NCCLCHECK(cmd) do {                         \
  ncclResult_t r = cmd;                             \
  if (r!= ncclSuccess) {                            \
    printf("Failed: NCCL error %s:%d '%s'\n",             \
        __FILE__,__LINE__,ncclGetErrorString(r));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

// ============================================================================
// KernelTimer: CUPTI Activity API-based per-kernel GPU timing
// ============================================================================
// Records per-kernel GPU execution times by matching kernel name substrings.
// Entirely benchmark-side — zero impact on the production nccl_ep library.
// Same mechanism used by PyTorch kineto (torch.profiler).

#define CUPTI_CALL(call) do {                                                  \
    CUptiResult _s = (call);                                                   \
    if (_s != CUPTI_SUCCESS) {                                                 \
        const char* _e; cuptiGetResultString(_s, &_e);                        \
        fprintf(stderr, "CUPTI error %s:%d: %s\n", __FILE__, __LINE__, _e);   \
    }                                                                          \
} while (0)

static const size_t CUPTI_BUF_SIZE = 8 * 1024 * 1024;  // 8 MB per buffer

struct KernelStat { uint64_t total_ns = 0; int count = 0; };
// Global accumulator populated by CUPTI buffer-completed callback
static std::map<std::string, KernelStat> g_kernel_stats;

static void CUPTIAPI cuptiBufferRequested(uint8_t** buf, size_t* sz, size_t* maxRecords) {
    // aligned_alloc requires size to be a multiple of alignment
    *buf = static_cast<uint8_t*>(aligned_alloc(8, CUPTI_BUF_SIZE));
    *sz = CUPTI_BUF_SIZE;
    *maxRecords = 0;
}

static void CUPTIAPI cuptiBufferCompleted(CUcontext /*ctx*/, uint32_t /*streamId*/,
                                           uint8_t* buf, size_t /*sz*/, size_t validSz) {
    CUpti_Activity* record = nullptr;
    while (cuptiActivityGetNextRecord(buf, validSz, &record) == CUPTI_SUCCESS) {
        if (record->kind == CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL) {
            auto* k = reinterpret_cast<CUpti_ActivityKernel5*>(record);
            if (k->name) {
                g_kernel_stats[k->name].total_ns += k->end - k->start;
                g_kernel_stats[k->name].count++;
            }
        }
    }
    free(buf);
}

class KernelTimer {
public:
    // Enable CUPTI kernel activity recording and clear accumulated stats.
    void start() {
        g_kernel_stats.clear();
        CUPTI_CALL(cuptiActivityRegisterCallbacks(cuptiBufferRequested, cuptiBufferCompleted));
        CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));
    }

    // Flush all pending CUPTI buffers and disable recording.
    void stop() {
        CUPTI_CALL(cuptiActivityFlushAll(0));
        CUPTI_CALL(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));
    }

    // Average GPU execution time (microseconds) across all kernels whose
    // mangled name contains substr.  Returns 0 if no matching kernel found.
    double get_avg_us(const char* substr) const {
        uint64_t total_ns = 0; int count = 0;
        for (const auto& kv : g_kernel_stats) {
            if (kv.first.find(substr) != std::string::npos) {
                total_ns += kv.second.total_ns;
                count    += kv.second.count;
            }
        }
        return count ? static_cast<double>(total_ns) / count / 1000.0 : 0.0;
    }

    // Print all captured kernel names and their stats to stdout (debug helper).
    void dump(int rank) const {
        if (rank != 0) return;
        printf("[KernelTimer] Captured %zu distinct kernel(s):\n", g_kernel_stats.size());
        for (const auto& kv : g_kernel_stats) {
            double avg_us = static_cast<double>(kv.second.total_ns) / kv.second.count / 1000.0;
            printf("  count=%3d  avg=%.2f us  %s\n", kv.second.count, avg_us, kv.first.c_str());
        }
        fflush(stdout);
    }
};

static uint64_t getHostHash(const char* string) {
  uint64_t result = 5381;
  for (int c = 0; string[c] != '\0'; c++){
    result = ((result << 5) + result) + string[c];
  }
  return result;
}

static void getHostName(char* hostname, int maxlen) {
  gethostname(hostname, maxlen);
  for (int i=0; i< maxlen; i++) {
    if (hostname[i] == '.') {
        hostname[i] = '\0';
        return;
    }
  }
}

// CUDA allocator callbacks for ncclEpCreateGroup
// These are used by ncclEpTensorCreate/Destroy to allocate/free tensor memory
static cudaError_t cudaAllocCallback(void** ptr, size_t size) {
    return cudaMalloc(ptr, size);
}

static cudaError_t cudaFreeCallback(void* ptr) {
    return cudaFree(ptr);
}

// Structure to hold all tensors needed for benchmarking
struct BenchmarkTensors {
    // Dispatch tensors
    ncclNDTensor_t inputs[3];
    ncclNDTensor_t outputs[3];
    ncclNDTensor_t local_tensors[1];
    int num_dispatch_inputs;
    int num_dispatch_outputs;

    // Combine tensors
    ncclNDTensor_t combine_inputs[2];
    ncclNDTensor_t combine_outputs[2];
    ncclNDTensor_t combine_local_tensors[1];
    int num_combine_inputs;
    int num_combine_outputs;
    int num_combine_local_tensors;

    // Owned tensors (for cleanup)
    ncclNDTensor_t dispatch_topk_weights;
    ncclNDTensor_t expert_outputs;
    ncclNDTensor_t combined_output;
    ncclNDTensor_t topk_weights;
    ncclNDTensor_t combine_output_topk_weights;

    bool is_ll_mode;
};

// Setup tensors for LOW_LATENCY mode using ncclEpTensorCreate
void setupLowLatencyTensors(
    ncclEpGroup_t ep_group,
    BenchmarkTensors& tensors,
    ncclNDTensor_t& topk_idx,
    unsigned int num_tokens,
    unsigned int hidden,
    unsigned int top_k,
    unsigned int num_local_experts,
    unsigned int max_tokens_per_rank,
    int nRanks
) {
    tensors.is_ll_mode = true;
    tensors.num_dispatch_inputs = 1;
    tensors.num_dispatch_outputs = 1;
    tensors.num_combine_inputs = 1;
    tensors.num_combine_outputs = 1;
    tensors.num_combine_local_tensors = 1;

    // Dispatch input: tokens
    NCCLCHECK(ncclEpTensorCreate(ep_group, &tensors.inputs[0], 2, ncclBfloat16,
                                   NCCL_EP_TENSOR_TAG_TOKENS,
                                   nullptr, num_tokens, hidden));

    // Dispatch output: 3D [num_local_experts, max_tokens_per_rank * nRanks, hidden]
    NCCLCHECK(ncclEpTensorCreate(ep_group, &tensors.outputs[0], 3, ncclBfloat16,
                                   NCCL_EP_TENSOR_TAG_TOKENS,
                                   nullptr, num_local_experts, max_tokens_per_rank * nRanks, hidden));

    // Local tensors: recv expert counter (device memory) - required for dispatch
    NCCLCHECK(ncclEpTensorCreate(ep_group, &tensors.local_tensors[0], 1, ncclInt32,
                                   NCCL_EP_TENSOR_TAG_RECV_EXPERT_COUNTER_DEVICE,
                                   nullptr, num_local_experts));

    // Combine input: 3D expert outputs
    NCCLCHECK(ncclEpTensorCreate(ep_group, &tensors.expert_outputs, 3, ncclBfloat16,
                                   NCCL_EP_TENSOR_TAG_TOKENS,
                                   nullptr, num_local_experts, max_tokens_per_rank * nRanks, hidden));

    // Combine output
    NCCLCHECK(ncclEpTensorCreate(ep_group, &tensors.combined_output, 2, ncclBfloat16,
                                   NCCL_EP_TENSOR_TAG_TOKENS,
                                   nullptr, num_tokens, hidden));

    // topk_weights as local tensor for combine
    NCCLCHECK(ncclEpTensorCreate(ep_group, &tensors.topk_weights, 2, ncclFloat32,
                                   NCCL_EP_TENSOR_TAG_TOPK_WEIGHTS,
                                   nullptr, num_tokens, top_k));

    // Setup combine arrays
    tensors.combine_inputs[0] = tensors.expert_outputs;
    tensors.combine_outputs[0] = tensors.combined_output;
    tensors.combine_local_tensors[0] = tensors.topk_weights;
}

// Setup tensors for HIGH_THROUGHPUT mode using ncclEpTensorCreate
void setupHighThroughputTensors(
    ncclEpGroup_t ep_group,
    BenchmarkTensors& tensors,
    ncclNDTensor_t& topk_idx,
    unsigned int num_tokens,
    unsigned int hidden,
    unsigned int top_k,
    unsigned int num_local_experts,
    unsigned int num_recv_tokens
) {
    tensors.is_ll_mode = false;
    tensors.num_dispatch_inputs = 3;
    tensors.num_dispatch_outputs = 3;
    // HT combine uses only 1 input (expert_outputs) and 1 output (combined_output)
    tensors.num_combine_inputs = 1;
    tensors.num_combine_outputs = 1;
    tensors.num_combine_local_tensors = 0;

    // Dispatch input: tokens - initialize with test pattern
    NCCLCHECK(ncclEpTensorCreate(ep_group, &tensors.inputs[0], 2, ncclBfloat16,
                                   NCCL_EP_TENSOR_TAG_TOKENS,
                                   nullptr, num_tokens, hidden));
    {
        void* input0_data;
        NCCLCHECK(ncclEpTensorGetData(tensors.inputs[0], &input0_data));
        CUDACHECK(cudaMemset(input0_data, 0, num_tokens * hidden * 2));
    }

    // Dispatch input: topk_weights - initialize with equal weights
    NCCLCHECK(ncclEpTensorCreate(ep_group, &tensors.dispatch_topk_weights, 2, ncclFloat32,
                                   NCCL_EP_TENSOR_TAG_TOPK_WEIGHTS,
                                   nullptr, num_tokens, top_k));
    {
        float *topk_weights_host = new float[num_tokens * top_k];
        for (unsigned int i = 0; i < num_tokens * top_k; i++) {
            topk_weights_host[i] = 1.0f / top_k;
        }
        void* dtw_data;
        NCCLCHECK(ncclEpTensorGetData(tensors.dispatch_topk_weights, &dtw_data));
        CUDACHECK(cudaMemcpy(dtw_data, topk_weights_host,
                             num_tokens * top_k * sizeof(float), cudaMemcpyHostToDevice));
        delete[] topk_weights_host;
    }
    tensors.inputs[1] = tensors.dispatch_topk_weights;

    tensors.inputs[2] = topk_idx;

    // Dispatch output: 2D [num_recv_tokens, hidden]
    NCCLCHECK(ncclEpTensorCreate(ep_group, &tensors.outputs[0], 2, ncclBfloat16,
                                   NCCL_EP_TENSOR_TAG_TOKENS,
                                   nullptr, num_recv_tokens, hidden));

    // Dispatch output: recv_topk_weights
    NCCLCHECK(ncclEpTensorCreate(ep_group, &tensors.outputs[1], 2, ncclFloat32,
                                   NCCL_EP_TENSOR_TAG_TOPK_WEIGHTS,
                                   nullptr, num_recv_tokens, top_k));

    // Dispatch output: recv_topk_idx
    NCCLCHECK(ncclEpTensorCreate(ep_group, &tensors.outputs[2], 2, ncclInt64,
                                   NCCL_EP_TENSOR_TAG_TOPK_IDX,
                                   nullptr, num_recv_tokens, top_k));

    // Local tensors: recv expert counter (device memory) - required for dispatch
    NCCLCHECK(ncclEpTensorCreate(ep_group, &tensors.local_tensors[0], 1, ncclInt32,
                                   NCCL_EP_TENSOR_TAG_RECV_EXPERT_COUNTER_DEVICE,
                                   nullptr, num_local_experts));

    // Combine input: 2D expert outputs - same size as dispatch output (received token count)
    NCCLCHECK(ncclEpTensorCreate(ep_group, &tensors.expert_outputs, 2, ncclBfloat16,
                                   NCCL_EP_TENSOR_TAG_TOKENS,
                                   nullptr, num_recv_tokens, hidden));
    {
        void* eo_data;
        NCCLCHECK(ncclEpTensorGetData(tensors.expert_outputs, &eo_data));
        CUDACHECK(cudaMemset(eo_data, 0, num_recv_tokens * hidden * 2));
    }

    // Combine output - sized to num_tokens (original token count per rank)
    NCCLCHECK(ncclEpTensorCreate(ep_group, &tensors.combined_output, 2, ncclBfloat16,
                                   NCCL_EP_TENSOR_TAG_TOKENS,
                                   nullptr, num_tokens, hidden));

    // topk_weights as regular input for combine
    NCCLCHECK(ncclEpTensorCreate(ep_group, &tensors.topk_weights, 2, ncclFloat32,
                                   NCCL_EP_TENSOR_TAG_TOPK_WEIGHTS,
                                   nullptr, num_tokens, top_k));

    // Combine output: topk_weights
    NCCLCHECK(ncclEpTensorCreate(ep_group, &tensors.combine_output_topk_weights, 2, ncclFloat32,
                                   NCCL_EP_TENSOR_TAG_TOPK_WEIGHTS,
                                   nullptr, num_tokens, top_k));

    // Setup combine arrays
    tensors.combine_inputs[0] = tensors.expert_outputs;
    tensors.combine_inputs[1] = tensors.topk_weights;
    tensors.combine_outputs[0] = tensors.combined_output;
    tensors.combine_outputs[1] = tensors.combine_output_topk_weights;
}

// Cleanup benchmark tensors using ncclEpTensorDestroy
void cleanupBenchmarkTensors(ncclEpGroup_t ep_group, BenchmarkTensors& tensors, ncclNDTensor_t topk_idx) {
    // topk_idx is created with ncclEpTensorCreate (user-provided data_ptr)
    {
        void* topk_data;
        ncclEpTensorGetData(topk_idx, &topk_data);
        if (topk_data) cudaFree(topk_data);
        ncclEpTensorDestroy(ep_group, topk_idx);
    }

    // All other tensors are created with ncclEpTensorCreate
    ncclEpTensorDestroy(ep_group, tensors.inputs[0]);

    if (!tensors.is_ll_mode) {
        ncclEpTensorDestroy(ep_group, tensors.dispatch_topk_weights);
    }

    ncclEpTensorDestroy(ep_group, tensors.outputs[0]);

    if (!tensors.is_ll_mode) {
        ncclEpTensorDestroy(ep_group, tensors.outputs[1]);
        ncclEpTensorDestroy(ep_group, tensors.outputs[2]);
    }

    ncclEpTensorDestroy(ep_group, tensors.local_tensors[0]);
    ncclEpTensorDestroy(ep_group, tensors.expert_outputs);
    ncclEpTensorDestroy(ep_group, tensors.combined_output);
    ncclEpTensorDestroy(ep_group, tensors.topk_weights);

    if (!tensors.is_ll_mode) {
        ncclEpTensorDestroy(ep_group, tensors.combine_output_topk_weights);
    }
}

// ============================================================================
// Data Validation Support (similar to DeepEP test_internode.py / test_low_latency.py)
//
// Methodology:
//   - Input tokens are fingerprinted with (source_rank, token_id) in BF16.
//   - Dispatch validation recomputes expected routing deterministically and
//     verifies each received token's identity and integrity.
//   - Combine validation computes expected weighted sums analytically and
//     compares against actual output using a cosine-similarity metric (calc_diff).
// ============================================================================

// Rank offset for BF16 precision: integers > 256 lose precision in BF16
// Using negative values (rank - 128) allows up to 256 ranks
static const int RANK_OFFSET = 128;

// Number of columns to embed token index (for full traceability)
// Matches DeepEP's approach: last 128 columns store token index
static const int TOKEN_ID_COLS = 128;

// Helper: Convert BF16 to float (CPU-side)
static float bf16ToFloat(uint16_t bf16) {
    uint32_t bits = (static_cast<uint32_t>(bf16)) << 16;
    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

// Helper: Convert float to BF16 (CPU-side, truncation — used only for initialization)
static uint16_t floatToBf16(float f) {
    uint32_t bits;
    memcpy(&bits, &f, sizeof(bits));
    return static_cast<uint16_t>(bits >> 16);
}

// Cosine-similarity-based discrepancy metric in double precision
// Returns 0 for perfect match, larger values for worse match
static double calc_diff(const double* x, const double* y, size_t n) {
    double dot_xy = 0, dot_xx = 0, dot_yy = 0;
    for (size_t i = 0; i < n; i++) {
        double xi = x[i] + 1.0;
        double yi = y[i] + 1.0;
        dot_xy += xi * yi;
        dot_xx += xi * xi;
        dot_yy += yi * yi;
    }
    double denom = dot_xx + dot_yy;
    if (denom == 0) return 0;
    return 1.0 - 2.0 * dot_xy / denom;
}

// Initialize input tensors with validation-friendly patterns (DeepEP style)
// Pattern: each element = (rank - RANK_OFFSET) except last TOKEN_ID_COLS columns = token_index
void initializeValidationData(
    BenchmarkTensors& tensors,
    unsigned int num_tokens,
    unsigned int hidden,
    unsigned int top_k,
    int myRank,
    bool is_ht_mode
) {
    // Calculate the rank value to use (handles BF16 precision limits)
    float rank_value = static_cast<float>(myRank - RANK_OFFSET);
    uint16_t rank_bf16 = floatToBf16(rank_value);

    // Allocate host buffer for token data
    size_t token_size = num_tokens * hidden;
    uint16_t* token_data_host = new uint16_t[token_size];

    // Fill token data with rank value, embed token index in last TOKEN_ID_COLS columns.
    // Token ID is split into high (t/256) and low (t%256) bytes to stay within BF16's
    // exact integer range (0-255). First TOKEN_ID column = high byte, rest = low byte.
    for (unsigned int t = 0; t < num_tokens; t++) {
        uint16_t token_hi = floatToBf16(static_cast<float>(t / 256));
        uint16_t token_lo = floatToBf16(static_cast<float>(t % 256));
        for (unsigned int h = 0; h < hidden; h++) {
            if (h == hidden - TOKEN_ID_COLS) {
                token_data_host[t * hidden + h] = token_hi;
            } else if (h > hidden - TOKEN_ID_COLS) {
                token_data_host[t * hidden + h] = token_lo;
            } else {
                token_data_host[t * hidden + h] = rank_bf16;
            }
        }
    }

    // Copy to GPU
    {
        void* input0_data;
        NCCLCHECK(ncclEpTensorGetData(tensors.inputs[0], &input0_data));
        CUDACHECK(cudaMemcpy(input0_data, token_data_host,
                             token_size * sizeof(uint16_t), cudaMemcpyHostToDevice));
    }

    // Generate random positive topk_weights: abs(randn)
    // LL: weights applied during combine → affects combined output
    // HT: weights forwarded during dispatch → does NOT affect combined output
    float* topk_weights_host = new float[num_tokens * top_k];
    std::mt19937 rng(42 + myRank);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    for (unsigned int i = 0; i < num_tokens * top_k; i++) {
        topk_weights_host[i] = std::abs(normal(rng));
        if (topk_weights_host[i] < 1e-6f) topk_weights_host[i] = 1e-6f;
    }

    if (is_ht_mode) {
        void* dtw_data;
        NCCLCHECK(ncclEpTensorGetData(tensors.dispatch_topk_weights, &dtw_data));
        CUDACHECK(cudaMemcpy(dtw_data, topk_weights_host,
                             num_tokens * top_k * sizeof(float), cudaMemcpyHostToDevice));
    }
    // Also initialize the combine topk_weights (used by both modes)
    {
        void* tw_data;
        NCCLCHECK(ncclEpTensorGetData(tensors.topk_weights, &tw_data));
        CUDACHECK(cudaMemcpy(tw_data, topk_weights_host,
                             num_tokens * top_k * sizeof(float), cudaMemcpyHostToDevice));
    }

    delete[] topk_weights_host;
    delete[] token_data_host;
}

// Validation result structure
struct ValidationResult {
    bool passed;
    int errors;
    double max_diff;
    std::string message;
};

// Forward declaration (defined later in the file)
void generateRandomTopkIndicesLL(
    int64_t* topk_idx_host, unsigned int num_tokens, unsigned int num_experts,
    unsigned int top_k, int rank, int seed = 1);

// Generate HT topk_idx for a given rank (deterministic)
// Randperm routing (uniform), consistent with Hybrid-EP (test_hybrid_ep.py)
static void generateTopkIndicesHT(
    int64_t* topk_idx_host,
    unsigned int num_tokens,
    unsigned int num_experts,
    unsigned int top_k,
    int rank
) {
    std::mt19937 gen(rank + 42);
    std::vector<int64_t> expert_perm(num_experts);
    std::iota(expert_perm.begin(), expert_perm.end(), 0);
    for (unsigned int i = 0; i < num_tokens; i++) {
        std::shuffle(expert_perm.begin(), expert_perm.end(), gen);
        for (unsigned int j = 0; j < top_k; j++) {
            topk_idx_host[i * top_k + j] = expert_perm[j];
        }
    }
}

// Extract (source_rank, token_id) from a received token row using first and last columns
static bool extractTokenIdentity(
    const uint16_t* row,
    unsigned int hidden,
    int nRanks,
    unsigned int num_tokens,
    int* out_source_rank,
    int* out_token_id
) {
    float rank_val = bf16ToFloat(row[0]);
    *out_source_rank = static_cast<int>(rank_val + RANK_OFFSET + 0.5f);

    float token_hi = bf16ToFloat(row[hidden - TOKEN_ID_COLS]);
    float token_lo = bf16ToFloat(row[hidden - 1]);
    *out_token_id = static_cast<int>(token_hi + 0.5f) * 256 + static_cast<int>(token_lo + 0.5f);

    return (*out_source_rank >= 0 && *out_source_rank < nRanks &&
            *out_token_id >= 0 && *out_token_id < static_cast<int>(num_tokens));
}

// Verify a received token row has consistent data (all rank cols match, all token_id cols match)
static bool verifyTokenIntegrity(
    const uint16_t* row,
    unsigned int hidden
) {
    uint16_t expected_rank_bf16 = row[0];
    for (unsigned int h = 1; h < hidden - TOKEN_ID_COLS; h++) {
        if (row[h] != expected_rank_bf16) return false;
    }
    // First TOKEN_ID column is the high byte (standalone), rest are low byte
    uint16_t expected_token_lo_bf16 = row[hidden - 1];
    for (unsigned int h = hidden - TOKEN_ID_COLS + 1; h < hidden - 1; h++) {
        if (row[h] != expected_token_lo_bf16) return false;
    }
    return true;
}

// Validate dispatch output: verify that expected tokens arrived at the correct experts.
// Recomputes every rank's topk_idx deterministically to build the expected set,
// then checks the dispatch output for missing, unexpected, or corrupted tokens.
ValidationResult validateDispatchOutput(
    BenchmarkTensors& tensors,
    unsigned int num_tokens,
    unsigned int hidden,
    unsigned int top_k,
    unsigned int num_experts,
    unsigned int num_local_experts,
    int myRank,
    int nRanks,
    bool is_ht_mode
) {
    ValidationResult result = {true, 0, 0.0, ""};
    int errors = 0;
    const int max_errors_to_print = 10;
    int errors_printed = 0;

    // Temp buffer to recompute each rank's topk_idx
    int64_t* src_topk = new int64_t[num_tokens * top_k];

    if (!is_ht_mode) {
        // ==================== LL Mode ====================
        // Output: 3D [num_local_experts, max_tokens_per_expert, hidden]

        const unsigned int* out0_sizes; unsigned int out0_ndim;
        NCCLCHECK(ncclEpTensorGetSizes(tensors.outputs[0], &out0_sizes, &out0_ndim));
        unsigned int max_tpe = out0_sizes[1];
        size_t total_size = static_cast<size_t>(num_local_experts) * max_tpe * hidden;
        uint16_t* recv_data = new uint16_t[total_size];
        void* output0_data_ll;
        NCCLCHECK(ncclEpTensorGetData(tensors.outputs[0], &output0_data_ll));
        CUDACHECK(cudaMemcpy(recv_data, output0_data_ll,
                             total_size * sizeof(uint16_t), cudaMemcpyDeviceToHost));

        int* tokens_per_expert = new int[num_local_experts];
        void* local0_data;
        NCCLCHECK(ncclEpTensorGetData(tensors.local_tensors[0], &local0_data));
        CUDACHECK(cudaMemcpy(tokens_per_expert, local0_data,
                             num_local_experts * sizeof(int), cudaMemcpyDeviceToHost));

        // Build expected set: expected[local_expert] = set of (source_rank, token_id)
        std::vector<std::set<std::pair<int,int>>> expected(num_local_experts);

        for (int r = 0; r < nRanks; r++) {
            generateRandomTopkIndicesLL(src_topk, num_tokens, num_experts, top_k, r);
            for (unsigned int t = 0; t < num_tokens; t++) {
                for (unsigned int k = 0; k < top_k; k++) {
                    int64_t expert_id = src_topk[t * top_k + k];
                    if (expert_id < 0) continue;
                    int expert_rank = static_cast<int>(expert_id) / static_cast<int>(num_local_experts);
                    int local_expert = static_cast<int>(expert_id) % static_cast<int>(num_local_experts);
                    if (expert_rank == myRank) {
                        expected[local_expert].insert({r, static_cast<int>(t)});
                    }
                }
            }
        }

        // Scan output and match against expected
        for (unsigned int e = 0; e < num_local_experts; e++) {
            int count = tokens_per_expert[e];
            if (count < 0 || count > static_cast<int>(max_tpe)) {
                if (errors_printed < max_errors_to_print) {
                    printf("[Rank %d] LL dispatch: expert %u has invalid count %d (max %u)\n",
                           myRank, e, count, max_tpe);
                    errors_printed++;
                }
                errors++;
                continue;
            }

            std::set<std::pair<int,int>> found;

            for (int j = 0; j < count; j++) {
                const uint16_t* row = recv_data + (e * max_tpe + j) * hidden;
                int source_rank = -1, token_id = -1;

                if (!extractTokenIdentity(row, hidden, nRanks, num_tokens, &source_rank, &token_id)) {
                    if (errors_printed < max_errors_to_print) {
                        printf("[Rank %d] LL dispatch: expert %u slot %d: invalid identity (rank=%d, token=%d)\n",
                               myRank, e, j, source_rank, token_id);
                        errors_printed++;
                    }
                    errors++;
                    continue;
                }

                if (!verifyTokenIntegrity(row, hidden)) {
                    if (errors_printed < max_errors_to_print) {
                        printf("[Rank %d] LL dispatch: expert %u slot %d: data corruption (rank=%d, token=%d)\n",
                               myRank, e, j, source_rank, token_id);
                        errors_printed++;
                    }
                    errors++;
                }

                auto key = std::make_pair(source_rank, token_id);
                if (expected[e].find(key) == expected[e].end()) {
                    if (errors_printed < max_errors_to_print) {
                        printf("[Rank %d] LL dispatch: expert %u slot %d: unexpected token (rank=%d, token=%d)\n",
                               myRank, e, j, source_rank, token_id);
                        errors_printed++;
                    }
                    errors++;
                }
                found.insert(key);
            }

            // Check for missing tokens
            for (const auto& key : expected[e]) {
                if (found.find(key) == found.end()) {
                    if (errors_printed < max_errors_to_print) {
                        printf("[Rank %d] LL dispatch: expert %u: missing token (rank=%d, token=%d)\n",
                               myRank, e, key.first, key.second);
                        errors_printed++;
                    }
                    errors++;
                }
            }
        }

        delete[] tokens_per_expert;
        delete[] recv_data;

    } else {
        // ==================== HT Mode ====================
        // FIXME: ncclEpHandleGetNumRecvTokens returns buffer max, not actual count — scan recv_topk_idx as workaround.
        // Output buffer is [nRanks * max_tokens_per_rank, hidden], tokens packed contiguously at 0..N-1.
        // We use recv_topk_idx (outputs[2]) to identify valid rows (expert index >= 0).

        const unsigned int* out0_sizes_ht; unsigned int out0_ndim_ht;
        NCCLCHECK(ncclEpTensorGetSizes(tensors.outputs[0], &out0_sizes_ht, &out0_ndim_ht));
        unsigned int buf_rows = out0_sizes_ht[0];
        size_t recv_size = static_cast<size_t>(buf_rows) * hidden;
        uint16_t* recv_data = new uint16_t[recv_size];
        void* output0_data;
        NCCLCHECK(ncclEpTensorGetData(tensors.outputs[0], &output0_data));
        CUDACHECK(cudaMemcpy(recv_data, output0_data,
                             recv_size * sizeof(uint16_t), cudaMemcpyDeviceToHost));

        // Read recv_topk_idx to identify valid rows
        int64_t* recv_topk_idx = new int64_t[static_cast<size_t>(buf_rows) * top_k];
        void* output2_data;
        NCCLCHECK(ncclEpTensorGetData(tensors.outputs[2], &output2_data));
        CUDACHECK(cudaMemcpy(recv_topk_idx, output2_data,
                             static_cast<size_t>(buf_rows) * top_k * sizeof(int64_t),
                             cudaMemcpyDeviceToHost));

        // Build expected set from deterministic routing
        std::set<std::pair<int,int>> expected;
        for (int r = 0; r < nRanks; r++) {
            generateTopkIndicesHT(src_topk, num_tokens, num_experts, top_k, r);
            for (unsigned int t = 0; t < num_tokens; t++) {
                for (unsigned int k = 0; k < top_k; k++) {
                    int64_t expert_id = src_topk[t * top_k + k];
                    int expert_rank = static_cast<int>(expert_id) / static_cast<int>(num_local_experts);
                    if (expert_rank == myRank) {
                        expected.insert({r, static_cast<int>(t)});
                        break;
                    }
                }
            }
        }

        // Scan ALL rows, but only validate rows where recv_topk_idx has valid entries
        std::set<std::pair<int,int>> found;

        for (unsigned int j = 0; j < buf_rows; j++) {
            // Check if this row has any valid expert index
            bool has_valid_expert = false;
            for (unsigned int k = 0; k < top_k; k++) {
                if (recv_topk_idx[j * top_k + k] >= 0) {
                    has_valid_expert = true;
                    break;
                }
            }
            if (!has_valid_expert) continue;

            const uint16_t* row = recv_data + j * hidden;
            int source_rank = -1, token_id = -1;

            if (!extractTokenIdentity(row, hidden, nRanks, num_tokens, &source_rank, &token_id)) {
                if (errors_printed < max_errors_to_print) {
                    printf("[Rank %d] HT dispatch: slot %u: invalid identity (rank=%d, token=%d)\n",
                           myRank, j, source_rank, token_id);
                    errors_printed++;
                }
                errors++;
                continue;
            }

            if (!verifyTokenIntegrity(row, hidden)) {
                if (errors_printed < max_errors_to_print) {
                    printf("[Rank %d] HT dispatch: slot %u: data corruption (rank=%d, token=%d)\n",
                           myRank, j, source_rank, token_id);
                    errors_printed++;
                }
                errors++;
            }

            auto key = std::make_pair(source_rank, token_id);
            if (expected.find(key) == expected.end()) {
                if (errors_printed < max_errors_to_print) {
                    printf("[Rank %d] HT dispatch: slot %u: unexpected token (rank=%d, token=%d)\n",
                           myRank, j, source_rank, token_id);
                    errors_printed++;
                }
                errors++;
            }
            found.insert(key);
        }

        // Check for missing tokens
        for (const auto& key : expected) {
            if (found.find(key) == found.end()) {
                if (errors_printed < max_errors_to_print) {
                    printf("[Rank %d] HT dispatch: missing token (rank=%d, token=%d)\n",
                           myRank, key.first, key.second);
                    errors_printed++;
                }
                errors++;
            }
        }

        delete[] recv_topk_idx;
        delete[] recv_data;
    }

    delete[] src_topk;

    result.errors = errors;
    result.passed = (errors == 0);
    if (!result.passed) {
        char buf[256];
        snprintf(buf, sizeof(buf), "%s dispatch validation: %d errors",
                 is_ht_mode ? "HT" : "LL", errors);
        result.message = buf;
    }

    return result;
}

// Compute is_token_in_rank.sum() - count of unique ranks each token is sent to
// This matches DeepEP's validation approach
// Returns array of size num_tokens with unique rank count per token
int* countUniqueRanksPerToken(
    const int64_t* topk_idx_host,
    unsigned int num_tokens,
    unsigned int num_experts,
    unsigned int top_k,
    int nRanks
) {
    int* unique_ranks = new int[num_tokens]();  // Zero-initialized
    unsigned int num_local_experts = num_experts / nRanks;

    for (unsigned int t = 0; t < num_tokens; t++) {
        std::set<int> ranks_set;
        for (unsigned int k = 0; k < top_k; k++) {
            int64_t expert_id = topk_idx_host[t * top_k + k];
            if (expert_id >= 0) {
                int target_rank = expert_id / num_local_experts;
                ranks_set.insert(target_rank);
            }
        }
        unique_ranks[t] = ranks_set.size();
    }
    return unique_ranks;
}

// Count valid experts for each token (experts with topk_idx >= 0)
int countValidExperts(const int64_t* topk_idx_host, unsigned int token_idx, unsigned int top_k) {
    int count = 0;
    for (unsigned int k = 0; k < top_k; k++) {
        if (topk_idx_host[token_idx * top_k + k] >= 0) {
            count++;
        }
    }
    return count;
}

// Validate combine output for Low Latency mode
// DeepEP formula: check = combined / is_token_in_rank.sum()
// LL combine applies weighted sum: combined[t] = x[t] * sum(valid weights)
// Compared using calc_diff in double precision with threshold 1e-5
ValidationResult validateCombineOutputLL(
    BenchmarkTensors& tensors,
    unsigned int num_tokens,
    unsigned int hidden,
    unsigned int num_experts,
    unsigned int top_k,
    int myRank,
    int nRanks,
    int64_t* topk_idx_host
) {
    (void)num_experts;
    (void)nRanks;

    ValidationResult result = {true, 0, 0.0, ""};

    size_t output_size = num_tokens * hidden;
    uint16_t* combined_data = new uint16_t[output_size];
    {
        void* co_data;
        NCCLCHECK(ncclEpTensorGetData(tensors.combined_output, &co_data));
        CUDACHECK(cudaMemcpy(combined_data, co_data,
                             output_size * sizeof(uint16_t), cudaMemcpyDeviceToHost));
    }

    float* topk_weights_host = new float[num_tokens * top_k];
    void* tw_data_ll;
    NCCLCHECK(ncclEpTensorGetData(tensors.topk_weights, &tw_data_ll));
    CUDACHECK(cudaMemcpy(topk_weights_host, tw_data_ll,
                         num_tokens * top_k * sizeof(float), cudaMemcpyDeviceToHost));

    float original_rank_val = static_cast<float>(myRank - RANK_OFFSET);

    size_t num_elements = 0;
    for (unsigned int t = 0; t < num_tokens; t++) {
        if (countValidExperts(topk_idx_host, t, top_k) > 0)
            num_elements += hidden;
    }

    double* ref = new double[num_elements];
    double* actual = new double[num_elements];
    size_t idx = 0;

    bool has_nan = false;
    for (unsigned int t = 0; t < num_tokens; t++) {
        int nv = countValidExperts(topk_idx_host, t, top_k);
        if (nv == 0) continue;

        double weight_sum = 0;
        for (unsigned int k = 0; k < top_k; k++) {
            if (topk_idx_host[t * top_k + k] >= 0)
                weight_sum += static_cast<double>(topk_weights_host[t * top_k + k]);
        }

        double rank_val = static_cast<double>(original_rank_val);
        double token_hi_val = static_cast<double>(bf16ToFloat(floatToBf16(static_cast<float>(t / 256))));
        double token_lo_val = static_cast<double>(bf16ToFloat(floatToBf16(static_cast<float>(t % 256))));

        for (unsigned int h = 0; h < hidden; h++) {
            double orig;
            if (h == hidden - TOKEN_ID_COLS)
                orig = token_hi_val;
            else if (h > hidden - TOKEN_ID_COLS)
                orig = token_lo_val;
            else
                orig = rank_val;
            ref[idx] = orig * weight_sum;
            float actual_f = bf16ToFloat(combined_data[t * hidden + h]);
            actual[idx] = static_cast<double>(actual_f);
            if (std::isnan(actual_f)) has_nan = true;
            idx++;
        }
    }

    double diff = calc_diff(ref, actual, num_elements);
    result.max_diff = diff;
    result.passed = (diff < 1e-5) && !has_nan;

    if (!result.passed) {
        char buf[256];
        snprintf(buf, sizeof(buf), "LL combine: calc_diff=%.6e (threshold=1e-5)%s",
                 diff, has_nan ? ", NaN detected" : "");
        result.message = buf;
    }

    delete[] ref;
    delete[] actual;
    delete[] topk_weights_host;
    delete[] combined_data;
    return result;
}

// Validate combine output for High Throughput mode
// DeepEP formula: check = combined / is_token_in_rank.sum()
// HT combine is unweighted sum: combined[t] = x[t] * num_unique_ranks
// Compared using calc_diff in double precision with threshold 5e-6
ValidationResult validateCombineOutputHT(
    BenchmarkTensors& tensors,
    unsigned int num_tokens,
    unsigned int hidden,
    unsigned int num_experts,
    unsigned int top_k,
    int myRank,
    int nRanks,
    int64_t* topk_idx_host
) {
    ValidationResult result = {true, 0, 0.0, ""};

    size_t output_size = num_tokens * hidden;
    uint16_t* combined_data = new uint16_t[output_size];
    {
        void* co_data;
        NCCLCHECK(ncclEpTensorGetData(tensors.combined_output, &co_data));
        CUDACHECK(cudaMemcpy(combined_data, co_data,
                             output_size * sizeof(uint16_t), cudaMemcpyDeviceToHost));
    }

    int* unique_ranks = countUniqueRanksPerToken(topk_idx_host, num_tokens,
                                                  num_experts, top_k, nRanks);

    float original_rank_val = static_cast<float>(myRank - RANK_OFFSET);

    size_t num_elements = 0;
    for (unsigned int t = 0; t < num_tokens; t++) {
        if (unique_ranks[t] > 0) num_elements += hidden;
    }

    double* ref = new double[num_elements];
    double* actual = new double[num_elements];
    size_t idx = 0;

    bool has_nan = false;
    for (unsigned int t = 0; t < num_tokens; t++) {
        int nr = unique_ranks[t];
        if (nr == 0) continue;

        double rank_val = static_cast<double>(original_rank_val);
        double token_hi_val = static_cast<double>(bf16ToFloat(floatToBf16(static_cast<float>(t / 256))));
        double token_lo_val = static_cast<double>(bf16ToFloat(floatToBf16(static_cast<float>(t % 256))));
        double scale = static_cast<double>(nr);

        for (unsigned int h = 0; h < hidden; h++) {
            double orig;
            if (h == hidden - TOKEN_ID_COLS)
                orig = token_hi_val;
            else if (h > hidden - TOKEN_ID_COLS)
                orig = token_lo_val;
            else
                orig = rank_val;
            ref[idx] = orig * scale;
            float actual_f = bf16ToFloat(combined_data[t * hidden + h]);
            actual[idx] = static_cast<double>(actual_f);
            if (std::isnan(actual_f)) has_nan = true;
            idx++;
        }
    }

    double diff = calc_diff(ref, actual, num_elements);
    result.max_diff = diff;
    result.passed = (diff < 5e-6) && !has_nan;

    if (!result.passed) {
        char buf[256];
        snprintf(buf, sizeof(buf), "HT combine: calc_diff=%.6e (threshold=5e-6)%s",
                 diff, has_nan ? ", NaN detected" : "");
        result.message = buf;
    }

    delete[] ref;
    delete[] actual;
    delete[] unique_ranks;
    delete[] combined_data;
    return result;
}

// Wrapper that calls appropriate validation based on mode
ValidationResult validateCombineOutput(
    BenchmarkTensors& tensors,
    unsigned int num_tokens,
    unsigned int hidden,
    unsigned int top_k,
    unsigned int num_experts,
    int myRank,
    int nRanks,
    bool is_ht_mode,
    int64_t* topk_idx_host
) {
    if (is_ht_mode) {
        return validateCombineOutputHT(tensors, num_tokens, hidden, num_experts,
                                        top_k, myRank, nRanks, topk_idx_host);
    } else {
        return validateCombineOutputLL(tensors, num_tokens, hidden, num_experts,
                                        top_k, myRank, nRanks, topk_idx_host);
    }
}

// Benchmark result structure
struct BenchResult {
    double avg_ms;
    double min_ms;
    double max_ms;
    double throughput_gbps;
};

// Structure to hold paired dispatch+combine benchmark results
struct PairedBenchResult {
    BenchResult dispatch;
    BenchResult combine;
    BenchResult total;
};

// Run paired dispatch+combine benchmark with separate timing for each phase
// This ensures dispatch and combine are always paired (required for correctness)
// while still measuring individual performance
PairedBenchResult runPairedBenchmark(
    std::function<void()> dispatch_fn,
    std::function<void()> combine_fn,
    int num_warmup,
    int num_iters,
    size_t dispatch_bytes,
    size_t combine_bytes,
    cudaStream_t stream
) {
    // Warmup with paired dispatch+combine
    // Note: cudaStreamSynchronize between dispatch and combine is required for HT mode
    // MPI_Barrier at end of each iteration ensures all ranks stay in sync (critical for HT mode)
    for (int i = 0; i < num_warmup; i++) {
        dispatch_fn();
        CUDACHECK(cudaStreamSynchronize(stream));
        combine_fn();
        CUDACHECK(cudaStreamSynchronize(stream));
        MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
    }

    // Create events for dispatch, combine, and total timing
    std::vector<cudaEvent_t> dispatch_start(num_iters);
    std::vector<cudaEvent_t> dispatch_end(num_iters);
    std::vector<cudaEvent_t> combine_start(num_iters);
    std::vector<cudaEvent_t> combine_end(num_iters);

    for (int i = 0; i < num_iters; i++) {
        CUDACHECK(cudaEventCreate(&dispatch_start[i]));
        CUDACHECK(cudaEventCreate(&dispatch_end[i]));
        CUDACHECK(cudaEventCreate(&combine_start[i]));
        CUDACHECK(cudaEventCreate(&combine_end[i]));
    }

    // Run paired benchmark with individual timing
    // Events are recorded immediately after kernel launch (before sync) to measure GPU time only
    // Sync happens after event recording to not affect timing
    // MPI_Barrier at end of each iteration ensures all ranks stay in sync (critical for HT mode)
    for (int i = 0; i < num_iters; i++) {
        CUDACHECK(cudaEventRecord(dispatch_start[i], stream));
        dispatch_fn();
        CUDACHECK(cudaEventRecord(dispatch_end[i], stream));    // Record before sync
        CUDACHECK(cudaStreamSynchronize(stream));              // Sync outside timing
        CUDACHECK(cudaEventRecord(combine_start[i], stream));  // Record after sync, before combine
        combine_fn();
        CUDACHECK(cudaEventRecord(combine_end[i], stream));    // Record before sync
        CUDACHECK(cudaStreamSynchronize(stream));             // Sync outside timing
        MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
    }

    // Collect times
    std::vector<float> dispatch_times(num_iters);
    std::vector<float> combine_times(num_iters);
    std::vector<float> total_times(num_iters);

    for (int i = 0; i < num_iters; i++) {
        CUDACHECK(cudaEventElapsedTime(&dispatch_times[i], dispatch_start[i], dispatch_end[i]));
        CUDACHECK(cudaEventElapsedTime(&combine_times[i], combine_start[i], combine_end[i]));
        CUDACHECK(cudaEventElapsedTime(&total_times[i], dispatch_start[i], combine_end[i]));
    }

    // Cleanup events
    for (int i = 0; i < num_iters; i++) {
        CUDACHECK(cudaEventDestroy(dispatch_start[i]));
        CUDACHECK(cudaEventDestroy(dispatch_end[i]));
        CUDACHECK(cudaEventDestroy(combine_start[i]));
        CUDACHECK(cudaEventDestroy(combine_end[i]));
    }

    // Helper to calculate stats from times vector (skip first iteration if we have more than 1)
    auto calc_stats = [](const std::vector<float>& times, size_t data_bytes) -> BenchResult {
        // For HT mode with only 1 iteration, don't skip any - use all data
        // For LL mode with multiple iterations, skip the first (warmup outlier)
        std::vector<float> times_trimmed;
        if (times.size() > 1) {
            times_trimmed.assign(times.begin() + 1, times.end());
        } else {
            times_trimmed = times;  // Use all data when we only have 1 iteration
        }

        BenchResult result;
        if (times_trimmed.empty()) {
            result.avg_ms = 0;
            result.min_ms = 0;
            result.max_ms = 0;
            result.throughput_gbps = 0;
        } else {
            result.avg_ms = std::accumulate(times_trimmed.begin(), times_trimmed.end(), 0.0) / times_trimmed.size();
            result.min_ms = *std::min_element(times_trimmed.begin(), times_trimmed.end());
            result.max_ms = *std::max_element(times_trimmed.begin(), times_trimmed.end());
            result.throughput_gbps = (data_bytes / 1e9) / (result.avg_ms / 1000.0);
        }
        return result;
    };

    PairedBenchResult result;
    result.dispatch = calc_stats(dispatch_times, dispatch_bytes);
    result.combine = calc_stats(combine_times, combine_bytes);
    result.total = calc_stats(total_times, dispatch_bytes + combine_bytes);

    return result;
}

// Structure to hold Low Latency byte calculation
// Matches DeepEP test_low_latency.py methodology
struct LowLatencyBytes {
    size_t dispatch_bytes;  // FP8 or BF16 format per selection
    size_t combine_bytes;   // BF16 format: hidden * 2 per selection
    unsigned int num_valid_selections;
    bool is_fp8;  // Whether dispatch uses FP8
};

// Calculate bytes for Low Latency mode
// Dispatch can be FP8 or BF16, combine is always BF16
LowLatencyBytes calculateLowLatencyBytes(
    const int64_t* topk_idx_host,
    unsigned int num_tokens,
    unsigned int top_k,
    unsigned int hidden,
    bool use_fp8
) {
    LowLatencyBytes bytes = {0, 0, 0, use_fp8};

    // Count valid selections (non-masked entries)
    for (unsigned int i = 0; i < num_tokens * top_k; i++) {
        if (topk_idx_host[i] >= 0) {
            bytes.num_valid_selections++;
        }
    }

    // FP8 bytes per selection: hidden + hidden/128*4 + 16 (scale factors + metadata)
    const size_t fp8_bytes_per_selection = hidden + (hidden / 128) * 4 + 16;
    // BF16 bytes per selection: hidden * 2
    const size_t bf16_bytes_per_selection = hidden * 2;

    // Dispatch: FP8 or BF16 based on config
    bytes.dispatch_bytes = static_cast<size_t>(bytes.num_valid_selections) *
                           (use_fp8 ? fp8_bytes_per_selection : bf16_bytes_per_selection);
    // Combine: always BF16
    bytes.combine_bytes = static_cast<size_t>(bytes.num_valid_selections) * bf16_bytes_per_selection;

    return bytes;
}

// Six bandwidth metrics for High Throughput mode, all dividing by measured time t:
//
//  Send-side (this rank dispatching tokens to experts):
//   total_send  = total_send_bytes / t   — all destinations (NVL+RDMA)
//   nvl_send    = nvl_send_bytes / t     — local node only (NVLink)
//   rdma_send   = rdma_send_bytes / t    — remote nodes only (RDMA outbound)
//
//  Recv-side (this rank's experts receiving tokens):
//   total_recv  = total_recv_bytes / t   — all sources (NVL+RDMA)
//   nvl_recv    = nvl_recv_bytes / t     — from local ranks (NVLink)
//   rdma_recv   = rdma_recv_bytes / t    — from remote ranks (RDMA inbound)
//
//  Derived: nvl_send = total_send - rdma_send
//           nvl_recv = total_recv - rdma_recv
struct HighThroughputBytes {
    size_t total_send_bytes;     // NVL + RDMA outbound
    size_t rdma_send_bytes;      // RDMA outbound only
    size_t total_recv_bytes;     // NVL + RDMA inbound
    size_t rdma_recv_bytes;      // RDMA inbound only (from remote ranks)
    unsigned int total_send_tokens;
    unsigned int rdma_send_tokens;
    unsigned int rdma_recv_tokens;
    unsigned int total_recv_tokens;
    bool is_fp8;
};

// Calculate all six byte metrics from topk_idx for High Throughput mode.
//
// Send side: count unique (token, node) pairs this rank sends to.
//   total_send_tokens = all nodes (local + remote)
//   rdma_send_tokens  = remote nodes only
//
// Recv side: simulate all source ranks' randperm routing (deterministic from
// seed = src_rank + 42) to count unique (src_rank, token) pairs where at least
// one selected expert belongs to myRank.
//   total_recv_tokens = all source ranks (NVL + RDMA)
//   rdma_recv_tokens = remote source ranks only
HighThroughputBytes calculateHighThroughputBytes(
    const int64_t* topk_idx_host,
    unsigned int num_tokens,
    unsigned int top_k,
    unsigned int num_experts,
    unsigned int hidden,
    int myRank,
    int nRanks,
    bool use_fp8,
    int num_ranks_per_node
) {
    HighThroughputBytes bytes = {0, 0, 0, 0, 0, 0, 0, 0, use_fp8};

    int num_nodes = (nRanks + num_ranks_per_node - 1) / num_ranks_per_node;
    unsigned int num_experts_per_node = static_cast<unsigned int>(num_experts / num_nodes);
    int local_node = myRank / num_ranks_per_node;
    unsigned int num_experts_per_rank = num_experts / static_cast<unsigned int>(nRanks);

    // Send side: count unique (token, node) pairs from this rank's topk_idx
    // A token routed to multiple experts on the same node is counted only once, even though
    // NCCL EP sends it to each target rank individually via NVLink P2P (not once per node).
    // TODO: switch to per-rank counting for both nvl_send and nvl_recv.
    for (unsigned int t = 0; t < num_tokens; t++) {
        std::set<int> nodes_for_token;
        for (unsigned int k = 0; k < top_k; k++) {
            int64_t expert_id = topk_idx_host[t * top_k + k];
            if (expert_id < 0) continue;
            int target_node = static_cast<int>(expert_id / num_experts_per_node);
            if (nodes_for_token.insert(target_node).second) {
                bytes.total_send_tokens++;
                if (target_node != local_node)
                    bytes.rdma_send_tokens++;
            }
        }
    }

    // Recv side: replay every source rank's randperm routing to count tokens
    // received by myRank. This is deterministic because each rank uses the
    // same seed (src_rank + 42) and same shuffle algorithm.
    // Each (src_rank, token) pair is counted once regardless of how many experts on myRank it targets.
    std::vector<int64_t> src_perm(num_experts);
    for (int src_rank = 0; src_rank < nRanks; src_rank++) {
        int src_node = src_rank / num_ranks_per_node;
        bool is_rdma = (src_node != local_node);

        std::mt19937 src_gen(src_rank + 42);
        std::iota(src_perm.begin(), src_perm.end(), 0);
        for (unsigned int t = 0; t < num_tokens; t++) {
            std::shuffle(src_perm.begin(), src_perm.end(), src_gen);
            for (unsigned int k = 0; k < top_k; k++) {
                int target_rank = static_cast<int>(src_perm[k] / num_experts_per_rank);
                if (target_rank == myRank) {
                    bytes.total_recv_tokens++;
                    if (is_rdma) bytes.rdma_recv_tokens++;
                    break;
                }
            }
        }
    }

    const size_t bf16_bytes_per_token = hidden * 2;
    const double fp8_factor = (1.0 + 4.0 / 128.0) / 2.0;
    const size_t bytes_per_token = use_fp8 ?
        static_cast<size_t>(bf16_bytes_per_token * fp8_factor) : bf16_bytes_per_token;

    bytes.total_send_bytes = bytes.total_send_tokens * bytes_per_token;
    bytes.rdma_send_bytes  = bytes.rdma_send_tokens  * bytes_per_token;
    bytes.total_recv_bytes = bytes.total_recv_tokens   * bytes_per_token;
    bytes.rdma_recv_bytes  = bytes.rdma_recv_tokens   * bytes_per_token;

    return bytes;
}

// Print benchmark results with MPI aggregation across ranks
// Print benchmark results for Low Latency mode
// Uses FP8 bytes for dispatch, BF16 bytes for combine (matching DeepEP test_low_latency.py)
void printLowLatencyResults(
    int myRank,
    int nRanks,
    const BenchResult& dispatch_result,
    const BenchResult& combine_result,
    const BenchResult& combined_result,
    const LowLatencyBytes& ll_bytes
) {
    // Print per-rank results
    printf("[Rank %d] Dispatch:         avg=%.2f us, min=%.2f us, max=%.2f us, throughput=%.2f GB/s\n",
           myRank,
           dispatch_result.avg_ms * 1000, dispatch_result.min_ms * 1000, dispatch_result.max_ms * 1000,
           dispatch_result.throughput_gbps);

    printf("[Rank %d] Combine:          avg=%.2f us, min=%.2f us, max=%.2f us, throughput=%.2f GB/s\n",
           myRank,
           combine_result.avg_ms * 1000, combine_result.min_ms * 1000, combine_result.max_ms * 1000,
           combine_result.throughput_gbps);

    printf("[Rank %d] Dispatch+Combine: avg=%.2f us, min=%.2f us, max=%.2f us, throughput=%.2f GB/s\n",
           myRank,
           combined_result.avg_ms * 1000, combined_result.min_ms * 1000, combined_result.max_ms * 1000,
           combined_result.throughput_gbps);

    // Aggregate latency results across ranks
    double local_dispatch_avg = dispatch_result.avg_ms;
    double local_dispatch_min = dispatch_result.min_ms;
    double local_dispatch_max = dispatch_result.max_ms;
    double local_combine_avg = combine_result.avg_ms;
    double local_combine_min = combine_result.min_ms;
    double local_combine_max = combine_result.max_ms;
    double local_total_avg = combined_result.avg_ms;
    double local_total_min = combined_result.min_ms;
    double local_total_max = combined_result.max_ms;

    double global_dispatch_avg, global_dispatch_min, global_dispatch_max;
    double global_combine_avg, global_combine_min, global_combine_max;
    double global_total_avg, global_total_min, global_total_max;

    MPI_Reduce(&local_dispatch_avg, &global_dispatch_avg, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_dispatch_min, &global_dispatch_min, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_dispatch_max, &global_dispatch_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_combine_avg, &global_combine_avg, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_combine_min, &global_combine_min, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_combine_max, &global_combine_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_total_avg, &global_total_avg, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_total_min, &global_total_min, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_total_max, &global_total_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // Gather throughput min/max with rank info using MPI_MINLOC/MPI_MAXLOC
    struct { double value; int rank; } local_dispatch_tp, local_combine_tp, local_total_tp;
    struct { double value; int rank; } global_dispatch_tp_min, global_dispatch_tp_max;
    struct { double value; int rank; } global_combine_tp_min, global_combine_tp_max;
    struct { double value; int rank; } global_total_tp_min, global_total_tp_max;

    local_dispatch_tp.value = dispatch_result.throughput_gbps;
    local_dispatch_tp.rank = myRank;
    local_combine_tp.value = combine_result.throughput_gbps;
    local_combine_tp.rank = myRank;
    local_total_tp.value = combined_result.throughput_gbps;
    local_total_tp.rank = myRank;

    MPI_Reduce(&local_dispatch_tp, &global_dispatch_tp_min, 1, MPI_DOUBLE_INT, MPI_MINLOC, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_dispatch_tp, &global_dispatch_tp_max, 1, MPI_DOUBLE_INT, MPI_MAXLOC, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_combine_tp, &global_combine_tp_min, 1, MPI_DOUBLE_INT, MPI_MINLOC, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_combine_tp, &global_combine_tp_max, 1, MPI_DOUBLE_INT, MPI_MAXLOC, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_total_tp, &global_total_tp_min, 1, MPI_DOUBLE_INT, MPI_MINLOC, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_total_tp, &global_total_tp_max, 1, MPI_DOUBLE_INT, MPI_MAXLOC, 0, MPI_COMM_WORLD);

    // Print summary on rank 0
    if (myRank == 0) {
        global_dispatch_avg /= nRanks;
        global_combine_avg /= nRanks;
        global_total_avg /= nRanks;

        double total_data_bytes = ll_bytes.dispatch_bytes + ll_bytes.combine_bytes;
        double avg_dispatch_tp = (ll_bytes.dispatch_bytes / 1e9) / (global_dispatch_avg / 1000.0);
        double avg_combine_tp = (ll_bytes.combine_bytes / 1e9) / (global_combine_avg / 1000.0);
        double avg_total_tp = (total_data_bytes / 1e9) / (global_total_avg / 1000.0);

        printf("\n=== Summary (Low Latency, across %d ranks) ===\n", nRanks);
        printf("Dispatch (%s):  avg=%.2f us, min=%.2f us, max=%.2f us\n",
               ll_bytes.is_fp8 ? "FP8" : "BF16",
               global_dispatch_avg * 1000,
               global_dispatch_min * 1000,
               global_dispatch_max * 1000);
        printf("                  throughput: avg=%.2f GB/s, min=%.2f GB/s (rank %d), max=%.2f GB/s (rank %d)\n",
               avg_dispatch_tp,
               global_dispatch_tp_min.value, global_dispatch_tp_min.rank,
               global_dispatch_tp_max.value, global_dispatch_tp_max.rank);
        printf("Combine (BF16):   avg=%.2f us, min=%.2f us, max=%.2f us\n",
               global_combine_avg * 1000,
               global_combine_min * 1000,
               global_combine_max * 1000);
        printf("                  throughput: avg=%.2f GB/s, min=%.2f GB/s (rank %d), max=%.2f GB/s (rank %d)\n",
               avg_combine_tp,
               global_combine_tp_min.value, global_combine_tp_min.rank,
               global_combine_tp_max.value, global_combine_tp_max.rank);
        printf("Total (D+C):      avg=%.2f us, min=%.2f us, max=%.2f us\n",
               global_total_avg * 1000,
               global_total_min * 1000,
               global_total_max * 1000);
        printf("                  throughput: avg=%.2f GB/s, min=%.2f GB/s (rank %d), max=%.2f GB/s (rank %d)\n",
               avg_total_tp,
               global_total_tp_min.value, global_total_tp_min.rank,
               global_total_tp_max.value, global_total_tp_max.rank);
        printf("\nByte counts: dispatch=%.2f MB (%s), combine=%.2f MB (BF16), selections=%u\n",
               ll_bytes.dispatch_bytes / 1e6,
               ll_bytes.is_fp8 ? "FP8" : "BF16",
               ll_bytes.combine_bytes / 1e6,
               ll_bytes.num_valid_selections);
    }
}

// Print results for High Throughput mode.
// local_kernel_dk_us / local_kernel_ck_us: per-rank CUPTI kernel times for the per-rank lines.
// All "global_*" parameters are raw MPI_SUM across all ranks (rank 0 only; 0 on other ranks).
// The function averages them by nRanks before use.
void printHighThroughputResults(
    int myRank,
    int nRanks,
    const BenchResult& dispatch_result,
    const BenchResult& combine_result,
    const BenchResult& combined_result,
    const HighThroughputBytes& ht_bytes,
    double local_kernel_dk_us,
    double local_kernel_ck_us,
    double global_kernel_dk_us,
    double global_kernel_ck_us,
    size_t global_total_send, size_t global_rdma_send,
    size_t global_total_recv, size_t global_rdma_recv
) {
    printf("[Rank %d] Dispatch:         total=%.2f us  kernel=%.2f us\n",
           myRank, dispatch_result.avg_ms * 1000, local_kernel_dk_us);
    printf("[Rank %d] Combine:          total=%.2f us  kernel=%.2f us\n",
           myRank, combine_result.avg_ms * 1000, local_kernel_ck_us);
    printf("[Rank %d] Dispatch+Combine: total=%.2f us\n", myRank, combined_result.avg_ms * 1000);

    double local_dispatch_avg = dispatch_result.avg_ms;
    double local_dispatch_min = dispatch_result.min_ms;
    double local_dispatch_max = dispatch_result.max_ms;
    double local_combine_avg = combine_result.avg_ms;
    double local_combine_min = combine_result.min_ms;
    double local_combine_max = combine_result.max_ms;
    double local_total_avg = combined_result.avg_ms;
    double local_total_min = combined_result.min_ms;
    double local_total_max = combined_result.max_ms;

    double global_dispatch_avg, global_dispatch_min, global_dispatch_max;
    double global_combine_avg, global_combine_min, global_combine_max;
    double global_total_avg, global_total_min, global_total_max;

    MPI_Reduce(&local_dispatch_avg, &global_dispatch_avg, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_dispatch_min, &global_dispatch_min, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_dispatch_max, &global_dispatch_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_combine_avg, &global_combine_avg, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_combine_min, &global_combine_min, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_combine_max, &global_combine_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_total_avg, &global_total_avg, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_total_min, &global_total_min, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_total_max, &global_total_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (myRank == 0) {
        global_dispatch_avg /= nRanks;
        global_combine_avg  /= nRanks;
        global_total_avg    /= nRanks;

        double avg_total_send = static_cast<double>(global_total_send) / nRanks;
        double avg_rdma_send  = static_cast<double>(global_rdma_send)  / nRanks;
        double avg_total_recv = static_cast<double>(global_total_recv) / nRanks;
        double avg_rdma_recv  = static_cast<double>(global_rdma_recv)  / nRanks;
        double avg_nvl_send   = avg_total_send - avg_rdma_send;
        double avg_nvl_recv   = avg_total_recv - avg_rdma_recv;

        double avg_kernel_dk_us = global_kernel_dk_us / nRanks;
        double avg_kernel_ck_us = global_kernel_ck_us / nRanks;
        double dk_s       = avg_kernel_dk_us / 1e6;
        double ck_s       = avg_kernel_ck_us / 1e6;
        double dk_total_s = global_dispatch_avg * 1e-3;  // avg total dispatch time in seconds
        double ck_total_s = global_combine_avg  * 1e-3;

        printf("\n=== Summary (High Throughput %s, across %d ranks) ===\n",
               ht_bytes.is_fp8 ? "FP8" : "BF16", nRanks);
        printf("NOTE: total time = kernel time + memcpyD2D + misc\n");

        // --- BW based on total time ---
        printf("--- BW based on total time ---\n");
        printf("Dispatch:    total=%.2f us (min=%.2f, max=%.2f)\n",
               global_dispatch_avg * 1000, global_dispatch_min * 1000, global_dispatch_max * 1000);
        if (dk_total_s > 0) {
            printf("             recv: total_bw=%.2f  nvl_bw=%.2f  rdma_bw=%.2f GB/s\n",
                   (avg_total_recv / 1e9) / dk_total_s, (avg_nvl_recv / 1e9) / dk_total_s, (avg_rdma_recv / 1e9) / dk_total_s);
            printf("             send: total_bw=%.2f  nvl_bw=%.2f  rdma_bw=%.2f GB/s\n",
                   (avg_total_send / 1e9) / dk_total_s, (avg_nvl_send / 1e9) / dk_total_s, (avg_rdma_send / 1e9) / dk_total_s);
        }
        printf("Combine:     total=%.2f us (min=%.2f, max=%.2f)\n",
               global_combine_avg * 1000, global_combine_min * 1000, global_combine_max * 1000);
        if (ck_total_s > 0) {
            printf("             send: total_bw=%.2f  nvl_bw=%.2f  rdma_bw=%.2f GB/s\n",
                   (avg_total_recv / 1e9) / ck_total_s, (avg_nvl_recv / 1e9) / ck_total_s, (avg_rdma_recv / 1e9) / ck_total_s);
            printf("             recv: total_bw=%.2f  nvl_bw=%.2f  rdma_bw=%.2f GB/s\n",
                   (avg_total_send / 1e9) / ck_total_s, (avg_nvl_send / 1e9) / ck_total_s, (avg_rdma_send / 1e9) / ck_total_s);
        }
        printf("Total (D+C): avg=%.2f us, min=%.2f us, max=%.2f us\n",
               global_total_avg * 1000, global_total_min * 1000, global_total_max * 1000);

        // --- BW based on kernel time ---
        printf("\n--- BW based on kernel time ---\n");
        printf("Dispatch:    kernel=%.2f us\n", avg_kernel_dk_us);
        if (dk_s > 0) {
            printf("             recv: total_bw=%.2f  nvl_bw=%.2f  rdma_bw=%.2f GB/s\n",
                   (avg_total_recv / 1e9) / dk_s, (avg_nvl_recv / 1e9) / dk_s, (avg_rdma_recv / 1e9) / dk_s);
            printf("             send: total_bw=%.2f  nvl_bw=%.2f  rdma_bw=%.2f GB/s\n",
                   (avg_total_send / 1e9) / dk_s, (avg_nvl_send / 1e9) / dk_s, (avg_rdma_send / 1e9) / dk_s);
        }
        printf("Combine:     kernel=%.2f us\n", avg_kernel_ck_us);
        if (ck_s > 0) {
            printf("             send: total_bw=%.2f  nvl_bw=%.2f  rdma_bw=%.2f GB/s\n",
                   (avg_total_recv / 1e9) / ck_s, (avg_nvl_recv / 1e9) / ck_s, (avg_rdma_recv / 1e9) / ck_s);
            printf("             recv: total_bw=%.2f  nvl_bw=%.2f  rdma_bw=%.2f GB/s\n",
                   (avg_total_send / 1e9) / ck_s, (avg_nvl_send / 1e9) / ck_s, (avg_rdma_send / 1e9) / ck_s);
        }
        printf("Total (D+C): kernel=%.2f us\n", avg_kernel_dk_us + avg_kernel_ck_us);

        if (dk_s > 0 || ck_s > 0) {
            printf("\nByte counts (per rank avg): total_send=%.2f MB (%u tokens), rdma_send=%.2f MB (%u tokens), "
                   "rdma_recv=%.2f MB (%u tokens), total_recv=%.2f MB (%u tokens)\n",
                   avg_total_send / 1e6, ht_bytes.total_send_tokens,
                   avg_rdma_send  / 1e6, ht_bytes.rdma_send_tokens,
                   avg_rdma_recv  / 1e6, ht_bytes.rdma_recv_tokens,
                   avg_total_recv / 1e6, ht_bytes.total_recv_tokens);
        }
    }
}

// Run NVTX profiling with labeled ranges for nsys analysis.
// Profiles one HandleCreate (to see AG + metadata processing) followed by
// num_iters paired Dispatch+Combine iterations.
void runNvtxProfiling(
    int myRank,
    int num_iters,
    std::function<void()> dispatch_fn,
    std::function<void()> combine_fn,
    std::function<void()> handle_create_fn,
    cudaStream_t stream
) {
    if (myRank == 0) {
        printf("\n=== NVTX Profiling Mode ===\n");
        printf("Run with: nsys profile --stats=true mpirun ...\n\n");
    }

    MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
    CUDACHECK(cudaStreamSynchronize(stream));

    // Start CUDA profiler (for nsys --capture-range=cudaProfilerApi)
    cudaProfilerStart();

    // Profile HandleCreate to expose AG and metadata processing phases
    nvtxRangePush("HandleCreate");
    handle_create_fn();
    CUDACHECK(cudaStreamSynchronize(stream));
    nvtxRangePop();

    MPICHECK(MPI_Barrier(MPI_COMM_WORLD));

    // Profile paired dispatch+combine iterations with individual labels
    // Note: cudaStreamSynchronize between dispatch and combine is required for HT mode
    // MPI_Barrier at end of each iteration ensures all ranks stay in sync (critical for HT mode)
    nvtxRangePush("Paired Dispatch+Combine Benchmark");
    for (int i = 0; i < num_iters; i++) {
        nvtxRangePush("Dispatch");
        dispatch_fn();
        CUDACHECK(cudaStreamSynchronize(stream));
        nvtxRangePop();
        nvtxRangePush("Combine");
        combine_fn();
        CUDACHECK(cudaStreamSynchronize(stream));
        nvtxRangePop();
        MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
    }
    nvtxRangePop();  // Paired Dispatch+Combine Benchmark

    cudaProfilerStop();

    if (myRank == 0) {
        printf("Profiling complete. Analyze with nsys-ui or nsys stats.\n");
    }
}

// Generate topk indices for LL mode (consistent with DeepEP test_low_latency.py)
// abs(randn)+1 scores → topk selection → random -1 masking (simulates dropped tokens)
void generateRandomTopkIndicesLL(
    int64_t* topk_idx_host,
    unsigned int num_tokens,
    unsigned int num_experts,
    unsigned int top_k,
    int rank,
    int seed
) {
    // Seed with (seed + rank) for reproducibility across ranks
    std::mt19937 gen(seed + rank);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    std::vector<std::pair<float, int>> score_idx(num_experts);

    for (unsigned int i = 0; i < num_tokens; i++) {
        // Generate random scores: abs(randn) + 1
        for (unsigned int e = 0; e < num_experts; e++) {
            float score = std::abs(dist(gen)) + 1.0f;
            score_idx[e] = {score, static_cast<int>(e)};
        }

        // Partial sort to get top-k (largest scores first)
        std::partial_sort(score_idx.begin(), score_idx.begin() + top_k, score_idx.end(),
                          [](const auto& a, const auto& b) { return a.first > b.first; });

        // Extract top-k expert indices (sorted by score, descending)
        for (unsigned int j = 0; j < top_k; j++) {
            topk_idx_host[i * top_k + j] = score_idx[j].second;
        }
    }

    // Randomly mask 10 positions with -1 (simulates dropped tokens)
    std::uniform_int_distribution<unsigned int> token_dist(0, num_tokens - 1);
    std::uniform_int_distribution<unsigned int> topk_dist(0, top_k - 1);
    for (int i = 0; i < 10; i++) {
        unsigned int token_idx = token_dist(gen);
        unsigned int k_idx = topk_dist(gen);
        topk_idx_host[token_idx * top_k + k_idx] = -1;
    }
}

void printUsage(const char* programName, int myRank) {
    if (myRank == 0) {
        printf("Usage: %s [OPTIONS]\n", programName);
        printf("Performance benchmark for NCCL EP operations\n\n");
        printf("Options:\n");
        printf("  --algorithm <mode>      Algorithm mode (default: ll)\n");
        printf("                          ll or low-latency:  Low latency mode\n");
        printf("                          ht or high-throughput:  High throughput mode\n");
        printf("  --tokens <num>          Number of tokens (default: LL=128, HT=4096)\n");
        printf("  --hidden <num>          Hidden dimension (default: 7168)\n");
        printf("  --top-k <num>           Top-k experts per token (default: 8)\n");
        printf("  --experts <num>         Total number of experts (default: 256)\n");
        printf("  --warmup <num>          Warmup iterations (default: 10)\n");
        printf("  --iters <num>           Benchmark iterations (default: 50)\n");
        printf("  --use-fp8               Use FP8 for dispatch (default: BF16)\n");
        printf("  --profile               Enable NVTX profiling mode (use with nsys)\n");
        printf("  --disable-nvlink        Disable NVLink, force RDMA for intranode communication (LL only)\n");
        printf("  --validate              Validate dispatch/combine data correctness\n");
        printf("  --dynamic-tokens        Enable dynamic token allocation (HT only, required for random topk)\n");
        printf("  --help                  Show this help message\n");
    }
}

int main(int argc, char* argv[]) {
    int myRank, nRanks, localRank = 0;

    // Default parameters
    ncclEpAlgorithm_t algorithm = NCCL_EP_ALGO_LOW_LATENCY;
    unsigned int num_tokens = 0;  // 0 means use algorithm-specific default
    unsigned int hidden = 7168;
    unsigned int top_k = 8;
    unsigned int num_experts = 256;
    int num_warmup = 10;
    int num_iters = 50;
    bool profile_mode = false;  // Enable NVTX profiling with nsys
    bool disable_nvlink = false;  // Force RDMA instead of NVLink
    bool use_fp8 = false;  // Use FP8 for dispatch (default: BF16)
    bool validate_data = false;  // Validate dispatch/combine correctness
    bool dynamic_tokens = false;  // Enable dynamic token allocation (HT only, for random topk)

    // Initialize MPI
    MPICHECK(MPI_Init(&argc, &argv));
    MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &myRank));
    MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &nRanks));

    // Parse command line arguments
    static struct option long_options[] = {
        {"algorithm",      required_argument, 0, 'a'},
        {"tokens",         required_argument, 0, 't'},
        {"hidden",         required_argument, 0, 'd'},
        {"top-k",          required_argument, 0, 'k'},
        {"experts",        required_argument, 0, 'e'},
        {"warmup",         required_argument, 0, 'w'},
        {"iters",          required_argument, 0, 'i'},
        {"profile",        no_argument,       0, 'p'},
        {"disable-nvlink", no_argument,       0, 'n'},
        {"use-fp8",        no_argument,       0, 'f'},
        {"validate",       no_argument,       0, 'V'},
        {"dynamic-tokens", no_argument,       0, 'M'},
        {"help",           no_argument,       0, 'h'},
        {0, 0, 0, 0}
    };

    int opt;
    int option_index = 0;
    while ((opt = getopt_long(argc, argv, "a:t:d:k:e:w:i:pnfVMh", long_options, &option_index)) != -1) {
        switch (opt) {
            case 'a':
                if (strcmp(optarg, "ll") == 0 || strcmp(optarg, "low-latency") == 0) {
                    algorithm = NCCL_EP_ALGO_LOW_LATENCY;
                } else if (strcmp(optarg, "ht") == 0 || strcmp(optarg, "high-throughput") == 0) {
                    algorithm = NCCL_EP_ALGO_HIGH_THROUGHPUT;
                } else {
                    if (myRank == 0) {
                        printf("Error: Invalid algorithm '%s'. Use 'll', 'low-latency', 'ht', or 'high-throughput'\n", optarg);
                    }
                    MPI_Finalize();
                    return 1;
                }
                break;
            case 't':
                num_tokens = static_cast<unsigned int>(atoi(optarg));
                break;
            case 'd':
                hidden = static_cast<unsigned int>(atoi(optarg));
                break;
            case 'k':
                top_k = static_cast<unsigned int>(atoi(optarg));
                break;
            case 'e':
                num_experts = static_cast<unsigned int>(atoi(optarg));
                break;
            case 'w':
                num_warmup = atoi(optarg);
                break;
            case 'n':
                disable_nvlink = true;
                break;
            case 'i':
                num_iters = atoi(optarg);
                break;
            case 'p':
                profile_mode = true;
                break;
            case 'f':
                use_fp8 = true;
                break;
            case 'V':
                validate_data = true;
                break;
            case 'M':
                dynamic_tokens = true;
                break;
            case 'h':
                printUsage(argv[0], myRank);
                MPI_Finalize();
                return 0;
            default:
                printUsage(argv[0], myRank);
                MPI_Finalize();
                return 1;
        }
    }

    // Set algorithm-specific default for num_tokens if not explicitly provided
    if (num_tokens == 0) {
        num_tokens = (algorithm == NCCL_EP_ALGO_HIGH_THROUGHPUT) ? 4096 : 128;
    }

    // Validate parameters
    if (num_experts % nRanks != 0) {
        if (myRank == 0) {
            printf("Error: num_experts (%u) must be divisible by nRanks (%d)\n", num_experts, nRanks);
        }
        MPI_Finalize();
        return 1;
    }

    // --dynamic-tokens (NCCL_EP_AUTO for max_tokens_per_rank) is intended for HT mode only.
    // Not yet supported in the current release; code paths are kept for future use.
    if (dynamic_tokens) {
        if (myRank == 0) {
            if (algorithm != NCCL_EP_ALGO_HIGH_THROUGHPUT)
                printf("Error: --dynamic-tokens is only applicable to HT mode (--algorithm ht)\n");
            else
                printf("Error: --dynamic-tokens (NCCL_EP_AUTO for max_tokens_per_rank) is not yet supported.\n"
                       "       This feature will be available in a future release for HT mode.\n");
        }
        MPI_Finalize();
        return 1;
    }

    unsigned int num_local_experts = num_experts / nRanks;

    // Calculate local rank based on hostname
    uint64_t hostHashs[nRanks];
    char hostname[1024];
    getHostName(hostname, 1024);
    hostHashs[myRank] = getHostHash(hostname);
    MPICHECK(MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, hostHashs, sizeof(uint64_t), MPI_BYTE, MPI_COMM_WORLD));
    for (int p = 0; p < nRanks; p++) {
        if (p == myRank) break;
        if (hostHashs[p] == hostHashs[myRank]) localRank++;
    }

    // Print configuration
    if (myRank == 0) {
        const char* algo_name = (algorithm == NCCL_EP_ALGO_LOW_LATENCY) ? "LOW_LATENCY" : "HIGH_THROUGHPUT";
        printf("=== NCCL EP Performance Benchmark ===\n");
        printf("Configuration:\n");
        printf("  Algorithm:       %s\n", algo_name);
        printf("  Ranks:           %d\n", nRanks);
        printf("  Tokens:          %u\n", num_tokens);
        printf("  Hidden:          %u\n", hidden);
        printf("  Top-k:           %u\n", top_k);
        printf("  Experts:         %u (local: %u)\n", num_experts, num_local_experts);
        printf("  Warmup iters:    %d\n", num_warmup);
        printf("  Benchmark iters: %d\n", num_iters);
        printf("  Dispatch dtype:  %s\n", use_fp8 ? "FP8" : "BF16");
        printf("  Profile mode:    %s\n", profile_mode ? "enabled" : "disabled");
        printf("  NVLink:          %s\n", disable_nvlink ? "disabled (force RDMA intranode, LL only)" : "enabled");
        printf("  Validate mode:   %s\n", validate_data ? "enabled" : "disabled");
        printf("  Dynamic tokens:  %s\n", dynamic_tokens ? "enabled (NCCL_EP_AUTO)" : "disabled");
        printf("\n");
    }

    // Disable NVLink/P2P if requested (LL mode only)
    // This forces RDMA communication even for intra-node communication
    // LL kernels use NCCL GIN, so NCCL_P2P_DISABLE is the relevant flag
    // NCCL_SHM_DISABLE is also set to avoid shared memory issues at scale
    if (disable_nvlink && algorithm == NCCL_EP_ALGO_LOW_LATENCY) {
        setenv("NCCL_P2P_DISABLE", "1", 1);
        setenv("NCCL_SHM_DISABLE", "1", 1);
    }

    // Setup CUDA
    CUDACHECK(cudaSetDevice(localRank));
    cudaStream_t stream;
    CUDACHECK(cudaStreamCreate(&stream));

    // Initialize NCCL
    ncclUniqueId id;
    ncclComm_t comm;
    if (myRank == 0) ncclGetUniqueId(&id);
    MPICHECK(MPI_Bcast(static_cast<void*>(&id), sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));
    NCCLCHECK(ncclCommInitRank(&comm, nRanks, id, myRank));

    // Create EP group
    if (myRank == 0) { printf("[DEBUG] Creating EP group...\n"); fflush(stdout); }
    ncclEpGroup_t ep_group;
    ncclEpGroupConfig_t config;
    config.version = 1;
    config.algorithm = algorithm;
    config.num_experts = num_experts;
    // max_tokens_per_rank is the per-rank batch size (max tokens any single rank will send).
    config.max_tokens_per_rank = dynamic_tokens ? NCCL_EP_AUTO : num_tokens;

    config.token_size_bytes = hidden * 2;  // bfloat16
    // Use NCCL_EP_AUTO for buffer sizes (required for dynamic tokens with larger batches)
    // For LL mode with disable_nvlink: NCCL_P2P_DISABLE env var handles NCCL GIN P2P
    config.rdma_buffer_size = NCCL_EP_AUTO;
    // num_qp_per_rank: LL mode requires >= num_local_experts, HT mode uses auto
    config.num_qp_per_rank = (algorithm == NCCL_EP_ALGO_LOW_LATENCY) ? num_local_experts : NCCL_EP_AUTO;
    config.num_channels = NCCL_EP_AUTO;

    printf("Rank %d: Testing ncclEpCreateGroup with algorithm: %s\n", myRank,
           (algorithm == NCCL_EP_ALGO_LOW_LATENCY) ? "LOW_LATENCY" : "HIGH_THROUGHPUT");
    MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
    double group_create_start = MPI_Wtime();
    NCCLCHECK(ncclEpCreateGroup(&ep_group, comm, &config, stream, cudaAllocCallback, cudaFreeCallback));
    CUDACHECK(cudaStreamSynchronize(stream));
    double group_create_end = MPI_Wtime();
    double group_create_ms = (group_create_end - group_create_start) * 1000.0;
    printf("Rank %d: ncclEpCreateGroup took %.2f ms\n", myRank, group_create_ms);

    // Initialize topk_idx tensor
    ncclNDTensor_t topk_idx;
    {
        void* topk_idx_data;
        CUDACHECK(cudaMalloc(&topk_idx_data, num_tokens * top_k * sizeof(int64_t)));
        NCCLCHECK(ncclEpTensorCreate(ep_group, &topk_idx, 2, ncclInt64, NCCL_EP_TENSOR_TAG_TOPK_IDX, topk_idx_data, num_tokens, top_k));
    }

    // Generate topk indices
    // HT: randperm (uniform), consistent with Hybrid-EP (test_hybrid_ep.py)
    // LL: abs(randn)+1 scores + topk + -1 masking, consistent with DeepEP (test_low_latency.py)
    int64_t *topk_idx_host = new int64_t[num_tokens * top_k];

    if (algorithm == NCCL_EP_ALGO_HIGH_THROUGHPUT) {
        generateTopkIndicesHT(topk_idx_host, num_tokens, num_experts, top_k, myRank);
        if (myRank == 0) {
            printf("Using randperm topk_idx for HT mode (uniform distribution)\n\n");
        }
    } else {
        generateRandomTopkIndicesLL(topk_idx_host, num_tokens, num_experts, top_k, myRank);
        if (myRank == 0) {
            printf("Using random topk_idx for LL mode (with -1 masking)\n\n");
        }
    }

    // Count valid token-expert pairs (excluding -1 masked entries)
    unsigned int num_valid_selections = 0;
    for (unsigned int i = 0; i < num_tokens * top_k; i++) {
        if (topk_idx_host[i] != -1) {
            num_valid_selections++;
        }
    }

    // Calculate byte metrics based on algorithm mode and FP8 setting
    LowLatencyBytes ll_bytes = {};
    HighThroughputBytes ht_bytes = {};
    if (algorithm == NCCL_EP_ALGO_LOW_LATENCY) {
        ll_bytes = calculateLowLatencyBytes(topk_idx_host, num_tokens, top_k, hidden, use_fp8);
    } else {
        ht_bytes = calculateHighThroughputBytes(
            topk_idx_host, num_tokens, top_k, num_experts, hidden, myRank, nRanks, use_fp8,
            ncclTeamLsa(comm).nRanks);
    }

    {
        void* topk_idx_data;
        NCCLCHECK(ncclEpTensorGetData(topk_idx, &topk_idx_data));
        CUDACHECK(cudaMemcpy(topk_idx_data, topk_idx_host, num_tokens * top_k * sizeof(int64_t), cudaMemcpyHostToDevice));
    }
    // Note: topk_idx_host is kept for validation, deleted at end

    // Create recv_expert_counter tensor for dynamic token allocation (HT + dynamic mode)
    ncclNDTensor_t recv_expert_counter_tensor = nullptr;
    if (dynamic_tokens) {
        void* recv_expert_counter_data;
        CUDACHECK(cudaHostAlloc(&recv_expert_counter_data, num_local_experts * sizeof(int), cudaHostAllocMapped));
        NCCLCHECK(ncclEpTensorCreate(ep_group, &recv_expert_counter_tensor, 1, ncclInt32,
                                     NCCL_EP_TENSOR_TAG_RECV_EXPERT_COUNTER_HOST, recv_expert_counter_data, num_local_experts));
    }

    // Create handle
    printf("Rank %d: Testing ncclEpCreateHandle\n", myRank);
    ncclEpHandle_t ep_handle;
    // ncclEpCreateHandle expects an array of local tensors and a count
    ncclNDTensor_t handle_local_tensors[1] = { recv_expert_counter_tensor };
    unsigned int handle_num_local_tensors = recv_expert_counter_tensor ? 1 : 0;
    MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
    double handle_create_start = MPI_Wtime();
    NCCLCHECK(ncclEpCreateHandle(&ep_handle, ep_group, topk_idx, handle_local_tensors, handle_num_local_tensors, nullptr, stream, use_fp8));
    CUDACHECK(cudaStreamSynchronize(stream));
    double handle_create_end = MPI_Wtime();
    double handle_create_ms = (handle_create_end - handle_create_start) * 1000.0;
    printf("Rank %d: ncclEpCreateHandle took %.2f ms\n", myRank, handle_create_ms);

    // max_tokens_per_rank is the per-rank dispatch count.
    // num_recv_tokens is the max tokens this rank can receive (nRanks * max_tokens_per_rank).
    unsigned int num_recv_tokens = 0;
    if (dynamic_tokens) {
        NCCLCHECK(ncclEpHandleGetNumRecvTokens(ep_handle, &num_recv_tokens));
        if (myRank == 0) {
            printf("[DEBUG] Dynamic tokens: num_recv_tokens=%u\n", num_recv_tokens);
            fflush(stdout);
        }
    } else {
        num_recv_tokens = config.max_tokens_per_rank * nRanks;
    }
    assert(num_recv_tokens);

    // HT recv bytes are pre-computed in calculateHighThroughputBytes via routing simulation
    if (algorithm == NCCL_EP_ALGO_HIGH_THROUGHPUT && myRank == 0) {
        printf("[DEBUG] HT bytes: send=%u tokens, rdma_send=%u, total_recv=%u tokens, rdma_recv=%u (buffer=%u)\n",
               ht_bytes.total_send_tokens, ht_bytes.rdma_send_tokens,
               ht_bytes.total_recv_tokens, ht_bytes.rdma_recv_tokens, num_recv_tokens);
        fflush(stdout);
    }

    // Setup benchmark tensors based on algorithm mode
    BenchmarkTensors tensors = {};

    if (myRank == 0) { printf("[DEBUG] Setting up tensors...\n"); fflush(stdout); }
    if (algorithm == NCCL_EP_ALGO_LOW_LATENCY) {
        setupLowLatencyTensors(ep_group, tensors, topk_idx, num_tokens, hidden, top_k,
                               num_local_experts, config.max_tokens_per_rank, nRanks);
    } else {
        setupHighThroughputTensors(ep_group, tensors, topk_idx, num_tokens, hidden, top_k,
                                   num_local_experts, num_recv_tokens);
    }
    if (myRank == 0) { printf("[DEBUG] Tensors set up\n"); fflush(stdout); }

    // Initialize validation data if enabled (fills tensors with rank-based patterns)
    if (validate_data) {
        if (myRank == 0) { printf("[DEBUG] Initializing validation data...\n"); fflush(stdout); }
        initializeValidationData(tensors, num_tokens, hidden, top_k, myRank,
                                 algorithm == NCCL_EP_ALGO_HIGH_THROUGHPUT);
        if (myRank == 0) { printf("[DEBUG] Validation data initialized\n"); fflush(stdout); }
    }

    ncclEpDispatchConfig_t dispatch_config;
    dispatch_config.round_scales = 0;

    // Synchronize before benchmarking
    MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
    CUDACHECK(cudaStreamSynchronize(stream));

    // Calculate data sizes for bandwidth calculation based on algorithm mode
    size_t dispatch_data_bytes, combine_data_bytes;
    if (algorithm == NCCL_EP_ALGO_LOW_LATENCY) {
        // LL mode: FP8 for dispatch, BF16 for combine
        dispatch_data_bytes = ll_bytes.dispatch_bytes;
        combine_data_bytes = ll_bytes.combine_bytes;
    } else {
        // HT mode: RDMA_send + total_recv (matches DeepEP methodology)
        dispatch_data_bytes = ht_bytes.rdma_send_bytes + ht_bytes.total_recv_bytes;
        combine_data_bytes = dispatch_data_bytes;  // Symmetric
    }

    // ==================== Paired Dispatch + Combine Benchmark ====================
    // Always run dispatch and combine paired to ensure correct internal state
    // (matching DeepEP's benchmarking approach)

    // Debug: print tensor setup for HT mode
    int num_dispatch_local = (algorithm == NCCL_EP_ALGO_HIGH_THROUGHPUT) ? 0 : 1;
    if (algorithm == NCCL_EP_ALGO_HIGH_THROUGHPUT && myRank == 0) {
        printf("HT Dispatch: %d inputs, %d outputs, %d local_tensors\n",
               tensors.num_dispatch_inputs, tensors.num_dispatch_outputs, num_dispatch_local);
        printf("HT Combine: %d inputs, %d outputs, %d local_tensors\n",
               tensors.num_combine_inputs, tensors.num_combine_outputs, tensors.num_combine_local_tensors);
        fflush(stdout);
    }
    if (myRank == 0) { printf("[DEBUG] Starting benchmark...\n"); fflush(stdout); }

    // HT mode: 0 local tensors, LL mode: 1 local tensor (tokens_per_experts)
    int num_dispatch_local_tensors = (algorithm == NCCL_EP_ALGO_HIGH_THROUGHPUT) ? 0 : 1;

    auto dispatch_fn = [&]() {
        NCCLCHECK(ncclEpDispatch(ep_handle, tensors.inputs, tensors.num_dispatch_inputs,
                                  tensors.outputs, tensors.num_dispatch_outputs,
                                  tensors.local_tensors, num_dispatch_local_tensors, false, &dispatch_config, stream));
        NCCLCHECK(ncclEpComplete(ep_handle, nullptr, stream));
    };

    auto combine_fn = [&]() {
        NCCLCHECK(ncclEpCombine(ep_handle, tensors.combine_inputs, tensors.num_combine_inputs,
                                 tensors.combine_outputs, tensors.num_combine_outputs,
                                 tensors.combine_local_tensors, tensors.num_combine_local_tensors,
                                 false, nullptr, stream));
        NCCLCHECK(ncclEpComplete(ep_handle, nullptr, stream));
    };

    // Use the requested number of iterations for both modes
    // HT mode uses "cached" mode for iterations after the first (handle state is reused)
    int actual_warmup = num_warmup;
    int actual_iters = num_iters;

    // CUPTI wraps the benchmark loop — records kernel GPU timestamps in hardware
    // alongside the cudaEvent timing, with zero interference.
    KernelTimer ktimer;
    ktimer.start();

    MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
    PairedBenchResult paired_result = runPairedBenchmark(
        dispatch_fn, combine_fn, actual_warmup, actual_iters,
        dispatch_data_bytes, combine_data_bytes, stream);

    ktimer.stop();

    // Extract individual results for printing
    BenchResult dispatch_result = paired_result.dispatch;
    BenchResult combine_result = paired_result.combine;
    BenchResult combined_result = paired_result.total;

    // ==================== NVTX Profiling Mode ====================
    if (profile_mode) {
        auto handle_create_fn = [&]() {
            NCCLCHECK(ncclEpHandleDestroy(ep_handle));
            NCCLCHECK(ncclEpCreateHandle(&ep_handle, ep_group, topk_idx,
                                          handle_local_tensors, handle_num_local_tensors,
                                          nullptr, stream, use_fp8));
        };
        runNvtxProfiling(myRank, actual_iters, dispatch_fn, combine_fn, handle_create_fn, stream);
    }

    // ==================== CUPTI Kernel-Only Timing (reduce before print) ====================
    // Debug: show all captured kernels on rank 0 (uncomment to inspect names)
    // ktimer.dump(myRank);

    double dispatch_kernel_us = ktimer.get_avg_us("dispatch_kernel");
    double combine_kernel_us  = ktimer.get_avg_us("combine_kernel");
    double global_dk_us = 0.0, global_ck_us = 0.0;
    MPI_Reduce(&dispatch_kernel_us, &global_dk_us, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&combine_kernel_us,  &global_ck_us, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    // Aggregate byte counts across ranks (HT only)
    size_t global_total_send = 0, global_rdma_send = 0;
    size_t global_total_recv = 0, global_rdma_recv = 0;
    if (algorithm == NCCL_EP_ALGO_HIGH_THROUGHPUT) {
        MPI_Reduce(&ht_bytes.total_send_bytes, &global_total_send, 1, MPI_UNSIGNED_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&ht_bytes.rdma_send_bytes,  &global_rdma_send,  1, MPI_UNSIGNED_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&ht_bytes.total_recv_bytes, &global_total_recv, 1, MPI_UNSIGNED_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&ht_bytes.rdma_recv_bytes,  &global_rdma_recv,  1, MPI_UNSIGNED_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    }

    // Print results and summary based on algorithm mode
    if (algorithm == NCCL_EP_ALGO_LOW_LATENCY) {
        printLowLatencyResults(myRank, nRanks, dispatch_result, combine_result, combined_result, ll_bytes);
        if (myRank == 0) {
            double avg_dk_us = global_dk_us / nRanks;
            double avg_ck_us = global_ck_us / nRanks;
            printf("\n=== Kernel-Only Timing (CUPTI, avg across %d ranks) ===\n", nRanks);
            printf("dispatch_kernel: avg=%.2f us\n", avg_dk_us);
            printf("combine_kernel:  avg=%.2f us\n", avg_ck_us);
            if (avg_dk_us == 0.0 || avg_ck_us == 0.0) {
                printf("  NOTE: 0 us means no matching kernel was captured.\n");
                printf("  Uncomment ktimer.dump() above to inspect captured kernel names.\n");
            }
            fflush(stdout);
        }
    } else {
        printHighThroughputResults(myRank, nRanks, dispatch_result, combine_result, combined_result, ht_bytes,
                                   dispatch_kernel_us, combine_kernel_us,
                                   global_dk_us, global_ck_us,
                                   global_total_send, global_rdma_send,
                                   global_total_recv, global_rdma_recv);
        if (myRank == 0 && (global_dk_us == 0.0 || global_ck_us == 0.0)) {
            printf("  NOTE: 0 us means no matching kernel was captured.\n");
            printf("  Uncomment ktimer.dump() above to inspect captured kernel names.\n");
            fflush(stdout);
        }
    }

    // Aggregate group/handle creation times across ranks
    {
        double local_group_ms = group_create_ms;
        double local_handle_ms = handle_create_ms;
        double global_group_avg, global_group_min, global_group_max;
        double global_handle_avg, global_handle_min, global_handle_max;

        MPI_Reduce(&local_group_ms, &global_group_avg, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&local_group_ms, &global_group_min, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
        MPI_Reduce(&local_group_ms, &global_group_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&local_handle_ms, &global_handle_avg, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&local_handle_ms, &global_handle_min, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
        MPI_Reduce(&local_handle_ms, &global_handle_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

        if (myRank == 0) {
            global_group_avg /= nRanks;
            global_handle_avg /= nRanks;

            printf("\n=== Setup Timing (across %d ranks) ===\n", nRanks);
            printf("ncclEpCreateGroup:   avg=%.2f ms, min=%.2f ms, max=%.2f ms\n",
                   global_group_avg, global_group_min, global_group_max);
            printf("ncclEpCreateHandle:  avg=%.2f ms, min=%.2f ms, max=%.2f ms\n",
                   global_handle_avg, global_handle_min, global_handle_max);
        }
    }

    // ==================== Data Validation ====================
    if (validate_data) {
        MPICHECK(MPI_Barrier(MPI_COMM_WORLD));

        // Run one more dispatch+combine with validation
        if (myRank == 0) { printf("\n=== Data Validation ===\n"); fflush(stdout); }

        // Re-initialize validation data (benchmark may have modified it)
        initializeValidationData(tensors, num_tokens, hidden, top_k, myRank,
                                 algorithm == NCCL_EP_ALGO_HIGH_THROUGHPUT);

        // Run dispatch
        dispatch_fn();
        CUDACHECK(cudaStreamSynchronize(stream));
        MPICHECK(MPI_Barrier(MPI_COMM_WORLD));

        ValidationResult dispatch_valid = validateDispatchOutput(
            tensors, num_tokens, hidden, top_k, num_experts, num_local_experts, myRank, nRanks,
            algorithm == NCCL_EP_ALGO_HIGH_THROUGHPUT);

        // Copy dispatch output to combine input (simulating expert processing)
        // In real usage, expert FFN processing would happen here
        // For validation, we just pass through the received tokens
        {
            void* eo_data;
            void* output0_data;
            NCCLCHECK(ncclEpTensorGetData(tensors.expert_outputs, &eo_data));
            NCCLCHECK(ncclEpTensorGetData(tensors.outputs[0], &output0_data));

            if (algorithm == NCCL_EP_ALGO_HIGH_THROUGHPUT) {
                // HT mode: 2D [num_recv_tokens, hidden]
                const unsigned int* eo_sizes;
                unsigned int eo_ndim;
                NCCLCHECK(ncclEpTensorGetSizes(tensors.expert_outputs, &eo_sizes, &eo_ndim));
                size_t data_size = eo_sizes[0] * eo_sizes[1] * sizeof(uint16_t);
                CUDACHECK(cudaMemcpy(eo_data, output0_data, data_size, cudaMemcpyDeviceToDevice));
            } else {
                // LL mode: 3D [num_local_experts, max_tokens_per_expert, hidden]
                const unsigned int* out0_sizes;
                unsigned int out0_ndim;
                NCCLCHECK(ncclEpTensorGetSizes(tensors.outputs[0], &out0_sizes, &out0_ndim));
                size_t data_size = out0_sizes[0] * out0_sizes[1] * out0_sizes[2] * sizeof(uint16_t);
                CUDACHECK(cudaMemcpy(eo_data, output0_data, data_size, cudaMemcpyDeviceToDevice));
            }
        }

        // Run combine
        combine_fn();
        CUDACHECK(cudaStreamSynchronize(stream));
        MPICHECK(MPI_Barrier(MPI_COMM_WORLD));

        // Validate combine output
        ValidationResult combine_valid = validateCombineOutput(
            tensors, num_tokens, hidden, top_k, num_experts, myRank, nRanks,
            algorithm == NCCL_EP_ALGO_HIGH_THROUGHPUT, topk_idx_host);

        // Print validation results (rank 0 only to avoid clutter)
        if (myRank == 0) {
            printf("Dispatch validation: %s", dispatch_valid.passed ? "PASSED" : "FAILED");
            if (!dispatch_valid.passed) {
                printf(" (%s)", dispatch_valid.message.c_str());
            }
            printf("\n");

            printf("Combine validation:  %s", combine_valid.passed ? "PASSED" : "FAILED");
            if (!combine_valid.passed) {
                printf(" (%s)", combine_valid.message.c_str());
            }
            printf(" (calc_diff=%.6e)\n", combine_valid.max_diff);
            fflush(stdout);
        }

        // Collect validation results across all ranks
        int local_dispatch_pass = dispatch_valid.passed ? 1 : 0;
        int local_combine_pass = combine_valid.passed ? 1 : 0;
        int global_dispatch_pass, global_combine_pass;

        MPICHECK(MPI_Allreduce(&local_dispatch_pass, &global_dispatch_pass, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD));
        MPICHECK(MPI_Allreduce(&local_combine_pass, &global_combine_pass, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD));

        if (myRank == 0) {
            printf("\nGlobal validation: Dispatch=%s, Combine=%s\n",
                   global_dispatch_pass ? "PASSED" : "FAILED",
                   global_combine_pass ? "PASSED" : "FAILED");
            fflush(stdout);
        }
    }

    // Cleanup (order matters: tensors -> handle -> group -> comm)
    cleanupBenchmarkTensors(ep_group, tensors, topk_idx);
    delete[] topk_idx_host;  // Now safe to delete after validation

    NCCLCHECK(ncclEpHandleDestroy(ep_handle));

    // Cleanup recv_expert_counter if allocated (must be before group destroy)
    if (dynamic_tokens && recv_expert_counter_tensor != nullptr) {
        void* rec_data;
        ncclEpTensorGetData(recv_expert_counter_tensor, &rec_data);
        if (rec_data) cudaFreeHost(rec_data);
        ncclEpTensorDestroy(ep_group, recv_expert_counter_tensor);
    }

    NCCLCHECK(ncclEpGroupDestroy(ep_group, stream));
    ncclCommDestroy(comm);

    CUDACHECK(cudaStreamDestroy(stream));

    MPICHECK(MPI_Finalize());
    cudaDeviceReset();

    return 0;
}
