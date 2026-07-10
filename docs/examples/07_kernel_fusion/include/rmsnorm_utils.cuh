/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef RMSNORM_UTILS_CUH_
#define RMSNORM_UTILS_CUH_

#include <cuda_runtime.h>
#include "nccl_device.h"
#include <cassert>
#include <random>
#include <cmath>
#include <cstring>

//==============================================================================
// Multimem operations (require SM 9.0+)
//==============================================================================
#if __CUDA_ARCH__ >= 900

/**
 * Load and reduce (sum) from multimem address across all LSA peers.
 * Uses PTX multimem.ld_reduce.global.add.f32 instruction.
 * @param addr  Multimem pointer (from ncclGetLsaMultimemPointer)
 * @return      Sum of values from all peers at this address
 */
__device__ __forceinline__ float multimemLoadSum(const float* addr) {
  float value;
  asm volatile("multimem.ld_reduce.global.add.f32 %0, [%1];"
               : "=f"(value)
               : "l"(__cvta_generic_to_global(addr))
               : "memory");
  return value;
}

/**
 * Store value to multimem address (broadcasts to all LSA peers).
 * Uses PTX multimem.st.global.b32 instruction.
 * @param addr  Multimem pointer (from ncclGetLsaMultimemPointer)
 * @param val   Value to store
 */
__device__ __forceinline__ void multimemStore(float* addr, float val) {
  asm volatile("multimem.st.global.b32 [%0], %1;"
               :: "l"(__cvta_generic_to_global(addr)), "f"(val)
               : "memory");
}

#else

__device__ __forceinline__ float multimemLoadSum(const float* addr) {
  (void)addr;  /* Suppress unused parameter warning */
  assert(false && "multimemLoadSum requires CUDA architecture >= 900 (sm_90 or higher)");
  return 0.0f;
}

__device__ __forceinline__ void multimemStore(float* addr, float val) {
  (void)addr;  /* Suppress unused parameter warning */
  (void)val;   /* Suppress unused parameter warning */
  assert(false && "multimemStore requires CUDA architecture >= 900 (sm_90 or higher)");
}

#endif // __CUDA_ARCH__ >= 900

/**
 * Block-level RMS Normalization
 *
 * Performs RMS normalization on a token using all threads in a block.
 * Uses shared memory for efficient parallel reduction.
 *
 * Algorithm:
 *   1. Each thread computes sum of squares for its elements
 *   2. Block-level reduction to get total sum of squares
 *   3. Compute RMS = sqrt(mean_sq + eps)
 *   4. Each thread normalizes its elements by dividing by RMS
 *
 * @param token_data        Pointer to token data (modified in-place)
 * @param hidden_dim        Dimensionality of the token
 * @param eps               Epsilon for numerical stability
 * @param reduction_buffer  Shared memory buffer (size >= blockDim.x floats)
 * @param coop              CTA cooperation handle (use ncclCoopCta() / coop.sync() for block sync)
 */
__device__ __forceinline__ void blockRMSNorm(
    float* token_data, int hidden_dim, float eps, float* reduction_buffer,
    ncclCoopCta coop) {
  float thread_sum = 0.0f;
  for (int i = threadIdx.x; i < hidden_dim; i += blockDim.x) {
    float val = token_data[i];
    thread_sum += val * val;
  }
  reduction_buffer[threadIdx.x] = thread_sum;
  coop.sync();

  for (int s = blockDim.x / 2; s >= 1; s /= 2) {
    if (threadIdx.x < s) {
      reduction_buffer[threadIdx.x] += reduction_buffer[threadIdx.x + s];
    }
    coop.sync();
  }

  float rms_scale = rsqrtf((reduction_buffer[0] / hidden_dim) + eps);

  for (int i = threadIdx.x; i < hidden_dim; i += blockDim.x) {
    token_data[i] *= rms_scale;
  }
  coop.sync();
}

/**
 * Initialize the data with random values
 * @param data          The data to initialize
 * @param tensor_size   The size of the tensor
 * @param rank          The rank of the process
 * @param initial_seed  The initial seed for the random number generator
 * @param clear_first   Whether to clear the data first
 */
inline void initialize_data(float* data, int tensor_size, int rank,
                            int initial_seed = 42, bool clear_first = true) {
  if (clear_first) {
    memset(data, 0, tensor_size * sizeof(float));
  }
  std::uniform_real_distribution<float> dis(0.0f, 1.0f);
  std::mt19937 gen(initial_seed + rank);
  for (int i = 0; i < tensor_size; i++) {
    data[i] += dis(gen);
  }
}

/**
 * Perform the RMSNorm operation on the host
 * @param data             The data to perform the RMSNorm operation on
 * @param tensor_size      The size of the tensor
 * @param sequence_length  The number of tokens in the sequence
 * @param hidden_size      The hidden dimension size
 * @param total_ranks      The total number of ranks
 * @param eps              Epsilon value for numerical stability
 * @param initial_seed     The initial seed for the random number generator
 */
inline void rms_norm_generate(float* data, int tensor_size, int sequence_length,
                              int hidden_size, int total_ranks, float eps,
                              int initial_seed = 42) {
  // Clear data only once to accumulate results from all ranks
  memset(data, 0, tensor_size * sizeof(float));
  for (int i = 0; i < total_ranks; i++) {
    initialize_data(data, tensor_size, i, initial_seed, false);
  }

  // Perform RMSNorm for each token
  for (int i = 0; i < sequence_length; i++) {
    float sum_of_squares = 0.0f;
    for (int j = 0; j < hidden_size; j++) {
      sum_of_squares += data[i * hidden_size + j] * data[i * hidden_size + j];
    }
    float rms = 1.0f / sqrtf(sum_of_squares / hidden_size + eps);
    for (int j = 0; j < hidden_size; j++) {
      data[i * hidden_size + j] *= rms;
    }
  }
}

/**
 * Verify the results
 * @param data           The data to verify
 * @param expected_data  The expected data
 * @param tensor_size    The size of the tensor
 * @param abs_tolerance  The absolute tolerance
 * @param rel_tolerance  The relative tolerance
 * @return true if the results are correct, false otherwise
 */
inline bool verify_results(const float* data, const float* expected_data,
                           int tensor_size, float abs_tolerance = 1e-5f,
                           float rel_tolerance = 1e-4f) {
  bool success = true;
  for (int i = 0; i < tensor_size; i++) {
    float abs_error = fabsf(data[i] - expected_data[i]);
    float rel_error = abs_error / (fabsf(expected_data[i]) + 1e-10f);
    if (abs_error >= abs_tolerance && rel_error >= rel_tolerance) {
      success = false;
    }
  }
  return success;
}

#endif // RMSNORM_UTILS_CUH_
