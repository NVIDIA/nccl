/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information.
 ************************************************************************/

/*************************************************************************
 * test_helpers.cu — kernel + host-launcher definitions for the
 * byte-pattern validators declared in test_helpers.h. This is the only
 * test-side TU that needs nvcc; the rest of tests/ is host-only and
 * compiles as plain .cc against host C++.
 ************************************************************************/

#include "test_helpers.h"

/* ======================================================================
 * Device kernels.
 * ====================================================================*/

static __global__ void testInitSourceDataKernel(char* buffer, size_t dim0, size_t dim1, size_t dim2, int ndims,
                                                size_t globalStart0, size_t globalStart1, size_t globalStart2,
                                                size_t globalDim1, size_t globalDim2) {
  size_t total = dim0 * dim1 * (ndims == 3 ? dim2 : 1);
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;

  for (size_t i = idx; i < total; i += stride) {
    size_t d2 = (ndims == 3) ? (i % dim2) : 0;
    size_t rem = (ndims == 3) ? (i / dim2) : i;
    size_t d1 = rem % dim1;
    size_t d0 = rem / dim1;

    size_t g0 = globalStart0 + d0;
    size_t g1 = globalStart1 + d1;
    size_t g2 = globalStart2 + d2;

    size_t globalIdx = g0 + g1 * globalDim1 + g2 * globalDim1 * globalDim2;
    buffer[i] = (char)(globalIdx % 256U);
  }
}

static __global__ void testValidateDestDataKernel(
  const char* buffer, size_t dim0, size_t dim1, size_t dim2, int ndims, size_t globalStart0, size_t globalStart1,
  size_t globalStart2, size_t globalDim1, size_t globalDim2, unsigned long long* errorCount, size_t* firstErrorIdx,
  char* firstErrorExpected, char* firstErrorActual) {
  size_t total = dim0 * dim1 * (ndims == 3 ? dim2 : 1);
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;

  for (size_t i = idx; i < total; i += stride) {
    size_t d2 = (ndims == 3) ? (i % dim2) : 0;
    size_t rem = (ndims == 3) ? (i / dim2) : i;
    size_t d1 = rem % dim1;
    size_t d0 = rem / dim1;

    size_t g0 = globalStart0 + d0;
    size_t g1 = globalStart1 + d1;
    size_t g2 = globalStart2 + d2;

    size_t globalIdx = g0 + g1 * globalDim1 + g2 * globalDim1 * globalDim2;
    char expected = (char)(globalIdx % 256U);
    char actual = buffer[i];

    if (actual != expected) {
      unsigned long long old = atomicAdd(errorCount, 1ULL);
      if (old == 0) {
        *firstErrorIdx = i;
        *firstErrorExpected = expected;
        *firstErrorActual = actual;
      }
    }
  }
}

/* ======================================================================
 * Host wrappers (definitions; declared in test_helpers.h).
 * ====================================================================*/

void testInitSourceData(char* buffer, const size_t localByteDims[3], int ndims, int shardDim, int shardIdx,
                        int shardCount, cudaStream_t stream) {
  size_t globalStart[3] = {0, 0, 0};
  size_t globalDims[3] = {localByteDims[0], localByteDims[1], ndims == 3 ? localByteDims[2] : 1};

  if (shardDim >= 0 && shardDim < ndims) {
    globalStart[shardDim] = (size_t)shardIdx * localByteDims[shardDim];
    globalDims[shardDim] = localByteDims[shardDim] * (size_t)shardCount;
  }

  int blockSize = 256;
  size_t total = localByteDims[0] * localByteDims[1] * (ndims == 3 ? localByteDims[2] : 1);
  if (total == 0) return;
  int numBlocks = (int)((total + blockSize - 1) / blockSize);
  if (numBlocks > 65535) numBlocks = 65535;

  testInitSourceDataKernel<<<numBlocks, blockSize, 0, stream>>>(
    buffer, localByteDims[0], localByteDims[1], ndims == 3 ? localByteDims[2] : 1, ndims, globalStart[0],
    globalStart[1], globalStart[2], globalDims[1], globalDims[2]);
}

bool testValidateDestData(const char* buffer, const size_t localByteDims[3], int ndims, int shardDim, int shardIdx,
                          int shardCount, int worldRank, cudaStream_t stream, unsigned long long* outErrorCount) {
  size_t globalStart[3] = {0, 0, 0};
  size_t globalDims[3] = {localByteDims[0], localByteDims[1], ndims == 3 ? localByteDims[2] : 1};

  if (shardDim >= 0 && shardDim < ndims) {
    globalStart[shardDim] = (size_t)shardIdx * localByteDims[shardDim];
    globalDims[shardDim] = localByteDims[shardDim] * (size_t)shardCount;
  }

  unsigned long long* dErrorCount = nullptr;
  size_t* dFirstIdx = nullptr;
  char* dFirstExp = nullptr;
  char* dFirstAct = nullptr;

  TEST_CUDACHECK(cudaMalloc(&dErrorCount, sizeof(unsigned long long)));
  TEST_CUDACHECK(cudaMalloc(&dFirstIdx, sizeof(size_t)));
  TEST_CUDACHECK(cudaMalloc(&dFirstExp, sizeof(char)));
  TEST_CUDACHECK(cudaMalloc(&dFirstAct, sizeof(char)));
  TEST_CUDACHECK(cudaMemset(dErrorCount, 0, sizeof(unsigned long long)));

  int blockSize = 256;
  size_t total = localByteDims[0] * localByteDims[1] * (ndims == 3 ? localByteDims[2] : 1);
  int numBlocks = total ? (int)((total + blockSize - 1) / blockSize) : 1;
  if (numBlocks > 65535) numBlocks = 65535;

  if (total > 0) {
    testValidateDestDataKernel<<<numBlocks, blockSize, 0, stream>>>(
      buffer, localByteDims[0], localByteDims[1], ndims == 3 ? localByteDims[2] : 1, ndims, globalStart[0],
      globalStart[1], globalStart[2], globalDims[1], globalDims[2], dErrorCount, dFirstIdx, dFirstExp, dFirstAct);
  }

  unsigned long long hErrorCount = 0;
  size_t hFirstIdx = 0;
  char hFirstExp = 0;
  char hFirstAct = 0;

  TEST_CUDACHECK(cudaMemcpyAsync(&hErrorCount, dErrorCount, sizeof(unsigned long long), cudaMemcpyDeviceToHost,
                                 stream));
  TEST_CUDACHECK(cudaMemcpyAsync(&hFirstIdx, dFirstIdx, sizeof(size_t), cudaMemcpyDeviceToHost, stream));
  TEST_CUDACHECK(cudaMemcpyAsync(&hFirstExp, dFirstExp, sizeof(char), cudaMemcpyDeviceToHost, stream));
  TEST_CUDACHECK(cudaMemcpyAsync(&hFirstAct, dFirstAct, sizeof(char), cudaMemcpyDeviceToHost, stream));
  TEST_CUDACHECK(cudaStreamSynchronize(stream));

  TEST_CUDACHECK(cudaFree(dErrorCount));
  TEST_CUDACHECK(cudaFree(dFirstIdx));
  TEST_CUDACHECK(cudaFree(dFirstExp));
  TEST_CUDACHECK(cudaFree(dFirstAct));

  if (outErrorCount) *outErrorCount = hErrorCount;

  if (hErrorCount > 0) {
    fprintf(stderr,
            "[Rank %d] VALIDATION FAILED: %llu errors, first at idx %zu "
            "(expected 0x%02x, got 0x%02x)\n",
            worldRank, hErrorCount, hFirstIdx, (unsigned char)hFirstExp, (unsigned char)hFirstAct);
    return false;
  }
  return true;
}
