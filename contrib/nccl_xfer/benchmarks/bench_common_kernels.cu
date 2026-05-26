/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information.
 ************************************************************************/

#include "bench_common_kernels.h"
#include "bench_common.h" // CUDACHECK

#include <cstdio>
#include <cstdlib>

// ============================================================================
// Validation Kernels
// ============================================================================

static __global__ void benchInitSourceDataKernel(char* pBuffer, size_t dim0, size_t dim1, size_t dim2, int nDims,
                                                 size_t globalStart0, size_t globalStart1, size_t globalStart2,
                                                 size_t globalDim1, size_t globalDim2, unsigned salt) {
  size_t total = dim0 * dim1 * (nDims == 3 ? dim2 : 1);
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;

  for (size_t i = idx; i < total; i += stride) {
    size_t d2 = (nDims == 3) ? (i % dim2) : 0;
    size_t rem = (nDims == 3) ? (i / dim2) : i;
    size_t d1 = rem % dim1;
    size_t d0 = rem / dim1;

    size_t g0 = globalStart0 + d0;
    size_t g1 = globalStart1 + d1;
    size_t g2 = globalStart2 + d2;

    size_t globalIdx = g0 + g1 * globalDim1 + g2 * globalDim1 * globalDim2;
    pBuffer[i] = (char)((globalIdx + salt) % 256U);
  }
}

static __global__ void benchValidateDestDataKernel(
  const char* pBuffer, size_t dim0, size_t dim1, size_t dim2, int nDims, size_t globalStart0, size_t globalStart1,
  size_t globalStart2, size_t globalDim1, size_t globalDim2, unsigned salt, unsigned long long* pErrorCount,
  size_t* pFirstErrorIdx, char* pFirstErrorExpected, char* pFirstErrorActual) {
  size_t total = dim0 * dim1 * (nDims == 3 ? dim2 : 1);
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;

  for (size_t i = idx; i < total; i += stride) {
    size_t d2 = (nDims == 3) ? (i % dim2) : 0;
    size_t rem = (nDims == 3) ? (i / dim2) : i;
    size_t d1 = rem % dim1;
    size_t d0 = rem / dim1;

    size_t g0 = globalStart0 + d0;
    size_t g1 = globalStart1 + d1;
    size_t g2 = globalStart2 + d2;

    size_t globalIdx = g0 + g1 * globalDim1 + g2 * globalDim1 * globalDim2;
    char expected = (char)((globalIdx + salt) % 256U);
    char actual = pBuffer[i];

    if (actual != expected) {
      unsigned long long old = atomicAdd(pErrorCount, 1ULL);
      if (old == 0) {
        *pFirstErrorIdx = i;
        *pFirstErrorExpected = expected;
        *pFirstErrorActual = actual;
      }
    }
  }
}

// ============================================================================
// Host launchers
// ============================================================================

void benchInitSourceData(char* pBuffer, const size_t pLocalDims[], int nDims, int shardDim, int shardIdx,
                         int shardCount, cudaStream_t stream, int iteration, int bufferId) {
  size_t globalStart[3] = {0, 0, 0};
  size_t globalDims[3] = {pLocalDims[0], pLocalDims[1], nDims == 3 ? pLocalDims[2] : 1};

  if (shardDim >= 0) {
    globalStart[shardDim] = shardIdx * pLocalDims[shardDim];
    globalDims[shardDim] = pLocalDims[shardDim] * shardCount;
  }

  unsigned salt = (unsigned)iteration * 37U + (unsigned)bufferId * 131U;

  int blockSize = 256;
  size_t total = pLocalDims[0] * pLocalDims[1] * (nDims == 3 ? pLocalDims[2] : 1);
  int numBlocks = (total + blockSize - 1) / blockSize;

  benchInitSourceDataKernel<<<numBlocks, blockSize, 0, stream>>>(
    pBuffer, pLocalDims[0], pLocalDims[1], nDims == 3 ? pLocalDims[2] : 1, nDims, globalStart[0], globalStart[1],
    globalStart[2], globalDims[1], globalDims[2], salt);
}

bool benchValidateDestData(const char* pBuffer, const size_t pLocalDims[], int nDims, int shardDim, int shardIdx,
                           int shardCount, int worldRank, cudaStream_t stream, int iteration, int bufferId) {
  size_t globalStart[3] = {0, 0, 0};
  size_t globalDims[3] = {pLocalDims[0], pLocalDims[1], nDims == 3 ? pLocalDims[2] : 1};

  if (shardDim >= 0) {
    globalStart[shardDim] = shardIdx * pLocalDims[shardDim];
    globalDims[shardDim] = pLocalDims[shardDim] * shardCount;
  }

  unsigned salt = (unsigned)iteration * 37U + (unsigned)bufferId * 131U;

  unsigned long long* pDevErrorCount;
  size_t* pDevFirstErrorIdx;
  char* pDevFirstErrorExpected;
  char* pDevFirstErrorActual;

  CUDACHECK(cudaMalloc(&pDevErrorCount, sizeof(unsigned long long)));
  CUDACHECK(cudaMalloc(&pDevFirstErrorIdx, sizeof(size_t)));
  CUDACHECK(cudaMalloc(&pDevFirstErrorExpected, sizeof(char)));
  CUDACHECK(cudaMalloc(&pDevFirstErrorActual, sizeof(char)));
  CUDACHECK(cudaMemset(pDevErrorCount, 0, sizeof(unsigned long long)));

  int blockSize = 256;
  size_t total = pLocalDims[0] * pLocalDims[1] * (nDims == 3 ? pLocalDims[2] : 1);
  int numBlocks = (total + blockSize - 1) / blockSize;

  benchValidateDestDataKernel<<<numBlocks, blockSize, 0, stream>>>(
    pBuffer, pLocalDims[0], pLocalDims[1], nDims == 3 ? pLocalDims[2] : 1, nDims, globalStart[0], globalStart[1],
    globalStart[2], globalDims[1], globalDims[2], salt, pDevErrorCount, pDevFirstErrorIdx, pDevFirstErrorExpected,
    pDevFirstErrorActual);

  unsigned long long hErrorCount;
  size_t hFirstErrorIdx;
  char hFirstErrorExpected, hFirstErrorActual;

  CUDACHECK(cudaMemcpyAsync(&hErrorCount, pDevErrorCount, sizeof(unsigned long long), cudaMemcpyDeviceToHost, stream));
  CUDACHECK(cudaMemcpyAsync(&hFirstErrorIdx, pDevFirstErrorIdx, sizeof(size_t), cudaMemcpyDeviceToHost, stream));
  CUDACHECK(cudaMemcpyAsync(&hFirstErrorExpected, pDevFirstErrorExpected, sizeof(char), cudaMemcpyDeviceToHost,
                            stream));
  CUDACHECK(cudaMemcpyAsync(&hFirstErrorActual, pDevFirstErrorActual, sizeof(char), cudaMemcpyDeviceToHost, stream));
  CUDACHECK(cudaStreamSynchronize(stream));

  CUDACHECK(cudaFree(pDevErrorCount));
  CUDACHECK(cudaFree(pDevFirstErrorIdx));
  CUDACHECK(cudaFree(pDevFirstErrorExpected));
  CUDACHECK(cudaFree(pDevFirstErrorActual));

  if (hErrorCount > 0) {
    printf("[Rank %d] VALIDATION FAILED: %llu errors, first at idx %zu "
           "(expected 0x%02x, got 0x%02x)\n",
           worldRank, hErrorCount, hFirstErrorIdx, (unsigned char)hFirstErrorExpected,
           (unsigned char)hFirstErrorActual);
    return false;
  }

  return true;
}
