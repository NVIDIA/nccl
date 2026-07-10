/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information.
 ************************************************************************/

#ifndef NCCL_M2N_BENCH_COMMON_KERNELS_H_
#define NCCL_M2N_BENCH_COMMON_KERNELS_H_

#include <cstddef>
#include <cuda_runtime.h>

void benchInitSourceData(char* pBuffer, const size_t pLocalDims[], int nDims, int shardDim, int shardIdx,
                         int shardCount, cudaStream_t stream, int iteration = 0, int bufferId = 0);

bool benchValidateDestData(const char* pBuffer, const size_t pLocalDims[], int nDims, int shardDim, int shardIdx,
                           int shardCount, int worldRank, cudaStream_t stream, int iteration = 0, int bufferId = 0);

#endif // NCCL_M2N_BENCH_COMMON_KERNELS_H_
