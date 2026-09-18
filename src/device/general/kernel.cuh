/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_DEVICE_GENERAL_KERNEL_H_
#define NCCL_DEVICE_GENERAL_KERNEL_H_

#include "../kernel_profiler.cuh"

#if __CUDA_ARCH__ >= 700
// __grid_constant__ appears to break cuda-gdb
#define NCCL_GRID_CONSTANT __grid_constant__
#else
#define NCCL_GRID_CONSTANT
#endif

// General kernel entrypoints take a leading compile-time `bool EnableProfiler`.
template <bool EnableProfiler>
__device__ __forceinline__ void ncclGenkRun_Broadcast_Ring_Simple(struct ncclGenkDevWorkArgs const* args);
template <bool EnableProfiler>
__device__ __forceinline__ void ncclGenkRun_Broadcast_Ring_LL(struct ncclGenkDevWorkArgs const* args);
template <bool EnableProfiler>
__device__ __forceinline__ void ncclGenkRun_Broadcast_Ring_LL128(struct ncclGenkDevWorkArgs const* args);

template <bool EnableProfiler, template <typename> typename Red, typename T>
__device__ __forceinline__ void ncclGenkRun_Reduce_Ring_Simple(struct ncclGenkDevWorkArgs const* args);
template <bool EnableProfiler, template <typename> typename Red, typename T>
__device__ __forceinline__ void ncclGenkRun_Reduce_Ring_LL(struct ncclGenkDevWorkArgs const* args);
template <bool EnableProfiler, template <typename> typename Red, typename T>
__device__ __forceinline__ void ncclGenkRun_Reduce_Ring_LL128(struct ncclGenkDevWorkArgs const* args);

template <bool EnableProfiler, template <typename> typename Red, typename T>
__device__ __forceinline__ void ncclGenkRun_AllReduce_Ring_Simple(struct ncclGenkDevWorkArgs const* args);
template <bool EnableProfiler, template <typename> typename Red, typename T>
__device__ __forceinline__ void ncclGenkRun_AllReduce_Tree_Simple(struct ncclGenkDevWorkArgs const* args);
template <bool EnableProfiler, template <typename> typename Red, typename T>
__device__ __forceinline__ void ncclGenkRun_AllReduce_Ring_LL(struct ncclGenkDevWorkArgs const* args);
template <bool EnableProfiler, template <typename> typename Red, typename T>
__device__ __forceinline__ void ncclGenkRun_AllReduce_Tree_LL(struct ncclGenkDevWorkArgs const* args);
template <bool EnableProfiler, template <typename> typename Red, typename T>
__device__ __forceinline__ void ncclGenkRun_AllReduce_Ring_LL128(struct ncclGenkDevWorkArgs const* args);
template <bool EnableProfiler, template <typename> typename Red, typename T>
__device__ __forceinline__ void ncclGenkRun_AllReduce_Tree_LL128(struct ncclGenkDevWorkArgs const* args);

template <bool EnableProfiler>
__device__ __forceinline__ void ncclGenkRun_AllGather_Ring_Simple(struct ncclGenkDevWorkArgs const* args);
template <bool EnableProfiler>
__device__ __forceinline__ void ncclGenkRun_AllGather_Ring_LL(struct ncclGenkDevWorkArgs const* args);
template <bool EnableProfiler>
__device__ __forceinline__ void ncclGenkRun_AllGather_Ring_LL128(struct ncclGenkDevWorkArgs const* args);

template <bool EnableProfiler, template <typename> typename Red, typename T>
__device__ __forceinline__ void ncclGenkRun_ReduceScatter_Ring_Simple(struct ncclGenkDevWorkArgs const* args);
template <bool EnableProfiler, template <typename> typename Red, typename T>
__device__ __forceinline__ void ncclGenkRun_ReduceScatter_Ring_LL(struct ncclGenkDevWorkArgs const* args);
template <bool EnableProfiler, template <typename> typename Red, typename T>
__device__ __forceinline__ void ncclGenkRun_ReduceScatter_Ring_LL128(struct ncclGenkDevWorkArgs const* args);

#endif // NCCL_DEVICE_GENERAL_KERNEL_H_
