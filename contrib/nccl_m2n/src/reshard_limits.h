/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information.
 ************************************************************************/

#ifndef NCCL_RESHARD_LIMITS_H_
#define NCCL_RESHARD_LIMITS_H_

/*
 * Central definition of all compile-time constants used by the
 * nccl-reshard library.  Every translation unit includes this header
 * instead of defining its own copy.
 */

/* Tensor dimensionality limit. */
#define MAX_TENSOR_DIMS 3

/* RING (hierarchical) algorithm array sizes. */
#define MAX_SOURCES 16
#define MAX_TARGETS 64
#define MAX_LOCAL_FOLLOWERS 128
#define MAX_WARP_GROUPS 15
#define MAX_SRC_WARPS 8

/* DIRECT algorithm array sizes. */
#define MAX_DIRECT_SOURCES 32
#define MAX_DIRECT_TARGETS 64

/* Default chunking parameters. */
#define DEFAULT_ELEMENTS_PER_CHUNK 32
#define CHUNK_SIZE_BYTES (256ULL * 1024ULL)

/* Default kernel-launch parameters.
 *
 * DEFAULT_KERNEL_MAX_NTHREADS must match the value baked into each
 * __launch_bounds__(DEFAULT_KERNEL_MAX_NTHREADS, 1) declaration on the
 * resharding kernels — keep them in sync (NCCL v2.30 register-pressure
 * fix from commit 420236f). */
#define DEFAULT_NUM_CTAS 8
#define DEFAULT_KERNEL_MAX_NTHREADS 512
#define DEFAULT_GIN_CONTEXT_COUNT 4

/* Cross-dim transpose threshold (bytes).  If the innermost transfer
   size is below this, the library transparently transposes the last
   two tensor dims to improve RDMA throughput. */
#ifndef CROSS_DIM_TRANSPOSE_THRESHOLD
#define CROSS_DIM_TRANSPOSE_THRESHOLD (256ULL * 1024ULL)
#endif

/* Cache capacities. */
#define MAX_WINDOW_CACHE_ENTRIES 128
#define MAX_DEVCOMM_CACHE_ENTRIES 64
#define MAX_TRANSPOSE_BUFFER_ENTRIES 16

/* Stream pool — used when callers pass the default stream
 * (nullptr / cudaStreamLegacy / cudaStreamPerThread).  Caps how many
 * distinct (ncclComm_t, cuda device) entries the pool will track,
 * one stream + one back-edge event per entry (1:1 mapping).  The
 * effective entry count is driven by NCCL_RESHARD_STREAM_POOL_SIZE
 * (default 4); STREAM_POOL_MAX_SIZE is the hard upper bound on
 * that env-var setting.  See gReshardStreamPoolSize in
 * reshard_internal.h. */
#define STREAM_POOL_MAX_SIZE 256

/* Placement helpers (shared by host prep and device kernels). */
#define NCCL_RESHARD_REPLICATE (-1)
#define NCCL_RESHARD_SHARD(td) (td)
#define IS_SHARD_PLACEMENT(p) ((p) >= 0)
#define GET_SHARD_TENSOR_DIM(p) (p)

#endif /* NCCL_RESHARD_LIMITS_H_ */
