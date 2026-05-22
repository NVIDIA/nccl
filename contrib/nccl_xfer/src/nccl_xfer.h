/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information.
 ************************************************************************/

/*************************************************************************
 * NCCL Xfer — Public C API (reshard release)
 *
 * This header exposes the complete public API for the NCCL Xfer library.
 * The sole resharding entry point is ncclXferReshardWithWindow.
 *
 * Callers include this one header; internal implementation details are
 * in src/reshard_*.h (not installed).
 ************************************************************************/

#ifndef NCCL_XFER_H_
#define NCCL_XFER_H_

#include <limits.h>
#include <stddef.h>

#include "cuda_runtime.h"
#include "nccl.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ======================================================================
 * Mesh Specification
 * ====================================================================*/

/**
 * 2-D mesh descriptor for one side of a reshard, analogous to PyTorch
 * DTensor's DeviceMesh + placements or JAX's Mesh + PartitionSpec.
 * The mesh owns ranks [startRank, startRank + dims[0] * dims[1]).
 *
 *   dims       2-D mesh dimensions.  Each entry must be positive.
 *              The product is the number of ranks on this side.
 *   startRank First world rank in this side's contiguous rank interval.
 *   placement  Per-mesh-axis tensor placement; use the helpers below.
 *              placement[i] is one of:
 *                  NCCLXFER_RESHARD_REPLICATE   Axis replicates the tensor slice.
 *                  NCCLXFER_RESHARD_SHARD(d)    Axis shards tensor dimension d.
 *              Exactly one axis per mesh should be a SHARD.
 */
typedef struct {
  int dims[2];
  int startRank;
  int placement[2];
} ncclXferReshardMesh_t;

/* Placement helpers for mesh.placement[]: */
#define NCCLXFER_RESHARD_REPLICATE (-1)
#define NCCLXFER_RESHARD_SHARD(td) (td)

/* ======================================================================
 * Distributed Tensor Descriptor
 * ====================================================================*/

/* Maximum tensor rank handled by ncclXferReshardWithWindow.  Mirrors the
 * library-internal MAX_TENSOR_DIMS (kept in src/reshard_limits.h). */
#define NCCLXFER_RESHARD_MAX_TENSOR_DIMS 3

/**
 * Distributed tensor descriptor — the per-rank tile + the topology under
 * which the global tensor is split.  Modeled after PyTorch DTensor's
 * (local_tensor, DeviceMesh, placements) and JAX's
 * (jax.Array, NamedSharding(mesh, spec)) — both bundle topology with
 * the per-rank tile.
 *
 *   dataPtr    Local buffer for this rank, or NULL if this rank does
 *               not participate as this side.  The window passed to
 *               ncclXferReshardWithWindow must be registered with this
 *               buffer as its base (zero-offset contract).
 *   localShape Per-axis element count on this rank.  Only the first
 *               `ndims` entries are read.  Ignored when dataPtr is NULL.
 *   ndims       Number of tensor dimensions (1, 2, or 3).
 *   dtype       Element data type.  Supported: ncclInt8, ncclUint8,
 *               ncclFloat8e4m3, ncclFloat8e5m2, ncclFloat16, ncclBfloat16,
 *               ncclInt32, ncclUint32, ncclFloat32, ncclInt64, ncclUint64,
 *               ncclFloat64.
 *   mesh        Caller-owned mesh descriptor — topology + placement.
 *               Required on every rank, including ranks where dataPtr
 *               is NULL: the library uses both meshes everywhere to
 *               compute who-talks-to-whom.
 */
typedef struct {
  void* dataPtr;
  size_t localShape[NCCLXFER_RESHARD_MAX_TENSOR_DIMS];
  int ndims;
  ncclDataType_t dtype;
  const ncclXferReshardMesh_t* mesh;
} ncclXferDistTensor_t;

/* ======================================================================
 * Library Configuration
 *
 * Modeled after ncclConfig_t.  Callers fill an ncclXferReshardConfig_t with
 * NCCLXFER_RESHARD_CONFIG_INITIALIZER, optionally override fields, and pass
 * a pointer to ncclXferReshardInit().  Passing NULL is equivalent to passing
 * an all-default-initialized config.  Fields left at
 * NCCLXFER_RESHARD_CONFIG_UNDEF_INT keep the library default.
 *
 *   maxCta    Max number of CTAs used by reshard kernel.
 * ====================================================================*/

#define NCCLXFER_RESHARD_CONFIG_UNDEF_INT INT_MIN
#define NCCLXFER_RESHARD_API_MAGIC 0x52455348u /* 'RESH' */

typedef struct ncclXferReshardConfig_v1 {
  size_t size;
  unsigned int magic;
  int maxCta;
} ncclXferReshardConfig_t;

#define NCCLXFER_RESHARD_CONFIG_INITIALIZER \
  {                                         \
    sizeof(ncclXferReshardConfig_t),        \
    NCCLXFER_RESHARD_API_MAGIC,             \
    NCCLXFER_RESHARD_CONFIG_UNDEF_INT,      \
  }

/* ======================================================================
 * Library Lifecycle
 * ====================================================================*/

/* Initialize process-global reshard state.  Passing NULL uses defaults;
 * environment variables override matching config fields.  This call is
 * idempotent; ncclXferReshardWithWindow also initializes with defaults on
 * first use if not already initialized. */
ncclResult_t ncclXferReshardInit(ncclXferReshardConfig_t* config);

/* Release library-owned caches and temporary transpose buffers.  This call is
 * idempotent and does not destroy caller-owned comms, windows, streams, or
 * buffers. */
ncclResult_t ncclXferReshardFinalize(void);

/* ======================================================================
 * Resharding Entry Point
 * ====================================================================*/

/**
 * Single-shot resharding using a caller-registered window.
 *
 * Both descriptors are required on every rank — they each carry one
 * side's mesh, and the library reads both meshes everywhere to compute
 * which ranks own source data and which receive it.  A rank that does
 * not participate on a given side passes a fully-formed descriptor
 * with `dataPtr = NULL` on that side (mirroring PyTorch DTensor's
 * size-0 local tensor on non-participating ranks).
 *
 * @param comm    NCCL communicator containing all ranks (src + dst).
 * @param window  ncclWindow_t registered on `comm` covering this rank's
 *                local tensor buffer.
 * @param src     Source-side tensor descriptor (non-NULL on every rank).
 *                `dataPtr` may be NULL on dest-only ranks.  `mesh`,
 *                `ndims`, and `dtype` are required and must match
 *                `dst->ndims` / `dst->dtype`.
 * @param dst     Destination-side tensor descriptor (non-NULL on every
 *                rank).  `dataPtr` may be NULL on source-only ranks.
 * @param stream  Explicit CUDA stream, or the default stream (NULL /
 *                `cudaStreamLegacy` / `cudaStreamPerThread`).  Default-
 *                stream callers run on a library-owned non-blocking
 *                stream from a per-(comm, device) pool; a back-edge
 *                event makes subsequent default-stream work see the
 *                result.  Prior default-stream work is NOT waited on —
 *                pass an explicit stream if you need that ordering.
 *
 * @return ncclSuccess on success, ncclInvalidArgument if any
 *         precondition is violated.
 */
ncclResult_t ncclXferReshardWithWindow(ncclComm_t comm, ncclWindow_t window, const ncclXferDistTensor_t* src,
                                       const ncclXferDistTensor_t* dst, cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif /* NCCL_XFER_H_ */
