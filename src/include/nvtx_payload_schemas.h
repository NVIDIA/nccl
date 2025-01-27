/*************************************************************************
 * Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

/// Definitions of NVTX payload types and schemas used for the NVTX
/// instrumentation in init.cc and collectives.cc.

#ifndef NVTX_PAYLOAD_SCHEMAS_H_
#define NVTX_PAYLOAD_SCHEMAS_H_


#include "nccl.h"
#include "nvtx3/nvToolsExtPayload.h"
#include "nvtx3/nvToolsExtPayloadHelper.h"

/**
 * \brief Define a C struct together with the matching schema entries.
 *
 * Does the same as `NCCL_NVTX_DEFINE_STRUCT_WITH_SCHEMA`, but without creating the
 * schema attributes. (Remove this helper when it is available in the NVTX headers.)
 */
#define NCCL_NVTX_DEFINE_STRUCT_WITH_SCHEMA_ENTRIES(struct_id, prefix, entries) \
  _NVTX_PAYLOAD_TYPEDEF_STRUCT(struct_id, _NVTX_PAYLOAD_PASS_THROUGH entries) \
  prefix _NVTX_PAYLOAD_SCHEMA_INIT_ENTRIES(struct_id, _NVTX_PAYLOAD_PASS_THROUGH entries)

// C strings used as NVTX payload entry names.
static constexpr char const* nccl_nvtxCommStr = "NCCL communicator ID";
static constexpr char const* nccl_nvtxCudaDevStr = "CUDA device";
static constexpr char const* nccl_nvtxRankStr = "Rank";
static constexpr char const* nccl_nvtxNranksStr = "No. of ranks";
static constexpr char const* nccl_nvtxMsgSizeStr = "Message size [bytes]";
static constexpr char const* nccl_nvtxReductionOpStrpStr = "Reduction operation";

NCCL_NVTX_DEFINE_STRUCT_WITH_SCHEMA_ENTRIES(NcclNvtxParamsCommInitAll, static constexpr,
  NCCL_NVTX_PAYLOAD_ENTRIES(
    (uint64_t, commhash, TYPE_UINT64, nccl_nvtxCommStr),
    (int, ndev, TYPE_INT, "No. of devices")
  )
)

NCCL_NVTX_DEFINE_STRUCT_WITH_SCHEMA_ENTRIES(NcclNvtxParamsCommInitRank, static constexpr,
  NCCL_NVTX_PAYLOAD_ENTRIES(
    (uint64_t, newcomm, TYPE_UINT64, nccl_nvtxCommStr),
    (int, nranks, TYPE_INT, nccl_nvtxNranksStr),
    (int, myrank, TYPE_INT, nccl_nvtxRankStr),
    (int, cudaDev, TYPE_INT, nccl_nvtxCudaDevStr)
  )
)
// The typedef and payload schema for ncclCommInitRank is also used for,
// ncclCommInitRankConfig, ncclCommInitRankScalable, ncclCommDestroy, and ncclCommAbort.
typedef NcclNvtxParamsCommInitRank NcclNvtxParamsCommInitRankConfig;
typedef NcclNvtxParamsCommInitRank NcclNvtxParamsCommInitRankScalable;
typedef NcclNvtxParamsCommInitRank NcclNvtxParamsCommAbort;
typedef NcclNvtxParamsCommInitRank NcclNvtxParamsCommDestroy;

NCCL_NVTX_DEFINE_STRUCT_WITH_SCHEMA_ENTRIES(NcclNvtxParamsCommSplit, static constexpr,
  NCCL_NVTX_PAYLOAD_ENTRIES(
    (uint64_t, newcomm, TYPE_UINT64, nccl_nvtxCommStr),
    (uint64_t, parentcomm, TYPE_UINT64, "Parent NCCL communicator ID"),
    (int, nranks, TYPE_INT, nccl_nvtxNranksStr),
    (int, myrank, TYPE_INT, nccl_nvtxRankStr),
    (int, cudaDev, TYPE_INT, nccl_nvtxCudaDevStr),
    (int, color, TYPE_INT, "Color"),
    (int, key, TYPE_INT, "Key")
  )
)

NCCL_NVTX_DEFINE_STRUCT_WITH_SCHEMA_ENTRIES(NcclNvtxParamsCommFinalize, static constexpr,
  NCCL_NVTX_PAYLOAD_ENTRIES(
    (uint64_t, comm, TYPE_UINT64, nccl_nvtxCommStr)
  )
)

NCCL_NVTX_DEFINE_STRUCT_WITH_SCHEMA_ENTRIES(NcclNvtxParamsAllGather, static constexpr,
  NCCL_NVTX_PAYLOAD_ENTRIES(
    (uint64_t, comm, TYPE_UINT64, nccl_nvtxCommStr),
    (size_t, bytes, TYPE_SIZE, nccl_nvtxMsgSizeStr)
  )
)

NCCL_NVTX_DEFINE_STRUCT_WITH_SCHEMA_ENTRIES(NcclNvtxParamsAllReduce, static constexpr,
  NCCL_NVTX_PAYLOAD_ENTRIES(
    (uint64_t, comm, TYPE_UINT64, nccl_nvtxCommStr),
    (size_t, bytes, TYPE_SIZE, nccl_nvtxMsgSizeStr),
    (ncclRedOp_t, op, NCCL_REDOP, nccl_nvtxReductionOpStrpStr)
  )
)

NCCL_NVTX_DEFINE_STRUCT_WITH_SCHEMA_ENTRIES(NcclNvtxParamsBroadcast, static constexpr,
  NCCL_NVTX_PAYLOAD_ENTRIES(
    (uint64_t, comm, TYPE_UINT64, nccl_nvtxCommStr),
    (size_t, bytes, TYPE_SIZE, nccl_nvtxMsgSizeStr),
    (int, root, TYPE_INT, "Root")
  )
)

NCCL_NVTX_DEFINE_STRUCT_WITH_SCHEMA_ENTRIES(NcclNvtxParamsReduce, static constexpr,
  NCCL_NVTX_PAYLOAD_ENTRIES(
    (uint64_t, comm, TYPE_UINT64, nccl_nvtxCommStr),
    (size_t, bytes, TYPE_SIZE, nccl_nvtxMsgSizeStr),
    (int, root, TYPE_INT, "Root"),
    (ncclRedOp_t, op, NCCL_REDOP, nccl_nvtxReductionOpStrpStr)
  )
)

NCCL_NVTX_DEFINE_STRUCT_WITH_SCHEMA_ENTRIES(NcclNvtxParamsReduceScatter, static constexpr,
  NCCL_NVTX_PAYLOAD_ENTRIES(
    (uint64_t, comm, TYPE_UINT64, nccl_nvtxCommStr),
    (size_t, bytes, TYPE_SIZE, nccl_nvtxMsgSizeStr),
    (ncclRedOp_t, op, NCCL_REDOP, nccl_nvtxReductionOpStrpStr)
  )
)

// Used in NCCL APIs `ncclSend` and `ncclRecv`.
NCCL_NVTX_DEFINE_STRUCT_WITH_SCHEMA_ENTRIES(NcclNvtxParamsSendRecv, static constexpr,
  NCCL_NVTX_PAYLOAD_ENTRIES(
    (uint64_t, comm, TYPE_UINT64, nccl_nvtxCommStr),
    (size_t, bytes, TYPE_SIZE, nccl_nvtxMsgSizeStr),
    (int, peer, TYPE_INT, "Peer rank")
  )
)

#endif // end include guard
