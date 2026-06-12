/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

/// Definitions of NVTX payload types and schemas used for NCCL NVTX
/// instrumentation.

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
static constexpr char const* nccl_nvtxCollectiveStr = "Collective";
static constexpr char const* nccl_nvtxAlgorithmStr = "Algorithm";
static constexpr char const* nccl_nvtxProtocolStr = "Protocol";
static constexpr char const* nccl_nvtxDatatypeStr = "Datatype";
static constexpr char const* nccl_nvtxChannelCountStr = "No. of channels";

NCCL_NVTX_DEFINE_STRUCT_WITH_SCHEMA_ENTRIES(
  NcclNvtxParamsCommInitAll, static constexpr,
  NCCL_NVTX_PAYLOAD_ENTRIES((uint64_t, commhash, TYPE_UINT64, nccl_nvtxCommStr),
                            (int, ndev, TYPE_INT, "No. of devices")))

NCCL_NVTX_DEFINE_STRUCT_WITH_SCHEMA_ENTRIES(
  NcclNvtxParamsCommInitRank, static constexpr,
  NCCL_NVTX_PAYLOAD_ENTRIES((uint64_t, newcomm, TYPE_UINT64, nccl_nvtxCommStr),
                            (int, nranks, TYPE_INT, nccl_nvtxNranksStr), (int, myrank, TYPE_INT, nccl_nvtxRankStr),
                            (int, cudaDev, TYPE_INT, nccl_nvtxCudaDevStr)))
// The typedef and payload schema for ncclCommInitRank is also used for,
// ncclCommInitRankConfig, ncclCommInitRankScalable, ncclCommDestroy, ncclCommAbort, and ncclCommRevoke.
typedef NcclNvtxParamsCommInitRank NcclNvtxParamsCommInitRankConfig;
typedef NcclNvtxParamsCommInitRank NcclNvtxParamsCommInitRankScalable;
typedef NcclNvtxParamsCommInitRank NcclNvtxParamsCommAbort;
typedef NcclNvtxParamsCommInitRank NcclNvtxParamsCommDestroy;
typedef NcclNvtxParamsCommInitRank NcclNvtxParamsCommRevoke;

NCCL_NVTX_DEFINE_STRUCT_WITH_SCHEMA_ENTRIES(
  NcclNvtxParamsCommSplit, static constexpr,
  NCCL_NVTX_PAYLOAD_ENTRIES((uint64_t, newcomm, TYPE_UINT64, nccl_nvtxCommStr),
                            (uint64_t, parentcomm, TYPE_UINT64, "Parent NCCL communicator ID"),
                            (int, nranks, TYPE_INT, nccl_nvtxNranksStr), (int, myrank, TYPE_INT, nccl_nvtxRankStr),
                            (int, cudaDev, TYPE_INT, nccl_nvtxCudaDevStr), (int, color, TYPE_INT, "Color"),
                            (int, key, TYPE_INT, "Key")))

NCCL_NVTX_DEFINE_STRUCT_WITH_SCHEMA_ENTRIES(
  NcclNvtxParamsCommShrink, static constexpr,
  NCCL_NVTX_PAYLOAD_ENTRIES((uint64_t, newcomm, TYPE_UINT64, nccl_nvtxCommStr),
                            (int, nranks, TYPE_INT, nccl_nvtxNranksStr), (int, myrank, TYPE_INT, nccl_nvtxRankStr),
                            (int, cudaDev, TYPE_INT, nccl_nvtxCudaDevStr), (int, num_exclude, TYPE_INT, "num_exclude")))

NCCL_NVTX_DEFINE_STRUCT_WITH_SCHEMA_ENTRIES(
  NcclNvtxParamsCommGrow, static constexpr,
  NCCL_NVTX_PAYLOAD_ENTRIES((uint64_t, newcomm, TYPE_UINT64, nccl_nvtxCommStr),
                            (uint64_t, parentcomm, TYPE_UINT64, "Parent NCCL communicator ID"),
                            (int, nranks, TYPE_INT, nccl_nvtxNranksStr), (int, myrank, TYPE_INT, nccl_nvtxRankStr),
                            (int, cudaDev, TYPE_INT, nccl_nvtxCudaDevStr)))

NCCL_NVTX_DEFINE_STRUCT_WITH_SCHEMA_ENTRIES(NcclNvtxParamsCommFinalize, static constexpr,
                                            NCCL_NVTX_PAYLOAD_ENTRIES((uint64_t, comm, TYPE_UINT64, nccl_nvtxCommStr)))

NCCL_NVTX_DEFINE_STRUCT_WITH_SCHEMA_ENTRIES(NcclNvtxParamsAllGather, static constexpr,
                                            NCCL_NVTX_PAYLOAD_ENTRIES((uint64_t, comm, TYPE_UINT64, nccl_nvtxCommStr),
                                                                      (size_t, bytes, TYPE_SIZE, nccl_nvtxMsgSizeStr)))

NCCL_NVTX_DEFINE_STRUCT_WITH_SCHEMA_ENTRIES(NcclNvtxParamsAlltoAll, static constexpr,
                                            NCCL_NVTX_PAYLOAD_ENTRIES((uint64_t, comm, TYPE_UINT64, nccl_nvtxCommStr),
                                                                      (size_t, bytes, TYPE_SIZE, nccl_nvtxMsgSizeStr)))

NCCL_NVTX_DEFINE_STRUCT_WITH_SCHEMA_ENTRIES(
  NcclNvtxParamsAllReduce, static constexpr,
  NCCL_NVTX_PAYLOAD_ENTRIES((uint64_t, comm, TYPE_UINT64, nccl_nvtxCommStr),
                            (size_t, bytes, TYPE_SIZE, nccl_nvtxMsgSizeStr),
                            (ncclRedOp_t, op, NCCL_REDOP, nccl_nvtxReductionOpStrpStr)))

NCCL_NVTX_DEFINE_STRUCT_WITH_SCHEMA_ENTRIES(NcclNvtxParamsBroadcast, static constexpr,
                                            NCCL_NVTX_PAYLOAD_ENTRIES((uint64_t, comm, TYPE_UINT64, nccl_nvtxCommStr),
                                                                      (size_t, bytes, TYPE_SIZE, nccl_nvtxMsgSizeStr),
                                                                      (int, root, TYPE_INT, "Root")))

NCCL_NVTX_DEFINE_STRUCT_WITH_SCHEMA_ENTRIES(NcclNvtxParamsGather, static constexpr,
                                            NCCL_NVTX_PAYLOAD_ENTRIES((uint64_t, comm, TYPE_UINT64, nccl_nvtxCommStr),
                                                                      (size_t, bytes, TYPE_SIZE, nccl_nvtxMsgSizeStr),
                                                                      (int, root, TYPE_INT, "Root")))

NCCL_NVTX_DEFINE_STRUCT_WITH_SCHEMA_ENTRIES(
  NcclNvtxParamsReduce, static constexpr,
  NCCL_NVTX_PAYLOAD_ENTRIES((uint64_t, comm, TYPE_UINT64, nccl_nvtxCommStr),
                            (size_t, bytes, TYPE_SIZE, nccl_nvtxMsgSizeStr), (int, root, TYPE_INT, "Root"),
                            (ncclRedOp_t, op, NCCL_REDOP, nccl_nvtxReductionOpStrpStr)))

NCCL_NVTX_DEFINE_STRUCT_WITH_SCHEMA_ENTRIES(
  NcclNvtxParamsReduceScatter, static constexpr,
  NCCL_NVTX_PAYLOAD_ENTRIES((uint64_t, comm, TYPE_UINT64, nccl_nvtxCommStr),
                            (size_t, bytes, TYPE_SIZE, nccl_nvtxMsgSizeStr),
                            (ncclRedOp_t, op, NCCL_REDOP, nccl_nvtxReductionOpStrpStr)))

NCCL_NVTX_DEFINE_STRUCT_WITH_SCHEMA_ENTRIES(NcclNvtxParamsScatter, static constexpr,
                                            NCCL_NVTX_PAYLOAD_ENTRIES((uint64_t, comm, TYPE_UINT64, nccl_nvtxCommStr),
                                                                      (size_t, bytes, TYPE_SIZE, nccl_nvtxMsgSizeStr),
                                                                      (int, root, TYPE_INT, "Root")))

// Used in NCCL APIs `ncclSend` and `ncclRecv`.
NCCL_NVTX_DEFINE_STRUCT_WITH_SCHEMA_ENTRIES(NcclNvtxParamsSendRecv, static constexpr,
                                            NCCL_NVTX_PAYLOAD_ENTRIES((uint64_t, comm, TYPE_UINT64, nccl_nvtxCommStr),
                                                                      (size_t, bytes, TYPE_SIZE, nccl_nvtxMsgSizeStr),
                                                                      (int, peer, TYPE_INT, "Peer rank")))

NCCL_NVTX_DEFINE_STRUCT_WITH_SCHEMA_ENTRIES(NcclNvtxParamsPut, static constexpr,
                                            NCCL_NVTX_PAYLOAD_ENTRIES((uint64_t, comm, TYPE_UINT64, nccl_nvtxCommStr),
                                                                      (size_t, bytes, TYPE_SIZE, nccl_nvtxMsgSizeStr),
                                                                      (int, peer, TYPE_INT, "Peer rank"),
                                                                      (int, ctx, TYPE_INT, "Context ID")))

NCCL_NVTX_DEFINE_STRUCT_WITH_SCHEMA_ENTRIES(NcclNvtxParamsSignal, static constexpr,
                                            NCCL_NVTX_PAYLOAD_ENTRIES((uint64_t, comm, TYPE_UINT64, nccl_nvtxCommStr),
                                                                      (int, peer, TYPE_INT, "Peer rank"),
                                                                      (int, ctx, TYPE_INT, "Context ID")))

NCCL_NVTX_DEFINE_STRUCT_WITH_SCHEMA_ENTRIES(NcclNvtxParamsWaitSignal, static constexpr,
                                            NCCL_NVTX_PAYLOAD_ENTRIES((uint64_t, comm, TYPE_UINT64, nccl_nvtxCommStr),
                                                                      (int, npeers, TYPE_INT, "Number of peers"),
                                                                      (int, ctx, TYPE_INT, "Context ID")))

NCCL_NVTX_DEFINE_STRUCT_WITH_SCHEMA_ENTRIES(NcclNvtxParamsKernelTask, static constexpr,
  NCCL_NVTX_PAYLOAD_ENTRIES(
    (uint64_t, comm, TYPE_UINT64, nccl_nvtxCommStr),
    (int, rank, TYPE_INT, nccl_nvtxRankStr),
    (size_t, bytes, TYPE_SIZE, nccl_nvtxMsgSizeStr),
    (int, func, TYPE_INT, nccl_nvtxCollectiveStr),
    (int, algorithm, TYPE_INT, nccl_nvtxAlgorithmStr),
    (int, protocol, TYPE_INT, nccl_nvtxProtocolStr),
    (int, datatype, TYPE_INT, nccl_nvtxDatatypeStr),
    (int, root, TYPE_INT, "Root"),
    (int, nChannels, TYPE_INT, nccl_nvtxChannelCountStr)
  )
)

#endif // end include guard
