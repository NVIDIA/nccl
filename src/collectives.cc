/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2015-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "argcheck.h" // Need some checks here since we access comm
#include "collectives.h"
#include "config/collconfig.h"
#include "enqueue.h"
#include "nccl.h"
#include "nvtx_payload_schemas.h"

const char* ncclFuncToString(ncclFunc_t fn) {
  switch (fn) {
  case ncclFuncAllGather:
    return "AllGather";
  case ncclFuncAllReduce:
    return "AllReduce";
  case ncclFuncAlltoAll:
    return "AlltoAll";
  case ncclFuncBroadcast:
    return "Broadcast";
  case ncclFuncGather:
    return "Gather";
  case ncclFuncRecv:
    return "Recv";
  case ncclFuncReduce:
    return "Reduce";
  case ncclFuncReduceScatter:
    return "ReduceScatter";
  case ncclFuncScatter:
    return "Scatter";
  case ncclFuncSendRecv:
    return "SendRecv";
  case ncclFuncSend:
    return "Send";
  case ncclFuncPutSignal:
    return "PutSignal";
  case ncclFuncSignal:
    return "Signal";
  case ncclFuncWaitSignal:
    return "WaitSignal";
  default:
    return "Invalid";
  }
}

const char* ncclDevRedOpToString(ncclDevRedOp_t op) {
  switch (op) {
  case ncclDevSum:
    return "Sum";
  case ncclDevProd:
    return "Prod";
  case ncclDevMinMax:
    return "MinMax";
  case ncclDevPreMulSum:
    return "PreMulSum";
  case ncclDevSumPostDiv:
    return "SumPostDiv";
  default:
    return "Unknown";
  }
}

const char* ncclDatatypeToString(ncclDataType_t type) {
  switch (type) {
  case ncclInt8:
    return "ncclInt8";
  case ncclInt32:
    return "ncclInt32";
  case ncclUint32:
    return "ncclUint32";
  case ncclInt64:
    return "ncclInt64";
  case ncclUint64:
    return "ncclUint64";
  case ncclFloat16:
    return "ncclFloat16";
  case ncclFloat32:
    return "ncclFloat32";
  case ncclFloat64:
    return "ncclFloat64";
  case ncclBfloat16:
    return "ncclBfloat16";
  case ncclFloat8e4m3:
    return "ncclFloat8e4m3";
  case ncclFloat8e5m2:
    return "ncclFloat8e5m2";
  default:
    return "Unknown";
  }
}

const char* ncclAlgoToString(int algo) {
  switch (algo) {
  case NCCL_ALGO_TREE:
    return "TREE";
  case NCCL_ALGO_RING:
    return "RING";
  case NCCL_ALGO_COLLNET_DIRECT:
    return "COLLNET_DIRECT";
  case NCCL_ALGO_COLLNET_CHAIN:
    return "COLLNET_CHAIN";
  case NCCL_ALGO_NVLS:
    return "NVLS";
  case NCCL_ALGO_NVLS_TREE:
    return "NVLS_TREE";
  case NCCL_ALGO_PAT:
    return "PAT";
  default:
    return "Unknown";
  }
}

const char* ncclProtoToString(int proto) {
  switch (proto) {
  case NCCL_PROTO_LL:
    return "LL";
  case NCCL_PROTO_LL128:
    return "LL128";
  case NCCL_PROTO_SIMPLE:
    return "SIMPLE";
  default:
    return "Unknown";
  }
}

static inline ncclResult_t ncclAllGatherConfigImpl(const void* sendbuff, void* recvbuff, size_t sendcount,
                                                   ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream,
                                                   const ncclCollConfig_t* config) {
  // clang-format off
  struct ncclInfo info = {ncclFuncAllGather, "AllGather",
                          sendbuff, recvbuff, sendcount, datatype, ncclSum, 0, comm, stream, /* Args */
                          ALLGATHER_CHUNKSTEPS, ALLGATHER_SLICESTEPS};
  // clang-format on
  NCCLCHECK(ncclParseCollConfig(config, &info.collConfig));
  return ncclEnqueueCheck(&info);
}

NCCL_API(ncclResult_t, ncclAllGather, const void* sendbuff, void* recvbuff, size_t sendcount, ncclDataType_t datatype,
         ncclComm_t comm, cudaStream_t stream);
ncclResult_t ncclAllGather(const void* sendbuff, void* recvbuff, size_t sendcount, ncclDataType_t datatype,
                           ncclComm_t comm, cudaStream_t stream) {
  // Just pass the size of one message and not the total bytes sent/received.
  NVTX3_FUNC_WITH_PARAMS(AllGather, NcclNvtxParamsAllGather,
                         NVTX3_PAYLOAD(comm ? comm->commHash : 0, sendcount * ncclTypeSize(datatype)));
  return ncclAllGatherConfigImpl(sendbuff, recvbuff, sendcount, datatype, comm, stream, nullptr);
}

NCCL_API(ncclResult_t, ncclAllGatherConfig, const void* sendbuff, void* recvbuff, size_t sendcount,
         ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream, const ncclCollConfig_t* config);
ncclResult_t ncclAllGatherConfig(const void* sendbuff, void* recvbuff, size_t sendcount, ncclDataType_t datatype,
                                 ncclComm_t comm, cudaStream_t stream, const ncclCollConfig_t* config) {
  // Just pass the size of one message and not the total bytes sent/received.
  NVTX3_FUNC_WITH_PARAMS(AllGatherConfig, NcclNvtxParamsAllGather,
                         NVTX3_PAYLOAD(comm ? comm->commHash : 0, sendcount * ncclTypeSize(datatype)));
  return ncclAllGatherConfigImpl(sendbuff, recvbuff, sendcount, datatype, comm, stream, config);
}

static inline ncclResult_t ncclAlltoAllConfigImpl(const void* sendbuff, void* recvbuff, size_t count,
                                                  ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream,
                                                  const ncclCollConfig_t* config) {
  // clang-format off
  struct ncclInfo info = {ncclFuncAlltoAll, "AlltoAll",
                          sendbuff, recvbuff, count, datatype, ncclSum, 0, comm, stream, /* Args */
                          ALLTOALL_CHUNKSTEPS, ALLTOALL_SLICESTEPS};
  // clang-format on
  NCCLCHECK(ncclParseCollConfig(config, &info.collConfig));
  return ncclEnqueueCheck(&info);
}

NCCL_API(ncclResult_t, ncclAlltoAll, const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
         ncclComm* comm, cudaStream_t stream);
ncclResult_t ncclAlltoAll(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, ncclComm* comm,
                          cudaStream_t stream) {
  NVTX3_FUNC_WITH_PARAMS(AlltoAll, NcclNvtxParamsAlltoAll,
                         NVTX3_PAYLOAD(comm ? comm->commHash : 0, count * ncclTypeSize(datatype)));
  return ncclAlltoAllConfigImpl(sendbuff, recvbuff, count, datatype, comm, stream, nullptr);
}

NCCL_API(ncclResult_t, ncclAlltoAllConfig, const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
         ncclComm_t comm, cudaStream_t stream, const ncclCollConfig_t* config);
ncclResult_t ncclAlltoAllConfig(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
                                ncclComm_t comm, cudaStream_t stream, const ncclCollConfig_t* config) {
  NVTX3_FUNC_WITH_PARAMS(AlltoAllConfig, NcclNvtxParamsAlltoAll,
                         NVTX3_PAYLOAD(comm ? comm->commHash : 0, count * ncclTypeSize(datatype)));
  return ncclAlltoAllConfigImpl(sendbuff, recvbuff, count, datatype, comm, stream, config);
}

static inline ncclResult_t ncclAllReduceConfigImpl(const void* sendbuff, void* recvbuff, size_t count,
                                                   ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm,
                                                   cudaStream_t stream, const ncclCollConfig_t* config) {
  // clang-format off
  struct ncclInfo info = {ncclFuncAllReduce, "AllReduce",
                          sendbuff, recvbuff, count, datatype, op, 0, comm, stream, /* Args */
                          ALLREDUCE_CHUNKSTEPS, ALLREDUCE_SLICESTEPS};
  // clang-format on
  NCCLCHECK(ncclParseCollConfig(config, &info.collConfig));
  return ncclEnqueueCheck(&info);
}

NCCL_API(ncclResult_t, ncclAllReduce, const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
         ncclRedOp_t op, ncclComm* comm, cudaStream_t stream);
ncclResult_t ncclAllReduce(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, ncclRedOp_t op,
                           ncclComm* comm, cudaStream_t stream) {
  NVTX3_FUNC_WITH_PARAMS(AllReduce, NcclNvtxParamsAllReduce,
                         NVTX3_PAYLOAD(comm ? comm->commHash : 0, count * ncclTypeSize(datatype), op));
  return ncclAllReduceConfigImpl(sendbuff, recvbuff, count, datatype, op, comm, stream, nullptr);
}

NCCL_API(ncclResult_t, ncclAllReduceConfig, const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
         ncclRedOp_t op, ncclComm_t comm, cudaStream_t stream, const ncclCollConfig_t* config);
ncclResult_t ncclAllReduceConfig(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
                                 ncclRedOp_t op, ncclComm_t comm, cudaStream_t stream, const ncclCollConfig_t* config) {
  NVTX3_FUNC_WITH_PARAMS(AllReduceConfig, NcclNvtxParamsAllReduce,
                         NVTX3_PAYLOAD(comm ? comm->commHash : 0, count * ncclTypeSize(datatype), op));
  return ncclAllReduceConfigImpl(sendbuff, recvbuff, count, datatype, op, comm, stream, config);
}

static inline ncclResult_t ncclBroadcastConfigImpl(const void* sendbuff, void* recvbuff, size_t count,
                                                   ncclDataType_t datatype, int root, ncclComm_t comm,
                                                   cudaStream_t stream, const ncclCollConfig_t* config) {
  // clang-format off
  struct ncclInfo info = {ncclFuncBroadcast, "Broadcast",
                          sendbuff, recvbuff, count, datatype, ncclSum, root, comm, stream, /* Args */
                          BROADCAST_CHUNKSTEPS, BROADCAST_SLICESTEPS};
  // clang-format on
  NCCLCHECK(ncclParseCollConfig(config, &info.collConfig));
  return ncclEnqueueCheck(&info);
}

NCCL_API(ncclResult_t, ncclBroadcast, const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
         int root, ncclComm_t comm, cudaStream_t stream);
ncclResult_t ncclBroadcast(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, int root,
                           ncclComm_t comm, cudaStream_t stream) {
  NVTX3_FUNC_WITH_PARAMS(Broadcast, NcclNvtxParamsBroadcast,
                         NVTX3_PAYLOAD(comm ? comm->commHash : 0, count * ncclTypeSize(datatype), root));
  return ncclBroadcastConfigImpl(sendbuff, recvbuff, count, datatype, root, comm, stream, nullptr);
}

NCCL_API(ncclResult_t, ncclBroadcastConfig, const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
         int root, ncclComm_t comm, cudaStream_t stream, const ncclCollConfig_t* config);
ncclResult_t ncclBroadcastConfig(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, int root,
                                 ncclComm_t comm, cudaStream_t stream, const ncclCollConfig_t* config) {
  NVTX3_FUNC_WITH_PARAMS(BroadcastConfig, NcclNvtxParamsBroadcast,
                         NVTX3_PAYLOAD(comm ? comm->commHash : 0, count * ncclTypeSize(datatype), root));
  return ncclBroadcastConfigImpl(sendbuff, recvbuff, count, datatype, root, comm, stream, config);
}
/* Deprecated original "in place" function, similar to MPI */
NCCL_API(ncclResult_t, ncclBcast, void* buff, size_t count, ncclDataType_t datatype, int root, ncclComm_t comm,
         cudaStream_t stream);
ncclResult_t ncclBcast(void* buff, size_t count, ncclDataType_t datatype, int root, ncclComm_t comm,
                       cudaStream_t stream) {
  return ncclBroadcast(buff, buff, count, datatype, root, comm, stream);
}

static inline ncclResult_t ncclGatherConfigImpl(const void* sendbuff, void* recvbuff, size_t count,
                                                ncclDataType_t datatype, int root, ncclComm_t comm, cudaStream_t stream,
                                                const ncclCollConfig_t* config) {
  // clang-format off
  struct ncclInfo info = {ncclFuncGather, "Gather",
                          sendbuff, recvbuff, count, datatype, ncclSum, root, comm, stream, /* Args */
                          GATHER_CHUNKSTEPS, GATHER_SLICESTEPS};
  // clang-format on
  NCCLCHECK(ncclParseCollConfig(config, &info.collConfig));
  return ncclEnqueueCheck(&info);
}

NCCL_API(ncclResult_t, ncclGather, const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
         int root, ncclComm* comm, cudaStream_t stream);
ncclResult_t ncclGather(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, int root,
                        ncclComm* comm, cudaStream_t stream) {
  NVTX3_FUNC_WITH_PARAMS(Gather, NcclNvtxParamsGather,
                         NVTX3_PAYLOAD(comm ? comm->commHash : 0, count * ncclTypeSize(datatype), root));
  return ncclGatherConfigImpl(sendbuff, recvbuff, count, datatype, root, comm, stream, nullptr);
}

NCCL_API(ncclResult_t, ncclGatherConfig, const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
         int root, ncclComm_t comm, cudaStream_t stream, const ncclCollConfig_t* config);
ncclResult_t ncclGatherConfig(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, int root,
                              ncclComm_t comm, cudaStream_t stream, const ncclCollConfig_t* config) {
  NVTX3_FUNC_WITH_PARAMS(GatherConfig, NcclNvtxParamsGather,
                         NVTX3_PAYLOAD(comm ? comm->commHash : 0, count * ncclTypeSize(datatype), root));
  return ncclGatherConfigImpl(sendbuff, recvbuff, count, datatype, root, comm, stream, config);
}

static inline ncclResult_t ncclReduceConfigImpl(const void* sendbuff, void* recvbuff, size_t count,
                                                ncclDataType_t datatype, ncclRedOp_t op, int root, ncclComm_t comm,
                                                cudaStream_t stream, const ncclCollConfig_t* config) {
  // clang-format off
  struct ncclInfo info = {ncclFuncReduce, "Reduce",
                          sendbuff, recvbuff, count, datatype, op, root, comm, stream, /* Args */
                          REDUCE_CHUNKSTEPS, REDUCE_SLICESTEPS};
  // clang-format on
  NCCLCHECK(ncclParseCollConfig(config, &info.collConfig));
  return ncclEnqueueCheck(&info);
}

NCCL_API(ncclResult_t, ncclReduce, const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
         ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream);
ncclResult_t ncclReduce(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, ncclRedOp_t op,
                        int root, ncclComm_t comm, cudaStream_t stream) {
  NVTX3_FUNC_WITH_PARAMS(Reduce, NcclNvtxParamsReduce,
                         NVTX3_PAYLOAD(comm ? comm->commHash : 0, count * ncclTypeSize(datatype), root, op));
  return ncclReduceConfigImpl(sendbuff, recvbuff, count, datatype, op, root, comm, stream, nullptr);
}

NCCL_API(ncclResult_t, ncclReduceConfig, const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
         ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream, const ncclCollConfig_t* config);
ncclResult_t ncclReduceConfig(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
                              ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream,
                              const ncclCollConfig_t* config) {
  NVTX3_FUNC_WITH_PARAMS(ReduceConfig, NcclNvtxParamsReduce,
                         NVTX3_PAYLOAD(comm ? comm->commHash : 0, count * ncclTypeSize(datatype), root, op));
  return ncclReduceConfigImpl(sendbuff, recvbuff, count, datatype, op, root, comm, stream, config);
}

static inline ncclResult_t ncclReduceScatterConfigImpl(const void* sendbuff, void* recvbuff, size_t recvcount,
                                                       ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm,
                                                       cudaStream_t stream, const ncclCollConfig_t* config) {
  // clang-format off
  struct ncclInfo info = {ncclFuncReduceScatter, "ReduceScatter",
                          sendbuff, recvbuff, recvcount, datatype, op, 0, comm, stream, /* Args */
                          REDUCESCATTER_CHUNKSTEPS, REDUCESCATTER_SLICESTEPS};
  // clang-format on
  NCCLCHECK(ncclParseCollConfig(config, &info.collConfig));
  return ncclEnqueueCheck(&info);
}

NCCL_API(ncclResult_t, ncclReduceScatter, const void* sendbuff, void* recvbuff, size_t recvcount,
         ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream);
ncclResult_t ncclReduceScatter(const void* sendbuff, void* recvbuff, size_t recvcount, ncclDataType_t datatype,
                               ncclRedOp_t op, ncclComm* comm, cudaStream_t stream) {
  NVTX3_FUNC_WITH_PARAMS(ReduceScatter, NcclNvtxParamsReduceScatter,
                         NVTX3_PAYLOAD(comm ? comm->commHash : 0, recvcount * ncclTypeSize(datatype), op));
  return ncclReduceScatterConfigImpl(sendbuff, recvbuff, recvcount, datatype, op, comm, stream, nullptr);
}

NCCL_API(ncclResult_t, ncclReduceScatterConfig, const void* sendbuff, void* recvbuff, size_t recvcount,
         ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm, cudaStream_t stream, const ncclCollConfig_t* config);
ncclResult_t ncclReduceScatterConfig(const void* sendbuff, void* recvbuff, size_t recvcount, ncclDataType_t datatype,
                                     ncclRedOp_t op, ncclComm_t comm, cudaStream_t stream,
                                     const ncclCollConfig_t* config) {
  NVTX3_FUNC_WITH_PARAMS(ReduceScatterConfig, NcclNvtxParamsReduceScatter,
                         NVTX3_PAYLOAD(comm ? comm->commHash : 0, recvcount * ncclTypeSize(datatype), op));
  return ncclReduceScatterConfigImpl(sendbuff, recvbuff, recvcount, datatype, op, comm, stream, config);
}

static inline ncclResult_t ncclScatterConfigImpl(const void* sendbuff, void* recvbuff, size_t count,
                                                 ncclDataType_t datatype, int root, ncclComm_t comm,
                                                 cudaStream_t stream, const ncclCollConfig_t* config) {
  // clang-format off
  struct ncclInfo info = {ncclFuncScatter, "Scatter",
                          sendbuff, recvbuff, count, datatype, ncclSum, root, comm, stream, /* Args */
                          SCATTER_CHUNKSTEPS, SCATTER_SLICESTEPS};
  // clang-format on
  NCCLCHECK(ncclParseCollConfig(config, &info.collConfig));
  return ncclEnqueueCheck(&info);
}

NCCL_API(ncclResult_t, ncclScatter, const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
         int root, ncclComm* comm, cudaStream_t stream);
ncclResult_t ncclScatter(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, int root,
                         ncclComm* comm, cudaStream_t stream) {
  NVTX3_FUNC_WITH_PARAMS(Scatter, NcclNvtxParamsScatter,
                         NVTX3_PAYLOAD(comm ? comm->commHash : 0, count * ncclTypeSize(datatype), root));
  return ncclScatterConfigImpl(sendbuff, recvbuff, count, datatype, root, comm, stream, nullptr);
}

NCCL_API(ncclResult_t, ncclScatterConfig, const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
         int root, ncclComm_t comm, cudaStream_t stream, const ncclCollConfig_t* config);
ncclResult_t ncclScatterConfig(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, int root,
                               ncclComm_t comm, cudaStream_t stream, const ncclCollConfig_t* config) {
  NVTX3_FUNC_WITH_PARAMS(ScatterConfig, NcclNvtxParamsScatter,
                         NVTX3_PAYLOAD(comm ? comm->commHash : 0, count * ncclTypeSize(datatype), root));
  return ncclScatterConfigImpl(sendbuff, recvbuff, count, datatype, root, comm, stream, config);
}

NCCL_API(ncclResult_t, ncclSend, const void* sendbuff, size_t count, ncclDataType_t datatype, int peer, ncclComm_t comm,
         cudaStream_t stream);
ncclResult_t ncclSend(const void* sendbuff, size_t count, ncclDataType_t datatype, int peer, ncclComm_t comm,
                      cudaStream_t stream) {
  NVTX3_FUNC_WITH_PARAMS(Send, NcclNvtxParamsSendRecv,
                         NVTX3_PAYLOAD(comm ? comm->commHash : 0, count * ncclTypeSize(datatype), peer));

  struct ncclInfo info = {ncclFuncSend,
                          "Send",
                          NULL,
                          (void*)sendbuff,
                          count,
                          datatype,
                          ncclSum,
                          peer,
                          comm,
                          stream, /* Args */
                          1,
                          1};
  return ncclEnqueueCheck(&info);
}

NCCL_API(ncclResult_t, ncclRecv, void* recvbuff, size_t count, ncclDataType_t datatype, int peer, ncclComm_t comm,
         cudaStream_t stream);
ncclResult_t ncclRecv(void* recvbuff, size_t count, ncclDataType_t datatype, int peer, ncclComm_t comm,
                      cudaStream_t stream) {
  NVTX3_FUNC_WITH_PARAMS(Recv, NcclNvtxParamsSendRecv,
                         NVTX3_PAYLOAD(comm ? comm->commHash : 0, count * ncclTypeSize(datatype), peer));

  struct ncclInfo info = {ncclFuncRecv,
                          "Recv",
                          NULL,
                          recvbuff,
                          count,
                          datatype,
                          ncclSum,
                          peer,
                          comm,
                          stream, /* Args */
                          1,
                          1};
  return ncclEnqueueCheck(&info);
}

NCCL_API(ncclResult_t, ncclPutSignal, const void* localbuff, size_t count, ncclDataType_t datatype, int peer,
         ncclWindow_t peerWin, size_t peerWinOffset, int sigIdx, int ctx, unsigned int flags, ncclComm_t comm,
         cudaStream_t stream);
ncclResult_t ncclPutSignal(const void* localbuff, size_t count, ncclDataType_t datatype, int peer, ncclWindow_t peerWin,
                           size_t peerWinOffset, int sigIdx, int ctx, unsigned int flags, ncclComm_t comm,
                           cudaStream_t stream) {
  NVTX3_FUNC_WITH_PARAMS(PutSignal, NcclNvtxParamsPut,
                         NVTX3_PAYLOAD(comm ? comm->commHash : 0, count * ncclTypeSize(datatype), peer, ctx));

  struct ncclInfo info = {ncclFuncPutSignal,
                          "PutSignal",
                          localbuff,
                          NULL,
                          count,
                          datatype,
                          ncclSum,
                          peer,
                          comm,
                          stream, /* Args */
                          1,
                          1, /* chunkSteps, sliceSteps */
                          peerWinOffset,
                          peerWin,
                          sigIdx,
                          ctx,
                          flags, /* peerWinOffset, peerWin, sigIdx, ctx, flags */
                          0,
                          NULL}; /* nDesc, signalDescs */
  return ncclEnqueueCheck(&info);
}

NCCL_API(ncclResult_t, ncclSignal, int peer, int sigIdx, int ctx, unsigned int flags, ncclComm_t comm,
         cudaStream_t stream);
ncclResult_t ncclSignal(int peer, int sigIdx, int ctx, unsigned int flags, ncclComm_t comm, cudaStream_t stream) {
  NVTX3_FUNC_WITH_PARAMS(Signal, NcclNvtxParamsSignal, NVTX3_PAYLOAD(comm ? comm->commHash : 0, peer, ctx));

  struct ncclInfo info = {ncclFuncSignal,
                          "Signal",
                          NULL,
                          NULL,
                          0,
                          ncclInt8,
                          ncclSum,
                          peer,
                          comm,
                          stream, /* Args */
                          1,
                          1, /* chunkSteps, sliceSteps */
                          0,
                          NULL,
                          sigIdx,
                          ctx,
                          flags, /* peerWinOffset, peerWin, sigIdx, ctx, flags */
                          0,
                          NULL}; /* nDesc, signalDescs */
  return ncclEnqueueCheck(&info);
}

NCCL_API(ncclResult_t, ncclWaitSignal, int nDesc, ncclWaitSignalDesc_t* signalDescs, ncclComm_t comm,
         cudaStream_t stream);
ncclResult_t ncclWaitSignal(int nDesc, ncclWaitSignalDesc_t* signalDescs, ncclComm_t comm, cudaStream_t stream) {
  NVTX3_FUNC_WITH_PARAMS(WaitSignal, NcclNvtxParamsWaitSignal, NVTX3_PAYLOAD(comm ? comm->commHash : 0, nDesc, 0));

  struct ncclInfo info = {ncclFuncWaitSignal,
                          "WaitSignal",
                          NULL,
                          NULL,
                          0,
                          ncclInt32,
                          ncclSum,
                          0,
                          comm,
                          stream, /* Args */
                          1,
                          1, /* chunkSteps, sliceSteps */
                          0,
                          NULL,
                          0,
                          0,
                          0, /* peerWinOffset, peerWin, sigIdx, ctx, flags */
                          nDesc,
                          signalDescs}; /* nDesc, signalDescs */
  return ncclEnqueueCheck(&info);
}
