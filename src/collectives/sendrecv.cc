/*************************************************************************
 * Copyright (c) 2015-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "enqueue.h"
#include "collectives.h"
#include "argcheck.h" // Need some checks here since we access comm

struct NvtxParamsSendRecv {
    size_t bytes;
    int peer;
};
constexpr const nvtxPayloadSchemaEntry_t SendRecvSchema[] = {
    {0, NVTX_PAYLOAD_ENTRY_TYPE_SIZE, "Bytes"},
    {0, NVTX_PAYLOAD_ENTRY_TYPE_INT, "Peer rank", nullptr, 0, offsetof(NvtxParamsSendRecv, peer)}
};

NCCL_API(ncclResult_t, ncclSend, const void* sendbuff, size_t count, ncclDataType_t datatype, int peer,
    ncclComm_t comm, cudaStream_t stream);
ncclResult_t ncclSend(const void* sendbuff, size_t count, ncclDataType_t datatype, int peer,
    ncclComm_t comm, cudaStream_t stream) {
  NvtxParamsSendRecv payload{count * ncclTypeSize(datatype), peer};
  NVTX3_FUNC_WITH_PARAMS(Send, SendRecvSchema, payload)

  struct ncclInfo info = { ncclFuncSend, "Send",
    NULL, (void*)sendbuff, count, datatype, ncclSum, peer, comm, stream, /* Args */
    1, 1 };
  ncclResult_t ret;
  NCCLCHECK(ncclGroupStart());
  ret = ncclEnqueueCheck(&info);
  NCCLCHECK(ncclGroupEnd());
  return ret;
}

NCCL_API(ncclResult_t, ncclRecv, void* recvbuff, size_t count, ncclDataType_t datatype, int peer,
    ncclComm_t comm, cudaStream_t stream);
ncclResult_t ncclRecv(void* recvbuff, size_t count, ncclDataType_t datatype, int peer,
    ncclComm_t comm, cudaStream_t stream) {
  NvtxParamsSendRecv payload{count * ncclTypeSize(datatype), peer};
  NVTX3_FUNC_WITH_PARAMS(Recv, SendRecvSchema, payload)

  struct ncclInfo info = { ncclFuncRecv, "Recv",
    NULL, recvbuff, count, datatype, ncclSum, peer, comm, stream, /* Args */
    1, 1 };
  ncclResult_t ret;
  NCCLCHECK(ncclGroupStart());
  ret = ncclEnqueueCheck(&info);
  NCCLCHECK(ncclGroupEnd());
  return ret;
}
