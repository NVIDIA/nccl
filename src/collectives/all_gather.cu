/*************************************************************************
 * Copyright (c) 2015-2018, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "core.h"
#include "common_coll.h"
#include "enqueue.h"
#include "collectives.h"

ncclResult_t ncclAllGatherFunc(const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream) {
  size_t nbytes = count*ncclTypeSize(datatype);
  INFO(COLL,"opCount %lx sendbuff %p recvbuff %p count %zi size %zi datatype %d op %d comm %p [nranks=%d] stream %p", comm->opCount, sendbuff, recvbuff, count, nbytes, datatype, op, comm, comm->nRanks, stream);
  if (comm->nRanks == 1) {
    if (sendbuff != recvbuff)
      CUDACHECK(cudaMemcpyAsync(recvbuff, sendbuff, nbytes, cudaMemcpyDeviceToDevice, stream));
  } else {
    NCCLCHECK(transportSaveProxies(ALLGATHER_SUBSTEPS, ALLGATHER_BUFCHUNKS, comm->nRanks-1, comm->nRanks, nbytes*comm->nRanks, proxyPatternRing, comm));
    NCCLCHECK(saveKernel(ncclCollAllGather, sendbuff, recvbuff, nbytes, ncclInt8, op, root, comm, stream, nbytes*comm->nRanks, 1));
  }
  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclAllGather, const void* sendbuff, void* recvbuff, size_t sendcount,
    ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream);
ncclResult_t ncclAllGather(const void* sendbuff, void* recvbuff, size_t sendcount,
    ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream) {
  return ncclEnqueueCheck(ncclAllGatherFunc, "AllGather", sendbuff, recvbuff, sendcount, datatype,
          ncclSum, 0, comm, stream);
}
