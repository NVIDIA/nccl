/*************************************************************************
 * Copyright (c) 2015-2018, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "core.h"
#include "common_coll.h"
#include "enqueue.h"
#include "collectives.h"

ncclResult_t ncclReduceScatterFunc(const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream) {
  size_t nbytes = count*ncclTypeSize(datatype);
  INFO(NCCL_COLL,"ReduceScatter: opCount %lx sendbuff %p recvbuff %p count %zi datatype %d op %d root %d comm %p [nranks=%d] stream %p", comm->opCount, sendbuff, recvbuff, count, datatype, op, root, comm, comm->nRanks, stream);
  if (comm->nRanks == 1) {
    if (sendbuff != recvbuff)
      CUDACHECK(cudaMemcpyAsync(recvbuff, sendbuff, nbytes, cudaMemcpyDeviceToDevice, stream));
  } else {
    NCCLCHECK(transportSaveProxies(REDUCESCATTER_SUBSTEPS, REDUCESCATTER_BUFCHUNKS, comm->nRanks-1, comm->nRanks, nbytes*comm->nRanks, proxyPatternRing, comm));
    NCCLCHECK(saveKernel(ncclCollReduceScatter, sendbuff, recvbuff, count, datatype, op, root, comm, stream, nbytes*comm->nRanks, 1));
  }
  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclReduceScatter, const void* sendbuff, void* recvbuff, size_t recvcount,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream);
ncclResult_t ncclReduceScatter(const void* sendbuff, void* recvbuff, size_t recvcount,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream) {
  return ncclEnqueueCheck(ncclReduceScatterFunc, "ReduceScatter", sendbuff, recvbuff, recvcount, datatype,
          op, 0, comm, stream);
}
