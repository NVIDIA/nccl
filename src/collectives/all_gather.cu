/*************************************************************************
 * Copyright (c) 2015-2018, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "nccl.h"
#include "core.h"
#include "common_coll.h"
#include "enqueue.h"
#include "collectives.h"

ncclResult_t ncclAllGatherFunc(const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream, ncclProf_t* nccl_prof) {
  size_t nbytes = count*ncclTypeSize(datatype);
  INFO(COLL,"AllGather: opCount %lx sendbuff %p recvbuff %p count %zi datatype %d op %d root %d comm %p [nranks=%d] stream %p", comm->opCount, sendbuff, recvbuff, count, datatype, op, root, comm, comm->nRanks, stream);
  if (comm->nRanks == 1) {
    if (sendbuff != recvbuff)
      CUDACHECK(cudaMemcpyAsync(recvbuff, sendbuff, nbytes, cudaMemcpyDeviceToDevice, stream));
  } else {
    NCCLCHECK(transportSaveProxies(ALLGATHER_SUBSTEPS, ALLGATHER_BUFCHUNKS, comm->nRanks-1, comm->nRanks, nbytes*comm->nRanks, proxyPatternRing, comm, nccl_prof));
    NCCLCHECK(saveKernel(ncclCollAllGather, sendbuff, recvbuff, nbytes, ncclInt8, op, root, comm, stream, nbytes*comm->nRanks, 1));
  }
  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclAllGather, const void* sendbuff, void* recvbuff, size_t sendcount,
    ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream, ncclProf_t* nccl_prof);
ncclResult_t ncclAllGather(const void* sendbuff, void* recvbuff, size_t sendcount,
    ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream, ncclProf_t* nccl_prof) {
  return ncclEnqueueCheck(ncclAllGatherFunc, "AllGather", sendbuff, recvbuff, sendcount, datatype,
          ncclSum, 0, comm, stream, nccl_prof);
}
