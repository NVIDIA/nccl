/*************************************************************************
 * Copyright (c) 2015-2018, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "core.h"
#include "common_coll.h"
#include "enqueue.h"
#include "collectives.h"

ncclResult_t ncclBroadcastFunc(const void* sendbuff, void* recvbuff, const size_t count,
    ncclDataType_t datatype, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream) {
  size_t nbytes = count*ncclTypeSize(datatype);
  INFO(NCCL_COLL,"Broadcast: opCount %lx sendbuff %p recvbuff %p count %zi datatype %d op %d root %d comm %p [nranks=%d] stream %p", comm->opCount, sendbuff, recvbuff, count, datatype, op, root, comm, comm->nRanks, stream);
  if (comm->nRanks == 1) {
    if (sendbuff != recvbuff)
      CUDACHECK(cudaMemcpyAsync(recvbuff, sendbuff, nbytes, cudaMemcpyDeviceToDevice, stream));
  } else {
    NCCLCHECK(transportSaveProxies(BROADCAST_SUBSTEPS, BROADCAST_BUFCHUNKS, 1, 1, nbytes, proxyPatternFrom(root), comm));
    NCCLCHECK(saveKernel(ncclCollBroadcast, sendbuff, recvbuff, nbytes, ncclInt8, op, root, comm, stream, nbytes, 1));
  }

  return ncclSuccess;
}

/* Deprecated original "in place" function, similar to MPI */
NCCL_API(ncclResult_t, ncclBcast, void* buff, size_t count, ncclDataType_t datatype, int root,
    ncclComm_t comm, cudaStream_t stream);
ncclResult_t ncclBcast(void* buff, size_t count, ncclDataType_t datatype, int root,
    ncclComm_t comm, cudaStream_t stream) {
  return ncclEnqueueCheck(ncclBroadcastFunc, "Bcast", buff, buff, count, datatype,
          ncclSum, root, comm, stream);
}

NCCL_API(ncclResult_t, ncclBroadcast, const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, int root,
    ncclComm_t comm, cudaStream_t stream);
ncclResult_t ncclBroadcast(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, int root,
    ncclComm_t comm, cudaStream_t stream) {
  return ncclEnqueueCheck(ncclBroadcastFunc, "Broadcast", sendbuff, recvbuff, count, datatype,
          ncclSum, root, comm, stream);
}
