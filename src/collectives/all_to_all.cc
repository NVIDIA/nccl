/*************************************************************************
 * Copyright (c) 2015-2019, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) Microsoft Corporation. Licensed under the MIT License.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "enqueue.h"
#include "collectives.h"
#include "graph/topo.h"

#include "msccl/msccl_lifecycle.h"

NCCL_API(ncclResult_t, ncclAllToAll, const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
  ncclComm_t comm, cudaStream_t stream);
ncclResult_t ncclAllToAll(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
  ncclComm_t comm, cudaStream_t stream) {
  if (mscclAvailable() && !mscclIsCaller()) {
    return mscclEnqueueCheck(
      sendbuff, nullptr, nullptr, recvbuff, nullptr, nullptr,
      count, datatype, 0, 0, ncclSum, mscclFuncAllToAll, comm, stream);
  }

  size_t rankOffset = count * ncclTypeSize(datatype);
  int nRanks;
  NCCLCHECK(ncclCommCount(comm, &nRanks));
  if (count == 0) return ncclSuccess;
  NCCLCHECK(ncclGroupStart());
  for (int r=0; r<nRanks; r++) {
    NCCLCHECK(ncclSend(((char*)sendbuff)+r*rankOffset, count, datatype, r, comm, stream));
    NCCLCHECK(ncclRecv(((char*)recvbuff)+r*rankOffset, count, datatype, r, comm, stream));
  }
  NCCLCHECK(ncclGroupEnd());
  return ncclSuccess;
}
