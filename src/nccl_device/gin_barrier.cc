/*************************************************************************
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "core.h"
#include "nccl_device/impl/gin_barrier__funcs.h"

NCCL_API(ncclResult_t, ncclGinBarrierCreateRequirement, ncclComm_t comm, ncclTeam_t team, int nBarriers, ncclGinBarrierHandle_t* outHandle, ncclDevResourceRequirements_t* outReq);
ncclResult_t ncclGinBarrierCreateRequirement(
    ncclComm_t comm, ncclTeam_t team, int nBarriers,
    ncclGinBarrierHandle_t* outHandle, ncclDevResourceRequirements_t* outReq
  ) {
  memset(outReq, 0, sizeof(*outReq));
  outReq->bufferSize = nBarriers*NCCL_GIN_MAX_CONTEXTS*sizeof(uint32_t);
  outReq->bufferAlign = alignof(uint32_t);
  outReq->outBufferHandle = &outHandle->bufHandle;
  outReq->ginSignalCount = nBarriers;
  outReq->outGinSignalStart = &outHandle->signal0;
  return ncclSuccess;
}
