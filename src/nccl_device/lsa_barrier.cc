/*************************************************************************
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "core.h"
#include "nccl_device/impl/lsa_barrier__funcs.h"

NCCL_API(ncclResult_t, ncclLsaBarrierCreateRequirement, ncclTeam_t team, int nBarriers, ncclLsaBarrierHandle_t* outHandle, ncclDevResourceRequirements_t* outReq);
ncclResult_t ncclLsaBarrierCreateRequirement(
    ncclTeam_t team, int nBarriers, ncclLsaBarrierHandle_t* outHandle,
    ncclDevResourceRequirements_t* outReq
  ) {
  memset(outReq, 0, sizeof(*outReq));
  outHandle->nBarriers = nBarriers;
  outReq->bufferSize = (3*nBarriers + nBarriers*team.nRanks)*sizeof(uint32_t);
  outReq->bufferAlign = alignof(uint32_t);
  outReq->outBufferHandle = &outHandle->bufHandle;
  return ncclSuccess;
}
