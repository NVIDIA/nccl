#include "core.h"
#include "nccl_device/impl/mem_barrier__funcs.h"

NCCL_API_CXX(ncclResult_t, ncclLsaBarrierCreateRequirement, ncclTeam team, int nBarriers, ncclLsaBarrierHandle* outHandle, ncclDevResourceRequirements* outReq);
ncclResult_t ncclLsaBarrierCreateRequirement(
    ncclTeam team, int nBarriers, ncclLsaBarrierHandle* outHandle,
    ncclDevResourceRequirements* outReq
  ) {
  memset(outReq, 0, sizeof(*outReq));
  outHandle->nBarriers = nBarriers;
  outReq->bufferSize = (3*nBarriers + nBarriers*team.nRanks)*sizeof(uint32_t);
  outReq->bufferAlign = alignof(uint32_t);
  outReq->outBufferHandle = &outHandle->bufHandle;
  return ncclSuccess;
}
