#include "core.h"
#include "nccl_device/impl/ll_a2a__funcs.h"

NCCL_API_CXX(int, ncclLLA2ACalcSlots, int maxElts, int maxEltSize);
int ncclLLA2ACalcSlots(int maxElts, int maxEltSize) {
  return maxElts*divUp(maxEltSize, 8);
}

NCCL_API_CXX(ncclResult_t, ncclLLA2ACreateRequirement, int nBlocks, int nSlots, ncclLLA2AHandle* outHandle, ncclDevResourceRequirements* outReq);
ncclResult_t ncclLLA2ACreateRequirement(
    int nBlocks, int nSlots, ncclLLA2AHandle* outHandle,
    ncclDevResourceRequirements* outReq
  ) {
  outHandle->nSlots = nSlots;
  memset(outReq, 0, sizeof(*outReq));
  outReq->bufferSize = nBlocks*(1 + 2*nSlots)*16;
  outReq->bufferAlign = 16;
  outReq->outBufferHandle = &outHandle->bufHandle;
  return ncclSuccess;
}
