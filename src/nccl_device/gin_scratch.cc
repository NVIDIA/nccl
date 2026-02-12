/*************************************************************************
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "core.h"
#include "../device/symmetric/gin_scratch__funcs.h"

NCCL_API(ncclResult_t, ncclGinOutboxCreateRequirement,
  int nBlocks, int size_log2,
  ncclGinOutboxHandle* outHandle, ncclDevResourceRequirements* outReq
);

ncclResult_t ncclGinOutboxCreateRequirement(
    int nBlocks, int size_log2,
    ncclGinOutboxHandle* outHandle, ncclDevResourceRequirements* outReq
  ) {
  memset(outReq, 0, sizeof(*outReq));
  size_log2 = std::max<int>(size_log2, /*log2(128)=*/7);
  outHandle->size_log2 = size_log2;
  outReq->bufferSize = nBlocks*(sizeof(ncclGinOutboxState) + alignUp(1<<size_log2, alignof(ncclGinOutboxState)));
  outReq->bufferAlign = 128;
  outReq->outBufferHandle = &outHandle->bufHandle;
  outReq->ginCounterCount = nBlocks << ncclGinScratchMaxBufs_log2;
  outReq->outGinCounterStart = &outHandle->counter0;
  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclGinInboxA2ACreateRequirement,
  ncclTeam peers, int nBlocks, int size_log2,
  ncclGinInboxA2AHandle* outHandle, ncclDevResourceRequirements* outReq
);

ncclResult_t ncclGinInboxA2ACreateRequirement(
    ncclTeam peers, int nBlocks, int size_log2,
    ncclGinInboxA2AHandle* outHandle, ncclDevResourceRequirements* outReq
  ) {
  int nPeers = peers.nRanks - 1;
  memset(outReq, 0, sizeof(*outReq));
  outHandle->size_log2 = size_log2;
  outHandle->nPeers_rcp32 = idivRcp32(nPeers);
  outReq->bufferSize = nBlocks*(sizeof(ncclGinInboxA2AState) + alignUp(1<<size_log2, alignof(ncclGinInboxA2AState)));
  outReq->bufferAlign = 128;
  outReq->outBufferHandle = &outHandle->bufHandle;
  outReq->ginSignalCount = nBlocks*4*(nPeers + (1<<ncclGinScratchMaxBufs_log2));
  outReq->outGinSignalStart = &outHandle->signals;
  return ncclSuccess;
}
