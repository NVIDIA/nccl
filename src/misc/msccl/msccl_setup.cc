/*************************************************************************
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT License.
 ************************************************************************/

#include "checks.h"
#include "collectives.h"
#include "proxy.h"
#include "transport.h"

#include "msccl/msccl_lifecycle.h"
#include "msccl/msccl_kernel.h"
#include "msccl/msccl_setup.h"
#include "msccl/msccl_status.h"

ncclResult_t mscclGetCaptureStatus(cudaStream_t stream) {
  mscclThreadLocalStatus& threadLocalStatus = mscclGetThreadLocalStatus();
  mscclSavedProxyArgs& savedProxyArgs = mscclGetSavedProxyArgs();
  cudaStreamCaptureStatus captureStatus;
  unsigned long long captureId;
  CUDACHECK(cudaStreamGetCaptureInfo_v2(stream, &captureStatus, &captureId, &threadLocalStatus.graph, nullptr, nullptr));
  if (captureStatus == cudaStreamCaptureStatusActive) {
    if (savedProxyArgs.count(captureId) == 0) {
      threadLocalStatus.captureStatus = mscclNewCapture;
      savedProxyArgs[captureId] = std::vector<struct mscclProxyArg>();
      // savedProxyArgs[captureId] = std::vector<mscclSavedCudaHostNodeParams>();
    } else {
      INFO(NCCL_INIT|NCCL_NET,"mscclGetCaptureStatus: captureId %llu is same with the previous one\n", captureId);
      threadLocalStatus.captureStatus = mscclExistingCapture;
    }
    threadLocalStatus.captureId = captureId;
  } else {
    threadLocalStatus.captureStatus = mscclNoCapture;
  }
  INFO(NCCL_INIT|NCCL_NET,"mscclGetCaptureStatus: %d, captureId: %llu, size: %lu\n", threadLocalStatus.captureStatus, threadLocalStatus.captureId, mscclGetSavedProxyArgs()[captureId].size());
  return ncclSuccess;
}

ncclResult_t mscclSetupCount(struct mscclAlgo* hostAlgo, ncclComm_t comm, size_t count, ncclDataType_t dataType) {
  mscclStatus& status = mscclGetStatus();
  status.stepSize = comm->buffSizes[hostAlgo->protocol] / NCCL_STEPS;
  status.chunkSteps = hostAlgo->protocol == NCCL_PROTO_SIMPLE ? hostAlgo->chunkSteps : 1;
  status.sliceSteps = hostAlgo->protocol == NCCL_PROTO_SIMPLE ? hostAlgo->sliceSteps : 1;
  status.chunkSize  = status.stepSize * status.chunkSteps;
  status.chunkEffectiveSize = status.chunkSize;
  if (hostAlgo->protocol == NCCL_PROTO_LL) status.chunkEffectiveSize /= 2;
  if (hostAlgo->protocol == NCCL_PROTO_LL128) status.chunkEffectiveSize = (status.chunkSize / NCCL_LL128_LINEELEMS) * NCCL_LL128_DATAELEMS;
  status.dataType = dataType;
  status.nBytes = count * ncclTypeSize(status.dataType) * hostAlgo->sizeMultiplier;
  status.maxAllowedCount = std::max((uint32_t)1, (uint32_t)(status.chunkEffectiveSize / DIVUP(status.nBytes, (size_t)(hostAlgo->nChunksPerLoop))));
  if (status.maxAllowedCount == 0){
    WARN("MSCCL: something went wrong. Max allowed count is 0\n");
    return ncclInternalError;
  }
  if (status.maxAllowedCount >= MSCCL_MAX_COUNT) {
    status.maxAllowedCount = MSCCL_MAX_COUNT - 1;
  }
  return ncclSuccess;
}

ncclResult_t mscclSetupScratch(struct mscclAlgo* hostAlgo, cudaStream_t stream) {
  mscclStatus& status = mscclGetStatus();
  size_t sizeNeeded = (status.nBytes * (size_t)(hostAlgo->nScratchChunks)) / (size_t)(hostAlgo->nChunksPerLoop);
  if (sizeNeeded > status.scratchBufferSize){
    NCCLCHECK(ncclCudaFree(status.scratchBuffer));
    NCCLCHECK(ncclCudaCalloc((char**)&status.scratchBuffer, sizeNeeded));
    status.scratchBufferSize = sizeNeeded;
  }
  return ncclSuccess;
}

ncclResult_t mscclSetupSyncFlags(cudaStream_t stream) {
  mscclStatus& status = mscclGetStatus();
  // if (mscclGetThreadLocalStatus().captureStatus == mscclNewCapture ||
  if ((status.graphEnabled && status.graphFirstKernel) ||
      status.workIndex > (1ULL << (8*sizeof(status.workIndex))) - 2 * NCCL_MAX_OPS - 1) {
    CUDACHECK(cudaMemsetAsync(status.syncFlags, 0, sizeof(struct mscclFlag) * MSCCL_MAX_NUM_THREAD_BLOCKS, stream));
    status.workIndex = 1; // setting the workIndex back to 1 for next iterations
    status.graphFirstKernel = false;
  }
  return ncclSuccess;
}

ncclResult_t mscclSetupConnections(struct mscclAlgo* hostAlgo, ncclComm_t comm) {
  // Check whether there is enough channels
  if (hostAlgo->nChannels > comm->nChannels) {
    WARN("MSCCL: number of channels available (%d) less than required (%d)", comm->nChannels, hostAlgo->nChannels);
    return ncclInvalidUsage;
  }

  // Flag MSCCL connections
  for (int i = 0; i < hostAlgo->nChannels; i++) {
    struct mscclChannelInfo* mCh = hostAlgo->mscclChannels + i;

    int sendPeers[MSCCL_MAX_NUM_THREAD_BLOCKS_PER_CHANNEL];
    for (int p = 0; p < mCh->nSendPeers; p++) {
      sendPeers[p] = mCh->sendPeerInfo[p].peer;
    }

    int recvPeers[MSCCL_MAX_NUM_THREAD_BLOCKS_PER_CHANNEL];
    for (int p = 0; p < mCh->nRecvPeers; p++) {
      recvPeers[p] = mCh->recvPeerInfo[p].peer;
    }

    NCCLCHECK(ncclTransportP2pConnect(comm, i, mCh->nRecvPeers, recvPeers, mCh->nSendPeers, sendPeers, 0 /*connIndex*/));
  }

  // Connect MSCCL connections
  mscclSetIsCallerFlag();
  NCCLCHECK(ncclTransportP2pSetup(comm, NULL, 0));
  mscclClearIsCallerFlag();

  return ncclSuccess;
}

// static ncclResult_t mscclSetupProxyImpl(struct mscclAlgo* hostAlgo, ncclComm_t comm, bool* justInquire) {
//   mscclStatus& status = mscclGetStatus();
//   mscclThreadLocalStatus& threadLocalStatus = mscclGetThreadLocalStatus();
//   struct ncclProxyOp proxyOp = {};
//   mscclSavedCudaHostNodeParams savedCudaHostNodeParams;

//   // proxyOp.connIndex = 0;
//   proxyOp.sliceSteps = status.sliceSteps;
//   proxyOp.chunkSteps = status.chunkSteps;
//   proxyOp.chunkSize = status.chunkSize;
//   proxyOp.protocol = hostAlgo->protocol;
//   proxyOp.dtype = status.dataType;
//   proxyOp.redOp = 0;
//   proxyOp.pattern = 0;
//   proxyOp.root = 0;
//   proxyOp.nbytes = status.stepSize*proxyOp.sliceSteps;
//   proxyOp.opCount = comm->collOpCount;
//   int nLoops = (int)(DIVUP(status.nBytes, (size_t)((size_t)hostAlgo->nChunksPerLoop*(size_t)status.chunkEffectiveSize)));
//   int nLoopsChunkSteps = nLoops * status.chunkSteps;
//   for (int ch = 0; ch < hostAlgo->nChannels; ch++) {
//     proxyOp.channelId = ch;
//     struct mscclChannelInfo* mscclChannel = hostAlgo->mscclChannels + ch;
//     struct ncclChannel* ncclChannel = comm->channels + ch;
//     for (int i = 0; i < mscclChannel->nRecvPeers; i++){
//       struct mscclChannelPeerInfo* recvPeer = mscclChannel->recvPeerInfo + i;
//       int nRecvs = 0;
//       for (int j = 0; j < recvPeer->nExistingCounts; j++){
//         int c = recvPeer->existingCounts[j];
//         int nStepsInCount = DIVUP(c+1, status.maxAllowedCount);
//         nRecvs += recvPeer->nTransmissionsOfCount[c] * nStepsInCount;
//       }
//       proxyOp.nsteps = nLoopsChunkSteps * nRecvs;
//       if (proxyOp.nsteps > 0) {
//         if (justInquire) {
//           savedCudaHostNodeParams[comm].emplace_back(ncclChannel, proxyRecv, recvPeer->peer, &proxyOp, 0);
//         }
//         else{
//           NCCLCHECK(mscclSaveProxy(ncclChannel, proxyRecv, recvPeer->peer, &proxyOp, 0, justInquire));
//         }
//       }
//     }
//     for (int i=0; i<mscclChannel->nSendPeers; i++){
//       struct mscclChannelPeerInfo* sendPeer = &mscclChannel->sendPeerInfo[i];
//       int nSends = 0;
//       for (int j = 0; j < sendPeer->nExistingCounts; j++){
//         int c = sendPeer->existingCounts[j];
//         int nStepsInCount = DIVUP(c+1, status.maxAllowedCount);
//         nSends += sendPeer->nTransmissionsOfCount[c] * nStepsInCount;
//       }
//       proxyOp.nsteps = nLoopsChunkSteps * nSends;
//       if (proxyOp.nsteps > 0) {
//         if (justInquire) {
//           savedCudaHostNodeParams[comm].emplace_back(ncclChannel, proxySend, sendPeer->peer, &proxyOp, 0);
//         }
//         else{
//           NCCLCHECK(SaveProxy(ncclChannel, proxySend, sendPeer->peer, &proxyOp, 0, justInquire));
//         }
//       }
//     }
//   }
//   if (justInquire) {
//     *justInquire=true;
//     INFO(NCCL_INIT|NCCL_NET,"mscclSetupProxyImpl: comm: %p proxy args size: %ld\n", comm, savedCudaHostNodeParams[comm].size());
//     mscclGetSavedProxyArgs()[threadLocalStatus.captureId].emplace_back(savedCudaHostNodeParams);
//     return ncclSuccess;
//   }
//   NCCLCHECK(ncclProxyStart(comm));
//   comm->collOpCount++;
//   return ncclSuccess;
// }

static ncclResult_t mscclSetupProxyImpl(struct mscclAlgo* hostAlgo, ncclComm_t comm, bool* justInquire) {
  mscclStatus& status = mscclGetStatus();
  struct ncclProxyOp proxyOp = {};

  // proxyOp.connIndex = 0;
  proxyOp.sliceSteps = status.sliceSteps;
  proxyOp.chunkSteps = status.chunkSteps;
  proxyOp.chunkSize = status.chunkSize;
  proxyOp.protocol = hostAlgo->protocol;
  proxyOp.dtype = status.dataType;
  proxyOp.redOp = 0;
  proxyOp.pattern = 0;
  proxyOp.root = 0;
  proxyOp.nbytes = status.stepSize*proxyOp.sliceSteps;
  proxyOp.opCount = comm->sharedRes->collOpCount;
  int nLoops = (int)(DIVUP(status.nBytes, (size_t)((size_t)hostAlgo->nChunksPerLoop*(size_t)status.chunkEffectiveSize)));
  int nLoopsChunkSteps = nLoops * status.chunkSteps;
  for (int ch = 0; ch < hostAlgo->nChannels; ch++) {
    proxyOp.channelId = ch;
    struct mscclChannelInfo* mscclChannel = hostAlgo->mscclChannels + ch;
    struct ncclChannel* ncclChannel = comm->channels + ch;
    for (int i = 0; i < mscclChannel->nRecvPeers; i++){
      struct mscclChannelPeerInfo* recvPeer = mscclChannel->recvPeerInfo + i;
      int nRecvs = 0;
      for (int j = 0; j < recvPeer->nExistingCounts; j++){
        int c = recvPeer->existingCounts[j];
        int nStepsInCount = DIVUP(c+1, status.maxAllowedCount);
        nRecvs += recvPeer->nTransmissionsOfCount[c] * nStepsInCount;
      }
      proxyOp.nsteps = nLoopsChunkSteps * nRecvs;
      if (proxyOp.nsteps > 0) {
        NCCLCHECK(mscclSaveProxy(comm, ncclChannel, proxyRecv, recvPeer->peer, &proxyOp, 0, justInquire));
      }
    }
    for (int i=0; i<mscclChannel->nSendPeers; i++){
      struct mscclChannelPeerInfo* sendPeer = &mscclChannel->sendPeerInfo[i];
      int nSends = 0;
      for (int j = 0; j < sendPeer->nExistingCounts; j++){
        int c = sendPeer->existingCounts[j];
        int nStepsInCount = DIVUP(c+1, status.maxAllowedCount);
        nSends += sendPeer->nTransmissionsOfCount[c] * nStepsInCount;
      }
      proxyOp.nsteps = nLoopsChunkSteps * nSends;
      if (proxyOp.nsteps > 0) {
        NCCLCHECK(mscclSaveProxy(comm, ncclChannel, proxySend, sendPeer->peer, &proxyOp, 0, justInquire));
      }
    }
  }
  NCCLCHECK(ncclProxyStart(comm));
  comm->sharedRes->collOpCount++;
  return ncclSuccess;
}


static void CUDART_CB mscclSetupProxyCallback(void *args) {
  std::vector<struct mscclProxyArg>* params = (std::vector<struct mscclProxyArg>*)args;
  INFO(NCCL_INIT|NCCL_NET,"mscclSetupProxyCallback: proxy args size: %ld\n", params->size());
  for (auto &p : *params) {
    mscclSetupProxyImpl(p.hostAlgo, p.comm, nullptr);
  }    
}

// static void CUDART_CB mscclSetupProxyCallback(void *args) {
//   std::vector<mscclSavedCudaHostNodeParams>* params = (std::vector<mscclSavedCudaHostNodeParams>*)args;
//   INFO(NCCL_INIT|NCCL_NET,"mscclSetupProxyCallback: proxy args size: %ld\n", params->size());
//   for (auto &p : *params) {
//     INFO(NCCL_INIT|NCCL_NET,"mscclSetupProxyCallback1: comm: %p, proxy args size: %ld\n", p.begin()->first, p.size());
//     for (auto &q : p) {
//       INFO(NCCL_INIT|NCCL_NET,"mscclSetupProxyCallback2: comm: %p, proxy args size: %ld\n", q.first, q.second.size());
//       for (auto &r : q.second) {
//         INFO(NCCL_INIT|NCCL_NET,"mscclSetupProxyCallback: comm: %p, channel: %p, opType: %d, peer: %d, proxyOp: %p, connIndex: %d\n", q.first, r.channel, r.opType, r.peer, r.proxyOp, r.connIndex);
//         mscclSaveProxy(r.channel, r.opType, r.peer, r.proxyOp, r.connIndex, nullptr);    
//       }
//       INFO(NCCL_INIT|NCCL_NET,"mscclSetupProxyCallback: start the proxy: comm: %p \n", q.first);
//       ncclProxyStart(q.first);
//       q.first->collOpCount++;
//     }
//   }
// }

ncclResult_t mscclSetupProxy(struct mscclAlgo* hostAlgo, ncclComm_t comm, cudaStream_t stream) {
  mscclThreadLocalStatus& threadLocalStatus = mscclGetThreadLocalStatus();
  mscclSavedProxyArgs& savedProxyArgs = mscclGetSavedProxyArgs();
  if (threadLocalStatus.captureStatus == mscclNoCapture) {
    INFO(NCCL_INIT|NCCL_NET,"mscclSetupProxy: no capture\n");
    NCCLCHECK(mscclSetupProxyImpl(hostAlgo, comm, nullptr));
  } else {
    INFO(NCCL_INIT|NCCL_NET,"mscclSetupProxy: capture\n");
    if (savedProxyArgs[threadLocalStatus.captureId].size() == 0) {
      INFO(NCCL_INIT|NCCL_NET,"mscclSetupProxy: adding callback\n");

      cudaGraphNode_t callbackNode;
      cudaHostNodeParams p;
      // p.fn = mscclSetupProxyCallback1;
      // auto params = new mscclCudaHostNodeParams(comm, threadLocalStatus.captureId);
      p.fn = mscclSetupProxyCallback;
      auto params = &savedProxyArgs[threadLocalStatus.captureId];
      p.userData = params;
      CUDACHECK(cudaGraphAddHostNode(&callbackNode, threadLocalStatus.graph, nullptr, 0, &p));
    }
    // bool justInquire = false;
    // NCCLCHECK(mscclSetupProxyImpl(hostAlgo, comm, &justInquire));
    mscclGetSavedProxyArgs()[threadLocalStatus.captureId].emplace_back(hostAlgo, comm);
  }
  return ncclSuccess;
}

static ncclResult_t hostToDevRedOp(
    ncclDevRedOpFull *opFull, ncclRedOp_t op, ncclDataType_t datatype, ncclComm *comm
  ) {
  union {
    int8_t i8;
    uint8_t u8;
    int32_t i32;
    uint32_t u32;
    int64_t i64;
    uint64_t u64;
    half f16;
    #if defined(__CUDA_BF16_TYPES_EXIST__)
      __nv_bfloat16 bf16;
    #endif
    #if defined(__CUDA_FP8_TYPES_EXIST__)
      __nv_fp8_e4m3 fp8_e4m3;
      __nv_fp8_e5m2 fp8_e5m2;
    #endif
    float f32;
    double f64;
    void *ptr;
  };
  u64 = 0;
  opFull->scalarArgIsPtr = false;
  switch (int(op)) {
  case ncclSum:  opFull->op = ncclDevSum;  break;
  case ncclProd: opFull->op = ncclDevProd; break;
  case ncclMax:  opFull->op = ncclDevMax;  break;
  case ncclMin:  opFull->op = ncclDevMin;  break;
  case ncclAvg:
    switch ((int)datatype) {
    case ncclInt8:  case ncclInt32:  case ncclInt64:
    case ncclUint8: case ncclUint32: case ncclUint64:
      opFull->op = ncclDevSumPostDiv;
      u64 = comm->nRanks;
      break;
    case ncclFloat16:
      opFull->op = ncclDevPreMulSum;
      f16 = __float2half(float(1.0/comm->nRanks)); // __double2half not supported pre CUDA 11.x
      break;
    #if defined(__CUDA_BF16_TYPES_EXIST__)
    case ncclBfloat16:
      opFull->op = ncclDevPreMulSum;
      bf16 = (__nv_bfloat16)(float(1.0/comm->nRanks));
      break;
    #endif  
    #if defined(__CUDA_FP8_TYPES_EXIST__)
    case ncclFp8E4M3:
      opFull->op = ncclDevPreMulSum;
      fp8_e4m3 = (__nv_fp8_e4m3)(float(1.0/comm->nRanks));
      break;
    case ncclFp8E5M2:
      opFull->op = ncclDevPreMulSum;
      fp8_e5m2 = (__nv_fp8_e5m2)(float(1.0/comm->nRanks));
      break;
    #endif
    case ncclFloat32:
      opFull->op = ncclDevPreMulSum;
      f32 = float(1.0/comm->nRanks);
      break;
    case ncclFloat64:
      opFull->op = ncclDevPreMulSum;
      f64 = 1.0/comm->nRanks;
      break;
    }
    opFull->scalarArgIsPtr = false;
    opFull->scalarArg = u64;
    break;
  default: // user created
    int ix = int(ncclUserRedOpMangle(comm, op)) - int(ncclNumOps);
    ncclUserRedOp *user = &comm->userRedOps[ix];
    if (datatype != user->datatype) {
      WARN("Data type supplied to user-created ncclRedOp_t does not match type "
           "given to reduction operation");
      return ncclInvalidArgument;
    }
    *opFull = user->opFull;
    break;
  }
  return ncclSuccess;
}

#define MSCCL_KERNEL_ENTRY_DEVREDOP_NULL() \
  nullptr, \
  nullptr, \
  nullptr

#define MSCCL_KERNEL_ENTRY_DEVREDOP_TYPE(devredop, type) \
  (void *)MSCCL_KERNEL_ENTRY_NAME(devredop, type, LL), \
  (void *)MSCCL_KERNEL_ENTRY_NAME(devredop, type, LL128), \
  (void *)MSCCL_KERNEL_ENTRY_NAME(devredop, type, Simple)

#if defined(__CUDA_BF16_TYPES_EXIST__) && defined(__CUDA_FP8_TYPES_EXIST__)
#define MSCCL_KERNEL_ENTRY_DEVREDOP(devredop) \
  MSCCL_KERNEL_ENTRY_DEVREDOP_TYPE(devredop, int8_t), \
  MSCCL_KERNEL_ENTRY_DEVREDOP_TYPE(devredop, uint8_t), \
  MSCCL_KERNEL_ENTRY_DEVREDOP_TYPE(devredop, int32_t), \
  MSCCL_KERNEL_ENTRY_DEVREDOP_TYPE(devredop, uint32_t), \
  MSCCL_KERNEL_ENTRY_DEVREDOP_TYPE(devredop, int64_t), \
  MSCCL_KERNEL_ENTRY_DEVREDOP_TYPE(devredop, uint64_t), \
  MSCCL_KERNEL_ENTRY_DEVREDOP_TYPE(devredop, half), \
  MSCCL_KERNEL_ENTRY_DEVREDOP_TYPE(devredop, float), \
  MSCCL_KERNEL_ENTRY_DEVREDOP_TYPE(devredop, double), \
  MSCCL_KERNEL_ENTRY_DEVREDOP_TYPE(devredop, __nv_bfloat16), \
  MSCCL_KERNEL_ENTRY_DEVREDOP_TYPE(devredop, __nv_fp8_e4m3), \
  MSCCL_KERNEL_ENTRY_DEVREDOP_TYPE(devredop, __nv_fp8_e5m2)
#elif defined(__CUDA_BF16_TYPES_EXIST__)
#define MSCCL_KERNEL_ENTRY_DEVREDOP(devredop) \
  MSCCL_KERNEL_ENTRY_DEVREDOP_TYPE(devredop, int8_t), \
  MSCCL_KERNEL_ENTRY_DEVREDOP_TYPE(devredop, uint8_t), \
  MSCCL_KERNEL_ENTRY_DEVREDOP_TYPE(devredop, int32_t), \
  MSCCL_KERNEL_ENTRY_DEVREDOP_TYPE(devredop, uint32_t), \
  MSCCL_KERNEL_ENTRY_DEVREDOP_TYPE(devredop, int64_t), \
  MSCCL_KERNEL_ENTRY_DEVREDOP_TYPE(devredop, uint64_t), \
  MSCCL_KERNEL_ENTRY_DEVREDOP_TYPE(devredop, half), \
  MSCCL_KERNEL_ENTRY_DEVREDOP_TYPE(devredop, float), \
  MSCCL_KERNEL_ENTRY_DEVREDOP_TYPE(devredop, double), \
  MSCCL_KERNEL_ENTRY_DEVREDOP_TYPE(devredop, __nv_bfloat16)
#else
#define MSCCL_KERNEL_ENTRY_DEVREDOP(devredop) \
  MSCCL_KERNEL_ENTRY_DEVREDOP_TYPE(devredop, int8_t), \
  MSCCL_KERNEL_ENTRY_DEVREDOP_TYPE(devredop, uint8_t), \
  MSCCL_KERNEL_ENTRY_DEVREDOP_TYPE(devredop, int32_t), \
  MSCCL_KERNEL_ENTRY_DEVREDOP_TYPE(devredop, uint32_t), \
  MSCCL_KERNEL_ENTRY_DEVREDOP_TYPE(devredop, int64_t), \
  MSCCL_KERNEL_ENTRY_DEVREDOP_TYPE(devredop, uint64_t), \
  MSCCL_KERNEL_ENTRY_DEVREDOP_TYPE(devredop, half), \
  MSCCL_KERNEL_ENTRY_DEVREDOP_TYPE(devredop, float), \
  MSCCL_KERNEL_ENTRY_DEVREDOP_TYPE(devredop, double)
#endif

#if defined(__CUDA_BF16_TYPES_EXIST__) && defined(__CUDA_FP8_TYPES_EXIST__)
#define MSCCL_KERNEL_ENTRY_DEVREDOP_NOFLOAT(devredop) \
  MSCCL_KERNEL_ENTRY_DEVREDOP_TYPE(devredop, int8_t), \
  MSCCL_KERNEL_ENTRY_DEVREDOP_TYPE(devredop, uint8_t), \
  MSCCL_KERNEL_ENTRY_DEVREDOP_TYPE(devredop, int32_t), \
  MSCCL_KERNEL_ENTRY_DEVREDOP_TYPE(devredop, uint32_t), \
  MSCCL_KERNEL_ENTRY_DEVREDOP_TYPE(devredop, int64_t), \
  MSCCL_KERNEL_ENTRY_DEVREDOP_TYPE(devredop, uint64_t), \
  MSCCL_KERNEL_ENTRY_DEVREDOP_NULL(), \
  MSCCL_KERNEL_ENTRY_DEVREDOP_NULL(), \
  MSCCL_KERNEL_ENTRY_DEVREDOP_NULL(), \
  MSCCL_KERNEL_ENTRY_DEVREDOP_NULL(), \
  MSCCL_KERNEL_ENTRY_DEVREDOP_NULL(), \
  MSCCL_KERNEL_ENTRY_DEVREDOP_NULL()
#elif defined(__CUDA_BF16_TYPES_EXIST__)
#define MSCCL_KERNEL_ENTRY_DEVREDOP_NOFLOAT(devredop) \
  MSCCL_KERNEL_ENTRY_DEVREDOP_TYPE(devredop, int8_t), \
  MSCCL_KERNEL_ENTRY_DEVREDOP_TYPE(devredop, uint8_t), \
  MSCCL_KERNEL_ENTRY_DEVREDOP_TYPE(devredop, int32_t), \
  MSCCL_KERNEL_ENTRY_DEVREDOP_TYPE(devredop, uint32_t), \
  MSCCL_KERNEL_ENTRY_DEVREDOP_TYPE(devredop, int64_t), \
  MSCCL_KERNEL_ENTRY_DEVREDOP_TYPE(devredop, uint64_t), \
  MSCCL_KERNEL_ENTRY_DEVREDOP_NULL(), \
  MSCCL_KERNEL_ENTRY_DEVREDOP_NULL(), \
  MSCCL_KERNEL_ENTRY_DEVREDOP_NULL(), \
  MSCCL_KERNEL_ENTRY_DEVREDOP_NULL()
#else
#define MSCCL_KERNEL_ENTRY_DEVREDOP_NOFLOAT(devredop) \
  MSCCL_KERNEL_ENTRY_DEVREDOP_TYPE(devredop, int8_t), \
  MSCCL_KERNEL_ENTRY_DEVREDOP_TYPE(devredop, uint8_t), \
  MSCCL_KERNEL_ENTRY_DEVREDOP_TYPE(devredop, int32_t), \
  MSCCL_KERNEL_ENTRY_DEVREDOP_TYPE(devredop, uint32_t), \
  MSCCL_KERNEL_ENTRY_DEVREDOP_TYPE(devredop, int64_t), \
  MSCCL_KERNEL_ENTRY_DEVREDOP_TYPE(devredop, uint64_t), \
  MSCCL_KERNEL_ENTRY_DEVREDOP_NULL(), \
  MSCCL_KERNEL_ENTRY_DEVREDOP_NULL(), \
  MSCCL_KERNEL_ENTRY_DEVREDOP_NULL()
#endif

#define MSCCL_KERNEL_ENTRY() \
  MSCCL_KERNEL_ENTRY_DEVREDOP(Sum), \
  MSCCL_KERNEL_ENTRY_DEVREDOP(Prod), \
  MSCCL_KERNEL_ENTRY_DEVREDOP(Max), \
  MSCCL_KERNEL_ENTRY_DEVREDOP(Min), \
  MSCCL_KERNEL_ENTRY_DEVREDOP(PreMulSum), \
  MSCCL_KERNEL_ENTRY_DEVREDOP_NOFLOAT(SumPostDiv)

void* mscclKernelEntries[ncclNumDevRedOps * ncclNumTypes * NCCL_NUM_PROTOCOLS] = {
  MSCCL_KERNEL_ENTRY()
};

// Returns maximum kernel stack size of all CUDA kernels
ncclResult_t mscclInitKernelsForDevice(int cudaArch, size_t* maxStackSize) {
  constexpr int KernelCount = ncclNumDevRedOps * ncclNumTypes * NCCL_NUM_PROTOCOLS;
  ncclResult_t result = ncclSuccess;

  if (maxStackSize) *maxStackSize = 0;
  int carveout = getEnvInt("NCCL_L1_SHARED_MEMORY_CARVEOUT", 0);
  // Keep track if we already visited a function pointer.
  void* lru[2] = {nullptr, nullptr};
  for (int i=0; i < KernelCount; i++) {
    void* fn = mscclKernelEntries[i];
    if (fn == lru[0] || fn == lru[1] || fn == nullptr) goto next_kernel;
    lru[1] = lru[0];
    lru[0] = fn;

    if (maxStackSize) {
      cudaFuncAttributes attr = {0};
      CUDACHECKGOTO(cudaFuncGetAttributes(&attr, fn), result, ignore0);
      if (attr.localSizeBytes > *maxStackSize) *maxStackSize = attr.localSizeBytes;
    ignore0:;
    }

    if (carveout) {
      CUDACHECKGOTO(cudaFuncSetAttribute(fn,
        cudaFuncAttributePreferredSharedMemoryCarveout, carveout),
        result, ignore1);
    ignore1:;
    }

    if (ncclShmemDynamicSize(cudaArch) != 0) {
      CUDACHECKGOTO(cudaFuncSetAttribute(fn,
        cudaFuncAttributeMaxDynamicSharedMemorySize, ncclShmemDynamicSize(cudaArch)),
        result, next_kernel);
    }
  next_kernel:;
  }
  return result;
}

ncclResult_t mscclSetupKernel(const void* sendBuff, void* recvBuff, size_t count,
    ncclDataType_t dataType, ncclRedOp_t op, struct mscclAlgo* hostAlgo, struct mscclAlgo* devAlgo,
    ncclComm_t comm, cudaStream_t stream) {
  mscclStatus& status = mscclGetStatus();
  
  if (status.lastStream != stream && status.lastStream != nullptr) {
    INFO(NCCL_INIT|NCCL_NET, "mscclSetupKernel - Waiting for last stream to finish");
    // TODO: Wait for last stream to finish, will refactor this later
    // CUDACHECK(cudaStreamWaitEvent(stream, comm->doneEvent, 0));
  }

  dim3 grid = {(uint32_t)hostAlgo->nBlocks, 1, 1};
  dim3 block;
  if (hostAlgo->protocol == NCCL_PROTO_SIMPLE) {
    block = {NCCL_SIMPLE_MAX_NTHREADS + WARP_SIZE, 1, 1};
  }
  else
  {
    block = {NCCL_MAX_NTHREADS, 1, 1};
  }
  struct ncclDevRedOpFull opFull;
  NCCLCHECK(hostToDevRedOp(&opFull, op, dataType, comm));
  size_t smem = ncclShmemDynamicSize(comm->cudaArch);
  
  mscclWork work;
  work.syncFlags = status.syncFlags;
  work.scratchBuffer = status.scratchBuffer;
  work.sendBuff = sendBuff;
  work.recvBuff = recvBuff;
  work.count = count * hostAlgo->sizeMultiplier; // count is sum of all ranks in MSCCL kernel
  work.redOpArg = opFull.scalarArg;
  work.workIndex = status.workIndex;
  work.nChunksPerLoop = hostAlgo->nChunksPerLoop;
  work.maxAllowedCount = status.maxAllowedCount;
  work.hasReduce = hostAlgo->hasReduce;
  work.redOpArgIsPtr = opFull.scalarArgIsPtr;
  INFO(NCCL_INIT|NCCL_NET, "MSCCL: Setup Kernel finished, smem %ld", smem);
  void *args[3] = {&comm->devComm, &devAlgo, &work};
  void *func = mscclKernelEntries[(opFull.op * ncclNumTypes + dataType) * NCCL_NUM_PROTOCOLS + hostAlgo->protocol];

  #if 0
  // #if CUDART_VERSION >= 11080
  int driverVersion;
  NCCLCHECK(ncclCudaDriverVersion(&driverVersion));
  if (driverVersion >= 11080) {
    int compCap = comm->compCap;
    unsigned int clusterSize = (compCap == 90) ? comm->config.cgaClusterSize : 0;

    cudaLaunchConfig_t launchConfig = {0};
    cudaLaunchAttribute launchAttrs[3];
    int attrs = 0;
    /* Cooperative Group Array (CGA)
     * On sm90 and later we have an extra level of hierarchy where we
     * can group together several blocks within the Grid, called
     * Thread Block Clusters.
     * Clusters enable multiple thread blocks running concurrently
     * across multiple SMs to synchronize and collaboratively fetch
     * and exchange data. A cluster of blocks are guaranteed to be
     * concurrently scheduled onto a group of SMs.
     * The maximum value is 8 and it must be divisible into the grid dimensions
     */
    if (clusterSize) {
      // Grid dimension must be divisible by clusterSize
      if (grid.x % clusterSize) clusterSize = 1;
      launchAttrs[attrs].id = cudaLaunchAttributeClusterDimension;
      launchAttrs[attrs++].val.clusterDim = {clusterSize, 1, 1};
      launchAttrs[attrs].id = cudaLaunchAttributeClusterSchedulingPolicyPreference;
      launchAttrs[attrs++].val.clusterSchedulingPolicyPreference = cudaClusterSchedulingPolicySpread;
    }
    #if CUDART_VERSION >= 12000
    if (compCap >= 90 && driverVersion >= 12000) {
      // Set the NCCL Mem Sync domain on CUDA 12.0 and later (sm90)
      launchAttrs[attrs].id = cudaLaunchAttributeMemSyncDomain;
      launchAttrs[attrs++].val.memSyncDomain = (cudaLaunchMemSyncDomain) ncclParamMemSyncDomain();
    }
    #endif
    launchConfig.gridDim = grid;
    launchConfig.blockDim = block;
    launchConfig.dynamicSmemBytes = smem;
    launchConfig.attrs = launchAttrs;
    launchConfig.numAttrs = attrs;
    launchConfig.stream = stream;

    CUDACHECK(cudaLaunchKernelExC(&launchConfig, func, args));
    return ncclSuccess;
  }
  #endif
  
  CUDACHECK(cudaLaunchKernel(func, grid, block, args, smem, stream));
  status.workIndex++;
  status.lastStream = stream;
  return ncclSuccess;
}
