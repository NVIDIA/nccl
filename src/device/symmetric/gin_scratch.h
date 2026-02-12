/*************************************************************************
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/
#ifndef _NCCL_DEVICE_GIN_SCRATCH_H_
#define _NCCL_DEVICE_GIN_SCRATCH_H_
#if 1 // When this file is not in "nccl_device/"
  #include "nccl_device.h"
#else // When this file is public in "nccl_device/"
  #include "core.h"
#endif

struct ncclGinOutboxHandle;
struct ncclGinInboxA2AHandle;

constexpr int ncclGinScratchMaxBufs_log2 = /*log2(512)=*/9;
constexpr int ncclGinScratchMaxBufsPerPeer_log2 = /*log2(4)=*/2;

NCCL_EXTERN_C __host__ ncclResult_t ncclGinOutboxCreateRequirement(
  int nBlocks, int size_log2,
  ncclGinOutboxHandle* outHandle, ncclDevResourceRequirements* outReq
);

NCCL_EXTERN_C __host__ ncclResult_t ncclGinInboxA2ACreateRequirement(
  ncclTeam peers, int nBlocks, int size_log2,
  ncclGinInboxA2AHandle* outHandle, ncclDevResourceRequirements* outReq
);

#if NCCL_CHECK_CUDACC
template<typename Coop, unsigned ginBackendMask>
struct ncclGinOutboxSession_internal;

struct ncclGinScratch_GetBufPtr;

template<typename Coop, unsigned ginBackendMask = NCCL_GIN_BACKEND_MASK_ALL>
struct ncclGinOutboxSession: ncclGinOutboxSession_internal<Coop, ginBackendMask> {
  NCCL_DEVICE_INLINE ncclGinOutboxSession(Coop, ncclGin_BackendMask<ginBackendMask> const&, ncclGinOutboxHandle handle, uint32_t index);
  NCCL_DEVICE_INLINE ~ncclGinOutboxSession();

  ncclGinOutboxSession(ncclGinOutboxSession const&) = delete; // non-copyable

  // Subdivide the capacity into (1<<nBufs_log2) buffers. Cooperative over all
  // threads in Coop but that can be partitioned across multiple subcoop's of which
  // exactly one must have subcoopIsNonTrivial=true. That is the one which will
  // do the heavy work.
  template<typename SubCoop>
  NCCL_DEVICE_INLINE void apportion(Coop, SubCoop, bool subcoopIsNonTrivial, int nBufs_log2, bool deferSync=false);
  template<typename SubCoop>
  NCCL_DEVICE_INLINE void waitBufs(SubCoop, int i0, int n);
  NCCL_DEVICE_INLINE ncclSymPtr<char> getBuf(int i) const;
  NCCL_DEVICE_INLINE ncclGinScratch_GetBufPtr make_getBufPtr(int i0) const;
  NCCL_DEVICE_INLINE ncclGinCounter_t getCounter(int i) const;
  NCCL_DEVICE_INLINE void advance(Coop, int n);
};
#endif

#if NCCL_CHECK_CUDACC
template<typename Coop, unsigned ginBackendMask>
struct ncclGinInboxA2ASession_internal;

struct ncclGinInboxA2A_GetBufPtr;

template<typename Coop, unsigned ginBackendMask = NCCL_GIN_BACKEND_MASK_ALL>
struct ncclGinInboxA2ASession: ncclGinInboxA2ASession_internal<Coop, ginBackendMask> {
  NCCL_DEVICE_INLINE ncclGinInboxA2ASession(Coop, ncclGin_BackendMask<ginBackendMask> const&, ncclTeam team, ncclGinInboxA2AHandle, uint32_t index);
  NCCL_DEVICE_INLINE ~ncclGinInboxA2ASession();

  ncclGinInboxA2ASession(ncclGinInboxA2ASession const&) = delete; // non-copyable

  // Subdivide the available space into individual buffers. Cooperative over all threads
  // in Coop but they can be partitioned into multiple subcoop's of which exactly
  // one must have subcoopIsNonTrivial=true. Before entry the nontrivial coop
  // must already be synced with threads doing waitRecvs/finishRecvs from the previous
  // round.
  // Required: nBufs_log2 <= ncclGinScratchMaxBufs_log2
  template<typename SubCoop>
  NCCL_DEVICE_INLINE void apportion(Coop, SubCoop, bool subcoopIsNonTrivial, int nBufs_log2);

  // When `stepLtPeers=true` we require `step < team.nRanks-1`
  NCCL_DEVICE_INLINE int getSendPeer(int step, bool stepLtPeers=false) const;
  NCCL_DEVICE_INLINE int getRecvPeer(int step, bool stepLtPeers=false) const;

  NCCL_DEVICE_INLINE ncclSymPtr<char> getBuf(int step) const;
  NCCL_DEVICE_INLINE ncclGinScratch_GetBufPtr make_getBufPtr(int step0) const;

  // Post sends for steps [step0, step0+nSteps). The lambdas take index in [0, nSteps).
  template<typename SubCoop, typename GetPtr, typename GetEltCount, typename GetCompletion>
  NCCL_DEVICE_INLINE void postSends(
    SubCoop, int step0, int nSteps,
    /*(int index, int peer)->ncclSymPtr<T>*/GetPtr getPtr,
    /*(int index, int peer)->int*/GetEltCount getEltCount,
    /*(int index, int peer)->ncclGin_???*/GetCompletion getCompletion
  );
  // Wait for recvs for steps [step0, step0+nSteps).
  template<typename SubCoop>
  NCCL_DEVICE_INLINE void waitRecvs(SubCoop, int step0, int nSteps);
  // Finish recvs for steps [step0, step0+nSteps).
  template<typename SubCoop>
  NCCL_DEVICE_INLINE void finishRecvs(SubCoop, int step0, int nSteps);

  // Move to next round of steps.
  NCCL_DEVICE_INLINE void endRound(Coop);
};
#endif

#endif // _NCCL_DEVICE_GIN_SCRATCH_H_

// Remove if we move to public "nccl_device/"
#include "gin_scratch__funcs.h"
