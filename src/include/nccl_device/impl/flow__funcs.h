/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more information
 *************************************************************************/

#ifndef NCCL_DEVICE_FLOW__FUNCS_H_
#define NCCL_DEVICE_FLOW__FUNCS_H_

#include "../flow.h"
#include "lsa_flow__funcs.h"
#if !defined(NCCL_OS_WINDOWS)
#include "gin_flow__funcs.h"
#endif

#if defined(__CUDACC__) && defined(__CUDACC_EXTENDED_LAMBDA__)
#include <cassert>
#include <new>

template <typename Backend, int SlotsPerProcess, int MaxPeers>
NCCL_DEVICE_INLINE int ncclFlowBase<Backend, SlotsPerProcess, MaxPeers>::threadRole(
  int nWorkerWarps, int nSendPeers, int nRecvPeers, int nParts, int part, int const* partWorkerWarps, int* rolePeer) {
  constexpr int warpSize = 32;
  constexpr int signalWarpsPerPart = ncclFlowGetSignalWarpsPerPart(Backend::Protocol);
  int const t = (int)threadIdx.x;
  int const workerWarp0 = ncclFlowPartWorkerWarp0(nWorkerWarps, nParts, part, partWorkerWarps, signalWarpsPerPart);
  int const workerWarps = ncclFlowPartWorkerWarps(nWorkerWarps, nParts, part, partWorkerWarps);
  int const workerThread = t - workerWarp0 * warpSize;
  int const sendWaitLane = workerThread;
  if (0 <= sendWaitLane && sendWaitLane < nSendPeers) {
    *rolePeer = sendWaitLane;
    return RoleSendWait;
  }
  int const recvWaitLane = workerThread - warpSize;
  if (0 <= recvWaitLane && recvWaitLane < nRecvPeers) {
    *rolePeer = recvWaitLane;
    return RoleRecvWait;
  }
  int const signalThread = (workerWarp0 + workerWarps) * warpSize;
  int const sendPostLane = t - signalThread;
  if (0 <= sendPostLane && sendPostLane < nSendPeers) {
    *rolePeer = sendPostLane;
    return RoleSendPost;
  }
  int const recvPostLane = t - signalThread - (signalWarpsPerPart == 1 ? warpSize / 2 : warpSize);
  if (0 <= recvPostLane && recvPostLane < nRecvPeers) {
    *rolePeer = recvPostLane;
    return RoleRecvPost;
  }
  *rolePeer = -1;
  return 0 <= workerThread && workerThread < workerWarps * warpSize ? RoleWorker : RoleIdle;
}

template <typename Backend, int SlotsPerProcess, int MaxPeers>
NCCL_DEVICE_INLINE int ncclFlowBase<Backend, SlotsPerProcess, MaxPeers>::roleType(int role) {
  if (role == RoleSendWait || role == RoleSendPost) return ncclFlowTypeSend;
  if (role == RoleRecvWait || role == RoleRecvPost) return ncclFlowTypeRecv;
  return ncclFlowTypeNone;
}

template <typename Backend, int SlotsPerProcess, int MaxPeers>
NCCL_DEVICE_INLINE ncclFlowBase<Backend, SlotsPerProcess, MaxPeers>::ncclFlowBase(
  uint32_t* abortFlag_, size_t slotSize_, int nSlots_, int nWorkerWarps_, int nSendPeers_, int nRecvPeers_, int nParts_,
  int part_, int const* sendSlots, int const* recvSlots, ncclFlowShmem& shmem, int const* partWorkerWarps)
  : role(RoleIdle), rolePeer(-1), roleSlot(-1), nSendPeers(nSendPeers_), nRecvPeers(nRecvPeers_), abortFlag(abortFlag_),
    slotSize(slotSize_), nSlots(nSlots_), processShmem(&shmem.parts[part_]),
    workers(/*warp0=*/ncclFlowPartWorkerWarp0(nWorkerWarps_, nParts_, part_, partWorkerWarps,
                                              ncclFlowGetSignalWarpsPerPart(Backend::Protocol)),
            /*nWarps=*/ncclFlowPartWorkerWarps(nWorkerWarps_, nParts_, part_, partWorkerWarps),
            /*id=*/part_),
    partThreads(/*warp0=*/ncclFlowPartWorkerWarp0(nWorkerWarps_, nParts_, part_, partWorkerWarps,
                                                  ncclFlowGetSignalWarpsPerPart(Backend::Protocol)),
                /*nWarps=*/ncclFlowPartWorkerWarps(nWorkerWarps_, nParts_, part_, partWorkerWarps) +
                  ncclFlowGetSignalWarpsPerPart(Backend::Protocol),
                /*id=*/nParts_ + part_) {
  assert(0 < nWorkerWarps_);
  assert(0 < nSlots);
  assert(0 < nParts_ && nParts_ <= NCCL_FLOW_MAX_PARTS);
  assert(0 <= part_ && part_ < nParts_);
  assert(2 * nParts_ <= nWorkerWarps_);
  assert((int)blockDim.x == 32 * (nWorkerWarps_ + nParts_ * ncclFlowGetSignalWarpsPerPart(Backend::Protocol)));
  assert(part_ ==
         ncclFlowThreadPart(nWorkerWarps_, nParts_, partWorkerWarps, ncclFlowGetSignalWarpsPerPart(Backend::Protocol)));
  assert(0 <= nSendPeers && nSendPeers <= MaxPeers);
  assert(0 <= nRecvPeers && nRecvPeers <= MaxPeers);
  assert(0 < slotSize);
  role = threadRole(nWorkerWarps_, nSendPeers, nRecvPeers, nParts_, part_, partWorkerWarps, &rolePeer);
  if (role == RoleSendWait || role == RoleSendPost) roleSlot = sendSlots == nullptr ? rolePeer : sendSlots[rolePeer];
  if (role == RoleRecvWait || role == RoleRecvPost) roleSlot = recvSlots == nullptr ? rolePeer : recvSlots[rolePeer];
  assert(roleSlot == -1 || 0 <= roleSlot && roleSlot < MaxPeers);
}

template <typename Backend, int SlotsPerProcess, int MaxPeers>
NCCL_DEVICE_INLINE void ncclFlowBase<Backend, SlotsPerProcess, MaxPeers>::initShmem() {
  if (partThreads.thread_rank() == 0) {
    for (int peer = 0; peer < MaxPeers; peer++) {
      processShmem->sendBuf[peer] = nullptr;
      processShmem->recvBuf[peer] = nullptr;
      processShmem->sendFlag[peer] = 0;
      processShmem->recvFlag[peer] = 0;
      processShmem->sendUsesFifo[peer] = false;
      processShmem->recvUsesFifo[peer] = false;
    }
  }
  partThreads.sync();
}

template <typename Backend, int SlotsPerProcess, int MaxPeers>
NCCL_DEVICE_INLINE void ncclFlowBase<Backend, SlotsPerProcess, MaxPeers>::close() {
  if (role != RoleSendPost && role != RoleRecvPost) return;
  static_cast<Backend*>(this)->closeBackend();
}

template <typename Backend, int SlotsPerProcess, int MaxPeers>
template <bool Send, bool DirectSend, bool Recv, bool DirectRecv, typename T, typename SrcLambda, typename DstLambda,
          typename RedOp>
NCCL_DEVICE_INLINE void ncclFlowBase<Backend, SlotsPerProcess, MaxPeers>::process(
  ncclFlowSendTag<Send, DirectSend> send, ncclFlowRecvTag<Recv, DirectRecv> recv, ncclSymPtr<T> input,
  ncclSymPtr<T> output, SrcLambda srcLambda, int nSrc, DstLambda dstLambda, int nDst, RedOp const& redOp,
  size_t nElts) {
  if (nElts == 0) return;
  process<Send, DirectSend, Recv, DirectRecv, T>(send.direct, recv.direct, input, output, srcLambda, nSrc, dstLambda,
                                                 nDst, redOp, nElts, ncclFlowProtocolTag<Backend::Protocol>{});
}

template <ncclFlowProtocol Protocol>
NCCL_DEVICE_INLINE void ncclFlowConnection<0, Protocol>::construct(
  ncclDevComm const& comm, ncclFlowConfig<0> const& config, int peer, int peerIndex, int maxPeers, int channelId,
  int nParts, int part, int type, bool /*waitRole*/, bool recvPost) {
  int const nSlots = config.nSlots;
  assert(config.channelSize % (nSlots * nParts * maxPeers) == 0);
  size_t const slotSize = config.channelSize / (nSlots * nParts * maxPeers);
  size_t const fifoOffset =
    (size_t)channelId * config.channelSize + ((size_t)part * maxPeers + peerIndex) * nSlots * slotSize;
  int const connection = part * maxPeers + peerIndex;
  assert(connection < config.connStatesPerChannel);
  size_t const stateIndex = (size_t)channelId * config.connStatesPerChannel + connection;
  ncclSymPtr<char> recvFifo = config.recvBuffer + fifoOffset;
  ncclSymPtr<ncclFlowConnState> state = config.connState + stateIndex;
  ::new (&storage.lsa) ncclLsaFlowConn<Protocol>(comm, recvFifo, state, peer, nSlots, slotSize, type, recvPost);
}

template <ncclFlowProtocol Protocol>
NCCL_DEVICE_INLINE void ncclFlowConnection<0, Protocol>::close() {
  storage.lsa.close();
}

template <ncclFlowProtocol Protocol>
NCCL_DEVICE_INLINE void* ncclFlowConnection<0, Protocol>::waitSend() {
  return storage.lsa.waitSend();
}

template <ncclFlowProtocol Protocol>
NCCL_DEVICE_INLINE void ncclFlowConnection<0, Protocol>::postSend(size_t bytes) {
  storage.lsa.postSend(bytes);
}

template <ncclFlowProtocol Protocol>
NCCL_DEVICE_INLINE void ncclFlowConnection<0, Protocol>::postSend(
  size_t bytes, bool /*direct*/, ncclSymPtr<char> /*output*/, ncclSymPtr<char> /*source*/) {
  storage.lsa.postSend(bytes);
}

template <ncclFlowProtocol Protocol>
NCCL_DEVICE_INLINE void* ncclFlowConnection<0, Protocol>::waitRecv() {
  return storage.lsa.waitRecv();
}

template <ncclFlowProtocol Protocol>
NCCL_DEVICE_INLINE void ncclFlowConnection<0, Protocol>::postRecv() {
  storage.lsa.postRecv();
}

template <ncclFlowProtocol Protocol>
NCCL_DEVICE_INLINE void ncclFlowConnection<0, Protocol>::advanceStep() {
  storage.lsa.advanceStep();
}

template <ncclFlowProtocol Protocol>
NCCL_DEVICE_INLINE void* ncclFlowConnection<0, Protocol>::advanceFifoSlot() {
  return storage.lsa.advanceFifoSlot();
}

template <ncclFlowProtocol Protocol>
NCCL_DEVICE_INLINE uint64_t ncclFlowConnection<0, Protocol>::stepFlag() const {
  return storage.lsa.stepFlag();
}

template <ncclFlowProtocol Protocol>
NCCL_DEVICE_INLINE bool ncclFlowConnection<0, Protocol>::usesGin() const {
  return false;
}

#if !defined(NCCL_OS_WINDOWS)
template <unsigned GinBackendMask, ncclFlowProtocol Protocol>
NCCL_DEVICE_INLINE void ncclFlowConnection<GinBackendMask, Protocol>::construct(
  ncclDevComm const& comm, ncclFlowConfig<GinBackendMask> const& config, int peer, int peerIndex, int maxPeers,
  int channelId, int nParts, int part, int type, bool waitRole, bool recvPost) {
  int const nSlots = config.nSlots;
  assert(config.channelSize % (nSlots * nParts * maxPeers) == 0);
  size_t const slotSize = config.channelSize / (nSlots * nParts * maxPeers);
  size_t const connectionOffset = ((size_t)part * maxPeers + peerIndex) * nSlots * slotSize;
  size_t const fifoOffset = (size_t)channelId * config.channelSize + connectionOffset;
  int const connection = part * maxPeers + peerIndex;
  assert(connection < config.connStatesPerChannel);
  size_t const stateIndex = (size_t)channelId * config.connStatesPerChannel + connection;
  ncclSymPtr<char> recvFifo = config.recvBuffer + fifoOffset;
  ncclSymPtr<ncclFlowConnState> state = config.connState + stateIndex;

  ncclTeam const world = ncclTeamWorld(comm);
  useGin = !ncclDevCommCanGetPeerPointer(comm, peer);
  if (useGin) {
    assert(ncclTeamRankIsMember(config.ginTeam, world, peer));
    int const ginPeer = ncclTeamRankToTeam(config.ginTeam, world, peer);
    ncclSymPtr<char> sendFifo;
    if (type == ncclFlowTypeSend) {
      assert(config.ginBuffer.window != nullptr);
      sendFifo = config.ginBuffer + connectionOffset;
    }
    int const contextId = config.ginContextIndex % comm.ginContextCount;
    ::new (&storage.gin) ncclGinFlowConn<GinBackendMask, Protocol>(comm, contextId, config.ginTeam, ginPeer, recvFifo,
                                                                   sendFifo, state, config.ginSignal0, stateIndex,
                                                                   nSlots, slotSize, type, waitRole, recvPost);
  } else {
    ::new (&storage.lsa) ncclLsaFlowConn<Protocol>(comm, recvFifo, state, peer, nSlots, slotSize, type, recvPost);
  }
}

template <unsigned GinBackendMask, ncclFlowProtocol Protocol>
NCCL_DEVICE_INLINE void ncclFlowConnection<GinBackendMask, Protocol>::close() {
  if (useGin) storage.gin.close();
  else storage.lsa.close();
}

template <unsigned GinBackendMask, ncclFlowProtocol Protocol>
NCCL_DEVICE_INLINE void* ncclFlowConnection<GinBackendMask, Protocol>::waitSend() {
  return useGin ? storage.gin.waitSend() : storage.lsa.waitSend();
}

template <unsigned GinBackendMask, ncclFlowProtocol Protocol>
NCCL_DEVICE_INLINE void ncclFlowConnection<GinBackendMask, Protocol>::postSend(size_t bytes) {
  if (useGin) storage.gin.postSend(bytes);
  else storage.lsa.postSend(bytes);
}

template <unsigned GinBackendMask, ncclFlowProtocol Protocol>
NCCL_DEVICE_INLINE void ncclFlowConnection<GinBackendMask, Protocol>::postSend(
  size_t bytes, bool direct, ncclSymPtr<char> output, ncclSymPtr<char> source) {
  if (useGin) {
    if (direct) storage.gin.postSendDirect(bytes, output, source);
    else if (source.window != nullptr) storage.gin.postSendFrom(bytes, source);
    else storage.gin.postSend(bytes);
  } else {
    storage.lsa.postSend(bytes);
  }
}

template <unsigned GinBackendMask, ncclFlowProtocol Protocol>
NCCL_DEVICE_INLINE void* ncclFlowConnection<GinBackendMask, Protocol>::waitRecv() {
  return useGin ? storage.gin.waitRecv() : storage.lsa.waitRecv();
}

template <unsigned GinBackendMask, ncclFlowProtocol Protocol>
NCCL_DEVICE_INLINE void ncclFlowConnection<GinBackendMask, Protocol>::postRecv() {
  if (useGin) storage.gin.postRecv();
  else storage.lsa.postRecv();
}

template <unsigned GinBackendMask, ncclFlowProtocol Protocol>
NCCL_DEVICE_INLINE void ncclFlowConnection<GinBackendMask, Protocol>::advanceStep() {
  if (useGin) storage.gin.advanceStep();
  else storage.lsa.advanceStep();
}

template <unsigned GinBackendMask, ncclFlowProtocol Protocol>
NCCL_DEVICE_INLINE void* ncclFlowConnection<GinBackendMask, Protocol>::advanceFifoSlot() {
  return useGin ? storage.gin.advanceFifoSlot() : storage.lsa.advanceFifoSlot();
}

template <unsigned GinBackendMask, ncclFlowProtocol Protocol>
NCCL_DEVICE_INLINE uint64_t ncclFlowConnection<GinBackendMask, Protocol>::stepFlag() const {
  return useGin ? storage.gin.stepFlag() : storage.lsa.stepFlag();
}

template <unsigned GinBackendMask, ncclFlowProtocol Protocol>
NCCL_DEVICE_INLINE bool ncclFlowConnection<GinBackendMask, Protocol>::usesGin() const {
  return useGin;
}
#endif

template <unsigned GinBackendMask, int SlotsPerProcess, int MaxPeers, ncclFlowProtocol Protocol>
NCCL_DEVICE_INLINE ncclFlowProcessor<GinBackendMask, SlotsPerProcess, MaxPeers, Protocol>::ncclFlowProcessor(
  ncclDevComm const& comm, ncclFlowConfig<GinBackendMask> const& config, int const* sendPeers, int nSendPeers,
  int const* recvPeers, int nRecvPeers, int channelId, int nWorkerWarps, ncclFlowShmem& shmem, int nParts, int sendPart,
  int recvPart, int const* sendSlots, int const* recvSlots, int const* partWorkerWarps)
  : Base(comm.abortFlag, config.channelSize / ((size_t)config.nSlots * nParts * MaxPeers), config.nSlots, nWorkerWarps,
         nSendPeers, nRecvPeers, nParts,
         ncclFlowThreadPart(nWorkerWarps, nParts, partWorkerWarps, ncclFlowGetSignalWarpsPerPart(Protocol)), sendSlots,
         recvSlots, shmem, partWorkerWarps),
    conn() {
  assert(0 < config.nSlots);
  assert(SlotsPerProcess <= config.nSlots);
  assert(0 < nParts && nParts <= NCCL_FLOW_MAX_PARTS);
  assert(config.channelSize % ((size_t)config.nSlots * nParts * MaxPeers) == 0);
  assert(0 <= sendPart && sendPart < nParts);
  assert(0 <= recvPart && recvPart < nParts);
  assert(nSendPeers == 0 || sendPeers != nullptr);
  assert(nRecvPeers == 0 || recvPeers != nullptr);
  int const type = this->roleType(this->role);
  if (type != ncclFlowTypeNone) {
    bool const send = type == ncclFlowTypeSend;
    int const peer = send ? sendPeers[this->rolePeer] : recvPeers[this->rolePeer];
    int const peerPart = send ? sendPart : recvPart;
    bool const waitRole = this->role == Base::RoleSendWait || this->role == Base::RoleRecvWait;
    conn.construct(comm, config, peer, this->roleSlot, MaxPeers, channelId, nParts, peerPart, type, waitRole,
                   this->role == Base::RoleRecvPost);
  }
  this->initShmem();
}

template <unsigned GinBackendMask, int SlotsPerProcess, int MaxPeers, ncclFlowProtocol Protocol>
NCCL_DEVICE_INLINE void ncclFlowProcessor<GinBackendMask, SlotsPerProcess, MaxPeers, Protocol>::closeBackend() {
  conn.close();
}

template <unsigned GinBackendMask, int SlotsPerProcess, int MaxPeers, ncclFlowProtocol Protocol>
NCCL_DEVICE_INLINE void* ncclFlowProcessor<GinBackendMask, SlotsPerProcess, MaxPeers, Protocol>::waitSend() {
  return conn.waitSend();
}

template <unsigned GinBackendMask, int SlotsPerProcess, int MaxPeers, ncclFlowProtocol Protocol>
NCCL_DEVICE_INLINE void ncclFlowProcessor<GinBackendMask, SlotsPerProcess, MaxPeers, Protocol>::postSend(size_t bytes) {
  conn.postSend(bytes);
}

template <unsigned GinBackendMask, int SlotsPerProcess, int MaxPeers, ncclFlowProtocol Protocol>
NCCL_DEVICE_INLINE void ncclFlowProcessor<GinBackendMask, SlotsPerProcess, MaxPeers, Protocol>::postSend(
  size_t bytes, bool direct, ncclSymPtr<char> output, ncclSymPtr<char> source) {
  conn.postSend(bytes, direct, output, source);
}

template <unsigned GinBackendMask, int SlotsPerProcess, int MaxPeers, ncclFlowProtocol Protocol>
NCCL_DEVICE_INLINE void* ncclFlowProcessor<GinBackendMask, SlotsPerProcess, MaxPeers, Protocol>::waitRecv() {
  return conn.waitRecv();
}

template <unsigned GinBackendMask, int SlotsPerProcess, int MaxPeers, ncclFlowProtocol Protocol>
NCCL_DEVICE_INLINE void ncclFlowProcessor<GinBackendMask, SlotsPerProcess, MaxPeers, Protocol>::postRecv() {
  conn.postRecv();
}

template <unsigned GinBackendMask, int SlotsPerProcess, int MaxPeers, ncclFlowProtocol Protocol>
NCCL_DEVICE_INLINE void ncclFlowProcessor<GinBackendMask, SlotsPerProcess, MaxPeers, Protocol>::advanceStep() {
  conn.advanceStep();
}

template <unsigned GinBackendMask, int SlotsPerProcess, int MaxPeers, ncclFlowProtocol Protocol>
NCCL_DEVICE_INLINE void* ncclFlowProcessor<GinBackendMask, SlotsPerProcess, MaxPeers, Protocol>::advanceFifoSlot() {
  return conn.advanceFifoSlot();
}

template <unsigned GinBackendMask, int SlotsPerProcess, int MaxPeers, ncclFlowProtocol Protocol>
NCCL_DEVICE_INLINE uint64_t ncclFlowProcessor<GinBackendMask, SlotsPerProcess, MaxPeers, Protocol>::stepFlag() const {
  return conn.stepFlag();
}

template <unsigned GinBackendMask, int SlotsPerProcess, int MaxPeers, ncclFlowProtocol Protocol>
NCCL_DEVICE_INLINE bool ncclFlowProcessor<GinBackendMask, SlotsPerProcess, MaxPeers, Protocol>::usesGin() const {
  return conn.usesGin();
}

#include "flow__simple_funcs.h"
#include "flow__ll_funcs.h"
#include "flow__ll128_funcs.h"
#endif // __CUDACC__ && __CUDACC_EXTENDED_LAMBDA__

#endif // NCCL_DEVICE_FLOW__FUNCS_H_
