/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more information
 *************************************************************************/

#ifndef NCCL_DEVICE_FLOW__INTERNALS_H_
#define NCCL_DEVICE_FLOW__INTERNALS_H_

// Included by flow.h after its public tags and configuration types.
#include "../lsa_flow.h"
#if !defined(NCCL_OS_WINDOWS)
#include "../gin_flow.h"
#endif

#ifdef __CUDACC__
template <ncclFlowProtocol Protocol>
struct ncclFlowProtocolTag {};

// Common processing protocol for point-to-point backends. Backend must provide:
//   waitSend()          wait for and return the next writable slot
//   waitRecv()          wait for and return a readable slot
//   advanceFifoSlot()   advance and return the next FIFO slot without waiting
//   advanceStep()       mirror the FIFO advancement in a post role
//   postSend(bytes)     publish the written slot
//   postRecv()          make the consumed slot eligible for credit return
//   stepFlag()          return the current protocol readiness flag
// Connections select the protocol-specific FIFO sequence internally. LL keeps an independent wrapping sequence;
// Simple and LL128 derive their slots and readiness values from the shared transport step.
// The CTA consists of nWorkerWarps divided between nParts, with protocol-specific signaling warps appended to each
// part. An optional partWorkerWarps array gives the number of contiguous worker warps assigned to each part and must
// sum to nWorkerWarps; the default divides workers evenly. Within each part, the first worker warp waits for send
// buffers and the second waits for receive buffers. Separate signaling warps normally publish sends and receives;
// LL128 uses the lower and upper halves of one warp. Each part uses independent shared state and named barriers.
template <typename Backend, int SlotsPerProcess, int MaxPeers>
struct ncclFlowBase {
  static_assert(0 < SlotsPerProcess, "ncclFlowProcessor must process at least one slot");
  static_assert(0 < MaxPeers && MaxPeers <= NCCL_FLOW_MAX_PEERS, "ncclFlowProcessor max peer count is out of range");

  NCCL_DEVICE_INLINE ncclFlowBase(uint32_t* abortFlag, size_t slotSize, int nSlots, int nWorkerWarps, int nSendPeers,
                                  int nRecvPeers, int nParts, int part, int const* sendSlots, int const* recvSlots,
                                  ncclFlowShmem& shmem, int const* partWorkerWarps);

  // Save persistent connection state. GIN drains accesses to external source windows, while connection-owned FIFOs
  // remain protected by their persistent credits. Every thread may call this function; only send-post and
  // receive-post roles act.
  NCCL_DEVICE_INLINE void close();

  template <bool Send, bool DirectSend, bool Recv, bool DirectRecv, typename T, typename SrcLambda, typename DstLambda,
            typename RedOp>
  NCCL_DEVICE_INLINE void process(ncclFlowSendTag<Send, DirectSend> send, ncclFlowRecvTag<Recv, DirectRecv> recv,
                                  ncclSymPtr<T> input, ncclSymPtr<T> output, SrcLambda srcLambda, int nSrc,
                                  DstLambda dstLambda, int nDst, RedOp const& redOp, size_t nElts);

protected:
  enum {
    RoleWorker,
    RoleSendWait,
    RoleRecvWait,
    RoleSendPost,
    RoleRecvPost,
    RoleIdle
  };

  int role;
  int rolePeer;
  int roleSlot;
  const int nSendPeers;
  const int nRecvPeers;
  uint32_t* const abortFlag;
  const size_t slotSize;
  const int nSlots;
  ncclFlowPartShmem* const processShmem;
  ncclCoopWarpSpan workers;
  ncclCoopWarpSpan partThreads;

  NCCL_DEVICE_INLINE void initShmem();
  NCCL_DEVICE_INLINE static int threadRole(int nWorkerWarps, int nSendPeers, int nRecvPeers, int nParts, int part,
                                           int const* partWorkerWarps, int* rolePeer);
  NCCL_DEVICE_INLINE static int roleType(int role);

  template <bool Send, bool DirectSend, bool Recv, bool DirectRecv, typename T, typename SrcLambda, typename DstLambda,
            typename RedOp>
  NCCL_DEVICE_INLINE void process(bool sendDirect, bool recvDirect, ncclSymPtr<T> input, ncclSymPtr<T> output,
                                  SrcLambda srcLambda, int nSrc, DstLambda dstLambda, int nDst, RedOp const& redOp,
                                  size_t nElts, ncclFlowProtocolTag<ncclFlowProtocolSimple>);
  template <bool Send, bool DirectSend, bool Recv, bool DirectRecv, typename T, typename SrcLambda, typename DstLambda,
            typename RedOp>
  NCCL_DEVICE_INLINE void process(bool sendDirect, bool recvDirect, ncclSymPtr<T> input, ncclSymPtr<T> output,
                                  SrcLambda srcLambda, int nSrc, DstLambda dstLambda, int nDst, RedOp const& redOp,
                                  size_t nElts, ncclFlowProtocolTag<ncclFlowProtocolLL>);
  template <bool Send, bool DirectSend, bool Recv, bool DirectRecv, typename T, typename SrcLambda, typename DstLambda,
            typename RedOp>
  NCCL_DEVICE_INLINE void process(bool sendDirect, bool recvDirect, ncclSymPtr<T> input, ncclSymPtr<T> output,
                                  SrcLambda srcLambda, int nSrc, DstLambda dstLambda, int nDst, RedOp const& redOp,
                                  size_t nElts, ncclFlowProtocolTag<ncclFlowProtocolLL128>);
};

template <unsigned GinBackendMask, ncclFlowProtocol Protocol>
struct ncclFlowConnection;

// Direct LSA connection adapter. Since GinBackendMask is zero, neither its object state nor its call graph can contain
// GIN.
template <ncclFlowProtocol Protocol>
struct ncclFlowConnection<0, Protocol> {
  union Storage {
    ncclLsaFlowConn<Protocol> lsa;

    NCCL_DEVICE_INLINE Storage() {}
    NCCL_DEVICE_INLINE ~Storage() {}
  } storage;

  NCCL_DEVICE_INLINE ncclFlowConnection() {}
  NCCL_DEVICE_INLINE void construct(ncclDevComm const& comm, ncclFlowConfig<0> const& config, int peer, int peerIndex,
                                    int maxPeers, int channelId, int nParts, int part, int type, bool waitRole,
                                    bool recvPost);
  NCCL_DEVICE_INLINE void close();
  NCCL_DEVICE_INLINE void* waitSend();
  NCCL_DEVICE_INLINE void postSend(size_t bytes);
  NCCL_DEVICE_INLINE void postSend(size_t bytes, bool direct, ncclSymPtr<char> output, ncclSymPtr<char> source);
  NCCL_DEVICE_INLINE void* waitRecv();
  NCCL_DEVICE_INLINE void postRecv();
  NCCL_DEVICE_INLINE void advanceStep();
  NCCL_DEVICE_INLINE void* advanceFifoSlot();
  NCCL_DEVICE_INLINE uint64_t stepFlag() const;
  NCCL_DEVICE_INLINE bool usesGin() const;
};

#if !defined(NCCL_OS_WINDOWS)
// Runtime LSA/GIN connection adapter. Each connection-role thread constructs exactly one member of Storage according
// to whether its world peer is accessible through ncclGetPeerPointer. Other workers construct neither member.
template <unsigned GinBackendMask, ncclFlowProtocol Protocol>
struct ncclFlowConnection {
  static_assert(GinBackendMask != 0, "A GIN-capable Flow connection requires a nonzero backend mask");

  union Storage {
    ncclLsaFlowConn<Protocol> lsa;
    ncclGinFlowConn<GinBackendMask, Protocol> gin;

    NCCL_DEVICE_INLINE Storage() {}
    NCCL_DEVICE_INLINE ~Storage() {}
  } storage;
  bool useGin;

  NCCL_DEVICE_INLINE ncclFlowConnection() : useGin(false) {}
  NCCL_DEVICE_INLINE void construct(ncclDevComm const& comm, ncclFlowConfig<GinBackendMask> const& config, int peer,
                                    int peerIndex, int maxPeers, int channelId, int nParts, int part, int type,
                                    bool waitRole, bool recvPost);
  NCCL_DEVICE_INLINE void close();
  NCCL_DEVICE_INLINE void* waitSend();
  NCCL_DEVICE_INLINE void postSend(size_t bytes);
  NCCL_DEVICE_INLINE void postSend(size_t bytes, bool direct, ncclSymPtr<char> output, ncclSymPtr<char> source);
  NCCL_DEVICE_INLINE void* waitRecv();
  NCCL_DEVICE_INLINE void postRecv();
  NCCL_DEVICE_INLINE void advanceStep();
  NCCL_DEVICE_INLINE void* advanceFifoSlot();
  NCCL_DEVICE_INLINE uint64_t stepFlag() const;
  NCCL_DEVICE_INLINE bool usesGin() const;
};
#endif
#endif

#endif // NCCL_DEVICE_FLOW__INTERNALS_H_
