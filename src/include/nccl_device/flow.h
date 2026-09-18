/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more information
 *************************************************************************/

#ifndef NCCL_DEVICE_FLOW_H_
#define NCCL_DEVICE_FLOW_H_

#include "impl/flow__types.h"
#include "ptr.h"
#include "utility.h"

#ifdef __CUDACC__
// Flow combines local inputs and data received from world-rank peers, then copies/reduces the result to local
// outputs and/or sends it to peers. To use it, provide the persistent resources described below, fill
// ncclFlowConfig, and construct one ncclFlowProcessor on every thread of the CTA. Call process()
// collectively for each transfer, then close() before the kernel exits. The processor chooses LSA or GIN for each
// peer when its GIN backend mask is nonzero.

// Return this thread's part before constructing a processor, for algorithms that select different work or peers
// for each part. Use the same warp counts as the processor: partWorkerWarps may split the workers unevenly;
// nullptr divides them evenly. Pass ncclFlowGetSignalWarpsPerPart(Protocol) for protocols whose signaling-warp
// count differs from the default.
NCCL_DEVICE_INLINE int ncclFlowThreadPart(int nWorkerWarps, int nParts, int const* partWorkerWarps = nullptr,
                                          int signalWarpsPerPart = NCCL_FLOW_SIGNAL_WARPS_PER_PART) {
  int const warp = (int)threadIdx.x / 32;
  for (int part = 0; part < nParts; part++) {
    int const partWarpEnd = ncclFlowPartWorkerWarp0(nWorkerWarps, nParts, part, partWorkerWarps, signalWarpsPerPart) +
                            ncclFlowPartWorkerWarps(nWorkerWarps, nParts, part, partWorkerWarps) + signalWarpsPerPart;
    if (warp < partWarpEnd) return part;
  }
  return nParts - 1;
}

// Placeholder pointer generator for an empty process source or destination list.
template <typename T>
struct ncclFlowNullPtrFn {
  NCCL_DEVICE_INLINE T* operator()(int) const {
    return nullptr;
  }
};

template <bool Send, bool Direct = false>
struct ncclFlowSendTag {
  static_assert(Send || !Direct, "A disabled Flow send cannot be direct");
  bool direct;

  NCCL_DEVICE_INLINE constexpr ncclFlowSendTag(bool direct_ = Direct) : direct(Direct && direct_) {}
};
using ncclFlowSend = ncclFlowSendTag<true>;
using ncclFlowDirectSend = ncclFlowSendTag<true, true>;
using ncclFlowNoSend = ncclFlowSendTag<false>;

template <bool Recv, bool Direct = false>
struct ncclFlowRecvTag {
  static_assert(Recv || !Direct, "A disabled Flow receive cannot be direct");
  bool direct;

  NCCL_DEVICE_INLINE constexpr ncclFlowRecvTag(bool direct_ = Direct) : direct(Direct && direct_) {}
};
using ncclFlowRecv = ncclFlowRecvTag<true>;
using ncclFlowDirectRecv = ncclFlowRecvTag<true, true>;
using ncclFlowNoRecv = ncclFlowRecvTag<false>;

// Direct tags are for Simple transfers involving a window-registered user output. They retain connection
// synchronization while bypassing the corresponding FIFO copy. Pass false to the tag constructor when a direct
// transfer is not eligible, so that process() uses the FIFO path. A registered input can also avoid GIN send
// staging in a send-only operation.

template <unsigned GinBackendMask>
struct ncclFlowConfig;

// Provide symmetric, window-backed receive FIFO and connection-state pointers so Flow can resolve both local
// and peer addresses. devComm resources are one way to allocate them. channelId is a caller-chosen index for
// independent Flow instances, not necessarily an NCCL channel. For channelId in [0, nChannels), one layout can use:
//   channelSize = nParts * MaxPeers * nSlots * slotBytes
//   recvBuffer: nChannels * channelSize bytes
//   connStatesPerChannel = nParts * MaxPeers
//   connState: nChannels * connStatesPerChannel * sizeof(ncclFlowConnState) bytes
// MaxPeers is the per-part capacity for send and receive peers, not the number active in every call; it must
// be at least one. The receive FIFO and connection state must be zero-initialized. nSlots must be at least
// SlotsPerProcess. slotBytes is the FIFO storage per slot, including protocol flags: it must be a multiple of
// sizeof(T) for Simple, sizeof(ncclLLFifoLine) for LL, or NCCL_LL128_LINESIZE for LL128. LL/LL128 require
// SlotsPerProcess = 1. Each process() call must fit in SlotsPerProcess slots.
//
// connState holds the connection state between peers (signals, counters, and current steps). Its pointer starts
// at the first state of this Flow layout, which may be a region of a larger allocation. State is indexed by
// channelId * connStatesPerChannel + part * MaxPeers + slot. The stride must be at least nParts * MaxPeers;
// allocate nChannels * connStatesPerChannel * sizeof(ncclFlowConnState) bytes for this layout. Mutually
// exclusive protocols using the same connections may reuse the state. LL/LL128 readiness flags are in the FIFO.
//
// Each connection-state slot that may use GIN needs two indexed signals, starting at ginSignal0. Thus
// 2 * nChannels * connStatesPerChannel signals suffice for the dense layout. GIN sends also
// need local, window-backed staging: up to channelSize bytes for each channel that may send to a GIN peer.
// ginBuffer points to that channel's staging, or may be empty when it cannot send through GIN.
// The devComm must have strong signals, at least one GIN context, and connections to possible GIN peers.
// With GinBackendMask = 0, no GIN signals or staging are needed, but all peers must be LSA-accessible.
// Communicating ranks must agree on the reciprocal channel, part, peer-slot, and nSlots layout. GIN staging is
// local; the receive FIFO, connection state, and GIN signal indices must also be valid at the peer.

template <>
struct ncclFlowConfig<0> {
  ncclSymPtr<char> recvBuffer;             // Persistent receive FIFO.
  ncclSymPtr<ncclFlowConnState> connState; // Persistent per-connection credits and LSA signals.
  size_t channelSize;                     // Receive-FIFO bytes per channel.
  int nSlots;                             // FIFO depth per peer slot.
  int connStatesPerChannel;               // Stride in connection states between channels.
};

#if !defined(NCCL_OS_WINDOWS)
template <unsigned GinBackendMask>
struct ncclFlowConfig {
  static_assert(GinBackendMask != 0, "A GIN-capable Flow configuration requires a nonzero backend mask");

  ncclSymPtr<char> recvBuffer;
  ncclSymPtr<ncclFlowConnState> connState;
  size_t channelSize;
  int nSlots;                       // FIFO depth per peer slot.
  int connStatesPerChannel;
  ncclTeam ginTeam;                 // Must contain every possible GIN peer.
  ncclGinSignal_t ginSignal0;       // First signal corresponding to connection-state index zero.
  // Local send staging for this channel; may be empty if every send peer is LSA-accessible.
  ncclSymPtr<char> ginBuffer;
  // Logical index mapped onto the devComm's GIN contexts.
  int ginContextIndex;
};
#endif

#include "impl/flow__internals.h"

// GinBackendMask = 0 compiles an LSA-only processor; otherwise it enables the selected GIN backends and chooses
// LSA or GIN per peer at runtime. SlotsPerProcess is the maximum number of FIFO slots in one process() call.
// MaxPeers is the per-part connection capacity. It bounds nSendPeers, nRecvPeers, and every index in sendSlots
// and recvSlots, even if some calls have no peers. Each slot has its own FIFO and connection state.
// Protocol selects the FIFO format.
template <unsigned GinBackendMask, int SlotsPerProcess = 1, int MaxPeers = 1,
          ncclFlowProtocol FlowProtocol = ncclFlowProtocolSimple>
struct ncclFlowProcessor;

#if defined(__CUDACC_EXTENDED_LAMBDA__)
template <unsigned GinBackendMask, int SlotsPerProcess, int MaxPeers, ncclFlowProtocol FlowProtocol>
struct ncclFlowProcessor : ncclFlowBase<ncclFlowProcessor<GinBackendMask, SlotsPerProcess, MaxPeers, FlowProtocol>,
                                        SlotsPerProcess, MaxPeers> {
  using Base =
    ncclFlowBase<ncclFlowProcessor<GinBackendMask, SlotsPerProcess, MaxPeers, FlowProtocol>, SlotsPerProcess, MaxPeers>;
  static constexpr ncclFlowProtocol Protocol = FlowProtocol;

  // Set up a Flow processor for this CTA to exchange data with the specified peers. sendPeers and recvPeers
  // contain world ranks, with each count at most MaxPeers. config supplies the FIFO layout (including nSlots),
  // connection state, and optional GIN resources; channelId selects their instance.
  // Every CTA thread constructs a processor; shmem resides in CTA shared memory.
  //
  // nParts splits the CTA into independent flows, such as reduce and broadcast. sendPart and recvPart select
  // the connection-state part for each direction. If a peer list is compacted or reordered, sendSlots and
  // recvSlots map its entries to stable slots; otherwise entry i uses slot i. Threads in a part use the
  // same peers and slot maps. partWorkerWarps optionally divides nWorkerWarps among parts. Launch
  // 32 * (nWorkerWarps + nParts * ncclFlowGetSignalWarpsPerPart(Protocol))
  // threads, with at least 2 * nParts worker warps.
  NCCL_DEVICE_INLINE ncclFlowProcessor(ncclDevComm const& comm, ncclFlowConfig<GinBackendMask> const& config,
                                       int const* sendPeers, int nSendPeers, int const* recvPeers, int nRecvPeers,
                                       int channelId, int nWorkerWarps, ncclFlowShmem& shmem, int nParts = 1,
                                       int sendPart = 0, int recvPart = 0, int const* sendSlots = nullptr,
                                       int const* recvSlots = nullptr, int const* partWorkerWarps = nullptr);

  // Process nElts elements collectively. send/recv tags select the active directions. src(i) and dst(i)
  // provide nSrc and nDst explicit T* operands; Flow appends non-direct receive FIFOs to the sources and
  // non-direct send FIFOs to the destinations. input/output preserve user-window metadata for direct Simple
  // transfers; pass an empty ncclSymPtr<T> when that window is not used. Direct LSA sends also need the remote
  // output in the explicit destinations. Every participating thread calls process() with matching nElts;
  // zero elements consume no FIFO slots. The transfer must fit in SlotsPerProcess slots.
  template <bool Send, bool DirectSend, bool Recv, bool DirectRecv, typename T, typename SrcLambda, typename DstLambda,
            typename RedOp>
  NCCL_DEVICE_INLINE void process(ncclFlowSendTag<Send, DirectSend> send, ncclFlowRecvTag<Recv, DirectRecv> recv,
                                  ncclSymPtr<T> input, ncclSymPtr<T> output, SrcLambda srcLambda, int nSrc,
                                  DstLambda dstLambda, int nDst, RedOp const& redOp, size_t nElts) {
    Base::process(send, recv, input, output, srcLambda, nSrc, dstLambda, nDst, redOp, nElts);
  }

  // Call after the final process() to save persistent connection state and drain external GIN sources.
  NCCL_DEVICE_INLINE void close() {
    Base::close();
  }

private:
  friend struct ncclFlowBase<ncclFlowProcessor<GinBackendMask, SlotsPerProcess, MaxPeers, FlowProtocol>,
                             SlotsPerProcess, MaxPeers>;

  ncclFlowConnection<GinBackendMask, FlowProtocol> conn;

  NCCL_DEVICE_INLINE void* waitSend();
  NCCL_DEVICE_INLINE void postSend(size_t bytes = 0);
  NCCL_DEVICE_INLINE void* waitRecv();
  NCCL_DEVICE_INLINE void postRecv();

  NCCL_DEVICE_INLINE void closeBackend();
  NCCL_DEVICE_INLINE void postSend(size_t bytes, bool direct, ncclSymPtr<char> output, ncclSymPtr<char> source);
  NCCL_DEVICE_INLINE void advanceStep();
  NCCL_DEVICE_INLINE void* advanceFifoSlot();
  NCCL_DEVICE_INLINE uint64_t stepFlag() const;
  NCCL_DEVICE_INLINE bool usesGin() const;
};
#else // __CUDACC_EXTENDED_LAMBDA__
template <unsigned GinBackendMask, int SlotsPerProcess, int MaxPeers, ncclFlowProtocol FlowProtocol>
struct ncclFlowProcessor {
  static_assert(nccl::utility::always_false<ncclFlowProcessor>::value,
                "NCCL Flow requires device side lambdas, please use '--extended-lambda' as compilation flag to enable "
                "it.");

  template <typename... Args>
  NCCL_DEVICE_INLINE ncclFlowProcessor(Args&&...) {}
};
#endif // __CUDACC_EXTENDED_LAMBDA__

template <int SlotsPerProcess = 1, int MaxPeers = 1, ncclFlowProtocol Protocol = ncclFlowProtocolSimple>
using ncclLsaFlowProcessor = ncclFlowProcessor<0, SlotsPerProcess, MaxPeers, Protocol>;
#endif

#endif // NCCL_DEVICE_FLOW_H_
