/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_DEVICE_LSA_FLOW_H_
#define NCCL_DEVICE_LSA_FLOW_H_

#include "impl/flow__types.h"
#include "ptr.h"

// One LSA Flow connection backed by a persistent symmetric receive FIFO and connection state. Send
// connections write directly into the peer's receive FIFO. Receive connections consume the local FIFO. Flow-level
// code assigns threads and maps topology edges to these resources.
#ifdef __CUDACC__
template <ncclFlowProtocol Protocol = ncclFlowProtocolSimple>
struct ncclLsaFlowConn {
  const int nSlices;
  const int sliceSteps;
  const size_t stepSize;
  const int type;
  uint32_t* abortFlag;
  ncclFlowConnState* state;
  char* fifo;
  // Transport credits are shared across protocols; LL tracks its FIFO slice and flag sequence separately.
  uint64_t step;
  uint32_t llStep;
  uint64_t signalValue;
  uint64_t cache;
  uint64_t* localSignal;
  uint64_t* peerSignal;

  // recvFifo and state identify the same persistent topology connection on every rank. peer is an LSA-accessible
  // world rank addressable through ncclGetPeerPointer. nSlices must divide ncclFlowFifoSlots.
  NCCL_DEVICE_INLINE ncclLsaFlowConn(ncclDevComm const& comm, ncclSymPtr<char> recvFifo,
                                     ncclSymPtr<ncclFlowConnState> state, int peer, int nSlices, size_t stepSize,
                                     int type, bool recvPost);

  NCCL_DEVICE_INLINE void close();
  NCCL_DEVICE_INLINE void* waitSend();
  NCCL_DEVICE_INLINE void postSend(size_t bytes);
  NCCL_DEVICE_INLINE void* waitRecv();
  NCCL_DEVICE_INLINE void postRecv(bool force = false);

  NCCL_DEVICE_INLINE void advanceStep();
  NCCL_DEVICE_INLINE void* advanceSlice();
  NCCL_DEVICE_INLINE uint64_t stepFlag() const;
};

#endif

#endif // NCCL_DEVICE_LSA_FLOW_H_
