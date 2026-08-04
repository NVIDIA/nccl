/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2015-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_CHANNEL_H_
#define NCCL_CHANNEL_H_
#include "comm.h"
#include "utils.h"

#include <algorithm>

ncclResult_t initChannel(struct ncclComm* comm, int channelid);
ncclResult_t initNvlsChannel(struct ncclComm* comm, int channelId, struct ncclComm* parent, bool share);
ncclResult_t initCollnetChannel(struct ncclComm* comm, int channelId, struct ncclComm* parent, bool share);
ncclResult_t freeChannel(struct ncclChannel* channel, int nRanks, int collnetNRanks, int nvlsNRanks,
                         struct ncclComm* comm);

static constexpr size_t ncclMinTrafficPerChannel = 32 << 10;

struct ncclCollChannelLayout {
  size_t countLo;
  size_t countMid;
  size_t countHi;
  size_t cellsLo;
  size_t cellsHi;
  size_t elementsPerCell;
  int channelOffset;
  int nMidChannels;
  int nChannels;
};

int ncclClampChannels(int nChannels, int minChannels, int maxChannels);
int ncclFuncTrafficPerByte(ncclFunc_t func, int nRanks);
size_t ncclCollTrafficPerChannel(size_t trafficBytes, int nChannels);
struct ncclCollChannelLayout ncclComputeCollChannelLayout(size_t count, size_t elementSize, int trafficPerByte,
                                                          size_t trafficPerChannel, size_t currentTraffic,
                                                          int channelId, int maxChannels);

inline uint8_t ncclP2pChannelBaseForRound(struct ncclComm* comm, int p2pRound) {
  int base;
  if (comm->nNodes > 1) {
    int localSize = comm->p2pSchedGroupSize;
    int groupDelta = p2pRound / localSize;
    int localDelta = p2pRound % localSize;
    base = groupDelta * divUp(localSize, NCCL_MAX_DEV_WORK_P2P_PER_BATCH);
    base += localDelta / NCCL_MAX_DEV_WORK_P2P_PER_BATCH;
  } else {
    base = p2pRound;
  }
  return reverseBits(base, log2Up(comm->p2pnChannels));
}

#endif
