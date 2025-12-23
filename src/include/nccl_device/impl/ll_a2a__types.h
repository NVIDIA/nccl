/*************************************************************************
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef _NCCL_DEVICE_LL_A2A__TYPES_H_
#define _NCCL_DEVICE_LL_A2A__TYPES_H_
#include "../ll_a2a.h"
#include "core__types.h"

struct ncclLLA2AHandle {
  ncclDevResourceHandle_t bufHandle;
  uint32_t nSlots;
};

#if NCCL_CHECK_CUDACC
template<typename Coop>
struct ncclLLA2ASession_internal {
  Coop coop;
  ncclDevComm const& comm;
  ncclTeam team;
  ncclLLA2AHandle handle;
  int block;
  int pitch;
  bool multimem;
  ncclMultimemHandle mmHandle;
  uint32_t epoch;
  uint32_t slotsOffset;

  NCCL_DEVICE_INLINE uint32_t calcSlotOffset() const {
    return block*(1 + 2*handle.nSlots) + 1 + (epoch & 1)*handle.nSlots;
  }
};
#endif

#endif // _NCCL_DEVICE_LL_A2A__TYPES_H_
