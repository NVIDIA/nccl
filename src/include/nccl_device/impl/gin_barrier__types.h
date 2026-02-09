/*************************************************************************
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef _NCCL_DEVICE_GIN_BARRIER__TYPES_H_
#define _NCCL_DEVICE_GIN_BARRIER__TYPES_H_
#include "../gin_barrier.h"
#include "core__types.h"
#include "gin__types.h"

struct ncclGinBarrierHandle {
  ncclGinSignal_t signal0;
  ncclDevResourceHandle_t bufHandle;
};

#if NCCL_CHECK_CUDACC
template<typename Coop>
struct ncclGinBarrierSession_internal {
  Coop coop;
  ncclGin net;
  ncclTeam team;
  ncclGinBarrierHandle handle;
  int index;
  uint32_t epoch;
  ncclGinSignal_t signal;
};
#endif

#endif // _NCCL_DEVICE_GIN_BARRIER__TYPES_H_
