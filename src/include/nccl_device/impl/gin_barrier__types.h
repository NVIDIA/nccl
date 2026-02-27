/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef _NCCL_DEVICE_GIN_BARRIER__TYPES_H_
#define _NCCL_DEVICE_GIN_BARRIER__TYPES_H_
#include "../gin_barrier.h"
#include "core__types.h"
#include "gin__types.h"

struct ncclGinBarrierHandle {
  ncclGinSignal_t signal0;
  ncclDevResourceHandle_t unused;
};

#if NCCL_CHECK_CUDACC
template<typename Coop>
struct ncclGinBarrierSession_internal {
  Coop coop;
  ncclGin net;
  ncclTeam team;
  ncclGinBarrierHandle handle;
  int index;
  ncclGinSignal_t signal;
};
#endif

#endif // _NCCL_DEVICE_GIN_BARRIER__TYPES_H_
