/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef _NCCL_DEVICE_COMM__TYPES_H_
#define _NCCL_DEVICE_COMM__TYPES_H_
#include "../comm.h"
#include "core__types.h"
#include "ll_a2a__types.h"
#include "lsa_barrier__types.h"
#include "gin_barrier__types.h"

#if __cplusplus
struct ncclDevCommWindowTable {
  struct Entry {
    uintptr_t base, size;
    ncclWindow_t window;
  } entries[32];
  struct ncclDevCommWindowTable* next;
};
#endif
typedef struct ncclDevCommWindowTable* ncclDevCommWindowTable_t;

struct ncclDevComm {
  int rank, nRanks;
  uint32_t nRanks_rcp32;
  int lsaRank, lsaSize;
  uint32_t lsaSize_rcp32;

  ncclDevCommWindowTable_t windowTable;

  ncclWindow_t resourceWindow;
  ncclWindow_vidmem_t resourceWindow_inlined;

  ncclMultimemHandle_t lsaMultimem;
  ncclLsaBarrierHandle_t lsaBarrier;
  ncclGinBarrierHandle_t railGinBarrier;

  uint8_t ginConnectionCount;
  uint8_t ginNetDeviceTypes[NCCL_GIN_MAX_CONNECTIONS];
  void* ginHandles[NCCL_GIN_MAX_CONNECTIONS];
  uint32_t ginSignalBase;
  int ginSignalCount;
  uint32_t ginCounterBase;
  int ginCounterCount;
  uint64_t* ginSignalShadows;
  uint32_t ginContextCount;
  uint32_t ginContextBase;
  bool ginIsRailed; // Whether the GIN connections are railed

  // FT related
  uint32_t* abortFlag;

  ncclLsaBarrierHandle_t hybridLsaBarrier;
  ncclGinBarrierHandle_t hybridRailGinBarrier;

  ncclGinBarrierHandle_t worldGinBarrier;
};

#endif // _NCCL_DEVICE_COMM__TYPES_H_
