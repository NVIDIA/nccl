/*************************************************************************
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef _NCCL_DEVICE_CORE__TYPES_H_
#define _NCCL_DEVICE_CORE__TYPES_H_
#include "../core.h"
#include "nccl_device/gin/gin_device_host_common.h"

// nccl.h has: typedef ncclWindow_vidmem* ncclWindow_t;
struct ncclWindow_vidmem {
  void* winHost;
  char* lsaFlatBase; // pointer to first byte for rank 0 of lsa team
  int lsaRank;
  int worldRank;
  uint32_t stride4G;
  uint32_t mcOffset4K;
  uint32_t ginOffset4K;
  ncclGinWindow_t ginWins[NCCL_GIN_MAX_CONTEXTS];
};

struct ncclMultimemHandle {
  void* mcBasePtr;
};

#endif
