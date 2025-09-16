/*************************************************************************
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef _NCCL_DEVICE_CORE__TYPES_H_
#define _NCCL_DEVICE_CORE__TYPES_H_
#include "../core.h"

// nccl.h has: typedef ncclWindow_vidmem* ncclWindow_t;
struct ncclWindow_vidmem {
  void* winHost;
  //ncclGinWindow_t ginWin;
  char* lsaFlatBase; // pointer to first byte for rank 0 of lsa team
  int lsaRank;
  int worldRank;
  uint32_t stride4G;
  uint32_t mcOffset4K;
};

struct ncclMultimemHandle {
  void* mcBasePtr;
};

#endif
