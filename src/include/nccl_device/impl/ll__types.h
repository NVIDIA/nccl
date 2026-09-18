/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2015-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_DEVICE_LL__TYPES_H_
#define NCCL_DEVICE_LL__TYPES_H_

#include "core__types.h"

#if __cplusplus
union ncclLLFifoLine {
  // Flags follow data so that a partial network receive cannot expose a flag before its data. This assumes
  // contiguous network writes or eight-byte write atomicity.
  struct {
    uint32_t data1;
    uint32_t flag1;
    uint32_t data2;
    uint32_t flag2;
  };
  uint64_t v[2];
  int4 i4;
};
static_assert(sizeof(ncclLLFifoLine) == 16, "Unexpected LL FIFO-line size");
#endif

#define NCCL_LL_LINES_PER_THREAD 8
#ifdef TEST_LL_CLEANUP
#define NCCL_LL_CLEAN_MASK 0x078 // Set to 0x100 to disable cleanup.
#define NCCL_LL_FLAG_MAX 0x100
#define NCCL_LL_FLAG(a) ((uint32_t)((a) % NCCL_LL_FLAG_MAX))
#else
#define NCCL_LL_CLEAN_MASK 0x7ffffff8
#define NCCL_LL_FLAG(a) ((uint32_t)(a))
#endif

#endif // NCCL_DEVICE_LL__TYPES_H_
