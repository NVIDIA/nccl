/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_REGISTER_INLINE_H_
#define NCCL_REGISTER_INLINE_H_

#include "comm.h"
#include "register.h"

static inline ncclResult_t ncclRegFind(struct ncclComm* comm, const void* data, size_t size, struct ncclReg** outReg) {
  struct ncclRegCache* cache = &comm->regCache;
  uintptr_t endAddr = (uintptr_t)data + size;
  constexpr int linearScanSlots = 32;
  int lo = 0, hi = cache->population;
  *outReg = nullptr;
  if (hi > linearScanSlots) {
    ncclReg* first = cache->slots[0];
    if ((uintptr_t)data < first->begAddr || endAddr > cache->slots[hi - 1]->maxEndAddr) return ncclSuccess;
    if (endAddr <= first->endAddr) {
      *outReg = first;
      return ncclSuccess;
    }
    lo = 1;
  }
  // The first prefix reaching endAddr ends at the first record large enough.
  // Later records start at the same or a higher address.
  while (hi - lo > linearScanSlots) {
    int mid = lo + (hi - lo) / 2;
    if (cache->slots[mid]->maxEndAddr < endAddr) lo = mid + 1;
    else hi = mid;
  }
  for (; lo < cache->population; ++lo) {
    ncclReg* reg = cache->slots[lo];
    if ((uintptr_t)data < reg->begAddr) break;
    if (endAddr <= reg->endAddr) {
      *outReg = reg;
      break;
    }
  }
  return ncclSuccess;
}

#endif
