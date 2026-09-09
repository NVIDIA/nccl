/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef _NCCL_DEVICE_CFT_H_
#define _NCCL_DEVICE_CFT_H_

#include "core.h"

#ifdef __CUDACC__

enum class ncclMemProxyType : uint32_t {
  Generic = 1u << 0,
  Fabric = 1u << 1,
};

enum class ncclMemFenceScope : uint32_t {
  Cta = 1u << 0,
  Sys = 1u << 1,
};

struct ncclCftSmem;

template <typename Coop>
struct ncclCft_internal;

template <typename Coop>
NCCL_DEVICE_INLINE void ncclMemFence(Coop coop, cuda::memory_order order, ncclMemProxyType producer,
                                     ncclMemProxyType consumer, ncclMemFenceScope scope);

template <typename Coop>
struct ncclCft : ncclCft_internal<Coop> {
  NCCL_DEVICE_INLINE ncclCft(Coop coop, ncclCftSmem& cftSmem);

  NCCL_DEVICE_INLINE ~ncclCft();

  template <typename OpCoop>
  NCCL_DEVICE_INLINE void submit(OpCoop coop);

  template <typename OpCoop>
  NCCL_DEVICE_INLINE void flushSmem(OpCoop coop);

  template <typename OpCoop>
  NCCL_DEVICE_INLINE void flush(OpCoop coop, bool* hasReport = nullptr, uint32_t* report = nullptr);

  template <typename OpCoop>
  NCCL_DEVICE_INLINE void put(OpCoop coop, ncclCftLeId leId, size_t leOffset, void* smemSource, uint32_t bytes);

  template <typename OpCoop>
  NCCL_DEVICE_INLINE void putCpMask(OpCoop coop, ncclCftLeId leId, size_t leOffset, void* smemSource, uint32_t bytes,
                                    uint16_t cpMask);

  template <typename OpCoop>
  NCCL_DEVICE_INLINE void putMultimem(OpCoop coop, ncclCftLeId leId, size_t leOffset, void* smemSource, uint32_t bytes);

  template <typename OpCoop>
  NCCL_DEVICE_INLINE void putMultimemCpMask(OpCoop coop, ncclCftLeId leId, size_t leOffset, void* smemSource,
                                            uint32_t bytes, uint16_t cpMask);

  template <typename OpCoop>
  NCCL_DEVICE_INLINE void get(OpCoop coop, ncclCftLeId leId, size_t leOffset, void* smemDestination, uint32_t bytes);

  template <typename RedOp, typename OpCoop>
  NCCL_DEVICE_INLINE void red(OpCoop coop, ncclCftLeId leId, size_t leOffset, RedOp const& red, void* smemSource,
                              uint32_t bytes);

  template <typename RedOp, typename OpCoop>
  NCCL_DEVICE_INLINE void redCpMask(OpCoop coop, ncclCftLeId leId, size_t leOffset, RedOp const& red, void* smemSource,
                                    uint32_t bytes, uint16_t cpMask);

  template <typename RedOp, typename OpCoop>
  NCCL_DEVICE_INLINE void redMultimem(OpCoop coop, ncclCftLeId leId, size_t leOffset, RedOp const& red,
                                      void* smemSource, uint32_t bytes);

  template <typename RedOp, typename OpCoop>
  NCCL_DEVICE_INLINE void redMultimemCpMask(OpCoop coop, ncclCftLeId leId, size_t leOffset, RedOp const& red,
                                            void* smemSource, uint32_t bytes, uint16_t cpMask);

  template <typename RedOp, typename OpCoop>
  NCCL_DEVICE_INLINE void pullRed(OpCoop coop, ncclCftLeId leId, size_t leOffset, RedOp const& red,
                                  void* smemDestination, uint32_t bytes);
};
#endif // __CUDACC__

#endif // _NCCL_DEVICE_CFT_H_
