/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_DEVICE_GENERAL_FLOW_REDUCE_H_
#define NCCL_DEVICE_GENERAL_FLOW_REDUCE_H_

#include "../common_kernel.h"

template <typename RedOp>
struct ncclGenkRedOpArg {
  NCCL_DEVICE_INLINE static uint64_t get(RedOp const&) {
    return 0;
  }
};

template <typename T>
struct ncclGenkRedOpArg<FuncMinMax<T>> {
  NCCL_DEVICE_INLINE static uint64_t get(FuncMinMax<T> const& redOp) {
    return redOp.xormask.native;
  }
};

// Keep the legacy reduction primitives private to Genk; Flow receives this policy through its constructor.
struct ncclGenkFlowOps {
  // Genk supports basic reductions but applies no pre-op or post-op.
  template <typename T, typename RedOp, int MaxSrcs, int MaxDsts, typename SrcFn, typename DstFn>
  NCCL_DEVICE_INLINE void reduceCopy(int thread, int nThreads, int nSrcs, SrcFn const& srcFn, int nDsts,
                                     DstFn const& dstFn, RedOp const& redOp, int nElts) const {
    ::reduceCopy<ncclCollUnroll(), RedOp, T, 0, 1, MaxSrcs, 0, 1, MaxDsts, /*PreOpSrcs=*/0>(
      thread, nThreads, ncclGenkRedOpArg<RedOp>::get(redOp), /*postOp=*/false, nSrcs, srcFn, nDsts, dstFn, nElts);
  }

  template <typename RedOp>
  NCCL_DEVICE_INLINE uint64_t reducePacked(RedOp const& redOp, uint64_t a, uint64_t b) const {
    return applyReduce(redOp, a, b);
  }
};

#endif // NCCL_DEVICE_GENERAL_FLOW_REDUCE_H_
