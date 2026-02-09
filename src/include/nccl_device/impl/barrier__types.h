/*************************************************************************
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef _NCCL_DEVICE_BARRIER__TYPES_H_
#define _NCCL_DEVICE_BARRIER__TYPES_H_
#include "../barrier.h"
#include "../utility.h"

#if NCCL_CHECK_CUDACC
template<typename Coop>
struct ncclBarrierSession_internal {
  Coop coop;
  nccl::utility::Optional<ncclGin> gin;
  nccl::utility::Optional<ncclLsaBarrierSession<Coop>> innerLsaBar;
  nccl::utility::Optional<ncclGinBarrierSession<Coop>> outerGinBar;

  template<typename GinInit, typename InnerInit, typename OuterInit>
  NCCL_DEVICE_INLINE ncclBarrierSession_internal(
      Coop coop, GinInit ginInit, InnerInit innerInit, OuterInit outerInit
    ):
    coop(coop), gin{ginInit}, innerLsaBar{innerInit}, outerGinBar{outerInit} {
  }
};
#endif

#endif // _NCCL_DEVICE_BARRIER__TYPES_H_
