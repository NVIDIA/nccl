/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more information
 *************************************************************************/

#include "nccl_device_ir.h"

enum ReduceCopyApi {
  LsaReduceSum,
  MultimemReduceSum,
  LsaCopy,
  MultimemCopy,
  LsaReduceSumCopy,
  MultimemReduceSumCopy,
  LocalReduceSumCopy,
};

extern "C" __global__ void runReduceCopyF32(
    ncclIrDevComm comm, int api, ncclWindow_t srcWindow, size_t srcOffset,
    float* src, float* mcSrc, ncclWindow_t dstWindow, size_t dstOffset,
    float* dst, float* mcDst, size_t displ, size_t count) {
#if __CUDA_ARCH__ >= 900
  ncclTeam team = ncclIrTeamLsa(&comm);
  ncclIrCoop coop;
  ncclIrCoopInitCta(&coop);

  ncclIrLsaBarrierSession session;
  ncclIrLsaBarrierSessionInitLsa(
      &session, &coop, &comm, blockIdx.x, /*multimem=*/false);
  ncclIrLsaBarrierSessionSync(&session, &coop, cuda::memory_order_acquire);

  if (team.rank == 0) {
    switch (api) {
      case LsaReduceSum:
        ncclIrLsaReduceSum_F32(
            &coop, srcWindow, srcOffset, dst, count, team);
        break;
      case MultimemReduceSum:
        ncclIrMultimemReduceSum_F32(&coop, mcSrc, dst, count);
        break;
      case LsaCopy:
        ncclIrLsaCopy_F32(
            &coop, src, dstWindow, dstOffset, count, team);
        break;
      case MultimemCopy:
        ncclIrMultimemCopy_F32(&coop, src, mcDst, count);
        break;
      case LsaReduceSumCopy:
        ncclIrLsaReduceSumCopy_F32(
            &coop, srcWindow, srcOffset, dstWindow, dstOffset, count, team);
        break;
      case MultimemReduceSumCopy:
        ncclIrMultimemReduceSumCopy_F32(&coop, mcSrc, mcDst, count);
        break;
      case LocalReduceSumCopy:
        ncclIrLocalReduceSumCopy_F32(
            &coop, 2, src, displ, 2, dst, displ, count);
        break;
    }
  }

  ncclIrLsaBarrierSessionSync(&session, &coop, cuda::memory_order_release);
  ncclIrLsaBarrierSessionDestroy(&session);
#endif
}
