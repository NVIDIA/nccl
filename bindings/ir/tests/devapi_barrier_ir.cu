#include "nccl_device_ir.h"

extern "C" __global__ void runDevice(ncclIrDevComm comm, ncclLsaBarrierHandle innerHandle, ncclGinBarrierHandle outerHandle) {
#if __CUDA_ARCH__ >= 700
  constexpr int kRounds = 10;
  int t = threadIdx.x;

  // Compute world team on device
  ncclTeam world = ncclIrTeamWorld(&comm);

  // Initialize the opaque CTA coop
  ncclIrCoop coop;
  ncclIrCoopInitCta(&coop);

  // Initialize the ALL-mask GIN object
  ncclIrGin net;
  ncclIrGinInit(&net, &comm, 0);

  // Initialize Barrier session (LSA + GIN) via C-ABI
  ncclIrBarrierSession session;

  ncclMultimemHandle mmHandle{}; // unused when multimem=false
  uint32_t* innerState =
      (uint32_t*)ncclIrGetResourceBufferLocalPointer(&comm, innerHandle.bufHandle);
  uint32_t initialInnerEpoch = innerState[innerHandle.nBarriers + blockIdx.x];

  ncclIrBarrierSessionInitWorld(
      &session, &coop, &net, blockIdx.x, /*multimem=*/false);

  // Barrier rounds
  for (int round = 0; round < kRounds; ++round) {
    ncclIrBarrierSessionSync(&session, &coop, cuda::memory_order_relaxed, ncclGinFenceLevel::None);
    if (t == 0 && blockIdx.x == 0 && world.rank == (round % world.nRanks)) {
      printf("Round %d\n", round);
    }
  }
  ncclIrBarrierSessionDestroy(&session);

  // FenceLevel::None always selects the hierarchical barrier when the inner
  // LSA session is present, so Destroy must persist exactly kRounds advances.
  if (t == 0) {
    uint32_t epoch = innerState[innerHandle.nBarriers + blockIdx.x];
    assert(epoch == initialInnerEpoch + kRounds);
  }

  // Reuse the same storage and slot after the nested LSA epoch was persisted.
  ncclIrBarrierSessionInit(
      &session, &coop, ncclIrTeamLsa(&comm), ncclIrTeamRail(&comm), &net,
      innerHandle, outerHandle, blockIdx.x, /*multimem=*/false, mmHandle);
  ncclIrBarrierSessionSync(
      &session, &coop, cuda::memory_order_relaxed, ncclGinFenceLevel::None);
  ncclIrBarrierSessionDestroy(&session);

  // Exercise the standalone GIN session lifecycle as well. Its current
  // destructor is empty, but the placement-new object still requires a
  // matching C-ABI destroy entrypoint.
  ncclIrGinBarrierSession ginSession;
  ncclIrGinBarrierSessionInit(
      &ginSession, &coop, &net, ncclIrTeamRail(&comm), outerHandle, blockIdx.x);
  ncclIrGinBarrierSessionDestroy(&ginSession);
#endif
}
