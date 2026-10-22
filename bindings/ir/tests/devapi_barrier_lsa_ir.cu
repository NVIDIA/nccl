#include "nccl_device_ir.h"

extern "C" __global__ void runDevice(ncclIrDevComm comm, ncclLsaBarrierHandle handle) {
#if __CUDA_ARCH__ >= 700
    constexpr int kRounds = 10;

    // Compute LSA team on device
    ncclTeam team = ncclIrTeamLsa(&comm);

    // Initialize CTA coop
    ncclIrCoop coop;
    ncclIrCoopInitCta(&coop);

    // Initialize LSA barrier session via C-ABI
    ncclIrLsaBarrierSession session;

    ncclMultimemHandle mmHandle{}; // unused when multimem=false
    uint32_t* state =
        (uint32_t*)ncclIrGetResourceBufferLocalPointer(&comm, handle.bufHandle);
    uint32_t initialEpoch = state[handle.nBarriers + blockIdx.x];

    ncclIrLsaBarrierSessionInitLsa(
        &session, &coop, &comm, blockIdx.x, /*multimem=*/false);

    // Barrier rounds
    for (int round = 0; round < kRounds; ++round) {
        ncclIrLsaBarrierSessionSync(&session, &coop, cuda::memory_order_relaxed);
        if (threadIdx.x == 0 && blockIdx.x == 0 && team.rank == (round % team.nRanks)) {
            printf("Round %d\n", round);
        }
    }
    ncclIrLsaBarrierSessionDestroy(&session);

    // Destroy must write the session's epoch back to the resource buffer. A
    // missing write-back does not hang: the next session restarts from the
    // previously persisted epoch, its wait predicate is already satisfied by
    // the stale inbox, and the barrier silently degrades to a no-op. So check
    // the slot explicitly -- non-multimem epochs live at nBarriers + index.
    if (threadIdx.x == 0) {
        uint32_t epoch = state[handle.nBarriers + blockIdx.x];
        assert(epoch == initialEpoch + kRounds);
    }

    // Reuse the same storage and slot, now that the epoch has been persisted.
    ncclIrLsaBarrierSessionInit(
        &session, &coop, &comm, team, handle, blockIdx.x,
        /*multimem=*/false, mmHandle);
    ncclIrLsaBarrierSessionSync(&session, &coop, cuda::memory_order_relaxed);
    ncclIrLsaBarrierSessionDestroy(&session);
#endif
}
