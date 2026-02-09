#include "sym_kernels.h"
#include "kernel.cuh"
#include "primitives.cuh"
#include "gin_scratch__types.h"

__device__ __forceinline__ void ncclSymkRun_AllGather_GinHier_MCRing(struct ncclSymkDevWorkArgs const* args) {
  ncclCoopCta cta;
  ncclSymkArgsHandler handler(args);
  ncclTeam rail = ncclTeamRail(handler.comm);
  ncclGin gin(handler.comm, (int)blockIdx.x);
  constexpr int chunkSize = ncclSymkGinRailBufSize;
  ncclGinSignal_t railSignals = handler.ginSyncHandle.railSignals + blockIdx.x * rail.nRanks;
  ncclBarrierSession<ncclCoopCta> bar(cta, ncclTeamTagWorld(), gin, blockIdx.x, /*multimem=*/true);
  int nextPeer = (rail.rank + 1) % rail.nRanks;
  int prevPeer = (rail.rank + rail.nRanks - 1) % rail.nRanks;
  uint64_t* localSignalPtr = gin.getSignalShadowPtr(railSignals + prevPeer);
  uint64_t localSignalValue = *localSignalPtr;
  const int ringThreads = WARP_SIZE;

  bar.sync(cta, cuda::memory_order_relaxed, ncclGinFenceLevel::Relaxed);

  handler.template forEachWorkNoFusion<uint8_t>(
    [&]__device__(size_t nElts, size_t nAllElts, ncclSymPtr<uint8_t> input, ncclSymPtr<uint8_t> output) {
      if (threadIdx.x < ringThreads) {
        ncclCoopWarpSpan warps(0, 1, 0);
        for (int step = 0; step < rail.nRanks - 1; step++) {
          int dataPeer = (rail.rank - step + rail.nRanks) % rail.nRanks;
          int dgrank = ncclTeamRankToWorld(handler.comm, rail, dataPeer);
          size_t remainingElts = nElts;
          size_t offset = 0;
          if (dataPeer == rail.rank) {
            while (remainingElts) {
              size_t chunkElts = min(remainingElts, size_t(chunkSize));
              // Send data chunk to next peer in ring
              gin.put(rail, nextPeer, output + dgrank * nAllElts + offset,
                input + offset, chunkElts,
                ncclGin_SignalInc{ railSignals + rail.rank }, ncclGin_None{}, warps);
              offset += chunkElts;
              remainingElts -= chunkElts;
            }
          } else {
            while (remainingElts) {
              size_t chunkElts = min(remainingElts, size_t(chunkSize));
              // Wait for ready signal from next peer before sending
              gin.waitSignal(warps, railSignals + prevPeer, localSignalValue + 1, 32);
              // Send data chunk to next peer in ring
              gin.put(rail, nextPeer, output + dgrank * nAllElts + offset,
                output + dgrank * nAllElts + offset, chunkElts,
                ncclGin_SignalInc{ railSignals + rail.rank }, ncclGin_None{}, warps);
              offset += chunkElts;
              remainingElts -= chunkElts;
              localSignalValue++;
            }
          }
        }
        gin.flush(warps);
      } else {
        ncclCoopWarpSpan warps(1, blockDim.x / WARP_SIZE - 1, 1);
        // Loop through rail ranks starting from itself
        for (int step = 0; step < rail.nRanks; step++) {
          int dataPeer = (rail.rank - step + rail.nRanks) % rail.nRanks;
          int dgrank = ncclTeamRankToWorld(handler.comm, rail, dataPeer);
          size_t remainingElts = nElts;
          size_t offset = 0;
          if (dataPeer == rail.rank) {
            while (remainingElts) {
              size_t chunkElts = min(remainingElts, size_t(chunkSize));
              // Put self rank's data
              bcastMultimem(handler, warps.num_threads(), warps.thread_rank(), input + offset, output + dgrank * nAllElts + offset, chunkElts);
              offset += chunkElts;
              remainingElts -= chunkElts;
            }
          } else {
            while (remainingElts) {
              size_t chunkElts = min(remainingElts, size_t(chunkSize));
              // Wait for signal from other peers before putting their data
              gin.waitSignal(warps, railSignals + prevPeer, localSignalValue + 1, 32);
              bcastMultimem(handler, warps.num_threads(), warps.thread_rank(), output + dgrank * nAllElts + offset, output + dgrank * nAllElts + offset, chunkElts);
              offset += chunkElts;
              remainingElts -= chunkElts;
              localSignalValue++;
            }
          }
        }
      }
    }
  );

  // update the shadow signal value
  if (threadIdx.x == ringThreads) {
    *localSignalPtr = localSignalValue;
  }
  bar.sync(cta, cuda::memory_order_release, ncclGinFenceLevel::Relaxed);
}
