#include "sym_kernels.h"
#include "kernel.cuh"
#include "primitives.cuh"
#include "data_ops.cuh"

template<template<typename> typename Red, typename T, bool multimem>
static __device__ void rsAlgoHier(ncclSymkDevWorkArgs const* args, BoolTag<multimem> multimemTag) {
  ncclCoopCta cta;
  ncclSymkArgsHandler handler{args};
  ncclTeam world = ncclTeamWorld(handler.comm);
  ncclTeam rail = ncclTeamRail(handler.comm);
  ncclTeam lsa = ncclTeamLsa(handler.comm);
  ncclGin gin{handler.comm, int(blockIdx.x % handler.comm.ginContextCount)};

  using AccT = typename ncclSymkGinAccumType<Red, T>::Type;
  Red<AccT> red(handler.devWork->redOpArg);
  Red<T> mmRed(handler.devWork->redOpArg);

  int nWorkWarps = blockDim.x/32 - 2;
  int stage0_nWorkWarps;
  if (lsa.nRanks == 1) {
    stage0_nWorkWarps = 0; // Stage 0 just posts sends so no workers.
  } else {
    // Only count reads because they dominate writes.
    int stage0_work = lsa.nRanks == 1 ? 0 : (multimem ? 1 : lsa.nRanks)*(rail.nRanks-1);
    int stage1_work = (multimem ? 1 : lsa.nRanks) + rail.nRanks-1;
    stage0_nWorkWarps = __float2int_rn(__fdividef(nWorkWarps*stage0_work, stage0_work + stage1_work));
    stage0_nWorkWarps = min(stage0_nWorkWarps, nWorkWarps-1); // Stage 1 requires at least 1 worker.
  }

  // 2 stage pipeline, one coop per stage.
  int stage = threadIdx.x/32 < 1 + stage0_nWorkWarps ? 0 : 1;
  ncclCoopWarpSpan coopStage{
    /*warp0=*/stage == 0 ? 0 : 1 + stage0_nWorkWarps,
    /*nWarps=*/1 + (stage == 0 ? stage0_nWorkWarps : nWorkWarps-stage0_nWorkWarps),
    /*id=*/stage
  };
  // Within each stage we have 2 roles: GIN warp, worker warps.
  bool roleIsWorker = 32 <= coopStage.thread_rank();
  ncclCoopWarpSpan coopRole{
    /*warp0=*/(stage == 0 ? 0 : 1 + stage0_nWorkWarps) + (roleIsWorker ? 1 : 0),
    /*nWarps=*/!roleIsWorker ? 1 : (stage == 0 ? stage0_nWorkWarps : nWorkWarps - stage0_nWorkWarps),
    /*id=*/2 + stage
  };

  // Construct outbox only for stage=0 when lsa peers exist.
  alignas(ncclGinOutboxSession<ncclCoopWarpSpan>) char outbox_storage[sizeof(ncclGinOutboxSession<ncclCoopWarpSpan>)];
  ncclGinOutboxSession<ncclCoopWarpSpan>& outbox =
    stage == 0 && lsa.nRanks != 1
      ? *::new(&outbox_storage) ncclGinOutboxSession<ncclCoopWarpSpan>
        {coopStage, gin, handler.ginOutbox, blockIdx.x}
      : reinterpret_cast<ncclGinOutboxSession<ncclCoopWarpSpan>&>(outbox_storage);

  __shared__ int totalSends;
  if (stage == 0 && !roleIsWorker && lsa.nRanks == 1) {
    if (coopRole.thread_rank() == 0) {
      totalSends = 0;
      // When pure rail we use a counter to track sends. We could leave them untracked
      // and end with a flush but by using a counter we can reuse same postSends code
      // for the rail-only and hybrid (lsa!=1) cases.
      gin.resetCounter(handler.ginCounterPerBlock + blockIdx.x);
    }
    coopRole.sync();
  }

  ncclGinInboxA2ASession<ncclCoopCta> inbox
    {cta, gin, rail, handler.ginInboxRail, blockIdx.x};

  ncclLsaBarrierSession<ncclCoopCta> lsaBar
    {cta, handler.comm, ncclTeamTagLsa(), blockIdx.x, multimem};
  lsaBar.sync(cta, cuda::memory_order_relaxed);

  int maxChunkElts = args->maxDynamicSmem/sizeof(AccT);

  handler.template forEachWorkNoFusion<T>(
    [&]__device__(size_t nElts, size_t nAllElts, ncclSymPtr<T> input, ncclSymPtr<T> output) {
      AccT* accum;
      ncclSymkSmemPartition(&accum, maxChunkElts);

      int chunkBytes_log2 = log2Up(nElts) + log2Up(sizeof(T));

      // Chunk size should not be larger than what host dictates and 1/2 of
      // total capacity so we enjoy some pipeline overlap. This is a soft constraint
      // because chunks which are too big can always be partially used.
      int maxChunkBytes_log2 = min(
        log2Up(maxChunkElts) + log2Up(sizeof(T)),
        handler.ginInboxRail.size_log2-1);
      chunkBytes_log2 = min(chunkBytes_log2, maxChunkBytes_log2);

      // Chunk size must not be so small that the chunk count exceeds either the
      // per peer number or total number. This is a hard constraint imposed
      // by inbox credit logic so we enforce after the max chunk size.
      int minChunkBytes_log2 = handler.ginInboxRail.size_log2 - min(
        log2Down(rail.nRanks-1) + ncclGinScratchMaxBufsPerPeer_log2,
        ncclGinScratchMaxBufs_log2);
      chunkBytes_log2 = max(chunkBytes_log2, minChunkBytes_log2);

      int nBufs_log2 = handler.ginInboxRail.size_log2 - chunkBytes_log2;

      maxChunkElts = min(maxChunkElts, (1u<<chunkBytes_log2)/(unsigned)sizeof(T));

      auto nop = []__device__(auto...) {};
      auto skeleton = [&]__device__(auto initFn, auto stepFn, auto finishFn) {
        size_t loopElts = nElts;
        ncclSymPtr<T> loopInput = input;
        ncclSymPtr<T> loopOutput = output;
        #pragma unroll 1
        while (loopElts != 0) {
          int nChunkElts = min(loopElts, (size_t)maxChunkElts);
          int nSteps = min(rail.nRanks-1, 1<<(nBufs_log2-1));
          int step = 0;
          initFn(nChunkElts, loopInput);
          do {
            nSteps = min(nSteps, rail.nRanks-1 - step);
            stepFn(step, nSteps, nChunkElts, loopInput);
            step += nSteps;
          } while (step != rail.nRanks-1);
          finishFn(nChunkElts, loopOutput);
          inbox.endRound(cta);
          loopInput += nChunkElts;
          loopOutput += nChunkElts;
          loopElts -= nChunkElts;
        }
      };

      if (stage == 0) { // !!! Pipeline Stage 0 !!!
        inbox.apportion(cta, /*subcoop=*/coopStage, /*subcoopIsNonTrivial=*/false, nBufs_log2);
        if (lsa.nRanks != 1) {
          outbox.apportion(coopStage, /*subcoop=*/coopRole, /*subcoopIsNonTrivial=*/roleIsWorker, nBufs_log2, /*deferSync=*/true);
        }

        // Generalize worker and non-worker logic with compile time BoolTag to differentiate.
        auto stage0_impl = [&](/*BoolTag<roleIsWorker>*/auto roleIsWorker_tag) {
          // Shadow runtime value with compile time value.
          constexpr bool roleIsWorker = roleIsWorker_tag.value;
          skeleton(
            /*initFn*/[&]__device__(int nChunkElts, ncclSymPtr<T> inPtr)->void {
              if (!roleIsWorker) {
                if (coopRole.thread_rank() == 0) {
                  // totalSends += rail.nRanks-1;
                  #if __CUDA_ARCH__ >= 700
                  asm volatile("red.relaxed.shared.add.s32 [%0],%1;" :: "r"((uint32_t)__cvta_generic_to_shared(&totalSends)), "r"(rail.nRanks-1) : "memory");
                  #else
                  __trap();
                  #endif
                }
              } else {
                // outbox.apportion() was told to defer sync so that we don't sync
                // the whole stage. We sync just this warp.
                coopRole.sync();
              }
            },
            /*stepFn=*/[&]__device__(int step0, int nSteps, int nChunkElts, ncclSymPtr<T> inPtr)->void {
              auto getInputOffset = [&]__device__(int step)->size_t {
                int peer = inbox.getSendPeer(step, /*step_lt_nPeers=*/true);
                int dstWorld = ncclTeamRankToTeam(world, rail, peer);
                return dstWorld*nAllElts;
              };

              if (lsa.nRanks != 1) { // No need to process data when we can send from input buf.
                if (roleIsWorker) {
                  // Wait for outbox bufs to free up. Since outbox advances with each call of this
                  // function we always index starting at 0.
                  outbox.waitBufs(coopRole, 0, nSteps);
                  coopRole.sync();
                  // Make `outbox.getBufPtr()` as cheap as possible within reduction loop.
                  auto outbox_getBufPtr = outbox.make_getBufPtr(0);
                  reduceLsaBatch(coopRole, /*nBatch=*/nSteps, nChunkElts,
                    /*dstMem=*/GMemTag(), /*dstAlignMin=*/16,
                    /*getDst=*/[&]__device__(int i)->T* {
                      return (T*)outbox_getBufPtr(i);
                    },
                    /*srcRedUc=*/red, /*srcRedMc=*/mmRed, /*srcBase=*/inPtr,
                    /*getSrcOffset=*/[&]__device__(int i)->size_t {
                      return getInputOffset(step0 + i);
                    },
                    handler.comm, multimemTag
                  );
                }
                coopStage.sync();
              }

              if (!roleIsWorker) {
                inbox.postSends(coopRole, step0, nSteps,
                  /*getPtr*/[&]__device__(int i, int peer) {
                    return lsa.nRanks == 1 ? inPtr + getInputOffset(step0 + i)
                                           : (ncclSymPtr<T>)outbox.getBuf(i);
                  },
                  /*getEltCount*/[&]__device__(int i, int peer) {
                    return nChunkElts;
                  },
                  /*getCompletion*/[&]__device__(int i, int peer) {
                    return ncclGin_CounterInc{
                      lsa.nRanks == 1 ? handler.ginCounterPerBlock + blockIdx.x
                                      : outbox.getCounter(i)
                    };
                  }
                );
              }

              if (lsa.nRanks != 1) {
                // We advance outbox with every iteration because it is only guaranteed
                // to have `nSteps` buffers. If we advanced it after the step loop it
                // would need rail.nRanks-1 buffers.
                outbox.advance(coopStage, nSteps);
              }
            },
            /*finishFn=*/nop
          );
        };

        // Instantiate stage0_impl specialized to each case of roleIsWorker.
        if (!roleIsWorker) stage0_impl(BoolTag</*roleIsWorker=*/false>());
        else stage0_impl(BoolTag</*roleIsWorker=*/true>());

      } else { // !!! Pipeline Stage 1 !!!

        // Generalize worker and non-worker logic with compile time BoolTag to differentiate.
        auto stage1_impl = [&]__device__(/*BoolTag<roleIsWorker>*/auto roleIsWorker_tag)->void {
          constexpr bool roleIsWorker = roleIsWorker_tag.value;
          inbox.apportion(cta, /*subcoop=*/coopRole, /*subcoopIsNonTrivial=*/!roleIsWorker, nBufs_log2);
          coopStage.sync();

          skeleton(
            /*initFn*/[&]__device__(int nChunkElts, ncclSymPtr<T> inPtr)->void {
              if (roleIsWorker) {
                reduceLsa(coopRole, nChunkElts,
                  /*dstMem=*/SMemTag(), /*dstAlignMin=*/16, /*dstPtr=*/accum,
                  /*srcRedUc=*/red, /*srcRedMc=*/mmRed,
                  /*srcPtr=*/inPtr + world.rank*nAllElts,
                  handler.comm, multimemTag
                );
              }
            },
            /*stepFn*/[&]__device__(int step0, int nSteps, int nChunkElts, ncclSymPtr<T> inPtr)->void {
              if (roleIsWorker) {
                inbox.waitRecvs(coopRole, step0, nSteps);
                coopRole.sync();
                // Make `inbox.getBufPtr()` as cheap as possible within reduction loop.
                auto inbox_getBufPtr = inbox.make_getBufPtr(step0);
                reduce(coopRole, red, /*inPlace=*/true, nChunkElts,
                  /*dstMem=*/SMemTag(), /*dstAlignMin=*/16, /*dst=*/accum,
                  /*nSrcs=*/nSteps, /*srcPtrCommonMask=*/16-1, /*srcPtrMasked=*/0,
                  /*getSrc=*/[&]__device__(int s)->T* {
                    return (T*)inbox_getBufPtr(s);
                  }
                );
              }
              coopStage.sync();
              if (!roleIsWorker) {
                inbox.finishRecvs(coopRole, step0, nSteps);
              }
            },
            /*finishFn*/[&]__device__(int nChunkElts, ncclSymPtr<T> outPtr)->void {
              if (roleIsWorker) {
                copy(coopRole, nChunkElts,
                  /*dstMem=*/GMemTag(), /*dst=*/outPtr.localPtr(),
                  /*srcMem=*/SMemTag(), /*src=*/accum);
                coopRole.sync(); // prevent initFn from trampling accum
              }
            }
          );
        };

        // Instantiate stage1_impl specialized to each case of roleIsWorker.
        if (!roleIsWorker) stage1_impl(BoolTag</*roleIsWorker=*/false>());
        else stage1_impl(BoolTag</*roleIsWorker=*/true>());
      }
    }
  );

  if (stage == 0) {
    if (lsa.nRanks == 1) {
      if (!roleIsWorker && coopRole.thread_rank() == 0) {
        gin.waitCounter(ncclCoopThread(), handler.ginCounterPerBlock + blockIdx.x, totalSends, 32);
      }
    } else {
      outbox.template ~ncclGinOutboxSession<ncclCoopWarpSpan>();
    }
  }

  lsaBar.sync(cta, cuda::memory_order_relaxed);
}

template<template<typename> typename Red, typename T>
__device__ __forceinline__ void ncclSymkRun_ReduceScatter_RailA2A_LsaLD(ncclSymkDevWorkArgs const* args) {
  rsAlgoHier<Red, T>(args, /*multimem=*/BoolTag<false>{});
}
template<template<typename> typename Red, typename T>
__device__ __forceinline__ void ncclSymkRun_ReduceScatter_RailA2A_LsaLDMC(ncclSymkDevWorkArgs const* args) {
  rsAlgoHier<Red, T>(args, /*multimem=*/BoolTag<true>{});
}
