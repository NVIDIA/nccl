/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "nvls_ub.h"
#include "alloc.h"
#include "comm.h"
#include "transport.h"
#include "register.h"
#include "shmutils.h"
#include "cudawrap.h"
#include "bitops.h"
#include "param.h"

#if CUDART_VERSION >= 12010

// UB carveout size: -1 auto (largest device memory in the domain, 4 GiB-rounded),
// 0 disables UB registration, > 0 is a size in bytes.
NCCL_PARAM(NvlsUbSize, "NVLS_UB_SIZE", -1);

// The two buffer roles one round resolves; also indices into the per-rank payloads below.
static constexpr int nvlsUbSend = 0;
static constexpr int nvlsUbRecv = 1;
static constexpr int nvlsUbRoles = 2;

namespace {

// What one role needs this round, derived only from the gathered eligibility payloads.
enum ubAction {
  ubActionFallback = 0, // no address every rank agrees on; the caller uses its unregistered path
  ubActionReuseRange,   // every rank already holds the same committed range
  ubActionNewRange      // every rank must allocate and bind
};

// What the domain did with one range, resolved from the gathered binds. Only ubRangeStateBound is
// committable; the rest name how the range goes back to the arena.
enum ubRangeState {
  ubRangeStateBound, // every rank bound it at the same offset
  ubRangeStatePartlyBound, // some rank bound it, some did not
  ubRangeStateUnbound, // no rank bound it
};

// What one rank knows about one role before anything is allocated or bound.
struct ubSlotEligibility {
  size_t userOffset;  // buffer address - registered base address; must agree across ranks
  size_t reqExtent;   // userOffset + this rank's collective extent; the bind must cover the cross-rank max
  size_t ucSize;      // registered extent, rounded up to the UC granularity
  size_t arenaOffset; // offset of the committed range; meaningful when committed
  bool eligible;    // passed every local pre-check
  bool committed;   // this rank already holds a committed range for this buffer
};

// Per-rank payload of the eligibility gather.
struct ubEligibility {
  struct ubSlotEligibility slot[nvlsUbRoles];
  size_t granularity;   // this rank's range alignment; ranks bind at the cross-rank max
  bool sendRecvAliased; // send and recv resolve to the same registration record
  bool hasNewRetired;   // this rank retired a range since its last reclaim exchange
  int retiredCount;     // retired ranges held by this rank; sizes the reclaim exchange
};

// Per-rank payload of the bind-result gather. The arena offset must be identical on
// every rank (the arenas are bit-identical); carrying it verifies that invariant so a
// divergence is caught here instead of corrupting data. SIZE_MAX = no allocation.
struct ubBindResult {
  size_t offset[nvlsUbRoles];
  bool bound[nvlsUbRoles]; // this rank's bind took
};

// What one eligibility exchange settled for the round, identical on every rank.
struct ubRoundPlan {
  enum ubAction action[nvlsUbRoles];
  size_t bindSize[nvlsUbRoles]; // per ubActionNewRange role: the extent every rank can bind
  int rangeOwner[nvlsUbRoles];  // the role whose range this role uses; an aliased recv uses send's
};

// One collective registration attempt: the send/recv pair under resolution, the plan the
// eligibility exchange settles, the gathered payloads, and the addresses the attempt resolves.
struct ubRegState {
  const void* buff[nvlsUbRoles];
  struct ncclReg* reg[nvlsUbRoles]; // registration records, or NULL per absent role
  size_t size[nvlsUbRoles];         // collective extents
  CUdeviceptr mcAddr[nvlsUbRoles];  // resolved multicast addresses; 0 = fallback
  struct ubRoundPlan plan;
  struct ubEligibility* allEligibility; // localRanks of gathered eligibility payloads
  struct ubBindResult* allBindResults;  // localRanks of gathered bind payloads
};

} // namespace

size_t ncclNvlsUbScratchTypeSize(void) {
  // The reclaim exchange chunks over this scratch, so it must fit at least one entry.
  return std::max(std::max(sizeof(struct ubEligibility), sizeof(struct ubBindResult)), sizeof(struct ncclMcRetiredReg));
}

bool ncclNvlsUbEnabled(struct ncclComm* comm) {
  return comm->nvlsResources->ubEnabled;
}

size_t ncclNvlsUbSize(struct ncclComm* comm) {
  int64_t request = ncclParamNvlsUbSize();
  size_t maxDeviceMem = 0;

  // nvlsRegSupport (0 under MNNVL) is a pure function of gathered peer info, so the skip
  // is uniform across ranks.
  if (!comm->nvlsRegSupport || request == 0) return 0;
  if (request > 0) return (size_t)request;
  // Auto: cover the largest device memory in the NVLS domain.
  for (int i = 0; i < comm->localRanks; i++)
    maxDeviceMem = std::max(maxDeviceMem, comm->peerInfo[comm->localRankToRank[i]].totalGlobalMem);
  return alignUp(maxDeviceMem, (size_t)4 << 30);
}

// The alignment a range must satisfy on this rank: the larger of the UC and MC bind
// granularities (both powers of two, so the larger is a multiple of the smaller).
static size_t ubRangeAlignment(const struct ncclMcArena* arena) {
  return std::max(arena->ucGranularity, arena->partition.minGranularity);
}

// The multicast address a range resolves for buff. The offset term is the userOffset the
// ranks gathered and agreed on.
static CUdeviceptr ubMcAddress(const struct ncclMcArenaReg* range, const struct ncclReg* reg, const void* buff) {
  return range->mcBaseAddress + ((uintptr_t)buff - reg->begAddr);
}

// Whether the registered extent satisfies every documented cuMulticastBindAddr
// precondition.
static bool ubRegistrationBindable(struct ncclComm* comm, const void* buff, const struct ncclReg* reg) {
  const struct ncclMcArena* arena = &comm->nvlsResources->ubArena;
  // ALLOWED_HANDLE_TYPES is a 64-bit bitmask; a narrower local overruns its stack slot.
  unsigned long long handleTypes = 0;

  if (CUPFN(cuPointerGetAttribute((void*)&handleTypes, CU_POINTER_ATTRIBUTE_ALLOWED_HANDLE_TYPES, (CUdeviceptr)buff)) !=
      CUDA_SUCCESS)
    return false;
  if ((handleTypes & ncclCuMemHandleType) == 0) return false;
  if (reg->begAddr % ubRangeAlignment(arena) != 0) return false;
  // The bind covers the UC-rounded extent, so backing must exist up to the rounding,
  // which an allocation created at a finer granularity can lack.
  return ncclMcAddrBindable((const void*)reg->begAddr, alignUp(reg->endAddr - reg->begAddr, arena->ucGranularity));
}

bool ncclNvlsUbEligible(struct ncclComm* comm, const void* buff, struct ncclReg* reg) {
  if (reg->state & NVLS_REG_NO_SUPPORT) return false;
  // A committed range is bound, so it passed at commit time.
  if (reg->state & (NVLS_REG_COMPLETE | NVLS_REG_POSSIBLE)) return true;
  if (!ubRegistrationBindable(comm, buff, reg)) {
    reg->state |= NVLS_REG_NO_SUPPORT;
    return false;
  }
  reg->state |= NVLS_REG_POSSIBLE;
  return true;
}

// Pick one role's action from the gathered payloads alone, so all ranks pick the same one
// even when the collective's arguments diverge. outBindSize is the cross-rank minimum of
// the UC-rounded sizes; alignment is the cross-rank maximum of the per-rank alignments.
static enum ubAction ubActionFromEligibility(const struct ubEligibility* allEligibility, int localRanks, int role,
                                             size_t alignment, size_t* outBindSize) {
  const struct ubSlotEligibility* rank0Slot = &allEligibility[0].slot[role];
  bool allCommitted = true, anyCommitted = false, offsetsAgree = true;
  size_t bindSize = SIZE_MAX;
  size_t reqExtent = 0;

  for (int i = 0; i < localRanks; i++) {
    const struct ubSlotEligibility* slot = &allEligibility[i].slot[role];
    if (!slot->eligible) return ubActionFallback;
    // The multicast address is a shared base plus this offset, so it must agree.
    if (slot->userOffset != rank0Slot->userOffset) return ubActionFallback;
    allCommitted &= slot->committed;
    anyCommitted |= slot->committed;
    // arenaOffset only means anything on committed slots; consulted only when allCommitted.
    offsetsAgree &= (!slot->committed || slot->arenaOffset == rank0Slot->arenaOffset);
    reqExtent = std::max(reqExtent, slot->reqExtent);
    bindSize = std::min(bindSize, slot->ucSize);
  }

  // Round down, never up: rounding up could bind past the memory backing a rank's
  // registration. The extent check below turns any loss into a fallback.
  bindSize = alignDown(bindSize, alignment);
  // bindSize is a cross-rank minimum, so a rank's buffer can extend past the bound extent.
  if (reqExtent > bindSize) return ubActionFallback;

  if (allCommitted) return offsetsAgree ? ubActionReuseRange : ubActionFallback;
  // A mix of committed and uncommitted ranks has no address every rank agrees on.
  if (anyCommitted || bindSize == 0) return ubActionFallback;
  *outBindSize = bindSize;
  return ubActionNewRange;
}

// Round one of the protocol: gather what every rank knows per role and settle the plan.
// Pre-check failures travel as payload, so a rank reaches this gather regardless of its
// local verdicts. Reuse addresses land in state->mcAddr.
static ncclResult_t ubEligibilityExchange(struct ncclComm* comm, struct ubRegState* state) {
  struct ncclNvlsSharedRes* resources = comm->nvlsResources;
  const void* const* buff = state->buff;
  struct ncclReg* const* reg = state->reg;
  const size_t* size = state->size;
  CUdeviceptr* mcAddr = state->mcAddr;
  struct ubEligibility* allEligibility = state->allEligibility;
  struct ubRoundPlan* plan = &state->plan;
  struct ubEligibility myEligibility = {};
  size_t alignment = 0; // the cross-rank maximum every range must satisfy

  // Callers pass only records that passed ncclNvlsUbEligible, so a present record is an
  // eligible slot.
  for (int r = 0; r < nvlsUbRoles; r++) {
    if (buff[r] == nullptr || reg[r] == nullptr) continue;
    struct ubSlotEligibility* slot = &myEligibility.slot[r];
    slot->userOffset = (uintptr_t)buff[r] - reg[r]->begAddr;
    slot->reqExtent = slot->userOffset + size[r];
    slot->ucSize = alignUp(reg[r]->endAddr - reg[r]->begAddr, resources->ubArena.ucGranularity);
    slot->committed = (reg[r]->state & NVLS_REG_COMPLETE) != 0;
    if (slot->committed) slot->arenaOffset = reg[r]->nvlsUbReg->offset;
    slot->eligible = true;
  }
  myEligibility.granularity = ubRangeAlignment(&resources->ubArena);
  myEligibility.sendRecvAliased = (reg[nvlsUbSend] != nullptr && reg[nvlsUbSend] == reg[nvlsUbRecv]);
  myEligibility.retiredCount = resources->ubArena.retiredQueue.nElems;
  myEligibility.hasNewRetired = resources->ubArena.hasNewRetiredSinceLastReclaim;

  // Fits the per-rank scratch reserved at init (ncclNvlsUbScratchTypeSize).
  NCCLCHECK(ncclShmemAllgather(comm, &resources->nvlsShmem, &myEligibility, allEligibility, sizeof(myEligibility)));

  // Aliasing decides whether one range serves both roles, so ranks must agree on it; a
  // mismatch resolves as a converged fallback (the zeroed plan).
  bool sendRecvAliased = allEligibility[0].sendRecvAliased;
  for (int i = 0; i < comm->localRanks; i++) {
    if (allEligibility[i].sendRecvAliased != sendRecvAliased) return ncclSuccess;
    alignment = std::max(alignment, allEligibility[i].granularity);
  }
  for (int r = 0; r < nvlsUbRoles; r++) {
    plan->rangeOwner[r] = r;
    plan->action[r] = ubActionFromEligibility(allEligibility, comm->localRanks, r, alignment, &plan->bindSize[r]);
    if (plan->action[r] == ubActionReuseRange) mcAddr[r] = ubMcAddress(reg[r]->nvlsUbReg, reg[r], buff[r]);
  }
  // Aliased roles share one range, allocated by send.
  if (sendRecvAliased && plan->action[nvlsUbSend] == ubActionNewRange && plan->action[nvlsUbRecv] == ubActionNewRange)
    plan->rangeOwner[nvlsUbRecv] = nvlsUbSend;
  return ncclSuccess;
}

// Reclaim is worth its exchange only when every rank holds a retired range and something was
// retired since the last one. Inputs come from the eligibility gather, so the answer is uniform.
static bool ubTriggerReclaim(const struct ubEligibility* allEligibility, int localRanks) {
  int fewestRetired = allEligibility[0].retiredCount;
  bool anyRetiredSinceLastReclaim = false;

  for (int i = 0; i < localRanks; i++) {
    fewestRetired = std::min(fewestRetired, allEligibility[i].retiredCount);
    anyRetiredSinceLastReclaim |= allEligibility[i].hasNewRetired;
  }
  return fewestRetired > 0 && anyRetiredSinceLastReclaim;
}

// Gather every rank's retired list into one rank-major array of *outEntriesPerRank entries per
// rank, chunked over the shmem scratch. The round count is a pure function of the gathered
// counts. The caller owns *outAllRetired.
static ncclResult_t ubRetiredExchange(struct ncclComm* comm, const struct ubEligibility* allEligibility,
                                      int* outEntriesPerRank, struct ncclMcRetiredReg** outAllRetired) {
  struct ncclMcArena* arena = &comm->nvlsResources->ubArena;
  int localRanks = comm->localRanks;
  int entriesPerRound = (int)(comm->nvlsResources->nvlsShmem.maxTypeSize / sizeof(struct ncclMcRetiredReg));
  struct ncclMcRetiredReg* myRetired = nullptr;    // this rank's list, offset-sorted
  struct ncclMcRetiredReg* roundChunk = nullptr;   // one round's worth, rank-major
  struct ncclMcRetiredReg* allRetired = nullptr;   // every rank's list, rank-major
  int mostRetired = 0;
  int exchangeRounds, entriesPerRank;
  ncclResult_t ret = ncclSuccess;

  for (int i = 0; i < localRanks; i++) mostRetired = std::max(mostRetired, allEligibility[i].retiredCount);
  exchangeRounds = DIVUP(mostRetired, entriesPerRound);
  entriesPerRank = exchangeRounds * entriesPerRound;

  NCCLCHECKGOTO(ncclCalloc(&myRetired, entriesPerRank), ret, exit);
  NCCLCHECKGOTO(ncclCalloc(&roundChunk, (size_t)localRanks * entriesPerRound), ret, exit);
  NCCLCHECKGOTO(ncclCalloc(&allRetired, (size_t)localRanks * entriesPerRank), ret, exit);

  // The queue is offset-sorted, so the snapshot is too; peers read only the retiredCount
  // entries this rank reported in the eligibility gather.
  ncclMcArenaRetiredSnapshot(arena, myRetired, entriesPerRank);

  for (int r = 0; r < exchangeRounds; r++) {
    NCCLCHECKGOTO(ncclShmemAllgather(comm, &comm->nvlsResources->nvlsShmem, myRetired + r * entriesPerRound, roundChunk,
                                     entriesPerRound * sizeof(*myRetired)),
                  ret, exit);
    for (int i = 0; i < localRanks; i++)
      memcpy(allRetired + i * entriesPerRank + r * entriesPerRound, roundChunk + i * entriesPerRound,
             entriesPerRound * sizeof(*myRetired));
  }

exit:
  if (ret != ncclSuccess) {
    free(allRetired);
    allRetired = nullptr;
  }
  *outAllRetired = allRetired;
  *outEntriesPerRank = entriesPerRank;
  free(myRetired);
  free(roundChunk);
  return ret;
}

// Free every range that all ranks have retired. Collective over the local ranks.
static ncclResult_t ubReclaimRetiredRanges(struct ncclComm* comm, const struct ubEligibility* allEligibility) {
  struct ncclMcArena* arena = &comm->nvlsResources->ubArena;
  struct ncclMcRetiredReg* allRetired = nullptr;    // every rank's retired list, rank-major
  struct ncclMcRetiredReg* sharedRetired = nullptr; // ranges every rank retired; the driver's slice, rewritten in place
  int* peerCursor = nullptr; // per-rank position in its gathered list
  int entriesPerRank = 0;
  int driverRank = 0; // The driver is based on the shortest reclaim list.
  int nShared = 0;
  size_t freedBytes = 0;
  ncclResult_t ret = ncclSuccess;

  for (int i = 1; i < comm->localRanks; i++)
    if (allEligibility[i].retiredCount < allEligibility[driverRank].retiredCount) driverRank = i;

  NCCLCHECKGOTO(ncclCalloc(&peerCursor, comm->localRanks), ret, exit);
  NCCLCHECKGOTO(ubRetiredExchange(comm, allEligibility, &entriesPerRank, &allRetired), ret, exit);
  // The shared set lands in the driver's own slice.
  sharedRetired = allRetired + (size_t)driverRank * entriesPerRank;

  // Copy the ranges present in every rank's retired list into sharedRetired, offset-sorted. The
  // shortest list drives: the peers' cursors only move forward, and every shared range is in
  // every list, so the walk ends once any peer's list is exhausted.
  for (int j = 0; j < allEligibility[driverRank].retiredCount; j++) {
    const struct ncclMcRetiredReg* cand = &allRetired[driverRank * entriesPerRank + j];
    bool onEveryRank = true;
    for (int i = 0; i < comm->localRanks && onEveryRank; i++) {
      const struct ncclMcRetiredReg* peer;
      if (i == driverRank) continue;

      while (peerCursor[i] < allEligibility[i].retiredCount &&
             allRetired[i * entriesPerRank + peerCursor[i]].offset < cand->offset)
        peerCursor[i]++;
      if (peerCursor[i] >= allEligibility[i].retiredCount) goto exit; // no possible matches left

      peer = &allRetired[i * entriesPerRank + peerCursor[i]];
      if (peer->offset != cand->offset) {
        onEveryRank = false;
      } else if (peer->bindSize != cand->bindSize) {
        // Cannot happen: an offset names one commit event. Skip to keep the arenas identical.
        WARN("rank %d NVLS UB retired range offset %zu has size %zu on rank %d but %zu on rank %d", comm->rank,
             cand->offset, cand->bindSize, comm->localRankToRank[driverRank], peer->bindSize, comm->localRankToRank[i]);
        onEveryRank = false;
      }
    }
    if (onEveryRank) sharedRetired[nShared++] = *cand;
  }

exit:
  // The shared set is the same on every rank, so the arenas stay identical. Reclaim runs on
  // every exit path, including an empty set: it re-arms the gate this exchange consumed, so
  // there is no re-fire until another retirement on some rank.
  NCCLCHECKIGNORE(ncclMcArenaReclaim(arena, sharedRetired, nShared, &freedBytes), ret);
  INFO(NCCL_REG, "rank %d NVLS UB reclaimed %d of %d retired arena ranges (%zu bytes)", comm->rank, nShared,
       allEligibility[comm->localRank].retiredCount, freedBytes);
  free(allRetired);
  free(peerCursor);
  return ret;
}

// Count what the domain did with one role's range, from the gathered binds alone, so every
// rank reaches the same state.
static enum ubRangeState ubRangeStateFromBinds(struct ncclComm* comm, const struct ubBindResult* allBindResults,
                                               int role) {
  bool everyRankBound = true, someRankBound = false, offsetsAgree = true;

  for (int i = 0; i < comm->localRanks; i++) {
    everyRankBound &= allBindResults[i].bound[role];
    someRankBound |= allBindResults[i].bound[role];
    // The arenas are bit-identical, so the offsets must be too; carrying the offset catches a
    // divergence here instead of letting it corrupt data.
    if (allBindResults[i].offset[role] != allBindResults[0].offset[role]) {
      WARN("rank %d NVLS UB arena divergence: rank %d allocated offset %zu, rank %d allocated offset %zu", comm->rank,
           comm->localRankToRank[0], allBindResults[0].offset[role], comm->localRankToRank[i],
           allBindResults[i].offset[role]);
      offsetsAgree = false;
    }
  }
  if (everyRankBound && offsetsAgree) return ubRangeStateBound;
  return someRankBound ? ubRangeStatePartlyBound : ubRangeStateUnbound;
}

// Round two of the protocol: reserve and bind the ranges the plan calls for, tell every peer
// how it went, then either commit every range to its ncclReg or dispose of every one back to
// the arena. Commit addresses land in state->mcAddr.
static ncclResult_t ubBindNewRanges(struct ncclComm* comm, struct ubRegState* state) {
  ncclResult_t ret = ncclSuccess;
  struct ncclNvlsSharedRes* resources = comm->nvlsResources;
  struct ncclMcArena* arena = &resources->ubArena;
  const void* const* buff = state->buff;
  struct ncclReg* const* reg = state->reg;
  CUdeviceptr* mcAddr = state->mcAddr;
  struct ubBindResult* allBindResults = state->allBindResults;
  const struct ubRoundPlan* plan = &state->plan;
  struct ncclMcArenaReg* newArenaRegs[nvlsUbRoles] = {}; // this rank's record per role
  enum ncclMcBindStatus bindStatus[nvlsUbRoles];
  struct ubBindResult myBindResult = {};
  const void* userBase[nvlsUbRoles] = {}; // the registration base each role binds, or NULL
  size_t reqBindSize[nvlsUbRoles] = {}; // the extent each role asks the arena for, 0 if none
  enum ubRangeState rangeState[nvlsUbRoles] = {ubRangeStatePartlyBound, ubRangeStatePartlyBound};

  // Allocate and bind ranges.
  for (int r = 0; r < nvlsUbRoles; r++) {
    if (plan->action[r] == ubActionNewRange && plan->rangeOwner[r] == r) {
      reqBindSize[r] = plan->bindSize[r];
      userBase[r] = (const void*)reg[r]->begAddr;
    }
    myBindResult.offset[r] = SIZE_MAX;
  }
  if (ncclMcArenaRegister(arena, nvlsUbRoles, userBase, reqBindSize, newArenaRegs, bindStatus) != ncclSuccess)
    INFO(NCCL_REG, "rank %d NVLS UB arena registration of %zu+%zu bytes failed, using internal buffer path", comm->rank,
         reqBindSize[nvlsUbSend], reqBindSize[nvlsUbRecv]);

  // Collect and exchange local bind results with peers.
  for (int r = 0; r < nvlsUbRoles; r++) {
    if (newArenaRegs[r] != nullptr) {
      myBindResult.offset[r] = newArenaRegs[r]->offset;
      myBindResult.bound[r] = (bindStatus[r] == ncclMcBindStatusOk);
    }
    // Store bind no support, so we don't need to check again next time.
    if (bindStatus[r] == ncclMcBindStatusNoSupport) reg[r]->state |= NVLS_REG_NO_SUPPORT;
  }
  NCCLCHECKGOTO(ncclShmemAllgather(comm, &resources->nvlsShmem, &myBindResult, allBindResults, sizeof(myBindResult)),
                ret, dispose);

  // Check for consistent state across all ranks before proceeding.
  // All ranks must agree that the send/recv buffers are new registrations or previously registered.
  // Otherwise, rollback registrations and fallback.
  for (int r = 0; r < nvlsUbRoles; r++) {
    rangeState[r] = (plan->action[r] == ubActionNewRange) ?
                      ubRangeStateFromBinds(comm, allBindResults, plan->rangeOwner[r]) :
                      ubRangeStateBound;
    if (rangeState[r] != ubRangeStateBound) goto dispose;
  }

  for (int r = 0; r < nvlsUbRoles; r++) {
    if (plan->action[r] != ubActionNewRange) continue;
    // A borrowing role's record is stored once, by its owner, through the ncclReg they share.
    if (newArenaRegs[r] != nullptr) {
      reg[r]->nvlsUbReg = newArenaRegs[r];
      reg[r]->state |= NVLS_REG_COMPLETE;
    }
    mcAddr[r] = ubMcAddress(newArenaRegs[plan->rangeOwner[r]], reg[r], buff[r]);
  }
  return ret;

dispose:
  // The state is gathered (or a failed gather left the retire default), so every rank mutates
  // its arena identically here: an offset no rank bound goes straight back, and anything else
  // recycles through reclaim. A bind refusal is a property of the buffer, not the offset, and
  // the NVLS_REG_NO_SUPPORT latch stops the refused buffer from consuming another range.
  for (int r = 0; r < nvlsUbRoles; r++) {
    if (newArenaRegs[r] == nullptr) continue;
    if (rangeState[r] == ubRangeStateUnbound) NCCLCHECKIGNORE(ncclMcArenaDeregister(arena, newArenaRegs[r]), ret);
    else ncclMcArenaRetireReg(arena, newArenaRegs[r]);
  }
  return ret;
}

// UB registration is a converged protocol over the NVLS domain's local ranks:
//   round one:  gather per-rank eligibility, settle a plan (reuse|newRange|fallback)
//   reclaim:    if triggered, gather the retired lists and free the intersection
//   round two:  reserve + bind new ranges, gather the verdicts, commit all or dispose all
// Every decision is a pure function of gathered payloads, so all ranks act identically and
// the arena's ncclSpace stays bit-identical across ranks. Local failures travel as payload;
// a rank reaches every gather its peers reach.
ncclResult_t ncclNvlsUbRegister(struct ncclComm* comm, const void* sendbuff, void* recvbuff, size_t sendbuffSize,
                                size_t recvbuffSize, struct ncclReg* sendReg, struct ncclReg* recvReg,
                                int* outRegBufUsed, void** outRegBufSend, void** outRegBufRecv) {
  int localRanks = comm->localRanks;
  struct ubRegState state = {{sendbuff, recvbuff}, {sendReg, recvReg}, {sendbuffSize, recvbuffSize}};
  bool anyNewRange = false;
  bool everyRoleResolved = true;
  ncclResult_t ret = ncclSuccess;

  // Callers may read these without checking the result, so clear them up front.
  *outRegBufUsed = 0;
  *outRegBufSend = nullptr;
  *outRegBufRecv = nullptr;

  NCCLCHECKGOTO(ncclCalloc(&state.allEligibility, localRanks), ret, exit);
  NCCLCHECKGOTO(ubEligibilityExchange(comm, &state), ret, exit);

  for (int r = 0; r < nvlsUbRoles; r++) {
    anyNewRange |= (state.plan.action[r] == ubActionNewRange);
    // Bind only if every passed buffer resolved: a committed range is never reclaimed.
    everyRoleResolved &= (state.buff[r] == nullptr || state.plan.action[r] != ubActionFallback);
  }
  // Reuse and fallback are settled by the exchange; only a new range needs more.
  if (anyNewRange && everyRoleResolved) {
    // Reclaiming what every rank retired is itself collective. A failure only leaves space
    // unrecovered, so it does not stop the round.
    if (ubTriggerReclaim(state.allEligibility, localRanks)) (void)ubReclaimRetiredRanges(comm, state.allEligibility);
    NCCLCHECKGOTO(ncclCalloc(&state.allBindResults, localRanks), ret, exit);
    NCCLCHECKGOTO(ubBindNewRanges(comm, &state), ret, exit);
  }

  *outRegBufUsed =
    ((sendbuff == nullptr || state.mcAddr[nvlsUbSend]) && (recvbuff == nullptr || state.mcAddr[nvlsUbRecv])) ? 1 : 0;
  if (*outRegBufUsed) {
    *outRegBufSend = (void*)state.mcAddr[nvlsUbSend];
    *outRegBufRecv = (void*)state.mcAddr[nvlsUbRecv];
  }
  INFO(NCCL_REG, "rank %d NVLS UB %s sendbuff %p size %zu -> %p, recvbuff %p size %zu -> %p", comm->rank,
       *outRegBufUsed ? "registered" : "fell back for", sendbuff, sendbuffSize, (void*)state.mcAddr[nvlsUbSend],
       recvbuff, recvbuffSize, (void*)state.mcAddr[nvlsUbRecv]);
exit:
  // On error the outputs keep their cleared values, so the caller falls back.
  free(state.allEligibility);
  free(state.allBindResults);
  return ret;
}

ncclResult_t ncclNvlsUbDeregister(struct ncclComm* comm, struct ncclReg* reg) {
  struct ncclNvlsSharedRes* resources = comm->nvlsResources;
  size_t offset, bindSize;

  if ((reg->state & NVLS_REG_COMPLETE) == 0) return ncclSuccess;
  offset = reg->nvlsUbReg->offset;
  bindSize = reg->nvlsUbReg->bindSize;
  // The unbind can fail when the user freed the buffer first; the binding is gone either
  // way, so the range still recycles.
  ncclMcArenaRetireReg(&resources->ubArena, reg->nvlsUbReg);
  reg->state &= ~(uint32_t)NVLS_REG_COMPLETE;
  reg->nvlsUbReg = NULL;
  INFO(NCCL_NVLS, "rank %d - NVLS deregistered buffer at arena offset %zu size %zu", comm->rank, offset, bindSize);
  return ncclSuccess;
}

#endif // CUDART_VERSION >= 12010
