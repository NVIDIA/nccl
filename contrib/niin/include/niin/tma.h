/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

// TMA (Tensor Memory Accelerator) backed RMA for NIIN.
//
// Mirrors NVSHMEM's TMA feature: an application hands a slice of its CTA shared
// memory to the runtime with nvshmemx_give_smem(), and NIIN's LSA (NVLink)
// put/get paths then move data with cp.async.bulk instead of vectorized
// load/store. Three routes exist, picked from where the local buffer lives:
//
//   smem -> peer gmem   direct cp.async.bulk, no staging (puts)
//   peer gmem -> smem   direct cp.async.bulk with an mbarrier (gets)
//   gmem <-> peer gmem  staged through the registered shared-memory tile
//
// Every route is opportunistic: alignment, size, registration, and CTA-shape
// constraints that TMA cannot meet make the helper return -1, and the caller
// falls back to the regular load/store path. That keeps the put/get contract
// unchanged regardless of whether TMA is available.
//
// TMA requires sm_90 or newer. On older architectures every helper here
// compiles to a no-op that returns -1.

#ifndef NIIN_TMA_H_
#define NIIN_TMA_H_

#include "niin/context.h"

// Threadgroup scope for the TMA helpers. Mirrors NVSHMEM's threadgroup_t for
// the scopes NIIN's device API exposes.
enum niinTmaScope {
  NIIN_TMA_THREAD = 0,
  NIIN_TMA_WARP   = 1,
  NIIN_TMA_BLOCK  = 2
};

__host__ __device__ __forceinline__ constexpr bool niin_tma_is_16b_aligned(size_t value) {
  return (value & (size_t)0xF) == 0;
}

__host__ __device__ __forceinline__ constexpr size_t niin_tma_align_down_16(size_t value) {
  return value & ~(size_t)0xF;
}

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900

// ---------------------------------------------------------------------------
// PTX primitives
// ---------------------------------------------------------------------------

// Elect one leader among the active threads of the calling warp. Returns true
// for exactly one thread. Callers must be warp-converged.
__device__ __forceinline__ bool niin_tma_elect_warp() {
  uint32_t isLeader;
  asm volatile(R"({
.reg .pred elect_p;
.reg .u32 elect_id;
elect.sync elect_id|elect_p, 0xffffffff;
selp.u32 %0, 1, 0, elect_p;
})"
               : "=r"(isLeader));
  return isLeader != 0;
}

// Elect exactly one thread across an entire CTA. Call from all threads.
//
// The __shfl_sync broadcast makes the warp id a compiler-visible warp-uniform
// value, which keeps the compiler from peeling the elect over active threads.
// A plain `threadIdx.x == 0` test does not give the compiler that information.
__device__ __forceinline__ bool niin_tma_block_is_elected() {
  unsigned int tid = threadIdx.x + threadIdx.y * blockDim.x +
                     threadIdx.z * blockDim.x * blockDim.y;
  unsigned int warpId = tid / warpSize;
  unsigned int uniformWarpId = __shfl_sync(0xffffffff, warpId, 0);
  return (uniformWarpId == 0) && niin_tma_elect_warp();
}

// Convert a generic pointer into a 32-bit shared-memory-space address.
__device__ __forceinline__ unsigned int niin_tma_cvta_to_shared(const void* ptr) {
  unsigned int smemAddr;
  asm(R"({
.reg .u64 smem_u64;
cvta.to.shared.u64 smem_u64, %1;
cvt.u32.u64 %0, smem_u64;
})"
      : "=r"(smemAddr)
      : "l"((uint64_t)(uintptr_t)ptr));
  return smemAddr;
}

__device__ __forceinline__ void niin_tma_bulk_shared_to_global(
    void* gmemDst, unsigned int smemAddr, uint32_t bytes) {
  asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
               :
               : "l"((uint64_t)(uintptr_t)gmemDst), "r"(smemAddr), "r"(bytes)
               : "memory");
}

__device__ __forceinline__ void niin_tma_bulk_global_to_shared(
    void* smemDst, const void* gmemSrc, uint32_t bytes, uint64_t* mbar) {
  unsigned int dstAddr = niin_tma_cvta_to_shared(smemDst);
  unsigned int mbarAddr = niin_tma_cvta_to_shared(mbar);
  asm volatile(
      "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes "
      "[%0], [%1], %2, [%3];"
      :
      : "r"(dstAddr), "l"((uint64_t)(uintptr_t)gmemSrc), "r"(bytes), "r"(mbarAddr)
      : "memory");
}

__device__ __forceinline__ void niin_tma_bulk_commit_group() {
  asm volatile("cp.async.bulk.commit_group;" ::: "memory");
}

// Wait until the bulk group has finished reading shared memory. The remote
// write may still be in flight; this only makes the source tile reusable.
__device__ __forceinline__ void niin_tma_bulk_wait_group_read_0() {
  asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory");
}

// Wait for full completion: shared-memory read and global write both done.
__device__ __forceinline__ void niin_tma_bulk_wait_group_0() {
  asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
}

// Order generic shared-memory stores against reads by the async proxy.
__device__ __forceinline__ void niin_tma_fence_proxy_async_shared_cta() {
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}

// mbarrier helpers for pairing with an inbound (gmem -> smem) cp.async.bulk.
// The barrier is an 8-byte shared-memory object with an arrive count of 1, so
// these are safe as long as the calling thread is the only arriver.
__device__ __forceinline__ void niin_tma_mbarrier_init(uint64_t* mbar) {
  unsigned int addr = niin_tma_cvta_to_shared(mbar);
  asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(addr));
}

__device__ __forceinline__ void niin_tma_mbarrier_arrive_expect_tx(uint64_t* mbar,
                                                                   uint32_t bytes) {
  unsigned int addr = niin_tma_cvta_to_shared(mbar);
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(addr),
               "r"(bytes));
}

__device__ __forceinline__ void niin_tma_mbarrier_try_wait(uint64_t* mbar, int phase) {
  unsigned int addr = niin_tma_cvta_to_shared(mbar);
  asm volatile(R"({
.reg .pred p;
niinWaitL_%=:
mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;
@!p bra niinWaitL_%=;
})" ::"r"(addr),
               "r"(phase)
               : "memory");
}

// Manually add to a barrier's transaction counter, for barriers that are not
// released by a cp.async.bulk completion.
__device__ __forceinline__ void niin_tma_mbarrier_complete_tx(uint64_t* mbar,
                                                              uint32_t txCount) {
  unsigned int addr = niin_tma_cvta_to_shared(mbar);
  asm volatile("mbarrier.complete_tx.relaxed.cta.shared::cta.b64 [%0], %1;" ::"r"(addr),
               "r"(txCount)
               : "memory");
}

#else  // pre-Hopper, or the host compilation pass

// No-op stubs for the primitives an application may call around an
// smem-sourced put, so that code compiles unchanged on architectures where TMA
// is unavailable and every RMA takes the load/store path.
__device__ __forceinline__ void niin_tma_bulk_commit_group() {}
__device__ __forceinline__ void niin_tma_bulk_wait_group_read_0() {}
__device__ __forceinline__ void niin_tma_bulk_wait_group_0() {}
__device__ __forceinline__ void niin_tma_fence_proxy_async_shared_cta() {}

#endif  // __CUDA_ARCH__ >= 900

// ---------------------------------------------------------------------------
// Per-CTA shared-memory registration
// ---------------------------------------------------------------------------

// These three stay outside any __CUDA_ARCH__ wrapper. nvcc parses __global__
// and __device__ bodies during the host pass too, so a helper that only exists
// when __CUDA_ARCH__ is defined cannot be named from device code at all. Guard
// the body, never the declaration.

// Linear CTA id within the grid, used to index the registration table.
__device__ __forceinline__ size_t niin_tma_block_id() {
#ifdef __CUDA_ARCH__
  return (size_t)blockIdx.x + (size_t)blockIdx.y * gridDim.x +
         (size_t)blockIdx.z * gridDim.x * gridDim.y;
#else
  return 0;
#endif
}

// True when this CTA has registered shared memory and TMA is not disabled.
// Gates the TMA dispatch in put, get, fence, and quiet.
__device__ __forceinline__ bool niin_tma_smem_registered() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  if (niin_tma_policy() == NVSHMEMX_TMA_DISABLE) return false;
  niinContext const& ctx = niin_ctx();
  uintptr_t* bases = ctx.tmaSmemBases;
  size_t blockId = niin_tma_block_id();
  return bases != nullptr && blockId < ctx.tmaSmemBasesLen && bases[blockId] != 0;
#else
  return false;
#endif
}

// Shared-memory size every CTA passed to give_smem (the API requires it to be
// uniform across the grid, so a single scalar covers the whole table).
__device__ __forceinline__ size_t niin_tma_smem_size() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  size_t* p = niin_ctx().tmaSmemSize;
  return p != nullptr ? *p : 0;
#else
  return 0;
#endif
}

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900

// Barrier slots live in the static carve at the base of the registered buffer:
// NIIN_TMA_NUM_BARRIER_SLOTS slots of 16 bytes, each holding one 8-byte
// mbarrier plus padding so the slot stride matches cp.async.bulk alignment.
//
// Slot assignment (must not collide between concurrently running TMA paths):
//   0       single-issuer staging mbarrier (thread and warp scope)
//   0, 1    block staging ready_bar[0], ready_bar[1]
//   2, 3    block staging done_bar[0], done_bar[1]
//   4       direct gmem -> smem get mbarrier
//   5..31   reserved
__device__ __forceinline__ uint64_t* niin_tma_barrier_slot(uintptr_t smemBase, int slot) {
  return reinterpret_cast<uint64_t*>(smemBase + (uintptr_t)slot * 16);
}

__device__ __forceinline__ char* niin_tma_data_buffer(uintptr_t smemBase) {
  return reinterpret_cast<char*>(smemBase + (uintptr_t)NIIN_SMEM_DATA_REGION_OFFSET);
}

// Elect the single thread that issues a transfer for the given scope. Thread
// scope has no election: every caller issues its own transfer.
template <int SCOPE>
__device__ __forceinline__ bool niin_tma_is_leader() {
  if constexpr (SCOPE == NIIN_TMA_THREAD) return true;
  else if constexpr (SCOPE == NIIN_TMA_WARP) return niin_tma_elect_warp();
  else return niin_tma_block_is_elected();
}

// Re-converge the threadgroup after the leader has issued its transfer.
template <int SCOPE>
__device__ __forceinline__ void niin_tma_group_sync() {
  if constexpr (SCOPE == NIIN_TMA_BLOCK) __syncthreads();
  else if constexpr (SCOPE == NIIN_TMA_WARP) __syncwarp();
}

#endif  // __CUDA_ARCH__ >= 900

// ---------------------------------------------------------------------------
// nvshmemx_give_smem / nvshmemx_release_smem
// ---------------------------------------------------------------------------

// nvshmemx_give_smem: lend a block of shared memory to NIIN for TMA transfers.
//
// Call from every thread of every CTA in the grid, once per kernel launch,
// before any TMA-backed put or get, and follow it with __syncthreads(). All
// CTAs must pass the same size. A CTA that skips the call falls back to
// load/store for every RMA it issues in that kernel.
//
// The __syncthreads() is required, not advisory: only one elected thread
// publishes the registration, and a warp- or block-scoped RMA needs every
// thread in the group to agree on whether TMA is in play.
//
// Every CTA that calls give_smem must call nvshmemx_release_smem() before the
// kernel returns. Without that, the registration outlives the launch and a
// later kernel whose CTAs reuse the same block ids would take the TMA path
// through a stale shared-memory pointer.
//
// TMA routing additionally requires 16-byte aligned source and destination
// pointers and a transfer size that is a multiple of 16 bytes. Transfers that
// miss those constraints keep the normal put/get contract by falling back to
// load/store. Block-scoped puts staged from a global-memory source need at
// least two full warps in the CTA; smaller CTAs fall back as well.
//
// Grids larger than NIIN_TMA_MAX_BLOCKS CTAs are supported, but CTAs with a
// linear block id at or beyond that bound cannot register.
//
//   smem: shared-memory pointer, 16-byte aligned
//   size: bytes, at least nvshmemx_ask_smem(NVSHMEMX_SMEM_BARRIERS_ONLY)
__device__ __forceinline__ void nvshmemx_give_smem(void* smem, size_t size) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  if (niin_tma_policy() == NVSHMEMX_TMA_DISABLE) return;
  if (smem == nullptr || size == 0) return;
  // A buffer that is not 16-byte aligned cannot host the mbarriers or serve as
  // a cp.async.bulk tile. Decline the registration rather than corrupt data;
  // this CTA then uses load/store.
  if (!niin_tma_is_16b_aligned((size_t)(uintptr_t)smem)) return;

  niinContext const& ctx = niin_ctx();
  uintptr_t* bases = ctx.tmaSmemBases;
  size_t blockId = niin_tma_block_id();
  if (bases == nullptr || blockId >= ctx.tmaSmemBasesLen) {
    // Grid is larger than the registration table; this CTA cannot use TMA.
    return;
  }
  // The leading bytes are reserved for mbarriers. A buffer too small to hold
  // them cannot be registered, so this CTA falls back to load/store.
  if (size < (size_t)NIIN_SMEM_DATA_REGION_OFFSET) return;

  if (niin_tma_block_is_elected()) {
    bases[blockId] = (uintptr_t)smem;
    size_t* sizeSlot = ctx.tmaSmemSize;
    if (sizeSlot != nullptr) *sizeSlot = size;
  }
#else
  (void)smem; (void)size;
#endif
}

// nvshmemx_release_smem: drop this CTA's shared-memory registration.
//
// Call from every CTA that called nvshmemx_give_smem(), before the kernel
// returns. Call from all threads; only the elected leader writes. Follow with
// __syncthreads() if other threads must observe the cleared state.
__device__ __forceinline__ void nvshmemx_release_smem() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  if (niin_tma_policy() == NVSHMEMX_TMA_DISABLE) return;
  niinContext const& ctx = niin_ctx();
  uintptr_t* bases = ctx.tmaSmemBases;
  size_t blockId = niin_tma_block_id();
  if (bases != nullptr && blockId < ctx.tmaSmemBasesLen) {
    if (niin_tma_block_is_elected()) bases[blockId] = 0;
  }
#endif
}

// ---------------------------------------------------------------------------
// Transfer implementations
// ---------------------------------------------------------------------------

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900

// Local shared memory -> local or peer global memory, no staging.
//
// The caller must have issued fence.proxy.async.shared::cta after writing the
// source tile and before calling this, so the async proxy sees those stores.
// One fence covers a run of puts as long as the source is not rewritten in
// between. This matches NVSHMEM's contract for smem-sourced TMA puts.
//
// BLOCKING waits for the remote write to complete before returning; otherwise
// the transfer is left in flight for a later fence/quiet to drain.
//
// Returns 0 when the transfer was issued, -1 to fall back.
template <int SCOPE, bool BLOCKING>
__device__ __forceinline__ int niin_tma_copy_shared_global(void* gmemDst, const void* smemSrc,
                                                           size_t bytes) {
  if (bytes == 0) return 0;
  if (!niin_tma_is_16b_aligned((size_t)(uintptr_t)gmemDst)) return -1;
  if (!niin_tma_is_16b_aligned((size_t)(uintptr_t)smemSrc)) return -1;
  if (!niin_tma_is_16b_aligned(bytes)) return -1;
  if (bytes > (size_t)UINT32_MAX) return -1;

  unsigned int smemAddr = niin_tma_cvta_to_shared(smemSrc);

  // One elected thread issues the whole transfer as a single bulk op: that
  // amortizes TMA setup better than splitting the buffer across warps.
  if (niin_tma_is_leader<SCOPE>()) {
    niin_tma_bulk_shared_to_global(gmemDst, smemAddr, (uint32_t)bytes);
    niin_tma_bulk_commit_group();
    if constexpr (BLOCKING) {
      niin_tma_bulk_wait_group_0();
      __threadfence_system();
    }
  }

  niin_tma_group_sync<SCOPE>();
  return 0;
}

// Local or peer global memory -> local shared memory, no staging. Used by gets
// whose destination is shared memory. Always blocking: the mbarrier wait means
// the data has landed when this returns.
//
// Returns 0 when the transfer completed, -1 to fall back.
template <int SCOPE>
__device__ __forceinline__ int niin_tma_copy_global_shared(void* smemDst, const void* gmemSrc,
                                                           size_t bytes) {
  if (bytes == 0) return 0;
  if (!niin_tma_is_16b_aligned((size_t)(uintptr_t)smemDst)) return -1;
  if (!niin_tma_is_16b_aligned((size_t)(uintptr_t)gmemSrc)) return -1;
  if (!niin_tma_is_16b_aligned(bytes)) return -1;
  if (bytes > (size_t)UINT32_MAX) return -1;

  uintptr_t base = niin_ctx().tmaSmemBases[niin_tma_block_id()];
  size_t smemSize = niin_tma_smem_size();
  constexpr size_t kReserve = (size_t)NIIN_SMEM_DATA_REGION_OFFSET;
  if (base == 0 || smemSize <= kReserve) return -1;

  // Refuse a destination that would land on the reserved mbarrier region.
  uintptr_t dstStart = (uintptr_t)smemDst;
  uintptr_t dstEnd = dstStart + bytes;
  if (dstStart < base + kReserve && dstEnd > base) return -1;

  uint64_t* mbar = niin_tma_barrier_slot(base, 4);
  if (niin_tma_is_leader<SCOPE>()) {
    niin_tma_mbarrier_init(mbar);
    niin_tma_fence_proxy_async_shared_cta();
    niin_tma_mbarrier_arrive_expect_tx(mbar, (uint32_t)bytes);
    niin_tma_bulk_global_to_shared(smemDst, gmemSrc, (uint32_t)bytes, mbar);
    niin_tma_mbarrier_try_wait(mbar, 0);
  }

  niin_tma_group_sync<SCOPE>();
  return 0;
}

// Single-issuer global -> global copy, staged through this CTA's registered
// tile. Used for thread and warp scope.
//
// The issuer pulls each chunk into the tile with an inbound cp.async.bulk that
// releases an mbarrier, then pushes the same tile out with an outbound
// cp.async.bulk. Before reusing the tile it waits for the outbound op to finish
// reading shared memory. Blocking calls also wait for the remote write.
//
// Because the staging tile and mbarrier are per-CTA state, at most one thread
// in a CTA may have a staged transfer in flight at a time. Two threads of the
// same CTA calling a thread-scoped put concurrently would corrupt each other's
// staging. NIIN inherits that constraint from NVSHMEM.
//
// Returns 0 when the transfer was issued, -1 to fall back.
template <int SCOPE, bool BLOCKING>
__device__ __forceinline__ int niin_tma_copy_global_global_single(void* gmemDst,
                                                                  const void* gmemSrc,
                                                                  size_t bytes) {
  static_assert(SCOPE == NIIN_TMA_THREAD || SCOPE == NIIN_TMA_WARP,
                "single-issuer staging is only for thread or warp scope");
  if (bytes == 0) return 0;
  if (!niin_tma_is_16b_aligned((size_t)(uintptr_t)gmemDst)) return -1;
  if (!niin_tma_is_16b_aligned((size_t)(uintptr_t)gmemSrc)) return -1;
  if (!niin_tma_is_16b_aligned(bytes)) return -1;

  uintptr_t base = niin_ctx().tmaSmemBases[niin_tma_block_id()];
  size_t smemSize = niin_tma_smem_size();
  constexpr size_t kReserve = (size_t)NIIN_SMEM_DATA_REGION_OFFSET;
  if (base == 0 || smemSize <= kReserve) return -1;

  size_t tileSize = niin_tma_align_down_16(smemSize - kReserve);
  if (tileSize == 0) return -1;
  // CTA shared memory is far below 4 GiB, so bulk byte counts fit in 32 bits.
  const uint32_t tile = (uint32_t)tileSize;

  if (niin_tma_is_leader<SCOPE>()) {
    uint64_t* mbar = niin_tma_barrier_slot(base, 0);
    char* dataBuf = niin_tma_data_buffer(base);
    const char* src = (const char*)gmemSrc;
    char* dst = (char*)gmemDst;
    size_t remaining = bytes;

    niin_tma_mbarrier_init(mbar);
    // Make the mbarrier init visible to the async proxy before the first bulk op.
    niin_tma_fence_proxy_async_shared_cta();
    int phase = 0;

    while (remaining > 0) {
      uint32_t chunk = remaining < (size_t)tile ? (uint32_t)remaining : tile;

      niin_tma_mbarrier_arrive_expect_tx(mbar, chunk);
      niin_tma_bulk_global_to_shared(dataBuf, src, chunk, mbar);
      niin_tma_mbarrier_try_wait(mbar, phase);
      phase ^= 1;

      // Inbound and outbound both run on the async proxy and the mbarrier
      // release orders them, so no fence is needed in between.
      niin_tma_bulk_shared_to_global(dst, niin_tma_cvta_to_shared(dataBuf), chunk);
      niin_tma_bulk_commit_group();

      remaining -= chunk;
      src += chunk;
      dst += chunk;

      if (remaining > 0) niin_tma_bulk_wait_group_read_0();
    }
    if constexpr (BLOCKING) {
      niin_tma_bulk_wait_group_0();
      __threadfence_system();
    }
  }

  niin_tma_group_sync<SCOPE>();
  return 0;
}

// Block-scoped, warp-specialized, double-buffered global -> global copy.
//
// Warp 0 lane 0 loads, warp 1 lane 0 stores, so the inbound TMA of tile N+1
// overlaps the outbound TMA of tile N. Needs at least two full warps.
//
// Shared layout: [reserved barriers][buf0: tile][buf1: tile], tile being
// (size - NIIN_SMEM_DATA_REGION_OFFSET) / 2 rounded down to 16 bytes.
//
//   ready_bar[i]  load warp signals "buf[i] filled" through the cp.async.bulk
//                 complete_tx; the store warp waits on it.
//   done_bar[i]   store warp signals "outbound read of buf[i] finished, safe to
//                 refill"; the load warp waits on it.
//
// The buffer alternates every iteration (slot = i & 1) and the barrier parity
// flips every full pipe cycle of two iterations (phase = (i >> 1) & 1), so the
// hot loop needs no __syncthreads().
//
// Returns 0 when the transfer was issued, -1 to fall back.
template <bool BLOCKING>
__device__ __forceinline__ int niin_tma_copy_global_global_block(void* gmemDst,
                                                                 const void* gmemSrc,
                                                                 size_t bytes) {
  if (bytes == 0) return 0;
  if (!niin_tma_is_16b_aligned((size_t)(uintptr_t)gmemDst)) return -1;
  if (!niin_tma_is_16b_aligned((size_t)(uintptr_t)gmemSrc)) return -1;
  if (!niin_tma_is_16b_aligned(bytes)) return -1;

  uintptr_t base = niin_ctx().tmaSmemBases[niin_tma_block_id()];
  size_t smemSize = niin_tma_smem_size();
  constexpr size_t kReserve = (size_t)NIIN_SMEM_DATA_REGION_OFFSET;
  if (base == 0 || smemSize <= kReserve) return -1;

  size_t tileSize = niin_tma_align_down_16((smemSize - kReserve) / 2);
  if (tileSize == 0) return -1;
  const uint32_t tile = (uint32_t)tileSize;

  unsigned int blockThreads = blockDim.x * blockDim.y * blockDim.z;
  if (blockThreads < 2u * (unsigned int)warpSize) return -1;

  unsigned int tid = threadIdx.x + threadIdx.y * blockDim.x +
                     threadIdx.z * blockDim.x * blockDim.y;
  // CUDA forms warps from the linear CTA rank, so these are warp 0 lane 0 and
  // warp 1 lane 0 for any CTA shape with at least two full warps.
  bool isLoad = (tid == 0);
  bool isStore = (tid == (unsigned int)warpSize);

  uint64_t* ready0 = niin_tma_barrier_slot(base, 0);
  uint64_t* ready1 = niin_tma_barrier_slot(base, 1);
  uint64_t* done0 = niin_tma_barrier_slot(base, 2);
  uint64_t* done1 = niin_tma_barrier_slot(base, 3);
  char* buf0 = niin_tma_data_buffer(base);
  char* buf1 = buf0 + tile;

  // Init all four barriers once, then fence so the async proxy sees the init
  // before any cp.async.bulk arrives on them.
  if (isLoad) {
    niin_tma_mbarrier_init(ready0);
    niin_tma_mbarrier_init(ready1);
    niin_tma_mbarrier_init(done0);
    niin_tma_mbarrier_init(done1);
    niin_tma_fence_proxy_async_shared_cta();
  }
  __syncthreads();

  if (isLoad) {
    const char* src = (const char*)gmemSrc;
    size_t remaining = bytes;
    for (size_t i = 0; remaining > 0; i++) {
      int slot = (int)(i & 1);
      int phase = (int)((i >> 1) & 1);
      uint32_t chunk = remaining < (size_t)tile ? (uint32_t)remaining : tile;
      uint64_t* ready = slot ? ready1 : ready0;
      uint64_t* done = slot ? done1 : done0;
      char* buf = slot ? buf1 : buf0;

      // Each slot's done barrier is fresh for the first two iterations, so the
      // wait is skipped there; after that we wait for the store warp's flip.
      if (i >= 2) niin_tma_mbarrier_try_wait(done, phase ^ 1);
      niin_tma_bulk_global_to_shared(buf, src, chunk, ready);
      niin_tma_mbarrier_arrive_expect_tx(ready, chunk);

      src += chunk;
      remaining -= chunk;
    }
  } else if (isStore) {
    unsigned int dataAddr0 = niin_tma_cvta_to_shared(buf0);
    unsigned int dataAddr1 = niin_tma_cvta_to_shared(buf1);
    char* dst = (char*)gmemDst;
    size_t remaining = bytes;
    for (size_t i = 0; remaining > 0; i++) {
      int slot = (int)(i & 1);
      int phase = (int)((i >> 1) & 1);
      uint32_t chunk = remaining < (size_t)tile ? (uint32_t)remaining : tile;
      uint64_t* ready = slot ? ready1 : ready0;
      uint64_t* done = slot ? done1 : done0;
      unsigned int dataAddr = slot ? dataAddr1 : dataAddr0;

      niin_tma_mbarrier_try_wait(ready, phase);
      niin_tma_bulk_shared_to_global(dst, dataAddr, chunk);
      niin_tma_bulk_commit_group();
      niin_tma_bulk_wait_group_read_0();
      niin_tma_mbarrier_arrive_expect_tx(done, 1);
      niin_tma_mbarrier_complete_tx(done, 1);

      dst += chunk;
      remaining -= chunk;
    }
  }

  if constexpr (BLOCKING) {
    if (isStore) {
      niin_tma_bulk_wait_group_0();
      __threadfence_system();
    }
  }
  __syncthreads();
  return 0;
}

// Route a global -> global copy by scope.
template <int SCOPE, bool BLOCKING>
__device__ __forceinline__ int niin_tma_copy_global_global(void* gmemDst, const void* gmemSrc,
                                                           size_t bytes) {
  if constexpr (SCOPE == NIIN_TMA_BLOCK) {
    return niin_tma_copy_global_global_block<BLOCKING>(gmemDst, gmemSrc, bytes);
  } else {
    return niin_tma_copy_global_global_single<SCOPE, BLOCKING>(gmemDst, gmemSrc, bytes);
  }
}

#endif  // __CUDA_ARCH__ >= 900

// ---------------------------------------------------------------------------
// Dispatch used by the LSA put/get paths
// ---------------------------------------------------------------------------

// Try to move `bytes` between two already-resolved pointers with TMA, choosing
// the route from where the local buffer lives. `dst` and `src` must both be
// directly addressable (local memory or an LSA peer pointer).
//
// BLOCKING mirrors NVSHMEM's split: a blocking put waits for the transfer to
// complete before returning, while an nbi put leaves it in flight for a later
// fence/quiet to drain. Gets are always completion-bound, so a get-shaped
// route ignores BLOCKING.
//
// Returns 0 when TMA handled the copy, -1 when the caller must fall back. All
// the reasons to fall back are uniform across a threadgroup, so a warp- or
// block-scoped caller either takes the TMA path with every thread or none.
template <int SCOPE, bool BLOCKING = true>
NIIN_NOINLINE_DEVICE int niin_tma_try_copy_enabled(void* dst, const void* src, size_t bytes) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  if (!niin_tma_smem_registered()) return -1;

  // A local-memory (stack) buffer is a legal RMA source but is not addressable
  // by the TMA engine, which only reads global and shared space. Decline it.
  if (__isLocal(src) || __isLocal(dst)) return -1;

  bool srcShared = __isShared(src);
  bool dstShared = __isShared(dst);
  // A shared-to-shared copy is not an RMA route; leave it to load/store.
  if (srcShared && dstShared) return -1;
  if (srcShared) return niin_tma_copy_shared_global<SCOPE, BLOCKING>(dst, src, bytes);
  // gmem -> smem is a get landing in shared memory: the mbarrier wait is the
  // transfer, so there is no non-blocking form of it.
  if (dstShared) return niin_tma_copy_global_shared<SCOPE>(dst, src, bytes);
  return niin_tma_copy_global_global<SCOPE, BLOCKING>(dst, src, bytes);
#else
  (void)dst; (void)src; (void)bytes;
  return -1;
#endif
}

template <int SCOPE, bool BLOCKING = true>
__device__ __forceinline__ int niin_tma_try_copy(void* dst, const void* src, size_t bytes) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  if (niin_tma_policy() == NVSHMEMX_TMA_DISABLE) return -1;
  return niin_tma_try_copy_enabled<SCOPE, BLOCKING>(dst, src, bytes);
#else
  (void)dst; (void)src; (void)bytes;
  return -1;
#endif
}

// Drain every bulk op this thread has issued. Used by fence and quiet so that
// TMA-initiated transfers are ordered and complete alongside the load/store and
// GIN paths. A no-op when this CTA has no registered shared memory.
NIIN_NOINLINE_DEVICE void niin_tma_drain_registered_slow() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  if (niin_tma_smem_registered()) {
    niin_tma_bulk_commit_group();
    niin_tma_bulk_wait_group_0();
  }
#endif
}

__device__ __forceinline__ void niin_tma_drain_if_registered() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  if (niin_tma_policy() == NVSHMEMX_TMA_DISABLE) return;
  niin_tma_drain_registered_slow();
#endif
}

#endif  // NIIN_TMA_H_
