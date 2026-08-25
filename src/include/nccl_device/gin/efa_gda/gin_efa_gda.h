/*************************************************************************
 * Copyright (c) 2026 Amazon.com, Inc. or its affiliates. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * EFA GDA implementations for NCCL GIN device-side APIs.
 *
 * This file provides ncclGinApi_*<NCCL_NET_DEVICE_GIN_EFA_GDA> template
 * specializations that target EFA via efa-dp-direct.
 *
 * Implemented: Put (data + signal/counter endpoints, signal-only via
 *              scratch buffer), PutValue (value staged through the
 *              per-endpoint slot pool), Get, Flush, GetSignalPtr,
 *              GetCounterPtr, ResetSignal, ResetCounter.
 * Stub: FlushAsync, Wait.
 *************************************************************************/

#ifndef _NCCL_DEVICE_GIN_EFA_GDA_H_
#define _NCCL_DEVICE_GIN_EFA_GDA_H_

#include <cstdint>
#include <cuda/atomic>
#include <cooperative_groups.h>

#include "../gin_device_common.h"
#include "gin_efa_gda_dev.h"

/* efa-dp-direct device functions (inline implementations) */
#include "../../transport/net_efa_gda/efa-dp-direct/include/device/efa_cuda_dp_impl.cuh"

namespace nccl {
namespace gin {
namespace efa_gda {

/* The plugin returns a contiguous array of per-context dev handles
 * in GPU memory; ctx.handle points at element 0. ctx.contextId
 * selects the entry for this caller. */
NCCL_DEVICE_INLINE static nccl_ofi_gin_gdaki_dev_handle* getDevHandle(ncclGinCtx ctx) {
  return &((nccl_ofi_gin_gdaki_dev_handle*)ctx.handle)[ctx.contextId];
}

/* ── Mode mapping: NCCL → efa-dp-direct ───────────────────────────── */

template <ncclGinResourceSharingMode mode>
static constexpr cuda::thread_scope ncclGinScope =
  (mode == NCCL_GIN_RESOURCE_SHARING_CTA) ? cuda::thread_scope_block : cuda::thread_scope_device;

/* The EFA hardware completion counters (FI_WRITE / FI_REMOTE_WRITE) wrap at
 * 2^31, while the kernel-side producer cursors are uint32 (wrap at 2^32).
 * Every comparison between a producer cursor and a HW counter must therefore
 * be a modular difference reduced to 31 bits: compute (producer - consumer)
 * and mask with EFA_CNTR_MASK. The true difference (in-flight / outstanding
 * work) is always far below 2^31 (bounded by sq_size == 4096), so the masked
 * difference recovers the exact value regardless of how many times either side
 * has wrapped. Never compare absolute counter values. */
static constexpr uint32_t EFA_CNTR_MASK = 0x7fffffffu;

/* Hardware cap on a single RDMA op (write or read): efadv_device_attr.max_rdma_size,
 * 1 GiB on current EFA devices. A WQE exceeding it fails on the NIC as a CQ
 * error, which this CQ-less path never observes, so the op hangs; the u32
 * SGE length field additionally truncates sizes >= 4 GiB. Larger transfers are
 * split into chunks of at most this size in putImplMode / getImplMode. */
static constexpr uint32_t EFA_GDA_MAX_RDMA_OP_SIZE = 1u << 30;

/* Which RDMA operation the post path issues. A write and a read populate identical
 * WR fields, so the opcode is the only difference between them and it is what
 * decides which way the bytes move. */
enum efaGdaRdmaOp {
  EFA_GDA_RDMA_WRITE,
  EFA_GDA_RDMA_READ
};

/* ── Atomic primitives parameterized on scope and memory order ────── */

template <cuda::thread_scope Scope, cuda::memory_order Order>
NCCL_DEVICE_INLINE static uint64_t scopedAtomicLoad(uint64_t* ptr) {
  cuda::atomic_ref<uint64_t, Scope> r(*ptr);
  return r.load(Order);
}

template <cuda::thread_scope Scope, cuda::memory_order Order>
NCCL_DEVICE_INLINE static void scopedAtomicAdd(uint64_t* ptr, uint64_t val) {
  cuda::atomic_ref<uint64_t, Scope> r(*ptr);
  r.fetch_add(val, Order);
}

/* ── NIC-written hardware counter (FI_WRITE / FI_REMOTE_WRITE) ────── */

/* Read a NIC-written hardware counter from GPU memory. System scope makes
 * the load coherent with the NIC's PCIe writes (bypasses GPU caches).
 * Acquire is the default and matches libfabric's local-completion contract:
 * when this load observes the counter has reached a target, the NIC's prior
 * side effects (e.g. source-buffer DMA-reads complete) are ordered-before
 * whatever this thread does next (e.g. overwriting that source buffer or
 * reusing the slot). */
template <cuda::memory_order Order = cuda::memory_order_acquire>
NCCL_DEVICE_INLINE static uint64_t hwCounterLoad(uint64_t* ptr) {
  return scopedAtomicLoad<cuda::thread_scope_system, Order>(ptr);
}

/* ── ringDoorbell: shared doorbell-ring used by the post-path ring sites ─

 * Rings the SQ doorbell to `target`, then advances the bookkeeping cursors:
 * submitted_count grows by the newly-rung span (target - db_rung) so Flush's
 * completion wait is exact, and db_rung (wqes_posted) is published to record
 * how far the doorbell has been rung.
 *
 * Preconditions the caller must satisfy:
 *   - The caller is allowed to ring up to `target` (it holds the doorbell
 *     turn, or is draining already-handed-off slots).
 *   - Every WQE in [db_rung, target) is already visible to the NIC. Each
 *     producing group publishes its own WQEs with __threadfence_system()
 *     before handoff, so the WQE data is system-visible by the time any slot
 *     is eligible to be rung here.
 *
 * Only a post-doorbell fence is emitted (to order the doorbell MMIO write).
 * A pre-doorbell publish fence would be useless: __threadfence_system()
 * orders only the calling thread's own writes, and the WQEs being rung were
 * written by other threads. */
template <ncclGinResourceSharingMode mode>
NCCL_DEVICE_INLINE static void ringDoorbell(efa_cuda_qp* qp, uint64_t* submitted_count_ptr,
                                            cuda::atomic_ref<uint32_t, ncclGinScope<mode>>& dbrung_ref,
                                            uint32_t db_rung, uint32_t target) {
  uint64_t dbAddr = (uint64_t)__cvta_generic_to_global(qp->sq.wq.db);
  asm volatile("st.mmio.relaxed.sys.global.b32 [%0], %1;" : : "l"(dbAddr), "r"(target) : "memory");
  /* Order the doorbell MMIO write. Use acq_rel (MEMBAR.ALL.SYS) instead of
   * __threadfence_system (MEMBAR.SC.SYS). */
  cuda::atomic_thread_fence(cuda::memory_order_acq_rel, cuda::thread_scope_system);
  scopedAtomicAdd<ncclGinScope<mode>, cuda::memory_order_relaxed>(submitted_count_ptr, (uint64_t)(target - db_rung));
  dbrung_ref.store(target, cuda::memory_order_release);
}

/* ── postRdmaOp: shared post path for Put, PutValue and Get ──────── */

/* Posts an RDMA write on `ep`'s local QP (its FI_WRITE counter tracks
 * local completion) to the remote QP given by the explicit
 * (ah, qpn, qkey) tuple. The local poster QP and the remote target QP
 * are chosen independently by the caller: counterId selects the local
 * poster (this `ep`), the target slot selects the remote tuple (via the
 * poster's [total_slots*nranks] target table).
 *
 * PutValue staging (pvSrcVal != nullptr): the caller posts from the
 * dedicated PutValue endpoint (pvdata), so each lane stages its value into
 * its own pool slot at pvSliceBase + (reserved SQ slot % sq_size) *
 * pvSlotSize and points the WR's SGE there. Put (pvSrcVal == nullptr) uses
 * the caller's fixed srcAddr/srcLkey. */
template <ncclGinResourceSharingMode mode, efaGdaRdmaOp op = EFA_GDA_RDMA_WRITE>
NCCL_DEVICE_INLINE static void postRdmaOp(nccl_ofi_gin_gdaki_dev_endpoint_handle* ep, uint16_t ah, uint16_t qpn,
                                          uint32_t qkey, uint64_t srcAddr, uint32_t srcLkey, uint32_t writeBytes,
                                          uint64_t dstAddr, uint32_t dstRkey,
                                          uint32_t optFlags = ncclGinOptFlagsDefault, const void* pvSrcVal = nullptr,
                                          uint32_t pvValBytes = 0, uint32_t pvLkey = 0, uint64_t pvSliceBase = 0,
                                          uint32_t pvSlotSize = 0) {
  efa_cuda_qp* qp = (efa_cuda_qp*)ep->qp;
  uint64_t* submitted_count_ptr = &ep->submitted_count;
  uint64_t* local_cntr_ptr = ep->local_cntr_value;
  uint32_t sq_size_val = ep->sq_size;

  /* WQE staging buffer. Always the 128B form: the builder zeroes and the MMIO
   * loop copies only the QP's negotiated wqe_size, but sizing the storage to
   * the larger layout keeps one buffer valid for both 64B and 128B QPs. */
  efa_io_tx_wqe_128 wr_storage;
  uint16_t wqe_size = qp->sq.wr_ctx.wqe_size;

  /* Only the SGE differs for PutValue, so set_sge is deferred. */
  EfaCudaWrBuilder wr(&qp->sq.wr_ctx, (uint8_t*)&wr_storage);
  /* The opcode is the only thing that differs between a Put and a Get here: both
   * carry the RDMA address pair (dstAddr, dstRkey) and the local buffer in the SGE,
   * and the opcode decides which way the bytes move. A read therefore arrives with
   * the REMOTE source in the RDMA pair and the LOCAL destination in the SGE. */
  if NCCL_IF_CONSTEXPR (op == EFA_GDA_RDMA_READ) {
    wr.init_rdma_read((uint64_t)threadIdx.x, dstRkey, dstAddr);
  } else {
    wr.init_rdma_write((uint64_t)threadIdx.x, dstRkey, dstAddr);
  }
  wr.set_remote(ah, (uint32_t)qpn, qkey);
  /* Tag the WQE as PPS-sensitive. GIN puts are small, high-rate writes, so
   * ask the NIC to optimize for packets-per-second (burst PPS) rather than
   * bandwidth. This sets the PROCESSING_HINTS field in the WQE meta
   * descriptor (ctrl3); it is a hint, so the device may ignore it. */
  wr.set_processing_hints(EFA_CUDA_PROCESSING_HINT_BURST_PPS_SENSITIVE);

  /* Sliding-window SQ post with warp coalescing (Stage 2).
   *
   * Inlines the reserve / write / doorbell sequence directly against
   * the efa_cuda_qp ring. Two shared cursors in the QP coordinate all
   * posters (across lanes, warps and CTAs):
   *
   *   pc             : monotonic reservation index. A group's leader
   *                    claims its whole range with one atomicAdd(+g).
   *   wqes_completed : "released" cursor — the rendezvous/handoff token.
   *                    Advances when a group passes the doorbell turn,
   *                    whether or not it actually rang (so deferred
   *                    groups still hand off in strict slot order).
   *   wqes_posted    : "doorbell rung" cursor (db_rung) — the value last
   *                    written to the SQ doorbell register. With request
   *                    aggregation (ncclGinOptFlagsAggregateRequests) the
   *                    doorbell is deferred: a group writes its WQEs and
   *                    hands off without ringing; a later non-aggregated
   *                    group rings to its own chunk_next, which
   *                    is >= every deferred slot (the doorbell is
   *                    monotonic), draining the whole batch in one ring.
   *                    Un-rung WQEs are bounded by max_batch via the
   *                    window check below (gated on db_rung), so the EFA
   *                    staging limit is always respected.
   *
   * Coalescing: lanes of a warp targeting the same QP form a group via
   * coalesced_threads() + labeled_partition(qp). The leader reserves g
   * = group.num_threads() contiguous slots; every member writes its own
   * WQE in parallel; the leader rings one doorbell for the batch.
   *
   * max_batch bound: a group may be larger than the EFA staging limit
   * (a warp can have up to 32 lanes on one QP), so the group is chunked
   * into windows of <= max_batch. For each chunk:
   *   - window-wait (leader): write only once the chunk fits within the
   *     released window [released, released + max_batch). This bounds
   *     un-doorbelled WQEs across ALL concurrent groups to max_batch.
   *   - SQ ring-overflow wait (leader): the chunk's high-water slot must
   *     be within sq_size of the NIC consumer (FI_WRITE counter).
   *   - members write their WQEs in parallel.
   *   - doorbell rendezvous (leader): wait until released == chunk_base
   *     (strict slot order across groups), ring the doorbell, then
   *     advance released to hand off to the next group. */
  cooperative_groups::coalesced_group active = cooperative_groups::coalesced_threads();
  auto group = cooperative_groups::labeled_partition(active, (unsigned long long)(uintptr_t)qp);

  int my_idx = group.thread_rank();
  int group_size = group.num_threads();
  bool is_leader = (my_idx == 0);
  uint32_t max_batch = qp->sq.wq.max_batch;

  cuda::atomic_ref<uint32_t, ncclGinScope<mode>> pc_ref(qp->sq.wq.pc);
  cuda::atomic_ref<uint32_t, ncclGinScope<mode>> base_ref(qp->sq.wq.wqes_completed);
  /* db_rung reuses the wqes_posted field, which the GIN path does not
   * otherwise use (zero-initialized by the plugin). */
  cuda::atomic_ref<uint32_t, ncclGinScope<mode>> dbrung_ref(qp->sq.wq.wqes_posted);
  const bool aggregate = (optFlags & ncclGinOptFlagsAggregateRequests) != 0;

  /* Leader reserves the whole group's contiguous slot range. */
  uint32_t base = 0;
  if (is_leader) {
    base = pc_ref.fetch_add((uint32_t)group_size, cuda::memory_order_relaxed);
  }
  base = group.shfl(base, 0);

  /* Chunk the group into windows of <= max_batch. */
  for (int chunk_start = 0; chunk_start < group_size; chunk_start += (int)max_batch) {
    int chunk_size = min((int)max_batch, group_size - chunk_start);
    uint32_t chunk_base = base + (uint32_t)chunk_start;
    uint32_t chunk_next = chunk_base + (uint32_t)chunk_size;

    if (is_leader) {
      /* Backpressure: keep the number of written-but-un-rung WQEs within
       * max_batch (the hard EFA staging limit). The un-rung depth this chunk
       * would reach is chunk_next - db_rung.
       *
       * We must ring to make room rather than just wait: a deferring group
       * leaves its WQEs un-rung and hands off without ringing, so db_rung is
       * only ever advanced by a group that actively rings. If this group
       * blocked passively, no one would ring the deferred slots below it and
       * it would wait forever.
       *
       * So while the depth would exceed max_batch, make room by ringing the
       * deferred work that is already written. Only the turn-holder
       * (base_ref == chunk_base) may ring, and it rings up to chunk_base:
       * those slots were written and published by earlier groups that handed
       * off before us, and the deferred span below chunk_base is itself
       * <= max_batch, so one doorbell drains it. If we do not yet hold the
       * turn, a lower group does and will either ring (advancing db_rung) or
       * hand off (advancing base_ref); keep checking until our chunk fits. */
      while (chunk_next - dbrung_ref.load(cuda::memory_order_relaxed) > max_batch) {
        if (base_ref.load(cuda::memory_order_acquire) == chunk_base) {
          uint32_t db_rung = dbrung_ref.load(cuda::memory_order_relaxed);
          if (chunk_base != db_rung) {   /* deferred, already-written batch */
            ringDoorbell<mode>(qp, submitted_count_ptr, dbrung_ref, db_rung, chunk_base);
          }
        }
      }
      /* SQ ring-overflow backpressure on the chunk's high-water slot.
       * Poll the NIC FI_WRITE counter with system-scope relaxed loads.
       *
       * In-flight count is computed as a 31-bit modular difference
       * (producer chunk_next minus the NIC FI_WRITE counter): the HW
       * counter wraps at 2^31 and chunk_next is uint32, so a plain
       * widened subtraction would underflow once either side wraps.
       * The true in-flight depth is bounded by sq_size (4096) « 2^31,
       * so the masked difference is exact. */
      while (((chunk_next - (uint32_t)hwCounterLoad<cuda::memory_order_relaxed>(local_cntr_ptr)) & EFA_CNTR_MASK) >
             sq_size_val) {
        /* spin */
      }
    }
    group.sync();   /* members wait for leader's backpressure before writing */

    /* Members in this window write their own WQE into their slot. */
    if (my_idx >= chunk_start && my_idx < chunk_start + chunk_size) {
      uint32_t my_slot = chunk_base + (uint32_t)(my_idx - chunk_start);

      uint64_t wrSrcAddr = srcAddr;
      uint32_t wrSrcLkey = srcLkey;
      if (pvSrcVal != nullptr) {
        /* PutValue: stage the value into this lane's pool slot, then point
         * the SGE at it. */
        uint64_t slot_idx = (uint64_t)my_slot % (uint64_t)sq_size_val;
        uint64_t local_addr = pvSliceBase + slot_idx * (uint64_t)pvSlotSize;
        for (uint32_t b = 0; b < pvValBytes; b++) ((uint8_t*)local_addr)[b] = ((const uint8_t*)pvSrcVal)[b];
        wrSrcAddr = local_addr;
        wrSrcLkey = pvLkey;
      }

      /* Only the source SGE is per-lane, the rest of wr was already built above. */
      wr.set_sge(wrSrcLkey, wrSrcAddr, writeBytes);

      uint32_t sq_idx = my_slot & qp->sq.wq.queue_mask;
      int wqe_phase = (int)((my_slot >> qp->sq.wq.queue_size_shift) & 1u);
      EFA_SET(&wr_storage.meta.ctrl2, EFA_IO_TX_META_DESC_PHASE, wqe_phase);
      uint64_t* src = (uint64_t*)&wr_storage;
      uint64_t* dst = (uint64_t*)(qp->sq.wq.buf + sq_idx * wqe_size);
      /* One final system-scope fence publishes the complete WQE after these
       * relaxed MMIO stores. */
      uint64_t dstAddr = (uint64_t)__cvta_generic_to_global(dst);
      uint32_t num_words = wqe_size / (uint32_t)sizeof(uint64_t);
      for (uint32_t i = 0; i < num_words; i++) {
        uint64_t value = src[i];
        asm volatile("st.mmio.relaxed.sys.global.b64 [%0], %1;"
                     :
                     : "l"(dstAddr + i * sizeof(uint64_t)), "l"(value)
                     : "memory");
      }
      /* Publish this group's WQE writes to system scope so they are visible
       * to the NIC whenever any doorbell rings a slot in this range. */
      cuda::atomic_thread_fence(cuda::memory_order_acq_rel, cuda::thread_scope_system);
    }
    group.sync();   /* all members' WQE writes for this chunk are done */

    if (is_leader) {
      /* Doorbell-order rendezvous: take the turn in strict slot order. */
      while (base_ref.load(cuda::memory_order_relaxed) != chunk_base) {
        /* spin */
      }

      /* Ring unless aggregating. Force a ring if deferring would leave
       * more than max_batch un-rung WQEs (db_rung is the last rung slot),
       * so the EFA staging limit is never exceeded. When we do ring, ring
       * to chunk_next: it is >= every deferred slot below us (we hold the
       * turn in slot order), so one doorbell drains the whole contiguous
       * batch. submitted_count advances by everything since db_rung. */
      uint32_t db_rung = dbrung_ref.load(cuda::memory_order_relaxed);
      bool must_ring = (!aggregate) || (chunk_next - db_rung >= max_batch);
      if (must_ring) {
        ringDoorbell<mode>(qp, submitted_count_ptr, dbrung_ref, db_rung, chunk_next);
      } else {
        /* Publish this group's WQE writes to system scope before handing off,
         * so they are visible to the NIC whenever any doorbell (this group's or
         * a later draining group's) rings a slot in this range.
         * Each group must publish its own writes: __threadfence_system()
         * orders only the calling thread's writes, and the handoff (base_ref)
         * is device/block scope, so a later thread's fence cannot publish this
         * group's writes for it. Runs only on the defer path. */
        cuda::atomic_thread_fence(cuda::memory_order_acq_rel, cuda::thread_scope_system);
      }
      base_ref.store(chunk_next, cuda::memory_order_release);   /* hand off to next group */
    }
    group.sync();   /* chunk fully posted before the next chunk */
  }
}

/* ── putImplMode: mode-templated Put implementation ─────────────── */

template <ncclGinResourceSharingMode mode, typename Coop>
NCCL_DEVICE_INLINE static void putImplMode(ncclGinCtx ctx, Coop coop, int peer, bool hasWins, ncclGinWindow_t dstWin,
                                           size_t dstOff, ncclGinWindow_t srcWin, size_t srcOff, size_t bytes,
                                           ncclGinSignalDescriptor signal, ncclGinSignalOp_t signalOp,
                                           uint64_t signalOpArg, bool hasCounter, ncclGinCounter_t counterId,
                                           bool hasDescriptor, ncclGinDescriptorSmem* descriptor,
                                           cuda::thread_scope required, cuda::thread_scope given, uint32_t optFlags) {
  coop.sync();
  if (coop.thread_rank() == 0) {
    nccl_ofi_gin_gdaki_dev_handle* dev = getDevHandle(ctx);

    bool hasPayload = hasWins && bytes > 0;

    /* This backend supports INDEXED signals only. EFA's FI_REMOTE_WRITE
     * counter ticks exactly once per inbound write and has no atomic-add,
     * so a signal Add-by-N is emulated as N inbound write events (see the
     * posting block below). VA-typed signals are not representable. */
    assert((signal.type == NCCL_GIN_SIGNAL_TYPE_NONE || signal.type == NCCL_GIN_SIGNAL_TYPE_INDEXED) &&
           "EFA GDA: only INDEXED signals are supported");
    assert((signal.type != NCCL_GIN_SIGNAL_TYPE_INDEXED || (int)signal.indexedSignal.signalId < dev->nSignals) &&
           "EFA GDA: signalId out of range");
    assert((!hasCounter || (int)counterId < dev->nCounters) && "EFA GDA: counterId out of range");

    /* A Put ALWAYS posts at least one WQE -- even with no payload and no
     * signal/counter, where it degenerates to a 0-byte write to the peer's
     * per-context scratch via the peer DATA endpoint (target slot 0), which
     * binds no FI_REMOTE_WRITE: remotely unobservable.
     *
     * Empty puts cannot be dropped as no-ops, because under
     * ncclGinOptFlagsAggregateRequests a put carries a LOCAL side effect: its
     * non-aggregated doorbell rendezvous in postRdmaOp is what publishes
     * earlier deferred WQEs on the QP. Dropping an "empty" put would make it
     * impossible to terminate a deferred stream whose last real put was
     * aggregated -- the tail WQEs (and their signals) would never be handed to
     * the NIC and the peer would wait forever. Callers that want to elide
     * empty puts for performance should skip at the call site, where "empty"
     * is actually known. */
    {
      /* Three WQE patterns:
       *
       * (a) Data put: posts an RDMA write of the user payload.
       *     Routed through the signal/counter endpoint when a signal or
       *     counter is attached, so the receiver's FI_REMOTE_WRITE fires
       *     on completion; otherwise routed through the data endpoint.
       *
       * (b) Signal-only: posts a 0-byte RDMA write into the peer's
       *     per-context scratch buffer. The write event bumps the
       *     receiver's FI_REMOTE_WRITE counter on the signal endpoint.
       *
       * (c) Empty (no payload, no signal/counter): same 0-byte scratch
       *     write as (b) but addressed to the peer DATA endpoint, which
       *     binds no FI_REMOTE_WRITE -- remotely unobservable. Posted for
       *     its local doorbell rendezvous (see the block comment above). */
      uint64_t absSrcAddr;
      uint64_t absDstAddr;
      uint32_t dstRkey;
      uint32_t srcLkey;
      uint32_t writeBytes;
      if (hasPayload) {
        /* Resolve the memory window to the rail this context is bound
         * to. The window is an array of per-rail mr_handle pointers;
         * dev->rail_id (= contextId % num_rails, pre-baked by the
         * plugin) selects this context's rail, keeping the path
         * rail-agnostic. */
        nccl_ofi_gin_gdaki_mr_handle* dstMh = ((nccl_ofi_gin_gdaki_mr_handle**)dstWin)[dev->rail_id];
        nccl_ofi_gin_gdaki_mr_handle* srcMh = ((nccl_ofi_gin_gdaki_mr_handle**)srcWin)[dev->rail_id];
        absSrcAddr = srcMh->local_addr + srcOff;
        absDstAddr = dstMh->peers[peer].remote_addr + dstOff;
        dstRkey = dstMh->peers[peer].rkey;
        srcLkey = srcMh->lkey;
        writeBytes = (uint32_t)bytes;
      } else {
        absSrcAddr = dev->scratch_local_addr;
        absDstAddr = dev->scratch_remote_addrs[peer];
        dstRkey = dev->scratch_remote_rkeys[peer];
        srcLkey = dev->scratch_lkey;
        writeBytes = 0;
      }

      /* A single RDMA write carries two independent QP choices:
       *
       *   - Local poster QP: the SQ we post from; its FI_WRITE counter
       *     ticks on local completion. This is a LOCAL property,
       *     selected by counterId. With a counter request we post from
       *     counter_handles[counterId] so its FI_WRITE is the counter;
       *     otherwise we post from the data endpoint (a signal-only or
       *     plain put has no local counter to track here).
       *
       *   - Remote target QP: the peer endpoint we address; the peer's
       *     FI_REMOTE_WRITE counter on THAT endpoint ticks (only when the
       *     target is a signal/sc endpoint). This is a TARGET property,
       *     selected below as a slot into the poster's target
       *     table (slot 0 = peer data EP, slot 1+signalId = peer sc EP).
       *
       * The signal (signalId) selects the remote (target) QP via the
       * slot; the counter (counterId) selects the local (poster) QP. */
      nccl_ofi_gin_gdaki_dev_endpoint_handle* main_ep =
        hasCounter ? &dev->counter_handles[counterId]->base : &dev->data;

      /* Target slot in the [total_slots*nranks] target addressing table
       * (targetSlot-major: idx = targetSlot*nranks + peer):
       *     signalling write (INDEXED) -> slot 1 + signalId (peer sc EP,
       *       whose FI_REMOTE_WRITE the receiver's waitSignal observes)
       *     plain put / counter-only    -> slot 0 (peer DATA EP, which
       *       binds no FI_REMOTE_WRITE, so the write ticks the local
       *       FI_WRITE counter without firing a signal on the receiver)
       * The local poster QP is chosen by counterId (main_ep); the slot
       * picks the remote target. */
      const bool isIndexed = (signal.type == NCCL_GIN_SIGNAL_TYPE_INDEXED);
      const uint32_t targetSlot = isIndexed ? (1u + (uint32_t)signal.indexedSignal.signalId) : 0u;
      const uint32_t targetIdx = targetSlot * (uint32_t)dev->nranks + (uint32_t)peer;

      /* Signal increment count.
       *
       * EFA's FI_REMOTE_WRITE counter advances by exactly 1 per inbound
       * write, so an Add-by-N signal is emulated as N inbound write
       * events. Inc is always +1 (signalOpArg is defined to be 1 for Inc
       * by the GIN API). A pure data put or counter request (no signal)
       * contributes a single write.
       *
       * Correctness-first: this issues the writes as separate posts (one
       * doorbell each). A future optimization can batch the doorbell over
       * a larger reservation via postRdmaOp's chunk loop; that must NOT
       * be done by suppressing doorbells across calls, which would break
       * the wqes_completed sliding-window / rendezvous invariant.
       *
       * TODO: batch the doorbells for a signal Add-by-N (and across the
       * payload + scratch writes) instead of ringing one doorbell per
       * write. Must reuse postRdmaOp's chunk loop (one doorbell per
       * max_batch reservation), NOT a doorbell-suppress flag across
       * separate calls. */
      uint32_t signalCount = 1u;
      if (isIndexed && signalOp == ncclGinSignalAdd) {
        signalCount = (uint32_t)signalOpArg;
      }

      /* Target tuples, both read from the target table at the same slot.
       *
       * The local FI_WRITE counter selected by counterId must tick
       * EXACTLY ONCE per Put, no matter how many physical writes the
       * signal Add-by-N expands into. So only the FIRST write rides
       * `main_ep` (the counterId-selected poster when hasCounter, else
       * the data EP); every remaining (signalCount - 1) increment rides
       * the DATA endpoint, whose FI_WRITE is not the caller's counter.
       * Both resolve the SAME remote target (slot), but each through its
       * own endpoint's AV — an address handle is AV-local, so the data EP
       * uses its own tuple, not main_ep's. */
      const uint16_t main_ah = main_ep->target_address_handles[targetIdx];
      const uint16_t main_qpn = main_ep->target_remote_qpns[targetIdx];
      const uint32_t main_qkey = main_ep->target_qkey[targetIdx];
      const uint16_t dataSigAh = dev->data.target_address_handles[targetIdx];
      const uint16_t dataSigQpn = dev->data.target_remote_qpns[targetIdx];
      const uint32_t dataSigQkey = dev->data.target_qkey[targetIdx];

      /* Chunk payloads that exceed the EFA per-op limit.
       *
       * A single RDMA write is capped at EFA_GDA_MAX_RDMA_OP_SIZE (1 GiB).
       * Split larger payloads into full-size leading chunks plus a tail
       * (<= cap); the tail is posted by the normal signal/counter-carrying
       * path below.
       *
       * Leading chunks target the peer's DATA EP (slot 0): no remote signal
       * fires and the caller's counter does not tick. They are posted
       * non-aggregated so their doorbells ring (the drain below only counts
       * rung WQEs). */
      if (hasPayload && bytes > (size_t)EFA_GDA_MAX_RDMA_OP_SIZE) {
        const size_t cap = (size_t)EFA_GDA_MAX_RDMA_OP_SIZE;
        const size_t nLeading = (bytes - 1) / cap; /* tail is (0, cap] */
        /* Peer's DATA EP tuple: target slot 0 -> idx = 0*nranks + peer. */
        const uint32_t dataIdx = (uint32_t)peer;
        const uint16_t dAh = dev->data.target_address_handles[dataIdx];
        const uint16_t dQpn = dev->data.target_remote_qpns[dataIdx];
        const uint32_t dQkey = dev->data.target_qkey[dataIdx];
        for (size_t i = 0; i < nLeading; i++) {
          postRdmaOp<mode>(&dev->data, dAh, dQpn, dQkey, absSrcAddr + i * cap, srcLkey, (uint32_t)cap,
                           absDstAddr + i * cap, dstRkey, ncclGinOptFlagsDefault);
        }
        /* EFA SRD is unordered: the tail landing does not imply the leading
         * chunks landed. So when the tail announces completion (signal or
         * counter), first wait for the leading chunks' local completions: a
         * local completion means the write has been queued towards the
         * receiver's PCIe, so any write issued after it to that same PCIe
         * destination is delivered after it by PCIe ordering rules. The tail
         * therefore lands behind the whole payload, making its signal/counter
         * mean the data is there and the source buffer safe to reuse.
         * A plain put announces nothing, so nothing can observe its chunks
         * out of order; the wait is deferred to the caller's later flush /
         * signaled put / barrier. The drain is whole-endpoint (outstanding
         * == 0, same loop as flushImplMode) and must be: the FI_WRITE
         * counter does not attribute completions to WQEs, so a fully
         * drained endpoint is the only state that proves THIS put's chunks
         * completed. That is required for correct signal delivery, even
         * though it also waits on concurrent posters' writes. */
        if (isIndexed || hasCounter) {
          cuda::atomic_ref<uint64_t, ncclGinScope<mode>> submitted_ref(dev->data.submitted_count);
          while (((((uint32_t)submitted_ref.load(cuda::memory_order_relaxed)) -
                   (uint32_t)hwCounterLoad(dev->data.local_cntr_value)) &
                  EFA_CNTR_MASK) != 0) {
            /* spin: leading chunks in flight */
          }
        }
        absSrcAddr += nLeading * cap;
        absDstAddr += nLeading * cap;
        writeBytes = (uint32_t)(bytes - nLeading * cap);
      }

      /* Final write: the payload tail (the WHOLE payload when no chunking
       * occurred above; hasPayload) or 0-byte scratch (signal-only),
       * on the counterId-selected poster so the local counter ticks once,
       * addressed to the resolved target so the receiver's FI_REMOTE_WRITE
       * fires once. absSrcAddr/absDstAddr/writeBytes already point at the
       * payload or the scratch region per the hasPayload branch above. */
      postRdmaOp<mode>(main_ep, main_ah, main_qpn, main_qkey, absSrcAddr, srcLkey, writeBytes, absDstAddr, dstRkey,
                       optFlags);

      /* Remaining (signalCount - 1) signal increments: 0-byte writes to
       * the peer scratch region on the DATA endpoint, so the caller's
       * counter is not over-incremented. The loop body is empty unless
       * signalCount > 1, which implies an INDEXED Add (and thus a
       * signal endpoint target). */
      for (uint32_t k = 1u; k < signalCount; k++) {
        postRdmaOp<mode>(&dev->data, dataSigAh, dataSigQpn, dataSigQkey, dev->scratch_local_addr, dev->scratch_lkey, 0u,
                         dev->scratch_remote_addrs[peer], dev->scratch_remote_rkeys[peer], optFlags);
      }
    }
  }
  (void)hasDescriptor;
  (void)descriptor;
  (void)required;
  (void)given;
  (void)optFlags;
  coop.sync();
}

/* ── putImpl: runtime mode dispatcher ─────────────────────────────── */

template <typename Coop>
NCCL_DEVICE_INLINE static void putImpl(ncclGinCtx ctx, Coop coop, int peer, bool hasWins, ncclGinWindow_t dstWin,
                                       size_t dstOff, ncclGinWindow_t srcWin, size_t srcOff, size_t bytes,
                                       ncclGinSignalDescriptor signal, ncclGinSignalOp_t signalOp, uint64_t signalOpArg,
                                       bool hasCounter, ncclGinCounter_t counterId, bool hasDescriptor,
                                       ncclGinDescriptorSmem* descriptor, cuda::thread_scope required,
                                       cuda::thread_scope given, uint32_t optFlags) {
  switch ((ncclGinResourceSharingMode)ctx.resourceSharingMode) {
  case NCCL_GIN_RESOURCE_SHARING_CTA:
    putImplMode<NCCL_GIN_RESOURCE_SHARING_CTA>(ctx, coop, peer, hasWins, dstWin, dstOff, srcWin, srcOff, bytes, signal,
                                               signalOp, signalOpArg, hasCounter, counterId, hasDescriptor, descriptor,
                                               required, given, optFlags);
    break;
  default:
    putImplMode<NCCL_GIN_RESOURCE_SHARING_GPU>(ctx, coop, peer, hasWins, dstWin, dstOff, srcWin, srcOff, bytes, signal,
                                               signalOp, signalOpArg, hasCounter, counterId, hasDescriptor, descriptor,
                                               required, given, optFlags);
    break;
  }
}

/* ── getImplMode: mode-templated Get implementation ──────────────── */

/* Fetches `bytes` from the peer's `remoteWin` into this rank's `localWin` with RDMA
 * reads on the data endpoint.
 *
 * Get is the reverse of Put and reuses the same post path: the RDMA address pair
 * carries the peer's window (the source the NIC reads) and the SGE carries the local
 * window (the destination it fills), with postRdmaOp's op selecting the
 * direction. The fetched bytes become observable to the caller once the read
 * completes, which the caller establishes with flush or flushAsync/wait. */
template <ncclGinResourceSharingMode mode, typename Coop>
NCCL_DEVICE_INLINE static void getImplMode(ncclGinCtx ctx, Coop coop, int peer, ncclGinWindow_t remoteWin,
                                           size_t remoteOff, ncclGinWindow_t localWin, size_t localOff, size_t bytes,
                                           uint32_t optFlags) {
  coop.sync();
  if (coop.thread_rank() == 0) {
    nccl_ofi_gin_gdaki_dev_handle* dev = getDevHandle(ctx);

    /* Both windows resolve to this context's rail, as in Put. The peer's window
     * supplies the remote address and rkey; this rank's window supplies the local
     * address and lkey. */
    nccl_ofi_gin_gdaki_mr_handle* remoteMh = ((nccl_ofi_gin_gdaki_mr_handle**)remoteWin)[dev->rail_id];
    nccl_ofi_gin_gdaki_mr_handle* localMh = ((nccl_ofi_gin_gdaki_mr_handle**)localWin)[dev->rail_id];
    uint64_t absRemoteAddr = remoteMh->peers[peer].remote_addr + remoteOff;
    uint32_t remoteRkey = remoteMh->peers[peer].rkey;
    uint64_t absLocalAddr = localMh->local_addr + localOff;
    uint32_t localLkey = localMh->lkey;

    /* Target slot 0 addresses the peer's data endpoint, which is the endpoint that
     * serves window memory. */
    const uint32_t targetIdx = 0u * (uint32_t)dev->nranks + (uint32_t)peer;
    const uint16_t ah = dev->data.target_address_handles[targetIdx];
    const uint16_t qpn = dev->data.target_remote_qpns[targetIdx];
    const uint32_t qkey = dev->data.target_qkey[targetIdx];

    /* One RDMA read carries at most EFA_GDA_MAX_RDMA_OP_SIZE, so a larger fetch splits
     * into full-size chunks plus a tail, advancing both sides together. A zero-byte
     * fetch posts one zero-length read rather than nothing, so that it still produces
     * a completion for flush and flushAsync/wait to terminate on. */
    const size_t cap = (size_t)EFA_GDA_MAX_RDMA_OP_SIZE;
    size_t remaining = bytes;
    uint64_t remoteAddr = absRemoteAddr;
    uint64_t localAddr = absLocalAddr;
    do {
      const size_t chunk = (remaining > cap) ? cap : remaining;
      postRdmaOp<mode, EFA_GDA_RDMA_READ>(&dev->data, ah, qpn, qkey, localAddr, localLkey, (uint32_t)chunk, remoteAddr,
                                          remoteRkey, optFlags);
      remoteAddr += chunk;
      localAddr += chunk;
      remaining -= chunk;
    } while (remaining > 0);
  }
  coop.sync();
}

/* ── getImpl: dispatch on the context's resource-sharing mode ────── */

template <typename Coop>
NCCL_DEVICE_INLINE static void getImpl(ncclGinCtx ctx, Coop coop, int peer, ncclGinWindow_t remoteWin, size_t remoteOff,
                                       ncclGinWindow_t localWin, size_t localOff, size_t bytes, uint32_t optFlags) {
  switch ((ncclGinResourceSharingMode)ctx.resourceSharingMode) {
  case NCCL_GIN_RESOURCE_SHARING_CTA:
    getImplMode<NCCL_GIN_RESOURCE_SHARING_CTA>(ctx, coop, peer, remoteWin, remoteOff, localWin, localOff, bytes,
                                               optFlags);
    break;
  default:
    getImplMode<NCCL_GIN_RESOURCE_SHARING_GPU>(ctx, coop, peer, remoteWin, remoteOff, localWin, localOff, bytes,
                                               optFlags);
    break;
  }
}

/* ── putValueImplMode: mode-templated PutValue implementation ─────── */

template <ncclGinResourceSharingMode mode, typename Coop, typename T>
NCCL_DEVICE_INLINE static void putValueImplMode(ncclGinCtx ctx, Coop coop, int peer, ncclGinWindow_t dstWin,
                                                size_t dstOff, T srcVal, ncclGinSignalDescriptor signal,
                                                ncclGinSignalOp_t signalOp, uint64_t signalOpArg, bool hasDescriptor,
                                                ncclGinDescriptorSmem* descriptor, cuda::thread_scope required,
                                                cuda::thread_scope given, uint32_t optFlags) {
  static_assert(sizeof(T) <= 8, "PutValue: T must fit in 8 bytes");
  coop.sync();
  if (coop.thread_rank() == 0) {
    nccl_ofi_gin_gdaki_dev_handle* dev = getDevHandle(ctx);

    /* This backend supports INDEXED signals only. */
    assert((signal.type == NCCL_GIN_SIGNAL_TYPE_NONE || signal.type == NCCL_GIN_SIGNAL_TYPE_INDEXED) &&
           "EFA GDA: only INDEXED signals are supported");

    /* Resolve the window to this context's rail. */
    nccl_ofi_gin_gdaki_mr_handle* dstMh = ((nccl_ofi_gin_gdaki_mr_handle**)dstWin)[dev->rail_id];

    /* All PutValues post from the dedicated PutValue endpoint (pvdata). */
    nccl_ofi_gin_gdaki_dev_endpoint_handle* ep = &dev->pvdata;

    /* Resolve the remote target (ah, qpn, qkey) via pvdata's target table
     * (targetSlot-major: idx = targetSlot*nranks + peer), same target slots
     * as Put:
     *     signalling write (INDEXED) -> slot 1 + signalId (peer sc EP,
     *       whose FI_REMOTE_WRITE the receiver's waitSignal observes)
     *     plain value write          -> slot 0 (peer DATA EP, no
     *       FI_REMOTE_WRITE bound, so no signal fires on the receiver)
     * The remote tuple is read through pvdata's own AV. */
    const bool isIndexed = (signal.type == NCCL_GIN_SIGNAL_TYPE_INDEXED);
    const uint32_t targetSlot = isIndexed ? (1u + (uint32_t)signal.indexedSignal.signalId) : 0u;
    const uint32_t targetIdx = targetSlot * (uint32_t)dev->nranks + (uint32_t)peer;
    uint16_t ah = ep->target_address_handles[targetIdx];
    uint16_t qpn = ep->target_remote_qpns[targetIdx];
    uint32_t qkey = ep->target_qkey[targetIdx];

    /* Signal increment count, mirroring Put: Inc (or no signal) is a single
     * write; an INDEXED Add-by-N expands into N inbound writes. signalOpArg
     * is defined to be 1 for Inc by the GIN API. */
    uint32_t signalCount = 1u;
    if (isIndexed && signalOp == ncclGinSignalAdd) {
      signalCount = (uint32_t)signalOpArg;
    }

    uint64_t absDstAddr = dstMh->peers[peer].remote_addr + dstOff;
    uint32_t dstRkey = dstMh->peers[peer].rkey;

    /* Value write: stage srcVal into pvdata's pool and RDMA-write it to the
     * destination. The arrival ticks the target sc EP's FI_REMOTE_WRITE once
     * (signalled) or no signal (no-signal). */
    postRdmaOp<mode>(ep, ah, qpn, qkey, /*srcAddr=*/0, /*srcLkey=*/0,
                     /*writeBytes=*/(uint32_t)sizeof(T), absDstAddr, dstRkey,
                     /*optFlags=*/ncclGinOptFlagsDefault,
                     /*pvSrcVal=*/&srcVal, /*pvValBytes=*/(uint32_t)sizeof(T),
                     /*pvLkey=*/dev->putvalue_lkey,
                     /*pvSliceBase=*/ep->putvalue_slice_base,
                     /*pvSlotSize=*/dev->putvalue_slot_size);

    /* Remaining (signalCount - 1) signal increments: 0-byte writes to
     * the peer scratch region on the DATA endpoint. The loop body is empty
     * unless signalCount > 1, which implies an INDEXED Add (and thus a
     * signal endpoint target). */
    for (uint32_t k = 1u; k < signalCount; k++) {
      postRdmaOp<mode>(ep, ah, qpn, qkey, dev->scratch_local_addr, dev->scratch_lkey, 0u,
                       dev->scratch_remote_addrs[peer], dev->scratch_remote_rkeys[peer]);
    }
  }
  (void)hasDescriptor;
  (void)descriptor;
  (void)required;
  (void)given;
  (void)optFlags;
  coop.sync();
}

/* ── putValueImpl: runtime mode dispatcher ────────────────────────── */

template <typename Coop, typename T>
NCCL_DEVICE_INLINE static void putValueImpl(ncclGinCtx ctx, Coop coop, int peer, ncclGinWindow_t dstWin, size_t dstOff,
                                            T srcVal, ncclGinSignalDescriptor signal, ncclGinSignalOp_t signalOp,
                                            uint64_t signalOpArg, bool hasDescriptor, ncclGinDescriptorSmem* descriptor,
                                            cuda::thread_scope required, cuda::thread_scope given, uint32_t optFlags) {
  switch ((ncclGinResourceSharingMode)ctx.resourceSharingMode) {
  case NCCL_GIN_RESOURCE_SHARING_CTA:
    putValueImplMode<NCCL_GIN_RESOURCE_SHARING_CTA>(ctx, coop, peer, dstWin, dstOff, srcVal, signal, signalOp,
                                                    signalOpArg, hasDescriptor, descriptor, required, given, optFlags);
    break;
  default:
    putValueImplMode<NCCL_GIN_RESOURCE_SHARING_GPU>(ctx, coop, peer, dstWin, dstOff, srcVal, signal, signalOp,
                                                    signalOpArg, hasDescriptor, descriptor, required, given, optFlags);
    break;
  }
}

/* ── flushImplMode: mode-templated Flush implementation ───────────── */

template <bool HasTimeout, ncclGinResourceSharingMode mode, typename Coop>
NCCL_DEVICE_INLINE static ncclResult_t flushImplMode(ncclGinCtx ctx, Coop coop, cuda::memory_order ord,
                                                     uint32_t* abortFlag, uint64_t timeoutCycles) {
  (void)ord;
  if NCCL_IF_CONSTEXPR (!HasTimeout) (void)timeoutCycles;

  coop.sync();
  ncclResult_t result = ncclSuccess;
  if (coop.thread_rank() == 0) {
    nccl_ofi_gin_gdaki_dev_handle* dev = getDevHandle(ctx);
    uint64_t startCycle = 0;
    if NCCL_IF_CONSTEXPR (HasTimeout) startCycle = clock64();
    else (void)startCycle; // referenced only when HasTimeout is true

    /* For each endpoint with outstanding work, spin on the NIC-written
     * FI_WRITE + FI_READ counter until it catches up with submitted_count.
     * submitted_count is re-read (scoped relaxed atomic load matching the
     * relaxed bumps from the post path) on every iteration rather than
     * snapshotted once: another thread may keep posting on this context
     * while we drain, and completions passing a stale snapshot would leave
     * the masked difference permanently non-zero. The HW counter is read
     * with system-scope acquire so the GPU bypasses caches and observes the
     * latest NIC update through PCIe-coherent memory. */
    auto wait_for_endpoint = [abortFlag, startCycle,
                              timeoutCycles](nccl_ofi_gin_gdaki_dev_endpoint_handle& ep) -> ncclResult_t {
      cuda::atomic_ref<uint64_t, ncclGinScope<mode>> target_ref(ep.submitted_count);

      /* Drain-to-zero: outstanding = (submitted - completed) reduced to
       * 31 bits, since the NIC FI_WRITE + FI_READ counter wraps at 2^31. Wait until
       * no work is outstanding. Outstanding is bounded by sq_size « 2^31,
       * so the masked difference is exact and cannot be fooled by a
       * counter wrap. */
      while (((((uint32_t)target_ref.load(cuda::memory_order_relaxed)) - (uint32_t)hwCounterLoad(ep.local_cntr_value)) &
              EFA_CNTR_MASK) != 0) {
        if NCCL_IF_CONSTEXPR (HasTimeout) {
          if (clock64() - startCycle >= timeoutCycles) return ncclTimeout;
        }
        if (abortFlag && *abortFlag) return ncclInProgress;
      }
      return ncclSuccess;
    };

    result = wait_for_endpoint(dev->data);
    if (result != ncclSuccess) goto done;

    /* The dedicated PutValue endpoint is a local poster too (all PutValue
     * writes ride it), so drain it as well. */
    result = wait_for_endpoint(dev->pvdata);
    if (result != ncclSuccess) goto done;

    /* Drain the counter endpoints only. With the decoupled model the
     * local poster QP is always the data endpoint, the PutValue endpoint,
     * or a counter endpoint (counterId selects the poster); a signal
     * endpoint is only ever a remote TARGET, never a local poster, so its
     * FI_WRITE counter never ticks from our writes and there is nothing to
     * drain. A signal QP needs no local completions at all. */
    for (int i = 0; i < dev->nCounters; i++) {
      result = wait_for_endpoint(dev->counter_handles[i]->base);
      if (result != ncclSuccess) goto done;
    }
  }
done:
  coop.sync();
  return (result == ncclInProgress) ? ncclSuccess : result;
}

/* ── flushImpl: runtime mode dispatcher ───────────────────────────── */

template <bool HasTimeout, typename Coop>
NCCL_DEVICE_INLINE static ncclResult_t flushImpl(ncclGinCtx ctx, Coop coop, cuda::memory_order ord, uint32_t* abortFlag,
                                                 uint64_t timeoutCycles) {
  switch ((ncclGinResourceSharingMode)ctx.resourceSharingMode) {
  case NCCL_GIN_RESOURCE_SHARING_CTA:
    return flushImplMode<HasTimeout, NCCL_GIN_RESOURCE_SHARING_CTA>(ctx, coop, ord, abortFlag, timeoutCycles);
  default:
    return flushImplMode<HasTimeout, NCCL_GIN_RESOURCE_SHARING_GPU>(ctx, coop, ord, abortFlag, timeoutCycles);
  }
}

} // namespace efa_gda
} // namespace gin
} // namespace nccl

/* ── Put ───────────────────────────────────────────────────────────── */

template <>
struct ncclGinApi_Put<NCCL_NET_DEVICE_GIN_EFA_GDA> {
  template <typename Coop>
  NCCL_DEVICE_INLINE static void call(ncclGinCtx ctx, Coop coop, int peer, bool hasWins, ncclGinWindow_t dstWin,
                                      size_t dstOff, ncclGinWindow_t srcWin, size_t srcOff, size_t bytes,
                                      ncclGinSignalDescriptor signal, ncclGinSignalOp_t signalOp, uint64_t signalOpArg,
                                      bool hasCounter, ncclGinCounter_t counterId, bool hasDescriptor,
                                      ncclGinDescriptorSmem* descriptor, cuda::thread_scope required,
                                      cuda::thread_scope given, uint32_t optFlags = ncclGinOptFlagsDefault) {
    nccl::gin::efa_gda::putImpl(ctx, coop, peer, hasWins, dstWin, dstOff, srcWin, srcOff, bytes, signal, signalOp,
                                signalOpArg, hasCounter, counterId, hasDescriptor, descriptor, required, given,
                                optFlags);
  }
};

/* ── PutValue ─────────────────────────────────────────────────────── */

template <>
struct ncclGinApi_PutValue<NCCL_NET_DEVICE_GIN_EFA_GDA> {
  template <typename Coop, typename T>
  NCCL_DEVICE_INLINE static void call(ncclGinCtx ctx, Coop coop, int peer, ncclGinWindow_t dstWin, size_t dstOff,
                                      T srcVal, ncclGinSignalDescriptor signal, ncclGinSignalOp_t signalOp,
                                      uint64_t signalOpArg, bool hasDescriptor, ncclGinDescriptorSmem* descriptor,
                                      cuda::thread_scope required, cuda::thread_scope given,
                                      uint32_t optFlags = ncclGinOptFlagsDefault) {
    nccl::gin::efa_gda::putValueImpl(ctx, coop, peer, dstWin, dstOff, srcVal, signal, signalOp, signalOpArg,
                                     hasDescriptor, descriptor, required, given, optFlags);
  }
};

/* ── Get ──────────────────────────────────────────────────────────── */

template <>
struct ncclGinApi_Get<NCCL_NET_DEVICE_GIN_EFA_GDA> {
  template <typename Coop>
  NCCL_DEVICE_INLINE static void call(ncclGinCtx ctx, Coop coop, int peer, ncclGinWindow_t remoteWin, size_t remoteOff,
                                      ncclGinWindow_t localWin, size_t localOff, size_t bytes, bool hasDescriptor,
                                      ncclGinDescriptorSmem* descriptor, uint32_t optFlags = ncclGinOptFlagsDefault) {
    (void)hasDescriptor;
    (void)descriptor;
    nccl::gin::efa_gda::getImpl(ctx, coop, peer, remoteWin, remoteOff, localWin, localOff, bytes, optFlags);
  }
};

/* ── FlushAsync ───────────────────────────────────────────────────── */

template <>
struct ncclGinApi_FlushAsync<NCCL_NET_DEVICE_GIN_EFA_GDA> {
  NCCL_DEVICE_INLINE static void call(ncclGinCtx ctx, int peer, ncclGinRequest_t* outRequest, bool hasDescriptor,
                                      ncclGinDescriptorSmem* descriptor, uint32_t optFlags) {
    (void)ctx;
    (void)peer;
    (void)outRequest;
    (void)hasDescriptor;
    (void)descriptor;
    (void)optFlags;
  }
};

/* ── Wait ─────────────────────────────────────────────────────────── */

template <>
struct ncclGinApi_Wait<NCCL_NET_DEVICE_GIN_EFA_GDA> {
  NCCL_DEVICE_INLINE static void call(ncclGinCtx ctx, ncclGinRequest_t& request, bool hasDescriptor,
                                      ncclGinDescriptorSmem* descriptor, cuda::memory_order ord, uint32_t* abortFlag) {
    (void)ctx;
    (void)request;
    (void)hasDescriptor;
    (void)descriptor;
    (void)ord;
    (void)abortFlag;
  }

  NCCL_DEVICE_INLINE static ncclResult_t call(ncclGinCtx ctx, ncclGinRequest_t& request, bool hasDescriptor,
                                              ncclGinDescriptorSmem* descriptor, cuda::memory_order ord,
                                              uint32_t* abortFlag, uint64_t timeoutCycles) {
    (void)ctx;
    (void)request;
    (void)hasDescriptor;
    (void)descriptor;
    (void)ord;
    (void)abortFlag;
    (void)timeoutCycles;
    return ncclSuccess;
  }
};

/* ── Flush ────────────────────────────────────────────────────────── */

template <>
struct ncclGinApi_Flush<NCCL_NET_DEVICE_GIN_EFA_GDA> {
  template <typename Coop>
  NCCL_DEVICE_INLINE static void call(ncclGinCtx ctx, Coop coop, bool hasDescriptor, ncclGinDescriptorSmem* descriptor,
                                      cuda::memory_order ord, uint32_t* abortFlag) {
    (void)hasDescriptor;
    (void)descriptor;
    (void)nccl::gin::efa_gda::flushImpl<false>(ctx, coop, ord, abortFlag, 0);
  }

  template <typename Coop>
  NCCL_DEVICE_INLINE static ncclResult_t call(ncclGinCtx ctx, Coop coop, bool hasDescriptor,
                                              ncclGinDescriptorSmem* descriptor, cuda::memory_order ord,
                                              uint32_t* abortFlag, uint64_t timeoutCycles) {
    (void)hasDescriptor;
    (void)descriptor;
    return nccl::gin::efa_gda::flushImpl<true>(ctx, coop, ord, abortFlag, timeoutCycles);
  };
};

/* ── SupportsStrongSignal ────────────────────────────────────────────────────────── */
template <>
struct ncclGinApi_SupportsStrongSignal<NCCL_NET_DEVICE_GIN_EFA_GDA> {
  NCCL_DEVICE_INLINE static bool call(ncclGinCtx) {
    return false;
  }
};

/* ── FlushesAllPutsOnAnySignal ───────────────────────────────────────────────────── */
template <>
struct ncclGinApi_FlushesAllPutsOnAnySignal<NCCL_NET_DEVICE_GIN_EFA_GDA> {
  // EFA is not MLX5 and does not flush previously-received puts on an unrelated signal, so the
  // capability is unconditionally absent.
  NCCL_DEVICE_INLINE static bool call(ncclGinCtx) {
    return false;
  }
};

/* ── GetSignalPtr ─────────────────────────────────────────────────── */

template <>
struct ncclGinApi_GetSignalPtr<NCCL_NET_DEVICE_GIN_EFA_GDA> {
  NCCL_DEVICE_INLINE static ncclGinOffsetPtr call(ncclGinCtx ctx, ncclGinSignal_t signalId) {
    nccl_ofi_gin_gdaki_dev_handle* dev = nccl::gin::efa_gda::getDevHandle(ctx);
    nccl_ofi_gin_gdaki_dev_counter_handle* h = dev->signal_handles[signalId];
    return {(uint64_t*)h->cntr_value, h->cntr_offset};
  }
};

/* ── GetCounterPtr ────────────────────────────────────────────────── */

template <>
struct ncclGinApi_GetCounterPtr<NCCL_NET_DEVICE_GIN_EFA_GDA> {
  NCCL_DEVICE_INLINE static ncclGinOffsetPtr call(ncclGinCtx ctx, ncclGinCounter_t counterId) {
    nccl_ofi_gin_gdaki_dev_handle* dev = nccl::gin::efa_gda::getDevHandle(ctx);
    nccl_ofi_gin_gdaki_dev_counter_handle* h = dev->counter_handles[counterId];
    return {(uint64_t*)h->cntr_value, h->cntr_offset};
  }
};

/* ── ResetSignal ──────────────────────────────────────────────────── */

template <>
struct ncclGinApi_ResetSignal<NCCL_NET_DEVICE_GIN_EFA_GDA> {
  NCCL_DEVICE_INLINE static void call(ncclGinCtx ctx, ncclGinSignalDescriptor signal) {
    nccl_ofi_gin_gdaki_dev_handle* dev = nccl::gin::efa_gda::getDevHandle(ctx);
    assert(signal.type == NCCL_GIN_SIGNAL_TYPE_INDEXED && "EFA GDA ResetSignal: only INDEXED signals are supported");
    assert((int)signal.indexedSignal.signalId < dev->nSignals && "EFA GDA ResetSignal: signalId out of range");
    /* Offset-based reset: the NIC counter cannot be written, so snapshot
     * its current value into cntr_offset. Subsequent reads/waits subtract
     * the offset, making the signal appear reset. */
    nccl_ofi_gin_gdaki_dev_counter_handle* h = dev->signal_handles[signal.indexedSignal.signalId];
    h->cntr_offset = nccl::gin::efa_gda::hwCounterLoad((uint64_t*)h->cntr_value);
  }
};

/* ── ResetCounter ─────────────────────────────────────────────────── */

template <>
struct ncclGinApi_ResetCounter<NCCL_NET_DEVICE_GIN_EFA_GDA> {
  NCCL_DEVICE_INLINE static void call(ncclGinCtx ctx, ncclGinCounter_t counterId) {
    nccl_ofi_gin_gdaki_dev_handle* dev = nccl::gin::efa_gda::getDevHandle(ctx);
    assert((int)counterId < dev->nCounters && "EFA GDA ResetCounter: counterId out of range");
    /* Offset-based reset: snapshot the NIC counter into cntr_offset
     * instead of writing the (NIC-owned) counter. */
    nccl_ofi_gin_gdaki_dev_counter_handle* h = dev->counter_handles[counterId];
    h->cntr_offset = nccl::gin::efa_gda::hwCounterLoad((uint64_t*)h->cntr_value);
  }
};

#endif /* _NCCL_DEVICE_GIN_EFA_GDA_H_ */
