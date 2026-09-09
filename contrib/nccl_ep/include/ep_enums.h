/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 * See LICENSE.txt for more license information.
 */

#pragma once

// Auto configuration constant for dynamic/automatic sizing
#define NCCL_EP_AUTO 0

// Communication algorithm (mode)
typedef enum {
    // Low-Latency (LL) mode
    NCCL_EP_ALGO_LOW_LATENCY = 0,
    // High-Throughput (HT) mode
    NCCL_EP_ALGO_HIGH_THROUGHPUT = 1
} ncclEpAlgorithm_t;

/**
 * Dispatch output layout
 *
 * Controls the shape and contents of the user-visible dispatch output tensors
 * (tokens, topk_weights, topk_idx) and their expected shape in Combine.
 *
 * Each layout is supported by a subset of algorithms:
 *   HT: NCCL_EP_LAYOUT_FLAT, NCCL_EP_LAYOUT_EXPERT_MAJOR.
 *   LL: NCCL_EP_LAYOUT_EXPERT_MAJOR, NCCL_EP_LAYOUT_RANK_MAJOR.
 */
typedef enum {
    /**
     * Sentinel for "layout not set" (zero-init default).
     *
     * Callers must override this with one of the layout values below before
     * passing the config to ncclEpInitHandle / ncclEpHandleMemSize; the
     * library does not auto-resolve based on algorithm. Leaving the field at
     * NCCL_EP_LAYOUT_UNSET is a programmer error and trips an assertion at
     * handle-init time.
     */
    NCCL_EP_LAYOUT_UNSET = NCCL_EP_AUTO,

    /**
     * Expert-major layout.
     *
     * LL dispatch output:
     *   tokens shape:            [num_local_experts, max_dispatch_tokens_per_rank * num_ranks, hidden]  (3D)
     *   topk_weights shape:       nullptr (not populated under LL EM).
     *   topk_idx shape:           nullptr (slot index encodes expert).
     *
     * HT dispatch output:
     *   tokens shape:            [num_recv_slots, hidden] 2D; expert grouping is internal ordering
     *   topk_weights shape:      [num_recv_slots] 1D, one weight per slot, where
     *                                each slot is per (source_token, local_expert), at most one match)
     *   topk_idx shape:          nullptr (slot index encodes expert).
     *
     *   Per-expert zones inside the tokens tensor are padded per
     *   ncclEpHandleConfig_t::dispatch_output_per_expert_alignment so each
     *   expert's token sequence start at an aligned offset.
     *
     * Combine input shape mirrors the dispatch output (post-expert activation
     * occupies the same slots).
     *
     * Each expert rank sends its post-expert activation back to the originating rank.
     *   * In HT mode, the user is expected to apply topk weights before passing the post-expert
     *     activations to ncclEpCombine.
     *   * In LL mode, the Combine kernel is applying the weights on the fly.
     * For all modes, the Combine kernel performs accumulation of up to num_topk
     * expert contributions, returning a single output token per input slot.
     */
    NCCL_EP_LAYOUT_EXPERT_MAJOR,

    /**
     * Rank-major layout  (LL mode only).
     *
     * Dispatch output:
     *   tokens shape:            [num_ranks, max_dispatch_tokens_per_rank, hidden] 3D
     *   topk_weights shape:      [num_ranks, max_dispatch_tokens_per_rank, num_topk] 3D
     *   topk_idx shape:          [num_ranks, max_dispatch_tokens_per_rank, num_topk] 3D
     *
     * Tokens arrive in rank-major order with no expert dimension.
     * The caller is responsible for running expert computation on each token
     * slot and pre-reducing across local experts using the per-expert weights
     * from topk_weights, producing one weighted output vector per slot.
     *
     * ncclEpCombine sends these pre-reduced vectors back to each token's home
     * rank. The home rank still performs a receive-side reduction, but it
     * accumulates one contribution per source expert rank (not per expert),
     * without applying the weights.
     */
    NCCL_EP_LAYOUT_RANK_MAJOR,

    /**
     * Flat layout (HT mode only).
     *
     * Dispatch output:
     *   tokens shape:            [num_recv_slots, hidden] 2D
     *   topk_weights shape:      [num_recv_slots, num_topk] 2D
     *   topk_idx shape:          [num_recv_slots, num_topk] 2D
     *
     * Combine input shape:       [num_recv_slots, hidden]
     *
     * num_recv_slots is the recv-slot dimension chosen by the caller and must
     * be no smaller than the actual number of tokens this rank will receive.
     *   - Static (the only mode supported in v0.1): choose a worst-case
     *     num_recv_slots (e.g. max_recv_tokens_per_rank) to guard against
     *     routing dynamism.
     *   - Query-then-allocate (PLANNED, NOT YET SUPPORTED IN v0.1): supply
     *     ncclEpLayoutInfo_t::recv_total_counter to either `ncclEpCreateHandle`
     *     or `ncclEpUpdateHandle`;
     *     the metadata kernel writes the actual recv count there, which the caller
     *     reads (requires a GPU→CPU sync) before sizing tokens. This depends on
     *     max_dispatch_tokens_per_rank = NCCL_EP_AUTO, which ncclEpCreateGroup
     *     currently rejects for HT.
     *
     * Tokens arrive as a single contiguous sequence with no rank-major or
     * expert-major structure.  The caller uses recv_topk_idx to route each
     * slot to the appropriate local expert(s) and topk_weights to apply
     * the weighted reduction before passing pre-reduced outputs to
     * ncclEpCombine.
     *
     */
    NCCL_EP_LAYOUT_FLAT,
} ncclEpLayout_t;

/**
 * Training pass direction for ncclEpDispatch / ncclEpCombine.
 *
 * Selects which side of the routing the call is participating in. FWD is the
 * zero-init default so callers that don't set the field get forward-pass
 * semantics.
 *
 *   FWD dispatch (HT): caller provides input topk_weights; routing is live.
 *   BWD dispatch (HT): no input topk_weights; reuses cached routing state.
 *   FWD combine  (HT): no input topk_weights; tokens only.
 *   BWD combine  (HT): caller provides input topk_weights and receives
 *                      combined topk_weights gradients alongside tokens.
 *
 * LL mode does not currently distinguish FWD/BWD and ignores this field.
 */
typedef enum {
    NCCL_EP_FWD_PASS = 0,
    NCCL_EP_BWD_PASS = 1,
} ncclEpPassDir_t;

// Zero-copy mode for dispatch / combine staging.
//   AUTO -- library picks (today: OFF).
//   OFF  -- always stage through library-owned buffers.
//   ON   -- skip staging; caller's tensors must be window-backed.
typedef enum {
    NCCL_EP_ZERO_COPY_AUTO = NCCL_EP_AUTO,
    NCCL_EP_ZERO_COPY_OFF,
    NCCL_EP_ZERO_COPY_ON
} ncclEpZeroCopyMode_t;

/**
 * Numbering convention for the recv_topk_idx output of dispatch.
 *
 * Applies to layouts that populate recv_topk_idx (LL rank-major, HT flat).
 * Selects whether per-slot expert ids are written in this rank's local
 * numbering or in the global numbering used on the wire.
 *
 *   AUTO   (zero-init default): library picks the historical default, which
 *          today resolves to LOCAL. Existing callers that leave the field
 *          zero-initialized stay on whatever the library currently considers
 *          the default; the resolved value may change in a future release
 *          without an ABI break. Callers that need a stable contract should
 *          pin LOCAL or GLOBAL explicitly.
 *   LOCAL: expert id in [0, num_local_experts), i.e.
 *          global_id - rank * num_local_experts. Slots not routed to this
 *          rank are -1.
 *   GLOBAL: unmodified wire-format global expert id in
 *          [rank*num_local_experts, (rank+1)*num_local_experts) for the
 *          local experts on this rank. Slots not routed to this rank are -1.
 *
 * The zero-init AUTO default keeps source/binary backward compatibility for
 * callers built against the pre-flag headers (the field reads as 0 and the
 * library treats it as today's behavior). Downstream consumers that prefer
 * the global id (e.g. fused MoE routers that want to skip the per-element
 * +rank*num_local_experts fold-back) opt in by setting GLOBAL on the layout
 * info struct.
 */
typedef enum {
    NCCL_EP_EXPERT_ID_AUTO = NCCL_EP_AUTO,
    NCCL_EP_EXPERT_ID_LOCAL = 1,
    NCCL_EP_EXPERT_ID_GLOBAL = 2,
} ncclEpExpertIdKind_t;
