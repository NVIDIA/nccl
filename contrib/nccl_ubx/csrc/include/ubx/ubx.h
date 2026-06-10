/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef UBX_UBX_H_
#define UBX_UBX_H_

#include <stddef.h>
#include <stdint.h>

#include <cuda_runtime.h>
#include <nccl.h>

#ifdef __cplusplus
extern "C" {
#endif

// Forward-declare the NCCL device-comm type. The full definition lives in
// <nccl_device/core.h>; downstream code that actually dereferences a
// ncclDevComm needs that header. Our public ABI only takes pointers
// (ncclDevComm_t const*), so an incomplete-type forward declaration is
// sufficient at this header layer.
#ifndef NCCL_DEV_COMM_FWD_DECLARED
#define NCCL_DEV_COMM_FWD_DECLARED
struct ncclDevComm;
typedef struct ncclDevComm ncclDevComm_t;
#endif

// Public ABI for the UB-X kernel launchers.
//
// The public surface takes a (ncclDevComm_t const* devcomm,
// ncclWindow_t window, uintptr_t pool_ptr) triple plus per-tensor byte
// offsets. The launcher resolves any required multicast pointers via
// ncclGetLsaMultimemDevicePointer(window, off, &out) and passes them
// to the kernel.
//
// Conventions:
//  - pool_ptr is the local-VA pool base (output of ncclMemAlloc). Used by
//    the kernels for REG0 commbuff[]/sync-flag access. Phase 4 cleanup:
//    drop pool_ptr once all kernels resolve via window.
//  - {in,out,scale,...}_offset are byte offsets within the registered
//    window where the corresponding tensor lives (i.e.,
//    `tensor.data_ptr() - pool_ptr`).
//  - Local non-pool-resident pointers (gamma, residual, ptr_in, etc.)
//    stay as uintptr_t — the kernel uses them directly.

void ubx_allreduce_2shot_mc(
    int ranks, int myrank,
    ncclDevComm_t const* devcomm, ncclWindow_t window, uintptr_t pool_ptr,
    size_t in_offset, size_t out_offset, size_t bytes,
    uintptr_t residual_in, uintptr_t residual_out, bool fuse_layernorm,
    uintptr_t gamma, float eps, const int hidden_size,
    int default_sms, int smlimit, int cgasize, int nchunk, bool multi_kernel,
    cudaStream_t stream);

void ubx_allreduce_2shot_uc(
    int ranks, int myrank,
    ncclDevComm_t const* devcomm, ncclWindow_t window, uintptr_t pool_ptr,
    size_t in_offset, size_t out_offset, size_t bytes,
    uintptr_t residual_in, uintptr_t residual_out, bool fuse_layernorm,
    uintptr_t gamma, float eps, const int hidden_size,
    int default_sms, int smlimit, int cgasize, int nchunk, bool multi_kernel,
    cudaStream_t stream);

void ubx_allreduce_2shot_mc_lamport(
    int ranks, int myrank,
    ncclDevComm_t const* devcomm, ncclWindow_t window, uintptr_t pool_ptr,
    uintptr_t ucptr_out, size_t in_offset, size_t out_offset,
    uintptr_t clear_ptr, size_t bytes, bool poisoned,
    uintptr_t residual_in, uintptr_t residual_out, bool fuse_layernorm,
    uintptr_t gamma, float eps, const int hidden_size,
    int default_sms, int smlimit, int cgasize, int nchunk, bool multi_kernel,
    cudaStream_t stream);

void ubx_allgather_mc(
    int ranks, int myrank,
    ncclDevComm_t const* devcomm, ncclWindow_t window, uintptr_t pool_ptr,
    uintptr_t ptr_in, size_t out_offset, size_t bytes,
    int default_sms, int smlimit, cudaStream_t stream);

// UC allgather: same I/O contract as ubx_allgather_mc but uses per-peer
// unicast writes + UC barrier. Required when multicast is unavailable
// (NVLS-disabled EP groups, large EP groups (>36 ranks), etc.).
void ubx_allgather_uc(
    int ranks, int myrank,
    ncclDevComm_t const* devcomm, ncclWindow_t window, uintptr_t pool_ptr,
    uintptr_t ptr_in, size_t out_offset, size_t bytes,
    int default_sms, int smlimit, cudaStream_t stream);

// alltoall: smlimit is set-exact (not cap-only) for runtime --smlimit sweeps;
// nthreads sets threads-per-block exactly when non-zero.
void ubx_alltoall(
    int ranks, int myrank,
    ncclDevComm_t const* devcomm, ncclWindow_t window, uintptr_t pool_ptr,
    uintptr_t ptr_in, size_t out_offset, size_t bytes,
    int default_sms, int smlimit, int nthreads, cudaStream_t stream);

// Variable-length alltoall. All offsets and counts are in BYTES. Kernel does
// uint4 (16 B) bulk loop + ushort (2 B) tail per source to support arbitrary
// per-destination byte counts (no 16 B alignment requirement). All element
// types in scope (bf16, fp16, fp32, fp64) are 2 B aligned, so tail handling
// in 2 B units is always sufficient.
//
// send_byte_offsets[r]: byte offset in ptr_in for data going to rank r.
// send_byte_counts[r]:  byte count to send to rank r.
// dest_byte_offsets[r]: byte offset in rank r's pool where data from myrank lands.
void ubx_alltoallv(
    int ranks, int myrank,
    ncclDevComm_t const* devcomm, ncclWindow_t window, uintptr_t pool_ptr,
    uintptr_t ptr_in,
    uintptr_t send_byte_offsets, uintptr_t send_byte_counts, uintptr_t dest_byte_offsets,
    int default_sms, int smlimit, cudaStream_t stream);

// alltoall_lamport: smlimit is cap-only; nthreads sets threads-per-block
// exactly when non-zero.
void ubx_alltoall_lamport(
    int ranks, int myrank,
    ncclDevComm_t const* devcomm, ncclWindow_t window, uintptr_t pool_ptr,
    uintptr_t ptr_in, size_t out_offset, uintptr_t clear_ptr,
    size_t bytes, bool poisoned,
    int default_sms, int smlimit, int nthreads, bool skip_barrier,
    cudaStream_t stream);

// Token-dispatch with bf16 -> mxfp8 quantization.
// token_offsets: int32[ntokens * ranks * experts_per_rank], dest slot or -1.
// Output fp8 data lands at commbuff[dest_rank] + lineoffset_out (uint4).
// Output E8M0 scales land at commbuff[dest_rank] + lineoffset_scales*16
// (1 byte/block).
// sync=1: synchronous, kernel polls until all ranks complete.
// sync=0: asynchronous, kernel returns after signalling barrier flag;
//         caller must invoke ubx_a2av_wait() before reading output.
void ubx_a2av_token_bf16_mxfp8(
    int ranks, int myrank, int ntokens, int blocks_per_token,
    int experts_per_rank, uintptr_t token_offsets,
    int64_t lineoffset_out, int64_t lineoffset_scales,
    ncclDevComm_t const* devcomm, ncclWindow_t window, uintptr_t pool_ptr,
    uintptr_t ptr_in_bf16,
    int default_sms, int smlimit, int sync,
    int expert_start, int expert_count,
    cudaStream_t stream);

// Out-of-place token-dispatch bf16 -> bf16 (no quantization).
// Same routing scheme as the mxfp8 variant but writes the raw bf16 block to
// the destination instead of fp8 + E8M0 scale. Output stride per slot is
// blocks_per_token*4 uint4 (= hidden bf16 elements / 4 per uint4).
void ubx_a2av_token_bf16_bf16(
    int ranks, int myrank, int ntokens, int blocks_per_token,
    int experts_per_rank, uintptr_t token_offsets,
    int64_t lineoffset_out,
    ncclDevComm_t const* devcomm, ncclWindow_t window, uintptr_t pool_ptr,
    uintptr_t ptr_in_bf16,
    int default_sms, int smlimit, int sync,
    int expert_start, int expert_count,
    cudaStream_t stream);

// Top-K LUT variant of bf16->bf16 token dispatch. K-loop runs only
// topk_max iterations per token (vs total_experts in the base launcher).
// Pre-computed topk_expert + topk_slot LUTs replace token_offsets.
void ubx_a2av_token_bf16_bf16_topk(
    int ranks, int myrank, int ntokens, int blocks_per_token,
    int experts_per_rank, int topk_max,
    uintptr_t topk_expert, uintptr_t topk_slot,
    int64_t lineoffset_out,
    ncclDevComm_t const* devcomm, ncclWindow_t window, uintptr_t pool_ptr,
    uintptr_t ptr_in_bf16,
    int default_sms, int smlimit, int sync,
    cudaStream_t stream);

// Wait for async a2av dispatch to complete on all ranks.
// Pairs with ubx_a2av_token_bf16_mxfp8(..., sync=0) or _bf16(..., sync=0).
void ubx_a2av_wait(
    int ranks,
    ncclDevComm_t const* devcomm, ncclWindow_t window, uintptr_t pool_ptr,
    cudaStream_t stream);

// Persistent (chunked) token-dispatch bf16 -> mxfp8.
// One kernel launch processes `nchunks` chunks internally, each with its
// own cross-rank barrier. Per-chunk barriers bump A2AV_ID by 1 and issue
// one ATOMIC_MCINC on BAR — same protocol as N non-persistent dispatches.
// nchunks must be in [1, 32].
void ubx_a2av_token_bf16_mxfp8_persistent(
    int ranks, int myrank, int ntokens, int blocks_per_token,
    int experts_per_rank, uintptr_t token_offsets,
    int64_t lineoffset_out, int64_t lineoffset_scales,
    ncclDevComm_t const* devcomm, ncclWindow_t window, uintptr_t pool_ptr,
    uintptr_t ptr_in_bf16,
    int default_sms, int smlimit,
    int nchunks, int nexperts_per_chunk,
    cudaStream_t stream);

// MoE token-combine: reverse of a2av_token dispatch. Top-K weighted sum
// per token, in bf16 local accumulation. Two flavors (bf16-wire and
// mxfp8-wire) sharing a two-phase PULL design:
//   Phase 1 (local): kernel reads bf16 expert outputs from ptr_in_bf16
//     and writes them into the temp symm buffer at lineoffset_temp
//     (bf16 wire = memcpy; mxfp8 wire = quantize to fp8 + E8M0 scales).
//   Cross-rank barrier on UBX_FLAG_COMBINE_BAR.
//   Phase 2 (remote): kernel iterates each local token's K experts via
//     token_offsets[t, e], PULLS bf16/fp8 from peer pools, dequant
//     (mxfp8 only), multiplies by gate_weights[t, e] if non-NULL,
//     fp32-accumulates, downcasts, writes to ptr_out_bf16.
//
// gate_weights: optional float [local_ntokens, total_experts]; pass 0
//   to disable the multiply. Sparse layout matches token_offsets — kernel
//   skips entries where token_offsets[t, e] < 0.
// max_tokens_per_rank: from compute_token_offsets(); the temp symm buffer
//   first dimension. Slots [0, max_tokens_per_rank) on every rank.
// sync=1: kernel polls COMBINE_BAR until all ranks finish; sync=0: signal
//   and return, caller invokes ubx_combine_wait().
void ubx_combine_bf16_bf16(
    int ranks, int myrank, int local_ntokens, int n_recv,
    int blocks_per_token,
    int experts_per_rank, int max_tokens_per_rank,
    uintptr_t token_offsets, uintptr_t gate_weights,
    int64_t lineoffset_temp,
    ncclDevComm_t const* devcomm, ncclWindow_t window, uintptr_t pool_ptr,
    uintptr_t ptr_in_bf16, uintptr_t ptr_out_bf16,
    int default_sms, int smlimit, int sync,
    cudaStream_t stream);

// Standalone barrier — no data movement, separate flag set
// (UBX_FLAG_BARRIER_*). Used for surrounding other UBX kernel calls
// with a hard sync that doesn't share flag slots with the data kernels.
void ubx_barrier(
    int ranks, int myrank,
    ncclDevComm_t const* devcomm, ncclWindow_t window, uintptr_t pool_ptr,
    int default_sms, int smlimit,
    cudaStream_t stream);

// Diagnostic: dump peer LSA pointers (one per rank) into a uint64 device
// buffer. After cudaStreamSynchronize, host can memcpy buf back and check
// values per rank for consistency. Used to verify NCCL's peer-ptr cache
// is populated identically across the EP group before the first UBX kernel.
void ubx_peer_ptr_dump(
    int ranks,
    ncclDevComm_t const* devcomm, ncclWindow_t window,
    uintptr_t out_ptrs,
    cudaStream_t stream);

// Diagnostic: minimal cross-rank peer atomic-inc test. Each rank's thread i
// issues ATOMIC_UCINC on peer i's UBX_FLAG_DIAG_PEER_TEST slot, then thread
// 0 polls own flag until peer atomics arrive (test_id * RANKS expected).
// Same raw cross-rank UC atomic pattern as combine's lastSM signaling, but
// isolated from combine's other logic. Used to test whether the cross-rank
// UC atomic mechanism itself is broken for the first EP group.
void ubx_peer_atomic_test(
    int ranks, int myrank, int test_id,
    ncclDevComm_t const* devcomm, ncclWindow_t window, uintptr_t pool_ptr,
    cudaStream_t stream);

// push3: 3-kernel PUSH combine (bf16 wire). Chained on the same stream.
// Phase 1 = many-CTA cross-rank NVLink writes (push to peers' dest bufs).
// Phase 2 = 1-CTA RANKS-thread cross-rank UCINC + spin-on-own-flag (with
//           compile-in timeout — only kernel that can hang).
// Phase 3 = many-CTA purely local sum + write (no NVLink, cannot hang).
// reduce_id is host-tracked.
void ubx_combine_push3_phase1_write(
    int ranks, int n_recv, int blocks_per_token, int topk_max,
    uintptr_t inverse_map, int max_tokens_per_rank, int64_t lineoffset_dest,
    ncclDevComm_t const* devcomm, ncclWindow_t window,
    uintptr_t ptr_in_bf16,
    int default_sms, int smlimit,
    cudaStream_t stream);
void ubx_combine_push3_phase2_signal(
    int ranks,
    ncclDevComm_t const* devcomm, ncclWindow_t window, uintptr_t pool_ptr,
    cudaStream_t stream);
void ubx_combine_push3_phase3_sum(
    int local_ntokens, int blocks_per_token, int topk_max, int total_experts,
    uintptr_t topk_idx, uintptr_t gate_weights,
    int64_t lineoffset_dest, uintptr_t pool_ptr,
    uintptr_t ptr_out_bf16,
    int default_sms, int smlimit,
    cudaStream_t stream);

// E22: combine v2 — true 2-kernel split (bf16 wire). Phase 1 = pure local
// data copy (no synchronization, kernel exit + stream order is the sync).
// Phase 2 = PDL grid-dependency-sync, CTA 0 issues RANKS peer ATOMIC_UCINCs
// to UBX_FLAG_COMBINE2_BAR, all CTAs poll until BAR >= reduce_id*RANKS,
// then run combine_phase2_bf16 (peer pull + fp32 sum). `reduce_id` is
// host-tracked (per-call counter).
void ubx_combine_v2_phase1_bf16(
    int n_recv, int blocks_per_token,
    uintptr_t pool_ptr, int64_t lineoffset_temp,
    uintptr_t ptr_in_bf16,
    int default_sms, int smlimit,
    cudaStream_t stream);

void ubx_combine_v2_phase2_bf16(
    int ranks, int reduce_id, int local_ntokens, int blocks_per_token,
    int experts_per_rank,
    uintptr_t token_offsets, uintptr_t gate_weights,
    int64_t lineoffset_temp,
    ncclDevComm_t const* devcomm, ncclWindow_t window, uintptr_t pool_ptr,
    uintptr_t ptr_out_bf16,
    int default_sms, int smlimit,
    cudaStream_t stream);

// mxfp8-wire combine. Phase 1 quantizes ptr_in_bf16 (the bf16 expert
// outputs) into the temp symm buffer at lineoffset_temp (fp8 data) +
// lineoffset_scales*16 (E8M0 scale bytes), using the same packing as
// ubx_a2av_token_bf16_mxfp8.
void ubx_combine_mxfp8_bf16(
    int ranks, int myrank, int local_ntokens, int n_recv,
    int blocks_per_token,
    int experts_per_rank, int max_tokens_per_rank,
    uintptr_t token_offsets, uintptr_t gate_weights,
    int64_t lineoffset_temp, int64_t lineoffset_scales,
    ncclDevComm_t const* devcomm, ncclWindow_t window, uintptr_t pool_ptr,
    uintptr_t ptr_in_bf16, uintptr_t ptr_out_bf16,
    int default_sms, int smlimit, int sync,
    cudaStream_t stream);

// PUSH-semantics combine, barrier-based (NOT Lamport).  Same PUSH routing
// as ubx_combine_bf16_bf16_lamport_push but Phase 2 uses a single
// cross-rank ATOMIC_MCINC barrier instead of per-element Lamport poll —
// avoids the polling overhead at large message sizes.  Caller rotates 2
// dest bufs (double-buffered) so call N+1 Phase 1 doesn't race with call
// N Phase 2 read of the same buf.  No warmup/poison needed.
void ubx_combine_bf16_bf16_push(
    int ranks, int myrank, int local_ntokens, int n_recv,
    int blocks_per_token, int experts_per_rank, int max_tokens_per_rank,
    int topk_max,
    uintptr_t inverse_map, uintptr_t topk_idx, uintptr_t gate_weights,
    int64_t lineoffset_dest,
    ncclDevComm_t const* devcomm, ncclWindow_t window, uintptr_t pool_ptr,
    uintptr_t ptr_in_bf16, uintptr_t ptr_out_bf16,
    int default_sms, int smlimit,
    cudaStream_t stream);

// Lamport-poll combine PUSH (bf16 wire only). Each writer pushes to peer's dest buf,
// reader polls OWN buf. Caller owns triple buffering (3 dest bufs of
// shape [local_ntokens, topk_max, hidden] bf16 rotated `call_n%3`),
// passes the current one's lineoffset_dest + the (call_n+2)%3 buf's
// address as clear_ptr, plus the inverse routing map and topk_idx
// computed via compute_combine_push_map (ubx/ops.py).
void ubx_combine_bf16_bf16_lamport_push(
    int ranks, int myrank, int local_ntokens, int n_recv,
    int blocks_per_token, int experts_per_rank, int max_tokens_per_rank,
    int topk_max,
    uintptr_t inverse_map, uintptr_t topk_idx, uintptr_t gate_weights,
    int64_t lineoffset_dest, uintptr_t clear_ptr,
    bool poisoned, bool skip_warmup_barrier,
    ncclDevComm_t const* devcomm, ncclWindow_t window, uintptr_t pool_ptr,
    uintptr_t ptr_in_bf16, uintptr_t ptr_out_bf16,
    int default_sms, int smlimit,
    cudaStream_t stream);

// Wait kernels for async combine. Pair with ubx_combine_*(..., sync=0):
// the main kernel did Phase 1 + signal + return; these poll the COMBINE_BAR
// flag and run Phase 2 themselves (read peer temp symm, gate-mul, sum,
// downcast bf16, write local output). Caller must keep the temp symm buffer
// (referenced by lineoffset_temp / lineoffset_scales) alive until after this
// kernel completes.
void ubx_combine_wait_bf16(
    int ranks, int local_ntokens, int blocks_per_token,
    int experts_per_rank,
    uintptr_t token_offsets, uintptr_t gate_weights,
    int64_t lineoffset_temp,
    ncclDevComm_t const* devcomm, ncclWindow_t window, uintptr_t pool_ptr,
    uintptr_t ptr_out_bf16,
    int default_sms, int smlimit,
    cudaStream_t stream);

void ubx_combine_wait_mxfp8(
    int ranks, int local_ntokens, int blocks_per_token,
    int experts_per_rank,
    uintptr_t token_offsets, uintptr_t gate_weights,
    int64_t lineoffset_temp, int64_t lineoffset_scales,
    ncclDevComm_t const* devcomm, ncclWindow_t window, uintptr_t pool_ptr,
    uintptr_t ptr_out_bf16,
    int default_sms, int smlimit,
    cudaStream_t stream);

// Set the kernel-side polling timeout in GPU clocks. Called once at
// SymmAllocator construction with int(UBX_TIMEOUT_SEC * 2e9). When the
// extension is built without UBX_BUILD_TIMEOUT, this is a no-op stub —
// the binding stays callable so Python doesn't need to know about the
// build mode. See csrc/ubx.cu / setup.py.
void ubx_set_timeout(unsigned long long clocks);

#ifdef __cplusplus
}
#endif

#endif  // UBX_UBX_H_
