#ifndef PACE_RS_KERNELS_CUH
#define PACE_RS_KERNELS_CUH

// Kernel template + device helpers extracted from rs.cu so that
// scripts/gen_rs_inst.py can emit per-instantiation .cu files that include
// this header (mirrors ep_kernels.cuh / gen_ep_inst.py). The host dispatcher
// (reduce_scatter_func) remains in rs.cu.

#include "util/error.hpp"
#include "collective/rs/rs.cuh"
#include "device/comm.cuh"
#include "util/math.hpp"
#include "device/mem.cuh"
#include "device/configs.cuh"
#include "device/launch.cuh"
#include "collective/rs/rs_defs.cuh"
#include <cstdint>
#include <cstdio>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <nccl_device/coop.h>
#include <nccl_device/core.h>
#include <nccl_device/gin.h>
#include <nccl_device/gin/gdaki/gin_gdaki.h>
#include <type_traits>
#include <cooperative_groups.h>

#ifdef PACE_TIMEOUT_DEBUG
// Override the default no-op timeout hook so that every LL_READ_64GE /
// LL_READ_64GE_STEP / LL_READ_64GE_STEP2 timeout in this kernel additionally
// dumps the NVL + rail-aligned arrival bitmap for the current round_n. All
// referenced names are in scope at every wait site in the kernel body below.
#undef LL_TIMEOUT_DUMP
#define LL_TIMEOUT_DUMP() \
    ::dump_arrival_status(devComm, num_local_ranks, num_nodes, \
                          local_rank, node_id, \
                          arrival_sig_base, round_n, rank, (int)blockIdx.x, \
                          cuda_device_id)
#endif

namespace pace {
constexpr int kRSRingThreads = 1024;



// Named-barrier helper for the warp-group split in the pure-intra path.
// Barrier ID 0 is reserved for __syncthreads(), so we use IDs 1..kRSRingWarpGroups
// (valid bar.sync IDs are 0..15, hence WG <= 15 when this branch is taken).
// When each WG is a single warp (kWGthreads == 32) we use __syncwarp() instead.
__device__ __forceinline__ void wg_sync(int wg) {
    constexpr int kWGthreads = kRSRingThreads / kRSRingWarpGroups;
    static_assert(kRSRingWarpGroups <= 15 || kWGthreads <= 32,
                  "bar.sync IDs are 0..15; named-barrier path requires WG <= 15");
    if constexpr (kWGthreads > 32) {
        asm volatile("bar.sync %0, %1;" :: "r"(wg + 1), "n"(kWGthreads));
    } else {
        __syncwarp();
    }
}

__device__ __forceinline__ uint64_t *signal_ptr(const ncclGin& gin, const int& qp_id, const uint32_t& signal_index) {
    const auto gdaki = static_cast<struct ncclGinGdakiGPUContext*>(gin._ginHandle) + qp_id;
    return reinterpret_cast<uint64_t*>(__ldg(reinterpret_cast<int64_t*>(&gdaki->signals_table.buffer))) + signal_index;
}

typedef struct RS_locator_t {
    int cache_bound, cache_lb, t;
    int cache_per_rank;
    uint64_t cache_ptr, full_elems;
} RS_locator;

// ===== Prolog: fused input adapter — multi-tensor list walk (RS_locator cache)
// ===== + per-rank strided index + bf16->f32 decode + tail zero-fill (+ pre-mul
// ===== in the _with_mul wrapper). Wrapped per ring scope by a named `ld_func`
// ===== lambda that captures the walk cache (see kernel-phase-model.md).
template <typename T, bool is_aligned>
__device__ __forceinline__ float4 unified_ld(const size_t& nargs, const uint64_t *args,
    const int *prefix_elem, const int *rank_elems, const uint64_t *full_elems, const int& wrank,
    const int& f4idx, RS_locator& locator) {
    constexpr int kNumTPerF4 = sizeof(float4) / sizeof(float);
    const int elem_base = f4idx * kNumTPerF4;
    bool need_update = false;
    if (locator.t == -1) {
        if (nargs == 1) {
            locator.t = 0;
        } else {
            locator.t = bound<false>(prefix_elem, nargs, elem_base) + 1;
        }
        locator.cache_bound = prefix_elem[locator.t];
        need_update = true;
    } else {
        while (elem_base >= locator.cache_bound) {
            locator.t += 1;
            locator.cache_bound = prefix_elem[locator.t];
            need_update = true;
        }
    }
    if (need_update) {
        locator.cache_lb = locator.t == 0 ? 0 : prefix_elem[locator.t - 1];
        locator.cache_ptr = __ldg(args + locator.t * 4);
        locator.full_elems = full_elems[locator.t];
        locator.cache_per_rank = rank_elems[locator.t];
    }
    // In global memory, tensor[t] is packed with real per-rank stride =
    // rank_elems[t] (= rank_chunk * chunk_elements). prefix_elem stores the
    // 4-aligned per-rank size just for f4 indexing, so we MUST NOT use
    // (cache_bound - cache_lb) as the rank stride when per_rank is not 4-aligned.
    const int per_rank = locator.cache_per_rank;
    const int elem_in_rank = elem_base - locator.cache_lb;                // 0..aligned_per_rank-1
    const uint64_t elem_start = static_cast<uint64_t>(per_rank) * wrank + elem_in_rank;
    float4 result = {0.0f, 0.0f, 0.0f, 0.0f};
    // Clamp load_width so that padding lanes (elem_in_rank >= per_rank) and
    // tail ranks beyond full_elems are zero-filled instead of reading the
    // next rank's data.
    int remaining_in_rank = per_rank - elem_in_rank;
    if (remaining_in_rank <= 0) return result;
    int load_width = std::min(kNumTPerF4,
                              std::min(remaining_in_rank,
                                       static_cast<int>(locator.cache_bound - elem_base)));
    load_width = static_cast<int>(std::min(static_cast<uint64_t>(load_width),
        elem_start < locator.full_elems ? (locator.full_elems - elem_start) : 0UL));
    if (load_width == 0) return result;
    if constexpr (std::is_same<T, float>()) {
        const float* ld_ptr = reinterpret_cast<const float*>(locator.cache_ptr) + elem_start;
        if constexpr (is_aligned) {
            // kAligned guarantees load_width is either 0 (early-returned above)
            // or exactly kNumTPerF4, so the runtime check is redundant.
            return ld_nc_global(reinterpret_cast<const float4*>(ld_ptr));
        } else {
            float* v_ptr = reinterpret_cast<float*>(&result);
            #pragma unroll 2
            for (int z = 0; z < load_width; ++z) {
                v_ptr[z] = ld_nc_global(ld_ptr + z);
            }
        }
    } else {
        const __nv_bfloat16* ld_ptr = reinterpret_cast<const __nv_bfloat16*>(locator.cache_ptr) + elem_start;
        float* v_ptr = reinterpret_cast<float*>(&result);
        if constexpr (is_aligned) {
            // kAligned guarantees load_width is either 0 (early-returned above)
            // or exactly kNumTPerF4, so the runtime check is redundant.
            int2 pack_v = ld_nc_global(reinterpret_cast<const int2*>(ld_ptr));
            reinterpret_cast<float2*>(v_ptr)[0] = __bfloat1622float2(reinterpret_cast<const __nv_bfloat162*>(&pack_v)[0]);
            reinterpret_cast<float2*>(v_ptr)[1] = __bfloat1622float2(reinterpret_cast<const __nv_bfloat162*>(&pack_v)[1]);
            return result;
        } else {
            #pragma unroll 2
            for (int z = 0; z < load_width; ++z) {
                v_ptr[z] = __bfloat162float(ld_nc_global(ld_ptr + z));
            }
        }
    }
    return result;
}

template <typename T, bool kAligned, bool kMul>
__device__ __forceinline__ float4 rs_unified_ld_with_mul(
    const size_t& nargs, const uint64_t* args, const int* prefix_elem, const int* rank_elems, const uint64_t* full_elems,
    const int& rank, const int& f4idx, RS_locator& locator, const float& extra_mul) {
    auto v = unified_ld<T, kAligned>(nargs, args, prefix_elem, rank_elems, full_elems, rank, f4idx, locator);
    if constexpr (kMul) {
        v.x *= extra_mul; v.y *= extra_mul;
        v.z *= extra_mul; v.w *= extra_mul;
    }
    return v;
}

// =============================================================================
// All-ring kernel
// =============================================================================
template <typename T, bool kMul = false, int kOutMode = RS_OUT_MODE_DIRECT, bool kAligned = false>
__global__ void __launch_bounds__(kRSRingThreads, 1)
reduce_scatter(
    const float extra_mul, const float extra_post_mul,
    uint64_t *args, size_t nargs, void* out_ptr,
    ncclWindow_t gin_win, void *gin_win_ptr,
    int *gmem_barrier, const int nvl_ring, const int rdma_ring,
    const size_t rdma_unroll, int round_n, const bool use_wg,
    int rank, int num_local_ranks, int num_ranks, ncclDevComm devComm,
    int arrival_sig_base, int cuda_device_id)
{
    (void)cuda_device_id;  // referenced only by LL_READ_64GE_STEP* timeout printf
    constexpr int kSlotThreads = GIN_CTA_THREADS;
    const int num_nodes = num_ranks / num_local_ranks;
    const int node_id = rank / num_local_ranks;
    const int local_rank = rank % num_local_ranks;
    constexpr unsigned int kNumTPerF4 = sizeof(float4) / sizeof(float);
    constexpr int kMaxTensors = 256;
    DEVICE_ASSERT(nargs <= kMaxTensors);

#ifdef PACE_TIMEOUT_DEBUG
    // Kernel-entry arrival beacon. Each block uses its own blockIdx.x as the
    // GIN context id on both send and read sides; sharing context 0 across
    // blocks ran into latent correctness issues with NCCL GIN. With per-block
    // contexts, sender block B routes through QP B and the dumper on the peer
    // reads from QP B's signal table, keeping the send/read pair self-
    // consistent.
    //
    // Signals: NVL siblings on this node + rail-aligned RDMA peers (same
    // local_rank, different node). Non-rail RDMA peers are skipped because
    // default NCCL_GIN_CONNECTION_RAIL doesn't connect them.
    //
    // Cumulative across launches — after the K-th call, peer M's QP-B
    // arrival slot for me reaches K+1 (= round_n + 1).
    if (num_nodes > 1) {
        // Match the existing kernel pattern: construct ncclGin / ncclTeam from
        // every thread of the CTA, but only issue gin.signal from thread 0
        // (signal isn't safe to invoke concurrently from multiple threads on
        // the same QP).
        ncclGin g_dbg{devComm, (int)blockIdx.x};
        ncclTeam world_dbg = ncclTeamWorld(devComm);
        if (threadIdx.x == 0) {
            // NVL siblings on this node.
            for (int p = 0; p < num_local_ranks; ++p) {
                const int peer = node_id * num_local_ranks + p;
                if (peer == rank) continue;
                g_dbg.signal(world_dbg, peer,
                             ncclGin_SignalInc{static_cast<uint32_t>(arrival_sig_base + rank)});
            }
            // Rail-aligned RDMA peers on other nodes.
            for (int n = 0; n < num_nodes; ++n) {
                if (n == node_id) continue;
                const int peer = n * num_local_ranks + local_rank;
                g_dbg.signal(world_dbg, peer,
                             ncclGin_SignalInc{static_cast<uint32_t>(arrival_sig_base + rank)});
            }
        }
    }
    __syncthreads();
#else
    (void)arrival_sig_base;
#endif
    // Required minima for the chunk-shift layout. The all-ring kernel uses an
    // extra (seg_slot in [0, pipe_depth)) axis on rdma_send/rdma_recv, so the
    // per-SM stride needs to fit pipe_depth * num_nodes slots.

    // Per-tensor element bookkeeping.
    __shared__ int prefix_elem[kMaxTensors];
    __shared__ int rank_elems[kMaxTensors];
    __shared__ uint64_t full_elems[kMaxTensors];
    for (int i = threadIdx.x; i < nargs; i += blockDim.x) {
        const uint32_t chunks = __ldg(args + i * 4 + 1);
        const uint32_t chunk_elements = __ldg(args + i * 4 + 2);
        const uint32_t rank_chunk = ceil_div(chunks, static_cast<uint32_t>(num_ranks));
        const uint32_t per_rank = rank_chunk * chunk_elements;
        rank_elems[i] = per_rank;
        prefix_elem[i] = align_up(per_rank, kNumTPerF4);
        full_elems[i] = static_cast<uint64_t>(chunks) * chunk_elements;
    }
    __syncthreads();
    if ((threadIdx.x >> 5) == 0) {
        hillis_steele_sum<EAGER_SCOPE_WARP>(prefix_elem, nargs, threadIdx.x);
    }
    __syncthreads();

    int smf4start, smf4end;
    const int total_f4 = ceil_div(prefix_elem[nargs - 1], static_cast<int>(kNumTPerF4));
    get_work_range(total_f4, gridDim.x, blockIdx.x, smf4start, smf4end);
    const int slot_f4 = static_cast<int>(rdma_unroll * kSlotThreads);
    const int unrolled_times = (smf4end - smf4start) / slot_f4;
    const int tailf4 = (smf4end - smf4start) - unrolled_times * slot_f4;
    const int seg_batch = unrolled_times + (tailf4 > 0);

    // Each SM uses a single fixed QP (stable single-stream RDMA on this rail).


    // ---- Pointer/signal helpers (sig area at gmem_barrier + 1024) ----
#define sig_ptr(peer) (((peer) == local_rank) ? (reinterpret_cast<uint8_t*>(gmem_barrier) + 1024) : reinterpret_cast<uint8_t*>(__ldg(reinterpret_cast<const uint64_t*>(gmem_barrier) + (peer))) + (reinterpret_cast<uint8_t*>(gmem_barrier) - reinterpret_cast<uint8_t*>(gin_win_ptr)) + 1024)
#define peer_ptr(peer, ptr) (((peer) == local_rank) ? reinterpret_cast<uint8_t*>(ptr) : (reinterpret_cast<uint8_t*>(__ldg(reinterpret_cast<const uint64_t*>(gmem_barrier) + (peer))) + (reinterpret_cast<uint8_t*>(ptr) - reinterpret_cast<uint8_t*>(gin_win_ptr))))

    // Sig area layout (per rank), with WG = kRSRingWarpGroups warp-group axis:
    //   [0,                  gridDim.x*WG): intra_data_sig per (sm, wg) -- monotonic counter,
    //                   incremented by sender (this rank's NVL prev neighbor).
    //   [gridDim.x*WG,    2*gridDim.x*WG): intra_ack_sig  per (sm, wg) -- monotonic counter,
    //                   incremented by receiver (this rank's NVL next neighbor).
    //   [2*gridDim.x*WG,  3*gridDim.x*WG): intra_gstart   per (sm, wg) -- persistent base
    //                   offset across launches; cumulative number of intra
    //                   ring steps performed.
    // Inter-only / mixed branches use wg = 0 only; pure-intra branch uses all WG.
    constexpr int WG = kRSRingWarpGroups;
    // When use_wg is false the host allocates buffer with WG=1 stride, so
    // buffer macros must use this runtime stride to match. Signal macros
    // still use the compile-time WG because gin_sigs is always allocated
    // with the kRSRingWarpGroups multiplier.
    const int wg_buf_stride = use_wg ? WG : 1;
#define sig_base(dst_local) reinterpret_cast<uint64_t*>(sig_ptr(dst_local))
#define intra_data_sig_st_ptr(dst_local, wg) (sig_base(dst_local) + (blockIdx.x * WG + (wg)))
#define intra_data_sig_ld_ptr(wg)            (sig_base(local_rank) + (blockIdx.x * WG + (wg)))
#define intra_ack_sig_st_ptr(dst_local, wg)  (sig_base(dst_local)  + gridDim.x * WG + (blockIdx.x * WG + (wg)))
#define intra_ack_sig_ld_ptr(wg)             (sig_base(local_rank) + gridDim.x * WG + (blockIdx.x * WG + (wg)))
#define gstart_ptr(wg)                       (sig_base(local_rank) + 2 * gridDim.x * WG + (blockIdx.x * WG + (wg)))

    const uint64_t gstart = __ldg(gstart_ptr(0));

    // Inter ring data/ack signals indexed per (sm, wg). Pure-inter branch uses
    // wg = 0 only; mixed branch uses all WG (each WG drives its own seg's full
    // inter+intra nested loop independently). Total inter sigs
    // = 2 * gridDim.x * WG (allocated by RSComm; scaled by WG when mixed).
#define inter_data_sig_w_index(wg)  static_cast<uint32_t>((blockIdx.x * WG + (wg)) * 2)
#define inter_data_sig_r_index(wg)  static_cast<uint32_t>((blockIdx.x * WG + (wg)) * 2)
#define inter_ack_sig_w_index(wg)   static_cast<uint32_t>((blockIdx.x * WG + (wg)) * 2 + 1)
#define inter_ack_sig_r_index(wg)   static_cast<uint32_t>((blockIdx.x * WG + (wg)) * 2 + 1)

    // Buffer macros.
    // nvl region (per rank, gin_win base): per SM, WG * nvl_ring slots.
    //   Each warp-group (wg ∈ [0, WG)) owns nvl_ring slots:
    //     Slots [0, num_local_ranks) := partial chunks.
    //     Slot  [num_local_ranks]    := single recv buffer (handshake-reused).
    //   Inter-only / mixed branches use wg = 0 only.
#define my_nvl_buffer(wg, s) (reinterpret_cast<float4*>(gin_win_ptr) + ((blockIdx.x * wg_buf_stride + (wg)) * nvl_ring + (s)) * (rdma_unroll * kSlotThreads))
#define next_nvl_buffer(wg, s) reinterpret_cast<float4*>(peer_ptr((local_rank == num_local_ranks - 1) ? 0 : (local_rank + 1), my_nvl_buffer((wg), (s))))
    // rdma_send region (per SM, per WG, rdma_ring slots laid out as
    // [sm=0..gridDim.x)[wg=0..WG)[s=0..rdma_ring)). Byte offset within gin_win.
    // The (wg) axis matches the kernel's WG specialization in the mixed
    // branch; pure-inter branch always passes wg=0.
#define rdma_send_ptr(wg, s) (reinterpret_cast<float4*>(gin_win_ptr) + ((num_local_ranks == 1 ? 0 : gridDim.x * wg_buf_stride) * nvl_ring + ((blockIdx.x * wg_buf_stride + (wg)) * rdma_ring + (s))) * (rdma_unroll * kSlotThreads))
#define rdma_send_offset(wg, s) ((rdma_send_ptr((wg), (s)) - reinterpret_cast<float4*>(gin_win_ptr)) * sizeof(float4))
    // rdma_recv region (per SM, per WG, rdma_ring slots, byte offset within gin_win).
#define rdma_recv_ptr(wg, s) (reinterpret_cast<float4*>(gin_win_ptr) + (gridDim.x * wg_buf_stride * (num_local_ranks == 1 ? 0 : 1) * nvl_ring + (((gridDim.x + blockIdx.x) * wg_buf_stride + (wg)) * rdma_ring + (s))) * (rdma_unroll * kSlotThreads))
#define rdma_recv_offset(wg, s) ((rdma_recv_ptr((wg), (s)) - reinterpret_cast<float4*>(gin_win_ptr)) * sizeof(float4))

#define out_st_ptr(working_iter) (reinterpret_cast<float4*>(out_ptr) + smf4start + (working_iter) * (rdma_unroll * kSlotThreads))

    constexpr bool kNeedAvg  = (kOutMode == RS_OUT_MODE_AVG  || kOutMode == RS_OUT_MODE_AVG_CAST_BF16);
    constexpr bool kNeedCast = (kOutMode == RS_OUT_MODE_CAST_BF16 || kOutMode == RS_OUT_MODE_AVG_CAST_BF16);
    const float inv_num_ranks = kNeedAvg ? (1.0f / static_cast<float>(num_ranks)) : 1.0f;
    // ===== Epilog: output-adapter lambda — avg/post-mul scale + optional
    // ===== f32->bf16 encode, fused into the terminal ring-step store.
    auto out_cast_if_needed = [&](float4* ptr, float4& val) -> void {
        out_cast_if_needed_dev<kNeedCast, kMul, kNeedAvg>(ptr, val, extra_post_mul, inv_num_ranks, out_ptr);
    };
    if (num_nodes == 1) {
        if (num_local_ranks == 1) {
            // just copy
            constexpr int current_rank = 0;
            RS_locator locator;
            locator.t = -1;
            // ===== Prolog: input-adapter lambda (single-rank pure copy) =====
            auto ld_func = [&](const int& f4idx) -> float4 {
                return rs_unified_ld_with_mul<T, kAligned, kMul>(nargs, args, prefix_elem, rank_elems, full_elems, current_rank, f4idx, locator, extra_mul);
            };
            // ===== Core: vectorized copy input -> output (single-rank pure copy) =====
            UNROLLED_BLOCK_COPY(3, smf4end - smf4start, out_st_ptr(0), smf4start, ld_func, out_cast_if_needed);
        } else {
            // pure intra ring reduce — warp-group parallel.
            // CTA (kRSRingThreads = 512 threads) is split into WG = kRSRingWarpGroups
            // warp groups of kWGthreads = 512 / WG threads each. Each WG advances a
            // different seg s independently with its own ring slots, sig counters,
            // and (named or warp-local) barrier — so a WG stalled on a forward sig
            // wait does not block other WGs from doing NVL load/store / reduce work
            // on their own segs. With WG = 16 each WG is exactly one warp; the
            // wg_sync helper picks __syncwarp() in that case.
            const bool wg_work = use_wg && (seg_batch >= WG); // only use WG split when it can help with intra-ring parallelism; otherwise all threads collaborate on each seg for best latency.
            constexpr int kWGthreads = kRSRingThreads / WG;  // 512 / WG
            const int wg     = wg_work ? (threadIdx.x / kWGthreads) : 0;     // 0..WG-1
            const int wg_tid = wg_work ? (threadIdx.x % kWGthreads) : threadIdx.x;     // 0..kWGthreads-1
            const int wg_size = wg_work ? kWGthreads : kRSRingThreads; // = kRSRingThreads when !wg_work
            const int wg_step = wg_work ? WG : 1;
            // Each WG holds a private gstart slot in the sig area.
            const uint64_t my_gstart = __ldg(gstart_ptr(wg));

            // Per-WG private cursors (live in registers).
            uint64_t prev_sig = my_gstart * (num_ranks - 1);
            uint64_t next_ack = my_gstart * (num_ranks - 1);

            // Round-robin partition of segs across WGs (best load balance for
            // arbitrary seg_batch). Each WG owns disjoint ring slots & counters.
            for (int s = wg; s < seg_batch; s += wg_step) {
                const int nfloat4 = s < unrolled_times ? slot_f4 : tailf4;
                for (int _w = 0, wrank = (rank - 1 + num_ranks) % num_ranks; _w < num_ranks; ++_w, wrank = (wrank > 0 ? (wrank - 1) : (num_ranks - 1))) {
                    if (_w != 0) {
                        // local nvl buffer ready
                        prev_sig += 1;
                        if (wg_tid == 0) {
                            // Sender of intra_data_sig is our NVL prev neighbor.
                            const int nvl_prev_peer = (rank == 0) ? (num_ranks - 1) : (rank - 1);
                            LL_READ_64GE_STEP(prev_sig, intra_data_sig_ld_ptr(wg), nvl_prev_peer, "_w", _w);
                        }
                    }
                    if (_w != num_ranks - 1) {
                        // next nvl buffer ready. When kWGthreads > 32 we put the ack
                        // wait on a different lane so it overlaps the data sig wait;
                        // when kWGthreads == 32 the WG is one warp and the two waits
                        // fall onto the same lane (still correct, no extra latency
                        // hiding within the WG, but different WGs still hide each
                        // other's waits).
                        constexpr int kAckLane = (kWGthreads > 32) ? 32 : 0;
                        if (wg_tid == kAckLane && next_ack >= nvl_ring) {
                            // Sender of intra_ack_sig is our NVL next neighbor.
                            const int nvl_next_peer = (rank == num_ranks - 1) ? 0 : (rank + 1);
                            LL_READ_64GE_STEP(next_ack - nvl_ring + 1, intra_ack_sig_ld_ptr(wg), nvl_next_peer, "_w", _w);
                        }
                        next_ack += 1;
                    }
                    wg_work ? wg_sync(wg) : __syncthreads(); // ensure all threads have observed the sig updates before we read/write the ring buffers
                    RS_locator locator;
                    locator.t = -1;
                    const int& current_rank = wrank;
                    // ===== Prolog: input-adapter lambda (this ring step's shifted chunk) =====
                    auto ld_func = [&](const int& f4idx) -> float4 {
                        return rs_unified_ld_with_mul<T, kAligned, kMul>(nargs, args, prefix_elem, rank_elems, full_elems, current_rank, f4idx, locator, extra_mul);
                    };
                    const int read_slot = prev_sig % nvl_ring;
                    const int write_slot = next_ack % nvl_ring;
                    auto out_ = (_w == num_ranks - 1) ? out_st_ptr(s) : next_nvl_buffer(wg, write_slot);
                    const auto f4start = smf4start + s * (rdma_unroll * kSlotThreads);
                    // ===== Core: just-copy or 2-way reduce (shifted chunk + peer NVL slot) -> output/next-stage =====
                    if (_w == 0) {
                        // just copy
                        UNROLLED_GROUP_COPY(8, wg_tid, wg_size, nfloat4, out_, f4start, ld_func, st_na_global);
                    } else {
                        // two way reduce, write to output/nvl next.
                        // V2: src0/src1 loads fully prefetched -> 2*UF inflight
                        // float4 loads per thread (vs UF in V1). Higher reg
                        // pressure but exposes more NVL load latency hiding,
                        // which is the bottleneck for the WG=4 path.
                        if (_w == num_ranks - 1) {
                            UNROLLED_GROUP_F32_2WAY_REDUCE_V2(8, wg_tid, wg_size, nfloat4, out_, f4start, my_nvl_buffer(wg, read_slot), ld_func, ld_nc_global, out_cast_if_needed);
                        } else {
                            UNROLLED_GROUP_F32_2WAY_REDUCE_V2(8, wg_tid, wg_size, nfloat4, out_, f4start, my_nvl_buffer(wg, read_slot), ld_func, ld_nc_global, st_na_global);
                        }
                    }
                    // Write to output does not need a fence.
                    wg_work ? wg_sync(wg) : __syncthreads();
                    // Both sig releases on lane 0. Originally these were split
                    // across two lanes (0 and 32) to overlap two store latencies,
                    // but with WG = 16 each WG only has 32 threads (lanes 0..31)
                    // so wg_tid == 32 would never be true and the forward sig
                    // would never be sent — which is what produced the partial
                    // ring-step accumulation bug. Putting both on lane 0 is
                    // correct for any kWGthreads; the inter-WG concurrency from
                    // the WG split is what hides the sig wait latency.
                    if (wg_tid == 0) {
                        if (_w != 0) {
                            // send ack
                            const int nvl_prev = (rank == 0) ? (num_ranks - 1) : (rank - 1);
                            st_release_sys_global(intra_ack_sig_st_ptr(nvl_prev, wg), prev_sig);
                        }
                        if (_w != num_ranks - 1) {
                            // send signal
                            const int nvl_next = (rank == num_ranks - 1) ? 0 : (rank + 1);
                            st_release_sys_global(intra_data_sig_st_ptr(nvl_next, wg), next_ack);
                        }
                    }
                }
            }

            // Per-WG persistent gstart update at end of pure-intra branch.
            // Each WG bumps its own counter by the number of segs it actually
            // processed under round-robin partition; WGs with my_segs == 0
            // simply rewrite the unchanged value.
            if (wg_tid == 0) {
                const int my_segs = wg_work ? ((seg_batch - wg + WG - 1) / WG) : seg_batch;
                st_na_global(gstart_ptr(wg), my_gstart + static_cast<uint64_t>(my_segs));
            }
        }
    } else {
        ncclGin gin { devComm, int(blockIdx.x) };
        ncclTeam world = ncclTeamWorld(devComm);
        if (num_local_ranks == 1) {
            // pure inter ring reduce
            uint64_t prev_sig = gstart * (num_ranks - 1), next_ack = gstart * (num_ranks - 1);
            for (int s = 0; s < seg_batch; s++) {
                const int nfloat4 = s < unrolled_times ? slot_f4 : tailf4;
                for (int _w = 0, wrank = (rank - 1 + num_ranks) % num_ranks; _w < num_ranks; ++_w, wrank = (wrank > 0 ? (wrank - 1) : (num_ranks - 1))) {
                    if (_w != 0) {
                        // local rdma recv data ready
                        prev_sig += 1;
                        if (threadIdx.x == 0) {
                            // Sender of inter_data_sig is our RDMA prev neighbor (gin.put from rank-1).
                            const int rdma_prev_peer = (rank == 0) ? (num_ranks - 1) : (rank - 1);
                            LL_READ_64GE_STEP(prev_sig, signal_ptr(gin, blockIdx.x, inter_data_sig_r_index(0)), rdma_prev_peer, "_w", _w);
                        }
                    }
                    if (_w != num_ranks - 1) {
                        // next rdma recv buffer ready
                        if (threadIdx.x == 32 && next_ack >= rdma_ring) {
                            // Sender of inter_ack_sig is our RDMA next neighbor (gin.signal from rank+1).
                            const int rdma_next_peer = (rank == num_ranks - 1) ? 0 : (rank + 1);
                            LL_READ_64GE_STEP(next_ack - rdma_ring + 1, signal_ptr(gin, blockIdx.x, inter_ack_sig_r_index(0)), rdma_next_peer, "_w", _w);
                        }
                        next_ack += 1;
                    }
                    __syncthreads();
                    RS_locator locator;
                    locator.t = -1;
                    const int& current_rank = wrank;
                    // ===== Prolog: input-adapter lambda (this ring step's shifted chunk) =====
                    auto ld_func = [&](const int& f4idx) -> float4 {
                        return rs_unified_ld_with_mul<T, kAligned, kMul>(nargs, args, prefix_elem, rank_elems, full_elems, current_rank, f4idx, locator, extra_mul);
                    };
                    const int read_slot = prev_sig % rdma_ring;
                    const int write_slot = next_ack % rdma_ring;
                    auto out_ = (_w == num_ranks - 1) ? out_st_ptr(s) : rdma_send_ptr(0, write_slot);
                    const auto f4start = smf4start + s * (rdma_unroll * kSlotThreads);
                    // ===== Core: just-copy or 2-way reduce (shifted chunk + RDMA recv slot) -> output/rdma-send =====
                    if (_w == 0) {
                        // just copy
                        UNROLLED_BLOCK_COPY(3, nfloat4, out_, f4start, ld_func, st_na_global);
                    } else {
                        // two way reduce, write to output/rdma send
                        if (_w == num_ranks - 1) {
                            UNROLLED_BLOCK_F32_2WAY_REDUCE(3, nfloat4, out_, f4start, rdma_recv_ptr(0, read_slot), ld_func, ld_nc_global, out_cast_if_needed);
                        } else {
                            UNROLLED_BLOCK_F32_2WAY_REDUCE(3, nfloat4, out_, f4start, rdma_recv_ptr(0, read_slot), ld_func, ld_nc_global, st_na_global);
                        }
                    }
                    // Write to output does not need a fence.
                    __syncthreads();
                    if (_w != 0 && threadIdx.x == 0) {
                        // send ack to previous rank, tell that rdma recv buffer is ready
                        const int rdma_prev = (rank == 0) ? (num_ranks - 1) : (rank - 1);
                        gin.signal(world, rdma_prev, ncclGin_SignalInc{inter_ack_sig_w_index(0)});
                    }
                    if (_w != num_ranks - 1 && threadIdx.x == 32) {
                        // send signal to next rank, tell that rdma recv data is ready
                        const int rdma_next = (rank == num_ranks - 1) ? 0 : (rank + 1);
                        gin.put(world, rdma_next, gin_win, rdma_recv_offset(0, write_slot), gin_win, rdma_send_offset(0, write_slot), nfloat4 * sizeof(float4), ncclGin_SignalInc{inter_data_sig_w_index(0)});
                    }
                }
            }
        } else {
            // mixed intra and inter ring reduce — WG-specialized: each warp-group
            // owns disjoint segs (round-robin partition) and drives its own full
            // inter+intra nested loop with private sig cursors. Intra signals are
            // indexed per (sm, wg) (already); inter GIN signals are also per
            // (sm, wg) — host RSComm scales gin_sigs by WG when this branch runs.
            const bool wg_work = use_wg && (seg_batch >= WG); // only use WG split when it can help with intra-ring parallelism; otherwise all threads collaborate on each seg for best latency.
            constexpr int kWGthreads = kRSRingThreads / WG;  // 512 / WG
            const int wg     = wg_work ? (threadIdx.x / kWGthreads) : 0;     // 0..WG-1
            const int wg_tid = wg_work ? (threadIdx.x % kWGthreads) : threadIdx.x;     // 0..kWGthreads-1
            const int wg_size = wg_work ? kWGthreads : kRSRingThreads; // = kRSRingThreads when !wg_work
            const int wg_step = wg_work ? WG : 1;

            const uint64_t my_gstart = __ldg(gstart_ptr(wg));

            // Per-WG private sig cursors.
            uint64_t rdma_prev_sig = my_gstart * (num_nodes - 1);
            uint64_t rdma_next_ack = my_gstart * (num_nodes - 1);
            uint64_t nvl_prev_sig  = my_gstart * num_nodes * (num_local_ranks - 1);
            uint64_t nvl_next_ack  = my_gstart * num_nodes * (num_local_ranks - 1);

            // Lane assignment within a WG. When the WG has >=128 threads we can
            // give each of the 4 sig waits/posts a separate lane (lanes 0/32/64/96)
            // so the four 64-bit reads/writes overlap. With fewer threads per WG
            // we fall back to lane 0 for everything (still correct; latency
            // hiding then comes purely from inter-WG concurrency).
            constexpr int kLaneRdmaData = 0;
            constexpr int kLaneRdmaAck  = (kWGthreads >= 64)  ? 32 : 0;
            constexpr int kLaneNvlData  = (kWGthreads >= 96)  ? 64 : 0;
            constexpr int kLaneNvlAck   = (kWGthreads >= 128) ? 96 : 0;

            for (int s = wg; s < seg_batch; s += wg_step) {
                const int nfloat4 = s < unrolled_times ? slot_f4 : tailf4;
                for (int _n = 0, wnode = (node_id - 1 + num_nodes) % num_nodes; _n < num_nodes; ++_n, wnode = (wnode > 0 ? (wnode - 1) : (num_nodes - 1))) {
                    if (_n != 0) {
                        // local rdma recv data ready
                        rdma_prev_sig += 1;
                        if (wg_tid == kLaneRdmaData) {
                            // Sender of inter_data_sig is the rail-aligned peer on our RDMA prev node.
                            const int rdma_prev_node = (node_id == 0) ? (num_nodes - 1) : (node_id - 1);
                            const int rdma_prev_peer = rdma_prev_node * num_local_ranks + local_rank;
                            LL_READ_64GE_STEP(rdma_prev_sig, signal_ptr(gin, blockIdx.x, inter_data_sig_r_index(wg)), rdma_prev_peer, "_n", _n);
                        }
                    }
                    if (_n != num_nodes - 1) {
                        // next rdma recv buffer ready
                        if (wg_tid == kLaneRdmaAck && rdma_next_ack >= rdma_ring) {
                            // Sender of inter_ack_sig is the rail-aligned peer on our RDMA next node.
                            const int rdma_next_node = (node_id == num_nodes - 1) ? 0 : (node_id + 1);
                            const int rdma_next_peer = rdma_next_node * num_local_ranks + local_rank;
                            LL_READ_64GE_STEP(rdma_next_ack - rdma_ring + 1, signal_ptr(gin, blockIdx.x, inter_ack_sig_r_index(wg)), rdma_next_peer, "_n", _n);
                        }
                        rdma_next_ack += 1;
                    }
                    const int rdma_read_slot = rdma_prev_sig % rdma_ring;
                    const int rdma_write_slot = rdma_next_ack % rdma_ring;
                    auto rdma_out_ = (_n == num_nodes - 1) ? out_st_ptr(s) : rdma_send_ptr(wg, rdma_write_slot);
                    for (int _l = 0, wlocal = (local_rank - num_nodes + _n + num_ranks) % num_local_ranks; _l < num_local_ranks; ++_l, wlocal = (wlocal > 0 ? (wlocal - 1) : (num_local_ranks - 1))) {
                        if (_l != 0) {
                            // local nvl data ready
                            nvl_prev_sig += 1;
                            if (wg_tid == kLaneNvlData) {
                                // Sender of intra_data_sig is our NVL prev neighbor (same node).
                                const int nvl_prev_local = (local_rank == 0) ? (num_local_ranks - 1) : (local_rank - 1);
                                const int nvl_prev_peer = node_id * num_local_ranks + nvl_prev_local;
                                LL_READ_64GE_STEP2(nvl_prev_sig, intra_data_sig_ld_ptr(wg), nvl_prev_peer, "_n", _n, "_l", _l);
                            }
                        }
                        if (_l != num_local_ranks - 1) {
                            // next nvl buffer ready
                            if (wg_tid == kLaneNvlAck && nvl_next_ack >= nvl_ring) {
                                // Sender of intra_ack_sig is our NVL next neighbor (same node).
                                const int nvl_next_local = (local_rank == num_local_ranks - 1) ? 0 : (local_rank + 1);
                                const int nvl_next_peer = node_id * num_local_ranks + nvl_next_local;
                                LL_READ_64GE_STEP2(nvl_next_ack - nvl_ring + 1, intra_ack_sig_ld_ptr(wg), nvl_next_peer, "_n", _n, "_l", _l);
                            }
                            nvl_next_ack += 1;
                        }
                        wg_work ? wg_sync(wg) : __syncthreads();
                        RS_locator locator;
                        locator.t = -1;
                        const int current_rank = wnode * num_local_ranks + wlocal;
                        // ===== Prolog: input-adapter lambda (this ring step's shifted chunk) =====
                        auto ld_func = [&](const int& f4idx) -> float4 {
                            return rs_unified_ld_with_mul<T, kAligned, kMul>(nargs, args, prefix_elem, rank_elems, full_elems, current_rank, f4idx, locator, extra_mul);
                        };
                        const int nvl_read_slot = nvl_prev_sig % nvl_ring;
                        const int nvl_write_slot = nvl_next_ack % nvl_ring;
                        auto nvl_out_ = (_l == num_local_ranks - 1) ? rdma_out_ : next_nvl_buffer(wg, nvl_write_slot);
                        const auto f4start = smf4start + s * (rdma_unroll * kSlotThreads);
                        /**
                         * _n == 0 && _l == 0: (input), dst is next nvl
                         * _n == 0 && _l != 0: (input + nvl), dst is next nvl/rdma send(_l == num_local_ranks - 1)
                         * _n != 0 && _l == 0: (rdma recv + input), dst is next nvl
                         * _n != 0 && _l != 0: (input + nvl), dst is next nvl/rdma send(_l == num_local_ranks - 1)/out(_n == num_nodes - 1 && _l == num_local_ranks - 1)
                         */
                        // ===== Core: just-copy or 2-way reduce (input + second input) -> output/nvl-next =====
                        if (_l == 0 && _n == 0) {
                            // just copy
                            UNROLLED_GROUP_COPY(8, wg_tid, wg_size, nfloat4, nvl_out_, f4start, ld_func, st_na_global);
                        } else {
                            // two way reduce, write to output/nvl next
                            auto second_input = (_l == 0) ? rdma_recv_ptr(wg, rdma_read_slot) : my_nvl_buffer(wg, nvl_read_slot);
                            if (_l == num_local_ranks - 1 && _n == num_nodes - 1) {
                                UNROLLED_GROUP_F32_2WAY_REDUCE_V2(8, wg_tid, wg_size, nfloat4, nvl_out_, f4start, second_input, ld_func, ld_nc_global, out_cast_if_needed);
                            } else {
                                UNROLLED_GROUP_F32_2WAY_REDUCE_V2(8, wg_tid, wg_size, nfloat4, nvl_out_, f4start, second_input, ld_func, ld_nc_global, st_na_global);
                            }
                        }
                        wg_work ? wg_sync(wg) : __syncthreads();
                        if (_l != 0 && wg_tid == kLaneNvlData) {
                            // send ack
                            const int nvl_prev = (local_rank == 0) ? (num_local_ranks - 1) : (local_rank - 1);
                            st_release_sys_global(intra_ack_sig_st_ptr(nvl_prev, wg), nvl_prev_sig);
                        }
                        if (_l != num_local_ranks - 1 && wg_tid == kLaneNvlAck) {
                            // send signal
                            const int nvl_next = (local_rank == num_local_ranks - 1) ? 0 : (local_rank + 1);
                            st_release_sys_global(intra_data_sig_st_ptr(nvl_next, wg), nvl_next_ack);
                        }
                    }
                    if (_n != 0 && wg_tid == kLaneRdmaData) {
                        // send ack
                        const int rdma_prev = (node_id == 0) ? (num_nodes - 1) : (node_id - 1);
                        gin.signal(world, rdma_prev * num_local_ranks + local_rank, ncclGin_SignalInc{inter_ack_sig_w_index(wg)});
                    }
                    if (_n != num_nodes - 1 && wg_tid == kLaneRdmaAck) {
                        // send signal
                        const int rdma_next = (node_id == num_nodes - 1) ? 0 : (node_id + 1);
                        gin.put(world, rdma_next * num_local_ranks + local_rank, gin_win, rdma_recv_offset(wg, rdma_write_slot), gin_win, rdma_send_offset(wg, rdma_write_slot), nfloat4 * sizeof(float4), ncclGin_SignalInc{inter_data_sig_w_index(wg)});
                    }
                }
            }

            // Per-WG persistent gstart update (mirrors pure-intra branch).
            if (wg_tid == 0) {
                const int my_segs = wg_work ? ((seg_batch - wg + WG - 1) / WG) : seg_batch;
                st_na_global(gstart_ptr(wg), my_gstart + static_cast<uint64_t>(my_segs));
            }
        }
    }
    __syncthreads();
    // Pure-intra and mixed branches update gstart per-WG inside the branch
    // (each WG writes its own gstart_ptr(wg)). The remaining branches
    // (pure-inter and pure-copy) still operate as a single warp-group, so they
    // update gstart_ptr(0) here. Skip the write entirely for the WG-specialized
    // branches — it has already happened.
    const bool wg_specialized = (num_nodes == 1 && num_local_ranks > 1) ||
                                (num_nodes > 1 && num_local_ranks > 1);
    if (!wg_specialized && threadIdx.x == 0) {
        st_na_global(gstart_ptr(0), gstart + seg_batch);
    }
#undef sig_ptr
#undef peer_ptr
#undef sig_base
#undef intra_data_sig_st_ptr
#undef intra_data_sig_ld_ptr
#undef intra_ack_sig_st_ptr
#undef intra_ack_sig_ld_ptr
#undef gstart_ptr
#undef inter_data_sig_w_index
#undef inter_data_sig_r_index
#undef inter_ack_sig_w_index
#undef inter_ack_sig_r_index
#undef rdma_send_ptr
#undef rdma_send_offset
#undef rdma_recv_ptr
#undef rdma_recv_offset
#undef out_st_ptr
}

} // namespace pace

#endif // PACE_RS_KERNELS_CUH
