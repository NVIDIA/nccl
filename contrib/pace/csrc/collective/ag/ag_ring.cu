#include "collective/ag/ag_ring.cuh"
#include "util/error.hpp"
#include "device/comm.cuh"
#include "util/math.hpp"
#include "device/launch.cuh"
#include "device/configs.cuh"
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <nccl_device/coop.h>
#include <nccl_device/core.h>
#include <nccl_device/gin.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <unordered_map>
#include <mutex>
#include <memory>
#include <vector>

#ifdef PACE_TIMEOUT_DEBUG
// Override the default no-op timeout hook so every LL_READ_64GE in this kernel
// also dumps the NVL + rail-aligned arrival bitmap. Inline RDMA timeout sites
// call LL_TIMEOUT_DUMP() directly (they don't go through LL_READ_64GE).
#undef LL_TIMEOUT_DUMP
#define LL_TIMEOUT_DUMP() \
    ::dump_arrival_status(dev_comm, num_local_ranks, num_nodes, local_rank, node_id, \
                          arrival_sig_base, round_n, rank, (int)blockIdx.x, \
                          cuda_device_id)
#endif

// ==========================================================================
// Multi-channel ring AllGather.
//
// See collective/ag/ag_ring.cuh for the topology and protocol definition. In short:
//   - num_sms rings, ring r seeded at GPU (r % num_local_ranks) on server 0.
//   - Each server walks 7 NVL hops alternating direction (ascending on even
//     servers, descending on odd) so that every RDMA hop is rail-aligned
//     (same local index on src and dst).
//   - Each SM drives exactly one ring; data is sliced across rings into a
//     virtual concatenated int4 stream (sum of int4-aligned input bytes).
//   - Two-slot recv ring per SM (configurable via nvl_ring; default 2).
//     Multi-node SMs additionally maintain a send staging ring of rdma_ring
//     slots used as the source of gin.put when next is RDMA.
//
// Per-gslot protocol (NCCL runRing):
//   round 0       : push my data to next (with optional fp32->bf16); also
//                   write to my output[my_rank] region.
//   rounds 1..num_ranks-2 : wait prev-data; read my recv slot; write output[src_rank];
//                   forward to next; ack prev.
//   round num_ranks-1     : wait prev-data; read; write output[src_rank]; ack prev.
//
// Cumulative monotonic counters per SM (across kernel calls):
//   data sig (incoming) value after my round k recv of gslot G = G*(num_ranks-1)+k
//   ack  sig (outgoing) value after my round k ack  of gslot G = G*(num_ranks-1)+k
//   data sig (outgoing) value after my round k push of gslot G = G*(num_ranks-1)+k+1
//   ack  sig (incoming) value after next consumes my round k push   = same
// ==========================================================================

namespace pace {

namespace {

constexpr int kRingMaxTensors = 256;



// Visit position t of ring with seed ring_seed in (num_local_ranks, num_nodes) cluster.
// Server s = t / num_local_ranks. Direction alternates: even servers ascending, odd descending.
// Even server visits ring_seed, ring_seed+1, ..., ring_seed-1 (mod num_local_ranks) in order.
// Odd  server visits ring_seed-1, ring_seed-2, ..., ring_seed   (mod num_local_ranks) in order.
__device__ __forceinline__ void ring_visit(
    int t, int ring_seed, int num_local_ranks, int /*num_nodes*/,
    int& out_server, int& out_local)
{
    out_server = t / num_local_ranks;
    int t_in = t - out_server * num_local_ranks;
    int local;
    if ((out_server & 1) == 0) {
        local = (ring_seed + t_in) % num_local_ranks;
    } else {
        // Need positive arg before %; ring_seed-1-t_in lies in (-num_local_ranks, num_local_ranks), add 2*num_local_ranks.
        local = (ring_seed - 1 - t_in + 2 * num_local_ranks) % num_local_ranks;
    }
    out_local = local;
}

// Inverse of ring_visit: t such that ring_visit(t)=(server, local).
__device__ __forceinline__ int ring_position(
    int server, int local, int ring_seed, int num_local_ranks)
{
    int t_in;
    if ((server & 1) == 0) {
        t_in = (local - ring_seed + 2 * num_local_ranks) % num_local_ranks;
    } else {
        t_in = (ring_seed - 1 - local + 2 * num_local_ranks) % num_local_ranks;
    }
    return server * num_local_ranks + t_in;
}

// 32 bytes of fp32 (8 floats from 2 int4 loads) -> 16 bytes of bf16 (8 bf16
// packed into 1 int4).
__device__ __forceinline__ int4 fp32x8_to_bf16x8(const int4& f_lo, const int4& f_hi) {
    int4 out;
    const float* fp_lo = reinterpret_cast<const float*>(&f_lo);
    const float* fp_hi = reinterpret_cast<const float*>(&f_hi);
    reinterpret_cast<__nv_bfloat162*>(&out)[0] =
        __float22bfloat162_rn(make_float2(fp_lo[0], fp_lo[1]));
    reinterpret_cast<__nv_bfloat162*>(&out)[1] =
        __float22bfloat162_rn(make_float2(fp_lo[2], fp_lo[3]));
    reinterpret_cast<__nv_bfloat162*>(&out)[2] =
        __float22bfloat162_rn(make_float2(fp_hi[0], fp_hi[1]));
    reinterpret_cast<__nv_bfloat162*>(&out)[3] =
        __float22bfloat162_rn(make_float2(fp_hi[2], fp_hi[3]));
    return out;
}

// ===== Prolog: fused input adapter — dtype wire-mapping (kImap) + partial-
// ===== byte tail zero-fill. Called directly at the ring copy sites.
// Load up to 16 bytes of input (one int4 of comm dtype). For FP32_TO_BF16,
// loads 32 bytes of fp32 input and converts. valid_comm_bytes is the number
// of comm-dtype bytes that are real (rest are tail padding -> zero).
template <int kImap>
__device__ __forceinline__ int4 load_input_int4(
    const uint8_t* in_p_comm_aligned,  // pointer into input (comm-dtype offset)
    int valid_comm_bytes)
{
    int4 val = {0, 0, 0, 0};
    if constexpr (kImap == AGCOMM_INPUT_DTYPE_MAPPING_FP32_TO_BF16) {
        // in_p_comm_aligned points to where the comm-dtype data WOULD live;
        // the real input lives at storage_dtype offset = comm_offset * 2.
        // Caller computes that, but we assume in_p_comm_aligned is already
        // the storage-pointer base + storage_offset (i.e., comm_byte_off * 2).
        const uint8_t* in_p = in_p_comm_aligned;
        int4 f_lo = {0, 0, 0, 0}, f_hi = {0, 0, 0, 0};
        if (valid_comm_bytes == static_cast<int>(sizeof(int4)) &&
            (reinterpret_cast<uintptr_t>(in_p) & 0xF) == 0) {
            f_lo = ld_nc_global(reinterpret_cast<const int4*>(in_p));
            f_hi = ld_nc_global(reinterpret_cast<const int4*>(in_p + sizeof(int4)));
        } else {
            uint8_t vbuf[32] = {0};
            int need = valid_comm_bytes * 2;
            for (int z = 0; z < need; ++z) vbuf[z] = ld_nc_global(in_p + z);
            f_lo = *reinterpret_cast<int4*>(vbuf);
            f_hi = *reinterpret_cast<int4*>(vbuf + sizeof(int4));
        }
        val = fp32x8_to_bf16x8(f_lo, f_hi);
    } else {
        const uint8_t* in_p = in_p_comm_aligned;
        if (valid_comm_bytes == static_cast<int>(sizeof(int4)) &&
            (reinterpret_cast<uintptr_t>(in_p) & 0xF) == 0) {
            val = ld_nc_global(reinterpret_cast<const int4*>(in_p));
        } else {
            uint8_t vbuf[16] = {0};
            for (int z = 0; z < valid_comm_bytes; ++z) vbuf[z] = ld_nc_global(in_p + z);
            val = *reinterpret_cast<int4*>(vbuf);
        }
    }
    return val;
}

// ===== Epilog: fused output adapter — vector store with partial-byte tail.
// ===== Called directly at the ring copy sites.
__device__ __forceinline__ void store_int4_partial(uint8_t* out_p, int4 val, int valid_bytes) {
    if (valid_bytes == static_cast<int>(sizeof(int4)) &&
        (reinterpret_cast<uintptr_t>(out_p) & 0xF) == 0) {
        st_na_global(reinterpret_cast<int4*>(out_p), val);
    } else {
        const uint8_t* vp = reinterpret_cast<const uint8_t*>(&val);
        for (int z = 0; z < valid_bytes; ++z) st_na_global(out_p + z, vp[z]);
    }
}

}  // namespace

// ==========================================================================
// Ring AllGather kernel
// ==========================================================================

template <bool kSingleNode, bool kSingleTensor, int kImap>
__global__ void __launch_bounds__(GIN_CTA_THREADS, 1)
allgather_ring_kernel(
    uint64_t* args_gpu,
    int num_tensors,
    ncclWindow_t gin_win,
    void* gin_buffer,
    void* signal_buffer,
    int unroll,
    int nvl_ring,
    int rdma_ring,
    uint64_t base_slot,
    int rank,
    int num_local_ranks,
    int num_ranks,
    ncclDevComm dev_comm,
    int arrival_sig_base,
    int round_n,
    int cuda_device_id)
{
    (void)cuda_device_id;  // referenced only by LL_READ_64GE_STEP* timeout printf
    const uint32_t sm_id = blockIdx.x;
    const uint32_t num_sms = gridDim.x;
    const int num_nodes = num_ranks / num_local_ranks;
    const int node_id = rank / num_local_ranks;
    const int local_rank = rank % num_local_ranks;

#ifdef PACE_TIMEOUT_DEBUG
    // Kernel-entry arrival beacon (multi-node only). Each block uses its own
    // blockIdx.x as the GIN context id on both send and read sides; sharing
    // context 0 across blocks ran into latent correctness issues with NCCL
    // GIN. Sender block B routes through QP B and the dumper on the peer
    // reads from QP B's signal table.
    if constexpr (!kSingleNode) {
        // Construct ncclGin/ncclTeam from every thread of the CTA (matches
        // existing kernel pattern) but only issue gin.signal from thread 0
        // (signal isn't safe to invoke concurrently from multiple threads
        // on one QP).
        ncclGin g_dbg{dev_comm, (int)sm_id};
        ncclTeam world_dbg = ncclTeamWorld(dev_comm);
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
                if (peer == rank) continue;
                g_dbg.signal(world_dbg, peer,
                             ncclGin_SignalInc{static_cast<uint32_t>(arrival_sig_base + rank)});
            }
        }
        __syncthreads();
    }
#else
    (void)arrival_sig_base;
    (void)round_n;
#endif

    // Ring identity for this SM.
    const int ring_seed = ((static_cast<int>(sm_id) % num_local_ranks) + num_local_ranks) % num_local_ranks;

    // Compute prev/next ranks in ring r.
    const int ring_pos = ring_position(node_id, local_rank, ring_seed, num_local_ranks);
    const int next_t = (num_ranks == 0) ? 0 : (ring_pos + 1) % num_ranks;
    const int prev_t = (num_ranks == 0) ? 0 : (ring_pos - 1 + num_ranks) % num_ranks;
    int next_node, next_local, prev_node, prev_local;
    ring_visit(next_t, ring_seed, num_local_ranks, num_nodes, next_node, next_local);
    ring_visit(prev_t, ring_seed, num_local_ranks, num_nodes, prev_node, prev_local);
    const int next_rank = next_node * num_local_ranks + next_local;
    const int prev_rank = prev_node * num_local_ranks + prev_local;
    const bool next_is_nvl = (next_node == node_id);
    const bool prev_is_nvl = (prev_node == node_id);

    // ---- P2P pointers from signal_buffer header ----
    __shared__ void* s_p2p[16];
    if (threadIdx.x < num_local_ranks) {
        s_p2p[threadIdx.x] = reinterpret_cast<void**>(signal_buffer)[threadIdx.x];
    }
    auto nvl_p2p = [&](auto* ptr, int dst_rank) -> decltype(ptr) {
        size_t off = reinterpret_cast<const uint8_t*>(ptr) - reinterpret_cast<const uint8_t*>(gin_buffer);
        return reinterpret_cast<decltype(ptr)>(reinterpret_cast<uint8_t*>(s_p2p[dst_rank % num_local_ranks]) + off);
    };

    // ---- Tensor metadata in shared memory ----
    __shared__ uint64_t s_tensor_bytes[kRingMaxTensors];
    __shared__ uint64_t s_prefix_aligned_f4[kRingMaxTensors];
    __shared__ uint64_t s_input_ptrs[kRingMaxTensors];
    __shared__ uint64_t s_output_ptrs[kRingMaxTensors];
    __shared__ uint64_t s_total_aligned_f4;

    if (threadIdx.x < num_tensors) {
        s_tensor_bytes[threadIdx.x] = __ldg(args_gpu + threadIdx.x * 4 + 1);
        s_input_ptrs[threadIdx.x]   = __ldg(args_gpu + threadIdx.x * 4 + 0);
        s_output_ptrs[threadIdx.x]  = __ldg(args_gpu + threadIdx.x * 4 + 2);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        uint64_t prefix = 0;
        for (int i = 0; i < num_tensors; ++i) {
            prefix += align_up(s_tensor_bytes[i], sizeof(int4));
            s_prefix_aligned_f4[i] = prefix / sizeof(int4);
        }
        s_total_aligned_f4 = prefix / sizeof(int4);
    }
    __syncthreads();

    const uint64_t total_f4 = s_total_aligned_f4;
    const int slot_f4 = GIN_CTA_THREADS * unroll;
    const size_t slot_bytes = static_cast<size_t>(slot_f4) * sizeof(int4);

    // Per-SM data slice in the virtual int4 stream.
    int smf4start, smf4end;
    get_work_range(static_cast<int>(total_f4), static_cast<int>(num_sms),
                   static_cast<int>(sm_id), smf4start, smf4end);

    // Compute global max_slots_per_sm so all SMs run the same number of slots
    // (must match host-side `max_slots_per_sm` exactly).
    const int per_sm_f4_max = ceil_div(static_cast<int>(total_f4), static_cast<int>(num_sms));
    const int global_max_slots_per_sm = ceil_div(per_sm_f4_max, slot_f4);

    const int my_work_f4 = smf4end - smf4start;
    const int unrolled_times = my_work_f4 / slot_f4;
    const int tail_f4 = my_work_f4 - unrolled_times * slot_f4;
    const int total_slots_for_sm = global_max_slots_per_sm;

    // ---- Buffer pointers per SM ----
    //   recv_ring  : nvl_ring × slot_bytes (peer-writable)
    //   send_stage : rdma_ring × slot_bytes (multi-node only, gin.put source)
    const size_t recv_ring_bytes  = slot_bytes * static_cast<size_t>(nvl_ring);
    const size_t send_stage_bytes = kSingleNode ? 0 : slot_bytes * static_cast<size_t>(rdma_ring);
    const size_t per_engine_block = recv_ring_bytes + send_stage_bytes;
    uint8_t* my_engine_base = reinterpret_cast<uint8_t*>(gin_buffer) + sm_id * per_engine_block;
    uint8_t* my_recv_ring   = my_engine_base;
    uint8_t* my_send_stage  = my_engine_base + recv_ring_bytes;

    // Per-round slot indexing (one slot per (gslot, round) push, modulo depth).
    // slot index for my push at round k of gslot G = (G*(num_ranks-1) + k) % depth
    // slot index for my read at round k of gslot G = (G*(num_ranks-1) + k - 1) % depth
    auto send_slot_idx = [&](uint64_t G, int k) -> uint64_t {
        return (G * static_cast<uint64_t>(num_ranks - 1) + static_cast<uint64_t>(k))
               % static_cast<uint64_t>(nvl_ring);
    };
    auto recv_slot_idx = [&](uint64_t G, int k) -> uint64_t {
        // round k >= 1
        return (G * static_cast<uint64_t>(num_ranks - 1) + static_cast<uint64_t>(k - 1))
               % static_cast<uint64_t>(nvl_ring);
    };
    auto stage_slot_idx = [&](uint64_t G, int k) -> uint64_t {
        return (G * static_cast<uint64_t>(num_ranks - 1) + static_cast<uint64_t>(k))
               % static_cast<uint64_t>(rdma_ring);
    };

    auto recv_slot_ptr_local_at_round = [&](uint64_t G, int k) -> int4* {
        return reinterpret_cast<int4*>(my_recv_ring + recv_slot_idx(G, k) * slot_bytes);
    };
    auto stage_slot_ptr_at_round = [&](uint64_t G, int k) -> int4* {
        return reinterpret_cast<int4*>(my_send_stage + stage_slot_idx(G, k) * slot_bytes);
    };
    // Peer's recv slot at the same per-round offset within their gin_buffer.
    auto peer_send_slot_ptr = [&](int peer_rank, uint64_t G, int k) -> int4* {
        int4* my_addr = reinterpret_cast<int4*>(my_recv_ring + send_slot_idx(G, k) * slot_bytes);
        if (peer_rank == rank) return my_addr;
        return nvl_p2p(my_addr, peer_rank);
    };
    // Byte offsets (within gin_buffer) for RDMA put.
    auto stage_slot_offset = [&](uint64_t G, int k) -> size_t {
        return static_cast<size_t>(reinterpret_cast<uint8_t*>(stage_slot_ptr_at_round(G, k))
                                   - reinterpret_cast<uint8_t*>(gin_buffer));
    };
    auto next_recv_slot_offset = [&](uint64_t G, int k) -> size_t {
        uint8_t* my_addr = my_recv_ring + send_slot_idx(G, k) * slot_bytes;
        return static_cast<size_t>(my_addr - reinterpret_cast<uint8_t*>(gin_buffer));
    };

    // ---- NVL signal pointers (in signal_buffer) ----
    // Header: num_local_ranks × void* (peer p2p table written by host).
    // Then per-SM block of 2 u64s: [data_sig, ack_sig].
    const size_t sig_header_bytes = static_cast<size_t>(num_local_ranks) * sizeof(void*);
    const size_t per_engine_sig_u64 = 2;  // data_sig + ack_sig (only 2 used per engine)
    auto sig_base_local = [&]() -> uint64_t* {
        return reinterpret_cast<uint64_t*>(
            reinterpret_cast<uint8_t*>(signal_buffer) + sig_header_bytes)
            + static_cast<size_t>(sm_id) * per_engine_sig_u64;
    };
    auto sig_base_peer = [&](int peer_rank) -> uint64_t* {
        if (peer_rank == rank) return sig_base_local();
        return nvl_p2p(sig_base_local(), peer_rank);
    };
    auto data_sig_local      = [&]() -> uint64_t* { return sig_base_local() + 0; };
    auto ack_sig_local       = [&]() -> uint64_t* { return sig_base_local() + 1; };
    auto data_sig_at_peer    = [&](int p) -> uint64_t* { return sig_base_peer(p) + 0; };
    auto ack_sig_at_peer     = [&](int p) -> uint64_t* { return sig_base_peer(p) + 1; };

    // ---- GIN context for RDMA ----
    ncclGin gin{dev_comm, static_cast<int>(sm_id)};
    ncclTeam world;
    if constexpr (!kSingleNode) world = ncclTeamWorld(dev_comm);
    const uint32_t rdma_data_sig_idx = ag_ring_rdma_data_sig(sm_id);
    const uint32_t rdma_ack_sig_idx  = ag_ring_rdma_ack_sig(sm_id);

    // Helper: find tensor id and bounds for a given f4 index in the virtual stream.
    auto find_tensor = [&](uint64_t f4idx, int& t, uint64_t& lb_f4, uint64_t& bound_f4) {
        if constexpr (kSingleTensor) {
            t = 0;
            lb_f4 = 0;
            bound_f4 = s_prefix_aligned_f4[0];
        } else {
            int tt = 0;
            while (tt < num_tensors && f4idx >= s_prefix_aligned_f4[tt]) tt++;
            t = tt;
            lb_f4 = (tt == 0) ? 0 : s_prefix_aligned_f4[tt - 1];
            bound_f4 = (tt < num_tensors) ? s_prefix_aligned_f4[tt] : lb_f4;
        }
    };

    // Issue an RDMA put of the staging slot for round k of gslot G to
    // next_rank's recv slot at the same per-round index, signaling
    // rdma_data_sig at next. When slot_f4_n == 0 (padded zero-work slot),
    // skip the data put and bump the remote rdma_data_sig with a
    // signal-only operation so peer counters stay in sync.
    auto rdma_put_to_next = [&](uint64_t G, int k, int slot_f4_n) {
        if (threadIdx.x == 0) {
            if (slot_f4_n > 0) {
                const size_t put_bytes = static_cast<size_t>(slot_f4_n) * sizeof(int4);
                const size_t src_off = stage_slot_offset(G, k);
                const size_t dst_off = next_recv_slot_offset(G, k);
                gin.put(world, next_rank,
                        gin_win, dst_off, gin_win, src_off, put_bytes,
                        ncclGin_SignalInc{rdma_data_sig_idx});
            } else {
                gin.signal(world, next_rank, ncclGin_SignalInc{rdma_data_sig_idx});
            }
        }
    };

    // Wait for next's ack to reach `expect`.
    auto wait_next_ack = [&](uint64_t expect) {
        if (next_is_nvl) {
            if (threadIdx.x == 0) {
                LL_READ_64GE(expect, ack_sig_local(), next_rank);
            }
        } else {
            if (threadIdx.x == 0) {
                auto start_t = clock64();
                while (gin.readSignal(rdma_ack_sig_idx) < expect) {
                    if (clock64() - start_t > NUM_TIMEOUT_CYCLES) {
                        printf("[rank %d dev %d sm %d] ag_ring rdma_ack wait timeout, expect %lu (next_rank=%d)\n",
                               rank, cuda_device_id, (int)sm_id, expect, next_rank);
                        LL_TIMEOUT_DUMP();
                        __trap();
                    }
                }
            }
        }
        __syncthreads();
    };

    // Wait for prev's data signal to reach `expect`.
    auto wait_prev_data = [&](uint64_t expect) {
        if (prev_is_nvl) {
            if (threadIdx.x == 0) {
                LL_READ_64GE(expect, data_sig_local(), prev_rank);
            }
        } else {
            if (threadIdx.x == 0) {
                auto start_t = clock64();
                while (gin.readSignal(rdma_data_sig_idx) < expect) {
                    if (clock64() - start_t > NUM_TIMEOUT_CYCLES) {
                        printf("[rank %d dev %d sm %d] ag_ring rdma_data wait timeout, expect %lu (prev_rank=%d)\n",
                               rank, cuda_device_id, (int)sm_id, expect, prev_rank);
                        LL_TIMEOUT_DUMP();
                        __trap();
                    }
                }
            }
        }
        __syncthreads();
    };

    // Signal next that I just pushed (round k of gslot G); new cumulative
    // push count = `value`. For NVL we just bump my data sig at next; for
    // RDMA the gin.put with SignalInc both transfers data and bumps the sig.
    auto signal_next_data = [&](uint64_t value, uint64_t G, int k, int slot_f4_n) {
        if (next_is_nvl) {
            if (threadIdx.x == 0) {
                st_release_sys_global(data_sig_at_peer(next_rank), value);
            }
        } else {
            (void)value;
            rdma_put_to_next(G, k, slot_f4_n);
        }
    };

    // Ack prev that I consumed their push; new cumulative ack count = `value`.
    auto ack_prev = [&](uint64_t value) {
        if (prev_is_nvl) {
            if (threadIdx.x == 0) {
                st_release_sys_global(ack_sig_at_peer(prev_rank), value);
            }
        } else {
            (void)value;
            if (threadIdx.x == 0) {
                gin.signal(world, prev_rank, ncclGin_SignalInc{rdma_ack_sig_idx});
            }
        }
    };

    // Special case: num_ranks == 1 (single rank). All-gather is a no-op copy.
    if (num_ranks == 1) {
        // We still need to copy input to output[my_rank] = output[0].
        // SM walks [smf4start, smf4end). Padded slots (s > unrolled_times) are
        // no-ops; harmless here since num_ranks==1 doesn't communicate.
        for (int s = 0; s < total_slots_for_sm; ++s) {
            int slot_f4_local;
            if (s < unrolled_times)       slot_f4_local = slot_f4;
            else if (s == unrolled_times) slot_f4_local = tail_f4;
            else                          slot_f4_local = 0;
            if (slot_f4_local == 0) continue;
            const uint64_t f4_global_start =
                static_cast<uint64_t>(smf4start) + static_cast<uint64_t>(s) * slot_f4;
            int t_init; uint64_t lb_init, bound_init;
            find_tensor(f4_global_start, t_init, lb_init, bound_init);
            for (int li = threadIdx.x; li < slot_f4_local; li += blockDim.x) {
                uint64_t f4idx = f4_global_start + static_cast<uint64_t>(li);
                int my_t = t_init;
                uint64_t my_lb = lb_init, my_bound = bound_init;
                if constexpr (!kSingleTensor) {
                    while (f4idx >= my_bound && my_t + 1 < num_tensors) {
                        my_t++; my_lb = my_bound; my_bound = s_prefix_aligned_f4[my_t];
                    }
                    if (f4idx >= my_bound) break;
                }
                const uint64_t comm_byte_off = (f4idx - my_lb) * sizeof(int4);
                const uint64_t tensor_bytes = s_tensor_bytes[my_t];
                const int valid_comm_bytes = static_cast<int>(min(
                    static_cast<uint64_t>(sizeof(int4)),
                    tensor_bytes - min(comm_byte_off, tensor_bytes)));
                if (valid_comm_bytes <= 0) continue;
                const uint8_t* in_p_storage = reinterpret_cast<const uint8_t*>(s_input_ptrs[my_t])
                    + ((kImap == AGCOMM_INPUT_DTYPE_MAPPING_FP32_TO_BF16) ? comm_byte_off * 2 : comm_byte_off);
                int4 val = load_input_int4<kImap>(in_p_storage, valid_comm_bytes);
                uint8_t* out_p = reinterpret_cast<uint8_t*>(s_output_ptrs[my_t]) + comm_byte_off;
                store_int4_partial(out_p, val, valid_comm_bytes);
            }
        }
        return;
    }

    // ==========================================================================
    // Main loop: process each gslot through num_ranks rounds (per-round slot pipeline).
    //
    // Slot layout (per-round): the slot index for the n-th push (1-indexed,
    // n = G*(num_ranks-1) + k + 1) is (n - 1) mod depth. Hence consecutive rounds use
    // distinct slot indices up to depth, and only need to wait for ack from
    // `depth` rounds back. With depth=2, the first two pushes never wait.
    //
    // Wait condition before pushing (push_count_after = G*(num_ranks-1) + k + 1):
    //   ack_sig >= push_count_after - eff_depth   (skip if <= 0)
    // where eff_depth = nvl_ring if next is NVL, else min(nvl_ring, rdma_ring)
    // (RDMA also reuses my staging slot of size rdma_ring).
    // ==========================================================================
    const int eff_depth = next_is_nvl ? nvl_ring
                                      : ((nvl_ring < rdma_ring) ? nvl_ring : rdma_ring);
    const uint64_t num_ranks_m1 = static_cast<uint64_t>(num_ranks - 1);

    // Per-thread unroll factor for the inner data loop.
    constexpr int kUnroll = 4;

    for (int s = 0; s < total_slots_for_sm; ++s) {
        const uint64_t G = base_slot + static_cast<uint64_t>(s);
        // For SMs with empty/short work, slot_f4_local is 0 for slots beyond
        // their real range. The inner data loops are no-ops in that case but
        // the signal protocol still advances so peer SMs' counters stay in
        // sync across calls (host advances total_send_slots by max_slots_per_sm
        // regardless of which SMs had work).
        int slot_f4_local;
        if (s < unrolled_times)       slot_f4_local = slot_f4;
        else if (s == unrolled_times) slot_f4_local = tail_f4;
        else                          slot_f4_local = 0;
        const uint64_t f4_global_start =
            static_cast<uint64_t>(smf4start) + static_cast<uint64_t>(s) * slot_f4;

        // Tensor metadata for this gslot's f4_global_start (constant for all
        // rounds — the data slice doesn't shift across rounds).
        int t_init; uint64_t lb_init, bound_init;
        find_tensor(f4_global_start, t_init, lb_init, bound_init);

        // ---- Rounds 0 .. num_ranks-1: place one shard, forward to next ----
        // Round 0 (from_input) sources from the input tensor, applies the dtype
        // mapping, and always forwards. Rounds k>=1 source from the recv slot that
        // prev filled (data already in output dtype), and the final round only
        // lands its shard without forwarding. The wait/ack protocol and the
        // src_rank/fwd/signal math below reduce to the round-0 case when k==0.
        for (int k = 0; k < num_ranks; ++k) {
            const bool from_input   = (k == 0);
            const bool will_forward = (k < num_ranks - 1);

            // Round 0 produces its own data; k>=1 waits for prev's push.
            if (!from_input) {
                const uint64_t recv_target = G * num_ranks_m1 + static_cast<uint64_t>(k);
                wait_prev_data(recv_target);
            }

            if (will_forward) {
                const uint64_t push_count_after = G * num_ranks_m1 + static_cast<uint64_t>(k + 1);
                if (push_count_after > static_cast<uint64_t>(eff_depth)) {
                    wait_next_ack(push_count_after - static_cast<uint64_t>(eff_depth));
                }
            }

            const int src_t = (ring_pos - k + num_ranks) % num_ranks;
            int src_node, src_local;
            ring_visit(src_t, ring_seed, num_local_ranks, num_nodes, src_node, src_local);
            const int src_rank = src_node * num_local_ranks + src_local;

            int4* fwd_dst = nullptr;
            if (will_forward) {
                if constexpr (kSingleNode) {
                    fwd_dst = peer_send_slot_ptr(next_rank, G, k);
                } else {
                    fwd_dst = next_is_nvl ? peer_send_slot_ptr(next_rank, G, k)
                                          : stage_slot_ptr_at_round(G, k);
                }
            }

            int4* my_recv = from_input ? nullptr : recv_slot_ptr_local_at_round(G, k);

            if (kSingleTensor) {
                const uint64_t tensor_bytes = s_tensor_bytes[0];
                // Round-0 source base into the input tensor (imap doubles the
                // stride for FP32->BF16); unused when reading from my_recv.
                const uint8_t* in_base_storage = reinterpret_cast<const uint8_t*>(s_input_ptrs[0])
                    + ((kImap == AGCOMM_INPUT_DTYPE_MAPPING_FP32_TO_BF16)
                           ? f4_global_start * (2 * sizeof(int4))
                           : f4_global_start * sizeof(int4));
                uint8_t* out_base = reinterpret_cast<uint8_t*>(s_output_ptrs[0])
                                    + static_cast<uint64_t>(src_rank) * tensor_bytes
                                    + f4_global_start * sizeof(int4);
                const uint64_t comm_byte_off_base = f4_global_start * sizeof(int4);
                const uint64_t tail_lim_f4 = (tensor_bytes > comm_byte_off_base)
                    ? ((tensor_bytes - comm_byte_off_base) / sizeof(int4))
                    : 0;
                const int safe_f4 = static_cast<int>(min(
                    static_cast<uint64_t>(slot_f4_local), tail_lim_f4));

                int4 vbuf[kUnroll];
                const int kStride = blockDim.x * kUnroll;
                const int kMain = (safe_f4 / kStride) * kStride;
                for (int i = threadIdx.x; i < kMain; i += kStride) {
                    #pragma unroll
                    for (int u = 0; u < kUnroll; ++u) {
                        const int li = i + u * blockDim.x;
                        if (from_input) {
                            if constexpr (kImap == AGCOMM_INPUT_DTYPE_MAPPING_FP32_TO_BF16) {
                                int4 f_lo = ld_nc_global(reinterpret_cast<const int4*>(
                                    in_base_storage) + 2 * li);
                                int4 f_hi = ld_nc_global(reinterpret_cast<const int4*>(
                                    in_base_storage) + 2 * li + 1);
                                vbuf[u] = fp32x8_to_bf16x8(f_lo, f_hi);
                            } else {
                                vbuf[u] = ld_nc_global(reinterpret_cast<const int4*>(
                                    in_base_storage) + li);
                            }
                        } else {
                            vbuf[u] = ld_nc_global(my_recv + li);
                        }
                    }
                    #pragma unroll
                    for (int u = 0; u < kUnroll; ++u) {
                        const int li = i + u * blockDim.x;
                        st_na_global(reinterpret_cast<int4*>(out_base) + li, vbuf[u]);
                        if (will_forward) {
                            st_na_global(fwd_dst + li, vbuf[u]);
                        }
                    }
                }
                for (int li = kMain + threadIdx.x; li < slot_f4_local; li += blockDim.x) {
                    const uint64_t comm_byte_off = comm_byte_off_base + (uint64_t)li * sizeof(int4);
                    const int valid_comm_bytes = static_cast<int>(min(
                        static_cast<uint64_t>(sizeof(int4)),
                        tensor_bytes - min(comm_byte_off, tensor_bytes)));
                    if (valid_comm_bytes <= 0) break;
                    int4 val;
                    if (from_input) {
                        const uint8_t* in_p = in_base_storage
                            + ((kImap == AGCOMM_INPUT_DTYPE_MAPPING_FP32_TO_BF16)
                                   ? (uint64_t)li * (2 * sizeof(int4))
                                   : (uint64_t)li * sizeof(int4));
                        val = load_input_int4<kImap>(in_p, valid_comm_bytes);
                    } else {
                        val = ld_nc_global(my_recv + li);
                    }
                    uint8_t* out_p = out_base + (uint64_t)li * sizeof(int4);
                    store_int4_partial(out_p, val, valid_comm_bytes);
                    if (will_forward) {
                        if (valid_comm_bytes == static_cast<int>(sizeof(int4))) {
                            st_na_global(fwd_dst + li, val);
                        } else {
                            const uint8_t* vp = reinterpret_cast<const uint8_t*>(&val);
                            for (int z = 0; z < valid_comm_bytes; ++z) {
                                st_na_global(reinterpret_cast<uint8_t*>(fwd_dst + li) + z, vp[z]);
                            }
                        }
                    }
                }
            } else {
                for (int li = threadIdx.x; li < slot_f4_local; li += blockDim.x) {
                    uint64_t f4idx = f4_global_start + static_cast<uint64_t>(li);
                    int my_t = t_init;
                    uint64_t my_lb = lb_init, my_bound = bound_init;
                    while (f4idx >= my_bound && my_t + 1 < num_tensors) {
                        my_t++; my_lb = my_bound; my_bound = s_prefix_aligned_f4[my_t];
                    }
                    if (f4idx >= my_bound) break;
                    const uint64_t comm_byte_off = (f4idx - my_lb) * sizeof(int4);
                    const uint64_t tensor_bytes = s_tensor_bytes[my_t];
                    const int valid_comm_bytes = static_cast<int>(min(
                        static_cast<uint64_t>(sizeof(int4)),
                        tensor_bytes - min(comm_byte_off, tensor_bytes)));
                    if (valid_comm_bytes <= 0) continue;
                    int4 val;
                    if (from_input) {
                        const uint8_t* in_p_storage = reinterpret_cast<const uint8_t*>(s_input_ptrs[my_t])
                            + ((kImap == AGCOMM_INPUT_DTYPE_MAPPING_FP32_TO_BF16) ? comm_byte_off * 2 : comm_byte_off);
                        val = load_input_int4<kImap>(in_p_storage, valid_comm_bytes);
                    } else {
                        val = ld_nc_global(my_recv + li);
                    }
                    uint8_t* out_p = reinterpret_cast<uint8_t*>(s_output_ptrs[my_t])
                                     + static_cast<uint64_t>(src_rank) * tensor_bytes
                                     + comm_byte_off;
                    store_int4_partial(out_p, val, valid_comm_bytes);
                    if (will_forward) {
                        st_na_global(fwd_dst + li, val);
                    }
                }
            }
            if (will_forward) {
                if constexpr (kSingleNode) {
                    __threadfence();
                } else {
                    __threadfence_system();
                }
                __syncthreads();
                const uint64_t push_after = G * num_ranks_m1 + static_cast<uint64_t>(k + 1);
                signal_next_data(push_after, G, k, slot_f4_local);
            } else {
                __syncthreads();
            }

            // Round 0 consumed nothing from prev; k>=1 acks the slot it drained.
            if (!from_input) {
                ack_prev(G * num_ranks_m1 + static_cast<uint64_t>(k));
            }
        }
    }
}

// ==========================================================================
// Host-side launcher
// ==========================================================================

namespace {

void dispatch_launch(
    int num_blocks,
    uint64_t* args_gpu, int num_tensors,
    ncclWindow_t gin_win, void* gin_buffer, void* signal_buffer,
    int unroll, int nvl_ring, int rdma_ring, uint64_t base_slot,
    int input_dtype_mapping, int rank, int num_local_ranks, int num_ranks,
    ncclDevComm dev_comm, cudaStream_t stream,
    int arrival_sig_base, int capture_round_n, int cuda_device_id)
{
    const bool single_node   = (num_local_ranks == num_ranks);
    const bool single_tensor = (num_tensors == 1);
    SETUP_LAUNCH_CONFIG(num_blocks, GIN_CTA_THREADS, stream);

#define CALL_K(kSingleNode_, kSingleTensor_, kImap_) \
    do { \
        auto kfn = allgather_ring_kernel<kSingleNode_, kSingleTensor_, kImap_>; \
        LAUNCH_KERNEL(&cfg, kfn, args_gpu, num_tensors, gin_win, gin_buffer, signal_buffer, \
                      unroll, nvl_ring, rdma_ring, base_slot, rank, num_local_ranks, num_ranks, \
                      dev_comm, arrival_sig_base, capture_round_n, cuda_device_id); \
    } while (0)

    if (input_dtype_mapping == AGCOMM_INPUT_DTYPE_MAPPING_FP32_TO_BF16) {
        if (single_node && single_tensor) CALL_K(true,  true,  AGCOMM_INPUT_DTYPE_MAPPING_FP32_TO_BF16);
        else if (single_node)             CALL_K(true,  false, AGCOMM_INPUT_DTYPE_MAPPING_FP32_TO_BF16);
        else if (single_tensor)           CALL_K(false, true,  AGCOMM_INPUT_DTYPE_MAPPING_FP32_TO_BF16);
        else                              CALL_K(false, false, AGCOMM_INPUT_DTYPE_MAPPING_FP32_TO_BF16);
    } else {
        if (single_node && single_tensor) CALL_K(true,  true,  AGCOMM_INPUT_DTYPE_MAPPING_NONE);
        else if (single_node)             CALL_K(true,  false, AGCOMM_INPUT_DTYPE_MAPPING_NONE);
        else if (single_tensor)           CALL_K(false, true,  AGCOMM_INPUT_DTYPE_MAPPING_NONE);
        else                              CALL_K(false, false, AGCOMM_INPUT_DTYPE_MAPPING_NONE);
    }
#undef CALL_K
}

}  // namespace

void allgather_ring_func(
    uint64_t* args_cpu, uint64_t* args_gpu, int num_tensors,
    ncclWindow_t gin_win, void* gin_buffer, void* signal_buffer,
    int unroll, int nvl_ring, int rdma_ring, size_t& total_send_slots,
    int capture_round_n, int input_dtype_mapping,
    int rank, int num_local_ranks, int num_ranks,
    ncclDevComm dev_comm,
    int num_sms, cudaStream_t stream,
    int arrival_sig_base, int cuda_device_id)
{
    const int num_nodes = num_ranks / num_local_ranks;
    // num_nodes must be 1 or even: the alternating-direction ring only closes
    // rail-aligned when num_nodes is even (server num_nodes-1, odd, ends at local ring_seed, so the
    // wrap-around RDMA back to server 0's ring_seed is rail-aligned; odd num_nodes breaks it).
    HOST_ASSERT(num_nodes == 1 || (num_nodes & 1) == 0);
    HOST_ASSERT(nvl_ring >= 1);
    HOST_ASSERT(num_nodes == 1 || rdma_ring >= 1);
    HOST_ASSERT(num_tensors > 0 && num_tensors <= kRingMaxTensors);

    // Compute per-SM data slice for slot accounting.
    size_t total_aligned_bytes = 0;
    for (int i = 0; i < num_tensors; ++i) {
        total_aligned_bytes += align_up<size_t>(static_cast<size_t>(args_cpu[4 * i + 1]),
                                                sizeof(int4));
    }
    const size_t total_f4 = total_aligned_bytes / sizeof(int4);
    const int slot_f4 = GIN_CTA_THREADS * unroll;
    const int eng = std::max(1, num_sms);
    const size_t per_sm_f4 = ceil_div<size_t>(total_f4, static_cast<size_t>(eng));
    const int max_slots_per_sm = static_cast<int>(
        ceil_div<size_t>(per_sm_f4, static_cast<size_t>(slot_f4)));
    const uint64_t base_slot = total_send_slots;

    // Push args to GPU.
    CUDACHECK(cudaMemcpyAsync(args_gpu, args_cpu, sizeof(uint64_t) * num_tensors * 4,
                              cudaMemcpyHostToDevice, stream));
    const int num_blocks = (num_sms > 0) ? num_sms : 1;
    dispatch_launch(num_blocks,
                    args_gpu, num_tensors,
                    gin_win, gin_buffer, signal_buffer,
                    unroll, nvl_ring, rdma_ring, base_slot,
                    input_dtype_mapping, rank, num_local_ranks, num_ranks,
                    dev_comm, stream,
                    arrival_sig_base, capture_round_n, cuda_device_id);

    total_send_slots += static_cast<size_t>(max_slots_per_sm);
}

}  // namespace pace
