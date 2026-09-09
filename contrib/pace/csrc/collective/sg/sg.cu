#include "collective/sg/sg.cuh"
#include "collective/sg/sg_kernels.cuh"  // device kernels (extern template — see sg_p2p_extern_decls.cuh)
#include "sg_p2p_extern_decls.cuh"  // generated: extern template decls (symbols from build/gen/sg/*.cu)
#include "util/error.hpp"
#include "device/comm.cuh"
#include "util/math.hpp"
#include <cstdint>
#include <cstdio>
#include <algorithm>
#include <map>
#include <vector>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <nccl_device/coop.h>
#include <nccl_device/core.h>
#include <nccl_device/gin.h>
#include <type_traits>
#include <cooperative_groups.h>
#include "device/launch.cuh"
#include "device/configs.cuh"
#include "collective/sg/tensor_layout.hpp"

// SG_F8_ARCH_SUPPORTED, SG_MAX_TENSORS, sg_unified_ld/st, sg_direct_ld/st,
// SG_DEBUG_RECORD_TS*, WG_BAR_SYNC, and the scattergather_kernel_p2p template
// have been moved to csrc/collective/sg/sg_kernels.cuh so that
// scripts/gen_sg_inst.py can emit per-instantiation .cu files that include
// the header (mirrors ep_kernels.cuh / gen_ep_inst.py). The host dispatchers
// (scattergatherfunc, scattergather_kernel_func, scattergathercordkernel,
// sg_mem_layout, sg_device_mem_layout) remain in this file.

// SG kernel constants
#define SG_UNROLL_FACTOR 2 // Increased from 3 for better throughput

namespace pace {

/**
 * sg.cu - Scatter-Gather (All-to-All) communication implementation
 * 
 * Design follows ag.cu single-stream pattern:
 *   copy_stream: all CE ops — c1 (input→send buf), c2 (peer NVL send→output),
 *                c3 (RDMA recv→output) — issued in order per slot, with c3
 *                lagged K=min(send_times,nvl_ring) slots to hide RDMA latency
 *                and avoid the symmetric deadlock (c3(0) must be queued on
 *                copy_stream before pre_c1(nvl_ring) stalls on a remote ack).
 *   stream (launch): runs the coordination kernel (cord) that drives RDMA/
 *                NVL signalling.
 *
 * Data flow for scatter_dim=1, gather_dim=0:
 *   Input: (S, H, D) -> Output: (S*N, H/N, D)
 *   Each rank sends its j-th H-chunk to rank j
 *   Each rank receives all ranks' local_rank-th H-chunk
 *
 * Data flow for scatter_dim=0, gather_dim=1:
 *   Input: (S, H, D) -> Output: (S/N, H*N, D)
 *   Each rank sends its j-th S-chunk to rank j
 *   Each rank receives all ranks' local_rank-th S-chunk
 */

/**
 * Memory layout class for scatter-gather operations
 * Similar to ag.cu mem_layout but adapted for all-to-all communication
 *
 * Memory layout per GPU:
 *   [RDMA send area: slot_bytes * num_local_ranks * nvl_ring * num_nodes] - stores local GPU's data for sending
 *   [RDMA recv area: slot_bytes * num_local_ranks * rdma_ring * num_nodes] - receives data from other nodes
 *   [Signals area] - synchronization signals
 */
class sg_mem_layout {
    void **ptrs;
    const int nvl_ring, rdma_ring, R;  // R = max(nvl_ring, rdma_ring): single ring for both send & recv.
    const size_t slot_bytes;        // Single sub-slot size
    const size_t batch_slot_bytes;  // num_local_ranks * slot_bytes (batch size for one iteration)
    size_t send_area_bytes, recv_area_bytes, data_bytes, sig_offset;
    const int rank, num_local_ranks, num_ranks;
    int local_rank, node_id, num_nodes;
public:
    __host__ __device__ sg_mem_layout(void **p2p_ptrs, int nvl_ring, int rdma_ring, size_t slot_bytes, int rank, int num_local_ranks, int num_ranks)
        : ptrs(p2p_ptrs), nvl_ring(nvl_ring), rdma_ring(rdma_ring), R(std::max(nvl_ring, rdma_ring)), slot_bytes(slot_bytes),
          batch_slot_bytes(slot_bytes * num_local_ranks), rank(rank), num_ranks(num_ranks), num_local_ranks(num_local_ranks) {
        num_nodes = num_ranks / num_local_ranks;
        local_rank = rank % num_local_ranks;
        node_id = rank / num_local_ranks;
        // Single ring: send & recv areas both sized for R slots per node.
        send_area_bytes = batch_slot_bytes * R * num_nodes;
        recv_area_bytes = (num_ranks != num_local_ranks) ? batch_slot_bytes * R * num_nodes : 0;
        data_bytes = send_area_bytes + recv_area_bytes;
        sig_offset = num_local_ranks * sizeof(void*);
    }

    // === NVL Send Area (push-based) ===
    // c1 NVL-store target: source writes its data for dst_rank into dst_local's
    // send buffer at source-slot = local_rank. When dst_local == local_rank
    // (dest is self) this reduces to ptrs[local_rank] (a local write).
    // Layout within a batch: [src0 | src1 | ... | srcN-1] (indexed by source).
    template <typename T>
    __host__ T* peer_send_st_ptr(const int& dst_rank, const size_t& slot) {
        HOST_ASSERT(dst_rank < num_ranks && dst_rank >= 0);
        const int dst_node = dst_rank / num_local_ranks;
        const int dst_local = dst_rank % num_local_ranks;
        return reinterpret_cast<T*>(reinterpret_cast<uint8_t*>(ptrs[dst_local]) + dst_node * R * batch_slot_bytes + batch_slot_bytes * (slot % R) + slot_bytes * local_rank);
    }

    // c2 local-read source: dest reads its own send buffer at source-slot =
    // src_local. Source src_local NVL-stored its data for us there in c1.
    template <typename T>
    __host__ T* my_send_ld_ptr(const size_t& slot, const int& src_local) {
        HOST_ASSERT(src_local < num_local_ranks && src_local >= 0);
        return reinterpret_cast<T*>(reinterpret_cast<uint8_t*>(ptrs[local_rank]) + node_id * R * batch_slot_bytes + batch_slot_bytes * (slot % R) + slot_bytes * src_local);
    }


    // Device offset for RDMA send (sends entire batch to remote node)
    __device__ uint64_t rdma_send_offset(const int& dst_node, const size_t& slot) {
        return dst_node * R * batch_slot_bytes + batch_slot_bytes * (slot % R);
    }

    // === RDMA Recv Area ===
    // Device offset for receiving batched data from a specific remote node
    __device__ uint64_t rdma_recv_offset(const int& src_node, const size_t& slot) {
        return send_area_bytes + batch_slot_bytes * (R * src_node + (slot % R));
    }

    // Push-based c3 read: read OWN recv buffer at source slot. The forwarder
    // (dest_local = local_rank on the source node) collected all sources' data
    // and RDMA-sent it to us (rail-targeted: remote local_rank → our local_rank).
    // We read source slot `src_local` from our own recv buffer — a local read.
    template <typename T>
    __host__ T* my_recv_ptr(const int& src_node, const int& src_local, const size_t& slot) {
        return reinterpret_cast<T*>(reinterpret_cast<uint8_t*>(ptrs[local_rank]) + send_area_bytes +
               batch_slot_bytes * (R * src_node + (slot % R)) + slot_bytes * src_local);
    }

    // === Signals (push-based 4-path protocol) ===
    // Layout per GPU (offsets in u64 from bsig(self/peer)):
    //   [0 .. num_nodes*Nlr-1)                            : c1_done[N][src] / c2_got[N][src]
    //   [num_nodes*Nlr .. 2*num_nodes*Nlr-1)              : c1_picked[N][dst] / c2_ack[N][dst]
    //   [2*num_nodes*Nlr .. ..+num_nodes)                 : c3_signal[N]
    //   [2*num_nodes*Nlr+num_nodes .. ..+num_nodes)       : c3_ack[N]
    // c1_done/c2_got: on the forwarder (peer=src wrote via NVL store; forwarder reads local).
    // c1_picked/c2_ack: on the source (forwarder=dst wrote via NVL store; source reads local).
    // c3_signal/c3_ack: local only (cord translates GIN signals ↔ these).

    __host__ __device__ uint64_t* bsig(const int& nvl_peer) {
        return reinterpret_cast<uint64_t*>(reinterpret_cast<uint8_t*>(ptrs[nvl_peer]) + data_bytes + sig_offset);
    }

    // c1_done[N][src]: source writes to forwarder's signal (NVL store).
    //   On the forwarder (self), indexed by (N, src). src = the source's local_rank.
    __host__ __device__ uint64_t* c1_done_st_ptr(const int& N, const int& dst_peer) {
        return bsig(dst_peer) + N * num_local_ranks + local_rank;
    }
    // c2_got[N][src]: forwarder reads own (same address as c1_done).
    __host__ __device__ uint64_t* c2_got_ld_ptr(const int& N, const int& src_peer) {
        return bsig(local_rank) + N * num_local_ranks + src_peer;
    }

    // c2_ack[N][dst]: forwarder writes to source's signal (NVL store).
    //   On the source (self), indexed by (N, dst). dst = the forwarder's local_rank.
    __host__ __device__ uint64_t* c2_ack_st_ptr(const int& N, const int& src_peer) {
        return bsig(src_peer) + num_nodes * num_local_ranks + N * num_local_ranks + local_rank;
    }
    // c1_picked[N][dst]: source reads own (same address as c2_ack).
    __host__ __device__ uint64_t* c1_picked_ld_ptr(const int& N, const int& dst_peer) {
        return bsig(local_rank) + num_nodes * num_local_ranks + N * num_local_ranks + dst_peer;
    }

    // c3_signal[N]: cord writes (from GIN arrival), host c3 reads. Local.
    __host__ __device__ uint64_t* c3_signal_ptr(const int& N) {
        return bsig(local_rank) + 2 * num_nodes * num_local_ranks + N;
    }
    // c3_ack[N]: host c3 writes, cord reads (→ GIN ack). Local.
    __host__ __device__ uint64_t* c3_ack_ptr(const int& N) {
        return bsig(local_rank) + 2 * num_nodes * num_local_ranks + num_nodes + N;
    }


    // Accessors for constants
    __host__ __device__ size_t get_slot_bytes() const { return slot_bytes; }
    __host__ __device__ size_t get_batch_slot_bytes() const { return batch_slot_bytes; }
    __host__ __device__ int get_num_local_ranks() const { return num_local_ranks; }
    __host__ __device__ int get_local_rank() const { return local_rank; }
    __host__ __device__ int get_num_nodes() const { return num_nodes; }
    __host__ __device__ int get_node_id() const { return node_id; }
    __host__ __device__ int get_nvl_ring() const { return nvl_ring; }
    __host__ __device__ int get_rdma_ring() const { return rdma_ring; }
    __host__ __device__ int get_R() const { return R; }
};

/**
 * Device-side memory layout for SG kernel (optimized for register usage)
 * Stores only essential values, computes derived values on-the-fly
 */
class sg_device_mem_layout {
    uint8_t *base_ptr;  // Local GPU's buffer base pointer
    const size_t slot_bytes;
    const int nvl_ring, num_local_ranks, local_rank, node_id, num_nodes;
    
public:
    __device__ sg_device_mem_layout(void *gin_win_ptr, int nvl_ring_, int rdma_ring_, size_t slot_bytes_, 
                                     int rank, int num_local_ranks_, int num_ranks)
        : base_ptr(reinterpret_cast<uint8_t*>(gin_win_ptr)), 
          slot_bytes(slot_bytes_),
          nvl_ring(nvl_ring_), 
          num_local_ranks(num_local_ranks_),
          local_rank(rank % num_local_ranks_),
          node_id(rank / num_local_ranks_),
          num_nodes(num_ranks / num_local_ranks_) {}

    // c1: Write to local send buffer for destination rank (per-block region)
    __device__ __forceinline__ float4* local_send_ptr(const size_t& slot, const int& dst_rank) {
        const size_t batch_slot_bytes = slot_bytes * num_local_ranks;
        const size_t block_send_area = batch_slot_bytes * nvl_ring * num_nodes;
        const int dst_node = dst_rank / num_local_ranks;
        const int dest_local = dst_rank % num_local_ranks;
        return reinterpret_cast<float4*>(base_ptr + blockIdx.x * block_send_area +
               dst_node * nvl_ring * batch_slot_bytes + 
               batch_slot_bytes * (slot % nvl_ring) + slot_bytes * dest_local);
    }
    
    // c2: Read from NVL peer's send buffer (per-block region)
    __device__ __forceinline__ float4* peer_send_ptr(void **p2p_ptrs, const int& src_local, const size_t& slot) {
        const size_t batch_slot_bytes = slot_bytes * num_local_ranks;
        const size_t block_send_area = batch_slot_bytes * nvl_ring * num_nodes;
        return reinterpret_cast<float4*>(reinterpret_cast<uint8_t*>(p2p_ptrs[src_local]) + 
               blockIdx.x * block_send_area +
               node_id * nvl_ring * batch_slot_bytes + batch_slot_bytes * (slot % nvl_ring) + slot_bytes * local_rank);
    }
    
    // c3: Read from RDMA recv buffer (per-block region) - only used in multi-node
    __device__ __forceinline__ float4* rdma_recv_ptr(void **p2p_ptrs, const int& src_node, const int& nvl_peer, const size_t& slot, const int rdma_ring) {
        const size_t batch_slot_bytes = slot_bytes * num_local_ranks;
        const size_t block_send_area = batch_slot_bytes * nvl_ring * num_nodes;
        const size_t block_recv_area = batch_slot_bytes * rdma_ring * num_nodes;
        return reinterpret_cast<float4*>(reinterpret_cast<uint8_t*>(p2p_ptrs[nvl_peer]) + 
               block_send_area * gridDim.x + blockIdx.x * block_recv_area +
               batch_slot_bytes * (rdma_ring * src_node + (slot % rdma_ring)) + slot_bytes * local_rank);
    }

    // RDMA offsets for GIN put operations (per-block region)
    __device__ __forceinline__ uint64_t rdma_send_offset(const int& dst_node, const size_t& slot) {
        const size_t batch_slot_bytes = slot_bytes * num_local_ranks;
        const size_t block_send_area = batch_slot_bytes * nvl_ring * num_nodes;
        return blockIdx.x * block_send_area +
               dst_node * nvl_ring * batch_slot_bytes + batch_slot_bytes * (slot % nvl_ring);
    }
    
    __device__ __forceinline__ uint64_t rdma_recv_offset(const int& src_node, const size_t& slot, const int rdma_ring) {
        const size_t batch_slot_bytes = slot_bytes * num_local_ranks;
        const size_t block_send_area = batch_slot_bytes * nvl_ring * num_nodes;
        const size_t block_recv_area = batch_slot_bytes * rdma_ring * num_nodes;
        return block_send_area * gridDim.x + blockIdx.x * block_recv_area +
               batch_slot_bytes * (rdma_ring * src_node + (slot % rdma_ring));
    }

    __device__ __forceinline__ int get_local_rank() const { return local_rank; }
    __device__ __forceinline__ int get_node_id() const { return node_id; }
    __device__ __forceinline__ int get_num_nodes() const { return num_nodes; }
    __device__ __forceinline__ int get_num_local_ranks() const { return num_local_ranks; }
};

// sg_unified_ld/st, sg_direct_ld/st, SG_DEBUG_RECORD_TS*, WG_BAR_SYNC,
// and scattergather_kernel_p2p template: moved to sg_kernels.cuh.

/**
 * Main scatter-gather kernel using time-division multiplexed lambdas
 * Similar to reduce_scatter_multimem_gin but for all-to-all communication
 *
 * Three phases (lambdas):
 *   c1_func: input -> send buffer (load from input for each dst_rank, write to NVL send area)
 *   c2_func: peer send buffer -> output (read from NVL peer's send buffer, write to output)
 *   c3_func: RDMA recv buffer -> output (read from RDMA recv buffer, write to output)
 *
 * Data flow for scatter_dim=1, gather_dim=0:
 *   Input: [X, num_ranks*Y, Z] -> each rank sends [X, Y, Z] chunk j to rank j
 *   Output: [num_ranks*X, Y, Z] -> each rank receives [X, Y, Z] from all ranks
 */
template <bool kF8 = false, bool kSingleNode = false, int kFlat = 0>
__global__ void __launch_bounds__(1024, 1)
scattergather_kernel(
    uint64_t *args,             // Interleaved [in_ptr, in_X, in_Y, in_Z, out_ptr, out_X, out_Y, out_Z] * nargs
    size_t nargs,               // Number of tensors
    void *gin_win_ptr,          // GIN window base pointer
    void **p2p_ptrs,            // P2P pointers for NVL access
    int *gmem_barrier,          // Signal area
    uint64_t *debug_buf,
    ncclWindow_t gin_win,      // GIN window handle
    const int nvl_ring,
    const int rdma_ring,
    const size_t rdma_unroll,
    int round_n,
    int rank,
    int num_local_ranks,
    int num_ranks,
    ncclDevComm dev_comm,
    const int c1_warps,
    const int cord_warps,
    const int c_warps,
    const int scatter_dim
) {
    const int num_nodes = num_ranks / num_local_ranks;
    const int node_id = rank / num_local_ranks;
    const int local_rank = rank % num_local_ranks;
    const uint32_t sm_id = blockIdx.x, num_sms = gridDim.x;
    // NOTE: We process data as raw bytes (float4 = 16 bytes), no type-specific handling needed
    constexpr bool kSrcFlat = kFlat & 1;
    constexpr bool kDstFlat = kFlat & 2;
    constexpr bool kLogZ = false;  // legacy kernel: no power-of-2 Z fast path (kLogZ is on the p2p kernel only)
    // Shared memory for tensor metadata
    __shared__ int prefix_f4[SG_MAX_TENSORS];   // Cumulative f4 count for input
    __shared__ int tensor_f4[SG_MAX_TENSORS];   // Per-tensor f4 count for input
    __shared__ uint64_t cached_args[SG_MAX_TENSORS * 8];  // Cached args from global memory
    
    DEVICE_ASSERT(nargs <= SG_MAX_TENSORS);
    
    // Load args into shared memory (8 uint64_t per tensor)
    // Each thread loads multiple elements to cover nargs * 8 elements
    for (int i = threadIdx.x; i < nargs * 8; i += blockDim.x) {
        cached_args[i] = args[i];
    }
    
    // Sync after loading cached_args
    __syncthreads();
    
    // Initialize tensor metadata from cached args (now in shared memory)
    // Format: [in_ptr, in_X, in_Y, in_Z, out_ptr, out_X, out_Y, out_Z] per tensor
    // Z is already in bytes, so f4_count = X * Z / 16 (where 16 = sizeof(float4))
    if (threadIdx.x < nargs) {
        // Input: [ptr, X, Y, Z] at offset i*8 - read from shared memory.
        // in-Y (arg +2) carries the S-row byte stride (== row_bytes when dense),
        // not the redundant num_ranks — so tensor_f4 = X*Z/16 = whole tensor.
        const uint64_t X = cached_args[threadIdx.x * 8 + 1];
        const uint64_t Z = cached_args[threadIdx.x * 8 + 3];  // Z in bytes
        const size_t f4_count = static_cast<int>(static_cast<uint64_t>(X) * Z / sizeof(float4));
        tensor_f4[threadIdx.x] = f4_count;
        prefix_f4[threadIdx.x] = f4_count;  // No extra alignment needed
    }
    __syncthreads();
    
    // Compute prefix sums
    if ((threadIdx.x >> 5) == 0) {
        hillis_steele_sum<EAGER_SCOPE_WARP>(prefix_f4, nargs, threadIdx.x);
    }
    __syncthreads();
    
    // Calculate work distribution
    const int total_f4 = prefix_f4[nargs - 1];
    const size_t slot_bytes = rdma_unroll * blockDim.x * sizeof(float4);
    const int slot_f4 = static_cast<int>(rdma_unroll * blockDim.x);
    
    int smf4start, smf4end;
    // Divide work in float8 (2×float4) units so smf4start is always even (32-byte aligned),
    // enabling the float8 vectorized path without misaligned address errors.
    if constexpr (kF8 && SG_F8_ARCH_SUPPORTED) {
        const int total_f8 = total_f4 / 2;
        int smf8start, smf8end;
        get_work_range(total_f8, num_sms, sm_id, smf8start, smf8end);
        smf4start = smf8start * 2;
        smf4end = smf8end * 2;
    } else {
        get_work_range(total_f4, num_sms, sm_id, smf4start, smf4end);
    }
    
    const int unrolled_times = (smf4end - smf4start) / slot_f4;
    const int tailf4 = (smf4end - smf4start) - unrolled_times * slot_f4;
    
    // Total segments per dst_local: each dst_local processes seg_batch slots
    const int seg_batch = unrolled_times + (tailf4 > 0);

    // GIN context
    ncclGin gin { dev_comm, static_cast<int>(sm_id) };
    ncclTeam world;
    if (num_nodes > 1) {
        world = ncclTeamWorld(dev_comm);
    }

    // Signal pointer lambdas
    auto sig_ptr = [&](const int& peer) -> uint8_t* {
        if (peer == local_rank) {
            return reinterpret_cast<uint8_t*>(gmem_barrier) + 1024;
        } else {
            return reinterpret_cast<uint8_t*>(__ldg(reinterpret_cast<const uint64_t*>(gmem_barrier) + peer)) +
                   (reinterpret_cast<uint8_t*>(gmem_barrier) - reinterpret_cast<uint8_t*>(gin_win_ptr)) + 1024;
        }
    };

    // Signal slot management
    auto c1_gstart_ptr = [&]() { return reinterpret_cast<uint64_t*>(sig_ptr(local_rank)) + 2 * num_local_ranks * gridDim.x + blockIdx.x; };
    const uint64_t gstart_id = __ldg(reinterpret_cast<const uint64_t*>(c1_gstart_ptr()));

    __syncthreads();  // ensure all setup (cached_args, prefix_f4) is visible before warp groups diverge
    
    // ===== 4-path signal layout (per-SM) =====
    // Offsets in u64 from sig_ptr(peer). The gstart slot counter occupies
    // [2*Nlr*num_sms, 2*Nlr*num_sms + num_sms) (see c1_gstart_ptr above), so
    // the 4-path signals MUST start after it to avoid overlap. For num_nodes≥2,
    // c1_picked alone (num_sms*num_nodes*Nlr) is wider than the pre-gstart area
    // (2*Nlr*num_sms), so everything goes after gstart.
    //   [base .. base + num_sms*num_nodes*Nlr)                    : c1_done[sm][N][src] / c2_got[sm][N][src]
    //   [base + num_sms*num_nodes*Nlr .. ..+num_sms*num_nodes*Nlr)  : c1_picked[sm][N][dst] / c2_ack[sm][N][dst]
    //   [base + 2*num_sms*num_nodes*Nlr .. ..+num_sms*num_nodes)   : c3_signal[sm][N]
    //   [base + 2*num_sms*num_nodes*Nlr+num_sms*num_nodes .. ..+num_sms*num_nodes): c3_ack[sm][N]
    const int R = std::max(nvl_ring, rdma_ring);
    const size_t sig_base = (size_t)2 * num_local_ranks * num_sms + num_sms;  // after gstart
    const size_t sig_base_c1done = sig_base;
    const size_t sig_base_c1picked = sig_base + (size_t)num_sms * num_nodes * num_local_ranks;
    const size_t sig_base_c3signal = sig_base + 2 * (size_t)num_sms * num_nodes * num_local_ranks;
    const size_t sig_base_c3ack = sig_base_c3signal + (size_t)num_sms * num_nodes;

    // c1_done[sm][N][src=local_rank]: source writes to forwarder (dst_peer)'s signal (NVL store).
    auto c1_done_st_ptr = [&](const int& N, const int& dst_peer) -> uint64_t* {
        return reinterpret_cast<uint64_t*>(sig_ptr(dst_peer)) + sig_base_c1done + ((size_t)sm_id * num_nodes + N) * num_local_ranks + local_rank;
    };
    // c2_got[sm][N][src]: forwarder (us) reads own signal.
    auto c2_got_ld_ptr = [&](const int& N, const int& src_peer) -> uint64_t* {
        return reinterpret_cast<uint64_t*>(sig_ptr(local_rank)) + sig_base_c1done + ((size_t)sm_id * num_nodes + N) * num_local_ranks + src_peer;
    };
    // c2_ack[sm][N][dst=local_rank]: forwarder (us) writes to source src_peer's signal (NVL store).
    auto c2_ack_st_ptr = [&](const int& N, const int& src_peer) -> uint64_t* {
        return reinterpret_cast<uint64_t*>(sig_ptr(src_peer)) + sig_base_c1picked + ((size_t)sm_id * num_nodes + N) * num_local_ranks + local_rank;
    };
    // c1_picked[sm][N][dst]: source (us) reads own signal.
    auto c1_picked_ld_ptr = [&](const int& N, const int& dst_peer) -> uint64_t* {
        return reinterpret_cast<uint64_t*>(sig_ptr(local_rank)) + sig_base_c1picked + ((size_t)sm_id * num_nodes + N) * num_local_ranks + dst_peer;
    };
    // c3_signal[sm][N]: cord writes, c3 group reads. Local.
    auto c3_signal_ptr = [&](const int& N) -> uint64_t* {
        return reinterpret_cast<uint64_t*>(sig_ptr(local_rank)) + sig_base_c3signal + (size_t)sm_id * num_nodes + N;
    };
    // c3_ack[sm][N]: c3 group writes, cord reads. Local.
    auto c3_ack_ptr = [&](const int& N) -> uint64_t* {
        return reinterpret_cast<uint64_t*>(sig_ptr(local_rank)) + sig_base_c3ack + (size_t)sm_id * num_nodes + N;
    };
    
    // GIN signal indices for multi-node
    auto rdma_ready_sig_w_index = [&]() { return static_cast<uint32_t>(sm_id * num_nodes + node_id); };
    auto rdma_ready_sig_r_index = [&](const int& src_node) { return static_cast<uint32_t>(sm_id * num_nodes + src_node); };
    auto rdma_picked_sig_w_index = [&]() { return static_cast<uint32_t>(num_sms * num_nodes + sm_id * num_nodes + node_id); };
    auto rdma_picked_sig_r_index = [&](const int& dst_node) { return static_cast<uint32_t>(num_sms * num_nodes + sm_id * num_nodes + dst_node); };
    
    // ===== Push data accessors (per-SM sliced) =====
    // Send buf layout within each GPU's gin_win: [sm][dst_node][R slot][src_local].
    // c1 NVL-store target: write to forwarder (dst_local)'s send buf, sub-slot = local_rank.
    auto peer_send_st_ptr = [&](const size_t& slot, const int& dst_rank) -> float4* {
        const int dst_node = dst_rank / num_local_ranks;
        const int dst_local = dst_rank % num_local_ranks;
        return reinterpret_cast<float4*>(reinterpret_cast<uint8_t*>(p2p_ptrs[dst_local]) +
            (((size_t)sm_id * num_nodes + dst_node) * R + (slot % R)) * num_local_ranks * slot_bytes + (size_t)local_rank * slot_bytes);
    };
    // c2 local-read: read OWN send buf at source-slot = src_local.
    auto my_send_ld_ptr = [&](const size_t& slot, const int& src_local) -> float4* {
        return reinterpret_cast<float4*>(reinterpret_cast<uint8_t*>(gin_win_ptr) +
            (((size_t)sm_id * num_nodes + node_id) * R + (slot % R)) * num_local_ranks * slot_bytes + (size_t)src_local * slot_bytes);
    };
    // c3 local-read: read OWN recv buf at source-slot = src_local.
    auto my_recv_ptr = [&](const int& src_node, const int& src_local, const size_t& slot) -> float4* {
        const size_t send_area = (size_t)num_sms * num_ranks * R * slot_bytes;
        return reinterpret_cast<float4*>(reinterpret_cast<uint8_t*>(gin_win_ptr) + send_area +
            (((size_t)sm_id * num_nodes + src_node) * R + (slot % R)) * num_local_ranks * slot_bytes + (size_t)src_local * slot_bytes);
    };
    // RDMA send offset (per-SM slice): from our send buf to remote recv.
    auto rdma_send_offset = [&](const int& dst_node, const size_t& slot) -> uint64_t {
        return (((size_t)sm_id * num_nodes + dst_node) * R + (slot % R)) * num_local_ranks * slot_bytes;
    };
    // RDMA recv offset (per-SM slice): remote writes into our recv buf at this offset.
    auto rdma_recv_offset = [&](const int& src_node, const size_t& slot) -> uint64_t {
        const size_t send_area = (size_t)num_sms * num_ranks * R * slot_bytes;
        return send_area + (((size_t)sm_id * num_nodes + src_node) * R + (slot % R)) * num_local_ranks * slot_bytes;
    };
    // ===== Warp specialization: 3 groups (c1 / cord / merged consumer) =====
    // Warp counts are runtime (kernel args); bar.sync accepts register count.
    // Sum must equal 32 (the CTA warp count) so every warp is assigned.
    // The merged consumer handles both local-node (c2) and remote-node (c3)
    // sources in one group — see the consumer branch below.
    const int C1_WARPS = c1_warps;
    const int CORD_WARPS = cord_warps;
    const int C_WARPS = c_warps;
    const int C1_THREADS = C1_WARPS * 32;
    const int C_THREADS = C_WARPS * 32;
    DEVICE_ASSERT(C1_WARPS + CORD_WARPS + C_WARPS == 32);
    const auto warp_id = threadIdx.x >> 5;
    const auto lane_id = threadIdx.x & 31;

    SG_DEBUG_RECORD_TS_AND_BATCH(seg_batch);

    if (warp_id < C1_WARPS) {
        // ===== Group A: c1 producer (input → NVL-store to forwarder's send buf) =====
        const int c1_tid = threadIdx.x;
        uint32_t c1_progress = 0;
        auto start_ts = clock64();
        while (c1_progress < seg_batch) {
            const uint32_t slot_idx = c1_progress;
            const size_t gslot = gstart_id + slot_idx;

            // Flow control: wait c1_picked[sm][N][dst] >= gslot+1-R for all (N, dst).
            // First R slots bootstrap (no ack needed). Per-thread distributed spin
            // (each thread owns a subset of (N, dst)); bar.sync after ensures all
            // threads passed before any writes to the reused send-buffer slot.
            if (slot_idx >= (uint32_t)R) {
                const uint64_t wait_val = gslot + 1 - R;
                const int total = num_nodes * num_local_ranks;
                for (int i = c1_tid; i < total; i += C1_THREADS) {
                    const int N = i / num_local_ranks;
                    const int dst = i % num_local_ranks;
                    auto bt = clock64();
                    while (true) {
                        if (ld_acquire_sys_global(c1_picked_ld_ptr(N, dst)) >= wait_val) break;
                        if (clock64() - bt > NUM_TIMEOUT_CYCLES) {
                            printf("[rank %d sm %d] c1 flow ctrl stuck N=%d dst=%d want=%lu\n", rank, sm_id, N, dst, wait_val);
                            __trap();
                        }
                    }
                }
                WG_BAR_SYNC(1, C1_THREADS);
            }

            for (int __p = 0; __p < num_nodes; ++__p) {
                const int _p = (__p + sm_id) % num_nodes;
                const auto f4start = smf4start + slot_idx * slot_f4;
                const auto nfloat4 = (slot_idx < unrolled_times) ? slot_f4 : tailf4;

                for (int __lr = 0; __lr < num_local_ranks; ++__lr) {
                    // Swizzle dst_local by (local_rank + 1) to spread NVL stores
                    // across all dst peers simultaneously. Per-source shift (not
                    // sm_id, which already swizzles the N dimension) — using sm_id
                    // for both would correlate N and dst_local on the same SM
                    // diagonal, causing (N, dst_local) pair clustering. +1 avoids
                    // the lr=0 / local_rank=0 self-loop landing on dst_local=0.
                    const int local_rank_ = (__lr + local_rank + 1) % num_local_ranks;
                    const auto dst_rank = _p * num_local_ranks + local_rank_;
                    int tidx = -1, cbound = -1, clowerbound = 0; uint64_t ciZ = 0, ciZ_log2 = 0, ciZ_mask = 0;
                    uint64_t ciptr = 0, ciStride = 0, ciBytes = 0;
                    auto ld_func = [&](const int& f4idx) {
                        return sg_unified_ld<kSrcFlat, kLogZ, float4>(nargs, cached_args, prefix_f4, tensor_f4, dst_rank, num_ranks, f4idx, tidx, cbound, clowerbound, ciZ, ciZ_log2, ciZ_mask, ciptr, ciStride, ciBytes, scatter_dim);
                    };
                    auto f8_ld_func = [&](const int& f8idx) {
                        return sg_unified_ld<kSrcFlat, kLogZ, float8>(nargs, cached_args, prefix_f4, tensor_f4, dst_rank, num_ranks, f8idx * 2, tidx, cbound, clowerbound, ciZ, ciZ_log2, ciZ_mask, ciptr, ciStride, ciBytes, scatter_dim);
                    };
                    if (dst_rank == rank) {
                        // Self-copy: input -> output directly.
                        int out_tidx = -1, out_cbound = -1, out_lb = 0; uint64_t out_Z = 0, out_Z_log2 = 0, out_Z_mask = 0;
                        uint64_t coptr = 0;
                        auto st_func = [&](const int& f4idx, const float4& value) {
                            sg_unified_st<kDstFlat, kLogZ, float4>(nargs, cached_args, prefix_f4, tensor_f4, dst_rank, num_ranks, f4idx, out_tidx, out_cbound, out_lb, out_Z, out_Z_log2, out_Z_mask, coptr, value);
                        };
                        auto f8_st_func = [&](const int& f8idx, const float8& value) {
                            sg_unified_st<kDstFlat, kLogZ, float8>(nargs, cached_args, prefix_f4, tensor_f4, dst_rank, num_ranks, f8idx * 2, out_tidx, out_cbound, out_lb, out_Z, out_Z_log2, out_Z_mask, coptr, value);
                        };
                        if constexpr (kF8 && SG_F8_ARCH_SUPPORTED) {
                            UNROLLED_GROUP_COPY(SG_UNROLL_FACTOR, c1_tid, C1_THREADS, nfloat4 / 2, f4start / 2, f4start / 2, f8_ld_func, f8_st_func);
                        } else {
                            UNROLLED_GROUP_COPY(SG_UNROLL_FACTOR, c1_tid, C1_THREADS, nfloat4, f4start, f4start, ld_func, st_func);
                        }
                    } else {
                        // NVL-store to forwarder (dst_local)'s send buf.
                        auto st_ptr = peer_send_st_ptr(gslot, dst_rank);
                        if constexpr (kF8 && SG_F8_ARCH_SUPPORTED) {
                            UNROLLED_GROUP_COPY(SG_UNROLL_FACTOR, c1_tid, C1_THREADS, nfloat4 / 2, reinterpret_cast<float8*>(st_ptr), f4start / 2, f8_ld_func, st_na_global);
                        } else {
                            UNROLLED_GROUP_COPY(SG_UNROLL_FACTOR, c1_tid, C1_THREADS, nfloat4, st_ptr, f4start, ld_func, st_na_global);
                        }
                    }
                }
                __threadfence_system();
                WG_BAR_SYNC(1, C1_THREADS);
                // Write c1_done[sm][N=_p][dst] = gslot+1 (NVL-store to forwarders' signals).
                if (warp_id == 0 && lane_id < num_local_ranks) {
                    st_release_sys_global(c1_done_st_ptr(_p, lane_id), gslot + 1);
                }
            }
            c1_progress++;
            start_ts = clock64();
        }
    } else if (warp_id < C1_WARPS + CORD_WARPS) {
        // ===== Group B: cord (4 warps, per-SM). Each warp a sub-role. =====
        if constexpr (!kSingleNode) {
            const int cord_subrole = warp_id - C1_WARPS;
            if (cord_subrole == 0) {
                // gin.put: poll c2_got[sm][N][src] min across src → gin.put.
                for (uint32_t slot = 0; slot < (uint32_t)seg_batch; ++slot) {
                    const size_t pending_slot = gstart_id + slot + 1;
                    for (int N = 0; N < num_nodes; ++N) {
                        if (N == node_id) continue;
                        auto bt = clock64();
                        uint64_t mn;
                        do {
                            mn = UINT64_MAX;
                            if (lane_id < num_local_ranks) mn = ld_acquire_sys_global(c2_got_ld_ptr(N, lane_id));
                            mn = warp_reduce_min(mn);
                            if (mn < pending_slot && clock64() - bt > NUM_TIMEOUT_CYCLES) {
                                if (lane_id == 0) printf("[rank %d sm %d] cord gin.put stuck N=%d slot=%u pending=%lu min=%lu\n", rank, sm_id, N, slot, pending_slot, mn);
                                __trap();
                            }
                        } while (mn < pending_slot);
                        // gin.put the whole per-SM batch from our send buf to remote
                        // (N, local_rank)'s recv buf. One thread issues — the device
                        // kernel's gin context is per-SM (sm_id), so multi-lane QP
                        // split would contend on one QP.
                        if (elect_one_sync()) {
                            const size_t batch_bytes = slot_bytes * num_local_ranks;
                            gin.put(world, N * num_local_ranks + local_rank, gin_win,
                                    rdma_recv_offset(node_id, pending_slot - 1),
                                    gin_win, rdma_send_offset(N, pending_slot - 1),
                                    batch_bytes, ncclGin_SignalInc{rdma_ready_sig_w_index()});
                        }
                        __syncwarp();
                    }
                }
            } else if (cord_subrole == 1) {
                // c2_ack broadcast: poll GIN rdma_picked → c2_ack[sm][N][src] for all src.
                for (uint32_t slot = 0; slot < (uint32_t)seg_batch; ++slot) {
                    const size_t wait_slot = gstart_id + slot + 1;
                    if (lane_id < GIN_QPS) {
                        for (int N = 0; N < num_nodes; ++N) {
                            if (N == node_id) continue;
                            LOOP_READ_GIN_SIGNAL_64GE(wait_slot, gin.readSignal(rdma_picked_sig_r_index(N)), N * num_local_ranks + local_rank, lane_id);
                        }
                    }
                    __syncwarp();
                    if (lane_id < num_local_ranks) {
                        for (int N = 0; N < num_nodes; ++N) {
                            if (N == node_id) continue;
                            st_release_sys_global(c2_ack_st_ptr(N, lane_id), wait_slot);
                        }
                    }
                    __syncwarp();
                }
            } else if (cord_subrole == 2) {
                // c3_signal: poll GIN rdma_ready → c3_signal[sm][N].
                for (uint32_t slot = 0; slot < (uint32_t)seg_batch; ++slot) {
                    const size_t wait_slot = gstart_id + slot + 1;
                    if (lane_id < GIN_QPS) {
                        for (int N = 0; N < num_nodes; ++N) {
                            if (N == node_id) continue;
                            LOOP_READ_GIN_SIGNAL_64GE(wait_slot, gin.readSignal(rdma_ready_sig_r_index(N)), N * num_local_ranks + local_rank, lane_id);
                        }
                    }
                    __syncwarp();
                    if (elect_one_sync()) {
                        for (int N = 0; N < num_nodes; ++N) {
                            if (N == node_id) continue;
                            st_release_sys_global(c3_signal_ptr(N), wait_slot);
                        }
                    }
                    __syncwarp();
                }
            } else {
                // c3_ack → gin.signal: poll c3_ack[sm][N] → gin.signal ack.
                for (uint32_t slot = 0; slot < (uint32_t)seg_batch; ++slot) {
                    const size_t wait_slot = gstart_id + slot + 1;
                    if (elect_one_sync()) {
                        for (int N = 0; N < num_nodes; ++N) {
                            if (N == node_id) continue;
                            LL_READ_64GE(wait_slot, c3_ack_ptr(N), N * num_local_ranks + local_rank);
                        }
                    }
                    __syncwarp();
                    if (elect_one_sync()) {
                        for (int N = 0; N < num_nodes; ++N) {
                            if (N == node_id) continue;
                            gin.signal(world, N * num_local_ranks + local_rank, ncclGin_SignalInc{rdma_picked_sig_w_index()});
                        }
                    }
                    __syncwarp();
                }
            }
        }
    } else {
        // ===== Merged consumer group: c2 (local-node peers) + c3 (remote nodes) =====
        // Bitmap wait-for-any across both source types. Per slot, the leader
        // polls c2_got[node_id][src_local] for each unprocessed local peer AND
        // c3_signal[N] for each unprocessed non-local N, builds two bitmasks,
        // publishes them via smem, and the group processes whatever's ready
        // this round. Acks are written per-source as each lands — c2_ack
        // unblocks the source's c1 flow control (c1_picked), c3_ack unblocks
        // cord's gin.signal for the remote RDMA path.
        //
        // Merging c2+c3 lets warps that finished c2 work (NVL-arrives early)
        // do c3 work (RDMA-arrives late) instead of idling in a separate
        // group — freeing warps for c1 (NVL store) without losing c3 throughput.
        DEVICE_ASSERT(num_local_ranks <= 32 && num_nodes <= 32);
        const int c_tid = threadIdx.x - (C1_WARPS + CORD_WARPS) * 32;
        const int c_first_warp = C1_WARPS + CORD_WARPS;
        // c2_target includes the self bit (src_local == local_rank): c1
        // self-copies input→output for the self-pair (no NVL store, no
        // c1_done written), so the consumer must synthesise the self-ack
        // (c2_ack[node_id][local_rank] = gslot+1) to unblock c1's flow
        // control on c1_picked[node_id][local_rank]. The leader treats the
        // self bit as always-ready; the processing loop skips the copy.
        uint32_t c2_target = 0;
        for (int s = 0; s < num_local_ranks; ++s) c2_target |= (1u << s);
        uint32_t c3_target = 0;
        if constexpr (!kSingleNode) {
            for (int N = 0; N < num_nodes; ++N) {
                if (N != node_id) c3_target |= (1u << N);
            }
        }
        __shared__ uint32_t c2_new_mask;
        __shared__ uint32_t c3_new_mask;
        uint32_t c_progress = 0;
        auto start_ts = clock64();
        while (c_progress < seg_batch) {
            const uint32_t slot_idx = c_progress;
            const size_t gslot = gstart_id + slot_idx;
            uint32_t c2_processed = 0;
            uint32_t c3_processed = 0;
            while (c2_processed != c2_target || c3_processed != c3_target) {
                // Leader polls both signal types, publishes two bitmasks via smem.
                if (warp_id == c_first_warp) {
                    uint32_t c2_new = 0, c3_new = 0;
                    auto bt = clock64();
                    while (c2_new == 0 && c3_new == 0) {
                        uint32_t c2_ready = 0, c3_ready = 0;
                        for (int s = 0; s < num_local_ranks; ++s) {
                            // Self bit is always ready (c1 self-copies, no signal).
                            if (s == local_rank) { c2_ready |= (1u << s); continue; }
                            if (c2_processed & (1u << s)) continue;
                            if (ld_acquire_sys_global(c2_got_ld_ptr(node_id, s)) >= gslot + 1) {
                                c2_ready |= (1u << s);
                            }
                        }
                        if constexpr (!kSingleNode) {
                            for (int N = 0; N < num_nodes; ++N) {
                                if (N == node_id) continue;
                                if (c3_processed & (1u << N)) continue;
                                if (ld_acquire_sys_global(c3_signal_ptr(N)) >= gslot + 1) {
                                    c3_ready |= (1u << N);
                                }
                            }
                        }
                        c2_new = c2_ready & ~c2_processed;
                        c3_new = c3_ready & ~c3_processed;
                        if (c2_new == 0 && c3_new == 0 && clock64() - bt > NUM_TIMEOUT_CYCLES) {
                            if (lane_id == 0) {
                                printf("[rank %d sm %d] consumer wait stuck slot=%u want=%lu", rank, sm_id, slot_idx, gslot + 1);
                                for (int s = 0; s < num_local_ranks; ++s) {
                                    if (s == local_rank) continue;
                                    printf(" c2got[s%d]=%lu", s, (uint64_t)ld_acquire_sys_global(c2_got_ld_ptr(node_id, s)));
                                }
                                if constexpr (!kSingleNode) {
                                    for (int N = 0; N < num_nodes; ++N) {
                                        if (N == node_id) continue;
                                        printf(" c3sig[N%d]=%lu", N, (uint64_t)ld_acquire_sys_global(c3_signal_ptr(N)));
                                    }
                                }
                                printf("\n");
                                __trap();
                            }
                        }
                    }
                    c2_new_mask = c2_new;
                    c3_new_mask = c3_new;
                }
                WG_BAR_SYNC(2, C_THREADS);
                const uint32_t c2_new = c2_new_mask;
                const uint32_t c3_new = c3_new_mask;
                const auto f4start = smf4start + slot_idx * slot_f4;
                const auto nfloat4 = (slot_idx < unrolled_times) ? slot_f4 : tailf4;
                // c2 sources (local peers): local-read own send buf → output.
                for (int __p = 0; __p < num_local_ranks; ++__p) {
                    const int _p = (__p + sm_id) % num_local_ranks;
                    if (_p == local_rank) continue;
                    if (!(c2_new & (1u << _p))) continue;
                    const int src_rank = node_id * num_local_ranks + _p;
                    auto ld_ptr = my_send_ld_ptr(gslot, _p);
                    int out_tidx = -1, out_cbound = -1, out_lb = 0; uint64_t out_Z = 0, out_Z_log2 = 0, out_Z_mask = 0;
                    uint64_t coptr = 0;
                    auto st_func = [&](const int& f4idx, const float4& value) {
                        sg_unified_st<kDstFlat, kLogZ, float4>(nargs, cached_args, prefix_f4, tensor_f4, src_rank, num_ranks, f4idx, out_tidx, out_cbound, out_lb, out_Z, out_Z_log2, out_Z_mask, coptr, value);
                    };
                    if constexpr (kF8 && SG_F8_ARCH_SUPPORTED) {
                        auto f8_st_func = [&](const int& f8idx, const float8& value) {
                            sg_unified_st<kDstFlat, kLogZ, float8>(nargs, cached_args, prefix_f4, tensor_f4, src_rank, num_ranks, f8idx * 2, out_tidx, out_cbound, out_lb, out_Z, out_Z_log2, out_Z_mask, coptr, value);
                        };
                        UNROLLED_GROUP_COPY(SG_UNROLL_FACTOR, c_tid, C_THREADS, nfloat4 / 2, f4start / 2, reinterpret_cast<const float8*>(ld_ptr), ld_nc_global, f8_st_func);
                    } else {
                        UNROLLED_GROUP_COPY(SG_UNROLL_FACTOR, c_tid, C_THREADS, nfloat4, f4start, ld_ptr, ld_nc_global, st_func);
                    }
                }
                // c3 sources (remote nodes): local-read own recv buf → output.
                if constexpr (!kSingleNode) {
                    for (int __node = 0; __node < num_nodes; ++__node) {
                        const int _node = (__node + sm_id) % num_nodes;
                        if (_node == node_id) continue;
                        if (!(c3_new & (1u << _node))) continue;
                        for (int __p = 0; __p < num_local_ranks; ++__p) {
                            const int _p = (__p + sm_id) % num_local_ranks;
                            const int src_rank = _node * num_local_ranks + _p;
                            auto ld_ptr = my_recv_ptr(_node, _p, gslot);
                            int out_tidx = -1, out_cbound = -1, out_lb = 0; uint64_t out_Z = 0, out_Z_log2 = 0, out_Z_mask = 0;
                            uint64_t coptr = 0;
                            auto st_func = [&](const int& f4idx, const float4& value) {
                                sg_unified_st<kDstFlat, kLogZ, float4>(nargs, cached_args, prefix_f4, tensor_f4, src_rank, num_ranks, f4idx, out_tidx, out_cbound, out_lb, out_Z, out_Z_log2, out_Z_mask, coptr, value);
                            };
                            if constexpr (kF8 && SG_F8_ARCH_SUPPORTED) {
                                auto f8_st_func = [&](const int& f8idx, const float8& value) {
                                    sg_unified_st<kDstFlat, kLogZ, float8>(nargs, cached_args, prefix_f4, tensor_f4, src_rank, num_ranks, f8idx * 2, out_tidx, out_cbound, out_lb, out_Z, out_Z_log2, out_Z_mask, coptr, value);
                                };
                                UNROLLED_GROUP_COPY(SG_UNROLL_FACTOR, c_tid, C_THREADS, nfloat4 / 2, f4start / 2, reinterpret_cast<const float8*>(ld_ptr), ld_nc_global, f8_st_func);
                            } else {
                                UNROLLED_GROUP_COPY(SG_UNROLL_FACTOR, c_tid, C_THREADS, nfloat4, f4start, ld_ptr, ld_nc_global, st_func);
                            }
                        }
                    }
                }
                __threadfence_system();
                WG_BAR_SYNC(2, C_THREADS);
                // Leader: write per-source acks as each lands.
                if (warp_id == c_first_warp) {
                    // c2_ack[node_id][src_local] for all newly-ready src_locals
                    // (incl. self — c1 self-copies, so the consumer synthesises
                    // the self-ack to unblock c1's flow control).
                    for (int s = 0; s < num_local_ranks; ++s) {
                        if (!(c2_new & (1u << s))) continue;
                        st_release_sys_global(c2_ack_st_ptr(node_id, s), gslot + 1);
                    }
                    // c3_ack[N] (local; cord_subrole 3 reads).
                    if constexpr (!kSingleNode) {
                        for (int N = 0; N < num_nodes; ++N) {
                            if (N == node_id) continue;
                            if (!(c3_new & (1u << N))) continue;
                            st_release_sys_global(c3_ack_ptr(N), gslot + 1);
                        }
                    }
                }
                // ALL threads advance masks in lockstep (broadcast via smem).
                c2_processed |= c2_new;
                c3_processed |= c3_new;
            }
            c_progress++;
            start_ts = clock64();
        }
    }
    __syncthreads();
    
    // Update global slot counter
    if (threadIdx.x == 0) {
        st_na_global(c1_gstart_ptr(), static_cast<uint64_t>(gstart_id + seg_batch));
    }
    SG_DEBUG_RECORD_TS(2 + seg_batch * 2 * (num_nodes + num_local_ranks));
}


/**
 * Coordination kernel for scatter-gather multinode push (Phase 2).
 *
 * 4 warps, NO data copies (the host does all CE copies on copy_stream).
 * Each warp runs its own `for slot` loop; warps execute concurrently
 * (warp-level parallelism). GIN signal indices: rdma_ready = node_id
 * (write) / src_node (read); rdma_picked = num_nodes + node_id (write) /
 * dst_node (read). Same scheme as the previous cord (cord is <<<1,128>>>,
 * so sm_id=0 and the per-SM index space collapses to these).
 *
 *   Warp 0 (A, RDMA send):     poll c2_got[N][src] (own signal, written by
 *                              sources' post_c1 via NVL store) min across
 *                              src → gin.put batch from own send buffer to
 *                              remote (N, local_rank)'s recv buffer.
 *   Warp 1 (B, RDMA ack → c2_ack): poll GIN rdma_picked (remote consumed
 *                              recv) → NVL-store c2_ack[N][src] to each
 *                              source peer's signal. Unblocks sources' c1.
 *   Warp 2 (C, RDMA arrival → c3_signal): poll GIN rdma_ready (data landed)
 *                              → write c3_signal[N] locally. Unblocks host c3.
 *   Warp 3 (D, c3_ack → RDMA ack): poll c3_ack[N] (local, written by host
 *                              c3) → gin.signal ack to remote. Unblocks
 *                              remote sources' c1.
 *
 * Liveness: c1 → c1_done → Warp A gin.put → remote c3_signal → remote c3
 * → remote c3_ack → remote Warp D gin.signal → local Warp B c2_ack →
 * local c1 unblock. R ≥ 1 (first R slots bootstrap). See push_design.md.
 */
__global__ void __launch_bounds__(128, 1) scattergathercordkernel(
    ncclWindow_t gin_win, void *signal_buffer, int unroll, int nvl_ring, int rdma_ring,
    size_t current_slot, size_t work_slots, int round_n, int rank, int num_local_ranks,
    int num_ranks, ncclDevComm dev_comm) {

    const int warp_id = threadIdx.x >> 5;
    const int lane_id = threadIdx.x & 31;
    ncclGin gin {dev_comm, lane_id & (GIN_QPS - 1)};
    ncclTeam world = ncclTeamWorld(dev_comm);
    const size_t slot_bytes = sizeof(int4) * GIN_CTA_THREADS * unroll;
    sg_mem_layout ml(reinterpret_cast<void**>(signal_buffer), nvl_ring, rdma_ring, slot_bytes, rank, num_local_ranks, num_ranks);
    const int node_id = rank / num_local_ranks;
    const int local_rank = rank % num_local_ranks;
    const int num_nodes = num_ranks / num_local_ranks;

    if (warp_id == 0) {
        // Warp A: c2_got (min across src) → gin.put.
        for (size_t slot = 0; slot < work_slots; ++slot) {
            const size_t pending_slot = current_slot + slot + 1;
            for (int N = 0; N < num_nodes; ++N) {
                if (N == node_id) continue;
                // Spin until every source has NVL-stored into our send buffer
                // for node N at this slot (c2_got = c1_done, written by sources).
                uint64_t min_val = (lane_id < num_local_ranks) ? UINT64_MAX : UINT64_MAX;
                auto start_t = clock64();
                uint64_t reduced;
                do {
                    min_val = UINT64_MAX;
                    if (lane_id < num_local_ranks) {
                        min_val = ld_acquire_sys_global(ml.c2_got_ld_ptr(N, lane_id));
                    }
                    reduced = warp_reduce_min(min_val);
                    if (reduced < pending_slot && clock64() - start_t > NUM_TIMEOUT_CYCLES) {
                        if (lane_id == 0) printf("[rank %d cord A]: timeout N=%d slot=%lu pending=%lu min=%lu\n",
                                                 rank, N, slot, pending_slot, reduced);
                        __trap();
                    }
                } while (reduced < pending_slot);
                // gin.put the batch (num_local_ranks source sub-slots) from our
                // send buffer to remote (N, local_rank)'s recv buffer.
                if (lane_id < GIN_QPS) {
                    const size_t batch_bytes = slot_bytes * num_local_ranks;
                    const size_t per_qp_bytes = batch_bytes / GIN_QPS;
                    const size_t qp_offset = lane_id * per_qp_bytes;
                    gin.put(world, N * num_local_ranks + local_rank, gin_win,
                            ml.rdma_recv_offset(node_id, pending_slot - 1) + qp_offset,
                            gin_win, ml.rdma_send_offset(N, pending_slot - 1) + qp_offset,
                            per_qp_bytes, ncclGin_SignalInc{static_cast<uint32_t>(node_id)});
                }
                __syncwarp();
            }
        }
    } else if (warp_id == 1) {
        // Warp B: GIN rdma_picked (remote ack) → c2_ack broadcast to sources.
        for (size_t slot = 0; slot < work_slots; ++slot) {
            const size_t wait_slot = current_slot + slot + 1;
            // Wait until remote consumed recv buffer for this slot (ack from
            // each non-local N). Lanes < GIN_QPS poll via per-QP gin context.
            if (lane_id < GIN_QPS) {
                for (int N = 0; N < num_nodes; ++N) {
                    if (N == node_id) continue;
                    LOOP_READ_GIN_SIGNAL_64GE(wait_slot, gin.readSignal(num_nodes + N), N * num_local_ranks + local_rank, lane_id);
                }
            }
            __syncwarp();
            // NVL-store c2_ack[N][src] = wait_slot to each source peer's signal
            // (unblocks their c1 flow control). Each lane handles one source.
            if (lane_id < num_local_ranks) {
                for (int N = 0; N < num_nodes; ++N) {
                    if (N == node_id) continue;
                    st_release_sys_global(ml.c2_ack_st_ptr(N, lane_id), wait_slot);
                }
            }
            __syncwarp();
        }
    } else if (warp_id == 2) {
        // Warp C: GIN rdma_ready (arrival) → c3_signal (local write).
        for (size_t slot = 0; slot < work_slots; ++slot) {
            const size_t wait_slot = current_slot + slot + 1;
            if (lane_id < GIN_QPS) {
                for (int N = 0; N < num_nodes; ++N) {
                    if (N == node_id) continue;
                    LOOP_READ_GIN_SIGNAL_64GE(wait_slot, gin.readSignal(N), N * num_local_ranks + local_rank, lane_id);
                }
            }
            __syncwarp();
            if (elect_one_sync()) {
                for (int N = 0; N < num_nodes; ++N) {
                    if (N == node_id) continue;
                    st_release_sys_global(ml.c3_signal_ptr(N), wait_slot);
                }
            }
            __syncwarp();
        }
    } else {
        // Warp D: c3_ack (local, written by host c3) → gin.signal ack to remote.
        for (size_t slot = 0; slot < work_slots; ++slot) {
            const size_t wait_slot = current_slot + slot + 1;
            // Wait until host c3 read node N's recv buffer (c3_ack written by
            // post_c3 on copy_stream, visible to us via ld_acquire_sys).
            if (elect_one_sync()) {
                for (int N = 0; N < num_nodes; ++N) {
                    if (N == node_id) continue;
                    LL_READ_64GE(wait_slot, ml.c3_ack_ptr(N), N * num_local_ranks + local_rank);
                }
            }
            __syncwarp();
            // Ack the remote sender — its recv buffer can be reused.
            if (lane_id < GIN_QPS) {
                for (int N = 0; N < num_nodes; ++N) {
                    if (N == node_id) continue;
                    gin.signal(world, N * num_local_ranks + local_rank, ncclGin_SignalInc{static_cast<uint32_t>(num_nodes + node_id)});
                }
            }
            __syncwarp();
        }
    }
}

void scattergatherfunc(std::vector<TensorLayout> layouts, std::vector<TensorLayout> rlayouts, ncclWindow_t gin_win, void *gin_buffer, void *signal_buffer, void **p2p_ptrs, int unroll, int nvl_ring, int rdma_ring, size_t& total_send_slots, int capture_round_n, int rank, int num_local_ranks, int num_ranks, CUstreamBatchMemOpParams *mparam, ncclDevComm dev_comm, int num_sms, cudaStream_t stream, cudaStream_t copy_stream, const std::vector<uint64_t>& src_row_strides) {
    if (num_sms > 0) {
        printf("[rank %d]: scattergatherfunc num_sms > 0 is not implemented yet\n", rank);
        return;
    }

    const int num_tensors = layouts.size();
    if (num_tensors == 0) return;
    
    const int local_rank = rank % num_local_ranks;
    const int node_id = rank / num_local_ranks;
    const int num_nodes = num_ranks / num_local_ranks;
    const bool is_single_node = (num_local_ranks == num_ranks);
    const size_t slot_bytes = sizeof(int4) * GIN_CTA_THREADS * unroll;
    
    sg_mem_layout ml(p2p_ptrs, nvl_ring, rdma_ring, slot_bytes, rank, num_local_ranks, num_ranks);
    
    // Calculate total size across all tensors (per-rank chunk size)
    size_t total_chunk_bytes = 0;
    for (int i = 0; i < num_tensors; ++i) {
        total_chunk_bytes += align_up(layouts[i].get_total_bytes() / num_ranks, sizeof(int4));
    }
    
    const int send_times = ceil_div(total_chunk_bytes, slot_bytes);
    
    // Early return if no work to do
    if (send_times == 0) {
        return;
    }

    // Pre/post signal lambdas (push-based 4-path protocol). All CE ops issue
    // on copy_stream, so every signal wait/write targets copy_stream too. c3
    // is lagged (see main loop). R = max(nvl_ring, rdma_ring) is the single
    // ring; the first R slots bootstrap (no ack needed).
    const int R = ml.get_R();

    auto pre_c1 = [&](const size_t& gslot) {
        if (gslot >= (size_t)R) {
            // Flow control: wait c1_picked[N][dst] >= gslot+1-R for all (N, dst).
            // c1_picked (= c2_ack) is written by the forwarder: local N acked by
            // the destination's post_c2 (NVL store to our signal); remote N acked
            // by the cord's Warp B after GIN rdma_picked (also NVL store). Both
            // land at c1_picked_ld_ptr(N, dst) on our own signal area.
            const int n_entries = num_nodes * num_local_ranks;
            memset(mparam, 0, sizeof(CUstreamBatchMemOpParams) * n_entries);
            for (int N = 0; N < num_nodes; ++N) {
                for (int dst = 0; dst < num_local_ranks; ++dst) {
                    const int idx = N * num_local_ranks + dst;
                    mparam[idx].operation = CU_STREAM_MEM_OP_WAIT_VALUE_64;
                    mparam[idx].waitValue.operation = CU_STREAM_MEM_OP_WAIT_VALUE_64;
                    mparam[idx].waitValue.address = reinterpret_cast<CUdeviceptr>(ml.c1_picked_ld_ptr(N, dst));
                    mparam[idx].waitValue.value64 = gslot + 1 - R;
                    mparam[idx].waitValue.flags = CU_STREAM_WAIT_VALUE_GEQ;
                }
            }
            CUCHECK(cuStreamBatchMemOp(copy_stream, n_entries, mparam, 0));
        }
    };

    auto post_c1 = [&](const size_t& gslot) {
        // Notify each forwarder `dst` that we wrote data for node N. NVL-store
        // to the forwarder's signal area (for dst==local_rank it's a local write).
        const int n_entries = num_nodes * num_local_ranks;
        memset(mparam, 0, sizeof(CUstreamBatchMemOpParams) * n_entries);
        for (int N = 0; N < num_nodes; ++N) {
            for (int dst = 0; dst < num_local_ranks; ++dst) {
                const int idx = N * num_local_ranks + dst;
                mparam[idx].operation = CU_STREAM_MEM_OP_WRITE_VALUE_64;
                mparam[idx].writeValue.operation = CU_STREAM_MEM_OP_WRITE_VALUE_64;
                mparam[idx].writeValue.address = reinterpret_cast<CUdeviceptr>(ml.c1_done_st_ptr(N, dst));
                mparam[idx].writeValue.value64 = gslot + 1;
                mparam[idx].writeValue.flags = CU_STREAM_WRITE_VALUE_DEFAULT;
            }
        }
        CUCHECK(cuStreamBatchMemOp(copy_stream, n_entries, mparam, 0));
    };

    auto pre_c2 = [&](const size_t& gslot) {
        // Wait for all sources to have NVL-stored into our send buffer for the
        // local node (c1_done = c2_got). Cord handles remote N's RDMA.
        memset(mparam, 0, sizeof(CUstreamBatchMemOpParams) * num_local_ranks);
        for (int src = 0; src < num_local_ranks; ++src) {
            mparam[src].operation = CU_STREAM_MEM_OP_WAIT_VALUE_64;
            mparam[src].waitValue.operation = CU_STREAM_MEM_OP_WAIT_VALUE_64;
            mparam[src].waitValue.address = reinterpret_cast<CUdeviceptr>(ml.c2_got_ld_ptr(node_id, src));
            mparam[src].waitValue.value64 = gslot + 1;
            mparam[src].waitValue.flags = CU_STREAM_WAIT_VALUE_GEQ;
        }
        CUCHECK(cuStreamBatchMemOp(copy_stream, num_local_ranks, mparam, 0));
    };

    auto post_c2 = [&](const size_t& gslot) {
        // Tell each source its data for the local node was consumed (c2_ack).
        // NVL-store to each source peer's signal. Remote N is acked by the cord.
        memset(mparam, 0, sizeof(CUstreamBatchMemOpParams) * num_local_ranks);
        for (int src = 0; src < num_local_ranks; ++src) {
            mparam[src].operation = CU_STREAM_MEM_OP_WRITE_VALUE_64;
            mparam[src].writeValue.operation = CU_STREAM_MEM_OP_WRITE_VALUE_64;
            mparam[src].writeValue.address = reinterpret_cast<CUdeviceptr>(ml.c2_ack_st_ptr(node_id, src));
            mparam[src].writeValue.value64 = gslot + 1;
            mparam[src].writeValue.flags = CU_STREAM_WRITE_VALUE_DEFAULT;
        }
        CUCHECK(cuStreamBatchMemOp(copy_stream, num_local_ranks, mparam, 0));
    };

    auto pre_c3 = [&](const size_t& gslot) {
        // Wait for RDMA data arrival from each non-local node (cord Warp C
        // translates GIN rdma_ready → c3_signal).
        const int n_entries = num_nodes - 1;
        if (n_entries <= 0) return;
        memset(mparam, 0, sizeof(CUstreamBatchMemOpParams) * n_entries);
        int idx = 0;
        for (int N = 0; N < num_nodes; ++N) {
            if (N == node_id) continue;
            mparam[idx].operation = CU_STREAM_MEM_OP_WAIT_VALUE_64;
            mparam[idx].waitValue.operation = CU_STREAM_MEM_OP_WAIT_VALUE_64;
            mparam[idx].waitValue.address = reinterpret_cast<CUdeviceptr>(ml.c3_signal_ptr(N));
            mparam[idx].waitValue.value64 = gslot + 1;
            mparam[idx].waitValue.flags = CU_STREAM_WAIT_VALUE_GEQ;
            ++idx;
        }
        CUCHECK(cuStreamBatchMemOp(copy_stream, n_entries, mparam, 0));
    };

    auto post_c3 = [&](const size_t& gslot) {
        // Tell the cord (Warp D) we consumed node N's recv buffer; it will
        // gin.signal the RDMA ack back to the remote sender.
        const int n_entries = num_nodes - 1;
        if (n_entries <= 0) return;
        memset(mparam, 0, sizeof(CUstreamBatchMemOpParams) * n_entries);
        int idx = 0;
        for (int N = 0; N < num_nodes; ++N) {
            if (N == node_id) continue;
            mparam[idx].operation = CU_STREAM_MEM_OP_WRITE_VALUE_64;
            mparam[idx].writeValue.operation = CU_STREAM_MEM_OP_WRITE_VALUE_64;
            mparam[idx].writeValue.address = reinterpret_cast<CUdeviceptr>(ml.c3_ack_ptr(N));
            mparam[idx].writeValue.value64 = gslot + 1;
            mparam[idx].writeValue.flags = CU_STREAM_WRITE_VALUE_DEFAULT;
            ++idx;
        }
        CUCHECK(cuStreamBatchMemOp(copy_stream, n_entries, mparam, 0));
    };
    
    // Launch coordination kernel ONCE for multi-node (flat across all tensors)
    if (!is_single_node) {
        scattergathercordkernel<<<1, 128, 0, stream>>>(gin_win, signal_buffer, unroll, nvl_ring, rdma_ring,
            total_send_slots, send_times, capture_round_n, rank, num_local_ranks, num_ranks, dev_comm);
        CUDACHECK(cudaGetLastError());
    }
    
    // Track position for flat iteration
    int tidx = 0;
    
    const bool debug = false;  // debug output off by default

    size_t common_bytes, most_blocks, most_massive, src_stride, dst_stride;
    size_t src_X, src_Z, dst_X, dst_Z, src_x, src_z, dst_x, dst_z;
    bool src_piecewise;
    // Strided-S input support (0-SM stream path): src_row_stride is the
    // S-row byte pitch (stride(0)*esize); src_strided when it differs from
    // the dense pitch. src_rows_per_rank = rows per dst chunk (s0g1 only).
    bool src_strided;
    size_t src_row_stride, src_rows_per_rank;
    auto src_ptr = [&](const int& dst_rank) {
        uint8_t* base = layouts[tidx].get_base_ptr();
        if (src_strided) {
            if (src_piecewise) {
                // s1g0: the S-rows are the X dim, pitched by src_row_stride.
                return base + src_x * src_row_stride + dst_rank * src_Z + src_z;
            }
            // s0g1: rank dst_rank's chunk is src_rows_per_rank strided S-rows.
            const size_t row = dst_rank * src_rows_per_rank + src_z / dst_Z;
            return base + row * src_row_stride + (src_z % dst_Z);
        }
        return base + src_x * num_ranks * src_Z + dst_rank * src_Z + src_z;
    };
    auto dst_ptr = [&](const int& dst_rank) {
        return rlayouts[tidx].get_base_ptr() + dst_x * num_ranks * dst_Z + dst_rank * dst_Z + dst_z;
    };
    auto update_tensor_meta = [&](){
        if (layouts[tidx].get_total_bytes() == 0 || rlayouts[tidx].get_total_bytes() == 0) return false;
        src_X = layouts[tidx].get_X(), src_Z = layouts[tidx].get_Z();
        dst_X = rlayouts[tidx].get_X(), dst_Z = rlayouts[tidx].get_Z();
        src_piecewise = src_Z < dst_Z;
        common_bytes = std::min(src_Z, dst_Z);
        most_blocks = std::max(src_Z, dst_Z) / common_bytes;
        most_massive = std::min(src_X, dst_X);
        // Dense S-row pitch: s1g0 the X-dim rows are num_ranks*src_Z apart;
        // s0g1 each per-rank chunk is rows of dst_Z bytes.
        src_row_stride = src_row_strides[tidx];
        const size_t dense_row_pitch = src_piecewise ? (num_ranks * src_Z) : dst_Z;
        src_strided = (src_row_stride != dense_row_pitch);
        src_rows_per_rank = src_Z / dst_Z;  // s0g1: rows per dst chunk
        src_stride = src_strided ? src_row_stride : ((src_piecewise ? layouts[tidx].get_Y() : 1) * common_bytes);
        dst_stride = (src_piecewise ? 1 : rlayouts[tidx].get_Y()) * common_bytes;
        src_x = 0, src_z = 0;
        dst_x = 0, dst_z = 0;
        return true;
    };
    while (tidx < num_tensors && !update_tensor_meta()) {
        tidx += 1;
    }
    // Pending tasks per dst_rank for each stream
    size_t work_remain = 0;
    
    auto tensor_proceed = [&](const size_t& bytes) {
        work_remain -= bytes;
        src_z += bytes;
        dst_z += bytes;
        src_x += src_z / src_Z;
        dst_x += dst_z / dst_Z;
        src_z %= src_Z;
        dst_z %= dst_Z;
    };

    // Recorded c2/c3 work per slot. On a single in-order copy_stream, c2 of
    // slot N must follow post_c1(N) (pre_c2 waits on c2_ready=c1_done(N),
    // which post_c1 writes — if pre_c2 were queued before post_c1 it would
    // wait for a signal behind it → symmetric deadlock). So c1 is issued
    // immediately during the work-iteration, and c2/c3 params are recorded
    // by value for phased replay after post_c1 / post_c2 respectively. c3 is
    // additionally lagged K slots (see main loop). Params are snapshotted at
    // record time — the tidx/src_x/src_z state advances between slots.
    struct CopyWork {
        const uint8_t* src;
        uint8_t* dst;
        size_t bytes;           // frag path (1D)
        size_t common_bytes;    // 2D path
        size_t blocks;
        size_t dst_stride;
        bool src_piecewise;     // true -> 1D, false -> 2D
        bool is_frag;
    };
    std::vector<std::vector<CopyWork>> c2_work(send_times);
    std::vector<std::vector<CopyWork>> c3_work(send_times);
    // total_send_slots == base_gslot + st during slot st's body (it increments
    // at the end of each iteration), so total_send_slots - base_gslot == st.
    // Used to index c2_work/c3_work from the copy lambdas (which are defined
    // before the for-loop's `st` is in scope).
    const size_t base_gslot = total_send_slots;

    auto ring_copy2d_src = [&](const uint8_t* iptr, uint8_t* dst_ptr, const size_t& blocks) {
        if (src_piecewise || src_strided) {
            // Strided input: the S-rows are pitched src_stride (== src_row_stride),
            // so c1 must be a 2D copy even when the logical src block is large.
            CUDACHECK(cudaMemcpy2DAsync(dst_ptr, common_bytes, iptr, src_stride, common_bytes, blocks, cudaMemcpyDeviceToDevice, copy_stream));
        } else {
            // flat src, just 1d copy
            CUDACHECK(cudaMemcpyAsync(dst_ptr, iptr, blocks * common_bytes, cudaMemcpyDeviceToDevice, copy_stream));
        }
    };
    // Push-based accessors (Phase 2: single-node + multinode both push).
    // c1 NVL-stores to the forwarder's (dst_local's) send buffer at source-slot
    // = local_rank. c2 reads OWN send buffer at source-slot = peer. c3 reads
    // OWN recv buffer at source-slot = peer (rail-targeted RDMA lands in our
    // own recv buffer; no NVL load needed).
    auto c1_dst_ptr = [&](int target_rank, size_t slot, size_t off) -> uint8_t* {
        return ml.peer_send_st_ptr<uint8_t>(target_rank, slot) + off;
    };
    auto c2_src_ptr = [&](int peer, int target_rank, size_t slot, size_t off) -> const uint8_t* {
        return ml.my_send_ld_ptr<uint8_t>(slot, peer) + off;
    };
    auto copy2d = [&](const size_t& work_remain, const size_t& blocks) {
        for (int _peer = 0; _peer < num_local_ranks; ++_peer) {
            const int peer = (_peer + local_rank) % num_local_ranks;
            for (int node = 0; node < num_nodes; ++node) {
                const int target_rank = node * num_local_ranks + peer;
                uint8_t* iptr = src_ptr(target_rank);
                uint8_t* optr = dst_ptr(target_rank);

                if (target_rank == rank) {
                    // Local copy: input -> output directly (handled in the
                    // final local-to-local loop below).
                } else {
                    // c1: input -> forwarder's send buffer (issued immediately)
                    uint8_t *dst_ptr = c1_dst_ptr(target_rank, total_send_slots, slot_bytes - work_remain);
                    ring_copy2d_src(iptr, dst_ptr, blocks);

                    if (node == node_id) {
                        // c2: own send buffer -> output (same node) — record
                        const uint8_t *src_ptr = c2_src_ptr(peer, target_rank, total_send_slots, slot_bytes - work_remain);
                        c2_work[total_send_slots - base_gslot].push_back({src_ptr, optr, 0, common_bytes, blocks, dst_stride, src_piecewise, false});
                    } else {
                        // c3: own recv buffer -> output (cross node) — record
                        const uint8_t *src_ptr = ml.my_recv_ptr<uint8_t>(node, peer, total_send_slots) + slot_bytes - work_remain;
                        c3_work[total_send_slots - base_gslot].push_back({src_ptr, optr, 0, common_bytes, blocks, dst_stride, src_piecewise, false});
                    }
                }
            }
        }
    };
    auto copy_frag = [&] (const size_t& work_bytes) {
        for (int _peer = 0; _peer < num_local_ranks; ++_peer) {
            const int peer = (_peer + local_rank) % num_local_ranks;
            for (int node = 0; node < num_nodes; ++node) {
                const int target_rank = node * num_local_ranks + peer;
                uint8_t* iptr = src_ptr(target_rank);
                uint8_t* optr = dst_ptr(target_rank);

                if (target_rank == rank) {
                    // Local copy: handled in the final local-to-local loop.
                } else {
                    // c1: input -> forwarder's send buffer (issued immediately)
                    uint8_t *dst_ptr = c1_dst_ptr(target_rank, total_send_slots, slot_bytes - work_remain);
                    CUDACHECK(cudaMemcpyAsync(dst_ptr, iptr, work_bytes, cudaMemcpyDeviceToDevice, copy_stream));

                    if (node == node_id) {
                        // c2: own send buffer -> output (same node) — record
                        const uint8_t *src_ptr = c2_src_ptr(peer, target_rank, total_send_slots, slot_bytes - work_remain);
                        c2_work[total_send_slots - base_gslot].push_back({src_ptr, optr, work_bytes, 0, 0, 0, false, true});
                    } else {
                        // c3: own recv buffer -> output (cross node) — record
                        const uint8_t *src_ptr = ml.my_recv_ptr<uint8_t>(node, peer, total_send_slots) + slot_bytes - work_remain;
                        c3_work[total_send_slots - base_gslot].push_back({src_ptr, optr, work_bytes, 0, 0, 0, false, true});
                    }
                }
            }
        }
    };

    // Replay a recorded copy work item onto copy_stream (used for c2 and c3).
    auto issue_copy_work = [&](const CopyWork& w) {
        if (w.is_frag) {
            CUDACHECK(cudaMemcpyAsync(w.dst, w.src, w.bytes, cudaMemcpyDeviceToDevice, copy_stream));
        } else if (w.src_piecewise) {
            CUDACHECK(cudaMemcpyAsync(w.dst, w.src, w.blocks * w.common_bytes, cudaMemcpyDeviceToDevice, copy_stream));
        } else {
            CUDACHECK(cudaMemcpy2DAsync(w.dst, w.dst_stride, w.src, w.common_bytes, w.common_bytes, w.blocks, cudaMemcpyDeviceToDevice, copy_stream));
        }
    };

    // c3 lag (multinode only). K = min(send_times, R): c3 of slot M is issued
    // K-1 slots after c1/c2 of slot M. This is load-bearing for liveness — see
    // the deadlock analysis in the file header. Single-node has no c3.
    const int K = is_single_node ? 0 : std::min(send_times, R);

    // Main slot loop - process all tensors in flat manner.
    // Phased per slot (mirrors ag.cu execute_ag_core): c1 then c2 then (lagged)
    // c3. On a single in-order copy_stream, pre_c2(N) MUST follow post_c1(N)
    // — it waits on c2_ready=c1_done(N) which post_c1 writes; queuing pre_c2
    // before post_c1 deadlocks symmetrically. So c1 is issued during the work
    // iteration, c2 is replayed from c2_work after post_c1+pre_c2, and c3 is
    // replayed from c3_work with a K-slot lag.
    for (int st = 0; st < send_times; ++st) {
        work_remain = slot_bytes;

        // --- c1 phase: pre_c1, c1 memcpys (records c2+c3), post_c1 ---
        pre_c1(total_send_slots);

        while (work_remain > 0 && tidx < num_tensors) {
            // 1. deal with fragment
            const size_t bytes_in_block = src_z % common_bytes; // src_z % common_bytes == dst_z % common_bytes
            if (bytes_in_block != 0) {
                const size_t work_bytes = std::min(work_remain, common_bytes - bytes_in_block);
                copy_frag(work_bytes);
                tensor_proceed(work_bytes);
            }
            if (work_remain == 0) break;
            // 2. deal with incomplete massive
            const size_t block_idx = (src_piecewise ? dst_z : src_z) / common_bytes;
            if (block_idx != 0) {
                // incomplete massive, 2dcopy <= most_blocks
                const size_t proceed_blocks = std::min(work_remain / common_bytes, most_blocks - block_idx);
                if (proceed_blocks > 0) {
                    copy2d(work_remain, proceed_blocks);
                    tensor_proceed(proceed_blocks * common_bytes);
                }
            }
            if (work_remain == 0) break;
            // 3. deal with massives
            const size_t massive_idx = src_piecewise ? dst_x : src_x;
            const size_t proceed_massives = std::min(work_remain / common_bytes / most_blocks, most_massive - massive_idx);
            for (int m = 0; m < proceed_massives; ++m) {
                copy2d(work_remain, most_blocks);
                tensor_proceed(most_blocks * common_bytes);
            }
            if (src_x == src_X) {
                // this tensor is done
                tidx += 1;  // move to next tensor first
                while (tidx < num_tensors && !update_tensor_meta()) {
                    tidx += 1;
                }
                continue;
            }
            // at lease a massive remain
            // 4. not enough for massive, but some blocks
            const size_t proceed_blocks = std::min(work_remain / common_bytes, most_blocks);
            if (proceed_blocks > 0) {
                copy2d(work_remain, proceed_blocks);
                tensor_proceed(proceed_blocks * common_bytes);
            }
            // 5. not enough for a block, but some remaining bytes
            const size_t work_bytes = std::min(work_remain, common_bytes);
            if (work_bytes > 0) {
                copy_frag(work_bytes);
                tensor_proceed(work_bytes);
            }
        }

        post_c1(total_send_slots);
        if (debug && local_rank == 0) printf("[rank %d]: slot %d post_c1 done\n", rank, st);

        // --- c2 phase: pre_c2 (waits on c1_done, now satisfied), c2 replay, post_c2 ---
        pre_c2(total_send_slots);
        for (const auto& w : c2_work[st]) {
            issue_copy_work(w);
        }
        post_c2(total_send_slots);
        if (debug && local_rank == 0) printf("[rank %d]: slot %d post_c2 done\n", rank, st);

        // --- c3 phase (lagged, multinode only): replay c3 for slot st-(K-1) ---
        // Guard K > 0: when K==0 (single-node) there is no c3, and `st >= K-1`
        // would be `st >= -1` (always true), computing c3_st = st+1 → OOB on
        // c3_work.
        if (K > 0 && st >= K - 1) {
            const int c3_st = st - (K - 1);
            const size_t c3_gslot = base_gslot + (size_t)c3_st;
            pre_c3(c3_gslot);
            for (const auto& w : c3_work[c3_st]) {
                issue_copy_work(w);
            }
            post_c3(c3_gslot);
        }

        total_send_slots += 1;
        if (debug && local_rank == 0) printf("[rank %d]: slot %d complete\n", rank, st);
    }
    // Tail: drain the remaining K-1 c3 slots (their c1/c2 already issued in
    // the main loop; only c3 lag remained). No-op when K<=1.
    for (int i = 0; i < K - 1; ++i) {
        const int c3_st = send_times - (K - 1) + i;
        const size_t c3_gslot = base_gslot + (size_t)c3_st;
        pre_c3(c3_gslot);
        for (const auto& w : c3_work[c3_st]) {
            issue_copy_work(w);
        }
        post_c3(c3_gslot);
    }
    // deal with local to local at last
    for (int t = 0; t < num_tensors; ++t) {
        const auto src_Z = layouts[t].get_Z(), dst_Z = rlayouts[t].get_Z();
        const auto copy_block_size = std::min(src_Z, dst_Z);
        HOST_ASSERT((src_Z == 0) == (dst_Z == 0));
        if (src_Z == 0) continue;
        const auto blocks = std::max(src_Z, dst_Z) / copy_block_size;
        auto src_ptr = layouts[t].get_base_ptr() + rank * src_Z;
        auto dst_ptr = rlayouts[t].get_base_ptr() + rank * dst_Z;
        auto src_stride = src_Z == copy_block_size ? (copy_block_size * num_ranks) : copy_block_size;
        const auto dst_stride = dst_Z == copy_block_size ? (copy_block_size * num_ranks) : copy_block_size;
        // Strided-S input: the S-rows are pitched src_row_strides[t]; for s0g1
        // the rank's chunk starts at row `rank * (src_Z / copy_block_size)`.
        const auto row_stride = src_row_strides[t];
        if (row_stride != src_stride) {
            src_stride = row_stride;
            if (src_Z != copy_block_size) {  // s0g1: chunk is rows pitched
                src_ptr = layouts[t].get_base_ptr() + rank * (src_Z / copy_block_size) * row_stride;
            }
        }
        CUDACHECK(cudaMemcpy2DAsync(dst_ptr, dst_stride, src_ptr, src_stride, copy_block_size, blocks, cudaMemcpyDeviceToDevice, copy_stream));
    }
    if (debug && local_rank == 0) printf("[rank %d]: scattergatherfunc complete, total_send_slots=%zu\n", rank, total_send_slots);
}

/**
 * Host-side launcher for the time-division multiplexed scatter-gather kernel
 * This function provides an alternative implementation that is functionally equivalent
 * to scattergatherfunc but uses a single kernel with three time-multiplexed lambdas.
 *
 * @param args_dev      Pre-allocated device memory for tensor metadata (interleaved in/out)
 *                      Format: [in_ptr, in_X, in_Y, in_Z, out_ptr, out_X, out_Y, out_Z] * num_tensors
 * @param num_tensors   Number of tensors
 * @param gin_win       GIN window handle
 * @param gin_buffer    GIN buffer pointer
 * @param signal_buffer Signal buffer containing p2p pointers (at offset 0)
 * @param unroll        Unroll factor
 * @param nvl_ring      NVL ring buffer size
 * @param rdma_ring     RDMA ring buffer size
 * @param total_send_slots Current slot counter (updated on return)
 * @param capture_round_n Round number for debugging
 * @param rank          Current rank
 * @param num_local_ranks Number of local ranks per node
 * @param num_ranks     Total number of ranks
 * @param total_chunk_bytes Total bytes per rank across all tensors
 * @param dev_comm      NCCL device communicator
 * @param num_sms       Number of SMs to use
 * @param stream        Main CUDA stream
 */
void scattergather_kernel_func(
    uint64_t *args_host,
    uint64_t *args_dev,
    int num_tensors,
    ncclWindow_t gin_win,
    void *gin_buffer,
    void *signal_buffer,
    uint64_t *debug_buf,
    int unroll,
    int nvl_ring,
    int rdma_ring,
    int capture_round_n,
    int rank,
    int num_local_ranks,
    int num_ranks,
    ncclDevComm dev_comm,
    int num_sms,
    cudaStream_t stream,
    const int scatter_dim
) {
    if (num_tensors == 0) return;

    const int is_single_node = (num_local_ranks == num_ranks) ? 1 : 0;

    // p2p_ptrs is at the beginning of signal_buffer
    void **p2p_ptrs_dev = reinterpret_cast<void**>(signal_buffer);
    
    
    // Launch kernel
    const size_t rdma_unroll = static_cast<size_t>(unroll);
    int flat_mode = 3;
    bool f4_aligned = true;
    int f8_aligned = 1;
    for (int i = 0; i < num_tensors; ++i) {
        const int this_tensor_flat = (args_host[8 * i + 1] == 1) | ((args_host[8 * i + 5] == 1) << 1);
        flat_mode &= this_tensor_flat;
        f4_aligned &= (args_host[8 * i + 0] % 16 == 0); // src_ptr
        f4_aligned &= (args_host[8 * i + 3] % 16 == 0); // src chunk (in-Z)
        f4_aligned &= (args_host[8 * i + 4] % 16 == 0); // dst_ptr
        f4_aligned &= (args_host[8 * i + 7] % 16 == 0); // dst chunk (out-Z)
        // in-Y now carries the S-row byte stride; every row start must be
        // 16B (int4) aligned so the strided row walk stays float4-safe.
        f4_aligned &= (args_host[8 * i + 2] % 16 == 0);
        f8_aligned &= (args_host[8 * i + 0] % 32 == 0);
        f8_aligned &= (args_host[8 * i + 3] % 32 == 0);
        f8_aligned &= (args_host[8 * i + 4] % 32 == 0);
        f8_aligned &= (args_host[8 * i + 7] % 32 == 0);
        // The float8 (.v8.f32, 32B) load reads two consecutive float4s; on a
        // strided source the second f4 can cross an S-row boundary into the
        // inter-row gap. Fall back to the float4 path whenever any tensor's
        // S-row stride is non-dense.
        const uint64_t row_bytes = (scatter_dim == 0) ? args_host[8 * i + 7]
                                                      : args_host[8 * i + 3] * static_cast<uint64_t>(num_ranks);
        if (args_host[8 * i + 2] != row_bytes) {
            f8_aligned = 0;
        }
    }
    HOST_ASSERT(f4_aligned);

    // ===== Intranode P2P-subgroup kernel dispatch =====
    // Auto-selected by sgcomm.cpp for the single-node num_sms>0 path. Falls
    // through to legacy scattergather_kernel for multinode or unsupported Nlr.
    //
    // Pre-instantiated (Nlr, kWP) pairs — each kNumThreads = kWP*(2*Nlr-1)*32
    // ≤ 1024 (CUDA's per-block thread limit). Adding a new pair requires
    // extending the SWITCH_P2P_PAIR macro and the kernel instantiation list.
    //   (Nlr=2, wp=10) → 960 threads   (1 peer/rank, max NVL bandwidth)
    //   (Nlr=4, wp=4)  → 896 threads
    //   (Nlr=8, wp=2)  → 960 threads   (sweet spot from Nlr=8 sweep)
    const bool p2p_enabled = is_single_node
        && (num_local_ranks == 2 || num_local_ranks == 4 || num_local_ranks == 8);
    int p2p_wp = -1;
    if (p2p_enabled) {
        // kWP per Nlr: pick the largest wp with wp*(2*Nlr-1)*32 ≤ 1024.
        p2p_wp = (num_local_ranks == 8) ? 2 : (num_local_ranks == 4) ? 4 : 10;
    }
    // kUnroll (per-thread copy unroll, sg.cu:1196): pre-instantiated values
    // {2, 4, 8}; fixed at 4. Higher = more ILP but more register pressure.
    int p2p_kunroll = 4;

    // kUnifiedView (sg.cu:1116): host-side decision. When OFF, the kernel
    // iterates tensors in an outer loop with direct addressing (no bound()
    // / while-loop / by-ref mutable state) — each slot only touches one
    // tensor. Profitable when each tensor occupies enough slots per SM to
    // amortize the per-tensor outer-loop setup, OR when there's only one
    // tensor (no need for bound() at all). Criterion:
    //   nargs == 1 OR (per-SM avg slots per tensor) >= 3.
    // per-SM avg slots = (total_f4 / nargs / num_sms) / slot_f4 where
    // slot_f4 = unroll * GIN_CTA_THREADS (matches the kernel's per-SM slot
    // iteration count). Inputs are always F8-aligned in this codebase, so
    // tensor_f4 is even — no f8 safety gating needed.
    bool p2p_unified_view = true;
    if (p2p_enabled) {
        const size_t slot_f4_host = rdma_unroll * GIN_CTA_THREADS;
        size_t total_f4_host = 0;
        for (int i = 0; i < num_tensors; ++i) {
            // in-Y (arg +2) is the S-row byte stride, not num_ranks, so the
            // whole-tensor f4 count is X*Z (the num_ranks factor is implicit).
            const uint64_t X = args_host[8 * i + 1];
            const uint64_t Z = args_host[8 * i + 3];
            total_f4_host += X * Z / sizeof(float4);
        }
        const size_t per_sm_f4 = (num_sms > 0)
            ? (total_f4_host / static_cast<size_t>(num_sms))
            : total_f4_host;
        const size_t avg_slots_per_tensor_per_sm =
            per_sm_f4 / static_cast<size_t>(num_tensors) / slot_f4_host;
        if (num_tensors == 1 || avg_slots_per_tensor_per_sm >= 3) {
            p2p_unified_view = false;
        }
    }

    // kLogZ (sg.cu scattergather_kernel_p2p template arg): host-side decision.
    // When ON, the kernel replaces the per-element integer division
    // (local_byte / cache_Z) and modulo (local_byte - x_coord * cache_Z)
    // with a shift (local_byte >> cache_Z_log2) and AND mask
    // (local_byte & cache_Z_mask) — valid only when every tensor's Z (src
    // AND dst, in bytes) is a power of 2. Auto-selected when all Z pass the
    // power-of-2 check; no env var. Z is in bytes (= D * esize); for
    // D=128 esize=2 → Z=256=2^8 ✓; D=64 esize=4 → Z=256 ✓; D=48 esize=4 →
    // Z=192 ✗. No effect on the kFlat path (no division there anyway).
    bool p2p_log_z = false;
    if (p2p_enabled) {
        bool all_z_pow2 = true;
        for (int i = 0; i < num_tensors; ++i) {
            const uint64_t src_Z = args_host[8 * i + 3];
            const uint64_t dst_Z = args_host[8 * i + 7];
            if (src_Z == 0 || (src_Z & (src_Z - 1)) != 0) { all_z_pow2 = false; break; }
            if (dst_Z == 0 || (dst_Z & (dst_Z - 1)) != 0) { all_z_pow2 = false; break; }
        }
        p2p_log_z = all_z_pow2;
    }

#define LAUNCH_SG_P2P_KERNEL(flat_mode, f8_mode, nlr, wp, kunroll, unified_view, log_z) \
    scattergather_kernel_p2p<nlr, wp, wp * (2 * nlr - 1) * 32, kunroll, f8_mode, flat_mode, unified_view, log_z><<<num_sms, wp * (2 * nlr - 1) * 32, 0, stream>>>( \
        args_dev, static_cast<size_t>(num_tensors), \
        gin_buffer, p2p_ptrs_dev, reinterpret_cast<int*>(signal_buffer), debug_buf, gin_win, \
        nvl_ring, rdma_ring, rdma_unroll, capture_round_n, \
        rank, num_ranks, dev_comm, scatter_dim);

// Combined (Nlr, wp) switch — only instantiates the 3 valid pairs, so
// invalid combinations like (Nlr=8, wp=10) → kNumThreads=4800 (over the
// 1024-thread-per-block limit) are never compiled and don't trigger
// ptxas `maxntid out of range` warnings.
#define SWITCH_P2P_PAIR(flat_mode, f8_mode, kunroll, unified_view, log_z) {\
    switch ((num_local_ranks << 16) | p2p_wp) {\
        case (2 << 16) | 10: LAUNCH_SG_P2P_KERNEL(flat_mode, f8_mode, 2, 10, kunroll, unified_view, log_z); break;\
        case (4 << 16) | 4:  LAUNCH_SG_P2P_KERNEL(flat_mode, f8_mode, 4, 4, kunroll, unified_view, log_z);  break;\
        case (8 << 16) | 2:  LAUNCH_SG_P2P_KERNEL(flat_mode, f8_mode, 8, 2, kunroll, unified_view, log_z);  break;\
        default: HOST_ASSERT(false && "unsupported (Nlr, wp) pair for p2p"); break;\
    }\
}
#define SWITCH_P2P_KUNROLL(flat_mode, f8_mode, unified_view, log_z) {\
    switch (p2p_kunroll) {\
        case 2: SWITCH_P2P_PAIR(flat_mode, f8_mode, 2, unified_view, log_z); break;\
        case 4: SWITCH_P2P_PAIR(flat_mode, f8_mode, 4, unified_view, log_z); break;\
        case 8: SWITCH_P2P_PAIR(flat_mode, f8_mode, 8, unified_view, log_z); break;\
        default: HOST_ASSERT(false && "unsupported kunroll"); break;\
    }\
}
#define P2P_L1(f8_mode, unified_view, log_z) { SWITCH_FLAT(flat_mode, SWITCH_P2P_KUNROLL, f8_mode, unified_view, log_z); }
#define P2P_L2() { \
    if (p2p_unified_view) { \
        if (p2p_log_z) { SWITCH_F8MODE(f8_aligned, P2P_L1, true,  true);  } \
        else            { SWITCH_F8MODE(f8_aligned, P2P_L1, true,  false); } \
    } else { \
        if (p2p_log_z) { SWITCH_F8MODE(f8_aligned, P2P_L1, false, true);  } \
        else            { SWITCH_F8MODE(f8_aligned, P2P_L1, false, false); } \
    } \
}

    // Warp partition of the 32 warps as "c1,cord,c" (merged consumer). Sum
    // must equal 32. In single-node mode the cord path is constexpr-skipped
    // inside the kernel — those warps would idle. The SN sweet spot from the
    // 8× H100 sweep is c1=16, c=16 (256/257 GBps, both directions balanced).
    int c1_warps = 8, cord_warps = 4, c_warps = 20;
    if (is_single_node) {
        c1_warps = 16; cord_warps = 0; c_warps = 16;
    }

#define LAUNCH_SG_KERNEL(flat_mode, single_node, f8_mode) \
    scattergather_kernel<f8_mode, single_node, flat_mode><<<num_sms, GIN_CTA_THREADS, 0, stream>>>( \
        args_dev, static_cast<size_t>(num_tensors), \
        gin_buffer, p2p_ptrs_dev, reinterpret_cast<int*>(signal_buffer), debug_buf, gin_win, \
        nvl_ring, rdma_ring, rdma_unroll, capture_round_n, \
        rank, num_local_ranks, num_ranks, dev_comm, \
        c1_warps, cord_warps, c_warps, scatter_dim);

#define SWITCH_FLAT(p_flat, macro, ...) {\
    switch (p_flat) {\
        case 0: macro(0, __VA_ARGS__);break;\
        case 1: macro(1, __VA_ARGS__);break;\
        case 2: macro(2, __VA_ARGS__);break;\
        case 3: macro(3, __VA_ARGS__);break;\
        default : assert(false && "unsupported flat mode");\
    }\
}
#define SWITCH_SINGLENODE(p_single, macro, ...) {\
    switch (p_single) {\
        case 0: macro(false, __VA_ARGS__);break;\
        case 1: macro(true, __VA_ARGS__);break;\
        default : assert(false && "unsupported single node mode");\
    }\
}
#define SWITCH_F8MODE(p_f8, macro, ...) {\
    switch (p_f8) {\
        case 0: macro(false, ##__VA_ARGS__);break;\
        case 1: macro(true, ##__VA_ARGS__);break;\
        default : assert(false && "unsupported f8 mode");\
    }\
}
#define L1_macro(p_single, p_f8) {\
    SWITCH_FLAT(flat_mode, LAUNCH_SG_KERNEL, p_single, p_f8);\
}
#define L2_macro(p_f8) {\
    SWITCH_SINGLENODE(is_single_node, L1_macro, p_f8);\
}
#define L3_macro() {\
    SWITCH_F8MODE(f8_aligned, L2_macro);\
}
    if (p2p_enabled) {
        P2P_L2();
    } else {
        L3_macro();
    }
    CUDACHECK(cudaGetLastError());
}
}
