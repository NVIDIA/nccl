#include "collective/ag/ag_zero_sm.cuh"
#include "util/error.hpp"
#include "device/comm.cuh"
#include "util/math.hpp"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <nccl_device/coop.h>
#include <nccl_device/core.h>
#include <nccl_device/gin.h>
#include <type_traits>
#include "device/launch.cuh"
#include "device/configs.cuh"
#include <unordered_map>
#include <vector>
#include <mutex>
#include <memory>
#include <cuda.h>
#include <algorithm>

namespace pace {

// ============================================================================
// CUDA Graph Cache for AllGather with BatchMemOp Node Tracking
// ============================================================================

// Graph mode is gated by AGZeroSMConfig.min_graph_nodes (default 32): tiny ops
// (few memcpys) pay the capture+instantiate setup tax without recouping enough
// through cached launches, so they fall back to direct submission. The estimate
// counts every node the captured graph would contain — see
// estimate_ag_zero_sm_graph_nodes() — so the threshold scales with workload (slot
// count, work pieces, world size) rather than just tensor count.

// Slot geometry shared across the cord-path host code (estimate, allgather_zero_sm_func,
// execute_ag_zero_sm_core, update_ag_zero_sm_exec, capture, without_graph). The cord
// kernel itself recomputes slot_bytes inline — device code can't call these.
static size_t ag_zero_sm_slot_bytes(int unroll) {
    return sizeof(int4) * GIN_CTA_THREADS * unroll;
}
// Number of slots needed to carry all num_tensors (each int4-aligned) at slot_bytes each.
static int ag_zero_sm_send_times(uint64_t* args_cpu, int num_tensors, int unroll) {
    size_t total_size = 0;
    for (int i = 0; i < num_tensors; ++i) {
        total_size += align_up(args_cpu[4 * i + 1], sizeof(int4));
    }
    return ceil_div(total_size, ag_zero_sm_slot_bytes(unroll));
}

// Returns the approximate number of graph nodes the captured AG would contain
// for this call. Used to gate graph mode against the per-call setup cost.
//
// Decomposition (matches execute_ag_zero_sm_core's single-graph emit order — c3 of
// slot M is interleaved K-1 slots after c1/c2 of slot M+K-1, with a tail
// drain afterwards; the per-slot counts are unchanged so we sum the same way):
//   - 1 cord kernel             (multi-node only, separate cord graph)
//   - 6 batch_mem_op per slot   (pre/post c1, pre/post c2, pre/post c3)
//                                or 4 per slot single-node (no c3 phase)
//   - per WorkPiece, one memcpy for each of:
//        2 c1 copies       (input -> send buffer; input -> local output)
//      + (num_local_ranks - 1) c2 copies (NVL peer's send buffer -> output)
//      + num_local_ranks * (num_nodes - 1) c3 copies (peer's RDMA recv -> output)
//     i.e. (num_ranks + 1) memcpys per piece, multi-node;
//          (num_local_ranks + 1) for single-node (no c3 fan-out).
//
// `pieces` is the total number of WorkPieces emitted by slot packing — equals
// num_tensors when no tensor straddles a slot boundary, and bumps by one per
// boundary crossing thereafter.
static int estimate_ag_zero_sm_graph_nodes(uint64_t* args_cpu, int num_tensors,
                                   int unroll, int num_local_ranks, int num_ranks,
                                   int send_times) {
    const bool is_single_node = (num_local_ranks == num_ranks);
    const size_t slot_bytes = ag_zero_sm_slot_bytes(unroll);

    // Replay the slot-packing loop from execute_ag_zero_sm_core, but only count pieces.
    int pieces = 0;
    if (num_tensors > 0) {
        int tidx = 0;
        size_t tensor_remain_bytes = args_cpu[1];
        size_t aligned_t_remain = align_up(tensor_remain_bytes, sizeof(int4));
        for (int st = 0; st < send_times && tidx < num_tensors; ++st) {
            size_t work_remain = slot_bytes;
            while (work_remain > 0 && tidx < num_tensors) {
                const size_t this_work = std::min(work_remain, aligned_t_remain);
                const size_t this_bytes = work_remain >= aligned_t_remain
                                            ? tensor_remain_bytes : work_remain;
                pieces += 1;
                work_remain -= this_work;
                aligned_t_remain -= this_work;
                tensor_remain_bytes -= this_bytes;
                if (tensor_remain_bytes == 0) {
                    tidx += 1;
                    if (tidx < num_tensors) {
                        tensor_remain_bytes = args_cpu[tidx * 4 + 1];
                        aligned_t_remain = align_up(tensor_remain_bytes, sizeof(int4));
                    }
                }
            }
        }
    }

    const int batch_per_slot = is_single_node ? 4 : 6;
    const int memcpys_per_piece = is_single_node ? (num_local_ranks + 1)
                                                  : (num_ranks + 1);
    const int cord = is_single_node ? 0 : 1;
    return cord + send_times * batch_per_slot + pieces * memcpys_per_piece;
}

// Cache key: scoped to a comm (signal_buffer) and includes every parameter
// that affects either the captured topology (num_tensors, ranks, ring shape)
// or the addresses baked into nodes. Per-call values that change between
// otherwise-identical calls (base_slot-derived wait/write counters, output
// pointers) are patched in via cuGraphExecUpdate, so they are not in the key.
struct AGZeroSmGraphKey {
    void* signal_buffer;            // comm identity; addresses baked into nodes
    int num_tensors;
    int unroll;
    int nvl_ring;
    int rdma_ring;
    int num_local_ranks;
    int num_ranks;
    bool use_ring;
    std::vector<size_t> tensor_bytes;

    bool operator==(const AGZeroSmGraphKey& o) const {
        return signal_buffer == o.signal_buffer
            && num_tensors == o.num_tensors
            && unroll == o.unroll
            && nvl_ring == o.nvl_ring
            && rdma_ring == o.rdma_ring
            && num_local_ranks == o.num_local_ranks
            && num_ranks == o.num_ranks
            && use_ring == o.use_ring
            && tensor_bytes == o.tensor_bytes;
    }
};

struct AGZeroSmGraphKeyHash {
    static void mix(size_t& h, size_t v) {
        h ^= v + 0x9e3779b97f4a7c15ULL + (h << 12) + (h >> 4);
    }
    size_t operator()(const AGZeroSmGraphKey& k) const {
        size_t h = std::hash<void*>()(k.signal_buffer);
        mix(h, std::hash<int>()(k.num_tensors));
        mix(h, std::hash<int>()(k.unroll));
        mix(h, std::hash<int>()(k.nvl_ring));
        mix(h, std::hash<int>()(k.rdma_ring));
        mix(h, std::hash<int>()(k.num_local_ranks));
        mix(h, std::hash<int>()(k.num_ranks));
        mix(h, std::hash<bool>()(k.use_ring));
        for (size_t b : k.tensor_bytes) mix(h, std::hash<size_t>()(b));
        return h;
    }
};

// Cache value: per-stream captured graphs. The cord kernel that drives the
// RDMA / NVL signalling lives in its own captured graph on the launch stream;
// putting it in a graph (rather than launching it bare each call) gives the
// driver a single entity to schedule against the captured c1/c2/c3 graphs and
// lets us update its kernel params (base_slot, send_times, …) per call via
// cuGraphExecKernelNodeSetParams.
struct AGZeroSmGraphCache {
    // Single captured graph for the slot loop: contains c1/c2/c3 nodes for
    // every slot (c3 of slot M lagged K-1 slots behind c1/c2 of slot M+K-1
    // — see execute_ag_zero_sm_core for the lag math). Single graph means single
    // cuGraphLaunch per call, which sidesteps the cuGraphLaunch host-side
    // back-pressure that the two-graph design hit at high slot counts.
    CUgraph graph = nullptr;
    CUgraphExec exec = nullptr;

    CUgraph cord_graph = nullptr;
    CUgraphExec cord_exec = nullptr;

    // Node handles cached after the first instantiation, in the topological
    // order produced by stream capture (which matches the insertion order
    // of the c1/c2/c3 ops in execute_ag_zero_sm_core). Used by update_ag_zero_sm_exec to
    // patch each node's params on cache hit without recapturing.
    std::vector<CUgraphNode> copy_nodes;
    CUgraphNode cord_node = nullptr;

    ~AGZeroSmGraphCache() {
        if (exec) cuGraphExecDestroy(exec);
        if (graph) cuGraphDestroy(graph);
        if (cord_exec) cuGraphExecDestroy(cord_exec);
        if (cord_graph) cuGraphDestroy(cord_graph);
    }
};

// Global cache map and mutex. shared_ptr (rather than unique_ptr) so that an
// in-flight launch keeps its AGZeroSmGraphCache alive even if clear_ag_zero_sm_graph_cache()
// or an eviction races with it.
static std::unordered_map<AGZeroSmGraphKey, std::shared_ptr<AGZeroSmGraphCache>, AGZeroSmGraphKeyHash> g_ag_zero_sm_graph_cache;
static std::mutex g_ag_zero_sm_graph_cache_mutex;

// Helper: create cache key from parameters
static AGZeroSmGraphKey make_ag_zero_sm_graph_key(uint64_t* args_cpu, int num_tensors, void* signal_buffer,
                                    int unroll, int nvl_ring, int rdma_ring,
                                    int num_local_ranks, int num_ranks, bool use_ring) {
    AGZeroSmGraphKey key;
    key.signal_buffer = signal_buffer;
    key.num_tensors = num_tensors;
    key.unroll = unroll;
    key.nvl_ring = nvl_ring;
    key.rdma_ring = rdma_ring;
    key.num_local_ranks = num_local_ranks;
    key.num_ranks = num_ranks;
    key.use_ring = use_ring;
    key.tensor_bytes.resize(num_tensors);
    for (int i = 0; i < num_tensors; ++i) {
        key.tensor_bytes[i] = static_cast<size_t>(args_cpu[4 * i + 1]);
    }
    return key;
}

// Forward declarations
static void execute_ag_zero_sm_without_graph(uint64_t* args_cpu, int num_tensors,
    ncclWindow_t gin_win, void *gin_buffer, void *signal_buffer, void **p2p_ptrs, int unroll, int nvl_ring,
    int rdma_ring, size_t& total_send_slots, int capture_round_n, int rank, int num_local_ranks, int num_ranks,
    CUstreamBatchMemOpParams *mparam, ncclDevComm dev_comm, cudaStream_t stream, cudaStream_t copy_stream, bool use_ring);

/**
ag_zero_sm.cu logic overview:
    copy_stream serially executes three phases (c1/c2/c3), within each slot:
      Phase 1 (c1): wait slot ready -> input -> send buffer + local output -> signal c1_done
      Phase 2 (c2): wait all peers' c1_done -> NVL peer send buffers -> output -> signal c2_done
      Phase 3 (c3, multi-node only): wait RDMA data ready -> RDMA recv buffers -> output -> signal c3_done
    stream launches the coordination kernel (multi-node RDMA signaling).

    In the kernel,
    warp 0 handles c1 done + RDMA ack (-rdma_ring) + c2 done (-rdma_ring) -> RDMA sig.
    warp 1 handles RDMA sig -> c3 ready.
    warp 2 handles RDMA ack (-nvl_ring) + c2 done (-nvl_ring) -> c1 ready.
    warp 3 handles c3 done -> RDMA ack.
*/
class mem_layout {
    void **ptrs;
    // Unified ring depth. AGComm equalizes nvl_ring/rdma_ring to their max
    // before we get here, so both inputs match; we still take max defensively
    // in case a direct caller drifts. The whole layout uses this single R.
    const int ring;
    const size_t slot_bytes;
    size_t data_bytes, sig_offset;
    const int rank, num_local_ranks, num_ranks;
    int local_rank, node_id, num_nodes;
public:
    __host__ __device__ mem_layout(void **p2p_ptrs, int nvl_ring, int rdma_ring, size_t slot_bytes, int rank, int num_local_ranks, int num_ranks)
        : ptrs(p2p_ptrs),
          ring(nvl_ring > rdma_ring ? nvl_ring : rdma_ring),
          slot_bytes(slot_bytes), rank(rank), num_ranks(num_ranks), num_local_ranks(num_local_ranks) {
        num_nodes = num_ranks / num_local_ranks;
        local_rank = rank % num_local_ranks;
        node_id = rank / num_local_ranks;
        // Single contiguous data area: ring slots per source-node, num_nodes
        // source-nodes. The slot at source = node_id is *also* the local
        // send area — c1 fills it and remote's gin.put writes into peer's
        // matching slot at source = our node_id, so no separate send region
        // is needed.
        data_bytes = slot_bytes * static_cast<size_t>(ring) * static_cast<size_t>(num_nodes);
        sig_offset = num_local_ranks * sizeof(void*);
    };

    // === Data area (R × num_nodes slots; slot[node_id] = local send area) ===
    // Write pointer for local GPU's c1: writes into its own slot[node_id].
    template <typename T>
    __host__ T* local_rdma_send_ptr(const size_t& slot) {
        return reinterpret_cast<T*>(reinterpret_cast<uint8_t*>(ptrs[local_rank])
            + slot_bytes * (static_cast<size_t>(ring) * node_id + slot % ring));
    };
    // Read pointer for NVL peer's c2: reads peer's slot[node_id], which holds
    // the data this rank's NVL peer wrote in c1.
    template <typename T>
    __host__ T* peer_rdma_send_ptr(const int& nvl_peer, const size_t& slot) {
        HOST_ASSERT(nvl_peer < num_local_ranks && nvl_peer >= 0);
        return reinterpret_cast<T*>(reinterpret_cast<uint8_t*>(ptrs[nvl_peer])
            + slot_bytes * (static_cast<size_t>(ring) * node_id + slot % ring));
    };
    // Device offset of the local rank's send slot, also the destination
    // offset for cord-warp-0 RDMA puts (= remote's slot[our node_id]).
    __device__ uint64_t rdma_send_offset(const size_t& slot) {
        return slot_bytes * (static_cast<size_t>(ring) * node_id + slot % ring);
    };

    // Offset for receiving data from a specific remote source node.
    __device__ uint64_t rdma_recv_offset(const int& src_node, const size_t& slot) {
        return slot_bytes * (static_cast<size_t>(ring) * src_node + slot % ring);
    }
    // Read pointer for c3: reads peer's slot[src_node].
    template <typename T>
    __host__ T* peer_rdma_recv_ptr(const int& src_node, const int& src_peer, const size_t& slot) {
        return reinterpret_cast<T*>(reinterpret_cast<uint8_t*>(ptrs[src_peer])
            + slot_bytes * (static_cast<size_t>(ring) * src_node + slot % ring));
    };

    // === Signals ===
    // Signal: local data ready for RDMA send
    __host__ __device__ uint64_t* signal_base(const int& nvl_peer) {
        return reinterpret_cast<uint64_t*>(reinterpret_cast<uint8_t*>(ptrs[nvl_peer]) + data_bytes + sig_offset);
    }

    // kernel w2 wait c2 & rdma (-nvl_ring) -> write it
    //
    // c1 wait it -> c1 copy in
    __host__ __device__ uint64_t* c1_ready_rw() {
        return signal_base(local_rank);
    }
    // c1 copy in -> c1 write it
    //
    // kernel w0 wait it -> w0 send/g0
    __host__ __device__ uint64_t* c1_done_rw(const int& nvl_peer) {
        return signal_base(nvl_peer) + 1;
    }
    // identical to c1_done_rw
    // c1 done -> c2 ready
    __host__ __device__ uint64_t* c2_ready_r(const int& nvl_peer) {
        return signal_base(nvl_peer) + 1;
    }
    // kernel w1 wait g0 -> kernel w1 write it
    //
    // c3 wait it -> c3 main copy out
    __host__ __device__ uint64_t* c3_ready_rw(const int& nvl_peer) {
        return signal_base(nvl_peer) + 2;
    }
    // c2 shortcut copy out -> c2 write it
    __host__ __device__ uint64_t* c2_done_w(const int& nvl_peer) {
        return signal_base(nvl_peer) + 3 + local_rank;
    }
    // kernel w0 wait it (-rdma_ring) -> w0 send/g0
    __host__ __device__ uint64_t* c2_done_r(const int& nvl_peer) {
        return signal_base(local_rank) + 3 + nvl_peer;
    }
    // c3 nvl copy out -> c3 write it
    __host__ __device__ uint64_t* c3_done_w(const int& nvl_peer) {
        return signal_base(nvl_peer) + 3 + num_local_ranks + local_rank;
    }
    __host__ __device__ uint64_t* c3_done_r(const int& nvl_peer) {
        return signal_base(local_rank) + 3 + num_local_ranks + nvl_peer;
    }
};

// allgather_zero_sm_cord_kernel — 1 block × 4 warp RDMA driver for the 0SM AllGather.
//
// `use_ring` selects the inter-node data shape:
//   false : every source broadcasts its slot to all (N-1) other nodes' rail
//           peers in parallel (the original behaviour).
//   true  : every source sends only to its next rail peer; receivers forward
//           down a per-rail ring until each source's data has reached every
//           rank. The receiver-side per-source-node slot indexing
//           (`rdma_recv_offset(src_node, slot)`) and the broadcast ack pattern
//           are unchanged, so the c1/c2/c3 pipeline carries over byte-for-byte.
//           Relay drain-before-overwrite is protected transitively: source
//           waits all (N-1) acks before reuse, and an ack from R_{k+1} implies
//           R_k drained. See docs/ag_ring_zsm_design.md for the full argument.
__global__ void __launch_bounds__(128, 1) allgather_zero_sm_cord_kernel(ncclWindow_t gin_win, void *signal_buffer, int unroll, int nvl_ring, int rdma_ring, size_t current_slot, size_t work_slots, int round_n, int rank, int num_local_ranks, int num_ranks, ncclDevComm dev_comm, int use_ring) {
    const int warp_id = threadIdx.x >> 5;
    const int lane_id = threadIdx.x & 31;
    ncclGin gin {dev_comm, lane_id & (GIN_QPS - 1)};
    ncclTeam world = ncclTeamWorld(dev_comm);
    const size_t slot_bytes = sizeof(int4) * GIN_CTA_THREADS * unroll;
    mem_layout layout(reinterpret_cast<void**>(signal_buffer), nvl_ring, rdma_ring, slot_bytes, rank, num_local_ranks, num_ranks);
    const int node_id = rank / num_local_ranks;
    const int local_rank = rank % num_local_ranks;
    const int num_nodes = num_ranks / num_local_ranks;
    const int next_node = (node_id + 1) % num_nodes;
    __shared__ int done_warps;
    if (threadIdx.x == 0) {
        done_warps = 0;
    }
    __syncthreads();
    if (warp_id == 0) {
        // warp 0 handles c1 done + RDMA ack (-rdma_ring) + c2 done (-rdma_ring) -> RDMA sig.
        for (size_t slot = 0; slot < work_slots; ++slot) {
            const size_t pending_slot = current_slot + slot + 1;
            // Wait until local data is ready (c1 phase done on copy_stream)
            if (elect_one_sync()) {
                LL_READ_64GE(pending_slot, layout.c1_done_rw(local_rank), rank);
            }
            __syncwarp();
            // Wait until remote RDMA recv slot is free (ack from previous round)
            if (current_slot + slot >= rdma_ring) {
                const size_t wait_slot = current_slot + slot + 1 - rdma_ring;
                if (lane_id < GIN_QPS) {
                    for (int node = 0; node < num_nodes; ++node) {
                        if (node == node_id) continue;
                        // Wait for ACK signal from remote node
                        LOOP_READ_GIN_SIGNAL_64GE(wait_slot, gin.readSignal(num_nodes + node), node * num_local_ranks + local_rank, lane_id);
                    }
                }
                __syncwarp();
                if (lane_id < num_local_ranks) {
                    LL_READ_64GE(wait_slot, layout.c2_done_r(lane_id), node_id * num_local_ranks + lane_id);
                }
            }
            // RDMA send local data. use_ring=false → broadcast to every other
            // node's rail peer. use_ring=true → send to the next-hop rail peer
            // only; relays propagate down the chain in warp 1.
            if (lane_id < GIN_QPS) {
                const size_t per_qp_bytes = slot_bytes / GIN_QPS;
                const size_t qp_offset = lane_id * per_qp_bytes;
                if (use_ring) {
                    gin.put(world, next_node * num_local_ranks + local_rank, gin_win, layout.rdma_recv_offset(node_id, pending_slot - 1) + qp_offset, gin_win, layout.rdma_send_offset(pending_slot - 1) + qp_offset, per_qp_bytes, ncclGin_SignalInc{static_cast<uint32_t>(node_id)});
                } else {
                    for (int node = 0; node < num_nodes; ++node) {
                        if (node == node_id) continue;
                        gin.put(world, node * num_local_ranks + local_rank, gin_win, layout.rdma_recv_offset(node_id, pending_slot - 1) + qp_offset, gin_win, layout.rdma_send_offset(pending_slot - 1) + qp_offset, per_qp_bytes, ncclGin_SignalInc{static_cast<uint32_t>(node_id)});
                    }
                }
            }
            __syncwarp();
        }
        if (elect_one_sync()) {
            atomicAdd_block(&done_warps, 1);
        }
        __syncwarp();
    } else if (warp_id == 1) {
        // warp 1 handles RDMA sig -> c3 ready.
        // Ring mode also forwards each source's data to next_node here, after
        // its arrival signal lands and before bumping c3_ready. The forward
        // re-uses the same per-source-node slot index on the next hop, and
        // re-emits SignalInc{source_node_id} so the receiver-side wait keeps
        // its existing per-source semantics.
        for (size_t slot = 0; slot < work_slots; ++slot) {
            const size_t wait_slot = current_slot + slot + 1;
            const size_t per_qp_bytes = slot_bytes / GIN_QPS;
            const size_t qp_offset = lane_id * per_qp_bytes;
            for (int node = 0; node < num_nodes; ++node) {
                if (node == node_id) continue;
                if (lane_id < GIN_QPS) {
                    // Read GIN signal: data arrived from `node`
                    LOOP_READ_GIN_SIGNAL_64GE(wait_slot, gin.readSignal(node), node * num_local_ranks + local_rank, lane_id);
                }
                __syncwarp();
                if (use_ring && next_node != node) {
                    // Relay: forward this source's slot to next-hop's matching
                    // per-source slot. Skipped on the last hop (next == source).
                    if (lane_id < GIN_QPS) {
                        gin.put(world, next_node * num_local_ranks + local_rank, gin_win, layout.rdma_recv_offset(node, wait_slot - 1) + qp_offset, gin_win, layout.rdma_recv_offset(node, wait_slot - 1) + qp_offset, per_qp_bytes, ncclGin_SignalInc{static_cast<uint32_t>(node)});
                    }
                    __syncwarp();
                }
            }
            // Notify that RDMA recv is done, NVL can start reading
            if (elect_one_sync()) {
                st_release_sys_global(layout.c3_ready_rw(local_rank), wait_slot);
            }
            __syncwarp();
        }
        if (elect_one_sync()) {
            atomicAdd_block(&done_warps, 1);
        }
        __syncwarp();
    } else if (warp_id == 2) {
        // warp 2 handles RDMA ack (-nvl_ring) + c2 done (-nvl_ring) -> c1 ready.
        for (size_t slot = 0; slot < work_slots; ++slot) {
            if (current_slot + slot >= nvl_ring) {
                const size_t wait_slot = current_slot + slot + 1 - nvl_ring;
                // wait for rdma space
                if (lane_id < GIN_QPS) {
                    for (int node = 0; node < num_nodes; ++node) {
                        if (node == node_id) continue;
                        // Wait for ACK signal from remote node
                        LOOP_READ_GIN_SIGNAL_64GE(wait_slot, gin.readSignal(num_nodes + node), node * num_local_ranks + local_rank, lane_id);
                    }
                }
                __syncwarp();
                if (lane_id < num_local_ranks) {
                    LL_READ_64GE(wait_slot, layout.c2_done_r(lane_id), node_id * num_local_ranks + lane_id);
                }
                __syncwarp();
                if (elect_one_sync()) {
                    st_release_sys_global(layout.c1_ready_rw(), wait_slot);
                }
                __syncwarp();
                if (__shfl_sync(0xffffffff, done_warps, 0) >= 3) {
                    break;
                }
            }
        }
        if (elect_one_sync()) {
            atomicAdd_block(&done_warps, 1);
        }
        __syncwarp();
    } else {
        // warp 3 handles c3 done -> RDMA ack.
        for (size_t slot = 0; slot < work_slots; ++slot) {
            const size_t wait_slot = current_slot + slot + 1;
            // Wait until output copy is done (c3 phase finished on copy_stream)
            if (lane_id < num_local_ranks) {
                LL_READ_64GE(wait_slot, layout.c3_done_r(lane_id), node_id * num_local_ranks + lane_id);
            }
            __syncwarp();
            // Send RDMA ack to all remote nodes (their recv buffer can be reused)
            if (lane_id < GIN_QPS) {
                for (int node = 0; node < num_nodes; ++node) {
                    if (node == node_id) continue;
                    gin.signal(world, node * num_local_ranks + local_rank, ncclGin_SignalInc{static_cast<uint32_t>(num_nodes + node_id)});
                }
            }
            __syncwarp();
        }
        if (elect_one_sync()) {
            atomicAdd_block(&done_warps, 1);
        }
        __syncwarp();
    }
}

static AGZeroSmGraphCache* capture_ag_zero_sm_graph(uint64_t* args_cpu, int num_tensors,
    ncclWindow_t gin_win, void *gin_buffer, void *signal_buffer, void **p2p_ptrs, int unroll, int nvl_ring,
    int rdma_ring, size_t& total_send_slots, int capture_round_n, int rank, int num_local_ranks, int num_ranks,
    CUstreamBatchMemOpParams *mparam, ncclDevComm dev_comm, cudaStream_t stream, cudaStream_t copy_stream, bool use_ring);

static CUresult update_ag_zero_sm_exec(uint64_t* args_cpu, int num_tensors,
    void *gin_buffer, void *signal_buffer, void **p2p_ptrs, int unroll, int nvl_ring,
    int rdma_ring, size_t base_slot, int send_times, int capture_round_n,
    int rank, int num_local_ranks, int num_ranks,
    ncclWindow_t gin_win, ncclDevComm dev_comm, bool use_ring,
    AGZeroSmGraphCache* cache, CUcontext ctx,
    CUstreamBatchMemOpParams* mparam);

void allgather_zero_sm_func(uint64_t* args_cpu, int num_tensors, ncclWindow_t gin_win, void *gin_buffer, void *signal_buffer, void **p2p_ptrs, int unroll, int nvl_ring, int rdma_ring, size_t& total_send_slots, int capture_round_n, int rank, int num_local_ranks, int num_ranks, CUstreamBatchMemOpParams *mparam, ncclDevComm dev_comm, cudaStream_t stream, cudaStream_t copy_stream, bool use_ring, bool use_graph, bool force_graph_capture, int min_graph_nodes) {
    const int send_times = ag_zero_sm_send_times(args_cpu, num_tensors, unroll);

    const bool is_single_node = (num_local_ranks == num_ranks);
    const int est_nodes = estimate_ag_zero_sm_graph_nodes(args_cpu, num_tensors, unroll,
                                                 num_local_ranks, num_ranks, send_times);
    const bool can_use_graph = use_graph && (est_nodes >= min_graph_nodes);
    AGZeroSmGraphKey key;
    std::shared_ptr<AGZeroSmGraphCache> active;
    cudaStreamCaptureStatus cs_status;

    if (!can_use_graph) {
        goto default_call;
    }
    // Bail out if any stream we capture or launch on is mid-capture by the
    // caller — we'd either fail BeginCapture or pollute the caller's graph.
    CUDACHECK(cudaStreamIsCapturing(stream, &cs_status));
    if (cs_status != cudaStreamCaptureStatusNone) goto default_call;
    CUDACHECK(cudaStreamIsCapturing(copy_stream, &cs_status));
    if (cs_status != cudaStreamCaptureStatusNone) goto default_call;

    if (force_graph_capture) {
        // Always recapture + reinstantiate; not stored in the cache. The
        // shared_ptr owns the fresh cache and frees it after launch.
        AGZeroSmGraphCache* fresh = capture_ag_zero_sm_graph(args_cpu, num_tensors, gin_win,
            gin_buffer, signal_buffer, p2p_ptrs, unroll, nvl_ring, rdma_ring, total_send_slots,
            capture_round_n, rank, num_local_ranks, num_ranks, mparam, dev_comm, stream,
            copy_stream, use_ring);
        if (!fresh) goto default_call;
        active.reset(fresh);
    } else {
        key = make_ag_zero_sm_graph_key(args_cpu, num_tensors, signal_buffer, unroll, nvl_ring, rdma_ring,
                                num_local_ranks, num_ranks, use_ring);
        {
            std::lock_guard<std::mutex> lock(g_ag_zero_sm_graph_cache_mutex);
            auto it = g_ag_zero_sm_graph_cache.find(key);
            if (it != g_ag_zero_sm_graph_cache.end()) active = it->second;
        }

        if (!active) {
            // Cache miss: capture + instantiate, then publish — or pick up
            // a concurrent insert if one landed first.
            AGZeroSmGraphCache* fresh = capture_ag_zero_sm_graph(args_cpu, num_tensors, gin_win,
                gin_buffer, signal_buffer, p2p_ptrs, unroll, nvl_ring, rdma_ring, total_send_slots,
                capture_round_n, rank, num_local_ranks, num_ranks, mparam, dev_comm, stream,
                copy_stream, use_ring);
            if (!fresh) goto default_call;
            std::shared_ptr<AGZeroSmGraphCache> ours(fresh);
            std::lock_guard<std::mutex> lock(g_ag_zero_sm_graph_cache_mutex);
            auto [it, inserted] = g_ag_zero_sm_graph_cache.try_emplace(key, std::move(ours));
            active = it->second;
        } else {
            // Cache hit: patch per-call params (base_slot, input/output
            // pointers, kernel args) directly onto the cached execs by
            // walking cache->copy_nodes / cache->cord_node and calling
            // cuGraphExec*NodeSetParams. No recapture needed. Set-params is
            // userspace (~0.25-0.5us per call) so this loop is cheap even
            // with hundreds of nodes.
            CUcontext cur_ctx = nullptr;
            CUresult res = cuCtxGetCurrent(&cur_ctx);
            if (res == CUDA_SUCCESS) {
                res = update_ag_zero_sm_exec(args_cpu, num_tensors, gin_buffer, signal_buffer,
                    p2p_ptrs, unroll, nvl_ring, rdma_ring, total_send_slots, send_times,
                    capture_round_n, rank, num_local_ranks, num_ranks, gin_win, dev_comm,
                    use_ring, active.get(), cur_ctx, mparam);
            }
            if (res != CUDA_SUCCESS) {
                // Topology drifted from what we cached. Evict so the next
                // call recaptures + reinstantiates cleanly instead of
                // failing the same update repeatedly.
                std::lock_guard<std::mutex> lock(g_ag_zero_sm_graph_cache_mutex);
                auto it = g_ag_zero_sm_graph_cache.find(key);
                if (it != g_ag_zero_sm_graph_cache.end() && it->second == active) {
                    g_ag_zero_sm_graph_cache.erase(it);
                }
                active.reset();
                goto default_call;
            }
        }
    }

    // Launch the cord kernel before the captured slot graph. exec begins
    // with a c1_ready wait (cord-set after peer c2_done acks) and would
    // otherwise stall its leading nodes on the GPU command queue —
    // historically that produced a host-side cuGraphLaunch back-pressure
    // hang at high slot counts. Launching cord first matches the
    // direct-submit ordering (allgather_zero_sm_cord_kernel is issued before the
    // slot loop in execute_ag_zero_sm_core).
    if (!is_single_node) {
        CUCHECK(cuGraphLaunch(active->cord_exec, stream));
    }
    CUCHECK(cuGraphLaunch(active->exec, copy_stream));
    total_send_slots += send_times;
    return;

default_call:
    execute_ag_zero_sm_without_graph(args_cpu, num_tensors, gin_win, gin_buffer,
        signal_buffer, p2p_ptrs, unroll, nvl_ring, rdma_ring, total_send_slots, capture_round_n,
        rank, num_local_ranks, num_ranks, mparam, dev_comm, stream, copy_stream, use_ring);
}

// ============================================================================
// Graph Capture Functions - Unified capture via fork/join
// ============================================================================

// One unit of per-slot copy work: which tensor, where in it, where in the slot,
// how many bytes, and the input pointer to load from.
struct AgZeroSmWorkPiece {
    int tensor_idx;
    size_t offset_in_tensor;
    size_t slot_offset;
    size_t bytes;
    const uint8_t* input_ptr;
};

// Pack the int4-aligned input stream across `send_times` slots of `slot_bytes`
// each, emitting per-slot AgZeroSmWorkPieces. Shared by execute_ag_zero_sm_core
// (emit) and update_ag_zero_sm_exec (cache-hit patch) so the two paths can't
// drift — the patch path must replay the exact slot/piece layout the captured
// graph was instantiated with (tensor sizes are part of the cache key, so the
// layout is identical across calls; only input/output pointers + base_slot
// differ, and those are patched per-call).
static std::vector<std::vector<AgZeroSmWorkPiece>> ag_zero_sm_pack_slot_work(
    const uint64_t* args_cpu, int num_tensors, size_t slot_bytes, int send_times,
    const std::vector<size_t>& tensor_sizes)
{
    std::vector<std::vector<AgZeroSmWorkPiece>> slot_work(send_times);
    int tidx = 0;
    size_t tensor_remain_bytes = args_cpu[1];
    size_t aligned_t_remain = align_up(tensor_remain_bytes, sizeof(int4));
    const uint8_t* tensor_remain_ptr = reinterpret_cast<const uint8_t*>(args_cpu[0]);
    for (int st = 0; st < send_times; ++st) {
        size_t work_remain = slot_bytes;
        while (work_remain > 0 && tidx < num_tensors) {
            const size_t this_work = std::min(work_remain, aligned_t_remain);
            const size_t this_bytes = work_remain >= aligned_t_remain ? tensor_remain_bytes : work_remain;
            const size_t start_offset = slot_bytes - work_remain;
            const size_t offset_in_tensor = tensor_sizes[tidx] - tensor_remain_bytes;
            slot_work[st].push_back({tidx, offset_in_tensor, start_offset, this_bytes, tensor_remain_ptr});
            work_remain -= this_work;
            aligned_t_remain -= this_work;
            tensor_remain_bytes -= this_bytes;
            tensor_remain_ptr += this_bytes;
            if (tensor_remain_bytes == 0) {
                tidx += 1;
                if (tidx < num_tensors) {
                    tensor_remain_bytes = args_cpu[tidx * 4 + 1];
                    aligned_t_remain = align_up(tensor_remain_bytes, sizeof(int4));
                    tensor_remain_ptr = reinterpret_cast<const uint8_t*>(args_cpu[tidx * 4]);
                }
            }
        }
    }
    return slot_work;
}

// Build a batch of CU_STREAM_MEM_OP_WAIT_VALUE_64 entries (one per i in [0,count)),
// each waiting (GEQ) on addr(i) >= value. Shared by the emit path
// (execute_ag_zero_sm_core: pre_c1/pre_c2/pre_c3) and the cache-hit patch path
// (update_ag_zero_sm_exec: set_c1c2/set_c3) so the wait/write semantics can't
// drift between capturing and patching.
template <typename AddrFn>
static void ag_zero_sm_build_wait_mparam(CUstreamBatchMemOpParams* mparam, int count,
                                        AddrFn addr, uint64_t value) {
    memset(mparam, 0, sizeof(CUstreamBatchMemOpParams) * count);
    for (int i = 0; i < count; ++i) {
        mparam[i].operation = CU_STREAM_MEM_OP_WAIT_VALUE_64;
        mparam[i].waitValue.operation = CU_STREAM_MEM_OP_WAIT_VALUE_64;
        mparam[i].waitValue.address = addr(i);
        mparam[i].waitValue.value64 = value;
        mparam[i].waitValue.flags = CU_STREAM_WAIT_VALUE_GEQ;
    }
}
// Build a batch of CU_STREAM_MEM_OP_WRITE_VALUE_64 entries: addr(i) := value.
template <typename AddrFn>
static void ag_zero_sm_build_write_mparam(CUstreamBatchMemOpParams* mparam, int count,
                                          AddrFn addr, uint64_t value) {
    memset(mparam, 0, sizeof(CUstreamBatchMemOpParams) * count);
    for (int i = 0; i < count; ++i) {
        mparam[i].operation = CU_STREAM_MEM_OP_WRITE_VALUE_64;
        mparam[i].writeValue.operation = CU_STREAM_MEM_OP_WRITE_VALUE_64;
        mparam[i].writeValue.address = addr(i);
        mparam[i].writeValue.value64 = value;
        mparam[i].writeValue.flags = CU_STREAM_WRITE_VALUE_DEFAULT;
    }
}

// ============================================================================
// Unified c1/c2/c3 schedule walker
//
// The copy-stream schedule (per-slot c1/c2/c3 phases, the K-1 slot c3 lag, and
// the tail drain) is walked exactly once, in ag_zero_sm_walk_schedule below. It
// computes every address, value, and op *order*, and dispatches each op to a
// backend that decides only *how* to commit it:
//   - AgZeroSmEmit  : issues the stream ops that stream-capture into graph nodes
//                     (also the direct-submit path when not capturing).
//   - AgZeroSmPatch : re-points the already-captured nodes via
//                     cuGraphExec*NodeSetParams, walking cache->copy_nodes in
//                     the same order the emit path created them.
// Because both backends are driven by the same walk, the capture order and the
// patch order cannot drift — the invariant is structural, not hand-mirrored.
// Each backend issues the same CUDA ops, so the captured graph is byte-identical.
//
// Emit ops CUCHECK/CUDACHECK internally (fatal on error) and return
// CUDA_SUCCESS; patch ops return the driver CUresult so a topology drift evicts
// + falls back cleanly. The walk early-returns on any non-success.
// ============================================================================

// Backend: emit stream ops (capture / direct submit).
struct AgZeroSmEmit {
    cudaStream_t copy_stream;
    CUstreamBatchMemOpParams* mparam;

    template <class AddrFn>
    CUresult wait_batch(int count, AddrFn addr, uint64_t value) {
        ag_zero_sm_build_wait_mparam(mparam, count, addr, value);
        CUCHECK(cuStreamBatchMemOp(copy_stream, count, mparam, 0));
        return CUDA_SUCCESS;
    }
    CUresult wait_single(CUdeviceptr addr, uint64_t value) {
        CUCHECK(cuStreamWaitValue64(copy_stream, addr, value, CU_STREAM_WAIT_VALUE_GEQ));
        return CUDA_SUCCESS;
    }
    template <class AddrFn>
    CUresult write_batch(int count, AddrFn addr, uint64_t value) {
        ag_zero_sm_build_write_mparam(mparam, count, addr, value);
        CUCHECK(cuStreamBatchMemOp(copy_stream, count, mparam, 0));
        return CUDA_SUCCESS;
    }
    CUresult write_single(CUdeviceptr addr, uint64_t value) {
        CUCHECK(cuStreamWriteValue64(copy_stream, addr, value, CU_STREAM_WRITE_VALUE_DEFAULT));
        return CUDA_SUCCESS;
    }
    CUresult copy(void* dst, const void* src, size_t bytes) {
        CUDACHECK(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice, copy_stream));
        return CUDA_SUCCESS;
    }
};

// Backend: patch previously-captured nodes in topological order. wait_single /
// write_single patch the count-1 batch-mem-op node that cuStreamWaitValue64 /
// cuStreamWriteValue64 captured on the emit side.
struct AgZeroSmPatch {
    CUgraphExec exec;
    const std::vector<CUgraphNode>* copy_nodes;
    size_t copy_idx;
    CUcontext ctx;
    CUstreamBatchMemOpParams* mparam;

    CUresult set_batch(int count) {
        if (copy_idx >= copy_nodes->size()) return CUDA_ERROR_INVALID_VALUE;
        CUDA_BATCH_MEM_OP_NODE_PARAMS p = {ctx, static_cast<unsigned>(count), mparam, 0};
        return cuGraphExecBatchMemOpNodeSetParams(exec, (*copy_nodes)[copy_idx++], &p);
    }
    template <class AddrFn>
    CUresult wait_batch(int count, AddrFn addr, uint64_t value) {
        ag_zero_sm_build_wait_mparam(mparam, count, addr, value);
        return set_batch(count);
    }
    CUresult wait_single(CUdeviceptr addr, uint64_t value) {
        ag_zero_sm_build_wait_mparam(mparam, 1, [&](int) { return addr; }, value);
        return set_batch(1);
    }
    template <class AddrFn>
    CUresult write_batch(int count, AddrFn addr, uint64_t value) {
        ag_zero_sm_build_write_mparam(mparam, count, addr, value);
        return set_batch(count);
    }
    CUresult write_single(CUdeviceptr addr, uint64_t value) {
        ag_zero_sm_build_write_mparam(mparam, 1, [&](int) { return addr; }, value);
        return set_batch(1);
    }
    CUresult copy(void* dst, const void* src, size_t bytes) {
        if (copy_idx >= copy_nodes->size()) return CUDA_ERROR_INVALID_VALUE;
        CUDA_MEMCPY3D p;
        memset(&p, 0, sizeof(p));
        p.srcMemoryType = CU_MEMORYTYPE_DEVICE;
        p.srcDevice = reinterpret_cast<CUdeviceptr>(src);
        p.dstMemoryType = CU_MEMORYTYPE_DEVICE;
        p.dstDevice = reinterpret_cast<CUdeviceptr>(dst);
        p.WidthInBytes = bytes;
        p.Height = 1;
        p.Depth = 1;
        return cuGraphExecMemcpyNodeSetParams(exec, (*copy_nodes)[copy_idx++], &p, ctx);
    }
};

// Walk the copy-stream schedule once, dispatching each op to `b`. Owns the
// slot packing, all addresses/values, phase order, c3 lag, and tail drain.
// The cord kernel (launch on emit, kernel-node patch on cache hit) is handled
// by the callers before this runs — it lives on a separate stream/graph and is
// independent of the copy-node walk here.
template <class B>
static CUresult ag_zero_sm_walk_schedule(B& b, uint64_t* args_cpu, int num_tensors,
    int unroll, int nvl_ring, int rdma_ring, size_t base_slot, int send_times,
    int rank, int num_local_ranks, int num_ranks, void** p2p_ptrs) {

    const size_t slot_bytes = ag_zero_sm_slot_bytes(unroll);
    const int local_rank = rank % num_local_ranks;
    const int node_id = rank / num_local_ranks;
    const int num_nodes = num_ranks / num_local_ranks;
    const bool is_single_node = (num_local_ranks == num_ranks);
    mem_layout layout(p2p_ptrs, nvl_ring, rdma_ring, slot_bytes, rank, num_local_ranks, num_ranks);

    std::vector<size_t> tensor_sizes(num_tensors);
    std::vector<void*> tensor_output_ptrs(num_tensors);
    for (int i = 0; i < num_tensors; ++i) {
        tensor_sizes[i] = args_cpu[i * 4 + 1];
        tensor_output_ptrs[i] = reinterpret_cast<void*>(args_cpu[i * 4 + 2]);
    }
    auto slot_work = ag_zero_sm_pack_slot_work(args_cpu, num_tensors, slot_bytes, send_times, tensor_sizes);
    auto out_ptr = [&](int src_rank, int tensor_idx, size_t offset_in_tensor) {
        return reinterpret_cast<uint8_t*>(tensor_output_ptrs[tensor_idx])
             + src_rank * tensor_sizes[tensor_idx] + offset_in_tensor;
    };
    auto D = [](void* p) { return reinterpret_cast<CUdeviceptr>(p); };

    auto do_c1c2 = [&](int st, size_t gslot) -> CUresult {
        CUresult r;
        // Phase 1 (c1): input -> local send buffer + local output.
        const uint64_t wait_value = (gslot >= static_cast<size_t>(nvl_ring)) ? (gslot + 1 - nvl_ring) : 0;
        if (is_single_node) {
            r = b.wait_batch(num_local_ranks, [&](int i) { return D(layout.c2_done_r(i)); }, wait_value);
        } else {
            r = b.wait_single(D(layout.c1_ready_rw()), wait_value);
        }
        if (r != CUDA_SUCCESS) return r;
        for (const auto& w : slot_work[st]) {
            r = b.copy(layout.local_rdma_send_ptr<uint8_t>(gslot) + w.slot_offset, w.input_ptr, w.bytes);
            if (r != CUDA_SUCCESS) return r;
            r = b.copy(out_ptr(rank, w.tensor_idx, w.offset_in_tensor), w.input_ptr, w.bytes);
            if (r != CUDA_SUCCESS) return r;
        }
        r = b.write_single(D(layout.c1_done_rw(local_rank)), gslot + 1);
        if (r != CUDA_SUCCESS) return r;

        // Phase 2 (c2): NVL peers' send buffers -> output (intra-node).
        r = b.wait_batch(num_local_ranks, [&](int i) { return D(layout.c2_ready_r(i)); }, gslot + 1);
        if (r != CUDA_SUCCESS) return r;
        for (const auto& w : slot_work[st]) {
            for (int peer_iter = 0; peer_iter < num_local_ranks; ++peer_iter) {
                int peer = (peer_iter + local_rank) % num_local_ranks;
                if (peer == local_rank) continue;
                int target_rank = node_id * num_local_ranks + peer;
                r = b.copy(out_ptr(target_rank, w.tensor_idx, w.offset_in_tensor),
                           layout.peer_rdma_send_ptr<uint8_t>(peer, gslot) + w.slot_offset, w.bytes);
                if (r != CUDA_SUCCESS) return r;
            }
        }
        return b.write_batch(num_local_ranks, [&](int i) { return D(layout.c2_done_w(i)); }, gslot + 1);
    };

    auto do_c3 = [&](int st, size_t gslot) -> CUresult {
        // Phase 3 (c3): RDMA recv buffers -> output (inter-node).
        CUresult r = b.wait_batch(num_local_ranks, [&](int i) { return D(layout.c3_ready_rw(i)); }, gslot + 1);
        if (r != CUDA_SUCCESS) return r;
        for (const auto& w : slot_work[st]) {
            for (int peer_iter = 0; peer_iter < num_local_ranks; ++peer_iter) {
                int peer = (peer_iter + local_rank) % num_local_ranks;
                for (int node = 0; node < num_nodes; ++node) {
                    if (node == node_id) continue;
                    int target_rank = node * num_local_ranks + peer;
                    r = b.copy(out_ptr(target_rank, w.tensor_idx, w.offset_in_tensor),
                               layout.peer_rdma_recv_ptr<uint8_t>(node, peer, gslot) + w.slot_offset, w.bytes);
                    if (r != CUDA_SUCCESS) return r;
                }
            }
        }
        return b.write_batch(num_local_ranks, [&](int i) { return D(layout.c3_done_w(i)); }, gslot + 1);
    };

    // c3 for slot M is issued K-1 iterations after c1/c2 of slot M (K capped at
    // nvl_ring for liveness — see the K=nvl_ring argument on execute_ag_zero_sm_core).
    const int c3_lag = is_single_node ? 0 : std::min(send_times, nvl_ring) - 1;
    for (int st = 0; st < send_times; ++st) {
        CUresult r = do_c1c2(st, base_slot + st);
        if (r != CUDA_SUCCESS) return r;
        if (!is_single_node && st >= c3_lag) {
            int c3_st = st - c3_lag;
            r = do_c3(c3_st, base_slot + c3_st);
            if (r != CUDA_SUCCESS) return r;
        }
    }
    if (!is_single_node) {
        for (int c3_st = send_times - c3_lag; c3_st < send_times; ++c3_st) {
            CUresult r = do_c3(c3_st, base_slot + c3_st);
            if (r != CUDA_SUCCESS) return r;
        }
    }
    return CUDA_SUCCESS;
}

// Core AG execution logic - shared by both capture and non-capture paths.
//
// Single-stream design: c1, c2, and c3 of each slot are all issued onto
// copy_stream in order. To hide RDMA latency, c3 of slot M is interleaved
// K-1 slots after c1/c2 of slot M, where K = min(send_times, nvl_ring).
// Concretely, for each iteration N in the main loop:
//   - issue pre_c1(N), c1(N), post_c1(N), pre_c2(N), c2(N), post_c2(N)
//   - if N >= K-1: issue pre_c3(N-K+1), c3(N-K+1), post_c3(N-K+1)
// After the main loop, a tail loop drains the remaining K-1 c3 slots.
//
// Correctness: pre_c3(M) still GEQ-waits on c3_ready(M) (cord-set after
// RDMA arrival), so the lag is an opportunistic optimization that lets
// the GPU find that wait already satisfied.
//
// The K = nvl_ring cap is load-bearing for liveness, not just throughput.
// pre_c1(N) on copy_stream waits for c1_ready_rw() >= N+1-nvl_ring, which
// the cord kernel sets only after receiving an RDMA ack from the remote,
// which the remote sends only after its post_c3 has run. So copy_stream
// stalls at iter N=nvl_ring until *some* remote rank's c3 phase
// completes. If we lag c3 by more than nvl_ring-1, c3(0) is queued on
// copy_stream behind pre_c1(nvl_ring) — c3(0) never executes, the remote
// never acks, all ranks deadlock symmetrically. K <= nvl_ring guarantees
// c3(0) is queued before that stall point. (K capped at rdma_ring would
// only protect recv-buffer reuse, which isn't the binding constraint
// here since the cord kernel already gates puts on receiver acks.)
//
// Output pointers are read from args_cpu[4*i+2].
static void execute_ag_zero_sm_core(uint64_t* args_cpu, int num_tensors,
    ncclWindow_t gin_win, void *gin_buffer, void *signal_buffer, void **p2p_ptrs, int unroll, int nvl_ring,
    int rdma_ring, size_t base_slot, int send_times, int capture_round_n, int rank, int num_local_ranks, int num_ranks,
    CUstreamBatchMemOpParams *mparam, ncclDevComm dev_comm, cudaStream_t stream, cudaStream_t copy_stream, bool launch_cord_kernel, bool use_ring) {
    
    const bool is_single_node = (num_local_ranks == num_ranks);

    // Cord kernel (multi-node) drives RDMA signalling on `stream`, ahead of the
    // copy schedule. In the capture path launch_cord_kernel is false — the cord
    // kernel is captured into its own graph instead of launched here.
    if (launch_cord_kernel && !is_single_node) {
        allgather_zero_sm_cord_kernel<<<1, 128, 0, stream>>>(gin_win, signal_buffer, unroll, nvl_ring, rdma_ring, base_slot, send_times, capture_round_n, rank, num_local_ranks, num_ranks, dev_comm, use_ring ? 1 : 0);
        CUDACHECK(cudaGetLastError());
    }

    // Emit the c1/c2/c3 copy schedule onto copy_stream (captured, or submitted
    // directly in the non-graph path). Emit ops are fatal-checked, so the walk
    // cannot return an error here.
    AgZeroSmEmit backend{copy_stream, mparam};
    (void)ag_zero_sm_walk_schedule(backend, args_cpu, num_tensors, unroll, nvl_ring, rdma_ring,
        base_slot, send_times, rank, num_local_ranks, num_ranks, p2p_ptrs);
}

// Cache-hit fast path: patch per-call params onto a previously-instantiated
// AG exec, without recapturing. Mirrors execute_ag_zero_sm_core's structural
// iteration (single stream, c3 lagged by K-1 slots, plus tail drain) so
// cache->copy_nodes and cache->cord_node — captured at instantiation in
// topological/insertion order — line up with the operations generated
// here. Each op site dispatches to cuGraphExec*NodeSetParams.
static CUresult update_ag_zero_sm_exec(uint64_t* args_cpu, int num_tensors,
    void *gin_buffer, void *signal_buffer, void **p2p_ptrs, int unroll, int nvl_ring,
    int rdma_ring, size_t base_slot, int send_times, int capture_round_n,
    int rank, int num_local_ranks, int num_ranks,
    ncclWindow_t gin_win, ncclDevComm dev_comm, bool use_ring,
    AGZeroSmGraphCache* cache, CUcontext ctx,
    CUstreamBatchMemOpParams* mparam) {

    const bool is_single_node = (num_local_ranks == num_ranks);

    // ---- Cord kernel: rebuild kernelParams[] with new per-call values ----
    if (!is_single_node && cache->cord_exec) {
        CUDA_KERNEL_NODE_PARAMS kp;
        CUresult r = cuGraphKernelNodeGetParams(cache->cord_node, &kp);
        if (r != CUDA_SUCCESS) return r;
        // Stack-resident args, kept alive until SetParams returns (the API
        // copies the pointed-to values, so post-return lifetime doesn't
        // matter, but they MUST be valid during the call).
        ncclWindow_t l_gin_win = gin_win;
        void* l_signal_buffer = signal_buffer;
        int l_unroll = unroll;
        int l_nvl_ring = nvl_ring;
        int l_rdma_ring = rdma_ring;
        size_t l_current_slot = base_slot;
        size_t l_work_slots = static_cast<size_t>(send_times);
        int l_round_n = capture_round_n;
        int l_rank = rank;
        int l_num_local_ranks = num_local_ranks;
        int l_num_ranks = num_ranks;
        ncclDevComm l_dev_comm = dev_comm;
        int l_use_ring = use_ring ? 1 : 0;
        void* args[] = {
            &l_gin_win, &l_signal_buffer, &l_unroll, &l_nvl_ring, &l_rdma_ring,
            &l_current_slot, &l_work_slots, &l_round_n, &l_rank,
            &l_num_local_ranks, &l_num_ranks, &l_dev_comm, &l_use_ring,
        };
        kp.kernelParams = args;
        r = cuGraphExecKernelNodeSetParams(cache->cord_exec, cache->cord_node, &kp);
        if (r != CUDA_SUCCESS) return r;
    }

    // Re-point the captured copy nodes by walking the same c1/c2/c3 schedule the
    // emit path captured, in topological order. AgZeroSmPatch consumes exactly
    // one cache->copy_nodes entry per op via copy_idx, so a full walk must land
    // on copy_nodes.size() — any mismatch means the cached topology drifted.
    AgZeroSmPatch backend{cache->exec, &cache->copy_nodes, 0, ctx, mparam};
    CUresult r = ag_zero_sm_walk_schedule(backend, args_cpu, num_tensors, unroll, nvl_ring,
        rdma_ring, base_slot, send_times, rank, num_local_ranks, num_ranks, p2p_ptrs);
    if (r != CUDA_SUCCESS) return r;
    if (backend.copy_idx != cache->copy_nodes.size()) return CUDA_ERROR_INVALID_VALUE;

    return CUDA_SUCCESS;
}

// Capture each stream's work into its own graph. `stream` (the launch stream)
// hosts the cord kernel; capturing it gives the driver one schedulable entity
// per stream and lets us patch its per-call args (base_slot, send_times,
// capture_round_n) via cuGraphExecKernelNodeSetParams instead of re-launching
// with `<<<>>>` each call.
//
// Order matters: agcomm.cpp's `stream_wait(c1_stream, launch_stream)` runs
// before allgather_zero_sm_func, leaving copy_stream with a pending wait on a launch
// _stream-recorded event. Once we BeginCapture on copy_stream, that link
// pulls launch_stream into the capture, and a subsequent BeginCapture on
// launch_stream errors with "stream is capturing". So we must capture the
// cord kernel on launch_stream *first*.
static AGZeroSmGraphCache* capture_ag_zero_sm_graph(uint64_t* args_cpu, int num_tensors,
    ncclWindow_t gin_win, void *gin_buffer, void *signal_buffer, void **p2p_ptrs, int unroll, int nvl_ring,
    int rdma_ring, size_t& total_send_slots, int capture_round_n, int rank, int num_local_ranks, int num_ranks,
    CUstreamBatchMemOpParams *mparam, ncclDevComm dev_comm, cudaStream_t stream, cudaStream_t copy_stream, bool use_ring) {

    auto cache = std::make_unique<AGZeroSmGraphCache>();
    const bool is_single_node = (num_local_ranks == num_ranks);

    const int send_times = ag_zero_sm_send_times(args_cpu, num_tensors, unroll);

    // Capture cord kernel into its own graph using a dedicated, never-shared
    // stream. We can't use launch_stream because PyTorch / NCCL's caching
    // allocator and prior cross-stream waits keep it tied up in some other
    // implicit capture context. The stream is created once per-thread and
    // reused — capture mode is exited at EndCapture, so the stream is safe
    // to reuse between calls.
    if (!is_single_node) {
        static thread_local cudaStream_t cap_stream = nullptr;
        if (!cap_stream) {
            CUDACHECK(cudaStreamCreateWithFlags(&cap_stream, cudaStreamNonBlocking));
        }
        CUDACHECK(cudaGraphCreate(&cache->cord_graph, 0));
        CUDACHECK(cudaStreamBeginCapture(cap_stream, cudaStreamCaptureModeRelaxed));
        allgather_zero_sm_cord_kernel<<<1, 128, 0, cap_stream>>>(
            gin_win, signal_buffer, unroll, nvl_ring, rdma_ring,
            total_send_slots, send_times, capture_round_n, rank,
            num_local_ranks, num_ranks, dev_comm, use_ring ? 1 : 0);
        CUDACHECK(cudaGetLastError());
        CUDACHECK(cudaStreamEndCapture(cap_stream, &cache->cord_graph));
    }

    CUDACHECK(cudaGraphCreate(&cache->graph, 0));
    CUDACHECK(cudaStreamBeginCapture(copy_stream, cudaStreamCaptureModeGlobal));

    // Call execute_ag_zero_sm_core directly (rather than execute_ag_zero_sm_without_graph) so
    // we can pass launch_cord_kernel=false: the cord kernel is already in
    // cord_graph, so we don't want execute_ag_zero_sm_core to re-launch it.
    execute_ag_zero_sm_core(args_cpu, num_tensors, gin_win, gin_buffer, signal_buffer,
        p2p_ptrs, unroll, nvl_ring, rdma_ring, total_send_slots, send_times, capture_round_n,
        rank, num_local_ranks, num_ranks, mparam, dev_comm, stream, copy_stream,
        false /* launch_cord_kernel */, use_ring);

    CUDACHECK(cudaStreamEndCapture(copy_stream, &cache->graph));
    if (cache->graph) {
        CUDACHECK(cudaGraphInstantiate(&cache->exec, cache->graph, nullptr, nullptr, 0));
        if (!is_single_node) {
            CUDACHECK(cudaGraphInstantiate(&cache->cord_exec, cache->cord_graph, nullptr, nullptr, 0));
        }

        // Cache node handles in topological order so update_ag_zero_sm_exec can
        // patch each node directly on cache hit without recapturing.
        size_t n = 0;
        CUCHECK(cuGraphGetNodes(cache->graph, nullptr, &n));
        cache->copy_nodes.resize(n);
        CUCHECK(cuGraphGetNodes(cache->graph, cache->copy_nodes.data(), &n));
        if (!is_single_node) {
            size_t cn = 0;
            CUCHECK(cuGraphGetNodes(cache->cord_graph, nullptr, &cn));
            if (cn != 1) return nullptr;  // expect exactly one kernel node
            CUCHECK(cuGraphGetNodes(cache->cord_graph, &cache->cord_node, &cn));
        }
        return cache.release();
    }
    return nullptr;
}

// Clear all cached graphs
void clear_ag_zero_sm_graph_cache() {
    std::lock_guard<std::mutex> lock(g_ag_zero_sm_graph_cache_mutex);
    g_ag_zero_sm_graph_cache.clear();
}

// Implementation without graph
static void execute_ag_zero_sm_without_graph(uint64_t* args_cpu, int num_tensors,
    ncclWindow_t gin_win, void *gin_buffer, void *signal_buffer, void **p2p_ptrs, int unroll, int nvl_ring,
    int rdma_ring, size_t& total_send_slots, int capture_round_n, int rank, int num_local_ranks, int num_ranks,
    CUstreamBatchMemOpParams *mparam, ncclDevComm dev_comm, cudaStream_t stream, cudaStream_t copy_stream, bool use_ring) {

    const int send_times = ag_zero_sm_send_times(args_cpu, num_tensors, unroll);

    execute_ag_zero_sm_core(args_cpu, num_tensors, gin_win, gin_buffer, signal_buffer,
        p2p_ptrs, unroll, nvl_ring, rdma_ring, total_send_slots, send_times, capture_round_n,
        rank, num_local_ranks, num_ranks, mparam, dev_comm, stream, copy_stream,
        true /* launch cord kernel */, use_ring);
    total_send_slots += send_times;
}

}