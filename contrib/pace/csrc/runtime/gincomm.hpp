#ifndef __GIN_COMM_HPP_H__
#define __GIN_COMM_HPP_H__

#include <optional>
#include <mutex>
#include <vector>
#include <functional>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <torch/types.h>
#include "nccl.h"
#include "nccl_device.h"
#include "services/event.hpp"

namespace pace {

#define SIGNAL_BYTES (sizeof(int) * 1048576)
constexpr int kMaxArgs = 256;

// ---------------------------------------------------------------------------
// Comm is composed of four single-concern components (docs/architecture.md
// "L3a"). The collective engines (RS/SG/AG/EP) HAVE-A Comm — no shared base
// class. Members are grouped so that call sites read, e.g., comm.topo.rank,
// comm.buffer.data_buffer, comm.transport.dev_comm, comm.launch.kstream.
// ---------------------------------------------------------------------------

// Where this rank sits in the group. Leaf value type; no owned resources.
struct Topology {
    int rank = 0, num_ranks = 0, num_local_ranks = 0;
    // nvl_local_ranks: effective NVL fabric scope (the set of ranks treated as
    // "intranode" for buffer sizing + p2p_ptrs indexing). NCCL is the source of
    // truth for the natural scope (queried via ncclCommQueryProperties().nLsaTeams
    // in Connect()); but the child *Comm ctor bodies run BEFORE Connect() and
    // size buffers based on this value, so we resolve a value here. At ctor time
    // we use num_local_ranks as a conservative placeholder (over-allocates
    // buffer for hypernode); Connect() overwrites with the authoritative NCCL
    // LSA team size (hypernode auto-detect — e.g. NVL72 spanning multiple
    // physical boxes).
    int nvl_local_ranks = 0;
    // Effective number of "LSA teams" = num_ranks / nvl_local_ranks — equals
    // NCCL's nLsaTeams (the natural scope). Set in Connect(); used as the gdaki
    // gate (and the Destroy gate — comms may be torn down before we could
    // re-query NCCL, so we stash the value).
    int n_lsa_teams = 0;
    // CUDA device ordinal (cudaGetDevice at construction). Plumbed into RS/AG
    // ring kernels so timeout/arrival dumps can identify the physical GPU,
    // which is otherwise ambiguous in railed comms where local_rank doesn't
    // map to device id.
    int cuda_device_id = 0;
    std::vector<int> topo_key;  // sorted global ranks of the group; identifies the ncclComm topology
};

// NCCL communicator + GIN device-comm handles (the cross-rank transport).
struct Transport {
    ncclComm_t global_comm = nullptr;
    ncclDevComm dev_comm{};
    uint32_t gin_sigs = 0, gin_qps = 0;
    bool active = false;
};

// The symmetric data/signal buffer, its registration window, and peer table.
struct CommBuffer {
    void *data_buffer = nullptr, *signal_buffer = nullptr;
    size_t buffer_bytes = 0;
    ncclWindow_t gin_win = nullptr;
    void **p2p_ptrs = nullptr;
    size_t send_slots = 0;
    int round_n = 0;
    bool use_ipc_path = false;  // true when intranode-only (num_nodes==1): use cudaMalloc + IPC instead of ncclMemAlloc
};

// Launch streams + kernel-argument marshaling (batched memops).
struct LaunchContext {
    at::cuda::CUDAStream kstream, c1_stream, c2_stream, c3_stream;
    uint64_t *args_gpu = nullptr, *args_host_buffer = nullptr;
    CUstreamBatchMemOpParams *mparam = nullptr;
    int unroll = 0, nvl_ring = 0, rdma_ring = 0, num_sms = 0;
    // Whether this comm marshals kernel args through the batched-memop path
    // (args_gpu/args_host_buffer/mparam). EP drives its kernels directly and
    // never uses this machinery, so it sets this false to skip the allocation.
    bool marshals_args = true;
    // Base GIN sig index for the per-peer kernel-entry arrival counters used
    // by rs / ag_ring under PACE_TIMEOUT_DEBUG. Set by the child
    // RSComm / AGComm constructor after gin_sigs has its base value; the
    // arrival region occupies sig slots [arrival_sig_base, arrival_sig_base +
    // num_ranks). Zero (and unused) when the build flag is not set.
    int arrival_sig_base = 0;
    LaunchContext();  // grabs the four launch streams from the pool
};

class Comm {
public:
    Topology topo;
    Transport transport;
    CommBuffer buffer;
    LaunchContext launch;

    Comm(int rank, int num_ranks, int num_local_ranks, int unroll, int nvl_ring, int rdma_ring, int num_sms, std::vector<int> topo_key, bool marshals_args = true);

    // Set by each *Comm (AG/RS/SG/EP) ctor; called by Connect() AFTER the
    // NCCL query has set the authoritative topo.nvl_local_ranks (and
    // topo.n_lsa_teams), but BEFORE ncclMemAlloc. Sizes buffer_bytes +
    // gin_sigs based on the authoritative scope — the child *Comm ctor
    // body ran BEFORE Connect (can't query NCCL there) and so couldn't size
    // buffers for the true hypernode case (NCCL lsaSize > num_local_ranks).
    std::function<void()> post_connect_fn = nullptr;

    pybind11::bytearray GenHeadInfo();

    // (stream_id, device_index, device_type) of the comm's internal stream.
    // Returned as plain ints rather than a torch::Stream so the binding does
    // not depend on THPStream_Wrap (absent in some torch builds); Python
    // reconstructs a torch.cuda.Stream from these fields.
    std::tuple<int64_t, int64_t, int64_t> GetStream() const;

    // Total shared symmetric memory this comm allocates per rank: the data
    // ring buffer (buffer.buffer_bytes) plus the fixed signal buffer
    // (SIGNAL_BYTES). Allocated once in Connect() via a single ncclMemAlloc /
    // cudaMalloc — nothing is allocated after that, so this is the comm's
    // whole shared-memory footprint.
    size_t GetSharedMemoryBytes() const { return buffer.buffer_bytes + SIGNAL_BYTES; }

    int Connect(pybind11::bytearray head_info);

    void FillArgs(const int& u64s);

    void Destroy();

    ~Comm();
};


class RSComm {
    bool use_wg;
    // Serializes ReduceScatter calls on this comm. With py::gil_scoped_release
    // on the pybind binding, two Python threads sharing one RSComm could
    // otherwise race on args_host_buffer, args_gpu, round_n, mparam, and the
    // launch streams.
    std::mutex lock;
public:
    Comm comm;
    RSComm(int rank, int num_ranks, int num_local_ranks, int unroll, int nvl_ring, int rdma_ring, int num_sms, bool use_wg, std::vector<int> topo_key);

    int out_numel_alignment() const;

    std::tuple<torch::Tensor, std::optional<EventHandle>> ReduceScatter(const std::vector<torch::Tensor>& src, const int& red_op, const float& extra_mul, const float& extra_post_mul, const int& out_cast_type, std::optional<torch::Tensor>& output, std::optional<EventHandle>& previous_event, const bool async);
};


class SGComm {
    // SG has two data-ring modes: the SM-resident kernel (sg.cu) when
    // num_sms > 0 and the 0-SM CE stream walk (scattergatherfunc) when
    // num_sms == 0. ring_mode is kept for the is_ring_mode() query (always true).
    bool ring_mode = true;
    std::mutex lock;
public:
    Comm comm;
    SGComm(int rank, int num_ranks, int num_local_ranks, int unroll, int nvl_ring, int rdma_ring, int num_sms, std::vector<int> topo_key);
    ~SGComm();

    bool is_ring_mode() const { return ring_mode; };

    // The SG kernel-selection gate. num_sms == 0 selects the 0-SM stream path
    // (scattergatherfunc); num_sms > 0 selects the SM-resident kernel
    // (scattergather_kernel_func, grid = num_sms), which further dispatches to
    // the p2p kernel for single-node Nlr in {2,4,8} and the legacy kernel
    // otherwise. Exposed to Python so callers can distinguish the 0/1-SM lean
    // variants (num_sms <= 1) from normal multi-SM kernels. All variants read
    // strided-S input directly.
    int num_sms() const { return comm.launch.num_sms; };
    
    std::tuple<std::vector<torch::Tensor>, std::optional<EventHandle>> ScatterGather(const std::vector<torch::Tensor>& src, const int scatter_dim, const int gather_dim, std::optional<EventHandle>& previous_event, const bool async, const std::optional<std::vector<torch::Tensor>>& out);

};


class AGComm {
public:
    Comm comm;
    // 0-SM (num_sms == 0) cord-path knobs, passed through from
    // CommConfig.ag_zero_sm (formerly the PACE_AG_FORCE_RING / GIN_AG_USE_GRAPH
    // / GIN_AG_FORCE_CAPTURE / GIN_AG_MIN_NODES env vars).
    bool use_ring;               // inter-node ring-forward vs broadcast
    bool use_graph;              // capture the 0-SM AG into a CUDA graph
    bool force_graph_capture;    // debug: re-capture every call
    int min_graph_nodes;         // min graph nodes before graph mode

    // Serializes AllGather calls on this comm. With py::gil_scoped_release on
    // the pybind binding, two Python threads sharing one AGComm could
    // otherwise race on args_host_buffer, args_gpu, round_n, mparam, and the
    // launch streams.
    std::mutex lock;

    AGComm(int rank, int num_ranks, int num_local_ranks, int unroll, int nvl_ring, int rdma_ring, int num_sms, bool use_ring, bool use_graph, bool force_graph_capture, int min_graph_nodes, std::vector<int> topo_key);
    ~AGComm();

    std::tuple<std::optional<torch::Tensor>, std::optional<EventHandle>> AllGather(const std::vector<torch::Tensor>& src, const std::optional<std::vector<torch::Tensor>>& output_tensors, std::optional<EventHandle>& previous_event, const bool async, const int input_dtype_mapping);
};

// PACE's AG / RS / SG collectives only. The EP (expert-parallel) and PP
// (pipeline-parallel) engines live in the upstream PACE repo and are NOT part
// of this contrib.
}

#endif