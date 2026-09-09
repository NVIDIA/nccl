
#include <cassert>
#include <cstddef>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDADataType.h>
#include <nccl.h>
#include <optional>
#include <pybind11/functional.h>
#include <torch/python.h>
#include <cstdlib>
#include <cstdio>
#include <cstdint>
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <torch/types.h>
#include "runtime/gincomm.hpp"
#include "util/error.hpp"
#include "collective/ag/ag_zero_sm.cuh"
#include "collective/ag/ag_ring.cuh"
#include "device/configs.cuh"
#include "services/event.hpp"
#include <unistd.h>
#include <fcntl.h>
#include <cstring>
#include <vector>
#include <string>
#include <stdexcept>

namespace pace {

AGComm::AGComm(int rank, int num_ranks, int num_local_ranks, int unroll, int nvl_ring, int rdma_ring, int num_sms, bool use_ring, bool use_graph, bool force_graph_capture, int min_graph_nodes, std::vector<int> topo_key): comm(rank, num_ranks, num_local_ranks, unroll, nvl_ring, rdma_ring, num_sms, std::move(topo_key)), use_ring(use_ring), use_graph(use_graph), force_graph_capture(force_graph_capture), min_graph_nodes(min_graph_nodes) {
    // The 0SM mem_layout aliases the local send buffer onto recv_area[node_id]
    // using a single ring-depth dimension R. The Python AGComm wrapper already
    // equalizes nvl_ring_size / rdma_ring_size to their max before constructing
    // us; coerce again here so direct-from-C++ callers (and any future drift)
    // see a consistent layout. nvl_ring/rdma_ring are public members of
    // Comm, so updating them here propagates to every downstream
    // function that reads them.
    const int R = std::max(nvl_ring, rdma_ring);
    comm.launch.nvl_ring = R;
    comm.launch.rdma_ring = R;
    nvl_ring = R;
    rdma_ring = R;
    comm.transport.gin_qps = std::max(GIN_QPS, num_sms);

    // Defer buffer/gin_sigs sizing to post_connect_fn — runs in Connect()
    // AFTER the NCCL query sets the authoritative topo.nvl_local_ranks (which
    // may exceed the conservative num_local_ranks placeholder set in
    // Comm::Comm for the hypernode case — e.g. NVL72 spanning 2 physical
    // boxes where NCCL reports one LSA team of 16). Sizing here would use
    // the conservative value, mismatching the kernel's view (kernel sees
    // the authoritative value post-Connect) — would set gin_sigs non-zero
    // (conservative num_nodes>1) for a hypernode where gdaki isn't set up.
    const int eng = std::max(1, num_sms);
    comm.post_connect_fn = [this, unroll, num_sms, eng, R]() {
        const int num_nodes = comm.topo.num_ranks / comm.topo.nvl_local_ranks;
        const size_t slot_bytes = sizeof(int4) * GIN_CTA_THREADS * unroll;
        // Buffer layout: num_sms==0 → cord path (single R×num_nodes data block,
        // send slot aliases recv_area[node_id]); num_sms>0 → ring kernel
        // (separate send + recv regions, recv = R per rail, replicated per engine).
        if (num_sms == 0) {
            const size_t per_node = slot_bytes * static_cast<size_t>(R);
            comm.buffer.buffer_bytes = per_node * static_cast<size_t>(num_nodes);
        } else {
            const size_t send_area = slot_bytes * static_cast<size_t>(R);
            const size_t recv_area = (num_nodes == 1)
                ? 0
                : slot_bytes * static_cast<size_t>(R);
            comm.buffer.buffer_bytes = (send_area + recv_area) * static_cast<size_t>(eng);
        }
        // GIN signals: ring kernel (num_sms>0) uses 2 per engine (data + ack);
        // cord path (num_sms==0) uses 2 per source-node (the cord kernel's
        // broadcast/ring wait-loops both index by source).
        comm.transport.gin_sigs = (num_nodes == 1)
            ? 0u
            : static_cast<uint32_t>((num_sms == 0) ? (2 * num_nodes) : (2 * eng));
#ifdef PACE_TIMEOUT_DEBUG
        // Reserve num_ranks GIN sig slots for the kernel-entry arrival debug. Only
        // meaningful in the multi-node case (single-node has no inter-rank gin
        // signaling needed for arrival), but allocate unconditionally for layout
        // simplicity. Base index = end of regular sigs.
        if (comm.transport.gin_sigs > 0) {
            comm.launch.arrival_sig_base = static_cast<int>(comm.transport.gin_sigs);
            comm.transport.gin_sigs += static_cast<uint32_t>(comm.topo.num_ranks);
        } else {
            comm.launch.arrival_sig_base = 0;
        }
#else
        comm.launch.arrival_sig_base = 0;
#endif
    };
}

AGComm::~AGComm() {
    // Evict this comm's cached AG graphs. The cache is keyed by signal_buffer
    // (a pointer the comm owns); without eviction, a destroyed comm leaves
    // dangling-key entries that a later comm reusing the same address + same
    // topology could hit as a stale graph (baking the old, now-freed
    // gin_win/addresses). shared_ptr keeps any in-flight launch's cache alive
    // across this clear. Clears the whole global cache (other comms re-capture
    // on next use) — dtors are rare, so the over-eviction is acceptable.
    clear_ag_zero_sm_graph_cache();
}

std::tuple<std::optional<torch::Tensor>, std::optional<EventHandle>> AGComm::AllGather(const std::vector<torch::Tensor>& src, const std::optional<std::vector<torch::Tensor>>& output_tensors, std::optional<EventHandle>& previous_event, const bool async, const int input_dtype_mapping) {
    // Serializes the host-side critical section (args buffer fill, round_n
    // increment, mparam upload, kernel launch). pybind releases the GIL for
    // this call, so absent this lock two Python threads sharing one AGComm
    // would race on those resources.
    std::lock_guard<std::mutex> guard(lock);
    const int num_tensors = src.size();
    std::optional<EventHandle> event;

    const size_t args_size = sizeof(uint64_t) * num_tensors * 4;
    HOST_ASSERT(num_tensors <= kMaxArgs);
    int64_t total_out_numel = 0l;
    uint64_t *args_data = comm.launch.args_host_buffer;
    for (int i = 0; i < num_tensors; ++i) {
        args_data[4 * i] = reinterpret_cast<uint64_t>(src[i].data_ptr());      // input ptr
        args_data[4 * i + 1] = static_cast<uint64_t>(src[i].numel() * (input_dtype_mapping == AGCOMM_INPUT_DTYPE_MAPPING_NONE ? src[i].element_size() : sizeof(__nv_bfloat16)));  // byte size
        // args_data[4 * i + 2] will be set to output ptr below
        // args_data[4 * i + 3] is reserved
        total_out_numel += src[i].numel() * comm.topo.num_ranks;
    }
    auto compute_stream = at::cuda::getCurrentCUDAStream();
    auto launch_stream = !async ? compute_stream : comm.launch.kstream;
    if (launch_stream != compute_stream) {
        if (previous_event.has_value()) {
            stream_wait(launch_stream, previous_event.value());
        } else {
            stream_wait(launch_stream, compute_stream);
        }
    }
    
    // Handle output tensors: use provided tensors or allocate new ones
    // Output pointers are stored in args_cpu[4*i+2]
    std::optional<torch::Tensor> result = std::nullopt;  // Only set when output_tensors is not provided
    
    if (output_tensors.has_value()) {
        // Use provided output tensors - store their pointers in args_cpu
        const auto& out_tensors = output_tensors.value();
        HOST_ASSERT(static_cast<int>(out_tensors.size()) == num_tensors);
        for (int i = 0; i < num_tensors; ++i) {
            HOST_ASSERT(out_tensors[i].numel() == src[i].numel() * comm.topo.num_ranks);
            args_data[4 * i + 2] = reinterpret_cast<uint64_t>(out_tensors[i].data_ptr());
        }
        // result remains nullopt - user should use their provided output tensors
    } else {
        // Allocate combined result tensor and store pointers in args_cpu.
        // When all input tensors share the same dtype, use a single typed
        // tensor. For mixed dtypes, fall back to a uint8 byte buffer and
        // compute byte-offsets per tensor.
        bool uniform_dtype = true;
        for (int i = 1; i < num_tensors; ++i) {
            if (src[i].scalar_type() != src[0].scalar_type()) {
                uniform_dtype = false;
                break;
            }
        }
        if (uniform_dtype) {
            if (input_dtype_mapping == AGCOMM_INPUT_DTYPE_MAPPING_NONE) {
                result = torch::empty({total_out_numel}, dtype(src[0].scalar_type()).device(torch::kCUDA));
            } else {
                result = torch::empty({total_out_numel}, dtype(torch::kBFloat16).device(torch::kCUDA));
            }
            int64_t offset = 0;
            for (int i = 0; i < num_tensors; ++i) {
                int64_t tensor_out_numel = src[i].numel() * comm.topo.num_ranks;
                void* out_ptr = reinterpret_cast<void*>(reinterpret_cast<uint8_t*>(result.value().data_ptr()) + offset * src[0].element_size());
                args_data[4 * i + 2] = reinterpret_cast<uint64_t>(out_ptr);
                offset += tensor_out_numel;
            }
        } else {
            size_t total_out_bytes = 0;
            std::vector<size_t> out_offsets(num_tensors);
            for (int i = 0; i < num_tensors; ++i) {
                out_offsets[i] = total_out_bytes;
                total_out_bytes += static_cast<size_t>(src[i].numel()) * comm.topo.num_ranks * src[i].element_size();
            }
            result = torch::empty({static_cast<int64_t>(total_out_bytes)},
                                  torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA));
            for (int i = 0; i < num_tensors; ++i) {
                void* out_ptr = reinterpret_cast<void*>(reinterpret_cast<uint8_t*>(result.value().data_ptr()) + out_offsets[i]);
                args_data[4 * i + 2] = reinterpret_cast<uint64_t>(out_ptr);
            }
        }
    }
    int capture_round_n = comm.buffer.round_n;
    comm.buffer.round_n += 1;
    if (comm.launch.num_sms == 0) {
        // allgather_zero_sm_func only uses c1_stream (single-stream c1/c2/c3 with
        // the c3 phase interleaved). c2_stream is only used by the
        // SM-mode allgather_ring_func dispatch under num_sms > 0.
        stream_wait(comm.launch.c1_stream, launch_stream);
    }
    // Dispatch:
    //   num_sms == 0 → allgather_zero_sm_func (cord path: copy-engine c1/c2/c3
    //     pipeline + cord kernel, graph-cached when node count pays for it).
    //     The cord kernel's `use_ring` flag selects ring-forward vs broadcast
    //     for the inter-node send; single-node forces use_ring=false.
    //   num_sms > 0 → allgather_ring_func (SM-resident ring kernel).
    if (comm.launch.num_sms == 0) {
        const bool eff_use_ring = use_ring && (comm.topo.nvl_local_ranks != comm.topo.num_ranks);
        allgather_zero_sm_func(args_data, num_tensors, comm.buffer.gin_win, comm.buffer.data_buffer,
                      comm.buffer.signal_buffer, comm.buffer.p2p_ptrs, comm.launch.unroll, comm.launch.nvl_ring, comm.launch.rdma_ring,
                      comm.buffer.send_slots, capture_round_n,
                      comm.topo.rank, comm.topo.nvl_local_ranks, comm.topo.num_ranks, comm.launch.mparam, comm.transport.dev_comm,
                      launch_stream, comm.launch.c1_stream,
                      eff_use_ring, use_graph, force_graph_capture, min_graph_nodes);
    } else {
        if (comm.topo.nvl_local_ranks != comm.topo.num_ranks) stream_wait(comm.launch.c2_stream, launch_stream);
        allgather_ring_func(args_data, comm.launch.args_gpu, num_tensors, comm.buffer.gin_win, comm.buffer.data_buffer,
                            comm.buffer.signal_buffer, comm.launch.unroll, comm.launch.nvl_ring, comm.launch.rdma_ring,
                            comm.buffer.send_slots, capture_round_n, input_dtype_mapping,
                            comm.topo.rank, comm.topo.nvl_local_ranks, comm.topo.num_ranks, comm.transport.dev_comm,
                            comm.launch.num_sms, launch_stream,
                            comm.launch.arrival_sig_base, comm.topo.cuda_device_id);
        if (comm.topo.nvl_local_ranks != comm.topo.num_ranks) stream_wait(launch_stream, comm.launch.c2_stream);
    }
    if (comm.launch.num_sms == 0) {
        stream_wait(launch_stream, comm.launch.c1_stream);
    }
    if (async) {
        event = EventHandle(launch_stream);
    } else if (launch_stream != compute_stream) {
        stream_wait(compute_stream, launch_stream);
    }
    return {std::move(result), event};
}

}