
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
#include "collective/rs/rs.cuh"
#include "device/configs.cuh"
#include "util/math.hpp"
#include "services/event.hpp"
#include <unistd.h>
#include <fcntl.h>
#include <cstring>
#include <vector>
#include <string>
#include <stdexcept>

namespace pace {

RSComm::RSComm(int rank, int num_ranks, int num_local_ranks, int unroll, int nvl_ring, int rdma_ring, int num_sms, bool use_wg, std::vector<int> topo_key): comm(rank, num_ranks, num_local_ranks, unroll, nvl_ring, rdma_ring, num_sms, std::move(topo_key)), use_wg(use_wg) {
    // Defer buffer/gin_sigs sizing to post_connect_fn (see AGComm rationale).
    comm.transport.gin_qps = num_sms;
    comm.post_connect_fn = [this, unroll, nvl_ring, rdma_ring, num_sms, use_wg]() {
        const int num_ultra_nodes = comm.topo.num_ranks / comm.topo.nvl_local_ranks;
        if (comm.topo.nvl_local_ranks == 1) comm.launch.nvl_ring = 0;
        const size_t slot_bytes = sizeof(float4) * GIN_CTA_THREADS * unroll * std::max(num_sms, 1);
        comm.buffer.buffer_bytes = (use_wg ? kRSRingWarpGroups : 1) * slot_bytes * ((comm.topo.nvl_local_ranks == 1 ? 0 : 1) * nvl_ring + (num_ultra_nodes == 1 ? 0 : 2) * rdma_ring);

        // All-ring path indexes inter GIN signals as (blockIdx.x * WG + wg) * 2,
        // so the kernel uses up to gridDim.x * WG * 2 sigs regardless of which
        // sub-branch (pure-inter or mixed) executes. Scale gin_sigs by WG for
        // any all-ring multi-node case so the index range is in bounds.
        comm.transport.gin_sigs = static_cast<uint32_t>(kRSRingWarpGroups) * 2u * static_cast<uint32_t>(std::max(1, num_sms));
#ifdef PACE_TIMEOUT_DEBUG
        // Kernel-entry arrival debug reserves num_ranks extra GIN sig slots so each
        // peer can increment our local arrival counter once per kernel call. Base
        // index = end of the regular sigs region so the existing kernel indexing
        // stays unchanged.
        comm.launch.arrival_sig_base = static_cast<int>(comm.transport.gin_sigs);
        comm.transport.gin_sigs += static_cast<uint32_t>(comm.topo.num_ranks);
#else
        comm.launch.arrival_sig_base = 0;
#endif
    };
}

int RSComm::out_numel_alignment() const {
    return 4; // 4 elements aligned
}

std::tuple<torch::Tensor, std::optional<EventHandle>> RSComm::ReduceScatter(const std::vector<torch::Tensor>& src, const int& red_op, const float& extra_mul, const float& extra_post_mul, const int& out_cast_type, std::optional<torch::Tensor>& output, std::optional<EventHandle>& previous_event, const bool async) {
    // Serializes the host-side critical section (args buffer fill, round_n
    // increment, mparam upload, kernel launch). pybind releases the GIL for
    // this call, so absent this lock two Python threads sharing one RSComm
    // would race on those resources.
    std::lock_guard<std::mutex> guard(lock);
    const int num_tensors = src.size();
    std::optional<EventHandle> event;

    const auto e_type = src[0].scalar_type();
    HOST_ASSERT(e_type == torch::kFloat32 || e_type == torch::kBFloat16);
    HOST_ASSERT(red_op == RS_RED_OP_AVG || red_op == RS_RED_OP_SUM);
    HOST_ASSERT(num_tensors <= kMaxArgs);
    const int64_t numel_align = out_numel_alignment();

    // --- Step 1: classify tensors by alignment ---
    // ptr aligned to input width (float in -> 16B / bf16 in -> 8B); elements 4-aligned (float4 load granularity)
    const size_t ptr_align = (e_type == torch::kFloat32) ? sizeof(float4) : sizeof(float2);
    const uint64_t elem_align = 4ULL;
    std::vector<int> aligned_indices, nonaligned_indices;
    aligned_indices.reserve(num_tensors);
    nonaligned_indices.reserve(num_tensors);
    for (int i = 0; i < num_tensors; ++i) {
        HOST_ASSERT(src[i].scalar_type() == e_type);
        uint64_t ptr = reinterpret_cast<uint64_t>(src[i].data_ptr());
        uint64_t chunk_elements = static_cast<uint64_t>(src[i].numel()) / static_cast<uint64_t>(src[i].size(0));
        bool is_aligned = (ptr % ptr_align == 0) && (chunk_elements % elem_align == 0);
        if (is_aligned)
            aligned_indices.push_back(i);
        else
            nonaligned_indices.push_back(i);
    }
    const int n_aligned = static_cast<int>(aligned_indices.size());
    const int n_nonaligned = static_cast<int>(nonaligned_indices.size());

    // --- Step 2: fill args_host_buffer (aligned first, then non-aligned) ---
    uint64_t *args_data = comm.launch.args_host_buffer;
    int64_t aligned_out_numel = 0;
    int64_t total_out_numel = 0;
    int slot = 0;
    for (int idx : aligned_indices) {
        args_data[4 * slot] = reinterpret_cast<uint64_t>(src[idx].data_ptr());
        args_data[4 * slot + 1] = static_cast<uint64_t>(src[idx].size(0));
        args_data[4 * slot + 2] = static_cast<uint64_t>(src[idx].numel()) / args_data[4 * slot + 1];
        const int64_t out_numel = ceil_div(args_data[4 * slot + 1], static_cast<uint64_t>(comm.topo.num_ranks)) * static_cast<int64_t>(args_data[4 * slot + 2]);
        aligned_out_numel += align_up(out_numel, numel_align);
        slot++;
    }
    total_out_numel = aligned_out_numel;
    for (int idx : nonaligned_indices) {
        args_data[4 * slot] = reinterpret_cast<uint64_t>(src[idx].data_ptr());
        args_data[4 * slot + 1] = static_cast<uint64_t>(src[idx].size(0));
        args_data[4 * slot + 2] = static_cast<uint64_t>(src[idx].numel()) / args_data[4 * slot + 1];
        const int64_t out_numel = ceil_div(args_data[4 * slot + 1], static_cast<uint64_t>(comm.topo.num_ranks)) * static_cast<int64_t>(args_data[4 * slot + 2]);
        total_out_numel += align_up(out_numel, numel_align);
        slot++;
    }

    // --- Step 3: stream sync + allocate output ---
    auto compute_stream = at::cuda::getCurrentCUDAStream();
    auto launch_stream = !async ? compute_stream : comm.launch.kstream;
    if (launch_stream != compute_stream) {
        if (previous_event.has_value()) {
            stream_wait(launch_stream, previous_event.value());
        } else {
            stream_wait(launch_stream, compute_stream);
        }
    }
    torch::Tensor result;
    const auto expected_out_dtype = (out_cast_type == RS_OUT_CAST_BF16) ? torch::kBFloat16 : torch::kFloat32;
    if (output.has_value()) {
        HOST_ASSERT(output->device().is_cuda() && output->is_contiguous());
        HOST_ASSERT(output->scalar_type() == expected_out_dtype);
        HOST_ASSERT(output->numel() == total_out_numel);
        result = *output;
    } else {
        result = torch::empty({total_out_numel}, dtype(expected_out_dtype).device(torch::kCUDA));
    }
    void *output_ptr = result.data_ptr();

    // --- Step 4: upload all args to GPU (single batch) ---
    const auto u64s = num_tensors * 4;
    comm.FillArgs(u64s);
    CUCHECK(cuStreamBatchMemOp(launch_stream, u64s, comm.launch.mparam, 0));

    // --- Step 5: dual kernel launch (aligned fast-path, non-aligned slow-path) ---
    const auto dtype_ = e_type == torch::kFloat32 ? RS_TYPE_FLOAT32 : RS_TYPE_BFLOAT16;
    const size_t aligned_out_bytes = aligned_out_numel * result.element_size();
    void *nonaligned_output_ptr = static_cast<char*>(output_ptr) + aligned_out_bytes;

    // Assign separate round_n for each launch
    int round_n_aligned = -1, round_n_nonaligned = -1;
    if (n_aligned > 0) { round_n_aligned = comm.buffer.round_n; comm.buffer.round_n += 1; }
    if (n_nonaligned > 0) { round_n_nonaligned = comm.buffer.round_n; comm.buffer.round_n += 1; }
    if (n_aligned > 0) {
        reduce_scatter_func(dtype_, red_op, extra_mul, extra_post_mul, out_cast_type, comm.launch.unroll, args_data, comm.launch.args_gpu, n_aligned, output_ptr, comm.buffer.gin_win, comm.buffer.data_buffer, reinterpret_cast<int*>(comm.buffer.signal_buffer), comm.launch.nvl_ring, comm.launch.rdma_ring, round_n_aligned, use_wg, comm.topo.rank, comm.topo.nvl_local_ranks, comm.topo.num_ranks, comm.transport.dev_comm, comm.launch.num_sms, launch_stream, comm.launch.arrival_sig_base, comm.topo.cuda_device_id);
    }
    if (n_nonaligned > 0) {
        reduce_scatter_func(dtype_, red_op, extra_mul, extra_post_mul, out_cast_type, comm.launch.unroll, args_data + n_aligned * 4, comm.launch.args_gpu + n_aligned * 4, n_nonaligned, nonaligned_output_ptr, comm.buffer.gin_win, comm.buffer.data_buffer, reinterpret_cast<int*>(comm.buffer.signal_buffer), comm.launch.nvl_ring, comm.launch.rdma_ring, round_n_nonaligned, use_wg, comm.topo.rank, comm.topo.nvl_local_ranks, comm.topo.num_ranks, comm.transport.dev_comm, comm.launch.num_sms, launch_stream, comm.launch.arrival_sig_base, comm.topo.cuda_device_id);
    }
    if (async) {
        event = EventHandle(launch_stream);
    } else if (launch_stream != compute_stream) {
        stream_wait(compute_stream, launch_stream);
    }
    return {result, event};
}

}