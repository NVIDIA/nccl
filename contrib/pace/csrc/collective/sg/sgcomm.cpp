
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
#include "collective/sg/sg.cuh"
#include "device/configs.cuh"
#include "util/math.hpp"
#include "services/event.hpp"
#include <unistd.h>
#include <fcntl.h>
#include <cstring>
#include <vector>
#include <string>
#include <stdexcept>
#include "collective/sg/tensor_layout.hpp"

// Signal buffer layout for SG kernel args:
// [p2p_ptrs area: num_local_ranks * sizeof(void*)] - existing
// [signal slots area] - existing
// [SG kernel args area: at offset SG_ARGS_OFFSET]
//   - in_args: SG_MAX_TENSORS * 4 * sizeof(uint64_t)
//   - out_args: SG_MAX_TENSORS * 4 * sizeof(uint64_t)
//   - p2p_ptrs_dev: NUM_MAX_NVL_PEERS * sizeof(void*)
#define SG_ARGS_OFFSET (512 * 1024)  // 512KB offset into signal buffer for SG args

namespace pace {

SGComm::SGComm(int rank, int num_ranks, int num_local_ranks, int unroll, int nvl_ring, int rdma_ring, int num_sms, std::vector<int> topo_key): comm(rank, num_ranks, num_local_ranks, unroll, nvl_ring, rdma_ring, num_sms, std::move(topo_key)) {
    // Defer buffer/gin_sigs sizing to post_connect_fn (see AGComm rationale).
    comm.post_connect_fn = [this, unroll, nvl_ring, rdma_ring, num_sms]() {
        const int num_nodes = comm.topo.num_ranks / comm.topo.nvl_local_ranks;
        // R = max(nvl_ring, rdma_ring) is the actual ring depth used by the
        // push-based SG p2p protocol (slot % R). The send_area must be sized
        // for R slots, not nvl_ring, otherwise rdma_ring > nvl_ring would
        // overflow the send buffer.
        const int R = std::max(nvl_ring, rdma_ring);
        const size_t slot_bytes = sizeof(float4) * GIN_CTA_THREADS * unroll * std::max(num_sms, 1);
        const size_t batch_slot_bytes = slot_bytes * comm.topo.nvl_local_ranks;
        const size_t send_area = batch_slot_bytes * R * num_nodes;
        const size_t recv_area = (num_nodes == 1) ? 0 : batch_slot_bytes * R * num_nodes;
        comm.buffer.buffer_bytes = send_area + recv_area;
        comm.transport.gin_sigs = num_nodes * 2 * std::max(1, num_sms);
        comm.transport.gin_qps = std::max(GIN_QPS, num_sms);
    };
}

SGComm::~SGComm() {
}

std::tuple<std::vector<torch::Tensor>, std::optional<EventHandle>> SGComm::ScatterGather(const std::vector<torch::Tensor>& src, const int scatter_dim, const int gather_dim, std::optional<EventHandle>& previous_event, const bool async, const std::optional<std::vector<torch::Tensor>>& out) {
    // Serializes the host-side critical section (args buffer fill, mparam
    // upload, kernel launch). pybind releases the GIL for this call, so absent
    // this lock two Python threads sharing one SGComm would race on
    // args_host_buffer and mparam.
    std::lock_guard<std::mutex> guard(lock);
    const int num_tensors = src.size();
    std::optional<EventHandle> event;
    std::vector<TensorLayout> layouts(num_tensors), rlayouts(num_tensors);
    HOST_ASSERT(num_tensors <= kMaxArgs);
    constexpr int64_t numel_align = 8; // float4 alignment for kernel output
    auto compute_stream = at::cuda::getCurrentCUDAStream();
    auto launch_stream = comm.launch.kstream;
    int capture_round_n = comm.buffer.round_n;
    comm.buffer.round_n += 1;
    if (previous_event.has_value()) {
        stream_wait(launch_stream, previous_event.value());
    } else {
        stream_wait(launch_stream, compute_stream);
    }
    std::vector<torch::Tensor> results(num_tensors);
    size_t total_chunk_bytes = 0;
    // Create host tensor for args (to keep reference and prevent early release)
    // Format: [in_args + out_args interleaved: (in_ptr, in_X, in_Y, in_Z, out_ptr, out_X, out_Y, out_Z) * num_tensors]
    uint64_t *args_host = comm.launch.args_host_buffer;
    size_t size2 = 0, esize = 0;
    if (out.has_value()) {
        HOST_ASSERT(static_cast<int>(out->size()) == num_tensors);
    }
    // Adopt a caller-provided out[i] (validated against the computed result
    // shape/dtype/alignment) or allocate a fresh result tensor.
    auto alloc_or_adopt = [&](int i, std::initializer_list<int64_t> out_shape) -> torch::Tensor {
        if (!out.has_value()) {
            return torch::empty(out_shape, dtype(src[i].scalar_type()).device(torch::kCUDA));
        }
        const torch::Tensor& o = (*out)[i];
        HOST_ASSERT(o.device().is_cuda() && o.is_contiguous());
        HOST_ASSERT(o.scalar_type() == src[i].scalar_type());
        HOST_ASSERT(o.sizes() == torch::IntArrayRef(out_shape));
        HOST_ASSERT((reinterpret_cast<int64_t>(o.data_ptr()) % sizeof(int4)) == 0);
        return o;
    };
    for (int i = 0; i < num_tensors; ++i) {
        HOST_ASSERT(src[i].dim() == 3);
        if (size2 == 0) {
            size2 = src[i].size(2);
            esize = src[i].element_size();
            HOST_ASSERT((size2 * esize) % sizeof(float4) == 0);
        } else {
            HOST_ASSERT(src[i].size(2) == size2);
            HOST_ASSERT(src[i].element_size() == esize);
        }
        // Strided-S contract (ulysses CP / USP qkv split): the only strided
        // axis may be S = src dim 0; dims 1,2 must be contiguous. Canonical
        // case: Q = storage[:,0] of a (S, G, H, D) C-order storage, stride
        // (G*H*D, D, 1). The SG kernels read the S-rows at the given stride
        // directly (no copy-in); the per-tensor stride is passed via the in-Y
        // arg slot below. (numel()==0 exempt: torch computes strides of size-0
        // dims as if they were size 1, so stride(0) != size(1)*size(2) on
        // empty inputs — the row-stride args are meaningless then but harmless.)
        HOST_ASSERT(src[i].stride(2) == 1);
        HOST_ASSERT(src[i].stride(1) == src[i].size(2));
        HOST_ASSERT(src[i].stride(0) >= src[i].size(1) * src[i].size(2));
        HOST_ASSERT((src[i].stride(0) * esize) % sizeof(float4) == 0);
        std::vector<size_t> shape(4);
        if (scatter_dim == 0) {
            shape[0] = uint64_t(comm.topo.num_ranks);
            shape[1] = uint64_t(src[i].size(0) / comm.topo.num_ranks);
            shape[2] = uint64_t(src[i].size(1));
            shape[3] = uint64_t(src[i].size(2));
            results[i] = alloc_or_adopt(i, {src[i].size(0) / comm.topo.num_ranks, src[i].size(1) * comm.topo.num_ranks, src[i].size(2)});
            rlayouts[i] = TensorLayout(results[i].data_ptr(), {uint64_t(src[i].size(0) / comm.topo.num_ranks), uint64_t(comm.topo.num_ranks), uint64_t(src[i].size(1)), uint64_t(src[i].size(2))}, results[i].element_size(), gather_dim);
        } else {
            shape[0] = uint64_t(src[i].size(0));
            shape[1] = uint64_t(comm.topo.num_ranks);
            shape[2] = uint64_t(src[i].size(1) / comm.topo.num_ranks);
            shape[3] = uint64_t(src[i].size(2));
            results[i] = alloc_or_adopt(i, {src[i].size(0) * comm.topo.num_ranks, src[i].size(1) / comm.topo.num_ranks, src[i].size(2)});
            rlayouts[i] = TensorLayout(results[i].data_ptr(), {uint64_t(comm.topo.num_ranks), uint64_t(src[i].size(0)), uint64_t(src[i].size(1) / comm.topo.num_ranks), uint64_t(src[i].size(2))}, results[i].element_size(), gather_dim);
        }
        layouts[i] = TensorLayout(src[i].data_ptr(), shape, src[i].element_size(), scatter_dim);
        // Calculate total chunk bytes for kernel version
        total_chunk_bytes += align_up(layouts[i].get_total_bytes() / comm.topo.num_ranks, sizeof(int4));
    }
    // Fill in args data: interleaved [in_ptr, in_X, in_Y, in_Z, out_ptr, out_X, out_Y, out_Z] per tensor
    for (int i = 0; i < num_tensors; ++i) {
        // Input: [base_ptr, X, Y=S-row byte stride, Z (in bytes)]. in-Y is
        // redundant as num_ranks (the kernels know num_ranks), so it carries
        // the per-tensor S-row byte stride = stride(0)*esize instead — the
        // SG kernels walk the rows at this pitch, which equals the dense
        // H*D*esize for contiguous inputs (strided addressing reduces exactly
        // to the original contiguous formula in that case).
        args_host[i * 8 + 0] = reinterpret_cast<uint64_t>(layouts[i].get_base_ptr());
        args_host[i * 8 + 1] = static_cast<uint64_t>(layouts[i].get_X());
        args_host[i * 8 + 2] = static_cast<uint64_t>(src[i].stride(0)) * esize;
        args_host[i * 8 + 3] = static_cast<uint64_t>(layouts[i].get_Z());

        // Output: [base_ptr, X, Y=num_ranks, Z (in bytes)]
        args_host[i * 8 + 4] = reinterpret_cast<uint64_t>(rlayouts[i].get_base_ptr());
        args_host[i * 8 + 5] = static_cast<uint64_t>(rlayouts[i].get_X());
        args_host[i * 8 + 6] = static_cast<uint64_t>(rlayouts[i].get_Y());
        args_host[i * 8 + 7] = static_cast<uint64_t>(rlayouts[i].get_Z());
    }
    // Device pointer in signal buffer (at SG_ARGS_OFFSET)
    // p2p_ptrs is already in signal_buffer, no need to copy
    uint64_t *args_dev = comm.launch.args_gpu;

    if (comm.launch.num_sms > 0) {
        // Copy args to device (single memcpy for both in and out args)
        const auto u64s = num_tensors * 8;
        comm.FillArgs(u64s);
        CUCHECK(cuStreamBatchMemOp(launch_stream, u64s, comm.launch.mparam, 0));
    }
    const bool use_kernel = (comm.launch.num_sms > 0);
    if (use_kernel) {
        // Kernel version: single kernel with time-multiplexed lambdas
        scattergather_kernel_func(
            args_host, args_dev, num_tensors,
            comm.buffer.gin_win, comm.buffer.data_buffer, comm.buffer.signal_buffer, nullptr,
            comm.launch.unroll, comm.launch.nvl_ring, comm.launch.rdma_ring, capture_round_n,
            comm.topo.rank, comm.topo.nvl_local_ranks, comm.topo.num_ranks,
            comm.transport.dev_comm, comm.launch.num_sms > 0 ? comm.launch.num_sms : 16, launch_stream,
            scatter_dim);
    } else {
        // Original stream-based version. scattergatherfunc now uses a
        // single copy_stream (c1_stream) for all CE ops — c1/c2/c3 are
        // issued in order on it with c3 lagged K slots (see sg.cu).
        if (comm.launch.num_sms == 0) {
            stream_wait(comm.launch.c1_stream, launch_stream);
        }
        // Per-tensor S-row byte stride for the 0-SM stream path (== dense row
        // size for contiguous inputs; the strided QKV-view pitch otherwise).
        std::vector<uint64_t> src_row_strides(num_tensors);
        for (int i = 0; i < num_tensors; ++i) {
            src_row_strides[i] = static_cast<uint64_t>(src[i].stride(0)) * esize;
        }
        scattergatherfunc(layouts, rlayouts, comm.buffer.gin_win, comm.buffer.data_buffer, comm.buffer.signal_buffer, comm.buffer.p2p_ptrs, comm.launch.unroll, comm.launch.nvl_ring, comm.launch.rdma_ring, comm.buffer.send_slots, capture_round_n, comm.topo.rank, comm.topo.nvl_local_ranks, comm.topo.num_ranks, comm.launch.mparam, comm.transport.dev_comm, comm.launch.num_sms, launch_stream, comm.launch.c1_stream, src_row_strides);
        if (comm.launch.num_sms == 0) {
            stream_wait(launch_stream, comm.launch.c1_stream);
        }
    }
    if (async) {
        event = EventHandle(launch_stream);
    } else {
        stream_wait(compute_stream, launch_stream);
    }
    // Return args_cpu to keep reference (prevents early release of pinned memory)
    return {results, event};
}

}