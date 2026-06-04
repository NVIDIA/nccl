/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <cstdint>

#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <ATen/cuda/CUDAContext.h>

#include <nccl.h>
#include <ubx/ubx.h>

namespace py = pybind11;

namespace {

// Reinterpret a Python-side integer (devcomm_handle / window_handle / raw
// device pointer) into the proper C type. The Phase 3 ABI passes every
// pointer-shaped argument as Python int (via uintptr_t) — caller responsible
// for handing in valid handles.
inline ncclDevComm_t const* as_devcomm(std::uintptr_t h) {
  return reinterpret_cast<ncclDevComm_t const*>(h);
}
inline ncclWindow_t as_window(std::uintptr_t h) {
  return reinterpret_cast<ncclWindow_t>(h);
}

}  // namespace

PYBIND11_MODULE(_C, m) {
  m.doc() = "UB-X (Ultra Bandwidth X): NVLink collectives — NCCL device-API ABI";

  // ===== UB-X kernel-launcher bindings =====
  // The Phase 3 ABI takes (devcomm, window, pool_ptr) handles + per-tensor
  // byte offsets where the legacy ABI took (uc0ptr, mc0ptr, mcptr_in/out).
  // Local non-pool pointers (gamma, residual, ptr_in, etc.) are passed as
  // plain Python ints (uintptr_t) — the launcher casts to void* internally.

  m.def(
      "ubx_allreduce_2shot_mc",
      [](int ranks, int myrank,
         std::uintptr_t devcomm, std::uintptr_t window, std::uintptr_t pool_ptr,
         std::size_t in_offset, std::size_t out_offset, std::size_t bytes,
         std::uintptr_t residual_in, std::uintptr_t residual_out, bool fuse_layernorm,
         std::uintptr_t gamma, float eps, const int hidden_size,
         int default_sms, int smlimit, int cgasize, int nchunk, bool multi_kernel) {
        ubx_allreduce_2shot_mc(
            ranks, myrank,
            as_devcomm(devcomm), as_window(window), pool_ptr,
            in_offset, out_offset, bytes,
            residual_in, residual_out, fuse_layernorm,
            gamma, eps, hidden_size,
            default_sms, smlimit, cgasize, nchunk, multi_kernel,
            at::cuda::getCurrentCUDAStream());
      },
      py::arg("ranks"), py::arg("myrank"),
      py::arg("devcomm"), py::arg("window"), py::arg("pool_ptr"),
      py::arg("in_offset"), py::arg("out_offset"), py::arg("bytes"),
      py::arg("residual_in"), py::arg("residual_out"), py::arg("fuse_layernorm"),
      py::arg("gamma"), py::arg("eps"), py::arg("hidden_size"),
      py::arg("default_sms") = 0, py::arg("smlimit") = 0,
      py::arg("cgasize"), py::arg("nchunk"), py::arg("multi_kernel"));

  m.def(
      "ubx_allreduce_2shot_uc",
      [](int ranks, int myrank,
         std::uintptr_t devcomm, std::uintptr_t window, std::uintptr_t pool_ptr,
         std::size_t in_offset, std::size_t out_offset, std::size_t bytes,
         std::uintptr_t residual_in, std::uintptr_t residual_out, bool fuse_layernorm,
         std::uintptr_t gamma, float eps, const int hidden_size,
         int default_sms, int smlimit, int cgasize, int nchunk, bool multi_kernel) {
        ubx_allreduce_2shot_uc(
            ranks, myrank,
            as_devcomm(devcomm), as_window(window), pool_ptr,
            in_offset, out_offset, bytes,
            residual_in, residual_out, fuse_layernorm,
            gamma, eps, hidden_size,
            default_sms, smlimit, cgasize, nchunk, multi_kernel,
            at::cuda::getCurrentCUDAStream());
      },
      py::arg("ranks"), py::arg("myrank"),
      py::arg("devcomm"), py::arg("window"), py::arg("pool_ptr"),
      py::arg("in_offset"), py::arg("out_offset"), py::arg("bytes"),
      py::arg("residual_in"), py::arg("residual_out"), py::arg("fuse_layernorm"),
      py::arg("gamma"), py::arg("eps"), py::arg("hidden_size"),
      py::arg("default_sms") = 0, py::arg("smlimit") = 0,
      py::arg("cgasize"), py::arg("nchunk"), py::arg("multi_kernel"));

  m.def(
      "ubx_allreduce_2shot_mc_lamport",
      [](int ranks, int myrank,
         std::uintptr_t devcomm, std::uintptr_t window, std::uintptr_t pool_ptr,
         std::uintptr_t ucptr_out, std::size_t in_offset, std::size_t out_offset,
         std::uintptr_t clear_ptr, std::size_t bytes, bool poisoned,
         std::uintptr_t residual_in, std::uintptr_t residual_out, bool fuse_layernorm,
         std::uintptr_t gamma, float eps, const int hidden_size,
         int default_sms, int smlimit, int cgasize, int nchunk, bool multi_kernel) {
        ubx_allreduce_2shot_mc_lamport(
            ranks, myrank,
            as_devcomm(devcomm), as_window(window), pool_ptr,
            ucptr_out, in_offset, out_offset,
            clear_ptr, bytes, poisoned,
            residual_in, residual_out, fuse_layernorm,
            gamma, eps, hidden_size,
            default_sms, smlimit, cgasize, nchunk, multi_kernel,
            at::cuda::getCurrentCUDAStream());
      },
      py::arg("ranks"), py::arg("myrank"),
      py::arg("devcomm"), py::arg("window"), py::arg("pool_ptr"),
      py::arg("ucptr_out"), py::arg("in_offset"), py::arg("out_offset"),
      py::arg("clear_ptr"), py::arg("bytes"), py::arg("poisoned"),
      py::arg("residual_in"), py::arg("residual_out"), py::arg("fuse_layernorm"),
      py::arg("gamma"), py::arg("eps"), py::arg("hidden_size"),
      py::arg("default_sms") = 0, py::arg("smlimit") = 0,
      py::arg("cgasize"), py::arg("nchunk"), py::arg("multi_kernel"));

  m.def(
      "ubx_allgather_mc",
      [](int ranks, int myrank,
         std::uintptr_t devcomm, std::uintptr_t window, std::uintptr_t pool_ptr,
         std::uintptr_t ptr_in, std::size_t out_offset, std::size_t bytes,
         int default_sms, int smlimit) {
        ubx_allgather_mc(
            ranks, myrank,
            as_devcomm(devcomm), as_window(window), pool_ptr,
            ptr_in, out_offset, bytes, default_sms, smlimit,
            at::cuda::getCurrentCUDAStream());
      },
      py::arg("ranks"), py::arg("myrank"),
      py::arg("devcomm"), py::arg("window"), py::arg("pool_ptr"),
      py::arg("ptr_in"), py::arg("out_offset"), py::arg("bytes"),
      py::arg("default_sms") = 0, py::arg("smlimit") = 0);

  m.def(
      "ubx_allgather_uc",
      [](int ranks, int myrank,
         std::uintptr_t devcomm, std::uintptr_t window, std::uintptr_t pool_ptr,
         std::uintptr_t ptr_in, std::size_t out_offset, std::size_t bytes,
         int default_sms, int smlimit) {
        ubx_allgather_uc(
            ranks, myrank,
            as_devcomm(devcomm), as_window(window), pool_ptr,
            ptr_in, out_offset, bytes, default_sms, smlimit,
            at::cuda::getCurrentCUDAStream());
      },
      py::arg("ranks"), py::arg("myrank"),
      py::arg("devcomm"), py::arg("window"), py::arg("pool_ptr"),
      py::arg("ptr_in"), py::arg("out_offset"), py::arg("bytes"),
      py::arg("default_sms") = 0, py::arg("smlimit") = 0);

  m.def(
      "ubx_alltoall",
      [](int ranks, int myrank,
         std::uintptr_t devcomm, std::uintptr_t window, std::uintptr_t pool_ptr,
         std::uintptr_t ptr_in, std::size_t out_offset, std::size_t bytes,
         int default_sms, int smlimit, int nthreads) {
        ubx_alltoall(
            ranks, myrank,
            as_devcomm(devcomm), as_window(window), pool_ptr,
            ptr_in, out_offset, bytes,
            default_sms, smlimit, nthreads,
            at::cuda::getCurrentCUDAStream());
      },
      py::arg("ranks"), py::arg("myrank"),
      py::arg("devcomm"), py::arg("window"), py::arg("pool_ptr"),
      py::arg("ptr_in"), py::arg("out_offset"), py::arg("bytes"),
      py::arg("default_sms") = 0, py::arg("smlimit") = 0, py::arg("nthreads") = 0);

  m.def(
      "ubx_alltoallv",
      [](int ranks, int myrank,
         std::uintptr_t devcomm, std::uintptr_t window, std::uintptr_t pool_ptr,
         std::uintptr_t ptr_in,
         std::uintptr_t send_byte_offsets, std::uintptr_t send_byte_counts, std::uintptr_t dest_byte_offsets,
         int default_sms, int smlimit) {
        ubx_alltoallv(
            ranks, myrank,
            as_devcomm(devcomm), as_window(window), pool_ptr,
            ptr_in,
            send_byte_offsets, send_byte_counts, dest_byte_offsets,
            default_sms, smlimit, at::cuda::getCurrentCUDAStream());
      },
      py::arg("ranks"), py::arg("myrank"),
      py::arg("devcomm"), py::arg("window"), py::arg("pool_ptr"),
      py::arg("ptr_in"),
      py::arg("send_byte_offsets"), py::arg("send_byte_counts"), py::arg("dest_byte_offsets"),
      py::arg("default_sms") = 0, py::arg("smlimit") = 0);

  m.def(
      "ubx_alltoall_lamport",
      [](int ranks, int myrank,
         std::uintptr_t devcomm, std::uintptr_t window, std::uintptr_t pool_ptr,
         std::uintptr_t ptr_in, std::size_t out_offset, std::uintptr_t clear_ptr,
         std::size_t bytes, bool poisoned,
         int default_sms, int smlimit, int nthreads, bool skip_barrier) {
        ubx_alltoall_lamport(
            ranks, myrank,
            as_devcomm(devcomm), as_window(window), pool_ptr,
            ptr_in, out_offset, clear_ptr, bytes, poisoned,
            default_sms, smlimit, nthreads, skip_barrier,
            at::cuda::getCurrentCUDAStream());
      },
      py::arg("ranks"), py::arg("myrank"),
      py::arg("devcomm"), py::arg("window"), py::arg("pool_ptr"),
      py::arg("ptr_in"), py::arg("out_offset"), py::arg("clear_ptr"),
      py::arg("bytes"), py::arg("poisoned"),
      py::arg("default_sms") = 0, py::arg("smlimit") = 0,
      py::arg("nthreads") = 0, py::arg("skip_barrier"));

  m.def(
      "ubx_a2av_token_bf16_mxfp8",
      [](int ranks, int myrank, int ntokens, int blocks_per_token, int experts_per_rank,
         std::uintptr_t token_offsets, int64_t lineoffset_out, int64_t lineoffset_scales,
         std::uintptr_t devcomm, std::uintptr_t window, std::uintptr_t pool_ptr,
         std::uintptr_t ptr_in_bf16,
         int default_sms, int smlimit, int sync,
         int expert_start, int expert_count) {
        ubx_a2av_token_bf16_mxfp8(
            ranks, myrank, ntokens, blocks_per_token, experts_per_rank,
            token_offsets, lineoffset_out, lineoffset_scales,
            as_devcomm(devcomm), as_window(window), pool_ptr,
            ptr_in_bf16,
            default_sms, smlimit, sync,
            expert_start, expert_count,
            at::cuda::getCurrentCUDAStream());
      },
      py::arg("ranks"), py::arg("myrank"), py::arg("ntokens"),
      py::arg("blocks_per_token"), py::arg("experts_per_rank"),
      py::arg("token_offsets"), py::arg("lineoffset_out"), py::arg("lineoffset_scales"),
      py::arg("devcomm"), py::arg("window"), py::arg("pool_ptr"),
      py::arg("ptr_in_bf16"),
      py::arg("default_sms") = 0, py::arg("smlimit") = 0, py::arg("sync") = 1,
      py::arg("expert_start") = 0, py::arg("expert_count") = 0);

  m.def(
      "ubx_a2av_token_bf16_bf16",
      [](int ranks, int myrank, int ntokens, int blocks_per_token, int experts_per_rank,
         std::uintptr_t token_offsets, int64_t lineoffset_out,
         std::uintptr_t devcomm, std::uintptr_t window, std::uintptr_t pool_ptr,
         std::uintptr_t ptr_in_bf16,
         int default_sms, int smlimit, int sync,
         int expert_start, int expert_count) {
        ubx_a2av_token_bf16_bf16(
            ranks, myrank, ntokens, blocks_per_token, experts_per_rank,
            token_offsets, lineoffset_out,
            as_devcomm(devcomm), as_window(window), pool_ptr,
            ptr_in_bf16,
            default_sms, smlimit, sync,
            expert_start, expert_count,
            at::cuda::getCurrentCUDAStream());
      },
      py::arg("ranks"), py::arg("myrank"), py::arg("ntokens"),
      py::arg("blocks_per_token"), py::arg("experts_per_rank"),
      py::arg("token_offsets"), py::arg("lineoffset_out"),
      py::arg("devcomm"), py::arg("window"), py::arg("pool_ptr"),
      py::arg("ptr_in_bf16"),
      py::arg("default_sms") = 0, py::arg("smlimit") = 0, py::arg("sync") = 1,
      py::arg("expert_start") = 0, py::arg("expert_count") = 0);

  m.def(
      "ubx_a2av_token_bf16_bf16_topk",
      [](int ranks, int myrank, int ntokens, int blocks_per_token,
         int experts_per_rank, int topk_max,
         std::uintptr_t topk_expert, std::uintptr_t topk_slot,
         int64_t lineoffset_out,
         std::uintptr_t devcomm, std::uintptr_t window, std::uintptr_t pool_ptr,
         std::uintptr_t ptr_in_bf16,
         int default_sms, int smlimit, int sync) {
        ubx_a2av_token_bf16_bf16_topk(
            ranks, myrank, ntokens, blocks_per_token, experts_per_rank, topk_max,
            topk_expert, topk_slot, lineoffset_out,
            as_devcomm(devcomm), as_window(window), pool_ptr,
            ptr_in_bf16,
            default_sms, smlimit, sync,
            at::cuda::getCurrentCUDAStream());
      },
      py::arg("ranks"), py::arg("myrank"), py::arg("ntokens"),
      py::arg("blocks_per_token"), py::arg("experts_per_rank"),
      py::arg("topk_max"),
      py::arg("topk_expert"), py::arg("topk_slot"),
      py::arg("lineoffset_out"),
      py::arg("devcomm"), py::arg("window"), py::arg("pool_ptr"),
      py::arg("ptr_in_bf16"),
      py::arg("default_sms") = 0, py::arg("smlimit") = 0, py::arg("sync") = 1);

  m.def(
      "ubx_a2av_wait",
      [](int ranks,
         std::uintptr_t devcomm, std::uintptr_t window, std::uintptr_t pool_ptr) {
        ubx_a2av_wait(
            ranks,
            as_devcomm(devcomm), as_window(window), pool_ptr,
            at::cuda::getCurrentCUDAStream());
      },
      py::arg("ranks"),
      py::arg("devcomm"), py::arg("window"), py::arg("pool_ptr"));

  m.def(
      "ubx_combine_bf16_bf16",
      [](int ranks, int myrank, int local_ntokens, int n_recv,
         int blocks_per_token, int experts_per_rank, int max_tokens_per_rank,
         std::uintptr_t token_offsets, std::uintptr_t gate_weights,
         int64_t lineoffset_temp,
         std::uintptr_t devcomm, std::uintptr_t window, std::uintptr_t pool_ptr,
         std::uintptr_t ptr_in_bf16, std::uintptr_t ptr_out_bf16,
         int default_sms, int smlimit, int sync) {
        ubx_combine_bf16_bf16(
            ranks, myrank, local_ntokens, n_recv,
            blocks_per_token, experts_per_rank, max_tokens_per_rank,
            token_offsets, gate_weights, lineoffset_temp,
            as_devcomm(devcomm), as_window(window), pool_ptr,
            ptr_in_bf16, ptr_out_bf16,
            default_sms, smlimit, sync,
            at::cuda::getCurrentCUDAStream());
      },
      py::arg("ranks"), py::arg("myrank"), py::arg("local_ntokens"),
      py::arg("n_recv"), py::arg("blocks_per_token"),
      py::arg("experts_per_rank"), py::arg("max_tokens_per_rank"),
      py::arg("token_offsets"), py::arg("gate_weights") = 0,
      py::arg("lineoffset_temp"),
      py::arg("devcomm"), py::arg("window"), py::arg("pool_ptr"),
      py::arg("ptr_in_bf16"), py::arg("ptr_out_bf16"),
      py::arg("default_sms") = 0, py::arg("smlimit") = 0, py::arg("sync") = 1);

  m.def(
      "ubx_combine_v2_phase1_bf16",
      [](int n_recv, int blocks_per_token,
         std::uintptr_t pool_ptr, int64_t lineoffset_temp,
         std::uintptr_t ptr_in_bf16,
         int default_sms, int smlimit) {
        ubx_combine_v2_phase1_bf16(
            n_recv, blocks_per_token, pool_ptr, lineoffset_temp,
            ptr_in_bf16, default_sms, smlimit,
            at::cuda::getCurrentCUDAStream());
      },
      py::arg("n_recv"), py::arg("blocks_per_token"),
      py::arg("pool_ptr"), py::arg("lineoffset_temp"),
      py::arg("ptr_in_bf16"),
      py::arg("default_sms") = 0, py::arg("smlimit") = 0);

  m.def(
      "ubx_combine_v2_phase2_bf16",
      [](int ranks, int reduce_id, int local_ntokens, int blocks_per_token,
         int experts_per_rank,
         std::uintptr_t token_offsets, std::uintptr_t gate_weights,
         int64_t lineoffset_temp,
         std::uintptr_t devcomm, std::uintptr_t window, std::uintptr_t pool_ptr,
         std::uintptr_t ptr_out_bf16,
         int default_sms, int smlimit) {
        ubx_combine_v2_phase2_bf16(
            ranks, reduce_id, local_ntokens, blocks_per_token,
            experts_per_rank, token_offsets, gate_weights,
            lineoffset_temp,
            as_devcomm(devcomm), as_window(window), pool_ptr,
            ptr_out_bf16, default_sms, smlimit,
            at::cuda::getCurrentCUDAStream());
      },
      py::arg("ranks"), py::arg("reduce_id"), py::arg("local_ntokens"),
      py::arg("blocks_per_token"), py::arg("experts_per_rank"),
      py::arg("token_offsets"), py::arg("gate_weights"),
      py::arg("lineoffset_temp"),
      py::arg("devcomm"), py::arg("window"), py::arg("pool_ptr"),
      py::arg("ptr_out_bf16"),
      py::arg("default_sms") = 0, py::arg("smlimit") = 0);

  m.def(
      "ubx_barrier",
      [](int ranks, int myrank,
         std::uintptr_t devcomm, std::uintptr_t window, std::uintptr_t pool_ptr,
         int default_sms, int smlimit) {
        ubx_barrier(ranks, myrank,
                    as_devcomm(devcomm), as_window(window), pool_ptr,
                    default_sms, smlimit,
                    at::cuda::getCurrentCUDAStream());
      },
      py::arg("ranks"), py::arg("myrank"),
      py::arg("devcomm"), py::arg("window"), py::arg("pool_ptr"),
      py::arg("default_sms") = 0, py::arg("smlimit") = 0);

  m.def(
      "ubx_peer_ptr_dump",
      [](int ranks,
         std::uintptr_t devcomm, std::uintptr_t window,
         std::uintptr_t out_ptrs) {
        ubx_peer_ptr_dump(ranks,
                          as_devcomm(devcomm), as_window(window),
                          out_ptrs,
                          at::cuda::getCurrentCUDAStream());
      },
      py::arg("ranks"),
      py::arg("devcomm"), py::arg("window"),
      py::arg("out_ptrs"));

  m.def(
      "ubx_peer_atomic_test",
      [](int ranks, int myrank, int test_id,
         std::uintptr_t devcomm, std::uintptr_t window,
         std::uintptr_t pool_ptr) {
        ubx_peer_atomic_test(ranks, myrank, test_id,
                             as_devcomm(devcomm), as_window(window),
                             pool_ptr,
                             at::cuda::getCurrentCUDAStream());
      },
      py::arg("ranks"), py::arg("myrank"), py::arg("test_id"),
      py::arg("devcomm"), py::arg("window"), py::arg("pool_ptr"));

  m.def(
      "ubx_combine_push3_phase1_write",
      [](int ranks, int n_recv, int blocks_per_token, int topk_max,
         std::uintptr_t inverse_map, int max_tokens_per_rank,
         int64_t lineoffset_dest,
         std::uintptr_t devcomm, std::uintptr_t window,
         std::uintptr_t ptr_in_bf16,
         int default_sms, int smlimit) {
        ubx_combine_push3_phase1_write(
            ranks, n_recv, blocks_per_token, topk_max,
            inverse_map, max_tokens_per_rank, lineoffset_dest,
            as_devcomm(devcomm), as_window(window),
            ptr_in_bf16,
            default_sms, smlimit,
            at::cuda::getCurrentCUDAStream());
      },
      py::arg("ranks"), py::arg("n_recv"), py::arg("blocks_per_token"),
      py::arg("topk_max"),
      py::arg("inverse_map"), py::arg("max_tokens_per_rank"),
      py::arg("lineoffset_dest"),
      py::arg("devcomm"), py::arg("window"), py::arg("ptr_in_bf16"),
      py::arg("default_sms") = 0, py::arg("smlimit") = 0);

  m.def(
      "ubx_combine_push3_phase2_signal",
      [](int ranks,
         std::uintptr_t devcomm, std::uintptr_t window,
         std::uintptr_t pool_ptr) {
        ubx_combine_push3_phase2_signal(
            ranks,
            as_devcomm(devcomm), as_window(window), pool_ptr,
            at::cuda::getCurrentCUDAStream());
      },
      py::arg("ranks"),
      py::arg("devcomm"), py::arg("window"), py::arg("pool_ptr"));

  m.def(
      "ubx_combine_push3_phase3_sum",
      [](int local_ntokens, int blocks_per_token, int topk_max,
         int total_experts,
         std::uintptr_t topk_idx, std::uintptr_t gate_weights,
         int64_t lineoffset_dest, std::uintptr_t pool_ptr,
         std::uintptr_t ptr_out_bf16,
         int default_sms, int smlimit) {
        ubx_combine_push3_phase3_sum(
            local_ntokens, blocks_per_token, topk_max, total_experts,
            topk_idx, gate_weights, lineoffset_dest, pool_ptr,
            ptr_out_bf16,
            default_sms, smlimit,
            at::cuda::getCurrentCUDAStream());
      },
      py::arg("local_ntokens"), py::arg("blocks_per_token"),
      py::arg("topk_max"), py::arg("total_experts"),
      py::arg("topk_idx"), py::arg("gate_weights"),
      py::arg("lineoffset_dest"), py::arg("pool_ptr"),
      py::arg("ptr_out_bf16"),
      py::arg("default_sms") = 0, py::arg("smlimit") = 0);

  m.def(
      "ubx_combine_mxfp8_bf16",
      [](int ranks, int myrank, int local_ntokens, int n_recv,
         int blocks_per_token, int experts_per_rank, int max_tokens_per_rank,
         std::uintptr_t token_offsets, std::uintptr_t gate_weights,
         int64_t lineoffset_temp, int64_t lineoffset_scales,
         std::uintptr_t devcomm, std::uintptr_t window, std::uintptr_t pool_ptr,
         std::uintptr_t ptr_in_bf16, std::uintptr_t ptr_out_bf16,
         int default_sms, int smlimit, int sync) {
        ubx_combine_mxfp8_bf16(
            ranks, myrank, local_ntokens, n_recv,
            blocks_per_token, experts_per_rank, max_tokens_per_rank,
            token_offsets, gate_weights,
            lineoffset_temp, lineoffset_scales,
            as_devcomm(devcomm), as_window(window), pool_ptr,
            ptr_in_bf16, ptr_out_bf16,
            default_sms, smlimit, sync,
            at::cuda::getCurrentCUDAStream());
      },
      py::arg("ranks"), py::arg("myrank"), py::arg("local_ntokens"),
      py::arg("n_recv"), py::arg("blocks_per_token"),
      py::arg("experts_per_rank"), py::arg("max_tokens_per_rank"),
      py::arg("token_offsets"), py::arg("gate_weights") = 0,
      py::arg("lineoffset_temp"), py::arg("lineoffset_scales"),
      py::arg("devcomm"), py::arg("window"), py::arg("pool_ptr"),
      py::arg("ptr_in_bf16"), py::arg("ptr_out_bf16"),
      py::arg("default_sms") = 0, py::arg("smlimit") = 0, py::arg("sync") = 1);

  m.def(
      "ubx_combine_bf16_bf16_push",
      [](int ranks, int myrank, int local_ntokens, int n_recv,
         int blocks_per_token, int experts_per_rank, int max_tokens_per_rank,
         int topk_max,
         std::uintptr_t inverse_map, std::uintptr_t topk_idx,
         std::uintptr_t gate_weights,
         int64_t lineoffset_dest,
         std::uintptr_t devcomm, std::uintptr_t window, std::uintptr_t pool_ptr,
         std::uintptr_t ptr_in_bf16, std::uintptr_t ptr_out_bf16,
         int default_sms, int smlimit) {
        ubx_combine_bf16_bf16_push(
            ranks, myrank, local_ntokens, n_recv,
            blocks_per_token, experts_per_rank, max_tokens_per_rank,
            topk_max,
            inverse_map, topk_idx, gate_weights,
            lineoffset_dest,
            as_devcomm(devcomm), as_window(window), pool_ptr,
            ptr_in_bf16, ptr_out_bf16,
            default_sms, smlimit,
            at::cuda::getCurrentCUDAStream());
      },
      py::arg("ranks"), py::arg("myrank"),
      py::arg("local_ntokens"), py::arg("n_recv"),
      py::arg("blocks_per_token"), py::arg("experts_per_rank"),
      py::arg("max_tokens_per_rank"), py::arg("topk_max"),
      py::arg("inverse_map"), py::arg("topk_idx"),
      py::arg("gate_weights") = 0,
      py::arg("lineoffset_dest"),
      py::arg("devcomm"), py::arg("window"), py::arg("pool_ptr"),
      py::arg("ptr_in_bf16"), py::arg("ptr_out_bf16"),
      py::arg("default_sms") = 0, py::arg("smlimit") = 0);

  m.def(
      "ubx_combine_bf16_bf16_lamport_push",
      [](int ranks, int myrank, int local_ntokens, int n_recv,
         int blocks_per_token, int experts_per_rank, int max_tokens_per_rank,
         int topk_max,
         std::uintptr_t inverse_map, std::uintptr_t topk_idx,
         std::uintptr_t gate_weights,
         int64_t lineoffset_dest, std::uintptr_t clear_ptr,
         bool poisoned, bool skip_warmup_barrier,
         std::uintptr_t devcomm, std::uintptr_t window, std::uintptr_t pool_ptr,
         std::uintptr_t ptr_in_bf16, std::uintptr_t ptr_out_bf16,
         int default_sms, int smlimit) {
        ubx_combine_bf16_bf16_lamport_push(
            ranks, myrank, local_ntokens, n_recv,
            blocks_per_token, experts_per_rank, max_tokens_per_rank,
            topk_max,
            inverse_map, topk_idx, gate_weights,
            lineoffset_dest, clear_ptr,
            poisoned, skip_warmup_barrier,
            as_devcomm(devcomm), as_window(window), pool_ptr,
            ptr_in_bf16, ptr_out_bf16,
            default_sms, smlimit,
            at::cuda::getCurrentCUDAStream());
      },
      py::arg("ranks"), py::arg("myrank"),
      py::arg("local_ntokens"), py::arg("n_recv"),
      py::arg("blocks_per_token"), py::arg("experts_per_rank"),
      py::arg("max_tokens_per_rank"), py::arg("topk_max"),
      py::arg("inverse_map"), py::arg("topk_idx"),
      py::arg("gate_weights") = 0,
      py::arg("lineoffset_dest"), py::arg("clear_ptr") = 0,
      py::arg("poisoned"), py::arg("skip_warmup_barrier"),
      py::arg("devcomm"), py::arg("window"), py::arg("pool_ptr"),
      py::arg("ptr_in_bf16"), py::arg("ptr_out_bf16"),
      py::arg("default_sms") = 0, py::arg("smlimit") = 0);

  m.def(
      "ubx_combine_wait_bf16",
      [](int ranks, int local_ntokens, int blocks_per_token,
         int experts_per_rank,
         std::uintptr_t token_offsets, std::uintptr_t gate_weights,
         int64_t lineoffset_temp,
         std::uintptr_t devcomm, std::uintptr_t window, std::uintptr_t pool_ptr,
         std::uintptr_t ptr_out_bf16,
         int default_sms, int smlimit) {
        ubx_combine_wait_bf16(
            ranks, local_ntokens, blocks_per_token, experts_per_rank,
            token_offsets, gate_weights, lineoffset_temp,
            as_devcomm(devcomm), as_window(window), pool_ptr,
            ptr_out_bf16,
            default_sms, smlimit,
            at::cuda::getCurrentCUDAStream());
      },
      py::arg("ranks"), py::arg("local_ntokens"), py::arg("blocks_per_token"),
      py::arg("experts_per_rank"),
      py::arg("token_offsets"), py::arg("gate_weights") = 0,
      py::arg("lineoffset_temp"),
      py::arg("devcomm"), py::arg("window"), py::arg("pool_ptr"),
      py::arg("ptr_out_bf16"),
      py::arg("default_sms") = 0, py::arg("smlimit") = 0);

  m.def(
      "ubx_combine_wait_mxfp8",
      [](int ranks, int local_ntokens, int blocks_per_token,
         int experts_per_rank,
         std::uintptr_t token_offsets, std::uintptr_t gate_weights,
         int64_t lineoffset_temp, int64_t lineoffset_scales,
         std::uintptr_t devcomm, std::uintptr_t window, std::uintptr_t pool_ptr,
         std::uintptr_t ptr_out_bf16,
         int default_sms, int smlimit) {
        ubx_combine_wait_mxfp8(
            ranks, local_ntokens, blocks_per_token, experts_per_rank,
            token_offsets, gate_weights,
            lineoffset_temp, lineoffset_scales,
            as_devcomm(devcomm), as_window(window), pool_ptr,
            ptr_out_bf16,
            default_sms, smlimit,
            at::cuda::getCurrentCUDAStream());
      },
      py::arg("ranks"), py::arg("local_ntokens"), py::arg("blocks_per_token"),
      py::arg("experts_per_rank"),
      py::arg("token_offsets"), py::arg("gate_weights") = 0,
      py::arg("lineoffset_temp"), py::arg("lineoffset_scales"),
      py::arg("devcomm"), py::arg("window"), py::arg("pool_ptr"),
      py::arg("ptr_out_bf16"),
      py::arg("default_sms") = 0, py::arg("smlimit") = 0);

  m.def(
      "ubx_set_timeout",
      [](unsigned long long clocks) { ubx_set_timeout(clocks); },
      py::arg("clocks"));

  m.def(
      "ubx_a2av_token_bf16_mxfp8_persistent",
      [](int ranks, int myrank, int ntokens, int blocks_per_token, int experts_per_rank,
         std::uintptr_t token_offsets, int64_t lineoffset_out, int64_t lineoffset_scales,
         std::uintptr_t devcomm, std::uintptr_t window, std::uintptr_t pool_ptr,
         std::uintptr_t ptr_in_bf16,
         int default_sms, int smlimit,
         int nchunks, int nexperts_per_chunk) {
        ubx_a2av_token_bf16_mxfp8_persistent(
            ranks, myrank, ntokens, blocks_per_token, experts_per_rank,
            token_offsets, lineoffset_out, lineoffset_scales,
            as_devcomm(devcomm), as_window(window), pool_ptr,
            ptr_in_bf16,
            default_sms, smlimit,
            nchunks, nexperts_per_chunk,
            at::cuda::getCurrentCUDAStream());
      },
      py::arg("ranks"), py::arg("myrank"), py::arg("ntokens"),
      py::arg("blocks_per_token"), py::arg("experts_per_rank"),
      py::arg("token_offsets"), py::arg("lineoffset_out"), py::arg("lineoffset_scales"),
      py::arg("devcomm"), py::arg("window"), py::arg("pool_ptr"),
      py::arg("ptr_in_bf16"),
      py::arg("default_sms") = 0, py::arg("smlimit") = 0,
      py::arg("nchunks"), py::arg("nexperts_per_chunk"));
}
