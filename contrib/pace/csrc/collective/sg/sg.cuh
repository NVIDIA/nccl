#ifndef PACE_KERNELS_INCLUDE_SG_CUH
#define PACE_KERNELS_INCLUDE_SG_CUH

#include "device/common.cuh"
#include "collective/sg/tensor_layout.hpp"

namespace pace {

void scattergatherfunc(std::vector<TensorLayout> layouts, std::vector<TensorLayout> rlayouts, ncclWindow_t gin_win, void *gin_buffer, void *signal_buffer, void **p2p_ptrs, int unroll, int nvl_ring, int rdma_ring, size_t& total_send_slots, int capture_round_n, int rank, int num_local_ranks, int num_ranks, CUstreamBatchMemOpParams *mparam, ncclDevComm dev_comm, int num_sms, cudaStream_t stream, cudaStream_t copy_stream, const std::vector<uint64_t>& src_row_strides);

void scattergather_kernel_func(
    uint64_t *args_host,
    uint64_t *args_dev,  // interleaved [in_ptr, in_X, in_Y, in_Z, out_ptr, out_X, out_Y, out_Z] * num_tensors
    int num_tensors,
    ncclWindow_t gin_win,
    void *gin_buffer,
    void *signal_buffer,  // p2p_ptrs is already in signal_buffer
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
    int scatter_dim
);

}

#endif
