#ifndef PACE_KERNELS_INCLUDE_AG_ZERO_SM_CUH
#define PACE_KERNELS_INCLUDE_AG_ZERO_SM_CUH

#include "device/common.cuh"

#define AGCOMM_INPUT_DTYPE_MAPPING_NONE 0
#define AGCOMM_INPUT_DTYPE_MAPPING_FP32_TO_BF16 1

namespace pace {

void allgather_zero_sm_func(uint64_t* args_cpu, int num_tensors, ncclWindow_t gin_win, void *gin_buffer, void *signal_buffer, void **p2p_ptrs, int unroll, int nvl_ring, int rdma_ring, size_t& total_send_slots, int capture_round_n, int rank, int num_local_ranks, int num_ranks, CUstreamBatchMemOpParams *mparam, ncclDevComm dev_comm, cudaStream_t stream, cudaStream_t copy_stream, bool use_ring = false, bool use_graph = false, bool force_graph_capture = false, int min_graph_nodes = 32);

// Clear AllGather CUDA graph cache (call when AGComm is destroyed)
void clear_ag_zero_sm_graph_cache();

}

#endif
