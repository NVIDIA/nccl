#ifndef PACE_KERNELS_INCLUDE_AG_RING_CUH
#define PACE_KERNELS_INCLUDE_AG_RING_CUH

#include "device/common.cuh"
#include "collective/ag/ag_zero_sm.cuh"  // for AGCOMM_INPUT_DTYPE_MAPPING_{NONE,FP32_TO_BF16}

namespace pace {

// ==========================================================================
// Ring AllGather (NCCL-style multi-channel, rail-aligned).
//
// Topology:
//   - P = num_local_ranks (GPUs per server), N = num_nodes, R = P*N = num_ranks.
//   - There are `num_sms` rings total, with ring r starting at GPU (r % P) on
//     server 0. Each ring has R hops total. On every server the ring enters at
//     local idx p_enter, walks ascending P-1 hops (local idx p_enter, p_enter+1,
//     ..., p_enter-1 mod P), then takes 1 RDMA hop to the same local idx on
//     the next server. After P*N hops the ring closes.
//   - Consequently every RDMA hop is rail-aligned (same local index across
//     servers), and different rings use different rails for their RDMA edges
//     (no two rings share an RDMA path).
//
// Pipelining:
//   - Each SM maintains two ring-slot buffers per direction (nvl_ring,
//     rdma_ring are both 2 by default, configurable). This realizes a 2-slot
//     ping-pong per ring: while slot A traverses the ring, slot B is being
//     staged / forwarded.
//
// Data partitioning:
//   - Input tensors are concatenated into a single virtual int4 stream of
//     total size B (sum over tensors of ag_align16(byte_size)). Each ring
//     (= each SM) owns a contiguous slice of B with ceil_div(B, num_sms)
//     bytes (last ring trimmed).
//
// Per-slot algorithm (NCCL runRing-style):
//   step 0           : copy input -> my output[my_rank] + push to next hop buf
//   step 1..R-2      : wait prev -> copy to output[src_rank] + forward to next
//   step R-1         : wait prev -> copy to output[src_rank]
// Intra-server hops use NVLink peer-to-peer writes directly into next hop's
// recv buffer. Inter-server hops use ncclGin.put with SignalInc.
//
// Signals (per-SM, counters that monotonically increase across calls):
//   nvl_data : NVL u64, producer is my nvl_prev, consumer is me
//   nvl_ack  : NVL u64, producer is me, consumer is nvl_prev (reuse)
//   rdma_data : GIN signal incremented by remote put (from rdma_prev)
//   rdma_ack  : GIN signal incremented by me to acknowledge rdma_prev so they
//               can reuse the buffer
//
// Args layout (same as TDM AG):
//   args[4*i + 0] = input ptr
//   args[4*i + 1] = byte size (of ONE rank's input tensor i)
//   args[4*i + 2] = output ptr (R contiguous rank regions of byte size)
//   args[4*i + 3] = reserved
// ==========================================================================

void allgather_ring_func(
    uint64_t* args_cpu,
    uint64_t* args_gpu,
    int num_tensors,
    ncclWindow_t gin_win,
    void* gin_buffer,
    void* signal_buffer,
    int unroll,
    int nvl_ring,
    int rdma_ring,
    size_t& total_send_slots,
    int capture_round_n,
    int input_dtype_mapping,
    int rank,
    int num_local_ranks,
    int num_ranks,
    ncclDevComm dev_comm,
    int num_sms,
    cudaStream_t stream,
    int arrival_sig_base = 0, int cuda_device_id = 0);

// 0SM Multi-node Ring AllGather: handled by `allgather_zero_sm_func` with
// `use_ring=true`. The cord kernel's send/forward shape switches from
// broadcast to a per-rail ring; buffers and signals stay non-ring.

// Per-SM GIN signal count (2: rdma_data, rdma_ack).
constexpr int AG_RING_GIN_SIGS_PER_ENGINE = 2;
inline __host__ __device__ uint32_t ag_ring_rdma_data_sig(uint32_t sm_id) {
    return sm_id * AG_RING_GIN_SIGS_PER_ENGINE + 0;
}
inline __host__ __device__ uint32_t ag_ring_rdma_ack_sig(uint32_t sm_id) {
    return sm_id * AG_RING_GIN_SIGS_PER_ENGINE + 1;
}

}  // namespace pace

#endif
