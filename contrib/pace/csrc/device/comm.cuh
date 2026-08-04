#ifndef PACE_DEVICE_COMM_CUH
#define PACE_DEVICE_COMM_CUH

// Cross-rank device primitives: low-latency (LL) barrier / signal-wait
// polling macros, GIN signal polls, and peer-pointer helpers. Built on
// top of the single-rank primitives in device/mem.cuh.

#include "device/mem.cuh"

#define INTRA_TAG(n) ((n & 0x3fffffff) | 0x40000000)
#define LL_ENCAP_BASE(round, value) ((static_cast<uint64_t>(round) << 32) | (static_cast<uint64_t>(value) & 0xffffffff))
#define LL_READ_BASE_NO_V(round, ptr, src_rank) {\
    auto l64 = reinterpret_cast<uint64_t*>(ptr);\
    auto start_t = clock64();\
    while (static_cast<int>(ld_acquire_sys_global(l64) >> 32) != round) {\
        if (clock64() - start_t > (NUM_TIMEOUT_CYCLES)) {\
            printf("[rank %d]: fail to read LL value from rank %d in time, %s:%d \n", rank, src_rank, __FILE__, __LINE__);\
            __trap();\
        }\
    }\
}

#define LOOP_READ_GIN_SIGNAL_64GE(exp_v, gin_sig_exp, src_rank, qp_id) {\
    auto start_t = clock64();\
    uint64_t value;\
    while ((value = (gin_sig_exp)) < (exp_v)) {\
        if (clock64() - start_t > (NUM_TIMEOUT_CYCLES)) {\
            printf("[rank %d]: [round 0x%08x] [qp %d] fail to read gin signal value from rank %d in time, %lu !>= %lu, %s:%d \n", rank, round_n, qp_id,(src_rank), value, (exp_v), __FILE__, __LINE__);\
            __trap();\
        }\
    }\
}

#define LL_READ_BASE(round, ptr, value, src_rank) {\
    auto l64 = reinterpret_cast<uint64_t*>(ptr);\
    auto start_t = clock64();\
    int64_t r;\
    while (static_cast<int>((r = ld_acquire_sys_global(l64)) >> 32) != round) {\
        if (clock64() - start_t > (NUM_TIMEOUT_CYCLES)) {\
            printf("[rank %d]: fail to read LL value from rank %d in time, %s:%d \n", rank, src_rank, __FILE__, __LINE__);\
            __trap();\
        }\
    }\
    value = static_cast<int>(r);\
}

#define LL_READ_64GE(expect, ptr, src_rank) {\
    auto l64 = reinterpret_cast<uint64_t*>(ptr);\
    auto start_t = clock64();\
    uint64_t value;\
    while ((value = ld_acquire_sys_global(l64)) < expect) {\
        if (clock64() - start_t > (NUM_TIMEOUT_CYCLES)) {\
            printf("[rank %d sm %d]: fail to read LL value from rank %d in time, %lu !>= %lu %s:%d \n", rank, (int)blockIdx.x, src_rank, value, expect, __FILE__, __LINE__);\
            LL_TIMEOUT_DUMP();\
            __trap();\
        }\
    }\
}

// Hook invoked inside LL_READ_64GE_STEP / LL_READ_64GE_STEP2 just before
// __trap() so callers can dump extra context at a timeout. Default is a no-op;
// kernels that want richer diagnostics #undef this and define their own (e.g.
// rs.cu / ag_ring.cu under PACE_TIMEOUT_DEBUG).
#ifndef LL_TIMEOUT_DUMP
#define LL_TIMEOUT_DUMP() ((void)0)
#endif

// Variant of LL_READ_64GE that also prints the current ring step. `step_desc`
// is a string-literal label (e.g. "_w", "_n._l") describing what `step` means
// in the calling kernel, so the same macro can be reused across ring kernels
// with different loop structures. Also prints `cuda_device_id` so railed
// comms (where local_rank ≠ device id) can be debugged from the log alone.
// Caller must have `cuda_device_id` in lexical scope.
#define LL_READ_64GE_STEP(expect, ptr, src_rank, step_desc, step) {\
    auto l64 = reinterpret_cast<uint64_t*>(ptr);\
    auto start_t = clock64();\
    uint64_t value;\
    while ((value = ld_acquire_sys_global(l64)) < expect) {\
        if (clock64() - start_t > (NUM_TIMEOUT_CYCLES)) {\
            printf("[rank %d dev %d sm %d " step_desc "=%d]: fail to read LL value from rank %d in time, %lu !>= %lu %s:%d \n", rank, cuda_device_id, (int)blockIdx.x, (int)(step), src_rank, value, expect, __FILE__, __LINE__);\
            LL_TIMEOUT_DUMP();\
            __trap();\
        }\
    }\
}

// Two-step variant for the mixed inter+intra ring (rs mixed branch).
#define LL_READ_64GE_STEP2(expect, ptr, src_rank, step1_desc, step1, step2_desc, step2) {\
    auto l64 = reinterpret_cast<uint64_t*>(ptr);\
    auto start_t = clock64();\
    uint64_t value;\
    while ((value = ld_acquire_sys_global(l64)) < expect) {\
        if (clock64() - start_t > (NUM_TIMEOUT_CYCLES)) {\
            printf("[rank %d dev %d sm %d " step1_desc "=%d " step2_desc "=%d]: fail to read LL value from rank %d in time, %lu !>= %lu %s:%d \n", rank, cuda_device_id, (int)blockIdx.x, (int)(step1), (int)(step2), src_rank, value, expect, __FILE__, __LINE__);\
            LL_TIMEOUT_DUMP();\
            __trap();\
        }\
    }\
}

#include "util/error.hpp"
#include "nccl.h"
#include "nccl_device.h"
#include "nccl_device/coop.h"
#include "nccl_device/core.h"
#include "nccl_device/gin.h"

#ifdef PACE_TIMEOUT_DEBUG
// One-line bitmap dump of which NVL siblings and rail-aligned RDMA peers have
// entered the current round, called from LL_TIMEOUT_DUMP() when a wait times
// out. Each peer increments my arrival sig slot once on kernel entry, so
// `arrival_sig[peer] >= round_n + 1` means the peer is "online" for this
// round. A 0 bit means we have not yet seen the peer enter — usually because
// their cooperative launch (e.g. DeepEP) is sitting in the stream queue.
//
// We only consider:
//   * NVL siblings on my node (same node_id, any local_rank). Bit i of
//     nvl_mask = 1 iff local rank i is online. My own bit is always 1.
//   * Rail-aligned RDMA peers (same local_rank, different node). Bit n of
//     rail_mask = 1 iff the rail peer at node n is online. My node's bit is
//     always 1.
//
// Non-rail RDMA peers are skipped because the default
// NCCL_GIN_CONNECTION_RAIL only routes signals along the rail, so we never
// could have signaled them (they'd permanently show as offline and clutter
// the log). Non-rail peers just won't be reflected in the bitmap above.
//
// Each block uses its own blockIdx.x as the GIN context id on BOTH the send
// and the read side. Sender block B signals via context B (= QP B), so the
// signal lands in the receiver's QP-B signal table; dumper block B reads from
// its own QP-B table and sees a consistent view of who sent through QP B.
// Sharing one context across blocks (pinning to QP 0) ran into latent
// correctness issues with NCCL GIN; per-block contexts keep each block's
// send/read pair self-consistent and decouple blocks from each other.
__device__ __forceinline__ void dump_arrival_status(
    ncclDevComm devComm, int num_local_ranks, int num_nodes,
    int local_rank, int node_id,
    int arrival_sig_base, int round_n, int my_rank, int sm_id,
    int cuda_device_id)
{
    if (num_nodes <= 1) {
        return;  // single-node has no inter-rank arrival debug
    }
    ncclGin g{devComm, sm_id};
    const uint64_t expected = static_cast<uint64_t>(round_n) + 1ULL;
    uint64_t nvl_mask  = (1ULL << local_rank);  // self
    uint64_t rail_mask = (1ULL << node_id);     // self's node
    for (int p = 0; p < num_local_ranks && p < 64; ++p) {
        if (p == local_rank) continue;
        const int peer = node_id * num_local_ranks + p;
        if (g.readSignal(arrival_sig_base + peer) >= expected) {
            nvl_mask |= (1ULL << p);
        }
    }
    for (int n = 0; n < num_nodes && n < 64; ++n) {
        if (n == node_id) continue;
        const int peer = n * num_local_ranks + local_rank;
        if (g.readSignal(arrival_sig_base + peer) >= expected) {
            rail_mask |= (1ULL << n);
        }
    }
    const int rail_hex = (num_nodes + 3) / 4;  // hex digits needed to fit N bits
    printf("[rank %d dev %d sm %d] arrival round=%d nvl=0x%02x/P=%d rail=0x%0*lx/N=%d "
           "(bit i set = peer at i-th local_rank / i-th node has entered)\n",
           my_rank, cuda_device_id, sm_id, round_n,
           (unsigned)(nvl_mask & 0xFFu), num_local_ranks,
           rail_hex, (unsigned long)rail_mask, num_nodes);
}
#endif

#endif  // PACE_DEVICE_COMM_CUH
