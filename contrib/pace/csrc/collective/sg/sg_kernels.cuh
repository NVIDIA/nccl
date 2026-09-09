#ifndef PACE_SG_KERNELS_CUH
#define PACE_SG_KERNELS_CUH

// Kernel template + device helpers extracted from sg.cu so that
// scripts/gen_sg_inst.py can emit per-instantiation .cu files that include
// this header (mirrors ep_kernels.cuh / gen_ep_inst.py pattern). The host
// dispatchers (scattergatherfunc, scattergather_kernel_func,
// scattergathercordkernel, sg_mem_layout, sg_device_mem_layout) remain in
// sg.cu.

#include "util/error.hpp"
#include "device/comm.cuh"
#include "util/math.hpp"
#include "device/mem.cuh"
#include "device/configs.cuh"
#include "device/launch.cuh"
#include "collective/sg/tensor_layout.hpp"
#include "collective/sg/sg.cuh"
#include <cstdint>
#include <cstdio>
#include <algorithm>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <nccl_device/coop.h>
#include <nccl_device/core.h>
#include <nccl_device/gin.h>
#include <type_traits>
#include <cooperative_groups.h>

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
#define SG_F8_ARCH_SUPPORTED 1
#else
#define SG_F8_ARCH_SUPPORTED 0
#endif

// SG kernel constants
#define SG_MAX_TENSORS 64

namespace pace {

// Byte offset into a strided-S source for s0g1 (scatter_dim == 0): rank
// dst_rank's chunk is (chunk_bytes / row_bytes) strided S-rows of row_bytes
// each, pitched by row_stride. `off` is the byte offset within that chunk
// (its X==1, so no x term). Standard head dims give a power-of-two row_bytes
// (e.g. D=128 bf16 -> row_bytes=2048=2^11), so take a shift/mask fast path
// instead of three per-element 64-bit divisions.
__device__ __forceinline__ uint64_t sg_strided_s0g1_offset(
    const uint64_t dst_rank, const uint64_t chunk_bytes, const uint64_t row_bytes,
    const uint64_t off, const uint64_t row_stride) {
    const uint64_t rb_mask = row_bytes - 1;
    if ((row_bytes & rb_mask) == 0) {
        // chunk_bytes is always an exact multiple of row_bytes (== rows_per_rank * row_bytes).
        const uint64_t rb_log2 = static_cast<uint64_t>(__ffsll(row_bytes) - 1);
        const uint64_t row = dst_rank * (chunk_bytes >> rb_log2) + (off >> rb_log2);
        return row * row_stride + (off & rb_mask);
    }
    const uint64_t row = dst_rank * (chunk_bytes / row_bytes) + off / row_bytes;
    return row * row_stride + off % row_bytes;
}

/**
 * unified_ld for scatter-gather: Load data from multiple tensors for a specific destination rank
 *
 * TensorLayout semantics:
 *   Input tensor is viewed as [X, Y=num_ranks, Z] where:
 *   - X = product of dimensions before pin_dim (scatter_dim)
 *   - Y = num_ranks (the scatter dimension)
 *   - Z = product of dimensions after pin_dim * elem_size (in bytes)
 *
 * For dst_rank j, we read data at position [x, j, z] for all valid x and z.
 * The data is stored in row-major order: base_ptr + x * Y * Z + j * Z + z
 *
 * @param nargs        Number of tensors
 * @param cached_args  Shared memory cached tensor metadata: [in_ptr, in_X, in_Y, in_Z, out_ptr, out_X, out_Y, out_Z] * nargs
 *                     Input at offset i*8: ptr, X, Y, Z
 * @param prefix_f4    Prefix sum of aligned f4 counts per tensor
 * @param tensor_f4    Per-tensor f4 count (X * Z / sizeof(float4))
 * @param dst_rank     Destination rank to load data for
 * @param num_ranks    Total number of ranks
 * @param f4idx        Global f4 index within the slot
 * @param t            [in/out] Current tensor index cache
 * @param cache_bound  [in/out] Current tensor's f4 bound cache
 */
template <bool kFlat = false, bool kLogZ = false, typename T>
__device__ __forceinline__ T sg_unified_ld(
    const size_t& nargs,
    const uint64_t *cached_args,  // Now from shared memory
    const int *prefix_f4,
    const int *tensor_f4,
    const int& dst_rank,
    const int& num_ranks,
    const int& f4idx,
    int& t,
    int& cache_bound,
    int& cache_lowerbound,
    uint64_t& cache_Z,
    uint64_t& cache_Z_log2,
    uint64_t& cache_Z_mask,
    uint64_t &cache_ptr,
    uint64_t &cache_row_stride,   // S-row byte stride (in-Y arg slot; == row_bytes when dense)
    uint64_t &cache_row_bytes,    // dense S-row bytes = H*D*esize (== out-Z for s0g1, num_ranks*Z for s1g0)
    const int scatter_dim
) {
    if (t == -1) {
        t = nargs == 1 ? 0 : bound<false>(prefix_f4, nargs, f4idx) + 1;
        cache_bound = prefix_f4[t];
        cache_lowerbound = (t > 0) ? prefix_f4[t - 1] : 0;
        cache_Z = cached_args[t * 8 + 3];
        if constexpr (kLogZ) {
            cache_Z_log2 = static_cast<uint64_t>(__ffsll(cache_Z) - 1);
            cache_Z_mask = cache_Z - 1;
        }
        cache_ptr = cached_args[t * 8];
        cache_row_stride = cached_args[t * 8 + 2];
        cache_row_bytes = cached_args[t * 8 + 7];
    } else {
        while (f4idx >= cache_bound) {
            t += 1;
            cache_lowerbound = cache_bound;
            cache_bound = prefix_f4[t];
            cache_Z = cached_args[t * 8 + 3];
            if constexpr (kLogZ) {
                cache_Z_log2 = static_cast<uint64_t>(__ffsll(cache_Z) - 1);
                cache_Z_mask = cache_Z - 1;
            }
            cache_ptr = cached_args[t * 8];
            cache_row_stride = cached_args[t * 8 + 2];
            cache_row_bytes = cached_args[t * 8 + 7];
        }
    }
    const int local_f4 = f4idx - cache_lowerbound;
    // Compute base address in float4 units (local_f4 is always in float4 units)
    const float4* f4_base;
    if constexpr (kFlat) {
        // Flat src (s0g1, or s1g0 with a single S row). A strided s0g1 source's
        // per-rank chunk is rows_per_rank S-rows of cache_row_bytes each, pitched
        // by cache_row_stride. Dense inputs (cache_row_stride == cache_row_bytes)
        // and single-row s1g0 keep the exact original contiguous addressing.
        if (scatter_dim == 0 && cache_row_stride != cache_row_bytes) {
            const uint64_t local_byte = static_cast<uint64_t>(local_f4) * sizeof(float4);
            f4_base = reinterpret_cast<const float4*>(cache_ptr) + sg_strided_s0g1_offset(dst_rank, cache_Z, cache_row_bytes, local_byte, cache_row_stride) / sizeof(float4);
        } else {
            f4_base = reinterpret_cast<const float4*>(cache_ptr) + dst_rank * static_cast<uint64_t>(cache_Z) / sizeof(float4) + local_f4;
        }
    } else {
        const uint64_t local_byte = static_cast<uint64_t>(local_f4) * sizeof(float4);
        uint64_t x_coord, z_byte;
        if constexpr (kLogZ) {
            x_coord = local_byte >> cache_Z_log2;
            z_byte = local_byte & cache_Z_mask;
        } else {
            x_coord = local_byte / cache_Z;
            z_byte = local_byte - x_coord * cache_Z;
        }
        if (scatter_dim == 1) {
            // s1g0: the S-rows are the X dim, pitched by cache_row_stride. Dense
            // inputs have cache_row_stride == num_ranks*cache_Z, reducing exactly
            // to x_coord*num_ranks*cache_Z + dst_rank*cache_Z + z_byte.
            f4_base = reinterpret_cast<const float4*>(cache_ptr) + (x_coord * cache_row_stride + static_cast<uint64_t>(dst_rank) * cache_Z + z_byte) / sizeof(float4);
        } else if (cache_row_stride == cache_row_bytes) {
            // Dense s0g1: contiguous per-rank chunk (x==1, so no x term).
            f4_base = reinterpret_cast<const float4*>(cache_ptr) + (static_cast<uint64_t>(dst_rank) * cache_Z + z_byte) / sizeof(float4);
        } else {
            // Strided s0g1: rank dst_rank's chunk is rows_per_rank strided S-rows.
            f4_base = reinterpret_cast<const float4*>(cache_ptr) + sg_strided_s0g1_offset(dst_rank, cache_Z, cache_row_bytes, z_byte, cache_row_stride) / sizeof(float4);
        }
    }
    if constexpr (std::is_same_v<T, float8>) {
        // .v8.f32 requires 32-byte alignment.
        return ld_nc_global(reinterpret_cast<const float8*>(f4_base));
    } else {
        return ld_nc_global(f4_base);
    }
}

/**
 * Get output pointer for a specific source rank
 *
 * TensorLayout semantics for output:
 *   Output tensor is viewed as [X, Y=num_ranks, Z] where:
 *   - X = product of dimensions before pin_dim (gather_dim)
 *   - Y = num_ranks (the gather dimension)
 *   - Z = product of dimensions after pin_dim * elem_size (in bytes)
 *
 * For src_rank j, we write data at position [x, j, z] for all valid x and z.
 * The data is stored in row-major order: base_ptr + x * Y * Z + j * Z + z
 *
 * @param cached_args  Shared memory cached tensor metadata: [in_ptr, in_X, in_Y, in_Z, out_ptr, out_X, out_Y, out_Z] * nargs
 *                     Output at offset i*8+4: ptr, X, Y, Z
 * @return             Pointer to output location, or nullptr if out of bounds
 */
template <bool kFlat = false, bool kLogZ = false, typename T>
__device__ __forceinline__ void sg_unified_st(
    const size_t& nargs,
    const uint64_t *cached_args,  // Now from shared memory
    const int *prefix_f4,
    const int *tensor_f4,
    const int& dst_rank,
    const int& num_ranks,
    const int& f4idx,
    int& t,
    int& cache_bound,
    int& cache_lowerbound,
    uint64_t& cache_Z,
    uint64_t& cache_Z_log2,
    uint64_t& cache_Z_mask,
    uint64_t &cache_ptr,
    const T& f4_value
) {
    if (t == -1) {
        t = nargs == 1 ? 0 : (bound<false>(prefix_f4, nargs, f4idx) + 1);
        cache_bound = prefix_f4[t];
        cache_lowerbound = (t > 0) ? prefix_f4[t - 1] : 0;
        cache_Z = cached_args[t * 8 + 7];
        if constexpr (kLogZ) {
            cache_Z_log2 = static_cast<uint64_t>(__ffsll(cache_Z) - 1);
            cache_Z_mask = cache_Z - 1;
        }
        cache_ptr = cached_args[t * 8 + 4];
    } else {
        while (f4idx >= cache_bound) {
            t += 1;
            cache_lowerbound = cache_bound;
            cache_bound = prefix_f4[t];
            cache_Z = cached_args[t * 8 + 7];
            if constexpr (kLogZ) {
                cache_Z_log2 = static_cast<uint64_t>(__ffsll(cache_Z) - 1);
                cache_Z_mask = cache_Z - 1;
            }
            cache_ptr = cached_args[t * 8 + 4];
        }
    }
    const int local_f4 = f4idx - cache_lowerbound;
    // Compute base address in float4 units (local_f4 is always in float4 units)
    float4* f4_base;
    if constexpr (kFlat) {
        f4_base = reinterpret_cast<float4*>(cache_ptr) + dst_rank * static_cast<uint64_t>(cache_Z) / sizeof(float4) + local_f4;
    } else {
        const uint64_t local_byte = static_cast<uint64_t>(local_f4) * sizeof(float4);
        uint64_t x_coord, z_byte;
        if constexpr (kLogZ) {
            x_coord = local_byte >> cache_Z_log2;
            z_byte = local_byte & cache_Z_mask;
        } else {
            x_coord = local_byte / cache_Z;
            z_byte = local_byte - x_coord * cache_Z;
        }
        f4_base = reinterpret_cast<float4*>(cache_ptr) + (x_coord * num_ranks * cache_Z + static_cast<uint64_t>(dst_rank) * cache_Z + z_byte) / sizeof(float4);
    }
    if constexpr (std::is_same_v<T, float8>) {
        // .v8.f32 requires 32-byte alignment.
        st_na_global(reinterpret_cast<float8*>(f4_base), f4_value);
    } else {
        st_na_global(f4_base, f4_value);
    }
}

/**
 * Direct-addressing variants of sg_unified_ld / sg_unified_st.
 *
 * Used when the caller iterates tensors in an outer loop and therefore
 * already knows which tensor a slot belongs to. The caller passes the
 * current tensor's (cache_ptr, cache_Z, tensor_lb) directly — no `bound()`
 * binary search over prefix_f4, no `while (f4idx >= cache_bound)`
 * tensor-advance branch, no by-reference mutable caching state. Address
 * math is identical to the unified variants.
 *
 * @param tensor_lb  Tensor t's f4 start in unified space (= prefix_f4[t-1],
 *                   0 for t==0). f4idx is global; local_f4 = f4idx - tensor_lb.
 */
template <bool kFlat = false, bool kLogZ = false, typename T>
__device__ __forceinline__ T sg_direct_ld(
    const uint64_t cache_ptr,
    const uint64_t cache_Z,
    const uint64_t cache_Z_log2,
    const uint64_t cache_Z_mask,
    const int tensor_lb,
    const int& dst_rank,
    const int& num_ranks,
    const int& f4idx,
    const uint64_t row_stride,    // S-row byte stride (in-Y arg slot; == row_bytes when dense)
    const uint64_t row_bytes,     // dense S-row bytes = H*D*esize (== out-Z for s0g1, num_ranks*Z for s1g0)
    const int scatter_dim
) {
    const int local_f4 = f4idx - tensor_lb;
    const float4* f4_base;
    if constexpr (kFlat) {
        if (scatter_dim == 0 && row_stride != row_bytes) {
            const uint64_t local_byte = static_cast<uint64_t>(local_f4) * sizeof(float4);
            f4_base = reinterpret_cast<const float4*>(cache_ptr) + sg_strided_s0g1_offset(dst_rank, cache_Z, row_bytes, local_byte, row_stride) / sizeof(float4);
        } else {
            f4_base = reinterpret_cast<const float4*>(cache_ptr) + dst_rank * static_cast<uint64_t>(cache_Z) / sizeof(float4) + local_f4;
        }
    } else {
        const uint64_t local_byte = static_cast<uint64_t>(local_f4) * sizeof(float4);
        uint64_t x_coord, z_byte;
        if constexpr (kLogZ) {
            x_coord = local_byte >> cache_Z_log2;
            z_byte = local_byte & cache_Z_mask;
        } else {
            x_coord = local_byte / cache_Z;
            z_byte = local_byte - x_coord * cache_Z;
        }
        if (scatter_dim == 1) {
            f4_base = reinterpret_cast<const float4*>(cache_ptr) + (x_coord * row_stride + static_cast<uint64_t>(dst_rank) * cache_Z + z_byte) / sizeof(float4);
        } else if (row_stride == row_bytes) {
            f4_base = reinterpret_cast<const float4*>(cache_ptr) + (static_cast<uint64_t>(dst_rank) * cache_Z + z_byte) / sizeof(float4);
        } else {
            f4_base = reinterpret_cast<const float4*>(cache_ptr) + sg_strided_s0g1_offset(dst_rank, cache_Z, row_bytes, z_byte, row_stride) / sizeof(float4);
        }
    }
    if constexpr (std::is_same_v<T, float8>) {
        return ld_nc_global(reinterpret_cast<const float8*>(f4_base));
    } else {
        return ld_nc_global(f4_base);
    }
}

template <bool kFlat = false, bool kLogZ = false, typename T>
__device__ __forceinline__ void sg_direct_st(
    const uint64_t cache_ptr,
    const uint64_t cache_Z,
    const uint64_t cache_Z_log2,
    const uint64_t cache_Z_mask,
    const int tensor_lb,
    const int& dst_rank,
    const int& num_ranks,
    const int& f4idx,
    const T& f4_value
) {
    const int local_f4 = f4idx - tensor_lb;
    float4* f4_base;
    if constexpr (kFlat) {
        f4_base = reinterpret_cast<float4*>(cache_ptr) + dst_rank * static_cast<uint64_t>(cache_Z) / sizeof(float4) + local_f4;
    } else {
        const uint64_t local_byte = static_cast<uint64_t>(local_f4) * sizeof(float4);
        uint64_t x_coord, z_byte;
        if constexpr (kLogZ) {
            x_coord = local_byte >> cache_Z_log2;
            z_byte = local_byte & cache_Z_mask;
        } else {
            x_coord = local_byte / cache_Z;
            z_byte = local_byte - x_coord * cache_Z;
        }
        f4_base = reinterpret_cast<float4*>(cache_ptr) + (x_coord * num_ranks * cache_Z + static_cast<uint64_t>(dst_rank) * cache_Z + z_byte) / sizeof(float4);
    }
    if constexpr (std::is_same_v<T, float8>) {
        st_na_global(reinterpret_cast<float8*>(f4_base), f4_value);
    } else {
        st_na_global(f4_base, f4_value);
    }
}

#define SG_DEBUG_RECORD_TS_AND_BATCH(batch) {\
    if (debug_buf != nullptr && blockIdx.x == 0 && threadIdx.x < 2) {\
        debug_buf[threadIdx.x] = threadIdx.x == 0 ? static_cast<uint64_t>(batch) : clock64();\
    }\
}

#define SG_DEBUG_RECORD_TS(index) {\
    if (debug_buf != nullptr && blockIdx.x == 0 && threadIdx.x == 0) {\
        debug_buf[index] = clock64();\
    }\
}

// Named-barrier group sync for warp-specialized groups. ID 0 is reserved for
// __syncthreads(); callers use 1..15. Both BAR_ID and N_THREADS are runtime
// values — PTX bar.sync accepts register operands for both. N_THREADS must be
// a multiple of 32 (warp size); ptxas rejects non-multiples. Since every
// group's thread count is warps*32, this holds for any non-empty group.
#define WG_BAR_SYNC(BAR_ID, N_THREADS) do {\
    asm volatile("bar.sync %0, %1;" :: "r"(BAR_ID), "r"(N_THREADS));\
} while(0)

// =============================================================================
// scattergather_kernel_p2p — intranode P2P-subgroup specialization.
// =============================================================================
/**
 * Intranode-only P2P-subgroup specialization of scattergather_kernel.
 *
 * Splits c1 (NVL store) and c2 (local read of own send buf) into independent
 * per-peer subgroups so all peer NVL-stores run concurrently within a rank
 * (instead of the sequential `for __lr < num_local_ranks` loop in
 * scattergather_kernel), and each peer's c1_done fires as soon as that
 * specific peer's write completes (no wait-for-all-peers).
 *
 * Layout for `Nlr` local ranks: `2*Nlr - 1` subgroups, each `kWP` warps
 * (uniform — no idle warps). blockDim = kNumThreads = kWP * (2*Nlr-1) * 32.
 *
 *   [0 .. Nlr)        : store subgroup for peer s (s == subgroup_id)
 *                       s == local_rank: self-store (local input → output)
 *                       s != local_rank: NVL-store input → peer s's send buf
 *   [Nlr .. 2*Nlr-1)  : recv subgroup for peer p (p = (r < local_rank) ? r : r+1,
 *                       r = subgroup_id - Nlr). Local-read own send buf → output.
 *
 * Each subgroup iterates its own seg_batch slot loop; subgroups are
 * independent (different memory regions — peer-store writes go to PEER's
 * GPU; recv reads come from OWN send buf). Cross-subgroup sync is only the
 * final __syncthreads() before the gstart update.
 *
 * Slot indexing / signal layout: identical to scattergather_kernel — reuses
 * the same 4-path signals (c1_done/c2_got/c1_picked/c2_ack) and the same
 * per-(sm, slot) ring-buffer indexing (R = max(nvl_ring, rdma_ring)).
 *
 * slot_bytes is decoupled from blockDim (uses GIN_CTA_THREADS=1024) so host
 * buffer sizing (sgcomm.cpp) stays unchanged when blockDim varies with
 * (Nlr, kWP).
 *
 * `bar.sync` IDs 1..2*Nlr-1 (one per subgroup) for subgroup-internal sync
 * after the copy + threadfence, before the leader writes the per-peer ack.
 * ID 0 reserved for __syncthreads().
 */
template <int Nlr, int kWP, int kNumThreads = kWP * (2 * Nlr - 1) * 32,
          int kUnroll = 4, bool kF8 = false, int kFlat = 0,
          bool kUnifiedView = true, bool kLogZ = false>
__global__ void __launch_bounds__(kNumThreads, 1)
scattergather_kernel_p2p(
    uint64_t *args,
    size_t nargs,
    void *gin_win_ptr,
    void **p2p_ptrs,
    int *gmem_barrier,
    uint64_t *debug_buf,
    ncclWindow_t gin_win,
    const int nvl_ring,
    const int rdma_ring,
    const size_t rdma_unroll,
    int round_n,
    int rank,
    int num_ranks,
    ncclDevComm dev_comm,
    const int scatter_dim
) {
    static_assert(Nlr >= 2 && Nlr <= 8, "Nlr must be in [2, 8]");
    static_assert(kWP >= 1 && kWP <= 16, "kWP must be in [1, 16]");
    static_assert(kNumThreads == kWP * (2 * Nlr - 1) * 32, "kNumThreads mismatch");
    static_assert(2 * Nlr - 1 <= 15, "too many subgroups for bar.sync ID space");

    // Single-node only — num_local_ranks == num_ranks.
    const int num_local_ranks = Nlr;
    const int num_nodes = 1;
    const int node_id = 0;
    const int local_rank = rank % num_local_ranks;
    const uint32_t sm_id = blockIdx.x, num_sms = gridDim.x;
    constexpr bool kSrcFlat = kFlat & 1;
    constexpr bool kDstFlat = kFlat & 2;

    __shared__ int prefix_f4[SG_MAX_TENSORS];
    __shared__ int tensor_f4[SG_MAX_TENSORS];
    __shared__ uint64_t cached_args[SG_MAX_TENSORS * 8];

    DEVICE_ASSERT(nargs <= SG_MAX_TENSORS);

    for (int i = threadIdx.x; i < nargs * 8; i += blockDim.x) {
        cached_args[i] = args[i];
    }
    __syncthreads();

    if (threadIdx.x < nargs) {
        // in-Y (arg +2) carries the S-row byte stride (== row_bytes when dense),
        // not the redundant num_ranks — so tensor_f4 = X*Z/16 = whole tensor.
        const uint64_t X = cached_args[threadIdx.x * 8 + 1];
        const uint64_t Z = cached_args[threadIdx.x * 8 + 3];
        const size_t f4_count = static_cast<int>(static_cast<uint64_t>(X) * Z / sizeof(float4));
        tensor_f4[threadIdx.x] = f4_count;
        prefix_f4[threadIdx.x] = f4_count;
    }
    __syncthreads();

    if ((threadIdx.x >> 5) == 0) {
        hillis_steele_sum<EAGER_SCOPE_WARP>(prefix_f4, nargs, threadIdx.x);
    }
    __syncthreads();

    const int total_f4 = prefix_f4[nargs - 1];
    // slot_bytes decoupled from blockDim — matches host sizing (GIN_CTA_THREADS=1024).
    const size_t slot_bytes = rdma_unroll * GIN_CTA_THREADS * sizeof(float4);
    const int slot_f4 = static_cast<int>(rdma_unroll * GIN_CTA_THREADS);

    int smf4start, smf4end;
    if constexpr (kF8 && SG_F8_ARCH_SUPPORTED) {
        const int total_f8 = total_f4 / 2;
        int smf8start, smf8end;
        get_work_range(total_f8, num_sms, sm_id, smf8start, smf8end);
        smf4start = smf8start * 2;
        smf4end = smf8end * 2;
    } else {
        get_work_range(total_f4, num_sms, sm_id, smf4start, smf4end);
    }

    const int unrolled_times = (smf4end - smf4start) / slot_f4;
    const int tailf4 = (smf4end - smf4start) - unrolled_times * slot_f4;
    const int seg_batch = unrolled_times + (tailf4 > 0);

    // Actual slots iterated by this SM. For kUnifiedView=true this equals
    // seg_batch. For kUnifiedView=false each tensor's SM-local slice is
    // ceiled separately, so the sum can exceed seg_batch
    // (ceil(a/s)+ceil(b/s) >= ceil((a+b)/s)). The gstart counter increment
    // and debug timestamp indices below use this actual count.
    int seg_batch_actual;
    if constexpr (kUnifiedView) {
        seg_batch_actual = seg_batch;
    } else {
        seg_batch_actual = 0;
        for (int t = 0; t < nargs; ++t) {
            const int t_lb = (t > 0) ? prefix_f4[t - 1] : 0;
            const int t_sm_start = (smf4start > t_lb) ? smf4start : t_lb;
            const int t_sm_end = (smf4end < prefix_f4[t]) ? smf4end : prefix_f4[t];
            if (t_sm_start >= t_sm_end) continue;
            seg_batch_actual += (t_sm_end - t_sm_start + slot_f4 - 1) / slot_f4;
        }
    }

    // Signal pointer lambdas (identical layout to scattergather_kernel).
    auto sig_ptr = [&](const int& peer) -> uint8_t* {
        if (peer == local_rank) {
            return reinterpret_cast<uint8_t*>(gmem_barrier) + 1024;
        } else {
            return reinterpret_cast<uint8_t*>(__ldg(reinterpret_cast<const uint64_t*>(gmem_barrier) + peer)) +
                   (reinterpret_cast<uint8_t*>(gmem_barrier) - reinterpret_cast<uint8_t*>(gin_win_ptr)) + 1024;
        }
    };

    auto c1_gstart_ptr = [&]() { return reinterpret_cast<uint64_t*>(sig_ptr(local_rank)) + 2 * num_local_ranks * gridDim.x + blockIdx.x; };
    const uint64_t gstart_id = __ldg(reinterpret_cast<const uint64_t*>(c1_gstart_ptr()));

    const int R = std::max(nvl_ring, rdma_ring);
    const size_t sig_base = (size_t)2 * num_local_ranks * num_sms + num_sms;
    const size_t sig_base_c1done = sig_base;
    const size_t sig_base_c1picked = sig_base + (size_t)num_sms * num_nodes * num_local_ranks;

    // c1_done[sm][N][src=local_rank]: source writes to forwarder (dst_peer)'s signal.
    auto c1_done_st_ptr = [&](const int& N, const int& dst_peer) -> uint64_t* {
        return reinterpret_cast<uint64_t*>(sig_ptr(dst_peer)) + sig_base_c1done + ((size_t)sm_id * num_nodes + N) * num_local_ranks + local_rank;
    };
    // c2_got[sm][N][src]: forwarder reads own signal.
    auto c2_got_ld_ptr = [&](const int& N, const int& src_peer) -> uint64_t* {
        return reinterpret_cast<uint64_t*>(sig_ptr(local_rank)) + sig_base_c1done + ((size_t)sm_id * num_nodes + N) * num_local_ranks + src_peer;
    };
    // c2_ack[sm][N][dst=local_rank]: forwarder writes to source src_peer's signal.
    auto c2_ack_st_ptr = [&](const int& N, const int& src_peer) -> uint64_t* {
        return reinterpret_cast<uint64_t*>(sig_ptr(src_peer)) + sig_base_c1picked + ((size_t)sm_id * num_nodes + N) * num_local_ranks + local_rank;
    };
    // c1_picked[sm][N][dst]: source reads own signal.
    auto c1_picked_ld_ptr = [&](const int& N, const int& dst_peer) -> uint64_t* {
        return reinterpret_cast<uint64_t*>(sig_ptr(local_rank)) + sig_base_c1picked + ((size_t)sm_id * num_nodes + N) * num_local_ranks + dst_peer;
    };

    // Data accessors (single-node variants; num_nodes==1 collapses indexing).
    auto peer_send_st_ptr = [&](const size_t& slot, const int& dst_rank) -> float4* {
        const int dst_node = dst_rank / num_local_ranks;
        const int dst_local = dst_rank % num_local_ranks;
        return reinterpret_cast<float4*>(reinterpret_cast<uint8_t*>(p2p_ptrs[dst_local]) +
            (((size_t)sm_id * num_nodes + dst_node) * R + (slot % R)) * num_local_ranks * slot_bytes + (size_t)local_rank * slot_bytes);
    };
    auto my_send_ld_ptr = [&](const size_t& slot, const int& src_local) -> float4* {
        return reinterpret_cast<float4*>(reinterpret_cast<uint8_t*>(gin_win_ptr) +
            (((size_t)sm_id * num_nodes + node_id) * R + (slot % R)) * num_local_ranks * slot_bytes + (size_t)src_local * slot_bytes);
    };

    // Subgroup dispatch.
    const auto warp_id = threadIdx.x >> 5;
    const int subgroup_id = warp_id / kWP;
    const int subgroup_tid = threadIdx.x - subgroup_id * kWP * 32;
    const int subgroup_size = kWP * 32;
    const bool is_leader = (subgroup_tid == 0);
    // bar.sync IDs 1..2*Nlr-1 (one per subgroup). ID 0 reserved for __syncthreads.
    const int bar_id = subgroup_id + 1;

    DEVICE_ASSERT(subgroup_id < 2 * Nlr - 1);

    // Compute recv_peer for subgroup_id >= Nlr (skips self).
    int recv_peer = -1;
    if (subgroup_id >= Nlr) {
        const int r = subgroup_id - Nlr;  // 0..Nlr-2
        recv_peer = (r < local_rank) ? r : (r + 1);
    }

    SG_DEBUG_RECORD_TS_AND_BATCH(seg_batch_actual);
    if (subgroup_id < Nlr) {
        const int s = subgroup_id;
        if (s == local_rank) {
            // ===== Self-store subgroup: input → output (local, no NVL, no signal) =====
            if constexpr (kUnifiedView) {
                for (int slot_idx = 0; slot_idx < seg_batch; ++slot_idx) {
                    const auto f4start = smf4start + slot_idx * slot_f4;
                    const auto nfloat4 = (slot_idx < unrolled_times) ? slot_f4 : tailf4;
                    int tidx = -1, cbound = -1, clowerbound = 0; uint64_t ciZ = 0, ciZ_log2 = 0, ciZ_mask = 0;
                    uint64_t ciptr = 0, ciStride = 0, ciBytes = 0;
                    auto ld_func = [&](const int& f4idx) {
                        return sg_unified_ld<kSrcFlat, kLogZ, float4>(nargs, cached_args, prefix_f4, tensor_f4, rank, num_ranks, f4idx, tidx, cbound, clowerbound, ciZ, ciZ_log2, ciZ_mask, ciptr, ciStride, ciBytes, scatter_dim);
                    };
                    int out_tidx = -1, out_cbound = -1, out_lb = 0; uint64_t out_Z = 0, out_Z_log2 = 0, out_Z_mask = 0;
                    uint64_t coptr = 0;
                    auto st_func = [&](const int& f4idx, const float4& value) {
                        sg_unified_st<kDstFlat, kLogZ, float4>(nargs, cached_args, prefix_f4, tensor_f4, rank, num_ranks, f4idx, out_tidx, out_cbound, out_lb, out_Z, out_Z_log2, out_Z_mask, coptr, value);
                    };
                    if constexpr (kF8 && SG_F8_ARCH_SUPPORTED) {
                        auto f8_ld_func = [&](const int& f8idx) {
                            return sg_unified_ld<kSrcFlat, kLogZ, float8>(nargs, cached_args, prefix_f4, tensor_f4, rank, num_ranks, f8idx * 2, tidx, cbound, clowerbound, ciZ, ciZ_log2, ciZ_mask, ciptr, ciStride, ciBytes, scatter_dim);
                        };
                        auto f8_st_func = [&](const int& f8idx, const float8& value) {
                            sg_unified_st<kDstFlat, kLogZ, float8>(nargs, cached_args, prefix_f4, tensor_f4, rank, num_ranks, f8idx * 2, out_tidx, out_cbound, out_lb, out_Z, out_Z_log2, out_Z_mask, coptr, value);
                        };
                        UNROLLED_GROUP_COPY(kUnroll, subgroup_tid, subgroup_size, nfloat4 / 2, f4start / 2, f4start / 2, f8_ld_func, f8_st_func);
                    } else {
                        UNROLLED_GROUP_COPY(kUnroll, subgroup_tid, subgroup_size, nfloat4, f4start, f4start, ld_func, st_func);
                    }
                }
            } else {
                // kUnifiedView=false: iterate tensors in an outer loop, each
                // slot only touches one tensor — no bound()/while-loop. The
                // direct-addressing lambdas hoist per-tensor (ptr, Z, lb) out
                // of the inner slot loop; the compiler can keep them in
                // registers across the unrolled copy (no by-ref mutable state).
                for (int t = 0; t < nargs; ++t) {
                    const int t_lb = (t > 0) ? prefix_f4[t - 1] : 0;
                    const int t_sm_start = (smf4start > t_lb) ? smf4start : t_lb;
                    const int t_sm_end = (smf4end < prefix_f4[t]) ? smf4end : prefix_f4[t];
                    if (t_sm_start >= t_sm_end) continue;
                    const int t_nslots = (t_sm_end - t_sm_start + slot_f4 - 1) / slot_f4;
                    const uint64_t t_src_ptr = cached_args[t * 8 + 0];
                    const uint64_t t_src_Z   = cached_args[t * 8 + 3];
                    const uint64_t t_src_Z_log2 = static_cast<uint64_t>(__ffsll(t_src_Z) - 1);
                    const uint64_t t_src_Z_mask = t_src_Z - 1;
                    const uint64_t t_src_stride = cached_args[t * 8 + 2];  // S-row byte stride
                    const uint64_t t_src_row_bytes = cached_args[t * 8 + 7];  // H*D*esize (out-Z)
                    const uint64_t t_dst_ptr = cached_args[t * 8 + 4];
                    const uint64_t t_dst_Z   = cached_args[t * 8 + 7];
                    const uint64_t t_dst_Z_log2 = static_cast<uint64_t>(__ffsll(t_dst_Z) - 1);
                    const uint64_t t_dst_Z_mask = t_dst_Z - 1;
                    for (int j = 0; j < t_nslots; ++j) {
                        const auto f4start = t_sm_start + j * slot_f4;
                        const auto nfloat4 = (slot_f4 < t_sm_end - f4start) ? slot_f4 : (t_sm_end - f4start);
                        auto ld_func = [&](const int& f4idx) {
                            return sg_direct_ld<kSrcFlat, kLogZ, float4>(t_src_ptr, t_src_Z, t_src_Z_log2, t_src_Z_mask, t_lb, rank, num_ranks, f4idx, t_src_stride, t_src_row_bytes, scatter_dim);
                        };
                        auto st_func = [&](const int& f4idx, const float4& value) {
                            sg_direct_st<kDstFlat, kLogZ, float4>(t_dst_ptr, t_dst_Z, t_dst_Z_log2, t_dst_Z_mask, t_lb, rank, num_ranks, f4idx, value);
                        };
                        if constexpr (kF8 && SG_F8_ARCH_SUPPORTED) {
                            auto f8_ld_func = [&](const int& f8idx) {
                                return sg_direct_ld<kSrcFlat, kLogZ, float8>(t_src_ptr, t_src_Z, t_src_Z_log2, t_src_Z_mask, t_lb, rank, num_ranks, f8idx * 2, t_src_stride, t_src_row_bytes, scatter_dim);
                            };
                            auto f8_st_func = [&](const int& f8idx, const float8& value) {
                                sg_direct_st<kDstFlat, kLogZ, float8>(t_dst_ptr, t_dst_Z, t_dst_Z_log2, t_dst_Z_mask, t_lb, rank, num_ranks, f8idx * 2, value);
                            };
                            UNROLLED_GROUP_COPY(kUnroll, subgroup_tid, subgroup_size, nfloat4 / 2, f4start / 2, f4start / 2, f8_ld_func, f8_st_func);
                        } else {
                            UNROLLED_GROUP_COPY(kUnroll, subgroup_tid, subgroup_size, nfloat4, f4start, f4start, ld_func, st_func);
                        }
                    }
                }
            }
        } else {
            // ===== Peer-store subgroup: NVL-store input → peer s's send buf =====
            const int dst_rank = s;  // single-node: full rank == local_rank
            if constexpr (kUnifiedView) {
                for (int slot_idx = 0; slot_idx < seg_batch; ++slot_idx) {
                    const size_t gslot = gstart_id + slot_idx;
                    // Flow control: wait c1_picked[node_id][s] >= gslot+1-R (slot reuse).
                    if (slot_idx >= R) {
                        const uint64_t wait_val = gslot + 1 - R;
                        auto bt = clock64();
                        while (ld_acquire_sys_global(c1_picked_ld_ptr(node_id, s)) < wait_val) {
                            if (clock64() - bt > NUM_TIMEOUT_CYCLES) {
                                if (is_leader) {
                                    printf("[rank %d sm %d] p2p store flow ctrl stuck peer=%d want=%lu got=%lu\n",
                                           rank, sm_id, s, wait_val,
                                           (uint64_t)ld_acquire_sys_global(c1_picked_ld_ptr(node_id, s)));
                                    __trap();
                                }
                            }
                        }
                    }
                    const auto f4start = smf4start + slot_idx * slot_f4;
                    const auto nfloat4 = (slot_idx < unrolled_times) ? slot_f4 : tailf4;
                    int tidx = -1, cbound = -1, clowerbound = 0; uint64_t ciZ = 0, ciZ_log2 = 0, ciZ_mask = 0;
                    uint64_t ciptr = 0, ciStride = 0, ciBytes = 0;
                    auto ld_func = [&](const int& f4idx) {
                        return sg_unified_ld<kSrcFlat, kLogZ, float4>(nargs, cached_args, prefix_f4, tensor_f4, dst_rank, num_ranks, f4idx, tidx, cbound, clowerbound, ciZ, ciZ_log2, ciZ_mask, ciptr, ciStride, ciBytes, scatter_dim);
                    };
                    auto st_ptr = peer_send_st_ptr(gslot, dst_rank);
                    if constexpr (kF8 && SG_F8_ARCH_SUPPORTED) {
                        auto f8_ld_func = [&](const int& f8idx) {
                            return sg_unified_ld<kSrcFlat, kLogZ, float8>(nargs, cached_args, prefix_f4, tensor_f4, dst_rank, num_ranks, f8idx * 2, tidx, cbound, clowerbound, ciZ, ciZ_log2, ciZ_mask, ciptr, ciStride, ciBytes, scatter_dim);
                        };
                        UNROLLED_GROUP_COPY(kUnroll, subgroup_tid, subgroup_size, nfloat4 / 2, reinterpret_cast<float8*>(st_ptr), f4start / 2, f8_ld_func, st_na_global);
                    } else {
                        UNROLLED_GROUP_COPY(kUnroll, subgroup_tid, subgroup_size, nfloat4, st_ptr, f4start, ld_func, st_na_global);
                    }
                    WG_BAR_SYNC(bar_id, subgroup_size);
                    if (is_leader) {
                        st_release_sys_global(c1_done_st_ptr(node_id, s), gslot + 1);
                    }
                }
            } else {
                // kUnifiedView=false: outer tensor loop + inner slot loop.
                // gslot is a running counter across tensors (slot_idx_global),
                // consistent across all subgroups of this SM (same tensor
                // order, same per-tensor slot counts derived from prefix_f4
                // and this SM's f4 slice).
                int slot_idx_global = 0;
                for (int t = 0; t < nargs; ++t) {
                    const int t_lb = (t > 0) ? prefix_f4[t - 1] : 0;
                    const int t_sm_start = (smf4start > t_lb) ? smf4start : t_lb;
                    const int t_sm_end = (smf4end < prefix_f4[t]) ? smf4end : prefix_f4[t];
                    if (t_sm_start >= t_sm_end) continue;
                    const int t_nslots = (t_sm_end - t_sm_start + slot_f4 - 1) / slot_f4;
                    const uint64_t t_src_ptr = cached_args[t * 8 + 0];
                    const uint64_t t_src_Z   = cached_args[t * 8 + 3];
                    const uint64_t t_src_Z_log2 = static_cast<uint64_t>(__ffsll(t_src_Z) - 1);
                    const uint64_t t_src_Z_mask = t_src_Z - 1;
                    const uint64_t t_src_stride = cached_args[t * 8 + 2];  // S-row byte stride
                    const uint64_t t_src_row_bytes = cached_args[t * 8 + 7];  // H*D*esize (out-Z)
                    for (int j = 0; j < t_nslots; ++j, ++slot_idx_global) {
                        const size_t gslot = gstart_id + slot_idx_global;
                        if (slot_idx_global >= R) {
                            const uint64_t wait_val = gslot + 1 - R;
                            auto bt = clock64();
                            while (ld_acquire_sys_global(c1_picked_ld_ptr(node_id, s)) < wait_val) {
                                if (clock64() - bt > NUM_TIMEOUT_CYCLES) {
                                    if (is_leader) {
                                        printf("[rank %d sm %d] p2p store flow ctrl stuck peer=%d want=%lu got=%lu\n",
                                               rank, sm_id, s, wait_val,
                                               (uint64_t)ld_acquire_sys_global(c1_picked_ld_ptr(node_id, s)));
                                        __trap();
                                    }
                                }
                            }
                        }
                        const auto f4start = t_sm_start + j * slot_f4;
                        const auto nfloat4 = (slot_f4 < t_sm_end - f4start) ? slot_f4 : (t_sm_end - f4start);
                        auto ld_func = [&](const int& f4idx) {
                            return sg_direct_ld<kSrcFlat, kLogZ, float4>(t_src_ptr, t_src_Z, t_src_Z_log2, t_src_Z_mask, t_lb, dst_rank, num_ranks, f4idx, t_src_stride, t_src_row_bytes, scatter_dim);
                        };
                        auto st_ptr = peer_send_st_ptr(gslot, dst_rank);
                        if constexpr (kF8 && SG_F8_ARCH_SUPPORTED) {
                            auto f8_ld_func = [&](const int& f8idx) {
                                return sg_direct_ld<kSrcFlat, kLogZ, float8>(t_src_ptr, t_src_Z, t_src_Z_log2, t_src_Z_mask, t_lb, dst_rank, num_ranks, f8idx * 2, t_src_stride, t_src_row_bytes, scatter_dim);
                            };
                            UNROLLED_GROUP_COPY(kUnroll, subgroup_tid, subgroup_size, nfloat4 / 2, reinterpret_cast<float8*>(st_ptr), f4start / 2, f8_ld_func, st_na_global);
                        } else {
                            UNROLLED_GROUP_COPY(kUnroll, subgroup_tid, subgroup_size, nfloat4, st_ptr, f4start, ld_func, st_na_global);
                        }
                        WG_BAR_SYNC(bar_id, subgroup_size);
                        if (is_leader) {
                            st_release_sys_global(c1_done_st_ptr(node_id, s), gslot + 1);
                        }
                    }
                }
            }
        }
    } else {
        // ===== Recv subgroup: poll c2_got[node_id][p], local-read own send buf → output =====
        const int p = recv_peer;  // peer (skips self)
        const int src_rank = p;  // single-node
        if constexpr (kUnifiedView) {
            for (int slot_idx = 0; slot_idx < seg_batch; ++slot_idx) {
                const size_t gslot = gstart_id + slot_idx;
                auto bt = clock64();
                while (ld_acquire_sys_global(c2_got_ld_ptr(node_id, p)) < gslot + 1) {
                    if (clock64() - bt > NUM_TIMEOUT_CYCLES) {
                        if (is_leader) {
                            printf("[rank %d sm %d] p2p recv stuck peer=%d want=%lu got=%lu\n",
                                   rank, sm_id, p, gslot + 1,
                                   (uint64_t)ld_acquire_sys_global(c2_got_ld_ptr(node_id, p)));
                            __trap();
                        }
                    }
                }
                const auto f4start = smf4start + slot_idx * slot_f4;
                const auto nfloat4 = (slot_idx < unrolled_times) ? slot_f4 : tailf4;
                auto ld_ptr = my_send_ld_ptr(gslot, p);
                int out_tidx = -1, out_cbound = -1, out_lb = 0; uint64_t out_Z = 0, out_Z_log2 = 0, out_Z_mask = 0;
                uint64_t coptr = 0;
                auto st_func = [&](const int& f4idx, const float4& value) {
                    sg_unified_st<kDstFlat, kLogZ, float4>(nargs, cached_args, prefix_f4, tensor_f4, src_rank, num_ranks, f4idx, out_tidx, out_cbound, out_lb, out_Z, out_Z_log2, out_Z_mask, coptr, value);
                };
                if constexpr (kF8 && SG_F8_ARCH_SUPPORTED) {
                    auto f8_st_func = [&](const int& f8idx, const float8& value) {
                        sg_unified_st<kDstFlat, kLogZ, float8>(nargs, cached_args, prefix_f4, tensor_f4, src_rank, num_ranks, f8idx * 2, out_tidx, out_cbound, out_lb, out_Z, out_Z_log2, out_Z_mask, coptr, value);
                    };
                    UNROLLED_GROUP_COPY(kUnroll, subgroup_tid, subgroup_size, nfloat4 / 2, f4start / 2, reinterpret_cast<const float8*>(ld_ptr), ld_nc_global, f8_st_func);
                } else {
                    UNROLLED_GROUP_COPY(kUnroll, subgroup_tid, subgroup_size, nfloat4, f4start, ld_ptr, ld_nc_global, st_func);
                }
                WG_BAR_SYNC(bar_id, subgroup_size);
                if (is_leader) {
                    st_release_sys_global(c2_ack_st_ptr(node_id, p), gslot + 1);
                }
            }
        } else {
            int slot_idx_global = 0;
            for (int t = 0; t < nargs; ++t) {
                const int t_lb = (t > 0) ? prefix_f4[t - 1] : 0;
                const int t_sm_start = (smf4start > t_lb) ? smf4start : t_lb;
                const int t_sm_end = (smf4end < prefix_f4[t]) ? smf4end : prefix_f4[t];
                if (t_sm_start >= t_sm_end) continue;
                const int t_nslots = (t_sm_end - t_sm_start + slot_f4 - 1) / slot_f4;
                const uint64_t t_dst_ptr = cached_args[t * 8 + 4];
                const uint64_t t_dst_Z   = cached_args[t * 8 + 7];
                const uint64_t t_dst_Z_log2 = static_cast<uint64_t>(__ffsll(t_dst_Z) - 1);
                const uint64_t t_dst_Z_mask = t_dst_Z - 1;
                for (int j = 0; j < t_nslots; ++j, ++slot_idx_global) {
                    const size_t gslot = gstart_id + slot_idx_global;
                    auto bt = clock64();
                    while (ld_acquire_sys_global(c2_got_ld_ptr(node_id, p)) < gslot + 1) {
                        if (clock64() - bt > NUM_TIMEOUT_CYCLES) {
                            if (is_leader) {
                                printf("[rank %d sm %d] p2p recv stuck peer=%d want=%lu got=%lu\n",
                                       rank, sm_id, p, gslot + 1,
                                       (uint64_t)ld_acquire_sys_global(c2_got_ld_ptr(node_id, p)));
                                __trap();
                            }
                        }
                    }
                    const auto f4start = t_sm_start + j * slot_f4;
                    const auto nfloat4 = (slot_f4 < t_sm_end - f4start) ? slot_f4 : (t_sm_end - f4start);
                    auto ld_ptr = my_send_ld_ptr(gslot, p);
                    auto st_func = [&](const int& f4idx, const float4& value) {
                        sg_direct_st<kDstFlat, kLogZ, float4>(t_dst_ptr, t_dst_Z, t_dst_Z_log2, t_dst_Z_mask, t_lb, src_rank, num_ranks, f4idx, value);
                    };
                    if constexpr (kF8 && SG_F8_ARCH_SUPPORTED) {
                        auto f8_st_func = [&](const int& f8idx, const float8& value) {
                            sg_direct_st<kDstFlat, kLogZ, float8>(t_dst_ptr, t_dst_Z, t_dst_Z_log2, t_dst_Z_mask, t_lb, src_rank, num_ranks, f8idx * 2, value);
                        };
                        UNROLLED_GROUP_COPY(kUnroll, subgroup_tid, subgroup_size, nfloat4 / 2, f4start / 2, reinterpret_cast<const float8*>(ld_ptr), ld_nc_global, f8_st_func);
                    } else {
                        UNROLLED_GROUP_COPY(kUnroll, subgroup_tid, subgroup_size, nfloat4, f4start, ld_ptr, ld_nc_global, st_func);
                    }
                    WG_BAR_SYNC(bar_id, subgroup_size);
                    if (is_leader) {
                        st_release_sys_global(c2_ack_st_ptr(node_id, p), gslot + 1);
                    }
                }
            }
        }
    }

    __syncthreads();
    if (threadIdx.x == 0) {
        st_na_global(c1_gstart_ptr(), static_cast<uint64_t>(gstart_id + seg_batch_actual));
    }
    SG_DEBUG_RECORD_TS(2 + seg_batch_actual * 2 * (num_nodes + num_local_ranks));
}


} // namespace pace

#endif // PACE_SG_KERNELS_CUH
