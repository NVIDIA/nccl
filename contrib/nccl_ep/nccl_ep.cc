/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 * See LICENSE.txt for more license information.
 */

#include <unistd.h>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <mutex>
#include <new>
#include <optional>
#include <set>
#include <cstring>
#include <string>
#include <vector>
#include <nccl.h>
#include <nccl_device.h>
#include "nccl_ep.h"
#include "nccl_ep_env.h"
#include "common.hpp"

// HT (High Throughput) includes
#include "device/hybridep_adapter.cuh"
#include "device/hybridep_configs.cuh"
#include "device/ll_ep_adapter.cuh"

// NCCLCHECK macro for public API error checking
// Undef any existing definition from internal NCCL headers to avoid using ncclDebugLog
#ifdef NCCLCHECK
#undef NCCLCHECK
#endif
#define NCCLCHECK(cmd) \
    do { \
        ncclResult_t res = cmd; \
        if (res != ncclSuccess) { \
            fprintf(stderr, "NCCL error %s:%d '%s'\n", __FILE__, __LINE__, ncclGetErrorString(res)); \
            return res; \
        } \
    } while (0)

// Forward declarations for HT functions
static ncclResult_t
init_hybridep_intranode(ncclEpGroup_t ep_group, const ncclEpGroupConfig_t* config, cudaStream_t stream);
static ncclResult_t destroy_hybridep_intranode(ncclEpGroup_t ep_group);
static ncclResult_t
init_hybridep_internode(ncclEpGroup_t ep_group, const ncclEpGroupConfig_t* config, cudaStream_t stream);
static ncclResult_t destroy_hybridep_internode(ncclEpGroup_t ep_group);

// Forward declaration for LL helper (defined alongside ll_init_handle).
static ncclResult_t ll_resize_rdma_buffer(ncclEpGroup_t ep_group, size_t new_size);

// Define NCCL_CHECK_RESULT macro for NCCL error checking
#ifndef NCCL_CHECK_RESULT
#define NCCL_CHECK_RESULT(cmd) \
    do { \
        ncclResult_t res = cmd; \
        if (res != ncclSuccess) { \
            fprintf(stderr, "NCCL error %d at %s:%d\n", res, __FILE__, __LINE__); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)
#endif

// Size-based ABI versioning: every cross-boundary struct starts with a `size`
// field set by the caller to sizeof(struct). The library checks that against
// its own known size; any mismatch means caller and library are from different
// releases. Strict equality for now — see nccl_ep.h for the planned future
// relaxation (all-zero-trailing-bytes escape hatch).
// Immediately after `size` there is a `magic` field pre-filled by NCCL_EP_*_INIT
// to catch unininitialized structures.
#define EP_REQUIRE_STRUCT(ptr) \
    do { \
        assert( \
            (ptr) != nullptr && (ptr)->size == sizeof(*(ptr)) && \
            "ABI struct size mismatch — caller and libnccl_ep.so must be from the same release"); \
        assert( \
            (ptr)->magic == NCCL_EP_MAGIC && \
            "struct magic mismatch — pointer is uninitialised or not an NCCL EP struct " \
            "(initialise with the corresponding NCCL_EP_*_INIT macro)"); \
    } while (0)
#define EP_OPTIONAL_STRUCT(ptr) \
    do { \
        assert( \
            ((ptr) == nullptr || (ptr)->size == sizeof(*(ptr))) && \
            "ABI struct size mismatch -- caller and libnccl_ep.so must be from the same release"); \
        assert( \
            ((ptr) == nullptr || (ptr)->magic == NCCL_EP_MAGIC) && \
            "struct magic mismatch -- pointer is uninitialised or not an NCCL EP struct " \
            "(initialise with the corresponding NCCL_EP_*_INIT macro)"); \
    } while (0)

// Minimum FP8 EXTERN scale block size supported by the pre-allocated output scale buffer.
// Buffer is sized at group create assuming this value; callers must use scale_block_size >= this.
// TODO: remove when ncclEpGroupConfig_t gains max_scale_bytes_per_token (future API update).
static constexpr int kFP8ExternScaleBlockSize = 128;

// Targeted forward-binary-compat for ncclEpLayoutInfo_t (only). The struct
// grows by appending; older callers built against a pre-flag header report a
// smaller `size`. Accept any size in [min-known, current-sizeof] and let
// per-field readers default to the corresponding AUTO sentinel when a field
// falls beyond the caller's reported size. Bumps to this struct must
// document the corresponding legacy size below.
static constexpr size_t kNcclEpLayoutInfoMinSize =
    offsetof(ncclEpLayoutInfo_t, recv_topk_idx_kind); // last legacy field end
#define EP_OPTIONAL_LAYOUT_INFO(ptr) \
    do { \
        if ((ptr) != nullptr) { \
            if ((ptr)->size < kNcclEpLayoutInfoMinSize || (ptr)->size > sizeof(*(ptr))) { \
                fprintf( \
                    stderr, \
                    "NCCL EP: ncclEpLayoutInfo_t size out of supported range: " \
                    "got %u, expected [%zu, %zu]\n", \
                    (ptr)->size, \
                    kNcclEpLayoutInfoMinSize, \
                    sizeof(*(ptr))); \
                return ncclInvalidArgument; \
            } \
            if ((ptr)->magic != NCCL_EP_MAGIC) { \
                fprintf( \
                    stderr, \
                    "NCCL EP: ncclEpLayoutInfo_t magic mismatch: got 0x%x, " \
                    "expected 0x%x (initialise with NCCL_EP_LAYOUT_INFO_INIT)\n", \
                    (ptr)->magic, \
                    NCCL_EP_MAGIC); \
                return ncclInvalidArgument; \
            } \
        } \
    } while (0)

// Safe field reader for ncclEpLayoutInfo_t::recv_topk_idx_kind. Returns AUTO
// when the caller's struct (size) does not cover the field, preserving the
// pre-flag default.
static inline ncclEpExpertIdKind_t layoutInfoRecvTopkIdxKind(const ncclEpLayoutInfo_t* lip) {
    if (lip == nullptr) return NCCL_EP_EXPERT_ID_AUTO;
    constexpr size_t field_end = offsetof(ncclEpLayoutInfo_t, recv_topk_idx_kind) + sizeof(ncclEpExpertIdKind_t);
    if (lip->size < field_end) return NCCL_EP_EXPERT_ID_AUTO;
    return lip->recv_topk_idx_kind;
}

// Resolve AUTO -- the public sentinel meaning "library default" -- to the
// concrete numbering the kernels need. AUTO maps to LOCAL today; pinning a
// non-AUTO value passes straight through.
static inline ncclEpExpertIdKind_t resolveRecvTopkIdxKind(ncclEpExpertIdKind_t k) {
    return (k == NCCL_EP_EXPERT_ID_AUTO) ? NCCL_EP_EXPERT_ID_LOCAL : k;
}

// Helper function to convert ncclDataType_t to cudaDataType_t
static cudaDataType_t ncclDataTypeToCudaDataType(ncclDataType_t nccl_type) {
    switch (nccl_type) {
    case ncclFloat16:
        return CUDA_R_16F;
    case ncclFloat32:
        return CUDA_R_32F;
    case ncclFloat64:
        return CUDA_R_64F;
    case ncclBfloat16:
        return CUDA_R_16BF;
    case ncclInt8:
        return CUDA_R_8I;
    case ncclInt32:
        return CUDA_R_32I;
    case ncclInt64:
        return CUDA_R_64I;
    case ncclUint8:
        return CUDA_R_8U;
    case ncclUint32:
        return CUDA_R_32U;
    case ncclUint64:
        return CUDA_R_64U;
    default:
        assert(false && "Unsupported ncclDataType_t for conversion to cudaDataType_t");
        return CUDA_R_16BF; // Default fallback
    }
}

static size_t ncclTypeSize(ncclDataType_t nccl_type) {
    switch (nccl_type) {
    case ncclInt8:
    case ncclUint8:
    case ncclFloat8e4m3:
    case ncclFloat8e5m2:
        return 1;
    case ncclFloat16:
    case ncclBfloat16:
        return 2;
    case ncclInt32:
    case ncclUint32:
    case ncclFloat32:
        return 4;
    case ncclInt64:
    case ncclUint64:
    case ncclFloat64:
        return 8;
    default:
        assert(false && "Unsupported ncclDataType_t for size query");
        return 0;
    }
}

// Supported token wire dtypes (runtime, since x->datatype comes from the descriptor).
// NONE mode is BF16/FP16/FP32; FP8 (e4m3/e5m2) is a separate dispatch wire format.
// Single source of truth for the accepted sets.
static bool validate_dtype(ncclDataType_t dt) {
    return dt == ncclBfloat16 || dt == ncclFloat16 || dt == ncclFloat32;
}
static bool validate_fp8_dtype(ncclDataType_t dt) {
    return dt == ncclFloat8e4m3 || dt == ncclFloat8e5m2;
}

// Dynamic NDTensor allocation
// Dynamic allocation is used for tensors that are returned by ncclEpTensorAlloc
// and must be released with ncclEpTensorDestroy.

// Internal magic cookie for tensors returned by ncclEpTensorAlloc. Kept out
// of the public header so callers can only spell the STATIC cookie
// (NCCL_EP_TENSOR_MAGIC) via NCCL_EP_TENSOR_INIT.
#define NCCL_EP_TENSOR_ALLOC_STATIC NCCL_EP_TENSOR_MAGIC
#define NCCL_EP_TENSOR_ALLOC_DYNAMIC 0xBEEFBEEF

#define NCCL_EP_TENSOR_INIT_DYNAMIC \
    ((ncclEpTensor_t){.size = (unsigned int)sizeof(ncclEpTensor_t), .magic = NCCL_EP_TENSOR_ALLOC_DYNAMIC})

typedef struct ncclEpTensorInternal_s {
    // Shadow of pub.sizes captured by ncclEpTensorAlloc; lets the library
    // detect callers that overwrote the sizes pointer on a dynamic descriptor.
    size_t* sizes_shadow;

    // Public field exposed to the user.
    ncclEpTensor_t pub;
} ncclEpTensorInternal_t;

// Recover the outer wrapper from a pointer to its embedded public descriptor.
static ncclEpTensorInternal_t* _getInternalTensor(ncclEpTensor_t* tensor) {
    return reinterpret_cast<ncclEpTensorInternal_t*>(
        reinterpret_cast<char*>(tensor) - offsetof(ncclEpTensorInternal_t, pub));
}
static const ncclEpTensorInternal_t* _getInternalTensor(const ncclEpTensor_t* tensor) {
    return reinterpret_cast<const ncclEpTensorInternal_t*>(
        reinterpret_cast<const char*>(tensor) - offsetof(ncclEpTensorInternal_t, pub));
}

// True if the tensor's `magic` cookie marks it as properly initialised by
// either NCCL_EP_TENSOR_INIT (static) or ncclEpTensorAlloc (dynamic).
// For dynamic tensors, also cross-check the public `sizes` pointer against
// the shadow captured at allocation — catches callers that overwrote the
// descriptor's library-owned sizes array.
static inline bool tensorIsInitialised(const ncclEpTensor_t* t) {
    if (t == nullptr) return false;
    if (t->magic == NCCL_EP_TENSOR_ALLOC_STATIC) return true;
    if (t->magic == NCCL_EP_TENSOR_ALLOC_DYNAMIC) {
        return _getInternalTensor(t)->sizes_shadow == t->sizes;
    }
    return false;
}

// True when the tensor is zero-extent in at least one dimension. Empty tensors
// carry no element to read or write, so the dispatch/combine kernels' per-token
// loops degrade to no-ops and the host side has no storage to address. The
// library treats them as a valid no-op input at the API boundary even though
// they legitimately lack a storage binding (PyTorch/NumPy hand out
// data_ptr() == 0 for such buffers).
static inline bool tensorIsEmpty(const ncclEpTensor_t* t) {
    if (t->sizes == nullptr) return false;
    for (unsigned int i = 0; i < t->ndim; ++i) {
        if (t->sizes[i] == 0) return true;
    }
    return false;
}

// True when the tensor advertises a storage binding -- either a device pointer
// or a window handle (offset may be 0, so we only check `win_hdl`). A tensor
// with neither is unusable by the library.
static inline bool tensorHasBinding(const ncclEpTensor_t* t) {
    return t->data != nullptr || t->win_hdl != ncclWindow_t{};
}

// Combined boundary check used by tensor_ptr / tensor_required: the descriptor
// must be initialised AND carry a storage binding. Empty tensors skip the
// binding check because they have no element to address; the kernels handle
// num_tokens == 0 as a no-op on the sender side and still participate in the
// collective on the receiver side.
static inline void tensorAssertValid(const ncclEpTensor_t* t) {
    assert(tensorIsInitialised(t) && "ncclEpTensor_t not initialised — use NCCL_EP_TENSOR_INIT or ncclEpTensorAlloc");
    assert(
        (tensorIsEmpty(t) || tensorHasBinding(t)) &&
        "ncclEpTensor_t has no storage binding — set `.data` or (`.win_hdl` [+ `.win_offset`])");
}

// Validate that a non-null tensor pointer points at an initialised descriptor.
// Returns the pointer unchanged when null (absent / optional field).
// Aborts with a clear message when the caller passed garbage or a zero-filled
// struct — catching the bug at the API boundary before any field is touched.
static inline ncclEpTensor_t* tensor_ptr(ncclEpTensor_t* t) {
    if (t == nullptr) return nullptr;
    tensorAssertValid(t);
    return t;
}
static inline const ncclEpTensor_t* tensor_ptr(const ncclEpTensor_t* t) {
    if (t == nullptr) return nullptr;
    tensorAssertValid(t);
    return t;
}

// Same as tensor_ptr but for tensors the library knows must be present (e.g.,
// inputs->tokens). Aborts when the pointer is null OR the cookie is wrong.
static inline ncclEpTensor_t* tensor_required(ncclEpTensor_t* t) {
    assert(t != nullptr && "required tensor field is NULL");
    tensorAssertValid(t);
    return t;
}
static inline const ncclEpTensor_t* tensor_required(const ncclEpTensor_t* t) {
    assert(t != nullptr && "required tensor field is NULL");
    tensorAssertValid(t);
    return t;
}

// Make a temporary stack copy of `src` for short-lived library-internal use
// (e.g. window-binding scratch). The copy carries the STATIC magic regardless
// of `src`'s magic. The copy shares `src`'s `sizes` pointer; the caller must
// keep `src->sizes` alive for `dst`'s lifetime.
static inline void tensor_temp_copy(ncclEpTensor_t* dst, const ncclEpTensor_t* src) {
    *dst = *src;
    dst->magic = NCCL_EP_TENSOR_MAGIC;
}

// Make a permanent (library-owned) copy of `src` into `dst`. Like
// tensor_temp_copy plus a heap-allocated `sizes` array deep-copied from
// `src->sizes`. Reuses `dst`'s existing sizes buffer when the shape (ndim)
// is unchanged.
static inline void tensor_permanent_copy(ncclEpTensor_t* dst, const ncclEpTensor_t* src) {
    size_t* sizes_buf = dst->sizes;
    const bool shape_matches = (sizes_buf != nullptr && dst->ndim == src->ndim);
    if (!shape_matches) {
        delete[] sizes_buf;  // no-op on nullptr
        sizes_buf = new size_t[src->ndim];
    }
    for (unsigned int i = 0; i < src->ndim; i++) {
        sizes_buf[i] = src->sizes[i];
    }
    tensor_temp_copy(dst, src);
    dst->sizes = sizes_buf;
}

ncclResult_t ncclEpTensorAlloc(
    ncclEpTensor_t** tensor,
    unsigned int ndim,
    ncclDataType_t datatype,
    const size_t* sizes,
    const ncclEpTensorAllocConfig_t* config) {
    EP_OPTIONAL_STRUCT(config);
    if (tensor == nullptr || sizes == nullptr || ndim == 0) {
        return ncclInvalidArgument;
    }

    ncclEpTensorInternal_t* internal = new ncclEpTensorInternal_t();
    internal->pub = NCCL_EP_TENSOR_INIT_DYNAMIC;
    internal->pub.ndim = ndim;
    internal->pub.datatype = datatype;

    size_t* sizes_copy = new size_t[ndim];
    for (unsigned int i = 0; i < ndim; i++) sizes_copy[i] = sizes[i];
    internal->pub.sizes = sizes_copy;
    internal->sizes_shadow = sizes_copy;

    *tensor = &internal->pub;
    return ncclSuccess;
}

ncclResult_t ncclEpTensorDestroy(ncclEpTensor_t* tensor) {
    if (tensor == nullptr) {
        return ncclSuccess;
    }
    if (!tensorIsInitialised(tensor) || tensor->magic != NCCL_EP_TENSOR_ALLOC_DYNAMIC) {
        return ncclInvalidArgument;
    }
    ncclEpTensorInternal_t* internal = _getInternalTensor(tensor);
    delete[] internal->sizes_shadow;
    delete internal;
    return ncclSuccess;
}

// Allgather on host memory using NCCL (used once for hostname exchange).
// This operates on a single in-place host buffer, unlike batchAllGatherIpcHandles
// which batches multiple IPC handles via a packed device buffer.
// Each rank contributes element_size bytes at offset rank * element_size.
static void
ncclAllGatherHost(void* host_buffer, size_t element_size, int rank, int nRanks, ncclComm_t comm, cudaStream_t stream) {
    const size_t total_size = element_size * nRanks;
    void* d_buffer;
    CUDA_CHECK(cudaMalloc(&d_buffer, total_size));
    CUDA_CHECK(cudaMemcpyAsync(
        static_cast<uint8_t*>(d_buffer) + rank * element_size,
        static_cast<uint8_t*>(host_buffer) + rank * element_size,
        element_size,
        cudaMemcpyHostToDevice,
        stream));
    NCCL_CHECK_RESULT(ncclAllGather(
        static_cast<uint8_t*>(d_buffer) + rank * element_size,
        d_buffer,
        element_size,
        ncclUint8,
        comm,
        stream));
    CUDA_CHECK(cudaMemcpyAsync(host_buffer, d_buffer, total_size, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaFree(d_buffer));
}

// NCCL barrier using AllReduce.
// If workspace is provided, it is used directly (must be at least sizeof(int) device bytes).
// Otherwise a temporary cudaMalloc/cudaFree pair is used as fallback.
static ncclResult_t ncclBarrier(ncclComm_t comm, cudaStream_t stream, void* workspace = nullptr) {
    int* nccl_barrier_var = nullptr;
    bool owns_memory = false;
    if (workspace) {
        nccl_barrier_var = static_cast<int*>(workspace);
    } else {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&nccl_barrier_var), sizeof(int)));
        owns_memory = true;
    }
    CUDA_CHECK(cudaMemset(nccl_barrier_var, 0, sizeof(int)));
    NCCL_CHECK_RESULT(ncclAllReduce(nccl_barrier_var, nccl_barrier_var, 1, ncclInt, ncclSum, comm, stream));
    CUDA_CHECK(cudaDeviceSynchronize());
    if (owns_memory) {
        CUDA_CHECK(cudaFree(nccl_barrier_var));
    }
    return ncclSuccess;
}

// Opaque struct definitions
struct ncclEpGroup {
    ncclComm_t comm;

    int nRanks;               // Total number of ranks (from ncclCommCount)
    int rank;                 // This rank's ID
    int nNodes;               // Number of nodes

    void* ep_workspace;       // Device workspace for EP operations
    int cuda_device_id;       // CUDA device ID
    int lsa_team_size;        // LSA team size: ncclTeamLsa(comm).nRanks
    int lsa_rank;             // Rank within LSA team: ncclTeamLsa(comm).rank
    int rdma_team_size;       // RDMA ranks
    int rdma_rank;            // RDMA rank
    void* rdma_buffer;
    // Bytes currently backing rdma_buffer. Layouts on live handles store
    // offsets only and are resolved against this base at use time, so the
    // buffer can grow safely while handles are live.
    //
    // Sizing is controlled by config.rdma_buffer_size:
    //   - NCCL_EP_AUTO: the buffer is lazily allocated on the first LL
    //     handle init using the actual (layout, num_topk), and grown later
    //     if a subsequent handle needs more.
    //   - any other positive value: the buffer is allocated at group time
    //     to exactly that size. Handle init rejects (returns an error) any
    //     handle whose layout does not fit; no growth is performed.
    size_t rdma_buffer_size_alloc;
    ncclEpGroupConfig_t config;         // Stored configuration

    // Environment-variable configuration, populated once at group creation
    // (nccl_ep_env_init) and read by this group's handles via the
    // nccl_ep_env_* accessors.
    ncclEpEnvConfig env;

    // HT inter-node RDMA stride: max_dispatch_tokens_per_rank rounded up to a whole
    // number of dispatch chunks so a partial final chunk stays within a node's region.
    int ht_aligned_max_tokens = 0;

    // Resolved HT dispatch/combine tokens-per-chunk for this group. RDMA configs
    // default to HT_TOKENS_PER_CHUNK_RDMA_DEFAULT (64); LSA-only configs default to
    // a grid-proportional size (NUM_OF_TOKENS_PER_GROUP * comm_num_sms, rounded up
    // to a multiple of 32). Either may be overridden by NCCL_EP_TOKENS_PER_CHUNK.
    int ht_tokens_per_chunk = 0;

    struct {
        // Device communicator (single comm, multiple contexts)
        // HT internode uses ncclTeamRail on the base communicator
        ncclDevComm_t* dcomms = nullptr;       // Host array of device communicators
        ncclDevComm_t* d_dcomms = nullptr;     // Device array of device communicators
        int num_comms = 0;                     // Number of communicators (always 1)
        int num_dcomms = 0;                    // Number of device comms
        int qps_per_rank = 0;                  // Total QPs (connections) per rank
        int num_ctx_per_comm = 0;              // Number of contexts per communicator

        // GIN memory base pointer and window
        void* gin_base_ptr = nullptr;         // Base pointer for all GIN memory
        ncclWindow_t nccl_window = {};        // Single registered window handle (pointer-sized)
        unsigned signals_base = 0;            // Base signal ID for dispatch
        unsigned combine_signal_offset = 0;   // Signal offset for combine operations
        int num_total_signals = 0;            // Total number of signals

        // Used by kernels to calculate actual addresses for RDMA puts
        size_t rdma_intra_node_red_token_offset = 0;
        size_t combine_rdma_inter_node_group_token_offset = 0;
        size_t rdma_intra_node_red_prob_offset = 0;
        size_t combine_rdma_inter_node_group_prob_offset = 0;
        size_t token_staging_offset = 0;
        size_t dense_prob_offset = 0;
        size_t scaling_factor_staging_offset = 0;

        // Layout: [NUM_LSA_TEAMS-1][BATCH_SIZE * bytes_per_entry]
        // bytes_per_entry = hidden * sizeof(TOKEN_DATA_TYPE) + prob_size + sf_size
        size_t rdma_send_staging_offset = 0;
        size_t rdma_inter_node_group_packed_offset = 0;  // Packed receive buffer (token+prob+sf)

        // RDMA sync-guard readiness-flag regions (per direction)
        size_t dispatch_guard_offset = 0;
        size_t combine_guard_offset = 0;

        unsigned signals_tail_base = 0;         // Base signal ID for tail tracking (sender -> receiver)
        int num_max_rdma_chunked_send_tokens = HYBRIDEP_DISPATCH_RDMA_BATCH_SIZE;

    } gin_config;

    int num_local_experts;    // Number of local experts (num_experts / comm->nRanks)
    int max_recv_tokens;      // Resolved per-rank IPC slot budget (= config.max_recv_tokens_per_rank).

    // SM-count configuration, all resolved once at ncclEpCreateGroup.
    unsigned int device_sm_count;   // Number of SMs on the device
    unsigned int comm_num_sms;      // Resolved SM count for EP kernels (from config.max_num_sms)
    unsigned int prolog_epilog_sms; // Resolved SM count for the prolog/epilog kernels (local_dup, local_reduce).
    unsigned int
        preprocess_num_sms; // Resolved SM count for the preprocessing scan kernels (NCCL_EP_PREPROCESS_NUM_SMS).

  // HT EM dispatch/combine code path. Resolved once at ncclEpCreateGroup.
    enum class HtEmMode : uint8_t {
        kLocalPermute = 0, // FLAT-dispatch + local_permute kernels
        kLocalDup = 1, // dispatch dedup + local_dup/local_reduce
        kNvlinkDup = 2, // in-kernel Expert-major scatter (sender duplicates over NVLink)
    };
    HtEmMode ht_em_mode;

    ncclEpAllocConfig_t alloc;

  // Active-mask buffer for FT support (LL only)
    int* mask_buffer = nullptr; // Device: int[nRanks], null if !enable_mask
    int* async_error_flag = nullptr; // Host-pinned: 0 = ok, 1 = timeout occurred
    uint64_t timeout_cycles = NUM_TIMEOUT_CYCLES; // GPU clock cycles for wait-loop timeout

  // Physical node properties (CUDA device assignment, IPC between co-located GPUs)
    int gpus_per_node; // Physical GPUs per node (nRanks / nNodes)
    int rank_in_node; // Per-node CUDA device ordinal (= cuda_device_id)
    int node_id; // Physical node index (rank / gpus_per_node)

  // NCCL device API
    size_t num_nccl_comms;
    std::vector<ncclComm_t> nccl_comms;
    ncclDevComm_t* nccl_dev_comms;
    ncclWindow_t* nccl_wins;
    int num_dispatch_signals;
    unsigned clean_barrier_signal_base;

  // Cross-rank sync region used by clean_low_latency_buffer.
    void* sync_buffer = nullptr; // device buffer, int[nRanks] aligned
    ncclWindow_t* sync_window = nullptr; // device ptr to the registered window handle

  // HT buffers for intranode communication
    struct {
    // IPC-mapped buffer pointer arrays (fixed-size, indexed by local NVL rank)
    // Host arrays for population and cleanup
        void** dispatch_expert_output_token_buffer_ptrs;
        float** dispatch_expert_output_prob_buffer_ptrs;
        float** dispatch_expert_output_scaling_factor_buffer_ptrs;
        uint16_t** combine_expert_input_token_buffer_ptrs;
        float** combine_expert_input_prob_buffer_ptrs;

    // Local buffers (owned by this rank)
        void* expert_output_token;
        float* expert_output_prob;
        float* expert_output_scaling_factor;
        uint16_t* expert_input_token;
        float* expert_input_prob;

    // Slot capacity of expert_output_token / expert_input_token.
        size_t token_staging_slots;

    // Sync flags (rank 0 allocates, others IPC-map)
        uint32_t* intra_node_write_completion_flags;
        uint32_t* combine_intra_node_write_completion_flags;
    // Single device-resident counter block (one alloc, one free) backing
    // the grid-barrier counters and the per-kernel expected-flag counters.
    // The expected counters are initialized at bootstrap and bumped in the
    // dispatch/combine kernel tails, so the bump is captured into any
    // enclosing CUDA graph and replays correctly.
        void* dev_counter_block = nullptr;
        uint32_t* dispatch_grid_barrier_counter = nullptr;
        uint32_t* combine_grid_barrier_counter = nullptr;
        uint64_t* dev_dispatch_expected_rdma = nullptr;
        uint32_t* dev_dispatch_expected_intra = nullptr;
        uint64_t* dev_combine_expected_rdma = nullptr;
        uint32_t* dev_combine_expected_intra = nullptr;

    // RDMA buffers (multi-node only)
        uint64_t* rdma_inter_node_group_flags;
        uint16_t* rdma_intra_node_red_token;
        float* rdma_intra_node_red_prob;
        uint16_t* combine_rdma_inter_node_group_token;
        float* combine_rdma_inter_node_group_prob;
        uint64_t* combine_rdma_inter_node_group_flags;

    // Pre-registered dispatch buffers (group-level, allocated during Group Create)
    // These are pre-registered with GIN to avoid ~60ms registration overhead during dispatch
        void* token_staging_buffer; // Pre-registered staging buffer for user tokens
        float* dense_prob_buffer; // Pre-registered buffer for sparse→dense prob conversion
        float* scaling_factor_staging_buffer; // Pre-registered staging buffer for FP8 scaling factors

    // Group-scoped routing bitmap; shared by all handles on this group.
        uint8_t* global_routing_map = nullptr;
        size_t global_routing_map_size = 0;

    // Merged IPC buffer (single cudaMalloc for all IPC-shared buffers)
        void* ipc_mega_buffer = nullptr;
        size_t ipc_mega_buffer_size = 0;
        ncclWindow_t intranode_mega_window = {};
        size_t ipc_dispatch_token_offset = 0;
        size_t ipc_dispatch_prob_offset = 0;
        size_t ipc_dispatch_scaling_factor_offset = 0; // FP8 EXTERN: per-block output scales region in the mega buffer
        size_t ipc_combine_token_offset = 0;
        size_t ipc_combine_prob_offset = 0;

        // Merged completion flags
        uint32_t* completion_flags_base = nullptr;
        ncclWindow_t completion_flags_window = {};

        void* host_ptr_block = nullptr; // Single cudaHostAlloc for all pointer arrays

        // Config
        bool initialized;
        bool internode_initialized;
    } ht_buffers;

    // Constructor to properly initialize all members
    ncclEpGroup()
        : comm(nullptr), nRanks(0), rank(0), nNodes(0), ep_workspace(nullptr), cuda_device_id(0), lsa_team_size(0),
          lsa_rank(0), rdma_team_size(0), rdma_rank(0), rdma_buffer(nullptr), rdma_buffer_size_alloc(0), config{},
          num_local_experts(0), max_recv_tokens(0), device_sm_count(0), comm_num_sms(0), prolog_epilog_sms(0),
          preprocess_num_sms(0), ht_em_mode(HtEmMode::kLocalPermute), alloc{}, gpus_per_node(0), rank_in_node(0),
          node_id(0), num_nccl_comms(0), nccl_comms{}, nccl_dev_comms(nullptr), nccl_wins(nullptr),
          num_dispatch_signals(0), clean_barrier_signal_base(0), ht_buffers{} {}
};

// For tensors w/o external window, lazily bind the internal GIN window and offset.
// Tensors created from a user window already carry their own window; resolve
// their local data pointer here once a group/comm is available.
static ncclResult_t
resolveTensorWindowBinding(const ncclEpGroup_t ep_group, ncclEpTensor_t* tensor, uint64_t default_offset) {
    if (tensor == nullptr) {
        return ncclInvalidArgument;
    }

    // Empty tensors have no storage to bind. Leave data == nullptr (and the
    // window unbound) -- consumers must already treat empty tensors as
    // no-ops at the per-element level. Without this guard, the
    // ncclWinGetUserPtr call below would fail on a NULL win_hdl.
    if (tensorIsEmpty(tensor)) {
        return ncclSuccess;
    }

    const bool internode_initialized = ep_group->ht_buffers.internode_initialized;
    if (internode_initialized) {
        if (tensor->win_hdl == ncclWindow_t{}) {
            tensor->win_hdl = ep_group->gin_config.nccl_window;
            tensor->win_offset = default_offset;
        }
    }

    if (tensor->data != nullptr) {
        return ncclSuccess;
    }

    void* base_ptr = nullptr;
    ncclResult_t result = ncclWinGetUserPtr(ep_group->comm, tensor->win_hdl, &base_ptr);
    if (result != ncclSuccess) {
        return result;
    }
    if (base_ptr == nullptr) {
        return ncclInvalidUsage;
    }

    tensor->data = static_cast<void*>(static_cast<char*>(base_ptr) + tensor->win_offset);
    return ncclSuccess;
}

// Const-input overload: decides internally whether the tensor needs mutation
// (win_hdl assignment or data resolution). If so, copies to local_storage,
// resolves there, and sets *out to local_storage. Otherwise sets *out to the
// original tensor unchanged. Call sites always get back a fully resolved pointer
// without managing the copy themselves.
static ncclResult_t resolveTensorWindowBinding(
    const ncclEpGroup_t ep_group,
    const ncclEpTensor_t* tensor,
    ncclEpTensor_t* local_storage,
    uint64_t default_offset,
    const ncclEpTensor_t** out) {
    const bool internode_initialized = ep_group->ht_buffers.internode_initialized;
    const bool needs_win = internode_initialized && tensor->win_hdl == ncclWindow_t{};
    const bool needs_data = tensor->data == nullptr;
    if (!needs_win && !needs_data) {
        *out = tensor;
        return ncclSuccess;
    }
    tensor_temp_copy(local_storage, tensor);
    NCCLCHECK(resolveTensorWindowBinding(ep_group, local_storage, default_offset));
    *out = local_storage;
    return ncclSuccess;
}

// A tensor is on the zero-copy path when its window is user-provided,
// not the group-owned internal GIN window.
static bool tensorUsesExternalWindow(const ncclEpGroup_t ep_group, const ncclEpTensor_t* tensor) {
    // No window yet means a regular tensor - it may be lazily bound to the
    // internal window later, so do not classify it as external.
    if (tensor->win_hdl == ncclWindow_t{}) {
        return false;
    }

    return tensor->win_hdl != ep_group->gin_config.nccl_window;
}

// Build per-LSA-rank pointers for a window-backed tensor so kernels can write
// directly to same-node peer buffers. The local pointer comes from tensor->data;
// peer pointers are resolved from the NCCL window plus the stored offset.
template <typename T>
static ncclResult_t
buildIntranodePtrArray(const ncclEpGroup_t group, const ncclEpTensor_t* tensor, std::vector<T*>& out_ptrs) {
    if (tensor == nullptr || tensor->win_hdl == ncclWindow_t{}) {
        return ncclInvalidUsage;
    }
    ncclEpTensor_t local;
    const ncclEpTensor_t* resolved;
    NCCLCHECK(resolveTensorWindowBinding(group, tensor, &local, 0, &resolved));

    ncclTeam lsa_team = ncclTeamLsa(group->comm);
    out_ptrs.resize(group->lsa_team_size, nullptr);
    out_ptrs[group->lsa_rank] = static_cast<T*>(resolved->data);

    for (int i = 0; i < group->lsa_team_size; i++) {
        if (i == group->lsa_rank) continue;

        int peer_global = ncclTeamRankToWorld(group->comm, lsa_team, i);
        void* peer_ptr = nullptr;
        NCCL_CHECK_RESULT(ncclGetPeerDevicePointer(resolved->win_hdl, resolved->win_offset, peer_global, &peer_ptr));

        out_ptrs[i] = static_cast<T*>(peer_ptr);
    }
    return ncclSuccess;
}

// True when dispatch writes EM staging by em_slot (max_recv_tokens-sized).
// Only kLocalDup/kNvlinkDup with non-zero-copy; reachable via env override since
// auto selection pairs these modes with zero_copy=ON (no staging allocated).
static bool em_staging_indexed_by_em_slot(ncclEpGroup_t group);

// HT Intranode Initialization (adapted for public NCCL APIs)
static ncclResult_t
init_hybridep_intranode(ncclEpGroup_t ep_group, const ncclEpGroupConfig_t* in_config, cudaStream_t stream) {
    ncclComm_t comm = ep_group->comm;
    // HT topology uses NCCL team semantics (rail/lsa)
    int lsa_ranks = ep_group->lsa_team_size;
    int lsa_rank = ep_group->lsa_rank;
    ncclTeam lsa_team = ncclTeamLsa(comm);
    size_t max_token_bytes = ep_group->config.max_token_bytes;
    int num_local_experts = ep_group->num_local_experts;
    int max_recv_tokens = ep_group->max_recv_tokens;

    ep_group->ht_buffers.initialized = false;

    // =========================================================================
    // Phase 1: Allocate all buffers upfront
    // =========================================================================

    // Consolidated intranode mega-buffer: single allocation for all 4 shared buffers.
    // Expert-prob buffers are sized by HT inner-domain cardinality (LSA team size).
    auto align_ipc = [](size_t s) -> size_t { return (s + 255) & ~size_t(255); };

    // max_recv_tokens is the resolved per-rank slot budget (see ncclEpCreateGroup).
    size_t max_output_slots = static_cast<size_t>(max_recv_tokens);
    // Token staging slots per direction; zero_copy elides both regions.
    // kLocalPermute uses rank-major writes, kLocalDup and kNvlinkDup use em_slot.
    const size_t flat_slots = static_cast<size_t>(ep_group->config.max_dispatch_tokens_per_rank) * ep_group->nRanks;
    size_t token_staging_slots = em_staging_indexed_by_em_slot(ep_group) ? max_output_slots : flat_slots;
    ep_group->ht_buffers.token_staging_slots = token_staging_slots;
    size_t expert_output_token_sz = token_staging_slots * max_token_bytes;
    size_t expert_output_prob_sz = max_output_slots * num_local_experts * lsa_ranks * sizeof(float);
    size_t expert_input_token_sz = token_staging_slots * max_token_bytes;
    size_t expert_input_prob_sz = max_output_slots * num_local_experts * lsa_ranks * sizeof(float);

    // FP8 EXTERN per-block output scales (FP32). hidden = max_token_bytes / sizeof(bf16);
    // FP8 EXTERN per-block output scales. Buffer sized assuming kFP8ExternScaleBlockSize (see
    // file-scope constant); callers must use scale_block_size >= kFP8ExternScaleBlockSize.
    size_t expert_output_scaling_factor_sz =
        max_output_slots * ((max_token_bytes / sizeof(uint16_t)) / kFP8ExternScaleBlockSize) * sizeof(float);

    // zero_copy elides both token regions (windowed tensors required).
    const bool zero_copy = ep_group->config.zero_copy == NCCL_EP_ZERO_COPY_ON;
    size_t dispatch_token_aligned = zero_copy ? 0 : align_ipc(expert_output_token_sz);
    size_t dispatch_prob_aligned = align_ipc(expert_output_prob_sz);
    size_t dispatch_sf_aligned = align_ipc(expert_output_scaling_factor_sz);
    size_t combine_token_aligned = zero_copy ? 0 : align_ipc(expert_input_token_sz);
    size_t combine_prob_aligned = align_ipc(expert_input_prob_sz);

    size_t mega_sz = dispatch_token_aligned + dispatch_prob_aligned + dispatch_sf_aligned + combine_token_aligned +
                     combine_prob_aligned;
    NCCL_CHECK_RESULT(ncclMemAlloc(&ep_group->ht_buffers.ipc_mega_buffer, mega_sz));
    ep_group->ht_buffers.ipc_mega_buffer_size = mega_sz;

    uint8_t* mega_base = static_cast<uint8_t*>(ep_group->ht_buffers.ipc_mega_buffer);
    ep_group->ht_buffers.ipc_dispatch_token_offset = 0;
    ep_group->ht_buffers.expert_output_token = zero_copy ? nullptr : mega_base;

    ep_group->ht_buffers.ipc_dispatch_prob_offset = dispatch_token_aligned;
    ep_group->ht_buffers.expert_output_prob = reinterpret_cast<float*>(mega_base + dispatch_token_aligned);

    // FP8 EXTERN output scales region (after token+prob; shifts the combine offsets by dispatch_sf_aligned).
    ep_group->ht_buffers.ipc_dispatch_scaling_factor_offset = dispatch_token_aligned + dispatch_prob_aligned;
    ep_group->ht_buffers.expert_output_scaling_factor =
        reinterpret_cast<float*>(mega_base + dispatch_token_aligned + dispatch_prob_aligned);

    ep_group->ht_buffers.ipc_combine_token_offset =
        dispatch_token_aligned + dispatch_prob_aligned + dispatch_sf_aligned;
    ep_group->ht_buffers.expert_input_token =
        zero_copy ? nullptr :
                    reinterpret_cast<uint16_t*>(
                        mega_base + dispatch_token_aligned + dispatch_prob_aligned + dispatch_sf_aligned);

    ep_group->ht_buffers.ipc_combine_prob_offset =
        dispatch_token_aligned + dispatch_prob_aligned + dispatch_sf_aligned + combine_token_aligned;
    ep_group->ht_buffers.expert_input_prob = reinterpret_cast<float*>(
        mega_base + dispatch_token_aligned + dispatch_prob_aligned + dispatch_sf_aligned + combine_token_aligned);

    // Host pointer arrays indexed by HT local rank within LSA team.
    size_t host_block_sz = sizeof(void*) * lsa_ranks + sizeof(float*) * lsa_ranks // dispatch prob
                           + sizeof(float*) * lsa_ranks // dispatch FP8 scales
                           + sizeof(uint16_t*) * lsa_ranks + sizeof(float*) * lsa_ranks;
    CUDA_CHECK(cudaHostAlloc(&ep_group->ht_buffers.host_ptr_block, host_block_sz, cudaHostAllocMapped));

    uint8_t* hptr = static_cast<uint8_t*>(ep_group->ht_buffers.host_ptr_block);
    ep_group->ht_buffers.dispatch_expert_output_token_buffer_ptrs = reinterpret_cast<void**>(hptr);
    hptr += sizeof(void*) * lsa_ranks;
    ep_group->ht_buffers.dispatch_expert_output_prob_buffer_ptrs = reinterpret_cast<float**>(hptr);
    hptr += sizeof(float*) * lsa_ranks;
    ep_group->ht_buffers.dispatch_expert_output_scaling_factor_buffer_ptrs = reinterpret_cast<float**>(hptr);
    hptr += sizeof(float*) * lsa_ranks;
    ep_group->ht_buffers.combine_expert_input_token_buffer_ptrs = reinterpret_cast<uint16_t**>(hptr);
    hptr += sizeof(uint16_t*) * lsa_ranks;
    ep_group->ht_buffers.combine_expert_input_prob_buffer_ptrs = reinterpret_cast<float**>(hptr);

    // Merged completion flags: allocate on all ranks as we will window register is collective
    NCCL_CHECK_RESULT(
        ncclMemAlloc(reinterpret_cast<void**>(&ep_group->ht_buffers.completion_flags_base), 2 * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemsetAsync(ep_group->ht_buffers.completion_flags_base, 0, 2 * sizeof(uint32_t), stream));
    ep_group->ht_buffers.intra_node_write_completion_flags = ep_group->ht_buffers.completion_flags_base;
    ep_group->ht_buffers.combine_intra_node_write_completion_flags = ep_group->ht_buffers.completion_flags_base + 1;

    // Per-rank (not IPC-shared) device counter block. Layout (offsets in bytes):
    //   [ 0..8)  dev_dispatch_expected_rdma  (uint64_t)
    //   [ 8..16) dev_combine_expected_rdma   (uint64_t)
    //   [16..20) dev_dispatch_expected_intra (uint32_t)
    //   [20..24) dev_combine_expected_intra  (uint32_t)
    //   [24..28) dispatch_grid_barrier_counter (uint32_t)
    //   [28..32) combine_grid_barrier_counter  (uint32_t)
    {
        constexpr size_t kCounterBlockBytes = 32;
        void* block = nullptr;
        CUDA_CHECK(ep_group->alloc.alloc_fn(&block, kCounterBlockBytes, ep_group->alloc.context));
        ep_group->ht_buffers.dev_counter_block = block;
        auto* base = reinterpret_cast<uint8_t*>(block);
        ep_group->ht_buffers.dev_dispatch_expected_rdma = reinterpret_cast<uint64_t*>(base + 0);
        ep_group->ht_buffers.dev_combine_expected_rdma = reinterpret_cast<uint64_t*>(base + 8);
        ep_group->ht_buffers.dev_dispatch_expected_intra = reinterpret_cast<uint32_t*>(base + 16);
        ep_group->ht_buffers.dev_combine_expected_intra = reinterpret_cast<uint32_t*>(base + 20);
        ep_group->ht_buffers.dispatch_grid_barrier_counter = reinterpret_cast<uint32_t*>(base + 24);
        ep_group->ht_buffers.combine_grid_barrier_counter = reinterpret_cast<uint32_t*>(base + 28);

        // Initialize the expected counters to the first-invocation value
        // (grid-barrier counters stay at 0). Bootstrap is outside any user
        // CUDA-graph capture, so a one-shot H2D memcpy + stream sync is safe.
        const bool is_single_node = (ep_group->nNodes == 1);
        const uint64_t init_rdma = is_single_node ? 0ull : 1ull;
        const uint32_t init_intra = static_cast<uint32_t>(ep_group->lsa_team_size);
        uint8_t host_init[kCounterBlockBytes] = {0};
        std::memcpy(host_init + 0, &init_rdma, sizeof(init_rdma));
        std::memcpy(host_init + 8, &init_rdma, sizeof(init_rdma));
        std::memcpy(host_init + 16, &init_intra, sizeof(init_intra));
        std::memcpy(host_init + 20, &init_intra, sizeof(init_intra));
        CUDA_CHECK(cudaMemcpyAsync(block, host_init, kCounterBlockBytes, cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    // =========================================================================
    // Phase 2: Register windows for shared intranode regions
    // Consolidated registration: mega buffer (token+prob+combine) & completion flags
    // =========================================================================
    // Register the mega buffer
    NCCL_CHECK_RESULT(ncclCommWindowRegister(
        comm,
        ep_group->ht_buffers.ipc_mega_buffer,
        mega_sz,
        &ep_group->ht_buffers.intranode_mega_window,
        NCCL_WIN_COLL_SYMMETRIC));

    // Register the completion flags
    NCCL_CHECK_RESULT(ncclCommWindowRegister(
        comm,
        ep_group->ht_buffers.completion_flags_base,
        2 * sizeof(uint32_t),
        &ep_group->ht_buffers.completion_flags_window,
        NCCL_WIN_COLL_SYMMETRIC));

    // =========================================================================
    // Phase 3: Resolve LSA-team peer pointers from NCCL windows.
    // Indexed by HT local rank (LSA team rank)
    // =========================================================================

    for (int i = 0; i < lsa_ranks; i++) {
        if (i == lsa_rank) {
            ep_group->ht_buffers.dispatch_expert_output_token_buffer_ptrs[i] = ep_group->ht_buffers.expert_output_token;
            ep_group->ht_buffers.dispatch_expert_output_prob_buffer_ptrs[i] = ep_group->ht_buffers.expert_output_prob;
            ep_group->ht_buffers.dispatch_expert_output_scaling_factor_buffer_ptrs[i] =
                ep_group->ht_buffers.expert_output_scaling_factor;
            ep_group->ht_buffers.combine_expert_input_token_buffer_ptrs[i] = ep_group->ht_buffers.expert_input_token;
            ep_group->ht_buffers.combine_expert_input_prob_buffer_ptrs[i] = ep_group->ht_buffers.expert_input_prob;
        } else {
            int peer_global = ncclTeamRankToWorld(comm, lsa_team, i);
            void* peer_base = nullptr;
            NCCL_CHECK_RESULT(
                ncclGetPeerDevicePointer(ep_group->ht_buffers.intranode_mega_window, 0, peer_global, &peer_base));
            uint8_t* pb = static_cast<uint8_t*>(peer_base);
            ep_group->ht_buffers.dispatch_expert_output_token_buffer_ptrs[i] =
                zero_copy ? nullptr : pb + ep_group->ht_buffers.ipc_dispatch_token_offset;
            ep_group->ht_buffers.dispatch_expert_output_prob_buffer_ptrs[i] =
                reinterpret_cast<float*>(pb + ep_group->ht_buffers.ipc_dispatch_prob_offset);
            ep_group->ht_buffers.dispatch_expert_output_scaling_factor_buffer_ptrs[i] =
                reinterpret_cast<float*>(pb + ep_group->ht_buffers.ipc_dispatch_scaling_factor_offset);
            ep_group->ht_buffers.combine_expert_input_token_buffer_ptrs[i] =
                zero_copy ? nullptr : reinterpret_cast<uint16_t*>(pb + ep_group->ht_buffers.ipc_combine_token_offset);
            ep_group->ht_buffers.combine_expert_input_prob_buffer_ptrs[i] =
                reinterpret_cast<float*>(pb + ep_group->ht_buffers.ipc_combine_prob_offset);
        }
    }

    // Merged completion flags: resolve rank0 pointer from window.
    if (lsa_rank != 0) {
        int node_rank0_global = ncclTeamRankToWorld(comm, lsa_team, 0);
        void* ptr = nullptr;
        NCCL_CHECK_RESULT(
            ncclGetPeerDevicePointer(ep_group->ht_buffers.completion_flags_window, 0, node_rank0_global, &ptr));
        ep_group->ht_buffers.intra_node_write_completion_flags = static_cast<uint32_t*>(ptr);
        ep_group->ht_buffers.combine_intra_node_write_completion_flags = static_cast<uint32_t*>(ptr) + 1;
    }

    ep_group->ht_buffers.initialized = true;
    CUDA_CHECK(cudaDeviceSynchronize());

    return ncclSuccess;
}

// HT Intranode Cleanup
static ncclResult_t destroy_hybridep_intranode(ncclEpGroup_t ep_group) {
    if (!ep_group->ht_buffers.initialized) return ncclSuccess;

    if (ep_group->ht_buffers.intranode_mega_window != ncclWindow_t{}) {
        NCCL_CHECK_RESULT(ncclCommWindowDeregister(ep_group->comm, ep_group->ht_buffers.intranode_mega_window));
        ep_group->ht_buffers.intranode_mega_window = {};
    }
    if (ep_group->ht_buffers.completion_flags_window != ncclWindow_t{}) {
        NCCL_CHECK_RESULT(ncclCommWindowDeregister(ep_group->comm, ep_group->ht_buffers.completion_flags_window));
        ep_group->ht_buffers.completion_flags_window = {};
    }

    // Free consolidated intranode mega-buffer (replaces 4 individual cudaFree calls)
    if (ep_group->ht_buffers.ipc_mega_buffer) {
        NCCL_CHECK_RESULT(ncclMemFree(ep_group->ht_buffers.ipc_mega_buffer));
        ep_group->ht_buffers.ipc_mega_buffer = nullptr;
        ep_group->ht_buffers.ipc_mega_buffer_size = 0;
        ep_group->ht_buffers.expert_output_token = nullptr;
        ep_group->ht_buffers.expert_output_prob = nullptr;
        ep_group->ht_buffers.expert_output_scaling_factor = nullptr;
        ep_group->ht_buffers.expert_input_token = nullptr;
        ep_group->ht_buffers.expert_input_prob = nullptr;
    }
    // Free the consolidated counter block (grid barriers + expected counters).
    if (ep_group->ht_buffers.dev_counter_block) {
        ep_group->alloc.free_fn(ep_group->ht_buffers.dev_counter_block, ep_group->alloc.context);
        ep_group->ht_buffers.dev_counter_block = nullptr;
        ep_group->ht_buffers.dispatch_grid_barrier_counter = nullptr;
        ep_group->ht_buffers.combine_grid_barrier_counter = nullptr;
        ep_group->ht_buffers.dev_dispatch_expected_rdma = nullptr;
        ep_group->ht_buffers.dev_dispatch_expected_intra = nullptr;
        ep_group->ht_buffers.dev_combine_expected_rdma = nullptr;
        ep_group->ht_buffers.dev_combine_expected_intra = nullptr;
    }

    // Free merged completion flags local allocation
    if (ep_group->ht_buffers.completion_flags_base) {
        NCCL_CHECK_RESULT(ncclMemFree(ep_group->ht_buffers.completion_flags_base));
        ep_group->ht_buffers.completion_flags_base = nullptr;
    }
    ep_group->ht_buffers.intra_node_write_completion_flags = nullptr;
    ep_group->ht_buffers.combine_intra_node_write_completion_flags = nullptr;

    // Free consolidated host pointer block
    if (ep_group->ht_buffers.host_ptr_block) {
        cudaFreeHost(ep_group->ht_buffers.host_ptr_block);
        ep_group->ht_buffers.host_ptr_block = nullptr;
    }

    ep_group->ht_buffers.initialized = false;
    return ncclSuccess;
}

// CUDACHECK_RET macro for CUDA calls in functions returning ncclResult_t
#ifndef CUDACHECK_RET
#define CUDACHECK_RET(cmd) \
    do { \
        cudaError_t err = cmd; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error %s:%d '%s'\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            return ncclInternalError; \
        } \
    } while (0)
#endif

// Constants for GIN configuration
static constexpr int HYBRIDEP_GIN_MAX_CONTEXTS = 32;
static constexpr int HYBRIDEP_GIN_CTXS_PER_COMM = 4;
static constexpr int MAX_BARRIER_SESSIONS = 32;

static ncclResult_t
init_hybridep_internode(ncclEpGroup_t ep_group, const ncclEpGroupConfig_t* in_config, cudaStream_t stream) {
    // Initialize using public NCCL APIs
    if (!in_config || !ep_group) {
        fprintf(stderr, "init_hybridep_internode: null config or ep_group\n");
        return ncclInvalidArgument;
    }

    int rdma_team_size = ep_group->rdma_team_size;
    // HT internode uses NCCL team semantics: outer domain=rail, inner domain=lsa.
    int lsa_team_size = ep_group->lsa_team_size;
    ep_group->ht_buffers.internode_initialized = false;

    if (rdma_team_size <= 1) {
        // Single HT outer-domain node — no internode RDMA, but the LSA guard needs a minimal devComm.
        ep_group->gin_config.num_dcomms = 1;
        ep_group->gin_config.dcomms = new ncclDevComm_t[1];
        ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
        reqs.lsaBarrierCount = HYBRIDEP_DISPATCH_NUM_OF_BLOCKS + 1; // dispatch per-block [0,NB) + elected combine [NB]
        NCCLCHECK(ncclDevCommCreate(ep_group->comm, &reqs, &ep_group->gin_config.dcomms[0]));
        CUDACHECK_RET(cudaMalloc(
            reinterpret_cast<void**>(&ep_group->gin_config.d_dcomms),
            sizeof(ncclDevComm_t) * ep_group->gin_config.num_dcomms));
        CUDACHECK_RET(cudaMemcpy(
            ep_group->gin_config.d_dcomms,
            ep_group->gin_config.dcomms,
            sizeof(ncclDevComm_t) * ep_group->gin_config.num_dcomms,
            cudaMemcpyHostToDevice));
        return ncclSuccess;
    }

    // =========================================================================
    // Phase 1: All local allocations (no collectives)
    // ncclMemAlloc + buffer partitioning moved here from after ncclDevCommCreate
    // to remove them from the collective critical path.
    // =========================================================================

    constexpr size_t GIN_ALIGNMENT = 4096;
    auto align_size = [](size_t sz, size_t alignment) { return (sz + alignment - 1) & ~(alignment - 1); };

    // Chunk-aligned stride for the inter-node RDMA buffers, resolved once at group
    // creation (ep_group->ht_aligned_max_tokens / ht_tokens_per_chunk); the
    // dispatch/combine kernels must use the same stride and chunk size.
    const int ht_max_tokens = ep_group->ht_aligned_max_tokens;
    const int ht_tokens_per_chunk = ep_group->ht_tokens_per_chunk;
    size_t rdma_intra_node_red_token_sz = align_size(
        static_cast<size_t>(ht_max_tokens) * (rdma_team_size - 1) * ep_group->config.max_token_bytes,
        GIN_ALIGNMENT);
    size_t combine_rdma_inter_node_group_token_sz = rdma_intra_node_red_token_sz;
    size_t rdma_intra_node_red_prob_sz = align_size(
        static_cast<size_t>(ht_max_tokens) * (rdma_team_size - 1) * (ep_group->num_local_experts * lsa_team_size) *
            sizeof(float),
        GIN_ALIGNMENT);
    size_t combine_rdma_inter_node_group_prob_sz = rdma_intra_node_red_prob_sz;

    int max_chunks_per_rank = (ht_max_tokens + ht_tokens_per_chunk - 1) / ht_tokens_per_chunk;
    size_t flags_sz =
        align_size(static_cast<size_t>(rdma_team_size - 1) * max_chunks_per_rank * sizeof(uint64_t), GIN_ALIGNMENT);
    // RDMA sync-guard: NUM_LSA_TEAMS uint64 internal-buffer readiness slots per direction.
    size_t guard_sz = align_size(static_cast<size_t>(rdma_team_size) * sizeof(uint64_t), GIN_ALIGNMENT);
    size_t token_staging_sz = align_size(
        static_cast<size_t>(ep_group->config.max_dispatch_tokens_per_rank) * ep_group->config.max_token_bytes,
        GIN_ALIGNMENT);
    size_t dense_prob_sz = align_size(
        static_cast<size_t>(ep_group->config.max_dispatch_tokens_per_rank) * ep_group->config.num_experts *
            sizeof(float),
        GIN_ALIGNMENT);
    size_t scaling_factor_staging_sz =
        align_size(static_cast<size_t>(ep_group->config.max_dispatch_tokens_per_rank) * sizeof(float), GIN_ALIGNMENT);

    size_t bytes_per_token_entry = ep_group->config.max_token_bytes;
    size_t bytes_per_prob_entry = (ep_group->num_local_experts * lsa_team_size) * sizeof(float);
    // FP8 scale-factor entry: per-128-bf16-element scale. Internal to the bf16->fp8 quantizer
    // (max_token_bytes / sizeof(bf16) / 128 = max_token_bytes / 256 floats).
    size_t bytes_per_sf_entry = (ep_group->config.max_token_bytes / 256) * sizeof(float);
    size_t bytes_per_entry = bytes_per_token_entry + bytes_per_prob_entry + bytes_per_sf_entry;
    size_t rdma_send_staging_sz = align_size(
        static_cast<size_t>(rdma_team_size - 1) * ep_group->config.max_dispatch_tokens_per_rank * bytes_per_entry,
        GIN_ALIGNMENT);
    size_t rdma_recv_packed_sz = align_size(
        static_cast<size_t>(rdma_team_size - 1) * ep_group->config.max_dispatch_tokens_per_rank * bytes_per_entry,
        GIN_ALIGNMENT);

    size_t total_gin_buffer_size = 0;
    total_gin_buffer_size += rdma_intra_node_red_token_sz;
    total_gin_buffer_size += combine_rdma_inter_node_group_token_sz;
    total_gin_buffer_size += rdma_intra_node_red_prob_sz;
    total_gin_buffer_size += combine_rdma_inter_node_group_prob_sz;
    total_gin_buffer_size += flags_sz * 2;
    total_gin_buffer_size += guard_sz * 2;
    total_gin_buffer_size += token_staging_sz;
    total_gin_buffer_size += dense_prob_sz;
    total_gin_buffer_size += scaling_factor_staging_sz;
    total_gin_buffer_size += rdma_send_staging_sz;
    total_gin_buffer_size += rdma_recv_packed_sz;

    NCCLCHECK(ncclMemAlloc(&ep_group->gin_config.gin_base_ptr, total_gin_buffer_size));

    // Partition the buffer into individual regions
    uint8_t* ptr = reinterpret_cast<uint8_t*>(ep_group->gin_config.gin_base_ptr);
    size_t offset = 0;

    ep_group->ht_buffers.rdma_intra_node_red_token = reinterpret_cast<uint16_t*>(ptr + offset);
    offset += rdma_intra_node_red_token_sz;

    ep_group->ht_buffers.combine_rdma_inter_node_group_token = reinterpret_cast<uint16_t*>(ptr + offset);
    offset += combine_rdma_inter_node_group_token_sz;

    ep_group->ht_buffers.rdma_intra_node_red_prob = reinterpret_cast<float*>(ptr + offset);
    offset += rdma_intra_node_red_prob_sz;

    ep_group->ht_buffers.combine_rdma_inter_node_group_prob = reinterpret_cast<float*>(ptr + offset);
    offset += combine_rdma_inter_node_group_prob_sz;

    ep_group->ht_buffers.rdma_inter_node_group_flags = reinterpret_cast<uint64_t*>(ptr + offset);
    CUDACHECK_RET(cudaMemset(ep_group->ht_buffers.rdma_inter_node_group_flags, 0, flags_sz));
    offset += flags_sz;

    ep_group->ht_buffers.combine_rdma_inter_node_group_flags = reinterpret_cast<uint64_t*>(ptr + offset);
    CUDACHECK_RET(cudaMemset(ep_group->ht_buffers.combine_rdma_inter_node_group_flags, 0, flags_sz));
    offset += flags_sz;

    // RDMA sync-guard regions (dispatch, combine): addressed by window+offset only.
    CUDACHECK_RET(cudaMemset(ptr + offset, 0, guard_sz));
    offset += guard_sz;
    CUDACHECK_RET(cudaMemset(ptr + offset, 0, guard_sz));
    offset += guard_sz;

    ep_group->ht_buffers.token_staging_buffer = reinterpret_cast<void*>(ptr + offset);
    offset += token_staging_sz;

    ep_group->ht_buffers.dense_prob_buffer = reinterpret_cast<float*>(ptr + offset);
    offset += dense_prob_sz;

    ep_group->ht_buffers.scaling_factor_staging_buffer = reinterpret_cast<float*>(ptr + offset);
    offset += scaling_factor_staging_sz;

    offset += rdma_send_staging_sz;

    // Calculate offsets for kernel mr_info
    size_t cur_offset = 0;
    ep_group->gin_config.rdma_intra_node_red_token_offset = cur_offset;
    cur_offset += rdma_intra_node_red_token_sz;

    ep_group->gin_config.combine_rdma_inter_node_group_token_offset = cur_offset;
    cur_offset += combine_rdma_inter_node_group_token_sz;

    ep_group->gin_config.rdma_intra_node_red_prob_offset = cur_offset;
    cur_offset += rdma_intra_node_red_prob_sz;

    ep_group->gin_config.combine_rdma_inter_node_group_prob_offset = cur_offset;
    cur_offset += combine_rdma_inter_node_group_prob_sz;

    cur_offset += flags_sz * 2;

    ep_group->gin_config.dispatch_guard_offset = cur_offset;
    cur_offset += guard_sz;
    ep_group->gin_config.combine_guard_offset = cur_offset;
    cur_offset += guard_sz;

    ep_group->gin_config.token_staging_offset = cur_offset;
    cur_offset += token_staging_sz;

    ep_group->gin_config.dense_prob_offset = cur_offset;
    cur_offset += dense_prob_sz;

    ep_group->gin_config.scaling_factor_staging_offset = cur_offset;
    cur_offset += scaling_factor_staging_sz;

    ep_group->gin_config.rdma_send_staging_offset = cur_offset;
    cur_offset += rdma_send_staging_sz;

    ep_group->gin_config.rdma_inter_node_group_packed_offset = cur_offset;
    cur_offset += rdma_recv_packed_sz;

    // =========================================================================
    // Phase 2: configure internode GIN resources
    // =========================================================================
    // Verify that configured HT node count matches NCCL rail team size.
    ncclTeam rail_team = ncclTeamRail(ep_group->comm);
    if (rail_team.nRanks != rdma_team_size) {
        fprintf(stderr, "[HT GIN] Error: rail team size (%d) must equal number of LSA domains (%d)\n", rail_team.nRanks,
                rdma_team_size);
        return ncclInvalidUsage;
    }

    int qps_per_rank = ep_group->config.num_qp_per_rank;
    int min_required_ctx = ep_group->comm_num_sms * HYBRIDEP_DISPATCH_N2N_WARPS;
    if (qps_per_rank == 0) qps_per_rank = min_required_ctx;
    if (qps_per_rank < min_required_ctx) {
        fprintf(stderr, "[HT GIN] Error: num_qp_per_rank(%d) must be >= %d for dedicated N2N warp contexts\n",
                qps_per_rank, min_required_ctx);
        return ncclInvalidUsage;
    }
    ep_group->gin_config.qps_per_rank = qps_per_rank;
    ep_group->gin_config.num_comms = 1;
    ep_group->gin_config.num_ctx_per_comm = qps_per_rank;

    int dispatch_signals = lsa_team_size * rdma_team_size * max_chunks_per_rank;
    int combine_signals = lsa_team_size * rdma_team_size * max_chunks_per_rank;
    int streaming_tail_signals = rdma_team_size * rdma_team_size * lsa_team_size * max_chunks_per_rank;
    int streaming_head_signals = rdma_team_size * rdma_team_size * lsa_team_size;
    ep_group->gin_config.num_total_signals =
        dispatch_signals + combine_signals + streaming_tail_signals + streaming_head_signals + MAX_BARRIER_SESSIONS;
    ep_group->gin_config.signals_base = 0;
    ep_group->gin_config.combine_signal_offset = dispatch_signals;
    ep_group->gin_config.signals_tail_base = dispatch_signals + combine_signals;

    // =========================================================================
    // Phase 3: comm setup (DevCommCreate + WindowRegister)
    // =========================================================================
    ep_group->gin_config.num_dcomms = 1;
    ep_group->gin_config.dcomms = new ncclDevComm_t[1];

    {
        ncclCommProperties_t props = NCCL_COMM_PROPERTIES_INITIALIZER;
        NCCL_CHECK_RESULT(ncclCommQueryProperties(ep_group->comm, &props));
        if (props.railedGinType == NCCL_GIN_TYPE_NONE) {
            fprintf(stderr, "[HT GIN] Error: NCCL EP internode requires GIN, but GIN is not supported\n");
            return ncclInvalidUsage;
        }
    }

    {
        ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
        reqs.ginSignalCount = ep_group->gin_config.num_total_signals;
        reqs.ginConnectionType = NCCL_GIN_CONNECTION_RAIL;
        reqs.ginContextCount = ep_group->gin_config.num_ctx_per_comm;
        reqs.ginQueueDepth = 3 * ht_tokens_per_chunk + 1;
        // LSA barriers for the HT sync-guard: per-block dispatch [0, NUM_OF_BLOCKS) + one
        // for the elected combine-tail block [NUM_OF_BLOCKS]. NUM_OF_BLOCKS <= HYBRIDEP_DISPATCH_NUM_OF_BLOCKS.
        reqs.lsaBarrierCount = HYBRIDEP_DISPATCH_NUM_OF_BLOCKS + 1; // dispatch per-block [0,NB) + elected combine [NB]
        NCCLCHECK(ncclDevCommCreate(ep_group->comm, &reqs, &ep_group->gin_config.dcomms[0]));
    }

    CUDACHECK_RET(cudaMalloc(
        reinterpret_cast<void**>(&ep_group->gin_config.d_dcomms),
        sizeof(ncclDevComm_t) * ep_group->gin_config.num_dcomms));
    CUDACHECK_RET(cudaMemcpy(
        ep_group->gin_config.d_dcomms,
        ep_group->gin_config.dcomms,
        sizeof(ncclDevComm_t) * ep_group->gin_config.num_dcomms,
        cudaMemcpyHostToDevice));

    // WindowRegister
    NCCLCHECK(ncclCommWindowRegister(
        ep_group->comm,
        ep_group->gin_config.gin_base_ptr,
        total_gin_buffer_size,
        &ep_group->gin_config.nccl_window,
        0));

    ep_group->ht_buffers.internode_initialized = true;
    return ncclSuccess;
}

static ncclResult_t destroy_hybridep_internode(ncclEpGroup_t ep_group) {
    if (!ep_group->ht_buffers.internode_initialized) return ncclSuccess;

    // =========================================================================
    // Cleanup using public NCCL APIs
    // =========================================================================

    // Destroy device communicator
    if (ep_group->gin_config.dcomms != nullptr) {
        ncclResult_t res = ncclDevCommDestroy(ep_group->comm, &ep_group->gin_config.dcomms[0]);
        if (res != ncclSuccess) {
            fprintf(stderr, "[HT GIN] Warning: Failed to destroy device comm: %s\n", ncclGetErrorString(res));
        }
        delete[] ep_group->gin_config.dcomms;
        ep_group->gin_config.dcomms = nullptr;
    }
    // Free device memory for dcomms
    if (ep_group->gin_config.d_dcomms != nullptr) {
        cudaFree(ep_group->gin_config.d_dcomms);
        ep_group->gin_config.d_dcomms = nullptr;
    }

    // Deregister the window
    if (ep_group->gin_config.gin_base_ptr != nullptr) {
        ncclCommWindowDeregister(ep_group->comm, ep_group->gin_config.nccl_window);
        ep_group->gin_config.nccl_window = {};
    }

    // Free the single GIN buffer (contains all RDMA regions)
    if (ep_group->gin_config.gin_base_ptr != nullptr) {
        ncclResult_t res = ncclMemFree(ep_group->gin_config.gin_base_ptr);
        if (res != ncclSuccess) {
            fprintf(stderr, "[HT GIN] Warning: Failed to free GIN memory: %s\n", ncclGetErrorString(res));
        }
        ep_group->gin_config.gin_base_ptr = nullptr;

        // Clear buffer pointers (they pointed into gin_base_ptr)
        ep_group->ht_buffers.rdma_intra_node_red_token = nullptr;
        ep_group->ht_buffers.combine_rdma_inter_node_group_token = nullptr;
        ep_group->ht_buffers.rdma_intra_node_red_prob = nullptr;
        ep_group->ht_buffers.combine_rdma_inter_node_group_prob = nullptr;
        ep_group->ht_buffers.rdma_inter_node_group_flags = nullptr;
        ep_group->ht_buffers.combine_rdma_inter_node_group_flags = nullptr;
        ep_group->ht_buffers.token_staging_buffer = nullptr;
        ep_group->ht_buffers.dense_prob_buffer = nullptr;
        ep_group->ht_buffers.scaling_factor_staging_buffer = nullptr;
    }

    ep_group->gin_config.num_comms = 0;

    ep_group->ht_buffers.internode_initialized = false;
    return ncclSuccess;
}

static cudaError_t default_alloc_fn(void** ptr, size_t size, void* /*context*/) {
    return cudaMalloc(ptr, size);
}
static cudaError_t default_free_fn(void* ptr, void* /*context*/) {
    return cudaFree(ptr);
}

ncclResult_t ncclEpGetVersion(int* version) {
    if (version == nullptr) return ncclInvalidArgument;
    *version = NCCL_EP_VERSION_CODE;
    return ncclSuccess;
}

// Pre-process the string so that running "strings" on the lib can quickly reveal the version.
// CUDA_MAJOR/CUDA_MINOR are injected at build time by makefiles/common.mk and by the top-level
// CMakeLists.txt — they reflect the CUDA toolkit this library was BUILT against.
#define NCCL_EP_STR2(v) #v
#define NCCL_EP_STR(v) NCCL_EP_STR2(v)
#define NCCL_EP_VERSION_STRING \
    "NCCL EP version " NCCL_EP_STR(NCCL_EP_MAJOR) "." NCCL_EP_STR(NCCL_EP_MINOR) "." NCCL_EP_STR( \
        NCCL_EP_PATCH) "+cuda" NCCL_EP_STR(CUDA_MAJOR) "." NCCL_EP_STR(CUDA_MINOR)

static void showVersion() {
    static std::once_flag once;
    std::call_once(once, []() {
        // No EP group exists at library-load time, so read a transient config
        // just to honor NCCL_EP_DEBUG for the banner. Per-group state is
        // populated later in ncclEpCreateGroup.
        ncclEpEnvConfig cfg;
        nccl_ep_env_init(&cfg);
        if (nccl_ep_env_flag_on(cfg.debug)) {
            fprintf(stderr, "%s\n", NCCL_EP_VERSION_STRING);
        }
    });
}

// Print the version banner when libnccl_ep.so is loaded (respects NCCL_EP_DEBUG).
__attribute__((constructor)) static void nccl_ep_lib_init() {
    showVersion();
}

ncclResult_t ncclEpCreateGroup(ncclEpGroup_t* out_ep_group, ncclComm_t comm, const ncclEpGroupConfig_t* in_config) {
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    // Parameter validation
    assert(out_ep_group != nullptr);
    int nRanks;
    assert(comm != nullptr && ncclCommCount(comm, &nRanks) == ncclSuccess && nRanks > 0);
    EP_REQUIRE_STRUCT(in_config); // null-checks and size-validates in_config
    if (in_config->version != NCCL_EP_API_VERSION) {
        fprintf(
            stderr,
            "NCCL EP WARN: ncclEpGroupConfig_t.version=%u, library API_VERSION=%u; "
            "behavior may differ across versions.\n",
            in_config->version,
            (unsigned)NCCL_EP_API_VERSION);
    }
    assert(
        (in_config->algorithm == NCCL_EP_ALGO_LOW_LATENCY || in_config->algorithm == NCCL_EP_ALGO_HIGH_THROUGHPUT) &&
        "ncclEpCreateGroup: invalid algorithm, supported: low_latency, high_throughput");

    bool low_latency_mode = (in_config->algorithm == NCCL_EP_ALGO_LOW_LATENCY);
    bool hybridep_mode = (in_config->algorithm == NCCL_EP_ALGO_HIGH_THROUGHPUT);
    assert(in_config->num_experts > 0 && "ncclEpCreateGroup: num_experts must be greater than 0");
    assert(in_config->max_token_bytes > 0 && "ncclEpCreateGroup: max_token_bytes must be greater than 0");
    assert(
        !(in_config->algorithm == NCCL_EP_ALGO_LOW_LATENCY && in_config->max_dispatch_tokens_per_rank == 0) &&
        "ncclEpCreateGroup: max_dispatch_tokens_per_rank must be greater than 0 for low latency mode");
    assert(
        !(in_config->algorithm == NCCL_EP_ALGO_HIGH_THROUGHPUT && in_config->max_dispatch_tokens_per_rank == 0) &&
        "ncclEpCreateGroup: max_dispatch_tokens_per_rank must be set for HT backend");
    // HT max_dispatch_tokens_per_rank need not be a chunk multiple; the inter-node
    // stride is chunk-aligned internally (ep_group->ht_aligned_max_tokens).
    // Create teams: LSA and Rail
    ncclTeam lsa_team = ncclTeamLsa(comm);
    ncclTeam rail_team = ncclTeamRail(comm);

    // Allocate EP group structure
    void* raw_memory = malloc(sizeof(ncclEpGroup));
    assert(raw_memory != nullptr && "Failed to malloc for ncclEpGroup");
    *out_ep_group = new (raw_memory) ncclEpGroup();
    ncclEpGroup_t ep_group = *out_ep_group;

    // Store configuration
    ep_group->comm = comm;
    ep_group->config = *in_config;

    // Populate this group's environment configuration once.
    nccl_ep_env_init(&ep_group->env);

    ep_group->alloc.alloc_fn = default_alloc_fn;
    ep_group->alloc.free_fn = default_free_fn;
    if (in_config->alloc.alloc_fn || in_config->alloc.free_fn) {
        if (!(in_config->alloc.alloc_fn && in_config->alloc.free_fn)) {
            fprintf(stderr, "NCCL EP: Failed to create group: Both alloc and free callbacks must be provided\n");
            return ncclInvalidUsage;
        }
        ep_group->alloc.alloc_fn = in_config->alloc.alloc_fn;
        ep_group->alloc.free_fn = in_config->alloc.free_fn;
        ep_group->alloc.context = in_config->alloc.context;
    }

    NCCL_CHECK_RESULT(ncclCommCount(comm, &ep_group->nRanks));
    NCCL_CHECK_RESULT(ncclCommUserRank(comm, &ep_group->rank));
    NCCL_CHECK_RESULT(ncclCommCuDevice(comm, &ep_group->cuda_device_id));

    // Stamp this process's rank into the env config
    nccl_ep_env_set_rank(&ep_group->env, ep_group->rank);

    // Dump the resolved environment configuration once if requested
    if (nccl_ep_env_verbose(ep_group->env)) nccl_ep_env_print(ep_group->env);

    CUDA_CHECK(cudaSetDevice(ep_group->cuda_device_id));
    cudaDeviceProp device_prop = {};
    CUDA_CHECK(cudaGetDeviceProperties(&device_prop, ep_group->cuda_device_id));
    ep_group->device_sm_count = device_prop.multiProcessorCount;

    // Resolve SM counts for EP kernels (dispatch, combine, preprocessing)
    if (in_config->max_num_sms == NCCL_EP_AUTO) {
        if (in_config->algorithm == NCCL_EP_ALGO_HIGH_THROUGHPUT) {
            ep_group->comm_num_sms = NCCL_EP_HT_DFLT_NUM_SMS;
        } else {
            ep_group->comm_num_sms = ep_group->device_sm_count;
        }
        ep_group->prolog_epilog_sms = ep_group->device_sm_count;
        ep_group->preprocess_num_sms = ep_group->device_sm_count;
    } else {
        if (in_config->max_num_sms > ep_group->device_sm_count) {
            fprintf(stderr, "Error: NCCL EP requires max_num_sms <= device_sm_count\n");
            return ncclInvalidUsage;
        }
        if (in_config->algorithm == NCCL_EP_ALGO_LOW_LATENCY) {
            // This reflects the current limitation of the LL backend
            // TODO: validate that the limitation is valid and if need - relax it in a follow-up fix
            constexpr int llMaxWarpGroupsLimit = 14;
            const int numWarpGroups = (in_config->num_experts + (in_config->max_num_sms - 1)) / in_config->max_num_sms;
            if (numWarpGroups > llMaxWarpGroupsLimit) {
                const int required_sms = (in_config->num_experts + (llMaxWarpGroupsLimit - 1)) / llMaxWarpGroupsLimit;
                fprintf(stderr,
                        "Error: insufficient 'max_num_sms'. In Low-Latency mode, NCCL EP requires at least %d SMs\n",
                        required_sms);
                return ncclInvalidUsage;
            }
        }
        ep_group->comm_num_sms = in_config->max_num_sms;
        ep_group->prolog_epilog_sms = in_config->max_num_sms;
        ep_group->preprocess_num_sms = in_config->max_num_sms;
    }

    // Apply an env-provided SM-count override, validated against [1, device_sm_count].
    // Out-of-range values are warned about and the resolved default is kept.
    const unsigned int dev_sms = ep_group->device_sm_count;
    auto apply_sms_override = [dev_sms, &env = ep_group->env](const ncclEpEnvVar& var, unsigned int& target) {
        if (!var.is_set) return;
        if (var.value.ul >= 1 && var.value.ul <= dev_sms) {
            target = static_cast<unsigned int>(var.value.ul);
        } else {
            fprintf(stderr, "[nccl_ep] %s=%lu out of range (must be in [1, %u]); using %u\n", var.name, var.value.ul,
                    dev_sms, target);
            // Dump the full env configuration to help diagnose the misconfig.
            nccl_ep_env_print(env);
        }
    };
    apply_sms_override(ep_group->env.comm_num_sms, ep_group->comm_num_sms);
    apply_sms_override(ep_group->env.prolog_epilog_sms, ep_group->prolog_epilog_sms);
    apply_sms_override(ep_group->env.preprocess_num_sms, ep_group->preprocess_num_sms);

    // Determine number of nodes by gathering hostnames and counting unique ones
    constexpr size_t HOSTNAME_LEN = 256;
    std::vector<char> all_hostnames(HOSTNAME_LEN * ep_group->nRanks, 0);
    gethostname(all_hostnames.data() + ep_group->rank * HOSTNAME_LEN, HOSTNAME_LEN);
    ncclAllGatherHost(all_hostnames.data(), HOSTNAME_LEN, ep_group->rank, ep_group->nRanks, comm, stream);
    std::set<std::string> unique_hosts;
    for (int i = 0; i < ep_group->nRanks; ++i) {
        unique_hosts.insert(std::string(all_hostnames.data() + i * HOSTNAME_LEN));
    }
    ep_group->nNodes = static_cast<int>(unique_hosts.size());

    ep_group->num_local_experts = ep_group->config.num_experts / ep_group->nRanks;
    // HT: caller must provide a slot budget >= max_dispatch_tokens_per_rank (NCCL_EP_AUTO==0).
    EP_HOST_ASSERT(
        !(in_config->algorithm == NCCL_EP_ALGO_HIGH_THROUGHPUT && ep_group->config.max_recv_tokens_per_rank == 0) &&
        "ncclEpCreateGroup: HT mode requires max_recv_tokens_per_rank > 0");
    EP_HOST_ASSERT(
        !(in_config->algorithm == NCCL_EP_ALGO_HIGH_THROUGHPUT &&
          ep_group->config.max_recv_tokens_per_rank < ep_group->config.max_dispatch_tokens_per_rank) &&
        "ncclEpCreateGroup: HT mode requires max_recv_tokens_per_rank >= max_dispatch_tokens_per_rank");
    // LL auto-budget (nRanks * max_dispatch_tokens_per_rank, layout-agnostic).
    if (ep_group->config.max_recv_tokens_per_rank == 0) {
        ep_group->config.max_recv_tokens_per_rank = ep_group->nRanks * ep_group->config.max_dispatch_tokens_per_rank;
    }
    ep_group->max_recv_tokens = static_cast<int>(ep_group->config.max_recv_tokens_per_rank);

    // Collective: all ranks must agree on the resolved budget (IPC buffers are sized from it).
    {
        std::vector<unsigned int> all_budgets(ep_group->nRanks, 0);
        all_budgets[ep_group->rank] = ep_group->config.max_recv_tokens_per_rank;
        ncclAllGatherHost(all_budgets.data(), sizeof(unsigned int), ep_group->rank, ep_group->nRanks, comm, stream);
        for (int r = 1; r < ep_group->nRanks; ++r) {
            EP_HOST_ASSERT(
                all_budgets[r] == all_budgets[0] &&
                "ncclEpCreateGroup: max_recv_tokens_per_rank must be identical across ranks");
        }
    }

    // Apply default values for auto-configured fields (when set to NCCL_EP_AUTO)
    if (ep_group->config.num_channels == NCCL_EP_AUTO) {
        ep_group->config.num_channels = 10;
    }

    if (ep_group->config.num_qp_per_rank == NCCL_EP_AUTO) {
        ep_group->config.num_qp_per_rank = ep_group->comm_num_sms * HYBRIDEP_DISPATCH_N2N_WARPS;
    }

    // Resolve timeout_cycles: env var > config field > compile-time default
    {
        int dev;
        int clock_khz_int;
        CUDA_CHECK(cudaGetDevice(&dev));
        CUDA_CHECK(cudaDeviceGetAttribute(&clock_khz_int, cudaDevAttrClockRate, dev));
        uint64_t clock_khz = static_cast<uint64_t>(clock_khz_int);

        uint64_t resolved = NUM_TIMEOUT_CYCLES;
        const char* source = "compile-time default";
        const uint64_t env_ms = static_cast<uint64_t>(ep_group->env.timeout_ms.value.ul);
        // Only a positive timeout overrides the default.
        const bool have_env_ms = ep_group->env.timeout_ms.is_set && env_ms > 0;

        if (have_env_ms) {
            resolved = clock_khz * 1000ULL * env_ms / 1000ULL;
            source = "NCCL_EP_TIMEOUT_MS env var";
            if (ep_group->config.timeout_ns != 0 && ep_group->rank == 0)
                fprintf(stderr, "NCCL EP: NCCL_EP_TIMEOUT_MS=%lu overrides config.timeout_ns=%lu\n",
                        (unsigned long)env_ms, (unsigned long)ep_group->config.timeout_ns);
        } else if (ep_group->config.timeout_ns != 0) {
            resolved = clock_khz * 1000ULL * (ep_group->config.timeout_ns / 1000000ULL) / 1000ULL;
            source = "config.timeout_ns";
        }

        ep_group->timeout_cycles = resolved;
        if (ep_group->rank == 0) {
            uint64_t timeout_ms = resolved / (clock_khz * 1000ULL / 1000ULL);
            char env_str[32];
            if (have_env_ms) snprintf(env_str, sizeof(env_str), "%llu", (unsigned long long)env_ms);
            else snprintf(env_str, sizeof(env_str), "unset");
            fprintf(stderr, "NCCL EP: using timeout=%llums (env=%s, config.timeout_ns=%llu, source=%s)\n",
                    (unsigned long long)timeout_ms, env_str, (unsigned long long)ep_group->config.timeout_ns, source);
        }
    }

    // Physical node properties. rank_in_node must lie in [0, gpus_per_node)
    // so the peer-access loop below can skip the self-device. Using the
    // within-comm rank (rather than the physical cuda_device_id) keeps this
    // invariant when multiple EP comms colocate on one physical node (e.g.
    // DP × EP mesh where ranks 4..7 form a second EP group on the same box).
    ep_group->gpus_per_node = ep_group->nRanks / ep_group->nNodes;
    ep_group->rank_in_node = ep_group->rank % ep_group->gpus_per_node;
    ep_group->node_id = ep_group->rank / ep_group->gpus_per_node;
    ep_group->lsa_team_size = lsa_team.nRanks;
    ep_group->lsa_rank = lsa_team.rank;
    if (hybridep_mode) {
        // HT uses rail-domain decomposition.
        ep_group->rdma_team_size = rail_team.nRanks;
        ep_group->rdma_rank = rail_team.rank;
    } else {
        // Preserve legacy semantics.
        // TODO: are we using this in LL?
        ep_group->rdma_team_size = ep_group->nRanks;
        ep_group->rdma_rank = ep_group->rank;
    }
    if (hybridep_mode) {
        assert(
            ep_group->rdma_team_size > 0 && ep_group->lsa_team_size > 0 &&
            "ncclEpCreateGroup: invalid HT team cardinalities");
        assert(
            ep_group->rdma_team_size * ep_group->lsa_team_size == ep_group->nRanks &&
            "ncclEpCreateGroup: HT requires rdma_team_size * lsa_team_size == nRanks");
    }

    // HT EM mode: env override wins, else auto-pick from (zero_copy, lsa_team_size).
    //   non zero_copy           -> kLocalPermute (FLAT dispatch + permute kernels)
    //   zero_copy, lsa > 1      -> kNvlinkDup    (sender duplicates per-expert over NVLink)
    //   zero_copy, lsa == 1     -> kLocalDup     (receiver-side fan-out via local_dup)
    {
        const bool want_local_dup = nccl_ep_env_flag_on(ep_group->env.ht_em_local_dup);
        const bool want_nvlink_dup = nccl_ep_env_flag_on(ep_group->env.ht_em_nvlink_dup);
        if (want_local_dup && want_nvlink_dup) {
            fprintf(stderr, "NCCL EP: NCCL_EP_HT_EM_LOCAL_DUP and NCCL_EP_HT_EM_NVLINK_DUP are mutually exclusive\n");
            return ncclInvalidUsage;
        }
        if (want_nvlink_dup) {
            ep_group->ht_em_mode = ncclEpGroup::HtEmMode::kNvlinkDup;
        } else if (want_local_dup) {
            ep_group->ht_em_mode = ncclEpGroup::HtEmMode::kLocalDup;
        } else if (in_config->zero_copy == NCCL_EP_ZERO_COPY_ON) {
            ep_group->ht_em_mode =
                (ep_group->lsa_team_size > 1) ? ncclEpGroup::HtEmMode::kNvlinkDup : ncclEpGroup::HtEmMode::kLocalDup;
        } else {
            ep_group->ht_em_mode = ncclEpGroup::HtEmMode::kLocalPermute;
        }
    }

    ep_group->rdma_buffer = nullptr;

    CUDA_CHECK(ep_group->alloc.alloc_fn(&ep_group->ep_workspace, NUM_WORKSPACE_BYTES, ep_group->alloc.context));
    CUDA_CHECK(cudaMemsetAsync(ep_group->ep_workspace, 0, NUM_WORKSPACE_BYTES, stream));

    ncclCommProperties_t props = NCCL_COMM_PROPERTIES_INITIALIZER;
    NCCL_CHECK_RESULT(ncclCommQueryProperties(ep_group->comm, &props));
    if (!props.deviceApiSupport) {
        fprintf(stderr, "Error: NCCL EP requires NCCL Device API support, but Device API is not supported\n");
        return ncclInvalidUsage;
    }

    // Initialize HT intranode buffers (windows, completion flags, etc.)
    if (hybridep_mode) {
        // Resolve the dispatch/combine tokens-per-chunk for this group. RDMA
        // (multi-node) configs use the tuned 64-token default; LSA-only configs
        // use a grid-proportional size so one chunk is one wave across all SMs
        // (NUM_OF_TOKENS_PER_GROUP tokens per SM). Either may be overridden by
        // NCCL_EP_TOKENS_PER_CHUNK. The chunk must be a multiple of 32 (warp width;
        // also covers the uint4 routing-map-load and token-group granularities), so
        // non-conforming values are rounded up with a warning.
        auto round_up_32 = [](int v) { return (v + 31) & ~31; };
        int chunk =
            (ep_group->rdma_team_size > 1) ?
                HT_TOKENS_PER_CHUNK_RDMA_DEFAULT :
                round_up_32(HYBRIDEP_COMBINE_NUM_OF_TOKENS_PER_GROUP * static_cast<int>(ep_group->comm_num_sms));
        if (ep_group->env.tokens_per_chunk.is_set && ep_group->env.tokens_per_chunk.value.ul > 0) {
            const int requested = static_cast<int>(ep_group->env.tokens_per_chunk.value.ul);
            chunk = round_up_32(requested);
            if (chunk != requested) {
                fprintf(stderr, "[nccl_ep] %s=%d rounded up to %d (must be a multiple of 32)\n",
                        ep_group->env.tokens_per_chunk.name, requested, chunk);
            }
        }
        ep_group->ht_tokens_per_chunk = chunk;
        // Chunk-aligned inter-node RDMA stride; also the max-tokens template arg for
        // the dispatch/combine kernels (guarantees MAX_NUM_OF_TOKENS_PER_RANK % chunk == 0).
        ep_group->ht_aligned_max_tokens = ((ep_group->config.max_dispatch_tokens_per_rank + chunk - 1) / chunk) * chunk;

        NCCL_CHECK_RESULT(init_hybridep_intranode(ep_group, in_config, stream));
        NCCL_CHECK_RESULT(init_hybridep_internode(ep_group, in_config, stream));

        // Group-scoped routing bitmap shared by all handles on this group.
        const size_t routing_bytes = static_cast<size_t>(ep_group->nRanks) *
                                     ep_group->config.max_dispatch_tokens_per_rank *
                                     ((ep_group->config.num_experts + 7) / 8);
        CUDA_CHECK(ep_group->alloc.alloc_fn(
            reinterpret_cast<void**>(&ep_group->ht_buffers.global_routing_map),
            routing_bytes,
            ep_group->alloc.context));
        ep_group->ht_buffers.global_routing_map_size = routing_bytes;
        // No init memset: AllGather in preprocessing overwrites every byte the kernel reads.
    }

    if (low_latency_mode) {
        ep_group->num_nccl_comms = 0; // no split comms created

        // Cleaning up any pending CUDA error
        CUDA_CHECK(cudaGetLastError());

        // Create device communicator on ep_group->comm with all GIN contexts.
        // This depends only on group-level parameters (num_experts, nRanks),
        // not on the per-handle layout/num_topk, so it stays at group time.
        ncclDevComm_t* nccl_dev_comms_host = new ncclDevComm_t[1];
        nccl_dev_comms_host[0] = ncclDevComm_t{};
        ep_group->num_dispatch_signals = ep_group->num_local_experts * ep_group->nRanks;
        int num_total_signals = ep_group->num_dispatch_signals;
        ep_group->clean_barrier_signal_base = 2 * num_total_signals;

        ncclCommProperties_t props = NCCL_COMM_PROPERTIES_INITIALIZER;
        NCCLCHECK(ncclCommQueryProperties(ep_group->comm, &props));
        if (props.nLsaTeams > 1 && props.ginType == NCCL_GIN_TYPE_NONE) {
            fprintf(stderr, "[LL] Error: NCCL EP requires GIN, but GIN is not supported\n");
            return ncclInvalidUsage;
        }

        ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
        if (props.nLsaTeams > 1) {
            reqs.ginContextCount = ep_group->config.num_qp_per_rank; // all contexts in single comm
            // Signal layout: combine [0, N), dispatch [N, 2N), clean barrier [2N]
            reqs.ginSignalCount = 2 * num_total_signals + 1;
            reqs.ginForceEnable = true;
            reqs.ginConnectionType = NCCL_GIN_CONNECTION_FULL;
            reqs.worldGinBarrierCount = 1;
        }
        NCCL_CHECK_RESULT(ncclDevCommCreate(ep_group->comm, &reqs, &nccl_dev_comms_host[0]));

        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ep_group->nccl_dev_comms), sizeof(ncclDevComm_t)));
        CUDA_CHECK(
            cudaMemcpy(ep_group->nccl_dev_comms, nccl_dev_comms_host, sizeof(ncclDevComm_t), cudaMemcpyHostToDevice));

        delete[] nccl_dev_comms_host;
        nccl_dev_comms_host = nullptr;

        if (ep_group->config.enable_mask) {
            // Allocate the cross-rank sync buffer used by clean_low_latency_buffer.
            // Its size depends only on nRanks (a group-level constant), so it is
            // sized and allocated here once, in its own registered NCCL window.
            // Keeping it out of rdma_buffer means lazy/grow reallocation of
            // rdma_buffer doesn't have to touch it, and ncclEpMaskClean can run
            // even before any LL handle has been created.
            const size_t sync_buffer_bytes = ((static_cast<size_t>(ep_group->nRanks) * sizeof(int) + 127) / 128) * 128;
            NCCL_CHECK_RESULT(ncclMemAlloc(&ep_group->sync_buffer, sync_buffer_bytes));
            CUDA_CHECK(cudaMemset(ep_group->sync_buffer, 0, sync_buffer_bytes));
            ncclWindow_t sync_win_host;
            NCCL_CHECK_RESULT(
                ncclCommWindowRegister(ep_group->comm, ep_group->sync_buffer, sync_buffer_bytes, &sync_win_host, 0));
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ep_group->sync_window), sizeof(ncclWindow_t)));
            CUDA_CHECK(cudaMemcpy(ep_group->sync_window, &sync_win_host, sizeof(ncclWindow_t), cudaMemcpyHostToDevice));
        }

        // If the user passed an explicit rdma_buffer_size (anything other than
        // NCCL_EP_AUTO), honor it verbatim now. ll_init_handle will then only
        // verify the requested layout fits; growth is not performed. With
        // NCCL_EP_AUTO, allocation is deferred until ll_init_handle so it can
        // size the buffer with the actual (layout, num_topk).
        if (ep_group->config.rdma_buffer_size != NCCL_EP_AUTO) {
            NCCLCHECK(ll_resize_rdma_buffer(ep_group, ep_group->config.rdma_buffer_size));
        }
    }

    // Allocate mask buffer and async error flag for active-mask support
    if (ep_group->config.enable_mask && ep_group->config.algorithm == NCCL_EP_ALGO_LOW_LATENCY) {
        size_t mask_bytes = ep_group->nRanks * sizeof(int);
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ep_group->mask_buffer), mask_bytes));
        // Initialize all ranks as active (1 = active, 0 = masked/failed)
        std::vector<int> all_active(ep_group->nRanks, 1);
        CUDA_CHECK(
            cudaMemcpyAsync(ep_group->mask_buffer, all_active.data(), mask_bytes, cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(
            cudaHostAlloc(reinterpret_cast<void**>(&ep_group->async_error_flag), sizeof(int), cudaHostAllocMapped));
        *ep_group->async_error_flag = 0;
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaStreamDestroy(stream));
    return ncclSuccess;
}

ncclResult_t ncclEpGroupDestroy(ncclEpGroup_t ep_group) {
    if (ep_group == nullptr) {
        return ncclSuccess;
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    // Clean up HT intranode resources
    if (ep_group->config.algorithm == NCCL_EP_ALGO_HIGH_THROUGHPUT && ep_group->ht_buffers.initialized) {
        destroy_hybridep_intranode(ep_group);
    }
    // Clean up HT internode resources (GIN deregistration must happen before ncclCommDestroy)
    if (ep_group->config.algorithm == NCCL_EP_ALGO_HIGH_THROUGHPUT && ep_group->ht_buffers.internode_initialized) {
        destroy_hybridep_internode(ep_group);
    }
    // Clean up mask buffer and async error flag
    if (ep_group->mask_buffer != nullptr) {
        CUDA_CHECK(cudaFree(ep_group->mask_buffer));
        ep_group->mask_buffer = nullptr;
    }
    if (ep_group->async_error_flag != nullptr) {
        CUDA_CHECK(cudaFreeHost(ep_group->async_error_flag));
        ep_group->async_error_flag = nullptr;
    }

    // Clean up workspace memory
    if (ep_group->ep_workspace != nullptr) {
        CUDA_CHECK(ep_group->alloc.free_fn(ep_group->ep_workspace, ep_group->alloc.context));
    }
    if (ep_group->ht_buffers.global_routing_map != nullptr) {
        CUDA_CHECK(ep_group->alloc.free_fn(ep_group->ht_buffers.global_routing_map, ep_group->alloc.context));
        ep_group->ht_buffers.global_routing_map = nullptr;
    }

    // Clean up RDMA resources (single-comm path: 1 window, 1 devcomm on ep_group->comm).
    // Gate on algorithm only — config.rdma_buffer_size may be NCCL_EP_AUTO
    // (== 0) for LL groups where the buffer is sized lazily.
    if (NCCL_EP_ALGO_LOW_LATENCY == ep_group->config.algorithm) {
        CUDA_CHECK(cudaDeviceSynchronize());
        cudaStream_t stream;
        CUDA_CHECK(cudaStreamCreate(&stream));
        NCCL_CHECK_RESULT(ncclBarrier(ep_group->comm, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaStreamDestroy(stream));

        // Deregister single NCCL window if it was ever registered. The
        // rdma_buffer and its window are allocated lazily on the first LL
        // handle init, so a group that was created but never had an LL
        // handle still has nccl_wins == nullptr here.
        if (ep_group->nccl_wins != nullptr) {
            ncclWindow_t win_host;
            CUDA_CHECK(cudaMemcpy(&win_host, ep_group->nccl_wins, sizeof(ncclWindow_t), cudaMemcpyDeviceToHost));
            NCCL_CHECK_RESULT(ncclCommWindowDeregister(ep_group->comm, win_host));
            CUDA_CHECK(cudaFree(ep_group->nccl_wins));
            ep_group->nccl_wins = nullptr;
        }

        // Free RDMA buffer (after window deregistered)
        if (ep_group->rdma_buffer) {
            NCCL_CHECK_RESULT(ncclMemFree(ep_group->rdma_buffer));
            ep_group->rdma_buffer = nullptr;
        }

        // Tear down the standalone sync buffer + its window.
        if (ep_group->sync_window != nullptr) {
            ncclWindow_t sync_win_host;
            CUDA_CHECK(cudaMemcpy(&sync_win_host, ep_group->sync_window, sizeof(ncclWindow_t), cudaMemcpyDeviceToHost));
            NCCL_CHECK_RESULT(ncclCommWindowDeregister(ep_group->comm, sync_win_host));
            CUDA_CHECK(cudaFree(ep_group->sync_window));
            ep_group->sync_window = nullptr;
        }
        if (ep_group->sync_buffer != nullptr) {
            NCCL_CHECK_RESULT(ncclMemFree(ep_group->sync_buffer));
            ep_group->sync_buffer = nullptr;
        }

        // Destroy single NCCL device communicator (copy back from device, destroy on ep_group->comm)
        ncclDevComm_t dc_host;
        CUDA_CHECK(cudaMemcpy(&dc_host, ep_group->nccl_dev_comms, sizeof(ncclDevComm_t), cudaMemcpyDeviceToHost));
        NCCL_CHECK_RESULT(ncclDevCommDestroy(ep_group->comm, &dc_host));
        CUDA_CHECK(cudaFree(ep_group->nccl_dev_comms));
        ep_group->nccl_dev_comms = nullptr;

        // No split comms to destroy (using ep_group->comm directly)
    }
    // Invoke destructor explicitly (placement new was used)
    ep_group->~ncclEpGroup();

    // Free the group structure
    free(ep_group);

    return ncclSuccess;
}

struct ncclEpHandle {
    ncclEpGroup_t group;

    ncclEpLayout_t layout;

    // User-owned (do not free). LL reads directly; HT uses cached hybridep.topk_idx.
    ncclEpTensor_t topk_idx;
    int num_tokens, num_topk;
    // True once `num_tokens` has been authoritatively populated by
    // `ncclEpUpdateHandle` (or `ncclEpCreateHandle` via its inner update call).
    // Distinguishes a legitimate zero-token configuration from the
    // "not yet bound" state -- the dispatch / combine lazy branches must not
    // treat a real 0 as "uninitialised" and re-fetch sizes[0] from inputs,
    // since the input tensors on zero-token ranks are empty (and now legally
    // carry data == nullptr per tensorHasBinding's empty-tensor relaxation).
    bool num_tokens_set;

    bool cached_mode;
    int num_scales;
    int hidden_int4;

    union {
        struct {
            // packed tensors for LL (descriptors only; data pointers reference handle_mem)
            ncclEpTensor_t expert_recv_source_indices;
            ncclEpTensor_t expert_dispatch_layout;

            // Persistent backing storage for the per-tensor `sizes` arrays above.
            size_t expert_recv_source_indices_sizes[1];
            size_t expert_dispatch_layout_sizes[2];

            // Backing storage for the two tensors above. Layout matches ll_handle_mem_size().
            void* handle_mem;
            bool owns_handle_mem;

            int buffer_idx = 0;

            std::function<void(unsigned int)> continue_fn;
            nccl_ep::LowLatencyLayout layout;
        } ll;
        struct {
            // Global routing map lives on ep_group->ht_buffers (group-scoped).

            // =================================================================================
            // PREPROCESSING OUTPUTS - Computed once per iteration, used by dispatch & combine
            // =================================================================================

            // Sparse-to-dense map: maps each (token, source_rank) pair to its position in
            // the destination rank's expert buffer. Used by NVLink (intra-node) warps.
            // Value of -1 indicates token is not routed to that rank.
            // dtype: int32_t
            // layout: [num_nodes * max_dispatch_tokens_per_rank, num_ranks_per_node]
            // usage: dispatch S2G warp group, combine G2S warp group
            // lifetime: valid after metadata_preprocessing, constant within iteration
            // vs NCCL HT: no direct equivalent.
            //   - HT: uses is_token_in_rank (local tokens only) + atomics to compute positions
            //   - HT: precomputes positions for ALL nodes' tokens, -1 sentinel for not routed
            int32_t* sparse_to_dense_map;

            // RDMA-to-attention map: boolean mask indicating which tokens this node needs
            // to RECEIVE from RDMA (inter-node). Indexed by [node_id, token_id].
            // Primarily used during combine to know which remote tokens to wait for.
            // dtype: bool
            // layout: [num_nodes, max_dispatch_tokens_per_rank_padded_to_16]
            //        (padding to 16 required for TMA alignment)
            // usage: dispatch G2S warp (polling), combine inter-node G2S/reduction warps
            // lifetime: valid after metadata_preprocessing, constant within iteration
            // vs NCCL HT: inverse perspective of is_token_in_rank.
            //   - is_token_in_rank: outbound - "where do MY tokens go?" [my_token, dest_rank]
            //   - rdma_to_attn_map: inbound - "which remote tokens do I receive?" [src_node, token]
            bool* rdma_to_attn_map;

            // Attention-to-RDMA map: boolean mask indicating which local tokens need to be
            // SENT via RDMA (inter-node) to each remote node.
            // Only allocated when num_nodes > 1.
            // dtype: bool
            // layout: [max_dispatch_tokens_per_rank, num_nodes - 1]
            // usage: dispatch N2N (RDMA) warp group
            // lifetime: valid after metadata_preprocessing, constant within iteration
            // vs NCCL HT: closest equivalent to is_token_in_rank for inter-node RDMA.
            //   - is_token_in_rank: per-rank granularity [num_tokens, num_ranks]
            //   - attn_to_rdma_map: per-node granularity [num_tokens, num_nodes-1] (RDMA only)
            bool* attn_to_rdma_map;

            // Per-token per-rank bitmask cache produced during preprocessing.
            // dtype: rank_mask_t<ceil(lsa_team_size/64)> (one uint64_t word per 64 ranks)
            // layout: [num_nodes * max_send_tokens_per_rank * ranks_per_node]
            void* token_rank_mask;

            // Local expert routing map: per-expert routing for tokens in this rank's buffer.
            // Used by subsequent expert MLP layers to route tokens to correct experts.
            // dtype: bool
            // layout: [max_recv_tokens, experts_per_rank]
            //        where max_recv_tokens = num_ranks * max_dispatch_tokens_per_rank
            // usage: passed to expert computation layers (not used by dispatch/combine directly)
            // lifetime: valid after metadata_preprocessing, constant within iteration
            // vs NCCL HT: similar purpose to num_tokens_per_expert but more detailed
            bool* local_expert_routing_map;

            // Number of tokens routed to local experts (total across all local experts).
            // Each token counted once even if routed to multiple local experts.
            // dtype: int32_t
            // layout: [1]
            // usage: buffer sizing, iteration control
            // lifetime: valid after metadata_preprocessing
            int32_t* num_tokens_for_experts;

            // =================================================================================
            // CONVERSION BUFFERS - Pre-allocated to avoid dispatch/combine-time malloc
            // =================================================================================

            // Dense prob buffer: shared scratch for sparse↔dense conversions
            // dtype: float
            // layout: [max_dispatch_tokens_per_rank, num_experts]
            // usage:
            //   - dispatch forward: sparse→dense input topk_weights conversion
            //   - combine backward: dense→sparse output prob conversion
            // lifetime: allocated at handle creation, freed at handle destroy
            // note: dispatch and combine are sequential, so one buffer suffices
            // For multi-node: points to group-level pre-registered buffer
            // For single-node: handle-owned buffer
            float* dense_prob_buffer;

            // Token staging buffer: pre-registered buffer to avoid GIN registration during dispatch
            // User tokens are copied here during dispatch, then this buffer is used for RDMA
            // dtype: uint16_t (bf16) or uint8_t (fp8)
            // layout: [max_dispatch_tokens_per_rank, hidden]
            // usage: copy user tokens → use for inter-node RDMA
            // lifetime: group-owned (allocated in Group Create, freed in Group Destroy)
            void* token_staging_buffer; // Pointer to group-level buffer (not handle-owned)

            // Scaling factor staging buffer: pre-registered buffer for FP8 scaling factors
            // User scaling factors are copied here during dispatch, then this buffer is used for RDMA
            // dtype: float
            // layout: [max_dispatch_tokens_per_rank]
            // usage: copy user scaling factors → use for inter-node RDMA
            // lifetime: group-owned (allocated in Group Create, freed in Group Destroy)
            float* scaling_factor_staging_buffer; // Pointer to group-level buffer (not handle-owned)

            // RDMA inter-node group flags: atomic completion flags for each remote node.
            // Remote ranks increment via RDMA atomic fetch-add to signal chunk completion.
            // Only allocated when num_nodes > 1.
            // dtype: uint64_t
            // layout: [num_nodes - 1]
            // usage: dispatch N2N warp (signaling), dispatch G2S warp (polling)
            // lifetime: reset to 0 at init, incremented by remote RDMA atomics
            // uint64_t* rdma_inter_node_group_flags;

            // Per-handle preprocessing block (single allocation for all preprocessing buffers)
            void* preprocessing_block;
            bool owns_handle_mem; // false = caller-owned (user path); destroy skips free
            size_t preprocessing_zero_region_size;
            size_t preprocessing_s2d_size;
            void* preprocessing_scan_tmp;

            // Expert-major fields (alignment set in InitHandle; offsets/counts set in UpdateHandle)
            size_t dispatch_output_per_expert_alignment;
            int64_t* expert_token_offsets; // [experts_per_rank] written by remap kernel
            int32_t* per_expert_counts_active; // alias to authoritative counts buffer

            // EM local-fanout dup-groups (allocated iff mode == kLocalDup). Written by EM
            // scan; consumed by local_dup (post-dispatch) and local_reduce (pre-combine).
            int32_t* emuf_group_buf;
            int32_t* emuf_group_count;
            int emuf_group_stride;
            int emuf_max_groups;

            // Cached user topk_idx; persists across dispatch/combine.
            // Native width matches handle->topk_idx.datatype (int32 or int64).
            void* topk_idx;

            // FLAT-dispatch + local-permute scratch (EM-permute path only).
            // flat2em_slot_map, recv_topk_weights_flat and token_to_recv_slot
            // are all null when em_local_permute_enabled() is false.
            // In em-permute mode sparse_to_dense_map is FLAT-shape
            // (inner=lsa_team_size).
            //
            // Mapping from the local-node FLAT slot index to the Expert-major
            // slot in the output tensor. Shape [max_recv_tokens, top_k]
            // (invalid positions are -1). Written by em_scan_kernel during
            // UpdateHandle, read by the local-permute kernels.
            int32_t* flat2em_slot_map;
            // Per-FLAT-slot topk weights; populated by dense_to_sparse_prob.
            float* recv_topk_weights_flat;
            // [num_total_attn_tokens] FLAT recv slot per global attention
            // token; -1 when the token has no local-rank hit. Populated by
            // scan_impl_flat in em-permute mode; consumed by em_scan_kernel
            // to populate the recv-indexed em_slot table.
            int32_t* token_to_recv_slot;

        } hybridep;
    };

    ncclEpHandle()
        : group(nullptr), layout(NCCL_EP_LAYOUT_UNSET), topk_idx(NCCL_EP_TENSOR_INIT), num_tokens(0), num_topk(0),
          num_tokens_set(false), cached_mode(false), num_scales(0), hidden_int4(0) {
        constexpr size_t union_size = std::max(sizeof(ll), sizeof(hybridep));
        memset(static_cast<void*>(&ll), 0, union_size);
    }

    ~ncclEpHandle() {}
};

static bool is_internode_available(ncclEpGroup_t ep_group) {
    // True when there are multiple HT outer-domain nodes
    return ep_group->rdma_team_size > 1;
}

// EM + mode == kLocalPermute. HT-only by design; zero_copy == OFF is enforced
// by the HtEmMode resolver (mode == kLocalPermute => zero_copy != ON).
static bool em_local_permute_enabled(ncclEpGroup_t group, ncclEpLayout_t layout) {
    return layout == NCCL_EP_LAYOUT_EXPERT_MAJOR && group->ht_em_mode == ncclEpGroup::HtEmMode::kLocalPermute;
}
static inline bool em_local_permute_enabled(ncclEpGroup_t group, ncclEpHandle_t handle) {
    return em_local_permute_enabled(group, handle->layout);
}

// EM + mode == kLocalDup. HT-only by design.
static bool em_local_dup_active(ncclEpGroup_t group, ncclEpLayout_t layout) {
    return layout == NCCL_EP_LAYOUT_EXPERT_MAJOR && group->ht_em_mode == ncclEpGroup::HtEmMode::kLocalDup;
}

static bool em_staging_indexed_by_em_slot(ncclEpGroup_t group) {
    return group->ht_em_mode == ncclEpGroup::HtEmMode::kNvlinkDup ||
           group->ht_em_mode == ncclEpGroup::HtEmMode::kLocalDup;
}

// Returns the total buffer size (in bytes) for a LL handle_mem block.
// Used by ncclEpHandleMemSize (public) and ncclEpInitHandle (internal).
static size_t ll_handle_mem_size(ncclEpGroup_t ep_group, int num_topk) {
    auto align256 = [](size_t s) -> size_t { return (s + 255) & ~size_t(255); };
    const size_t local_experts = static_cast<size_t>(ep_group->num_local_experts);
    const size_t nRanks = static_cast<size_t>(ep_group->nRanks);
    const size_t max_tokens = static_cast<size_t>(ep_group->config.max_dispatch_tokens_per_rank);
    // Layout: nRanks per-rank counts + nRanks * max_tokens * (num_topk+1) token entries
    size_t sz_recv_src = align256(nRanks * (1 + static_cast<size_t>(num_topk + 1) * max_tokens) * sizeof(int32_t));
    size_t sz_dispatch = align256(local_experts * nRanks * sizeof(int64_t));
    return sz_recv_src + sz_dispatch;
}

// All individual buffer sizes for a HT handle_mem block plus derived totals.
// Single source of truth shared by ht_handle_mem_size() and ht_init_handle().
struct HtBlockLayout {
    // global_routing_map is group-scoped (ep_group->ht_buffers); not part of this block.
    size_t sz_r2a, sz_a2r, sz_ler, sz_ntfe;
    size_t sz_s2d, sz_rank_mask, sz_scan_tmp, sz_prob;
    size_t sz_topk_idx; // cached topk_idx
    size_t sz_pec_active; // EM only
    size_t sz_eto; // EM only
    // Local-dup local-fanout scratch.
    size_t sz_emuf_group_buf, sz_emuf_group_count;
    int emuf_group_stride; // row width actually allocated; <= experts_per_rank
    int emuf_max_groups; // capacity in rows; kernel must not exceed
    // EM-permute scratch (handle mem).
    size_t sz_flat2em_slot_map;
    size_t sz_recv_topk_weights_flat;
    size_t sz_token_to_recv_slot;
    size_t zero_region, no_memset_region, total;

    static HtBlockLayout compute(ncclEpGroup_t ep_group, ncclEpLayout_t layout, int num_topk = 0) {
        auto align256 = [](size_t s) -> size_t { return (s + 255) & ~size_t(255); };
        const int num_experts = ep_group->config.num_experts;
        const int max_tokens = ep_group->config.max_dispatch_tokens_per_rank;
        const int lsa_team_size = ep_group->lsa_team_size;
        const int rdma_team_size = ep_group->rdma_team_size;
        const int experts_per_rank = ep_group->num_local_experts;
        const int max_recv_tokens = ep_group->max_recv_tokens;
        const int padded_max_tokens = ((max_tokens + 15) / 16) * 16;
        const bool has_expert_major = (layout == NCCL_EP_LAYOUT_EXPERT_MAJOR);
        const bool em_permute = em_local_permute_enabled(ep_group, layout);

        HtBlockLayout L = {};
        L.sz_r2a = align256(static_cast<size_t>(rdma_team_size) * padded_max_tokens * sizeof(bool));
        L.sz_a2r =
            (rdma_team_size > 1) ? align256(static_cast<size_t>(max_tokens) * (rdma_team_size - 1) * sizeof(bool)) : 0;
        L.sz_ler = align256(static_cast<size_t>(ep_group->max_recv_tokens) * experts_per_rank * sizeof(bool));
        L.sz_ntfe = align256(sizeof(int32_t));
        // S2D inner_dim: num_topk for nvlink_dup/local_dup EM (packed rank/slot); lsa_team_size
        // for FLAT and for EM-permute (unified FLAT-shape s2d — em_scan_kernel's
        // EM-shape writes are suppressed in em-permute mode).
        const int s2d_inner_dim = (has_expert_major && !em_permute) ? num_topk : lsa_team_size;
        L.sz_s2d = align256(static_cast<size_t>(rdma_team_size) * max_tokens * s2d_inner_dim * sizeof(int32_t));
        L.sz_rank_mask = align256(
            static_cast<size_t>(rdma_team_size) * max_tokens * lsa_team_size *
            nccl_ep::hybridep::get_rank_mask_elem_size(lsa_team_size));
        // Size for device_sm_count blocks: the upper bound on the preprocessing
        // scan's block count (ep_group->preprocess_num_sms, incl. any
        // NCCL_EP_PREPROCESS_NUM_SMS override) so the buffer always fits.
        L.sz_scan_tmp = align256(
            nccl_ep::hybridep::get_preprocessing_scan_tmp_size(
                static_cast<int>(ep_group->device_sm_count),
                lsa_team_size));
        L.sz_prob = !is_internode_available(ep_group) ?
                        align256(static_cast<size_t>(max_tokens) * num_experts * sizeof(float)) :
                        0;
        L.sz_topk_idx = (num_topk > 0) ? align256(static_cast<size_t>(max_tokens) * num_topk * sizeof(int64_t)) : 0;
        L.sz_pec_active = has_expert_major ? align256(static_cast<size_t>(experts_per_rank) * sizeof(int32_t)) : 0;
        // [experts_per_rank + 1]: em_scan_kernel publishes the EM-padded total at `experts_per_rank` index.
        L.sz_eto = has_expert_major ? align256(static_cast<size_t>(experts_per_rank + 1) * sizeof(int64_t)) : 0;
        // EM local-fanout dup-groups. Bounds: num_groups <= max_recv_tokens / 2
        // (each multi-hit group occupies >= 2 em_slots); row width <= min(num_topk, experts_per_rank).
        const bool emuf_enabled = has_expert_major && ep_group->ht_em_mode == ncclEpGroup::HtEmMode::kLocalDup;
        const int emuf_row_width = (num_topk > 0) ? std::min(num_topk, experts_per_rank) : experts_per_rank;
        const size_t emuf_max_groups = static_cast<size_t>(ep_group->max_recv_tokens) / 2;
        const size_t emuf_group_entries = emuf_enabled ? emuf_max_groups * static_cast<size_t>(emuf_row_width) : 0;
        L.emuf_group_stride = emuf_enabled ? emuf_row_width : 0;
        L.emuf_max_groups = emuf_enabled ? static_cast<int>(emuf_max_groups) : 0;
        L.sz_emuf_group_buf = align256(emuf_group_entries * sizeof(int32_t));
        L.sz_emuf_group_count = emuf_enabled ? align256(sizeof(int32_t)) : 0;
        // EM-permute scratch: only when EM + !zero_copy_on (env-var flip handled at call time).
        L.sz_flat2em_slot_map = (em_permute && num_topk > 0) ?
                                    align256(static_cast<size_t>(max_recv_tokens) * num_topk * sizeof(int32_t)) :
                                    0;
        L.sz_recv_topk_weights_flat = (em_permute && num_topk > 0) ?
                                          align256(static_cast<size_t>(max_recv_tokens) * num_topk * sizeof(float)) :
                                          0;
        // EM-shape LERM mirrors sz_ler size; needs to be zero-initialised so
        // em_scan_kernel can write only the local-rank's em_slot rows.
        // num_total_attn_tokens = max_tokens * lsa_team_size * rdma_team_size.
        L.sz_token_to_recv_slot =
            em_permute ? align256(static_cast<size_t>(max_tokens) * lsa_team_size * rdma_team_size * sizeof(int32_t)) :
                         0;
        L.zero_region = L.sz_r2a + L.sz_a2r + L.sz_ler + L.sz_ntfe;
        L.no_memset_region = L.sz_rank_mask + L.sz_scan_tmp + L.sz_prob + L.sz_topk_idx + L.sz_pec_active + L.sz_eto +
                             L.sz_emuf_group_buf + L.sz_emuf_group_count + L.sz_flat2em_slot_map +
                             L.sz_recv_topk_weights_flat + L.sz_token_to_recv_slot;
        L.total = L.zero_region + L.sz_s2d + L.no_memset_region;
        return L;
    }
};

static size_t ht_handle_mem_size(ncclEpGroup_t ep_group, ncclEpLayout_t layout, int num_topk) {
    return HtBlockLayout::compute(ep_group, layout, num_topk).total;
}

ncclResult_t ncclEpHandleMemSize(
    ncclEpGroup_t ep_group,
    ncclEpLayout_t layout,
    const ncclEpHandleConfig_t* config,
    size_t* size_out,
    int num_topk) {
    assert(ep_group != nullptr && size_out != nullptr);
    EP_OPTIONAL_STRUCT(config);
    EP_HOST_ASSERT(layout != NCCL_EP_LAYOUT_UNSET && "ncclEpHandleMemSize: layout must be set explicitly");
    if (ep_group->config.algorithm == NCCL_EP_ALGO_HIGH_THROUGHPUT) {
        assert(num_topk > 0 && "HT mode requires num_topk > 0 for ncclEpHandleMemSize");
        *size_out = ht_handle_mem_size(ep_group, layout, num_topk);
    } else if (ep_group->config.algorithm == NCCL_EP_ALGO_LOW_LATENCY) {
        assert(num_topk > 0 && "LL mode requires num_topk > 0 for ncclEpHandleMemSize");
        *size_out = ll_handle_mem_size(ep_group, num_topk);
    } else {
        return ncclInvalidUsage;
    }
    return ncclSuccess;
}

// Collectively (re-)size rdma_buffer to `new_size`. Two modes:
//   - First allocation (rdma_buffer == nullptr): allocate, register, stash
//     window handle in a newly-allocated device slot.
//   - Grow (rdma_buffer != nullptr): cross-rank fence, deregister window,
//     free, reallocate, re-register, update existing device slot.
// In both cases the buffer is zeroed and rdma_buffer_size_alloc is updated.
// Must be called by all ranks in the comm with the same new_size.
static ncclResult_t ll_resize_rdma_buffer(ncclEpGroup_t ep_group, size_t new_size) {
    EP_HOST_ASSERT(new_size > ep_group->rdma_buffer_size_alloc);
    // With NCCL_EP_AUTO, config.rdma_buffer_size is 0 (the AUTO sentinel) and
    // imposes no cap. With a user-specified value, the caller (ll_init_handle)
    // rejects layouts that don't fit before reaching here, and the group-time
    // path passes exactly config.rdma_buffer_size.
    EP_HOST_ASSERT(ep_group->config.rdma_buffer_size == NCCL_EP_AUTO || new_size <= ep_group->config.rdma_buffer_size);

    const bool first_alloc = (ep_group->rdma_buffer == nullptr);

    if (!first_alloc) {
        // Cross-rank fence before teardown: cudaDeviceSynchronize drains
        // local work, but a remote rank may still be touching this rank's
        // window via RDMA. ncclMemFree/ncclCommWindowDeregister/
        // ncclMemAlloc/ncclCommWindowRegister are themselves collective,
        // so no further barrier is needed afterward.
        CUDA_CHECK(cudaDeviceSynchronize());
        cudaStream_t stream;
        CUDA_CHECK(cudaStreamCreate(&stream));
        NCCL_CHECK_RESULT(ncclBarrier(ep_group->comm, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaStreamDestroy(stream));

        ncclWindow_t win_host;
        CUDA_CHECK(cudaMemcpy(&win_host, ep_group->nccl_wins, sizeof(ncclWindow_t), cudaMemcpyDeviceToHost));
        NCCL_CHECK_RESULT(ncclCommWindowDeregister(ep_group->comm, win_host));
        NCCL_CHECK_RESULT(ncclMemFree(ep_group->rdma_buffer));
        ep_group->rdma_buffer = nullptr;
    }

    NCCL_CHECK_RESULT(ncclMemAlloc(&ep_group->rdma_buffer, new_size));
    CUDA_CHECK(cudaMemset(ep_group->rdma_buffer, 0, new_size));
    ep_group->rdma_buffer_size_alloc = new_size;

    ncclWindow_t win_host;
    NCCL_CHECK_RESULT(ncclCommWindowRegister(ep_group->comm, ep_group->rdma_buffer, new_size, &win_host, 0));
    if (first_alloc) {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ep_group->nccl_wins), sizeof(ncclWindow_t)));
    }
    CUDA_CHECK(cudaMemcpy(ep_group->nccl_wins, &win_host, sizeof(ncclWindow_t), cudaMemcpyHostToDevice));

    return ncclSuccess;
}

static ncclResult_t
ll_init_handle(ncclEpHandle_t handle, ncclEpGroup_t ep_group, const ncclEpTensor_t* handle_mem, int num_topk) {
    assert(num_topk > 0 && "LL mode requires num_topk > 0 (pass top_k to ncclEpInitHandle)");
    assert(
        (ep_group->config.max_dispatch_tokens_per_rank * ep_group->num_local_experts) % 4 == 0 &&
        "TMA requires the number of tokens to be multiple of 4");

    auto layout = nccl_ep::LowLatencyLayout(
        ep_group->config.max_dispatch_tokens_per_rank,
        ep_group->config.max_token_bytes,
        ep_group->nRanks,
        ep_group->config.num_experts,
        num_topk,
        handle->layout);

    // Ensure rdma_buffer is large enough for this handle's actual layout.
    // - NCCL_EP_AUTO: lazily allocate (first handle) or grow (later handle
    //   needing more, e.g. different layout or larger num_topk). Stored
    //   layouts on other live handles are pure offsets, so reallocation
    //   does not invalidate them.
    // - User-specified size: the buffer was already allocated to that exact
    //   value at group create. Reject the handle if it doesn't fit.
    if (layout.total_bytes > ep_group->rdma_buffer_size_alloc) {
        if (ep_group->config.rdma_buffer_size != NCCL_EP_AUTO) {
            fprintf(
                stderr,
                "ncclEpInitHandle: requested layout needs %zu bytes but user-specified "
                "rdma_buffer_size is %zu bytes; either increase rdma_buffer_size or use "
                "NCCL_EP_AUTO to allow lazy sizing\n",
                layout.total_bytes,
                static_cast<size_t>(ep_group->config.rdma_buffer_size));
            return ncclInvalidUsage;
        }
        const size_t new_size = ((layout.total_bytes + NUM_BUFFER_ALIGNMENT_BYTES - 1) / NUM_BUFFER_ALIGNMENT_BYTES) *
                                NUM_BUFFER_ALIGNMENT_BYTES;
        NCCLCHECK(ll_resize_rdma_buffer(ep_group, new_size));
    }
    assert(layout.total_bytes <= ep_group->rdma_buffer_size_alloc);

    if (handle_mem != nullptr) {
        assert(handle_mem->ndim == 1 && handle_mem->datatype == ncclUint8);
        assert(
            handle_mem->sizes[0] >= ll_handle_mem_size(ep_group, num_topk) &&
            "handle_mem too small; use ncclEpHandleMemSize to query required size");
        handle->ll.handle_mem = handle_mem->data;
        handle->ll.owns_handle_mem = false;
    } else {
        CUDA_CHECK(ep_group->alloc.alloc_fn(
            &handle->ll.handle_mem,
            ll_handle_mem_size(ep_group, num_topk),
            ep_group->alloc.context));
        handle->ll.owns_handle_mem = true;
    }
    char* base = static_cast<char*>(handle->ll.handle_mem);

    const size_t recv_src_count =
        static_cast<size_t>(ep_group->nRanks) *
        (1 + static_cast<size_t>(num_topk + 1) * ep_group->config.max_dispatch_tokens_per_rank);
    {
        handle->ll.expert_recv_source_indices_sizes[0] = recv_src_count;
        handle->ll.expert_recv_source_indices = (ncclEpTensor_t){
            NCCL_EP_TENSOR_INIT_INLINE,
            .ndim = 1,
            .datatype = ncclInt32,
            .data = base,
            .sizes = handle->ll.expert_recv_source_indices_sizes,
        };
    }

    {
        auto align256 = [](size_t s) -> size_t { return (s + 255) & ~size_t(255); };
        const size_t recv_src_bytes = align256(recv_src_count * sizeof(int32_t));
        handle->ll.expert_dispatch_layout_sizes[0] = static_cast<size_t>(ep_group->num_local_experts);
        handle->ll.expert_dispatch_layout_sizes[1] = static_cast<size_t>(ep_group->nRanks);
        handle->ll.expert_dispatch_layout = (ncclEpTensor_t){
            NCCL_EP_TENSOR_INIT_INLINE,
            .ndim = 2,
            .datatype = ncclInt64,
            .data = base + recv_src_bytes,
            .sizes = handle->ll.expert_dispatch_layout_sizes,
        };
    }
    handle->num_topk = num_topk;
    handle->ll.layout = layout;
    return ncclSuccess;
}

static ncclResult_t
ht_init_handle(ncclEpHandle_t handle, ncclEpGroup_t ep_group, const ncclEpTensor_t* handle_mem, int num_topk) {
    assert(ep_group->config.max_dispatch_tokens_per_rank > 0 && "HT requires max_dispatch_tokens_per_rank > 0");
    assert(num_topk > 0 && "HT mode requires num_topk > 0 (pass top_k to ncclEpInitHandle)");
    // kNvlinkDup/kLocalDup write the staging buffer in the same per-expert
    // expanded shape as the dispatch output (each token can occupy up to
    // num_topk slots), so max_recv_tokens must cover the worst case
    // nRanks * max_dispatch_tokens_per_rank * num_topk.
    if (em_staging_indexed_by_em_slot(ep_group) && ep_group->config.zero_copy != NCCL_EP_ZERO_COPY_ON) {
        const size_t needed =
            static_cast<size_t>(ep_group->nRanks) * ep_group->config.max_dispatch_tokens_per_rank * num_topk;
        EP_HOST_ASSERT(
            static_cast<size_t>(ep_group->max_recv_tokens) >= needed &&
            "ncclEpInitHandle: max_recv_tokens too small for em_slot staging");
    }
    handle->num_topk = num_topk;
    const auto L = HtBlockLayout::compute(ep_group, handle->layout, num_topk);

    if (handle_mem != nullptr) {
        assert(handle_mem->ndim == 1 && handle_mem->datatype == ncclUint8);
        assert(
            handle_mem->sizes[0] >= L.total && "handle_mem too small; use ncclEpHandleMemSize to query required size");
        ncclEpTensor_t handle_mem_local;
        const ncclEpTensor_t* resolved_handle_mem;
        NCCLCHECK(resolveTensorWindowBinding(ep_group, handle_mem, &handle_mem_local, 0, &resolved_handle_mem));
        handle->hybridep.preprocessing_block = resolved_handle_mem->data;
        handle->hybridep.owns_handle_mem = false;
    } else {
        CUDA_CHECK(ep_group->alloc.alloc_fn(&handle->hybridep.preprocessing_block, L.total, ep_group->alloc.context));
        handle->hybridep.owns_handle_mem = true;
    }
    handle->hybridep.preprocessing_zero_region_size = L.zero_region;
    handle->hybridep.preprocessing_s2d_size = L.sz_s2d;

    char* ptr = static_cast<char*>(handle->hybridep.preprocessing_block);
    size_t offset = 0;

    // global_routing_map: read directly from ep_group->ht_buffers (not duplicated on the handle).
    handle->hybridep.rdma_to_attn_map = reinterpret_cast<bool*>(ptr + offset);
    offset += L.sz_r2a;
    handle->hybridep.attn_to_rdma_map = (ep_group->nNodes > 1) ? reinterpret_cast<bool*>(ptr + offset) : nullptr;
    offset += L.sz_a2r;
    handle->hybridep.local_expert_routing_map = reinterpret_cast<bool*>(ptr + offset);
    offset += L.sz_ler;
    handle->hybridep.num_tokens_for_experts = reinterpret_cast<int32_t*>(ptr + offset);
    offset += L.sz_ntfe;
    // --- end of zero_region (memset 0x00) ---
    handle->hybridep.sparse_to_dense_map = reinterpret_cast<int32_t*>(ptr + offset);
    offset += L.sz_s2d;
    // --- end of s2d region (memset 0xFF) ---
    handle->hybridep.token_rank_mask = ptr + offset;
    offset += L.sz_rank_mask;
    handle->hybridep.preprocessing_scan_tmp = reinterpret_cast<void*>(ptr + offset);
    offset += L.sz_scan_tmp;
    if (!is_internode_available(ep_group)) {
        handle->hybridep.dense_prob_buffer = reinterpret_cast<float*>(ptr + offset);
        offset += L.sz_prob;
    } else {
        handle->hybridep.dense_prob_buffer = nullptr;
    }
    handle->hybridep.topk_idx = (L.sz_topk_idx > 0) ? reinterpret_cast<void*>(ptr + offset) : nullptr;
    offset += L.sz_topk_idx;
    // EM remap counts/offsets live in handle_mem (EM only; FLAT must not read them).
    handle->hybridep.per_expert_counts_active =
        (L.sz_pec_active > 0) ? reinterpret_cast<int32_t*>(ptr + offset) : nullptr;
    offset += L.sz_pec_active;
    handle->hybridep.expert_token_offsets = (L.sz_eto > 0) ? reinterpret_cast<int64_t*>(ptr + offset) : nullptr;
    offset += L.sz_eto;
    if (L.sz_emuf_group_buf > 0) {
        handle->hybridep.emuf_group_buf = reinterpret_cast<int32_t*>(ptr + offset);
        offset += L.sz_emuf_group_buf;
        handle->hybridep.emuf_group_count = reinterpret_cast<int32_t*>(ptr + offset);
        offset += L.sz_emuf_group_count;
        handle->hybridep.emuf_group_stride = L.emuf_group_stride;
        handle->hybridep.emuf_max_groups = L.emuf_max_groups;
    } else {
        handle->hybridep.emuf_group_buf = nullptr;
        handle->hybridep.emuf_group_count = nullptr;
        handle->hybridep.emuf_group_stride = 0;
        handle->hybridep.emuf_max_groups = 0;
    }
    handle->hybridep.flat2em_slot_map =
        (L.sz_flat2em_slot_map > 0) ? reinterpret_cast<int32_t*>(ptr + offset) : nullptr;
    offset += L.sz_flat2em_slot_map;
    handle->hybridep.recv_topk_weights_flat =
        (L.sz_recv_topk_weights_flat > 0) ? reinterpret_cast<float*>(ptr + offset) : nullptr;
    offset += L.sz_recv_topk_weights_flat;
    handle->hybridep.token_to_recv_slot =
        (L.sz_token_to_recv_slot > 0) ? reinterpret_cast<int32_t*>(ptr + offset) : nullptr;
    offset += L.sz_token_to_recv_slot;
    handle->hybridep.dispatch_output_per_expert_alignment = 0;

    if (is_internode_available(ep_group)) {
        handle->hybridep.dense_prob_buffer = ep_group->ht_buffers.dense_prob_buffer;
        handle->hybridep.token_staging_buffer = ep_group->ht_buffers.token_staging_buffer;
        handle->hybridep.scaling_factor_staging_buffer = ep_group->ht_buffers.scaling_factor_staging_buffer;
    } else {
        handle->hybridep.token_staging_buffer = nullptr;
        handle->hybridep.scaling_factor_staging_buffer = nullptr;
    }
    return ncclSuccess;
}

// No collective; allocates routing buffers only.
// handle_mem == nullptr → alloc_fn owns the block; freed on destroy.
// handle_mem != nullptr → wraps caller buffer; destroy frees only the struct.
ncclResult_t ncclEpInitHandle(
    ncclEpHandle_t* out_handle,
    ncclEpGroup_t ep_group,
    ncclEpLayout_t layout,
    const ncclEpHandleConfig_t* config,
    int num_topk,
    const ncclEpTensor_t* handle_mem) {
    assert(ep_group != nullptr && out_handle != nullptr);
    assert(ep_group->comm != nullptr);
    EP_OPTIONAL_STRUCT(config);
    handle_mem = tensor_ptr(handle_mem); // NULL passthrough; otherwise validates magic
    EP_HOST_ASSERT(layout != NCCL_EP_LAYOUT_UNSET && "ncclEpInitHandle: layout must be set explicitly");
    EP_HOST_ASSERT(
        !(ep_group->config.algorithm == NCCL_EP_ALGO_HIGH_THROUGHPUT && layout != NCCL_EP_LAYOUT_FLAT &&
          layout != NCCL_EP_LAYOUT_EXPERT_MAJOR) &&
        "ncclEpInitHandle: HT mode supports flat and expert-major layouts");
    EP_HOST_ASSERT(
        !(ep_group->config.algorithm == NCCL_EP_ALGO_LOW_LATENCY && layout != NCCL_EP_LAYOUT_EXPERT_MAJOR &&
          layout != NCCL_EP_LAYOUT_RANK_MAJOR) &&
        "ncclEpInitHandle: LL mode supports only expert-major and rank-major layouts");
    assert(ep_group->config.num_experts > 0);
    assert(ep_group->config.num_experts % ep_group->nRanks == 0);

    // Validate EM padding alignment up-front (pow2 required) before any allocation.
    const bool is_ht_em =
        ep_group->config.algorithm != NCCL_EP_ALGO_LOW_LATENCY && layout == NCCL_EP_LAYOUT_EXPERT_MAJOR;
    const size_t em_align = (is_ht_em && config && config->dispatch_output_per_expert_alignment > 1) ?
                                config->dispatch_output_per_expert_alignment :
                                1;
    assert((em_align & (em_align - 1)) == 0 && "dispatch_output_per_expert_alignment must be a power of two");

    *out_handle = new ncclEpHandle();
    ncclEpHandle_t handle = *out_handle;
    handle->group = ep_group;
    handle->layout = layout;

    ncclResult_t res;
    if (ep_group->config.algorithm == NCCL_EP_ALGO_LOW_LATENCY) {
        res = ll_init_handle(handle, ep_group, handle_mem, num_topk);
    } else {
        res = ht_init_handle(handle, ep_group, handle_mem, num_topk);
        if (res == ncclSuccess && is_ht_em) {
            handle->hybridep.dispatch_output_per_expert_alignment = em_align;
        }
    }

    return res;
}

ncclResult_t ncclEpUpdateHandle(
    ncclEpHandle_t handle,
    const ncclEpTensor_t* topk_idx,
    const ncclEpLayoutInfo_t* layout_info,
    cudaStream_t stream) {
    assert(handle != nullptr);
    topk_idx = tensor_required(topk_idx);
    EP_OPTIONAL_LAYOUT_INFO(layout_info);
    assert(topk_idx->ndim == 2);

    ncclEpGroup_t ep_group = handle->group;
    assert(ep_group != nullptr);

    // LL and HT both accept ncclInt32 or ncclInt64; the cached idx keeps the
    // caller's native width.
    assert(
        (topk_idx->datatype == ncclInt32 || topk_idx->datatype == ncclInt64) &&
        "topk_idx must be ncclInt32 or ncclInt64");

    // Take ownership of the descriptor with a library-owned tensor to
    // ensure no dependency on the caller's descriptor.
    tensor_permanent_copy(&handle->topk_idx, topk_idx);

    handle->num_tokens = static_cast<int>(handle->topk_idx.sizes[0]);
    handle->num_tokens_set = true;

    if (ep_group->config.algorithm == NCCL_EP_ALGO_LOW_LATENCY) {
        assert(
            static_cast<int>(topk_idx->sizes[1]) == handle->num_topk &&
            "LL: num_topk mismatch between ncclEpInitHandle and ncclEpUpdateHandle");
        assert(layout_info == nullptr && "LL mode does not accept local tensors in ncclEpUpdateHandle");
        return ncclSuccess;
    }
    if (handle->topk_idx.win_hdl != ncclWindow_t{}) {
        NCCLCHECK(resolveTensorWindowBinding(ep_group, &handle->topk_idx, 0));
    }

    int num_topk = static_cast<int>(topk_idx->sizes[1]);
    if (handle->num_topk > 0)
        assert(
            handle->num_topk == num_topk &&
            "Given topk_idx has unmatched num_topk that ncclEpHandle was created with!");
    else handle->num_topk = num_topk;

    assert(
        handle->num_tokens <= static_cast<int>(ep_group->config.max_dispatch_tokens_per_rank) &&
        "Token count exceeds HT buffer capacity");

    const ncclEpTensor_t* recv_expert_counter = layout_info ? tensor_ptr(layout_info->expert_counters) : nullptr;

    const int num_experts = ep_group->config.num_experts;
    const int max_tokens = ep_group->config.max_dispatch_tokens_per_rank;
    const int n_ranks_per_node = ep_group->lsa_team_size;
    const int nNodes = ep_group->rdma_team_size;
    const int experts_per_rank = ep_group->num_local_experts;
    const int num_experts_packed = (num_experts + 7) / 8;

    // Zero the entire preprocessing zero region (routing, r2a, a2r, ler, ntfe) in one call.
    // Buffers are allocated at max_tokens capacity, so this clears beyond the active num_tokens
    // region — safe because allgather/preprocessing will overwrite the relevant portions.
    CUDA_CHECK(cudaMemsetAsync(
        handle->hybridep.preprocessing_block,
        0,
        handle->hybridep.preprocessing_zero_region_size,
        stream));
    // sparse_to_dense_map (unified S2D) uses 0xFF sentinel (not zero)
    if (handle->hybridep.preprocessing_s2d_size > 0) {
        CUDA_CHECK(cudaMemsetAsync(
            handle->hybridep.sparse_to_dense_map,
            0xFF,
            handle->hybridep.preprocessing_s2d_size,
            stream));
    }

    uint8_t* global_routing_map = ep_group->ht_buffers.global_routing_map;
    uint8_t* local_routing_send_ptr = global_routing_map + (max_tokens * num_experts_packed) * ep_group->rank;

    // ===== Step 1: Convert sparse topk_idx to bitmap routing map =====
    // Pass max_tokens so the kernel zeroes the tail rows in the local send slot;
    // ncclAllGather below ships max_tokens rows and stale tail bits would otherwise
    // be interpreted as live routing by peers.
    // Cache the idx in the caller's native width (int32 or int64).
    if (handle->topk_idx.datatype == ncclInt32) {
        nccl_ep::hybridep::convert_topk_to_routing_map(
            static_cast<const int32_t*>(handle->topk_idx.data),
            local_routing_send_ptr,
            static_cast<int32_t*>(handle->hybridep.topk_idx),
            handle->num_tokens,
            max_tokens,
            handle->num_topk,
            num_experts_packed,
            stream);
    } else {
        nccl_ep::hybridep::convert_topk_to_routing_map(
            static_cast<const int64_t*>(handle->topk_idx.data),
            local_routing_send_ptr,
            static_cast<int64_t*>(handle->hybridep.topk_idx),
            handle->num_tokens,
            max_tokens,
            handle->num_topk,
            num_experts_packed,
            stream);
    }

    // ===== Step 2: Allgather bitmap routing maps =====
    NCCL_CHECK_RESULT(ncclAllGather(
        local_routing_send_ptr,
        global_routing_map,
        static_cast<size_t>(max_tokens) * num_experts_packed,
        ncclUint8,
        ep_group->comm,
        stream));

    // ===== Step 3: Run metadata_preprocessing =====
    const bool expert_major = (handle->layout == NCCL_EP_LAYOUT_EXPERT_MAJOR);

    // layout_info->expert_counters: per-expert counts (HT flat unpadded int32; EM padded int32/64).
    // layout_info->expert_offsets: EM-only per-expert offsets (int32/64).
    const ncclEpTensor_t* recv_expert_offsets_tensor = layout_info ? tensor_ptr(layout_info->expert_offsets) : nullptr;
    auto check_int32_or_int64 = [&](const ncclEpTensor_t* t, const char* name) {
        assert(t->ndim == 1 && "tensor must be 1D");
        assert((t->datatype == ncclInt32 || t->datatype == ncclInt64) && "tensor must be ncclInt32 or ncclInt64");
        assert(
            t->sizes[0] >= static_cast<size_t>(ep_group->num_local_experts) &&
            "tensor size must be >= num_local_experts");
        assert(t->data != nullptr && "tensor data must not be null");
        (void)name;
    };
    // Caller must use the same int dtype across the 3 preprocessing output tensors.
    bool out_is_int64 = true;
    bool out_dtype_set = false;
    auto track_out_dtype = [&](const ncclEpTensor_t* t) {
        const bool is64 = (t->datatype == ncclInt64);
        if (!out_dtype_set) {
            out_is_int64 = is64;
            out_dtype_set = true;
        } else assert(is64 == out_is_int64 && "all preprocessing int output tensors must share dtype");
    };
    void* padded_out_counts = nullptr;
    if (expert_major && recv_expert_counter != nullptr) {
        check_int32_or_int64(recv_expert_counter, "expert_counters");
        padded_out_counts = recv_expert_counter->data;
        track_out_dtype(recv_expert_counter);
    }
    void* out_offsets = nullptr;
    if (expert_major && recv_expert_offsets_tensor != nullptr) {
        check_int32_or_int64(recv_expert_offsets_tensor, "expert_offsets");
        out_offsets = recv_expert_offsets_tensor->data;
        track_out_dtype(recv_expert_offsets_tensor);
    }

    // EM: counts/offsets buffers live in handle_mem (wired by ht_init_handle).
    // FLAT: authoritative counts go to caller's recv_expert_counter when provided.
    int32_t* per_expert_counts_device = nullptr;
    if (expert_major) {
        per_expert_counts_device = handle->hybridep.per_expert_counts_active;
    } else if (recv_expert_counter != nullptr) {
        assert(recv_expert_counter->ndim == 1 && "recv_expert_counter must be 1D");
        assert(recv_expert_counter->datatype == ncclInt32 && "HT flat: recv_expert_counter must be ncclInt32");
        assert(
            recv_expert_counter->sizes[0] >= static_cast<unsigned int>(ep_group->num_local_experts) &&
            "recv_expert_counter size must be >= num_local_experts");
        ncclEpTensor_t recv_expert_counter_local;
        const ncclEpTensor_t* resolved_recv_expert_counter;
        NCCLCHECK(resolveTensorWindowBinding(
            ep_group,
            recv_expert_counter,
            &recv_expert_counter_local,
            0,
            &resolved_recv_expert_counter));
        assert(resolved_recv_expert_counter->data != nullptr && "recv_expert_counter data must not be null");
        per_expert_counts_device = static_cast<int32_t*>(resolved_recv_expert_counter->data);
    }
    // hybridep.expert_token_offsets is already the handle_mem slot for EM; FLAT path never reads it.

    // layout_info->recv_total_counter: scalar total recv tokens (size 1, int32/64), written by metadata.
    void* recv_total_counter = nullptr;
    {
        const ncclEpTensor_t* recv_total_counter_tensor =
            layout_info ? tensor_ptr(layout_info->recv_total_counter) : nullptr;
        if (recv_total_counter_tensor != nullptr) {
            assert(recv_total_counter_tensor->ndim == 1);
            assert(recv_total_counter_tensor->sizes[0] >= 1);
            assert(
                recv_total_counter_tensor->datatype == ncclInt32 || recv_total_counter_tensor->datatype == ncclInt64);
            assert(recv_total_counter_tensor->data != nullptr);
            recv_total_counter = recv_total_counter_tensor->data;
            track_out_dtype(recv_total_counter_tensor);
        }
    }

    // Zero the EM-unfused dup-group counter before scan (when allocated).
    if (handle->hybridep.emuf_group_count != nullptr) {
        CUDA_CHECK(cudaMemsetAsync(handle->hybridep.emuf_group_count, 0, sizeof(int32_t), stream));
    }

    const bool em_permute_active = em_local_permute_enabled(ep_group, handle);

    NCCLCHECK(
        nccl_ep::hybridep::call_metadata_preprocessing(
            global_routing_map,
            handle->hybridep.sparse_to_dense_map,
            handle->hybridep.rdma_to_attn_map,
            handle->hybridep.attn_to_rdma_map,
            handle->hybridep.token_rank_mask,
            handle->hybridep.num_tokens_for_experts,
            handle->hybridep.local_expert_routing_map,
            per_expert_counts_device,
            handle->hybridep.preprocessing_scan_tmp,
            ep_group->rdma_rank,
            ep_group->lsa_rank,
            max_tokens,
            nNodes,
            n_ranks_per_node,
            experts_per_rank,
            expert_major,
            expert_major ? handle->hybridep.expert_token_offsets : nullptr,
            padded_out_counts,
            out_offsets,
            expert_major ? handle->hybridep.dispatch_output_per_expert_alignment : size_t(0),
            // Remap kernel writes authoritative per-expert counts (scan overcounts secondary hits).
            expert_major ? per_expert_counts_device : nullptr,
            expert_major ? handle->num_topk : 0,
            recv_total_counter,
            out_is_int64,
            static_cast<int>(ep_group->config.max_recv_tokens_per_rank),
            handle->hybridep.emuf_group_buf,
            handle->hybridep.emuf_group_count,
            handle->hybridep.emuf_group_stride,
            handle->hybridep.emuf_max_groups,
            ep_group->preprocess_num_sms,
            // EM cooperative scan scratch carved from ep_workspace.
            expert_major ? ep_group->ep_workspace : nullptr,
            // EM-permute mode: FLAT scan replaces scan_em and writes the unified
            // sparse_to_dense_map (now FLAT-shaped) + FLAT-shape LERM; em_scan_kernel
            // runs only for EM-only offsets / counts / alignment.
            em_permute_active,
            // EM-permute bridge: scan_impl_flat fills token_to_recv_slot with the FLAT
            // recv slot per global token, and em_scan_kernel consumes it to
            // populate the recv-indexed em_slot table.
            em_permute_active ? handle->hybridep.token_to_recv_slot : nullptr,
            em_permute_active ? handle->hybridep.flat2em_slot_map : nullptr,
            em_permute_active ? handle->num_topk : 0,
            stream));

    return ncclSuccess;
}

ncclResult_t ncclEpCreateHandle(
    ncclEpHandle_t* out_handle,
    ncclEpGroup_t ep_group,
    ncclEpLayout_t layout,
    const ncclEpTensor_t* topk_idx,
    const ncclEpLayoutInfo_t* layout_info,
    const ncclEpHandleConfig_t* config,
    cudaStream_t stream) {
    topk_idx = tensor_required(topk_idx);
    assert(out_handle != nullptr);
    NCCL_CHECK_RESULT(ncclEpInitHandle(
        out_handle,
        ep_group,
        layout,
        config,
        static_cast<int>(topk_idx->sizes[1]),
        /*handle_mem=*/nullptr));
    return ncclEpUpdateHandle(*out_handle, topk_idx, layout_info, stream);
}

ncclResult_t ncclEpHandleDestroy(ncclEpHandle_t handle) {
    if (!handle) return ncclSuccess;

    if (handle->group->config.algorithm == NCCL_EP_ALGO_LOW_LATENCY) {
        if (handle->ll.owns_handle_mem && handle->ll.handle_mem) {
            handle->group->alloc.free_fn(handle->ll.handle_mem, handle->group->alloc.context);
            handle->ll.handle_mem = nullptr;
        }
    } else if (handle->group->config.algorithm == NCCL_EP_ALGO_HIGH_THROUGHPUT) {
        if (handle->hybridep.owns_handle_mem) {
            if (handle->hybridep.preprocessing_block) {
                handle->group->alloc.free_fn(handle->hybridep.preprocessing_block, handle->group->alloc.context);
                handle->hybridep.preprocessing_block = nullptr;
            }
        }
    }

    delete[] handle->topk_idx.sizes;
    delete handle;
    return ncclSuccess;
}

// EP Operations

ncclResult_t ncclEpDispatch(
    ncclEpHandle_t handle,
    const ncclEpDispatchInputs_t* inputs,
    const ncclEpDispatchOutputs_t* outputs,
    const ncclEpLayoutInfo_t* layout_info,
    const ncclEpDispatchConfig_t* config,
    cudaStream_t stream) {
    EP_REQUIRE_STRUCT(inputs);
    EP_REQUIRE_STRUCT(outputs);
    EP_OPTIONAL_LAYOUT_INFO(layout_info);
    EP_OPTIONAL_STRUCT(config);
    const unsigned int send_only = config ? config->send_only : 0;
    const ncclEpPassDir_t pass_direction = config ? config->pass_direction : NCCL_EP_FWD_PASS;
    if (pass_direction != NCCL_EP_FWD_PASS && handle->group->config.algorithm == NCCL_EP_ALGO_LOW_LATENCY) {
        fprintf(stderr, "ncclEpDispatch: backward pass (pass_direction=%d) is not supported in LL mode\n",
                (int)pass_direction);
        return ncclInvalidUsage;
    }
    ncclEpGroup_t group = handle->group;

    // Lazy num_tokens for callers that skip UpdateHandle (e.g. backward reusing forward's handle_mem).
    // Guard on `num_tokens_set` -- a real zero-token configuration is valid
    // and must not trigger a re-fetch from inputs->tokens (whose data pointer
    // may be NULL on the zero-token rank per the empty-tensor relaxation).
    if (!handle->num_tokens_set) {
        const ncclEpTensor_t* t = tensor_required(inputs->tokens);
        assert(t->ndim > 0);
        handle->num_tokens = static_cast<int>(t->sizes[0]);
        handle->num_tokens_set = true;
    }

    // Consolidated token-dtype gate for all dispatch paths (LL/HT): NONE-mode
    // (bf16/fp16/fp32) or FP8 EXTERN (e4m3/e5m2). Checked once here so the
    // per-mode branches below need not re-validate.
    {
        const ncclEpTensor_t* xt = tensor_required(inputs->tokens);
        if (!validate_dtype(xt->datatype) && !validate_fp8_dtype(xt->datatype)) {
            fprintf(stderr, "NCCL EP: dispatch unsupported token dtype %d\n", static_cast<int>(xt->datatype));
            return ncclInvalidArgument;
        }
    }

    if (group->config.algorithm == NCCL_EP_ALGO_LOW_LATENCY) {
        const ncclEpTensor_t* x = tensor_required(inputs->tokens);
        assert(x->ndim == 2);
        assert(x->sizes[0] == handle->num_tokens);
        assert(x->sizes[0] <= group->config.max_dispatch_tokens_per_rank);
        assert(x->sizes[1] % sizeof(int4) == 0);
        assert(x->sizes[1] % 128 == 0);
        assert(x->sizes[1] * ncclTypeSize(x->datatype) <= group->config.max_token_bytes);
        const int hidden = static_cast<int>(x->sizes[1]);

        // Find and validate output tensors
        const ncclEpTensor_t* recv_x = tensor_required(outputs->tokens);
        const ncclEpTensor_t* scales = tensor_ptr(outputs->scales);
        assert(recv_x->ndim > 0);

        // The FP8 wire path is selected by the presence of a scales tensor, not by
        // x->datatype. An FP8 input dtype with no scales would otherwise fall through
        // to the NONE path and be read at 2 B/elem instead of 1 B — out-of-bounds /
        // corruption. Require scales for FP8 inputs (mirrors the HT dispatch guard).
        if ((x->datatype == ncclFloat8e4m3 || x->datatype == ncclFloat8e5m2) && scales == nullptr) {
            fprintf(stderr, "NCCL EP: LL dispatch with FP8 token dtype requires an outputs->scales tensor\n");
            return ncclInvalidArgument;
        }

        // Read rank-major-specific tensors unconditionally so we can assert
        // their presence (rank-major) or absence (expert-major) in the switch below.
        const ncclEpTensor_t* topk_weights_in = tensor_ptr(inputs->topk_weights);
        const ncclEpTensor_t* recv_topk_weights = tensor_ptr(outputs->topk_weights);
        const ncclEpTensor_t* recv_topk_idx = tensor_ptr(outputs->topk_idx);
        const ncclEpTensor_t* src_rank_counter = layout_info ? tensor_ptr(layout_info->src_rank_counters) : nullptr;

        const unsigned num_recv_tokens =
            static_cast<unsigned>(group->nRanks) * group->config.max_dispatch_tokens_per_rank;
        switch (handle->layout) {
        case NCCL_EP_LAYOUT_RANK_MAJOR:
            // Rank-major output is 3D so the per-rank, per-slot structure is
            // explicit in the descriptor — the caller must acknowledge it.
            assert(recv_x->ndim == 3);
            assert(recv_x->sizes[0] == static_cast<unsigned>(group->nRanks));
            assert(recv_x->sizes[1] == group->config.max_dispatch_tokens_per_rank);
            assert(recv_x->sizes[2] == static_cast<unsigned>(hidden));
            assert(topk_weights_in != nullptr);
            assert(recv_topk_weights != nullptr);
            assert(recv_topk_idx != nullptr);
            assert(topk_weights_in->ndim == 2);
            assert(topk_weights_in->datatype == ncclFloat32);
            assert(topk_weights_in->sizes[0] == static_cast<unsigned>(handle->num_tokens));
            assert(topk_weights_in->sizes[1] == static_cast<unsigned>(handle->num_topk));
            assert(recv_topk_weights->ndim == 3);
            assert(recv_topk_weights->datatype == ncclFloat32);
            assert(recv_topk_weights->sizes[0] == static_cast<unsigned>(group->nRanks));
            assert(recv_topk_weights->sizes[1] == group->config.max_dispatch_tokens_per_rank);
            assert(recv_topk_weights->sizes[2] == static_cast<unsigned>(handle->num_topk));
            assert(recv_topk_idx->ndim == 3);
            assert(recv_topk_idx->datatype == ncclInt32);
            assert(recv_topk_idx->sizes[0] == static_cast<unsigned>(group->nRanks));
            assert(recv_topk_idx->sizes[1] == group->config.max_dispatch_tokens_per_rank);
            assert(recv_topk_idx->sizes[2] == static_cast<unsigned>(handle->num_topk));
            assert(src_rank_counter != nullptr);
            assert(src_rank_counter->ndim == 1);
            assert(src_rank_counter->datatype == ncclInt32);
            assert(src_rank_counter->sizes[0] == static_cast<unsigned>(group->nRanks));
            break;
        default:
            assert(recv_x->ndim == 3);
            assert(recv_x->sizes[0] == group->num_local_experts);
            assert(recv_x->sizes[1] == num_recv_tokens);
            assert(recv_x->sizes[2] == static_cast<unsigned>(hidden));
            assert(topk_weights_in == nullptr);
            assert(recv_topk_weights == nullptr);
            assert(recv_topk_idx == nullptr);
            assert(src_rank_counter == nullptr);
            break;
        }

        if (scales != nullptr) {
            assert(scales->ndim == 3);
            assert(scales->datatype == ncclFloat32);
            assert(scales->sizes[0] == group->num_local_experts);
            assert(scales->sizes[1] == group->config.max_dispatch_tokens_per_rank * group->nRanks);
            EP_HOST_ASSERT(
                scales->sizes[2] > 0 && hidden % static_cast<int>(scales->sizes[2]) == 0 &&
                "LL FP8 output scales: sizes[2] must be a non-zero divisor of hidden");
            // Infer scale block size from the caller-supplied output scales tensor.
            const int scale_block_size = hidden / static_cast<int>(scales->sizes[2]);
            assert(hidden % 512 == 0);
            const size_t fp8_payload_bytes =
                static_cast<size_t>(hidden) + static_cast<size_t>(scales->sizes[2]) * sizeof(float);
            EP_HOST_ASSERT(
                fp8_payload_bytes <= group->config.max_token_bytes && "FP8 dispatch bytes exceed max_token_bytes");
            (void)scale_block_size; // used below when threading to kernel
        }

        // EXTERN FP8: token dtype is the signal. FP8 input → caller pre-quantized;
        // inputs->scales is required and must supply one FP32 scale per block.
        const bool extern_fp8_outer = (x->datatype == ncclFloat8e4m3 || x->datatype == ncclFloat8e5m2);
        const ncclEpTensor_t* in_scales_outer = tensor_ptr(inputs->scales);
        if (extern_fp8_outer) {
            EP_HOST_ASSERT(in_scales_outer != nullptr && "LL FP8 EXTERN requires inputs->scales");
            assert(in_scales_outer->ndim == 2);
            assert(in_scales_outer->datatype == ncclFloat32 || in_scales_outer->datatype == ncclUint8);
            assert(in_scales_outer->sizes[0] == handle->num_tokens);
            EP_HOST_ASSERT(
                in_scales_outer->sizes[1] > 0 && hidden % static_cast<int>(in_scales_outer->sizes[1]) == 0 &&
                "LL FP8 input scales: sizes[1] must be a non-zero divisor of hidden");
            // scale_block_size inferred: hidden / numScales; any divisor of hidden is valid.
        } else {
            assert(in_scales_outer == nullptr && "inputs->scales must be null for BF16 LL dispatch");
        }

        // RECV_EXPERT_COUNTER_DEVICE is required for expert-major (per-expert atomic slot allocator)
        // and must be absent for rank-major (outCnt is unused in the rank-major kernel path).
        const ncclEpTensor_t* recv_count = layout_info ? tensor_ptr(layout_info->expert_counters) : nullptr;
        if (handle->layout == NCCL_EP_LAYOUT_RANK_MAJOR) {
            assert(recv_count == nullptr);
        } else {
            assert(recv_count != nullptr);
            assert(recv_count->ndim == 1);
            assert(recv_count->datatype == ncclInt32);
            assert(recv_count->sizes[0] == group->num_local_experts);
        }

        const auto& buffer = handle->ll.layout.buffers[handle->ll.buffer_idx];
        auto& next_buffer = handle->ll.layout.buffers[handle->ll.buffer_idx ^= 1];
        const auto next_clean_meta = next_buffer.clean_meta_offset();

        unsigned signal_base = group->num_dispatch_signals;
        // Zero-copy dispatch output: when the user supplied a window for the
        // output token tensor and we are on the rank-major / nvlinkOnly / bf16
        // path, sender writes payload directly into the user's recv_x buffer
        // (via P2P peer pointer) and the receiver skips the staging→recv_x copy.
        const bool nvlink_only = (group->lsa_team_size == group->nRanks);
        const bool use_fp8_outer = (scales != nullptr) || extern_fp8_outer;
        const bool zero_copy_recv_x = nvlink_only && handle->layout == NCCL_EP_LAYOUT_RANK_MAJOR && !use_fp8_outer &&
                                      recv_x->win_hdl != ncclWindow_t{};
        // Strict zero_copy=ON contract: all opportunistic conditions must
        // hold, else the call is misconfigured. AUTO/OFF stay opportunistic
        // (matches pre-enum behavior; missing windows just stage through
        // library buffers).
        if (group->config.zero_copy == NCCL_EP_ZERO_COPY_ON && !zero_copy_recv_x) {
            const char* reason =
                !nvlink_only ? "requires nvlink-only topology (lsa_team_size == nRanks)" :
                handle->layout != NCCL_EP_LAYOUT_RANK_MAJOR ? "requires NCCL_EP_LAYOUT_RANK_MAJOR" :
                use_fp8_outer                               ? "is not supported with FP8 outputs" :
                                "requires outputs->tokens to be backed by an NCCL window (ncclCommWindowRegister)";
            fprintf(stderr, "NCCL EP: zero_copy=ON on LL dispatch %s\n", reason);
            return ncclInvalidArgument;
        }
        const ncclWindow_t recv_data_window = zero_copy_recv_x ? recv_x->win_hdl : ncclWindow_t{};
        const size_t recv_data_offset = zero_copy_recv_x ? static_cast<size_t>(recv_x->win_offset) : 0;
        auto dispatch_fn = [=](int phases) {
            char* rdma_base = static_cast<char*>(group->rdma_buffer);
            void* dispatch_send_ptr = rdma_base + buffer.dispatch_rdma_send_buffer_offset;
            void* dispatch_recv_ptr = rdma_base + buffer.dispatch_rdma_recv_data_buffer_offset;
            int* dispatch_count_ptr = reinterpret_cast<int*>(rdma_base + buffer.dispatch_rdma_recv_count_buffer_offset);
            int* next_clean_ptr = reinterpret_cast<int*>(rdma_base + next_clean_meta.first);
            // Prepare data pointers
            auto* recv_x_data = recv_x->data;
            auto* scales_data = scales ? scales->data : nullptr;
            auto* expert_recv_source_indices_data = static_cast<int*>(handle->ll.expert_recv_source_indices.data);
            auto* src_rank_counter_data = src_rank_counter ? static_cast<int*>(src_rank_counter->data) : nullptr;
            auto* expert_dispatch_layout_data = static_cast<int64_t*>(handle->ll.expert_dispatch_layout.data);
            auto* recv_count_data = recv_count ? static_cast<int*>(recv_count->data) : nullptr;
            auto* x_data = x->data;
            auto* topk_weights_in_data = topk_weights_in ? static_cast<const float*>(topk_weights_in->data) : nullptr;
            auto* recv_topk_weights_data = recv_topk_weights ? static_cast<float*>(recv_topk_weights->data) : nullptr;
            auto* recv_topk_idx_data = recv_topk_idx ? static_cast<int32_t*>(recv_topk_idx->data) : nullptr;

            // INTERN FP8: x->datatype == ncclBfloat16 && outputs->scales != nullptr
            // EXTERN FP8: x->datatype == ncclFloat8e4m3/e5m2 (inputs->scales validated above)
            const bool extern_fp8 = (x->datatype == ncclFloat8e4m3 || x->datatype == ncclFloat8e5m2);
            const bool use_fp8 = (scales != nullptr) || extern_fp8;
            const bool round_scale = config ? config->round_scales : false;
            // recv_topk_idx numbering selector. Read in a version-safe way
            // (older callers' smaller layout_info reports AUTO), then resolve
            // AUTO -> LOCAL so the kernel sees only concrete kinds.
            const ncclEpExpertIdKind_t recv_topk_idx_kind =
                resolveRecvTopkIdxKind(layoutInfoRecvTopkIdxKind(layout_info));

            // LL accepts int32 or int64 topk_idx; the cached dtype picks the JIT
            // kernel specialization (TopkIdxT). The lambda packs the shared
            // DispatchParams and threads extern-FP8 + zero-copy through.
            auto launch_dispatch = [&](auto* topk_idx_data, bool topk_is_int64) {
                auto* in_scales_data = extern_fp8 ? static_cast<const uint8_t*>(in_scales_outer->data) : nullptr;
                nccl_ep::internode_ll::DispatchParams params{};
                params.inData = x_data;
                params.inScalesBuf = in_scales_data;
                params.inTopkIdx = topk_idx_data;
                params.topkIdxIsInt64 = topk_is_int64;
                params.scalesPerToken = extern_fp8 ? static_cast<int>(in_scales_outer->sizes[1]) : (hidden / 128);
                params.inTopkWeights = topk_weights_in_data;
                params.outDataBuf = recv_x_data;
                params.outScalesBuf = scales_data;
                params.outSrcInfo = expert_recv_source_indices_data;
                params.outRecvRankCounter = src_rank_counter_data;
                params.outLayout = expert_dispatch_layout_data;
                params.outCnt = recv_count_data;
                params.outRecvTopkWeights = recv_topk_weights_data;
                params.outRecvTopkIdx = recv_topk_idx_data;
                params.sendBuf = dispatch_send_ptr;
                params.recvBuf = dispatch_recv_ptr;
                params.recvCntBuf = dispatch_count_ptr;
                params.sendOff = buffer.dispatch_rdma_send_buffer_offset;
                params.recvOff = buffer.dispatch_rdma_recv_data_buffer_offset;
                params.recvCntOff = buffer.dispatch_rdma_recv_count_buffer_offset;
                params.nextRecvCntBuf = next_clean_ptr;
                params.nextRecvCntBufSize = next_clean_meta.second;
                params.recvStats = nullptr;
                params.waitStats = nullptr;
                params.numTokens = handle->num_tokens;
                params.hidden = hidden;
                params.maxTokensPerRank = group->config.max_dispatch_tokens_per_rank;
                params.numTopk = handle->num_topk;
                params.numExperts = group->config.num_experts;
                params.currRank = group->rank;
                params.numRanks = group->nRanks;
                params.layout = handle->layout;
                params.numComms = group->num_nccl_comms;
                params.devComms = group->nccl_dev_comms;
                params.windows = group->nccl_wins;
                params.signalsBase = signal_base;
                params.workspace = group->ep_workspace;
                params.numDeviceSms = group->comm_num_sms;
                params.rankMask = group->mask_buffer;
                params.asyncErrorFlag = group->async_error_flag;
                params.timeoutCycles = group->timeout_cycles;
                params.useUe8m0 = false;
                params.roundScale = round_scale;
                params.nvlinkOnly = nvlink_only;
                params.recvDataWindow = recv_data_window;
                params.recvDataOffset = recv_data_offset;
                params.recvTopkIdxKind = recv_topk_idx_kind;
                params.phases = phases;
                params.tokenDtype = x->datatype;
                nccl_ep::internode_ll::call_dispatch(params, use_fp8, stream);
            };
            switch (handle->topk_idx.datatype) {
            case ncclInt32:
                launch_dispatch(static_cast<const int32_t*>(handle->topk_idx.data), /*topk_is_int64=*/false);
                break;
            case ncclInt64:
                launch_dispatch(static_cast<const int64_t*>(handle->topk_idx.data), /*topk_is_int64=*/true);
                break;
            default:
                assert(false && "LL topk_idx must be ncclInt32 or ncclInt64");
            }
        };

        // Execute dispatch with appropriate phase flags
        const int dispatch_phases =
            send_only ? LOW_LATENCY_SEND_PHASE : (LOW_LATENCY_SEND_PHASE | LOW_LATENCY_RECV_PHASE);
        dispatch_fn(dispatch_phases);

        if (send_only) {
            handle->ll.continue_fn = dispatch_fn;
        }
    } else { // HT

        bool is_single_node = !is_internode_available(group);

        const bool expert_major = (handle->layout == NCCL_EP_LAYOUT_EXPERT_MAJOR);

        const ncclEpTensor_t* x = tensor_required(inputs->tokens);
        const ncclEpTensor_t* topk_weights = tensor_ptr(inputs->topk_weights);
        const ncclEpTensor_t* scales = tensor_ptr(inputs->scales);
        // Local copies for tensors that need window resolution; only populated when used.
        ncclEpTensor_t x_local, scales_local, topk_weights_local;
        ncclEpTensor_t recv_x_local, recv_scales_local, recv_topk_weights_local, recv_topk_idx_local;

        if (x->ndim == 0) {
            return ncclInvalidArgument;
        }
        assert(x->ndim == 2);
        assert(x->sizes[0] == handle->num_tokens);
        assert(x->sizes[0] <= group->config.max_dispatch_tokens_per_rank);
        assert(
            x->sizes[1] * ncclTypeSize(x->datatype) <= group->config.max_token_bytes &&
            "HT dispatch token bytes must not exceed group's max_token_bytes");
        const int hidden = static_cast<int>(x->sizes[1]);
        NCCLCHECK(resolveTensorWindowBinding(
            group,
            x,
            &x_local,
            static_cast<uint64_t>(group->gin_config.token_staging_offset),
            &x));

        // For multi-node: copy user buffers to pre-registered staging buffers
        // The staging buffers were allocated and GIN-registered during Group Create
        // This avoids ~60ms GIN registration overhead on the dispatch hot path
        void* token_ptr = x->data; // Default: use user buffer directly
        const bool x_uses_external_window = tensorUsesExternalWindow(group, x);
        if (!is_single_node && handle->hybridep.token_staging_buffer != nullptr && !x_uses_external_window) {
            // Copy user tokens to pre-registered staging buffer (D2D copy is ~0.1ms vs ~30ms GIN registration)
            size_t token_size = x->sizes[0] * x->sizes[1] * ncclTypeSize(x->datatype);
            CUDA_CHECK(cudaMemcpyAsync(
                handle->hybridep.token_staging_buffer,
                x->data,
                token_size,
                cudaMemcpyDeviceToDevice,
                stream));
            token_ptr = handle->hybridep.token_staging_buffer;
        }
        // EXTERN quantization is requested by the presence of input scales (per Phuong):
        // scales != NULL => the caller pre-quantized to FP8 and we forward the FP8 tokens +
        // scales to the experts unchanged (block size is irrelevant here — pure transport).
        bool use_fp8 = (scales != nullptr);
        if (use_fp8 && x->datatype != ncclFloat8e4m3 && x->datatype != ncclFloat8e5m2) {
            return ncclInvalidArgument; // scales given => tokens must be an FP8 type
        }
        // local_dup / local_reduce JIT hard-codes uint16_t token type; FP8 is unsupported.
        if (use_fp8 && em_local_dup_active(group, handle->layout)) {
            return ncclInvalidArgument;
        }
        // fp8 is incompatible with the FLAT-dispatch + local-permute path: the
        // permute kernel is bf16-only and UpdateHandle has already populated
        // the main LERM in FLAT shape (mismatched for the nvlink_dup
        // dense_to_sparse_prob consumer). Hard-reject early before any kernel
        // runs; the user can set NCCL_EP_HT_EM_NVLINK_DUP=1 to take that path.
        if (use_fp8 && em_local_permute_enabled(group, handle)) {
            fprintf(
                stderr,
                "NCCL EP: HT EM with fp8 is not supported on the local-permute path; "
                "set NCCL_EP_HT_EM_NVLINK_DUP=1 to use the nvlink_dup path.\n");
            return ncclInvalidArgument;
        }
        // Derive scale element size from the dtype in the caller-supplied scales tensor.
        // FP32 (ncclFloat32) = 4 bytes/scale, UE8M0 (ncclUint8) = 1 byte/scale.
        const int scale_elem_bytes = (use_fp8 && scales->datatype == ncclUint8) ? 1 : static_cast<int>(sizeof(float));
        if (use_fp8) {
            NCCLCHECK(resolveTensorWindowBinding(
                group,
                scales,
                &scales_local,
                static_cast<uint64_t>(group->gin_config.scaling_factor_staging_offset),
                &scales));
        }

        // For FP8: copy user scaling factors to pre-registered staging buffer
        void* scales_ptr = use_fp8 ? scales->data : nullptr; // Default: use user buffer directly
        const bool scales_uses_external_window = use_fp8 && tensorUsesExternalWindow(group, scales);
        if (use_fp8 && !is_single_node && handle->hybridep.scaling_factor_staging_buffer != nullptr &&
            !scales_uses_external_window) {
            // Copy user scaling factors to pre-registered staging buffer (D2D copy is ~0.1ms vs ~30ms GIN registration)
            size_t scales_size = scales->sizes[0] * scales->sizes[1] * ncclTypeSize(scales->datatype);
            CUDA_CHECK(cudaMemcpyAsync(
                handle->hybridep.scaling_factor_staging_buffer,
                scales->data,
                scales_size,
                cudaMemcpyDeviceToDevice,
                stream));
            scales_ptr = handle->hybridep.scaling_factor_staging_buffer;
        }
        if (use_fp8) {
            assert(scales->ndim == 2);
            assert(scales->datatype == ncclFloat32 || scales->datatype == ncclUint8);
        }

        // HT dispatch kernel uses TMA for token/prob/scaling-factor payloads.
        // Keep these constraints at API-entry to fail fast on unsupported shapes.
        const int experts_per_node = group->num_local_experts * group->lsa_team_size;
        assert(
            (experts_per_node * static_cast<int>(sizeof(float))) % 16 == 0 &&
            "HT dispatch requires experts_per_node to be multiple of 4 (16B prob TMA alignment)");

        const size_t token_bytes_per_token = static_cast<size_t>(hidden) * ncclTypeSize(x->datatype);
        assert(
            (token_bytes_per_token % 16) == 0 &&
            "HT dispatch requires token bytes per token to be 16B aligned for TMA");

        // Infer FP8 scale block size from the caller's input scales tensor.
        // Buffer was pre-allocated for scale_block_size >= kFP8ExternScaleBlockSize (see group create).
        const int num_scales_per_token = use_fp8 ? static_cast<int>(scales->sizes[1]) : 0;
        if (use_fp8) {
            EP_HOST_ASSERT(
                num_scales_per_token > 0 && hidden % num_scales_per_token == 0 &&
                "HT dispatch FP8: scales->sizes[1] must be a non-zero divisor of hidden");
            const int scale_block_size = hidden / num_scales_per_token;
            EP_HOST_ASSERT(
                scale_block_size >= kFP8ExternScaleBlockSize &&
                "HT dispatch FP8: scale_block_size < kFP8ExternScaleBlockSize; "
                "pre-allocated output scale buffer too small (future: set max_scale_bytes_per_token in config)");
            assert(
                ((num_scales_per_token * scale_elem_bytes) % 16) == 0 &&
                "HT dispatch FP8 requires scaling-factor bytes per token to be 16B aligned");
        }

        // Output tensors
        const ncclEpTensor_t* recv_x = tensor_required(outputs->tokens);
        const ncclEpTensor_t* recv_topk_weights = tensor_ptr(outputs->topk_weights);
        const ncclEpTensor_t* recv_topk_idx = tensor_ptr(outputs->topk_idx);
        const ncclEpTensor_t* recv_scales = nullptr;
        if (recv_x->ndim == 0) {
            return ncclInvalidArgument;
        }
        NCCLCHECK(resolveTensorWindowBinding(group, recv_x, &recv_x_local, 0, &recv_x));
        if (use_fp8) {
            recv_scales = tensor_ptr(outputs->scales);
            if (recv_scales == nullptr) {
                return ncclInvalidArgument;
            }
            NCCLCHECK(resolveTensorWindowBinding(group, recv_scales, &recv_scales_local, 0, &recv_scales));
        }

        // Pass direction is the source of truth (default FWD via zero-init).
        // FWD: topk_weights required (routing live). BWD: topk_weights forbidden.
        const bool forward_dispatch = (pass_direction == NCCL_EP_FWD_PASS);

        if (forward_dispatch) {
            if (topk_weights == nullptr) {
                return ncclInvalidArgument;
            }
            assert(topk_weights->ndim == 2 && topk_weights->datatype == ncclFloat32);
            assert(topk_weights->sizes[0] == handle->num_tokens);
            assert(topk_weights->sizes[1] == handle->num_topk);
            NCCLCHECK(resolveTensorWindowBinding(
                group,
                topk_weights,
                &topk_weights_local,
                static_cast<uint64_t>(group->gin_config.dense_prob_offset),
                &topk_weights));
        } else {
            if (topk_weights != nullptr || recv_topk_weights != nullptr || recv_topk_idx != nullptr) {
                return ncclInvalidArgument;
            }
        }

        // FWD only: rebuild dense prob from cached topk_idx + caller topk_weights.
        float* dense_prob = handle->hybridep.dense_prob_buffer;
        if (forward_dispatch) {
            assert(
                handle->hybridep.topk_idx != nullptr &&
                "HT FWD dispatch: hybridep.topk_idx missing (ncclEpUpdateHandle not called?)");
            size_t dense_prob_size =
                static_cast<size_t>(handle->num_tokens) * group->config.num_experts * sizeof(float);
            CUDA_CHECK(cudaMemsetAsync(dense_prob, 0, dense_prob_size, stream));

            if (handle->topk_idx.datatype == ncclInt32) {
                nccl_ep::hybridep::sparse_to_dense_prob(
                    static_cast<const int32_t*>(handle->hybridep.topk_idx),
                    static_cast<const float*>(topk_weights->data),
                    dense_prob,
                    handle->num_tokens,
                    handle->num_topk,
                    group->config.num_experts,
                    stream);
            } else {
                nccl_ep::hybridep::sparse_to_dense_prob(
                    static_cast<const int64_t*>(handle->hybridep.topk_idx),
                    static_cast<const float*>(topk_weights->data),
                    dense_prob,
                    handle->num_tokens,
                    handle->num_topk,
                    group->config.num_experts,
                    stream);
            }
        }

        /* ===== Build DispatchParams ===== */
        // DispatchParams encapsulates all buffers and metadata needed by HT dispatch kernel:
        //   - Input buffers: attn_input_token, attn_input_prob, attn_input_scaling_factor
        //   - Intranode output buffers: expert_output_token_ptrs, expert_output_prob_ptrs (per-rank pointers)
        //   - RDMA staging buffers: rdma_inter_node_group_* (for multi-node only)
        //   - Metadata: sparse_to_dense_map, rdma_to_attn_map, attn_to_rdma_map
        //   - Sync flags: expected_*_flag_value, intra_node_write_completion_flags
        nccl_ep::hybridep::DispatchParams params;
        params.hidden_dim = hidden;
        params.experts_per_rank = group->num_local_experts;
        params.num_ranks_per_node = group->lsa_team_size;
        params.attn_input_token = token_ptr;
        params.attn_input_prob = forward_dispatch ? dense_prob : nullptr;
        params.attn_input_scaling_factor = use_fp8 ? static_cast<const uint8_t*>(scales_ptr) : nullptr;
        // Use HOST pointer arrays - these get copied into the kernel param struct for fast __grid_constant__ access.
        // For external output tensors with windows, resolve full per-rank output pointers
        // (local + same-node peers) so all writers target user buffers directly.
        std::vector<void*> dispatch_output_token_ptrs;
        const bool recv_x_uses_external_window = tensorUsesExternalWindow(group, recv_x);
        if (group->config.zero_copy == NCCL_EP_ZERO_COPY_ON && !recv_x_uses_external_window) {
            fprintf(
                stderr,
                "NCCL EP: zero_copy requires ncclEpDispatch outputs->tokens to be backed by a "
                "user-registered NCCL window (ncclCommWindowRegister)\n");
            return ncclInvalidArgument;
        }
        // em_permute always writes to staging (the permute kernel below copies
        // staging -> recv_x->data); external-window vs. plain recv_x is irrelevant
        // since the permute kernel just dereferences recv_x->data.
        const bool em_permute_active = em_local_permute_enabled(group, handle);
        if (recv_x_uses_external_window && !em_permute_active) {
            NCCLCHECK(buildIntranodePtrArray<void>(group, recv_x, dispatch_output_token_ptrs));
            params.expert_output_token_ptrs = dispatch_output_token_ptrs.data();
        } else {
            params.expert_output_token_ptrs = group->ht_buffers.dispatch_expert_output_token_buffer_ptrs;
        }
        params.expert_output_prob_ptrs = group->ht_buffers.dispatch_expert_output_prob_buffer_ptrs;
        std::vector<uint8_t*> dispatch_output_sf_ptrs;
        const bool recv_scales_uses_external_window = use_fp8 && tensorUsesExternalWindow(group, recv_scales);
        if (recv_scales_uses_external_window) {
            NCCLCHECK(buildIntranodePtrArray<uint8_t>(group, recv_scales, dispatch_output_sf_ptrs));
            params.expert_output_scaling_factor_ptrs = dispatch_output_sf_ptrs.data();
        } else {
            params.expert_output_scaling_factor_ptrs =
                use_fp8 ? reinterpret_cast<uint8_t* const*>(
                              group->ht_buffers.dispatch_expert_output_scaling_factor_buffer_ptrs) :
                          nullptr;
        }
        // EM-permute path runs the dispatch kernel in FLAT layout and reshuffles
        // FLAT staging into EM zones via the local permute kernel below. BWD
        // reuses the FWD-populated handle maps; only the FWD-only weight scatter
        // is skipped (recv_topk_weights is null for BWD).
        params.rdma_to_attn_map = handle->hybridep.rdma_to_attn_map;
        params.attn_to_rdma_map = handle->hybridep.attn_to_rdma_map;
        params.sparse_to_dense_map = handle->hybridep.sparse_to_dense_map;
        params.s2d_inner_dim = (expert_major && !em_permute_active) ? handle->num_topk : group->lsa_team_size;
        params.layout = em_permute_active ? NCCL_EP_LAYOUT_RANK_MAJOR : handle->layout;
        // s2d_inner_dim must pair with layout (mismatch → OOB in combine reduction).
        assert(
            (params.layout == NCCL_EP_LAYOUT_EXPERT_MAJOR) ? (params.s2d_inner_dim == handle->num_topk) :
                                                             (params.s2d_inner_dim == group->lsa_team_size));
        // Expert-major zero-padding inputs for the in-kernel PAD warp; suppressed
        // in EM-permute mode (permute kernel writes pads after dispatch).
        const bool fused_em_pad = expert_major && !em_permute_active;
        params.pad_actual_counts = fused_em_pad ? handle->hybridep.per_expert_counts_active : nullptr;
        params.pad_expert_token_offsets = fused_em_pad ? handle->hybridep.expert_token_offsets : nullptr;
        params.pad_alignment =
            fused_em_pad ? static_cast<int>(handle->hybridep.dispatch_output_per_expert_alignment) : 0;
        // Always pass a valid device pointer — the kernel unconditionally
        // dereferences this even in single-node mode (the value is just unused).
        params.expected_rdma_flag_value = group->ht_buffers.dev_dispatch_expected_rdma;
        params.rdma_inter_node_group_flags = is_single_node ? nullptr : group->ht_buffers.rdma_inter_node_group_flags;
        params.expected_intra_node_flag_value = group->ht_buffers.dev_dispatch_expected_intra;
        params.intra_node_write_completion_flags = group->ht_buffers.intra_node_write_completion_flags;
        params.dispatch_grid_barrier_counter = group->ht_buffers.dispatch_grid_barrier_counter;
        params.guard_enabled = !nccl_ep_env_flag_on(group->env.disable_guard);
        // Pass device communicators and windows
        // Always pass a valid devComm (single-node too): the HT LSA sync-guard uses the NCCL LSA
        // barrier (needs comm.lsaBarrier). GIN/RDMA paths stay if-constexpr-gated (out single-node).
        // TODO: remove multiple gin comm notion from group
        params.dcomm = group->gin_config.dcomms[0];
        params.nccl_token_window = x->win_hdl;
        params.nccl_prob_window = forward_dispatch ? group->gin_config.nccl_window : ncclWindow_t{};
        params.nccl_sf_window = use_fp8 ? scales->win_hdl : ncclWindow_t{};
        params.nccl_internal_window = group->gin_config.nccl_window;
        params.num_ctx_per_comm = is_single_node ? 0 : group->gin_config.num_ctx_per_comm;
        params.gin_base_ptr = is_single_node ? nullptr : group->gin_config.gin_base_ptr;
        // Use offsets relative to gin_base_ptr
        // All buffers are part of one large registered window
        // Calculate bytes_per_entry for batched staging
        size_t bytes_per_token_entry = group->config.max_token_bytes; // token data
        size_t bytes_per_prob_entry = (group->num_local_experts * group->lsa_team_size) * sizeof(float); // prob data
        size_t bytes_per_sf_entry =
            use_fp8 ? static_cast<size_t>(num_scales_per_token) * scale_elem_bytes : 0; // FP8 scaling factor
        size_t bytes_per_entry = bytes_per_token_entry + bytes_per_prob_entry + bytes_per_sf_entry;

        params.mr_info = {
            .attn_input_token_offset = is_single_node ? 0 : x->win_offset,
            .attn_input_prob_offset = (is_single_node || !forward_dispatch) ? 0 : group->gin_config.dense_prob_offset,
            .attn_input_scaling_factor_offset = (is_single_node || !use_fp8) ? 0 : scales->win_offset,
            // Batched staging parameters (packed layout)
            .rdma_send_staging_offset = is_single_node ? 0 : group->gin_config.rdma_send_staging_offset,
            .rdma_inter_node_group_packed_offset =
                is_single_node ? 0 : group->gin_config.rdma_inter_node_group_packed_offset,
            .guard_offset = is_single_node ? 0 : group->gin_config.dispatch_guard_offset,
            .bytes_per_entry = bytes_per_entry,
            .max_tokens_per_dest = static_cast<size_t>(group->config.max_dispatch_tokens_per_rank),
            // Streaming signal parameters
            .signals_tail_base = is_single_node ? 0 : static_cast<unsigned>(group->gin_config.signals_tail_base),
            .num_max_rdma_chunked_send_tokens = is_single_node ? 0 : group->gin_config.num_max_rdma_chunked_send_tokens,
        };
        params.local_rank = group->lsa_rank;
        params.node_rank = group->rdma_rank;
        params.num_tokens_per_rank = group->config.max_dispatch_tokens_per_rank;
        // EM local-fanout: dispatch dedups S2G; receiver local_dup fills secondaries.
        const bool em_unfused_active = em_local_dup_active(group, handle->layout);
        params.local_dup_num_sms = em_unfused_active ? static_cast<int>(group->prolog_epilog_sms) : 0;

        // Call dispatch kernel
        const int sf_bytes_per_token = use_fp8 ? num_scales_per_token * scale_elem_bytes : 0;
        NCCLCHECK(
            nccl_ep::hybridep::call_dispatch(
                params,
                group->ht_aligned_max_tokens,
                group->ht_tokens_per_chunk,
                group->rdma_team_size,
                use_fp8,
                forward_dispatch,
                static_cast<int>(group->comm_num_sms),
                sf_bytes_per_token,
                &group->env,
                stream,
                x->datatype));

        // Fan primaries out to secondary em_slots in this rank's recv buffer.
        if (em_unfused_active) {
            nccl_ep::hybridep::call_local_dup(
                /*expert_output_token=*/
                group->ht_buffers.dispatch_expert_output_token_buffer_ptrs[group->lsa_rank],
                /*expert_output_prob=*/
                forward_dispatch ? group->ht_buffers.dispatch_expert_output_prob_buffer_ptrs[group->lsa_rank] : nullptr,
                handle->hybridep.emuf_group_buf,
                handle->hybridep.emuf_group_count,
                handle->hybridep.emuf_group_stride,
                group->ht_buffers.intra_node_write_completion_flags,
                /*expected_intra_node_flag_value=*/params.expected_intra_node_flag_value,
                /*grid_barrier_counter=*/params.dispatch_grid_barrier_counter,
                params.hidden_dim,
                params.experts_per_rank,
                params.num_ranks_per_node,
                forward_dispatch,
                params.local_dup_num_sms,
                stream,
                x->datatype);
        }

        const unsigned int max_recv_tokens = static_cast<unsigned int>(handle->group->max_recv_tokens);

        // Pick the staging→user copy size based on stream-capture state:
        //   - Capturing (CUDA Graph): copy the full caller-allocated bound
        //     (max_recv_tokens). No host sync, so the call records cleanly.
        //     The caller must size recv_x (and recv_scales) >= max_recv_tokens.
        //   - Not capturing: blocking D2H readback of the actual recv-token
        //     total and copy only that many rows, keeping bandwidth cost
        //     proportional to real traffic.
        cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
        CUDA_CHECK(cudaStreamIsCapturing(stream, &capture_status));
        const bool is_capturing = (capture_status == cudaStreamCaptureStatusActive);
        unsigned int actual_recv_tokens = 0;
        // The D2H readback exists only to size the legacy byte_copy and the
        // FP8-scales D2D; the em-permute path reads num_tokens_for_experts on
        // device, so skip the sync entirely when it fires.
        if (!is_capturing && !em_permute_active) {
            CUDA_CHECK(cudaMemcpyAsync(
                &actual_recv_tokens,
                handle->hybridep.num_tokens_for_experts,
                sizeof(actual_recv_tokens),
                cudaMemcpyDeviceToHost,
                stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            assert(actual_recv_tokens <= max_recv_tokens);
        }
        const unsigned int recv_copy_rows = is_capturing ? max_recv_tokens : actual_recv_tokens;

        /* ===== Copy intranode staging → caller outputs ===== */
        // External-window outputs are written directly by the kernel; regular tensors
        // need a D2D copy from the shared intranode staging buffers. On the
        // EM-permute path (HT + EM + zero_copy != ON), the copy is deferred until
        // after dense_to_sparse_prob has populated recv_topk_idx_flat, since the
        // permute kernel uses that table to map FLAT slots to EM zones.
        assert(recv_x->ndim == 2);
        if (!recv_x_uses_external_window && !em_permute_active) {
            if (recv_x->sizes[0] < recv_copy_rows) {
                return ncclInvalidArgument;
            }
            // Clamp to staging capacity (see token_staging_slots).
            const unsigned int clamped_rows =
                static_cast<unsigned int>(std::min<size_t>(recv_copy_rows, group->ht_buffers.token_staging_slots));
            size_t copy_size = static_cast<size_t>(clamped_rows) * recv_x->sizes[1] * ncclTypeSize(recv_x->datatype);
            void* src = group->ht_buffers.dispatch_expert_output_token_buffer_ptrs[group->lsa_rank];
            CUDA_CHECK(cudaMemcpyAsync(recv_x->data, src, copy_size, cudaMemcpyDeviceToDevice, stream));
        }

        /* ===== Convert dense output → sparse format ===== */
        if (forward_dispatch) {
            // recv_topk_weights required; shape depends on layout:
            //   EM: 1D [N] (each slot is per-(token, local_expert), at most 1 weight).
            //   FLAT: 2D [N, top_k] paired with required recv_topk_idx [N, top_k].
            if (recv_topk_weights == nullptr) {
                return ncclInvalidArgument;
            }
            NCCLCHECK(
                resolveTensorWindowBinding(group, recv_topk_weights, &recv_topk_weights_local, 0, &recv_topk_weights));
            const bool em = (handle->layout == NCCL_EP_LAYOUT_EXPERT_MAJOR);
            if (em) {
                if (recv_topk_idx != nullptr) {
                    return ncclInvalidArgument;
                }
            } else {
                if (recv_topk_idx == nullptr) {
                    return ncclInvalidArgument;
                }
                NCCLCHECK(resolveTensorWindowBinding(group, recv_topk_idx, &recv_topk_idx_local, 0, &recv_topk_idx));
            }
            assert(recv_topk_weights->datatype == ncclFloat32);
            if (em) {
                assert(recv_topk_weights->ndim == 1 && "HT EM recv_topk_weights must be 1D [num_recv_tokens]");
                if (recv_topk_weights->sizes[0] < max_recv_tokens) {
                    return ncclInvalidArgument;
                }
            } else {
                assert(recv_topk_weights->ndim == 2 && "HT FLAT recv_topk_weights must be 2D [num_recv_tokens, top_k]");
                assert(recv_topk_idx->ndim == 2);
                assert(recv_topk_idx->datatype == ncclInt64);
                if (recv_topk_weights->sizes[0] < max_recv_tokens || recv_topk_idx->sizes[0] < max_recv_tokens ||
                    recv_topk_weights->sizes[0] != recv_topk_idx->sizes[0]) {
                    return ncclInvalidArgument;
                }
            }

            int num_recv_tokens = static_cast<int>(max_recv_tokens);
            int experts_per_node = group->num_local_experts * group->lsa_team_size;
            // recv_topk_idx numbering selector (matches LL rank-major path):
            // version-safe read of layout_info, resolve AUTO -> LOCAL, pass the
            // concrete kind plus the per-group GLOBAL offset to the kernel.
            const ncclEpExpertIdKind_t recv_topk_idx_kind =
                resolveRecvTopkIdxKind(layoutInfoRecvTopkIdxKind(layout_info));
            const int global_expert_offset = group->rank * group->num_local_experts;

            // em-permute path: dense_to_sparse_prob runs in FLAT shape.
            // Weights are written to FLAT scratch (recv_topk_weights_flat) and
            // the topk_idx write is skipped to preserve the em_slot values
            // published by em_scan_kernel. The local permute kernel below
            // reshuffles FLAT scratch into the caller's EM recv buffer.
            const bool dsp_em = em && !em_permute_active;
            float* dsp_topk_weights = em_permute_active ? handle->hybridep.recv_topk_weights_flat :
                                                          static_cast<float*>(recv_topk_weights->data);
            int64_t* dsp_topk_idx =
                em_permute_active ? nullptr : (recv_topk_idx ? static_cast<int64_t*>(recv_topk_idx->data) : nullptr);

            nccl_ep::hybridep::dense_to_sparse_prob(
                group->ht_buffers.dispatch_expert_output_prob_buffer_ptrs[group->lsa_rank],
                handle->hybridep.local_expert_routing_map,
                dsp_topk_weights,
                dsp_topk_idx,
                num_recv_tokens,
                handle->num_topk,
                group->num_local_experts,
                experts_per_node,
                group->lsa_rank,
                global_expert_offset,
                recv_topk_idx_kind,
                dsp_em,
                stream);
        }

        // EM-permute path: FLAT staging now holds rank-major token rows, the
        // FLAT scratch holds recv_topk_idx/weights. Run the local permute kernel
        // to write the caller's EM recv_x + EM 1D recv_topk_weights with the
        // correct per-expert zone offsets and zero-pad rows.
        if (em_permute_active) {
            // BWD passes recv_topk_weights == nullptr (weights are FWD-only).
            assert(forward_dispatch ? (recv_topk_weights != nullptr) : (recv_topk_weights == nullptr));
            if (!validate_dtype(recv_x->datatype)) {
                // local_permute_dup is a byte-relocation kernel: any NONE-mode
                // wire dtype (bf16/fp16/fp32) works; FP8 is not supported here.
                return ncclInvalidArgument;
            }
            // local_permute_dup writes recv_x by em_slot up to max_recv_tokens.
            if (recv_x->sizes[0] < max_recv_tokens) {
                return ncclInvalidArgument;
            }
            const int row_bytes = static_cast<int>(recv_x->sizes[1]) * ncclTypeSize(recv_x->datatype);
            if (row_bytes <= 0 || (row_bytes % 16) != 0) {
                return ncclInvalidArgument; // int4-vectorized row copy requires 16B-aligned row
            }
            nccl_ep::hybridep::launch_dispatch_permute(
                recv_x->data,
                recv_topk_weights ? static_cast<float*>(recv_topk_weights->data) : nullptr,
                group->ht_buffers.dispatch_expert_output_token_buffer_ptrs[group->lsa_rank],
                forward_dispatch ? handle->hybridep.recv_topk_weights_flat : nullptr,
                handle->hybridep.flat2em_slot_map,
                handle->hybridep.num_tokens_for_experts,
                handle->hybridep.expert_token_offsets,
                handle->hybridep.per_expert_counts_active,
                handle->num_topk,
                group->num_local_experts,
                row_bytes,
                static_cast<int>(group->device_sm_count),
                group->prolog_epilog_sms,
                stream);
        }

        // FP8 scales output (async D2D, sized by caller).
        if (use_fp8) {
            assert(recv_scales->ndim == 2);
            if (!recv_scales_uses_external_window) {
                if (recv_scales->sizes[0] < recv_copy_rows) {
                    return ncclInvalidArgument;
                }
                size_t copy_size =
                    static_cast<size_t>(recv_copy_rows) * recv_scales->sizes[1] * ncclTypeSize(recv_scales->datatype);
                CUDA_CHECK(cudaMemcpyAsync(
                    recv_scales->data,
                    group->ht_buffers.dispatch_expert_output_scaling_factor_buffer_ptrs[group->lsa_rank],
                    copy_size,
                    cudaMemcpyDeviceToDevice,
                    stream));
            }
        }
    }
    return ncclSuccess;
}

ncclResult_t ncclEpCombine(
    ncclEpHandle_t handle,
    const ncclEpCombineInputs_t* inputs,
    const ncclEpCombineOutputs_t* outputs,
    const ncclEpCombineConfig_t* config,
    cudaStream_t stream) {
    EP_REQUIRE_STRUCT(inputs);
    EP_REQUIRE_STRUCT(outputs);
    EP_OPTIONAL_STRUCT(config);
    const unsigned int send_only = config ? config->send_only : 0;
    const ncclEpPassDir_t pass_direction = config ? config->pass_direction : NCCL_EP_FWD_PASS;
    if (pass_direction != NCCL_EP_FWD_PASS && handle->group->config.algorithm == NCCL_EP_ALGO_LOW_LATENCY) {
        fprintf(stderr, "ncclEpCombine: backward pass (pass_direction=%d) is not supported in LL mode\n",
                (int)pass_direction);
        return ncclInvalidUsage;
    }

    // Lazy num_tokens for callers that skip UpdateHandle (e.g. handle relocation between prepare and combine).
    // Guard on `num_tokens_set` so a legitimate zero-token configuration is
    // preserved (the outputs->tokens descriptor for a zero-token rank may
    // carry data == nullptr, which is now legal per tensorHasBinding's
    // empty-tensor relaxation but would still fail the assert on dim 0).
    if (!handle->num_tokens_set) {
        const ncclEpTensor_t* lazy_combined = tensor_required(outputs->tokens);
        assert(lazy_combined->ndim > 0);
        handle->num_tokens = static_cast<int>(lazy_combined->sizes[0]);
        handle->num_tokens_set = true;
    }

    // Consolidated token-dtype gate for all combine paths (LL/HT): NONE-mode
    // (bf16/fp16/fp32). There is no FP8 combine. Checked once here.
    {
        const ncclEpTensor_t* xt = tensor_required(inputs->tokens);
        if (!validate_dtype(xt->datatype)) {
            fprintf(stderr, "NCCL EP: combine unsupported token dtype %d\n", static_cast<int>(xt->datatype));
            return ncclInvalidArgument;
        }
    }

    if (handle->group->config.algorithm == NCCL_EP_ALGO_LOW_LATENCY) {
        // Find and validate input tensors
        const ncclEpTensor_t* x = tensor_required(inputs->tokens);
        assert(x->ndim > 0);

        const ncclEpTensor_t* topk_idx = &handle->topk_idx;
        const ncclEpTensor_t* src_info = &handle->ll.expert_recv_source_indices;
        const ncclEpTensor_t* layout_range = &handle->ll.expert_dispatch_layout;

        // topk_weights: expert-major requires it in outputs; rank-major does not
        // (weights are applied by the caller in preReduceRankMajor before ncclEpCombine).
        const ncclEpTensor_t* topk_weights;
        if (handle->layout == NCCL_EP_LAYOUT_RANK_MAJOR) {
            topk_weights = nullptr;
        } else {
            topk_weights = tensor_ptr(outputs->topk_weights);
            EP_HOST_ASSERT(topk_weights != nullptr && "expert-major combine requires topk_weights in outputs");
        }

        // Extract configuration values
        const int num_experts = handle->group->config.num_experts;
        const int num_ranks = handle->group->nRanks;
        const int num_max_dispatch_tokens_per_rank = handle->group->config.max_dispatch_tokens_per_rank;

        // Extract hidden dimension (index differs by layout).
        int hidden;
        switch (handle->layout) {
        case NCCL_EP_LAYOUT_RANK_MAJOR:
            // Rank-major input is 3D so the per-rank, per-slot structure is
            // explicit in the descriptor — the caller must acknowledge it.
            assert(x->ndim == 3);
            assert(x->sizes[0] == static_cast<unsigned>(num_ranks));
            assert(x->sizes[1] == static_cast<unsigned>(num_max_dispatch_tokens_per_rank));
            assert(x->sizes[2] % sizeof(int4) == 0);
            assert(x->sizes[2] % 128 == 0);
            hidden = static_cast<int>(x->sizes[2]);
            break;
        default:
            assert(x->ndim == 3);
            assert(x->sizes[0] == num_experts / num_ranks);
            assert(x->sizes[1] == static_cast<unsigned>(num_ranks) * num_max_dispatch_tokens_per_rank);
            assert(x->sizes[2] % sizeof(int4) == 0);
            assert(x->sizes[2] % 128 == 0);
            hidden = static_cast<int>(x->sizes[2]);
            break;
        }

        // Validate topk_idx tensor (LL: int32 or int64; dtype must
        // match what ncclEpUpdateHandle saw).
        assert(topk_idx->ndim == 2);
        assert(
            (topk_idx->datatype == ncclInt32 || topk_idx->datatype == ncclInt64) &&
            "LL topk_idx must be ncclInt32 or ncclInt64");

        // Validate src_info tensor
        assert(src_info->ndim == 1);
        assert(src_info->datatype == ncclInt32);

        // Validate topk_weights tensor (expert-major only; rank-major applies weights before combine)
        if (topk_weights != nullptr) {
            assert(topk_weights->ndim == 2);
            assert(topk_weights->sizes[0] == topk_idx->sizes[0]);
            assert(topk_weights->sizes[1] == topk_idx->sizes[1]);
            assert(topk_weights->datatype == ncclFloat32);
            assert(topk_weights->sizes[0] <= num_max_dispatch_tokens_per_rank);
        }

        // Extract dimensions (hidden already set in the layout switch above)
        const int num_topk = static_cast<int>(topk_idx->sizes[1]);
        const int num_combined_tokens = static_cast<int>(topk_idx->sizes[0]);

        // Manage double-buffering
        const auto& buffer = handle->ll.layout.buffers[handle->ll.buffer_idx];
        auto& next_buffer = handle->ll.layout.buffers[handle->ll.buffer_idx ^= 1];
        const auto next_clean_meta = next_buffer.clean_meta_offset();

        // Validate buffer layout
        assert(handle->ll.layout.total_bytes <= handle->group->rdma_buffer_size_alloc);

        // Find and validate output tensor
        const ncclEpTensor_t* out = tensor_required(outputs->tokens);

        assert(out->ndim > 0);
        assert(out->ndim == 2);
        assert(out->sizes[0] == num_combined_tokens);
        assert(out->sizes[1] == hidden);
        assert(out->datatype == x->datatype);

        // Define combine lambda
        unsigned signal_base = handle->ll.buffer_idx * (handle->group->num_dispatch_signals / 2);
        auto combine_fn = [=](int phases) {
            char* rdma_base = static_cast<char*>(handle->group->rdma_buffer);
            void* combine_send_ptr = rdma_base + buffer.combine_rdma_send_buffer_offset;
            void* combine_recv_ptr = rdma_base + buffer.combine_rdma_recv_data_buffer_offset;
            int* combine_flag_ptr = reinterpret_cast<int*>(rdma_base + buffer.combine_rdma_recv_flag_buffer_offset);
            int* next_clean_ptr = reinterpret_cast<int*>(rdma_base + next_clean_meta.first);
            // Prepare data pointers
            auto* out_data = out->data;
            auto* x_data = x->data;
            auto* topk_weights_data = topk_weights ? static_cast<float*>(topk_weights->data) : nullptr;
            auto* src_info_data = static_cast<int*>(src_info->data);
            auto* layout_range_data = static_cast<int64_t*>(layout_range->data);

            // LL accepts int32 or int64 topk_idx; the cached dtype picks the JIT
            // combine specialization (TopkIdxT), same as dispatch. zeroCopy stays
            // false: combine reads inData directly and uses windows only to
            // translate peer recv-buffer ptrs; the kernel's zeroCopy mode is
            // dispatch-shaped, so config.zero_copy is a dispatch-only switch.
            auto launch_combine = [&](auto* topk_idx_data, bool topk_is_int64) {
                nccl_ep::internode_ll::CombineParams params{};
                params.inData = x_data;
                params.srcInfo = src_info_data;
                params.layoutRange = layout_range_data;
                params.inTopkIdx = topk_idx_data;
                params.topkIdxIsInt64 = topk_is_int64;
                params.topkWeights = topk_weights_data;
                params.outData = out_data;
                params.sendBuf = combine_send_ptr;
                params.recvBuf = combine_recv_ptr;
                params.recvFlagBuf = combine_flag_ptr;
                params.sendOff = buffer.combine_rdma_send_buffer_offset;
                params.recvOff = buffer.combine_rdma_recv_data_buffer_offset;
                params.recvFlagOff = buffer.combine_rdma_recv_flag_buffer_offset;
                params.nextRecvCntBuf = next_clean_ptr;
                params.nextRecvCntBufSize = next_clean_meta.second;
                params.waitStats = nullptr;
                params.numCombinedTokens = num_combined_tokens;
                params.hidden = hidden;
                params.maxTokensPerRank = num_max_dispatch_tokens_per_rank;
                params.numTopk = num_topk;
                params.numExperts = num_experts;
                params.currRank = handle->group->rank;
                params.numRanks = handle->group->nRanks;
                params.layout = handle->layout;
                params.numComms = handle->group->num_nccl_comms;
                params.devComms = handle->group->nccl_dev_comms;
                params.windows = handle->group->nccl_wins;
                params.signalsBase = signal_base;
                params.workspace = handle->group->ep_workspace;
                params.numDeviceSms = handle->group->comm_num_sms;
                params.rankMask = handle->group->mask_buffer;
                params.asyncErrorFlag = handle->group->async_error_flag;
                params.timeoutCycles = handle->group->timeout_cycles;
                params.useLogFmt = false;
                params.zeroCopy = false;
                params.phases = phases;
                params.tokenDtype = x->datatype;
                nccl_ep::internode_ll::call_combine(params, stream);
            };
            switch (topk_idx->datatype) {
            case ncclInt32:
                launch_combine(static_cast<const int32_t*>(topk_idx->data), /*topk_is_int64=*/false);
                break;
            case ncclInt64:
                launch_combine(static_cast<const int64_t*>(topk_idx->data), /*topk_is_int64=*/true);
                break;
            default:
                assert(false && "LL topk_idx must be ncclInt32 or ncclInt64");
            }
        };

        // Execute combine with appropriate phase flags
        const int combine_phases =
            send_only ? LOW_LATENCY_SEND_PHASE : (LOW_LATENCY_SEND_PHASE | LOW_LATENCY_RECV_PHASE);
        combine_fn(combine_phases);

        if (send_only) {
            handle->ll.continue_fn = combine_fn;
        }
    } else if (handle->group->config.algorithm == NCCL_EP_ALGO_HIGH_THROUGHPUT) {
        // ===== HT mode =====
        // Combine: gather expert outputs back to original token positions
        // Forward combine: just tokens (inference)
        // Backward combine: tokens + gradients (training, topk_weights provided)
        ncclEpGroup_t group = handle->group;
        bool is_single_node = !is_internode_available(group);
        // assert(is_single_node && "HT mode only supports single-node");

        /* ===== Inputs validation ===== */
        const ncclEpTensor_t* x = tensor_required(inputs->tokens);
        if (x->ndim == 0) {
            return ncclInvalidArgument;
        }
        assert(x->ndim == 2);
        // Local copies for tensors that need window resolution; only populated when used.
        ncclEpTensor_t x_local, topk_weights_local, combined_topk_weights_local, combined_x_local;
        NCCLCHECK(resolveTensorWindowBinding(
            group,
            x,
            &x_local,
            static_cast<uint64_t>(group->gin_config.rdma_intra_node_red_token_offset),
            &x));

        // Get dimensions from input tensor
        auto num_tokens = static_cast<int>(x->sizes[0]);
        auto hidden = static_cast<int>(x->sizes[1]);

        // Validate int4 alignment for TMA
        assert((hidden * ncclTypeSize(x->datatype)) % sizeof(int4) == 0);

        // Number of tokens to combine back (original token count from this rank)
        auto num_combined_tokens = handle->num_tokens;

        // Top-k checks (for backward mode)
        // Output combined_topk_weights is always 2D [num_combined_tokens, source_top_k].
        // Input topk_weights shape MUST match the FWD recv_topk_weights shape by layout:
        //   FLAT/RM: 2D [num_recv_tokens, source_top_k]
        //   EM:      1D [num_recv_tokens]
        // The scatter kernel sparse_to_dense_prob_combine_kernel scans local experts per
        // recv slot; under EM each slot maps to exactly one local expert, so the inner
        // stride is 1.
        int num_topk = 0;
        int input_topk_stride = 0;
        const ncclEpTensor_t* topk_weights = tensor_ptr(inputs->topk_weights);
        const ncclEpTensor_t* combined_topk_weights = tensor_ptr(outputs->topk_weights);

        // Pass direction is the source of truth (default FWD via zero-init).
        // BWD combine requires inputs->topk_weights and outputs->topk_weights;
        // FWD combine forbids inputs->topk_weights (outputs->topk_weights is unused).
        const bool backward_combine = (pass_direction == NCCL_EP_BWD_PASS);
        const bool expert_major_in = (handle->layout == NCCL_EP_LAYOUT_EXPERT_MAJOR);

        if (backward_combine) {
            if (topk_weights == nullptr) {
                return ncclInvalidArgument;
            }
            if (combined_topk_weights == nullptr) {
                return ncclInvalidArgument;
            }
            assert(combined_topk_weights->ndim == 2);
            assert(combined_topk_weights->sizes[0] == num_combined_tokens);
            assert(combined_topk_weights->datatype == ncclFloat32);
            num_topk = static_cast<int>(combined_topk_weights->sizes[1]);
            // Input shape validation by layout — must match FWD recv_topk_weights.
            assert(topk_weights->datatype == ncclFloat32);
            if (expert_major_in) {
                assert(topk_weights->ndim == 1 && "HT EM BWD combine: input topk_weights must be 1D [num_recv_tokens]");
                input_topk_stride = 1;
            } else {
                assert(
                    topk_weights->ndim == 2 &&
                    "HT FLAT/RM BWD combine: input topk_weights must be 2D [num_recv, top_k]");
                assert(
                    static_cast<int>(topk_weights->sizes[1]) == num_topk &&
                    "HT FLAT/RM BWD combine: input top_k must equal output top_k");
                input_topk_stride = num_topk;
            }
            NCCLCHECK(resolveTensorWindowBinding(
                group,
                topk_weights,
                &topk_weights_local,
                static_cast<uint64_t>(group->gin_config.rdma_intra_node_red_prob_offset),
                &topk_weights));
            NCCLCHECK(resolveTensorWindowBinding(
                group,
                combined_topk_weights,
                &combined_topk_weights_local,
                0,
                &combined_topk_weights));
        } else {
            // FWD combine: input topk_weights forbidden. outputs->topk_weights is unused
            // by the kernel here and left to caller bookkeeping (not validated).
            if (topk_weights != nullptr) {
                return ncclInvalidArgument;
            }
        }

        /* ===== Output tensors ===== */
        const ncclEpTensor_t* combined_x = tensor_required(outputs->tokens);
        if (combined_x->ndim == 0) {
            return ncclInvalidArgument;
        }
        assert(combined_x->ndim == 2);
        assert(combined_x->sizes[0] == num_combined_tokens); // Output should match original token count
        assert(combined_x->sizes[1] == hidden); // Should match input hidden dimension
        NCCLCHECK(resolveTensorWindowBinding(group, combined_x, &combined_x_local, 0, &combined_x));

        /* ===== Copy input to IPC staging buffers ===== */
        // Expert MLP output needs to be in IPC buffer so other ranks can read it
        const bool combine_x_uses_external_window = tensorUsesExternalWindow(group, x);
        if (group->config.zero_copy == NCCL_EP_ZERO_COPY_ON && !combine_x_uses_external_window) {
            fprintf(
                stderr,
                "NCCL EP: zero_copy requires ncclEpCombine inputs->tokens to be backed by a "
                "user-registered NCCL window (ncclCommWindowRegister)\n");
            return ncclInvalidArgument;
        }
        // em-permute path: gather caller EM x into FLAT staging via
        // local_permute_reduce so combine can run with FLAT layout. The kernel
        // also fuses a 1D EM to [num_flat, top_k] FLAT weight gather on BWD.
        const bool em_permute_combine = em_local_permute_enabled(group, handle);
        const bool em_permute_bwd_weights = em_permute_combine && backward_combine && expert_major_in;
        // em-permute combine always runs the local_permute_reduce gather
        // (caller EM x -> FLAT staging), regardless of external-window — x->data
        // is the resolved device pointer either way.
        if (em_permute_combine) {
            // x dtype already validated as NONE-mode at the combine entry gate;
            // local_permute_reduce handles bf16/fp16/fp32.
            const int row_bytes = hidden * ncclTypeSize(x->datatype);
            if (row_bytes <= 0 || (row_bytes % 16) != 0) {
                return ncclInvalidArgument; // int4-vectorized row copy requires 16B-aligned row
            }
            nccl_ep::hybridep::launch_combine_reduce(
                group->ht_buffers.expert_input_token,
                x->data,
                handle->hybridep.flat2em_slot_map,
                handle->hybridep.num_tokens_for_experts,
                em_permute_bwd_weights ? static_cast<const float*>(topk_weights->data) : nullptr,
                em_permute_bwd_weights ? handle->hybridep.recv_topk_weights_flat : nullptr,
                handle->num_topk,
                row_bytes,
                static_cast<int>(group->device_sm_count),
                group->prolog_epilog_sms,
                stream,
                x->datatype);
        } else if (!combine_x_uses_external_window) {
            // Clamp to staging capacity (nvlink_dup/local_dup EM only).
            const size_t clamped_tokens =
                std::min<size_t>(static_cast<size_t>(num_tokens), group->ht_buffers.token_staging_slots);
            size_t token_copy_size = clamped_tokens * hidden * ncclTypeSize(x->datatype);
            CUDA_CHECK(cudaMemcpyAsync(
                group->ht_buffers.expert_input_token,
                x->data,
                token_copy_size,
                cudaMemcpyDeviceToDevice,
                stream));
        }

        /* ===== Convert sparse topk_weights to dense prob for backward combine ===== */
        // For backward combine, convert sparse input weights to dense format for HT kernel
        if (backward_combine) {
            int experts_per_node = group->num_local_experts * group->lsa_team_size;
            size_t dense_prob_size = static_cast<size_t>(num_tokens) * experts_per_node * sizeof(float);

            // Zero-initialize the dense prob buffer before scattering
            CUDA_CHECK(cudaMemsetAsync(
                group->ht_buffers.combine_expert_input_prob_buffer_ptrs[group->lsa_rank],
                0,
                dense_prob_size,
                stream));

            // Scatter sparse weights into dense prob [num_recv, experts_per_node].
            // em-permute uses FLAT inputs (gathered above for EM, or passed
            // through for FLAT) keyed by the FLAT main LERM; nvlink_dup/local_dup EM keeps
            // the 1D EM input keyed by the EM-shape LERM scratch.
            const bool use_flat_inputs = em_permute_combine;
            const float* prob_input = (use_flat_inputs && expert_major_in) ?
                                          handle->hybridep.recv_topk_weights_flat :
                                          static_cast<const float*>(topk_weights->data);
            const int prob_stride = use_flat_inputs ? num_topk : input_topk_stride;
            const bool* lerm_for_combine = handle->hybridep.local_expert_routing_map;
            nccl_ep::hybridep::sparse_to_dense_prob_combine(
                prob_input,
                lerm_for_combine,
                group->ht_buffers.combine_expert_input_prob_buffer_ptrs[group->lsa_rank],
                num_tokens,
                prob_stride,
                group->num_local_experts, // experts_per_rank
                experts_per_node,
                group->lsa_rank,
                stream);
        }

        // Use pre-allocated dense prob buffer for backward combine
        float* dense_output_prob = handle->hybridep.dense_prob_buffer;
        if (backward_combine) {
            size_t dense_output_prob_size =
                static_cast<size_t>(num_combined_tokens) * group->config.num_experts * sizeof(float);
            CUDA_CHECK(cudaMemsetAsync(dense_output_prob, 0, dense_output_prob_size, stream));
        }

        /* ===== Build CombineParams ===== */
        // CombineParams encapsulates all buffers and metadata needed by HT combine kernel:
        //   - IPC input buffers: expert_input_token_ptrs, expert_input_prob_ptrs (per-rank pointers)
        //   - Output buffers: attn_output_token, attn_output_prob (user-provided)
        //   - RDMA buffers: rdma_intra_node_red_*, combine_rdma_inter_node_group_* (for multi-node)
        //   - Metadata: sparse_to_dense_map, rdma_to_attn_map, attn_to_rdma_map, local_expert_routing_map
        //   - Sync flags: combine_expected_*_flag_value, combine_intra_node_write_completion_flags
        nccl_ep::hybridep::CombineParams params;
        params.hidden_dim = hidden;
        params.experts_per_rank = group->num_local_experts;
        params.num_ranks_per_node = group->lsa_team_size;
        // Use HOST pointer arrays - these get copied into the kernel param struct for fast __grid_constant__ access
        std::vector<uint16_t*> combine_input_token_ptrs;
        if (combine_x_uses_external_window) {
            NCCLCHECK(buildIntranodePtrArray<uint16_t>(group, x, combine_input_token_ptrs));
            params.expert_input_token_ptrs = combine_input_token_ptrs.data();
        } else {
            params.expert_input_token_ptrs = group->ht_buffers.combine_expert_input_token_buffer_ptrs;
        }
        params.expert_input_prob_ptrs =
            backward_combine ? group->ht_buffers.combine_expert_input_prob_buffer_ptrs : nullptr;
        params.attn_output_token = combined_x->data;
        params.attn_output_prob = backward_combine ? dense_output_prob : nullptr;
        params.rdma_intra_node_red_token = is_single_node ? nullptr : group->ht_buffers.rdma_intra_node_red_token;
        params.rdma_intra_node_red_prob =
            (!is_single_node && backward_combine) ? group->ht_buffers.rdma_intra_node_red_prob : nullptr;
        params.combine_rdma_inter_node_group_token =
            is_single_node ? nullptr : group->ht_buffers.combine_rdma_inter_node_group_token;
        params.combine_rdma_inter_node_group_prob =
            (!is_single_node && backward_combine) ? group->ht_buffers.combine_rdma_inter_node_group_prob : nullptr;
        // Unified s2d: FLAT-shape for FLAT layout and EM em-permute mode;
        // EM-shape (packed rank/slot) only for EM kNvlinkDup / kLocalDup modes.
        params.sparse_to_dense_map = handle->hybridep.sparse_to_dense_map;
        const bool expert_major = (handle->layout == NCCL_EP_LAYOUT_EXPERT_MAJOR);
        params.s2d_inner_dim = (expert_major && !em_permute_combine) ? handle->num_topk : group->lsa_team_size;
        params.layout = em_permute_combine ? NCCL_EP_LAYOUT_RANK_MAJOR : handle->layout;
        assert(
            (params.layout == NCCL_EP_LAYOUT_EXPERT_MAJOR) ? (params.s2d_inner_dim == handle->num_topk) :
                                                             (params.s2d_inner_dim == group->lsa_team_size));
        params.rdma_to_attn_map = handle->hybridep.rdma_to_attn_map;
        params.attn_to_rdma_map = handle->hybridep.attn_to_rdma_map;
        params.local_expert_routing_map = handle->hybridep.local_expert_routing_map;
        // Always pass a valid device pointer — see dispatch path comment.
        params.combine_expected_rdma_flag_value = group->ht_buffers.dev_combine_expected_rdma;
        params.combine_rdma_inter_node_group_flags =
            is_single_node ? nullptr : group->ht_buffers.combine_rdma_inter_node_group_flags;
        params.combine_expected_intra_node_flag_value = group->ht_buffers.dev_combine_expected_intra;
        params.combine_grid_barrier_counter = group->ht_buffers.combine_grid_barrier_counter;
        params.combine_intra_node_write_completion_flags = group->ht_buffers.combine_intra_node_write_completion_flags;
        params.guard_enabled = !nccl_ep_env_flag_on(group->env.disable_guard);
        const ncclWindow_t combine_token_window =
            !combine_x_uses_external_window ? x->win_hdl : group->gin_config.nccl_window;
        const size_t combine_token_offset =
            is_single_node ? 0 :
                             (!combine_x_uses_external_window ? static_cast<size_t>(x->win_offset) :
                                                                group->gin_config.rdma_intra_node_red_token_offset);
        // Pass device communicators and windows
        // Always pass the devComm (single-node too): the HT LSA sync-guard now uses the
        // NCCL LSA barrier (needs comm.lsaBarrier). RDMA paths stay if-constexpr-gated.
        params.dcomms = group->gin_config.d_dcomms;
        params.nccl_token_window = combine_token_window;
        params.nccl_prob_window = !backward_combine ? ncclWindow_t{} : group->gin_config.nccl_window;
        params.nccl_internal_window = group->gin_config.nccl_window;
        params.num_gin_comms = is_single_node ? 0 : group->gin_config.num_comms;
        params.num_ctx_per_comm = is_single_node ? 0 : group->gin_config.num_ctx_per_comm;
        params.gin_base_ptr = is_single_node ? nullptr : group->gin_config.gin_base_ptr;
        params.signals_base = group->gin_config.signals_base;
        params.combine_signal_offset = group->gin_config.combine_signal_offset;
        // Use offsets relative to gin_base_ptr
        params.mr_info = {
            .rdma_intra_node_red_token_offset = combine_token_offset,
            .combine_rdma_inter_node_group_token_offset =
                is_single_node ? 0 : group->gin_config.combine_rdma_inter_node_group_token_offset,
            .rdma_intra_node_red_prob_offset = is_single_node ? 0 : group->gin_config.rdma_intra_node_red_prob_offset,
            .combine_rdma_inter_node_group_prob_offset =
                is_single_node ? 0 : group->gin_config.combine_rdma_inter_node_group_prob_offset,
            .guard_offset = is_single_node ? 0 : group->gin_config.combine_guard_offset,
        };
        params.local_rank = group->lsa_rank;
        params.node_rank = group->rdma_rank;
        params.num_tokens_per_rank = group->config.max_dispatch_tokens_per_rank;
        params.num_real_tokens = num_combined_tokens;
        params.num_recv_tokens = num_tokens;
        params.combine_local_reduce_enabled = em_local_dup_active(group, handle->layout);

        // Pre-sum secondary em_slots into primaries before combine reads them.
        if (params.combine_local_reduce_enabled) {
            assert(
                handle->hybridep.emuf_group_buf != nullptr && handle->hybridep.emuf_group_count != nullptr &&
                "unfused combine requires emuf dup-group buf from dispatch scan");
            nccl_ep::hybridep::call_local_reduce(
                /*expert_input_token=*/
                params.expert_input_token_ptrs[group->lsa_rank],
                /*expert_input_prob=*/
                backward_combine ? params.expert_input_prob_ptrs[group->lsa_rank] : nullptr,
                handle->hybridep.emuf_group_buf,
                handle->hybridep.emuf_group_count,
                handle->hybridep.emuf_group_stride,
                params.hidden_dim,
                params.experts_per_rank,
                params.num_ranks_per_node,
                backward_combine,
                static_cast<int>(group->prolog_epilog_sms),
                stream,
                x->datatype);
        }

        /* ===== Call combine kernel ===== */
        params.token_dtype = x->datatype;
        nccl_ep::hybridep::call_combine(
            params,
            group->ht_aligned_max_tokens, // chunk-aligned stride
            group->ht_tokens_per_chunk, // tokens per dispatch/combine chunk
            group->rdma_team_size, // num_nodes (RDMA domain size)
            backward_combine, // backward mode flag
            static_cast<int>(group->comm_num_sms),
            &group->env,
            stream);

        // BWD combine: scatter dense output back to FWD k-slot ordering via hybridep.topk_idx.
        if (backward_combine) {
            assert(
                handle->hybridep.topk_idx != nullptr &&
                "HT BWD combine: hybridep.topk_idx missing (ncclEpUpdateHandle not called?)");
            if (handle->topk_idx.datatype == ncclInt32) {
                nccl_ep::hybridep::dense_to_sparse_prob_combine(
                    dense_output_prob,
                    static_cast<const int32_t*>(handle->hybridep.topk_idx),
                    static_cast<float*>(combined_topk_weights->data),
                    num_combined_tokens,
                    num_topk,
                    group->config.num_experts,
                    stream);
            } else {
                nccl_ep::hybridep::dense_to_sparse_prob_combine(
                    dense_output_prob,
                    static_cast<const int64_t*>(handle->hybridep.topk_idx),
                    static_cast<float*>(combined_topk_weights->data),
                    num_combined_tokens,
                    num_topk,
                    group->config.num_experts,
                    stream);
            }
        }

        handle->cached_mode = true;
    }

    return ncclSuccess;
}

ncclResult_t ncclEpComplete(ncclEpHandle_t handle, const ncclEpCompleteConfig_t* config, cudaStream_t stream) {
    EP_OPTIONAL_STRUCT(config);
    if (handle->group->config.algorithm == NCCL_EP_ALGO_LOW_LATENCY) {
        if (handle->ll.continue_fn) {
            handle->ll.continue_fn(LOW_LATENCY_RECV_PHASE);
            handle->ll.continue_fn = nullptr;
        }
    } else if (handle->group->config.algorithm == NCCL_EP_ALGO_HIGH_THROUGHPUT) {
        // HT mode - no continue needed (synchronous)
    }
    return ncclSuccess;
}

ncclResult_t ncclEpMaskQuery(ncclEpGroup_t ep_group, int* mask_status, cudaStream_t stream) {
    EP_HOST_ASSERT(ep_group != nullptr);
    if (!ep_group->config.enable_mask) {
        return ncclInvalidUsage;
    }
    EP_HOST_ASSERT(ep_group->mask_buffer != nullptr && "ncclEpMaskQuery: enable_mask must be true");
    EP_HOST_ASSERT(mask_status != nullptr);
    CUDA_CHECK(cudaMemcpyAsync(
        mask_status,
        ep_group->mask_buffer,
        ep_group->nRanks * sizeof(int),
        cudaMemcpyDeviceToDevice,
        stream));
    return ncclSuccess;
}

ncclResult_t ncclEpMaskUpdate(ncclEpGroup_t ep_group, const int* mask, cudaStream_t stream) {
    EP_HOST_ASSERT(ep_group != nullptr);
    if (!ep_group->config.enable_mask) {
        return ncclInvalidUsage;
    }
    EP_HOST_ASSERT(ep_group->mask_buffer != nullptr && "ncclEpMaskUpdate: enable_mask must be true");
    EP_HOST_ASSERT(mask != nullptr);
    CUDA_CHECK(
        cudaMemcpyAsync(ep_group->mask_buffer, mask, ep_group->nRanks * sizeof(int), cudaMemcpyHostToDevice, stream));
    return ncclSuccess;
}

ncclResult_t ncclEpMaskClean(ncclEpGroup_t ep_group, cudaStream_t stream) {
    EP_HOST_ASSERT(ep_group != nullptr);

    if (!ep_group->config.enable_mask) {
        return ncclInvalidUsage;
    }
    EP_HOST_ASSERT(ep_group->mask_buffer != nullptr && "ncclEpMaskClean: enable_mask must be true");
    EP_HOST_ASSERT(ep_group->config.algorithm == NCCL_EP_ALGO_LOW_LATENCY);
    EP_HOST_ASSERT(
        ep_group->rdma_buffer != nullptr &&
        "ncclEpMaskClean: rdma_buffer not yet allocated; create at least one LL handle first");
    EP_HOST_ASSERT(ep_group->sync_buffer != nullptr && ep_group->sync_window != nullptr);

    // Reset internal RDMA recv-count/flag counters for both double-buffer slots
    // via a GPU kernel that includes a cross-rank barrier, then clear the mask.
    // Query the layout to read the clean_meta_offset values.
    nccl_ep::LowLatencyLayout layout(
        ep_group->config.max_dispatch_tokens_per_rank,
        ep_group->config.max_token_bytes,
        ep_group->nRanks,
        ep_group->config.num_experts,
        MAX_NUM_TOPK,
        NCCL_EP_LAYOUT_RANK_MAJOR);
    char* rdma_base = static_cast<char*>(ep_group->rdma_buffer);
    auto clean_0 = layout.buffers[0].clean_meta_offset();
    auto clean_1 = layout.buffers[1].clean_meta_offset();
    int* clean_0_ptr = reinterpret_cast<int*>(rdma_base + clean_0.first);
    int* clean_1_ptr = reinterpret_cast<int*>(rdma_base + clean_1.first);

    nccl_ep::internode_ll::CleanLowLatencyBufferParams clean_params{};
    clean_params.clean_0 = clean_0_ptr;
    clean_params.num_clean_int_0 = clean_0.second;
    clean_params.clean_1 = clean_1_ptr;
    clean_params.num_clean_int_1 = clean_1.second;
    clean_params.rankMask = ep_group->mask_buffer;
    clean_params.syncBuffer = static_cast<int*>(ep_group->sync_buffer);
    clean_params.syncWindow = ep_group->sync_window;
    clean_params.devComms = ep_group->nccl_dev_comms;
    clean_params.barrierSignalBase = ep_group->clean_barrier_signal_base;
    clean_params.timeoutCycles = ep_group->timeout_cycles;

    nccl_ep::internode_ll::call_clean_low_latency_buffer(clean_params, stream);

    // Reset all ranks to active (1 = active).
    // Sync the stream before returning so all_active outlives the async copy.
    std::vector<int> all_active(ep_group->nRanks, 1);
    CUDA_CHECK(cudaMemcpyAsync(
        ep_group->mask_buffer,
        all_active.data(),
        ep_group->nRanks * sizeof(int),
        cudaMemcpyHostToDevice,
        stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    return ncclSuccess;
}

ncclResult_t ncclEpGetAsyncError(ncclEpGroup_t ep_group, int* error_out) {
    EP_HOST_ASSERT(ep_group != nullptr);
    if (!ep_group->config.enable_mask) {
        return ncclInvalidUsage;
    }
    EP_HOST_ASSERT(ep_group->async_error_flag != nullptr && "ncclEpGetAsyncError: enable_mask must be true");
    EP_HOST_ASSERT(error_out != nullptr);
    *error_out = __atomic_load_n(ep_group->async_error_flag, __ATOMIC_ACQUIRE);
    return ncclSuccess;
}

ncclResult_t ncclEpErrorClear(ncclEpGroup_t ep_group) {
    EP_HOST_ASSERT(ep_group != nullptr);
    if (!ep_group->config.enable_mask) {
        return ncclInvalidUsage;
    }
    EP_HOST_ASSERT(ep_group->async_error_flag != nullptr && "ncclEpErrorClear: enable_mask must be true");
    __atomic_store_n(ep_group->async_error_flag, 0, __ATOMIC_RELEASE);
    return ncclSuccess;
}
// ── Test-only internal helpers (declared in nccl_ep_test_internal.h) ─────────
// Exposed via a separate test header so library consumers cannot accidentally
// depend on these implementation details.

const int32_t* ncclEpHandle_test_getSparseToDenseMap(ncclEpHandle_t handle) {
    return handle->hybridep.sparse_to_dense_map;
}

int ncclEpHandle_test_getNumTopk(ncclEpHandle_t handle) {
    return handle->num_topk;
}

int ncclEpHandle_test_getMaxTokensPerRank(ncclEpHandle_t handle) {
    return static_cast<int>(handle->group->config.max_dispatch_tokens_per_rank);
}

int ncclEpHandle_test_getNRanksPerNode(ncclEpHandle_t handle) {
    return handle->group->lsa_team_size;
}

int ncclEpHandle_test_getExpertsPerRank(ncclEpHandle_t handle) {
    return handle->group->num_local_experts;
}

ncclResult_t ncclEpHandle_test_getNumRecvTokens(ncclEpHandle_t handle, unsigned int* num_recv_tokens) {
    if (handle->group->config.algorithm != NCCL_EP_ALGO_HIGH_THROUGHPUT) {
        return ncclInvalidUsage;
    }
    // EM modes: total padded slots is the user-visible recv-token count.
    // em_scan_kernel publishes it as em_internal_offsets[experts_per_rank].
    // FLAT mode: num_tokens_for_experts holds the raw recv count.
    if (handle->layout == NCCL_EP_LAYOUT_EXPERT_MAJOR) {
        int64_t em_padded_total;
        CUDA_CHECK(cudaMemcpy(
            &em_padded_total,
            handle->hybridep.expert_token_offsets + handle->group->num_local_experts,
            sizeof(em_padded_total),
            cudaMemcpyDeviceToHost));
        assert(em_padded_total >= 0);
        *num_recv_tokens = static_cast<unsigned int>(em_padded_total);
        return ncclSuccess;
    }
    int32_t actual_recv_tokens;
    CUDA_CHECK(cudaMemcpy(
        &actual_recv_tokens,
        handle->hybridep.num_tokens_for_experts,
        sizeof(actual_recv_tokens),
        cudaMemcpyDeviceToHost));
    assert(actual_recv_tokens >= 0);
    *num_recv_tokens = static_cast<unsigned int>(actual_recv_tokens);
    return ncclSuccess;
}

void ncclEpHandle_test_clearTopkIdx(ncclEpHandle_t handle) {
    handle->topk_idx.data = nullptr;
}
