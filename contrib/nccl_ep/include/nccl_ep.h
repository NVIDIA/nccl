/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 * See LICENSE.txt for more license information.
 */

#pragma once

#include <cuda.h>
#include <nccl.h>
#include "ep_enums.h"

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// ABI + API versioning
//
// Every struct that crosses the API boundary starts with `unsigned int size`,
// which the caller MUST set to sizeof(struct). The library validates it
// against its own sizeof and rejects mismatches (layout / ABI check).
//
// ncclEpGroupConfig_t additionally carries `unsigned int version`
// (= NCCL_EP_API_VERSION) for catching feature-level incompatibilities that
// size can't detect; the library warns on mismatch.
//
// Convenience macros NCCL_EP_xxx_INIT expand to compound literals that pre-fill
// these fields. They work in declaration init, assignment, and expression
// contexts:
//
//   ncclEpDispatchInputs_t inputs = NCCL_EP_DISPATCH_INPUTS_INIT;
//   inputs.tokens = my_tokens;
//
//   // also valid as a post-declaration assignment / reset:
//   inputs = NCCL_EP_DISPATCH_INPUTS_INIT;
//
// To set additional fields inline, write the compound literal directly:
//
//   inputs = (ncclEpDispatchInputs_t){
//       .size   = sizeof(ncclEpDispatchInputs_t),
//       .tokens = my_tokens,
//   };
//
// FUTURE IMPROVEMENT: relax the strict size-equality check to allow forward
// compat when the caller's struct is larger than the library's known size, by
// scanning the trailing bytes; if all zero, accept silently (the caller didn't
// actually fill any unknown fields).
// ============================================================================
#define NCCL_EP_API_VERSION 1

// Opaque N-dimensional tensor handle used to describe various user inputs
// (i.e., tokens, top-k indices, weights, scales, etc.)
typedef struct ncclNDTensor* ncclNDTensor_t;

// Allocator and free function pointer types.
// context is the value stored in ncclEpAllocConfig_t::context and is forwarded unchanged
// on every call.
typedef cudaError_t (*ncclEpAllocFn_t)(void** ptr, size_t size, void* context);
typedef cudaError_t (*ncclEpFreeFn_t)(void* ptr, void* context);

// Device memory allocator configuration embedded in ncclEpGroupConfig_t.
typedef struct {
    // Optional custom device memory allocator (NULL for default cudaMalloc).
    ncclEpAllocFn_t alloc_fn;
    // Optional custom device memory free function (NULL for default cudaFree).
    ncclEpFreeFn_t free_fn;
    // Opaque pointer forwarded verbatim to every alloc_fn/free_fn call.
    void* context;
} ncclEpAllocConfig_t;

// EP group configuration structure
typedef struct {
    unsigned int size;                   // = sizeof(this struct); first field, never moves
    unsigned int version;                // = NCCL_EP_API_VERSION; caller's feature-set version
    ncclEpAlgorithm_t algorithm;         // low_latency or high_throughput
    // Receive buffer layout for the dispatch and combine path.
    // Determines the shape of recv_x on dispatch output and the expected input shape for combine.
    // Default (NCCL_EP_LAYOUT_AUTO / zero-init): auto-selected based on algorithm
    //   (EXPERT_MAJOR for LL, FLAT for HT).
    // HT mode only supports NCCL_EP_LAYOUT_FLAT.
    ncclEpLayout_t layout;
    unsigned int num_experts;            // Number of experts (required)
    // Maximum number of tokens any single rank will dispatch. Must be the same across all ranks.
    // Each rank should be prepared to receive up to max_send_tokens_per_rank * num_ranks tokens.
    // REQUIRED for both LL and HT modes (must be > 0).
    // In a future release, NCCL_EP_AUTO will be supported for HT mode,
    // in which case the received token count will be determined by ncclEpCreateHandle.
    unsigned int max_send_tokens_per_rank;
    // Upper bound on per-token bytes, covering both dispatch and combine.
    // The group sizes all token buffers from this; per-call sizes flow through
    // the input tensors' sizes/datatype and may be smaller. Independent of
    // element type — purely byte-oriented.
    unsigned int max_token_bytes;
    unsigned long int rdma_buffer_size;  // RDMA buffer size in bytes (NCCL_EP_AUTO for auto, defaults to a sufficiently large buffer for any algorithm)
    unsigned int num_qp_per_rank;        // Number of QPs per rank (NCCL_EP_AUTO for auto)
    // Number of channels per rank (NCCL_EP_AUTO for auto).
    // In high throughput collectives, each channel occupies 2 SMs
    unsigned int num_channels;
    // Total recv-slot budget per rank (across all source ranks).
    //   FLAT/RM: one slot per recv token; same token from different source ranks
    //            occupies distinct slots, but never duplicated within one source rank.
    //   EM:      additionally covers intra-rank duplication (token → multiple local
    //            experts) and per-expert alignment/padding to
    //            dispatch_output_per_expert_alignment.
    //   HT: required, must be >= max_send_tokens_per_rank.
    //   LL: AUTO/0 → nRanks*max_send_tokens_per_rank.
    unsigned int max_recv_token_slots_per_rank;
    // Maximum number of SMs to use for EP kernels (dispatch, combine, preprocessing).
    // Default: NCCL_EP_AUTO — algorithm-dependent default.
    unsigned int max_num_sms;
    // Device memory allocator; zero-init (all NULL) uses cudaMalloc/cudaFree.
    ncclEpAllocConfig_t alloc;
    // Enable active-mask support for fault tolerance (LL mode only).
    // When enabled, a per-rank mask buffer is allocated. If a remote rank times out
    // during dispatch or combine, it is automatically masked (skipped) rather than
    // causing a GPU trap. The mask can be queried, updated, and cleared via the
    // ncclEpMaskQuery / ncclEpMaskUpdate / ncclEpMaskClean APIs.
    // A host-visible error flag is also set on timeout, pollable via ncclEpGetAsyncError().
    bool enable_mask;
    // Timeout for GPU-side wait loops, in nanoseconds. 0 = use default (~100 s).
    // Can be overridden by the NCCL_EP_TIMEOUT_MS environment variable.
    // Setting too low risks false positives (slow ranks marked as failed).
    uint64_t timeout_ns;
} ncclEpGroupConfig_t;

#define NCCL_EP_GROUP_CONFIG_INIT ((ncclEpGroupConfig_t){ \
    .size    = (unsigned int)sizeof(ncclEpGroupConfig_t), \
    .version = NCCL_EP_API_VERSION })

// Opaque type forward declaration
typedef struct ncclEpGroup* ncclEpGroup_t;

// Create an EP group from an NCCL communicator
//   This call is collective and must be invoked by all ranks in the group.
//
// Arguments:
//   ep_group   - [OUT] Pointer to newly created EP group
//   comm       - [IN]  Existing NCCL communicator
//   config     - [IN]  Pointer to EP configuration structure.
//
// Returns:
//   ncclResult_t error code

ncclResult_t ncclEpCreateGroup(
    ncclEpGroup_t* ep_group,
    ncclComm_t comm,
    const ncclEpGroupConfig_t* config
);

// Destroy an EP group and release associated resources.
//
// Arguments:
//   ep_group     - [IN]  EP group to destroy
//
// Returns:
//   ncclResult_t error code

ncclResult_t ncclEpGroupDestroy(
    ncclEpGroup_t ep_group
);


// Wrap a caller-provided device buffer in a tensor descriptor.
//   The implementation guarantees that the tensor is contiguous in memory (including accordingly
//   setting the strides to 1 for all dimensions).
//
//   The buffer is NOT owned by the tensor; the caller is responsible for the lifetime of `data`
//   and must keep it valid until ncclEpTensorDestroy returns. Releasing `data` while the tensor
//   handle is still live leaves the handle dangling — any subsequent use is undefined behavior.
//   Recommended teardown order: ncclEpTensorDestroy(t) first, then free the buffer.
//
// Arguments:
//   tensor   - [OUT] Pointer to newly created tensor
//   ndim     - [IN]  Number of dimensions
//   datatype - [IN]  Data type
//   data     - [IN]  Non-null device pointer to the tensor's storage
//   sizes    - [IN]  Array of ndim dimension sizes
//
// Returns: ncclResult_t error code

ncclResult_t ncclEpTensorCreate(
    ncclNDTensor_t* tensor,
    unsigned int ndim,
    ncclDataType_t datatype,
    void* data,
    const size_t* sizes
);

// Create a tensor from a registered NCCL window.
//   The tensor stores the window handle and offset. Its local data pointer is
//   resolved lazily when the tensor is used with an EP group.
//   ncclEpTensorGetData returns ncclInvalidUsage until that resolution happens.
//
//   The window is NOT owned by the tensor; the caller must keep the window
//   registered and valid until ncclEpTensorDestroy returns.
ncclResult_t ncclEpTensorCreateFromWindow(
    ncclNDTensor_t* tensor,
    unsigned int ndim,
    ncclDataType_t datatype,
    ncclWindow_t win,
    uint64_t win_offset,
    const size_t* sizes
);

// Destroy a tensor descriptor.
//   Only the descriptor is freed; the underlying data buffer is the caller's responsibility.
//
// Arguments:
//   tensor       - [IN]  Tensor handle to destroy
//
// Returns: ncclResult_t error code

ncclResult_t ncclEpTensorDestroy(
    ncclNDTensor_t tensor
);

// Layout info passed to ncclEpCreateHandle / ncclEpUpdateHandle and ncclEpDispatch.
// All fields are optional (NULL = not provided).
typedef struct {
    unsigned int   size;                // = sizeof(this struct); first field, never moves
    ncclNDTensor_t expert_counters; // 1D [num_local_experts] int32 (or int64 for HT EM)
                                        //   HT (handle time): per-expert recv counts. Flat: unpadded int32.
                                        //                     EM: padded counts (sum equals output slot count).
                                        //   LL expert-major: per-expert received token counts (dispatch time).
    ncclNDTensor_t src_rank_counters;    // 1D [num_ranks] int32
                                        //   LL rank-major only: per-source-rank token counts (dispatch time).
    ncclNDTensor_t expert_offsets; // 1D [num_local_experts] int32 or int64
                                        //   HT expert-major only: prefix sum of padded per-expert counts.
    ncclNDTensor_t recv_total_counter;  // 1D [1] int32 or int64
                                        //   HT: scalar total recv token count. Flat: unpadded. EM: padded slot total.
} ncclEpLayoutInfo_t;

#define NCCL_EP_LAYOUT_INFO_INIT ((ncclEpLayoutInfo_t){ .size = (unsigned int)sizeof(ncclEpLayoutInfo_t) })

// Input tensors for ncclEpDispatch.
// All fields except tokens are optional (NULL = not provided).
typedef struct {
    unsigned int   size;         // = sizeof(this struct); first field, never moves
    ncclNDTensor_t tokens;       // required; 2D [num_tokens, hidden]
    ncclNDTensor_t topk_weights; // optional; 2D [num_tokens, top_k], ncclFloat32
                                 //   LL rank-major: per-token routing weights
                                 //   HT forward: routing weights (topk_idx taken from handle)
    ncclNDTensor_t scales;       // optional; HT FP8 only; 2D [num_tokens, hidden/128], ncclFloat32
} ncclEpDispatchInputs_t;

#define NCCL_EP_DISPATCH_INPUTS_INIT ((ncclEpDispatchInputs_t){ .size = (unsigned int)sizeof(ncclEpDispatchInputs_t) })

// Output tensors for ncclEpDispatch.
// All fields except tokens are optional (NULL = not provided).
typedef struct {
    unsigned int   size;         // = sizeof(this struct); first field, never moves
    ncclNDTensor_t tokens;       // required; received tokens
    ncclNDTensor_t topk_weights; // optional; LL rank-major or HT: received top-k weights
    ncclNDTensor_t scales;       // optional; FP8 only; received per-token scaling factors
    ncclNDTensor_t topk_idx;     // optional; LL rank-major or HT: received top-k expert indices
} ncclEpDispatchOutputs_t;

#define NCCL_EP_DISPATCH_OUTPUTS_INIT ((ncclEpDispatchOutputs_t){ .size = (unsigned int)sizeof(ncclEpDispatchOutputs_t) })

// Input tensors for ncclEpCombine.
// All fields except tokens are optional (NULL = not provided).
typedef struct {
    unsigned int   size;         // = sizeof(this struct); first field, never moves
    ncclNDTensor_t tokens;       // required; post-expert activation tensor
    ncclNDTensor_t topk_weights; // optional; HT backward combine only:
                                 //   2D [num_recv_tokens, top_k], ncclFloat32
} ncclEpCombineInputs_t;

#define NCCL_EP_COMBINE_INPUTS_INIT ((ncclEpCombineInputs_t){ .size = (unsigned int)sizeof(ncclEpCombineInputs_t) })

// Output tensors for ncclEpCombine.
// All fields except tokens are optional (NULL = not provided).
typedef struct {
    unsigned int   size;         // = sizeof(this struct); first field, never moves
    ncclNDTensor_t tokens;       // required; combined output in original token order
    ncclNDTensor_t topk_weights; // optional; 2D [num_tokens, top_k], ncclFloat32
                                 //   LL expert-major: per-token routing weights applied on receive side
                                 //   HT backward: combined routing weights output
} ncclEpCombineOutputs_t;

#define NCCL_EP_COMBINE_OUTPUTS_INIT ((ncclEpCombineOutputs_t){ .size = (unsigned int)sizeof(ncclEpCombineOutputs_t) })

// Opaque type forward declaration
typedef struct ncclEpHandle* ncclEpHandle_t;
typedef struct {
    unsigned int size;  // = sizeof(this struct); first field, never moves
    unsigned int use_fp8;  // enable FP8 for dispatch (0 = false, non-zero = true; default: 0)
    // HT expert-major only: per-expert zone alignment in tokens (pow2; 0/1 = no padding).
    // Padded slots are zero-filled by dispatch.
    size_t dispatch_output_per_expert_alignment;
} ncclEpHandleConfig_t;

#define NCCL_EP_HANDLE_CONFIG_INIT ((ncclEpHandleConfig_t){ .size = (unsigned int)sizeof(ncclEpHandleConfig_t) })

// Create and initialize an EP handle.
//   * Performs dispatch setup and (in HT mode only) metadata exchange.
//   * This call is collective and must be invoked by all ranks in the group.
//
// Arguments:
//   handle              - [OUT] Pointer to newly created and initialized EP handle
//   ep_group            - [IN]  A valid EP group
//   topk_idx            - [IN]  Tensor holding top-K expert indices (routing information)
//   layout_info         - [IN/OUT, optional] Layout info (see ncclEpLayoutInfo_t). NULL = none.
//                         HT: set expert_counters when max_send_tokens_per_rank is NCCL_EP_AUTO.
//                         LL mode: must be NULL.
//   config              - [IN]  Handle configuration (see ncclEpHandleConfig_t). NULL = defaults.
//   stream              - [IN]  CUDA stream
//
// Notes:
//   - If max_send_tokens_per_rank in ncclEpGroupConfig_t was set to NCCL_EP_AUTO,
//     this call may block as the host allocates memory for the actual number
//     of received tokens.
//
// Returns: ncclResult_t error code

ncclResult_t ncclEpCreateHandle(
    ncclEpHandle_t* handle,
    ncclEpGroup_t ep_group,
    ncclNDTensor_t topk_idx,
    const ncclEpLayoutInfo_t* layout_info,  // NULL = none
    const ncclEpHandleConfig_t* config,  // NULL = defaults
    cudaStream_t stream
);

// Destroy an EP handle and release all associated resources.
//
// Arguments:
//   handle         - [IN]  EP handle to destroy
//
// Returns: ncclResult_t error code

ncclResult_t ncclEpHandleDestroy(
    ncclEpHandle_t handle
);

// Query the device bytes required for a handle's routing buffers.
//
// Arguments:
//   ep_group  - [IN]  A valid EP group
//   config    - [IN]  Handle configuration (see ncclEpHandleConfig_t). NULL = defaults.
//   size_out  - [OUT] Required bytes for handle_mem
//   num_topk  - [IN]  Required for LL (> 0); optional for HT
//
// Returns: ncclResult_t error code

ncclResult_t ncclEpHandleMemSize(
    ncclEpGroup_t               ep_group,
    const ncclEpHandleConfig_t* config,
    size_t*                     size_out,
    int                         num_topk
);

// Allocate handle buffers without performing any collective.
// Call ncclEpUpdateHandle before the first ncclEpDispatch/ncclEpCombine.
//
// handle_mem == NULL:  NCCL EP allocates via alloc_fn; handle owns the memory.
// handle_mem != NULL:  wraps caller-owned 1D ncclUint8 tensor (>= ncclEpHandleMemSize);
//                      handle owns no memory; ncclEpHandleDestroy frees only the struct.
//
// Arguments:
//   handle     - [OUT] Newly created handle
//   ep_group   - [IN]  A valid EP group
//   config     - [IN]  Handle configuration (see ncclEpHandleConfig_t). NULL = defaults.
//   num_topk   - [IN]  Required for LL (> 0); pass -1 for HT
//   handle_mem - [IN]  NULL = internal alloc; non-NULL = caller-owned device buffer
//
// Returns: ncclResult_t error code

ncclResult_t ncclEpInitHandle(
    ncclEpHandle_t*             handle,
    ncclEpGroup_t               ep_group,
    const ncclEpHandleConfig_t* config,
    int                         num_topk,
    ncclNDTensor_t              handle_mem
);

// Per-step collective: prepare the handle for the given top-k routing decisions.
// Must be called after ncclEpInitHandle and before ncclEpDispatch.
//
// Arguments:
//   handle             - [IN]  Handle from ncclEpInitHandle
//   topk_idx           - [IN]  [num_tokens, top_k] int64
//   layout_info      - [IN/OUT, optional] Named local tensors (NULL = none provided).
//                         HT: layout_info->expert_counters is required when
//                         max_send_tokens_per_rank is NCCL_EP_AUTO.
//                         LL mode: must be NULL.
//   stream             - [IN]  CUDA stream
//
// Returns: ncclResult_t error code

ncclResult_t ncclEpUpdateHandle(
    ncclEpHandle_t handle,
    ncclNDTensor_t topk_idx,
    const ncclEpLayoutInfo_t* layout_info,  // NULL = none
    cudaStream_t stream
);

// EP dispatch configuration structure
typedef struct {
    unsigned int size;         // = sizeof(this struct); first field, never moves
    unsigned int send_only;    // if non-zero, only initiate transfers; requires ncclEpComplete() afterward
                               //   supported for LL mode only; output tensors must still be preallocated
    unsigned int round_scales; // whether to round the scaling factors tensor into a power of 2
} ncclEpDispatchConfig_t;

#define NCCL_EP_DISPATCH_CONFIG_INIT ((ncclEpDispatchConfig_t){ .size = (unsigned int)sizeof(ncclEpDispatchConfig_t) })

// Perform EP dispatch
//   * Sends tokens and metadata to the experts according to routing decisions.
//   * This call is collective and must be invoked by all ranks in the group.
//
// Arguments:
//   handle        - [IN,OUT] EP handle. The handle's topk_idx (set via ncclEpUpdateHandle / ncclEpCreateHandle)
//                            is used by HT forward dispatch. For HT backward dispatch or LL mode,
//                            set topk_idx to NULL when calling ncclEpUpdateHandle.
//   inputs        - [IN]     Named input tensors (see ncclEpDispatchInputs_t).
//                            inputs->tokens is required; other fields are optional.
//   outputs       - [IN,OUT] Named preallocated output tensors (see ncclEpDispatchOutputs_t).
//                            outputs->tokens is required; other fields are optional.
//                            For HT (NCCL_EP_LAYOUT_FLAT): outputs->tokens is [N(r) x hidden] (2D),
//                                    where N(r) = num_ranks * max_send_tokens_per_rank for static allocation,
//                                    or the actual received count when max_send_tokens_per_rank is NCCL_EP_AUTO.
//                            For LL expert-major: outputs->tokens is [local_experts x num_recv_tokens x hidden] (3D).
//                            For LL rank-major: outputs->tokens is [num_recv_tokens x hidden] (2D);
//                                    outputs->topk_weights and outputs->topk_idx must also be provided.
//   layout_info - [IN,OUT] Named local tensors (see ncclEpLayoutInfo_t). NULL = none.
//                            LL expert-major: layout_info->expert_counters receives per-expert token counts.
//                            LL rank-major: layout_info->src_rank_counters receives per-source-rank token counts.
//   config        - [IN]     Dispatch configuration (see ncclEpDispatchConfig_t). NULL = defaults.
//   stream        - [IN]     CUDA stream. If `ncclEpDispatch()` is called on a different stream than the stream used in
//                            `ncclEpCreateHandle()`,
//                            it is the responsibility of the user to synchronize between streams to ensure correctness.
//
// Returns:
//   ncclResult_t error code

ncclResult_t ncclEpDispatch(
    ncclEpHandle_t handle,
    const ncclEpDispatchInputs_t* inputs,
    const ncclEpDispatchOutputs_t* outputs,
    const ncclEpLayoutInfo_t* layout_info,  // NULL = none
    const ncclEpDispatchConfig_t* config,   // NULL = defaults
    cudaStream_t stream
);

typedef struct {
    unsigned int size;         // = sizeof(this struct); first field, never moves
    unsigned int send_only;    // if non-zero, only initiate transfers; requires ncclEpComplete() afterward
                               //   supported for LL mode only; output tensors must still be preallocated
} ncclEpCombineConfig_t;

#define NCCL_EP_COMBINE_CONFIG_INIT ((ncclEpCombineConfig_t){ .size = (unsigned int)sizeof(ncclEpCombineConfig_t) })

// Perform EP combine
//   * Gathers outputs from experts and returns them to their source in original token order.
//   * This call is collective and must be invoked by all ranks in the group.
//
// Arguments:
//   handle           - [IN,OUT] EP handle that was used for `ncclEpDispatch()` operation
//   inputs           - [IN]     Named input tensors (see ncclEpCombineInputs_t).
//                               inputs->tokens is required; other fields are optional.
//                               For HT (NCCL_EP_LAYOUT_FLAT): inputs->tokens is [N(r) x hidden] (2D).
//                               For LL expert-major: inputs->tokens is [local_experts x num_recv_tokens x hidden] (3D).
//                               For LL rank-major: inputs->tokens is [num_recv_tokens x hidden] (2D),
//                                       pre-reduced across local experts by the caller before this call.
//                               HT backward: inputs->topk_weights must also be provided.
//   outputs          - [IN,OUT] Named preallocated output tensors (see ncclEpCombineOutputs_t).
//                               outputs->tokens is required; 2D [num_tokens x hidden], restored to original order.
//                               outputs->topk_weights:
//                                 LL expert-major: per-token routing weights applied on the combine receive side.
//                                 HT backward: must also be provided; receives combined routing weights.
//   config           - [IN]     Combine configuration (see ncclEpCombineConfig_t). NULL = defaults.
//   stream           - [IN]     CUDA stream. If `ncclEpCombine()` is called on a different stream than the stream
//                               used in `ncclEpCreateHandle()`, it is the responsibility of the user to synchronize
//                               between streams to ensure correctness.
//
// Returns:
//   ncclResult_t error code

ncclResult_t ncclEpCombine(
    ncclEpHandle_t handle,
    const ncclEpCombineInputs_t* inputs,
    const ncclEpCombineOutputs_t* outputs,
    const ncclEpCombineConfig_t* config,    // NULL = defaults
    cudaStream_t stream
);

// Reserved config struct; callers must pass NULL today. The single
// placeholder member exists so the struct has a complete type (cybind /
// pycparser require this), and so the typedef stays struct-form (rather
// than pointer-form) — that way callers can spell pointer-to-const as
// `const ncclEpCompleteConfig_t*` naturally.
typedef struct ncclEpCompleteConfig {
    char _reserved;
} ncclEpCompleteConfig_t;

// Continues a staged EP operation to completion.
//   * This should be called after a prior `ncclEpDispatch()` or `ncclEpCombine()` call with `send_only` flag set.
//
// Arguments:
//   handle     - [IN,OUT] EP handle used in the preceding staged operation
//   config     - [IN]     Reserved for future options (must be NULL).
//   stream     - [IN]     CUDA stream
//
// Notes:
//   - If `ncclEpComplete()` is called on a different stream than the operation initiation call
//     (i.e., `ncclEpDispatch()` or `ncclEpCombine()`), it is the responsibility of the user to
//     synchronize between streams to ensure correctness.
//   - Only LL mode is supported.
//
// Returns:
//   ncclResult_t error code

ncclResult_t ncclEpComplete(
    ncclEpHandle_t handle,
    const ncclEpCompleteConfig_t* config,
    cudaStream_t stream
);

// Get the data pointer from a tensor.
//   Window-backed tensors return ncclInvalidUsage until they have been used
//   with an EP group and their data pointer has been resolved.
//
// Arguments:
//   tensor   - [IN]   Tensor handle
//   data     - [OUT]  Pointer to receive the data pointer
//
// Returns: ncclResult_t error code

ncclResult_t ncclEpTensorGetData(
    ncclNDTensor_t tensor,
    void** data
);

// Get the sizes and number of dimensions of a tensor.
//
// Arguments:
//   tensor   - [IN]   Tensor handle
//   sizes    - [OUT]  Pointer to receive the sizes array pointer (not a copy)
//   ndim     - [OUT]  Pointer to receive the number of dimensions
//
// Returns: ncclResult_t error code

ncclResult_t ncclEpTensorGetSizes(
    ncclNDTensor_t tensor,
    const size_t** sizes,
    unsigned int* ndim
);


// Query the active-mask status of all ranks.
//   Copies the mask buffer to a user-provided device tensor.
//   Requires enable_mask=true in the group config.
//
// Arguments:
//   ep_group     - [IN]  EP group with masking enabled
//   mask_status  - [OUT] Device pointer to int[nRanks]. 1 = active, 0 = masked (failed).
//   stream       - [IN]  CUDA stream
//
// Returns: ncclResult_t error code

ncclResult_t ncclEpMaskQuery(
    ncclEpGroup_t ep_group,
    int* mask_status,
    cudaStream_t stream
);

// Set the mask for all ranks at once.
//   Requires enable_mask=true in the group config.
//
// Arguments:
//   ep_group   - [IN] EP group with masking enabled
//   mask       - [IN] Host pointer to int[nRanks]. 1 = active, 0 = masked (failed).
//   stream     - [IN] CUDA stream
//
// Returns: ncclResult_t error code

ncclResult_t ncclEpMaskUpdate(
    ncclEpGroup_t ep_group,
    const int* mask,
    cudaStream_t stream
);

// Reset masks and RDMA buffers so previously masked ranks can re-join.
//   Collective: all surviving ranks must call simultaneously.
//   Resets RDMA buffers via a cross-rank barrier and sets all masks to active.
//   Does NOT reset the async error flag — call ncclEpErrorClear() separately.
//   Note: this API is for re-admitting a delayed rank within the same
//   communicator. Rank replacement requires a new communicator (e.g.,
//   ncclCommGrow) and a new EP group.
//   Requires enable_mask=true in the group config.
//
// Arguments:
//   ep_group - [IN] EP group with masking enabled
//   stream   - [IN] CUDA stream
//
// Returns: ncclResult_t error code

ncclResult_t ncclEpMaskClean(
    ncclEpGroup_t ep_group,
    cudaStream_t stream
);

// Poll for asynchronous errors (e.g., rank timeout).
//   Lightweight host-side check — reads a pinned CPU flag, no GPU sync required.
//   The flag is set by the kernel when a timeout masks a rank; clear it
//   explicitly via ncclEpErrorClear().
//   Requires enable_mask=true in the group config.
//
// Arguments:
//   ep_group  - [IN]  EP group with masking enabled
//   error_out - [OUT] 0 = no error, 1 = timeout occurred (one or more ranks masked)
//
// Returns: ncclResult_t error code

ncclResult_t ncclEpGetAsyncError(
    ncclEpGroup_t ep_group,
    int* error_out
);

// Clear the async error flag.
//   Lightweight host-side reset — writes zero to the pinned CPU flag.
//   Use after detecting an error (via ncclEpGetAsyncError) to re-arm the flag
//   for detecting new failures. Should be called after ncclEpMaskClean (full
//   recovery) or standalone when surviving ranks continue in degraded mode.
//   Requires enable_mask=true in the group config.
//
// Arguments:
//   ep_group - [IN] EP group with masking enabled
//
// Returns: ncclResult_t error code

ncclResult_t ncclEpErrorClear(
    ncclEpGroup_t ep_group
);

#ifdef __cplusplus
}
#endif
