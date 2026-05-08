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

typedef enum {
    NCCL_EP_TENSOR_FLAG_NONE = 0,
} ncclEpTensorFlags_t; // Reserved for future use

// Opaque N-dimensional tensor handle used to describe various user inputs
// (i.e., tokens, top-k indices, weights, scales, etc.)
typedef struct ncclNDTensor* ncclNDTensor_t;

// EP group configuration structure
typedef struct {
    unsigned int version;                // Structure version (set to 1.0.0)
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
    unsigned int token_size_bytes;       // Token size for buffer allocation (independent of datatype)
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
} ncclEpGroupConfig_t;

// Opaque type forward declaration
typedef struct ncclEpGroup* ncclEpGroup_t;

// Allocator and free function pointer types (matching cudaMalloc/cudaFree signatures)
typedef cudaError_t (*ncclEpAllocFn_t)(void** ptr, size_t size);
typedef cudaError_t (*ncclEpFreeFn_t)(void* ptr);

// Create an EP group from an NCCL communicator
//   This call is collective and must be invoked by all ranks in the group.
//
// Arguments:
//   ep_group   - [OUT] Pointer to newly created EP group
//   comm       - [IN]  Existing NCCL communicator
//   config     - [IN]  Pointer to EP configuration structure
//   alloc_fn   - [IN]  Optional custom allocator function (NULL for default cudaMalloc)
//   free_fn    - [IN]  Optional custom free function (NULL for default cudaFree)
//
// Returns:
//   ncclResult_t error code

ncclResult_t ncclEpCreateGroup(
    ncclEpGroup_t* ep_group,
    ncclComm_t comm,
    const ncclEpGroupConfig_t* config,
    ncclEpAllocFn_t alloc_fn = nullptr,
    ncclEpFreeFn_t free_fn = nullptr
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
//   tensor       - [OUT] Pointer to newly created tensor
//   ndim         - [IN]  Number of dimensions (1..5)
//   datatype     - [IN]  Data type
//   tag          - [IN]  Tensor identification tag
//   data         - [IN]  Non-null device pointer to the tensor's storage
//   size0..size4 - [IN]  Dimension sizes
//
// Returns: ncclResult_t error code

ncclResult_t ncclEpTensorCreate(
    ncclNDTensor_t* tensor,
    unsigned int ndim,
    ncclDataType_t datatype,
    ncclEpTensorTag_t tag,
    void* data,
    unsigned int size0,
    unsigned int size1 = 1,
    unsigned int size2 = 1,
    unsigned int size3 = 1,
    unsigned int size4 = 1
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
    ncclEpTensorTag_t tag,
    ncclWindow_t win,
    uint64_t win_offset,
    unsigned int size0,
    unsigned int size1 = 1,
    unsigned int size2 = 1,
    unsigned int size3 = 1,
    unsigned int size4 = 1
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

// Per-handle config (pass to ncclEpCreateHandle/InitHandle; NULL = defaults).
struct ncclEpHandleConfig {
    // HT expert-major only: per-expert zone alignment in tokens (pow2; 0/1 = no padding).
    // Padded slots are zero-filled by dispatch.
    size_t dispatch_output_per_expert_alignment;
};

// Opaque type forward declaration
typedef struct ncclEpHandle* ncclEpHandle_t;
typedef struct ncclEpHandleConfig ncclEpHandleConfig_t;

// Create and initialize an EP handle.
//   * Performs dispatch setup and (in HT mode only) metadata exchange.
//   * This call is collective and must be invoked by all ranks in the group.
//
// Arguments:
//   handle              - [OUT] Pointer to newly created and initialized EP handle
//   ep_group            - [IN]  A valid EP group
//   topk_idx            - [IN]  Tensor holding top-K expert indices (routing information)
//   local_tensors       - [IN/OUT, optional] Array of pointers to local tensors.
//                         Same set of tags as ncclEpUpdateHandle (see below).
//                         Required when max_send_tokens_per_rank is NCCL_EP_AUTO.
//                         LL mode: does not accept local tensors (num_local_tensors must be 0).
//   num_local_tensors   - [IN]  Number of local tensors.
//   config              - [IN]  Optional handle config (NULL uses defaults)
//   stream              - [IN]  CUDA stream
//   use_fp8             - [IN]  Enable FP8 for dispatch (default: false)
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
    const ncclNDTensor_t* local_tensors,
    unsigned int num_local_tensors,
    const ncclEpHandleConfig_t* config,
    cudaStream_t stream,
    bool use_fp8 = false
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
//   config    - [IN]  Optional handle config (NULL uses defaults)
//   size_out  - [OUT] Required bytes for handle_mem
//   num_topk  - [IN]  Required for LL (> 0); optional for HT
//
// Returns: ncclResult_t error code

ncclResult_t ncclEpHandleMemSize(
    ncclEpGroup_t               ep_group,
    const ncclEpHandleConfig_t* config,
    size_t*                     size_out,
    int                         num_topk = -1
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
//   config     - [IN]  Optional handle config (NULL uses defaults)
//   num_topk   - [IN]  Required for LL (> 0); optional for HT (default: -1)
//   use_fp8    - [IN]  Enable FP8 dispatch (default: false)
//   handle_mem - [IN]  NULL = internal alloc; non-NULL = caller-owned device buffer
//
// Returns: ncclResult_t error code

ncclResult_t ncclEpInitHandle(
    ncclEpHandle_t*             handle,
    ncclEpGroup_t               ep_group,
    const ncclEpHandleConfig_t* config,
    int                         num_topk   = -1,
    bool                        use_fp8    = false,
    ncclNDTensor_t              handle_mem = nullptr
);

// Per-step collective: prepare the handle for the given top-k routing decisions.
// Must be called after ncclEpInitHandle and before ncclEpDispatch.
//
// Arguments:
//   handle             - [IN]  Handle from ncclEpInitHandle
//   topk_idx           - [IN]  [num_tokens, top_k] int64
//   local_tensors      - [IN/OUT, optional] Array of pointers to local tensors.
//                         HT optional 1D outputs (ncclInt32 or ncclInt64; sizes per tag):
//                           RECV_EXPERT_COUNTER_DEVICE [num_local_experts] — per-expert recv
//                             counts (FLAT unpadded int32 only; EM padded). Optional.
//                           RECV_EXPERT_OFFSETS_DEVICE [num_local_experts] — EM only,
//                             prefix-sum of padded counts (start index per expert).
//                           RECV_TOTAL_COUNTER_DEVICE [1] — total recv tokens
//                             (FLAT: unpadded; EM: padded slot total).
//                         LL mode: does not accept local tensors (num_local_tensors must be 0).
//   num_local_tensors  - [IN]  Number of local tensors
//   stream             - [IN]  CUDA stream
//
// Returns: ncclResult_t error code

ncclResult_t ncclEpUpdateHandle(
    ncclEpHandle_t handle,
    ncclNDTensor_t topk_idx,
    const ncclNDTensor_t* local_tensors,
    unsigned int num_local_tensors,
    cudaStream_t stream
);

// EP dispatch configuration structure
typedef struct {
    unsigned int round_scales;          // whether to round the scaling factors tensor into a power of 2
} ncclEpDispatchConfig_t;

// Perform EP dispatch
//   * Sends tokens and metadata to the experts according to routing decisions.
//   * This call is collective and must be invoked by all ranks in the group.
//   * All tensors are tagged using tags with `NCCL_EP_TENSOR_TAG` prefix
//     to indicate the types of tensor (i.e., tokens, topK indices, weights, etc.)
//
// Arguments:
//   handle        - [IN,OUT] EP handle
//   inputs        - [IN]     Array of pointers to input tensors;
//                            all must be 2D [num_tokens x data_size].
//                            The number of tokens must be equal across all tensors, but data_size may vary.
//                            Tensors are used to describe distinct pieces of data exchanged with experts.
//                            Must include token tensor (NCCL_EP_TENSOR_TAG_TOKENS) and
//                            (depending on the algorithm) may optionally be extended with metadata tensors
//                            (i.e., topK indices, weights, scales, etc.).
//   num_inputs    - [IN]     Number of input tensors
//   outputs       - [IN,OUT] Array of pointers to preallocated output tensors, provided in the same order
//                            as input tensors;
//                            If the datatypes of input and output token tensors are diffent,
//                            then the additional output tensor with tag (NCCL_EP_TENSOR_TAG_SCALES) for scaling
//                            factors must be supplied.
//                            Scaling will be applied during the collective and the output tensor will be scaled.
//                            For HT (NCCL_EP_LAYOUT_FLAT): output tensors are [N(r) x data_size] (2D),
//                                    where N(r) = num_ranks * max_send_tokens_per_rank for static allocation,
//                                    or the actual received count when max_send_tokens_per_rank is NCCL_EP_AUTO.
//                                    Tokens arrive as a contiguous flat sequence; no rank or expert structure.
//                                    Optional routing outputs:
//                                    TOPK_WEIGHTS:      [OUT] 2D [N(r) x num_topk], ncclFloat32.
//                                    TOPK_IDX:          [OUT] 2D [N(r) x num_topk], ncclInt64.
//                                                       (recv topk indices/weights reuse the TOPK_IDX /
//                                                        TOPK_WEIGHTS tags in outputs.)
//                            For HT (NCCL_EP_LAYOUT_EXPERT_MAJOR): outputs are
//                                    [N(r) x data_size] (2D, same as FLAT). Caller derives per-expert
//                                    slot zones from RECV_EXPERT_OFFSETS_DEVICE / RECV_EXPERT_COUNTER_DEVICE.
//                                    Each slot is per (source_token, local_expert), so:
//                                    TOPK_WEIGHTS:      [OUT] 1D [N(r)], ncclFloat32 — single weight per slot.
//                                    TOPK_IDX:          not populated (slot encodes expert); omit from outputs.
//                            For LL: the dimensions of output tensors are
//                                    [local_experts x num_recv_tokens x data_size] (3D, expert-major).
//                                    The dimensions of the scaling factors tensor are:
//                                    [local_experts x num_recv_tokens x (hidden / 128)] (3D, ncclFloat32)
//                                    where num_recv_tokens = num_ranks * max_send_tokens_per_rank.
//                            For LL rank-major: outputs are [num_recv_tokens x data_size] (2D),
//                                    where num_recv_tokens = num_ranks * max_send_tokens_per_rank.
//                                    Additionally requires two routing output tensors:
//                                    TOPK_IDX:          [OUT] 2D [num_recv_tokens x num_topk], ncclInt32.
//                                        Top-k expert indices received from source ranks
//                                        (recv topk indices reuse the TOPK_IDX tag in outputs).
//                                    TOPK_WEIGHTS:      [OUT] 2D [num_recv_tokens x num_topk], ncclFloat32.
//                                        Top-k weights received from source ranks
//                                        (recv topk weights reuse the TOPK_WEIGHTS tag in outputs).
//   num_outputs   - [IN]     Number of output tensors (equal to num_inputs plus number of scaling tensors)
//   local_tensors - [IN,OUT] Array of pointers to preallocated tensors, with information that is local to the rank.
//                            LL mode: accepts 1 optional local tensor:
//                                    RECV_EXPERT_COUNTER_DEVICE: [OUT] 1D ncclInt32 [num_local_experts] —
//                                    per-expert recv tokens.
//                            HT mode: no local tensors (num_local_tensors must be 0).
//                            HT per-expert metadata comes from ncclEpUpdateHandle, not here.
//   num_local_tensors - [IN] Number of local tensors.
//   send_only     - [IN]     If true, the dispatch will only initiate data transfers and immediately
//                            release GPU resources (without waiting for the data to be received).
//                            When set, a blocking `ncclEpComplete()` must be used to complete the operation.
//                            Note:
//                              - Supported for LL mode only.
//                              - The output tensors must still be preallocated even when send_only is set.
//   config        - [IN]     Dispatch configuration.
//   stream        - [IN]     CUDA stream. If `ncclEpDispatch()` is called on a different stream than the stream used in
//                            `ncclEpCreateHandle()`,
//                            it is the responsibility of the user to synchronize between streams to ensure correctness.
//
//
// Returns:
//   ncclResult_t error code

ncclResult_t ncclEpDispatch(
    ncclEpHandle_t handle,
    const ncclNDTensor_t* inputs,
    unsigned int num_inputs,
    const ncclNDTensor_t* outputs,
    unsigned int num_outputs,
    const ncclNDTensor_t* local_tensors,
    unsigned int num_local_tensors,
    unsigned int send_only,
    const ncclEpDispatchConfig_t* config,
    cudaStream_t stream
);

// Opaque config struct (reserved, must be NULL)
typedef struct ncclEpCombineConfig ncclEpCombineConfig_t;

// Perform EP combine
//   * Gathers outputs from experts and returns them to their source in original token order.
//   * This call is collective and must be invoked by all ranks in the group.
//   * All tensors are tagged using tags with `NCCL_EP_TENSOR_TAG` prefix
//     to indicate the types of tensor (i.e., tokens, topK indices, weights, etc.)
//   This call is collective and must be invoked by all ranks in the group.
//
// Arguments:
//   handle           - [IN,OUT] EP handle that was used for `ncclEpDispatch()` operation
//   inputs           - [IN]     Array of pointers to input tensors, each containing expert outputs.
//                               For HT (NCCL_EP_LAYOUT_FLAT): inputs are [N(r) x data_size] (2D),
//                                       where N(r) is the flat received token count (same as dispatch output dim 0).
//                               For LL: inputs are [local_experts x num_recv_tokens x data_size] (3D, expert-major),
//                                       where num_recv_tokens = num_ranks * max_send_tokens_per_rank.
//                               For LL rank-major: inputs are [num_recv_tokens x data_size] (2D),
//                                       pre-reduced across local experts by the caller before this call.
//   num_inputs       - [IN]     Number of input tensors
//   outputs          - [IN,OUT] Array of pointers to preallocated output nd tensors, same number & order as inputs.
//                               All must be 2D [num_tokens x data_size]; tokens and metadata are restored to original order.
//   num_outputs      - [IN]     Number of output tensors (must equal num_inputs)
//   local_tensors    - [IN,OUT] Array of pointers to preallocated tensors, with information that is local to the rank.
//                               LL mode: accepts 1 optional local tensor:
//                                       TOP_K_WEIGHTS - IN [num_tokens x top_k] - top-k weights for each token.
//                               Expert-major: applied as per-expert weights on the combine receive side.
//                               Rank-major: not used; weight reduction and application are the caller's responsibility.
//   num_local_tensors - [IN]    Number of local tensors.
//   send_only        - [IN]     If true, the combine will only initiate data transfers and immediately
//                               release GPU resources (without waiting for the data to be received).
//                               When set, a blocking `ncclEpComplete()` must be used to complete the operation.
//                               Note:
//                                 - Supported for LL mode only.
//                                 - The output tensors must still be preallocated even when send_only is set.
//   config           - [IN]     Reserved for future options (must be NULL).
//   stream           - [IN]     CUDA stream. If `ncclEpCombine()` is called on a different stream than the stream
//                               used in `ncclEpCreateHandle()`, it is the responsibility of the user to synchronize
//                               between streams to ensure correctness.
//
// Returns:
//   ncclResult_t error code

ncclResult_t ncclEpCombine(
    ncclEpHandle_t handle,
    const ncclNDTensor_t* inputs,
    unsigned int num_inputs,
    const ncclNDTensor_t* outputs,
    unsigned int num_outputs,
    const ncclNDTensor_t* local_tensors,
    unsigned int num_local_tensors,
    unsigned int send_only,
    const ncclEpCombineConfig_t* config,
    cudaStream_t stream
);

// Opaque config struct (reserved, must be NULL)
typedef struct ncclEpCompleteConfig ncclEpCompleteConfig_t;

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
    const unsigned int** sizes,
    unsigned int* ndim
);


#ifdef __cplusplus
}
#endif
