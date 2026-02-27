/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 * See LICENSE.txt for more license information.
 */

#pragma once

#include <cuda.h>
#include <nccl.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    NCCL_EP_TENSOR_FLAG_NONE = 0,
} ncclEpTensorFlags_t; // Reserved for future use

// Tensor tags required to identify the type of tensors in `ncclEpDispatch` and `ncclEpCombine`
typedef enum {
    NCCL_EP_TENSOR_TAG_NONE = 0,
    // Tensor containing tokens
    NCCL_EP_TENSOR_TAG_TOKENS = 1,
    // Tensor containing top-k expert indices
    NCCL_EP_TENSOR_TAG_TOPK_IDX = 2,
    // Tensor containing top-k weights
    NCCL_EP_TENSOR_TAG_TOPK_WEIGHTS = 3,
    // Tensor containing scales
    NCCL_EP_TENSOR_TAG_SCALES = 4,
    // Tensor containing tokens received per expert (device memory)
    NCCL_EP_TENSOR_TAG_RECV_EXPERT_COUNTER_DEVICE = 5,
    // Tensor containing tokens received per expert (pinned host memory)
    NCCL_EP_TENSOR_TAG_RECV_EXPERT_COUNTER_HOST = 6,
    // Tensor containing per-expert token counts
    NCCL_EP_TENSOR_TAG_TOKENS_PER_EXPERTS = 7,
} ncclEpTensorTag_t;


// N-dimensional tensor used to describe various user inputs
// (i.e., tokens, top-k indices, weights, scales, etc.)
typedef struct {
    unsigned int version;             // Structure version (set to 1.0.0)
    unsigned int ndim;                // Number of dimensions
    unsigned int* sizes;              // Dimension sizes [ndim]
    unsigned int* strides;            // Strides in elements [ndim]
    ncclDataType_t datatype;          // Element data type
    void* data;                       // Pointer to tensor data
    unsigned int tag;                 // Tensor identification tag
    ncclEpTensorFlags_t flags;       // Tensor flags (set to 0)
} ncclNDTensor_t;

// Communication algorithm (mode)
typedef enum {
    // Low-Latency (LL) mode
    NCCL_EP_ALGO_LOW_LATENCY = 0,
    // High-Throughput (HT) mode
    NCCL_EP_ALGO_HIGH_THROUGHPUT = 1
} ncclEpAlgorithm_t;

// Auto configuration constant for dynamic/automatic sizing
#define NCCL_EP_AUTO 0

// EP group configuration structure
typedef struct {
    unsigned int version;                // Structure version (set to 1.0.0)
    ncclEpAlgorithm_t algorithm;         // low_latency or high_throughput
    unsigned int num_experts;            // Number of experts (required)
    // Maximum number of tokens any single rank will dispatch. Must be the same across all ranks.
    // Each rank should be prepared to receive up to max_tokens_per_rank * num_ranks tokens.
    // REQUIRED for both LL and HT modes (must be > 0).
    // In a future release, NCCL_EP_AUTO will be supported for HT mode,
    // in which case the received token count will be determined by ncclEpCreateHandle.
    unsigned int max_tokens_per_rank;
    unsigned int token_size_bytes;       // Token size for buffer allocation (independent of datatype)
    unsigned long int rdma_buffer_size;  // RDMA buffer size in bytes (NCCL_EP_AUTO for auto, defaults to a sufficiently large buffer for any algorithm)
    unsigned int num_qp_per_rank;        // Number of QPs per rank (NCCL_EP_AUTO for auto)
    // Number of channels per rank (NCCL_EP_AUTO for auto).
    // In high throughput collectives, each channel occupies 2 SMs
    unsigned int num_channels;
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
//   stream     - [IN]  CUDA stream
//   alloc_fn   - [IN]  Optional custom allocator function (NULL for default cudaMalloc)
//   free_fn    - [IN]  Optional custom free function (NULL for default cudaFree)
//
// Returns:
//   ncclResult_t error code

ncclResult_t ncclEpCreateGroup(
    ncclEpGroup_t* ep_group,
    ncclComm_t comm,
    const ncclEpGroupConfig_t* config,
    cudaStream_t stream,
    ncclEpAllocFn_t alloc_fn = nullptr,
    ncclEpFreeFn_t free_fn = nullptr
);

// Destroy an EP group and release associated resources.
//
// Arguments:
//   ep_group     - [IN]  EP group to destroy
//   stream       - [IN]  CUDA stream on which the group is being destroyed
//
// Returns:
//   ncclResult_t error code

ncclResult_t ncclEpGroupDestroy(
    ncclEpGroup_t ep_group,
    cudaStream_t stream
);


// Create a tensor with the given dimensions and data type using the EP group's allocator.
//   The implementation guarantees that the tensor is contiguous in memory (including accordingly
//   setting the strides to 1 for all dimensions).
//
// Arguments:
//   ep_group     - [IN]  EP group to create the tensor for
//   tensor       - [OUT] Pointer to newly created tensor
//   ndim         - [IN]  Number of dimensions
//   datatype     - [IN]  Data type
//   tag          - [IN]  Tensor identification tag
//   size0..size4 - [IN]  Dimension sizes
//
// Returns: ncclResult_t error code

ncclResult_t ncclEpTensorCreate(
    ncclEpGroup_t ep_group,
    ncclNDTensor_t* tensor,
    unsigned int ndim,
    ncclDataType_t datatype,
    ncclEpTensorTag_t tag,
    unsigned int size0,
    unsigned int size1 = 1,
    unsigned int size2 = 1,
    unsigned int size3 = 1,
    unsigned int size4 = 1
);


// Destroy a tensor and free its memory using the group's allocator.
//
// Arguments:
//   ep_group     - [IN]  EP group to destroy the tensor for
//   tensor       - [IN] Pointer to the tensor to destroy
//
// Returns: ncclResult_t error code

ncclResult_t ncclEpTensorDestroy(
    ncclEpGroup_t ep_group,
    ncclNDTensor_t* tensor
);

// Opaque type forward declaration
typedef struct ncclEpHandle* ncclEpHandle_t;
typedef struct ncclEpHandleConfig* ncclEpHandleConfig_t;  // Reserved for future use

// Create and initialize an EP handle.
//   * Performs dispatch setup and (in HT mode only) metadata exchange.
//   * This call is collective and must be invoked by all ranks in the group.
//
// Arguments:
//   handle              - [OUT] Pointer to newly created and initialized EP handle
//   ep_group            - [IN]  A valid EP group
//   topk_idx            - [IN]  Tensor holding top-K expert indices (routing information)
//   local_tensors       - [IN/OUT, optional] Array of pointers to local tensors.
//                         HT: accepts optional RECV_EXPERT_COUNTER tensor (1D, ncclInt32, size=num_local_experts)
//                         with tag NCCL_EP_TENSOR_TAG_RECV_EXPERT_COUNTER_HOST (pinned+mapped) or _DEVICE.
//                         Required when max_tokens_per_rank is NCCL_EP_AUTO.
//                         LL mode: does not accept local tensors (num_local_tensors must be 0).
//   num_local_tensors   - [IN]  Number of local tensors.
//   config              - [IN]  Reserved for future options (should be set to NULL)
//   stream              - [IN]  CUDA stream
//   use_fp8             - [IN]  Enable FP8 for dispatch (default: false)
//
// Notes:
//   - If max_tokens_per_rank in ncclEpGroupConfig_t was set to NCCL_EP_AUTO,
//     this call may block as the host allocates memory for the actual number
//     of received tokens.
//   - The config argument is reserved; must be set to NULL for now.
//
// Returns: ncclResult_t error code

ncclResult_t ncclEpCreateHandle(
    ncclEpHandle_t* handle,
    ncclEpGroup_t ep_group,
    const ncclNDTensor_t* topk_idx,
    ncclNDTensor_t* const* local_tensors,
    unsigned int num_local_tensors,
    const ncclEpHandleConfig_t* config,  // Reserved, should be set to NULL
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
//                            For HT: the dimensions of output tensors are [num_recv_tokens x data_size] (2D),
//                                    where num_recv_tokens = num_ranks * max_tokens_per_rank.
//                            For LL: the dimensions of output tensors are
//                                    [local_experts x num_recv_tokens x data_size] (3D, expert-major).
//                                    The dimensions of the scaling factors tensor are:
//                                    [local_experts x num_recv_tokens x (hidden / 128)] (3D, ncclFloat32)
//                                    where num_recv_tokens = num_ranks * max_tokens_per_rank.
//   num_outputs   - [IN]     Number of output tensors (equal to num_inputs plus number of scaling tensors)
//   local_tensors - [IN,OUT] Array of pointers to preallocated tensors, with information that is local to the rank.
//                            LL mode: accepts 1 optional local tensor:
//                                    NUM_TOKENS_PER_EXPERTS: [OUT] a 1D tensor of unsigned int [num_experts]
//                                    that contains the number of tokens received by each expert on this rank.
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
    const ncclNDTensor_t* const* inputs,
    unsigned int num_inputs,
    ncclNDTensor_t* const* outputs,
    unsigned int num_outputs,
    ncclNDTensor_t* const* local_tensors,
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
//                               For HT: inputs are [num_recv_tokens x data_size] (2D),
//                                       where num_recv_tokens = num_ranks * max_tokens_per_rank.
//                               For LL: inputs are [local_experts x num_recv_tokens x data_size] (3D, expert-major),
//                                       where num_recv_tokens = num_ranks * max_tokens_per_rank.
//   num_inputs       - [IN]     Number of input tensors
//   outputs          - [IN,OUT] Array of pointers to preallocated output nd tensors, same number & order as inputs.
//                               All must be 2D [num_tokens x data_size]; tokens and metadata are restored to original order.
//   num_outputs      - [IN]     Number of output tensors (must equal num_inputs)
//   local_tensors    - [IN,OUT] Array of pointers to preallocated tensors, with information that is local to the rank.
//                               LL mode: accepts 1 optional local tensor:
//                                       TOP_K_WEIGHTS - IN [num_tokens x top_k] - top-k weights for each token.
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
    const ncclNDTensor_t* const* inputs,
    unsigned int num_inputs,
    ncclNDTensor_t* const* outputs,
    unsigned int num_outputs,
    ncclNDTensor_t* const* local_tensors,
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

// Query the number of received tokens that will be received after a call to EP dispatch (HT mode only).
//
// Arguments:
//   handle           - [IN]   A valid EP handle.
//   num_recv_tokens  - [OUT]  Pointer to int, will be set to the actual number of tokens expected to be received on this rank
//
// Notes:
//   - This API is only supported in HIGH_THROUGHPUT (HT) mode.
//
// Returns:
//   ncclResult_t error code (e.g., ncclInvalidArgument if called in LL mode)

ncclResult_t ncclEpHandleGetNumRecvTokens(
    ncclEpHandle_t handle,
    unsigned int* num_recv_tokens
);


#ifdef __cplusplus
}
#endif
