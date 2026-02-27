# NCCL EP (Expert Parallelism) API

NCCL EP is a high-performance NCCL API extension for efficient Mixture-of-Experts (MoE) communication.
It provides optimized dispatch and combine primitives for Expert Parallelism (EP) across distributed GPU systems
implemented on top of NCCL Device API: Load-Store Accessible (LSA) and GPU-Initiated Networking (GIN) operations.


# Table of Contents

- [Overview](#overview)
- [Usage](#usage)
- [Core Concepts](#core-concepts)
- [API Reference](#api-reference)
- [Execution Modes](#execution-modes)
- [Usage Examples](#usage-examples)

# Overview

The NCCL EP API extends NCCL with native support for efficient MoE communication patterns. It provides optimized implementations for token "dispatch" and expert output "combine" operations, which are an important components of modern large language models employing sparse MoE architectures.

NCCL EP brings the performance benefits of modern device-initiated MoE libraries into the NCCL ecosystem with a unified API. NCCL EP provides unified `ncclEpDispatch` and `ncclEpCombine` primitives that allow selecting the appropriate algorithm based on workload characteristics.

Currently, two distinct communication algorithms tailored for different workload characteristics are supported:

* **Low-Latency (LL)** - optimized for small batch sizes and latency-sensitive workloads (i.e., LLM inference). To minimize latency, it uses direct point-to-point all-to-all communication with experts.

* **High-Throughput (HT)** - optimized for training and inference prefilling with large batch sizes. HT mode implements hierarchical communication patterns, relying on NVLink for intra-node aggregation, and on RDMA for inter-node communication. The implementation leverages Hopper architecture features, including warp-specialized pipelines, and TMA (Tensor Memory Accelerator) operations.

NCCL EP relies on NCCL Device API, using GIN `put`/`signal` operations for RDMA and LSA load/store operations for NVLink communication, eliminating CPU involvement in the critical path while inheriting NCCL topology detection and plugin architecture.

## Key Features

- **Staged Execution** (LL mode only): Enable computation-communication overlap through a `send-only` flag.
- **Automatic Tuning**: Let the API auto-tune buffer sizes, queue pairs, and channels.
- **Type Conversion**: Automatic scaling support for different input and output data types (for a restricted set of combinations).

## Quick Start

### C API

```c
// Group management
ncclEpCreateGroup(&ep_group, comm, &config, stream, alloc_fn, free_fn);
ncclEpGroupDestroy(ep_group, stream);

// Handle management
ncclEpCreateHandle(&handle, ep_group, &topk_idx, local_tensors, num_local, config, stream);
ncclEpHandleDestroy(handle);

// Communication operations
ncclEpDispatch(handle, inputs, num_in, outputs, num_out, local, num_local, send_only, config, stream);
ncclEpCombine(handle, inputs, num_in, outputs, num_out, local, num_local, send_only, config, stream);
ncclEpComplete(handle, config, stream);  // LL mode only
```

### Python API

Install Python bindings

```bash
$ pip install -e contrib/nccl_ep/python
```

Import and use NCCL EP in a python application
```python
from nccl_ep import NCCLLibrary, NCCL_EP_ALGO_LOW_LATENCY

nccl_lib = NCCLLibrary()
# Use nccl_lib.ncclEpDispatch, ncclEpCombine, etc.
```

### Reference performance numbers

For microbenchmarking, NCCL EP provides the performance evaluation tool [`ep_bench`](ep_bench.cu).

Below, the reference performance numbers collected
on NVIDIA H100 platform for Low-Latency mode
using **BF16 dispatch and combine** (same data type) are presented.
Note, the data was obtained for NCCL v2.29u1 release.

#### Test Configuration
- Hidden: 7168
- Top-k: 8
- Experts: 256
- Tokens: 128 per rank
- 8 GPUs per node
- Up to 8 nodes

#### Performance

| Number Of GPUs | Node Count | Dispatch BW (GB/s) | Combine BW (GB/s)  |
|:--------------:|:----------:|:------------------:|:------------------:|
| 8              | 1          | 224.3              | 185.2              |
| 16             | 2          | 76.7               | 73.0               |
| 32             | 4          | 53.6               | 50.0               |
| 64             | 8          | 48.8               | 43.8               |

### Common scenarios

This section provides a high-level overview of the input, output, and local tensors expected by the API for common expected scenarios.

#### Used notation

**Dimensions:**
* B = batch size
* H = hidden dimension
* S = scales dimension
* L = number of local experts
* K = top K
* N(r) = number of tokens targeting rank r

**Tags:**
* TOKENS = NCCL_EP_TENSOR_TAG_TOKENS
* WEIGHT = NCCL_EP_TENSOR_TAG_TOPK_WEIGHTS
* INDEX = NCCL_EP_TENSOR_TAG_TOPK_IDX
* SCALES = NCCL_EP_TENSOR_TAG_SCALES
* CNTR_D = NCCL_EP_TENSOR_TAG_RECV_EXPERT_COUNTER_DEVICE
* CNTR_H = NCCL_EP_TENSOR_TAG_RECV_EXPERT_COUNTER_HOST


#### LL mode (same data type)

| Operation | Tensor Index | Input tag | Input dims  | Output tag | Output dims | Local tags | Local dims |
|:---------:|:------------:|:---------:|:-----------:|:----------:|:-----------:|:----------:|:----------:|
| Dispatch  | 0            | TOKEN     | [B x H]     | TOKENS     | [L x B x H] | CNTR_D     | [L]        |
| Combine   | 0            | TOKEN     | [L x B x H] | TOKENS     | [B x H]     |            |            |
|           | 1            | WEIGHTS   | [B x K]     |            |             |            |            |


#### HT mode (same data type)

**Handle creation**

| Operation | Local tags    | Local dims |
|-----------|:-------------:|:----------:|
| Create    | CNTR_D/CNTR_H | [L]        |


**Forward pass**

| Operation | Tensor Index | Input tag | Input dims  | Output tag | Output dims | Local tags | Local dims |
|:---------:|:------------:|:---------:|:-----------:|:----------:|:-----------:|:----------:|:----------:|
| Dispatch  | 0            | TOKEN     | [B x H]     | TOKENS     | [N(r) x H]  |            |            |
|           | 1            | WEIGHTS   | [B x K]     | WEIGHTS    | [N(r) x K]  |            |            |
|           | 2            | INDEX     | [B x K]     | INDEX      | [N(r) x K]  |            |            |
| Combine   | 0            | TOKEN     | [[N(r) x H] | TOKENS     | [B x H]     |            |            |
|           | 1            | WEIGHTS   | [N(r) x H]  |            |             |            |            |

**Backward pass**

| Operation | Tensor Index | Input tag | Input dims  | Output tag | Output dims | Local tags | Local dims |
|:---------:|:------------:|:---------:|:-----------:|:----------:|:-----------:|:----------:|:----------:|
| Dispatch  | 0            | TOKEN     | [B x H]     | TOKENS     | [N(r) x H]  |            |            |
|           | 1            | WEIGHTS   | [B x K]     | WEIGHTS    | [N(r) x K]  |            |            |
|           | 2            | INDEX     | [B x K]     | INDEX      | [N(r) x K]  |            |            |
| Combine   | 0            | TOKEN     | [[N(r) x H] | TOKENS     | [B x H]     |            |            |
|           | 1            | WEIGHTS   | [N(r) x H]  | **WEIGHTS**| [B x K]     |            |            |

#### LL mode (FP16 -> FP8 conversion - NOT SUPPORTED)

| Operation | Tensor Index | Input tag | Input dims  | Output tag | Output dims | Local tags | Local dims |
|:---------:|:------------:|:---------:|:-----------:|:----------:|:-----------:|:----------:|:----------:|
| Dispatch  | 0            | TOKEN     | [B x H]     | TOKENS     | [L x B x H] | CNTR_D     | [L]        |
|           | 1            |           |             | **SCALES** | [L x B x S] |            |            |
| Combine   | 0            | TOKEN     | [L x B x H] | TOKENS     | [B x H]     |            |            |
|           | 1            | WEIGHTS   | [B x K]     |            |             |            |            |

#### HT mode (FP16 -> FP8 conversion - NOT SUPPORTED)

**Forward pass**

| Operation | Tensor Index | Input tag | Input dims  | Output tag | Output dims | Local tags | Local dims |
|:---------:|:------------:|:---------:|:-----------:|:----------:|:-----------:|:----------:|:----------:|
| Dispatch  | 0            | TOKEN     | [B x H]     | TOKENS     | [N(r) x H]  |            |            |
|           | 1            | WEIGHTS   | [B x K]     | WEIGHTS    | [N(r) x K]  |            |            |
|           | 2            | INDEX     | [B x K]     | INDEX      | [N(r) x K]  |            |            |
|           | 3            | **SCALES**| [B x S]     | **SCALES** | [N(r) x S]  |            |            |
| Combine   | 0            | TOKEN     | [[N(r) x H] | TOKENS     | [B x H]     |            |            |
|           | 1            | WEIGHTS   | [N(r) x H]  |            |             |            |            |

# Usage

## Prerequisites

### Dependencies

| Component | Version | Notes |
|-----------|---------|-------|
| CUDA | 13+ | Required |
| NCCL | 2.29+ | With Device API and GIN support |
| MPI | Any (OpenMPI, MPICH, etc.) | Required for multi-process launch |
| GPU | Hopper (H100) or Blackwell | Tested configurations |

### Discover compute capabilities

Use `nvidia-smi` command to detect the compute capabilities of your NVIDIA GPU.

For example, on Hopper system with `compute_cap` of `90`, the output looks like below:
```bash
$ nvidia-smi --query-gpu=compute_cap --format=csv
compute_cap
9.0
...
```

### Set the environment

The following paths are required for building and running the

```
export COMPUTE_CAP=<discovered compute_cap>
export CUDA_HOME=/path/to/cuda
export MPI_HOME=/path/to/openmpi
export NCCL_HOME=/path/to/nccl/build # The desired NCCL build
export LD_LIBRARY_PATH="${CUDA_HOME}/lib:${CUDA_HOME}/lib64:${CUDA_HOME}/extras/CUPTI/lib64:${NCCL_HOME}/lib:$LD_LIBRARY_PATH"
export PATH="${CUDA_HOME}/bin:${NCCL_HOME}/bin:${MPI_HOME}/bin:$PATH"
```

## Building

### Step 1: Build NCCL with Device API Support

```bash
cd /path/to/nccl-source
make -j src.build BUILDDIR=${NCCL_HOME}
```

This creates the NCCL build artifacts in `BUILDDIR` (`./build` by default):
- `${NCCL_HOME}/lib/libnccl.so` - NCCL library with EP support
- `${NCCL_HOME}/include/` - Header files

### Step 2: Build NCCL EP Library and Test


```bash
make -C contrib/nccl_ep MPI=1 BUILDDIR=${NCCL_HOME} \
       NVCC_GENCODE="-gencode=arch=compute_${COMPUTE_CAP},code=sm_${COMPUTE_CAP}"
```

Once `make` command is successfuly completed, the following files will be created:
- `${NCCL_HOME}/lib/libnccl_ep.a` - Static library
- `${NCCL_HOME}/lib/libnccl_ep.so` - Shared library (for Python bindings)
- `${NCCL_HOME}/include/nccl_ep.h` - C API header
- `${NCCL_HOME}/test/nccl_ep/ep_test` - Test application for both Low-Latency and High-Throughput modes
- `${NCCL_HOME}/test/nccl_ep/ep_bench` - Benchmark application for both Low-Latency and High-Throughput modes

## Running

### Environment Setup

Make sure to set the generic environment according to the
[Set the environment](#set-the-environment) section.

```bash
# NCCL GIN configuration recommended for multi-node RDMA:
export NCCL_GIN_TYPE=3  # GDAKI - GPU Direct Async Kernel-Initiated
```

For debugging, the following variables can be set

```bash
export NCCL_DEBUG=INFO        # Enable NCCL debug output
export NCCL_DEBUG_SUBSYS=ALL  # All subsystems
```

### `ep_test` application

The `ep_test` application (`ep_test.cu`) is a comprehensive working example that demonstrates both Low-Latency and High-Throughput modes.
It can be used as a reference implementation when integrating NCCL EP into an application.

#### Command-Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `-a <ll\|ht>` | Algorithm mode: `ll` (Low-Latency) or `ht` (High-Throughput) | `ll` |
| `-t <num>` | Number of tokens | 50 |
| `-d <num>` | Hidden dimension size | 7168 |
| `-m` | Disable max_tokens_per_rank (HT mode only) | disabled |
| `-s <mode>` | Send-only mode: `none`, `dispatch`, `combine`, `both` | `none` |
| `-c` | Enable cached mode (HT only) | disabled |
| `-r` | Enable random mode (random topk_idx) | disabled |

#### Single-node execution

```bash
# Low-Latency mode (default)
mpirun -np 8 ./build/test/nccl_ep/ep_test -a ll -t 128 -d 7168

# High-Throughput mode
mpirun -np 8 ./build/test/nccl_ep/ep_test -a ht -t 4096 -d 7168

```

#### Multi-Node Execution (MPI)

```bash
# 2 nodes, 8 GPUs per node
mpirun -np 16 \
  --map-by ppr:8:node \
  -x NCCL_GIN_TYPE=3 \
  -x LD_LIBRARY_PATH \
  ./build/test/nccl_ep/ep_test -a ll -t 128 -d 7168
```

# Core Concepts

## Key Data Structures

### `ncclNDTensor_t` - Multi-dimensional Tensor Descriptor

Encapsulates tensor metadata and data layout information:

```c
typedef struct {
    unsigned int version;           // Structure version (set to 1)
    unsigned int ndim;              // Number of dimensions
    unsigned int* sizes;            // Dimension sizes [ndim]
    unsigned int* strides;          // Strides in elements [ndim]
    ncclDataType_t datatype;        // Element data type
    void* data;                     // Pointer to tensor data
    unsigned int tag;               // Tensor identification tag
    ncclEpTensorFlags_t flags;     // Tensor flags (set to 0)
} ncclNDTensor_t;
```

### `ncclEpGroup_t` - EP Group Configuration

Created from an NCCL communicator, manages the distributed EP configuration across all ranks in the group:

```c
typedef struct {
    unsigned int version;           // Structure version (set to 1)
    ncclEpAlgorithm_t algorithm;   // HT or LL mode
    unsigned int num_experts;       // Total experts across all ranks
    unsigned int max_tokens_per_rank;  // Max tokens per rank
    unsigned int token_size_bytes;  // Maximum token size
    unsigned int rdma_buffer_size;  // RDMA buffer size (0=auto)
    unsigned int num_qp_per_rank;   // Queue pairs per rank (0=auto)
    unsigned int num_channels;      // Channels per rank (0=auto)
} ncclEpGroupConfig_t;
```

### `ncclEpHandle_t` - Operation Handle

Maintains state for a sequence of related MoE operations, i.e. dispatch and combine pairs for forward and (optionally) backward passes. The handle encapsulates routing metadata and communication buffers.

## Algorithm-related configurations

**High Throughput (HT)**:
- Output tokens must have 2D format: `[num_recv_tokens x hidden]` where `num_recv_tokens = num_ranks * max_tokens_per_rank`
- Supports dynamic `max_tokens_per_rank` (set to `NCCL_EP_AUTO`)

**Low Latency (LL)**:
- Output tokens must have 3D format: `[num_experts x max_tokens x hidden]`
- Expert-major data layout for efficient expert processing
- Supports `send_only` parameter to enable computation/communication overlapping
- Does not support dynamic `max_tokens_per_rank` detection

## Custom Allocators

The API supports custom memory allocators for custom buffer management. This enables integration with memory pools, custom allocation strategies, or framework-specific allocators.

### Function Signatures

```c
typedef cudaError_t (*ncclEpAllocFn_t)(void** ptr, size_t size);
typedef cudaError_t (*ncclEpFreeFn_t)(void* ptr);
```

Allocators must match the `cudaMalloc`/`cudaFree` signatures and return `cudaSuccess` on success.

### Example

```c
// Custom allocator using a memory pool
cudaError_t my_alloc(void** ptr, size_t size) {
    *ptr = myMemoryPool.allocate(size);
    return (*ptr != nullptr) ? cudaSuccess : cudaErrorMemoryAllocation;
}

cudaError_t my_free(void* ptr) {
    myMemoryPool.deallocate(ptr);
    return cudaSuccess;
}

// Pass to ncclEpCreateGroup
ncclEpCreateGroup(&ep_group, comm, &config, stream, my_alloc, my_free);
```

If `NULL` is passed for both allocator functions, the default `cudaMalloc`/`cudaFree` are used.


## API Reference

### Group Management

#### `ncclEpCreateGroup()`

```c
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
    ncclEpAllocFn_t alloc_fn,
    ncclEpFreeFn_t free_fn
);
```

#### `ncclEpGroupDestroy()`

```c
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
```

### Tensor Management

#### `ncclEpTensorCreate()`

```c
// Create a tensor with the given dimensions and data type using the EP group's allocator.
// The implementation guarantees that the tensor is contiguous in memory (including accordingly
// setting the strides to 1 for all dimensions).
//
// Arguments:
//   ep_group     - [IN]  EP group to create the tensor for
//   tensor       - [OUT] Pointer to the newly created tensor
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
```

#### `ncclEpTensorDestroy()`

```c
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
```

### Handle Management

#### `ncclEpCreateHandle()`

```c
// Create and initialize an EP handle.
//   Performs dispatch setup and (in HT mode only) metadata exchange.
//   This call is collective and must be invoked by all ranks in the group.
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
```

#### `ncclEpHandleDestroy()`

```c
// Destroy an EP handle and release all associated resources.
//
// Arguments:
//   handle         - [IN]  EP handle to destroy
//
// Returns: ncclResult_t error code

ncclResult_t ncclEpHandleDestroy(
    ncclEpHandle_t handle
);
```

#### `ncclEpHandleGetNumRecvTokens()` (HT mode only)

```c
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
```

### Communication Operations

#### `ncclEpDispatch()`

Perform EP dispatch: send tokens to experts according to routing decisions.

```c
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
//                            then the additional output tensor for scaling factors must be supplied (NCCL_EP_TENSOR_TAG_SCALES).
//
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
//   send_only     - [IN]     If true, the dispatch kernel will only initiate data transfers and
//                            release GPU resources before the data is received.
//                            When set, a blocking `ncclEpComplete` call must be used to complete the operation.
//                            Not supported in HT mode.
//                            Note that the output tensors must be preallocated even when send_only is set.
//   config        - [IN]     Dispatch configuration.
//   stream        - [IN]     CUDA stream. If ncclEpDispatch is called on a different stream than the stream used in
//                            `ncclEpCreateHandle`,
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
```

#### `ncclEpCombine()`

Perform EP combine: gather expert outputs and return in original token order.

```c
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
```

#### `ncclEpComplete()` (LL mode only)

```c
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
```

# Execution Modes

Both `ncclEpDispatch()` and `ncclEpCombine()` operations support synchronous and staged semantics.

## Synchronous Mode (Default)

Corresponds to a synchronous version of an operation where GPU resources are allocated during the whole operation execution
including the time to wait for the data to be received. This mode doesn't allow for
computation/communication overlap.

```c
int send_only = false;
ncclEpDispatch(handle, inputs, num_inputs, outputs, num_outputs,
                NULL, 0, send_only, NULL, stream);
// Dispatch is complete when this returns
```

## Staged Mode (Low Latency Only)

In this mode, the operation is split into the send and receive phases enabling computation computation/communication overlap:
  * The operation is invoked with `send_only = true` flag
  * In this case, after all data transfers are initiated, the corresponding GPU resources are released
  * This enables an application to utilize all avaialble GPU resources for computation while offloaded data transfers are performed.
  * To complete the operation, the `ncclEpComplete()` primitive is used.

This mode is particularly beneficial for inference with multiple micro-batches.

```c
// Stage 1: Post send requests without waiting for completion
int send_only = true;
ncclEpDispatch(handle, inputs, num_inputs, outputs, num_outputs,
                NULL, 0, send_only, NULL, stream);
// Returns after initiating the operations

// Stage 2: Continue other computations...

// Stage 3: Wait for actual completion
ncclEpCompleteConfig_t continue_config;
ncclEpComplete(handle, &continue_config, stream);
// Now all data is actually sent/received
```

# Usage Examples

> **Note:** For a complete working example, see `ep_test.cu` which demonstrates both LL and HT modes with all API calls.

## Example 1: High Throughput Mode - Forward and Backward Pass

```c
#include "nccl.h"
#include "nccl_ep.h"
#include "cuda_runtime.h"

// Initialize NCCL communicator
ncclComm_t comm;
ncclCommInitRank(&comm, nRanks, id, myRank);

cudaStream_t stream;
cudaStreamCreate(&stream);

unsigned int top_k = 8;
unsigned int hidden = 4096;

// Configure for High Throughput mode
ncclEpGroupConfig_t config;
config.version = 1;
config.algorithm = NCCL_EP_ALGO_HIGH_THROUGHPUT;
config.num_experts = 256;
config.max_tokens_per_rank = 4;  // Or NCCL_EP_AUTO for dynamic sizing
config.token_size_bytes = hidden * 2;  // bfloat16
config.rdma_buffer_size = NCCL_EP_AUTO;     // Auto-size
config.num_qp_per_rank = NCCL_EP_AUTO;      // Auto-size
config.num_channels = NCCL_EP_AUTO;         // Auto-size

ncclEpGroup_t ep_group;
ncclEpCreateGroup(&ep_group, comm, &config, stream, my_alloc, my_free);

ncclNDTensor_t topk_idx;
ncclEpTensorCreate(ep_group, &topk_idx, 2, ncclInt64,
                    NCCL_EP_TENSOR_TAG_TOPK_IDX,
                    num_tokens, top_k);

// Create recv_expert_counter local tensor for ncclEpCreateHandle (optional, for HT mode)
// This tensor will receive the number of tokens per expert after metadata exchange
ncclNDTensor_t recv_expert_counter;
ncclNDTensor_t* local_tensors[1] = {nullptr};
unsigned int num_local_tensors = 0;
if (config.max_tokens_per_rank == NCCL_EP_AUTO) {
    recv_expert_counter.ndim = 1;
    recv_expert_counter.datatype = ncclInt32;
    recv_expert_counter.strides = new unsigned int[1];
    recv_expert_counter.strides[0] = 1;
    recv_expert_counter.tag = NCCL_EP_TENSOR_TAG_RECV_EXPERT_COUNTER_HOST;
    recv_expert_counter.flags = NCCL_EP_TENSOR_FLAG_NONE;
    recv_expert_counter.sizes = new unsigned int[1];
    recv_expert_counter.sizes[0] = num_local_experts;
    cudaHostAlloc(&recv_expert_counter.data, num_local_experts * sizeof(int), cudaHostAllocMapped);
    local_tensors[0] = &recv_expert_counter;
    num_local_tensors = 1;
}

// Create EP handle (can be reused for forward and backward)
ncclEpHandle_t handle;
ncclEpCreateHandle(&handle, ep_group, &topk_idx, local_tensors, num_local_tensors, NULL, stream);

// max_tokens_per_rank is the per-rank dispatch count.
// num_recv_tokens is the max tokens this rank can receive (nRanks * max_tokens_per_rank).
unsigned int num_recv_tokens;
if (config.max_tokens_per_rank == NCCL_EP_AUTO) {
    ncclEpHandleGetNumRecvTokens(handle, &num_recv_tokens);
} else {
    num_recv_tokens = config.max_tokens_per_rank * nRanks;
}

// === FORWARD PASS ===

// Create input tensors (HT mode uses 3 inputs)
ncclNDTensor_t input_tokens;
ncclEpTensorCreate(ep_group, &input_tokens, 2, ncclBfloat16,
                    NCCL_EP_TENSOR_TAG_TOKENS,
                    num_tokens, hidden);

ncclNDTensor_t topk_weights;
ncclEpTensorCreate(ep_group, &topk_weights, 2, ncclFloat32,
                    NCCL_EP_TENSOR_TAG_TOPK_WEIGHTS,
                    num_tokens, top_k);

topk_idx.tag = NCCL_EP_TENSOR_TAG_TOPK_IDX;

ncclNDTensor_t* forward_inputs[3] = {&input_tokens, &topk_weights, &topk_idx};

// Create output tensors (HT mode: 3 outputs, all 2D)
ncclNDTensor_t output_tokens;
ncclEpTensorCreate(ep_group, &output_tokens, 2, ncclBfloat16,
                    NCCL_EP_TENSOR_TAG_TOKENS,
                    num_recv_tokens, hidden);

ncclNDTensor_t recv_topk_weights;
ncclEpTensorCreate(ep_group, &recv_topk_weights, 2, ncclFloat32,
                    NCCL_EP_TENSOR_TAG_TOPK_WEIGHTS,
                    num_recv_tokens, top_k);

ncclNDTensor_t recv_topk_idx;
ncclEpTensorCreate(ep_group, &recv_topk_idx, 2, ncclInt64,
                    NCCL_EP_TENSOR_TAG_TOPK_IDX,
                    num_recv_tokens, top_k);

ncclNDTensor_t* forward_outputs[3] = {&output_tokens, &recv_topk_weights, &recv_topk_idx};

// Local tensors for dispatch
unsigned int num_local_experts = config.num_experts / nRanks;
ncclNDTensor_t tokens_per_expert;
ncclEpTensorCreate(ep_group, &tokens_per_expert, 1, ncclInt32,
                    NCCL_EP_TENSOR_TAG_RECV_EXPERT_COUNTER_DEVICE,
                    num_local_experts);

ncclNDTensor_t* dispatch_local_tensors[1] = {&tokens_per_expert};

// Dispatch tokens to experts
ncclEpDispatchConfig_t dispatch_config;
ncclEpDispatch(handle, forward_inputs, 3, forward_outputs, 3,
                dispatch_local_tensors, 1, 0, &dispatch_config, stream);

// Expert forward computation
// ... process output_tokens using tokens_per_expert counts ...

// Create expert output tensor
ncclNDTensor_t expert_outputs;
ncclEpTensorCreate(ep_group, &expert_outputs, 2, ncclBfloat16,
                    NCCL_EP_TENSOR_TAG_TOKENS,
                    num_recv_tokens, hidden);

ncclNDTensor_t combined_output;
ncclEpTensorCreate(ep_group, &combined_output, 2, ncclBfloat16,
                    NCCL_EP_TENSOR_TAG_TOKENS,
                    num_tokens, hidden);

ncclNDTensor_t* combine_inputs[1] = {&expert_outputs};
ncclNDTensor_t* combine_outputs[1] = {&combined_output};

ncclEpCombine(handle, combine_inputs, 1, combine_outputs, 1,
               nullptr, 0, 0, nullptr, stream);

// === BACKWARD PASS ===
// Use the same handle - routing information is reused

ncclNDTensor_t* backward_dispatch_inputs[1] = {&grad_combined};
ncclNDTensor_t* backward_dispatch_outputs[1] = {&grad_at_experts};

ncclEpDispatch(handle, backward_dispatch_inputs, 1, backward_dispatch_outputs, 1,
                nullptr, 0, 0, &dispatch_config, stream);

// Expert backward computation
// ... compute gradients for each expert ...

// Combine gradients
ncclNDTensor_t* backward_combine_inputs[2] = {&grad_expert_outputs, combine_topk_weights_input};
ncclNDTensor_t* backward_combine_outputs[2] = {&grad_tokens, combine_topk_weights_output};

ncclEpCombine(handle, backward_combine_inputs, 2, backward_combine_outputs, 2,
               nullptr, 0, 0, nullptr, stream);

// Cleanup
ncclEpHandleDestroy(handle);
ncclEpGroupDestroy(ep_group, stream);
ncclCommDestroy(comm);
cudaStreamDestroy(stream);
```

## Example 2: Low Latency Mode - Forward Pass

```c
#include "nccl.h"
#include "nccl_ep.h"
#include "cuda_runtime.h"

// Initialize NCCL communicator
ncclComm_t comm;
ncclCommInitRank(&comm, nRanks, id, myRank);

cudaStream_t stream;
cudaStreamCreate(&stream);

unsigned int top_k = 8;
unsigned int hidden = 4096;
unsigned int num_local_experts = config.num_experts / nRanks;

// Configure for Low Latency mode
ncclEpGroupConfig_t config;
config.version = 1;
config.algorithm = NCCL_EP_ALGO_LOW_LATENCY;
config.num_experts = 256;
config.max_tokens_per_rank = 128;  // Must be set for LL mode
config.token_size_bytes = hidden * 2;  // bfloat16
config.rdma_buffer_size = NCCL_EP_AUTO;     // Auto-size
config.num_qp_per_rank = NCCL_EP_AUTO;      // Auto-size (or specify for LL)
config.num_channels = NCCL_EP_AUTO;         // Auto-size

ncclEpGroup_t ep_group;
ncclEpCreateGroup(&ep_group, comm, &config, stream, my_alloc, my_free);

// Create routing tensor (topk_idx)
ncclNDTensor_t topk_idx;
ncclEpTensorCreate(ep_group, &topk_idx, 2, ncclInt64,
                    NCCL_EP_TENSOR_TAG_TOPK_IDX,
                    num_tokens, top_k);

// Create EP handle
ncclEpHandle_t handle;
ncclEpCreateHandle(&handle, ep_group, &topk_idx, NULL, 0, NULL, stream);

// === FORWARD PASS ===

// Create input tensor (LL mode uses 1 input)
ncclNDTensor_t input_tokens;
ncclEpTensorCreate(ep_group, &input_tokens, 2, ncclBfloat16,
                    NCCL_EP_TENSOR_TAG_TOKENS,
                    num_tokens, hidden);

ncclNDTensor_t* dispatch_inputs[1] = {&input_tokens};

// Create output tensor (LL mode: 3D format [num_local_experts, nRanks * max_tokens, hidden])
ncclNDTensor_t output_tokens;
ncclEpTensorCreate(ep_group, &output_tokens, 3, ncclBfloat16,
                    NCCL_EP_TENSOR_TAG_TOKENS,
                    num_local_experts, nRanks * config.max_tokens_per_rank, hidden);

ncclNDTensor_t* dispatch_outputs[1] = {&output_tokens};

// Create local tensors for LL mode
ncclNDTensor_t tokens_per_expert;
ncclEpTensorCreate(ep_group, &tokens_per_expert, 1, ncclInt32,
                    NCCL_EP_TENSOR_TAG_RECV_EXPERT_COUNTER_DEVICE,
                    num_local_experts);

ncclNDTensor_t* local_tensors[1] = {&tokens_per_expert};

// Dispatch tokens to experts (staged execution for overlap)
ncclEpDispatchConfig_t dispatch_config;
dispatch_config.round_scales = 0;

ncclEpDispatch(handle, dispatch_inputs, 1, dispatch_outputs, 1,
                local_tensors, 1, 1 /* send_only */, &dispatch_config, stream);

// Overlap with other computation...
// doOtherWork(stream);

// Wait for dispatch to complete
ncclEpComplete(handle, nullptr, stream);
cudaStreamSynchronize(stream);

// Expert forward computation
// Process output_tokens in 3D layout [experts x tokens x hidden]
// Use tokens_per_expert to know how many valid tokens per expert
// ... expertCompute(output_tokens, expert_outputs, tokens_per_expert, stream) ...

// Create expert output tensor (also 3D in LL mode)
ncclNDTensor_t expert_outputs;
ncclEpTensorCreate(ep_group, &expert_outputs, 3, ncclBfloat16,
                    NCCL_EP_TENSOR_TAG_TOKENS,
                    num_local_experts, nRanks * config.max_tokens_per_rank, hidden);

// Create topk_weights for combine
ncclNDTensor_t topk_weights;
ncclEpTensorCreate(ep_group, &topk_weights, 2, ncclFloat32,
                    NCCL_EP_TENSOR_TAG_TOPK_WEIGHTS,
                    num_tokens, top_k);

ncclNDTensor_t* combine_local_tensors[1] = {&topk_weights};

// Combine expert outputs back to original token order
ncclNDTensor_t combined_output;
ncclEpTensorCreate(ep_group, &combined_output, 2, ncclBfloat16,
                    NCCL_EP_TENSOR_TAG_TOKENS,
                    num_tokens, hidden);

ncclNDTensor_t* combine_inputs[1] = {&expert_outputs};
ncclNDTensor_t* combine_outputs[1] = {&combined_output};

ncclEpCombine(handle, combine_inputs, 1, combine_outputs, 1,
               combine_local_tensors, 1, 0 /* send_only */, nullptr, stream);

ncclEpComplete(handle, nullptr, stream);
cudaStreamSynchronize(stream);

// Cleanup
ncclEpHandleDestroy(handle);
ncclEpGroupDestroy(ep_group, stream);
ncclCommDestroy(comm);
cudaStreamDestroy(stream);
```
