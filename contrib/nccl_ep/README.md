# NCCL EP (Expert Parallelism) API

NCCL EP is a high-performance NCCL API extension for efficient Mixture-of-Experts (MoE) communication.
It provides optimized dispatch and combine primitives for Expert Parallelism (EP) across distributed GPU systems
implemented on top of NCCL Device API: Load-Store Accessible (LSA) and GPU-Initiated Networking (GIN) operations.

# Maintainers

| GitHub | Areas |
|--------|------|
| @artpol84 | APIs, new features, layouts |
| @kwen2501 | APIs, integration |
| @sb17v | Kernels, build systems |
| @nv-lschneider | Kernels, mnnvl |
| @kgioioso | GIN, NCCL |

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
ncclEpCreateGroup(&ep_group, comm, &config, alloc_fn, free_fn);
ncclEpGroupDestroy(ep_group);

// Handle management. `layout_info` is an optional ncclEpLayoutInfo_t* whose
// fields advertise device-side metadata tensors (expert_counters,
// src_rank_counters, expert_offsets, recv_total_counter).
ncclEpCreateHandle(&handle, ep_group, topk_idx, layout_info, handle_cfg, stream);
ncclEpHandleDestroy(handle);

// Communication operations. inputs / outputs are named-struct pointers
// (ncclEpDispatchInputs_t / ncclEpDispatchOutputs_t /
// ncclEpCombineInputs_t / ncclEpCombineOutputs_t); each cross-boundary
// tensor lives in a named field, no parallel arrays.
ncclEpDispatch(handle, topk_idx, &dispatch_in, &dispatch_out, layout_info, &dispatch_cfg, stream);
ncclEpCombine(handle, &combine_in, &combine_out, &combine_cfg, stream);
ncclEpComplete(handle, config, stream);  // LL mode only
```

### Python API

Install nccl4py, which includes the NCCL EP Python bindings as `nccl.ep`. Only CUDA 13 is supported as of now.

```bash
$ pip install nccl4py[cu13]
```

Import and use NCCL EP in a python application
```python
from nccl.ep import NCCLLibrary, NCCL_EP_ALGO_LOW_LATENCY

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

This section provides a high-level overview of the input, output, and layout
metadata tensors expected by the API for common scenarios. Each cross-boundary
tensor lives in a named field of one of the API's struct types
(`ncclEpDispatchInputs_t`, `ncclEpDispatchOutputs_t`, `ncclEpLayoutInfo_t`,
`ncclEpCombineInputs_t`, `ncclEpCombineOutputs_t`); the **Struct** column
names the struct and the **Field** column names the field within it.

#### Used notation

**Dimensions:**
* B = batch size
* H = hidden dimension
* S = scales dimension
* L = number of local experts
* K = top K
* R = number of ranks (nRanks)
* N(r) = number of tokens targeting rank r


#### LL mode (same data type)

| Operation | Struct             | Field             | Dims             |
|:---------:|:-------------------|:------------------|:----------------:|
| Dispatch  | dispatch_inputs    | tokens            | [B x H]          |
|           | dispatch_outputs   | tokens            | [L x R x B x H]  |
|           | layout_info        | expert_counters   | [L]              |
| Combine   | combine_inputs     | tokens            | [L x R x B x H]  |
|           | combine_outputs    | tokens            | [B x H]          |
|           | combine_outputs    | topk_weights      | [B x K]          |


#### HT mode (same data type)

HT mode uses the **flat layout** (`NCCL_EP_LAYOUT_FLAT`): dispatch output is a contiguous 2D sequence of
N(r) received tokens with no rank-major or expert-major structure.
This is the only layout supported by HT mode and is selected automatically when `NCCL_EP_LAYOUT_AUTO` is used.

**Handle creation**

| Operation | Struct        | Field           | Dims |
|-----------|:--------------|:----------------|:----:|
| Create    | layout_info   | expert_counters | [L]  |


**Forward pass**

| Operation | Struct             | Field             | Dims       |
|:---------:|:-------------------|:------------------|:----------:|
| Dispatch  | dispatch_inputs    | tokens            | [B x H]    |
|           | dispatch_inputs    | topk_weights      | [B x K]    |
|           | _(top-level arg)_  | topk_idx          | [B x K]    |
|           | dispatch_outputs   | tokens            | [N(r) x H] |
|           | dispatch_outputs   | topk_weights      | [N(r) x K] |
|           | dispatch_outputs   | topk_idx          | [N(r) x K] |
| Combine   | combine_inputs     | tokens            | [N(r) x H] |
|           | combine_outputs    | tokens            | [B x H]    |

**Backward pass**

Compared to the Forward pass, the Backward pass requires per-token routing
weights to be passed as `combine_inputs.topk_weights` and returned via
`combine_outputs.topk_weights`.

| Operation | Struct             | Field             | Dims       |
|:---------:|:-------------------|:------------------|:----------:|
| Dispatch  | dispatch_inputs    | tokens            | [B x H]    |
|           | dispatch_inputs    | topk_weights      | [B x K]    |
|           | _(top-level arg)_  | topk_idx          | [B x K]    |
|           | dispatch_outputs   | tokens            | [N(r) x H] |
|           | dispatch_outputs   | topk_weights      | [N(r) x K] |
|           | dispatch_outputs   | topk_idx          | [N(r) x K] |
| Combine   | combine_inputs     | tokens            | [N(r) x H] |
|           | combine_inputs     | **topk_weights**  | [N(r) x K] |
|           | combine_outputs    | tokens            | [B x H]    |
|           | combine_outputs    | **topk_weights**  | [B x K]    |

#### LL mode (FP16 -> FP8 conversion - NOT SUPPORTED)

In the token data type conversion scenario, in LL mode the Dispatch operation
will perform precision reduction and return lower-precision tokens via
`dispatch_outputs.tokens`. In addition, `dispatch_outputs.scales` must be
provided to return the scaling information.

| Operation | Struct             | Field             | Dims             |
|:---------:|:-------------------|:------------------|:----------------:|
| Dispatch  | dispatch_inputs    | tokens            | [B x H]          |
|           | dispatch_outputs   | tokens            | [L x R x B x H]  |
|           | dispatch_outputs   | **scales**        | [L x R x B x S]  |
|           | layout_info        | expert_counters   | [L]              |
| Combine   | combine_inputs     | tokens            | [L x R x B x H]  |
|           | combine_outputs    | tokens            | [B x H]          |
|           | combine_outputs    | topk_weights      | [B x K]          |

#### HT mode (FP16 -> FP8 conversion - NOT SUPPORTED)

**Forward pass**

In the HT case, scales are expected to be calculated by the user and passed
via `dispatch_inputs.scales`. NCCL EP communicates them alongside the token
data and returns them via `dispatch_outputs.scales`.

| Operation | Struct             | Field             | Dims       |
|:---------:|:-------------------|:------------------|:----------:|
| Dispatch  | dispatch_inputs    | tokens            | [B x H]    |
|           | dispatch_inputs    | topk_weights      | [B x K]    |
|           | _(top-level arg)_  | topk_idx          | [B x K]    |
|           | dispatch_inputs    | **scales**        | [B x S]    |
|           | dispatch_outputs   | tokens            | [N(r) x H] |
|           | dispatch_outputs   | topk_weights      | [N(r) x K] |
|           | dispatch_outputs   | topk_idx          | [N(r) x K] |
|           | dispatch_outputs   | **scales**        | [N(r) x S] |
| Combine   | combine_inputs     | tokens            | [N(r) x H] |
|           | combine_inputs     | topk_weights      | [N(r) x K] |
|           | combine_outputs    | tokens            | [B x H]    |

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
| `-m` | Disable max_dispatch_tokens_per_rank (HT mode only) | disabled |
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

### `ncclNDTensor_t` - Opaque Multi-dimensional Tensor Handle

An opaque handle that encapsulates tensor metadata and data layout information.
Tensors are created with `ncclEpTensorCreate` and accessed through getter functions.
The library does not own the buffer: the caller passes a non-null device pointer and is
responsible for freeing it after `ncclEpTensorDestroy`.

```c
typedef struct ncclNDTensor* ncclNDTensor_t;

// Access tensor properties:
ncclEpTensorGetData(tensor, &data);          // Get data pointer
ncclEpTensorGetSizes(tensor, &sizes, &ndim); // Get dimensions
```

### `ncclEpGroup_t` - EP Group Configuration

Created from an NCCL communicator, manages the distributed EP configuration across all ranks in the group:

```c
typedef struct {
    unsigned int size;                          // = sizeof(struct); ABI-size check
    unsigned int version;                       // = NCCL_EP_API_VERSION; feature-set version
    ncclEpAlgorithm_t algorithm;                // HT or LL mode
    ncclEpLayout_t layout;                      // recv-buffer layout (AUTO selects per algorithm)
    unsigned int num_experts;                   // Total experts across all ranks
    unsigned int max_dispatch_tokens_per_rank;      // Max tokens any single rank dispatches
    unsigned int max_token_bytes;               // Maximum token size
    unsigned long int rdma_buffer_size;         // RDMA buffer size (0=auto)
    unsigned int num_qp_per_rank;               // Queue pairs per rank (0=auto)
    unsigned int num_channels;                  // Channels per rank (0=auto)
    unsigned int max_recv_tokens_per_rank; // Per-rank recv slot budget (0=auto, LL only)
} ncclEpGroupConfig_t;

// Use NCCL_EP_GROUP_CONFIG_INIT to pre-fill `size` and `version` correctly.
```

### `ncclEpHandle_t` - Operation Handle

Maintains state for a sequence of related MoE operations, i.e. dispatch and combine pairs for forward and (optionally) backward passes. The handle encapsulates routing metadata and communication buffers.

## Algorithm-related configurations

**High Throughput (HT)**:
- Uses flat layout (`NCCL_EP_LAYOUT_FLAT`), the only layout supported by HT mode.
- Dispatch output tokens are a contiguous flat sequence: `[N(r) x hidden]` where `N(r)` is the total number of tokens targeting this rank.
  - Static allocation: `N(r) = num_ranks * max_dispatch_tokens_per_rank`.
  - Dynamic allocation (`max_dispatch_tokens_per_rank = NCCL_EP_AUTO`): `N(r)` is the actual received count, written by the metadata kernel into the optional `ncclEpLayoutInfo_t.recv_total_counter` scalar tensor.
- `dispatch_outputs.topk_idx` and `dispatch_outputs.topk_weights` carry per-slot routing metadata alongside the received tokens.
- The caller uses `topk_idx` to route each slot to the correct local expert(s), applies the weighted reduction using `topk_weights`, and passes the pre-reduced `[N(r) x hidden]` tensor as `combine_inputs.tokens` to `ncclEpCombine`.
- Supports dynamic `max_dispatch_tokens_per_rank` (set to `NCCL_EP_AUTO`)

**Low Latency (LL)**:
- Output tokens must have 3D format: `[num_experts x max_dispatch_tokens_per_rank x hidden]`
- Expert-major data layout for efficient expert processing
- Supports `send_only` (in `ncclEpDispatchConfig_t` / `ncclEpCombineConfig_t`) to enable computation/communication overlapping
- Does not support dynamic `max_dispatch_tokens_per_rank` detection

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
ncclEpCreateGroup(&ep_group, comm, &config, my_alloc, my_free);
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
//   alloc_fn   - [IN]  Optional custom allocator function (NULL for default cudaMalloc)
//   free_fn    - [IN]  Optional custom free function (NULL for default cudaFree)
//
// Returns:
//   ncclResult_t error code

ncclResult_t ncclEpCreateGroup(
    ncclEpGroup_t* ep_group,
    ncclComm_t comm,
    const ncclEpGroupConfig_t* config,
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
//
// Returns:
//   ncclResult_t error code

ncclResult_t ncclEpGroupDestroy(
    ncclEpGroup_t ep_group
);
```

### Tensor Management

#### `ncclEpTensorCreate()`

```c
// Wrap a caller-provided device buffer in a tensor descriptor.
//   Contiguous in memory (strides set to 1 for all dimensions).
//   The buffer is NOT owned by the tensor; the caller is responsible for the lifetime
//   of `data` and must keep it valid until ncclEpTensorDestroy returns.
//
// Arguments:
//   tensor   - [OUT] Pointer to the newly created tensor
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
```

#### `ncclEpTensorDestroy()`

```c
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
```

#### Tensor Accessors

```c
// Get data pointer
ncclResult_t ncclEpTensorGetData(ncclNDTensor_t tensor, void** data);

// Get sizes and dimensions
ncclResult_t ncclEpTensorGetSizes(ncclNDTensor_t tensor, const unsigned int** sizes, unsigned int* ndim);
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
//   layout_info         - [IN/OUT, optional] Named-struct pointer carrying device-side
//                         metadata tensors. Set `expert_counters` (1D ncclInt32/ncclInt64,
//                         size = num_local_experts) to receive per-expert recv counts;
//                         set `recv_total_counter` (scalar) when
//                         max_dispatch_tokens_per_rank is NCCL_EP_AUTO. NULL = no metadata.
//   config              - [IN]  Optional handle configuration (alignment, FP8 flag, ...);
//                               NULL = defaults.
//   stream              - [IN]  CUDA stream
//
// Notes:
//   - If max_dispatch_tokens_per_rank in ncclEpGroupConfig_t was set to NCCL_EP_AUTO,
//     this call may block as the host allocates memory for the actual number
//     of received tokens.
//
// Returns: ncclResult_t error code

ncclResult_t ncclEpCreateHandle(
    ncclEpHandle_t* handle,
    ncclEpGroup_t ep_group,
    ncclNDTensor_t topk_idx,
    const ncclEpLayoutInfo_t* layout_info,
    const ncclEpHandleConfig_t* config,
    cudaStream_t stream
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

### Communication Operations

#### `ncclEpDispatch()`

Perform EP dispatch: send tokens to experts according to routing decisions.

```c
// Perform EP dispatch
//   * Sends tokens and metadata to the experts according to routing decisions.
//   * This call is collective and must be invoked by all ranks in the group.
//   * Cross-boundary tensors are carried in named-struct fields
//     (ncclEpDispatchInputs_t / ncclEpDispatchOutputs_t / ncclEpLayoutInfo_t).
//
// Arguments:
//   handle        - [IN,OUT] EP handle
//   topk_idx      - [IN]     Top-k expert index tensor used by HT mode for forward dispatch.
//                            HT: 2D [num_tokens, top_k] int64. LL: not used; NULL.
//   inputs        - [IN]     Named preallocated input tensors. `inputs->tokens` is required;
//                            other fields (topk_weights, scales) are optional and depend on
//                            algorithm/layout.
//   outputs       - [IN,OUT] Named preallocated output tensors. `outputs->tokens` is required;
//                            other fields (topk_weights, topk_idx, scales) are optional and
//                            depend on algorithm/layout. For HT, outputs are 2D [N(r), data_size].
//                            For LL expert-major, outputs are 3D [num_local_experts, N(r), data_size].
//                            If FP8 conversion is requested, `outputs->scales` must be supplied.
//   layout_info   - [IN,OUT, optional] Named-struct pointer for device-side metadata tensors.
//                            LL mode: optional `expert_counters` (1D ncclInt32 / ncclInt64,
//                            size = num_local_experts) receives per-expert recv counts.
//                            NULL = no metadata.
//   config        - [IN]     Dispatch configuration.
//   stream        - [IN]     CUDA stream. If ncclEpDispatch is called on a different stream than
//                            the stream used in `ncclEpCreateHandle`, it is the responsibility
//                            of the user to synchronize between streams to ensure correctness.
//
// Returns:
//   ncclResult_t error code

ncclResult_t ncclEpDispatch(
    ncclEpHandle_t handle,
    ncclNDTensor_t topk_idx,
    const ncclEpDispatchInputs_t* inputs,
    const ncclEpDispatchOutputs_t* outputs,
    const ncclEpLayoutInfo_t* layout_info,
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
//   * Cross-boundary tensors are carried in named-struct fields
//     (ncclEpCombineInputs_t / ncclEpCombineOutputs_t).
//
// Arguments:
//   handle           - [IN,OUT] EP handle that was used for `ncclEpDispatch()` operation.
//   inputs           - [IN]     Named preallocated input tensors. `inputs->tokens` is required;
//                               LL expert-major: 3D [num_local_experts, N(r), data_size].
//                               HT / LL rank-major: 2D [N(r), data_size].
//                               Backward combine: also set `inputs->topk_weights`
//                               [N(r), top_k] (HT only).
//   outputs          - [IN,OUT] Named preallocated output tensors. `outputs->tokens` is required;
//                               2D [num_tokens, data_size] in original token order.
//                               LL expert-major: `outputs->topk_weights` [num_tokens, top_k]
//                               is used by the receive-side reduction.
//                               HT backward combine: `outputs->topk_weights` [num_tokens, top_k]
//                               receives per-token routing weights.
//   config           - [IN]     Combine configuration (e.g. `send_only` for LL staged mode).
//   stream           - [IN]     CUDA stream. If `ncclEpCombine()` is called on a different stream than
//                               the stream used in `ncclEpCreateHandle()`, it is the responsibility
//                               of the user to synchronize between streams to ensure correctness.
//
// Returns:
//   ncclResult_t error code

ncclResult_t ncclEpCombine(
    ncclEpHandle_t handle,
    const ncclEpCombineInputs_t* inputs,
    const ncclEpCombineOutputs_t* outputs,
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
ncclEpDispatchConfig_t dispatch_cfg = NCCL_EP_DISPATCH_CONFIG_INIT;
ncclEpDispatch(handle, topk_idx, &dispatch_in, &dispatch_out,
               /*layout_info=*/NULL, &dispatch_cfg, stream);
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
ncclEpDispatchConfig_t dispatch_cfg = NCCL_EP_DISPATCH_CONFIG_INIT;
dispatch_cfg.send_only = 1;
ncclEpDispatch(handle, /*topk_idx=*/NULL, &dispatch_in, &dispatch_out,
               /*layout_info=*/NULL, &dispatch_cfg, stream);
// Returns after initiating the operations

// Stage 2: Continue other computations...

// Stage 3: Wait for actual completion
ncclEpComplete(handle, /*config=*/NULL, stream);
// Now all data is actually sent/received
```

# Usage Examples

> **Note:** For a complete working example, see `ep_test.cu` which demonstrates both LL and HT modes with all API calls.

## Example 1: High Throughput Mode - Forward and Backward Pass

```c
#include "nccl.h"
#include "nccl_ep.h"
#include "cuda_runtime.h"

// The library does not own tensor memory. The caller allocates a device
// buffer and passes it to ncclEpTensorCreate; the snippets below use these
// helpers to keep the call sites compact (see ep_test.cu for the full version).
static size_t dtype_bytes(ncclDataType_t dt) { /* 1, 2, 4, 8 by type */ }

static ncclResult_t make_tensor(ncclNDTensor_t* t, unsigned int ndim,
                                ncclDataType_t dt,
                                unsigned int s0, unsigned int s1 = 1, unsigned int s2 = 1,
                                unsigned int s3 = 1, unsigned int s4 = 1) {
    size_t dims[5] = {s0, s1, s2, s3, s4};
    size_t total = dtype_bytes(dt);
    for (unsigned int i = 0; i < ndim; i++) total *= dims[i];
    void* data = nullptr;
    cudaMalloc(&data, total);
    return ncclEpTensorCreate(t, ndim, dt, data, dims);
}

// Inverse: destroy the descriptor first, then free the backing buffer
// (mirror of make_tensor's cudaMalloc + Create order).
static void free_tensor(ncclNDTensor_t t) {
    if (!t) return;
    void* data = nullptr;
    ncclEpTensorGetData(t, &data);
    ncclEpTensorDestroy(t);
    if (data) cudaFree(data);
}

// Initialize NCCL communicator
ncclComm_t comm;
ncclCommInitRank(&comm, nRanks, id, myRank);

cudaStream_t stream;
cudaStreamCreate(&stream);

unsigned int top_k = 8;
unsigned int hidden = 4096;

// Configure for High Throughput mode
ncclEpGroupConfig_t config = NCCL_EP_GROUP_CONFIG_INIT;
config.algorithm = NCCL_EP_ALGO_HIGH_THROUGHPUT;
config.num_experts = 256;
config.max_dispatch_tokens_per_rank = 4;
config.max_recv_tokens_per_rank = nRanks * config.max_dispatch_tokens_per_rank;
config.max_token_bytes = hidden * 2;  // bfloat16
config.rdma_buffer_size = NCCL_EP_AUTO;     // Auto-size
config.num_qp_per_rank = NCCL_EP_AUTO;      // Auto-size
config.num_channels = NCCL_EP_AUTO;         // Auto-size

ncclEpGroup_t ep_group;
ncclEpCreateGroup(&ep_group, comm, &config, my_alloc, my_free);

unsigned int num_local_experts = config.num_experts / nRanks;

ncclNDTensor_t topk_idx;
make_tensor(&topk_idx, 2, ncclInt64, num_tokens, top_k);

// Optional: expert_counters receives per-expert recv counts written by the
// metadata kernel during handle creation. Required when the application
// needs that breakdown (e.g. for ragged expert-side compute).
ncclNDTensor_t expert_counters = NULL;
ncclEpLayoutInfo_t handle_layout = NCCL_EP_LAYOUT_INFO_INIT;
make_tensor(&expert_counters, 1, ncclInt32, num_local_experts);
handle_layout.expert_counters = expert_counters;

// Create EP handle (can be reused for forward and backward)
ncclEpHandle_t handle;
ncclEpCreateHandle(&handle, ep_group, topk_idx, &handle_layout,
                   /*config=*/NULL, stream);

// num_recv_tokens is the max number of tokens this rank can receive.
unsigned int num_recv_tokens = config.max_recv_tokens_per_rank;

// === FORWARD PASS ===

ncclEpDispatchInputs_t  dispatch_in  = NCCL_EP_DISPATCH_INPUTS_INIT;
ncclEpDispatchOutputs_t dispatch_out = NCCL_EP_DISPATCH_OUTPUTS_INIT;

make_tensor(&dispatch_in.tokens,       2, ncclBfloat16, num_tokens, hidden);
make_tensor(&dispatch_in.topk_weights, 2, ncclFloat32,  num_tokens, top_k);

make_tensor(&dispatch_out.tokens,       2, ncclBfloat16, num_recv_tokens, hidden);
make_tensor(&dispatch_out.topk_weights, 2, ncclFloat32,  num_recv_tokens, top_k);
make_tensor(&dispatch_out.topk_idx,     2, ncclInt64,    num_recv_tokens, top_k);

ncclEpDispatchConfig_t dispatch_cfg = NCCL_EP_DISPATCH_CONFIG_INIT;
ncclEpDispatch(handle, topk_idx, &dispatch_in, &dispatch_out,
               &handle_layout, &dispatch_cfg, stream);

// Expert forward computation
// ... process dispatch_out.tokens using expert_counters to size each expert's slab ...

// Combine expert outputs back to original token order
ncclEpCombineInputs_t  combine_in  = NCCL_EP_COMBINE_INPUTS_INIT;
ncclEpCombineOutputs_t combine_out = NCCL_EP_COMBINE_OUTPUTS_INIT;
make_tensor(&combine_in.tokens,  2, ncclBfloat16, num_recv_tokens, hidden);
make_tensor(&combine_out.tokens, 2, ncclBfloat16, num_tokens, hidden);

ncclEpCombineConfig_t combine_cfg = NCCL_EP_COMBINE_CONFIG_INIT;
ncclEpCombine(handle, &combine_in, &combine_out, &combine_cfg, stream);

// === BACKWARD PASS ===
// Reuse the same handle — routing information stays the same.

ncclEpDispatchInputs_t  bwd_dispatch_in  = NCCL_EP_DISPATCH_INPUTS_INIT;
ncclEpDispatchOutputs_t bwd_dispatch_out = NCCL_EP_DISPATCH_OUTPUTS_INIT;
bwd_dispatch_in.tokens   = grad_combined;       // user-supplied grad
bwd_dispatch_out.tokens  = grad_at_experts;     // preallocated buffer

ncclEpDispatch(handle, topk_idx, &bwd_dispatch_in, &bwd_dispatch_out,
               &handle_layout, &dispatch_cfg, stream);

// Expert backward computation
// ... compute gradients for each expert ...

// Combine gradients (backward combine: also carry per-token routing weights).
ncclEpCombineInputs_t  bwd_combine_in  = NCCL_EP_COMBINE_INPUTS_INIT;
ncclEpCombineOutputs_t bwd_combine_out = NCCL_EP_COMBINE_OUTPUTS_INIT;
bwd_combine_in.tokens        = grad_expert_outputs;
bwd_combine_in.topk_weights  = combine_topk_weights_input;
bwd_combine_out.tokens       = grad_tokens;
bwd_combine_out.topk_weights = combine_topk_weights_output;

ncclEpCombine(handle, &bwd_combine_in, &bwd_combine_out, &combine_cfg, stream);

// Cleanup
ncclEpHandleDestroy(handle);
ncclEpGroupDestroy(ep_group);
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

// Configure for Low Latency mode
ncclEpGroupConfig_t config = NCCL_EP_GROUP_CONFIG_INIT;
config.algorithm = NCCL_EP_ALGO_LOW_LATENCY;
config.num_experts = 256;
config.max_dispatch_tokens_per_rank = 128;  // Required for LL mode
config.max_token_bytes = hidden * 2;    // bfloat16
config.rdma_buffer_size = NCCL_EP_AUTO; // Auto-size
config.num_qp_per_rank = NCCL_EP_AUTO;  // Auto-size (or specify for LL)
config.num_channels = NCCL_EP_AUTO;     // Auto-size

ncclEpGroup_t ep_group;
ncclEpCreateGroup(&ep_group, comm, &config, my_alloc, my_free);

unsigned int num_local_experts = config.num_experts / nRanks;

// Create routing tensor (topk_idx) — required at handle creation in LL mode.
ncclNDTensor_t topk_idx;
make_tensor(&topk_idx, 2, ncclInt64, num_tokens, top_k);

// Create EP handle (no layout_info needed for the LL handle itself).
ncclEpHandle_t handle;
ncclEpCreateHandle(&handle, ep_group, topk_idx, /*layout_info=*/NULL,
                   /*config=*/NULL, stream);

// === FORWARD PASS ===

ncclEpDispatchInputs_t  dispatch_in  = NCCL_EP_DISPATCH_INPUTS_INIT;
ncclEpDispatchOutputs_t dispatch_out = NCCL_EP_DISPATCH_OUTPUTS_INIT;
ncclEpLayoutInfo_t      layout_info  = NCCL_EP_LAYOUT_INFO_INIT;

// Input: tokens [B x H].
make_tensor(&dispatch_in.tokens, 2, ncclBfloat16, num_tokens, hidden);

// Output: expert-major 3D [num_local_experts, nRanks * max_dispatch_tokens_per_rank, hidden].
make_tensor(&dispatch_out.tokens, 3, ncclBfloat16,
            num_local_experts,
            nRanks * config.max_dispatch_tokens_per_rank,
            hidden);

// expert_counters [num_local_experts] receives per-expert token counts.
make_tensor(&layout_info.expert_counters, 1, ncclInt32, num_local_experts);

// Dispatch tokens to experts (staged execution for overlap)
ncclEpDispatchConfig_t dispatch_cfg = NCCL_EP_DISPATCH_CONFIG_INIT;
dispatch_cfg.send_only = 1;
ncclEpDispatch(handle, /*topk_idx=*/NULL, &dispatch_in, &dispatch_out,
               &layout_info, &dispatch_cfg, stream);

// Overlap with other computation...
// doOtherWork(stream);

// Wait for dispatch to complete
ncclEpComplete(handle, /*config=*/NULL, stream);
cudaStreamSynchronize(stream);

// Expert forward computation:
// Process dispatch_out.tokens in 3D layout [experts x tokens x hidden],
// using layout_info.expert_counters to size each expert's valid range.

// Combine inputs: per-expert post-processed activation, same shape as dispatch_out.tokens.
ncclEpCombineInputs_t  combine_in  = NCCL_EP_COMBINE_INPUTS_INIT;
ncclEpCombineOutputs_t combine_out = NCCL_EP_COMBINE_OUTPUTS_INIT;
make_tensor(&combine_in.tokens, 3, ncclBfloat16,
            num_local_experts,
            nRanks * config.max_dispatch_tokens_per_rank,
            hidden);

// Combine output: [B x H] back to original token order; per-token routing
// weights drive the receive-side reduction.
make_tensor(&combine_out.tokens,       2, ncclBfloat16, num_tokens, hidden);
make_tensor(&combine_out.topk_weights, 2, ncclFloat32,  num_tokens, top_k);

ncclEpCombineConfig_t combine_cfg = NCCL_EP_COMBINE_CONFIG_INIT;
combine_cfg.send_only = 1;
ncclEpCombine(handle, &combine_in, &combine_out, &combine_cfg, stream);

ncclEpComplete(handle, /*config=*/NULL, stream);
cudaStreamSynchronize(stream);

// Cleanup
ncclEpHandleDestroy(handle);
ncclEpGroupDestroy(ep_group);
ncclCommDestroy(comm);
cudaStreamDestroy(stream);
```
