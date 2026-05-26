<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0

  See LICENSE.txt for more license information
-->

# Fusing Computation and Communication Examples

This directory contains examples that demonstrate how to fuse computation with communication using NCCL Device API. By fusing computation with communication at a fine-grained level, these examples showcase advanced techniques for achieving optimal performance in distributed GPU applications.

## Overview

**Note**: These examples are designed to showcase the NCCL Device API and the fusion of computation with communication. Simplicity and clarity are prioritized to illustrate the concepts; the implementations are not optimized for maximum performance.

**Note**: The sources in this directory are written and tested against **NCCL 2.30 and later**. Device API types, `ncclDevCommRequirements` fields, and symbol names can differ in older NCCL releases. If you build against an earlier NCCL version, expect to adjust includes, requirement flags, and API calls to match that version’s headers and documentation.

**Note**: Throughout these examples, the sequence length (number of tokens) is assumed to be divisible by the number of ranks. That assumption keeps partitioning and indexing straightforward. In real deployments you typically need to handle edge cases—for example when the sequence length is not evenly divisible by the rank count—using padding, remainder slices, or other strategies appropriate to your workload. That is why the examples still use `sequence_length` through host and device APIs even when a given kernel does not use it: a production implementation would rely on it for bounds, padding, and remainder handling. Under the even-split assumption here, it is often redundant in the kernel body on purpose, so the code stays simple while the signature matches what you would extend for those cases.

Traditional approaches separate computation and communication into distinct phases. However, modern GPU architectures and NCCL's Device API enables kernels to perform both computation and communication simultaneously, reducing overall latency and improving throughput. These examples demonstrate RMSNorm (Root Mean Square Normalization) operations fused with reduce-scatter and all-gather communication patterns.

**Implementation assumption:** We consider an implementation that **requires** a **reduce-scatter** step (each rank contributes partial activations so that, after reduction, every rank holds full activations for the tokens it owns) and an **all-gather** step (normalized results are broadcast so every rank has what it needs for the next layer). The three fused phases in the kernels correspond to that schedule; device code uses LSA, GIN, or hybrid memory operations rather than separate host-launched collectives, but the logical pattern is still reduce-scatter, local RMSNorm, then all-gather.

### RMSNorm Operation

RMSNorm is a normalization technique commonly used in transformer models. For a vector $`\mathbf{x} = \bigl[\, x_1,\, x_2,\, \ldots,\, x_n \,\bigr]`$, the RMSNorm is computed as:

```math
\text{RMS}(\mathbf{x}) = \sqrt{\frac{1}{n}\sum_{i=1}^{n} x_i^2 + \epsilon}
```

```math
y_j = \frac{x_j}{\text{RMS}(\mathbf{x})}
```

where $\mathbf{y}$ is the normalized output vector, $j$ indexes each element of the output, $i$ indexes the features (dimensions) of the hidden representation, $n$ is the hidden dimension size, and $\epsilon$ is a small constant for numerical stability.

**Note**: As in PyTorch's [RMSNorm](https://docs.pytorch.org/docs/stable/generated/torch.nn.RMSNorm.html), $\epsilon$ is added **inside** the square root (to the mean of squares before taking the root), not outside. The complete RMSNorm typically includes learnable scale parameters $\gamma_j$ such that $`y_j = \frac{x_j}{\text{RMS}(\mathbf{x})} \cdot \gamma_j`$. For simplicity, these examples assume $\gamma_j = 1$ for all $j$, focusing on the communication and reduction patterns rather than the learned scaling.

### Distributed RMSNorm: A Concrete Example

Consider a **4×8 matrix** with $L=4$ tokens, hidden dimension $H=8$, distributed across $N=2$ GPUs:

In tensor parallelism, the previous layer (e.g., a feed-forward network or attention layer) is split across multiple GPUs. Each GPU computes a partial result for all tokens. Before applying RMSNorm, we need to:
1. **Gather** partial contributions from all GPUs
2. **Sum** them to get the complete activation for each token
3. **Normalize** using RMSNorm
4. **Distribute** the normalized results back to all GPUs for the next layer

The partial contributions come from the previous layer in a tensor-parallel setup: when layers such as feed-forward networks or attention are partitioned across GPUs along the hidden dimension, each GPU produces a partial result for all tokens.

**Why RMSNorm Requires Reduce-Scatter and All-Gather Communication**

RMSNorm must be computed on the **complete** activations (after summing all partial contributions), not on the partial results from individual GPUs. This is because:
- The RMS normalization depends on all features:

    ```math
    \text{RMS}(\mathbf{x}) = \sqrt{\frac{1}{H}\sum_{d=0}^{H-1} x_d^2}
    ```

- Computing RMSNorm on partial contributions would yield incorrect statistics

Therefore, the distributed RMSNorm operation must follow these steps:
1. **Phase 1 (Reduce-Scatter)**: Gather and sum the partial contributions from all GPUs for each token to reconstruct the complete activation
2. **Phase 2 (RMSNorm)**: Apply RMSNorm on the complete activation
3. **Phase 3 (All-Gather)**: Distribute the normalized results back to all GPUs for the next layer

This is the motivation for the three-phase pattern demonstrated in these examples.

**Initial State** - Each GPU holds partial contributions for all tokens from the previous layer:

GPU 0 holds:
```math
\mathbf{X}^0 = \begin{bmatrix}
x^0_{0,0} & x^0_{0,1} & \cdots & x^0_{0,7} \\
x^0_{1,0} & x^0_{1,1} & \cdots & x^0_{1,7} \\
x^0_{2,0} & x^0_{2,1} & \cdots & x^0_{2,7} \\
x^0_{3,0} & x^0_{3,1} & \cdots & x^0_{3,7}
\end{bmatrix}
```

GPU 1 holds:
```math
\mathbf{X}^1 = \begin{bmatrix}
x^1_{0,0} & x^1_{0,1} & \cdots & x^1_{0,7} \\
x^1_{1,0} & x^1_{1,1} & \cdots & x^1_{1,7} \\
x^1_{2,0} & x^1_{2,1} & \cdots & x^1_{2,7} \\
x^1_{3,0} & x^1_{3,1} & \cdots & x^1_{3,7}
\end{bmatrix}
```

where the superscript denotes the GPU rank and subscripts denote $x^{\text{rank}}_{\text{token}, \text{dimension}}$.

#### Phase 1: Reduce-Scatter

Each GPU becomes responsible for specific tokens:
- **GPU 0**: Computes tokens 0 and 1
- **GPU 1**: Computes tokens 2 and 3

After the reduce-scatter communication and reduction, each GPU holds the summed contributions.

**Computing the sums** - Each element is obtained by summing contributions from all GPUs. For example, GPU 0 computes for tokens 0 and 1:

```math
s_{0,0} = x^0_{0,0} + x^1_{0,0}, \quad s_{0,1} = x^0_{0,1} + x^1_{0,1}, \quad \ldots, \quad s_{0,7} = x^0_{0,7} + x^1_{0,7}
```

```math
s_{1,0} = x^0_{1,0} + x^1_{1,0}, \quad s_{1,1} = x^0_{1,1} + x^1_{1,1}, \quad \ldots, \quad s_{1,7} = x^0_{1,7} + x^1_{1,7}
```

Similarly, GPU 1 computes for tokens 2 and 3:

```math
s_{2,0} = x^0_{2,0} + x^1_{2,0}, \quad s_{2,1} = x^0_{2,1} + x^1_{2,1}, \quad \ldots, \quad s_{2,7} = x^0_{2,7} + x^1_{2,7}
```

```math
s_{3,0} = x^0_{3,0} + x^1_{3,0}, \quad s_{3,1} = x^0_{3,1} + x^1_{3,1}, \quad \ldots, \quad s_{3,7} = x^0_{3,7} + x^1_{3,7}
```

In general, $s_{t,d} = \sum_{r=0}^{1} x^r_{t,d}$ represents the complete sum across all GPU contributions.

**Resulting matrices:**

GPU 0 receives and sums:
```math
\mathbf{S}^0 = \begin{bmatrix}
s_{0,0} & s_{0,1} & \cdots & s_{0,7} \\
s_{1,0} & s_{1,1} & \cdots & s_{1,7}
\end{bmatrix}
```

GPU 1 receives and sums:
```math
\mathbf{S}^1 = \begin{bmatrix}
s_{2,0} & s_{2,1} & \cdots & s_{2,7} \\
s_{3,0} & s_{3,1} & \cdots & s_{3,7}
\end{bmatrix}
```

where $s_{t,d} = x^0_{t,d} + x^1_{t,d}$ represents the complete summed value (no longer a partial GPU contribution).

#### Phase 2: RMSNorm Computation

Each GPU computes RMSNorm for its assigned tokens. For token $t$ on GPU $g$:

```math
\text{RMS}_t = \sqrt{\frac{1}{8}\sum_{d=0}^{7} (\mathbf{S}^{g}_{t,d})^2 + \epsilon}
```

```math
y_{t,d} = \frac{\mathbf{S}^{g}_{t,d}}{\text{RMS}_t}
```

After normalization:

GPU 0 produces:
```math
\mathbf{Y}^0 = \begin{bmatrix}
y_{0,0} & y_{0,1} & \cdots & y_{0,7} \\
y_{1,0} & y_{1,1} & \cdots & y_{1,7}
\end{bmatrix}
```

GPU 1 produces:
```math
\mathbf{Y}^1 = \begin{bmatrix}
y_{2,0} & y_{2,1} & \cdots & y_{2,7} \\
y_{3,0} & y_{3,1} & \cdots & y_{3,7}
\end{bmatrix}
```

#### Phase 3: All-Gather

Each GPU broadcasts its normalized tokens back to all GPUs. After this phase, every GPU has the complete normalized result:

Both GPU 0 and GPU 1 hold:
```math
\mathbf{Y} = \begin{bmatrix}
y_{0,0} & y_{0,1} & \cdots & y_{0,7} \\
y_{1,0} & y_{1,1} & \cdots & y_{1,7} \\
y_{2,0} & y_{2,1} & \cdots & y_{2,7} \\
y_{3,0} & y_{3,1} & \cdots & y_{3,7}
\end{bmatrix}
```

### Fusion Benefits

By fusing these phases within a single kernel, we can:
- Minimize kernel launch overhead
- Save the round trip to global memory compared to multiple kernel launches

### Equivalent Using NCCL Collectives

Before diving into the Device API implementations, it is useful to see how the same distributed RMSNorm pattern can be expressed using standard NCCL collective semantics. The three-phase pattern (reduce-scatter → RMSNorm → all-gather) maps directly to `ncclReduceScatter`, a custom kernel, and `ncclAllGather`:

```cpp
// Phase 1: Reduce-Scatter — sum partial contributions; each rank receives its tokens' sums
// sendbuff: full tensor (sequence_length * hidden_dim) of partial contributions
// recvbuff: tokens_per_gpu * hidden_dim — reduced sum for this rank's tokens
size_t recv_count = tokens_per_gpu * hidden_dim;
NCCLCHECK(ncclReduceScatter(sendbuff, recvbuff, recv_count, ncclFloat, ncclSum, comm, stream));

// Phase 2: RMS Normalization — kernel on the reduced data
rmsnormKernel<<<...>>>(recvbuff, tokens_per_gpu, hidden_dim, eps);

// Phase 3: All-Gather — broadcast normalized results to all ranks
// Send from recvbuff (our normalized tokens); receive full tensor into sendbuff
size_t send_count = tokens_per_gpu * hidden_dim;
NCCLCHECK(ncclAllGather(recvbuff, sendbuff, send_count, ncclFloat, comm, stream));
```

**Key differences from the Device API approach:**
- **Separate phases**: Each collective and kernel is a distinct launch; the CPU orchestrates the sequence.
- **Buffer management**: Typically requires separate send/recv buffers or careful in-place handling.
- **No fusion**: Communication and computation cannot overlap within a single kernel.

The Device API examples below achieve the same logical pattern but **fuse** all three phases into a single kernel launch, with communication initiated directly from GPU code (Load Store Accessible (LSA), GPU-Initiated Networking (GIN), or hybrid). This eliminates kernel launch overhead between phases and enables tighter overlap of computation with communication.

## Examples

| # | Example | Description | Target Architecture / Topology |
|---|---------|-------------|----------|
| 01 | [01_rmsnorm_lsa](01_rmsnorm_lsa/) | Pure LSA RMSNorm | Single-node / single NVLink domain |
| 02 | [02_rmsnorm_multimem](02_rmsnorm_multimem/) | Multimem RMSNorm (SM 9.0+) | Single-node / single NVLink domain (Hopper+) |
| 03 | [03_rmsnorm_gin](03_rmsnorm_gin/) | Pure GIN RMSNorm | Multi-node |
| 04 | [04_rmsnorm_hybrid](04_rmsnorm_hybrid/) | Hybrid LSA/GIN RMSNorm | Multi-node with NVLink per node |

### Device Communicator Requirements

All examples create a device communicator via `ncclResult_t ncclDevCommCreate(ncclComm_t comm, ncclDevCommRequirements_t const* reqs, ncclDevComm_t* outDevComm)`.
The `ncclDevCommRequirements` structure specifies which resources each kernel needs.

**01 LSA**

| Requirement | Value | Purpose |
|-------------|-------|---------|
| `lsaBarrierCount` | `tokens_per_gpu` | One LSA barrier per block; synchronizes across LSA peers between phases |

**02 Multimem**

| Requirement | Value | Purpose |
|-------------|-------|---------|
| `lsaBarrierCount` | `tokens_per_gpu` | One LSA barrier per block |
| `lsaMultimem` | `true` | Enables Multimem for `ncclGetLsaMultimemPointer` and PTX multimem instructions (requires SM 9.0+) |

**03 GIN**

| Requirement | Value | Purpose |
|-------------|-------|---------|
| `ginConnectionType` | `NCCL_GIN_CONNECTION_FULL` | Enables full GIN connectivity for remote PUT |
| `worldGinBarrierCount` | `tokens_per_gpu` | One world GIN barrier per block; used by `ncclGinBarrierSession` with `ncclTeamTagWorld` for fence-level sync |
| `ginSignalCount` | `tokens_per_gpu` | One signal per block; `gin.waitSignal()` observes inbound signaled PUTs from peers (see GIN signals below) |

**04 Hybrid**

| Requirement | Value | Purpose |
|-------------|-------|---------|
| `ginConnectionType` | `NCCL_GIN_CONNECTION_FULL` | Enables GIN for remote peer PUT |
| `barrierCount` | `tokens_per_gpu` | One hybrid barrier per block; `ncclBarrierSession` with `ncclTeamTagWorld` and `ncclGin` provides both `bar.sync` and `bar.lsaBarrier()` (not a separate `lsaBarrierCount`) |
| `ginSignalCount` | `tokens_per_gpu` | One signal per block; counts inbound signaled PUTs from **remote** peers only (LSA peers do not signal) |

**Notes:**
- All examples (01–04) use `ncclCommQueryProperties` to verify requirements before proceeding. All require Device API support.
- LSA examples (01, 02) require a single LSA team (`nLsaTeams == 1`).
- Example 02 tests for multimem support (`props.multimemSupport`).
- GIN examples (03, 04) require GIN support (`ginType != NCCL_GIN_TYPE_NONE`).

### Host Program Setup Flow

All examples follow the same host setup pattern. Host memory allocation is performed **after** verifying Device API (and GIN/LSA) support, so that early exits on unsupported configurations do not require freeing host buffers.

**Host buffers**: `h_data` holds the initial input (populated by `initialize_data()` with rank-specific values) and is used to compute expected results for verification. `h_data_validation` receives the GPU output for comparison. Both are allocated in Step 2.

1. **Step 1: Initialize NCCL Communicator** — Create the communicator and verify capability support. Example 02 checks GPU compute capability (SM 9.0+) before creating the comm, then uses `ncclCommQueryProperties` to verify multimem support (`props.multimemSupport`). All examples use `ncclCommQueryProperties` to verify Device API support; LSA examples (01, 02) also check LSA topology (`nLsaTeams == 1`); GIN examples (03, 04) check GIN support (`ginType != NCCL_GIN_TYPE_NONE`). Exit with comm cleanup if unsupported.
2. **Step 2: Allocate Host Memory and Initialize Data** — Allocate `h_data` and `h_data_validation` (each `sequence_length * hidden_size * sizeof(float)` bytes). Call `initialize_data(h_data, tensor_size, my_rank)` to populate input. On allocation failure, finalize and destroy the communicator before returning.
3. **Step 3: Allocate Device Memory and Register Windows** — Allocate device buffers, register windows, copy `h_data` to device, create CUDA stream. Examples 01 and 02 use a single symmetric window; examples 03 and 04 use separate send and recv windows.
4. **Step 4: Create Device Communicator** — Call `ncclDevCommCreate` with the required resources (LSA barriers, GIN signals, etc.).
5. **Step 5: Launch Fused RMSNorm Kernel**
6. **Step 6: Verify Results Against CPU Reference** — Compute expected results from `h_data`, copy GPU output to `h_data_validation`, compare.
7. **Step 7: Cleanup Resources** — Destroy device communicator, deregister windows, free device memory, finalize/destroy NCCL comm, destroy stream, free `h_data` and `h_data_validation`.
8. **Step 8: Report Results**

### [01_rmsnorm_lsa](01_rmsnorm_lsa/)

**Pure Load Store Accessible (LSA) RMSNorm Implementation**

This example demonstrates fused computation and communication using NCCL's Load Store Accessible (LSA) mechanism. LSA enables GPUs to directly access memory on peer GPUs within a single-node / single NVLink domain using high-bandwidth NVLink connections.

**Key Features:**
- Uses `ncclLsaBarrierSession` for synchronization
- Leverages `ncclGetLsaPointer` for direct peer memory access

#### Implementation Details

This example implements the complete distributed RMSNorm operation within a single GPU kernel using NCCL's LSA APIs. Here's how each phase is implemented:

##### Kernel Configuration

The kernel is launched with:
- **Grid Dimensions**: `tokens_per_gpu` blocks (each GPU processes a subset of tokens)
- **Block Dimensions**: 256 threads per block
- **Shared Memory**: Dynamic allocation for block-level reductions (one float per thread)

```cuda
const size_t shared_mem_size = threads_per_block * sizeof(float);
RMSNormLSA<<<tokens_per_gpu, threads_per_block, shared_mem_size, stream>>>(
    window, devComm, sequence_length, hidden_size, eps);
```

##### Phase 1: Reduce-Scatter via LSA

Each thread block is responsible for one token and performs reduction across all GPUs:

```cuda
ncclCoopCta coop = ncclCoopCta();

// Initialize LSA barrier session for this block
ncclLsaBarrierSession<ncclCoopCta> bar {
    coop, devComm, ncclTeamTagLsa(), blockIdx.x
};

// Initial synchronization across all GPUs
bar.sync(coop, cuda::memory_order_acquire);

const int rank = devComm.rank;
const int nRanks = devComm.nRanks;
const int token_idx = rank * gridDim.x + blockIdx.x;
const int window_offset = token_idx * hidden_dim * sizeof(float);
float* local_pointer = (float*)ncclGetLocalPointer(window, window_offset);

// Accumulate contributions from all peers
for (int i = threadIdx.x; i < hidden_dim; i += blockDim.x) {
    float sum = local_pointer[i];
    for (int peer = 0; peer < nRanks; peer++) {
        if (peer == rank) continue;  // Skip self
        float* peer_token_data = (float*)ncclGetLsaPointer(window, window_offset, peer);
        sum += peer_token_data[i];
    }
    local_pointer[i] = sum;
}

coop.sync();
```

**Key LSA APIs:**
- `ncclGetLocalPointer(window, offset)`: Returns a pointer to the local GPU's memory at the specified window offset. Used to store the reduced results directly in the registered window.
- `ncclGetLsaPointer(window, offset, peer)`: Returns a pointer to peer GPU's memory at the specified window offset. This pointer can be directly dereferenced for read/write operations.
- `ncclLsaBarrierSession`: Provides fine-grained synchronization between GPUs, ensuring all peers have completed their memory operations before proceeding.

**Memory Access Pattern:**
- The ranks collectively register the memory symmetrically with `ncclCommWindowRegister()`
- The window contains the full tensor (all tokens)
- `window_offset` calculates the position of token `token_idx` in the window

##### Phase 2: RMS Normalization

After the block synchronizes following Phase 1, each block normalizes its assigned token using the `blockRMSNorm()` device function:

```cuda
blockRMSNorm(local_pointer, hidden_dim, eps, reduction_buffer, coop);
```


The `blockRMSNorm()` function (in `include/rmsnorm_utils.cuh`) implements a block-level parallel reduction to calculate the sum of squared values, required for the RMS calculation:

1. **Thread-level accumulation**: Each thread computes partial sum of squares for its assigned elements:
   ```cuda
   float thread_sum = 0.0f;
   for (int i = threadIdx.x; i < hidden_dim; i += blockDim.x) {
       float val = token_data[i];
       thread_sum += val * val;
   }
   reduction_buffer[threadIdx.x] = thread_sum;
   coop.sync();
   ```

2. **Block-level reduction**: Parallel reduction across all threads using shared memory:
   ```cuda
   for (int s = blockDim.x / 2; s >= 1; s /= 2) {
       if (threadIdx.x < s) {
           reduction_buffer[threadIdx.x] += reduction_buffer[threadIdx.x + s];
       }
       coop.sync();
   }
   float rms_scale = rsqrtf((reduction_buffer[0] / hidden_dim) + eps);
   ```

3. **Apply normalization**: All threads apply the computed RMS factor:
   ```cuda
   for (int i = threadIdx.x; i < hidden_dim; i += blockDim.x) {
       token_data[i] *= rms_scale;
   }
   coop.sync();
   ```

##### Phase 3: All-Gather via LSA

The normalized results are written back to all peer GPUs using direct LSA writes:

```cuda
// Write to all peer GPUs (including self)
for (int peer = 0; peer < nRanks; peer++) {
    float* peer_token_data = (float*)ncclGetLsaPointer(window, window_offset, peer);
    for (int i = threadIdx.x; i < hidden_dim; i += blockDim.x) {
        peer_token_data[i] = local_pointer[i];
    }
}
bar.sync(coop, cuda::memory_order_release);
```

**Synchronization Strategy:**
- `cuda::memory_order_acquire`: Used for the initial barrier before reading peer data, ensuring visibility of setup
- `cuda::memory_order_release`: Used for the final barrier to ensure all memory writes are visible to other GPUs
- `ncclCoopCta coop`: CTA cooperation handle passed to `bar.sync()` and to `blockRMSNorm()` (which uses `coop.sync()` for block-wide synchronization inside the helper)

##### Memory Layout

Host buffers `h_data` and `h_data_validation` are allocated and initialized in Step 2 of the Host Program Setup Flow. The device layout uses a single allocation:

1. **Symmetric Window** (`d_data`): Size = `sequence_length * hidden_size * sizeof(float)`
   - Registered with `ncclCommWindowRegister()` for LSA access
   - Contains the full tensor (all tokens across all dimensions)
   - Each GPU can access any part of any peer's window
   - This is where input data resides and final results are written
   - The kernel uses `ncclGetLocalPointer()` to obtain a pointer for in-place reduction and normalization

For a system with $N$ GPUs and sequence length $L$, each GPU's symmetric window holds all $L$ tokens (initially the partial contributions $\mathbf{X}^r$, where $r$ is the rank).

##### Performance Characteristics

**Advantages of LSA:**
- **High Bandwidth**: Utilizes full NVLink bandwidth
- **No Intermediate Copies**: Data moves directly from peer memory to registers

**Block-level Parallelism:**
- Each block operates independently on one token
- `tokens_per_gpu` blocks execute concurrently (limited by GPU resources)
- Barriers synchronize only when necessary for correctness

### [02_rmsnorm_multimem](02_rmsnorm_multimem/)

**Multimem RMSNorm Implementation**

This example demonstrates fused computation and communication using NCCL's Multimem capability. Multimem leverages hardware multicast memory operations (SM 9.0+) to perform reduce-scatter and gather with a single pointer, simplifying the kernel logic compared to explicit peer iteration.

**Key Features:**
- Uses `ncclGetLsaMultimemPointer` for multicast memory access
- `multimemLoadSum()` for Phase 1: hardware-accelerated sum across all LSA peers
- `multimemStore()` for Phase 3: broadcast normalized results to all peers
- Each rank processes only its own tokens (reduce-scatter semantics)
- Requires GPU compute capability 9.0 or higher (Hopper H100+)


#### Implementation Details

##### Phase 1: Reduce-Scatter via Multimem

Each block loads and sums from all peers using a single multimem pointer. The reduced result is stored locally only (each rank handles its own tokens). The barrier setup follows the LSA example (`01_rmsnorm_lsa`) with the multimem-enabled constructor:

```cuda
ncclCoopCta coop = ncclCoopCta();

ncclLsaBarrierSession<ncclCoopCta> bar {
    coop, devComm, ncclTeamTagLsa(), blockIdx.x, /*multimem=*/true
};

// Initial synchronization across all GPUs
bar.sync(coop, cuda::memory_order_acquire);

const int rank = devComm.rank;
const int token_idx = rank * gridDim.x + blockIdx.x;  // Global token index
const int window_offset = token_idx * hidden_dim * sizeof(float);
float* local_pointer = (float*)ncclGetLocalPointer(window, window_offset);
float* multimem_pointer = reinterpret_cast<float*>(
    ncclGetLsaMultimemPointer(window, window_offset, devComm));

// Load and sum from all peers. Each rank only needs the reduced result for its
// own tokens, so we store to local only (no multimemStore here).
for (int i = threadIdx.x; i < hidden_dim; i += blockDim.x) {
    local_pointer[i] = multimemLoadSum(multimem_pointer + i);
}

coop.sync();
```

**Key Multimem APIs:**
- `ncclGetLsaMultimemPointer(window, offset, devComm)`: Returns a multicast pointer; loads read from all LSA peers, stores write to all LSA peers
- `multimemLoadSum(addr)`: PTX `multimem.ld_reduce.global.add.f32` - loads and sums from all peers
- `multimemStore(addr, val)`: PTX `multimem.st.global.b32` - stores value to all peers

##### Phase 2: RMS Normalization

Identical to the LSA example — `blockRMSNorm()` on the local reduced data:

```cuda
blockRMSNorm(local_pointer, hidden_dim, eps, reduction_buffer, coop);
```

##### Phase 3: All-Gather via Multimem

Broadcast normalized results to all peers using `multimemStore`:

```cuda
for (int i = threadIdx.x; i < hidden_dim; i += blockDim.x) {
    float val = local_pointer[i];  // Read once from local
    multimemStore(multimem_pointer + i, val);
}
bar.sync(coop, cuda::memory_order_release);
```

**Synchronization Strategy:** Same as LSA — initial barrier uses `cuda::memory_order_acquire` before Phase 1; final barrier uses `cuda::memory_order_release` after Phase 3.

### [03_rmsnorm_gin](03_rmsnorm_gin/)

**Pure GPU-Initiated Networking (GIN) RMSNorm Implementation**

This example demonstrates fused computation and communication using NCCL's GPU-Initiated Networking (GIN). GIN allows GPU kernels to directly initiate remote communication without CPU involvement, enabling efficient multi-node communication.

**Key Features:**
- Uses `ncclGin` for GPU-initiated PUT operations across the full communicator
- Uses `ncclGinBarrierSession` with `ncclTeamTagWorld()` for GIN-only cross-rank barriers (pairs with `worldGinBarrierCount` on the host)
- Issues remote **PUT** operations via `gin.put()`
- Implements signal-based synchronization with `gin.waitSignal()`
- Applies a uniform GIN communication pattern across all ranks and is primarily aimed at multi-node setups


#### Implementation Details

The GPU-Initiated Networking (GIN) example implements the complete distributed RMSNorm operation within a single GPU kernel using NCCL's GPU-Initiated Networking APIs. Here's how each phase is implemented:

##### Kernel Configuration

The kernel is launched with:
- **Grid Dimensions**: `tokens_per_gpu` blocks (each GPU processes a subset of tokens)
- **Block Dimensions**: 256 threads per block
- **Shared Memory**: Dynamic allocation for block-level reductions (one float per thread)

```cuda
const size_t shared_mem_size = threads_per_block * sizeof(float);
RMSNormGIN<<<tokens_per_gpu, threads_per_block, shared_mem_size, stream>>>(
    window_send, window_recv, devComm, sequence_length, hidden_size, eps);
```

Host code creates the device communicator with `worldGinBarrierCount` (see table above):

```cuda
ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
reqs.ginConnectionType = NCCL_GIN_CONNECTION_FULL;
reqs.worldGinBarrierCount = tokens_per_gpu;
reqs.ginSignalCount = tokens_per_gpu;
NCCLCHECK(ncclDevCommCreate(comm, &reqs, &devComm));
```

##### Phase 1: Reduce-Scatter via GPU-Initiated Networking (GIN) PUT

In pure GIN, **each thread block is responsible for a single token** on this rank. Block `blockIdx.x` owns global token index `token_idx = rank * gridDim.x + blockIdx.x`. Phase 1 has two logical steps for that block: **(1)** exchange—each rank issues **PUTs** (`gin.put`) to every rank in the communicator, **including itself**, so each rank’s receive window gets the partials it needs; for this block, **every rank’s contribution for `token_idx`** lands in this rank’s `window_recv`. **(2)** **reduce**—`gin.waitSignal` blocks until **all ranks** have finished those **signaled** **PUTs** for `token_idx`, and `gin.flush` completes **outbound** GIN work locally so **`window_send` regions used as PUT sources** are safe to reuse.

**PUT pattern:** For each rank `peer` in the communicator, each **PUT** (`gin.put`) sends **this rank’s** partial activations for the token that **that rank** will own at the same block index—`peer_token_idx = peer * gridDim.x + blockIdx.x`—from `window_send` at `peer_token_idx`, into that rank’s `window_recv` at the offset for `token_idx`. This includes the self case (`peer == rank`) for a uniform code path and matching signal accounting. Symmetrically, this rank receives each rank’s contribution for `token_idx` into its own `window_recv`. The strided receive layout lines up with the per-block reduction that follows.

```cuda
ncclCoopCta coop = ncclCoopCta();

// Initialize GIN context with per-block signal
// Use multiple GIN contexts to spread blocks across communication channels
int ginContext = blockIdx.x % devComm.ginContextCount;
unsigned int signalIndex = blockIdx.x;
ncclGin gin { devComm, ginContext };
uint64_t signalValue = gin.readSignal(signalIndex);

const int rank = devComm.rank;
const int nRanks = devComm.nRanks;
const int token_idx = rank * gridDim.x + blockIdx.x;

ncclGinBarrierSession<ncclCoopCta> bar { coop, gin, ncclTeamTagWorld(), blockIdx.x };
bar.sync(coop, cuda::memory_order_acquire, ncclGinFenceLevel::None);

size_t my_window_offset = (token_idx * hidden_dim) * sizeof(float);

for (int peer = threadIdx.x; peer < nRanks; peer += blockDim.x) {
  const int peer_token_idx = peer * gridDim.x + blockIdx.x;
  size_t peer_window_offset = (peer_token_idx * hidden_dim) * sizeof(float);

  // PUT: send our token data to peer's receive window
  gin.put(ncclTeamWorld(devComm), peer, window_recv, my_window_offset,
          window_send, peer_window_offset, sizeof(float) * hidden_dim,
          ncclGin_WeakSignalInc{signalIndex});
}

// Wait until every peer has completed its signaled PUT delivering its contribution for
// token_idx into this rank's window_recv (reduce-scatter gather for this block's token).
gin.waitSignal(coop, signalIndex, signalValue + devComm.nRanks);
// Flush outbound Phase 1 PUTs so window_send regions used as sources are safe to reuse.
gin.flush(coop);

//----------------------------------------------------------------------------
// Reduction: Sum contributions from all peers
//----------------------------------------------------------------------------
float *my_token_data = (float*)ncclGetLocalPointer(window_recv, blockIdx.x * hidden_dim * sizeof(float));

// Sum across all peer contributions for this dimension, using the first row of the
// corresponding token as a buffer, therefore skip index 0 to avoid adding twice
for (int peer = 1; peer < nRanks; peer++) {
  for (int j = threadIdx.x; j < hidden_dim; j += blockDim.x) {
    my_token_data[j] += my_token_data[j + peer * hidden_dim * gridDim.x];
  }
}
```

**Important**: All GPUs perform the reduction by summing contributions from all peers and storing the result in the **first part** of each block's section in the receive window. Specifically:
- Each block's data starts at offset `blockIdx.x * hidden_dim` in `window_recv`
- Contributions from peer $i$ are at offset `blockIdx.x * hidden_dim + i * hidden_dim * tokens_per_gpu`
- The sum overwrites position `blockIdx.x * hidden_dim` (the first peer's contribution space)
- This in-place reduction saves memory bandwidth and simplifies Phase 2 access

**Key GIN APIs:**
- `ncclGinBarrierSession<ncclCoopCta> bar { coop, gin, ncclTeamTagWorld(), blockIdx.x }`: World-team GIN barrier session; requires `worldGinBarrierCount` in `ncclDevCommRequirements` (see table above).
- `ncclGin gin { devComm, ginContext }`: Creates a GIN handle for initiating remote operations. The `ginContext` parameter selects which communication channel to use.
- `devComm.ginContextCount`: The number of available GIN contexts. Multiple contexts allow parallel operations across different communication channels.
- `gin.put(team, peer, dest_window, dest_offset, src_window, src_offset, size, signal)`: Initiates a one-sided **PUT** to rank `peer` in the communicator; in this example the loop includes all ranks, including self
- `gin.waitSignal(scope, signalIndex, expectedValue)`: Waits until **this rank's** local signal reaches the threshold—i.e. enough **inbound** signaled **PUTs** from peers have completed (remote `ncclGin_WeakSignalInc` adds one completion increment on the **destination's** counter)
- `gin.flush(scope)`: Completes local consumption of pending GIN operations this context issued (e.g. safe to reuse source buffers per API semantics)
- `gin.readSignal(signalIndex)`: Reads the current signal value to establish a baseline for waiting

**Communication Pattern:**
- Each GPU sends its full token data to all peer GPUs
- Data is placed in the receive window in **peer-major** order: contiguous floats are **token0_peer0, token1_peer0, …, token_{B-1}_peer0**, then **token0_peer1, token1_peer1, …** (for each peer, all `B` local tokens in order; then the next peer’s block). Offset `peer·H·B + b·H + j` is peer `peer`, local token `b`, dimension `j` (`H = hidden_dim`, `B = tokens_per_gpu`). The reduction uses `my_token_data[j + peer·H·B]` from token `b`’s base—**not** interleaving peers per token (e.g. not token0_peer0, token0_peer1, token1_peer0, …).
- Threads within a block divide the work: each thread handles `ceil(nRanks / blockDim.x)` peers
- Signal-based synchronization ensures every **inbound** signaled **PUT** for this block is accounted for before reduction

**GIN signals (per thread block):**
- **`signalIndex = blockIdx.x`**: Each token block uses its own signal slot so different blocks in the same kernel do not share one counter.
- **`gin.readSignal(signalIndex)`** before Phase 1: Captures the baseline `signalValue` for this slot immediately before this block issues any **PUT** (`gin.put`) in the current launch. Later waits are expressed relative to that baseline.
- **`ncclGin_WeakSignalInc{signalIndex}`** on each **PUT** (`gin.put`): **Remote action** on the **peer**: their **PUT** **to you** adds one completion increment to **this rank's** signal `signalIndex` when the transfer is ordered per the API. So **your** counter rises once per peer that issues such a **PUT** with that signal (not when **you** finish sending to them).
- **After Phase 1**: `gin.waitSignal(..., signalValue + nRanks)` waits until **each** peer has completed one signaled **PUT** with its partial for **`token_idx`** (this rank and block’s token) **into this rank's** `window_recv`.
- **Phase 3**: Issues another **nRanks** outbound **PUTs** (`gin.put`) with `ncclGin_WeakSignalInc{signalIndex}`; peers’ signals rise as they receive. The per-block signal is **not** reset. **`gin.waitSignal(..., signalValue + 2 * nRanks)`** waits until **each** peer has completed the signaled Phase 3 **PUT** into **this** rank’s `window_send` (all-gather inbound round).
- **`gin.flush`**: After each wait, completes **this rank’s** outbound GIN operations so **`window_send`** (Phase 1) and **`window_recv`** (Phase 3) regions used as **PUT sources** can be safely reused; barriers still order cross-rank phases.

##### Phase 2: RMS Normalization

After the Phase 1 reduction, threads in the block synchronize with `coop.sync()` (the same `ncclCoopCta` instance passed to `gin.waitSignal`, `flush`, and `bar.sync`) before normalizing:

```cuda
coop.sync();

//============================================================================
// Phase 2: RMS Normalization
//============================================================================
// Normalize the reduced token data using block-level RMSNorm.
// All threads in the block collaborate to compute RMS and apply normalization.
//----------------------------------------------------------------------------
blockRMSNorm(my_token_data, hidden_dim, eps, reduction_buffer, coop);
```

The `blockRMSNorm()` function implements a block-level parallel reduction (identical to the LSA implementation):

1. **Thread-level accumulation**: Each thread computes partial sum of squares
2. **Block-level reduction**: Parallel reduction across all threads using shared memory
3. **Apply normalization**: All threads apply the computed RMS factor

##### Phase 3: All-Gather via GPU-Initiated Networking (GIN) PUT

The normalized results are written back to all peer GPUs using remote **PUT** (`gin.put`):

```cuda
// Release: publish normalization writes before Phase 3 PUTs
bar.sync(coop, cuda::memory_order_release, ncclGinFenceLevel::None);

size_t final_token_offset = (token_idx * hidden_dim) * sizeof(float);
my_window_offset = (blockIdx.x * hidden_dim) * sizeof(float);

for (int peer = threadIdx.x; peer < nRanks; peer += blockDim.x) {
  // PUT: send normalized data to peer's send window
  gin.put(ncclTeamWorld(devComm), peer, window_send, final_token_offset,
          window_recv, my_window_offset, sizeof(float) * hidden_dim,
          ncclGin_WeakSignalInc{signalIndex});
}

// Wait until every peer has completed its signaled PUT for Phase 3 all-gather into
// this rank's window_send (each peer sends its normalized slice where we expect it).
gin.waitSignal(coop, signalIndex, signalValue + 2 * devComm.nRanks);
// Flush outbound Phase 3 PUTs so window_recv staging used as source is consumed.
gin.flush(coop);
bar.sync(coop, cuda::memory_order_release, ncclGinFenceLevel::None);
```

**Synchronization Strategy:**
- **Initial barrier** (`cuda::memory_order_acquire`, `ncclGinFenceLevel::None`): Before Phase 1, aligns the per-block signal baseline before issuing **PUTs** (`gin.put`).
- **Phase 1 signal/flush**: **`waitSignal`** accounts for **inbound** signaled **PUTs** for **`token_idx`**; **`flush`** finishes **outbound** GIN so **PUT sources** are reusable.
- **Barrier before Phase 3** (`cuda::memory_order_release`, `ncclGinFenceLevel::None`): Participates in the world-team barrier so each rank **releases** its Phase 2 stores before any rank starts Phase 3 **PUTs**; keeps cross-GPU phase ordering consistent with cumulative GIN signals.
- **Signal accumulation (ties to `readSignal` / `waitSignal` above)**: The per-block counter is not reset between Phase 1 and Phase 3. After Phase 1 **`waitSignal`**, the signal has advanced by `nRanks` from the baseline (one **inbound** signaled **PUT** per peer for **`token_idx`**). After Phase 3 **`waitSignal`**, by another `nRanks` (second **inbound** all-gather round), so the final threshold is `signalValue + 2 * nRanks`.
- **Barrier after Phase 3** (`cuda::memory_order_release`): World-team barrier so all ranks finish Phase 3 (GIN + visibility) before the kernel returns.

##### Memory Layout

Host buffers `h_data` and `h_data_validation` are allocated and initialized in Step 2 of the Host Program Setup Flow. The GIN example uses two symmetric device memory windows:

1. **Send Window** (`window_send`): Size = `sequence_length * hidden_size * sizeof(float)`
   - Registered with `ncclCommWindowRegister()` for GIN access
   - Contains the input data initially
   - Receives the final normalized results after Phase 3
   - Each GPU can write to any peer's send window using **PUT** (`gin.put()`)

2. **Receive Window** (`window_recv`): Size = `sequence_length * hidden_size * sizeof(float)`
   - Registered with `ncclCommWindowRegister()` for GIN access
   - Used as staging area for intermediate results
   - After Phase 1 **PUT** exchanges, contains contributions from all peers in a strided layout
   - **Critical**: After the reduction step, each GPU stores the summed result in the **first part** of its assigned token sections (offsets `0` to `tokens_per_gpu * hidden_dim`)
   - The reduction overwrites the first peer's contribution with the sum across all peers
   - Phase 2 normalization operates on this compacted first section
   - Phase 3 reads from the first section to broadcast normalized results

The receive window's strided layout **after Phase 1 PUT** can be represented as (shown for **rank 0**):

```math
\mathbf{W}_{\text{recv}}^0 = \left[\begin{array}{c}
\text{Token 0 from GPU 0} \\
\text{Token 1 from GPU 0} \\
\vdots \\
\text{Token } (L/N-1) \text{ from GPU 0} \\[10pt]
\text{Token 0 from GPU 1} \\
\text{Token 1 from GPU 1} \\
\vdots \\
\text{Token } (L/N-1) \text{ from GPU 1} \\[10pt]
\vdots \\[10pt]
\text{Token 0 from GPU } (N-1) \\
\text{Token 1 from GPU } (N-1) \\
\vdots \\
\text{Token } (L/N-1) \text{ from GPU } (N-1)
\end{array}\right] \in \mathbb{R}^{L \times H}
```

Each "Token $t$ from GPU $p$" block contains $H$ elements. Note that "Token $t$" refers to the local token index on GPU $p$ (i.e., `blockIdx.x = t`), which corresponds to global token $`p \cdot \frac{L}{N} + t`$. For rank 0, the tokens in the first section are global tokens 0 through $`\frac{L}{N} - 1`$. Other ranks have the same structure but process different global tokens.

**Reduction into the First Section**: After receiving all contributions, each GPU performs the reduction **in-place** by summing across peer contributions and storing the results in the **first part** of the buffer (offsets `0` to `tokens_per_gpu * hidden_dim - 1`). Specifically:
- GPU with rank $`r`$ sums contributions for its assigned tokens, where *k* is a global token index satisfying $`r \cdot \frac{L}{N} \leq k \leq (r+1) \cdot \frac{L}{N} - 1`$.
- For block $`b`$ processing token $`t = r \cdot \frac{L}{N} + b`$:
  - Reads from positions: `my_token_data[j + i * hidden_dim * gridDim.x]` for peer $`i \in \{0, 1, \ldots, N-1\}`$
  - Writes sum to: `my_token_data[j]` (the first section, overwriting peer 0's contribution)
- After reduction, only the first `tokens_per_gpu * hidden_dim` elements contain valid summed data
- The remaining space (contributions from peers 1 through $N-1$) becomes unused scratch space

This in-place reduction strategy:
- Provides a contiguous layout for Phase 2 (normalization)
- Simplifies pointer arithmetic for Phase 3 (distribution)

##### Performance Characteristics

**Advantages of GIN:**
- **Zero CPU Involvement**: GPU kernel directly initiates remote transfers
- **Scalability**: Efficient across multiple nodes with high-speed interconnects
- **RDMA**: Leverages RDMA capabilities of modern interconnects
- **Multiple Contexts**: Uses multiple GIN contexts to spread blocks across communication channels, improving parallel throughput

**Thread-level Parallelism:**
- Each of the 256 threads processes `hidden_dim / 256` elements (4 elements for hidden_dim=1024)
- Threads also divide communication work: each thread handles `ceil(nRanks / 256)` peers
- Coalesced memory access pattern for both computation and data transfers

**Block-level Parallelism:**
- Each block operates independently on one token
- `tokens_per_gpu` blocks execute concurrently (limited by GPU resources)
- Per-block signals enable independent progress tracking
- Blocks are distributed across GIN contexts for parallel communication utilization

### [04_rmsnorm_hybrid](04_rmsnorm_hybrid/)

**Hybrid LSA/GIN RMSNorm Implementation**

This example demonstrates fused computation and communication using a hybrid approach that combines both Load Store Accessible (LSA) for intra-node communication and GPU-Initiated Networking (GIN) for inter-node communication. This pattern is ideal for multi-node systems where GPUs within a node can leverage fast NVLink (LSA) while communicating across nodes via GIN.

**Key Features:**
- Uses `ncclBarrierSession` with `ncclTeamTagWorld()` and `ncclGin` for world-team barriers (`bar.sync`) and the LSA sub-barrier (`bar.lsaBarrier()`); both pair with `barrierCount` on the host
- Leverages `ncclGetLsaPointer` for local peer memory access within the same node
- Issues remote **PUT** to peers on other nodes via `gin.put()`
- Optimally selects communication mechanism based on peer location

#### Implementation Details

The Hybrid example implements the complete distributed RMSNorm operation within a single GPU kernel using both NCCL's LSA and GIN APIs. Here's how each phase is implemented:

##### Kernel Configuration

The kernel is launched with:
- **Grid Dimensions**: `tokens_per_gpu` blocks (each GPU processes a subset of tokens)
- **Block Dimensions**: 256 threads per block
- **Shared Memory**: Dynamic allocation for block-level reductions (one float per thread)

```cuda
const size_t shared_mem_size = threads_per_block * sizeof(float);
RMSNormHybrid<<<tokens_per_gpu, threads_per_block, shared_mem_size, stream>>>(
    window_send, window_recv, devComm, sequence_length, hidden_size, eps);
```

Host code creates the device communicator with `barrierCount` (see table above):

```cuda
ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
reqs.ginConnectionType = NCCL_GIN_CONNECTION_FULL;
reqs.barrierCount = tokens_per_gpu;
reqs.ginSignalCount = tokens_per_gpu;
NCCLCHECK(ncclDevCommCreate(comm, &reqs, &devComm));
```

##### Phase 1: Reduce-Scatter via Hybrid LSA/GIN

The same block/token mapping, offsets (`token_idx`, `peer_token_idx`), and in-place reduction as the pure GIN example apply here; see that section for the layout. **Transport is split by peer kind:** remote (non-LSA) peers use **PUT** (`gin.put`) with `ncclGin_WeakSignalInc`, and **LSA** peers get the same logical data via **stores** (no GIN signal). **`gin.waitSignal(signalValue + numRemotePeers)`** therefore waits only until **every remote peer** has finished its signaled **PUT** with its partial for **`token_idx`** into this rank’s `window_recv`; **`gin.flush`** completes **outbound** GIN so **`window_send`** Phase 1 sources are reusable.

For same-node LSA peers, **`bar.lsaBarrier().sync(coop, cuda::memory_order_acq_rel)`** then publishes this rank’s LSA stores and acquires same-node peers’ LSA stores before reduction.

```cuda
ncclCoopCta coop = ncclCoopCta();

// Initialize GIN context with per-block signal
// Use multiple GIN contexts to spread blocks across communication channels
int ginContext = blockIdx.x % devComm.ginContextCount;
unsigned int signalIndex = blockIdx.x;
ncclGin gin { devComm, ginContext };
uint64_t signalValue = gin.readSignal(signalIndex);

const int rank = devComm.rank;
const int nRanks = devComm.nRanks;
const int token_idx = rank * gridDim.x + blockIdx.x;

ncclTeam world = ncclTeamWorld(devComm);
ncclTeam lsa = ncclTeamLsa(devComm);
const int startLsa = world.rank - lsa.rank;
const int lsaSize = lsa.nRanks;
const int numRemotePeers = world.nRanks - lsa.nRanks;

ncclBarrierSession<ncclCoopCta> bar { coop, ncclTeamTagWorld(), gin, blockIdx.x };
bar.sync(coop, cuda::memory_order_acquire, ncclGinFenceLevel::None);

size_t my_window_offset = (token_idx * hidden_dim) * sizeof(float);

// Remote peers: PUT (peers before LSA team)
for (int peer = threadIdx.x; peer < startLsa; peer += blockDim.x) {
  const int peer_token_idx = peer * gridDim.x + blockIdx.x;
  size_t peer_window_offset = (peer_token_idx * hidden_dim) * sizeof(float);

  gin.put(ncclTeamWorld(devComm), peer, window_recv, my_window_offset,
          window_send, peer_window_offset, sizeof(float) * hidden_dim,
          ncclGin_WeakSignalInc{signalIndex});
}

// Remote peers: PUT (peers after LSA team)
for (int peer = startLsa + lsaSize + threadIdx.x; peer < nRanks; peer += blockDim.x) {
  const int peer_token_idx = peer * gridDim.x + blockIdx.x;
  size_t peer_window_offset = (peer_token_idx * hidden_dim) * sizeof(float);

  gin.put(ncclTeamWorld(devComm), peer, window_recv, my_window_offset,
          window_send, peer_window_offset, sizeof(float) * hidden_dim,
          ncclGin_WeakSignalInc{signalIndex});
}

// Send to local peers using LSA direct writes
for (size_t offset = threadIdx.x; offset < hidden_dim; offset += blockDim.x) {
  for (int lp = 0; lp < lsa.nRanks; lp++) {
    const int peer_token_idx = (lp + startLsa) * gridDim.x + blockIdx.x;
    size_t peer_window_offset = (peer_token_idx * hidden_dim) * sizeof(float);
    float* sendPtr = (float*)ncclGetLocalPointer(window_send, peer_window_offset);
    float* recvPtr = (float*)ncclGetLsaPointer(window_recv, my_window_offset, lp);
    recvPtr[offset] = sendPtr[offset];
  }
}

// Remote peers only: wait until each has completed its signaled PUT with its
// contribution for token_idx into this rank's window_recv (reduce-scatter GIN leg).
gin.waitSignal(coop, signalIndex, signalValue + numRemotePeers);
// Flush outbound GIN Phase 1 PUTs so window_send sources are safe to reuse.
gin.flush(coop);

// LSA sub-barrier: publish our same-node stores and acquire peer same-node stores.
// Remote GIN contributions are covered by the waitSignal above.
bar.lsaBarrier().sync(coop, cuda::memory_order_acq_rel);

//----------------------------------------------------------------------------
// Reduction: Sum contributions from all peers
//----------------------------------------------------------------------------
float *my_token_data = (float*)ncclGetLocalPointer(window_recv, blockIdx.x * hidden_dim * sizeof(float));

// Sum across all peer contributions for this dimension, using the first row of the
// corresponding token as a buffer, therefore skip index 0 to avoid adding twice
for (int peer = 1; peer < nRanks; peer++) {
  for (int j = threadIdx.x; j < hidden_dim; j += blockDim.x) {
    my_token_data[j] += my_token_data[j + peer * hidden_dim * gridDim.x];
  }
}
```

**Key Hybrid Communication Concepts:**

- **Team-based Peer Classification**: NCCL organizes ranks into teams based on their communication capabilities:
  - `ncclTeamWorld(devComm)`: All ranks in the communicator
  - `ncclTeamLsa(devComm)`: Ranks that can directly access each other's memory via LSA (typically single-node / single NVLink domain)
  - The hybrid approach identifies which peers are local (LSA team) vs remote (outside LSA team)

- **Dual Communication Paths**:
  - **Remote Peers** (outside LSA team): Use **PUT** (`gin.put`) with signal-based synchronization
  - **Local Peers** (within LSA team): Use direct LSA memory writes (no remote **PUT**); visibility before reduction uses the LSA sub-barrier after the Phase 1 GIN wait

- **Synchronization Strategy**:
  - `gin.waitSignal(..., signalValue + numRemotePeers)` after Phase 1: **Remote peers only**—each must finish one signaled **PUT** with its partial for **`token_idx`** into this rank’s `window_recv`. The counter rises by `numRemotePeers`, not `nRanks`; LSA does not increment it.
  - `gin.flush` after Phase 1 (and Phase 3): Completes **outbound** GIN locally so **`window_send`** / **`window_recv`** slices used as **PUT sources** can be reused.
  - `gin.waitSignal(..., signalValue + 2 * numRemotePeers)` after Phase 3: Second **remote-only** inbound round (all-gather GIN into `window_send`); LSA Phase 3 copies are fenced by `bar.sync`, not signals.
  - `bar.lsaBarrier().sync(coop, cuda::memory_order_acq_rel)` after the Phase 1 wait/flush: LSA-only synchronization so same-node stores are both published and acquired before reduction.
  - Together, remote **signal** waits, **`flush`**, and the LSA sub-barrier guarantee all contributions (remote GIN + local LSA) are available before reduction; the later world `bar.sync` calls order cross-node phases.

**Communication Pattern:**
- Each GPU classifies its peers into local (LSA) and remote (GIN) groups
- Remote peers: **PUT** (`gin.put`) with signal increments
- Local peers: Perform direct LSA memory writes
- Signal counting tracks only remote operations (`numRemotePeers` instead of `nRanks`)
- Data is placed in the receive window in the same strided layout as the pure GIN example
- Remote signal waits cover remote GIN contributions; the LSA sub-barrier covers same-node stores before reduction

**Hybrid GIN signals (per thread block):**
- **`signalIndex`** and baseline **`gin.readSignal(signalIndex)`** behave like the pure GIN example: one signal slot per `blockIdx.x`, and `signalValue` is sampled before any **PUT** (`gin.put`) in this block.
- **Only remote PUT** (`gin.put`) uses `ncclGin_WeakSignalInc{signalIndex}` as a **remote** action, adding one completion increment to **the destination rank's** signal when the **PUT** is ordered. **LSA copies do not** touch that counter—local peers are ordered via the LSA sub-barrier.
- **After Phase 1 remote GIN**: `gin.waitSignal(..., signalValue + numRemotePeers)` waits until every **remote** peer has completed one signaled **PUT** with its contribution for **`token_idx`** **into this rank's** `window_recv` (`numRemotePeers` increments only). **`gin.flush`** then reuses **`window_send`** sources used for outbound **PUTs**.
- **After Phase 1 local LSA**: **`bar.lsaBarrier().sync(coop, cuda::memory_order_acq_rel)`** publishes and acquires **LSA** writes before reduction.
- **Phase 3**: Another `numRemotePeers` remote **PUTs** (`gin.put`) with `ncclGin_WeakSignalInc`; final **`waitSignal(..., signalValue + 2 * numRemotePeers)`** waits for the **remote** all-gather round into **`window_send`**. **`flush`** consumes Phase 3 **`window_recv`** sources used for outbound **PUTs**. LSA broadcast is fenced by **`bar.sync`**, not signals.

##### Phase 2: RMS Normalization

After hybrid synchronization, each block normalizes its assigned token using the `blockRMSNorm()` device function:

```cuda
coop.sync();

//============================================================================
// Phase 2: RMS Normalization
//============================================================================
// Normalize the reduced token data using block-level RMSNorm.
// All threads in the block collaborate to compute RMS and apply normalization.
//----------------------------------------------------------------------------
blockRMSNorm(my_token_data, hidden_dim, eps, reduction_buffer, coop);
```

The normalization logic is identical to the LSA and GIN examples.

##### Phase 3: All-Gather via Hybrid LSA/GIN

The normalized results are written back to all peer GPUs using the optimal communication mechanism:

```cuda
// Release: publish normalization writes before Phase 3 PUTs / LSA writes
bar.sync(coop, cuda::memory_order_release, ncclGinFenceLevel::None);

size_t final_token_offset = (token_idx * hidden_dim) * sizeof(float);
my_window_offset = (blockIdx.x * hidden_dim) * sizeof(float);

// Remote peers: PUT (peers before LSA team)
for (int peer = threadIdx.x; peer < startLsa; peer += blockDim.x) {
  gin.put(ncclTeamWorld(devComm), peer, window_send, final_token_offset,
          window_recv, my_window_offset, sizeof(float) * hidden_dim,
          ncclGin_WeakSignalInc{signalIndex});
}

// Remote peers: PUT (peers after LSA team)
for (int peer = startLsa + lsaSize + threadIdx.x; peer < nRanks; peer += blockDim.x) {
  gin.put(ncclTeamWorld(devComm), peer, window_send, final_token_offset,
          window_recv, my_window_offset, sizeof(float) * hidden_dim,
          ncclGin_WeakSignalInc{signalIndex});
}

// Send to local peers using LSA direct writes
for (size_t offset = threadIdx.x; offset < hidden_dim; offset += blockDim.x) {
  for (int lp = 0; lp < lsa.nRanks; lp++) {
    float* sendPtr = (float*)ncclGetLsaPointer(window_send, final_token_offset, lp);
    sendPtr[offset] = my_token_data[offset];
  }
}

// Remote peers only: wait until each has completed signaled Phase 3 PUT into
// this rank's window_send (all-gather network leg). LSA writes use barriers, not signals.
gin.waitSignal(coop, signalIndex, signalValue + 2 * numRemotePeers);
// Flush outbound GIN Phase 3 PUTs so window_recv staging used as source is consumed.
gin.flush(coop);
bar.sync(coop, cuda::memory_order_release, ncclGinFenceLevel::None);
```

**Hybrid Synchronization Strategy:**
- **Initial barrier** (`cuda::memory_order_acquire`): Before Phase 1, ensures visibility of setup before reading/writing peer data.
- **GIN signal after Phase 1** (**remote only**): `waitSignal` on `signalValue + numRemotePeers` until every remote peer has completed its **PUT** for its partial for **`token_idx`**; **`flush`** for outbound GIN source reuse; then **`bar.lsaBarrier().sync(..., acq_rel)`** so local LSA stores are visible before reduction.
- **Barrier before Phase 3** (`cuda::memory_order_release`, `ncclGinFenceLevel::None`): Each rank releases its Phase 2 stores before any rank starts Phase 3; keeps cross-GPU phase ordering consistent with cumulative **remote** GIN signals.
- **GIN signal after Phase 3** (**remote only**): `waitSignal` on `signalValue + 2 * numRemotePeers`; **`flush`** after; LSA Phase 3 copies fenced by **`bar.sync`**.
- **Barrier after Phase 3** (`cuda::memory_order_release`): World-team barrier so all ranks finish Phase 3 (GIN + LSA + visibility) before kernel exit.

##### Memory Layout

Host buffers `h_data` and `h_data_validation` are allocated and initialized in Step 2 of the Host Program Setup Flow. The Hybrid example uses two symmetric device memory windows (similar to the GIN example):

1. **Send Window** (`window_send`): Size = `sequence_length * hidden_size * sizeof(float)`
   - Contains the input data initially
   - Receives the final normalized results after Phase 3
   - Accessible via both **PUT** (`gin.put` for remote peers) and LSA pointers (local peers)

2. **Receive Window** (`window_recv`): Size = `sequence_length * hidden_size * sizeof(float)`
   - Used as staging area for intermediate results
   - After Phase 1, contains contributions from all peers in a strided layout
   - After reduction, the first section contains summed results for normalization
   - Accessible via both GIN and LSA mechanisms

The memory layout and reduction strategy follow the same pattern as the pure GIN example.

##### Performance Characteristics

**Advantages of Hybrid Approach:**
- **Optimal Path Selection**: Uses the best communication mechanism for each peer
- **Intra-node Performance**: Full NVLink bandwidth via LSA for local peers
- **Resource Efficiency**: Only remote operations use GIN signals, reducing signal pressure
- **Communication Utilization**: Multiple GIN contexts spread blocks across communication channels

**Trade-offs:**
- **Code Complexity**: More complex than pure LSA or pure GIN implementations
- **Synchronization Overhead**: Requires GIN signal waits, an LSA sub-barrier after Phase 1, and world `bar.sync` for cross-phase ordering
- **Memory Requirements**: Two windows required (like GIN), not the single window of LSA

**When to Use Hybrid vs Pure Approaches:**
- **Pure LSA**: Best for single-node / single NVLink domain workloads
- **Pure GIN**: Best when all peers are remote or uniform communication pattern is preferred
- **Hybrid**: Best for multi-node systems with multiple GPUs per node, maximizing both intra-node and inter-node performance

## Building the Examples

From the `07_kernel_fusion` directory:

```bash
# Build all examples
make

# Build a specific example
make -C 01_rmsnorm_lsa
make -C 02_rmsnorm_multimem
make -C 03_rmsnorm_gin
make -C 04_rmsnorm_hybrid

# Build with MPI support for multi-node execution
make MPI=1

# Clean build artifacts
make clean
```

## Running the Examples

### Single-Node Execution (Thread-based)

```bash
# Run example 01 (LSA)
cd 01_rmsnorm_lsa
./rmsnorm_lsa

# Run example 02 (Multimem)
cd 02_rmsnorm_multimem
./rmsnorm_multimem

# Run example 03 (GIN)
cd 03_rmsnorm_gin
./rmsnorm_gin

# Run example 04 (Hybrid)
cd 04_rmsnorm_hybrid
./rmsnorm_hybrid
```

### Multi-Node Execution (MPI-based)

```bash
# Run with 2 MPI ranks (requires MPI=1 build)
cd 01_rmsnorm_lsa
mpirun -np 2 ./rmsnorm_lsa

cd 02_rmsnorm_multimem
mpirun -np 2 ./rmsnorm_multimem

cd 03_rmsnorm_gin
mpirun -np 2 ./rmsnorm_gin

cd 04_rmsnorm_hybrid
mpirun -np 2 ./rmsnorm_hybrid
```

## Performance Considerations

### Example 01 (LSA)
- **Best Performance**: On systems with NVLink/NVSwitch
- **Memory Access**: Direct peer-to-peer with low latency
- **Scalability**: Limited to single-node / single NVLink domain

### Example 02 (Multimem)
- **Best Performance**: On Hopper+ GPUs with NVLink/NVSwitch
- **Memory Access**: Hardware multicast reduce and store
- **Scalability**: Limited to single-node / single NVLink domain

### Example 03 (GIN)
- **Best Performance**: With high-speed interconnects
- **Communication**: Optimized for remote transfers
- **Scalability**: Scales to multiple nodes

### Example 04 (Hybrid)
- **Best Performance**: Multi-node systems with NVLink within nodes
- **Communication**: Hybrid approach using LSA for intra-node, GIN for inter-node
- **Scalability**: Optimal for large-scale multi-node deployments with multiple GPUs per node

## Configuration

All examples use the following default configuration:
- **Sequence Length**: 4096 tokens (must be divisible by number of GPUs)
- **Hidden Dimension**: 1024 elements per token
- **Threads per Block**: 256
- **Epsilon**: 1e-6 (for numerical stability)

These parameters can be modified in the source code to match your specific use case.

## Prerequisites

- CUDA-capable GPUs
- **NCCL 2.30 or later** (recommended; these examples target the Device API as shipped in that line). Older NCCL versions may require source changes to match their APIs and requirement structures.
- CUDA Toolkit 12.0 or later
- For Multimem example (02_rmsnorm_multimem): Compute Capability 9.0+ required (Hopper H100+)
- For multi-node: MPI implementation (OpenMPI, MPICH, etc.)

## Common Issues and Solutions

### Compilation Errors

**Issue**: Unresolved symbols related to `rms_norm_generate`, `blockRMSNorm`, or `verify_results`
- **Solution**: Ensure `rmsnorm_utils.cuh` is included and the include path is set correctly

### Runtime Errors

**Issue**: Sequence length not divisible by number of GPUs
- **Solution**: Adjust `sequence_length` in the code to be a multiple of the GPU count

**Issue**: LSA barriers failing
- **Solution**: Verify NVLink connectivity with `nvidia-smi topo -m`

**Issue**: GIN operations timing out
- **Solution**: Check interconnect connectivity and ensure NCCL_DEBUG=INFO for detailed diagnostics

**Issue**: Multimem example reports "compute capability 9.0 or higher" required
- **Solution**: Multimem requires SM 9.0+ (Hopper H100 or newer). Verify GPU with `nvidia-smi` or use the LSA example on older GPUs

## Additional Resources

- [NCCL Device API Documentation](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/device.html)
- [NCCL User Guide](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/index.html)
- [Main Examples README](../README.md)

## Contributing

When adding new examples to this directory:
1. Follow the existing naming convention (`XX_description`)
2. Reuse `rmsnorm_utils.cuh` for common operations
3. Include a detailed comment header in the source
4. Add appropriate build targets to the Makefile
5. Update this README with the new example description
