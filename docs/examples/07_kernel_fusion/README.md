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

Each example below has its own README with the language-agnostic overview and a
`c/README.md` with the build, run, and kernel code walk-through.

### [01_rmsnorm_lsa](01_rmsnorm_lsa/)

**Pure Load Store Accessible (LSA) RMSNorm** — direct peer memory access within a single-node / single NVLink domain, using `ncclGetLsaPointer` and `ncclLsaBarrierSession`. See [`01_rmsnorm_lsa/README.md`](01_rmsnorm_lsa/README.md) for purpose, requirements, and performance characteristics, and [`01_rmsnorm_lsa/c/README.md`](01_rmsnorm_lsa/c/README.md) for the kernel walk-through.

### [02_rmsnorm_multimem](02_rmsnorm_multimem/)

**Multimem RMSNorm (SM 9.0+)** — hardware multicast reduce/broadcast via a single multimem pointer (`ncclGetLsaMultimemPointer`, `multimemLoadSum`/`multimemStore`), simplifying the kernel relative to explicit peer iteration. See [`02_rmsnorm_multimem/README.md`](02_rmsnorm_multimem/README.md) and [`02_rmsnorm_multimem/c/README.md`](02_rmsnorm_multimem/c/README.md).

### [03_rmsnorm_gin](03_rmsnorm_gin/)

**Pure GPU-Initiated Networking (GIN) RMSNorm** — multi-node fusion that issues remote PUTs directly from the kernel (`ncclGin`, `gin.put`/`gin.waitSignal`) with signal-based synchronization. See [`03_rmsnorm_gin/README.md`](03_rmsnorm_gin/README.md) and [`03_rmsnorm_gin/c/README.md`](03_rmsnorm_gin/c/README.md).

### [04_rmsnorm_hybrid](04_rmsnorm_hybrid/)

**Hybrid LSA/GIN RMSNorm** — multi-node fusion that selects transport per peer: LSA writes for same-node peers, GIN PUTs for remote peers (`ncclBarrierSession`, `ncclGetLsaPointer`, `gin.put`). See [`04_rmsnorm_hybrid/README.md`](04_rmsnorm_hybrid/README.md) and [`04_rmsnorm_hybrid/c/README.md`](04_rmsnorm_hybrid/c/README.md).

## Building the Examples

From the `07_kernel_fusion` directory:

```bash
# Build all examples
make

# Build a specific example
make -C 01_rmsnorm_lsa/c
make -C 02_rmsnorm_multimem/c
make -C 03_rmsnorm_gin/c
make -C 04_rmsnorm_hybrid/c

# Build with MPI support for multi-node execution
make MPI=1

# Clean build artifacts
make clean
```

## Running the Examples

### Single-Node Execution (Thread-based)

```bash
# Run example 01 (LSA)
cd 01_rmsnorm_lsa/c
./rmsnorm_lsa

# Run example 02 (Multimem)
cd 02_rmsnorm_multimem/c
./rmsnorm_multimem

# Run example 03 (GIN)
cd 03_rmsnorm_gin/c
./rmsnorm_gin

# Run example 04 (Hybrid)
cd 04_rmsnorm_hybrid/c
./rmsnorm_hybrid
```

### Multi-Node Execution (MPI-based)

```bash
# Run with 2 MPI ranks (requires MPI=1 build)
cd 01_rmsnorm_lsa/c
mpirun -np 2 ./rmsnorm_lsa

cd 02_rmsnorm_multimem/c
mpirun -np 2 ./rmsnorm_multimem

cd 03_rmsnorm_gin/c
mpirun -np 2 ./rmsnorm_gin

cd 04_rmsnorm_hybrid/c
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
