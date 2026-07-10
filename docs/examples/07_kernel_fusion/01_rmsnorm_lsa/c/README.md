<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0

  See LICENSE.txt for more license information
-->

# C: Fused RMSNorm with LSA

This is the C/CUDA implementation of fused RMSNorm using NCCL's Load Store
Accessible (LSA) device API. It implements the complete distributed RMSNorm
operation within a single GPU kernel.

## Build

From this directory:

```shell
make [MPI=1] [MPI_HOME=<path-to-mpi>] [NCCL_HOME=<path-to-nccl>] [CUDA_HOME=<path-to-cuda>]
```

## Run

### When compiled for pthreads (default)

```shell
[NTHREADS=N] ./rmsnorm_lsa
```

### When compiled for MPI

```shell
mpirun -np <num_processes> ./rmsnorm_lsa
```

## Code walk-through

This example implements the complete distributed RMSNorm operation within a single GPU kernel using NCCL's LSA APIs. Here's how each phase is implemented:

### Kernel Configuration

The kernel is launched with:
- **Grid Dimensions**: `tokens_per_gpu` blocks (each GPU processes a subset of tokens)
- **Block Dimensions**: 256 threads per block
- **Shared Memory**: Dynamic allocation for block-level reductions (one float per thread)

```cuda
const size_t shared_mem_size = threads_per_block * sizeof(float);
RMSNormLSA<<<tokens_per_gpu, threads_per_block, shared_mem_size, stream>>>(
    window, devComm, sequence_length, hidden_size, eps);
```

### Phase 1: Reduce-Scatter via LSA

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

### Phase 2: RMS Normalization

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

### Phase 3: All-Gather via LSA

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

### Memory Layout

Host buffers `h_data` and `h_data_validation` are allocated and initialized in Step 2 of the Host Program Setup Flow. The device layout uses a single allocation:

1. **Symmetric Window** (`d_data`): Size = `sequence_length * hidden_size * sizeof(float)`
   - Registered with `ncclCommWindowRegister()` for LSA access
   - Contains the full tensor (all tokens across all dimensions)
   - Each GPU can access any part of any peer's window
   - This is where input data resides and final results are written
   - The kernel uses `ncclGetLocalPointer()` to obtain a pointer for in-place reduction and normalization

For a system with $N$ GPUs and sequence length $L$, each GPU's symmetric window holds all $L$ tokens (initially the partial contributions $\mathbf{X}^r$, where $r$ is the rank).
