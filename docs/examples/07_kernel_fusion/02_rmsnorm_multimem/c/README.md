<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0

  See LICENSE.txt for more license information
-->

# C: Fused RMSNorm with Multimem

This is the C/CUDA implementation of fused RMSNorm using NCCL's Multimem device
API (hardware multicast memory operations, SM 9.0+).

## Build

From this directory:

```shell
make [MPI=1] [MPI_HOME=<path-to-mpi>] [NCCL_HOME=<path-to-nccl>] [CUDA_HOME=<path-to-cuda>]
```

## Run

### When compiled for pthreads (default)

```shell
[NTHREADS=N] ./rmsnorm_multimem
```

### When compiled for MPI

```shell
mpirun -np <num_processes> ./rmsnorm_multimem
```

## Code walk-through

### Phase 1: Reduce-Scatter via Multimem

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

### Phase 2: RMS Normalization

Identical to the LSA example — `blockRMSNorm()` on the local reduced data:

```cuda
blockRMSNorm(local_pointer, hidden_dim, eps, reduction_buffer, coop);
```

### Phase 3: All-Gather via Multimem

Broadcast normalized results to all peers using `multimemStore`:

```cuda
for (int i = threadIdx.x; i < hidden_dim; i += blockDim.x) {
    float val = local_pointer[i];  // Read once from local
    multimemStore(multimem_pointer + i, val);
}
bar.sync(coop, cuda::memory_order_release);
```

**Synchronization Strategy:** Same as LSA — initial barrier uses `cuda::memory_order_acquire` before Phase 1; final barrier uses `cuda::memory_order_release` after Phase 3.
