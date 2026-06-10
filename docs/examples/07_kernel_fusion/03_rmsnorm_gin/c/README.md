<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0

  See LICENSE.txt for more license information
-->

# C: Fused RMSNorm with GIN

This is the C/CUDA implementation of fused RMSNorm using NCCL's GPU-Initiated
Networking (GIN) device API. It implements the complete distributed RMSNorm
operation within a single GPU kernel.

## Build

From this directory:

```shell
make [MPI=1] [MPI_HOME=<path-to-mpi>] [NCCL_HOME=<path-to-nccl>] [CUDA_HOME=<path-to-cuda>]
```

## Run

### When compiled for pthreads (default)

```shell
[NTHREADS=N] ./rmsnorm_gin
```

### When compiled for MPI

```shell
mpirun -np <num_processes> ./rmsnorm_gin
```

## Code walk-through

The GPU-Initiated Networking (GIN) example implements the complete distributed RMSNorm operation within a single GPU kernel using NCCL's GPU-Initiated Networking APIs. Here's how each phase is implemented:

### Kernel Configuration

The kernel is launched with:
- **Grid Dimensions**: `tokens_per_gpu` blocks (each GPU processes a subset of tokens)
- **Block Dimensions**: 256 threads per block
- **Shared Memory**: Dynamic allocation for block-level reductions (one float per thread)

```cuda
const size_t shared_mem_size = threads_per_block * sizeof(float);
RMSNormGIN<<<tokens_per_gpu, threads_per_block, shared_mem_size, stream>>>(
    window_send, window_recv, devComm, sequence_length, hidden_size, eps);
```

Host code creates the device communicator with `worldGinBarrierCount` (see the requirements table in the parent `README.md`):

```cuda
ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
reqs.ginConnectionType = NCCL_GIN_CONNECTION_FULL;
reqs.worldGinBarrierCount = tokens_per_gpu;
reqs.ginSignalCount = tokens_per_gpu;
ncclDevCommCreate(comm, &reqs, &devComm);
```

### Phase 1: Reduce-Scatter via GPU-Initiated Networking (GIN) PUT

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
- `ncclGinBarrierSession<ncclCoopCta> bar { coop, gin, ncclTeamTagWorld(), blockIdx.x }`: World-team GIN barrier session; requires `worldGinBarrierCount` in `ncclDevCommRequirements` (see the requirements table in the parent `README.md`).
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

### Phase 2: RMS Normalization

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

### Phase 3: All-Gather via GPU-Initiated Networking (GIN) PUT

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

### Memory Layout

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
