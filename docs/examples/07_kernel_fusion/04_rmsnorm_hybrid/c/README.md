<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0

  See LICENSE.txt for more license information
-->

# C: Fused RMSNorm with Hybrid LSA/GIN

This is the C/CUDA implementation of fused RMSNorm using a hybrid of NCCL's LSA
(intra-node) and GIN (inter-node) device APIs. It implements the complete
distributed RMSNorm operation within a single GPU kernel.

## Build

From this directory:

```shell
make [MPI=1] [MPI_HOME=<path-to-mpi>] [NCCL_HOME=<path-to-nccl>] [CUDA_HOME=<path-to-cuda>]
```

## Run

### When compiled for pthreads (default)

```shell
[NTHREADS=N] ./rmsnorm_hybrid
```

### When compiled for MPI

```shell
mpirun -np <num_processes> ./rmsnorm_hybrid
```

## Code walk-through

The Hybrid example implements the complete distributed RMSNorm operation within a single GPU kernel using both NCCL's LSA and GIN APIs. Here's how each phase is implemented:

### Kernel Configuration

The kernel is launched with:
- **Grid Dimensions**: `tokens_per_gpu` blocks (each GPU processes a subset of tokens)
- **Block Dimensions**: 256 threads per block
- **Shared Memory**: Dynamic allocation for block-level reductions (one float per thread)

```cuda
const size_t shared_mem_size = threads_per_block * sizeof(float);
RMSNormHybrid<<<tokens_per_gpu, threads_per_block, shared_mem_size, stream>>>(
    window_send, window_recv, devComm, sequence_length, hidden_size, eps);
```

Host code creates the device communicator with `barrierCount` (see the requirements table in the parent `README.md`):

```cuda
ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
reqs.ginConnectionType = NCCL_GIN_CONNECTION_FULL;
reqs.barrierCount = tokens_per_gpu;
reqs.ginSignalCount = tokens_per_gpu;
ncclDevCommCreate(comm, &reqs, &devComm);
```

### Phase 1: Reduce-Scatter via Hybrid LSA/GIN

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

### Phase 2: RMS Normalization

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

### Phase 3: All-Gather via Hybrid LSA/GIN

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

### Memory Layout

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
