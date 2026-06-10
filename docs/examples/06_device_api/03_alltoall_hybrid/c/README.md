<!--
  SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0

  See LICENSE.txt for more license information
-->

# C: Device API AlltoAll Hybrid

This is the C/CUDA implementation using NCCL Device API with hybrid LSA+GIN.

## Build

From this directory:

```shell
make [MPI=1] [MPI_HOME=<path-to-mpi>] [NCCL_HOME=<path-to-nccl>] [CUDA_HOME=<path-to-cuda>]
```
## Run

### When compiled for pthreads (default)

From this directory:

```shell
[NTHREADS=N] ./alltoall_hybrid
```

### When compiled for MPI

From this directory:

```shell
mpirun -np <num_processes> ./alltoall_hybrid
```

## Expected output

```
Starting Hybrid AlltoAll initialization
  Rank 0 using GPU device 0
  Rank 1 using GPU device 1
  Rank 2 using GPU device 2
  Rank 3 using GPU device 3
  Rank 0 initialized NCCL communicator for 4 total ranks
  Rank 1 initialized NCCL communicator for 4 total ranks
  Rank 2 initialized NCCL communicator for 4 total ranks
  Rank 3 initialized NCCL communicator for 4 total ranks
  Rank 0 initialized send data
  Rank 1 initialized send data
  Rank 2 initialized send data
  Rank 3 initialized send data
  Rank 0 created device communicator with hybrid support
  Rank 1 created device communicator with hybrid support
  Rank 2 created device communicator with hybrid support
  Rank 3 created device communicator with hybrid support
Starting Hybrid AlltoAll with 1024 elements per rank (4096 total elements, 0 MB)
Using LSA for local peers and GIN for remote peers

=== Executing Hybrid AlltoAll ===
  Rank 0 completed hybrid AlltoAll kernel
  Rank 1 completed hybrid AlltoAll kernel
  Rank 2 completed hybrid AlltoAll kernel
  Rank 3 completed hybrid AlltoAll kernel
Hybrid AlltoAll result: PASSED
✓ All 4096 elements correctly exchanged using hybrid communication
```

## Code walk-through

### Device Communicator Creation (Host-side)

For hybrid communication, the device communicator is configured with both LSA and GIN
resources. This dual setup enables optimal communication for each peer type.

```cpp
ncclDevComm devComm;
ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
// LSA barriers enable direct memory access coordination for local peers
reqs.lsaBarrierCount = NCCL_DEVICE_CTA_COUNT;
// GIN barriers enable cross-node synchronization over the network
reqs.railGinBarrierCount = NCCL_DEVICE_CTA_COUNT;
// GIN signals provide completion notifications for asynchronous network operations
reqs.ginSignalCount = 1;
// Enable full GIN connectivity, i.e., connect each rank to all other ranks
reqs.ginConnectionType = NCCL_GIN_CONNECTION_FULL;

// Create device communicator with hybrid LSA+GIN support
ncclDevCommCreate(comm, &reqs, &devComm);
```

### Symmetric Memory Windows (Host-side)

The same memory windows support both LSA direct access and GIN network operations.
The kernel selects LSA for local peers and GIN for remote peers.

```cpp
ncclWindow_t send_win, recv_win;

ncclCommWindowRegister(comm, d_sendbuff, size_bytes, &send_win, NCCL_WIN_COLL_SYMMETRIC);
ncclCommWindowRegister(comm, d_recvbuff, size_bytes, &recv_win, NCCL_WIN_COLL_SYMMETRIC);
```

### Hybrid Barriers (Device-side)

Hybrid barriers coordinate both local LSA operations and remote GIN operations.
The barrier uses the world team and GIN context to ensure synchronization across
all ranks, regardless of their communication method.

```cpp
// Hybrid barriers coordinate both LSA and GIN operations across all ranks
ncclBarrierSession<ncclCoopCta> bar {
    ncclCoopCta(),              // Barrier scope: entire CTA (thread block)
    ncclTeamTagWorld(),         // Team spanning all ranks (local + remote)
    gin,                        // GIN context for network coordination
    blockIdx.x                  // Barrier index: matches our CTA index
};
bar.sync(ncclCoopCta(), cuda::memory_order_acquire, ncclGinFenceLevel::Relaxed);
```

### Peer Classification (Device-side)

The hybrid kernel classifies peers into local (LSA-accessible) and remote (GIN-only)
categories, then uses the optimal communication method for each.

```cpp
// Classify peers into local (LSA) and remote (GIN) categories
ncclTeam world = ncclTeamWorld(devComm);  // All ranks
ncclTeam lsa = ncclTeamLsa(devComm);      // Local ranks only
const int startLsa = world.rank - lsa.rank;  // First local rank in world
const int lsaSize = lsa.nRanks;              // Number of local peers
```

### Memory Access (Device-side)

Local peers use direct load/store via `ncclGetLsaPointer`, while remote peers
use GIN put operations over the network.

```cpp
// Local peers: direct memory access (LSA)
T* sendLocal = (T*)ncclGetLocalPointer(sendwin, sendoffset);
T* recvPtr = (T*)ncclGetLsaPointer(recvwin, recvoffset, lp);

// Remote peers: network operations (GIN)
gin.put(world, r, recvwin, recvoffset + world.rank * size,
        sendwin, sendoffset + r * size, size, ncclGin_SignalInc{signalIndex});
```

### Receiving CTA (Device-side)

`signalIndex` is `blockIdx.x`, so each `gin.put` increments the destination rank's signal for that sender CTA index. Only the CTA with `blockIdx.x == receivingCta` calls `waitSignal`; `receivingCta = (world.rank % nthreads) / blockDim.x` picks one waiting CTA per destination rank (same idea as the pure GIN AlltoAll example). The wait uses `signalValue + numRemotePeers` because only **remote** peers perform GIN puts toward this rank; LSA peers are not part of that GIN put count.

### Key advantages

- **Best of both worlds**: LSA latency for local, GIN scalability for remote
- **Optimal performance**: Chooses best method based on peer locality
- **Multi-node capable**: Works across nodes using GIN for remote peers
