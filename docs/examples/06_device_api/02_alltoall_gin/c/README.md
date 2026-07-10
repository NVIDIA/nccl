<!--
  SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0

  See LICENSE.txt for more license information
-->

# C: Device API AlltoAll with GIN

This is the C/CUDA implementation using NCCL Device API with GPU-Initiated Networking (GIN).

## Build

From this directory:

```shell
make [MPI=1] [MPI_HOME=<path-to-mpi>] [NCCL_HOME=<path-to-nccl>] [CUDA_HOME=<path-to-cuda>]
```
## Run

### When compiled for pthreads (default)

From this directory:

```shell
[NTHREADS=N] ./alltoall_gin
```

### When compiled for MPI

From this directory:

```shell
mpirun -np <num_processes> ./alltoall_gin
```

## Expected output

```
Starting Pure GIN AlltoAll initialization
  Rank 0 using GPU device 0
  Rank 1 using GPU device 1
  Rank 0 initialized NCCL communicator for 2 total ranks
  Rank 1 initialized NCCL communicator for 2 total ranks
  Rank 0 initialized send data
  Rank 1 initialized send data
  Rank 0 created device communicator with GIN support
  Rank 1 created device communicator with GIN support
Starting Pure GIN AlltoAll with 1024 elements per rank (2048 total elements, 0 MB)

=== Executing Pure GIN AlltoAll ===
  Rank 0 completed pure GIN AlltoAll kernel
  Rank 1 completed pure GIN AlltoAll kernel
Pure GIN AlltoAll result: PASSED
```

## Code walk-through

### Device Communicator Creation (Host-side)

For pure GIN communication, the device communicator is configured with GIN-specific
resources. Unlike LSA-based examples, we don't need LSA barriers since all communication
goes through the network.

```cpp
ncclDevComm devComm;
ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
// GIN barriers enable cross-node synchronization over the network
reqs.railGinBarrierCount = NCCL_DEVICE_CTA_COUNT;
// GIN signals provide completion notifications for asynchronous operations
reqs.ginSignalCount = 1;
// Enable full GIN connectivity, i.e., connect each rank to all other ranks
reqs.ginConnectionType = NCCL_GIN_CONNECTION_FULL;

// Create device communicator with pure GIN support
ncclDevCommCreate(comm, &reqs, &devComm);
```

### Symmetric Memory Windows (Host-side)

The device API requires symmetric memory windows registered with
`NCCL_WIN_COLL_SYMMETRIC`. Unlike LSA which provides direct memory access, GIN
windows are accessed through network put/get operations.

```cpp
ncclWindow_t send_win, recv_win;

ncclCommWindowRegister(comm, d_sendbuff, size_bytes, &send_win, NCCL_WIN_COLL_SYMMETRIC);
ncclCommWindowRegister(comm, d_recvbuff, size_bytes, &recv_win, NCCL_WIN_COLL_SYMMETRIC);
```

### GIN Barriers (Device-side)

GIN barriers enable cross-node synchronization from device code over the network.
Each thread block uses `blockIdx.x` to select its dedicated barrier, allowing blocks
to progress independently while coordinating with corresponding blocks on other nodes.

```cpp
// GIN barriers coordinate GPU threads across different nodes over network
ncclGinBarrierSession<ncclCoopCta> bar {
    ncclCoopCta(),                    // Barrier scope: entire CTA (thread block)
    gin,                              // GIN context for network operations
    ncclTeamWorld(devComm),          // Team spanning all ranks
    devComm.railGinBarrier,          // GIN barrier handle
    blockIdx.x                       // Barrier index: matches our CTA index
};
bar.sync(ncclCoopCta(), cuda::memory_order_acquire, ncclGinFenceLevel::Relaxed);
```

### GIN Put Operations (Device-side)

GIN provides one-sided put operations for direct remote memory writes over the
network. The `ncclGin_SignalInc` parameter increments a signal counter, enabling
asynchronous completion detection.

```cpp
// Send data to all peers via GIN network operations
const size_t size = count * sizeof(T);
for (int r = tid; r < devComm.nRanks; r += nthreads) {
    gin.put(ncclTeamWorld(devComm), r,
        recvwin, recvoffset + devComm.rank * size,  // Destination: peer r's buffer
        sendwin, sendoffset + r * size,             // Source: data for peer r
        size, ncclGin_SignalInc{signalIndex});      // Signal increment for completion
}
```

### Signal-based Completion (Device-side)

GIN uses signals for asynchronous completion detection. The kernel waits for the
signal value to reach the expected count, indicating all put operations have completed.

```cpp
// Wait for all remote puts to complete
gin.waitSignal(ncclCoopCta(), signalIndex, signalValue + devComm.nRanks);
gin.flush(ncclCoopCta());  // Ensure all operations are committed
```

### Receiving CTA (Device-side)

`signalIndex` is `blockIdx.x`, so each `gin.put` increments the destination rank's signal for that sender CTA index. Only the CTA with `blockIdx.x == receivingCta` calls `waitSignal`; `receivingCta = (devComm.rank % nthreads) / blockDim.x` picks one waiting CTA per destination rank. The wait uses `signalValue + devComm.nRanks` because every rank, including self, sends one GIN put to this rank.

### Key advantages

- **Pure GPU-initiated communication**: Network operations initiated directly from GPU threads
- **No host synchronization**: Eliminates host-device round trips
- **Scalability**: Works across nodes, not limited to single-node like LSA
