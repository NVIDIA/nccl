<!--
  SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0

  See LICENSE.txt for more license information
-->

# C: Device API AllReduce with LSA

This is the C/CUDA implementation using NCCL Device API with Load Store Accessible (LSA) barriers.

## Build

The advanced examples can be built using either pthread or MPI for parallelization. pthread is the default choice.

From this directory:

```shell
make [MPI=1] [MPI_HOME=<path-to-mpi>] [NCCL_HOME=<path-to-nccl>] [CUDA_HOME=<path-to-cuda>]
```
## Run

### When compiled for pthreads (default)

From this directory:

```shell
[NTHREADS=N] ./allreduce_lsa
```

### When compiled for MPI

From this directory:

```shell
mpirun -np <num_processes> ./allreduce_lsa
```

Run with NCCL debug output:

```shell
NCCL_DEBUG=INFO ./allreduce_lsa
```

## Expected output

```
Starting Device API AllReduce initialization
  Rank 0 using GPU device 0
  Rank 1 using GPU device 1
  Rank 2 using GPU device 2
  Rank 3 using GPU device 3
  Rank 0 initialized NCCL communicator for 4 total ranks
  Rank 1 initialized NCCL communicator for 4 total ranks
  Rank 2 initialized NCCL communicator for 4 total ranks
  Rank 3 initialized NCCL communicator for 4 total ranks
  Rank 0 initialized data with value 0
  Rank 1 initialized data with value 1
  Rank 2 initialized data with value 2
  Rank 3 initialized data with value 3
  Rank 0 created device communicator with 16 LSA barriers
  Rank 1 created device communicator with 16 LSA barriers
  Rank 2 created device communicator with 16 LSA barriers
  Rank 3 created device communicator with 16 LSA barriers
Starting AllReduce with 1048576 elements (4 MB) using Device API
Expected result: sum of ranks 0 to 3 = 6 per element
  Rank 0 completed AllReduce kernel execution
  Rank 1 completed AllReduce kernel execution
  Rank 2 completed AllReduce kernel execution
  Rank 3 completed AllReduce kernel execution
AllReduce completed. Result verification: PASSED
All elements correctly sum to 6 (ranks 0-3)
```

## Code walk-through

### Device Communicator Creation

The NCCL Device API enables GPU kernels to perform inter-GPU communication directly:

```c
ncclDevComm devComm;
ncclDevCommRequirements requirements = {0};
requirements.lsaBarrierCount = numBlocks;  // One barrier per thread block

// Create device communicator
ncclDevCommCreate(&devComm, comm, &requirements);
```

### Symmetric Memory Windows (Host-side)

The device API requires symmetric memory windows registered with
`NCCL_WIN_COLL_SYMMETRIC`. See the [symmetric memory
example](../../05_symmetric_memory/) for allocation and requirements details.

```c
ncclWindow_t send_win, recv_win;

// Register symmetric windows for device-side peer access
ncclCommWindowRegister(comm, d_sendbuff, size_bytes, &send_win, NCCL_WIN_COLL_SYMMETRIC);
ncclCommWindowRegister(comm, d_recvbuff, size_bytes, &recv_win, NCCL_WIN_COLL_SYMMETRIC);
```

### LSA Barriers (Device-side)

LSA barriers enable cross-GPU synchronization from device code. Each thread
block uses `blockIdx.x` to select its dedicated barrier, allowing blocks to
progress independently while coordinating with corresponding blocks on other
GPUs.

```cpp
// Create a barrier session scoped to this CTA (thread block).
// Each block gets its own barrier (indexed by blockIdx.x), so blocks
// synchronize independently with their counterparts on other GPUs.
ncclLsaBarrierSession<ncclCoopCta> bar {
    ncclCoopCta(),           // Barrier scope: entire CTA (thread block)
    devComm, ncclTeamLsa(devComm), devComm.lsaBarrier,
    blockIdx.x               // Barrier index: matches our CTA index (0 to lsaBarrierCount-1)
};

// Acquire barrier — wait until all ranks reach this point
bar.sync(ncclCoopCta(), cuda::memory_order_acquire);

// ... perform peer memory reads and local accumulation ...

// Release barrier — ensures we received data from everyone before
// unblocking the stream and allowing the next kernel(s) to proceed.
// Critical for correctness in device-side collective operations.
bar.sync(ncclCoopCta(), cuda::memory_order_release);
```

### Peer Memory Access (Device-side)

`ncclGetLsaPointer` allows CUDA kernels to directly access other GPUs' memory
within the LSA team:

```cpp
// Access peer memory directly using LSA (Load/Store Accessible) pointers
float* peerPtr = (float*)ncclGetLsaPointer(sendwin, sendoffset, peer);
```

### Key advantages

- **Computation-communication fusion**: Overlap computation with communication in the same kernel
- **Reduced host-device synchronization**: No CPU involvement during collective operations
- **Lower latency**: Direct GPU-to-GPU communication without going through the host
- **Fine-grained control**: Per-thread-block synchronization with LSA barriers
