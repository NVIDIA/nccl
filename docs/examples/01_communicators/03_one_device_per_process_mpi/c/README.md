<!--
  SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0

  See LICENSE.txt for more license information
-->

# C: One Device per Process (MPI)

This is the C implementation using MPI and `ncclCommInitRank`.

## Build

From this directory:

```shell
make MPI=1 [MPI_HOME=<path-to-mpi>] [NCCL_HOME=<path-to-nccl>] [CUDA_HOME=<path-to-cuda>]
```
## Run

From this directory:

```shell
mpirun -np <num_processes> ./one_device_per_process_mpi
```

Run with NCCL debug output:

```shell
NCCL_DEBUG=INFO mpirun -np <num_processes> ./one_device_per_process_mpi
```

## Expected output

### Single Node (4 processes)
```
Starting NCCL communicator lifecycle example with 4 processes
  MPI initialized - Process 0 of 4 total processes
  Found 4 CUDA devices on this node
  MPI rank 0 assigned to CUDA device 0
Rank 0 generated NCCL unique ID for all processes
  Rank 0 received NCCL unique ID
  Rank 0 created NCCL communicator
  MPI rank 0 → NCCL rank 0/4 on GPU device 0

[Similar output for ranks 1-3]

All communicators initialized successfully! Beginning cleanup...
  Rank 0 destroyed NCCL communicator

All NCCL communicators created and cleaned up properly!
This example demonstrated the complete NCCL communicator lifecycle.
Next steps: Try running NCCL collective operations (AllReduce, etc.)
```

### Multi-node (8 processes, 2 nodes)
```
Starting NCCL communicator lifecycle example with 8 processes
  MPI initialized - Process 0 of 8 total processes
  MPI initialized - Process 1 of 8 total processes
  MPI initialized - Process 2 of 8 total processes
  MPI initialized - Process 3 of 8 total processes
  MPI initialized - Process 4 of 8 total processes
  MPI initialized - Process 5 of 8 total processes
  MPI initialized - Process 6 of 8 total processes
  MPI initialized - Process 7 of 8 total processes
...

  MPI rank 0 → NCCL rank 0/8 on GPU device 0
  MPI rank 1 → NCCL rank 1/8 on GPU device 1
  MPI rank 2 → NCCL rank 2/8 on GPU device 2
  MPI rank 3 → NCCL rank 3/8 on GPU device 3
  MPI rank 4 → NCCL rank 4/8 on GPU device 0
  MPI rank 5 → NCCL rank 5/8 on GPU device 1
  MPI rank 6 → NCCL rank 6/8 on GPU device 2
  MPI rank 7 → NCCL rank 7/8 on GPU device 3

All NCCL communicators created and cleaned up properly!
```

## Code walk-through

### Key function: `ncclCommInitRank` with MPI

Each MPI process creates its own communicator:

```c
ncclUniqueId nccl_id;

// Rank 0 generates unique ID
if (mpi_rank == 0) {
  ncclGetUniqueId(&nccl_id);
}

// Broadcast to all processes
MPI_Bcast(&nccl_id, sizeof(ncclUniqueId), MPI_BYTE, 0, MPI_COMM_WORLD);

// Each process initializes its communicator
ncclCommInitRank(&comm, mpi_size, nccl_id, mpi_rank);
```

### Multi-node GPU assignment

The example automatically determines local rank on each node:

```c
// Split by node to get local rank
MPI_Comm local_comm;
MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0,
                    MPI_INFO_NULL, &local_comm);
int local_rank;
MPI_Comm_rank(local_comm, &local_rank);

// Set GPU based on local rank
cudaSetDevice(local_rank);
```

This approach:
- Maps each process to the GPU whose index equals its node-local rank (device = local rank)
- Each node assigns GPUs starting from 0
- Works for both single-node and multi-node deployments

### Important considerations

- **Unique ID distribution**: MPI broadcasts the unique ID from rank 0 to all processes
- **Device assignment**: Uses local rank within each node to assign GPUs
- **Coordination**: MPI provides synchronization primitives for coordination
- **Cleanup**: Always finalize NCCL communicators before MPI_Finalize()
