<!-- Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

See LICENSE.txt for license information -->

# NCCL Example: One Device per Process (MPI)

This example demonstrates NCCL communicator lifecycle management using MPI, with
one GPU per MPI process.

## Overview

This example shows one of the most common NCCL deployment pattern: one GPU
device per process. This approach is ideal for distributed training across
multiple nodes and provides the foundation for scalable multi-GPU applications.
MPI is used as it provides a parallel launcher and broadcast functions. It is,
however, not a requirement for multi-node NCCL applications.

Other approaches use server-client models or spawn parallel processes using
sockets. NCCL only requires that the unique ID is distributed among each
thread/process taking part in collective communication and all threads/processes
call some NCCL initialization function.

## What This Example Does

1. **Multi-node Support**:
   - Determines local rank on each node automatically
   - Maps MPI processes to GPUs on each node
   - Uses `MPI_Comm_split_type` with `MPI_COMM_TYPE_SHARED` to assign each local
     rank a GPU.

2. **Communicator Creation**:
   - Uses `ncclCommInitRank` with MPI-coordinated unique ID
   - Rank `0` generates and broadcasts NCCL unique ID
   - Each process joins the distributed communicator

3. **Verification**:
   - Displays MPI rank → NCCL rank → GPU device mapping
   - Confirms successful initialization across all processes

4. **Cleanup**:
   - Proper resource cleanup order
   - MPI synchronization for clean shutdown

## Building and Running

### Build
```shell
make MPI=1 [MPI_HOME=<path-to-mpi>] [NCCL_HOME=<path-to-nccl>] [CUDA_HOME=<path-to-cuda>]
```

### Run example
```shell
mpirun -np <num_processes> ./one_device_per_process_mpi
```

### Run with NCCL debug output
```shell
NCCL_DEBUG=INFO mpirun -np <num_processes> ./one_device_per_process_mpi
```

## Code Walk-through

This approach:
- Automatically handles multi-node GPU assignment
- Uses MPI for coordination and NCCL for GPU communication
- Supports both single-node and multi-node deployments

### Unique ID Distribution
The NCCL unique ID must be shared with all process which call `ncclCommInitRank`. We use MPI for that:
```c
// Rank 0 generates unique ID
if (mpi_rank == 0) {
    NCCLCHECK(ncclGetUniqueId(&nccl_id));
}

// Broadcast to all processes
MPI_Bcast(&nccl_id, sizeof(ncclUniqueId), MPI_BYTE, 0, MPI_COMM_WORLD);
```

### Key Function: Multi-node GPU assignment
```c
// Separate function to determine the node local rank via `MPI_Comm_split_type`
int local_rank = getLocalRank(MPI_COMM_WORLD);

// Use the local rank as the GPU device number. This assumes you only start as many processes as available GPUs
CUDACHECK(cudaSetDevice(local_rank));

ncclComm_t comm;
int mpi_rank, mpi_size; // mpi_rank & mpi_size are set during MPI initialization
ncclUniqueId nccl_id; // nccl_id is generated and broadcasted as above

// Initialize NCCL communicator across all processes
NCCLCHECK(ncclCommInitRank(&comm, mpi_size, nccl_id, mpi_rank));
```

## Expected Output

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

## When to Use MPI Approach

### Ideal Use Cases
- **Multi-node clusters**: Scales across multiple machines
- **Production deployments**: Industry standard for distributed training,
  inference, and most HPC codes
- **Process isolation**: Each GPU in separate process for robustness
- **Large scale**: Supports thousands of GPUs

### When NOT to Use
- **Single-node testing**: Simpler approaches available
- **No MPI available**: Some environments don't support MPI
- **Shared memory needs**: Single-process approaches may be simpler

## Performance Considerations

- **Advantages**:
  - MPI has been optimized for large parallel startup.
  - Industry standard deployment pattern
  - Optimal for large-scale training

- **Disadvantages**:
  - MPI setup complexity
  - Inter-process communication overhead
  - Requires MPI runtime environment

## Common Issues and Solutions

1. **More MPI processes than GPUs on a node**:
   - The example reports an error if local rank exceeds available devices
   - Use fewer processes per node or more GPUs

2. **MPI broadcast hangs**:
   - Ensure all ranks participate in collective operations
   - Check MPI installation and network connectivity

3. **Multi-node communication fails**:
   - Check firewall settings and network configuration
   - Set `NCCL_SOCKET_IFNAME` to specify network interface

## Error Handling

The example uses simplified error handling with CHECK macros:
- **CUDACHECK**: Exits immediately on CUDA errors
- **NCCLCHECK**: Exits immediately on NCCL errors
- **No async error checking**: Simplified for clarity
- **No global error coordination**: Each process exits on its own errors

## Next Steps

After understanding this example:
1. Try running collective operations (AllReduce, AllGather, etc.)
2. Experiment with multi-node deployments
