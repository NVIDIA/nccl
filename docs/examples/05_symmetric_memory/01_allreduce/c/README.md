<!--
  SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0

  See LICENSE.txt for more license information
-->

# C: AllReduce with Symmetric Memory

This is the C implementation using symmetric memory windows for optimized AllReduce.

## Build

From this directory:

```shell
make [MPI=1] [MPI_HOME=<path-to-mpi>] [NCCL_HOME=<path-to-nccl>] [CUDA_HOME=<path-to-cuda>]
```
## Run

### When compiled for pthreads (default)

```shell
[NTHREADS=N] ./allreduce_sm
```

### When compiled for MPI

```shell
mpirun -np <num_processes> ./allreduce_sm
```

## Expected output

### With 4 GPUs (using pthreads/MPI)
```
Starting AllReduce example with 4 ranks
  Rank 0 communicator initialized using device 0
  Rank 1 communicator initialized using device 1
  Rank 2 communicator initialized using device 2
  Rank 3 communicator initialized using device 3
Symmetric Memory allocation
  Rank 0 allocating 4.00 MB per buffer
  Rank 1 allocating 4.00 MB per buffer
  Rank 2 allocating 4.00 MB per buffer
  Rank 3 allocating 4.00 MB per buffer
  Rank 0 data initialized (value: 0)
  Rank 1 data initialized (value: 1)
  Rank 2 data initialized (value: 2)
  Rank 3 data initialized (value: 3)
Starting AllReduce with 1048576 elements (4 MB)
AllReduce completed successfully
Verification - Expected: 6.0, Got: 6.0
Results verified correctly
  Rank 0 symmetric memory windows deregistered
  Rank 1 symmetric memory windows deregistered
  Rank 2 symmetric memory windows deregistered
  Rank 3 symmetric memory windows deregistered
All resources cleaned up successfully
Example completed - demonstrated symmetric memory lifecycle
```

## Code walk-through

### Symmetric Memory Windows

Register symmetric memory windows for low-latency peer access:

```c
void* window;
size_t buffer_size = count * sizeof(float);

// Register symmetric memory window for LSA access
ncclCommWindowRegister(comm, &window, buffer, buffer_size,
                       ncclCommWindowNone, ncclCommWindowLSA);

// Perform AllReduce with symmetric memory optimization
ncclAllReduce(sendbuff, recvbuff, count, ncclFloat, ncclSum,
              comm, stream);

// Deregister when done
ncclCommWindowDeregister(comm, window);
```

### Performance benefits

- **Direct peer access**: GPUs can directly access peer memory
- **Reduced latency**: Eliminates intermediate copies and protocol overhead
- **LSA (Load-Store Accessible)**: Memory accessible via direct load/store operations

### When to use

- Single-node scenarios where all GPUs can directly access each other
- Latency-sensitive applications
- Repeated operations on the same buffers
- When using NCCL Device API for kernel-level communication
