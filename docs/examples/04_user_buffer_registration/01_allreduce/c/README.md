<!--
  SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0

  See LICENSE.txt for more license information
-->

# C: AllReduce with User Buffer Registration

This is the C implementation using user buffer registration for optimized AllReduce.

## Build

From this directory:

```shell
make [MPI=1] [MPI_HOME=<path-to-mpi>] [NCCL_HOME=<path-to-nccl>] [CUDA_HOME=<path-to-cuda>]
```
## Run

### When compiled for pthreads (default)

```shell
[NTHREADS=N] ./allreduce_ub
```

### When compiled for MPI

```shell
mpirun -np <num_processes> ./allreduce_ub
```

## Expected output

### With 4 GPUs (using pthreads/MPI)
```
Starting AllReduce example with 4 ranks
  Rank 0 communicator initialized using device 0
  Rank 1 communicator initialized using device 1
  Rank 2 communicator initialized using device 2
  Rank 3 communicator initialized using device 3
User Buffer allocation:
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
  Rank 0 buffers deregistered
  Rank 1 buffers deregistered
  Rank 2 buffers deregistered
  Rank 3 buffers deregistered
All resources cleaned up successfully
```

## Code walk-through

### Buffer Allocation and Registration

Buffers must be allocated with `ncclMemAlloc` (or a compatible allocator) for
registration to work. Registration returns a handle used for later deregistration.

```c
size_t size_bytes; // Size of the send/receive buffers
void *d_sendbuff, *d_recvbuff;

// Allocate buffers using ncclMemAlloc on the device
ncclMemAlloc(&d_sendbuff, size_bytes);
ncclMemAlloc(&d_recvbuff, size_bytes);

ncclComm_t comm;   // Communicator from ncclCommInitRank
void *send_handle, *recv_handle;

// Register buffers with NCCL — handle is returned for deregistration
ncclCommRegister(comm, d_sendbuff, size_bytes, &send_handle);
ncclCommRegister(comm, d_recvbuff, size_bytes, &recv_handle);
```

### AllReduce with Registered Buffers

```c
size_t count;        // Number of floats to reduce
cudaStream_t stream; // From cudaStreamCreate

ncclAllReduce(d_sendbuff, d_recvbuff, count, ncclFloat, ncclSum,
              comm, stream);
```

### Deregistration and Cleanup

```c
// Deregister buffers using handles from ncclCommRegister
ncclCommDeregister(comm, send_handle);
ncclCommDeregister(comm, recv_handle);

// Free buffers allocated with ncclMemAlloc
ncclMemFree(d_sendbuff);
ncclMemFree(d_recvbuff);
```

### Performance benefits

- **Reduced overhead**: Registration amortized across multiple operations
- **Optimized transfers**: NCCL can operate directly on registered buffers (zero-copy)
- **Reusable**: Same buffer used for many operations

### When to use

- Buffers reused across many iterations (e.g., training loops)
- Performance-critical sections with repeated communication
- Large buffers where registration overhead is amortized
