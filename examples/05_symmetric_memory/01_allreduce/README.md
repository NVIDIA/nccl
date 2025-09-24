<!-- Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

See LICENSE.txt for license information -->

# NCCL Symmetric Memory AllReduce Example

This example demonstrates how to use NCCL's symmetric memory feature for
optimized collective operations. Symmetric memory provides optimized performance
by leveraging consistent memory layouts across all participating ranks, enabling
advanced communication algorithms.

## Overview

Symmetric memory windows provide a way to register memory buffers that benefit
from optimized collective operations. When using `NCCL_WIN_COLL_SYMMETRIC`, all
ranks must provide symmetric buffers, enabling optimized communication patterns
and better performance for large-scale multi-GPU operations.

## What This Example Does

1. **Allocates memory using NCCL allocator** (`ncclMemAlloc`) which provides
   memory compatible with symmetric windows
2. **Registers buffers as symmetric windows** using `ncclCommWindowRegister`
   with `NCCL_WIN_COLL_SYMMETRIC` flag
3. **Performs AllReduce sum operation** using the symmetric memory for optimized
   communication performance

## Building and Running

The advanced examples can be built using either pthread or MPI for
parallelization. pthread is the default choice. To use MPI the user needs to set
`MPI=1` at build time and can optionally provide a valid MPI installation under
`MPI_HOME`.

### Build
```shell
make [MPI=1] [MPI_HOME=<path-to-mpi>] [NCCL_HOME=<path-to-nccl>] [CUDA_HOME=<path-to-cuda>]
```

### Run when compiled for pthreads (default)
```shell
[NTHREADS=N] ./allreduce_sm
```

### Run when compiled for MPI
```shell
mpirun -np <num_processes> ./allreduce_sm
```

## Code Structure

### Key Components

1. **Buffer Allocation and Window Registration**:
```c
size_t size_bytes; // Is set to the size of the send/receive buffers
void *d_sendbuff;
void *d_recvbuff;

// Allocate buffers using ncclMemAlloc (compatible with symmetric memory)
NCCLCHECK(ncclMemAlloc(&d_sendbuff, size_bytes));
NCCLCHECK(ncclMemAlloc(&d_recvbuff, size_bytes));

ncclComm_t comm;
ncclWindow_t send_win;
ncclWindow_t recv_win;

// Register buffers as symmetric windows
NCCLCHECK(ncclCommWindowRegister(comm, d_sendbuff, size_bytes, &send_win, NCCL_WIN_COLL_SYMMETRIC));
NCCLCHECK(ncclCommWindowRegister(comm, d_recvbuff, size_bytes, &recv_win, NCCL_WIN_COLL_SYMMETRIC));
```

2. **AllReduce Operation**:
```c
size_t count;  // set to number of floats to exchange
cudaStream_t stream; // stream is set in cudaStreamCreate

// Perform AllReduce with symmetric memory optimization
NCCLCHECK(ncclAllReduce(d_sendbuff, d_recvbuff, count, ncclFloat, ncclSum,
                        comm, stream));
```

3. **Window Deregistration and Cleanup**:
```c
// Deregister symmetric memory windows
NCCLCHECK(ncclCommWindowDeregister(comm, send_win));
NCCLCHECK(ncclCommWindowDeregister(comm, recv_win));

// Free buffers allocated with ncclMemAlloc
NCCLCHECK(ncclMemFree(d_sendbuff));
NCCLCHECK(ncclMemFree(d_recvbuff));
```

## Expected Output

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

## Performance Benefits of Symmetric Memory

Symmetric memory registration provides several performance advantages:

- **Optimized Communication Algorithms**: NCCL can apply advanced optimizations
  when all ranks have symmetric layouts
- **Better Memory Access Patterns**: Consistent layouts enable better caching
  and memory access optimization

For more information on the performance benefits see the [Enabling Fast
Inference and Resilient Training with NCCL
2.27]()https://developer.nvidia.com/blog/enabling-fast-inference-and-resilient-training-with-nccl-2-27/)
blog.

**Important**: Buffers must be allocated using the CUDA Virtual Memory
Management (VMM) API. NCCL provides the `ncclMemAlloc` convenience function for
symmetric memory registration. The `NCCL_WIN_COLL_SYMMETRIC` flag requires all
ranks to provide symmetric buffers consistently.

## Key Insights

- **Symmetric Memory Windows** are most beneficial for:
  - Large-scale collective operations with consistent memory patterns
  - Latency-sensitive kernels
  - Applications with predictable allocation patterns
- **ncclCommInitRank** can be used for pthread or MPI parallel case
- **Window registration** must happen on all ranks for collective operations
- **Memory management** is critical - always deregister windows before freeing
  memory

## Common Issues and Solutions

1. **Window Registration Failure**: Buffers MUST be allocated with (VMM) API,
   e.g. `ncclMemAlloc` (not `cudaMalloc`) for symmetric memory.
2. **Allocation Error**: If `ncclMemAlloc` fails, check NCCL version (requires
   at least v2.27) and available memory
3. **Deregistration Order**: Always deregister windows before freeing memory or
   destroying communicators
4. **Symmetric Requirement**: All ranks must use `NCCL_WIN_COLL_SYMMETRIC`
   consistently in collective operations
5. **Memory Leaks**: Always use `ncclMemFree` for buffers allocated with
   `ncclMemAlloc`
