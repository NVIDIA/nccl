<!-- Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

See LICENSE.txt for license information -->

# NCCL User Buffer Registration AllReduce Example

This example demonstrates how to use NCCL's user buffer registration feature to
optimize performance for repeated collective operations on the same buffers.
User Buffer Registration is a feature that allows NCCL to directly
send/receive/operate data through the user buffer without extra internal copy
(zero-copy).

## Overview

User buffer registration allows NCCL to pre-register memory buffers with
communicators, eliminating registration overhead on each operation. This is
particularly beneficial for applications that repeatedly perform collective
operations on the same memory regions, such as iterative training loops.

## What This Example Does

1. **Allocates memory using NCCL allocator** (`ncclMemAlloc`) which is provided
   by NCCL as convenience function
2. **Registers buffers with communicator** using `ncclCommRegister` for
   optimized performance
3. **Performs AllReduce sum operation** using the registered buffers for
   efficient communication

## Building and Running

The advanced examples can be built using either pthread or MPI for
parallelization. pthread is the default choice. To use MPI the user needs to
provide a valid MPI installation under `MPI_HOME`.

### Build
```shell
make [MPI=1] [MPI_HOME=<path-to-mpi>] [NCCL_HOME=<path-to-nccl>] [CUDA_HOME=<path-to-cuda>]
```

### Run when compiled for pthreads (default)
```shell
[NTHREADS=N] ./allreduce_ub
```

### Run when compiled for MPI
```shell
mpirun -np <num_processes> ./allreduce_ub
```

## Code Structure

### Key Components

1. **Buffer Allocation and Registration**:
```c
size_t size_bytes; // Is set to the size of the send/receive buffers
void *d_sendbuff;
void *d_recvbuff;

// Allocate buffers using ncclMemAlloc (or another qualified allocator) on the device
NCCLCHECK(ncclMemAlloc(&d_sendbuff, size_bytes));
NCCLCHECK(ncclMemAlloc(&d_recvbuff, size_bytes));

ncclComm_t comm;   // comms is set during ncclCommInitRank
void *send_handle;
void *recv_handle;

// Register buffers with NCCL, handle is returned for De-registration
NCCLCHECK(ncclCommRegister(comm, d_sendbuff, size_bytes, &send_handle));
NCCLCHECK(ncclCommRegister(comm, d_recvbuff, size_bytes, &recv_handle));
```

2. **AllReduce with Group Operations**:
```c
size_t count;  // set to number of floats to exchange
cudaStream_t stream; // stream is set in cudaStreamCreate

NCCLCHECK(ncclAllReduce(d_sendbuff, d_recvbuff, count, ncclFloat, ncclSum,
                        comm, stream));
```

3. **Buffer Deregistration and Cleanup**:
```c
// Deregister buffers using handle from ncclCommRegister
NCCLCHECK(ncclCommDeregister(comm, send_handle));
NCCLCHECK(ncclCommDeregister(comm, recv_handle));

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

## Performance Benefits of User Buffer Registration

User buffer registration provides several performance advantages:

1. **Reduced Overhead**: Pre-registration eliminates the need to
   register/deregister buffers for each operation
2. **Better Memory Pinning**: Registered buffers are pinned in memory,
   preventing page faults
3. **Lower Latency**: Especially beneficial for repeated operations on the same
   buffers

**Important**: Buffers must be allocated with `ncclMemAlloc` or a compatible
allocator for registration to work. See The [General Buffer Registration
](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/bufferreg.html#general-buffer-registration)
section of the user guide.

**Important**: If any rank in a communicator passes registered buffers to a NCCL
communication function, all other ranks in the same communicator must pass their
registered buffers; otherwise, mixing registered and non-registered buffers can
result in undefined behavior.

## Key Insights

- **User Buffer Registration** is most beneficial for:
  - Large data transfers
  - Repeated operations on the same buffers
  - Performance-critical applications
- **Memory management** is critical - always deregister buffers before freeing

## Common Issues and Solutions

1. **Registration Failure**: Buffers MUST be allocated with `ncclMemAlloc` or
   another qualified allocator (not `cudaMalloc`) for registration. See [Buffer
   Registration](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/bufferreg.html)
   section for details.
2. **Allocation Error**: If `ncclMemAlloc` fails, check NCCL version (requires
   2.19.x+) and available memory
3. **Deregistration Order**: Always deregister before freeing memory
4. **Handle Management**: Keep track of registration handles for proper cleanup
5. **Memory Leaks**: Always use `ncclMemFree` for buffers allocated with
   `ncclMemAlloc`
