<!-- Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

See LICENSE.txt for license information -->

# NCCL Example: One Device per Thread (pthread)

This example demonstrates NCCL communicator lifecycle management using pthreads, with one GPU per
thread.

## Overview

This example shows how to use NCCL in a multi-threaded environment where each pthread manages one
GPU device. It demonstrates the proper initialization and cleanup sequence for NCCL communicators
within threads.

## What This Example Does

1. **Thread Creation**:
   - Creates one pthread per available GPU or `NTHREADS` if set
   - Each thread manages its own CUDA device context

2. **Communicator Creation**:
   - Uses `ncclCommInitRank` with unique ID across threads
   - Each thread initializes its own communicator
   - Demonstrates thread-safe NCCL initialization

3. **Verification**:
   - Queries communicator properties (rank, size, device)
   - Confirms successful initialization across all threads

4. **Cleanup**:
   - Proper resource cleanup order within each thread
   - Demonstrates correct NCCL and CUDA resource management

## Building and Running

### Build
```shell
make  [NCCL_HOME=<path-to-nccl>] [CUDA_HOME=<path-to-cuda>]
```

### Run with specific thread count (number of GPUs)
```shell
[NTHREADS=n] ./one_device_per_pthread
```

### Run with NCCL debug output
```shell
NCCL_DEBUG=INFO ./one_device_per_pthread
```

## Code Walk-through

### Key Function: ncclCommInitRank in threads
```c
// Each thread creates it's own copy of struct `threadData_t`.
typedef struct {
  int thread_id; // thread_id is set when thread is created
  int num_gpus; // num_gpus is set by querying the CUDA devices
  ncclUniqueId commId; // commId is set by ncclGetUniqueId
  ncclComm_t* comms;
} threadData_t;
threadData_t* data;

// Each thread initializes its own communicator
NCCLCHECK(ncclCommInitRank(&data->comms[thread_id], data->num_gpus, data->commId, data->thread_id));
```

In this approach:
- Each thread gets its own NCCL rank (0, 1, 2...)
- Does not need explicit distribution of `uniqueId` since it uses a global variable.

## Expected Output

```
Using 4 devices with pthreads
Creating 4 threads for NCCL communicators
  Thread 0: Set device 0 and created stream
  Thread 1: Set device 1 and created stream
  Thread 2: Set device 2 and created stream
  Thread 3: Set device 3 and created stream
  Thread 0: NCCL communicator initialized
  Thread 1: NCCL communicator initialized
  Thread 2: NCCL communicator initialized
  Thread 3: NCCL communicator initialized
All threads synchronized - communicators ready
  Thread 0: Communicator rank 0 of 4
  Thread 1: Communicator rank 1 of 4
  Thread 2: Communicator rank 2 of 4
  Thread 3: Communicator rank 3 of 4
  Thread 0: Destroyed NCCL communicator
  Thread 1: Destroyed NCCL communicator
  Thread 2: Destroyed NCCL communicator
  Thread 3: Destroyed NCCL communicator
  Thread 0: Resources cleaned up
  Thread 1: Resources cleaned up
  Thread 2: Resources cleaned up
  Thread 3: Resources cleaned up
All threads completed
Success
```

## When to Use pthread Approach

### Ideal Use Cases
- **Thread-based applications**: When your application is already threaded
- **Single-node workloads**: All GPUs on one machine
- **Shared memory**: Need to share data structures between GPU contexts

### When NOT to Use
- **Multi-node clusters**: Cannot scale beyond one node
- **Process isolation**: When GPU contexts should be isolated
- **Complex applications**: Multi-process approach may be cleaner

## Performance Considerations

- **Advantages**:
  - Shared address space between threads
  - Easier data sharing between GPU contexts
  - No MPI overhead

- **Disadvantages**:
  - Thread synchronization complexity
  - Limited to single node

## Common Issues and Solutions

1. **Thread synchronization errors**:
   - Ensure all threads use the same NCCL unique ID
   - Proper pthread synchronization (barriers, joins)

2. **CUDA context conflicts**:
   - Each thread must call `cudaSetDevice()` before CUDA operations
   - Don't share CUDA streams between threads

3. **Resource cleanup order**:
   - Always destroy NCCL communicators before CUDA resources
   - Synchronize streams before destroying communicators

## Error Handling

The example uses simplified error handling with CHECK macros:
- **CUDACHECK**: Exits immediately on CUDA errors
- **NCCLCHECK**: Exits immediately on NCCL errors
- **No async error checking**: Simplified for clarity
- **Thread safety**: Each thread handles its own errors

## Highlighted Environment Variables

- `NTHREADS`: Number of threads to create (defaults to number of GPUs)

See examples/README.md for the full list.

## Next Steps

After understanding this example:
1. Try using the collective examples and add the pthread approach
2. Compare with MPI-based multi-process approach
