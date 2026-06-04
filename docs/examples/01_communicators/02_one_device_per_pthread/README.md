<!--
  SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0

  See LICENSE.txt for more license information
-->

# NCCL Example: One Device per Thread (pthread)

This example demonstrates NCCL communicator lifecycle management using pthreads, with one GPU per
thread.

## Overview

This example shows how to use NCCL in a multi-threaded environment where each pthread manages one
GPU device. It demonstrates the proper initialization and cleanup sequence for NCCL communicators
within threads.

## Runtime Requirements

This example runs on a single node only, using pthreads (not MPI) for parallelization with one thread per GPU. It is most useful with multiple GPUs, but can still run with one thread and one GPU.

## What This Example Does

1. **Thread Creation**: Creates one pthread per available GPU (or `NTHREADS` if set)
2. **Communicator Creation**: Each thread initializes its own communicator using a shared unique ID
3. **Verification**: Queries and validates communicator properties across all threads
4. **Cleanup**: Proper resource cleanup order within each thread

## Variants

- **C**: `c/` (uses pthreads + `ncclCommInitRank`)
  - How to run + walkthrough: `c/README.md`

Note: This example is C-only as it demonstrates pthread-based parallelism, which is specific to C/C++.

## Expected Output

You should see:

- Thread creation for each GPU
- Successful communicator initialization per thread
- Per-thread rank/size/device information
- Clean shutdown from all threads

## When to Use pthread Approach

### Ideal Use Cases
- **Thread-based applications**: When your application is already threaded
- **Single-node workloads**: All GPUs on one machine
- **Shared memory**: Need to share data structures between GPU contexts

### When NOT to Use
- **Multi-node clusters**: Cannot scale beyond one node (use MPI-based examples instead)
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

This example uses simplified error checking that immediately exits on any failure, with each thread
handling its own errors independently and no asynchronous error checking. In production code,
consider more graceful error handling and recovery mechanisms.

## Highlighted Environment Variables

- `NTHREADS`: Number of threads to create (defaults to number of GPUs)

See examples/README.md for the full list.

## Next Steps

After understanding this example:
1. Try using the collective examples and add the pthread approach
2. Compare with MPI-based multi-process approach
