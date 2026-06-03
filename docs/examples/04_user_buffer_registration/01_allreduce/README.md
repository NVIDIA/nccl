<!--
  SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0

  See LICENSE.txt for more license information
-->

# NCCL Example: User Buffer Registration AllReduce

This example demonstrates how to use NCCL's user buffer registration feature to
optimize performance for repeated collective operations on the same buffers.

## Overview

User buffer registration allows NCCL to pre-register memory buffers with
communicators, enabling zero-copy communication. This eliminates internal buffer
copies and registration overhead on each operation, which is particularly
beneficial for applications that repeatedly perform collective operations on the
same memory regions (e.g., iterative training loops).

## Runtime Requirements

The C variant can be built with either pthreads or MPI (and with MPI for multi-node runs), while the Python variant uses a single-process, single-node setup. It is most useful with multiple GPUs, but can still run with one visible GPU.

## What This Example Does

1. **Allocate memory** using NCCL's allocator (required for buffer registration)
2. **Register buffers** with the communicator for optimized performance
3. **Perform AllReduce** sum operation using the registered buffers
4. **Deregister and free** buffers in the correct cleanup order

## Variants

- **C**: `c/` (uses `ncclMemAlloc()` + `ncclCommRegister()` + `ncclAllReduce()`)
  - How to run + walkthrough: `c/README.md`
- **Python (nccl4py + mpi4py)**: `python/` (uses `nccl.cupy.empty()` + `comm.register_buffer()` + `comm.allreduce()`)
  - How to run + walkthrough: `python/README.md`

## Expected Output

You should see:

- Communicator initialization for each rank
- Buffer allocation and registration
- AllReduce operation completion
- Verification showing all ranks received the correct sum
- Clean deregistration and resource cleanup

## Performance Benefits of User Buffer Registration

User buffer registration provides several advantages:

1. **Reduced overhead**: Pre-registration eliminates per-operation registration costs
2. **Better memory pinning**: Registered buffers are pinned in memory, preventing page faults
3. **Lower latency**: Especially beneficial for repeated operations on the same buffers

**Important**: Buffers must be allocated with `ncclMemAlloc()` or another compatible allocator (C), or an
NCCL-compatible allocator in Python such as `nccl.cupy.empty()`, for registration to work. See the
[Buffer Registration](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/bufferreg.html)
section of the user guide.

**Important**: If any rank passes registered buffers, **all** ranks in the same communicator must also
pass registered buffers. Mixing registered and non-registered buffers is undefined behavior.

## Common Issues and Solutions

1. **Registration failure**: Buffers must be from `ncclMemAlloc()` or compatible allocator (not `cudaMalloc()`)
2. **Allocation error**: Check NCCL version (requires 2.19+) and available memory
3. **Deregistration order**: Always deregister before freeing memory
4. **Handle management**: Keep track of registration handles for proper cleanup
5. **Memory leaks**: Always free buffers allocated with NCCL's allocator
