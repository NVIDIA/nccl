<!--
  SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0

  See LICENSE.txt for more license information
-->

# NCCL Example: Symmetric Memory AllReduce

This example demonstrates how to use NCCL's symmetric memory feature for
optimized collective operations.

## Overview

Symmetric memory windows provide a way to register memory buffers that benefit
from optimized collective algorithms. When all ranks provide symmetric buffers
(same size, allocated with the NCCL allocator), NCCL can apply advanced
communication optimizations for better performance.

## Runtime Requirements

The C variant can be built with either pthreads or MPI (and with MPI for multi-node runs), while the Python variant uses a single-process setup; symmetric-memory benefits are most natural on a single node. It is most useful with multiple GPUs, but can still run with one visible GPU.

## What This Example Does

1. **Allocate memory** using NCCL's allocator (required for symmetric windows)
2. **Register symmetric windows** with the communicator (collective call - all ranks participate)
3. **Perform AllReduce** using the symmetric buffers for optimized communication
4. **Deregister and free** windows and buffers in the correct cleanup order

## Variants

- **C**: `c/` (uses `ncclMemAlloc()` + `ncclCommWindowRegister()` + `ncclAllReduce()`)
  - How to run + walkthrough: `c/README.md`
- **Python (nccl4py + mpi4py)**: `python/` (uses `nccl.cupy.empty()` + `comm.register_window()` + `comm.allreduce()`)
  - How to run + walkthrough: `python/README.md`

## Expected Output

You should see:

- Communicator initialization for each rank
- Symmetric memory allocation and window registration
- AllReduce operation completion
- Verification showing all ranks received the correct sum
- Clean window deregistration and resource cleanup

## Performance Benefits of Symmetric Memory

Symmetric memory registration provides several advantages:

- **Optimized communication algorithms**: NCCL can apply advanced optimizations
  when all ranks have symmetric layouts
- **Better memory access patterns**: Consistent layouts enable better caching
  and memory access optimization

For more information see the [Enabling Fast Inference and Resilient Training
with NCCL 2.27](https://developer.nvidia.com/blog/enabling-fast-inference-and-resilient-training-with-nccl-2-27/)
blog.

**Important**: Buffers must be allocated using the CUDA Virtual Memory Management (VMM) API.
NCCL provides `ncclMemAlloc()` (C), and the Python example uses `nccl.cupy.empty()` to create
NCCL-backed CuPy arrays from the same allocator family. The `NCCL_WIN_COLL_SYMMETRIC` (C) /
`WindowFlag.CollSymmetric` (Python) flag requires all ranks to provide symmetric buffers
consistently.

## Common Issues and Solutions

1. **Window registration failure**: Buffers must be from VMM-compatible allocator (e.g., `ncclMemAlloc()`, not `cudaMalloc()`)
2. **Allocation error**: Check NCCL version (requires 2.27+) and available memory
3. **Deregistration order**: Always deregister windows before freeing memory
4. **Symmetric requirement**: All ranks must use the symmetric flag consistently
5. **Memory leaks**: Always free buffers allocated with NCCL's allocator
