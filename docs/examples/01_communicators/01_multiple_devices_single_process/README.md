<!--
  SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0

  See LICENSE.txt for more license information
-->

# NCCL Example: Multiple Devices Single Process

This example demonstrates the simplest NCCL communicator initialization pattern:
**one process managing multiple GPUs**, with ranks assigned locally from 0 to \(N-1\).

## Overview

This example demonstrates how to initialize NCCL communicators for multiple GPUs within a
single process. This is the simplest NCCL setup and is ideal for learning NCCL basics or for
applications that want to use multiple GPUs without the complexity of multi-process coordination.

`ncclCommInitAll()` / `Communicator.init_all()` provides a simplified way to create communicators
when all of the following apply:

- All GPUs are managed by a single process
- Running on a single node
- No multi-process coordination is needed

This approach is ideal for single-node multi-GPU applications where simplicity is preferred over
the flexibility of multi-process setups. For the full API comparison between `ncclCommInitAll()`
and `ncclCommInitRank()`, see `c/README.md`.

## Runtime Requirements

This example is intended for single-node execution without any MPI or pthread parallelization. It is most useful with multiple GPUs, but can still run with one visible GPU.

## What This Example Does

1. Detect all available CUDA devices
2. Create communicators for all devices
3. Verify communicator properties (rank, size, device assignment)
4. Clean up all resources properly

## Variants

- **C**: `c/` (uses `ncclCommInitAll()`)
  - How to run + walkthrough: `c/README.md`
- **Python (nccl4py)**: `python/` (uses `nccl.Communicator.init_all()`)
  - How to run + walkthrough: `python/README.md`

## Expected Output

You should see:

- GPU count and basic device info
- Successful communicator initialization
- Per-rank info printed (rank/size/device)
- Clean shutdown

## When to Use ncclCommInitAll

- **Ideal**: single-node, single-process applications that want to use all GPUs with minimal setup
- **Not supported**: multi-node workloads (use the MPI-based examples instead)

## Performance Considerations

- **Advantages**:
  - Lower overhead than multi-process coordination
  - Simpler memory management model
  - Direct access to all GPUs from one process

- **Disadvantages**:
  - Limited by single-process resources
  - Cannot scale beyond one node

## Common Issues and Solutions

1. **Not all GPUs visible**:
   - Check `CUDA_VISIBLE_DEVICES`
   - Ensure user has permissions for all GPUs
   - Verify no other process is using GPUs exclusively

2. **Out of memory**:
   - A single process must handle memory for all GPUs
   - Consider using multiple processes if memory is limited

## Next Steps

After understanding this example:
1. Try the collective operation examples using `ncclCommInitAll()`
2. Compare this pattern with the MPI-based multi-process approach
3. Experiment with different GPU combinations
