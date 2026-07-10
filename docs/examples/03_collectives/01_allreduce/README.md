<!--
  SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0

  See LICENSE.txt for more license information
-->

# NCCL Example: AllReduce Collective Operation

This example demonstrates the fundamental AllReduce collective operation using
NCCL's single-process, multi-GPU approach in which a single process manages all
GPUs to perform a sum reduction.

## Overview

AllReduce combines data from all participants using a reduction operation (sum,
max, min, etc.) and distributes the result to all participants. This is one of
the most important collective operations in distributed and parallel computing.

## Runtime Requirements

This example is intended for single-node execution without any MPI or pthread parallelization; use the MPI communicator example to scale across nodes. It is most useful with multiple GPUs, but can still run with one visible GPU.

## What This Example Does

1. **Initialize**: Detects available GPUs and initializes NCCL communicators
2. **Data preparation**: Each GPU contributes its rank value (GPU 0→0, GPU 1→1, etc.)
3. **AllReduce operation**: All GPU values are combined using the reduction operation (sum) and distributed to all participants
4. **Verification**: Checks that all GPUs received the expected result: 0+1+2+...+(n-1)

## Variants

- **C**: `c/` (uses `ncclAllReduce()`)
  - How to run + walkthrough: `c/README.md`
- **Python (nccl4py)**: `python/` (uses `comm.allreduce()`)
  - How to run + walkthrough: `python/README.md`

## Expected Output

You should see:

- GPU count and initialization
- Per-GPU data initialization (each GPU contributes its rank)
- Successful AllReduce operation
- Verification results showing all GPUs received the correct sum

Example with 4 GPUs:
- Each GPU sends: GPU 0→0, GPU 1→1, GPU 2→2, GPU 3→3
- All GPUs receive: 6 (0+1+2+3=6)

## When to Use

- **Deep learning**: Gradient averaging in data-parallel training (most common use case)
- **Scientific computing**: Global reductions in parallel algorithms
- **Statistics**: Computing global sums, averages, or other reductions
- **Distributed algorithms**: Any scenario requiring collective reduction operations

## Common Issues and Solutions

### Issue: Verification failures
**Solution:** Ensure each GPU initializes its buffer correctly with its rank value.

### Issue: Out of memory errors
**Solution:** Reduce the buffer size in the code or use fewer GPUs.

## Error Handling

This example uses comprehensive error checking that immediately exits on any failure. In production code, consider more graceful error handling and recovery mechanisms.

## Next Steps

After understanding AllReduce, explore:
- **Point-to-point communication**: Examples in `02_point_to_point/`
- **Other collectives**: Implement Broadcast, Reduce, AllGather operations
- **Multi-node approach**: Use the MPI implementation from `01_communicators` to scale across nodes
