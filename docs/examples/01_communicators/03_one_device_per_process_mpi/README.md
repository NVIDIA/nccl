<!--
  SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0

  See LICENSE.txt for more license information
-->

# NCCL Example: One Device per Process (MPI)

This example demonstrates NCCL communicator lifecycle management using MPI, with
one GPU per MPI process.

## Overview

This example shows one of the most common NCCL deployment patterns: one GPU
device per process. This approach is ideal for distributed training across
multiple nodes and provides the foundation for scalable multi-GPU applications.
MPI is used as it provides a parallel launcher and broadcast functions. It is,
however, not a requirement for multi-node NCCL applications.

Other approaches use server-client models or spawn parallel processes using
sockets. NCCL only requires that the unique ID is distributed among each
thread/process taking part in collective communication and all threads/processes
call some NCCL initialization function.

## Runtime Requirements

This example requires MPI: it is launched with `mpirun`, with one MPI process per GPU, and can run on a single node or across multiple nodes. It assigns exactly one GPU per MPI process, so multiple GPUs are expected.

## What This Example Does

1. **Multi-node Support**: Determines each process's node-local rank and assigns it the GPU with that index (device = local rank)
2. **Communicator Creation**: Rank 0 generates and broadcasts NCCL unique ID, then each process initializes its communicator
3. **Verification**: Displays MPI rank → NCCL rank → GPU device mapping
4. **Cleanup**: Proper resource cleanup order with MPI synchronization

## Variants

- **C**: `c/` (uses MPI + `ncclCommInitRank()`)
  - How to run + walkthrough: `c/README.md`
- **Python (nccl4py + mpi4py)**: `python/` (uses `mpi4py` + `nccl.Communicator.init()`)
  - How to run + walkthrough: `python/README.md`

## Expected Output

You should see:
- MPI process initialization (across one or more nodes)
- GPU assignment by node-local rank (each process uses the GPU whose index equals its local rank, restarting from 0 on each node)
- Successful communicator initialization per process
- MPI rank → NCCL rank → GPU device mapping spanning all processes
- Clean shutdown

## When to Use MPI Approach

### Ideal Use Cases
- **Multi-node clusters**: Scales across multiple machines
- **Production deployments**: One-process-per-GPU is the standard pattern for distributed training, inference, and HPC; MPI is a common launcher in HPC, while ML frameworks often use their own (e.g., `torchrun`)
- **Process isolation**: Each GPU in separate process for robustness
- **Large scale**: Supports thousands of GPUs

### When NOT to Use
- **Single-node testing**: Simpler approaches available (use single-process examples)
- **No MPI available**: Some environments don't support MPI
- **Shared memory needs**: Single-process approaches may be simpler

## Performance Considerations

- **Advantages**:
  - MPI has been optimized for large parallel startup.
  - Well-established one-process-per-GPU deployment pattern
  - Scales to large-scale training

- **Disadvantages**:
  - MPI setup complexity
  - Inter-process communication overhead
  - Requires MPI runtime environment

## Common Issues and Solutions

1. **More MPI processes than GPUs on a node**:
   - The example reports an error if local rank exceeds available devices
   - Use fewer processes per node or more GPUs

2. **MPI broadcast hangs**:
   - Ensure all ranks participate in collective operations
   - Check MPI installation and network connectivity

3. **Multi-node communication fails**:
   - Check firewall settings and network configuration
   - Set `NCCL_SOCKET_IFNAME` to specify network interface

## Error Handling

This example uses simplified error checking that immediately exits on any failure, with each process
handling its own errors independently (no global error coordination) and no asynchronous error
checking. In production code, consider more graceful error handling and recovery mechanisms.

## Next Steps

After understanding this example:
1. Try running collective operations (AllReduce, AllGather, etc.)
2. Experiment with multi-node deployments
