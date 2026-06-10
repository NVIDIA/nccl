<!--
  SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0

  See LICENSE.txt for more license information
-->

# NCCL Example: Device API AllReduce (LSA)

This example shows how to implement AllReduce sum operation directly in a GPU
kernel using the NCCL Device API with Load Store Accessible (LSA) barriers and
peer memory access.

## Overview

The NCCL Device API enables GPU kernels to perform inter-GPU communication
directly, without returning to the host. This allows fusing computation with
communication in a single kernel launch. This example demonstrates the LSA
(Load Store Accessible) approach, where GPU threads directly read/write peer
memory through symmetric windows.

## Runtime Requirements

The C variant can be built with either pthreads or MPI. LSA is intended for local peer access on a single node. It is most useful with multiple GPUs, but can still run with one visible GPU.

## What This Example Does

1. **Create device communicators** with LSA barrier resources
2. **Register symmetric memory windows** for direct peer-to-peer access
3. **Launch GPU kernel** that performs AllReduce entirely on device:
   - Synchronize with LSA barriers across all ranks
   - Read peer data via LSA pointers
   - Accumulate sum locally
   - Write result and release barrier
4. **Verify results** and clean up resources

## Key concepts

### Device communicator
The `ncclDevComm` is the core component of the device API, enabling GPU kernels
to perform inter-GPU communication. The `ncclDevCommRequirements` structure specifies
what resources the device communicator should allocate — in this example,
`lsaBarrierCount` is set to match the thread block count, giving each block its
own barrier for independent cross-GPU synchronization.

### Symmetric memory windows
The device API requires symmetric memory windows registered with
`NCCL_WIN_COLL_SYMMETRIC`. See the [symmetric memory example](../../05_symmetric_memory/)
for allocation and requirements details. These windows enable GPU kernels to directly
access other GPUs' memory within the LSA team.

### LSA barriers
LSA barriers enable cross-GPU synchronization from device code. Each thread block uses
`blockIdx.x` to select its dedicated barrier, allowing blocks to progress independently
while coordinating with corresponding blocks on other GPUs. Barriers support memory
ordering semantics (`memory_order_acquire` to wait for peers before reading,
`memory_order_release` to ensure data is visible before unblocking the stream).

### Peer memory access
`ncclGetLsaPointer` allows CUDA kernels to directly access other GPUs' memory within the
LSA team via load/store operations — no host involvement or network traffic needed.

For code examples of each concept, see `c/README.md`.

## Variants

- **C/CUDA**: `c/` (uses `ncclDevCommCreate` + LSA barriers + `ncclGetLsaPointer`)
  - How to run + walkthrough: `c/README.md`

Note: This example is C/CUDA-only as it demonstrates GPU kernel programming with the NCCL Device API.

## Expected Output

You should see:

- Device communicator creation for each rank with LSA barrier count
- AllReduce kernel execution on all ranks
- Verification that all elements correctly sum to `0 + 1 + ... + (N-1)`

## When to Use

- **Kernel-level communication**: When compute kernels need immediate access to communication results
- **Low-latency scenarios**: Reduced host-device synchronization overhead
- **Custom collectives**: Implementing specialized reduction or communication patterns
- **Computation-communication fusion**: Overlapping compute and communication in the same kernel

## Performance Considerations

**Advantages:**
- Lower latency for small to medium message sizes
- Eliminates host-device synchronization bottlenecks
- Enables computation-communication fusion within kernels
- Direct peer memory access without CPU involvement

**Disadvantages:**
- More complex programming model requiring LSA barriers
- Requires careful memory ordering and synchronization
- Higher development complexity compared to host API
- Requires CUDA Compute Capability 7.0+ and GPUs with P2P support (e.g., NVLink)

## Common Issues and Solutions

1. **Communicator does not support symmetric memory**: NCCL selects support based on GPU connectivity.
   Use `nvidia-smi` to select GPUs connected through NVLink or PCIe via `CUDA_VISIBLE_DEVICES`.
2. **LSA barrier synchronization failures**: Ensure `lsaBarrierCount` matches the number of thread blocks.
3. **Memory access violations**: Verify windows are registered as symmetric and all ranks use identical buffer sizes.
4. **Race conditions**: Use proper memory ordering in LSA barriers (`memory_order_acquire` vs `memory_order_release`).

## Performance Notes

- These are educational examples, not optimized for performance
- Real implementations should use vectorization, loop unrolling, and memory coalescing
- See NCCL's internal device kernels and perf tests for optimized reference implementations

## Error Handling

The example uses comprehensive error checking for both CUDA and NCCL operations.
Device kernels should implement proper error handling for LSA operations and
memory access patterns.

## Next Steps

After understanding this example, explore:
- **Custom reduction operations**: Implement non-standard reduction patterns
- **Mixed host-device patterns**: Combine host and device API for complex
  workflows
- **Performance optimization**: Fine-tune LSA barrier usage and memory access
  patterns
