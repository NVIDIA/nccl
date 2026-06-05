<!--
  SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0

  See LICENSE.txt for more license information
-->

# NCCL Example: Device API AlltoAll (Pure GIN)

This example demonstrates NCCL's GPU-Initiated Networking (GIN) capabilities for
performing AlltoAll collective operations directly from GPU kernels using only
network-based communication.

## Overview

Pure GIN communication sends all data through the network, without Load Store
Access (LSA) optimizations. This is the approach for multi-node environments
where ranks cannot use direct peer memory access, and serves as the baseline
for understanding GIN communication patterns.

## Runtime Requirements

The C variant can be built with either pthreads or MPI. It can run on a single node but is especially relevant for multi-node communication. It is most useful with multiple GPUs, but can still run with one visible GPU.

## What This Example Does

1. **Create device communicators** with GIN barriers, signals, and full connectivity
2. **Register symmetric memory windows** for GIN network access
3. **Launch GPU kernel** that performs AlltoAll entirely on device:
   - Synchronize with GIN barriers across all ranks
   - Use `gin.put()` to write data to each peer's receive buffer
   - Wait for signal-based completion of all remote puts
   - Flush to ensure all operations are committed
4. **Verify results** and clean up resources

## Key concepts

### Device communicator with GIN resources
For pure GIN communication, the device communicator is configured with GIN-specific
resources: `railGinBarrierCount` for network barriers, `ginSignalCount` for async
completion signals, and `ginConnectionType` (e.g., `NCCL_GIN_CONNECTION_FULL`) to
establish connectivity to all peers. Unlike LSA, no LSA barriers are needed since all
communication goes through the network.

### GIN barriers
GIN barriers enable cross-node synchronization from device code over the network. Like
LSA barriers, each thread block gets its own barrier indexed by `blockIdx.x`. The key
difference is that GIN barriers operate over the network rather than through direct
memory access.

### GIN put operations
GIN provides one-sided put operations for direct remote memory writes over the network.
Each put increments a signal counter (`ncclGin_SignalInc`), enabling asynchronous
completion detection without blocking.

### Signal-based completion
After issuing all puts, the kernel waits for the signal value to reach the expected count
(indicating all remote writes have completed), then flushes to ensure operations are
committed. This asynchronous pattern enables efficient overlapping of computation and
communication.

For code examples of each concept, see `c/README.md`.

## Variants

- **C/CUDA**: `c/` (uses `ncclDevCommCreate` + GIN barriers + `gin.put()`)
  - How to run + walkthrough: `c/README.md`

Note: This example is C/CUDA-only as it demonstrates GPU kernel programming with the NCCL Device API.

## Expected Output

You should see:

- Device communicator creation for each rank with GIN support
- AlltoAll kernel execution on all ranks via pure GIN
- Verification that all ranks received the correct data

## When to Use

- **Multi-node environments**: When ranks cannot use LSA (direct peer memory access)
- **Network-only communication**: Testing or benchmarking network performance
- **Cross-node collectives**: AlltoAll and other patterns that span network boundaries

## Performance Considerations

- **Network overhead**: All communication goes through the network stack
- **Signal-based completion**: Enables asynchronous operation patterns
- **Entry barrier** (`ncclGinBarrierSession` before puts): aligns ranks before the exchange
- **Multiple GIN contexts**: Can improve parallel communication performance

## Common Issues and Solutions

1. **Deadlock at initialization**: Ensure you're running with multiple GPUs/processes
2. **CUDA out of memory**: Reduce the data size in the example
3. **Network errors**: Ensure proper network configuration for multi-node setups
4. **Communicator does not support symmetric memory**: Select GPUs with NVLink/PCIe connectivity via `CUDA_VISIBLE_DEVICES`

## Performance Notes

- These are educational examples, not optimized for performance
- Real implementations should consider optimal GIN context usage, signal pool management,
  memory coalescing, and network topology-aware strategies

## Error Handling

The example uses comprehensive error checking for CUDA, NCCL, and GIN operations. Device kernels should implement proper error handling for network operations and signal management.

## Next Steps

After understanding this example, explore:
- **Performance optimization**: Fine-tune GIN context usage and signal management
- **Hybrid approaches**: Combine GIN with LSA for topology-aware optimizations
- **Integration with compute**: Fuse network communication with computation kernels
