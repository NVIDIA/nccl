<!--
  SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0

  See LICENSE.txt for more license information
-->

# NCCL Example: Device API AlltoAll (Hybrid LSA + GIN)

This example shows how to implement AlltoAll operations using a hybrid approach
that combines Load Store Access (LSA) for local peers with GPU-Initiated
Networking (GIN) for remote peers.

## Overview

Real-world multi-GPU deployments often have mixed connectivity: GPUs on the same
node can use direct peer memory access (LSA), while GPUs on different nodes must
communicate over the network (GIN). This hybrid approach intelligently selects
the optimal communication method for each peer:

- **LSA** for local peers (same node, direct memory access)
- **GIN** for remote peers (different nodes, network-based)

## Runtime Requirements

The C variant can be built with either pthreads or MPI. It can run on a single node but is most relevant when local and remote peers are both present (multiple nodes with multiple GPUs per node). It is most useful with multiple GPUs, but can still run with one visible GPU.

## What This Example Does

1. **Create hybrid device communicators** with both LSA barriers and GIN resources
2. **Register symmetric memory windows** for both LSA and GIN access
3. **Launch GPU kernel** that performs AlltoAll:
   - Classify each peer as LSA-reachable or GIN-reachable
   - Use LSA direct memory access for local peers
   - Use GIN put operations for remote peers
   - Synchronize with appropriate barriers for each path
4. **Verify results** and clean up resources

## Key concepts

### Hybrid device communicator
The device communicator is configured with both LSA and GIN resources:
`lsaBarrierCount` for local peer synchronization, `railGinBarrierCount` for
network synchronization, `ginSignalCount` for async completion, and
`ginConnectionType` for cross-node connectivity. This dual setup enables optimal
communication for each peer type.

### Peer classification
The kernel classifies each peer at runtime as either LSA-reachable or GIN-only.
`ncclTeamWorld()` spans all ranks, while `ncclTeamLsa()` contains only the local
(same-node) ranks. The difference between the two determines which peers require
network communication.

### Hybrid barriers
Hybrid barriers coordinate both local LSA operations and remote GIN operations in
a unified synchronization. The barrier uses the world team and GIN context to ensure
all ranks — local and remote — reach the same point before proceeding with data exchange.

### Dual access patterns
Local peers are accessed via direct load/store through `ncclGetLsaPointer` (same as
the LSA example), while remote peers use `gin.put()` for one-sided network writes
(same as the GIN example). The kernel selects the appropriate path per peer, combining
the latency benefits of LSA with the scalability of GIN.

For code examples of each concept, see `c/README.md`.

## Variants

- **C/CUDA**: `c/` (uses `ncclDevCommCreate` + LSA barriers + GIN put + peer classification)
  - How to run + walkthrough: `c/README.md`

Note: This example is C/CUDA-only as it demonstrates GPU kernel programming with the NCCL Device API.

## Expected Output

You should see:

- Device communicator creation with both LSA and GIN support
- Peer classification showing which peers use LSA vs GIN
- Hybrid AlltoAll kernel execution on all ranks
- Verification that all ranks received the correct data

## When to Use

- **Multi-node clusters**: Mixed LSA and GIN connectivity between GPUs
- **Production deployments**: Optimal performance across all peer types
- **Topology-aware applications**: Adapting communication to the hardware layout
- **Large-scale training**: Combining fast intra-node and network inter-node communication

## Performance Considerations

**Advantages:**
- Optimal communication method for each peer type
- Combines LSA's low latency with GIN's multi-node reach
- Topology-aware for best overall performance

**Disadvantages:**
- Most complex programming model (both LSA and GIN in one kernel)
- Requires understanding of GPU connectivity topology
- More code paths to test and maintain

## Common Issues and Solutions

1. **All peers classified as GIN**: GPUs may lack direct P2P access. Check NVLink/PCIe connectivity.
2. **Deadlock at initialization**: Ensure you're running with multiple GPUs/processes.
3. **Communicator does not support symmetric memory**: Select GPUs with NVLink/PCIe connectivity via `CUDA_VISIBLE_DEVICES`.
4. **Signal management**: Ensure GIN signals are properly initialized and waited on.

## Performance Notes

- These are educational examples, not optimized for performance
- Real implementations should consider memory coalescing, optimal GIN context usage,
  and overlap of LSA and GIN operations for maximum throughput

## Error Handling

The example uses comprehensive error checking for CUDA, NCCL, LSA, and GIN operations. Device kernels should implement proper error handling for both direct memory access patterns and network operations.

## Next Steps

After understanding this example, explore:
- **Topology-aware optimization**: Fine-tune LSA/GIN balance based on hardware topology
- **Custom hybrid patterns**: Implement specialized communication strategies
- **Performance profiling**: Analyze LSA vs GIN performance characteristics
- **Advanced synchronization**: Optimize barrier usage for complex communication patterns
