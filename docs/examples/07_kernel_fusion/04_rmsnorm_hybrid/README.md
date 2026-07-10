<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0

  See LICENSE.txt for more license information
-->

# NCCL Example: Fused RMSNorm with Hybrid LSA/GIN

**Hybrid LSA/GIN RMSNorm Implementation**

This example demonstrates fused computation and communication using a hybrid approach that combines both Load Store Accessible (LSA) for intra-node communication and GPU-Initiated Networking (GIN) for inter-node communication. This pattern is ideal for multi-node systems where GPUs within a node can leverage fast NVLink (LSA) while communicating across nodes via GIN.

See the [category README](../README.md) for the shared background on distributed RMSNorm (the reduce-scatter → RMSNorm → all-gather pattern), the host program setup flow, and prerequisites.

## Overview

Implement the fused distributed RMSNorm on multi-node systems by selecting the optimal transport per peer: LSA direct memory writes for same-node peers and GIN remote PUTs for peers on other nodes.

## Target Architecture / Topology

Multi-node with NVLink per node.

## Key Features

- Uses `ncclBarrierSession` with `ncclTeamTagWorld()` and `ncclGin` for world-team barriers (`bar.sync`) and the LSA sub-barrier (`bar.lsaBarrier()`); both pair with `barrierCount` on the host
- Leverages `ncclGetLsaPointer` for local peer memory access within the same node
- Issues remote **PUT** to peers on other nodes via `gin.put()`
- Optimally selects communication mechanism based on peer location

## What This Example Does

The kernel fuses the three logical phases described in the [category README](../README.md), splitting transport by peer kind:

1. **Phase 1 (Reduce-Scatter via Hybrid LSA/GIN)**: remote peers receive contributions via signaled GIN PUTs; same-node peers via direct LSA stores; the block then reduces all contributions in place
2. **Phase 2 (RMSNorm)**: each block normalizes its reduced token with the shared `blockRMSNorm()` helper
3. **Phase 3 (All-Gather via Hybrid LSA/GIN)**: normalized results are broadcast back using GIN PUTs to remote peers and LSA writes to local peers

## Device Communicator Requirements

| Requirement | Value | Purpose |
|-------------|-------|---------|
| `ginConnectionType` | `NCCL_GIN_CONNECTION_FULL` | Enables GIN for remote peer PUT |
| `barrierCount` | `tokens_per_gpu` | One hybrid barrier per block; `ncclBarrierSession` with `ncclTeamTagWorld` and `ncclGin` provides both `bar.sync` and `bar.lsaBarrier()` (not a separate `lsaBarrierCount`) |
| `ginSignalCount` | `tokens_per_gpu` | One signal per block; counts inbound signaled PUTs from **remote** peers only (LSA peers do not signal) |

Requires GIN support (`ginType != NCCL_GIN_TYPE_NONE`) and Device API support (verified via `ncclCommQueryProperties`).

## Variants

- **C/CUDA**: `c/` (uses `ncclBarrierSession` + `ncclGetLsaPointer` + `gin.put`)
  - How to run + walkthrough: `c/README.md`

Note: This example is C/CUDA-only as it demonstrates device-side (kernel) communication, which is not exposed by the nccl4py host-side API.

## Expected Output

You should see:

- Device API and GIN support capability checks per rank
- The fused RMSNorm kernel launching across `tokens_per_gpu` blocks
- Verification of the GPU output against a CPU reference, with a pass/fail result

## Performance Characteristics

**Advantages of Hybrid Approach:**
- **Optimal Path Selection**: Uses the best communication mechanism for each peer
- **Intra-node Performance**: Full NVLink bandwidth via LSA for local peers
- **Resource Efficiency**: Only remote operations use GIN signals, reducing signal pressure
- **Communication Utilization**: Multiple GIN contexts spread blocks across communication channels

**Trade-offs:**
- **Code Complexity**: More complex than pure LSA or pure GIN implementations
- **Synchronization Overhead**: Requires GIN signal waits, an LSA sub-barrier after Phase 1, and world `bar.sync` for cross-phase ordering
- **Memory Requirements**: Two windows required (like GIN), not the single window of LSA

**When to Use Hybrid vs Pure Approaches:**
- **Pure LSA**: Best for single-node / single NVLink domain workloads
- **Pure GIN**: Best when all peers are remote or uniform communication pattern is preferred
- **Hybrid**: Best for multi-node systems with multiple GPUs per node, maximizing both intra-node and inter-node performance

**Performance considerations:**
- **Best Performance**: Multi-node systems with NVLink within nodes
- **Communication**: Hybrid approach using LSA for intra-node, GIN for inter-node
- **Scalability**: Optimal for large-scale multi-node deployments with multiple GPUs per node
