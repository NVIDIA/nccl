<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0

  See LICENSE.txt for more license information
-->

# NCCL Example: Fused RMSNorm with GIN

**Pure GPU-Initiated Networking (GIN) RMSNorm Implementation**

This example demonstrates fused computation and communication using NCCL's GPU-Initiated Networking (GIN). GIN allows GPU kernels to directly initiate remote communication without CPU involvement, enabling efficient multi-node communication.

See the [category README](../README.md) for the shared background on distributed RMSNorm (the reduce-scatter → RMSNorm → all-gather pattern), the host program setup flow, and prerequisites.

## Overview

Implement the fused distributed RMSNorm across nodes by issuing remote PUT operations directly from the GPU kernel via GIN, with signal-based synchronization — no host-launched collectives and no CPU involvement in the data path.

## Target Architecture / Topology

Multi-node.

## Key Features

- Uses `ncclGin` for GPU-initiated PUT operations across the full communicator
- Uses `ncclGinBarrierSession` with `ncclTeamTagWorld()` for GIN-only cross-rank barriers (pairs with `worldGinBarrierCount` on the host)
- Issues remote **PUT** operations via `gin.put()`
- Implements signal-based synchronization with `gin.waitSignal()`
- Applies a uniform GIN communication pattern across all ranks and is primarily aimed at multi-node setups

## What This Example Does

The kernel fuses the three logical phases described in the [category README](../README.md):

1. **Phase 1 (Reduce-Scatter via GIN PUT)**: each block issues PUTs to every rank (including itself), waits on per-block signals until all peers' contributions for its token have arrived, then reduces them in place
2. **Phase 2 (RMSNorm)**: each block normalizes its reduced token with the shared `blockRMSNorm()` helper
3. **Phase 3 (All-Gather via GIN PUT)**: each block PUTs its normalized token back to all peers

## Device Communicator Requirements

| Requirement | Value | Purpose |
|-------------|-------|---------|
| `ginConnectionType` | `NCCL_GIN_CONNECTION_FULL` | Enables full GIN connectivity for remote PUT |
| `worldGinBarrierCount` | `tokens_per_gpu` | One world GIN barrier per block; used by `ncclGinBarrierSession` with `ncclTeamTagWorld` for fence-level sync |
| `ginSignalCount` | `tokens_per_gpu` | One signal per block; `gin.waitSignal()` observes inbound signaled PUTs from peers (see GIN signals in `c/README.md`) |

Requires GIN support (`ginType != NCCL_GIN_TYPE_NONE`) and Device API support (verified via `ncclCommQueryProperties`).

## Variants

- **C/CUDA**: `c/` (uses `ncclGin` + `gin.put`/`gin.waitSignal` + `ncclGinBarrierSession`)
  - How to run + walkthrough: `c/README.md`

Note: This example is C/CUDA-only as it demonstrates device-side (kernel) communication, which is not exposed by the nccl4py host-side API.

## Expected Output

You should see:

- Device API and GIN support capability checks per rank
- The fused RMSNorm kernel launching across `tokens_per_gpu` blocks
- Verification of the GPU output against a CPU reference, with a pass/fail result

## Performance Characteristics

**Advantages of GIN:**
- **Zero CPU Involvement**: GPU kernel directly initiates remote transfers
- **Scalability**: Efficient across multiple nodes with high-speed interconnects
- **RDMA**: Leverages RDMA capabilities of modern interconnects
- **Multiple Contexts**: Uses multiple GIN contexts to spread blocks across communication channels, improving parallel throughput

**Thread-level Parallelism:**
- Each of the 256 threads processes `hidden_dim / 256` elements (4 elements for hidden_dim=1024)
- Threads also divide communication work: each thread handles `ceil(nRanks / 256)` peers
- Coalesced memory access pattern for both computation and data transfers

**Block-level Parallelism:**
- Each block operates independently on one token
- `tokens_per_gpu` blocks execute concurrently (limited by GPU resources)
- Per-block signals enable independent progress tracking
- Blocks are distributed across GIN contexts for parallel communication utilization

**Performance considerations:**
- **Best Performance**: With high-speed interconnects
- **Communication**: Optimized for remote transfers
- **Scalability**: Scales to multiple nodes
