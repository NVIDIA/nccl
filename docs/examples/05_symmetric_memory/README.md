<!--
  SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0

  See LICENSE.txt for more license information
-->

# NCCL Symmetric Memory Examples

## Overview
This directory contains minimal examples that demonstrate NCCL symmetric memory
windows for improving performance of collective operations when all ranks use
consistent memory layouts.

## Examples

### [01_allreduce](01_allreduce/)
**AllReduce with Symmetric Memory Windows**
- **Pattern**: Register symmetric windows per rank and use them for collectives
- **API**: `ncclCommWindowRegister`, `ncclCommWindowDeregister`, `ncclMemAlloc`,
  `ncclAllReduce`
- **Use case**: Large-scale collectives with consistent buffer layouts across
  ranks
- **Key features**:
  - Buffers allocated via `ncclMemAlloc` for symmetric compatibility
  - Windows registered as `NCCL_WIN_COLL_SYMMETRIC`
  - Collective operations executed on symmetric windows
  - Correct deregistration and cleanup

### [02_allgather](02_allgather/)
**AllGather with Symmetric Memory Windows and Copy Engine**
- **Pattern**: Register symmetric windows and use copy engine for zero SM usage
- **API**: `ncclCommInitRankConfig`, `ncclCommWindowRegister`,
  `ncclCommWindowDeregister`, `ncclMemAlloc`, `ncclAllGather`
- **Use case**: Large-scale collectives with computation overlap
- **Key features**:
  - NCCL config with `CTAPolicy=2` to enable copy engine
  - Zero SM usage during collective operations
  - Enables true overlap of communication with computation
  - Buffers allocated via `ncclMemAlloc` for symmetric compatibility
  - Higher peak bandwidth for large message sizes (higher latency for small messages)

## Choosing the Right Pattern

*Scenario* : Large-scale training with consistent memory patterns
*Addresses* : Low-latency, high-bandwidth collectives on supported systems
*Dependencies* : pthread or MPI

### Why Symmetric Windows?
Symmetric windows enable NCCL to apply optimized collective protocols when all
ranks use consistent layouts. The memory needs to be allocated through the CUDA
Virtual Memory Management (VMM) API and registered with NCCL.

## Building

### **Quick Start**
```shell
# Build example by directory name
make 01_allreduce
make 02_allgather
```

### **Individual Examples**
```shell
# Build and run AllReduce with symmetric windows
cd 01_allreduce/c && make
./allreduce_sm

# Build and run AllGather with symmetric windows + copy engine
cd 02_allgather/c && make
./allgather_ce
```

### **Python**
See the `python/README.md` in each example directory.

## References
- [NCCL User Guide:
  Examples](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/examples.html)
- [NCCL API
  Reference](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api.html)
- [CUDA Programming
  Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
