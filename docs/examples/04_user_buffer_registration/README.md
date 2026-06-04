<!--
  SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0

  See LICENSE.txt for more license information
-->

# NCCL User Buffer Registration Examples

## Overview
This directory contains minimal examples that demonstrate NCCL user buffer
registration for improving performance by allowing NCCL to operate directly on
user-allocated buffers.

## Examples

### [01_allreduce](01_allreduce/)
**AllReduce with User Buffer Registration**
- **Pattern**: Register communication buffers once and reuse across operations
- **API**: `ncclCommRegister`, `ncclCommDeregister`, `ncclMemAlloc`,
  `ncclAllReduce`
- **Use case**: Repeated collectives on the same buffers; performance-critical
  workloads
- **Key features**:
  - Buffers allocated via `ncclMemAlloc` for registration compatibility
  - Registration handles managed explicitly (register → use → deregister)
  - Collective operations executed on registered buffers
  - Correct cleanup and verification

## Choosing the Right Pattern

*Scenario* : Optimize performance for repeated collectives on same buffers
*Addresses* : Throughput-sensitive training loops
*Dependencies* : pthread or MPI

### Why Buffer Registration?
Pre-registering buffers eliminates per-call registration overhead and enables
direct access. It can accelerate collectives and greatly reduce the resource
usage (e.g. #channel usage). Also, this is a prerequisite for advanced features
such as symmetric memory or device API calls.

## Building

### **Quick Start**
```shell
# Build example by directory name
make 01_allreduce
```

### **Individual Examples**
```shell
# Build and run AllReduce with user buffer registration
cd 01_allreduce/c && make
./allreduce_ub
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
