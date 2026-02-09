<!-- Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

See LICENSE.txt for license information -->

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

## Choosing the Right Pattern

*Scenario* : Large-scale training with consistent memory patterns
*Addresses* : Low-latency, high-bandwidth collectives on supported systems
*Dependencies* : pthread or MPI

### Why Symmetric Windows?
Symmetric windows enable NCCL to apply optimized collective protocols when all
ranks use consistent layouts. The memory needs to be allocated through the CUDA
Virtual Memory Management (VMM) API and registered with NCCL.

```c
// Allocate using NCCL provided convenience function and register symmetric windows
NCCLCHECK(ncclMemAlloc(&buffer, size_bytes));
NCCLCHECK(ncclCommWindowRegister(comm, buffer, size_bytes, &win, NCCL_WIN_COLL_SYMMETRIC));

// Collective using symmetric windows
NCCLCHECK(ncclAllReduce(buffer, buffer, count, ncclFloat, ncclSum, comm, stream));

// Deregister and free
NCCLCHECK(ncclCommWindowDeregister(comm, win));
NCCLCHECK(ncclMemFree(buffer));
```

## Building

### **Quick Start**
```shell
# Build example by directory name
make 01_allreduce
```

### **Individual Examples**
```shell
# Build and run AllReduce with symmetric windows
cd 01_allreduce && make
./allreduce_sm
```

## References
- [NCCL User Guide:
  Examples](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/examples.html)
- [NCCL API
  Reference](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api.html)
- [CUDA Programming
  Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
