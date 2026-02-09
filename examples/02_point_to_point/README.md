<!-- Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

See LICENSE.txt for license information -->

# NCCL Point-to-Point Communication Examples

## Overview
This directory contains minimal examples that demonstrate NCCL point-to-point
(P2P) communication patterns on a single node. These examples focus on clarity
and correct communicator usage, resource management, and verification.

## Examples

### [01_ring_pattern](01_ring_pattern/)
**Ring Communication Pattern**
- **Pattern**: Circular data flow among all GPUs
- **API**: `ncclCommInitAll` with P2P operations (`ncclSend`/`ncclRecv`)
- **Use case**: Learning P2P communication; pipeline/data movement patterns on a
  single node
- **Key features**:
  - Initializes all GPUs in a single process
  - Computes ring neighbors with modulo arithmetic
  - Uses `ncclGroupStart/End` to prevent deadlocks
  - Verifies data correctness after transfers

## Choosing the Right Pattern

*Scenario* : Pipeline parallel training needs to send data from one GPU to
another
*Addresses* : Individual transfers between two ranks
*Dependencies* : A functional NCCL library and its dependencies

### Why `ncclCommInitAll` here?
For single-node collective examples we use `ncclCommInitAll` as it creates a
clique of communicators in one call.

```c
// Initialize all GPUs in one call
ncclComm_t* comms;
int num_gpus;
NCCLCHECK(ncclCommInitAll(comms, num_gpus, NULL));
```

## Building

### **Quick Start**
```shell
# Build example by directory name
make 01_ring_pattern
```

### **Individual Examples**
```shell
# Build and run the ring pattern
cd 01_ring_pattern && make
./ring_pattern
```

## References
- [NCCL User Guide:
  Examples](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/examples.html)
- [NCCL API
  Reference](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api.html)
- [CUDA Programming
  Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
