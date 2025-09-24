<!-- Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

See LICENSE.txt for license information -->

# NCCL Collective Communication Examples

## Overview
This directory contains minimal examples that demonstrate NCCL collective
communication operations on a single node using a single process managing all
GPUs. The focus is clarity, correct resource management, and result
verification.

## Examples

### [01_allreduce](01_allreduce/)
**AllReduce Collective Operation**
- **Pattern**: All participants reduce and distribute the result
- **API**: `ncclCommInitAll`, `ncclAllReduce`
- **Use case**: Global reductions in ML and HPC (e.g., gradient averaging)
- **Key features**:
  - Initializes all GPUs in a single process
  - Each GPU contributes its rank value
  - Executes AllReduce sum across all GPUs
  - Verifies the expected global sum

## Choosing the Right Pattern

*Scenario* : Parallel training needs efficient global communication
*Addresses* : Most commonly used collective algorithms
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

A more advanced setup using MPI to initialize communicators across multiple
nodes is shown in
[01_communicators/03_one_device_per_process_mpi](../01_communicators/03_one_device_per_process_mpi)

## Building

### **Quick Start**
```shell
# Build example by directory name
make 01_allreduce
```

### **Individual Examples**
```shell
# Build and run AllReduce
cd 01_allreduce && make
./allreduce
```

## References
- [NCCL User Guide:
  Examples](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/examples.html)
- [NCCL API
  Reference](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api.html)
- [CUDA Programming
  Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
