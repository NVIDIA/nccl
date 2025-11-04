<!-- Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

See LICENSE.txt for license information -->

# NCCL Device API Examples

## Overview
This directory contains minimal examples that demonstrate NCCL's device API,
enabling users to perform inter-GPU communication within their own kernels.

## Examples

### [01_allreduce](01_allreduce/)
**AllReduce with Device Kernel Implementation**
- **Pattern**: GPU kernel performs collectives using device communicators
- **API**: `ncclDevCommCreate`, `ncclCommWindowRegister`, device-side LSA
  barriers, `ncclAllReduce`
- **Use case**: Allreduce operations with custom operations, fusing allreduce
  operation with previous/next compute operation.
- **Key features**:
  - Device communicator creation with LSA barrier support
  - Symmetric memory windows for peer memory access
  - Device kernels coordinating via LSA barriers
  - Host launches kernel; kernel performs AllReduce on-device

### [02_gin_alltoall_pure](02_gin_alltoall_pure/)
**Pure GIN AlltoAll - Network-Only Communication**
- **Pattern**: GPU kernel performs AlltoAll using only GIN for all peers
- **API**: `ncclDevCommCreate` with GIN support, `ncclGin`, GIN barriers and signals
- **Use case**: Multi-node AlltoAll with consistent network-based communication
- **Key features**:
  - Pure GIN implementation (no LSA optimizations)
  - Network barriers for cross-node synchronization
  - Signal-based completion detection
  - Baseline network performance measurements

### [03_gin_alltoall_hybrid](03_gin_alltoall_hybrid/)
**Hybrid AlltoAll - Optimized Communication**
- **Pattern**: GPU kernel performs AlltoAll using LSA for local peers, GIN for remote
- **API**: `ncclDevCommCreate` with both LSA and GIN support, peer classification
- **Use case**: Multi-node AlltoAll with optimal performance across topologies
- **Key features**:
  - Hybrid implementation for optimal performance
  - Intelligent peer classification (local vs remote)
  - Combined LSA and GIN synchronization
  - Production-ready optimized communication patterns

## Choosing the Right Pattern

*Scenario* : Custom kernels fusing computation and communication.
*Addresses* : Schedule communication from inside a CUDA kernel.
*Dependencies* : pthread or MPI

### Why the Device API?
The device API allows NCCL communication within CUDA kernels, fusing communication and computation steps:
```cpp
// Host:
// 1) Create device communicator + requirements
// 2) Register symmetric memory window for peer access
ncclDevComm devComm; ncclDevCommRequirements reqs{};
reqs.lsaBarrierCount = NCCL_DEVICE_CTA_COUNT;
NCCLCHECK(ncclDevCommCreate(comm, &reqs, &devComm));
NCCLCHECK(ncclCommWindowRegister(comm, buffer, size, &win, NCCL_WIN_COLL_SYMMETRIC));

// Device:
// - Use barriers for cross-GPU synchronization
// - Access peers via symmetric window (LSA pointers)
myAllReduceKernel<<<grid, block>>>(win, devComm);
```

## Building

### **Quick Start**
```shell
# Build example by directory name
make 01_allreduce
make 02_gin_alltoall_pure
make 03_gin_alltoall_hybrid
```

### **Individual Examples**
```shell
# Build and run the device API AllReduce
cd 01_allreduce && make
./allreduce_device_api

# Build and run the Pure GIN AlltoAll example
cd 02_gin_alltoall_pure && make
./gin_alltoall_pure_device_api

# Build and run the Hybrid AlltoAll example
cd 03_gin_alltoall_hybrid && make
./gin_alltoall_hybrid_device_api
```

## References
- [NCCL User Guide:
  Examples](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/examples.html)
- [NCCL API
  Reference](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api.html)
- [CUDA Programming
  Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
