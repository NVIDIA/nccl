<!--
  SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0

  See LICENSE.txt for more license information
-->

# NCCL Communicator Examples

## Overview
This directory contains minimal examples that demonstrate NCCL communicator
lifecycle management (creation, query, and destruction) using different
initialization patterns.

## Examples

### [01_multiple_devices_single_process](01_multiple_devices_single_process/)
**Multiple Devices Single Process**
- **Pattern**: Single process manages all GPUs
- **API**:
  - **C**: `ncclCommInitAll` (no external coordination)
  - **Python (nccl4py)**: `nccl.Communicator.init_all()`
- **Use case**: Simple single-node applications
- **Key features**:
  - Simplest initialization method
  - No MPI or threading required
  - Automatic rank assignment (0 to n-1)
  - Cannot span multiple nodes

**Run command (C):**
```shell
./01_multiple_devices_single_process/c/multiple_devices_single_process
```

**Run command (Python):**
```shell
python ./01_multiple_devices_single_process/python/multiple_devices_single_process.py
```

### [02_one_device_per_pthread](02_one_device_per_pthread/)
**One Device per Thread with pthreads**
- **Pattern**: One thread per GPU within single process
- **API**: `ncclCommInitRank` with pthread coordination
- **Use case**: Single-node multi-GPU, thread-based parallelism
- **Key features**:
  - pthread barriers for synchronization
  - Shared memory for unique ID
  - Lower overhead than multi-process
  - Cannot span multiple nodes

**Run command:**
```shell
[NTHREADS=n] ./02_one_device_per_pthread/c/one_device_per_pthread
```

### [03_one_device_per_process_mpi](03_one_device_per_process_mpi/)
**One Device per Process with MPI**
- **Pattern**: One MPI process per GPU
- **API**:
  - **C**: `ncclCommInitRank` with MPI coordination
  - **Python**: `nccl.Communicator.init()` with `mpi4py`
- **Use case**: Multi-node clusters, distributed training
- **Key features**:
  - MPI broadcast for unique ID distribution
  - Process-to-GPU mapping by local MPI ranks
  - Scalable to multiple nodes

**Run command (C):**
```shell
mpirun -np <num_processes> ./03_one_device_per_process_mpi/c/one_device_per_process_mpi
```

**Run command (Python):**
```shell
mpirun -np <num_processes> python ./03_one_device_per_process_mpi/python/one_device_per_process_mpi.py
```

## Choosing the Right Approach

| Feature                | ncclCommInitAll | pthread          | MPI      |
|------------------------|-----------------|------------------|----------|
| **Multi-node support** | ✗               | ✗                | ✓        |
| **Process isolation**  | ✗               | ✗                | ✓        |
| **Setup complexity**   | Low             | Medium           | High     |
| **Memory overhead**    | Low             | Medium           | High     |
| **Best for**           | Simple test     | Single-node apps | Clusters |

### When to use each:
- **ncclCommInitAll**: Development, testing, simple single-node apps
- **pthread**: Single-node with thread-based parallelism needs
- **MPI**: Production distributed training, multi-node setups

## Building

### **Quick Start**
```shell
# Build all examples [or single directory]
make [directory]

# Test all examples
make test
```

### **Individual Examples**
```shell
# Build specific example
make 01_multiple_devices_single_process
make 02_one_device_per_pthread
make 03_one_device_per_process_mpi

# Test individual example
cd 01_multiple_devices_single_process/c && make test
cd 02_one_device_per_pthread/c && make test
cd 03_one_device_per_process_mpi/c && make test
```

### **Python**
See the `python/README.md` in each example directory.

## References
- [NVIDIA NCCL User Guide
  Examples](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/examples.html)
- [NCCL API
  Reference](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api.html)
- [CUDA Programming
  Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [MPI Standard](https://www.mpi-forum.org/docs/)
