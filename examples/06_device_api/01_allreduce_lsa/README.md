<!-- Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

See LICENSE.txt for license information -->

# NCCL Device API AllReduce Example

This example shows how to implement AllReduce sum operation directly in a kernel
using the NCCL device API. We first create a device communicator with
`ncclDevCommCreate` to enable kernel-initiated communication. After that,
device-side synchronization is performed with barriers and symmetric memory
windows are used to enable Load Store Accessible (LSA) memory access of peers.

## Overview

This example shows how to implement AllReduce sum operation using a GPU kernel
that directly performs the collective operations. The device communicators are
created with `ncclDevCommCreate` and device-side synchronization is ensured with
Load Store Accessible (LSA) barriers. LSA windows are used for peer memory
access.

## What This Example Does

1. **Creates device communicators** using `ncclDevCommCreate` for GPU kernel
   access to NCCL operations
2. **Registers symmetric memory windows** with `ncclCommWindowRegister` for
   direct peer-to-peer access
3. **Launches GPU kernel** that performs AllReduce sum operation entirely on
   device using LSA barriers

## Building and Running

The advanced examples can be built using either pthread or MPI for
parallelization. pthread is the default choice. To use MPI the user needs to set
`MPI=1` at build time and can optionally provide a valid MPI installation under
`MPI_HOME`.

### Build
```shell
make [MPI=1] [MPI_HOME=<path-to-mpi>] [NCCL_HOME=<path-to-nccl>] [CUDA_HOME=<path-to-cuda>]
```

### Run when compiled for pthreads (default)
```shell
[NTHREADS=N] ./allreduce_lsa
```

### Run when compiled for MPI
```shell
mpirun -np <num_processes> ./allreduce_lsa
```

## Code Walk-through

### Device Communicator Creation (Host-side)
The `ncclDevComm` is the core component of the device API, enabling GPU kernels
to perform inter-GPU communication and fuse computation with communication. The
`ncclDevCommRequirements` specifies what resources the device communicator
should allocate. In this example, we set `lsaBarrierCount` to match our thread
block count, giving each block its own barrier for independent cross-GPU
synchronization.

```cpp
ncclDevComm devComm;
ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
// Allocate one barrier per CTA we intend to launch
reqs.lsaBarrierCount = NCCL_DEVICE_CTA_COUNT;

// Create device communicator with LSA barrier support
NCCLCHECK(ncclDevCommCreate(comm, &reqs, &devComm));
```

### Memory Window Registration (Host-side)
The device API requires symmetric memory windows registered using
`NCCL_WIN_COLL_SYMMETRIC`. See the [symmetric memory
example](../../05_symmetric_memory/) for allocation and requirements details.

```cpp
ncclComm_t comm;
void* d_sendbuff;
void* d_recvbuff;
ncclWindow_t send_win;
ncclWindow_t recv_win;

// Register symmetric windows for device-side peer access
NCCLCHECK(ncclCommWindowRegister(comm, d_sendbuff, size_bytes, &send_win, NCCL_WIN_COLL_SYMMETRIC));
NCCLCHECK(ncclCommWindowRegister(comm, d_recvbuff, size_bytes, &recv_win, NCCL_WIN_COLL_SYMMETRIC));
```

### LSA Barriers (Device-side)
LSA barriers enable cross-GPU synchronization from device code. Each thread
block uses `blockIdx.x` to select its dedicated barrier, allowing blocks to
progress independently while coordinating with corresponding blocks on other
GPUs.

```cpp
// LSA barriers enable coordination between GPU threads across different ranks
// This ensures all ranks reach the same synchronization point before proceeding
ncclLsaBarrierSession<ncclCoopCta> bar {
    ncclCoopCta(),           // Barrier scope: entire CTA (thread block)
    devComm, ncclTeamLsa(devComm), devComm.lsaBarrier,
    blockIdx.x               // Barrier index: matches our CTA index (0 to lsaBarrierCount-1)
};
bar.sync(ncclCoopCta(), cuda::memory_order_relaxed);

// ...

// Release barrier ensures that we received data from everyone before we unblock the stream and allow the next kernel(s) to process the data.
// Critical for correctness in device-side collective operations
bar.sync(ncclCoopCta(), cuda::memory_order_release);
```
### Memory Access (Device-side)
`ncclGetLsaPointer` allows CUDA kernels to directly access other GPUs' memory
within the LSA team.

```cpp
// Access peer memory directly using LSA (Load/Store Accessible) pointers
float* peerPtr = (float*)ncclGetLsaPointer(sendwin, sendoffset, peer);
```

## Expected Output

```
Starting Device API AllReduce initialization
  Rank 0 using GPU device 0
  Rank 1 using GPU device 1
  Rank 2 using GPU device 2
  Rank 3 using GPU device 3
  Rank 0 initialized NCCL communicator for 4 total ranks
  Rank 1 initialized NCCL communicator for 4 total ranks
  Rank 2 initialized NCCL communicator for 4 total ranks
  Rank 3 initialized NCCL communicator for 4 total ranks
  Rank 0 initialized data with value 0
  Rank 1 initialized data with value 1
  Rank 2 initialized data with value 2
  Rank 3 initialized data with value 3
  Rank 0 created device communicator with 16 LSA barriers
  Rank 1 created device communicator with 16 LSA barriers
  Rank 2 created device communicator with 16 LSA barriers
  Rank 3 created device communicator with 16 LSA barriers
Starting AllReduce with 1048576 elements (4 MB) using Device API
Expected result: sum of ranks 0 to 3 = 6 per element
  Rank 0 completed AllReduce kernel execution
  Rank 1 completed AllReduce kernel execution
  Rank 2 completed AllReduce kernel execution
  Rank 3 completed AllReduce kernel execution
AllReduce completed. Result verification: PASSED
All elements correctly sum to 6 (ranks 0-3)
```

## When to Use

- **Kernel-level communication**: When compute kernels need immediate access to
  communication results
- **Low-latency scenarios**: Reduced host-device synchronization overhead
- **Custom collectives**: Implementing specialized reduction or communication
  patterns
- **Iterative algorithms**: Repeated communication with minimal CPU involvement

## Performance Considerations

**Advantages:**
- Lower latency for small to medium message sizes
- Eliminates host-device synchronization bottlenecks
- Enables computation-communication fusion within kernels
- Direct peer memory access without CPU copying

**Disadvantages:**
- More complex programming model requiring LSA barriers
- Requires careful memory ordering and synchronization
- Higher development complexity compared to host API
- CUDA Compute Capability 7.0+ and GPUs with P2P support (e.g., NVLink or PCI)
  required.

## Common Issues and Solutions

### Issue: NCCL warning communicator does not support symmetric memory
NCCL selects support for symmetric memory operations based on GPU connectivity.
If the GPUs on a node are only connected through e.g. the inter-CPU link,
symmetric memory will not be supported. **Solution:** Use `nvidia-smi` to
identify and select a subset of GPUs (e.g. via `CUDA_VISIBLE_DEVICES`) connected
through NVlink or PCIe.

### Issue: LSA barrier synchronization failures
**Solution:** Ensure `lsaBarrierCount` matches the number of thread blocks in
kernel launch configuration.

### Issue: Memory access violations in device kernel
**Solution:** Verify memory windows are registered as `NCCL_WIN_COLL_SYMMETRIC`
and all ranks use identical buffer sizes.

### Issue: Incomplete results or race conditions
**Solution:** Use proper memory ordering in LSA barriers
(`cuda::memory_order_relaxed` vs `cuda::memory_order_release`).

## Performance Notes

- These are educational examples, not optimized for performance
- Real implementations should use vectorization, loop unrolling, and memory
  coalescing
- Consider NCCL's optimized device kernels for best practices related to
  performance
  - NCCL library implementation of device kernels for collective operations
  - NCCL perf tests implementations of optimized device kernels

## Error Handling

The example uses comprehensive error checking for both CUDA and NCCL operations.
Device kernels should implement proper error handling for LSA operations and
memory access patterns.

## Next Steps

After understanding this example, explore:
- **Custom reduction operations**: Implement non-standard reduction patterns
- **Mixed host-device patterns**: Combine host and device API for complex
  workflows
- **Performance optimization**: Fine-tune LSA barrier usage and memory access
  patterns
