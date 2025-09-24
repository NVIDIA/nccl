<!-- Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

See LICENSE.txt for license information -->

# NCCL AllReduce Collective Operation Example

This example demonstrates the fundamental AllReduce collective operation using
NCCL's single-process, multi-GPU approach in which a single process manages all
GPUs to perform a sum reduction.

## Overview

AllReduce combines data from all participants using a reduction operation (sum,
max, min, etc.) and distributes the result to all participants. This example
shows how each GPU contributes its rank value and all GPUs receive the combined
sum using `ncclCommInitAll` for simplified setup.

## What This Example Does

1. **Detects available GPUs** and initializes NCCL communicators for all devices
   using `ncclCommInitAll`
2. **Initializes data** with each GPU contributing its rank value (GPU 0→0, GPU
   1→1, etc.)
3. **Performs AllReduce sum operation** where all GPU values are summed and
   distributed to all participants
4. **Verifies correctness** by checking that all GPUs received the expected sum:
   0+1+2+...+(n-1)

## Building and Running

### Build the Example
```bash
cd examples/03_collectives/01_allreduce
make [NCCL_HOME=<path-to-nccl>] [CUDA_HOME=<path-to-cuda>]
```

### Run with All Available GPUs
```bash
./allreduce
```

### Run with Specific GPUs
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 ./allreduce
```

## Code Walk-through

### Data Initialization
Each GPU sets a send buffer allocated on the GPU to its rank value:
```cpp
float** sendbuff;
float rank_value = (float)i;
size_t size; // size is the number of float to be sent

// Allocate device memory for send buffers
CUDACHECK(cudaMalloc((void **)&sendbuff[i], size * sizeof(float)));

// Each GPU contributes its rank (GPU i contributes value i)
// Zero the entire buffer, then set first element to rank
CUDACHECK(cudaMemset(sendbuff[i], 0, size * sizeof(float)));
CUDACHECK(cudaMemcpy(sendbuff[i], &rank_value, sizeof(float), cudaMemcpyHostToDevice));
```

### AllReduce Operation
All GPUs participate in the sum reduction. The operations are evaluated in parallel within a NCCL group to avoid any deadlocks.
```cpp
float** recvbuff;
ncclComm_t *comms;  // comms are set during ncclCommInitAll
cudaStream_t *streams; // streams are set in cudaStreamCreate

// Allocate device memory for receive buffers
CUDACHECK(cudaMalloc((void **)&recvbuff[i], size * sizeof(float)));

NCCLCHECK(ncclGroupStart());
for (int i = 0; i < num_gpus; i++) {
    NCCLCHECK(ncclAllReduce(sendbuff[i], recvbuff[i], size, ncclFloat,
                            ncclSum, comms[i], streams[i]));
}
NCCLCHECK(ncclGroupEnd());
```

## Expected Output

```
Using 4 devices for collective communication
Memory allocated for 4 communicators and streams
NCCL communicators initialized for all devices
  Device 0 initialized with data value 0
  Device 1 initialized with data value 1
  Device 2 initialized with data value 2
  Device 3 initialized with data value 3
Starting collective sum operation across all devices
Collective operation completed
Verifying results (expected sum: 6)
  Device 0 correctly received sum: 6
  Device 1 correctly received sum: 6
  Device 2 correctly received sum: 6
  Device 3 correctly received sum: 6
Example completed successfully!
```

## When to Use

- **Deep learning**: Gradient averaging in data-parallel training
- **Scientific computing**: Global reductions in parallel algorithms
- **Statistics**: Computing global sums, averages, or other reductions
- **Distributed algorithms**: Any scenario requiring collective reduction
  operations

## Key Insights
- `ncclCommInitAll` simplifies single-node multi-GPU setup
- No MPI or pthreads needed for single-node patterns
- Allocate device buffer via ``cudaMalloc` and initialize via `cudaMemset`.
- Best practices to wrap all collective calls in ncclGroupStart/End
- All communication happens in parallel

## Common Issues and Solutions

### Issue: Verification failures
**Solution:** Ensure each GPU initializes its buffer correctly with its rank
value.

### Issue: Out of memory errors
**Solution:** Reduce the buffer size in the code or use fewer GPUs.

## Error Handling

This example uses comprehensive error checking with `NCCLCHECK` and `CUDACHECK`
macros that immediately exit on any failure. In production code, consider more
graceful error handling and recovery mechanisms.

## Next Steps

After understanding AllReduce, explore:
- **Point-to-point communication**: Examples in `02_point_to_point/`
- **Other collectives**: Implement Broadcast, Reduce, AllGather operations using
  this example
- **Multi-node approach**: Use the MPI implementation from `01_communicators` to
  send data across nodes.

