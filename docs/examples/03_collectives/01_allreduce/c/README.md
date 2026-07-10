<!--
  SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0

  See LICENSE.txt for more license information
-->

# C: AllReduce

This is the C implementation using `ncclAllReduce`.

## Build

From this directory:

```shell
make [NCCL_HOME=<path-to-nccl>] [CUDA_HOME=<path-to-cuda>]
```
## Run

From this directory:

```shell
./allreduce
```

Run with NCCL debug output:

```shell
NCCL_DEBUG=INFO ./allreduce
```

## Expected output

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

## Code walk-through

### Data Initialization

Each GPU sets a send buffer allocated on the GPU to its rank value:

```c
float** sendbuff;
float rank_value = (float)i;
size_t size; // size is the number of float to be sent

// Allocate device memory for send buffers
cudaMalloc((void **)&sendbuff[i], size * sizeof(float));

// Each GPU contributes its rank (GPU i contributes value i)
// Zero the entire buffer, then set first element to rank
cudaMemset(sendbuff[i], 0, size * sizeof(float));
cudaMemcpy(sendbuff[i], &rank_value, sizeof(float), cudaMemcpyHostToDevice);
```

### Key function: `ncclAllReduce`

The AllReduce operation combines data from all GPUs and distributes the result to all:

```c
float **sendbuff;  // Input data from each GPU
float **recvbuff;  // Output result to each GPU
size_t size;       // Number of elements to reduce
ncclComm_t *comms; // Communicators for each GPU
cudaStream_t *streams; // CUDA streams for each GPU

// Perform AllReduce with Sum operation
ncclGroupStart();
for (int i = 0; i < num_gpus; i++) {
  ncclAllReduce(sendbuff[i], recvbuff[i], size, ncclFloat, ncclSum,
                comms[i], streams[i]);
}
ncclGroupEnd();
```

### AllReduce operation

- **Input**: Each GPU provides its own data in `sendbuff`
- **Operation**: Data is combined using a reduction operation (Sum, Max, Min, etc.)
- **Output**: Every GPU receives the same result in `recvbuff`

For example, with 4 GPUs using Sum:
- GPU 0 sends: [0, 0, 0, ...]
- GPU 1 sends: [1, 0, 0, ...]
- GPU 2 sends: [2, 0, 0, ...]
- GPU 3 sends: [3, 0, 0, ...]
- All GPUs receive: [6, 0, 0, ...] (0+1+2+3=6)

### Reduction operations

NCCL supports various reduction operations:
- `ncclSum`: Sum of all values
- `ncclProd`: Product of all values
- `ncclMax`: Maximum value
- `ncclMin`: Minimum value
- `ncclAvg`: Average of all values
