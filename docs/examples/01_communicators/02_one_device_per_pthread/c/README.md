<!--
  SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0

  See LICENSE.txt for more license information
-->

# C: One Device per Thread (pthread)

This is the C implementation using pthreads and `ncclCommInitRank`.

## Build

From this directory:

```shell
make [NCCL_HOME=<path-to-nccl>] [CUDA_HOME=<path-to-cuda>]
```
## Run

From this directory:

```shell
./one_device_per_pthread
```

Run with specific number of threads:

```shell
NTHREADS=2 ./one_device_per_pthread
```

Run with NCCL debug output:

```shell
NCCL_DEBUG=INFO ./one_device_per_pthread
```

## Expected output

```
Using 4 devices with pthreads
Creating 4 threads for NCCL communicators
  Thread 0: Set device 0 and created stream
  Thread 1: Set device 1 and created stream
  Thread 2: Set device 2 and created stream
  Thread 3: Set device 3 and created stream
  Thread 0: NCCL communicator initialized
  Thread 1: NCCL communicator initialized
  Thread 2: NCCL communicator initialized
  Thread 3: NCCL communicator initialized
All threads synchronized - communicators ready
  Thread 0: Communicator rank 0 of 4
  Thread 1: Communicator rank 1 of 4
  Thread 2: Communicator rank 2 of 4
  Thread 3: Communicator rank 3 of 4
  Thread 0: Destroyed NCCL communicator
  Thread 1: Destroyed NCCL communicator
  Thread 2: Destroyed NCCL communicator
  Thread 3: Destroyed NCCL communicator
  Thread 0: Resources cleaned up
  Thread 1: Resources cleaned up
  Thread 2: Resources cleaned up
  Thread 3: Resources cleaned up
All threads completed
Success
```

## Code walk-through

### Key function: `ncclCommInitRank` in threads

Each thread creates its own communicator using the shared unique ID:

```c
typedef struct {
  int thread_id;
  int num_gpus;
  ncclUniqueId commId;
  ncclComm_t* comms;
} threadData_t;

void* thread_worker(void* arg) {
  threadData_t* data = (threadData_t*)arg;

  // Set device context
  cudaSetDevice(data->thread_id);

  // Initialize communicator for this thread
  ncclCommInitRank(&data->comms[data->thread_id],
                   data->num_gpus,
                   data->commId,
                   data->thread_id);

  // ... perform work ...

  // Cleanup
  ncclCommFinalize(data->comms[data->thread_id]);
  ncclCommDestroy(data->comms[data->thread_id]);
}
```

### Thread synchronization

The main thread:
1. Creates a unique ID with `ncclGetUniqueId()`
2. Spawns one pthread per GPU
3. Each thread initializes its communicator independently
4. Waits for all threads to complete with `pthread_join()`

### Important considerations

- **Device context**: Each thread must call `cudaSetDevice()` before any CUDA operations
- **Unique ID sharing**: All threads share the same `ncclUniqueId` to join the same communicator group
- **Thread safety**: NCCL initialization is thread-safe when each thread manages its own communicator
- **Cleanup order**: Always finalize/destroy communicators before destroying CUDA resources
