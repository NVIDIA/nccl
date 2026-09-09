<!--
  SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0

  See LICENSE.txt for more license information
-->

# C: Multiple Devices Single Process

This is the C implementation of the example using `ncclCommInitAll`.

## Build

From this directory:

```shell
make [NCCL_HOME=<path-to-nccl>] [CUDA_HOME=<path-to-cuda>]
```
## Run

From this directory:

```shell
./multiple_devices_single_process
```

Run with specific GPUs:

```shell
CUDA_VISIBLE_DEVICES=0,1 ./multiple_devices_single_process
```

Run with NCCL debug output:

```shell
NCCL_DEBUG=INFO ./multiple_devices_single_process
```

## Expected output

```
Found 4 CUDA device(s) available

Available GPU devices:
  GPU 0: NVIDIA A100-SXM4-40GB (CUDA Device 0)
    Compute Capability: 8.0
    Memory: 40.0 GB
  GPU 1: NVIDIA A100-SXM4-40GB (CUDA Device 1)
    Compute Capability: 8.0
    Memory: 40.0 GB
  GPU 2: NVIDIA A100-SXM4-40GB (CUDA Device 2)
    Compute Capability: 8.0
    Memory: 40.0 GB
  GPU 3: NVIDIA A100-SXM4-40GB (CUDA Device 3)
    Compute Capability: 8.0
    Memory: 40.0 GB
Using ncclCommInitAll() to create all communicators simultaneously
All 4 NCCL communicators initialized successfully

Communicator Details:
  Communicator 0: Rank 0/4 on CUDA device 0
  Communicator 1: Rank 1/4 on CUDA device 1
  Communicator 2: Rank 2/4 on CUDA device 2
  Communicator 3: Rank 3/4 on CUDA device 3
All communicators have the expected size of 4

Synchronizing all CUDA streams...
All streams synchronized
Destroying NCCL communicators...
All NCCL communicators destroyed
Destroying CUDA streams...
All CUDA streams destroyed

=============================================================
SUCCESS: Multiple devices single process example completed!
=============================================================
```

## Code walk-through

### Key function: `ncclCommInitAll`

For single-node examples we use `ncclCommInitAll` as it creates a clique of communicators in one call:

```c
int num_gpus; // num_gpus is set by querying the CUDA devices
ncclComm_t* comms;
int* devices; // devices needs to be populated with CUDA devices used

// Create communicators for all devices in one call
ncclCommInitAll(comms, num_gpus, devices);
```

This single call:

- Creates `num_gpus` communicators
- Assigns ranks 0 to (num_gpus-1)
- Sets up internal communication paths
- No unique ID distribution needed

### Comparison with `ncclCommInitRank`

`ncclCommInitAll` is a convenience function and has the same functionality as:

```c
ncclUniqueId id;

ncclGetUniqueId(&id);

ncclGroupStart();
for(int i = 0; i < num_gpus; i++) {
  cudaSetDevice(i);
  ncclCommInitRank(comms[i], num_gpus, id, devices[i]);
}
ncclGroupEnd();
```

### Cleanup

Finalize all communicators within a single group, then destroy them. With multiple
GPUs per thread, `ncclCommFinalize` is a collective and must be grouped to avoid a hang;
`ncclCommDestroy` only frees local resources and runs after the group.

```c
ncclGroupStart();
for(int i = 0; i < num_gpus; i++) {
  ncclCommFinalize(comms[i]);
}
ncclGroupEnd();
for(int i = 0; i < num_gpus; i++) {
  ncclCommDestroy(comms[i]);
}
```
