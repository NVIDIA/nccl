# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# See LICENSE.txt for more license information

"""
NCCL Example (nccl4py): Multiple Devices Single Process
========================================================

Demonstrates how to initialize NCCL communicators in a single process.

Usage:
```shell
python multiple_devices_single_process.py
```
"""

import sys

import nccl.core as nccl
from cuda.core import Device, system

EXAMPLE_NAME = "multiple_devices_single_process"


def main() -> int:
    # =========================================================================
    # STEP 1: Detect Available GPUs
    # =========================================================================

    num_gpus = system.get_num_devices()
    if num_gpus < 1:
        print("ERROR: No CUDA devices found", file=sys.stderr)
        return 1

    print(f"Starting {EXAMPLE_NAME} example with {num_gpus} GPUs")

    # Display basic information for all visible devices.
    print()
    print("Available GPU devices:")
    devices = list(range(num_gpus))
    for i in devices:
        device = Device(i)
        device.set_current()
        mem_info = device.to_system_device().memory_info
        print(f"  GPU {i}: {device.name} (CUDA Device {i})")
        print(f"    Compute Capability: {device.compute_capability.major}.{device.compute_capability.minor}")
        print(f"    Memory: {mem_info.total / (1024**3):.1f} GB")

    print()

    # =========================================================================
    # STEP 2: Initialize NCCL Communicators
    # =========================================================================

    # Communicator.init_all() creates one communicator per visible device.
    print("Using Communicator.init_all() to create all communicators simultaneously")
    comms = nccl.Communicator.init_all()
    print("All NCCL communicators initialized successfully")

    print()

    # =========================================================================
    # STEP 3: Verify Communicator Properties
    # =========================================================================

    print("Communicator Details:")
    all_sizes_match = True
    for i, comm in enumerate(comms):
        rank = comm.rank
        size = comm.nranks
        device_id = comm.device.device_id

        print(f"  Communicator {i}: Rank {rank}/{size} on CUDA device {device_id}", end="")

        if rank != i:
            print(f" [WARNING: Expected rank {i}]", end="")
        if device_id != devices[i]:
            print(f" [WARNING: Expected device {devices[i]}]", end="")
        print()

        if size != num_gpus:
            print(f"WARNING: Communicator {i} has size {size}, expected {num_gpus}")
            all_sizes_match = False
    if all_sizes_match:
        print(f"All communicators have the expected size of {num_gpus}")

    print()

    # =========================================================================
    # STEP 4: Cleanup Resources
    # =========================================================================

    print("Cleaning up resources")
    # Finalize and destroy NCCL communicators
    with nccl.group():
        for comm in comms:
            comm.finalize()
    for comm in comms:
        comm.destroy()

    print()

    print("=============================================================")
    print(f"SUCCESS: {EXAMPLE_NAME} example completed!")
    print("=============================================================")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
