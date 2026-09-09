# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# See LICENSE.txt for more license information

"""
NCCL Example (nccl4py): AllReduce
==================================

Demonstrates how to run collective operations.
Usage:
```shell
python allreduce.py
```
"""

import sys

import nccl.core as nccl
import cupy as cp
import numpy as np

EXAMPLE_NAME = "AllReduce"


def main() -> int:
    # =========================================================================
    # STEP 1: Detect Available GPUs
    # =========================================================================

    num_gpus = cp.cuda.runtime.getDeviceCount()
    if num_gpus < 1:
        print("ERROR: No CUDA devices found", file=sys.stderr)
        return 1

    print(f"Starting {EXAMPLE_NAME} example with {num_gpus} GPUs")

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
    # STEP 3: Allocate Buffers
    # =========================================================================

    print("Allocating buffers")

    count = 32 * 1024 * 1024  # 32M floats per GPU
    size_bytes = count * 4  # 4 bytes per float32
    dtype = cp.float32

    send_buffers = []
    recv_buffers = []

    for i in range(num_gpus):
        with cp.cuda.Device(i):
            # Allocate arrays
            send_buffers.append(cp.empty(count, dtype=dtype))
            recv_buffers.append(cp.empty(count, dtype=dtype))
        print(f"  Rank {i} allocated {size_bytes / (1024 * 1024):.0f} MB per buffer")

    print()

    # =========================================================================
    # STEP 4: Initialize Data and Prepare for Communication
    # =========================================================================

    print("Filling buffers with data")

    # Initialize data - each rank contributes its rank value
    for i in range(num_gpus):
        with cp.cuda.Device(i):
            send_buffers[i].fill(float(i))
        print(f"  Rank {i} data initialized (value: {i})")

    print()

    # =========================================================================
    # STEP 5: Perform AllReduce Operation
    # =========================================================================

    print(
        f"Starting AllReduce with {count} elements ({size_bytes // (1024 * 1024)} MB)"
    )

    # Group the collective calls so NCCL can schedule them together.
    with nccl.group():
        for i in range(num_gpus):
            comms[i].allreduce(send_buffers[i], recv_buffers[i], nccl.SUM)

    # Synchronize all devices to ensure completion
    for i in range(num_gpus):
        cp.cuda.Device(i).synchronize()

    print("AllReduce completed successfully")

    print()

    # =========================================================================
    # STEP 6: Verify Results
    # =========================================================================

    expected = float(num_gpus * (num_gpus - 1)) / 2  # sum of 0 + 1 + ... + (num_gpus-1)
    result = cp.asnumpy(recv_buffers[0])

    print(f"Verification - Expected: {expected:.1f}, Got: {result[0]:.1f}")

    success = np.allclose(result, expected, atol=0.001)

    if success:
        print("Verification passed")
    else:
        print("Verification failed")

    print()

    # =========================================================================
    # STEP 7: Cleanup Resources
    # =========================================================================

    print("Cleaning up resources")

    # Finalize and destroy NCCL communicators
    with nccl.group():
        for comm in comms:
            comm.finalize()
    for comm in comms:
        comm.destroy()

    if success:
        print()
        print("=============================================================")
        print(f"SUCCESS: {EXAMPLE_NAME} example completed!")
        print("=============================================================")

    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
