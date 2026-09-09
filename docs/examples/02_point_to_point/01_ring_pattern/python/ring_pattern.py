# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# See LICENSE.txt for more license information

"""
NCCL Example (nccl4py): Ring Communication Pattern
==================================================

Demonstrates how to use NCCL send() / recv() to implement ring communication pattern

Usage:
```shell
python ring_pattern.py
```
"""

import sys

import nccl.core as nccl
import cupy as cp

EXAMPLE_NAME = "Ring Pattern"


def main() -> int:
    # =========================================================================
    # STEP 1: Detect Available GPUs
    # =========================================================================

    num_gpus = cp.cuda.runtime.getDeviceCount()
    if num_gpus < 1:
        print("ERROR: No CUDA devices found", file=sys.stderr)
        return 1

    if num_gpus < 2:
        print("At least 2 GPUs are necessary to create inter-GPU traffic")
        print(f"Found only {num_gpus} GPU(s) - pattern will be limited")

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

    print("Ring topology")
    print(f"Data flow -> GPU 0 -> ... -> GPU {num_gpus - 1} -> GPU 0")

    print("Allocating buffers")

    count = 256 * 1024 * 1024  # 256M floats = 1GB per GPU
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

    # Fill send buffer with a recognizable pattern: first element is i*1000,
    # so after the ring transfer GPU j can verify it received from GPU (j-1).
    for i in range(num_gpus):
        with cp.cuda.Device(i):
            send_buffers[i][:] = i * 1000 + (cp.arange(count, dtype=dtype) % 1000)
        print(f"  Rank {i} data initialized (value: {i})")

    print()

    # =========================================================================
    # STEP 5: Perform Ring Communication Pattern
    # =========================================================================

    # Group the send/recv calls so NCCL can schedule them without deadlock.
    with nccl.group():
        for i in range(num_gpus):
            next_rank = (i + 1) % num_gpus
            prev_rank = (i - 1 + num_gpus) % num_gpus
            print(f"  GPU {i} sends to GPU {next_rank}, receives from GPU {prev_rank}")

            comms[i].send(send_buffers[i], next_rank)
            comms[i].recv(recv_buffers[i], prev_rank)

    # Synchronize all devices to ensure completion
    for i in range(num_gpus):
        cp.cuda.Device(i).synchronize()

    print("Ring communication completed successfully")

    print()

    # =========================================================================
    # STEP 6: Verify Results
    # =========================================================================

    success = True

    for i in range(num_gpus):
        result = cp.asnumpy(recv_buffers[i][:1])

        prev_rank = (i - 1 + num_gpus) % num_gpus
        expected = float(prev_rank * 1000)
        correct = (result[0] == expected)

        print(f"  GPU {i} received data from GPU {prev_rank}: {'CORRECT' if correct else 'ERROR'}")

        if not correct:
            success = False
            print(f"  Expected {expected:.0f}, got {result[0]:.0f}")

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
