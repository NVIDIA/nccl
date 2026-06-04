<!--
  SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0

  See LICENSE.txt for more license information
-->

# Python (nccl4py): Ring Communication Pattern

This is the nccl4py implementation using `nccl.core` with CuPy arrays.

## Run

From this directory:

```shell
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-cu12.txt  # or: requirements-cu13.txt
python ring_pattern.py
```

## Code walk-through

- **Communicator init**: `nccl.Communicator.init_all()` creates a host-local communicator,
  returning one object per visible device, matching C's `ncclCommInitAll`.
- **Device buffers**: CuPy arrays keep the example close to familiar Python array code while
  still passing GPU buffers directly to NCCL.
- **Ring topology**: Calculates `next_rank = (i + 1) % num_gpus` and `prev_rank = (i - 1 + num_gpus) % num_gpus`.
- **Send/Recv**: `comm.send(sendbuff, peer)` and `comm.recv(recvbuff, peer)` for point-to-point communication.
- **Deadlock avoidance**: `with nccl.group():` wraps all send/recv operations to prevent deadlock
  (matches C's `ncclGroupStart`/`ncclGroupEnd`).
