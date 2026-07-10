<!--
  SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0

  See LICENSE.txt for more license information
-->

# NCCL Example: Ring Communication Pattern

This example demonstrates a ring communication pattern using NCCL P2P
operations. It runs on a single node where a single process manages all GPUs and
data flows in a circular pattern.

## Overview

The ring communication pattern creates a circular data flow where each GPU sends
data to its "next" neighbor and receives from its "previous" neighbor in the
ring.

## Runtime Requirements

This example is intended for single-node execution without any MPI or pthread parallelization. It needs at least two GPUs to demonstrate inter-GPU ring traffic.

## What This Example Does

1. **Initialize**: Detects and initializes communicators for all available GPUs
2. **Ring topology**: Each GPU calculates its next and previous neighbors using modulo arithmetic
3. **Communication**: Executes simultaneous point-to-point communication with each GPU sending to next and receiving from previous
4. **Verification**: Checks that each GPU received the expected data from its predecessor

## Variants

- **C**: `c/` (uses `ncclSend()`/`ncclRecv()` + `ncclGroupStart()`/`ncclGroupEnd()`)
  - How to run + walkthrough: `c/README.md`
- **Python (nccl4py)**: `python/` (uses `comm.send()`/`comm.recv()` + `nccl.group()`)
  - How to run + walkthrough: `python/README.md`

## Expected Output

You should see:

- GPU count and initialization
- Ring topology description (GPU 0 → GPU 1 → ... → GPU N-1 → GPU 0)
- Per-GPU send/receive neighbor information
- Successful data transfer completion
- Data verification results (CORRECT/ERROR for each GPU)

## When to Use

- **Learning NCCL fundamentals**: Understanding point-to-point communication patterns
- **Algorithm development**: Building custom collective operations based on point to point communications
- **Single-node applications**: Pipeline parallelism or custom data distribution patterns

## Common Issues and Solutions

### Issue: Deadlock without group operations
**Solution:** Always use group operations (C: `ncclGroupStart()`/`ncclGroupEnd()`, Python: `nccl.group()`) when performing simultaneous send/recv operations.

### Issue: Verification failures
**Solution:** Check ring topology calculations and data initialization patterns. Ensure correct neighbor calculations using modulo arithmetic.

## Error Handling

This example uses comprehensive error checking that immediately exits on any failure. In production code, consider more graceful error handling and recovery mechanisms.

## Next Steps

After this example, try:
- **Collective operations**: Examples in `03_collectives/`
- **Multi-node approach**: Use the MPI implementation from `01_communicators` to
  send data across nodes.
