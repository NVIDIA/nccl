<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0

  See LICENSE.txt for more license information
-->

# NCCL Example: GIN Ring Exchange

This example compares GPU-Initiated Networking (GIN) put optimization patterns
in NCCL Device API kernels. It runs the same ring-exchange traffic and buffer
layout across different producer models and completion patterns so their costs
can be compared side by side.

The example focuses on two choices:

- How to choose the producer granularity.
- How to provide request-aggregation hints while retaining weak per-put signals.

See the [category README](../README.md) for the shared overview of the
optimization dimensions, buffer layout, and timing interpretation.

## Overview

The benchmark compares CTA-cooperative puts against one put per producer thread
inside each CTA. It also compares the default request flags against an
aggregate-request pattern where every put carries a weak signal and non-final
puts hint to the backend that more requests are coming.

`ncclGinOptFlagsAggregateRequests` is a hint to the backend, not a prescribed
doorbell operation. One way a backend can use the hint is to delay ringing its
NIC doorbell for hinted requests, then ring after a later unhinted request to
publish the batch with one notification.

Each rank sends to `(rank + 1) % nRanks` and receives from
`(rank - 1 + nRanks) % nRanks`, so the benchmark is a ring exchange with one
outgoing peer and one incoming peer per rank.

Speedups are reported relative to the baseline configuration: producer
`cta_coop` and completion `weak per-put signals`.

## Variants

- **C/CUDA**: `c/` (creates an `ncclGin` context and calls `context.put` and
  `context.waitSignal`; also uses `ncclGinOptFlagsAggregateRequests`,
  `ncclCoopCta`, and `ncclCoopThread`)
  - How to build, run, and inspect the implementation: `c/README.md`

Note: This example is C/CUDA-only because it demonstrates device-side GIN
posting from CUDA kernels.

## Expected Output

You should see:

- GIN support checks for each rank
- The fixed workload and four implementation variants
- Rank-0 whole-batch timing, amortized per-put timing, speedup versus the
  baseline, and validation results aggregated across all ranks
