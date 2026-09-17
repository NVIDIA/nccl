<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0

  See LICENSE.txt for more license information
-->

# NCCL GIN Optimizations

## Overview

This directory contains focused examples for understanding and tuning
GPU-Initiated Networking (GIN) in NCCL Device API applications. The examples
demonstrate how to structure, measure, and validate reusable GIN optimization
techniques.

Each subdirectory defines its own communication pattern, compared
implementations, workload, measurement methodology, and correctness checks.
Consult the individual example README for those details.

## Examples

### [01_ring_exchange](01_ring_exchange/)

**GIN Ring Exchange**

This example measures GIN producer models and aggregate-request hints under one
comparable ring-exchange workload. Each rank sends to one peer and receives from
one peer. The executable launches multiple CTAs, maps each CTA to
`ctaIndex % devComm.ginContextCount`, and uses one signal index per CTA.

The executable compares:

- **Producer model**: CTA-cooperative puts with `ncclCoopCta` versus one
  `ncclCoopThread` put per producer thread inside each CTA.
- **Request aggregation**: weak per-put signals using default request flags
  versus the same signals with aggregate-request hints on non-final puts.

These two dimensions form the four implementations:

| Case | Producer model | Request flags |
| ---: | --- | --- |
| 1 | CTA-cooperative | Default flags on every put |
| 2 | CTA-cooperative | `AggregateRequests` on non-final puts |
| 3 | One producer thread per put | Default flags on every put |
| 4 | One producer thread per put | `AggregateRequests` on non-final puts |

`ncclGinOptFlagsAggregateRequests` hints to the backend that more requests are
coming. A backend may use this hint to reduce notification overhead; for
example, it can delay ringing its NIC doorbell until a later unhinted request
and publish several requests with one ring.

All four implementations run the same fixed workload and use only weak payload
signals. Speedups are reported relative to the CTA-cooperative configuration
using default request flags. Rank 0 reports whole-batch time and an end-to-end
amortized cost per logical put, while validation checks the complete payload
layout across all ranks.

#### Buffer Layout

Each `ctaIndex` owns a contiguous buffer region, and each `putIndex` inside that
CTA identifies a contiguous put-sized region:

```text
idx = ((ctaIndex * putsPerCta) + putIndex) * elemsPerPut + elemIndex
```

The send buffer encodes source rank, CTA index, put index, and element index in
the payload. Receive-side validation uses the same layout, so it detects an
incorrect source rank, `ctaIndex`, `putIndex`, or element placement.

The example creates an `ncclGin` context and calls `context.put` and
`context.waitSignal`. It also uses `ncclGinOptFlagsAggregateRequests`,
`ncclCoopCta`, `ncclCoopThread`, `ncclGin_WeakSignalInc`, and
`ncclGinBarrierSession`.
