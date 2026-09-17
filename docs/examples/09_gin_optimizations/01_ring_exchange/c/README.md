<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0

  See LICENSE.txt for more license information
-->

# C: GIN Ring Exchange

This is the C/CUDA implementation of the NCCL Device API GIN ring-exchange
example.

The executable runs the optimization choices described in the example overview
on one fixed ring-exchange workload and buffer layout so the results are
comparable. Each rank sends to `(rank + 1) % nRanks` and receives from
`(rank - 1 + nRanks) % nRanks`, giving each rank one outgoing peer and one
incoming peer.

The example is split by responsibility: `kernels.cuh` contains the workload
geometry and four case-specific producer kernels, while `main.cu` contains
buffer initialization and verification, NCCL setup, timing, and result
reporting.

Each configuration has its own kernel and contains its GIN put sequence
directly. The measured kernels do not select completion behavior through a
runtime flag: the per-put kernels use the default flags for every put, while the
aggregate-request kernels mark non-final puts with
`ncclGinOptFlagsAggregateRequests`.

The comparison spans two dimensions:

- **Producer model**: CTA-cooperative puts with `ncclCoopCta` versus one
  `ncclCoopThread` put per producer thread inside each CTA.
- **Request aggregation**: weak per-put signals without versus with
  aggregate-request hints.

## Configuration Cases

The executable runs these numbered configurations:

| Case | Producer | Completion |
| ---: | --- | --- |
| 1 | `cta_coop` | `weak per-put signals` |
| 2 | `cta_coop` | `weak AggregateRequests` |
| 3 | `thread_per_put` | `weak per-put signals` |
| 4 | `thread_per_put` | `weak AggregateRequests` |

These rows map directly to `ctaCooperativePutKernel`,
`ctaCooperativePutAggregateRequestsKernel`, `threadPerPutKernel`, and
`threadPerPutAggregateRequestsKernel`, respectively.

All printed speedups are relative to configuration case 1. That row is the
simple baseline where every put uses the default flags and signals individually.

The host does not request a GIN context count. The kernels use the
communicator's default count and map each CTA to
`ctaIndex % devComm.ginContextCount`. Each CTA uses its own signal index, and
its selected context participates in the world GIN barrier and the measured
put/flush path.

The communicator requirements set `ginStrongSignalsRequired = false` because
every payload put uses `ncclGin_WeakSignalInc`; the example does not mix strong
and weak payload signals. The GIN barriers remain part of kernel coordination
and are separate from the payload completion pattern being compared.

## Producer Models

### CTA-Cooperative

The CTA-cooperative kernels loop over the puts and post each one with
`ncclCoopCta`. The puts are therefore issued serially within each CTA. The
cooperative leader performs the backend post, while the other threads
participate in the required cooperative synchronization but do not issue
independent puts. This makes the implementation straightforward, but it can be
slower because most threads are not performing useful posting work.

### Thread per Put

The thread-per-put kernels assign one put to each CUDA thread and post with
`ncclCoopThread`. This exposes the puts in a CTA in parallel and gives each
producer thread independent posting work.

## Completion Patterns

Both producer models run the same two request patterns. They move the same
payload bytes to the same destination offsets and attach a weak signal to every
put; the difference is whether non-final puts carry an aggregate-request hint.

`ncclGinOptFlagsAggregateRequests` hints to the backend that more requests are
coming. The backend decides how to use that information. For example, a backend
may delay ringing its NIC doorbell for hinted requests, then ring after a later
unhinted request so several requests become visible to the networking engine
with one notification.

### Weak Per-Put Signals

- **Purpose**: Simple baseline with straightforward per-put completion.
- **Hint behavior**: Every put uses the default flags, so no request indicates
  that more requests are coming.
- **Signal behavior**: Every put increments the CTA-owned signal with
  `ncclGin_WeakSignalInc`. Each weak signal covers only the put it is attached
  to.
- **Signal wait behavior**: The receiver waits for `PUTS_PER_CTA` signal
  increments per CTA batch, because every put advances the CTA-owned signal.
- **Source pattern**: Attach a weak signal to every put.

This path favors the simplest posting sequence and leaves request handling
entirely to the backend's default behavior.

```cpp
context.put(world, dstRank, recvWin, offset, sendWin, offset, bytesPerPut,
                ncclGin_WeakSignalInc{signalIndex}, ncclGin_None{}, coop);
```

The wait target advances by the number of puts in each CTA batch:

```cpp
context.waitSignal(coop, signalIndex, signalBase + (batch + 1) * PUTS_PER_CTA);
```

### Weak Signals with Aggregate-Request Hints

- **Purpose**: Tell the backend that related requests follow while retaining
  weak completion for every put.
- **Hint behavior**: Non-final puts use `ncclGinOptFlagsAggregateRequests`; the
  final put uses the default flags to end the hinted sequence.
- **Example backend behavior**: A backend may use the hint to defer doorbell
  ringing for non-final puts and publish the batch when it processes the final
  unhinted put.
- **Signal behavior**: Every put increments the CTA-owned signal with
  `ncclGin_WeakSignalInc`, so each signal covers only its attached put.
- **Signal wait behavior**: The receiver waits for `PUTS_PER_CTA` signal
  increments per CTA batch, confirming every put independently.
- **Source pattern**: Attach a weak signal to every put, mark the non-final puts
  with the aggregate-request hint, and use the default flags on the final put.

The non-final puts carry the aggregate-request hint while preserving their
individual weak signals:

```cpp
context.put(world, dstRank, recvWin, offset, sendWin, offset, bytesPerPut,
                ncclGin_WeakSignalInc{signalIndex}, ncclGin_None{}, coop, ncclGin_None{},
                cuda::thread_scope_thread, cuda::thread_scope_device,
                ncclGinOptFlagsAggregateRequests);
```

The final put also carries a weak signal but omits the aggregate-request hint:

```cpp
context.put(world, dstRank, recvWin, offset, sendWin, offset, bytesPerPut,
                ncclGin_WeakSignalInc{signalIndex}, ncclGin_None{}, coop,
                ncclGin_None{}, cuda::thread_scope_thread, cuda::thread_scope_device,
                ncclGinOptFlagsDefault);
```

The wait target advances by the number of puts in each CTA batch:

```cpp
context.waitSignal(coop, signalIndex, signalBase + (batch + 1) * PUTS_PER_CTA);
```

The CTA-cooperative kernel posts serially, so its final unhinted put naturally
follows all hinted puts. The thread-per-put kernel posts concurrently, so
nonzero threads post the hinted puts, the CTA synchronizes, and thread 0 posts
the final unhinted put. Thread 0 is only the designated final producer.

## Buffer Layout

The CTA-cooperative and thread-per-put producer models issue the same number of
puts per CTA. Each `ctaIndex` owns a contiguous buffer region, and each
`putIndex` inside that CTA identifies a contiguous put-sized region:

For a small illustrative case with `putsPerCta = 4`, the rank-local send and
receive windows are laid out as:

| `ctaIndex` | `putIndex = 0` | `putIndex = 1` | `putIndex = 2` | `putIndex = 3` |
| --- | --- | --- | --- | --- |
| `ctaIndex = 0` | `(0, 0)` | `(0, 1)` | `(0, 2)` | `(0, 3)` |
| `ctaIndex = 1` | `(1, 0)` | `(1, 1)` | `(1, 2)` | `(1, 3)` |
| `ctaIndex = 2` | `(2, 0)` | `(2, 1)` | `(2, 2)` | `(2, 3)` |
| `ctaIndex = 3` | `(3, 0)` | `(3, 1)` | `(3, 2)` | `(3, 3)` |

Each table cell is one contiguous put-sized region containing `elemsPerPut`
`int` values. In memory, the table is flattened row-major: all `putIndex`
regions for `ctaIndex = 0` come first, then those for `ctaIndex = 1`, and so on.

```text
idx = ((ctaIndex * putsPerCta) + putIndex) * elemsPerPut + elemIndex
```

In the CTA-cooperative kernel, `putIndex` is the loop index. In the
thread-per-put kernel, `putIndex` is `threadIdx.x`. Both kernels compute the
same byte offset:

```cpp
const size_t bytesPerPut = ELEMS_PER_PUT * sizeof(int);
const size_t offset = ((size_t)ctaIndex * PUTS_PER_CTA + putIndex) * bytesPerPut;
```

The send buffer stores an `int` validation tag derived from
`(srcRank, ctaIndex, putIndex, elemIndex)` in every payload value. Receive
validation uses the same layout, so it catches an incorrect source rank,
`ctaIndex`, `putIndex`, or element placement.

Because the validation tag is stored in a plain `int`, the example assumes a
bounded rank count. With the current constants, validation supports up to
`MAX_VALIDATION_RANKS` ranks, which is 4096. Larger jobs should use a wider
validation payload or a different tag encoding; the executable warns and exits
before running if `totalRanks` exceeds that limit.

## Workload

The example uses one fixed workload so the implementation comparison stays
easy to follow: 4 CTAs x 32 puts/CTA = 128 puts/batch, with 256 elements per put
(128 KiB per batch).

| CTAs | Puts per CTA | CTA threads | Elements per put | Warmup iterations | Timed iterations |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 32 | 32 | 256 | 10 | 100 |

## Measurement Scope

The benchmark measures elapsed kernel time with CUDA events around the whole
timed run, after a separate warmup run. The timed interval includes GIN put
posting, receive-side signal waits, `context.flush`, CTA synchronization, and
the entry/exit GIN barriers spread across `TIMED_ITERS`.

Validation results are reduced across all ranks outside the timed interval.
Rank 0 prints a case and the final result as `PASSED` only when every rank
validates its complete receive buffer; any rank that finds a mismatch prints
its local error before the aggregate result.

The printed `us/batch` value is rank 0's average time for one complete batch. A
batch is one full pass over the rank-local buffer layout. It includes every
`ctaIndex` row and every `putIndex` column in that row:

```text
CTA_COUNT * PUTS_PER_CTA
```

The workload has 4 CTAs and 32 puts per CTA, so one batch contains 128 logical
puts. The CTA-cooperative producer issues those 32 puts per CTA with
`ncclCoopCta`; the thread-per-put producer issues the same 32 puts with 32
producer threads. The `us/batch` value is the time for that whole set of puts
plus the waits, flushes, and synchronization in the timed loop.

The printed `us/put` column is an amortized value derived from the whole-batch
time:

```text
us_per_batch / (CTA_COUNT * PUTS_PER_CTA)
```

It is an end-to-end amortized cost per logical put for comparing
implementations. It is not the isolated latency of a single `context.put` call.

## Building and Running

This C/CUDA variant can be built using either pthread or MPI for
parallelization. pthread is the default choice. To use MPI, set `MPI=1` at
build time and optionally provide `MPI_HOME`.

### Build

From this directory:

```bash
make [MPI=1] [MPI_HOME=<path-to-mpi>] [NCCL_HOME=<path-to-nccl>] [CUDA_HOME=<path-to-cuda>]
```

### Run when compiled for pthreads

From this directory:

```bash
[NTHREADS=N] ./ring_exchange
```

### Run when compiled for MPI

From this directory:

```bash
mpirun -np <num_processes> ./ring_exchange
```

## Expected Output

```text
Starting GIN Ring Exchange initialization
GIN Ring Exchange: ctas=4, puts_per_cta=32, elems_per_put=256

=== Comparing GIN ring-exchange implementations ===
Timing columns: us/batch is whole-batch time; us/put is derived from us/batch / (ctas * puts_per_cta).
Speedups are relative to configuration case 1.

Configuration cases:
  Case Producer       Completion
  ---- -------------- ----------------------
  1    cta_coop       weak per-put signals  (baseline)
  2    cta_coop       weak AggregateRequests
  3    thread_per_put weak per-put signals
  4    thread_per_put weak AggregateRequests

  Case     us/batch       us/put       speedup   result
  ---- ------------ ------------ ------------- --------
  1            *.**         *.**        *.**x   PASSED
  2            *.**         *.**        *.**x   PASSED
  3            *.**         *.**        *.**x   PASSED
  4            *.**         *.**        *.**x   PASSED
GIN Ring Exchange result: PASSED
```

The launcher can print additional MPI or pthread setup lines before the example
output. Actual timings depend on GPU, NIC, topology, message size, rank count,
and GIN backend.

## Common Issues and Solutions

### Issue: Example reports that GIN is unsupported

**Solution:** Run on a system where NCCL reports GIN support through
`ncclCommQueryProperties`.

### Issue: Only one GPU or rank is available

**Solution:** Run with at least two ranks, for example `NTHREADS=2` in pthread
mode or `mpirun -np 2` in MPI mode.
