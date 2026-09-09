<!--
  SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0

  See LICENSE.txt for more license information
-->

# NCCL RAS Examples

## What is RAS?

**RAS (Reliability, Availability, Serviceability)** is a diagnostic subsystem
built into NCCL (since 2.24) that runs automatically in the background of every
NCCL job.  It requires no application code changes.

When your application calls `ncclCommInitRank`, NCCL spawns a lightweight
background thread per process.  These threads connect to each other over a
bidirectional ring on the out-of-band NIC (the same NIC used by the bootstrap)
and maintain a globally consistent view of:

- Every process in the job — hostname, PID, GPU index.
- Every communicator — its ranks, per-rank collective operation counts, and
  health status.
- Dead processes — processes that have been unreachable for over 60 seconds.

The ring operates independently of the RDMA/high-bandwidth data plane, so it
keeps working even when the data plane is sick.

You can query the current state at any time from outside the job:

```shell
# Verbose text report
ncclras -v

# JSON (for scripted analysis)
ncclras -f json

# Real-time event stream
ncclras -m

# Or use nc/telnet directly
echo "verbose status" | nc localhost 28028
```

Default listening address: `localhost:28028`.  Override with
`NCCL_RAS_ADDR=<host>:<port>`.  Disable entirely with `NCCL_RAS_ENABLE=0`.

---

## Examples

### [01_fault_detection](01_fault_detection/)

An interactive playground that:
1. Creates a communicator and keeps a small AllReduce workload running
   (one operation per second), leaving you 60s (`--wait`) to log in and
   attach the RAS client while the job is healthy.
2. Injects a fault on one or more ranks (configurable kind and rank set):
   `exit` and `suspend` kill or freeze the rank; `sleep` turns it into a
   straggler that lags behind on collective launches.
3. Stays alive and queryable for another 120s (`--observe`) so you can
   watch the progression yourself — the full `RUNNING → INCOMPLETE → DEAD`
   timeline for the fatal modes, or the recoverable `MISMATCH → OK` cycle
   for the straggler mode — then terminates.

At startup the program prints the hostname and port where RAS is listening
so you know exactly where to connect:

```shell
srun -N2 --ntasks-per-node=4 ./01_fault_detection/c/ras_fault_detection
# find the node the job landed on (RAS listens on port 28028 by default):
squeue --me
# query RAS on that node:
echo "verbose status" | nc <hostname> 28028
```

---

## Building

```shell
# Build all RAS examples
make

# Build a single example
make -C 01_fault_detection/c
```

All RAS examples require MPI (`MPI=1` is set in each leaf Makefile).

---

## Interpreting RAS output

The status/error terms below are summarized for convenience; the
[NCCL User Guide — RAS troubleshooting](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting/ras.html)
is the authoritative reference if the vocabulary changes across releases.

### Status vocabulary

| Status | Meaning |
|--------|---------|
| `RUNNING` | All ranks are progressing and reachable. |
| `INCOMPLETE` | One or more ranks did not respond to the RAS collective. Their data is missing from the report. |
| `DEAD` | One or more processes have been unreachable for ≥ 60s and are declared permanently dead. |
| `MISMATCH` | Ranks within a communicator disagree on collective operation counts or status flags — often transient during normal operation, but persistent mismatches indicate a straggler. |
| `NOCOMM` | A rank has already called `ncclCommDestroy` and left the communicator gracefully. |

### Error / warning vocabulary

| Code | Meaning |
|------|---------|
| `OK` | No issues detected. |
| `TIMEOUT` | The RAS collective leg for this rank timed out — the rank is unresponsive but not yet declared dead. |
| `DEAD` | The rank's process has been declared dead. |
| `MISMATCH` | Op counts or status differ from other ranks in this communicator. |
| `INCOMPLETE` | Global data is partially missing; report may be incomplete. |

### Operation counts

The **AllReduce / AllGather / …** columns show how many of each collective
type each rank has *launched* (counted host-side when the operation is
enqueued, not when the kernel completes).  In a healthy job all ranks in a
communicator show identical counts.  A divergence — for example `50 50 50 0`
— means rank 3 is stuck or lagging behind rank 0–2, even if RAS has not yet
timed it out.  This is often the first visible sign of a straggler, and it is
exactly what `--type sleep` in [01_fault_detection](01_fault_detection/)
demonstrates.

### The RUNNING → INCOMPLETE → DEAD timeline

```
 t=0    Fault occurs (process exits or is stopped)

        ── Exit mode (socket closes immediately) ──────────────────
 t~0    EOF on TCP socket; RAS retries connection; startRetryTime = t
 t~2s   Next ncclras query sees INCOMPLETE (data missing from dead rank)
 t~60s  60s unreachable since startRetryTime → RAS_BC_DEADPEER → DEAD

        ── Suspend mode (SIGSTOP, socket stays open) ───────────────
 t~5s   No keep-alive → connection flagged as delayed → INCOMPLETE
        RAS adds a ring fallback via an alternate neighbor
 t~20s  Socket aborted; startRetryTime = t; connection retries every 1s
 t~80s  60s unreachable since startRetryTime → DEAD
```

The straggler mode (`--type sleep`) never enters this timeline: the lagging
process stays alive and its RAS thread keeps answering keepalives, so no
timeout fires.  Instead its collective launch counter falls behind the other
ranks and the report shows `MISMATCH` (with everyone still `RUNNING`) until
the straggler catches up, at which point it returns to `OK`.

### Dead-peer semantics

- Death is **irreversible**.  Once RAS broadcasts `RAS_BC_DEADPEER`, that
  process stays in the dead list permanently for this job.  If the process
  somehow comes back, RAS will reject its reconnect attempt.
- RAS **cannot distinguish** a clean crash from a network fault — both look
  like unreachability.  The 60s threshold is the safeguard against treating
  transient network hiccups as fatal.

### What RAS does NOT do

RAS is a **diagnostic** tool.  It surfaces information but does not:
- Repair the job or reschedule work.
- Kill surviving ranks when a peer dies.
- Distinguish a hardware fault from a software crash.

---

## References

- [NCCL User Guide — RAS troubleshooting](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting/ras.html)
- [NCCL User Guide — Environment variables](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html)
- [NCCL API Reference](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api.html)
