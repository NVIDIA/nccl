<!--
  SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0

  See LICENSE.txt for more license information
-->

# RAS Fault Detection Example

This example is an interactive playground for NCCL's **RAS (Reliability,
Availability, Serviceability)** subsystem.  The program itself is
deliberately simple:

1. It runs a small AllReduce workload — **one operation per second** — so
   the job always has live collective activity for RAS to report on.
2. After `--wait` seconds (default **60**) it injects a configurable fault
   on one or more ranks.
3. It then simply stays alive for `--observe` seconds (default **120**) and
   terminates.

Everything interesting happens on **your** side: while the job runs, you
connect to it with the RAS client (`ncclras`, or plain `nc`) from another
terminal and watch the live report change as the fault is detected.

Three fault kinds are available via `--type`: two fatal ones (`exit`,
`suspend`) that end with the rank declared `DEAD`, and a recoverable one
(`sleep`) where the rank becomes a **straggler** and RAS reports a
`MISMATCH` in per-rank collective launch counts.

## What you will see

### `--type exit` / `--type suspend` (fatal fault)

| Phase | When (fault = t 0) | Report shows |
|-------|--------------------|--------------|
| HEALTHY | before the fault | All ranks `RUNNING`, errors `OK`; workload progressing |
| INCOMPLETE | from ~2s | Status still `RUNNING`, errors `INCOMPLETE`; faulted process named by PID/node/GPU, group size drops |
| DEAD | ~60s (`exit`) / ~80s (`suspend`) | Faulted process declared `DEAD` |
| end | t = `--observe` (120s) | Surviving ranks clean up and exit |

### `--type sleep` (recoverable straggler)

| Phase | When (fault = t 0) | Report shows |
|-------|--------------------|--------------|
| HEALTHY | before the fault | All ranks `RUNNING`, errors `OK`; workload progressing |
| MISMATCH | 0 – 40s | Status still `RUNNING`, errors `MISMATCH`; the `MISMATCH` warning names the lagging rank and its launch count |
| RECOVERED | 40s → `--observe` | Straggler caught up; back to `RUNNING` / `OK` |

## Build

This example requires MPI — it launches with `mpirun`/`srun` and uses MPI for
rank bootstrap and barriers.  The leaf Makefile sets `MPI=1`, so an MPI
compiler wrapper (`mpicxx`) must be on your `PATH`.

```shell
cd docs/examples/08_ras/01_fault_detection/c
make NCCL_HOME=<path-to-nccl>
```

## Run

```shell
# Default: last rank exits 60s in → INCOMPLETE within seconds, DEAD at ~60s
mpirun -np 4 ./ras_fault_detection

# Suspend mode (SIGSTOP — exercises the keep-alive timeout path, DEAD at ~80s)
mpirun -np 4 ./ras_fault_detection --type suspend

# Straggler mode: rank 7 stops launching collectives → MISMATCH, then recovers
mpirun -np 8 ./ras_fault_detection --type sleep --ranks 7

# Fail multiple specific ranks
mpirun -np 8 ./ras_fault_detection --ranks 2,5

# Allow extra time (3 min) to log in and attach the RAS client
mpirun -np 4 ./ras_fault_detection --wait 180

# Keep the faulted job around longer for leisurely exploration
mpirun -np 4 ./ras_fault_detection --observe 300
```

On startup the program prints something like:

```
════════════════════════════════════════════════════════════
  NCCL RAS Fault Detection Example
════════════════════════════════════════════════════════════
  Ranks   : 8
  Mode    : exit  (fast — socket close)
  Fault   : rank(s) 7, injected after 60s of healthy activity
  Observe : job stays queryable for 120s after the fault

  RAS is now listening on  gpu-node042:28028
    echo "verbose status" | nc gpu-node042 28028
    ncclras -h gpu-node042 -p 28028 -v
════════════════════════════════════════════════════════════
```

## Observing

RAS is queryable for the whole run — from the submitting shell (SLURM leaves
it free once the job is on the reservation) or from any node.  Find the node
the job landed on with `squeue --me`; the RAS port is `28028` by default:

```shell
# One-shot verbose report (-h host, -p port)
ncclras -h <hostname> -p 28028 -v

# The same over a raw socket
echo "verbose status" | nc <hostname> 28028

# Poll every 5s and watch the progression hands-free
watch -n 5 'echo "verbose status" | nc <hostname> 28028'

# Live event stream — shows PEER_DEAD the moment it fires
ncclras -h <hostname> -p 28028 -m

# JSON output for scripted inspection
ncclras -h <hostname> -p 28028 -f json
```

`ncclras` is the RAS client built alongside NCCL (e.g.
`<nccl-build>/bin/ncclras`).  On the job's own node these commands work as
shown — the hostname resolves to the loopback RAS binds to by default, and
`-h`/`-p` default to `localhost`/`28028` (so a bare `ncclras -v` works too).
To query from another machine, set `NCCL_RAS_ADDR` (see below).

## Options

| Option | Default | Description |
|--------|---------|-------------|
| `--type MODE` | `exit` | Fault to inject — `exit`, `suspend`, or `sleep` (described above) |
| `--ranks N[,M,…]` | last rank | Comma-separated ranks to fail; rank 0 is always preserved |
| `--wait S` | `60` | Seconds of healthy AllReduce activity before the fault — your window to connect |
| `--observe S` | `120` | Seconds the job stays alive (and queryable) after the fault |
| `--help` | | Print usage |

### Environment variables

| Variable | Default | Effect |
|----------|---------|--------|
| `NCCL_RAS_ADDR` | `localhost:28028` | RAS client bind address (host **and** port) — the runtime way to change the port. Set to `<host-ip>:28028` to allow connections from other hosts (`<host-ip>` = `0.0.0.0` accepts *any* host — mind the security exposure). |
| `NCCL_RAS_ENABLE` | `1` | Master switch for RAS. Set to `0` to disable the subsystem entirely (no daemon, no client socket) — this example has nothing to query in that case. |

The default port `28028` is a compile-time constant (`NCCL_RAS_CLIENT_PORT`).

---

## Annotated output walkthrough

The blocks below are real `ncclras -v` reports (with identifying details
masked) from an 8-rank job across 2 nodes (`srun -N2 --ntasks-per-node=4`),
failing rank 7 — the last rank, on GPU 3 of the second node.

Every report opens with a version banner and a **Job summary** (node /
process / GPU inventory), then a **Communicators** table, then the **Errors**
and **Warnings** sections.  Note the per-rank collective *launch* counts are
**not** in the summary table — they appear only in the `MISMATCH` warning
detail (see the straggler walkthrough).

### Healthy baseline

All 8 ranks reporting, one communicator, no errors or warnings:

```
$ ncclras -v
NCCL version 2.30.7 compiled with CUDA 12.8
CUDA runtime version 12080, driver version 13000

Job summary
===========

  Nodes    Processes          GPUs  Processes      GPUs
  (total)   per node   per process    (total)   (total)
      2            4             1          8         8

Communicators... (0.002s)
=============

Group     Comms      Nodes     Ranks     Ranks     Ranks   Status  Errors
    #  in group   per comm  per node  per comm  in group
    0         1          2         4         8         8  RUNNING      OK

Errors
======


Warnings
========
```

**What to notice:**

- **`RUNNING` / `OK`** — every rank is progressing and reachable over the RAS overlay.
- **`Ranks in group 8`** of an 8-rank job — all ranks running as expected.
- The **Errors** and **Warnings** sections are empty.
- Re-run the query as the job proceeds; the picture stays `RUNNING` / `OK`
  while the workload makes progress.

---

### `exit` fault — INCOMPLETE

Rank 7 has called `exit(0)`; its socket closed and RAS can no longer gather
that process's data.  The communicator **Status stays `RUNNING`** — the
surviving ranks are healthy — while the **Errors** column flags `INCOMPLETE`:

```
Communicators... (0.002s)
=============

Group     Comms      Nodes     Ranks     Ranks     Ranks   Status  Errors
    #  in group   per comm  per node  per comm  in group
    0         1          2       3-4         8         7  RUNNING  INCOMPLETE

Errors
======

INCOMPLETE
  Missing communicator data from 1 job process
  Process 3748990 on node node1 managing GPU 3

#0-0 (1bdb83b038775869) INCOMPLETE
  Missing communicator data from 1 rank
  Rank 7 -- GPU 3 managed by process 3748990 on node node1

Warnings
========
```

**What to notice:**

- **`INCOMPLETE`** in the Errors column — RAS is missing data from one
  process.  Only 7 ranks report in (`Ranks in group 7`), and one node now has
  3 ranks (`Ranks per node 3-4`).
- The Errors section names the missing process precisely — PID, node, GPU —
  both at the job level and per communicator (rank 7).
- The status is still `RUNNING`: RAS has not yet declared the peer dead.  It
  keeps retrying for 60s before a verdict, so re-querying during this window
  shows no change.

---

### `exit` fault — DEAD

60s after the socket broke, RAS declares the peer dead.  By this point the
surviving ranks have finished the run and torn down the communicator, so no
live communicator data remains to report — but the dead-peer verdict stands:

```
Communicators... (0.002s)
=============

No communicator data collected!

Errors
======

DEAD
  1 job process is considered dead (unreachable via the RAS network)
  Process 3748990 on node node1 managing GPU 3

Warnings
========
```

**What to notice:**

- **`DEAD`** — definitive and irreversible; RAS will not reconnect to this
  peer, and the dead process stays identified by PID / node / GPU.
- **`No communicator data collected!`** — there is no live communicator left
  to report here because the surviving ranks have already exited.  Earlier in
  the window (while they were still running) the table showed the reduced
  group, as in the INCOMPLETE report above.

---

## Straggler walkthrough (`--type sleep`)

In sleep mode nothing dies.  The faulted rank simply stops launching
collectives for a while, while the surviving ranks keep launching — without
waiting for completion.  NCCL launches are asynchronous, so the survivors'
per-rank launch counters (which RAS samples live) keep advancing while the
straggler's stays put.  Its RAS thread keeps answering keepalives, so the job
never becomes `INCOMPLETE` and no timeout is involved — the divergence shows
up in the very next query.

### While the straggler is behind

Status is still `RUNNING`; the Errors column flags `MISMATCH`, and the detail
lands under **Warnings**:

```
Group     Comms      Nodes     Ranks     Ranks     Ranks   Status  Errors
    #  in group   per comm  per node  per comm  in group
    0         1          2         4         8         8  RUNNING  MISMATCH

Errors
======


Warnings
========

#0-0 (78db921d1c117b4a) MISMATCH
  Communicator ranks have different AllReduce operation counts
  7 ranks have launched up to operation 64
  Rank 7 has launched up to operation 60 -- GPU 3 managed by process 1650386 on node node1
```

**What to notice:**

- **Status is still `RUNNING`** — the straggler's process is alive and
  reachable.  That is what distinguishes a straggler from the `INCOMPLETE`
  state of a crashed or frozen process.
- **`MISMATCH`** — RAS pinpoints exactly which rank is behind and by how much
  (7 ranks at operation 64, rank 7 at 60), with its PID, node, and GPU.
  Re-query and the gap grows while the straggler sleeps.
- These are **launch** counts sampled host-side, so a rank late to *call*
  `ncclAllReduce` shows up immediately — long before any timeout.  Often the
  first visible sign of a straggler in a real job.
- The detail is a **Warning**, not an Error: a mismatch caught between two
  snapshots can be transient (ranks are never perfectly in lockstep); a
  *persistent* one marks a genuine straggler.

### After the straggler catches up

Once the straggler resumes and the queued collectives drain, the launch
counts re-converge and the report returns to the healthy `RUNNING` / `OK`
state shown in the baseline above — with no trace of the earlier divergence.
Unlike `DEAD`, a `MISMATCH` is fully recoverable.

---

## Timing

The fault modes exercise different RAS detection paths.  In every timeline
below `t=0` is the moment of fault injection — `--wait` seconds (default
60) after the workload starts.

### Exit mode (fast path)

```
 t=0s    rank exits, OS closes TCP socket; survivor counts freeze
 t~0s    peer RAS threads see EOF; startRetryTime = t
 t~2s    queries now show INCOMPLETE (data missing for the dead rank)
 t~60s   60s since startRetryTime → RAS_BC_DEADPEER broadcast → DEAD
 t=120s  (--observe) surviving ranks clean up and exit
```

### Suspend mode (keep-alive path)

```
 t=0s    SIGSTOP freezes the process including its RAS thread
 t~5s    no keep-alive received → RAS flags the connection as delayed
         (INCOMPLETE); peer adds a fallback on a different ring neighbor
 t~20s   still no keep-alive → socket aborted; startRetryTime = t
 t~80s   60s since startRetryTime → DEAD
 t=120s  (--observe) surviving ranks clean up and exit
```

### Sleep mode (straggler / MISMATCH path)

```
 t=0s    straggler stops launching; survivors keep launching 1 op/s
         without waiting for completion (their kernels queue up,
         spin-waiting for the straggler)
 t~10s   queries show a growing MISMATCH, e.g. 70 70 70 60
 t=40s   straggler wakes, catches up back-to-back; queued kernels
         drain; counts converge
 t>40s   healthy 1 op/s workload resumes → RUNNING / OK
 t=120s  (--observe) all ranks clean up and exit
```

No RAS timeout participates in the sleep path — the launch counters are
sampled live, so the divergence is visible in the first query after
injection.

---

## Cleanup note

In exit/suspend mode the example calls `ncclCommAbort()` (not
`ncclCommFinalize()`) on the surviving ranks.  `ncclCommFinalize()` is a
collective; it could block indefinitely waiting for the dead peer.
`ncclCommAbort()` tears down the communicator immediately and unconditionally.

In sleep mode every rank is alive at the end, so the example performs a
normal collective teardown with `ncclCommDestroy()` and keeps MPI active for
the whole run.
