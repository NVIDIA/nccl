<!-- SPDX-FileCopyrightText: Copyright (c) 2026 Kuaishou Technology. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->
# Contributing to TIDE

## Sign-off

Every commit carries a Developer Certificate of Origin sign-off (`git commit -s`, a
`Signed-off-by:` line matching the author), certifying the terms at
<https://developercertificate.org/>. Contributions are licensed under Apache-2.0 (`LICENSE`).
New source and script files start with the SPDX header the others carry; the offline gate checks it.

## What a change must pass

`scripts/gate.sh` runs every leg as its own process and requires exit 0; negatives inject a fault
(`TIDE_FAULT=...`) and pass only when the check catches it, so they need the fault-hook build
(`scripts/build.sh`, not a plain `pip install`).

| change | stage |
|---|---|
| any | `offline` (no GPU: wave-plan check, Python compile, shell syntax, SPDX headers) |
| reduce-scatter, FSDP2 adapter | `core1` on one node + `coremulti` on >= 2 nodes |
| expert subgroups (communicators over `r % ep`) | `ep` on >= 2 nodes |
| decoupled all-gathers | `decouple` |
| all-gather, runtime (windows, flags, credits, NIC rounds) | `1` + `multi` |
| protocol or wave planning | `all` on the largest fleet available; at <= 8 nodes the multi-wave paths run with `TIDE_RS_WAVE_NODES` below the node count, which `multi` and `coremulti` do |

Run the stages on the first host of the hostfile (`HOSTFILE`, default `/etc/mpi/hostfile`), `N`
nodes for the multi-node legs, `ONLY=<regex>` to rerun some legs, `OUTDIR` for the logs and
`gate_summary.txt`. `scripts/run_multinode.sh` launches one torchrun per node over ssh and waits for
idle GPUs first; its `NCCL_IB_*` / socket settings are those of the RoCE fabric TIDE was developed
on, override them with `EXTRA="K=V ..."`.

Performance claims come with the benchmark and its conditions (fleet, fabric, statistic, number of
iterations; interleave the arms being compared): `bench/ag_bench.py`, `bench/rs_bench.py`,
`bench/interference.py`, `bench/rs_host_gap.py`.

## Code

- C++17 host code in `csrc/`; device kernels in `csrc/collective/rs/kernels.cu` (sums, landing) and `devctl.cu`
  (the experimental device network phase). Outside that phase every wait is a stream memop or a signal wait,
  never an SM spin on a peer.
- Knobs are parsed in one place, `csrc/runtime/config.{h,cpp}`; document a new one in
  `docs/usage.md`.
- Comments state constraints the code cannot show (ordering, reuse, why a wait exists); design
  rationale and measurements go to `docs/design_*.md`.
- Docs describe the current design and its measurements (fleet, fabric, statistic), not the project's
  history: no dates, commit ids or "removed on" notes; an alternative that lost stays only as its
  measured reason.
