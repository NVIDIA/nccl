<!-- SPDX-FileCopyrightText: Copyright (c) 2026 Kuaishou Technology. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->
# TIDE usage reference

How each collective runs, its Python API, knobs and tests. Start with the [README](../README.md);
the design rationale and measurements are in `design_ag.md` and `design_rs.md`.

## How the all-gather runs

Single node: every rank writes its own pieces in place (or stages the packed
shard), then one `cudaMemcpyBatchAsync` per call scatters them — to the NVLS
multicast address when the window is NVLS-bound (one write lands on every
GPU), else peer-to-peer into each LSA peer's window. Entry/exit fences are
`cuStreamBatchMemOp` waits on a flag window.

Multi node (rail-aligned, `csrc/collective/ag/issue.cpp: issueMultiNode`): the caller
writes its packed shard into stage slot 0 of a stage area of *M* slots; slot
*i* holds the shard of the same-index GPU on node+*i*. Round *r* = 0 ..
ceil(log2 *M*)−1 puts everything held so far -- slots [0, 2^*r*) -- with
`ncclPutSignal` into the slots [2^*r*, 2^(*r*+1)) of node−2^*r*'s same-index GPU
(one chunk per RMA context, so the transfer is striped over as many QPs) while
receiving the same block from node+2^*r* (Bruck's partial last round for
non-power-of-two *M*): recursive doubling, ceil(log2 *M*) rounds for the same
bytes as an *M*−1 round rotation, one
incoming flow per NIC at a time. The rounds are paced
(`TIDE_NIC_PACE`, default on for shards of at least two leads): round *r*+1's
source is released by a signal that leaves once all but
`TIDE_NIC_PACE_LEAD_MIB` (8) of round *r* have landed, so the signal's round
trip overlaps the round's tail; this keeps a drifting node from putting a
second full flow into a busy port and triggering RoCE congestion control
(unpaced, 6 nodes x 400G: 10 M ECN marks and a 1.5-2x median; paced: 16 k).
As a block arrives (`ncclWaitSignal`) the copy engines scatter it into the
param-major landing across the node -- consecutive slots are consecutive
ranks, so a segment's pieces are one strided descriptor. Signals carry counts,
not identities, so buffer reuse is tracked explicitly: every distinct stage
area owns a credit channel (signal index 2 + slot), the receiver signals it
once the whole call's slots have been scattered, and a source waits for the
credits of its previous put into that area before overwriting it. Every byte
crosses a NIC once; the collective launches no kernel.

All calls on one communicator must be issued in the same order on every rank;
each call is ordered behind the previous one whatever stream it is issued on.
`comm.close()` ends a communicator at a program point of your choosing
(`ncclCommDestroy` barriers within the node, so it must not be left to garbage
collection); `comm.free_arena(t)` forgets a window and its registrations.
A TIDE communicator never runs a collective kernel. Device memory outside torch's allocator, 8 nodes x 8
GPUs, 64 ranks: with NCCL's default runtime connect (`NCCL_RUNTIME_CONNECT=1`) the communicator costs
0.21 GiB per GPU (0.36 at 2 and 4 nodes: it does not grow with the node count), ~14 MiB per RMA
context, nothing per signal index, and the first put allocates nothing more. It is created with a
one-channel cap (`TIDE_COMM_CTAS`, 0 = NCCL's default), so forcing eager connect (`=0`) preconnects
one channel's collective buffers: 0.66 GiB. `NCCL_MIN_NCHANNELS` in the environment overrides the cap
(`=32`: 4.87 GiB), and the process group of the framework grows under it too (torch's own, on NCCL
2.31.2: 5.6 GiB eager against 1.1 lazy), so keep lazy connect; TIDE warns when eager connect is on.
None of this shows in `torch.cuda` memory stats or snapshots (the arenas do: they come from a torch MemPool on
the process group's NCCL allocator unless `TIDE_ARENA_TORCH=0`): `comm.device_bytes` is
what the communicator took at construction, and `tide.fsdp.install()` logs the arena (ring slots, dedicated
slots, the sharded parameters moved into it, reduce-scatter area) and the communicators.
Start-up time is logged on rank 0 as well: `TideComm()` prints how long the unique-id broadcast, `ncclCommInitRank`,
the first window registration and the config check took, and `install()` prints its phases (checks, collect, allocate
split into arena + symmetric windows / reduce-scatter tuning / registration / slot binding, patch). Both run on the
calling thread after `fully_shard`. NCCL 2.31's host RMA connects at a communicator's first window registration:
every RMA context (`TIDE_RMA_CTX`, 4) opens a queue pair to **every rank of that communicator** with a ring barrier
over it after each (`src/rma/rma_proxy.cc` `ncclRmaProxyConnectOnce`, `src/transport/net_ib/gin.cc`
`ncclRmaIbProxyCreateContext`), ~`ctx x ranks^2`. TIDE only ever puts to its M - 1 rail peers, so the main
communicator is created without RMA contexts (its windows still give the LSA peer pointers and the multicast
address) and the RMA runs on a rail communicator split from it (`ncclCommSplit`, one rank per node, rank = node), which
is where the connect happens: M ranks instead of M x L (on the full communicator it took 81-184 s at 512 ranks; the
`rail split` term of the start-up log shows the rail one). Fewer contexts would cut it further but cost bandwidth: on
400G single-port rails at 8 nodes a 298 MiB all-gather takes 49 / 54 / 74 ms with 4 / 2 / 1 contexts (the proxy
chains an atomic signal behind each write and polls every rank's queue), so the context count is not a start-up knob.
The NIC leg runs on a rail communicator (one rank per node, split from the main one, which keeps no RMA
contexts): each RMA context connects to every rank of *that* communicator with one queue pair per direction
(NCCL's host RMA), so a rank holds about `2 * num_rma_ctx * M` QPs, the M-1 rail peers it puts to plus itself.

## Reduce-scatter (RS phase)

`comm.register_rs(arena, stage_off, grads or (shapes, itemsize))` / `handle.reduce_scatter(grads, out=, scale=, accumulate=)`
reduces FSDP2's unsharded gradients (bf16 or fp32) read in place into the packed fp32 layout of
`torch._chunk_cat + reduce_scatter_tensor`; no input window, no chunk_cat, no producer contract.
Inside a node every rank pushes its pieces for the other seven owners into their staging slot
with the copy engines (bf16 on the wire) and the owner sums the eight pieces in fp32 on a few
CTAs (TMA-streamed, fixed source order, bitwise reproducible run to run in our tests); across nodes fp32 partial sums
travel over the rail exactly like the all-gather's shards. The cross-node sum is a **multicast
landing** (NVLS multimem, required on multi-node): each node's partial is computed on the
local rank that makes it land, at the same offset, on a different local rank of the owner's node,
and the owner reads the sum of a wave of up to 8 nodes with one `multimem.ld_reduce` per 16 B
(fp32 add in the switch), accumulating wave after wave (double-buffered inbox, any node count)
-- no chain of add kernels, a fixed handful of CTAs whatever the node count. The landing
applies `scale` (AVG) and `accumulate` (gradient-accumulation microbatches). Every wait is a
stream memop or a signal wait. See `docs/design_rs.md`; tests in `test/test_rs_bitwise.py`,
benchmark `bench/rs_bench.py`. Needs nvcc (the extension is a `CUDAExtension` since the RS
kernels) and an sm_90+ GPU with ~207 KB of shared memory per block: the sum kernel streams its
sources with `cp.async.bulk` (TMA), and registration refuses a device without it.

Pushes for consecutive destination ranks travel as one strided copy descriptor (`cudaMemcpy3DBatchAsync`)
from 4 rows on and as flat ones below (a strided op costs the host 1.7 us whatever its size but runs
alone on a copy engine; a flat one 0.75 us): in multicast mode the 8 destinations of a node round are
one descriptor per (owner, segment) -- 6 nodes x 16 segments, host per call 2.0 -> 0.8 ms. Tables
sharing a stage area may be registered in any order before the area's first call; size the area
with `comm.rs_area_bytes(tables)`.

Sizing: `comm.tune_rs(tables)` (collective, before the first registration; the FSDP2 adapter calls it)
sets the sum / landing CTAs, k and the staging chunk from `tide.solver` for the largest table, with the
rates in `$TIDE_CALIB` (a `bench/calibrate.py` profile) or the built-in ones (400G RoCE); `comm.rs_tuning`
shows what the next registration uses. At 8 nodes x 400G it picks 4 + 4 CTAs, k = 2 or 4 (a NIC round waits
~1.3 ms for its peers whatever it carries, so k doubles from 2 while the rounds' waits pass 10% of the NIC time and
doubling cuts rounds, up to the 4 peers the device network phase takes: 64 nodes x a 74 MiB output, k = 4 in 19
rounds beat k = 2 in 33 by 4-5%; 8 nodes x a 592 MiB one, k = 2 in 4 rounds beat k = 4 in 3 by 6-8%) and a staging
chunk of at most 8 MiB on multi-node (whole sub-slices of a large layer at a small rank count made a 10 GiB stage area);
once the inbox of whole sub-slices would reach 2 GiB (8 nodes x a 592 MiB shard: 4.6 GiB) it runs 4 passes over the
sub-slices at k = 2 (`comm.rs_parts`). Without it the extension's defaults apply; a sizing knob set in the
environment wins over the solver ([Environment variables](#environment-variables)).

Release (efficiency mode, the default; throughput mode copies the gradients instead, see [FSDP2](#fsdp2)): TIDE reads
the gradients in place (no copy-in), so they live until the call has read them. The adapter plans
the multi-node calls pipelined (`tune_rs(read_early=True, pipeline=2)`): a read phase (pushes and level-1 sums into an
outbox bank -- one pass, staging chunks up to 32 MiB, an outbox block per remote destination, so the sums read every
destination's inputs back to back -- then `read_event`) that waits for the gradients' ready event only, and a network
phase on the caller's stream behind the previous call's; two outbox / own-spare banks alternate, so call e+1 reads
while call e is on the NIC. FSDP2 keeps the gradients until its next post-backward waited on `read_event` (its own
reference-plus-event release: by then the reads are long done, so the wait costs nothing and a slow rank's jitter is
absorbed), which keeps one layer of gradients alive across the next layer's backward (freeing them from a hook inside
the next layer instead made every layer wait for its slowest reader). Lanes with one rank per node read their inputs in
every round of the call and keep FSDP's release at the next post-backward, after the whole call.
Before the next group's reduce-scatter the adapter issues FSDP2's own next backward all-gather
(`TIDE_FSDP_EARLY_PREFETCH`), so on a merged comm stream it runs ahead of the network phase. Set
`CUDA_DEVICE_MAX_CONNECTIONS=32` before CUDA starts (the adapter warns otherwise): with the default 8 hardware queues
the next call's pushes queue behind the previous network phase's stream waits and the two phases serialize. The sum
and landing streams run two levels above the default priority (`TIDE_RS_KERNEL_PRIORITY`): the reads overlap the
backward's full-device GEMMs, whose CTAs leave no room for the sum kernel's, and the node's staging slots (2,
`TIDE_RS_STAGE_SLOTS`) stall together behind one late sum; persistent GEMMs hand the SMs to a higher-priority kernel at
tile boundaries.

`TIDE_RS_NET=device` (experimental; default `host`; pipelined calls, GDAKI GIN required) runs the network phase in
device kernels: one 1-CTA kernel per landing wave on a highest-priority stream does the wave's rounds (credits, puts
with a strong signal on each context's last put to a peer, arrivals, flush, the next round's credits) and a final kernel
the next call's credits, so the host issues a handful of launches instead of the rounds' NCCL calls and stream memops
(which queue up to hundreds of ms per call behind a full command buffer). Same plans and sums: bitwise. A device wait
longer than `TIDE_RS_NET_TIMEOUT_S` records the fault, never returns (no completion event fires, so nothing downstream
consumes a partial result) and a monitor thread exits the process with status 70, first sending SIGKILL to the process's
own children (data-loader workers would otherwise hold the launcher's output pipe open and keep the job alive). The wave
kernel occupies one SM while it waits: a persistent kernel launched with one CTA per SM (each filling an SM) has to
leave one free, or its last CTA waits out another CTA's whole loop (e.g. a MoE router GEMM). The host path is the
default and the one the technical report measures; the device path passes the regression gate but has run far less
training.

The device network phase sends on NCCL's GIN GDAKI QPs, which DOCA creates over DEVX: the kernel never applies an mlx5
port's global traffic class (`/sys/class/infiniband/<dev>/tc/<port>/traffic_class`) to them. Where that class moves
RoCE onto the PFC priority, GDAKI traffic would stay on class 0, lossy: with other RDMA traffic on the NICs its packets
are dropped, a QP ends in a retry-exceeded error (CQE syndrome 0x15) and the flush that waits on it never returns.
`TideComm` therefore sets `NCCL_GIN_IB_TC` to that global class when the caller has not set it (NCCL reads it when GIN
connects); the class appears on the `[TIDE-RS net]` startup line.

Limits: at most 8 ranks per NVLink domain, node-major ranks, identical call sequence on every rank.

## Known limitation: NCCL 2.31.2 PAT and AVG

On a communicator with one rank per node (an expert subgroup, for instance) NCCL 2.31.2's PAT
reduce-scatter (2.32.3's too) drops the scaling of AVG and PreMulSum and returns the plain sum. TIDE's own
reduce-scatter does not use NCCL's, but the stock collectives in the same process do (FSDP groups
TIDE leaves to stock among them): exclude PAT, e.g. `NCCL_ALGO=^PAT` or any list without it.

## FSDP2

`tide.fsdp.install(model, comm, slots=2, reduce_scatter=True)` (after `fully_shard` of the root
`model`, after any other FSDP patch, before the first forward) replaces `foreach_all_gather` /
`foreach_all_gather_copy_out` (and `foreach_reduce`): each parameter group is pinned to an arena slot
at its first all-gather and its unsharded parameters become views of the
landing, so copy-out is gone. Groups that reshard after forward share `slots`
ring slots -- two, ping-pong: with prefetch depth 1 only the computing and the
prefetched group are unsharded at once, and the slot a prefetch reuses was
released before the prefetch is issued (a deeper explicit prefetch needs more
and fails loudly otherwise); groups that stay unsharded (including the root
when FSDP decides so at lazy init) get a dedicated slot. A slot is a landing area followed, on
multi-node, by a receive-stage area at a fixed offset, so a stage never overlaps a
landing that another group may still be computing on. Without expert groups the ring slots share one
receive stage (`TIDE_FSDP_SHARED_STAGE`), a stage of M packed shards less per lane: overlapping stage
areas share a credit channel, so a group's remote writes wait until the previous ring all-gather has
scattered out of it, which under FSDP2's prefetch order it already has. The sharded parameters of a
TIDE group live in the arena (moved there once at binding, the parameter objects and
optimizer state untouched; `uninstall()` moves them back): every all-gather reads them
in place, so there is no copy-in at all -- a group that casts (fp32 shards, bf16
all-gather) keeps the cast kernel, which writes the packed shard. `install()` first checks FSDP2's internals it patches (`tide.fsdp.compat_problems`: signatures of
the patched entry points, `AllGatherResult` fields, per-parameter attributes) and raises with the list
on a mismatch (`allow_fallback=True`: log and stay stock). A slot is rewritten only after the
compute stream released the previous occupant and the previous all-gather on
it completed. With `reduce_scatter`, the backward reduce-scatter reads the
unsharded gradients in place and lands the packed fp32 shard grads; an
accumulating microbatch reduces straight into the existing grads. Under HSDP the
shard group's reduce-scatter runs on TIDE and the replica all-reduce on the stock
collective, after which the microbatch is accumulated. A parameter with an FSDP2
all-gather extension (`fsdp_pre_all_gather` / `fsdp_post_all_gather`, e.g. an MXFP8
weight gathered as E4M3 payload + E8M0 scales) is served as one segment per input:
the inputs are rebuilt (`FSDPParam.all_gather_inputs`) and copied in at every call,
each lands in its own area, and FSDP's post hook runs over the landed views; the
inputs' dtypes and numels must not change between calls (`TIDE_FSDP_EXTENSIONS=0`
keeps such groups on the stock path). Groups TIDE declines (CPU offload, non-dim-0
sharding, post-forward meshes, HSDP microbatches without the replica all-reduce)
keep the entry points that were in place at install.
`tide.fsdp.uninstall()` restores them and frees the arena; `install()` may
follow again. `test/test_fsdp_adapter.py` trains a toy stack with stock FSDP2
and with TIDE and requires bitwise-equal losses and parameters (1e-4 and
run-to-run bitwise with `--rs`, whose sum order differs from NCCL's), also
across `--uninstall-after N [--reinstall]` and `--accum N` microbatches.

Stream ordering follows FSDP2's own conventions and adds nothing per call. An
all-gather is ordered after the optimizer by FSDP2's root pre-forward wait and
after its slot's previous occupant by the slot release event; ordering every
all-gather behind the compute stream's queued work would serialize FSDP2's
implicit prefetch. The adapter's all-gathers are decoupled: a call leaves the caller's stream once its
writes are issued, the done fence (every local rank's writes into my landing, the node's slowest rail) and
the slot's event run on an internal stream, the next call's `ready` waits for the slot's release instead
of the stream, and on multi-node the stage credits leave at the next call (a cast kernel, which writes the
packed shard into the stage, first waits for this rank's previous scatters out of it); compute waits on the slot's
event, so the comm stream no longer re-synchronizes the node on every call. The reduce-scatter returns the unsharded gradients to FSDP2
as the input it keeps until its compute stream has waited on the reduce-scatter
event, the way stock FSDP2 holds its packed input, so they are freed in stream
order; `record_stream` instead would hold them until the GPU reaches the free,
which with the host running ahead in backward keeps one more layer of gradients.

Two modes. The default, **efficiency mode** (*In-Place* in the technical report), reads the gradients in place as
above. **Throughput mode** (*Slot-Reuse*; `TIDE_FSDP_COPYIN=1`, the same on every rank) is for runs whose communication hides behind compute or that need the
memory for a larger batch. At a group's post-backward the adapter copies
the gradients on the compute stream into the slot the group has just resharded out of, so FSDP2 frees them right after
the copy and no layer of gradients outlives its post-backward; the reduce-scatter reads the slot, whose release event
is recorded again after those reads, so the slot's next all-gather (its own writes and, through `ready`, the peers')
waits for them. The cost is the copy on the compute stream; the early backward all-gather is off, the next all-gather
into the slot following the reduce-scatter (on one comm stream it queues behind the whole network phase). Nothing being
freed early, the reduce-scatter is planned without the early reads (`tune_rs(read_early=False, pipeline=2)`): each
outbox bank holds two NIC rounds instead of a block per remote node, the reads after the first two rounds follow the
NIC and the slot stays held until they end. Groups
without a slot, groups that stay unsharded after backward, all-gather extension groups and gradients of another dtype
than the slot (fp32-accumulated ones) keep the in-place read.

### Expert groups

`install(model, comm, subgroups=True)` also serves the parameter groups whose process group is a regular subset of
the ranks: ascending, the same number on every node, node-major -- the subgroups `r % ep` over which an
expert-parallel model shards each expert's weights with FSDP2 (torchtitan's `dp_shard_mod_ep`). Each subset gets its
own lane: a `TideComm` over its process group (a second one when `install()` was given a separate reduce-scatter
communicator), its own arena and ring slots, beside the lane of `comm`. The lanes are created while `install()`
collects the groups, collectively over each subset (every rank reaches the same module in the same order), and
closed by `uninstall()`; a subset that is not regular logs why and stays on the stock path.

A subset with several ranks per node runs the paths above. With one rank per node (`ep` equal to the GPUs per node)
nothing is left inside a node: the all-gather runs the same recursive doubling from landing to landing (no stage, no
scatter; a round's put waits for its target's `ready`), and the reduce-scatter packs each destination's pieces with
the copy engines into an outbox block per k-way round, puts them into the owner's inbox (two parities alternate) and
folds each round into the fp32 output on the sum stream, in one phase (`TIDE_RS_RAIL_PARTS` passes give the same
result; `TIDE_RS_NET=device` applies to the multi-rank lanes only). On such a lane the round width regroups the
cross-node sum ([numerics.md](numerics.md)). Both modes run on every lane. With expert groups the ring slots keep a
receive stage each (`TIDE_FSDP_SHARED_STAGE` defaults to 0), and the stock collectives in the process need PAT off
([known limitation](#known-limitation-nccl-2312-pat-and-avg)). `scripts/gate.sh ep` runs the all-gather,
reduce-scatter, fence and FSDP2 legs over `ep` = 8, 4 and 2 (8 is one rank per node).

## Tests

The regression gate is `scripts/gate.sh offline | core1 | coremulti | ep | decouple | 1 | multi | all` (run on the
hostfile's first host; `N` nodes for the multi-node legs, `ONLY=<regex>` to rerun some): every leg must exit 0,
negatives included, and `gate_summary.txt` lists the verdicts. `offline` needs no GPU (the exhaustive wave-plan
check and a Python compile); `core1` + `coremulti` is the short gate every reduce-scatter or adapter change must
pass, `ep` adds the expert subgroups and `decouple` the decoupled all-gathers (CONTRIBUTING.md).

`torchrun` scripts under `test/` (multi-node through `scripts/run_multinode.sh`; the reduce-scatter test is `test_rs_bitwise.py`, negatives `TIDE_FAULT=nowrite`, `nofence,slowpush`, `nofree,slowsum`, `nocredit,slow`, `nodone,slow`, `nozero`):
`test_ag_bitwise.py` (byte-exact against stock all-gather + permutation),
`test_fence.py` (two-stream FSDP-like schedule without host sync, lane reuse
in round-robin or burst order, a slow consumer), `test_concurrent_rs.py`
(TIDE all-gather alongside torch/NCCL reduce-scatter), `test_fsdp_adapter.py`,
`test_rs_invariance.py` (sum/landing CTAs, k and chunk leave the result bitwise unchanged),
`test_rs_nonfinite.py` (NaN / ±Inf propagate as in an fp64 reduce-scatter), `test_abort.py` (a call that fails mid-protocol leaves a
communicator that `close()` aborts instead of draining; `TIDE_FAULT=abort`). `test_fsdp_adapter.py --no-mp --vocab 4099`
puts padded shards in the root's dedicated group; `--stock-between N` runs stock steps between uninstall and reinstall.
Negatives are fault injections that must be caught: `TIDE_FAULT=nowrite`,
`nofence`, `nodone,slow`, `nocredit,slow` (`slow` delays one rank's data
movement by `TIDE_FAULT_US` on `TIDE_FAULT_RANK`; alone it is the matching
positive). The pipelined and device-network legs (`pipe_*`, `dc_*`, `fsdp_async*`) rerun the reduce-scatter and
adapter checks in those modes; the `fsdp_copyin*` legs (`test_fsdp_adapter.py --copyin`) a second run with copy-in,
bitwise against the in-place one, whose negative `TIDE_FAULT=noreadwait,slowpush` lets the slot's next all-gather skip
its wait for the reads; `dc_stall` (`TIDE_FAULT=dcstall` on one rank) must end the job within its deadline
with a device-wait fault logged. The hooks exist only in builds with `TIDE_FAULT_INJECTION=1`
(`scripts/build.sh`); a plain `pip install` leaves them out and refuses to
start with `TIDE_FAULT` set.

## Profiling

The extension marks its host work with NVTX ranges in a `TIDE` domain, one color per phase: `tide.ag` (with
`tide.ag.sync`, `tide.ag.round`, `tide.ag.tail`, `tide.ag.credits`) and `tide.rs` (with `tide.rs.read`, `tide.rs.net`,
`tide.rs.wave`, `tide.rs.land`, `tide.rs.tail`), and names its internal streams (`TIDE ag scatter / ready / done`,
`TIDE rs push 0..3 / sum / land / zero / read event / net`). Nsight Systems records them with `-t cuda,nvtx` and projects
the ranges onto the GPU rows over the work they launched. The FSDP2 adapter wraps each all-gather and
reduce-scatter call in a torch profiler user annotation (`tide::ag <group>`, `tide::rs <group>`), which the torch
profiler spans over the GPU work the call launched. Without a tool attached a range costs one branch.

## Environment variables

The extension's knobs are parsed in `csrc/runtime/config.cpp` when a communicator is created, the Python ones
(`tide/comm.py`, `tide/fsdp.py`) when used; unset means the default.

### Deployment

| Variable | Default | Effect |
|---|---|---|
| `TIDE_CALIB` | built-in rates (400G RoCE) | `bench/calibrate.py` profile the solver sizes the reduce-scatter from |
| `TIDE_RS_NET` | host | `device` (experimental): the network phase of pipelined calls in GIN device kernels (needs GDAKI; `NCCL_GIN_IB_TC` defaults to the NICs' global traffic class); `host`: rounds issued by the host |
| `TIDE_RS_NET_TIMEOUT_S` | 300 | `device`: a device wait longer than this is a protocol fault: the process exits (70). Keep it above the job's collective timeout, or a stalled peer the rest of the job would wait out ends it |
| `TIDE_FSDP_COPYIN` | 0 | 1: throughput mode, the FSDP2 adapter copies each group's gradients into the slot it has just left and frees them there (one layer of gradients less, one copy more; [FSDP2](#fsdp2)); 0: efficiency mode. The same on every rank |
| `TIDE_FSDP_EXTENSIONS` | 1 | 0: groups with an FSDP2 all-gather extension stay on the stock path |

Retired: `TideComm` refuses to start with `TIDE_RS_DEVCTL*` (now `TIDE_RS_NET*`), `TIDE_FSDP_AG_DECOUPLE` (the
adapter's all-gathers are always decoupled), `TIDE_FSDP_EARLY_RELEASE` or `TIDE_FSDP_RELEASE_MODE` set.

### Tuning, calibration and debug

The solver (`comm.tune_rs`, which the FSDP2 adapter calls) or an automatic rule picks the sizing knobs, the rest are
fixed defaults; a value set in the environment wins. For calibrating a new fleet and for debugging: these may change
meaning or go away; do not set them in production.

| Variable | Default | Effect |
|---|---|---|
| `TIDE_FSDP_SHARED_STAGE` | 1 without expert groups, 0 with them | 1: the ring slots share one receive stage |
| `TIDE_ARENA_TORCH` | 1 | arenas from a torch MemPool, in torch's memory statistics; 0: TIDE's own `ncclMemAlloc` |
| `TIDE_RMA_CTX` | 4 | RMA contexts (QPs) a shard is striped over; `TideComm(num_rma_ctx=)` > 0 wins. Each context connects to every rank of the rail communicator at its first window registration (start-up ~ctx x nodes^2, see above); fewer contexts lose bandwidth (1: -33% on a 298 MiB all-gather at 8 nodes) |
| `TIDE_RS_CTAS` | solver; else 4 (8 on a single node, 12 with one rank per node) | sum CTAs |
| `TIDE_RS_MC_CTAS` | solver; else 4 | CTAs of one wave's multicast landing, in total over its blocks (0 skips it: debug) |
| `TIDE_RS_KWAY` | solver (2 or 4); else 2 | remote destinations per NIC round (at most 4 with `TIDE_RS_NET=device`) |
| `TIDE_RS_CHUNK_MIB` | solver; else 16 | staging chunk hint per source (the chunk count nearest to span / hint, so up to 1.5x) |
| `TIDE_RS_PARTS` | solver; else 1 | passes over the sub-slices on multi-node, the same result bitwise; ignored when a landing wave would hold the own node alone (M = 1 mod the wave size, min(`TIDE_RS_WAVE_NODES`, ranks per node)); lanes with one rank per node use `TIDE_RS_RAIL_PARTS`. Each pass queues its own NIC and staging rounds: many passes at k = 1 block the host in `cuStreamBatchMemOp` |
| `TIDE_RS_OUTBOX` | solver; else two rounds' worth (pipelined: full bank) | outbox blocks per bank (at least two consecutive NIC rounds, up to the remote destinations of a call; each an fp32 block of 1/parts sub-slice): efficiency mode asks for a full bank so the sums read every destination back to back, throughput mode for two rounds' worth per bank |
| `TIDE_RS_STAGE_SLOTS` | 2 | local-staging slots (2..8), each L x L staging chunks; more let the pushes run ahead of the sums, so the reads end earlier (the solver sizes the chunk for two, the others add their staging; 4 measured within noise of 2 at 8 nodes in both FSDP2 modes, for 8 S more memory) |
| `TIDE_RS_KERNEL_PRIORITY` | 2 | levels above the default stream priority for the sum and landing kernels (capped at the device's greatest); 0: the default priority |
| `TIDE_RS_PIPELINE` | 1 (2 from the FSDP2 adapter) | 2: multi-node calls in a read and a network phase with two outbox / own-spare banks; 1 under the adapter: one-phase calls, its tests' bitwise reference |
| `TIDE_FSDP_EARLY_PREFETCH` | 1 | issue FSDP2's next backward all-gather before a pipelined reduce-scatter's network phase (off with `TIDE_FSDP_COPYIN=1`) |
| `TIDE_RS_RAIL_PARTS` | 4 | staging parts of the one-rank-per-node reduce-scatter |
| `TIDE_NIC_PACE` | auto | all-gather round pacing: 1 always, 0 never, auto: shards of at least two leads |
| `TIDE_NIC_PACE_LEAD_MIB` | 8 | bytes of a round still in flight when its pace signal leaves |
| `TIDE_NIC_GROUPS` | 4 | slot ranges the last all-gather round is sent and scattered in |
| `TIDE_COMM_CTAS` | 1 | channel cap of TIDE's NCCL communicator (0: NCCL's default) |
| `TIDE_STRIDED_ROWS` | 4 | rows from which one strided copy descriptor replaces flat ones (0: never) |
| `TIDE_RING_TAIL` | 1 | 1: the last block's group may sit in the ring; 0: a dedicated slot |
| `TIDE_RS_WAVE_NODES` | 8 | nodes per landing wave; below the node count it runs the multi-wave path on a small fleet |
| `TIDE_RS_DEBUG` | 0 | 1: serialize and report the multicast phases; 2: dump the flags of a call that does not finish; 3: host time per call by category, read phase apart, averaged over each 20 calls |
| `TIDE_FSDP_ASYNC_STATS_EVERY` | 0 | pipelined calls: every N calls rank 0 logs read (to `read_event`, the last level-1 sum; under a partial outbox bank it includes the earlier NIC rounds) / full times from the gradients' ready event, the post-backward and all-gather waits |
| `TIDE_RS_NET_DEBUG` | 0 | 1: the device network kernels leave breadcrumbs (step, round, peer, signal value, time waited) the monitor thread prints every 5 s |
| `TIDE_RS_NET_TRACE` | 0 | N: the wave kernels stamp each round (credit, puts, data, release, flush); every rank logs the per-call medians every N calls, the credit wait split into the waves' first rounds and their later rounds, and the mean wait per round by peer node (credits by target, data by source) |
| `TIDE_RS_CHECK_ORDER` | 0 | 1: every reduce-scatter checks across ranks that all issue the same call (a host collective per call) |
| `TIDE_TRACE_AG` | 0 | 1: GPU timestamps of the multi-node all-gather phases (`comm.trace_ag()`) |
| `TIDE_TRACE_AG_EVERY` | 0 | with `TIDE_TRACE_AG=1`: the FSDP2 adapter logs the phase medians every N calls |
| `TIDE_FSDP_RELEASE_POISON` | 0 | 1: NaN-fill a pipelined call's gradients when FSDP frees them, copied-in ones right after the copy (a late read shows in the result) |
| `TIDE_FAULT`, `TIDE_FAULT_RANK`, `TIDE_FAULT_US` | unset | fault injection for the negative tests ([Tests](#tests)); only in builds with `TIDE_FAULT_INJECTION=1` |
