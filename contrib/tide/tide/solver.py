# SPDX-FileCopyrightText: Copyright (c) 2026 Kuaishou Technology. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sizing the collectives for a fleet: CTAs, NIC round width k, staging chunk, and the memory and
time they imply, from a few calibrated rates (bench/calibrate.py writes them; the defaults are the
numbers measured on 400G RoCE rails in docs/design_rs.md).

    python -m tide.solver --nodes 128 --shard-mib 20 --rs-out-mib 40 --segments 30
    python -m tide.solver --nodes 6 --calib calibration.json --env      # export lines for the knobs

The model is first order: bytes / rate per link, fixed cost per NIC round, exposed tails. What it
cannot know -- the fabric's incast tolerance (k), skew between ranks -- comes in as constants.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass, field

MIB = 2**20
# Multi-node staging chunk ceiling: whole sub-slices of a large layer at a small rank count make the staging tens of
# GiB, and every chunk-round costs host issue time.
RS_CHUNK_CAP = 8 * MIB
# Passes over the sub-slices once the inbox of whole sub-slices reaches RS_PARTS_INBOX. Every pass queues its own NIC
# and staging rounds as stream memops: 4 passes need k = 2 to stay at the one-pass count, or the host blocks in
# cuStreamBatchMemOp.
RS_PARTS_INBOX = 2 * 2**30
RS_AUTO_PARTS = 4
# Pipelined calls free their inputs once read: one pass (from the third pass on, a pass's own-node sum waits on the
# landing two passes back) and chunks up to this let the sums read every destination back to back.
RS_CHUNK_CAP_EARLY = 32 * MIB


@dataclass
class Calib:
    nic_gbs: float = 47.5             # one rail port per rank (400G RoCE, measured 47.5-48.7)
    ksum_gbs_per_cta: float = 99.0    # K_sum bytes moved per CTA (7 staged + own read + fp32 write), TMA
    ksum_rail_gbs_per_cta: float = 16.0  # the one-rank-per-node sum kernel (grid-stride; 8 nodes: 4/8/12/16 CTAs -> 6.4/4.3/3.8/3.8 ms at 32 MiB)
    land_gbs_per_cta: float = 21.0    # multimem landing per CTA in situ (48 in the microbenchmark)
    land_gbs_gpu_max: float = 90.0    # 8-issuer fabric ceiling per GPU
    ce_multicast_gbs: float = 650.0   # copy-engine multicast scatter into the landing, per GPU
    round_fixed_ms: float = 0.10      # launch + arrival detection per NIC round
    pace_rtt_ms: float = 0.12         # pace / release signal round trip
    desc_strided_us: float = 0.65     # one strided descriptor inside a cudaMemcpy3DBatchAsync (host; nsys, 2 nodes, P = 16-64)
    desc_flat_us: float = 0.75        # one cudaMemcpyBatchAsync descriptor (host)
    memop_us: float = 4.0             # one cuStreamBatchMemOp (host)
    launch_us: float = 8.0            # one kernel launch (host)
    put_host_us: float = 5.5          # one ncclPutSignal / wait / signal (host; 2 nodes: ~45 us per 4-context round)
    padded_frac: float = 0.3          # segments whose shortened last row goes out as a flat descriptor
    incast_max: int = 2               # flows per receiving port of the one-rank-per-node form
    ksum_margin: float = 1.5          # CTAs over the pace requirement (rank skew, HBM contention)
    tail_frac: float = 0.05           # exposed tail allowed as a fraction of the NIC time
    rs_round_ms: float = 1.3          # a multicast NIC round's wait for its peers (64 nodes, device trace: 1.24 at k=2, 1.64 at k=4)
    rs_round_frac: float = 0.10       # rounds' waits allowed, as a fraction of the NIC time, before the rounds widen
    rs_k_max: int = 4                 # widest multicast round (the device network phase's tables hold 4 peers)


@dataclass
class Fleet:
    nodes: int
    local: int = 8
    segments: int = 30
    shard_bytes: int = 20 * MIB       # bf16 all-gather shard per rank of the largest group
    rs_out_bytes: int = 40 * MIB      # fp32 reduce-scatter output per rank of the largest group
    rs_itemsize: int = 2              # gradient dtype on the wire inside the node (bf16)
    rs_area_budget: int = 2 * 2**30   # stage area (staging + inbox + outbox)
    rs_rail_parts: int = 4            # TIDE_RS_RAIL_PARTS (one rank per node)
    rs_parts: int = 0                 # TIDE_RS_PARTS (multicast passes); 0: plan_rs chooses
    rs_stage_slots: int = 2           # TIDE_RS_STAGE_SLOTS: the chunk is sized for two, the others add their staging
    read_early: bool = False          # pipelined calls (multi-rank nodes): one pass, chunks up to RS_CHUNK_CAP_EARLY
    pipeline: int = 1                 # 2: pipelined calls hold two outbox banks and two spares per own wave (area only)
    ring_slots: int = 2
    total_shard_bytes: int = 0        # every group's sharded parameters (0: unknown)


@dataclass
class AgPlan:
    rounds: int
    nic_groups: int
    paced: bool
    nic_ms: float
    fixed_ms: float
    tail_ms: float
    total_ms: float
    host_ms: float
    descriptors: int
    recv_bytes_per_slot: int
    landing_bytes_per_slot: int
    arena_bytes: int


@dataclass
class RsPlan:
    ksum_ctas: int
    landing_ctas: int
    kway: int
    wave_nodes: int
    waves: int
    rounds: int
    chunk_bytes: int
    chunks_per_round: int
    nic_ms: float
    fixed_ms: float
    tail_ms: float
    total_ms: float
    host_ms: float
    staging_bytes: int
    inbox_bytes: int
    outbox_bytes: int
    area_bytes: int
    notes: list[str] = field(default_factory=list)
    parts: int = 1
    outbox_blocks: int = 0  # asked of the runtime (read_early: every remote destination); 0: two rounds' worth
    spare_bytes: int = 0    # own-node partials summed aside (in area_bytes)


def _round_widths(remote: int, k: int, wave: int) -> list[int]:
    """Per-round destination counts buildWaves produces: ramp within the first wave, k-wide in between,
    ramp down within the last wave (waves of `wave` contributors, the own node last)."""
    if remote <= 0:
        return []

    def ramp(r: int) -> list[int]:
        v, d, left = [], 1, r
        while left > 0:
            take = min(d, k, left)
            if left - take < take:
                take = min(left, k)
            v.append(take)
            left -= take
            d *= 2
        return v

    contributors = remote + 1
    n_waves = math.ceil(contributors / wave)
    out: list[int] = []
    first = 1
    for w in range(n_waves):  # filled in order, my node last (collective/rs/waves.cpp)
        count = min(wave, contributors - w * wave)
        own = first + count - 1 == contributors
        rem = count - (1 if own else 0)
        first += count
        if rem == 0:
            continue
        if w == 0:
            out.extend(ramp(rem))
        elif w == n_waves - 1:
            out.extend(reversed(ramp(rem)))
        else:
            while rem > 0:
                take = min(k, rem)
                out.append(take)
                rem -= take
    return out


def _round_sizes(remote: int, k: int, wave: int) -> int:
    return len(_round_widths(remote, k, wave))


def rs_area_budget(local: int, rs_out_bytes: int, c: Calib) -> int:
    """Stage-area cap the reduce-scatter sizing works within: 2 GiB, or room for a two-parity inbox, the
    widest outbox the fabric takes (2 x incast_max blocks) and one whole sub-slice per staging chunk,
    (3L + 2 incast_max) S -- a fixed cap squeezes k and the chunk while the inbox alone is already over
    it. The area itself follows the k and chunk the plan picks."""
    return max(2 * 2**30, (3 * local + 2 * c.incast_max) * rs_out_bytes)


def plan_ag(f: Fleet, c: Calib) -> AgPlan:
    M, S, P = f.nodes, f.shard_bytes, f.segments
    R = math.ceil(math.log2(M)) if M > 1 else 0
    G = 4  # TIDE_NIC_GROUPS default
    paced = S >= 16 * MIB
    nic_ms = (M - 1) * S / (c.nic_gbs * 1e9) * 1e3
    fixed_ms = R * (c.round_fixed_ms + (c.pace_rtt_ms if paced else 0.0))
    last_block = math.ceil(M / 2) if M > 1 else 0            # shards forwarded in the last round
    tail_ms = (last_block / G) * S / (c.ce_multicast_gbs * 1e9) * 1e3
    descriptors = P * (R + 1)                                 # one strided descriptor per segment per round (+ own)
    puts = 2 * R * 4                                          # own + receive-run puts, 4 contexts each
    host_ms = (descriptors * c.desc_strided_us + puts * c.put_host_us) / 1e3
    recv = (M - 1) * S
    landing = S * M * f.local
    arena = f.ring_slots * (recv + landing) + f.total_shard_bytes
    return AgPlan(R, G, paced, nic_ms, fixed_ms, tail_ms, nic_ms + fixed_ms + tail_ms, host_ms, descriptors,
                  recv, landing, arena)


def _rail_rounds(remote: int, k: int) -> list[int]:
    """Round widths of the one-rank-per-node form (railRounds in rs/plan.cpp)."""
    v, d, left = [], 1, remote
    while left > 0:
        take = min(d, k, left)
        if left - take < take:
            take = min(left, k)
        v.append(take)
        left -= take
        d = min(2 * d, k)
    return v


def _plan_rs_rail(f: Fleet, c: Calib) -> RsPlan:
    # one rank per node: blocks travel in the gradient dtype, no staging; the rounds run once per part,
    # a block holding 1/parts of a destination's pieces; a round of k blocks costs the owner k block
    # reads plus one read-modify-write of the part's fp32 output
    M, S, parts = f.nodes, f.rs_out_bytes, f.rs_rail_parts
    notes: list[str] = []
    blk = S * f.rs_itemsize / 4
    stride = math.ceil(blk / parts / 256) * 256

    def outbox_blocks(k: int) -> int:  # two consecutive rounds (outboxSlots in rs/plan.cpp)
        r = _rail_rounds(M - 1, k)
        return max(2, max(a + b for a, b in zip(r, [0] + r)))

    kway = k_asked = min(c.incast_max, 8, M - 1)  # the sum kernel stages up to 8 blocks
    while kway > 1 and (2 * kway + outbox_blocks(kway)) * stride > f.rs_area_budget:
        kway -= 1
    if kway != k_asked:
        notes.append(f"k lowered from {k_asked} to {kway}: the inbox / outbox blocks would not fit rs_area_budget")
    rounds = _rail_rounds(M - 1, kway)
    moved = blk + 2 * S * len(rounds) / (M - 1)  # per destination, amortized
    ksum = min(16, max(1, math.ceil(moved / blk * c.nic_gbs / c.ksum_rail_gbs_per_cta * c.ksum_margin)))
    nic_ms = (M - 1) * blk / (c.nic_gbs * 1e9) * 1e3
    fixed_ms = len(rounds) * parts * (c.round_fixed_ms + c.pace_rtt_ms)
    tail_ms = (rounds[-1] * blk + 2 * S) / parts / (ksum * c.ksum_rail_gbs_per_cta * 1e9) * 1e3
    inbox = 2 * kway * stride
    outbox = outbox_blocks(kway) * stride
    host_ms = (M - 1) * ((f.segments + parts - 1) * c.desc_flat_us + parts * 2 * 4 * c.put_host_us) / 1e3
    return RsPlan(ksum, 0, kway, 1, 0, len(rounds) * parts, int(stride), parts, nic_ms, fixed_ms, tail_ms,
                  nic_ms + fixed_ms + tail_ms, host_ms, 0, inbox, outbox, inbox + outbox, notes)


def plan_rs(f: Fleet, c: Calib) -> RsPlan:
    M, L, S, P = f.nodes, f.local, f.rs_out_bytes, f.segments
    notes: list[str] = []
    nic_bytes_per_s = c.nic_gbs * 1e9
    if L == 1 and M > 1:
        return _plan_rs_rail(f, c)
    # K_sum must produce one partial block (S fp32) per block time on the NIC; it moves
    # 7 + 1 pieces of S * itemsize / 4 in and S out per block.
    moved_per_block = 8 * S * f.rs_itemsize / 4 + S
    ksum_need = moved_per_block / S * c.nic_gbs / c.ksum_gbs_per_cta if M > 1 else 4
    ksum = max(1, math.ceil(ksum_need * c.ksum_margin)) if M > 1 else 8
    wave = min(8, L)
    waves = math.ceil(M / wave) if M > 1 else 0
    # passes over the sub-slices (rs_parts 0: chosen here); refused when a wave holds my node alone (repeatWaves)
    inbox_whole = (2 if waves > 1 else 1) * L * math.ceil(S / L / 32) * 32 * L if M > 1 else 0
    auto = f.rs_parts <= 0
    early = f.read_early and M > 1
    one_pass = (f.read_early or f.pipeline == 2) and M > 1
    parts = (RS_AUTO_PARTS if inbox_whole >= RS_PARTS_INBOX and not one_pass else 1) if auto else f.rs_parts
    parts = parts if M > 1 and wave > 1 and M % wave != 1 else 1
    nic_ms = (M - 1) * S / nic_bytes_per_s * 1e3
    # landing: hides behind the NIC except the last wave; CTAs for the tail budget, capped at the fabric ceiling
    per_cta = c.land_gbs_per_cta * 1e9
    land_pace = S / per_cta / (wave * S / nic_bytes_per_s) if M > 1 else 0      # CTAs to keep pace with a wave's arrival
    tail_budget_s = c.tail_frac * nic_ms / 1e3
    land_tail = S / per_cta / tail_budget_s if tail_budget_s > 0 else 0
    cap = max(1, int(c.land_gbs_gpu_max // c.land_gbs_per_cta))
    landing = min(cap, max(1, math.ceil(max(land_pace, land_tail)))) if M > 1 else 0
    if M > 1 and land_tail / parts > cap:
        notes.append(f"landing tail {S / parts / per_cta / cap * 1e3:.2f} ms exceeds the {c.tail_frac:.0%} budget at {cap} CTAs (small M: the tail is one block)")
    # k: a round waits rs_round_ms for its peers however few bytes it carries, so with many nodes the rounds, not the
    # bytes, set the network phase; a round of k puts k flows on every receive port. k = 2, doubled while the rounds'
    # waits exceed rs_round_frac of the NIC time and doubling cuts rounds (8 nodes: k = 2 in 4 rounds beat k = 4 in
    # 3 by 6-8%; 64 nodes: k = 4 in 19 rounds beat k = 2 in 33 by 4-5%)
    kway = min(2, max(1, M - 1))
    while 2 * kway <= min(c.rs_k_max, M - 1) and \
            _round_sizes(M - 1, kway, wave) * c.rs_round_ms > c.rs_round_frac * nic_ms and \
            _round_sizes(M - 1, 2 * kway, wave) < _round_sizes(M - 1, kway, wave):
        kway *= 2
    # memory: inbox 2 parities x L blocks, outbox two rounds (2k blocks; read_early: every remote destination),
    # staging budgeted as 2 slots x L x L x chunk (further slots add theirs on top); the 2k outbox grows with k, so k gives way before the staging chunk drops
    # below 4 MiB. A block holds one part of L sub-slices; from two parts on the inbox has two parities.
    stride = math.ceil(S / L / parts / 32) * 32 * L if M > 1 else 0
    inbox = 2 * L * stride if waves > 1 or parts > 1 else L * stride
    sub_bytes = int(math.ceil(S / L / parts) * f.rs_itemsize / 4) if M > 1 else int(S * f.rs_itemsize / 4)
    min_staging = 2 * L * L * min(4 * MIB, sub_bytes) if M > 1 else 0
    k_asked = kway
    while not one_pass and kway > 1 and inbox + max(2, 2 * kway) * stride + min_staging > f.rs_area_budget:
        kway -= 1
    if kway != k_asked:
        notes.append(f"k lowered from {k_asked} to {kway}: a wider outbox (2k blocks) would not fit rs_area_budget")
    r_widths = _round_widths(M - 1, kway, wave) * parts if M > 1 else []
    two_rounds = max([2] + [a + b for a, b in zip(r_widths, [0] + r_widths)])
    outbox_blocks = (parts * (M - 1) if early
                     else (min(parts * (M - 1), two_rounds) if f.pipeline == 2 and M > 1 else 0))
    outbox = (outbox_blocks or max(2, 2 * kway)) * stride
    rounds = len(r_widths)
    fixed_ms = rounds * c.rs_round_ms
    if fixed_ms > c.rs_round_frac * nic_ms and M > 2:
        notes.append(f"{rounds} NIC rounds at k = {kway} wait ~{fixed_ms:.0f} ms ({fixed_ms / nic_ms:.0%} of the NIC time)")
    tail_ms = S / parts / (landing * per_cta) * 1e3 if M > 1 else 0.0
    # a round moves one sub-slice per source (S / L of the output, in the gradient dtype); the chunk is
    # the whole sub-slice unless the staging budget forces several chunk-rounds
    # single node: several chunks per source keep the pushes and the sum pipelined; the budget never
    # pushes a chunk below 4 MiB (an inbox over budget is reported, not paid in chunk-rounds)
    slot_max = (f.rs_area_budget - inbox - outbox) // 2
    fit = slot_max // (L * L if M > 1 else L)
    cap = min(sub_bytes, (RS_CHUNK_CAP_EARLY if one_pass else RS_CHUNK_CAP) if M > 1 else 16 * MIB)
    want = min(cap, max(fit, 4 * MIB))
    # one chunk for the whole sub-slice rounds up; a chunk the budget bounds rounds down
    chunk = -(-want // 4096) * 4096 if want == cap else max(4096, want // 4096 * 4096)
    if fit < min(cap, 4 * MIB):
        notes.append(f"stage area over rs_area_budget ({f.rs_area_budget / 2**30:.1f} GiB): inbox {inbox / MIB:.0f} MiB")
    chunks_per_round = max(1, math.ceil(sub_bytes / chunk))
    if chunks_per_round > 1 and chunk < 4 * MIB:
        notes.append(f"staging chunk {chunk / MIB:.1f} MiB (x{chunks_per_round} per round) below 4 MiB: raise rs_area_budget")
    staging = f.rs_stage_slots * (L * L if M > 1 else L) * chunk
    # the runtime's layout past the one outbox sized above: pipelined calls hold two outbox banks (read_early: one
    # block per remote destination; else two rounds' worth) and two spares per own-node wave (one per part); one-phase
    # calls a spare per own wave from wave 2 on
    if M > 1 and f.pipeline == 2:
        outbox, spares = 2 * outbox_blocks * stride, 2 * parts * stride
    else:
        spares = sum((q + 1) * waves - 1 >= 2 for q in range(parts)) * stride if M > 1 else 0
    area = staging + inbox + outbox + spares
    # host per node round: one strided descriptor per (local owner, segment) per chunk -- segments are
    # separate tensors, so they cannot share one -- plus a flat one per padded segment tail, ~5 memops
    # and a launch; per remote destination one head + tail put per RMA context (measured: the
    # descriptors are ~80% of the host time, charged per descriptor with no fixed part)
    per_round_us = (7 * P * chunks_per_round * (c.desc_strided_us + c.padded_frac * c.desc_flat_us)
                    + 5 * c.memop_us + chunks_per_round * c.launch_us)
    host_ms = parts * ((M if M > 1 else 1) * per_round_us + (M - 1) * 2 * 4 * c.put_host_us) / 1e3
    total = max(nic_ms, moved_per_block * M / (ksum * c.ksum_gbs_per_cta * 1e9) * 1e3) + fixed_ms + tail_ms
    return RsPlan(ksum, landing, kway, wave, waves, rounds, chunk, chunks_per_round, nic_ms, fixed_ms, tail_ms,
                  total, host_ms, staging, inbox, outbox, area, notes, parts, outbox_blocks, spares)


def load_calib(path: str = "") -> Calib:
    """calibration.json from bench/calibrate.py; unknown keys are an error, missing ones keep the default."""
    return Calib(**json.load(open(path))) if path else Calib()


def env_lines(rs: RsPlan) -> list[str]:
    return [f"export TIDE_RS_CTAS={rs.ksum_ctas}", f"export TIDE_RS_MC_CTAS={rs.landing_ctas}",
            f"export TIDE_RS_KWAY={rs.kway}", f"export TIDE_RS_CHUNK_MIB={max(1, math.ceil(rs.chunk_bytes / MIB))}"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--nodes", type=int, required=True)
    ap.add_argument("--local", type=int, default=8)
    ap.add_argument("--segments", type=int, default=30)
    ap.add_argument("--shard-mib", type=float, default=20.0, help="bf16 all-gather shard per rank (largest group)")
    ap.add_argument("--rs-out-mib", type=float, default=40.0, help="fp32 reduce-scatter output per rank (largest group)")
    ap.add_argument("--rs-area-gib", type=float, default=2.0)
    ap.add_argument("--rail-parts", type=int, default=4, help="TIDE_RS_RAIL_PARTS (--local 1)")
    ap.add_argument("--parts", type=int, default=0, help="TIDE_RS_PARTS (multicast passes; 0: chosen)")
    ap.add_argument("--read-early", action="store_true", help="early reads (the FSDP2 adapter, with --pipeline 2)")
    ap.add_argument("--pipeline", type=int, default=1, choices=(1, 2),
                    help="2: pipelined calls, two outbox banks and spares in the area")
    ap.add_argument("--total-shard-gib", type=float, default=0.0, help="all sharded parameters (live in the arena)")
    ap.add_argument("--nic-gbps", type=float, default=None, help="rail port speed in Gb/s (overrides calib nic_gbs at 95%%)")
    ap.add_argument("--calib", default="", help="calibration.json from bench/calibrate.py")
    ap.add_argument("--env", action="store_true", help="print export lines for the reduce-scatter knobs")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    c = load_calib(a.calib)
    if a.nic_gbps:
        c.nic_gbs = a.nic_gbps / 8 * 0.95
    f = Fleet(a.nodes, a.local, a.segments, int(a.shard_mib * MIB), int(a.rs_out_mib * MIB),
              rs_area_budget=int(a.rs_area_gib * 2**30), rs_rail_parts=a.rail_parts, rs_parts=a.parts,
              read_early=a.read_early and a.local > 1, pipeline=a.pipeline,
              total_shard_bytes=int(a.total_shard_gib * 2**30))
    ag, rs = plan_ag(f, c), plan_rs(f, c)
    if a.json:
        print(json.dumps({"fleet": asdict(f), "calib": asdict(c), "ag": asdict(ag), "rs": asdict(rs)}, indent=1))
        return
    print(f"fleet: {f.nodes} nodes x {f.local}, {f.segments} segments, AG shard {f.shard_bytes / MIB:.0f} MiB, "
          f"RS out {f.rs_out_bytes / MIB:.0f} MiB, NIC {c.nic_gbs:.1f} GB/s")
    print(f"AG : {ag.rounds} rounds (G={ag.nic_groups}, paced={ag.paced}); NIC {ag.nic_ms:.1f} ms + fixed {ag.fixed_ms:.2f} "
          f"+ tail {ag.tail_ms:.2f} = {ag.total_ms:.1f} ms; host {ag.host_ms:.2f} ms ({ag.descriptors} descriptors); "
          f"0 CTAs; per ring slot: receive {ag.recv_bytes_per_slot / MIB:.0f} MiB + landing {ag.landing_bytes_per_slot / MIB:.0f} MiB"
          + (f"; arena {ag.arena_bytes / 2**30:.2f} GiB" if f.total_shard_bytes else ""))
    print(f"RS : K_sum {rs.ksum_ctas} CTAs + landing {rs.landing_ctas} CTAs (peak {rs.ksum_ctas + rs.landing_ctas}); k={rs.kway}, "
          f"waves {rs.waves} x {rs.wave_nodes}, {rs.parts} parts, {rs.rounds} rounds, chunk {rs.chunk_bytes / MIB:.1f} MiB x {rs.chunks_per_round}; "
          f"NIC {rs.nic_ms:.1f} ms + fixed {rs.fixed_ms:.2f} + tail {rs.tail_ms:.2f} = {rs.total_ms:.1f} ms; host {rs.host_ms:.2f} ms; "
          f"area {rs.area_bytes / 2**30:.2f} GiB (staging {rs.staging_bytes / MIB:.0f} + inbox {rs.inbox_bytes / MIB:.0f} + outbox {rs.outbox_bytes / MIB:.0f} "
          f"+ spares {rs.spare_bytes / MIB:.0f} MiB)")
    for n in rs.notes:
        print("     note:", n)
    if a.env:
        print("\n".join(env_lines(rs)))


if __name__ == "__main__":
    main()
