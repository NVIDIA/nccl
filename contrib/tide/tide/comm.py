# SPDX-FileCopyrightText: Copyright (c) 2026 Kuaishou Technology. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import glob
import logging
import os
import re
import time
import warnings
from typing import Sequence

import torch
import torch.distributed as dist

from tide import _C

log = logging.getLogger("tide.comm")

# a gradient list, or its description (shapes = [(numel, dim0)] per gradient, itemsize 2 or 4)
RsTable = Sequence[torch.Tensor] | tuple[Sequence[tuple[int, int]], int]


_check_order = os.environ.get("TIDE_RS_CHECK_ORDER", "") == "1"


_WINDOW_ALIGN = 2 << 20  # ncclMemAlloc granularity (csrc/runtime/comm_internal.h kWindowAlign)


def nic_global_tclass() -> dict[int, list[str]]:
    """mlx5's per-port global traffic class (sysfs tc/<port>/traffic_class) -> devices: the kernel writes it into every
    verbs QP it modifies, e.g. to put all RoCE traffic on the PFC priority."""
    found: dict[int, list[str]] = {}
    for f in sorted(glob.glob("/sys/class/infiniband/*/tc/*/traffic_class")):
        try:
            with open(f) as fh:
                m = re.search(r"tclass\s*=\s*(-?\d+)", fh.read())
        except OSError:
            continue
        if m and int(m.group(1)) >= 0:
            found.setdefault(int(m.group(1)), []).append(f.split("/")[4])
    return found


def align_gin_traffic_class() -> str:
    """Give NCCL's GIN GDAKI QPs (TIDE_RS_NET=device) the traffic class the host's verbs QPs get; returns what it did.

    GDAKI QPs are created over DEVX, past the kernel's global tclass, and otherwise send with class 0: a lossy priority
    where that tclass moves RoCE onto the PFC one, so drops under load end in a retry-exceeded QP and a flush that never
    returns. NCCL reads the variable when GIN connects, after torch's own communicators exist."""
    if os.environ.get("NCCL_GIN_IB_TC"):
        return f"NCCL_GIN_IB_TC={os.environ['NCCL_GIN_IB_TC']} (set by the caller)"
    found = nic_global_tclass()
    if not found:
        return "no NIC global traffic class: GIN QPs and verbs QPs both use NCCL's"
    if len(found) > 1:
        warnings.warn(f"[TIDE] NICs disagree on the global traffic class ({found}); GIN QPs keep NCCL's class and may send "
                      "on another priority than the host's traffic: set NCCL_GIN_IB_TC", stacklevel=2)
        return f"NICs disagree on the global traffic class {sorted(found)}: left to NCCL"
    (tc, devs), = found.items()
    os.environ["NCCL_GIN_IB_TC"] = str(tc)
    return f"NCCL_GIN_IB_TC={tc} (the global traffic class of {devs[0]}..{devs[-1]})"


# earlier TIDE read these; set now, they would be ignored without a word
_RETIRED_ENV = {
    "TIDE_RS_DEVCTL": "use TIDE_RS_NET=device|host",
    "TIDE_RS_DEVCTL_TIMEOUT_S": "use TIDE_RS_NET_TIMEOUT_S",
    "TIDE_RS_DEVCTL_TRACE": "use TIDE_RS_NET_TRACE",
    "TIDE_RS_DEVCTL_DEBUG": "use TIDE_RS_NET_DEBUG",
    "TIDE_FSDP_AG_DECOUPLE": "the FSDP2 adapter's all-gathers are always decoupled now",
    "TIDE_FSDP_EARLY_RELEASE": "removed: pipelined calls free their inputs once read",
    "TIDE_FSDP_RELEASE_MODE": "removed: pipelined calls free their inputs once read",
}


def _all_same(group, value, what: str) -> None:
    objs = [None] * dist.get_world_size(group)
    dist.all_gather_object(objs, value, group=group)
    if any(o != objs[0] for o in objs):
        raise RuntimeError(f"tide: {what} differs across ranks: {objs}")


class TideComm:
    """A dedicated NCCL communicator plus the symmetric windows registered on it.

    Never issues NCCL collectives; only windows, peer pointers and (multi-node)
    host RMA. Windows are registered once and never grow; free_arena() forgets
    one, close() ends the communicator at a program point of your choosing.

    num_rma_ctx: QPs the inter-node leg stripes a shard over (0 = TIDE_RMA_CTX
    or 4). One context halves the six-node all-gather bandwidth (7.2 vs 4.5 ms
    at 32 MiB) but saves a launch per round; worth it only where the all-gather
    is hidden behind compute (FSDP: -0.15% step time), never as the default.
    num_rma_sig: signal channels; data arrival, round pacing, then one credit
    channel per distinct stage area (default 64).
    """

    def __init__(self, group=None, device: torch.device | None = None, num_rma_ctx: int = 0, num_rma_sig: int = 0):
        stale = [f"{k} ({why})" for k, why in _RETIRED_ENV.items() if k in os.environ]
        if stale:
            raise RuntimeError("tide: retired environment variables are set: " + "; ".join(stale))
        self.group = group if group is not None else dist.group.WORLD
        self.rank = dist.get_rank(self.group)
        self.nranks = dist.get_world_size(self.group)
        self.device = torch.device(device) if device is not None else torch.device("cuda", torch.cuda.current_device())
        torch.cuda.set_device(self.device)
        t0 = time.perf_counter()
        obj = [_C.get_unique_id() if self.rank == 0 else None]
        dist.broadcast_object_list(obj, src=dist.get_global_rank(self.group, 0), group=self.group, device=self.device)
        t1 = time.perf_counter()
        if os.environ.get("NCCL_RUNTIME_CONNECT", "1") == "0" and self.rank == 0:
            # Eager connect preallocates ring/tree/NVLS buffers this communicator never uses; the extension caps
            # the communicator at one channel (TIDE_COMM_CTAS), but NCCL_MIN_NCHANNELS duplicates channels past it.
            warnings.warn("[TIDE] NCCL_RUNTIME_CONNECT=0 preconnects collective buffers on a TIDE communicator "
                          "(one channel's worth, or NCCL_MIN_NCHANNELS of them); leave it at NCCL's default (1)",
                          stacklevel=2)
        self.gin_traffic_class = align_gin_traffic_class()
        free0 = torch.cuda.mem_get_info(self.device)[0]
        self._c = _C.Comm(obj[0], self.nranks, self.rank, num_rma_ctx, num_rma_sig)
        t2 = time.perf_counter()
        # taken from the driver directly: invisible to torch.cuda memory stats and snapshots
        self.device_bytes = free0 - torch.cuda.mem_get_info(self.device)[0]
        self.props = self._c.props()
        self.lsa_peers = list(self._c.lsa_peers())
        self.node, self.n_nodes, self.local_ranks = self._c.node, self._c.n_nodes, self._c.local_ranks
        self.rma_ctx, self.nic_groups = self._c.rma_ctx, self._c.nic_groups
        self.rail = self.local_ranks == 1 and self.n_nodes > 1  # one rank per node (e.g. an EP=8 expert group)
        _all_same(self.group, (self.rma_ctx, self._c.rma_sig, self.nic_groups, self._c.pace_mode, self._c.pace_lead,
                               self._c.rs_stage_slots), "communicator config")
        t3 = time.perf_counter()
        comm_s, sync_s, rail_s = self._c.init_s
        self.init_s = {"id_broadcast": t1 - t0, "nccl_init": comm_s, "first_window": sync_s, "rail_split": rail_s,
                       "other": t2 - t1 - comm_s - sync_s - rail_s, "config_check": t3 - t2}  # startup accounting
        if self.rank == 0:
            # where a large job's start goes: ncclCommInitRank's bootstrap grows with the rank count; the rail split
            # carries NCCL's RMA connect (every context to every rank of the rail communicator, a ring barrier per
            # peer: ~ctx x nodes^2, docs/usage.md)
            log.info("[TIDE] communicator over %d ranks (%d nodes) up in %.1f s: unique-id broadcast %.1f, "
                     "ncclCommInitRank %.1f, first window %.1f, rail split / RMA connect (%d ctx) %.1f, other %.1f, "
                     "config check %.1f", self.nranks, self.n_nodes, t3 - t0, t1 - t0, comm_s, sync_s, self.rma_ctx,
                     rail_s, t2 - t1 - comm_s - sync_s - rail_s, t3 - t2)
        self._windows: list = []
        self._rs_registered = 0

    def stage_bytes(self, shard_bytes: int, own_separate: bool = False) -> int:
        """Bytes of packed-shard stage a region needs: the own shard (unless it lives at its own
        `own_off`) plus, on multi-node, one receive slot per other node. One rank per node: none (the
        NIC puts land straight in the landing)."""
        if self.rail:
            return 0
        recv = int(shard_bytes) * (self.n_nodes - 1)
        return recv if own_separate else recv + int(shard_bytes)

    def alloc_arena(self, nbytes: int) -> torch.Tensor:
        """Collective. Returns a uint8 tensor over a freshly registered symmetric window. The memory comes
        from a torch MemPool on the process group's NCCL allocator, so torch's memory statistics, snapshots
        and OOM accounting include it (TIDE_ARENA_TORCH=0, or no NCCL allocator: TIDE's own ncclMemAlloc)."""
        _all_same(self.group, int(nbytes), "arena size")
        pool = self._arena_pool()
        if pool is None:
            win = self._c.alloc_window(int(nbytes))
            t = win.tensor(self.device.index)
            t._tide_window = win  # type: ignore[attr-defined]
        else:
            with torch.cuda.use_mem_pool(pool):
                t = torch.empty(-(-int(nbytes) // _WINDOW_ALIGN) * _WINDOW_ALIGN, dtype=torch.uint8, device=self.device)
            # The window holds t and the pool until it is deregistered. No reference back from t: torch keeps
            # a tensor's Python object alive while C++ holds the tensor, so the pair would never be collected.
            win = self._c.register_window(t, pool)
        self.arena_in_torch = pool is not None
        self._windows.append(win)
        return t

    def _arena_pool(self):
        """A MemPool of its own per arena (freeing the arena hands its segment back with the pool)."""
        if os.environ.get("TIDE_ARENA_TORCH", "1") == "0":
            return None
        try:
            allocator = self.group._get_backend(self.device).mem_allocator
        except (AttributeError, RuntimeError):
            return None
        return torch.cuda.MemPool(allocator)

    def free_arena(self, arena: torch.Tensor) -> None:
        """Collective in effect (same program point on every rank, nothing in flight). Forgets the
        arena's window and every registration on it; handles over it must not be used again. The
        memory itself is released when the last tensor over the window goes away (from a torch MemPool:
        back to the pool, whose segment the driver gets at torch's next release of cached blocks)."""
        win, off = self._window_of(arena)
        if off != 0 or arena.numel() != int(win.bytes):
            raise ValueError("free_arena takes the tensor alloc_arena returned, not a view into it")
        self._c.free_window(win)
        self._windows.remove(win)

    def close(self) -> None:
        """Collective (ncclCommDestroy barriers within the node). Drains the device and destroys the
        communicator; tensors still over an arena keep their window mapped and defer the destroy."""
        self._windows.clear()
        self._c.close()

    @property
    def closed(self) -> bool:
        return bool(self._c.closed)

    def _window_of(self, arena: torch.Tensor):
        win = getattr(arena, "_tide_window", None)
        if win is None:
            for w in self._windows:
                if w.base <= arena.data_ptr() < w.base + w.bytes:
                    return w, arena.data_ptr() - w.base
            raise ValueError("tensor is not inside a TIDE window")
        return win, arena.data_ptr() - win.base

    def register_ag(self, arena: torch.Tensor, region_off: int, sizes: Sequence[int], offsets: Sequence[int],
                    stage_off: int = -1, own_off: int = -1) -> "AgHandle":
        """Collective (cross-rank table check). Prebuilds every descriptor for this region.

        stage_off >= 0: packed-shard stage at that arena offset (stage_bytes(shard) bytes; required on
        multi-node): the own shard first, then one receive slot per other node. own_off >= 0 instead
        puts the own shard there (the sharded parameters themselves may live at own_off; the stage is
        then receive slots only, stage_bytes(shard, own_separate=True)); with neither the shard stays in
        place in the landing (single node). Multi-node plans whose receive areas overlap share one credit
        channel, so a receive area must never overlap a landing another plan may be using, and an own
        shard may touch neither."""
        win, base_off = self._window_of(arena)
        region_off = int(region_off) + int(base_off)
        if stage_off >= 0:
            stage_off = int(stage_off) + int(base_off)
        elif self.n_nodes > 1 and not self.rail:
            raise ValueError("multi-node all-gather needs stage_off (see stage_bytes)")
        if own_off >= 0:
            own_off = int(own_off) + int(base_off)
        sizes, offsets = [int(s) for s in sizes], [int(o) for o in offsets]
        _all_same(self.group, (region_off, sizes, offsets, stage_off, own_off), "all-gather table")
        plan = self._c.register_ag(win, region_off, sizes, offsets, stage_off, own_off)
        # the window (registrations pair up by order), the copy mode and everything that decides how
        # many signals a call sends must agree, or ranks address different memory / the counters drift
        _all_same(self.group, (win.id, plan.rounds, plan.stage_slot, plan.paced, plan.pace_tails),
                  "all-gather protocol")
        return AgHandle(self, plan, win, region_off, sizes, offsets, base_off)

    def all_gather(self, handle: "AgHandle", stream: torch.cuda.Stream | None = None) -> torch.cuda.Event:
        """The caller has written its pieces in place (or its packed shard at own_pieces())."""
        s = stream if stream is not None else torch.cuda.current_stream(self.device)
        self._c.all_gather(handle.plan, s.cuda_stream)
        ev = torch.cuda.Event()
        ev.record(s)
        return ev

    def all_gather_on(self, handle: "AgHandle", stream: torch.cuda.Stream,
                      free_event: torch.cuda.Event | None = None, done_event: torch.cuda.Event | None = None) -> None:
        """all_gather() without recording an event (the caller records its own). With done_event the call is
        decoupled: done_event is recorded once every piece has landed here and `stream` does not wait for the
        other ranks' writes, so consumers wait on done_event, not on the stream; peers may write into the
        landing once free_event (the release of its previous tenant; None or never recorded: none) has passed."""
        done = 0
        if done_event is not None:
            if not done_event.cuda_event:
                done_event.record(stream)  # torch creates events lazily and waits on an uncreated one as a no-op
            done = done_event.cuda_event
        free = free_event.cuda_event if free_event is not None else 0
        self._c.all_gather(handle.plan, stream.cuda_stream, free, done)

    def await_stage_reads(self, stream: torch.cuda.Stream) -> None:
        """Before writing a packed shard into a receive stage (a plan's own slot, e.g. a cast kernel): `stream` waits
        for the previous decoupled all-gather's scatters, which may still read the stage."""
        self._c.await_stage_reads(stream.cuda_stream)

    def trace_ag(self) -> list[tuple[int, list[float]]]:
        """TIDE_TRACE_AG=1 only: completed multi-node calls since the last dump, as (seq, ms per phase);
        phases: start>credits, credits>fence, fence>round 1, round 1>2, ..., last round>scatter join, join>done."""
        return self._c.trace_ag()

    # -- reduce-scatter over gradients read in place ------------------------------------------------
    @staticmethod
    def _shapes(grads: Sequence[torch.Tensor]) -> tuple[list[tuple[int, int]], int]:
        if not grads:
            raise ValueError("empty gradient list")
        dt = grads[0].dtype
        if dt not in (torch.bfloat16, torch.float32):
            raise ValueError(f"reduce-scatter input must be bf16 or fp32, got {dt}")
        shapes = []
        for g in grads:
            if g.dtype != dt or not g.is_contiguous() or g.dim() == 0 or g.numel() == 0:
                raise ValueError("gradients must be contiguous, non-empty, same dtype")
            shapes.append((int(g.numel()), int(g.shape[0])))
        return shapes, dt.itemsize

    @classmethod
    def _table(cls, t: "RsTable") -> tuple[list[tuple[int, int]], int]:
        """A gradient list, or its `(shapes, itemsize)` description: shapes = (numel, dim0) per gradient,
        itemsize 2 (bf16) or 4 (fp32)."""
        if not t:
            raise ValueError("empty reduce-scatter table")
        if isinstance(t[0], torch.Tensor):
            return cls._shapes(t)
        shapes, itemsize = t
        shapes = [(int(n), int(d)) for n, d in shapes]
        if int(itemsize) not in (2, 4) or not shapes or any(d <= 0 or n <= 0 or n % d for n, d in shapes):
            raise ValueError("reduce-scatter table: itemsize 2 or 4, shapes (numel, dim0) with 0 < dim0 | numel")
        return shapes, int(itemsize)

    def rs_stage_bytes(self, table: "RsTable") -> int:
        """Stage bytes a reduce-scatter over `table` needs (put it at a 256 B offset)."""
        return int(self._c.rs_stage_bytes(*self._table(table)))

    def rs_area_bytes(self, tables: Sequence["RsTable"]) -> int:
        """Stage bytes of one area shared by reduce-scatters over every table in `tables`. The largest
        table in each dimension sets the area's geometry; register them all before the first call on it."""
        return int(self._c.rs_area_bytes([self._table(t) for t in tables]))

    def tune_rs(self, tables: Sequence["RsTable"], calib: str | None = None, read_early: bool = False,
                pipeline: int = 0):
        """Collective. Sizes the reduce-scatter (sum / landing CTAs, NIC round width k, staging chunk)
        with tide.solver for the largest of `tables`, before the first registration; environment knobs
        win. `calib`: calibration.json from bench/calibrate.py (default $TIDE_CALIB, else the built-in
        400G RoCE rates). read_early: one pass with larger staging chunks and an outbox block for every
        remote destination of a call, so the sums read the inputs back to back without waiting for the NIC or
        earlier passes' landings (multi-rank nodes; a rail lane reads its inputs in every round anyway).
        pipeline=2 (TIDE_RS_PIPELINE wins): multi-node multicast calls in a read and a network phase over two
        outbox / own-spare banks, so a call's reads overlap the previous call's network phase (ready_event:
        where the gradients are complete; default the call stream). The FSDP2 adapter asks for both.
        Returns the solver's plan, or None once a reduce-scatter is registered."""
        from tide import solver

        if self._rs_registered:
            return None
        tabs = [self._table(t) for t in tables]
        n = self.nranks

        def out_bytes(shapes):
            return 4 * sum(-(-int(d) // n) * (int(e) // int(d)) for e, d in shapes)

        shapes, itemsize = max(tabs, key=lambda t: out_bytes(t[0]))
        path = calib if calib is not None else os.environ.get("TIDE_CALIB", "")
        c = solver.load_calib(path)
        early = read_early and self.n_nodes > 1 and self.local_ranks > 1
        fleet = solver.Fleet(self.n_nodes, self.local_ranks, len(shapes), rs_out_bytes=out_bytes(shapes),
                             rs_itemsize=int(itemsize),
                             rs_area_budget=solver.rs_area_budget(self.local_ranks, out_bytes(shapes), c),
                             rs_rail_parts=self._c.rs_rail_parts, rs_parts=self._c.rs_parts_env,
                             rs_stage_slots=self._c.rs_stage_slots, read_early=early,
                             pipeline=2 if int(pipeline) == 2 else 1)
        rs = solver.plan_rs(fleet, c)
        self._c.set_rs_tuning(rs.ksum_ctas, rs.landing_ctas, rs.kway, rs.chunk_bytes, rs.parts, rs.outbox_blocks,
                              int(pipeline))
        _all_same(self.group, (tuple(self._c.rs_tuning), self._c.rs_pipeline), "reduce-scatter tuning")
        return rs

    @property
    def rs_pipeline(self) -> int:
        """2: multi-node multicast reduce-scatters run as a read and a network phase (TIDE_RS_PIPELINE / tune_rs)."""
        return int(self._c.rs_pipeline)

    @property
    def rs_tuning(self) -> tuple[int, int, int, int]:
        """(sum CTAs, landing CTAs, k, staging chunk bytes) the next registration uses."""
        return tuple(self._c.rs_tuning)

    @property
    def rs_parts(self) -> int:
        """Passes a multi-node reduce-scatter makes over the sub-slices (TIDE_RS_PARTS)."""
        return int(self._c.rs_parts)

    @property
    def rs_outbox(self) -> int:
        """Outbox blocks asked per bank (TIDE_RS_OUTBOX, or tune_rs); 0: two rounds' worth on one-phase calls, one
        block per remote destination per bank on pipelined ones. See a plan's outbox_slots / outbox_bank."""
        return int(self._c.rs_outbox)

    def register_rs(self, arena: torch.Tensor, stage_off: int, table: "RsTable") -> "RsHandle":
        """Collective. Registers a reduce-scatter over `table` (gradients dim-0 sharded, padded to nranks
        like FSDP2); the stage area starts at `stage_off` in `arena`."""
        win, base_off = self._window_of(arena)
        shapes, itemsize = self._table(table)
        stage_off = int(stage_off) + int(base_off)
        _all_same(self.group, (stage_off, shapes, itemsize), "reduce-scatter table")
        plan = self._c.register_rs(win, stage_off, shapes, itemsize)
        _all_same(self.group, (win.id, plan.chunks, plan.area, plan.paced, plan.pace_tail, plan.wave_nodes, plan.waves,
                               tuple(plan.k_rounds)), "reduce-scatter protocol")
        self._rs_registered += 1
        return RsHandle(self, plan, shapes, itemsize, self._rs_registered - 1)

    def _reduce_scatter(self, handle, ptrs: list[int], out, scale, accumulate, stream, read_event=None,
                        ready_event=None, gate_event=None):
        if handle.comm is not self:
            raise ValueError("handle belongs to another TideComm")
        s = stream if stream is not None else torch.cuda.current_stream(self.device)
        if s.device != self.device:
            raise ValueError(f"stream on {s.device}, communicator on {self.device}")
        if out is None:
            if accumulate:
                raise ValueError("accumulate=True needs the output to accumulate into (out=)")
            with torch.cuda.stream(s):
                out = torch.empty(handle.out_numel, dtype=torch.float32, device=self.device)
        elif out.dtype != torch.float32 or not out.is_contiguous() or out.numel() != handle.out_numel:
            raise ValueError("out must be a contiguous fp32 tensor of handle.out_numel elements")
        elif out.device != self.device:
            raise ValueError(f"out on {out.device}, communicator on {self.device}")
        if _check_order:
            _all_same(self.group, (handle.index, int(handle.plan.calls)), "reduce-scatter call order")
        read = 0
        if read_event is not None:
            if not read_event.cuda_event:
                read_event.record(s)  # torch creates events lazily
            read = read_event.cuda_event
        # recorded events only: a wait on a never-recorded event would pass at once
        ready = ready_event.cuda_event if ready_event is not None else 0
        gate = gate_event.cuda_event if gate_event is not None else 0
        if (ready_event is not None and not ready) or (gate_event is not None and not gate):
            raise ValueError("ready_event / gate_event must have been recorded")
        if int(handle.plan.pipeline) == 2 and not ready:  # the gradients are complete where the stream is now
            ready_event = torch.cuda.Event()
            ready_event.record(s)
            ready = ready_event.cuda_event
        self._c.reduce_scatter(handle.plan, ptrs, out.data_ptr(), float(scale), bool(accumulate), s.cuda_stream, read,
                               ready, gate)
        ev = torch.cuda.Event()
        ev.record(s)
        return out, ev

    def reduce_scatter(self, handle: "RsHandle", grads: Sequence[torch.Tensor], out: torch.Tensor | None = None,
                       scale: float = 1.0, accumulate: bool = False, stream: torch.cuda.Stream | None = None,
                       read_event: torch.cuda.Event | None = None, ready_event: torch.cuda.Event | None = None,
                       gate_event: torch.cuda.Event | None = None) -> tuple[torch.Tensor, torch.cuda.Event]:
        """out[i] = (out[i] if accumulate else 0) + scale * sum over ranks; out is the packed fp32 of
        this rank's pieces (torch._chunk_cat + reduce_scatter_tensor layout). The gradients must stay
        alive until the returned event -- or, with read_event, until a stream that reuses them waited on
        read_event, which this call records once it has read them all (before it completes).

        Every rank must issue the same sequence of reduce-scatter calls on this communicator (the
        flags are counters); TIDE_RS_CHECK_ORDER=1 verifies (handle, call index) across ranks on
        every call, for debugging only. A call that raises mid-protocol poisons the communicator;
        with accumulate=True the output may then hold a partial update, so the process must exit."""
        shapes, itemsize = self._shapes(grads)
        if shapes != handle.shapes or itemsize != handle.itemsize:
            raise ValueError("gradient shapes differ from the registered table")
        for g in grads:
            if g.device != self.device:
                raise ValueError(f"gradient on {g.device}, communicator on {self.device}")
        return self._reduce_scatter(handle, [g.data_ptr() for g in grads], out, scale, accumulate, stream, read_event,
                                    ready_event, gate_event)


class RsHandle:
    def __init__(self, comm: TideComm, plan, shapes: list[tuple[int, int]], itemsize: int, index: int = 0):
        self.comm, self.plan, self.shapes, self.itemsize = comm, plan, shapes, itemsize
        self.index = index  # registration ordinal on the communicator (same on every rank)
        self.pieces = list(plan.pieces)  # (pieceElems, outOff) per segment
        self.out_numel = int(plan.out_elems)

    def reduce_scatter(self, grads, out=None, scale: float = 1.0, accumulate: bool = False, stream=None,
                       read_event=None, ready_event=None, gate_event=None):
        return self.comm.reduce_scatter(self, grads, out, scale, accumulate, stream, read_event, ready_event, gate_event)

    def split(self, out: torch.Tensor) -> list[torch.Tensor]:
        """Per-segment padded piece views of a packed output (stock's reduce_output slices)."""
        return [out.narrow(0, off, n) for n, off in self.pieces]

    @property
    def stage_bytes(self) -> int:
        return int(self.plan.stage_bytes)

    @property
    def calls(self) -> int:
        return self.plan.calls


class AgHandle:
    def __init__(self, comm: TideComm, plan, win, region_off: int, sizes: list[int], offsets: list[int],
                 base_off: int = 0):
        self.comm, self.plan, self.win = comm, plan, win
        # region_off/stage are window-relative; the views below subtract the arena tensor's own offset
        self.win_region_off = region_off
        self.region_off, self.sizes, self.offsets = region_off - base_off, sizes, offsets
        self.own_off = (plan.own_off - base_off) if plan.staged else -1  # my packed shard (staged plans)
        self.nranks, self.rank = comm.nranks, comm.rank
        self.shard_bytes = sum(sizes)
        self.region_bytes = self.shard_bytes * self.nranks

    def all_gather(self, stream: torch.cuda.Stream | None = None) -> torch.cuda.Event:
        return self.comm.all_gather(self, stream)

    def all_gather_on(self, stream: torch.cuda.Stream, free_event: torch.cuda.Event | None = None,
                      done_event: torch.cuda.Event | None = None) -> None:
        self.comm.all_gather_on(self, stream, free_event, done_event)

    # Byte views into a window-backed uint8 tensor `arena` (window-relative addressing).
    def region(self, arena: torch.Tensor) -> torch.Tensor:
        return arena.narrow(0, self.region_off, self.region_bytes)

    def landing_tensor(self, p: int) -> torch.Tensor:
        """Same bytes as landing(), but as a tensor whose storage starts there (storage_offset 0) with
        its own version counter -- what FSDP2 expects of an all-gather output buffer."""
        return self.win.tensor_range(self.win_region_off + self.offsets[p] * self.nranks, self.sizes[p] * self.nranks,
                                     self.comm.device.index)

    def own_pieces(self, arena: torch.Tensor) -> list[torch.Tensor]:
        """Where segment p must be before a call: in the packed shard (read in place by every call) when
        staged, else this rank's slot of the in-place landing."""
        if self.own_off >= 0:
            return [arena.narrow(0, self.own_off + o, n) for o, n in zip(self.offsets, self.sizes)]
        return [arena.narrow(0, self.region_off + o * self.nranks + self.rank * n, n) for o, n in zip(self.offsets, self.sizes)]

    def own_piece_tensors(self) -> list[torch.Tensor]:
        """own_pieces() as tensors whose storage starts at the piece (storage offset 0, like landing_tensor): a DTensor
        over such a local tensor keeps its wrapper's data_ptr at 0, which deepcopy's clone path relies on."""
        if self.own_off >= 0:
            offs = [self.plan.own_off + o for o in self.offsets]
        else:
            offs = [self.win_region_off + o * self.nranks + self.rank * n for o, n in zip(self.offsets, self.sizes)]
        return [self.win.tensor_range(off, n, self.comm.device.index) for off, n in zip(offs, self.sizes)]

    @property
    def num_descriptors(self) -> int:
        return self.plan.num_descriptors

    @property
    def calls(self) -> int:
        return self.plan.calls
