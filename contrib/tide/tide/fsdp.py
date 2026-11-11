# SPDX-FileCopyrightText: Copyright (c) 2026 Kuaishou Technology. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FSDP2 adapter: the all-gather lands parameter-major straight into each parameter's unsharded
storage, so FSDP's copy-out kernel (`split_with_sizes_copy`) disappears and no SM is used for the
collective.

    comm = tide.TideComm(...)
    model = fully_shard(...)            # as usual; `model` is the root
    adapter = tide.fsdp.install(model, comm)
    ...
    tide.fsdp.uninstall()               # optional: back to the stock path, arena released
    comm.close()

The sharded parameters of a TIDE group live in the arena (moved there once, at binding; the
parameter objects stay, their local tensors become window views): every call reads them in place,
so there is no copy-in either -- unless the group casts (fp32 shards, bf16 all-gather), where the
cast kernel writes the packed shard. Every FSDP parameter group is pinned to a slot of a symmetric
arena at its first all-gather (forward order, identical on all ranks). Groups that reshard after forward share `slots` ring slots
(round-robin; two suffice: with prefetch depth 1 only the computing group and the prefetched one
are unsharded at any time, and consecutive groups alternate slots in either traversal order);
groups that stay unsharded get a dedicated slot, except the last of them in module order (the last block,
unsharded only from its forward to its backward), which shares the ring (TIDE_RING_TAIL=0: dedicated). A slot is a landing area followed,
on multi-node, by a stage area at a fixed offset, so a stage never overlaps a landing another group
may still be computing on. A slot is written into only after the compute stream finished with its
previous occupant (event recorded at reshard) and after the previous all-gather on it completed --
that is the entry fence. A parameter with an FSDP2 all-gather extension (fsdp_pre_all_gather /
fsdp_post_all_gather, e.g. an MXFP8 weight gathered as E4M3 payload + E8M0 scales) is served as one
segment per input: the inputs are rebuilt and copied in at every call (like a cast group), each lands
in its own area, and FSDP's post hook runs over the landed views; the inputs' dtypes and numels must
not change between calls (TIDE_FSDP_EXTENSIONS=0 keeps such groups on the stock path). Ineligible
groups (CPU offload, non-dim-0 sharding, post-forward mesh, a process group TIDE does not cover) keep
the stock path. With subgroups=True, a
group whose process group is a regular subset of the ranks (e.g. the expert groups' dp_shard_mod_ep:
same rank count on every node, node-major) gets its own lane -- a communicator over that process
group, its own arena and ring -- beside the lane of the main communicator. install() wraps whatever
FSDP entry points are in place at that moment, so it goes last when other patches are installed.
"""
from __future__ import annotations

import collections
import inspect
import logging
import os
import statistics
import time
import typing
from dataclasses import dataclass, field

import torch
import torch.distributed as dist

# FSDP2 internals the adapter patches or reads; install() checks them (compat_problems) before
# touching anything, so a torch whose internals moved fails there with a list, not mid-step.
_TESTED_TORCH = ("2.13.",)
try:
    from torch.distributed.fsdp._fully_shard import _fsdp_collectives as _coll
    from torch.distributed.fsdp._fully_shard import _fsdp_param as _fp
    from torch.distributed.fsdp._fully_shard import _fsdp_param_group as _pg
    from torch.distributed.fsdp._fully_shard._fsdp_collectives import _get_gradient_divide_factors
    from torch.distributed.fsdp._fully_shard._fsdp_common import TrainingState as _TrainingState
    from torch.distributed.fsdp._fully_shard._fsdp_param import free_storage
    from torch.distributed.fsdp._fully_shard._fsdp_state import _get_module_fsdp_state
    _IMPORT_PROBLEM = ""
except ImportError as e:  # noqa: F841 - reported by compat_problems()
    _coll = _fp = _pg = _get_gradient_divide_factors = free_storage = _get_module_fsdp_state = _TrainingState = None
    _IMPORT_PROBLEM = f"FSDP2 internals not importable: {e}"

from tide.comm import AgHandle, RsHandle, TideComm, _all_same
from tide.table import byte_table, check_alignment

log = logging.getLogger("tide.fsdp")

# FSDP2's own collective classes differ only in where they allocate buffers (TIDE uses its arena);
# any other class is a user's custom collective (set_custom_all_gather / _reduce_scatter), left alone
_BUILTIN_AG = tuple(c for c in (getattr(_coll, n, None) for n in ("DefaultAllGather", "ProcessGroupAllocAllGather",
                                                                   "SymmMemAllGather")) if c is not None) if _coll else ()
_BUILTIN_RS = tuple(c for c in (getattr(_coll, n, None) for n in ("DefaultReduceScatter", "ProcessGroupAllocReduceScatter",
                                                                   "SymmMemReduceScatter")) if c is not None) if _coll else ()


def _custom(comm, builtin) -> bool:
    return comm is not None and bool(builtin) and type(comm) not in builtin


_ALIGN = 4096
_RS_MAX_SEGMENTS = 256  # kRsMaxSegments: the reduce-scatter kernels stage a plan's segment table in shared memory


def _round_up(x: int, a: int = _ALIGN) -> int:
    return (x + a - 1) // a * a


def _repoint(fp, data: torch.Tensor) -> None:
    """Make `data` (1-D, padded sharded numel) the storage of fp's sharded parameter -- what FSDP2's own
    reset_sharded_param does when a module op replaces the storage; the DTensor object stays."""
    padded = data.view(fp.padded_sharded_param_size)
    length = fp.sharded_param._local_tensor.size(0)
    fp._sharded_param_data = data
    fp.sharded_param._local_tensor = padded.narrow(0, 0, length)


class _PipelinedInputHold:
    """A pipelined reduce-scatter's first foreach_reduce value, the gradients only, paired with the read event;
    FSDP drops it once its current stream waited on that event. poison (test): NaN-fill them at the drop, on that
    stream, so a read after the read event shows in the result."""
    __slots__ = ("grads", "poison")

    def __init__(self, grads: tuple, poison: bool):
        self.grads, self.poison = grads, poison

    def __del__(self):
        if self.poison and self.grads:
            with torch.no_grad():
                for t in self.grads:
                    t.fill_(float("nan"))


@dataclass
class _Slot:
    off: int                                     # landing area
    stage_off: int                               # packed-shard stage area (-1 on a single node)
    region: int = 0                              # landing / stage bytes the slot was carved with
    stage: int = 0
    occupant: "_Group | None" = None
    free_event: torch.cuda.Event | None = None   # compute stream released the previous occupant
    ag_event: torch.cuda.Event | None = None     # previous all-gather on this slot completed


@dataclass
class _Lane:
    """One communicator pair and everything allocated on it: the groups over one process group."""
    comm: TideComm
    comm_rs: TideComm
    owned: bool = False              # communicators created by the adapter (closed at uninstall)
    groups: list = field(default_factory=list)
    arena: torch.Tensor | None = None
    rs_arena: torch.Tensor | None = None
    rs_stage_off: int = -1
    ring: list = field(default_factory=list)
    dedicated: list = field(default_factory=list)
    next_ring: int = 0
    tail: object = None              # TIDE_RING_TAIL: the unsharded-after-forward group moved into the ring


@dataclass
class _Group:
    pg: object                       # FSDPParamGroup
    lane: _Lane
    dtypes: list[torch.dtype]
    numels: list[int]                # padded sharded numel per param (= all-gather input numel)
    sizes: list[int]                 # bytes per segment
    offsets: list[int]
    shard_bytes: int
    region_bytes: int
    ring: bool                       # shares a ring slot (reshards after forward)
    segs: list[list[int]] = field(default_factory=list)  # per parameter: its segments (indices into dtypes / numels / sizes)
    ext: bool = False                # a parameter with an FSDP2 all-gather extension: the copy-in runs
                                     # `all_gather_inputs` (fsdp_pre_all_gather) each call, one piece per input
    fqn: str = "?"
    stock_ag: bool = False           # unshards after the ring tail in forward: keeps the stock all-gather
    own_off: int = -1                # packed shard: the sharded parameters' storage (-1: cast group)
    adopted: bool = False            # the sharded parameters live at own_off
    slot: _Slot | None = None
    handle: AgHandle | None = None
    landing: list[torch.Tensor] = field(default_factory=list)   # unsharded storages, one per param
    ag_meta: tuple = ()              # AllGatherResult constants (dtypes, numels, split sizes)
    region_view: torch.Tensor | None = None
    pieces: list[torch.Tensor] = field(default_factory=list)    # typed copy-in targets (cast path only)
    cast: bool = False               # sharded data dtype != all-gather dtype: the copy-in is a cast kernel
    bound: bool = False
    freed: int = 0
    calls: int = 0
    # reduce-scatter: unsharded-gradient table (numel, dim0) per param, input itemsize, plan
    rs_params: list = field(default_factory=list)  # the trainable params the table covers, in group order
    rs_shapes: list[tuple[int, int]] = field(default_factory=list)
    rs_itemsize: int = 0
    rs_handle: RsHandle | None = None
    rs_calls: int = 0
    rs_fused: int = 0                # reduce-scatters that accumulated straight into the existing grads
    # host seconds in this group's all-gather / reduce-scatter calls, waits for room in a full stream queue included;
    # the throughput-mode copy-in's launches (on the compute stream, whose queue is full in backward) apart
    host_ag_s: float = 0.0
    host_rs_s: float = 0.0
    host_copyin_s: float = 0.0


class TideFSDP:
    supports_extensions = True  # FSDP2 all-gather extensions served as multi-segment groups (TIDE_FSDP_EXTENSIONS)

    def __init__(self, comm: TideComm, slots: int = 2, comm_rs: TideComm | None = None, subgroups: bool = False):
        self.comm = comm
        # NCCL runs every host-RMA op of one communicator in issue order, so with one communicator a
        # backward prefetch all-gather waits behind the previous layer's reduce-scatter rounds; a
        # separate reduce-scatter communicator decouples the two chains (its own window for the stage).
        self.comm_rs = comm_rs if comm_rs is not None else comm
        ranks = tuple(dist.get_process_group_ranks(comm.group))
        if comm_rs is not None and (tuple(dist.get_process_group_ranks(comm_rs.group)) != ranks or comm_rs.device != comm.device):
            raise ValueError("tide.fsdp: comm_rs must span the same ranks and device as comm")
        self.main = _Lane(comm, self.comm_rs)
        self.lanes: dict[tuple, _Lane | None] = {ranks: self.main}  # global ranks -> lane (None: declined)
        self.subgroups = bool(subgroups)
        self._lane_by_pg: dict = {}  # process group -> lane or None (keyed by the object: no id reuse)
        if int(slots) < 2:
            raise ValueError("tide.fsdp: slots must be >= 2 (current group + prefetched group)")
        self.nslots = int(slots)
        self.groups: dict[int, _Group] = {}     # id(fsdp_params list) -> group
        self.stock_calls = 0
        self.all_gather = True    # False: reduce-scatter only, the all-gather stays on the stock path
        self.reduce_scatter = True
        # FSDP2 all-gather extensions (fsdp_pre_all_gather / fsdp_post_all_gather) as multi-segment groups;
        # TIDE_FSDP_EXTENSIONS=0 keeps them on the stock path as before
        self.extensions = os.environ.get("TIDE_FSDP_EXTENSIONS", "1") == "1"
        self.stock_rs_calls = 0
        self.stock_rs_reasons: collections.Counter = collections.Counter()  # (group fqn, reason) -> calls
        # TIDE_TRACE_AG_EVERY=N (with TIDE_TRACE_AG=1 on the comm): log GPU-side phase medians every N calls
        self._trace_every = int(os.environ.get("TIDE_TRACE_AG_EVERY", "0"))
        self._ag_calls = 0
        self._release_poison = os.environ.get("TIDE_FSDP_RELEASE_POISON", "0") == "1"  # test: NaN-fill at release
        # TIDE_FSDP_COPYIN=1: a group's gradients go into its slot once it reshards and are freed there, in stream
        # order; the reduce-scatter reads the slot and the slot's next all-gather waits for those reads
        self.copyin = os.environ.get("TIDE_FSDP_COPYIN", "0") == "1"  # install() agrees it
        self.rs_copyin_calls = 0
        self._slot_stream: torch.cuda.Stream | None = None
        self._noreadwait = "noreadwait" in os.environ.get("TIDE_FAULT", "").split(",")  # test: skip that wait
        # Multi-node multicast reduce-scatters are pipelined (read phase, then network phase): FSDP frees the inputs at
        # the next group's post_backward once read, and the next backward all-gather is issued before the network
        # phase (docs/usage.md, "Reduce-scatter")
        self.early_prefetch = os.environ.get("TIDE_FSDP_EARLY_PREFETCH", "1") == "1"  # install() agrees it
        self.rs_pipelined_calls = 0
        self.early_prefetches = 0
        self._out_hold: list[torch.Tensor] = []   # outputs of the last pipelined calls (their post-processing reads them)
        self._backward_cb = False
        # TIDE_FSDP_ASYNC_STATS_EVERY=N: per-call GPU timings of the pipelined path, logged every N calls on rank 0
        self._stats_every = int(os.environ.get("TIDE_FSDP_ASYNC_STATS_EVERY", "0"))
        self._stats_on = self._stats_every > 0
        self._rs_rec: list[tuple] = []            # (ready, read, full, post_backward entry or None)
        self._ag_rec: list[tuple] = []            # (before, after, backward) around the copy-out wait
        self.alloc_s: dict = {}                   # install-time accounting per phase (seconds, summed over lanes)

    # the main lane's allocation, under the names callers used before lanes existed
    arena = property(lambda self: self.main.arena)
    rs_arena = property(lambda self: self.main.rs_arena)
    rs_stage_off = property(lambda self: self.main.rs_stage_off)
    ring = property(lambda self: self.main.ring)
    dedicated = property(lambda self: self.main.dedicated)

    def _lane_of(self, group, create: bool = False) -> _Lane | None:
        """The lane serving process group `group`; with create (collect() only: collective over the
        group, reached at the same module on every rank) a new lane for a regular subgroup."""
        if group in self._lane_by_pg:
            return self._lane_by_pg[group]
        ranks = tuple(dist.get_process_group_ranks(group))
        if ranks not in self.lanes and create and self.subgroups:
            self.lanes[ranks] = self._new_lane(group, ranks)
        lane = self._lane_by_pg[group] = self.lanes.get(ranks)
        return lane

    def _new_lane(self, group, ranks: tuple) -> _Lane | None:
        why = None
        if list(ranks) != sorted(ranks) or len(ranks) < 2:
            why = "ranks not ascending"
        else:
            comm = TideComm(group=group, device=self.comm.device)
            shapes = [None] * comm.nranks
            dist.all_gather_object(shapes, (comm.n_nodes, comm.local_ranks), group=group)
            if len(set(shapes)) != 1 or comm.n_nodes * comm.local_ranks != comm.nranks:
                why = f"irregular (nodes x ranks per node per rank: {sorted(set(shapes))} for {comm.nranks} ranks)"
                comm.close()
        if why is not None:
            log.info("[TIDE-FSDP] process group of %d ranks stays on the stock path: %s", len(ranks), why)
            return None
        comm_rs = TideComm(group=group, device=self.comm.device) if self.comm_rs is not self.comm else comm
        if self.comm.rank == 0:
            log.info("[TIDE-FSDP] lane over %d ranks (%d nodes x %d)%s", comm.nranks, comm.n_nodes, comm.local_ranks,
                     " -- one rank per node" if comm.rail else "")
        return _Lane(comm, comm_rs, owned=True)

    # -- setup --------------------------------------------------------------------------------
    def collect(self, model: torch.nn.Module) -> None:
        root = _get_module_fsdp_state(model)
        if root is None:
            raise ValueError("tide.fsdp: install() expects the root fully_shard module")
        order: list[tuple[_Group, bool]] = []  # (group, is root) in module order
        for name, module in model.named_modules():
            state = _get_module_fsdp_state(module)
            if state is None:
                continue
            # FSDP's lazy init turns off reshard-after-forward for the root unless the user set it
            # explicitly; classify the way FSDP will behave, not the way it is configured now.
            pinned = state is root and getattr(root, "_auto_reshard_after_forward", True)
            for pg in state._fsdp_param_groups:
                if id(pg.fsdp_params) in self.groups:
                    continue
                g = self._describe(pg, ring=pg._reshard_after_forward and not pinned)
                if g is not None:
                    g.fqn = name or "<root>"
                    self.groups[id(pg.fsdp_params)] = g
                    order.append((g, state is root))
        if os.environ.get("TIDE_RING_TAIL", "1") == "1":
            self._ring_tail(order)

    @staticmethod
    def _ring_tail(order: list[tuple[_Group, bool]]) -> None:
        # The last group that stays unsharded after forward (the last block) lives only from its
        # forward to its own backward, alternating with its predecessor like any ring group, so it
        # shares the ring instead of taking a dedicated slot (a forward without autograd reshards it
        # at once: _post_forward). _claim still refuses a held slot.
        for lane in {id(g.lane): g.lane for g, _ in order}.values():
            mine = [(g, is_root) for g, is_root in order if g.lane is lane]
            tail = next((i for i in range(len(mine) - 1, -1, -1) if not mine[i][0].ring and not mine[i][1]), None)
            if tail is None or not any(g.ring for g, _ in mine[:tail]) or any(g.ring for g, _ in mine[tail + 1:]):
                continue
            g = mine[tail][0]
            g.ring = True
            lane.tail = g
            log.info("[TIDE-FSDP] group %s stays unsharded after forward but is the last one: it shares the ring", g.fqn)

    def _describe(self, pg, ring: bool) -> _Group | None:
        def decline(why: str):
            log.info("[TIDE-FSDP] group %s stays on the stock path: %s", getattr(pg, "_module_fqn", "?"), why)

        try:
            ag_group = pg._all_gather_process_group
        except Exception as e:  # noqa: BLE001 - FSDP raises for world-size-1 / unusual meshes
            return decline(f"no all-gather process group ({e})")
        lane = self._lane_of(ag_group, create=True)
        if lane is None:
            return decline("all-gather group not covered (differs from the TIDE communicator; subgroups=True for a regular subset)")
        if pg._use_post_forward_mesh:
            return decline("reshards to a post-forward mesh")
        if _custom(getattr(pg, "_all_gather_comm", None), _BUILTIN_AG):
            return decline("custom all-gather collective (set_custom_all_gather)")
        if any(not hasattr(fp, "param_dtype") for fp in pg.fsdp_params):
            pg._init_mp_dtypes()  # FSDP does this lazily at the first forward; same call, idempotent
        # One segment per all-gather input. A plain parameter has one (its shard, in the compute dtype);
        # a parameter with an FSDP2 all-gather extension (fsdp_pre_all_gather / fsdp_post_all_gather, e.g.
        # an FP8 weight gathered as payload + scales) has one per tensor the extension returns: TIDE
        # lands each in its own area, and FSDP's init_unsharded_param runs the post hook over the landed
        # views exactly as over its own buffers. The extension is probed once here (its outputs' dtypes
        # and numels must not change between calls; foreach_all_gather checks).
        dtypes, numels, sizes, segs, grad_dtypes, ext, cast = [], [], [], [], [], False, False
        for fp in pg.fsdp_params:
            if fp.offload_to_cpu:
                return decline("CPU offload")
            if fp.fsdp_placement.dim != 0:
                return decline("sharded on a dim other than 0")
            dt = fp.param_dtype or fp.orig_dtype
            grad_dtypes.append(dt)
            if hasattr(fp._sharded_local_tensor, "fsdp_pre_all_gather"):
                if not self.extensions:
                    return decline("all-gather extension (TIDE_FSDP_EXTENSIONS=0)")
                try:
                    inputs = fp.all_gather_inputs
                except Exception as e:  # noqa: BLE001 - an extension that cannot run outside an unshard
                    return decline(f"all-gather extension failed to produce its inputs ({type(e).__name__}: {e})")
                if not inputs:
                    return decline("all-gather extension with no input")
                mine = []
                for t in inputs:
                    n = int(t.numel())
                    if n == 0:
                        return decline("all-gather extension with an empty input")
                    mine.append(len(dtypes))
                    dtypes.append(t.dtype)
                    numels.append(n)
                    sizes.append(n * t.element_size())
                segs.append(mine)
                ext = True
                continue
            n = int(fp._sharded_param_data.numel())
            if n == 0:
                return decline("empty shard")
            segs.append([len(dtypes)])
            dtypes.append(dt)
            numels.append(n)
            sizes.append(n * dt.itemsize)
            cast |= fp._sharded_param_data.dtype != dt
        sizes, offsets = byte_table(sizes)
        world = lane.comm.nranks
        why = check_alignment(sizes, offsets, [dt.itemsize for dt in dtypes], world)
        if why is not None:
            return decline(why)
        shard = sum(sizes)
        # an extension group copies in every call like a cast group (its inputs are built per unshard)
        g = _Group(pg=pg, lane=lane, dtypes=dtypes, numels=numels, sizes=sizes, offsets=offsets, shard_bytes=shard,
                   region_bytes=shard * world, ring=ring, segs=segs, ext=ext, cast=cast or ext)
        if ext:
            log.info("[TIDE-FSDP] group %s: %d all-gather extension segment(s) over %d parameter(s)",
                     getattr(pg, "_module_fqn", "?"), len(dtypes), len(pg.fsdp_params))
        # reduce-scatter table: the unsharded gradient of param p has the unsharded shape and the
        # compute dtype (an extension changes what is gathered, not the gradient). fp32 reduction only
        # (the kernels accumulate in fp32): FSDP reduces in `_reduce_dtype`, or in the gradient dtype
        # when that is None -- the same test foreach_reduce applies at call time, so a bf16-reducing
        # group registers no plan and reserves no stage.
        dtypes = grad_dtypes
        effective_reduce = getattr(pg, "_reduce_dtype", None) or dtypes[0]
        if self.reduce_scatter and effective_reduce == torch.float32 and \
                not _custom(getattr(pg, "_reduce_scatter_comm", None), _BUILTIN_RS) and \
                len(set(dtypes)) == 1 and dtypes[0] in (torch.bfloat16, torch.float32):
            # FSDP2 reduce-scatters only the params that get gradients; frozen ones (requires_grad False) never
            # do, so the table covers the trainable params, in group order
            rs_params = [fp for fp in pg.fsdp_params if fp.sharded_param.requires_grad]
            if len(rs_params) > _RS_MAX_SEGMENTS:
                log.info("[TIDE-FSDP] group %s keeps the stock reduce-scatter: %d trainable params, a plan holds %d",
                         getattr(pg, "_module_fqn", "?"), len(rs_params), _RS_MAX_SEGMENTS)
            else:
                g.rs_params = rs_params
                g.rs_shapes = [(int(torch.Size(fp._orig_size).numel()), int(fp._orig_size[0])) for fp in rs_params]
                g.rs_itemsize = dtypes[0].itemsize
        lane.groups.append(g)
        return g

    @staticmethod
    def _layout(lane: _Lane, groups: list[_Group]) -> tuple[int, int]:
        """(landing bytes, receive-stage bytes) of a slot able to hold any of `groups`."""
        region = max((g.region_bytes for g in groups), default=0)
        stage = max((lane.comm.stage_bytes(g.shard_bytes, own_separate=not g.cast) for g in groups),
                    default=0) if lane.comm.n_nodes > 1 else 0
        return _round_up(region), _round_up(stage)

    def allocate(self) -> None:
        """Collective. Per lane, one arena: ring slots sized for the largest ring group + dedicated slots."""
        for lane in self.lanes.values():
            if lane is not None:
                self._allocate(lane)

    def _allocate(self, lane: _Lane) -> None:
        comm, comm_rs = lane.comm, lane.comm_rs
        for g in lane.groups:  # foreach_reduce finds the group through its params (a stock fallback then names it)
            for fp in g.pg.fsdp_params:
                fp._tide_group_params = g.pg.fsdp_params
        ring_groups = [g for g in lane.groups if g.ring] if self.all_gather else []
        ded_groups = [g for g in lane.groups if not g.ring] if self.all_gather else []
        slots: list[_Slot] = []  # in allocation order
        total = 0

        def carve(region: int, stage: int) -> None:
            nonlocal total
            slots.append(_Slot(total, total + region if stage else -1, region, stage))
            total += region + stage

        # Ring groups take the slots in turn at their first all-gather, in module order (the tail rule relies on it
        # too): slot k is sized for groups k, k + nring, ..., so one outsized group widens one slot; _group_for checks.
        nring = min(self.nslots, len(ring_groups))
        layouts = [self._layout(lane, ring_groups[k::nring]) for k in range(nring)]
        # default only without expert groups (with them the main lane's stage is small)
        expert = any(ln is not self.main for ln in self._lane_by_pg.values())
        if os.environ.get("TIDE_FSDP_SHARED_STAGE", "0" if expert else "1") == "1" and any(s for _, s in layouts):
            # one receive stage for every ring slot: overlapping stage areas share a credit channel, so a group's
            # remote writes wait until the previous ring all-gather has scattered out of it
            stage = max(s for _, s in layouts)
            stage_off, total = total, total + stage
            for region, _ in layouts:
                slots.append(_Slot(total, stage_off, region, stage))
                total += region
        else:
            for region, stage in layouts:
                carve(region, stage)
        ring_region = max((s.region for s in slots), default=0)
        ring_slot = max((s.region + s.stage for s in slots), default=0)
        ring_bytes = total
        for g in ded_groups:
            region, stage = self._layout(lane, [g])
            if ring_region and region >= ring_region // 2 and self.comm.rank == 0:
                log.warning("[TIDE-FSDP] group %s stays unsharded after forward and takes a dedicated slot of %.2f GiB "
                            "(a ring slot is %.2f GiB); only the last such group in module order shares the ring",
                            g.fqn, (region + stage) / 2**30, ring_slot / 2**30)
            carve(region, stage)
        ded_bytes = total - ring_bytes
        for g in ring_groups + ded_groups:  # the sharded parameters themselves; a cast target lives in its slot
            if g.cast:
                continue
            g.own_off = total
            total += _round_up(g.shard_bytes)
        own_bytes = total - ring_bytes - ded_bytes
        rs_groups = [g for g in lane.groups if g.rs_shapes]
        tick = time.perf_counter()

        def lap(key: str) -> None:
            nonlocal tick
            now = time.perf_counter()
            self.alloc_s[key] = self.alloc_s.get(key, 0.0) + now - tick
            tick = now

        if rs_groups:
            # copy-in frees the gradients at the copy and the reads only hold the slot: no early reads, so each outbox
            # bank holds two NIC rounds instead of a block per remote node
            rs_plan = comm_rs.tune_rs([(g.rs_shapes, g.rs_itemsize) for g in rs_groups], read_early=not self.copyin,
                                      pipeline=2)
            if rs_plan is not None and self.comm.rank == 0:
                for note in rs_plan.notes:
                    log.warning("[TIDE-FSDP] reduce-scatter solver: %s", note)
        lap("rs_tune")
        rs_bytes = comm_rs.rs_area_bytes([(g.rs_shapes, g.rs_itemsize) for g in rs_groups]) if rs_groups else 0
        split = comm_rs is not comm
        if rs_bytes and not split:
            lane.rs_stage_off = total
            total += _round_up(rs_bytes)
        if total == 0 and not (rs_bytes and split):
            return
        if total:
            lane.arena = comm.alloc_arena(total)
        if rs_bytes and split:  # stage area in its own window on the reduce-scatter communicator
            lane.rs_arena = comm_rs.alloc_arena(_round_up(rs_bytes))
            lane.rs_stage_off = 0
        else:
            lane.rs_arena = lane.arena
        lap("arena")
        lane.ring = slots[:nring]
        for g, s in zip(ded_groups, slots[nring:]):
            lane.dedicated.append(s)
            self._bind(g, s)
        lap("bind")
        # one reduce-scatter stage area shared by every group (plans that overlap share a channel)
        for g in rs_groups:
            g.rs_handle = comm_rs.register_rs(lane.rs_arena, lane.rs_stage_off, (g.rs_shapes, g.rs_itemsize))
        lap("rs_register")
        if rs_groups:
            p = rs_groups[0].rs_handle.plan
            if not self.copyin and p.pipeline == 2 and p.outbox_bank < p.parts * (comm_rs.n_nodes - 1):
                raise RuntimeError(f"tide.fsdp: efficiency mode needs a full outbox bank ({p.parts * (comm_rs.n_nodes - 1)} "
                                   f"blocks, got {p.outbox_bank} from TIDE_RS_OUTBOX): a partial bank makes "
                                   "read_event wait for the NIC rounds, which stalls FSDP's next post-backward")
        gib = 2**30
        comm_bytes = comm.device_bytes + (comm_rs.device_bytes if split else 0)
        parts = (ring_bytes, ded_bytes, own_bytes, _round_up(rs_bytes), comm_bytes)
        if self.comm.rank == 0:
            if rs_groups:  # the registered plan (outbox blocks and banks, passes, pipeline)
                p = rs_groups[0].rs_handle.plan
                box = (f", outbox {p.outbox_slots} blocks in {p.pipeline} bank{'s' if p.pipeline > 1 else ''}"
                       if comm_rs.n_nodes > 1 and not comm_rs.rail else "")
                log.info("[TIDE-FSDP] reduce-scatter over %d ranks: %s CTAs sum / %s landing, k=%s, chunk %d KiB, %d parts%s, "
                         "pipeline %d (solver; env wins)", comm_rs.nranks, *comm_rs.rs_tuning[:3],
                         comm_rs.rs_tuning[3] // 1024, p.parts, box, p.pipeline)
            log.info("[TIDE-FSDP] %d groups over %d ranks: arena %.2f GiB (ring %.2f, dedicated %.2f, sharded parameters "
                     "%.2f, rs %.2f) %s; communicator %.2f GiB outside torch stats", len(lane.groups), comm.nranks,
                     sum(parts[:4]) / gib, *(b / gib for b in parts[:4]),
                     "in torch's reserved memory" if getattr(comm, "arena_in_torch", False) else "outside torch stats",
                     parts[4] / gib)

    def _bind(self, g: _Group, slot: _Slot) -> None:
        """Collective (table check). Pins the group to `slot`; landing views are fixed from now on."""
        g.slot = slot
        g.handle = g.lane.comm.register_ag(g.lane.arena, slot.off, g.sizes, g.offsets, stage_off=slot.stage_off, own_off=g.own_off)
        # own-storage tensors: FSDP2 builds the unsharded parameter with as_strided(storage_offset=0)
        g.landing = [g.handle.landing_tensor(p).view(g.dtypes[p]) for p in range(len(g.sizes))]
        g.ag_meta = ([[g.dtypes[i] for i in segs] for segs in g.segs], [[g.numels[i] for i in segs] for segs in g.segs],
                     list(g.numels))
        g.region_view = g.handle.region(g.lane.arena)
        # mixed precision (fp32 shards, bf16 all-gather): the cast kernel writes the slot's own shard each
        # call (the stage's first slot, or the landing in place on one node) after the slot's entry wait;
        # with matching dtypes the sharded parameters move into the packed shard at the first all-gather
        # storage offset 0: the DTensor wrappers torch derives from an adopted shard (state_dict, detach) keep
        # data_ptr 0, else deepcopy (EMA's original_state_dict) copies their data-less storage and fails
        g.pieces = [t.view(dt) for t, dt in zip(g.handle.own_piece_tensors(), g.dtypes)]
        for fp in g.pg.fsdp_params:  # a stock all-gather since the last install built one over its own buffers
            fp.__dict__.pop("_unsharded_param", None)
        if slot.ag_event is None:
            slot.ag_event = torch.cuda.Event()  # one per slot, re-recorded each call (the slot's group waits it before the next reuse)
        for fp in g.pg.fsdp_params:
            fp._tide_group = g

    @torch.no_grad()
    def _adopt(self, g: _Group) -> None:
        """Move the group's sharded parameters into its packed shard (one copy, on the current stream). Not
        before FSDP's lazy init: its reset_sharded_param re-pads a padded shard into a new buffer."""
        for fp, piece in zip(g.pg.fsdp_params, g.pieces):
            old = fp._sharded_param_data
            if piece.numel() != old.numel() or piece.dtype != old.dtype:
                raise RuntimeError("tide.fsdp: sharded parameter changed shape or dtype since install()")
            piece.copy_(old)
            _repoint(fp, piece)
        g.adopted = True

    @torch.no_grad()
    def _evict(self, g: _Group) -> None:
        """Move the sharded parameters back to allocator memory (uninstall)."""
        for fp, piece in zip(g.pg.fsdp_params, g.pieces):
            _repoint(fp, piece.clone())
        g.adopted = False

    def _group_for(self, fsdp_params) -> _Group | None:
        g = self.groups.get(id(fsdp_params))
        if g is None or g.stock_ag or g.lane.arena is None or not self.all_gather:
            return None
        if g.slot is None:  # first all-gather of a ring group: pin the next ring slot
            lane = g.lane
            if lane.tail is not None and lane.tail.slot is not None and g is not lane.tail:
                # module order was not forward order: the tail holds its ring slot until its backward, so this
                # group would collide with it -- it keeps the stock all-gather (its reduce-scatter stays on TIDE)
                g.stock_ag = True
                if self.comm.rank == 0:
                    log.warning("[TIDE-FSDP] %s unshards after the ring tail %s in forward (module order is not forward "
                                "order): it keeps the stock all-gather; TIDE_RING_TAIL=0 gives the tail a dedicated slot",
                                g.fqn, lane.tail.fqn)
                return None
            region, stage = self._layout(lane, [g])
            fits = [region <= s.region and stage <= s.stage for s in lane.ring]
            k = lane.next_ring % len(lane.ring)
            if not fits[k] and lane.next_ring == 0 and any(fits):
                k = fits.index(True)  # the lane's first group may start the rotation at any slot that holds it
            if not fits[k]:
                # first all-gathers out of module order: the slot was sized for other groups
                g.stock_ag = True
                if self.comm.rank == 0:
                    log.warning("[TIDE-FSDP] %s does not fit ring slot %d (sized for the groups in module order): it keeps "
                                "the stock all-gather", g.fqn, k)
                return None
            self._bind(g, lane.ring[k])
            lane.next_ring = k + 1
        return g

    def _claim(self, g: _Group) -> _Slot:
        slot = g.slot
        occ = slot.occupant
        if occ is not None and occ is not g:
            # An occupant whose all-gather was dropped by FSDP's iteration reset never came back
            # through free_unsharded_param; its compute never ran, so the slot is free.
            if occ.pg._all_gather_result is None and not occ.pg.is_unsharded:
                self._release(slot)
            else:
                raise RuntimeError("tide.fsdp: slot still held by another unsharded group; raise slots=")
        slot.occupant = g
        g.freed = 0
        return slot

    def _release(self, slot: _Slot) -> None:
        if slot.free_event is None:
            slot.free_event = torch.cuda.Event()
        slot.free_event.record(torch.cuda.current_stream())
        slot.occupant = None

    # -- the two patched FSDP entry points -------------------------------------------------------
    @torch.no_grad()
    def foreach_all_gather(self, fsdp_params, group, async_op, copy_in_stream, ag_stream, device, all_gather_comm):
        g = self._group_for(fsdp_params)
        if g is None:
            self.stock_calls += 1
            return _stock.foreach_all_gather(fsdp_params, group, async_op, copy_in_stream, ag_stream, device,
                                             all_gather_comm)
        if async_op:
            raise RuntimeError("tide.fsdp: async unshard is not supported on TIDE-managed groups")
        if _custom(all_gather_comm, _BUILTIN_AG):
            raise RuntimeError("tide.fsdp: set_custom_all_gather on a TIDE-managed group after install(); "
                               "set it before install() to keep that group on the stock path")
        torch.cuda.nvtx.range_push("tide.fsdp.ag")  # host time per call, for profiles
        t0 = time.perf_counter()
        slot = self._claim(g)
        # No wait on the current stream: it would hold every prefetch behind the compute already queued.
        # FSDP's root pre-forward orders this stream after the optimizer; the slot is written only after
        # the previous occupant's compute released it.
        if slot.free_event is not None:
            ag_stream.wait_event(slot.free_event)
        if g.ext:
            # Each parameter's all-gather inputs (the extension's fsdp_pre_all_gather, which also records
            # the sizes / metadata init_unsharded_param hands to fsdp_post_all_gather; a plain parameter's
            # cast shard) copied into its pieces, on the all-gather stream like the cast copy-in.
            torch.cuda.nvtx.range_push("tide.fsdp.ag.extcopy")
            g.lane.comm.await_stage_reads(ag_stream)  # the target is the stage's own slot
            with torch.cuda.stream(ag_stream):
                srcs = []
                for fp, segs in zip(fsdp_params, g.segs):
                    inputs = fp.all_gather_inputs
                    if len(inputs) != len(segs) or any(t.numel() != g.numels[i] or t.dtype != g.dtypes[i]
                                                       for t, i in zip(inputs, segs)):
                        raise RuntimeError(
                            f"tide.fsdp: group {g.fqn}: all-gather inputs changed since install() "
                            f"(got {[(tuple(t.shape), t.dtype) for t in inputs]}, registered "
                            f"{[(g.numels[i], g.dtypes[i]) for i in segs]}); an extension whose inputs vary "
                            f"between unshards cannot be served by TIDE -- keep that group on the stock path")
                    srcs.extend(t.reshape(-1) for t in inputs)
                torch._foreach_copy_([piece.view(-1) for piece in g.pieces], srcs)
            torch.cuda.nvtx.range_pop()
        elif g.cast:
            torch.cuda.nvtx.range_push("tide.fsdp.ag.castcopy")
            g.lane.comm.await_stage_reads(ag_stream)  # the target is the stage's own slot
            with torch.cuda.stream(ag_stream):
                torch._foreach_copy_(g.pieces, [fp._sharded_param_data for fp in fsdp_params])
            torch.cuda.nvtx.range_pop()
        else:
            if not g.adopted:
                self._adopt(g)
                ag_stream.wait_stream(torch.cuda.current_stream())
            for fp, piece in zip(fsdp_params, g.pieces):  # the shards must still be where the plan reads them
                if fp._sharded_param_data.data_ptr() != piece.data_ptr():
                    raise RuntimeError("tide.fsdp: a sharded parameter's storage moved (load_state_dict(assign=True)? "
                                       "module._apply?) -- uninstall() before that and install() again")
        torch.cuda.nvtx.range_push("tide.fsdp.ag.call")
        # torch profiler: a user annotation, which kineto spans over the GPU work this call launches
        with torch.profiler.record_function("tide::ag " + g.fqn):
            # decoupled: ag_stream does not wait for the node's other writes into the landing (its slowest rail);
            # compute waits on slot.ag_event, the next writer on slot.free_event
            g.lane.comm.all_gather_on(g.handle, ag_stream, slot.free_event, slot.ag_event)
        torch.cuda.nvtx.range_pop()
        g.calls += 1
        g.host_ag_s += time.perf_counter() - t0
        torch.cuda.nvtx.range_pop()
        self._ag_calls += 1
        if self._trace_every and self._ag_calls % self._trace_every == 0:
            self._log_trace()
        dtypes, numels, splits = g.ag_meta
        return _coll.AllGatherResult(
            all_gather_output=g.region_view,
            all_gather_event=slot.ag_event,
            all_gather_work=None,
            param_all_gather_input_dtypes=dtypes,
            param_all_gather_input_numels=numels,
            all_gather_input_split_sizes=splits,
        )

    def _log_trace(self) -> None:
        for ranks, lane in self.lanes.items():  # every lane: expert subgroups trace on their own communicators
            if lane is None:
                continue
            rows = lane.comm.trace_ag()
            if not rows:
                continue
            n = len(rows[0][1])
            rows = [r for r in rows if len(r[1]) == n]
            med = [statistics.median(r[1][p] for r in rows) for p in range(n)]
            p90 = [sorted(r[1][p] for r in rows)[int(0.9 * (len(rows) - 1))] for p in range(n)]
            names = ["credits", "fence"] + [f"r{j}" for j in range(1, n - 3)] + ["scatter", "done"]
            log.info("[TIDE-FSDP] trace_ag lane %d ranks, rank %d n=%d total med %.1f ms | %s", len(ranks),
                     lane.comm.rank, len(rows), sum(med), " ".join(f"{k}={m:.1f}/{q:.1f}" for k, m, q in zip(names, med, p90)))

    @torch.no_grad()
    def foreach_all_gather_copy_out(self, all_gather_result, fsdp_params, group):
        g = self.groups.get(id(fsdp_params))
        if g is None or g.slot is None:
            return _stock.foreach_all_gather_copy_out(all_gather_result, fsdp_params, group)
        if all_gather_result.all_gather_event is not None:
            cur = torch.cuda.current_stream()
            before = after = None
            if self._stats_on:
                before = torch.cuda.Event(enable_timing=True)
                before.record(cur)
            cur.wait_event(all_gather_result.all_gather_event)
            if before is not None:
                after = torch.cuda.Event(enable_timing=True)
                after.record(cur)
                self._ag_rec.append((before, after, torch._C._current_graph_task_id() != -1))
        if not g.bound:
            for fp, segs in zip(fsdp_params, g.segs):
                fp.all_gather_outputs = [g.landing[i] for i in segs]
            g.bound = True

    # -- reduce-scatter -------------------------------------------------------------------------
    def _rs_stock_reason(self, g, world, fsdp_params, unsharded_grads, reduce_scatter_group, reduce_scatter_comm,
                         reduce_dtype, all_reduce_group, all_reduce_grads, partial_reduce_output, all_reduce_hook) -> str | None:
        if g is None:
            return "not a TIDE group"
        if g.rs_handle is None:
            return "no reduce-scatter plan"
        if world == 1:
            return "single-rank group"
        if all_reduce_group is not None and not all_reduce_grads:
            return "HSDP microbatch without the replica all-reduce"
        if _custom(reduce_scatter_comm, _BUILTIN_RS):
            return "custom reduce-scatter collective"
        if partial_reduce_output is not None:
            return "partial reduce output"
        if all_reduce_hook is not None:
            return "all-reduce hook"
        dtype = reduce_dtype or unsharded_grads[0].dtype
        if dtype != torch.float32:
            return f"reduce dtype {dtype}"
        if len(fsdp_params) != len(g.rs_params) or any(a is not b for a, b in zip(fsdp_params, g.rs_params)):
            have = {id(p) for p in fsdp_params}
            missing = [getattr(p, "_param_fqn", "?") for p in g.rs_params if id(p) not in have]
            more = " ..." if len(missing) > 3 else ""
            return (f"{len(fsdp_params)} params have gradients, the table covers {len(g.rs_params)} trainable ones "
                    f"(missing {', '.join(missing[:3])}{more})")
        if [(int(t.numel()), int(t.shape[0])) for t in unsharded_grads] != g.rs_shapes:
            return "gradient shapes differ from the plan"
        if any(t.dtype.itemsize != g.rs_itemsize or not t.is_contiguous() for t in unsharded_grads):
            return "gradient dtype or layout differs from the plan"
        if self._lane_of(reduce_scatter_group) is not g.lane:
            return "reduce-scatter group is not the group's lane"
        # a hook would run while a pipelined call's network phase is still writing the sharded gradient
        if int(g.rs_handle.plan.pipeline) == 2 and any(getattr(fp.sharded_param, "_post_accumulate_grad_hooks", None)
                                                       for fp in fsdp_params):
            return "post-accumulate-grad hooks on a pipelined reduce-scatter"
        return None

    def _copyin_views(self, g: _Group, fsdp_params, unsharded_grads) -> list[torch.Tensor] | None:
        """Each gradient's place in the slot its group has just left (resharded, not claimed yet), or None: no slot,
        the group keeps its parameters after backward, a gradient of another dtype (the copy must not round), or an
        extension group (its landing is per segment, not per parameter)."""
        if not self.copyin or g.ext or g.slot is None or not g.landing or g.slot.occupant is not None or \
                g.pg.is_unsharded:
            return None
        landing = {id(fp): t.view(-1) for fp, t in zip(g.pg.fsdp_params, g.landing)}
        views = []
        for fp, grad in zip(fsdp_params, unsharded_grads):
            t = landing.get(id(fp))
            if t is None or t.dtype != grad.dtype or t.numel() < grad.numel():
                return None
            views.append(t.narrow(0, 0, grad.numel()).view(grad.shape))
        return views

    @torch.no_grad()
    def foreach_reduce(self, fsdp_params, unsharded_grads, reduce_scatter_group, reduce_scatter_stream,
                       reduce_scatter_comm, orig_dtype, reduce_dtype, device, gradient_divide_factor,
                       all_reduce_group, all_reduce_stream, all_reduce_grads, partial_reduce_output,
                       all_reduce_hook, force_sum_reduction_for_comms=False):
        # post_backward passes a fresh list (the params that have gradients): find the group by its first param
        g = self.groups.get(id(getattr(fsdp_params[0], "_tide_group_params", None))) if fsdp_params else None
        world = reduce_scatter_group.size() if reduce_scatter_group is not None else 1
        why = self._rs_stock_reason(g, world, fsdp_params, unsharded_grads, reduce_scatter_group, reduce_scatter_comm,
                                    reduce_dtype, all_reduce_group, all_reduce_grads, partial_reduce_output, all_reduce_hook)
        if why is not None:
            self.stock_rs_calls += 1
            key = (getattr(g.pg, "_module_fqn", "?") if g is not None else "?", why)
            if key not in self.stock_rs_reasons:
                log.warning("[TIDE-FSDP] reduce-scatter of group %s falls back to stock: %s", *key)
            self.stock_rs_reasons[key] += 1
            return _stock.foreach_reduce(fsdp_params, unsharded_grads, reduce_scatter_group, reduce_scatter_stream,
                                         reduce_scatter_comm, orig_dtype, reduce_dtype, device, gradient_divide_factor,
                                         all_reduce_group, all_reduce_stream, all_reduce_grads, partial_reduce_output,
                                         all_reduce_hook, force_sum_reduction_for_comms)
        t0 = time.perf_counter()
        hsdp = all_reduce_group is not None
        pre, post, rs_op, ar_op = _get_gradient_divide_factors(reduce_scatter_group, all_reduce_group, torch.float32,
                                                                device.type, gradient_divide_factor,
                                                                force_sum_reduction_for_comms)
        scale = 1.0
        if pre:
            scale /= pre
        if post:
            scale /= post
        op_type = rs_op if isinstance(rs_op, dist.ReduceOp.RedOpType) else rs_op.op
        if op_type == dist.ReduceOp.AVG:
            scale /= world
        elif op_type == dist.ReduceOp.PREMUL_SUM:
            scale *= float(rs_op.__getstate__()[1])
        h = g.rs_handle
        fp32_out = orig_dtype in (None, torch.float32)
        # Accumulating microbatch: the sharded grads set by the previous call are views of one packed
        # fp32 buffer -- reduce straight into it (out += scale * sum) instead of into a new buffer
        # that is then added on (one fewer pass over the fp32 gradient, no allocation).
        # (HSDP: the replica all-reduce must see this microbatch alone, so the accumulation follows it)
        packed = self._packed_grads(g, fsdp_params, world) if fp32_out and not hsdp else None
        current = torch.cuda.current_stream()
        views = self._copyin_views(g, fsdp_params, unsharded_grads)
        copyin_s = 0.0
        if views is not None:  # on the compute stream, so the gradients can be freed right after it in stream order
            tc = time.perf_counter()
            torch.cuda.nvtx.range_push("tide.fsdp.copyin")
            torch._foreach_copy_(views, unsharded_grads)
            if self._release_poison:
                for t in unsharded_grads:
                    t.fill_(float("nan"))
            torch.cuda.nvtx.range_pop()
            copyin_s = time.perf_counter() - tc
            g.host_copyin_s += copyin_s
        torch.cuda.nvtx.range_push("tide.fsdp.rs")  # after the copy-in: profiles read it as the reduce-scatter's host time
        src = unsharded_grads if views is None else views
        out = packed
        pipelined = int(h.plan.pipeline) == 2
        if pipelined:
            # the next backward all-gather first: on a shared comm stream it then precedes this network phase
            gate = self._early_prefetch(g, reduce_scatter_stream) if self.early_prefetch else None
            ready = torch.cuda.Event(enable_timing=self._stats_on)
            ready.record(current)  # the gradients are complete: the reads wait for this, not for the comm stream
            reduce_scatter_stream.wait_event(ready)
            read_event = torch.cuda.Event(enable_timing=self._stats_on)
        else:
            reduce_scatter_stream.wait_stream(current)  # the gradients are complete
            read_event = None
        with torch.cuda.stream(reduce_scatter_stream):
            if out is None:
                out = torch.empty(h.out_numel, dtype=torch.float32, device=device)
            with torch.profiler.record_function("tide::rs " + g.fqn):
                if pipelined:
                    g.lane.comm_rs.reduce_scatter(h, src, out=out, scale=scale, accumulate=packed is not None,
                                                  stream=reduce_scatter_stream, read_event=read_event, ready_event=ready,
                                                  gate_event=gate)
                else:
                    g.lane.comm_rs.reduce_scatter(h, src, out=out, scale=scale,
                                                  accumulate=packed is not None, stream=reduce_scatter_stream)
            reduce_scatter_event = torch.cuda.Event(enable_timing=self._stats_on and pipelined)
            reduce_scatter_event.record(reduce_scatter_stream)
        if views is not None:
            self.rs_copyin_calls += 1
            # the slot's next all-gather waits on free_event (here and in the AG's ready fence): after the reads now,
            # not just after the compute released the parameters (reshard recorded that)
            if not self._noreadwait:
                if self._slot_stream is None:
                    self._slot_stream = torch.cuda.Stream(device=device)
                self._slot_stream.wait_event(read_event if read_event is not None else reduce_scatter_event)
                if g.slot.free_event is None:
                    g.slot.free_event = torch.cuda.Event()
                g.slot.free_event.record(self._slot_stream)
        post_stream, all_reduce_input, all_reduce_event = reduce_scatter_stream, None, None
        if hsdp:  # the replicas' sum stays on the stock collective, as in FSDP's foreach_reduce
            all_reduce_stream.wait_stream(reduce_scatter_stream)
            with torch.cuda.stream(all_reduce_stream):
                dist.all_reduce(out, group=all_reduce_group, op=ar_op)
                all_reduce_input, all_reduce_event = out, all_reduce_stream.record_event()
            post_stream = all_reduce_stream
        with torch.cuda.stream(post_stream):
            reduce_output = out if fp32_out else out.to(orig_dtype)
            off = 0
            for (n, d0), fp in zip(g.rs_shapes, fsdp_params):
                if packed is None:
                    new_sharded_grad = torch.as_strided(reduce_output, size=fp.sharded_size,
                                                        stride=fp.contiguous_sharded_stride, storage_offset=off)
                    if fp.sharded_param.grad is not None:
                        fp.sharded_param.grad._local_tensor += new_sharded_grad
                    else:
                        fp.sharded_param.grad = fp.to_sharded_dtensor(new_sharded_grad)
                for hook in (getattr(fp.sharded_param, "_post_accumulate_grad_hooks", {}) or {}).values():
                    hook(fp.sharded_param)
                off += ((d0 + world - 1) // world) * (n // d0)
            post_reduce_event = post_stream.record_event()
        # The gradients are read until reduce_scatter_event (pipelined: read_event): FSDP holds this first return value
        # until its compute stream waited on that event, then frees it in stream order (record_stream would add the
        # host's lead). Copied in, nothing reads them any more: nothing to hold.
        grads = tuple(unsharded_grads) if views is None else ()
        if pipelined:
            # FSDP frees the gradients once its stream waited on read_event; the output stays held here until the
            # next call (its post-processing above reads it; reuse is on its allocation stream, after that)
            keepalive = _PipelinedInputHold(grads, self._release_poison)
            self._out_hold = self._out_hold[-1:] + [out]
            if not self._backward_cb and torch._C._current_graph_task_id() != -1:
                torch.autograd.Variable._execution_engine.queue_callback(self._end_of_backward)
                self._backward_cb = True
            self.rs_pipelined_calls += 1
            if self._stats_on:
                self._rs_rec.append((ready, read_event, reduce_scatter_event, g.pg.__dict__.pop("_tide_pbe", None)))
                if self.rs_pipelined_calls % self._stats_every == 0:
                    self._log_pipeline_stats()
            unsharded_grads.clear()
            g.rs_calls += 1
            g.rs_fused += int(packed is not None)
            g.host_rs_s += time.perf_counter() - t0 - copyin_s
            torch.cuda.nvtx.range_pop()
            # copied in: FSDP's compute stream has nothing to wait for (FSDP skips a None event)
            return (keepalive, None if views is not None else read_event, post_stream, post_reduce_event,
                    all_reduce_input, all_reduce_event, None)
        keepalive = (out, grads)
        unsharded_grads.clear()
        g.rs_calls += 1
        g.rs_fused += int(packed is not None)
        g.host_rs_s += time.perf_counter() - t0 - copyin_s
        torch.cuda.nvtx.range_pop()
        return (keepalive, None if views is not None else reduce_scatter_event, post_stream, post_reduce_event,
                all_reduce_input, all_reduce_event, None)

    def _early_prefetch(self, g: _Group, reduce_scatter_stream) -> torch.cuda.Event | None:
        """Issue now the backward all-gather FSDP2 issues at the next group's pre_backward (_backward_prefetch's
        default target, chosen the same way), before this group's reduce-scatter: on a shared comm stream it then
        runs ahead of the network phase; on a separate reduce-scatter stream its event (returned) gates it. FSDP's
        own prefetch then finds the target unsharding and returns. Every rank takes the same decision."""
        pg = g.pg
        idx = pg.__dict__.pop("_tide_bw_index", None)  # this group's forward index, saved before pre_backward popped it
        order = pg.comm_ctx.post_forward_order
        # a grouped fully_shard (backward may cover part of its modules) or a group that keeps its parameters (its
        # slot is not free yet) is left to FSDP's own prefetch; the same test on every rank
        if idx is None or not 0 < idx <= len(order) or len(pg.modules) > 1 or not pg.reshard_after_backward:
            return None
        nxt = order[idx - 1]  # the group whose backward follows (FSDP2 prefetched it at this group's pre_backward)
        if getattr(nxt, "_num_param_groups", 1) != 1 or not nxt._post_forward_indices or len(nxt.modules) > 1:
            return None
        tidx = nxt._post_forward_indices[-1]  # nxt pops this at its pre_backward and prefetches order[tidx - 1]
        if tidx < 1:
            return None
        target = order[tidx - 1]
        if target is pg or target.is_unsharded or target._all_gather_result is not None or len(target.modules) > 1 \
                or target.unshard_async_op:
            return None
        tg = self.groups.get(id(target.fsdp_params))
        if tg is not None and tg.slot is not None and self.all_gather and not tg.stock_ag:
            occ = tg.slot.occupant
            if occ is not None and occ is not tg and not (occ.pg._all_gather_result is None and not occ.pg.is_unsharded):
                raise RuntimeError(f"tide.fsdp: early prefetch of {tg.fqn}: its ring slot is still held by {occ.fqn}")
        _pg.FSDPParamGroup._prefetch_unshard(target, "backward")
        self.early_prefetches += 1
        res = target._all_gather_result
        if res is None or res.all_gather_event is None or reduce_scatter_stream == pg.comm_ctx.all_gather_stream:
            return None  # one comm stream: stream order already puts the all-gather first
        return res.all_gather_event

    def _end_of_backward(self) -> None:
        """Queued behind FSDP's root post-backward callback (which waited on every post-reduce event)."""
        self._backward_cb = False
        self._out_hold.clear()

    def _log_pipeline_stats(self) -> None:
        done = [r for r in self._rs_rec if all(e.query() for e in r[:3]) and (r[3] is None or r[3].query())]
        ids = {id(r) for r in done}
        self._rs_rec = [r for r in self._rs_rec if id(r) not in ids]
        ag_done = [r for r in self._ag_rec if r[1].query()]
        self._ag_rec = [r for r in self._ag_rec if not r[1].query()]
        if self.comm.rank != 0 or not done:
            return

        def med_p90(v):
            v = sorted(v)
            return (statistics.median(v), v[int(0.9 * (len(v) - 1))]) if v else (0.0, 0.0)

        read = [r[0].elapsed_time(r[1]) for r in done]
        full = [r[0].elapsed_time(r[2]) for r in done]
        pbw = [r[3].elapsed_time(r[0]) for r in done if r[3] is not None]
        bw = [b.elapsed_time(a) for b, a, backward in ag_done if backward]
        fw = [b.elapsed_time(a) for b, a, backward in ag_done if not backward]
        # read: to read_event (the last level-1 sum; with a partial outbox bank, as in throughput mode, later sums
        # follow the NIC rounds)
        log.info("[TIDE-FSDP] pipelined rs n=%d: read %.1f/%.1f ms, full %.1f/%.1f ms (med/p90 from gradients ready); "
                 "post_backward wait %.1f/%.1f ms (sum %.0f); all-gather wait backward sum %.0f ms over %d, forward sum "
                 "%.0f ms over %d; early prefetches %d", len(done), *med_p90(read), *med_p90(full), *med_p90(pbw),
                 sum(pbw), sum(bw), len(bw), sum(fw), len(fw), self.early_prefetches)

    def early_prefetch_problems(self, model: torch.nn.Module) -> list[str]:
        """Why the early backward all-gather of the pipelined reduce-scatter does not apply ([] = it does)."""
        why = []
        if not self.all_gather:
            why.append("the all-gather is not on TIDE")
        if any(getattr(st, "_states_to_backward_prefetch", None)
               for st in (_get_module_fsdp_state(m) for m in model.modules()) if st is not None):
            why.append("explicit backward prefetch (set_modules_to_backward_prefetch)")
        # (grouped fully_shard, reshard_after_backward=False, async unshard: those calls skip it, _early_prefetch)
        why += [f"{g.fqn}: several parameter groups per module" for g in self.groups.values()
                if getattr(g.pg, "_num_param_groups", 1) != 1]
        return why

    @staticmethod
    def _packed_grads(g: _Group, fsdp_params, world: int) -> torch.Tensor | None:
        """The fp32 buffer every parameter's sharded grad is an in-order view of (the layout this
        adapter and stock FSDP2 produce: padded shard after padded shard), or None."""
        storage, off = None, 0
        for (n, d0), fp in zip(g.rs_shapes, fsdp_params):
            grad = fp.sharded_param.grad
            lt = getattr(grad, "_local_tensor", None) if grad is not None else None
            if lt is None or lt.dtype != torch.float32 or lt.storage_offset() != off or \
                    tuple(lt.shape) != tuple(fp.sharded_size) or lt.stride() != tuple(fp.contiguous_sharded_stride):
                return None
            st = lt.untyped_storage()
            if storage is None:
                storage = st
            elif st.data_ptr() != storage.data_ptr():
                return None
            off += ((d0 + world - 1) // world) * (n // d0)
        if storage is None or off != g.rs_handle.out_numel or storage.nbytes() < off * 4:
            return None
        return torch.empty(0, dtype=torch.float32, device=fsdp_params[0].sharded_param.device).set_(storage, 0, (off,))

    def on_free(self, fsdp_param) -> None:
        """Called for every parameter FSDP reshards; the slot is released once the whole group is."""
        g: _Group = fsdp_param._tide_group
        if g.slot is None or g.slot.occupant is not g:
            return
        g.freed += 1
        if g.freed == len(g.pg.fsdp_params):
            g.freed = 0
            self._release(g.slot)

    def release(self) -> None:
        """Undo the per-parameter bindings so the stock path rebuilds its own buffers, and give the
        arena back. Every group must be sharded with no all-gather in flight: the unsharded parameters
        are views of the arena. (After a failed all-gather this cannot hold; the process is done for.)"""
        self._out_hold.clear()
        self._rs_rec.clear()
        self._ag_rec.clear()
        for g in self.groups.values():
            if g.pg.is_unsharded or g.pg._all_gather_result is not None:
                raise RuntimeError("tide.fsdp: uninstall() while a TIDE-managed group is unsharded "
                                   "(after a forward without backward, call model.reshard() first)")
        torch.cuda.synchronize(self.comm.device)
        for g in self.groups.values():
            if g.adopted:
                self._evict(g)
            for fp in g.pg.fsdp_params:
                fp.__dict__.pop("_tide_group", None)
                fp.__dict__.pop("_tide_group_params", None)
                if g.bound:
                    fp.all_gather_outputs = []
                    fp.__dict__.pop("_unsharded_param", None)  # FSDP builds it once; force a rebuild
            g.handle = g.rs_handle = g.slot = None
            g.landing, g.pieces, g.region_view, g.bound = [], [], None, False
        for lane in self.lanes.values():
            if lane is None:
                continue
            for slot in lane.ring + lane.dedicated:
                slot.occupant = None
            if lane.rs_arena is not None and lane.rs_arena is not lane.arena:
                lane.comm_rs.free_arena(lane.rs_arena)
            lane.rs_arena = None
            if lane.arena is not None:
                lane.comm.free_arena(lane.arena)  # memory goes with the last view, all of which were just dropped
                lane.arena = None
            if lane.owned:
                if lane.comm_rs is not lane.comm:
                    lane.comm_rs.close()
                lane.comm.close()
        # an arena in torch's MemPool went back to that pool, which hands its segment to the driver only
        # when the allocator releases cached blocks
        if any(getattr(c, "arena_in_torch", False) for lane in self.lanes.values() if lane is not None
               for c in (lane.comm, lane.comm_rs)):
            torch.cuda.empty_cache()


@dataclass
class _Stock:
    """The FSDP entry points in place when install() ran (stock or another patch's), restored by uninstall()."""
    foreach_all_gather: object
    foreach_all_gather_copy_out: object
    foreach_reduce: object
    free_unsharded_param: object
    alloc_all_gather_outputs: object
    post_forward: object
    backward_prefetch: object = None
    post_backward: object = None


_adapter: TideFSDP | None = None
_stock: _Stock | None = None


def _foreach_reduce(*args, **kwargs):
    if _adapter is None:
        return _stock.foreach_reduce(*args, **kwargs)
    return _adapter.foreach_reduce(*args, **kwargs)


def _foreach_all_gather(*args, **kwargs):
    if _adapter is None:
        return _stock.foreach_all_gather(*args, **kwargs)
    return _adapter.foreach_all_gather(*args, **kwargs)


def _foreach_all_gather_copy_out(*args, **kwargs):
    if _adapter is None:
        return _stock.foreach_all_gather_copy_out(*args, **kwargs)
    return _adapter.foreach_all_gather_copy_out(*args, **kwargs)


def _free_unsharded_param(self):
    if getattr(self, "_tide_group", None) is None:
        return _stock.free_unsharded_param(self)
    for t in self._unsharded_inner_tensors:
        free_storage(t)
    _adapter.on_free(self)  # landing memory belongs to the arena; only the slot changes hands


def _alloc_all_gather_outputs(self):
    if getattr(self, "_tide_group", None) is None:
        return _stock.alloc_all_gather_outputs(self)


def _backward_prefetch_cursor(self):
    # the forward index this pre_backward pops, for the early prefetch at this group's pipelined reduce-scatter
    # (the exact invocation; the module's position in post_forward_order is not unique under accumulation)
    if self._training_state == _TrainingState.PRE_BACKWARD and self._post_forward_indices:
        self._tide_bw_index = self._post_forward_indices[-1]
    return _stock.backward_prefetch(self)


def _post_backward_mark(self, *args, **kwargs):
    # TIDE_FSDP_ASYNC_STATS_EVERY: where the compute stream enters post_backward (before FSDP's wait on the reads)
    if _adapter is not None and _adapter._stats_on:
        ev = torch.cuda.Event(enable_timing=True)
        ev.record()
        self._tide_pbe = ev
    return _stock.post_backward(self, *args, **kwargs)


def _post_forward(self, module, input, output):
    output = _stock.post_forward(self, module, input, output)
    # A forward without autograd (validation) gets no backward: the ring group that stays unsharded after
    # forward would keep its ring slot into the next forward, where an earlier group bound to it collides.
    if _adapter is not None and not torch.is_grad_enabled() and not _pg.is_bw() and self.is_unsharded:
        g = getattr(self.fsdp_params[0], "_tide_group", None) if self.fsdp_params else None
        if g is not None and g.ring and g.slot is not None and g.slot.occupant is g:
            self.reshard()
    return output


# Per-object FSDP2 attributes the adapter reads or rewrites.
_PARAM_ATTRS = ("_sharded_param_data", "sharded_param", "padded_sharded_param_size", "fsdp_placement", "_orig_size",
                "offload_to_cpu", "all_gather_outputs", "to_sharded_dtensor")  # orig_dtype: set by _init_mp_dtypes
_GROUP_ATTRS = ("fsdp_params", "_all_gather_process_group", "_use_post_forward_mesh", "_reshard_after_forward",
                "_all_gather_result", "is_unsharded", "_init_mp_dtypes")
_AG_RESULT_FIELDS = ("all_gather_output", "all_gather_event", "all_gather_work", "param_all_gather_input_dtypes",
                     "param_all_gather_input_numels", "all_gather_input_split_sizes")


def compat_problems(model: torch.nn.Module | None = None) -> list[str]:
    """Why this torch's FSDP2 does not match what the adapter patches ([] = compatible). With `model`
    (after fully_shard), also checks the attributes of its parameter groups and parameters."""
    if _IMPORT_PROBLEM:
        return [_IMPORT_PROBLEM]
    out = []

    def params(f):
        return [p for n, p in inspect.signature(f).parameters.items() if n != "self"]

    # FSDP calls the patched functions positionally: our wrappers must take as many parameters, with
    # defaults in the same places (names may differ)
    for name, ours in (("foreach_all_gather", TideFSDP.foreach_all_gather),
                       ("foreach_all_gather_copy_out", TideFSDP.foreach_all_gather_copy_out),
                       ("foreach_reduce", TideFSDP.foreach_reduce)):
        stock = getattr(_pg, name, None)
        if stock is None:
            out.append(f"_fsdp_param_group.{name} is gone")
        elif any(p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD) for p in inspect.signature(stock).parameters.values()):
            # another patch's (*args, **kwargs) wrapper is in place: its parameter list says nothing
            log.warning("[TIDE-FSDP] _fsdp_param_group.%s is a (*args, **kwargs) wrapper; its signature is not checked", name)
        elif [p.default is p.empty for p in params(stock)] != [p.default is p.empty for p in params(ours)]:
            out.append(f"_fsdp_param_group.{name}{inspect.signature(stock)} differs from the adapter's "
                       f"({', '.join(p.name for p in params(ours))})")
    try:  # the adapter returns foreach_reduce's 7-tuple; the caller unpacks it positionally
        ret = typing.get_args(typing.get_type_hints(_coll.foreach_reduce).get("return"))
        if ret and len(ret) != 7:
            out.append(f"foreach_reduce returns a {len(ret)}-tuple, the adapter a 7-tuple")
    except Exception:  # noqa: BLE001 - annotations not resolvable here: nothing to compare
        pass
    for name in ("free_unsharded_param", "alloc_all_gather_outputs"):
        if not callable(getattr(_fp.FSDPParam, name, None)):
            out.append(f"FSDPParam.{name} is gone")
    # all-gather extensions: the inputs come from the property FSDP itself uses, the post hook from
    # init_unsharded_param over `all_gather_outputs` (one per input, flattened)
    for name in ("all_gather_inputs", "init_unsharded_param", "_unflatten_all_gather_outputs"):
        if not hasattr(_fp.FSDPParam, name):
            out.append(f"FSDPParam.{name} is gone")
    for name in ("post_forward", "reshard"):
        if not callable(getattr(_pg.FSDPParamGroup, name, None)):
            out.append(f"FSDPParamGroup.{name} is gone")
    if not callable(getattr(_pg, "is_bw", None)):
        out.append("_fsdp_param_group.is_bw is gone")
    for name in ("post_backward", "_backward_prefetch", "_prefetch_unshard"):  # the early backward all-gather
        if not callable(getattr(_pg.FSDPParamGroup, name, None)):
            out.append(f"FSDPParamGroup.{name} is gone")
    missing = [f for f in _AG_RESULT_FIELDS if f not in getattr(_coll.AllGatherResult, "_fields", ())]
    if missing:
        out.append(f"AllGatherResult lacks {missing}")
    if len(inspect.signature(_get_gradient_divide_factors).parameters) != 6:
        out.append(f"_get_gradient_divide_factors{inspect.signature(_get_gradient_divide_factors)} is not the 6-argument form")
    if model is not None:
        groups = [pg for m in model.modules() if (st := _get_module_fsdp_state(m)) is not None
                  for pg in getattr(st, "_fsdp_param_groups", ())]
        if not groups:
            out.append("no FSDP parameter groups found (call after fully_shard; FSDPState._fsdp_param_groups gone?)")
        for pg in groups[:1]:
            out += [f"FSDPParamGroup.{a} is gone" for a in _GROUP_ATTRS if not hasattr(pg, a)]
            for fp in list(getattr(pg, "fsdp_params", ()))[:1]:
                out += [f"FSDPParam.{a} is gone" for a in _PARAM_ATTRS if not hasattr(fp, a)]
                if hasattr(fp, "sharded_param") and not hasattr(fp.sharded_param, "_local_tensor"):
                    out.append("the sharded parameter has no _local_tensor (not a DTensor?)")
    return out


def env_problems(comm: TideComm) -> list[str]:
    """What this rank's hardware / layout lacks for the adapter ([] = fine), known before anything is allocated
    (the communicator itself already refused non-node-major ranks and multi-node without host RMA)."""
    out = []
    if comm.n_nodes > 1 and not comm.props.get("multimem"):
        out.append(f"{comm.n_nodes} nodes but no NVLS multimem (multi-node landings are multicast): "
                   "NCCL_NVLS_ENABLE=1, fabric manager / IMEX up?")
    if comm.local_ranks > 8:
        out.append(f"{comm.local_ranks} ranks per NVLink domain; the kernels hold at most 8")
    return out


def install(model: torch.nn.Module, comm: TideComm, slots: int = 2, reduce_scatter: bool = True,
            all_gather: bool = True, comm_rs: TideComm | None = None, allow_fallback: bool = False,
            subgroups: bool = False) -> TideFSDP | None:
    """Collective. Call after fully_shard() on every rank, before the first forward, after any other
    FSDP patch (they keep serving the groups TIDE declines). With reduce_scatter, every eligible
    group's backward reduce-scatter also runs on TIDE (gradients read in place, no chunk_cat); the
    reduction order differs from NCCL's, so losses are not bitwise equal to stock -- the all-gather
    alone is. all_gather=False leaves the all-gather on the stock path (reduce-scatter only).
    subgroups: groups over a regular subset of the ranks (expert groups) get their own communicators
    (created here, collective over each subset; closed by uninstall).
    Multi-node reduce-scatters on lanes with several ranks per node are pipelined: FSDP frees a group's
    gradients at the next post-backward once they are read, while the network phase runs on (docs/usage.md).
    TIDE_FSDP_COPYIN=1 (for runs whose communication hides behind compute): the gradients go into the slot their
    group just left and are freed at once; the reduce-scatter reads the slot, the slot's next all-gather waits.

    Checks FSDP2's internals (compat_problems) and the hardware (env_problems, agreed over the group: one
    rank's problem is every rank's) first: a problem raises, listing them, before anything is allocated or
    patched -- or, with allow_fallback, logs them and returns None on every rank with FSDP untouched (the
    run is then stock, not TIDE). A torch outside the tested versions only warns when every check passes."""
    global _adapter, _stock
    if _adapter is not None:
        raise RuntimeError("tide.fsdp.install() called twice")
    t_start = time.perf_counter()
    problems = compat_problems(model)
    if comm is not None:  # the FSDP2 check alone needs no communicator
        env = env_problems(comm) + (env_problems(comm_rs) if comm_rs is not None else [])
        anyone = torch.tensor([int(bool(env))], device=comm.device)
        dist.all_reduce(anyone, op=dist.ReduceOp.MAX, group=comm.group)
        if anyone.item() and not env:
            env = ["another rank of the group reported a hardware / layout problem (see its log)"]
        problems += env
    if problems:
        msg = f"FSDP2 of torch {torch.__version__} / this hardware does not suit the adapter:\n  " + "\n  ".join(problems)
        if not allow_fallback:
            raise RuntimeError("tide.fsdp: " + msg)
        log.warning("[TIDE-FSDP] %s\n-> staying on the stock FSDP2 path", msg)
        return None
    if not torch.__version__.startswith(_TESTED_TORCH):
        log.warning("[TIDE-FSDP] torch %s is outside the tested versions %s (FSDP2 internals checked and match)",
                    torch.__version__, _TESTED_TORCH)
    if not (all_gather or reduce_scatter):
        raise ValueError("tide.fsdp.install(): nothing to install")
    adapter = TideFSDP(comm, slots, comm_rs, subgroups)
    adapter.all_gather = bool(all_gather)
    adapter.reduce_scatter = bool(reduce_scatter)
    if adapter.copyin and not (adapter.all_gather and adapter.reduce_scatter):
        adapter.copyin = False
        if comm.rank == 0:
            log.warning("[TIDE-FSDP] TIDE_FSDP_COPYIN=1 needs both the all-gather and the reduce-scatter on TIDE: off")
    # both decide which all-gathers go early: differing across ranks they would split the call order
    _all_same(comm.group, (adapter.copyin, adapter.early_prefetch), "TIDE_FSDP_COPYIN / TIDE_FSDP_EARLY_PREFETCH")
    # pipelined calls only; with copy-in the next all-gather goes after the reduce-scatter that frees its slot
    adapter.early_prefetch = adapter.early_prefetch and adapter.reduce_scatter and comm.n_nodes > 1 and not adapter.copyin
    t_checks = time.perf_counter()
    adapter.collect(model)
    t_collect = time.perf_counter()
    adapter.allocate()
    t_alloc = time.perf_counter()
    if adapter.early_prefetch:  # agreed over the group: an early all-gather on some ranks only would split the call order
        why = adapter.early_prefetch_problems(model)
        bad = torch.tensor([int(bool(why))], device=comm.device)
        dist.all_reduce(bad, op=dist.ReduceOp.MAX, group=comm.group)
        if bad.item():
            adapter.early_prefetch = False
            log.warning("[TIDE-FSDP] pipelined reduce-scatter without the early backward all-gather: %s",
                        "; ".join(why) or "another rank reported a reason (see its log)")
    _adapter = adapter
    _stock = _Stock(_pg.foreach_all_gather, _pg.foreach_all_gather_copy_out, _pg.foreach_reduce,
                    _fp.FSDPParam.free_unsharded_param, _fp.FSDPParam.alloc_all_gather_outputs,
                    _pg.FSDPParamGroup.post_forward, _pg.FSDPParamGroup._backward_prefetch,
                    _pg.FSDPParamGroup.post_backward)
    _pg.foreach_all_gather = _foreach_all_gather
    _pg.foreach_all_gather_copy_out = _foreach_all_gather_copy_out
    _pg.foreach_reduce = _foreach_reduce
    _fp.FSDPParam.free_unsharded_param = _free_unsharded_param
    _fp.FSDPParam.alloc_all_gather_outputs = _alloc_all_gather_outputs
    _pg.FSDPParamGroup.post_forward = _post_forward
    if adapter.early_prefetch:
        _pg.FSDPParamGroup._backward_prefetch = _backward_prefetch_cursor
    if adapter._stats_on:
        _pg.FSDPParamGroup.post_backward = _post_backward_mark
    if adapter.reduce_scatter and comm.n_nodes > 1 and comm.rank == 0:
        log.info("[TIDE-FSDP] reduce-scatter: multi-node calls pipelined, inputs freed once read, early backward "
                 "all-gather %s", "on" if adapter.early_prefetch else "off")
        conns = os.environ.get("CUDA_DEVICE_MAX_CONNECTIONS", "")
        if not conns.isdigit() or int(conns) < 32:
            log.warning("[TIDE-FSDP] CUDA_DEVICE_MAX_CONNECTIONS=%s: set 32 before CUDA starts; on the 8 default hardware "
                        "queues the next call's reads wait behind the previous network phase", conns or "unset")
    if adapter.copyin and comm.rank == 0:
        log.info("[TIDE-FSDP] copy-in: a group's gradients go into its slot once it reshards and are freed there; the "
                 "slot's next all-gather waits for the reduce-scatter's reads")
    if comm.rank == 0:  # startup accounting (the communicators' own time is logged where they are created)
        t_end = time.perf_counter()
        a = adapter.alloc_s
        log.info("[TIDE-FSDP] install %.1f s: checks %.1f, collect %.1f, allocate %.1f (arena + windows %.1f, reduce-scatter "
                 "tuning %.1f + registration %.1f, slot binding %.1f), patch %.1f", t_end - t_start, t_checks - t_start,
                 t_collect - t_checks, t_alloc - t_collect, a.get("arena", 0.0), a.get("rs_tune", 0.0), a.get("rs_register", 0.0),
                 a.get("bind", 0.0), t_end - t_alloc)
    return adapter


def uninstall() -> None:
    """Collective in effect (every rank must uninstall before its next forward). Releases the arena;
    the communicator stays usable for a later install() or comm.close()."""
    global _adapter, _stock
    if _adapter is None:
        return
    _adapter.release()
    _pg.foreach_all_gather = _stock.foreach_all_gather
    _pg.foreach_all_gather_copy_out = _stock.foreach_all_gather_copy_out
    _pg.foreach_reduce = _stock.foreach_reduce
    _fp.FSDPParam.free_unsharded_param = _stock.free_unsharded_param
    _fp.FSDPParam.alloc_all_gather_outputs = _stock.alloc_all_gather_outputs
    _pg.FSDPParamGroup.post_forward = _stock.post_forward
    if _stock.backward_prefetch is not None:
        _pg.FSDPParamGroup._backward_prefetch = _stock.backward_prefetch
    if _stock.post_backward is not None:
        _pg.FSDPParamGroup.post_backward = _stock.post_backward
    _adapter = _stock = None
