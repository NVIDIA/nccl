# SPDX-FileCopyrightText: Copyright (c) 2026 Kuaishou Technology. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FSDP2 x TIDE: a toy stack of fully_shard-wrapped blocks trained for a few steps with stock FSDP2,
then the same model/data again with tide.fsdp installed (optionally uninstalled part-way, continuing
on the stock path or on a second install over a fresh arena). Losses and final parameters must
match bitwise (the all-gather is exact; compute is untouched).

--rs also routes every group's reduce-scatter through TIDE (gradients read in place); its fp32 sum
order differs from NCCL's, so that run is checked against stock at 1e-4 and a second TIDE run must
match it bitwise (determinism). --accum N backs N microbatches per step; from the second one on the
reduce-scatter must accumulate straight into the existing sharded grads.

--ep N gives every block an expert module laid out the way torchtitan lays out MoE experts: weights
[E, ., .] sharded Shard(0) over an "ep" mesh dimension, then fully_shard over "efsdp" (the ranks
r % N == rank % N) -- FSDP places them _StridedShard; the adapter serves them on a second lane.

--wide N (with --fp32-group) puts N more small params into the fp32 group, beyond the 256 segments a
reduce-scatter plan holds: that group must keep the stock reduce-scatter and the TIDE all-gather.

--ext wraps the experts' weights (--ep N) in a toy FSDP2 all-gather extension (ExtWeight): each shard is
gathered as its bf16 payload plus a uint8 tag plane stamped with the sending rank, the post hook checks the
tags (gather order / completeness) and builds the parameter over the payload plus one derived inner tensor
-- the shape of an MXFP8 weight's payload + scales gather. TIDE serves such groups as multi-segment groups;
with TIDE_FSDP_EXTENSIONS=0 they stay on the stock path (the run must still match).

With --rs on several nodes the reduce-scatters are pipelined: FSDP frees the inputs once read, the next backward
all-gather is issued before the network phase. The second TIDE run (the determinism reference) then runs one-phase
calls (TIDE_RS_PIPELINE=1) on fresh communicators, so a pipelined sum that differs fails bitwise. --poison NaN-fills
the gradients when FSDP frees them, so a read after the reduce-scatter's read event corrupts the result. Negative:
TIDE_FAULT=earlyread,slowpush (the event recorded before any read) must be caught (exit 0 iff it is).

--copyin (with --rs): the second TIDE run copies each group's gradients into the slot it has just left
(TIDE_FSDP_COPYIN=1, same communicators and plan) and must match the first, in-place run
bitwise; --poison NaN-fills the gradients right after the copy. Negative: TIDE_FAULT=noreadwait,slowpush (the slot's next all-gather does not wait
for the reads) must be caught -- on separate streams and, on several nodes, separate communicators (--split-comm): one comm
stream or one NCCL communicator queues that all-gather behind the reads anyway.

torchrun --standalone --nproc_per_node 8 test/test_fsdp_adapter.py --layers 6 --hidden 2048 --steps 6 [--rs]
"""
from __future__ import annotations

import argparse
import copy
import json
import os

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict, set_model_state_dict
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.distributed.tensor import Shard, distribute_tensor

import tide
import tide.fsdp


class Experts(nn.Module):
    def __init__(self, h: int, f: int, n: int):
        super().__init__()
        self.w1 = nn.Parameter(torch.randn(n, h, f) * h ** -0.5)
        self.w2 = nn.Parameter(torch.randn(n, f, h) * f ** -0.5)

    def forward(self, x):  # every local expert on every token (routing is not the point here)
        w1, w2 = (w.to_local() if hasattr(w, "to_local") else w for w in (self.w1, self.w2))
        y = torch.einsum("bsh,ehf->ebsf", x, w1).relu()
        return torch.einsum("ebsf,efh->bsh", y, w2) / w1.shape[0]


_EXT_REWRAP = {  # the view / allocation ops FSDP2 runs on a parameter while building and resetting its shard
    torch.ops.aten.detach.default, torch.ops.aten.view.default, torch.ops.aten.as_strided.default,
    torch.ops.aten.slice.Tensor, torch.ops.aten.split.Tensor, torch.ops.aten.split_with_sizes.default,
    torch.ops.aten.new_zeros.default, torch.ops.aten.new_empty.default, torch.ops.aten.empty_like.default,
    torch.ops.aten.clone.default, torch.ops.aten._to_copy.default, torch.ops.aten.t.default,
    torch.ops.aten.alias.default, torch.ops.aten.narrow.default, torch.ops.aten.contiguous.default,
}
EXT_TAG_COLS = 128  # one 128-byte line of uint8 per row


class ExtWeight(torch.Tensor):
    """Toy FSDP2 all-gather extension (fsdp_pre_all_gather / fsdp_post_all_gather): the sharded parameter is
    gathered as two inputs -- its payload in the all-gather dtype and a uint8 tag plane [rows, 128] holding
    the sending rank + 1 -- like an MXFP8 weight gathers E4M3 payload + E8M0 scales. The post hook verifies
    the tags of every row (a wrong gather order, a missing segment or stale bytes fail loudly), returns the
    payload as the parameter and keeps one derived inner tensor FSDP frees at reshard."""

    @staticmethod
    def __new__(cls, t: torch.Tensor):
        return torch.Tensor._make_wrapper_subclass(
            cls, t.shape, strides=t.stride(), storage_offset=t.storage_offset(), dtype=t.dtype, device=t.device,
            requires_grad=t.requires_grad,
        )

    def __init__(self, t: torch.Tensor):
        self._t = t

    def __repr__(self):  # noqa: D105
        return f"ExtWeight({self._t!r})"

    def __tensor_flatten__(self):
        return ["_t"], None

    @staticmethod
    def __tensor_unflatten__(inner, ctx, outer_size, outer_stride):
        return ExtWeight(inner["_t"])

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        from torch.utils._pytree import tree_map

        kwargs = kwargs or {}
        unwrap = lambda x: x._t if isinstance(x, ExtWeight) else x  # noqa: E731
        out = func(*tree_map(unwrap, args), **tree_map(unwrap, kwargs))
        if func._schema.name.endswith("_") and isinstance(args[0], ExtWeight):  # in-place: the wrapper itself
            return args[0]
        if func in _EXT_REWRAP:
            return tree_map(lambda x: ExtWeight(x) if isinstance(x, torch.Tensor) else x, out)
        return out

    def fsdp_pre_all_gather(self, mesh, outer_size, outer_stride, module, mp_policy):
        payload = self._t.to(mp_policy.param_dtype or self._t.dtype)
        tag = torch.full((payload.shape[0], EXT_TAG_COLS), mesh.get_local_rank() + 1, dtype=torch.uint8,
                         device=payload.device)
        return (payload, tag), (int(outer_size[0]), mesh.size())

    def fsdp_post_all_gather(self, all_gather_outputs, metadata, param_dtype, *, out=None):
        rows, world = metadata
        payload, tags = all_gather_outputs  # [world * rows_per_rank, k], [world * rows_per_rank, 128]
        per_rank = tags.shape[0] // world
        want = torch.arange(1, world + 1, dtype=torch.uint8, device=tags.device).repeat_interleave(per_rank)
        if not torch.equal(tags[:, 0], want) or not torch.equal(tags.min(dim=1).values, tags.max(dim=1).values):
            raise RuntimeError("ExtWeight: gathered tag plane is wrong (order, completeness or stale bytes)")
        if out is not None:  # refill the inner tensor FSDP re-allocated (same object) for this unshard
            self._ext_derived.copy_(tags[:, :1].float())
            return None
        self._ext_derived = derived = tags[:, :1].float()  # the sharded wrapper outlives every unshard
        return payload[:rows], (derived,)


class Block(nn.Module):
    def __init__(self, h: int, odd: bool, experts: int = 0, frozen: bool = False):
        super().__init__()
        self.experts = Experts(h, h // 2 + (8 if odd else 0), experts) if experts else None
        # uneven parameter sizes on purpose: several segments per group, some tiny
        self.fc1 = nn.Linear(h, h * 2 + (24 if odd else 0), bias=True)
        self.fc2 = nn.Linear(h * 2 + (24 if odd else 0), h, bias=False)
        self.norm = nn.LayerNorm(h)
        self.fz = nn.Parameter(torch.zeros(h), requires_grad=False) if frozen else None  # frozen: no gradient

    def forward(self, x):
        if self.fz is not None:
            x = x + self.fz
        x = x + self.fc2(torch.relu(self.fc1(x)))
        if self.experts is not None:
            x = x + self.experts(x)
        return self.norm(x).to(x.dtype)  # fp32 group (--fp32-group) returns fp32


class Wide(nn.Module):  # a production MoE model's fp32 group holds every router and norm: hundreds of small params
    def __init__(self, n: int, h: int):
        super().__init__()
        self.p = nn.ParameterList(nn.Parameter(torch.zeros(h)) for _ in range(n))

    def forward(self, x):
        return x + torch.stack(list(self.p)).sum(0).to(x.dtype)


class Toy(nn.Module):
    def __init__(self, layers: int, h: int, vocab: int, experts: int = 0, frozen: bool = False, reverse: bool = False,
                 wide: int = 0):
        super().__init__()
        self.reverse = reverse  # run the blocks last-to-first: module order is not forward order
        self.emb = nn.Embedding(vocab, h)
        self.wide = Wide(wide, h) if wide else None
        self.blocks = nn.ModuleList(Block(h, i % 2 == 1, experts, frozen) for i in range(layers))
        self.head = nn.Linear(h, vocab, bias=False)
        self.frozen = nn.Parameter(torch.zeros(h), requires_grad=False) if frozen else None
        if frozen:  # frozen last-block params, as in a production model: one lands in the fp32 group (--fp32-group)
            self.blocks[-1].norm.bias.requires_grad_(False)

    def forward(self, idx):
        x = self.emb(idx)
        if self.frozen is not None:
            x = x + self.frozen
        if self.wide is not None:
            x = self.wide(x)
        for b in (reversed(self.blocks) if self.reverse else self.blocks):
            x = b(x)
        return self.head(x)


def _inner(b):
    return getattr(b, "_checkpoint_wrapped_module", b)


def build(args, dev, seed):
    torch.manual_seed(seed)
    world = dist.get_world_size()
    rep = args.replicate
    # E = ep x efsdp experts: every ep rank holds efsdp of them, one row per efsdp rank
    model = Toy(args.layers, args.hidden, args.vocab, world if args.ep > 1 else 0, args.frozen, args.reverse_forward,
                args.wide).to(dev)
    if args.ac:
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper
        for i, b in enumerate(model.blocks):
            model.blocks[i] = checkpoint_wrapper(b)
    mp = MixedPrecisionPolicy(param_dtype=None if args.no_mp else torch.bfloat16, reduce_dtype=torch.float32)
    # HSDP (--replicate R): torchtitan's dim order, replicate outermost
    dmesh = init_device_mesh("cuda", (rep, world // rep), mesh_dim_names=("rep", "shard")) if rep > 1 else None
    if args.ep > 1:
        # rank = efsdp_index * ep + ep_index: the efsdp group of a rank is r % ep == rank % ep
        if rep > 1:
            mesh = init_device_mesh("cuda", (rep, world // rep // args.ep, args.ep), mesh_dim_names=("rep", "efsdp", "ep"))
            emesh = mesh["rep", "efsdp"]
        else:
            mesh = init_device_mesh("cuda", (world // args.ep, args.ep), mesh_dim_names=("efsdp", "ep"))
            emesh = mesh["efsdp"]
        for b in model.blocks:
            ex = _inner(b).experts
            for name in ("w1", "w2"):
                w = getattr(ex, name)
                local = distribute_tensor(w.data, mesh["ep"], [Shard(0)])
                if args.ext:  # the extension lives on the local tensor FSDP will shard
                    from torch.distributed.tensor import DTensor
                    local = DTensor.from_local(ExtWeight(local.to_local()), local.device_mesh, local.placements,
                                               shape=local.shape, stride=local.stride())
                setattr(ex, name, nn.Parameter(local))
            fully_shard(ex, mesh=emesh, mp_policy=mp, reshard_after_forward=not (args.last_no_reshard and b is model.blocks[-1]))
            if args.divide_factor:
                ex.set_gradient_divide_factor(args.divide_factor)
    if args.fp32_group:  # every block's precision-critical small weights in one fp32 group that stays unsharded
        norms = [_inner(b).norm for b in model.blocks] + ([model.wide] if model.wide is not None else [])
        fully_shard(norms, mesh=dmesh, reshard_after_forward=False,
                    mp_policy=MixedPrecisionPolicy(param_dtype=torch.float32, reduce_dtype=torch.float32))
        if args.ac:  # a recompute guard: the nested fp32 group is re-entered by the recompute before its pre-backward
            from torch.distributed.fsdp._fully_shard._fsdp_common import TrainingState
            from torch.distributed.fsdp._fully_shard._fsdp_state import _get_module_fsdp_state
            state = _get_module_fsdp_state(norms[0])
            model.register_forward_pre_hook(lambda *_: setattr(state, "_training_state", TrainingState.IDLE))
            model.register_forward_hook(lambda *_: setattr(state, "_training_state", TrainingState.PRE_BACKWARD))
    for b in model.blocks:
        fully_shard(b, mesh=dmesh, mp_policy=mp, reshard_after_forward=not (args.last_no_reshard and b is model.blocks[-1]))
    # --root-auto leaves the root's reshard_after_forward to FSDP, which turns it off at lazy init
    if args.root_auto:
        fully_shard(model, mesh=dmesh, mp_policy=mp)
    else:
        fully_shard(model, mesh=dmesh, mp_policy=mp, reshard_after_forward=False)
    if args.block_prefetch:  # each block (and root) prefetches the next block with its experts
        blocks = list(model.blocks)
        targets = [[b] + ([_inner(b).experts] if _inner(b).experts is not None else []) for b in blocks]
        model.set_modules_to_forward_prefetch(targets[0])
        for src, nxt in zip(blocks[:-1], targets[1:]):
            src.set_modules_to_forward_prefetch(nxt)
        for src, prev in zip(blocks[:0:-1], targets[-2::-1]):
            src.set_modules_to_backward_prefetch(prev)
    # frozen params (--frozen): their reduce-scatter must leave .grad None, as stock does
    model.no_grad_params = ([_inner(b).fz for b in model.blocks] + [model.frozen, _inner(model.blocks[-1]).norm.bias]
                            if args.frozen else [])
    model.grad_seen_on_no_grad = False
    return model


def run(model, args, dev, gen, steps):
    opt = torch.optim.SGD(model.parameters(), lr=0.05)
    slow = dist.get_rank() == args.slow_rank and args.slow_ms > 0
    if slow:  # one straggler widens every window between a group's all-gathers and its compute
        for b in model.blocks:
            b.register_forward_pre_hook(lambda *_: torch.cuda._sleep(int(args.slow_ms * 1e-3 * 1.5e9)))
    losses = []
    for step in range(steps):
        for _ in range(args.accum):  # every microbatch reduce-scatters; grads accumulate in the sharded grads
            idx = torch.randint(0, args.vocab, (args.batch, args.seq), device=dev, generator=gen)
            tgt = torch.randint(0, args.vocab, (args.batch, args.seq), device=dev, generator=gen)
            loss = nn.functional.cross_entropy(model(idx).float().view(-1, args.vocab), tgt.view(-1))
            loss.backward()
            losses.append(loss.detach().clone())
            model.grad_seen_on_no_grad |= any(p.grad is not None for p in model.no_grad_params)
        if args.slow_opt_ms > 0:  # late shard writes: the next step's all-gathers must still read the updated shards
            torch.cuda._sleep(int(args.slow_opt_ms * 1e-3 * 1.5e9))
        opt.step()
        opt.zero_grad(set_to_none=True)
        if args.eval_batches and step == steps // 2:  # validation mid-run: forwards without backward, then training goes on
            with torch.no_grad():
                for swap in (False, True) if args.ema_swap else (False,):
                    if swap:  # EMA validation: deepcopy the live state dict, load other weights, load it back after
                        orig = copy.deepcopy(get_model_state_dict(model))
                        set_model_state_dict(model, {k: v.detach().clone().mul_(0.5) for k, v in orig.items()},
                                             options=StateDictOptions(strict=True))
                    for _ in range(args.eval_batches):
                        idx = torch.randint(0, args.vocab, (args.batch, args.seq), device=dev, generator=gen)
                        tgt = torch.randint(0, args.vocab, (args.batch, args.seq), device=dev, generator=gen)
                        losses.append(nn.functional.cross_entropy(model(idx).float().view(-1, args.vocab), tgt.view(-1)))
                    if swap:
                        set_model_state_dict(model, orig)
    torch.cuda.synchronize()
    return losses


def params_of(model):
    return [p.detach().to_local().clone() if hasattr(p, "to_local") else p.detach().clone() for p in model.parameters()]


def _tail_in_ring(model, adapter) -> bool:
    from torch.distributed.fsdp._fully_shard._fsdp_state import _get_module_fsdp_state
    last = model.blocks[-1]
    for m in [last] + ([_inner(last).experts] if _inner(last).experts is not None else []):
        g = adapter.groups.get(id(_get_module_fsdp_state(m)._fsdp_param_group.fsdp_params))
        if g is None or not g.ring or not any(g.slot is s for s in g.lane.ring):
            return False
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--hidden", type=int, default=2048)
    ap.add_argument("--vocab", type=int, default=4096)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--slots", type=int, default=2)
    ap.add_argument("--num-rma-ctx", type=int, default=0)
    ap.add_argument("--root-auto", action="store_true", help="let FSDP decide the root's reshard_after_forward")
    ap.add_argument("--slow-rank", type=int, default=-1)
    ap.add_argument("--slow-ms", type=float, default=0.0, help="sleep before every block on --slow-rank")
    ap.add_argument("--slow-opt-ms", type=float, default=0.0, help="sleep on the compute stream before every optimizer step")
    ap.add_argument("--uninstall-after", type=int, default=0, help="uninstall TIDE after this many steps and continue")
    ap.add_argument("--env-problem-rank", type=int, default=-1,
                    help="inject a hardware problem on this rank: install(allow_fallback=True) must decline on every rank")
    ap.add_argument("--reinstall", action="store_true", help="with --uninstall-after: install again (fresh arena) instead")
    ap.add_argument("--stock-between", type=int, default=0,
                    help="with --reinstall: this many stock steps between uninstall and the second install")
    ap.add_argument("--accum", type=int, default=1, help="microbatches per optimizer step")
    ap.add_argument("--rs", action="store_true", help="TIDE reduce-scatter too (tolerance + determinism check)")
    ap.add_argument("--rs-only", action="store_true", help="TIDE reduce-scatter with the stock all-gather")
    ap.add_argument("--split-comm", action="store_true", help="reduce-scatter on a second communicator")
    ap.add_argument("--no-mp", action="store_true", help="fp32 all-gather (no cast: copy-in is a copy-engine memcpy)")
    ap.add_argument("--ep", type=int, default=1, help="expert modules over efsdp groups (r %% ep); TIDE serves them on a second lane")
    ap.add_argument("--ext", action="store_true",
                    help="with --ep: the experts' weights carry a toy FSDP2 all-gather extension (payload + tag plane)")
    ap.add_argument("--divide-factor", type=float, default=0.0,
                    help="experts' gradient divide factor (torchtitan EP uses the data-parallel world): PREMUL_SUM")
    ap.add_argument("--block-prefetch", action="store_true", help="explicit forward/backward prefetch of block + experts")
    ap.add_argument("--fp32-group", action="store_true", help="all blocks' norms in one fp32 group, never resharded")
    ap.add_argument("--wide", type=int, default=0, help="with --fp32-group: that many more small params in the fp32 group")
    ap.add_argument("--last-no-reshard", action="store_true",
                    help="the last block (and its experts) stay unsharded after forward")
    ap.add_argument("--reverse-forward", action="store_true",
                    help="blocks run last-to-first (module order is not forward order): the ring-tail fallback")
    ap.add_argument("--ac", action="store_true", help="activation checkpointing around every block")
    ap.add_argument("--replicate", type=int, default=1,
                    help="HSDP replicas (TIDE reduce-scatter over the shard group, the replica all-reduce stays stock)")
    ap.add_argument("--frozen", action="store_true",
                    help="frozen params (requires_grad False) in every block, the root and the fp32 group")
    ap.add_argument("--eval-batches", type=int, default=0,
                    help="forwards under no_grad after the middle step (validation), their losses compared too")
    ap.add_argument("--poison", action="store_true", help="with --rs: NaN-fill the gradients when FSDP frees them")
    ap.add_argument("--ema-swap", action="store_true",
                    help="with --eval-batches: a second pass under swapped-in weights, as an EMA validation does")
    ap.add_argument("--merged-stream", action="store_true",
                    help="FSDP's all-gather and reduce-scatter on one comm stream, as some training scripts set them")
    ap.add_argument("--copyin", action="store_true",
                    help="with --rs: the second TIDE run copies the gradients into the slots (bitwise against the first)")
    args = ap.parse_args()
    if args.merged_stream:
        from torch.distributed.fsdp._fully_shard._fsdp_param_group import FSDPCommContext
        stock_init = FSDPCommContext.lazy_init

        def merged_init(self, device):
            stock_init(self, device)
            self.reduce_scatter_stream = self.all_gather_stream

        FSDPCommContext.lazy_init = merged_init
    if args.wide and not args.fp32_group:
        ap.error("--wide adds to the fp32 group: pass --fp32-group")
    if args.ext and args.ep <= 1:
        ap.error("--ext wraps the experts' weights: pass --ep N")
    if args.rs_only:
        args.rs = True
    if args.copyin and (not args.rs or args.rs_only):
        ap.error("--copyin needs --rs with the TIDE all-gather (the slots)")
    if args.poison:
        os.environ["TIDE_FSDP_RELEASE_POISON"] = "1"
    if args.ep > 1:  # NCCL 2.31.2 PAT (one rank per node) reduce-scatter returns SUM for AVG: the stock reference would be wrong
        os.environ.setdefault("NCCL_ALGO", "^PAT")
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    dev = torch.device("cuda", int(os.environ.get("LOCAL_RANK", rank % torch.cuda.device_count())))
    torch.cuda.set_device(dev)

    # HSDP: per-rank data, so the replicas' gradients differ and a missing or mis-scaled replica all-reduce shows
    data_seed = 7 + (rank if args.replicate > 1 else 0)
    ref_model = build(args, dev, 1)
    init_params = params_of(ref_model)
    ref_losses = torch.stack(run(ref_model, args, dev, torch.Generator(device=dev).manual_seed(data_seed), args.steps))
    ref_params = params_of(ref_model)

    comm = tide.TideComm(device=dev, num_rma_ctx=args.num_rma_ctx)
    comm_rs = tide.TideComm(device=dev, num_rma_ctx=args.num_rma_ctx) if args.split_comm else None
    if args.env_problem_rank >= 0:  # one rank's hardware problem: every rank falls back, FSDP left as it was
        real = tide.fsdp.env_problems
        tide.fsdp.env_problems = lambda c: real(c) + (["injected"] if rank == args.env_problem_rank else [])
        model = build(args, dev, 1)
        adapter = tide.fsdp.install(model, comm, slots=args.slots, reduce_scatter=args.rs, comm_rs=comm_rs,
                                    allow_fallback=True, subgroups=args.ep > 1 or args.replicate > 1)
        everyone = torch.tensor([int(adapter is None)], device=dev)
        dist.all_reduce(everyone, op=dist.ReduceOp.MIN)
        losses = torch.stack(run(model, args, dev, torch.Generator(device=dev).manual_seed(data_seed), args.steps))
        ok = bool(everyone.item()) and torch.equal(losses, ref_losses)
        if rank == 0:
            print(json.dumps({"env_fallback_everywhere": bool(everyone.item()),
                              "stock_losses_bitwise": torch.equal(losses, ref_losses)}), flush=True)
            print("PASS" if ok else "FAIL", flush=True)
        dist.barrier()
        for c in (comm, comm_rs):
            if c is not None:
                c.close()
        dist.destroy_process_group()
        return 0 if ok else 1
    released = []  # device memory given back by uninstall() as a fraction of the arena
    in_torch = []  # torch's reserved memory grown by install() as a fraction of the arena

    def uninstall(adapter):
        arena_bytes = adapter.arena.numel() if adapter.arena is not None else 0
        torch.cuda.synchronize()
        free0 = torch.cuda.mem_get_info(dev)[0]
        tide.fsdp.uninstall()
        dist.barrier()  # the physical pages go only once every peer dropped its mapping of them too
        torch.cuda.synchronize()
        # small arenas (reduce-scatter only) drown in the allocator's own traffic: judge >= 256 MiB only
        if arena_bytes >= 2**28:
            released.append((torch.cuda.mem_get_info(dev)[0] - free0) / arena_bytes)

    def note_plans(adapter):  # recorded at install: uninstall() drops the handles and closes the lanes it created
        rs_groups = [g for g in adapter.groups.values() if g.rs_handle is not None]
        adapter.plan_pipeline = max((int(g.rs_handle.plan.pipeline) for g in rs_groups), default=0)
        plans = [(g.rs_handle.plan, g.lane.comm_rs.n_nodes) for g in rs_groups
                 if g.lane.comm_rs.n_nodes > 1 and g.lane.comm_rs.local_ranks > 1]
        adapter.pipelined_multi = bool(plans)
        # the registered plans: pipelined, two outbox banks (efficiency mode: a block per remote destination of each
        # pass; throughput mode: two consecutive NIC rounds); one pass unless TIDE_RS_PARTS
        adapter.outbox_ok = all(p.pipeline == 2 and p.outbox_bank == p.parts * (m - 1) and p.outbox_slots == 2 * p.outbox_bank
                                for p, m in plans)
        adapter.outbox_rounds = all(
            p.pipeline == 2 and p.outbox_slots == 2 * p.outbox_bank and
            p.outbox_bank == min(p.parts * (m - 1), max([2] + [a + b for a, b in zip(p.k_rounds, [0] + list(p.k_rounds))]))
            for p, m in plans)
        adapter.one_pass = all(p.parts == 1 for p, _ in plans)

    def tide_run(comms=None):
        c, c_rs = comms if comms is not None else (comm, comm_rs)
        model = build(args, dev, 1)
        reserved0 = torch.cuda.memory_reserved(dev)
        adapter = tide.fsdp.install(model, c, slots=args.slots, reduce_scatter=args.rs, all_gather=not args.rs_only,
                                    comm_rs=c_rs, subgroups=args.ep > 1 or args.replicate > 1)
        note_plans(adapter)
        if adapter.arena is not None and adapter.arena.numel() >= 2**28:  # share of the arena in torch's reserved memory
            in_torch.append((torch.cuda.memory_reserved(dev) - reserved0) / adapter.arena.numel())
        gen = torch.Generator(device=dev).manual_seed(data_seed)
        if args.uninstall_after > 0:
            # TIDE for the first steps, then either the stock path or a second install in the same
            # process: the unsharded parameters are rebuilt off the (new) arena, the run still matches
            losses = run(model, args, dev, gen, args.uninstall_after)
            uninstall(adapter)
            if args.reinstall:
                losses += run(model, args, dev, gen, args.stock_between)  # stock rebuilds its unsharded params
                adapter = tide.fsdp.install(model, c, slots=args.slots, reduce_scatter=args.rs, all_gather=not args.rs_only,
                                            comm_rs=c_rs, subgroups=args.ep > 1 or args.replicate > 1)
                note_plans(adapter)
            losses += run(model, args, dev, gen, args.steps - args.uninstall_after - (args.stock_between if args.reinstall else 0))
            if args.reinstall:
                uninstall(adapter)
        else:
            losses = run(model, args, dev, gen, args.steps)
            if args.last_no_reshard and os.environ.get("TIDE_RING_TAIL", "1") == "1":
                adapter.tail_ok = _tail_in_ring(model, adapter)
            uninstall(adapter)
        return adapter, torch.stack(losses), params_of(model), model.grad_seen_on_no_grad

    adapter, tide_losses, tide_params, tide_grad_seen = tide_run()
    comms2 = None  # the second run's communicators (None: the first run's)
    if args.rs and adapter.plan_pipeline == 2 and not args.copyin:  # the reference: one-phase calls on fresh communicators
        prev = os.environ.get("TIDE_RS_PIPELINE")
        os.environ["TIDE_RS_PIPELINE"] = "1"  # read at communicator creation; wins over the adapter's tune_rs
        comms2 = (tide.TideComm(device=dev, num_rma_ctx=args.num_rma_ctx),
                  tide.TideComm(device=dev, num_rma_ctx=args.num_rma_ctx) if args.split_comm else None)
        if prev is None:
            os.environ.pop("TIDE_RS_PIPELINE")
        else:
            os.environ["TIDE_RS_PIPELINE"] = prev
    elif args.copyin:  # fresh too: a communicator keeps the tuning of its first registration, and copy-in plans its own
        comms2 = (tide.TideComm(device=dev, num_rma_ctx=args.num_rma_ctx),
                  tide.TideComm(device=dev, num_rma_ctx=args.num_rma_ctx) if args.split_comm else None)
    adapter2 = None
    if args.rs:
        # same inputs again: the reduce-scatter order is fixed (and pipelining changes no sum, nor does the copy-in)
        if args.copyin:
            os.environ["TIDE_FSDP_COPYIN"] = "1"  # read at install
        adapter2, losses2, params2, _ = tide_run(comms=comms2)
        os.environ.pop("TIDE_FSDP_COPYIN", None)
        det = bool(torch.equal(tide_losses, losses2)) and all(torch.equal(a, b) for a, b in zip(tide_params, params2))
        if rank == 0 and not det:
            print("run1 losses", [float(x) for x in tide_losses], "run2 losses", [float(x) for x in losses2])
            for i, (a, b) in enumerate(zip(tide_params, params2)):
                if not torch.equal(a, b):
                    print(f"param {i} shape {tuple(a.shape)} mismatches {int((a != b).sum())} maxdiff {float((a - b).abs().max())}")
        loss_eq = bool(torch.allclose(ref_losses, tide_losses, rtol=1e-4, atol=1e-4))
        bad = [int((~torch.isclose(a, b, rtol=1e-3, atol=1e-4)).sum().item()) for a, b in zip(ref_params, tide_params)]
        param_bad = sum(bad)
        if param_bad:
            for i, (n, a, b, c) in enumerate(zip(bad, ref_params, tide_params, init_params)):
                if n:  # update ratio tide/reference where the reference moved most: a scale bug shows as a constant
                    du, dt = (a - c).flatten(), (b - c).flatten()
                    big = du.abs() > du.abs().max() * 0.3
                    print(f"rank {rank} param {i} shape {tuple(a.shape)} off-reference {n} "
                          f"maxdiff {float((a - b).abs().max()):.3g} update ratio {float((dt[big] / du[big]).median()):.4g}",
                          flush=True)
    else:
        det = True
        loss_eq = bool(torch.equal(ref_losses, tide_losses))
        param_bad = sum(int((a != b).sum().item()) for a, b in zip(ref_params, tide_params))
    tide_calls = sum(g.calls for g in adapter.groups.values())
    rs_calls = sum(g.rs_calls for g in adapter.groups.values())
    rs_fused = sum(g.rs_fused for g in adapter.groups.values())
    res = {"rank": rank, "rs": args.rs, "rs_only": args.rs_only, "accum": args.accum, "loss_match": loss_eq, "param_mismatch_elems": param_bad,
           "deterministic": det,
           "ref_losses": [round(float(x), 6) for x in ref_losses.tolist()],
           "tide_losses": [round(float(x), 6) for x in tide_losses.tolist()],
           "groups": len(adapter.groups), "ring_slots": len(adapter.ring), "dedicated": len(adapter.dedicated),
           "lanes": sorted((len(r), len(l.groups)) for r, l in adapter.lanes.items() if l is not None),
           "strided": sorted({type(fp.fsdp_placement).__name__ for g in adapter.groups.values() for fp in g.pg.fsdp_params}),
           "ext_groups": sum(1 for g in adapter.groups.values() if getattr(g, "ext", False)),
           "ext_segments": sum(len(g.dtypes) for g in adapter.groups.values() if getattr(g, "ext", False)),
           "tide_ag_calls": tide_calls, "stock_fallback_calls": adapter.stock_calls,
           "tide_rs_calls": rs_calls, "tide_rs_fused": rs_fused, "stock_rs_calls": adapter.stock_rs_calls,
           "arena_released": [round(x, 2) for x in released], "arena_in_torch": [round(x, 2) for x in in_torch]}
    wide = [g for g in adapter.groups.values() if len(g.pg.fsdp_params) > 256]
    wide_ok = True
    # (counters only: uninstall() has dropped the groups' handles and slots)
    if args.wide:  # that group runs the TIDE all-gather; with --rs exactly it keeps the stock reduce-scatter (no plan)
        wide_ok = len(wide) == 1 and wide[0].calls > 0 and wide[0].rs_calls == 0
        if args.rs:
            wide_ok = wide_ok and set(adapter.stock_rs_reasons) == {
                (getattr(wide[0].pg, "_module_fqn", "?"), "no reduce-scatter plan")}
        res["wide_groups"] = [{"params": len(g.pg.fsdp_params), "ag_calls": g.calls, "rs_calls": g.rs_calls}
                              for g in wide]
        res["stock_rs_reasons"] = {f"{k[0][:40]}: {k[1]}": n for k, n in adapter.stock_rs_reasons.items()}
    # HSDP: the reduce-scatter is TIDE's, the all-reduce over the replicas FSDP's stock collective
    rs_ok = (rs_calls > 0 and (adapter.stock_rs_calls == 0 or args.wide > 0)) if args.rs else True
    # accumulating microbatches (all but the first of a step) must reduce straight into the grads -- except under
    # HSDP, where the replica all-reduce comes first
    fused_ok = rs_fused == (0 if args.replicate > 1 else rs_calls * (args.accum - 1) // args.accum) if args.rs else True
    released_ok = all(x >= 0.5 for x in released)  # uninstall() gave the arena's device memory back
    # the arena is torch memory (MemPool on NCCL's allocator) unless TIDE_ARENA_TORCH=0
    in_torch_ok = all((x >= 0.99) == (os.environ.get("TIDE_ARENA_TORCH", "1") != "0") for x in in_torch)
    ag_ok = (tide_calls == 0) if args.rs_only else (tide_calls > 0 and adapter.stock_calls == 0)
    if args.reverse_forward and args.last_no_reshard:  # the groups unsharding after the ring tail fell back to stock
        ag_ok = tide_calls > 0 and adapter.stock_calls > 0
    no_grad_ok = not (ref_model.grad_seen_on_no_grad or tide_grad_seen)
    tail_ok = getattr(adapter, "tail_ok", True)  # TIDE_RING_TAIL=1 + --last-no-reshard: the last block sits in the ring
    pipe_ok = True
    if args.rs:  # every multi-node TIDE reduce-scatter took the pipelined path; the early all-gather ran where it applies
        multi = adapter.pipelined_multi
        if multi:  # an expert lane of one rank per node stays one-phase
            pipe_ok = (adapter.rs_pipelined_calls == rs_calls or args.ep > 1) and adapter.rs_pipelined_calls > 0 and \
                adapter.plan_pipeline == 2
            pipe_ok = pipe_ok and adapter.outbox_ok and ("TIDE_RS_PARTS" in os.environ or adapter.one_pass)
        else:
            pipe_ok = adapter.rs_pipelined_calls == 0
        if multi and not (args.block_prefetch or args.rs_only or args.ep > 1) and args.layers >= 3:
            pipe_ok = pipe_ok and adapter.early_prefetches > 0
        res.update(rs_pipelined_calls=adapter.rs_pipelined_calls, early_prefetches=adapter.early_prefetches,
                   plan_pipeline=adapter.plan_pipeline)
    copyin_ok = True
    if args.copyin:  # every TIDE reduce-scatter of the second run read the slots, no early all-gather, two-round outbox banks
        rs_calls2 = sum(g.rs_calls for g in adapter2.groups.values())
        # (--reverse-forward: the groups left on the stock all-gather have no slot and are read in place;
        # --ext: so are the extension groups)
        copyin_ok = adapter2.rs_copyin_calls > 0 and adapter2.early_prefetches == 0 and \
            (adapter2.rs_copyin_calls == rs_calls2 or args.reverse_forward or args.ext) and \
            (not adapter2.pipelined_multi or adapter2.outbox_rounds or "TIDE_RS_OUTBOX" in os.environ)
        res.update(rs_copyin_calls=adapter2.rs_copyin_calls, copyin_run_rs_calls=rs_calls2)
    # --ext: every expert group is an extension group on TIDE (two segments per weight), unless TIDE_FSDP_EXTENSIONS=0
    # keeps them on the stock path (then the all-gather / reduce-scatter counters include stock calls by design)
    ext_groups = [g for g in adapter.groups.values() if getattr(g, "ext", False)]
    ext_on = os.environ.get("TIDE_FSDP_EXTENSIONS", "1") == "1"
    ext_ok = True
    if args.ext:
        if ext_on:
            ext_ok = len(ext_groups) == args.layers and all(len(g.dtypes) == 2 * len(g.pg.fsdp_params) and g.calls > 0
                                                             for g in ext_groups)
        else:
            ext_ok = not ext_groups and adapter.stock_calls > 0
            ag_ok = tide_calls > 0  # the main lane runs TIDE, the extension groups stock
            rs_ok = (rs_calls > 0) if args.rs else True
    checks = torch.tensor([int(loss_eq), int(param_bad == 0), int(det), int(ag_ok), int(rs_ok), int(fused_ok),
                           int(released_ok), int(no_grad_ok), int(tail_ok), int(in_torch_ok), int(wide_ok),
                           int(pipe_ok), int(copyin_ok), int(ext_ok)], device=dev)
    per_rank = [torch.zeros_like(checks) for _ in range(dist.get_world_size())]
    dist.all_gather(per_rank, checks)
    ok = all(bool(c.min().item()) for c in per_rank)
    faults = os.environ.get("TIDE_FAULT", "").split(",")
    if args.rs and ("earlyread" in faults or (args.copyin and "noreadwait" in faults)):
        ok = not all(bool(c[:3].min().item()) for c in per_rank)  # negative: loss / params / det must show the corruption
        res["negative_caught"] = ok
    if rank == 0:
        names = ["loss", "params", "det", "ag", "rs", "rs_fused", "arena_released", "no_grad", "tail", "arena_in_torch", "wide",
                 "pipelined", "copyin", "ext"]
        res["failing_ranks"] = {r: [n for n, v in zip(names, c.tolist()) if not v]
                                for r, c in enumerate(per_rank) if not bool(c.min().item())}
        print(json.dumps(res), flush=True)
        print("PASS" if ok else "FAIL", flush=True)
    dist.barrier()
    comm.close()
    assert comm.closed
    for c in (comm_rs, *(comms2 or ())):
        if c is not None:
            c.close()
    dist.destroy_process_group()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
