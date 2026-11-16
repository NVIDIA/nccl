# SPDX-FileCopyrightText: Copyright (c) 2026 Kuaishou Technology. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TIDE reduce-scatter against the stock FSDP2 reference (torch._chunk_cat + reduce_scatter_tensor).

Integer-valued inputs make every fp32 sum exact, so TIDE and stock must agree bitwise whatever the
reduction order; random inputs are checked against an fp64 reduce-scatter; the same inputs reduced
`--repeat` times must give bitwise identical outputs (fixed source order). scale / accumulate exercise
the fused landing. Segments mix dim-0 sizes that are and are not multiples of the world size, tiny
odd vectors and 16 B-misaligned rows.

On one node and on one rank per node the kernels' summation order is fixed end to end, so the output
must also equal, bitwise, an fp32 host sum in that order (on several nodes of several ranks the
multicast landing adds in the switch, in an order of its own).

Negatives (exit 0 iff the outcome matches the mode); the delays act on TIDE_FAULT_RANK for TIDE_FAULT_US:
  TIDE_FAULT=nowrite            nothing moves -> wrong
  TIDE_FAULT=nofence,slowpush   owners sum before the slow rank's pieces landed
  TIDE_FAULT=nofree,slowsum     sources overwrite a staging slot the slow rank is still summing (needs >= 2
                                chunks: set TIDE_RS_CHUNK_MIB=4, the default round policy fits 32 MiB in one)
  TIDE_FAULT=nocredit,slow      multi-node: the next call's partial overwrites an inbox the slow rank is still
                                accumulating (TIDE_FAULT_US longer than one call)
  TIDE_FAULT=nodone,slow        multi-wave (TIDE_RS_WAVE_NODES below the node count): a reused inbox parity is
                                overwritten before the slow owner landed it
  TIDE_FAULT=nozero             multi-wave: blocks a wave fed are not zeroed for the wave reusing the parity

torchrun --standalone --nproc_per_node 8 test/test_rs_bitwise.py
"""
from __future__ import annotations

import argparse
import json
import os
import random

import torch
import torch.distributed as dist

import tide
from tide.testing import sub_group


def make_shapes(rng: random.Random, world: int, target_out_elems: int, segments: int) -> list[tuple[int, int]]:
    """(dim0, rest) per segment: big matrices plus a tail of small/odd vectors."""
    shapes = []
    remaining = target_out_elems * world
    n_big = max(1, segments - 3)
    for i in range(n_big):
        rest = rng.choice([1024, 1536, 2048, 4096, 1000, 4104])  # 4104: rows not 16 B multiples in bf16? (yes: 8208 B)
        share = remaining // (n_big - i)
        dim0 = max(world, share // rest)
        if rng.random() < 0.5:
            dim0 += rng.randrange(1, world)  # not a multiple of world -> padding
        shapes.append((dim0, rest))
        remaining -= dim0 * rest
    shapes += [(100, 1), (world * 3 + 1, 5), (7, 3)]  # tiny, odd, padded
    return shapes[:segments] if segments < len(shapes) else shapes


def stock_reduce_scatter(grads: list[torch.Tensor], world: int, dtype: torch.dtype, group) -> torch.Tensor:
    padded = [((g.shape[0] + world - 1) // world * world) * (g.numel() // g.shape[0]) for g in grads]
    inp = torch.empty(sum(padded), dtype=dtype, device=grads[0].device).view(world, -1)
    torch._chunk_cat([g.to(dtype) for g in grads], dim=0, num_chunks=world, out=inp)
    out = torch.empty(inp.shape[1], dtype=dtype, device=inp.device)
    dist.reduce_scatter_tensor(out, inp, op=dist.ReduceOp.SUM, group=group)
    return out


def ordered_reference(grads: list[torch.Tensor], world: int, me: int, rounds, group) -> torch.Tensor:
    """fp32 host sum in the kernels' order. One node (rounds None): the other ranks' pieces in rank
    order, then the own piece. One rank per node: per NIC round its blocks in order -- the j-th
    destination's block comes from rank me - j -- the own piece after the first round's blocks, the
    rounds' sums added to the output in order."""
    padded = [((g.shape[0] + world - 1) // world * world) * (g.numel() // g.shape[0]) for g in grads]
    inp = torch.empty(sum(padded), dtype=grads[0].dtype, device=grads[0].device).view(world, -1)
    torch._chunk_cat(grads, dim=0, num_chunks=world, out=inp)
    recv = torch.empty_like(inp)
    dist.all_to_all_single(recv, inp, group=group)  # recv[r]: rank r's pieces for me
    pieces = recv.float().cpu()
    if rounds is None:
        groups = [[r for r in range(world) if r != me] + [me]]
    else:
        groups, j = [], 1
        for i, k in enumerate(rounds):
            groups.append([(me - j - b) % world for b in range(k)] + ([me] if i == 0 else []))
            j += k
    out = None
    for g in groups:
        acc = torch.zeros(pieces.shape[1])
        for r in g:
            acc += pieces[r]
        out = acc if out is None else out + acc
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard-mib", type=float, default=32.0, help="fp32 output bytes per rank")
    ap.add_argument("--segments", type=int, default=12)
    ap.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    ap.add_argument("--iters", type=int, default=4, help="distinct inputs")
    ap.add_argument("--repeat", type=int, default=3, help="determinism: reductions per input")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--plans", type=int, default=1, help="distinct tables sharing one stage area, calls interleaved")
    ap.add_argument("--ep", type=int, default=1, help="run over the subgroup r %% ep (expert-group layout)")
    args = ap.parse_args()

    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    dev = torch.device("cuda", int(os.environ.get("LOCAL_RANK", rank % torch.cuda.device_count())))
    torch.cuda.set_device(dev)
    group = sub_group(args.ep)
    world = dist.get_world_size(group)
    in_dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    fault = os.environ.get("TIDE_FAULT", "")
    expect_bad = bool({"nowrite", "nofence", "nofree", "nocredit", "nodone", "nozero", "noputwait"} & set(fault.split(",")))

    rng = random.Random(args.seed)
    out_elems = int(args.shard_mib * 2**20) // 4
    all_shapes = [make_shapes(rng, world, int(out_elems * (1 + 0.3 * i)), args.segments + i) for i in range(args.plans)]
    comm = tide.TideComm(group=group, device=dev)
    templates = [[torch.empty(d0, rest, dtype=in_dtype, device=dev) for d0, rest in sh] for sh in all_shapes]
    stage = comm.rs_area_bytes(templates)
    arena = comm.alloc_arena(stage)
    handles = [comm.register_rs(arena, 0, t) for t in templates]  # any order: the area grows to the largest
    del templates
    shapes, h = all_shapes[0], handles[0]
    order_ref = "rail" if comm.rail else ("local" if comm.n_nodes == 1 else None)
    rounds = list(h.plan.k_rounds) if comm.rail else None

    bad_exact = bad_close = bad_det = bad_acc = bad_order = 0
    checks = 0
    for it in range(args.iters):
        g = torch.Generator(device=dev).manual_seed(args.seed * 7919 + it * 131 + rank)
        integer = it % 2 == 0
        grads = []
        for d0, rest in shapes:
            if integer:
                t = torch.randint(-8, 9, (d0, rest), generator=g, device=dev).to(in_dtype)
            else:
                t = torch.randn(d0, rest, generator=g, device=dev).to(in_dtype)
            grads.append(t)
        # a second input set (+1 per element, exact in bf16 for the integer range) is interleaved so
        # that a stale or early buffer read shows up as the other set's data
        grads_b = [t + 1 for t in grads]
        ref_dtype = torch.float32 if integer else torch.float64
        ref = stock_reduce_scatter(grads, world, ref_dtype, group)
        ref_b = stock_reduce_scatter(grads_b, world, ref_dtype, group)
        # other plans' inputs (real values), interleaved to exercise the shared stage area
        others = []
        for i in range(1, args.plans):
            gi = torch.Generator(device=dev).manual_seed(args.seed * 31 + it * 17 + i * 5 + rank)
            others.append((handles[i], [torch.randn(d0, rest, generator=gi, device=dev).to(in_dtype) for d0, rest in all_shapes[i]]))
        outs, outs_b, outs_o = [], [], [[] for _ in others]
        for r in range(args.repeat):
            outs.append(h.reduce_scatter(grads)[0])
            for k, (hi, gi) in enumerate(others):
                outs_o[k].append(hi.reduce_scatter(gi)[0])
            outs_b.append(h.reduce_scatter(grads_b)[0])
        for k in range(len(others)):
            for o in outs_o[k][1:]:
                bad_det += int((o != outs_o[k][0]).sum().item())
        # scale + accumulate landing: out2 = 1.0 + 0.5 * sum
        out2 = torch.ones(h.out_numel, dtype=torch.float32, device=dev)
        h.reduce_scatter(grads, out=out2, scale=0.5, accumulate=True)
        torch.cuda.synchronize()
        if integer:
            bad_exact += int((outs[0] != ref).sum().item()) + int((outs_b[0] != ref_b).sum().item())
            bad_acc += int((out2 != 1.0 + 0.5 * ref).sum().item())
        else:
            tol = 1e-5 * ref.abs().max().clamp_min(1.0)
            bad_close += int(((outs[0].double() - ref).abs() > tol).sum().item())
            bad_close += int(((outs_b[0].double() - ref_b).abs() > tol).sum().item())
            bad_acc += int(((out2.double() - (1.0 + 0.5 * ref)).abs() > tol).sum().item())
        for o in outs[1:]:
            bad_det += int((o != outs[0]).sum().item())
        for o in outs_b[1:]:
            bad_det += int((o != outs_b[0]).sum().item())
        if order_ref:
            bad_order += int((outs[0].cpu() != ordered_reference(grads, world, comm.rank, rounds, group)).sum().item())
        checks += 1
    torch.cuda.synchronize()
    tot = torch.tensor([bad_exact, bad_close, bad_det, bad_acc, bad_order], device=dev)
    dist.all_reduce(tot)
    bad_all = int(tot.sum().item())
    ok = (bad_all > 0) if expect_bad else (bad_all == 0)
    if rank == 0:
        print(json.dumps({"fault": fault, "world": world, "nodes": comm.n_nodes, "ep": args.ep, "rail_blocks": h.plan.rail_blocks, "k_rounds": h.plan.k_rounds, "dtype": args.dtype,
                          "segments": len(shapes), "out_numel": h.out_numel, "stage_bytes": h.stage_bytes,
                          "chunks": h.plan.chunks, "waves": h.plan.waves, "bad_exact": int(tot[0]), "bad_close": int(tot[1]),
                          "bad_det": int(tot[2]), "bad_acc": int(tot[3]), "bad_order": int(tot[4]),
                          "order_ref": order_ref, "iters": args.iters}))
        print(("PASS" if ok else "FAIL") + (f" (fault '{fault}' detected)" if expect_bad and ok else ""), flush=True)
    dist.barrier()
    del h, arena
    comm.close()
    dist.destroy_process_group()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
