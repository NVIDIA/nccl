# SPDX-FileCopyrightText: Copyright (c) 2026 Kuaishou Technology. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TideComm.tune_rs: the solver's sizing reaches the plan, environment knobs win, tuning is refused
once a reduce-scatter is registered, and a tuned plan reduces integer-valued gradients bitwise like
stock (chunk_cat + reduce_scatter_tensor).

torchrun --standalone --nproc_per_node 8 test/test_rs_tuning.py
TIDE_RS_KWAY=3 TIDE_RS_CTAS=3 torchrun ... test/test_rs_tuning.py      # env precedence
"""
from __future__ import annotations

import os

import torch
import torch.distributed as dist

import tide
from tide import solver

MIB = 2**20


def main() -> None:
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    dev = torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
    torch.cuda.set_device(dev)
    comm = tide.TideComm(device=dev)

    shapes = [(4096 * 2048, 4096), (1024 * 1000, 1024), (world * 3 + 1, world * 3 + 1), (100, 100)]
    big = [(2 * n, 2 * d) for n, d in shapes]  # the larger table sizes the solver
    rs = comm.tune_rs([(shapes, 2), (big, 2)])
    out_bytes = 4 * sum(-(-d // world) * (n // d) for n, d in big)
    ref = solver.plan_rs(solver.Fleet(comm.n_nodes, comm.local_ranks, len(big), rs_out_bytes=out_bytes, rs_itemsize=2,
                                      rs_area_budget=solver.rs_area_budget(comm.local_ranks, out_bytes,
                                                                          solver.load_calib(os.environ.get("TIDE_CALIB", ""))),
                                      rs_rail_parts=comm._c.rs_rail_parts, rs_parts=comm._c.rs_parts_env),
                         solver.load_calib(os.environ.get("TIDE_CALIB", "")))
    assert rs == ref, (rs, ref)
    want = [ref.ksum_ctas, ref.landing_ctas, ref.kway, ref.chunk_bytes]
    for i, env in enumerate(["TIDE_RS_CTAS", "TIDE_RS_MC_CTAS", "TIDE_RS_KWAY", "TIDE_RS_CHUNK_MIB"]):
        if env in os.environ:
            v = int(os.environ[env])
            want[i] = v << 20 if i == 3 else v
        elif i == 1 and ref.landing_ctas <= 0:
            want[i] = comm.rs_tuning[1]  # single node: no landing, the default stays
    assert list(comm.rs_tuning) == want, (comm.rs_tuning, want)
    assert comm.rs_parts == (comm._c.rs_parts_env or ref.parts), (comm.rs_parts, ref.parts)

    arena = comm.alloc_arena(comm.rs_area_bytes([(shapes, 2), (big, 2)]))
    h = comm.register_rs(arena, 0, (shapes, 2))
    assert comm.tune_rs([(shapes, 2)]) is None
    try:
        comm._c.set_rs_tuning(1, 1, 1, MIB)
        raise AssertionError("set_rs_tuning accepted after a registration")
    except RuntimeError:
        pass

    g = torch.Generator(device=dev).manual_seed(1234 + rank)
    grads = [torch.randint(-8, 9, (d, n // d), generator=g, device=dev).to(torch.bfloat16) for n, d in shapes]
    out = torch.empty(h.out_numel, dtype=torch.float32, device=dev)
    comm.reduce_scatter(h, grads, out=out)
    padded = [(-(-g.shape[0] // world) * world) * (g.numel() // g.shape[0]) for g in grads]
    inp = torch.empty(sum(padded), dtype=torch.float32, device=dev).view(world, -1)
    torch._chunk_cat(grads, dim=0, num_chunks=world, out=inp)
    exp = torch.empty(inp.shape[1], dtype=torch.float32, device=dev)
    dist.reduce_scatter_tensor(exp, inp.view(-1), op=dist.ReduceOp.SUM)
    ok = torch.equal(out, exp)
    flags = torch.tensor([int(ok)], device=dev)
    dist.all_reduce(flags, op=dist.ReduceOp.MIN)
    if rank == 0:
        print(f"tune_rs: tuning {comm.rs_tuning} chunks {h.plan.chunks} bitwise {'PASS' if flags.item() else 'FAIL'}")
    comm.free_arena(arena)
    comm.close()
    dist.destroy_process_group()
    if not flags.item():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
