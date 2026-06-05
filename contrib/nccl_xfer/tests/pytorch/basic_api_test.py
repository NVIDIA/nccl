#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# See LICENSE.txt for more license information.

"""PyTorch-level correctness matrix for ``symm_mem.mxn_cast``.

This mirrors the C-level ``tests/basic_api_test_core.h`` matrix and the
PyTorch-level pytest matrix as closely as the binding allows. It is
intentionally a standalone distributed program instead of a pytest file.
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import math
import os
import sys
import warnings
from typing import Iterable

torch = None
dist = None
symm_mem = None
DeviceMesh = None
Replicate = None
Shard = None


def import_torch_modules() -> None:
    global torch, dist, symm_mem, DeviceMesh, Replicate, Shard

    if torch is not None:
        return

    import torch as torch_mod
    import torch.distributed as dist_mod
    import torch.distributed._symmetric_memory as symm_mem_mod
    from torch.distributed.device_mesh import DeviceMesh as device_mesh_cls
    from torch.distributed.tensor.placement_types import Replicate as replicate_cls
    from torch.distributed.tensor.placement_types import Shard as shard_cls

    torch = torch_mod
    dist = dist_mod
    symm_mem = symm_mem_mod
    DeviceMesh = device_mesh_cls
    Replicate = replicate_cls
    Shard = shard_cls


PL_RS = "rs"      # {Replicate(), Shard(d)}
PL_SR = "sr"      # {Shard(d), Replicate()}
PL_REPL = "repl"  # semantic full replication


@dataclasses.dataclass(frozen=True)
class DtypeSpec:
    name: str
    torch_attr: str
    element_size: int
    tag: str | None = None

    @property
    def torch_dtype(self) -> torch.dtype:
        return getattr(torch, self.torch_attr)


@dataclasses.dataclass(frozen=True)
class TestCase:
    group: str
    global_shape: tuple[int, ...]
    src_dim0: int
    dst_dim0: int
    src_shard_dim: int
    dst_shard_dim: int
    src_pl: str
    dst_pl: str
    dtype: DtypeSpec
    world_min: int
    world_divisor: int
    src_ratio_num: int = 0
    dst_ratio_num: int = 0

    @property
    def ndims(self) -> int:
        return len(self.global_shape)

    @property
    def name(self) -> str:
        gd = "x".join(str(x) for x in self.global_shape)
        esz = self.dtype.element_size
        if self.src_pl == PL_REPL and self.dst_pl == PL_REPL:
            out = f"{self.group}[gd={gd},esz={esz}]"
        elif self.src_dim0 == 0 and self.dst_dim0 == 0:
            out = (
                f"{self.group}[gd={gd},sd={self.src_shard_dim}/"
                f"{self.dst_shard_dim},esz={esz}]"
            )
        elif self.src_ratio_num == 0 and self.dst_ratio_num == 0:
            out = (
                f"{self.group}[gd={gd},m={self.src_dim0}x{self.dst_dim0}_"
                f"{self.src_pl}/{self.dst_pl},sd={self.src_shard_dim}/"
                f"{self.dst_shard_dim},esz={esz}]"
            )
        else:
            out = (
                f"{self.group}[gd={gd},m={self.src_dim0}x{self.dst_dim0}_"
                f"{self.src_pl}/{self.dst_pl},sd={self.src_shard_dim}/"
                f"{self.dst_shard_dim},esz={esz},ratio={self.src_ratio_num}:"
                f"{self.dst_ratio_num}]"
            )
        if self.dtype.tag is not None:
            out += f",dt={self.dtype.tag}"
        return out


@dataclasses.dataclass(frozen=True)
class ResolvedShape:
    src_total: int
    dst_total: int
    src_dim0: int
    src_dim1: int
    dst_dim0: int
    dst_dim1: int
    src_shard_count: int
    dst_shard_count: int


BASE_DTYPES = (
    DtypeSpec("fp32", "float32", 4),
    DtypeSpec("bf16", "bfloat16", 2),
    DtypeSpec("uint8", "uint8", 1),
)

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--list", action="store_true", help="List selected cases and exit")
    p.add_argument("--filter", default="", help="Run only cases whose name contains this substring")
    p.add_argument("--start-index", type=int, default=0, help="Skip the first N selected cases")
    p.add_argument("--min-world", type=int, default=0, help="Select cases whose minimum feasible world is >= N")
    p.add_argument("--max-world", type=int, default=0, help="Select cases whose minimum feasible world is <= N")
    p.add_argument("--limit", type=int, default=0, help="Run at most N selected cases")
    p.add_argument("--fail-fast", action="store_true", help="Stop after the first failed case")
    p.add_argument(
        "--reuse-world-group",
        action="store_true",
        help="Reuse dist.WORLD for every case instead of creating a fresh NCCL group per case",
    )
    p.add_argument(
        "--stream-mode",
        default="non-default",
        choices=("default", "non-default", "both"),
        help="CUDA stream path to test (default: non-default)",
    )
    return p.parse_args()


def emit_full_replication(cases: list[TestCase]) -> None:
    for dtype in BASE_DTYPES:
        cases.append(
            TestCase(
                group="full_replication",
                global_shape=(200, 200),
                src_dim0=0,
                dst_dim0=0,
                src_shard_dim=-1,
                dst_shard_dim=-1,
                src_pl=PL_REPL,
                dst_pl=PL_REPL,
                dtype=dtype,
                world_min=4,
                world_divisor=2,
            )
        )


def emit_full_sharding(cases: list[TestCase]) -> None:
    for src_sd, dst_sd in ((0, 0), (0, 1), (1, 0), (1, 1)):
        for dtype in BASE_DTYPES:
            cases.append(
                TestCase(
                    group="full_sharding",
                    global_shape=(200, 200),
                    src_dim0=0,
                    dst_dim0=0,
                    src_shard_dim=src_sd,
                    dst_shard_dim=dst_sd,
                    src_pl=PL_RS,
                    dst_pl=PL_RS,
                    dtype=dtype,
                    world_min=4,
                    world_divisor=2,
                )
            )


def emit_2d_placement_matrix(
    cases: list[TestCase],
    *,
    group: str,
    global_shape: tuple[int, int],
    n_shards: tuple[int, int],
    ratio: tuple[int, int] = (0, 0),
) -> None:
    for src_sd, dst_sd in ((0, 0), (0, 1), (1, 0), (1, 1)):
        for src_pl, dst_pl in (
            (PL_SR, PL_SR),
            (PL_RS, PL_RS),
            (PL_SR, PL_RS),
            (PL_RS, PL_SR),
        ):
            cases.append(
                TestCase(
                    group=group,
                    global_shape=global_shape,
                    src_dim0=n_shards[0],
                    dst_dim0=n_shards[1],
                    src_shard_dim=src_sd,
                    dst_shard_dim=dst_sd,
                    src_pl=src_pl,
                    dst_pl=dst_pl,
                    dtype=BASE_DTYPES[1],
                    world_min=4,
                    world_divisor=2 if ratio == (0, 0) else sum(ratio),
                    src_ratio_num=ratio[0],
                    dst_ratio_num=ratio[1],
                )
            )


def emit_2d_placement(cases: list[TestCase]) -> None:
    for n_shards in ((2, 4), (4, 2), (2, 2)):
        emit_2d_placement_matrix(
            cases,
            group="2d_placement",
            global_shape=(200, 200),
            n_shards=n_shards,
        )


def emit_uneven_ratio(cases: list[TestCase]) -> None:
    for ratio in ((3, 1), (1, 3)):
        emit_2d_placement_matrix(
            cases,
            group="uneven_ratio",
            global_shape=(240, 240),
            n_shards=(2, 2),
            ratio=ratio,
        )


def emit_tensor_size_sensitivity(cases: list[TestCase]) -> None:
    for shape in ((576, 576), (3072, 6144), (3072, 3072)):
        emit_2d_placement_matrix(
            cases,
            group="tensor_size_sensitivity",
            global_shape=shape,
            n_shards=(4, 4),
        )


def emit_nd_tensors(cases: list[TestCase]) -> None:
    for shape in ((64, 128, 128), (128, 64, 64)):
        for src_sd, dst_sd in ((0, 0), (0, 1), (0, 2), (2, 0), (1, 0), (1, 1), (1, 2)):
            if shape == (64, 128, 128) and (src_sd, dst_sd) == (0, 1):
                continue
            cases.append(
                TestCase(
                    group="nd_tensors",
                    global_shape=shape,
                    src_dim0=0,
                    dst_dim0=0,
                    src_shard_dim=src_sd,
                    dst_shard_dim=dst_sd,
                    src_pl=PL_RS,
                    dst_pl=PL_RS,
                    dtype=BASE_DTYPES[1],
                    world_min=4,
                    world_divisor=2,
                )
            )


def emit_1d_full_sharding(cases: list[TestCase]) -> None:
    for dtype in BASE_DTYPES:
        cases.append(
            TestCase(
                group="1d_full_sharding",
                global_shape=(8192,),
                src_dim0=0,
                dst_dim0=0,
                src_shard_dim=0,
                dst_shard_dim=0,
                src_pl=PL_RS,
                dst_pl=PL_RS,
                dtype=dtype,
                world_min=4,
                world_divisor=2,
            )
        )


def emit_1d_placement_matrix(
    cases: list[TestCase],
    *,
    group: str,
    global_shape: tuple[int, ...],
    n_shards: tuple[int, int],
    ratio: tuple[int, int] = (0, 0),
) -> None:
    for src_pl, dst_pl in (
        (PL_SR, PL_SR),
        (PL_RS, PL_RS),
        (PL_SR, PL_RS),
        (PL_RS, PL_SR),
    ):
        cases.append(
            TestCase(
                group=group,
                global_shape=global_shape,
                src_dim0=n_shards[0],
                dst_dim0=n_shards[1],
                src_shard_dim=0,
                dst_shard_dim=0,
                src_pl=src_pl,
                dst_pl=dst_pl,
                dtype=BASE_DTYPES[1],
                world_min=4,
                world_divisor=2 if ratio == (0, 0) else sum(ratio),
                src_ratio_num=ratio[0],
                dst_ratio_num=ratio[1],
            )
        )


def emit_1d_2d_placement(cases: list[TestCase]) -> None:
    for n_shards in ((2, 4), (4, 2), (2, 2)):
        emit_1d_placement_matrix(
            cases,
            group="1d_2d_placement",
            global_shape=(8192,),
            n_shards=n_shards,
        )


def emit_1d_uneven_ratio(cases: list[TestCase]) -> None:
    for ratio in ((3, 1), (1, 3)):
        emit_1d_placement_matrix(
            cases,
            group="1d_uneven_ratio",
            global_shape=(16384,),
            n_shards=(2, 2),
            ratio=ratio,
        )


def emit_1d_tensor_size_sensitivity(cases: list[TestCase]) -> None:
    for n in (16384, 1048576, 4194304):
        emit_1d_placement_matrix(
            cases,
            group="1d_tensor_size_sensitivity",
            global_shape=(n,),
            n_shards=(4, 4),
        )


def emit_cross_dim_regression(cases: list[TestCase]) -> None:
    for src_pl, dst_pl in (
        (PL_SR, PL_SR),
        (PL_RS, PL_RS),
        (PL_SR, PL_RS),
        (PL_RS, PL_SR),
    ):
        cases.append(
            TestCase(
                group="cross_dim_regression",
                global_shape=(200, 200),
                src_dim0=2,
                dst_dim0=4,
                src_shard_dim=0,
                dst_shard_dim=1,
                src_pl=src_pl,
                dst_pl=dst_pl,
                dtype=BASE_DTYPES[1],
                world_min=8,
                world_divisor=2,
            )
        )
    cases.append(
        TestCase(
            group="cross_dim_regression",
            global_shape=(64, 128, 128),
            src_dim0=0,
            dst_dim0=0,
            src_shard_dim=0,
            dst_shard_dim=1,
            src_pl=PL_RS,
            dst_pl=PL_RS,
            dtype=BASE_DTYPES[1],
            world_min=8,
            world_divisor=2,
        )
    )


def build_all_cases() -> list[TestCase]:
    cases: list[TestCase] = []
    emit_full_replication(cases)
    emit_full_sharding(cases)
    emit_2d_placement(cases)
    emit_uneven_ratio(cases)
    emit_tensor_size_sensitivity(cases)
    emit_nd_tensors(cases)
    emit_1d_full_sharding(cases)
    emit_1d_2d_placement(cases)
    emit_1d_uneven_ratio(cases)
    emit_1d_tensor_size_sensitivity(cases)
    emit_cross_dim_regression(cases)
    return cases


def shard_count(kind: str, dim0: int, dim1: int) -> int:
    if kind == PL_REPL:
        return 1
    if kind == PL_RS:
        return dim1
    if kind == PL_SR:
        return dim0
    raise ValueError(f"unknown placement kind: {kind}")


def case_feasible_at(case: TestCase, world_size: int) -> tuple[ResolvedShape | None, str | None]:
    if world_size < case.world_min:
        return None, "world_size below minimum"
    if case.world_divisor != 0 and world_size % case.world_divisor != 0:
        return None, "world_size not divisible by required factor"

    if case.src_ratio_num == 0 and case.dst_ratio_num == 0:
        src_total = world_size // 2
        dst_total = world_size - src_total
    else:
        total_ratio = case.src_ratio_num + case.dst_ratio_num
        src_total = world_size * case.src_ratio_num // total_ratio
        dst_total = world_size - src_total
    if src_total <= 0 or dst_total <= 0 or src_total + dst_total != world_size:
        return None, "ratio yields empty side"

    if case.src_pl == PL_REPL:
        src_dim0, src_dim1 = src_total, 1
    else:
        src_dim0 = 1 if case.src_dim0 == 0 else case.src_dim0
        if src_total % src_dim0 != 0:
            return None, "src_total not divisible by src_dim0"
        src_dim1 = src_total // src_dim0

    if case.dst_pl == PL_REPL:
        dst_dim0, dst_dim1 = dst_total, 1
    else:
        dst_dim0 = 1 if case.dst_dim0 == 0 else case.dst_dim0
        if dst_total % dst_dim0 != 0:
            return None, "dst_total not divisible by dst_dim0"
        dst_dim1 = dst_total // dst_dim0

    src_shards = shard_count(case.src_pl, src_dim0, src_dim1)
    dst_shards = shard_count(case.dst_pl, dst_dim0, dst_dim1)

    if case.src_shard_dim >= 0 and case.global_shape[case.src_shard_dim] % src_shards != 0:
        return None, "global dim not divisible by src shard count"
    if case.dst_shard_dim >= 0 and case.global_shape[case.dst_shard_dim] % dst_shards != 0:
        return None, "global dim not divisible by dst shard count"

    # Preserve the external pytest's large rs/rs skip for smaller worlds.
    if (
        case.group == "tensor_size_sensitivity"
        and world_size < 32
        and len(case.global_shape) == 2
        and case.global_shape[0] > 1024
        and case.global_shape[1] > 1024
        and case.src_pl == PL_RS
        and case.dst_pl == PL_RS
    ):
        return None, "pytest-compatible large rs/rs skip below world_size 32"

    return (
        ResolvedShape(
            src_total=src_total,
            dst_total=dst_total,
            src_dim0=src_dim0,
            src_dim1=src_dim1,
            dst_dim0=dst_dim0,
            dst_dim1=dst_dim1,
            src_shard_count=src_shards,
            dst_shard_count=dst_shards,
        ),
        None,
    )


def compute_min_world(case: TestCase, bound: int = 4096) -> int:
    divisor = case.world_divisor if case.world_divisor > 0 else 1
    start = case.world_min
    if start % divisor != 0:
        start += divisor - (start % divisor)
    start = max(start, divisor)
    for world in range(start, bound + 1, divisor):
        shape, _ = case_feasible_at(case, world)
        if shape is not None:
            return world
    return -1


def select_cases(cases: Iterable[TestCase], args: argparse.Namespace) -> list[TestCase]:
    out: list[TestCase] = []
    selected_idx = 0
    for case in cases:
        if args.filter and args.filter not in case.name:
            continue
        min_world = compute_min_world(case)
        if min_world < 0:
            continue
        if args.min_world and min_world < args.min_world:
            continue
        if args.max_world and min_world > args.max_world:
            continue
        if selected_idx < args.start_index:
            selected_idx += 1
            continue
        out.append(case)
        selected_idx += 1
        if args.limit and len(out) >= args.limit:
            break
    return out


def local_shape(global_shape: tuple[int, ...], shard_dim: int, shard_count: int) -> tuple[int, ...]:
    shape = list(global_shape)
    if shard_dim >= 0:
        shape[shard_dim] //= shard_count
    return tuple(shape)


def local_shard(
    global_tensor: torch.Tensor,
    shard_dim: int,
    shard_idx: int,
    shard_count: int,
) -> torch.Tensor:
    if shard_dim < 0:
        return global_tensor.contiguous()
    block = global_tensor.shape[shard_dim] // shard_count
    return global_tensor.narrow(shard_dim, shard_idx * block, block).contiguous()


def shard_idx_for_rank(kind: str, rank: int, start_rank: int, dim1: int, is_1d_mesh: bool) -> int:
    if kind == PL_REPL:
        return 0
    local_rank = rank - start_rank
    if is_1d_mesh:
        return local_rank
    if kind == PL_RS:
        return local_rank % dim1
    if kind == PL_SR:
        return local_rank // dim1
    raise ValueError(f"unknown placement kind: {kind}")


def mesh_and_placements(
    *,
    case: TestCase,
    resolved: ResolvedShape,
    is_src: bool,
    device_type: str,
) -> tuple[DeviceMesh, list[object]]:
    if is_src:
        total = resolved.src_total
        start = 0
        dim0, dim1 = resolved.src_dim0, resolved.src_dim1
        kind = case.src_pl
        shard_dim = case.src_shard_dim
        original_dim0 = case.src_dim0
    else:
        total = resolved.dst_total
        start = resolved.src_total
        dim0, dim1 = resolved.dst_dim0, resolved.dst_dim1
        kind = case.dst_pl
        shard_dim = case.dst_shard_dim
        original_dim0 = case.dst_dim0

    ranks = torch.arange(start, start + total, dtype=torch.int64)
    if kind == PL_REPL:
        # Match the C tests' one-shard full-replication encoding while
        # keeping local_shape == global_shape. A pure {REPLICATE, REPLICATE}
        # mesh is a degenerate prepare-time case the test suite does not
        # currently exercise.
        mesh_ranks = ranks.reshape(total, 1)
        placements = [Replicate(), Shard(0)]
    elif original_dim0 == 0:
        mesh_ranks = ranks
        placements = [Shard(shard_dim)]
    else:
        mesh_ranks = ranks.reshape(dim0, dim1)
        if kind == PL_RS:
            placements = [Replicate(), Shard(shard_dim)]
        elif kind == PL_SR:
            placements = [Shard(shard_dim), Replicate()]
        else:
            raise ValueError(f"unknown placement kind: {kind}")
    return DeviceMesh(device_type, mesh_ranks, _init_backend=False), placements


def pattern_tensor(shape: tuple[int, ...], dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    numel = math.prod(shape)
    values = torch.arange(numel, dtype=torch.int64, device=device)
    if dtype == torch.uint8:
        values = values.remainder(256)
    else:
        values = values.remainder(127)
    return values.to(dtype=dtype).reshape(shape)


def enable_symmetric_memory_group(group_name: str) -> None:
    enable = getattr(symm_mem, "enable_symm_mem_for_group", None)
    if enable is None or symm_mem.is_symm_mem_enabled_for_group(group_name):
        return

    # Some PyTorch builds still require explicit group-info setup before
    # symmetric-memory pool allocations, even though the helper is deprecated.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        enable(group_name)


def run_one_case(
    *,
    case: TestCase,
    resolved: ResolvedShape,
    stream_mode: str,
    rank: int,
    device: torch.device,
    collective_group: dist.ProcessGroup,
    group_name: str,
) -> bool:
    dtype = case.dtype.torch_dtype
    is_src = rank < resolved.src_total
    is_dst = not is_src

    src_shape = local_shape(case.global_shape, case.src_shard_dim, resolved.src_shard_count)
    dst_shape = local_shape(case.global_shape, case.dst_shard_dim, resolved.dst_shard_count)
    src_mesh, src_placements = mesh_and_placements(
        case=case, resolved=resolved, is_src=True, device_type=device.type
    )
    dst_mesh, dst_placements = mesh_and_placements(
        case=case, resolved=resolved, is_src=False, device_type=device.type
    )

    # The mxn_cast binding performs the symmetric-memory rendezvous. That
    # rendezvous requires the same allocation shape and dtype on every rank in
    # the group. Individual src/dst roles may use different local shapes, so
    # allocate the per-case maximum and only use the role-specific prefix.
    alloc_numel = max(math.prod(src_shape), math.prod(dst_shape))
    buf = symm_mem.empty(alloc_numel, dtype=dtype, device=device)

    global_values = pattern_tensor(case.global_shape, dtype, device)
    buf.fill_(0)
    if is_src:
        src_is_1d_mesh = case.src_pl != PL_REPL and case.src_dim0 == 0
        src_shard_idx = shard_idx_for_rank(
            case.src_pl, rank, 0, resolved.src_dim1, src_is_1d_mesh
        )
        src_local = local_shard(
            global_values, case.src_shard_dim, src_shard_idx, resolved.src_shard_count
        )
        buf[: src_local.numel()].copy_(src_local.reshape(-1))
    dist.barrier(group=collective_group)

    def call() -> None:
        inactive_shape = [0] * case.ndims
        symm_mem.mxn_cast(
            buf,
            src_local_shape=list(src_shape) if is_src else inactive_shape,
            src_mesh=src_mesh,
            src_placements=src_placements,
            dst_local_shape=list(dst_shape) if is_dst else inactive_shape,
            dst_mesh=dst_mesh,
            dst_placements=dst_placements,
            group=group_name,
        )

    if stream_mode == "non-default":
        op_stream = torch.cuda.Stream(device=device)
        op_stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(op_stream):
            call()
        op_stream.synchronize()
    else:
        call()
        torch.cuda.synchronize(device)
    dist.barrier(group=collective_group)

    ok = True
    if is_dst:
        dst_is_1d_mesh = case.dst_pl != PL_REPL and case.dst_dim0 == 0
        dst_shard_idx = shard_idx_for_rank(
            case.dst_pl, rank, resolved.src_total, resolved.dst_dim1, dst_is_1d_mesh
        )
        expected = local_shard(
            global_values, case.dst_shard_dim, dst_shard_idx, resolved.dst_shard_count
        )
        actual = buf[: expected.numel()].view(expected.shape)
        if not torch.equal(actual, expected):
            ok = False
            neq = (actual != expected).reshape(-1)
            first = int(torch.nonzero(neq, as_tuple=False)[0].item()) if bool(neq.any()) else -1
            print(
                f"[rank {rank}] mismatch in {case.name} stream={stream_mode} "
                f"first_flat_idx={first}",
                flush=True,
            )

    fail = torch.tensor([0 if ok else 1], device=device, dtype=torch.int32)
    dist.all_reduce(fail, op=dist.ReduceOp.MAX, group=collective_group)

    del buf, global_values, src_mesh, dst_mesh
    gc.collect()
    torch.cuda.empty_cache()
    return fail.item() == 0


def main() -> int:
    args = parse_args()
    cases = select_cases(build_all_cases(), args)

    if args.list:
        print(f"# total_cases={len(cases)}")
        print("# columns: idx min_world name")
        for idx, case in enumerate(cases, start=args.start_index):
            print(f"[{idx:4d}] {compute_min_world(case):4d} {case.name}")
        return 0

    import_torch_modules()
    symm_mem.set_backend("NCCL")
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank % torch.cuda.device_count()))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    dist.all_reduce(torch.ones(1, device=device))
    world_group = dist.group.WORLD
    world_group_name = dist.group.WORLD.group_name
    enable_symmetric_memory_group(world_group_name)
    streams = ("default", "non-default") if args.stream_mode == "both" else (args.stream_mode,)

    if rank == 0:
        print("=" * 80)
        print("  symm_mem.mxn_cast PyTorch basic_api correctness matrix")
        print(f"  world_size={world_size} streams={','.join(streams)} cases_selected={len(cases)}")
        print("=" * 80, flush=True)

    passed = 0
    skipped = 0
    failed = 0
    for idx, case in enumerate(cases, start=args.start_index):
        resolved, reason = case_feasible_at(case, world_size)
        if resolved is None:
            skipped += 1
            if rank == 0:
                min_world = compute_min_world(case)
                print(f"[SKIP] {idx:4d} {case.name} :: {reason}; min_world={min_world}", flush=True)
            continue

        for stream_mode in streams:
            if rank == 0:
                print(f"[RUN ] {idx:4d} {case.name} stream={stream_mode}", flush=True)
            if args.reuse_world_group:
                collective_group = world_group
                group_name = world_group_name
                destroy_group = False
            else:
                collective_group = dist.new_group(ranks=list(range(world_size)), backend="nccl")
                group_name = collective_group.group_name
                destroy_group = True
            try:
                enable_symmetric_memory_group(group_name)
                ok = run_one_case(
                    case=case,
                    resolved=resolved,
                    stream_mode=stream_mode,
                    rank=rank,
                    device=device,
                    collective_group=collective_group,
                    group_name=group_name,
                )
            finally:
                torch.cuda.synchronize(device)
                if destroy_group:
                    dist.destroy_process_group(collective_group)
            if ok:
                passed += 1
                if rank == 0:
                    print(f"[PASS] {idx:4d} {case.name} stream={stream_mode}", flush=True)
            else:
                failed += 1
                if rank == 0:
                    print(f"[FAIL] {idx:4d} {case.name} stream={stream_mode}", flush=True)
                if args.fail_fast:
                    break
        if failed and args.fail_fast:
            break

    if rank == 0:
        print("=" * 80)
        print(f"  SUMMARY pass={passed} skip={skipped} fail={failed}")
        print("=" * 80, flush=True)

    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(1 if failed else 0)


if __name__ == "__main__":
    main()
