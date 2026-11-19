# SPDX-FileCopyrightText: Copyright (c) 2026 Kuaishou Technology. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Helpers shared by tests and benchmarks."""
from __future__ import annotations

import torch
import torch.distributed as dist


def make_segments(shard_bytes: int, nseg: int, small_frac: float, seed: int) -> list[int]:
    """Skewed segment sizes (tiny norms + big matrices), 16 B multiples, summing to shard_bytes."""
    if nseg == 1:
        return [shard_bytes]
    g = torch.Generator().manual_seed(seed)
    nsmall = max(0, min(nseg - 1, int(round(nseg * small_frac))))
    small = [int(torch.randint(1, 1024, (1,), generator=g)) * 16 for _ in range(nsmall)]
    rest = shard_bytes - sum(small)
    nbig = nseg - nsmall
    w = torch.rand(nbig, generator=g) + 0.2
    big = [int(x) // 16 * 16 for x in (w / w.sum() * rest).tolist()]
    big[-1] += rest - sum(big)
    sizes = small + big
    assert sum(sizes) == shard_bytes and all(s > 0 and s % 16 == 0 for s in sizes), sizes
    # interleave tiny and huge pieces (worst case for descriptor batching)
    order = sorted(range(nseg), key=lambda i: (i % 2, sizes[i]))
    return [sizes[i] for i in order]


def sub_group(ep: int):
    """Collective over the world. The expert-group-style subgroup of this rank: the ranks r with
    r % ep == rank % ep, as FSDP's dp_shard_mod_ep mesh dimension forms them (ep <= 1: the world).
    Every rank creates every subgroup, in the same order."""
    if ep <= 1:
        return dist.group.WORLD
    world, rank = dist.get_world_size(), dist.get_rank()
    if world % ep:
        raise ValueError(f"world {world} is not a multiple of ep {ep}")
    mine = None
    for i in range(ep):
        g = dist.new_group([r for r in range(world) if r % ep == i])
        if i == rank % ep:
            mine = g
    # NCCL creates a group's communicator at its first collective: do it now, not behind a TIDE call
    dist.all_reduce(torch.zeros(1, device="cuda"), group=mine)
    torch.cuda.synchronize()
    return mine


def param_major_permute(shard: torch.Tensor, sizes, offsets, group) -> torch.Tensor:
    """Reference: stock all-gather (rank-major) followed by the parameter-major permutation."""
    N = dist.get_world_size(group)
    S = shard.numel()
    ref = torch.empty(N * S, dtype=torch.uint8, device=shard.device)
    dist.all_gather_into_tensor(ref, shard, group=group)
    exp = torch.empty_like(ref)
    r2 = ref.view(N, S)
    for s, o in zip(sizes, offsets):
        exp[o * N : (o + s) * N].view(N, s).copy_(r2[:, o : o + s])
    return exp
