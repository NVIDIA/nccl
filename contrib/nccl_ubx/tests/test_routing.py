"""Tests for compute_token_offsets — single-GPU, no distributed setup needed."""

import pytest
import torch
from ubx import compute_token_offsets


def _routing(rows, dtype=torch.uint8, device="cpu"):
    """Build a routing matrix from a list of per-token expert lists."""
    nranks_experts = max(max(r) for r in rows if r) + 1 if any(rows) else 1
    total_experts = max(max(max(r) for r in rows if r) + 1, 1)
    ntokens = len(rows)
    t = torch.zeros(ntokens, total_experts, dtype=dtype, device=device)
    for i, experts in enumerate(rows):
        for e in experts:
            t[i, e] = 1
    return t


# ---------------------------------------------------------------------------
# Basic correctness
# ---------------------------------------------------------------------------

def test_k1_each_token_one_expert():
    """K=1: token t goes to expert t, 4 tokens, 4 experts, 2 ranks, 2 epr.

    Expert sub-ranges (max_slots=1):
      e0 (e_local=0): slot [0, 1)  e1 (e_local=1): slot [1, 2)
      e2 (e_local=0): slot [0, 1)  e3 (e_local=1): slot [1, 2)

    rank 0 (tokens 0,1): token0→e0 slot 0, token1→e1 slot 1
    rank 1 (tokens 2,3): token2→e2 slot 0, token3→e3 slot 1
    """
    routing = _routing([[0], [1], [2], [3]])  # [4, 4]
    offsets0, max0, _, _ = compute_token_offsets(routing, experts_per_rank=2, myrank=0, nranks=2)
    offsets1, max1, _, _ = compute_token_offsets(routing, experts_per_rank=2, myrank=1, nranks=2)

    assert max0 == max1 == 2  # 2 experts_per_rank * 1 max_slot
    assert offsets0.shape == (2, 4)
    assert offsets1.shape == (2, 4)

    # rank 0 local token 0 → expert 0, slot 0 (base=0*1=0, prefix=0)
    assert offsets0[0, 0] == 0
    assert (offsets0[0, 1:] == -1).all()

    # rank 0 local token 1 → expert 1, slot 1 (base=1*1=1, prefix=0)
    assert offsets0[1, 1] == 1
    assert offsets0[1, 0] == -1
    assert (offsets0[1, 2:] == -1).all()

    # rank 1 local token 2 → expert 2, slot 0 (base=0*1=0, prefix=0)
    assert offsets1[0, 2] == 0
    # rank 1 local token 3 → expert 3, slot 1 (base=1*1=1, prefix=0)
    assert offsets1[1, 3] == 1


def test_k2_token_to_two_experts():
    """K=2: each token routed to two experts.

    4 tokens, 4 experts, 2 ranks (epr=2).
    token 0: experts 0,1   token 1: experts 2,3
    token 2: experts 0,2   token 3: experts 1,3

    Expert counts: e0=2, e1=2, e2=2, e3=2 → max_slots=2
    Sub-ranges: e0→[0,2), e1→[2,4), e2→[0,2), e3→[2,4)
    """
    routing = _routing([[0, 1], [2, 3], [0, 2], [1, 3]])  # [4, 4]
    offsets0, max0, _, _ = compute_token_offsets(routing, experts_per_rank=2, myrank=0, nranks=2)
    offsets1, max1, _, _ = compute_token_offsets(routing, experts_per_rank=2, myrank=1, nranks=2)

    assert max0 == max1 == 4  # 2 epr * 2 max_slots

    # rank 0, token 0 → e0 slot 0 (prefix=0), e1 slot 2 (base=2, prefix=0)
    assert offsets0[0, 0] == 0
    assert offsets0[0, 1] == 2
    assert offsets0[0, 2] == -1
    assert offsets0[0, 3] == -1

    # rank 0, token 1 → e2 slot 0 (base=0, prefix=0), e3 slot 2 (base=2, prefix=0)
    assert offsets0[1, 0] == -1
    assert offsets0[1, 1] == -1
    assert offsets0[1, 2] == 0
    assert offsets0[1, 3] == 2

    # rank 1, token 2 → e0 slot 1 (base=0, prefix=1), e2 slot 1 (base=0, prefix=1)
    assert offsets1[0, 0] == 1
    assert offsets1[0, 2] == 1

    # rank 1, token 3 → e1 slot 3 (base=2, prefix=1), e3 slot 3 (base=2, prefix=1)
    assert offsets1[1, 1] == 3
    assert offsets1[1, 3] == 3


def test_no_slot_collisions():
    """Every (dest_rank, slot) pair used by token_offsets is unique across all source ranks."""
    torch.manual_seed(0)
    ntokens, total_experts, nranks, epr = 16, 8, 4, 2
    # K=2 routing: each token goes to exactly 2 random experts
    routing = torch.zeros(ntokens, total_experts, dtype=torch.uint8)
    for t in range(ntokens):
        chosen = torch.randperm(total_experts)[:2]
        routing[t, chosen] = 1

    all_offsets = []
    for r in range(nranks):
        offs, max_tpr, _, _ = compute_token_offsets(routing, epr, myrank=r, nranks=nranks)
        all_offsets.append((r, offs))

    _, first_offs = all_offsets[0]
    _, first_max, _, _ = compute_token_offsets(routing, epr, myrank=0, nranks=nranks)
    max_tpr = first_max

    for dest_rank in range(nranks):
        # Collect all (slot) values written to dest_rank's experts by any source rank
        dest_expert_start = dest_rank * epr
        dest_expert_end = dest_expert_start + epr
        used_slots = set()
        for src_rank, offs in all_offsets:
            for local_t in range(ntokens // nranks):
                for e in range(dest_expert_start, dest_expert_end):
                    s = offs[local_t, e].item()
                    if s >= 0:
                        key = (e, s)
                        assert key not in used_slots, (
                            f"Slot collision: dest_rank={dest_rank} expert={e} slot={s} "
                            f"from src_rank={src_rank} token={local_t}"
                        )
                        used_slots.add(key)


def test_max_tokens_per_rank_is_symmetric():
    """max_tokens_per_rank must be identical for all myrank values."""
    # 8 tokens, 4 experts, 4 ranks, 1 expert per rank
    routing = _routing([[0], [1], [2], [3], [0], [3], [2], [1]])
    results = [
        compute_token_offsets(routing, experts_per_rank=1, myrank=r, nranks=4)
        for r in range(4)
    ]
    maxes = [m for _, m, _ in results]
    assert len(set(maxes)) == 1, f"max_tokens_per_rank differs across ranks: {maxes}"


def test_unrouted_entries_are_minus_one():
    """All entries where routing=0 must produce token_offsets=-1."""
    routing = _routing([[0], [3], [1], [2]])  # [4, 4], sparse
    offsets, _, _, _ = compute_token_offsets(routing, experts_per_rank=2, myrank=0, nranks=2)
    routed_mask = routing[:2] != 0  # local tokens for rank 0
    assert (offsets[~routed_mask] == -1).all()


def test_all_tokens_same_expert():
    """All tokens route to expert 0. max_slots_per_expert = ntokens."""
    ntokens, total_experts, nranks, epr = 8, 4, 2, 2
    routing = torch.zeros(ntokens, total_experts, dtype=torch.uint8)
    routing[:, 0] = 1  # all tokens → expert 0

    offsets, max_tpr, _, _ = compute_token_offsets(routing, epr, myrank=0, nranks=nranks)
    assert max_tpr == epr * ntokens  # max_slots=ntokens, total=epr*ntokens

    # rank 0 local tokens 0..3 all go to expert 0 with sequential slots 0,1,2,3
    for local_t in range(ntokens // nranks):
        assert offsets[local_t, 0] == local_t
        assert (offsets[local_t, 1:] == -1).all()


def test_empty_routing():
    """No tokens routed anywhere. max_tokens_per_rank=0, all offsets -1."""
    routing = torch.zeros(4, 4, dtype=torch.uint8)
    offsets, max_tpr, _, _ = compute_token_offsets(routing, experts_per_rank=2, myrank=0, nranks=2)
    assert max_tpr == 0
    assert (offsets == -1).all()


def test_bool_and_int_routing_dtypes():
    """routing can be bool or int32, not just uint8."""
    base = _routing([[0, 1], [2, 3], [0, 2], [1, 3]])
    for dtype in (torch.bool, torch.int32, torch.int64):
        routing = base.to(dtype)
        offsets, max_tpr, _, _ = compute_token_offsets(routing, experts_per_rank=2, myrank=0, nranks=2)
        assert offsets.dtype == torch.int32
        assert max_tpr > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_gpu_routing():
    """Function works on CUDA tensors and returns CUDA token_offsets."""
    routing = _routing([[0], [1], [2], [3]], device="cuda")
    offsets, max_tpr, _, _ = compute_token_offsets(routing, experts_per_rank=2, myrank=0, nranks=2)
    assert offsets.is_cuda
    assert offsets.dtype == torch.int32
    assert max_tpr == 2
