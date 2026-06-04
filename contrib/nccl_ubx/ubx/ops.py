"""Module-level convenience functions.

Provides a global allocator registry (_allocator_map) and convenience functions
that manage allocator lifecycles and dispatch to the appropriate collective ops.
"""

from __future__ import annotations

import os
import torch
from typing import Dict, Optional, Tuple

from .allocator import SymmAllocator, AUTO_SWITCH_BYTES
from .tensor import SymmTensor
from .fused import allreduce_fused

# Global allocator registry: maps process groups to (max_size, allocator) pairs
_allocator_map: Dict[torch.distributed.group, Tuple[int, Optional[SymmAllocator]]] = {}


def request_allocator(
    dist_group: torch.distributed.group,
    shape: Optional[torch.Size] = None,
    dtype: torch.dtype = torch.bfloat16,
) -> None:
    """Register a request for a SymmAllocator for the given process group.

    Multiple calls accumulate the maximum tensor size needed. The allocator
    is lazily created on the first call to get_sym_tensor().
    """
    if shape is not None:
        num_elements = torch.Size(shape).numel()
        element_size = torch.tensor(0, dtype=dtype).element_size()
        tensor_size = num_elements * element_size
    else:
        tensor_size = 0

    if dist_group not in _allocator_map:
        _allocator_map[dist_group] = (tensor_size, None)
    else:
        old_size, allocator = _allocator_map[dist_group]
        assert allocator is None, "Second element of tuple must be None"
        max_size = max(old_size, tensor_size)
        _allocator_map[dist_group] = (max_size, None)


def get_sym_tensor(
    shape: torch.Size, dtype: torch.dtype, dist_group: torch.distributed.group
) -> Optional[torch.Tensor]:
    """Create a SymmTensor for the given shape/dtype/group.

    Lazily creates the SymmAllocator on first call. Returns None if the
    dtype is not supported (only bf16 currently) or no allocator was requested.
    """
    if dtype != torch.bfloat16:
        return None  # Unsupported dtype, fallback to NCCL
    if dist_group not in _allocator_map:
        return None  # No allocator requested, fallback to NCCL
    (max_size, allocator) = _allocator_map[dist_group]
    if allocator is None:
        new_max_size = int(
            os.environ.get("UBX_SYMM_POOL_SIZE", ((6 * max_size + 1048575) / 1024 / 1024))
        )
        allocator = SymmAllocator(
            new_max_size * 1024 * 1024,
            torch.device(f"cuda:{torch.cuda.current_device()}"),
            dist_group,
        )
        _allocator_map[dist_group] = (new_max_size, allocator)
    return allocator.create_tensor(shape, dtype)


def allreduce(
    tensor_in: SymmTensor,
    gamma: Optional[torch.Tensor] = None,
    eps: Optional[float] = None,
    smlimit: int = 0,
    cgasize: int = 0,
) -> SymmTensor:
    """Performs allreduce on the given SymmTensor using the best algorithm.

    Four modes:
     - standalone allreduce: no residual, no layernorm
     - first PROJ layer: layernorm fused, global residual in, internal residual out
     - middle layers: layernorm fused, internal residual in, internal residual out
     - last FC2 layer: no layernorm, internal residual in, no residual out
    """
    if tensor_in._allocator.dummy:
        return tensor_in
    if tensor_in._allocator.debug:
        print(f"UBX ALLREDUCE: {tensor_in.shape} gamma None:{gamma is None} eps None:{eps is None}")

    fuse_layernorm = gamma is not None and eps is not None
    internal_residual = tensor_in._allocator.residual
    residual_global = tensor_in._allocator.residual_global
    num_ranks = tensor_in._allocator.world_size
    hidden_size = (
        tensor_in.shape[-1]
        if fuse_layernorm or internal_residual is not None or residual_global is not None
        else tensor_in.numel() // num_ranks
    )
    assert (tensor_in.numel() // hidden_size) % tensor_in._allocator.nchunks == 0, \
        "Token count must be divisible by nchunks"

    num_tokens = (tensor_in.numel() // hidden_size) // tensor_in._allocator.nchunks
    myrank = tensor_in._allocator.myrank
    if residual_global is not None and tensor_in._allocator.current_chunk == 0 and (
        internal_residual is None
        or tensor_in._allocator.residual_tokens != num_tokens
        or tensor_in._allocator.residual_chunks != tensor_in._allocator.nchunks
    ):
        my_tokens = num_tokens // num_ranks
        extra_tokens = num_tokens % num_ranks
        first_token = myrank * my_tokens
        if myrank < extra_tokens:
            my_tokens += 1
            first_token += myrank
        else:
            first_token += extra_tokens
        if my_tokens == 0:
            my_tokens = 1  # avoid empty residual shard
        if tensor_in._allocator.residual is not None:
            del tensor_in._allocator.residual
        tensor_in._allocator.residual = torch.empty(
            my_tokens * tensor_in._allocator.nchunks * hidden_size,
            dtype=tensor_in.dtype, device=tensor_in.device,
        )
        tensor_in._allocator.residual_tokens = num_tokens
        tensor_in._allocator.residual_chunks = tensor_in._allocator.nchunks
        internal_residual = tensor_in._allocator.residual

    residual_in = residual_global if residual_global is not None else internal_residual

    residual_out = (
        internal_residual if fuse_layernorm else None
    )  # without layernorm new full residual is output of allreduce

    if tensor_in._allocator.current_chunk == tensor_in._allocator.nchunks - 1:
        tensor_in._allocator.residual_global = None

    if tensor_in.numel() * tensor_in.element_size() > AUTO_SWITCH_BYTES:
        return tensor_in._allocator.allreduce_mc(
            tensor_in, hidden_size, residual_in, residual_out, fuse_layernorm, gamma, eps,
            smlimit, cgasize,
        )
    else:
        return tensor_in._allocator.allreduce_lamport(
            tensor_in, hidden_size, residual_in, residual_out, fuse_layernorm, gamma, eps,
            smlimit, cgasize,
        )


def free_residual(tensor_in: SymmTensor):
    """Free the internal residual buffer associated with the tensor's allocator."""
    if tensor_in._allocator.residual is not None:
        del tensor_in._allocator.residual
        tensor_in._allocator.residual_tokens = 0
        tensor_in._allocator.residual = None


def restore(tensor: torch.Tensor, dist_group: torch.distributed.group) -> SymmTensor:
    """Restore a torch.Tensor to a SymmTensor if its data pointer is within the pool.

    If the tensor's data pointer falls within an allocator's pool, wraps it
    as a SymmTensor. Otherwise returns the original tensor unchanged.
    """
    if dist_group not in _allocator_map:
        return tensor
    (_, allocator) = _allocator_map[dist_group]
    if allocator is None:
        return tensor
    ptr = tensor.data_ptr()
    pool_ptr = allocator.pool_ptr
    pool_size = allocator.pool_size
    if pool_ptr <= ptr < pool_ptr + pool_size:
        offset = ptr - pool_ptr
        num_elements = tensor.numel()
        element_size = tensor.element_size()
        nbytes = element_size * num_elements
        if allocator.internal_pool.numel() < offset + nbytes:
            raise ValueError(
                f"Offset {offset} + {nbytes} bytes exceeds pool size "
                f"{allocator.internal_pool.numel()}"
            )
        symm_tensor = SymmTensor.__new__(
            SymmTensor,
            allocator.internal_pool,
            offset,
            tensor.shape,
            tensor.dtype,
            allocator,
            blocked_format=None,
            metadata_offset=None,
        )
        del tensor
        return symm_tensor
    else:
        return tensor


def compute_token_offsets(
    routing: torch.Tensor,
    experts_per_rank: int,
    myrank: int,
    nranks: int,
) -> Tuple[torch.Tensor, int]:
    """Compute slot assignments for a2av_token_bf16_mxfp8 from a routing matrix.

    Tokens are block-distributed: rank r owns global token rows
    [r * local_n, (r+1) * local_n] where local_n = ntokens // nranks.

    For each expert e, tokens routed there receive sequential slot indices
    (0, 1, 2, ...) ordered by global token index.  Expert
    e_local = e % experts_per_rank on the destination rank is assigned the
    slot sub-range [e_local * max_slots, (e_local+1) * max_slots), so
    multiple experts on the same destination rank never write to the same
    buffer offset.

    max_slots_per_expert is the global maximum across all experts — the
    symmetric buffer must have the same total_slots on every rank.

    Args:
        routing: [ntokens, total_experts] uint8 (or bool/int) on any device.
            Non-zero means token is routed to that expert.
            ntokens must be divisible by nranks.
        experts_per_rank: Number of experts hosted on each rank.
            total_experts must equal nranks * experts_per_rank.
        myrank: This rank's index (0-based).
        nranks: Total number of ranks (= world_size).

    Returns:
        token_offsets: [local_n, total_experts] int32 on the same device as
            routing.  Entry [local_t, e] is the destination slot index (>= 0)
            for this rank's token local_t at expert e, or -1 if not routed.
        max_tokens_per_rank: int.  Total buffer slots per rank needed for a
            collision-free symmetric allocation:
            experts_per_rank * max_tokens_per_expert.  Use this to size or
            validate the mxfp8 SymmTensor passed to a2av_token_bf16_mxfp8.
        tokens_per_expert: [total_experts] int32 on the same device as
            routing.  Number of tokens routed to each expert globally.
            For expert e on this rank, tokens occupy slots
            [e_local * max_slots, e_local * max_slots + tokens_per_expert[e])
            where e_local = e % experts_per_rank and
            max_slots = max_tokens_per_rank // experts_per_rank.
        expert_offsets: [experts_per_rank + 1] int32 on the same device as
            routing.  Start slot offset for each local expert on this rank.
            expert_offsets[e_local] is the first slot for local expert e_local,
            expert_offsets[e_local + 1] - expert_offsets[e_local] is the count.
            expert_offsets[-1] == total tokens received by this rank.
    """
    ntokens, total_experts = routing.shape
    assert ntokens % nranks == 0, (
        f"ntokens ({ntokens}) must be divisible by nranks ({nranks})"
    )
    assert total_experts == nranks * experts_per_rank, (
        f"total_experts ({total_experts}) must equal "
        f"nranks * experts_per_rank = {nranks} * {experts_per_rank}"
    )
    local_n = ntokens // nranks
    local_start = myrank * local_n

    routed = routing != 0  # bool

    # Exclusive prefix sum per column: sequential slot index for each (token, expert)
    cumsum = routed.int().cumsum(dim=0)       # [ntokens, total_experts], inclusive
    prefix = cumsum - routed.int()            # [ntokens, total_experts], exclusive

    # Width of each expert's sub-range = max tokens any single expert receives
    max_slots = int(cumsum[-1].max().item())  # scalar int

    # Sub-range base for each expert: expert e_local = e % experts_per_rank
    # gets slots [e_local * max_slots, (e_local+1) * max_slots)
    e_idx = torch.arange(total_experts, device=routing.device)
    slot_base = (e_idx % experts_per_rank * max_slots).to(torch.int32)  # [total_experts]

    local_routed = routed[local_start : local_start + local_n]           # [local_n, total_experts]
    local_prefix = prefix[local_start : local_start + local_n].to(torch.int32)

    token_offsets = torch.where(
        local_routed,
        slot_base.unsqueeze(0) + local_prefix,
        torch.full((), -1, dtype=torch.int32, device=routing.device),
    )  # [local_n, total_experts]

    tokens_per_expert = cumsum[-1].to(torch.int32)  # [total_experts]

    # Expert offsets for this rank's local experts: prefix sum of local expert counts
    my_expert_start = myrank * experts_per_rank
    local_counts = tokens_per_expert[my_expert_start:my_expert_start + experts_per_rank]
    expert_offsets = torch.zeros(experts_per_rank + 1, dtype=torch.int32, device=routing.device)
    expert_offsets[1:] = local_counts.cumsum(0)

    max_tokens_per_rank = experts_per_rank * max_slots
    return token_offsets, max_tokens_per_rank, tokens_per_expert, expert_offsets


def compute_combine_push_map(
    routing: torch.Tensor,
    experts_per_rank: int,
    myrank: int,
    nranks: int,
    *,
    max_slots_hint: Optional[int] = None,
    topk_max_hint: Optional[int] = None,
    slot_per_K_t_e: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Compute the inverse routing tables needed for PUSH-semantics combine.

    Companion to `compute_token_offsets`. While the forward map answers
    "where does this rank's token-t routed-to-expert-e land in the dispatch
    output?", the inverse map answers "for slot s in MY rank's dispatch
    output, who originally sent that token (and as which top-K position)?".

    PUSH-semantics combine uses this to read its local FFN expert outputs
    and PUSH each one into the originating rank's combine buffer at the
    correct (origin_token, k_idx) position.  Compare PULL semantics
    (combine_bf16_bf16_lamport) which has each receiver read peer bufs at
    forward-map slot indices — that approach has a known race with the
    triple-buffer re-poison protocol; PUSH avoids the race by mirroring
    alltoall_lamport's read-own-buf pattern.

    Args:
        routing: [ntokens, total_experts] uint8/bool/int.  Same as for
            compute_token_offsets — every rank must see the same matrix.
        experts_per_rank, myrank, nranks: see compute_token_offsets.

    Returns:
        inverse_map: [max_tokens_per_rank, 4] int32 on the same device as
            routing.  Entry [s] is (origin_rank, origin_local_token, k_idx,
            valid) for a slot s in MY rank's dispatch output:
              - origin_rank: the rank whose token created this slot.
              - origin_local_token: that rank's local token index in
                [0, local_n).
              - k_idx: the position of MY expert among the origin token's
                top-K experts (0-indexed, sorted by expert id ascending).
              - valid: 1 if the slot is populated by some source, 0 if
                unused (slot index falls in the per-expert padding region
                between [tokens_per_expert[e], max_slots)).
            Shape padding: rows for unused slots have valid=0; the kernel
            uses this to skip pushing.
        topk_idx: [local_n, topk_max] int32 on the same device.  Entry
            [t, k] is the global expert id of the k-th routed expert for
            MY rank's token t (sorted ascending), or -1 if t routes to
            fewer than k+1 experts.  topk_max = max top-K count over all
            tokens (often equal to the user's topk arg, but we infer from
            the routing matrix to avoid a separate parameter).  The
            combine kernel uses this to look up gate_weights[t, expert]
            for the k-th contribution received in Phase 2.
        max_tokens_per_rank: identical to the same return from
            compute_token_offsets — the symmetric buffer dimension.
    """
    ntokens, total_experts = routing.shape
    assert total_experts == nranks * experts_per_rank
    assert ntokens % nranks == 0
    local_n = ntokens // nranks
    device = routing.device

    routed = (routing != 0).int()                              # [ntokens, te] bool→int
    if slot_per_K_t_e is None:
        cumsum = routed.cumsum(dim=0)                          # [ntokens, te], inclusive
        prefix = cumsum - routed                               # exclusive
        if max_slots_hint is None:
            max_slots = int(cumsum[-1].max().item())
        else:
            max_slots = int(max_slots_hint)
    else:
        # slot_per_K_t_e is [nranks, local_n, total_experts]; max_slots is provided
        # via hint (drop_and_pad → max_slots = capacity).
        assert max_slots_hint is not None, (
            "slot_per_K_t_e requires max_slots_hint (= capacity under drop_and_pad)"
        )
        max_slots = int(max_slots_hint)
        prefix = None  # unused on fast path
    max_tokens_per_rank = experts_per_rank * max_slots

    # ---- inverse_map ---------------------------------------------------------
    # Closed-form vectorised fill: for every (origin_K, origin_t, e_local) in my
    # owned-expert slice, compute the slot and write (origin_K, origin_t, k_idx,
    # valid) into inverse_map. Invalid (unrouted) entries write into an extra
    # "junk" row at index `max_tokens_per_rank`, which is discarded at the end.
    # No nonzero(), no .item() — no host syncs.
    my_e_start = myrank * experts_per_rank
    my_e_end = my_e_start + experts_per_rank
    my_routed = routed[:, my_e_start:my_e_end]                 # [ntokens, ne_local] int32 0/1
    my_routed_bool = my_routed.bool()
    ne_local = experts_per_rank

    # k_idx along the expert axis: # of routed experts with smaller id for the
    # SAME origin_t. rowwise_prefix is needed for both inverse_map (k_idx of
    # peers' tokens) and topk_idx (k_idx of my own tokens). Compute once.
    rowwise_prefix = routed.cumsum(dim=1) - routed             # [ntokens, te]

    if slot_per_K_t_e is not None:
        # Fast path: slot index already computed by caller. The caller may
        # pass either [nranks, local_n, total_experts] (legacy) or — to save
        # memory traffic — the pre-sliced [nranks, local_n, ne_local] for our
        # owned-expert columns only. Both shapes resolve to [ntokens, ne_local]
        # after reshape.
        if slot_per_K_t_e.shape[-1] == total_experts:
            sliced = slot_per_K_t_e[:, :, my_e_start:my_e_end]
        else:
            assert slot_per_K_t_e.shape[-1] == ne_local, (
                f"slot_per_K_t_e last dim must be total_experts ({total_experts}) "
                f"or num_local_experts ({ne_local}); got {slot_per_K_t_e.shape[-1]}"
            )
            sliced = slot_per_K_t_e
        my_slot_grid = sliced.reshape(ntokens, ne_local).to(torch.int64)
    else:
        e_local_grid = (
            torch.arange(ne_local, device=device, dtype=torch.int64)
            .view(1, -1).expand(ntokens, -1)
        )
        my_slot_grid = (
            e_local_grid * max_slots + prefix[:, my_e_start:my_e_end].to(torch.int64)
        )                                                      # [ntokens, ne_local] int64

    t_grid = (
        torch.arange(ntokens, device=device, dtype=torch.int64)
        .view(-1, 1).expand(-1, ne_local)
    )
    origin_rank_grid = (t_grid // local_n).to(torch.int32)
    origin_token_grid = (t_grid % local_n).to(torch.int32)
    k_idx_grid = rowwise_prefix[:, my_e_start:my_e_end].to(torch.int32)

    # Route invalid entries to a junk row that's discarded after the scatter.
    junk_slot = torch.full((), max_tokens_per_rank, dtype=torch.int64, device=device)
    slot_safe = torch.where(my_routed_bool, my_slot_grid, junk_slot)

    inverse_map_ext = torch.zeros((max_tokens_per_rank + 1, 4),
                                  dtype=torch.int32, device=device)
    slot_flat = slot_safe.flatten()
    # UBX_PUSH_MAP_DIAG=1: print min/max of slot_flat per rank with the
    # bound (max_tokens_per_rank). Helps localize OOB asserts in
    # inverse_map_ext[slot_flat, ...] without needing CUDA_LAUNCH_BLOCKING.
    import os as _os_pmd
    if _os_pmd.environ.get("UBX_PUSH_MAP_DIAG", "0") == "1":
        import torch as _t
        _t.cuda.synchronize()
        _smin = int(slot_flat.min().item())
        _smax = int(slot_flat.max().item())
        _rank = _t.distributed.get_rank() if _t.distributed.is_initialized() else 0
        print(f"[PUSH_MAP_DIAG r{_rank}] slot_flat: min={_smin} max={_smax} "
              f"bound=max_tokens_per_rank={max_tokens_per_rank} "
              f"OOB={1 if _smax > max_tokens_per_rank else 0} "
              f"my_routed.sum={int(my_routed.sum().item())} "
              f"my_e=[{my_e_start}:{my_e_end})",
              flush=True)
    inverse_map_ext[slot_flat, 0] = origin_rank_grid.flatten()
    inverse_map_ext[slot_flat, 1] = origin_token_grid.flatten()
    inverse_map_ext[slot_flat, 2] = k_idx_grid.flatten()
    inverse_map_ext[slot_flat, 3] = my_routed.flatten()
    inverse_map = inverse_map_ext[:max_tokens_per_rank].contiguous()

    # ---- topk_idx for MY rank's tokens --------------------------------------
    # For each routed (t, e) on this rank, k_idx = # routed entries with
    # smaller expert id for the same t = exclusive per-row cumsum of `routed`
    # at column e.
    local_routed_bool = (
        routed[myrank * local_n : (myrank + 1) * local_n] != 0
    )                                                               # [local_n, te]
    local_per_row_excl = (
        local_routed_bool.int().cumsum(dim=1) - local_routed_bool.int()
    )                                                               # [local_n, te]
    # NOTE: topk_max must be IDENTICAL on every rank — peers push tokens to
    # this rank's dest_buf using their kernel's `topk_max` as the per-token
    # stride. If A's topk_max differs from B's, A's writes land at wrong
    # offsets in B's dest_buf → OOB. Compute the GLOBAL max over ALL tokens
    # (the routing matrix is the same on every rank, so the max is too).
    if topk_max_hint is None:
        global_topk_counts = (routing != 0).int().sum(dim=1)        # [ntokens]
        topk_max = int(global_topk_counts.max().item()) if global_topk_counts.numel() else 0
    else:
        topk_max = int(topk_max_hint)

    topk_idx = torch.full((local_n, max(topk_max, 1)), -1,
                          dtype=torch.int32, device=device)
    if topk_max > 0:
        # Closed-form vectorised fill: for each (t, e) on my rank, write
        # the global expert id at column k_idx_local; unrouted entries
        # write to a junk column that's sliced off.
        e_id_grid = (
            torch.arange(total_experts, device=device, dtype=torch.int32)
            .view(1, -1).expand(local_n, -1)
        )
        junk_col = torch.full((), topk_max, dtype=torch.int64, device=device)
        k_safe = torch.where(local_routed_bool, local_per_row_excl, junk_col.to(torch.int32)).to(torch.int64)
        e_id_safe = torch.where(local_routed_bool, e_id_grid, torch.full((), -1, dtype=torch.int32, device=device))

        topk_idx_ext = torch.full((local_n, topk_max + 1), -1,
                                  dtype=torch.int32, device=device)
        row_idx = (
            torch.arange(local_n, device=device, dtype=torch.int64)
            .view(-1, 1).expand(-1, total_experts)
        )
        topk_idx_ext[row_idx.flatten(), k_safe.flatten()] = e_id_safe.flatten()
        topk_idx = topk_idx_ext[:, :topk_max].contiguous()

    return inverse_map, topk_idx, max_tokens_per_rank


def compute_dispatch_topk_map(
    routing: torch.Tensor,
    token_offsets: torch.Tensor,
    experts_per_rank: int,
    myrank: int,
    nranks: int,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Per-token LUT of (expert_id, dest_slot) for the K experts each token routes to.

    Companion to ``compute_token_offsets`` for the *top-K-indexed* fused
    dispatch kernel (``ubx_a2av_token_bf16_bf16_topk``).  The base fused
    kernel iterates every (token, global_expert) pair — for top-K=6 out
    of total_experts=128 that's 95 % dead checks.  This LUT lets the
    kernel iterate only the K routed entries per token.

    Args:
        routing: ``[ntokens, total_experts]`` uint8/bool/int. Non-zero = routed.
            Must match what was passed to ``compute_token_offsets``.
        token_offsets: ``[ntokens, total_experts]`` int32 from
            ``compute_token_offsets``.  ``token_offsets[t, e] >= 0`` is the
            destination slot on the rank hosting expert ``e``;  ``-1`` means
            token ``t`` is not routed to expert ``e``.
        experts_per_rank, myrank, nranks: same as compute_token_offsets.

    Returns:
        topk_expert: ``[local_n, topk_max]`` int32 (on the same device as
            ``routing``).  ``topk_expert[t, k]`` is the global expert id of
            ``t``'s ``k``-th routed expert sorted ascending, or ``-1`` for
            unused slots.  Only this rank's local tokens are included.
        topk_slot:   ``[local_n, topk_max]`` int32.  ``topk_slot[t, k]`` is
            ``token_offsets[t, topk_expert[t,k]]`` for valid entries, ``-1``
            otherwise.  By construction has the same ``-1`` mask as
            ``topk_expert``.
        topk_max:    ``int`` — global max top-K count over all tokens in
            ``routing``.  Same value all ranks compute (deterministic from
            the global routing matrix).
    """
    ntokens, total_experts = routing.shape
    assert total_experts == nranks * experts_per_rank
    assert ntokens % nranks == 0
    local_n = ntokens // nranks
    # `token_offsets` from `compute_token_offsets` is already per-rank
    # shape [local_n, total_experts] (NOT global). Validate that.
    assert token_offsets.shape == (local_n, total_experts), (
        f"token_offsets must be [{local_n}, {total_experts}] (local slice), "
        f"got {list(token_offsets.shape)}"
    )
    device = routing.device

    # Slice routing to this rank's local tokens. token_offsets is already
    # the local slice (per compute_token_offsets contract).
    local_routing = routing[myrank * local_n:(myrank + 1) * local_n]
    local_offsets = token_offsets

    local_routed = (local_routing != 0)                                # [local_n, total_experts]
    # k_idx within each token row: exclusive cumsum along expert axis.
    routed_int = local_routed.int()
    excl_cumsum = routed_int.cumsum(dim=1) - routed_int                # [local_n, total_experts]

    # Global topk_max from the FULL routing matrix — must be identical on
    # every rank so the kernel's per-token stride is consistent.  (Mirrors
    # the same fix in compute_combine_push_map.)
    topk_max = int((routing != 0).int().sum(dim=1).max().item())
    if topk_max == 0:
        # Degenerate case: nothing routed. Still allocate (local_n, 1) so the
        # kernel can early-exit cleanly on the -1 sentinel.
        topk_max = 1

    topk_expert = torch.full((local_n, topk_max), -1,
                             dtype=torch.int32, device=device)
    topk_slot   = torch.full((local_n, topk_max), -1,
                             dtype=torch.int32, device=device)
    # Scatter routed (t, e) -> (t, k_idx)
    t_idx, e_idx = local_routed.nonzero(as_tuple=True)
    k_idx        = excl_cumsum[t_idx, e_idx].to(torch.int64)
    topk_expert[t_idx, k_idx] = e_idx.to(torch.int32)
    topk_slot  [t_idx, k_idx] = local_offsets[t_idx, e_idx]

    return topk_expert, topk_slot, topk_max


def mem_stats():
    """Print memory statistics for all active allocators."""
    for dist_group, (_, allocator) in _allocator_map.items():
        if allocator is not None:
            print(f"Rank {allocator.myrank} Graph pool used size: "
                  f"{sum(size for _, size, _ in allocator.allocated[True]) / 1024 / 1024} MB")
            print(f"Rank {allocator.myrank} Non-graph pool used size: "
                  f"{sum(size for _, size, _ in allocator.allocated[False]) / 1024 / 1024} MB")
