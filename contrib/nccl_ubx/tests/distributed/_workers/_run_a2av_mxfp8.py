"""Worker script for UB-X a2av token bf16→mxfp8 dispatch tests.

Supports two modes selected by --mode:

  manual  -- 2 tokens (all-ones, all-twos), each routed to a single expert on
             the matching rank.  Prints per-rank, per-slot token summaries and
             asserts the dequantized values match the input.

  random  -- Random bf16 tokens, each routed to exactly one randomly chosen
             expert.  Verifies that every received token dequantizes within the
             mxfp8 tolerance of the original bf16 input.

Usage (2 GPUs):
    torchrun --nproc_per_node=2 _run_a2av_mxfp8.py --mode manual
    torchrun --nproc_per_node=2 _run_a2av_mxfp8.py --mode random --ntokens 32 --hidden 128
"""

import argparse
import math
import os
import sys
import torch
import torch.distributed as dist

os.environ.setdefault("UBX_GRAPH_POOL_SHARE", "0.1")


# --------------------------------------------------------------------------- #
# Distributed init                                                             #
# --------------------------------------------------------------------------- #

def get_rank_info():
    if "RANK" in os.environ:
        rank       = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
    elif "OMPI_COMM_WORLD_RANK" in os.environ:
        rank       = int(os.environ["OMPI_COMM_WORLD_RANK"])
        world_size = int(os.environ["OMPI_COMM_WORLD_SIZE"])
        local_rank = int(os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", rank))
    elif "SLURM_PROCID" in os.environ:
        rank       = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ["SLURM_NTASKS"])
        local_rank = int(os.environ.get("SLURM_LOCALID", rank))
    else:
        raise RuntimeError("Cannot determine rank")
    return rank, world_size, local_rank


def init_distributed():
    rank, world_size, local_rank = get_rank_info()
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://",
                                world_size=world_size, rank=rank)
    return rank, world_size, local_rank


# --------------------------------------------------------------------------- #
# Dequantize helper                                                            #
# --------------------------------------------------------------------------- #

def dequantize_from_symm_tensor(output, allocator, blocks_per_token: int) -> torch.Tensor:
    """Dequantize a mxfp8 SymmTensor back to float32.

    Reads fp8 bytes and E8M0 scale bytes directly from the symmetric pool
    via the SymmTensor's data_ptr / metadata_ptr.

    Returns float32 tensor of shape [nslots, hidden].
    """
    pool     = allocator.internal_pool   # uint8 view of pool
    pool_ptr = allocator.pool_ptr
    nslots   = output.shape[0]
    hidden   = output.shape[1]

    data_offset  = output.data_ptr()   - pool_ptr
    scale_offset = output.metadata_ptr - pool_ptr

    fp8_bytes  = pool[data_offset  : data_offset  + nslots * hidden          ].view(nslots, hidden)
    scale_bytes = pool[scale_offset : scale_offset + nslots * blocks_per_token].view(nslots, blocks_per_token)

    result = torch.empty(nslots, hidden, dtype=torch.float32, device=pool.device)
    for s in range(nslots):
        for b in range(blocks_per_token):
            e8m0  = scale_bytes[s, b].item()
            scale = 2.0 ** (e8m0 - 127)
            raw   = fp8_bytes[s, b * 32 : (b + 1) * 32]
            result[s, b * 32 : (b + 1) * 32] = raw.view(torch.float8_e4m3fn).float() * scale
    return result


# --------------------------------------------------------------------------- #
# Manual test                                                                  #
# --------------------------------------------------------------------------- #

def run_manual_test(async_dispatch=False):
    """2 ranks, 1 expert per rank, 1 token each (all-ones / all-twos).

    Rank 0 sends token 0 (all 1.0) to expert 0 (on rank 0, slot 0).
    Rank 1 sends token 1 (all 2.0) to expert 1 (on rank 1, slot 0).

    After dispatch each rank prints the dequantized values of its received
    token, then asserts they match the expected constant.

    Scale derivation (E8M0, ceil rule):
        amax=1.0: ceil(log2(1/448))=ceil(-8.807)=-8 → e8m0=119 → scale=1/256
                  quant=256 → fp8=0x78 → dequant=256*(1/256)=1.0
        amax=2.0: ceil(log2(2/448))=ceil(-7.807)=-7 → e8m0=120 → scale=1/128
                  quant=256 → fp8=0x78 → dequant=256*(1/128)=2.0
    """
    rank, world_size, local_rank = init_distributed()
    assert world_size == 2, "manual test requires exactly 2 ranks"
    device = torch.device(f"cuda:{local_rank}")

    hidden           = 32
    blocks_per_token = 1
    experts_per_rank = 1
    total_experts    = world_size * experts_per_rank
    ntokens          = 1

    fill_val    = float(rank + 1)
    tokens_bf16 = torch.full((ntokens, hidden), fill_val,
                             dtype=torch.bfloat16, device=device)

    token_offsets = torch.full((ntokens, total_experts), -1,
                               dtype=torch.int32, device=device)
    token_offsets[0, rank] = 0   # each rank's token goes to its own expert, slot 0

    pool_bytes = 8 * 1024 * 1024
    from ubx import SymmAllocator
    allocator = SymmAllocator(pool_bytes, device, dist.group.WORLD)

    max_slots_per_expert = 1
    total_slots = experts_per_rank * max_slots_per_expert
    output = allocator.create_tensor(
        [total_slots, hidden], torch.float8_e4m3fn, blocked='mxfp8')

    sync_mode = "async+wait" if async_dispatch else "sync"
    print(f"[rank{rank}] manual test: sending token fill={fill_val} ({sync_mode})", flush=True)

    output = allocator.a2av_token_bf16_mxfp8(
        tokens_bf16, token_offsets, experts_per_rank, output,
        sync=not async_dispatch,
    )
    if async_dispatch:
        allocator.a2av_wait()
    torch.cuda.synchronize()

    dequant = dequantize_from_symm_tensor(output, allocator, blocks_per_token)

    scale_offset = output.metadata_ptr - allocator.pool_ptr
    scale_byte   = allocator.internal_pool[scale_offset].item()

    print(f"[rank{rank}] received slot 0, first 8 dequant values: "
          f"{dequant[0, :8].cpu().tolist()}", flush=True)
    print(f"[rank{rank}] e8m0 scale byte = {scale_byte}, "
          f"scale_f = {2.0**(scale_byte-127):.6g}", flush=True)

    try:
        torch.testing.assert_close(dequant, torch.full_like(dequant, fill_val),
                                   atol=0.005, rtol=0.01)
        print(f"PASS rank={rank} mode=manual fill={fill_val}", flush=True)
    except AssertionError as e:
        print(f"FAIL rank={rank} mode=manual: {e}", flush=True)
        sys.exit(1)

    dist.destroy_process_group()


# --------------------------------------------------------------------------- #
# Random test                                                                  #
# --------------------------------------------------------------------------- #

def run_random_test(args):
    """Random bf16 tokens, each routed to exactly one random expert.

    Uses compute_token_offsets for collision-free slot assignment across ranks.
    Global tokens are block-distributed: rank r owns global rows
    [r*local_n, (r+1)*local_n].  Each rank generates its own local tokens,
    and all ranks allgather tokens for cross-rank verification.
    """
    rank, world_size, local_rank = init_distributed()
    device = torch.device(f"cuda:{local_rank}")

    hidden           = args.hidden
    local_ntokens    = args.ntokens
    experts_per_rank = args.experts_per_rank
    total_experts    = world_size * experts_per_rank
    global_ntokens   = local_ntokens * world_size
    blocks_per_token = hidden // 32
    assert hidden % 32 == 0, "hidden must be a multiple of 32"

    # Each rank generates its own local tokens with a per-rank seed.
    torch.manual_seed(args.seed + rank)
    tokens_bf16 = torch.randn(local_ntokens, hidden, dtype=torch.bfloat16, device=device)

    # Allgather tokens so every rank can verify received data against source.
    all_tokens_list = [torch.empty_like(tokens_bf16) for _ in range(world_size)]
    dist.all_gather(all_tokens_list, tokens_bf16)
    all_tokens = torch.cat(all_tokens_list, dim=0)  # [global_ntokens, hidden]

    # Build global routing matrix: each token routed to exactly one expert.
    # Deterministic from rank 0 seed so all ranks agree.
    torch.manual_seed(args.seed)
    routing = torch.zeros(global_ntokens, total_experts, dtype=torch.uint8, device=device)
    expert_ids = torch.randint(0, total_experts, (global_ntokens,), device=device)
    routing[torch.arange(global_ntokens, device=device), expert_ids] = 1

    # compute_token_offsets returns this rank's local offsets with globally
    # unique slots (collision-free via prefix sum).
    from ubx import compute_token_offsets
    token_offsets, max_tokens_per_rank, _, _ = compute_token_offsets(
        routing, experts_per_rank, rank, world_size)

    pool_bytes = 16 * 1024 * 1024
    from ubx import SymmAllocator
    allocator = SymmAllocator(pool_bytes, device, dist.group.WORLD)

    total_slots = max_tokens_per_rank
    output = allocator.create_tensor(
        [total_slots, hidden], torch.float8_e4m3fn, blocked='mxfp8')

    async_dispatch = args.async_dispatch
    sync_mode = "async+wait" if async_dispatch else "sync"
    print(f"[rank{rank}] random test: local_ntokens={local_ntokens} hidden={hidden} "
          f"total_experts={total_experts} max_tokens_per_rank={max_tokens_per_rank} "
          f"({sync_mode})", flush=True)

    output = allocator.a2av_token_bf16_mxfp8(
        tokens_bf16, token_offsets, experts_per_rank, output,
        sync=not async_dispatch,
    )
    if async_dispatch:
        allocator.a2av_wait()
    torch.cuda.synchronize()

    dequant = dequantize_from_symm_tensor(output, allocator, blocks_per_token)

    my_expert_start = rank * experts_per_rank
    my_expert_end   = my_expert_start + experts_per_rank

    # Verify: check all global tokens routed to this rank's experts.
    # token_offsets only covers this rank's local tokens, so we need to
    # recompute slots for ALL source ranks to find what landed here.
    errors = []
    for src_rank in range(world_size):
        src_offsets, _, _, _ = compute_token_offsets(
            routing, experts_per_rank, src_rank, world_size)
        src_start = src_rank * local_ntokens
        for local_t in range(local_ntokens):
            global_t = src_start + local_t
            e = expert_ids[global_t].item()
            if not (my_expert_start <= e < my_expert_end):
                continue
            slot = src_offsets[local_t, e].item()
            if slot < 0:
                continue
            orig = all_tokens[global_t].float().cpu()
            got  = dequant[slot].cpu()

            scale_offset = output.metadata_ptr - allocator.pool_ptr
            blk_start    = (slot * blocks_per_token)
            e8m0_vals    = allocator.internal_pool[
                scale_offset + blk_start : scale_offset + blk_start + blocks_per_token
            ].float().cpu()
            max_scale = float((2.0 ** (e8m0_vals - 127)).max())
            # Worst-case fp8 e4m3 error: 0.5 * max_ulp * scale.
            # At the top of e4m3 range (448), ULP=32, so error ≤ 16 * scale.
            atol = max_scale * 16.0 + 1e-5

            if not torch.allclose(got, orig, atol=atol, rtol=0.0):
                max_err = (got - orig).abs().max().item()
                errors.append(
                    f"global_token {global_t} expert {e} slot {slot}: "
                    f"max_err={max_err:.4g} atol={atol:.4g}"
                )

    if errors:
        for err in errors:
            print(f"FAIL rank={rank} mode=random: {err}", flush=True)
        sys.exit(1)

    print(f"PASS rank={rank} mode=random local_ntokens={local_ntokens} hidden={hidden}", flush=True)
    dist.destroy_process_group()


# --------------------------------------------------------------------------- #
# Entry point                                                                  #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=["manual", "random"])
    parser.add_argument("--ntokens",           type=int, default=16)
    parser.add_argument("--hidden",            type=int, default=128,
                        help="Token hidden size (multiple of 32)")
    parser.add_argument("--experts_per_rank",  type=int, default=2)
    parser.add_argument("--seed",              type=int, default=42)
    parser.add_argument("--async_dispatch",    action="store_true",
                        help="Use async dispatch (sync=False) + a2av_wait()")
    args = parser.parse_args()

    if args.mode == "manual":
        run_manual_test(async_dispatch=args.async_dispatch)
    else:
        run_random_test(args)
