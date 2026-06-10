"""UB-X MoE token combine — round-trip with an identity expert.

Pairs with `examples/moe_dispatch.py`:

  1. Build a deterministic global routing matrix (top-K).
  2. Dispatch this rank's tokens to remote experts via the bf16 wire
     (`a2av_token_bf16_bf16`).
  3. Skip the FFN — pretend each expert is the identity, so the dispatch
     output is also the "expert output".
  4. Combine the expert outputs back to originating ranks via
     `combine_bf16_bf16` (PULL-barrier).

With identity experts and unweighted combine, the reconstructed value
for each local token equals (number of experts that token was routed
to) × original token. We validate against that closed form.

Launch (2 or more GPUs on a single node):

    torchrun --nproc-per-node=2 examples/moe_combine.py
"""

import os
import torch
import torch.distributed as dist


def main():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")
    device = torch.device(f"cuda:{local_rank}")

    from ubx import SymmAllocator, compute_token_offsets

    hidden = 128                                  # must be a multiple of 32
    experts_per_rank = 2
    total_experts = world_size * experts_per_rank
    local_ntokens = 8
    global_ntokens = local_ntokens * world_size
    topk = 2                                      # each token to 2 experts

    # 1. Deterministic top-K routing (same matrix on every rank — the API
    #    expects ranks to agree on slot assignments without an extra broadcast).
    g = torch.Generator(device=device).manual_seed(0)
    routing = torch.zeros(global_ntokens, total_experts,
                          dtype=torch.uint8, device=device)
    for t in range(global_ntokens):
        chosen = torch.randperm(total_experts, generator=g, device=device)[:topk]
        routing[t, chosen] = 1

    # 2. Per-rank slot table for dispatch and PULL combine.
    token_offsets, max_tokens_per_rank, _, _ = compute_token_offsets(
        routing, experts_per_rank, myrank=rank, nranks=world_size,
    )

    # 3. Allocate the symmetric pool — needs to fit one bf16 dispatch buffer
    #    plus one bf16 combine temp buffer of [max_tokens_per_rank, hidden],
    #    with headroom.
    pool_bytes = max(64 * 1024 * 1024, 8 * max_tokens_per_rank * hidden * 2)
    allocator = SymmAllocator(pool_bytes, device, dist.group.WORLD)

    # This rank's input tokens.
    torch.manual_seed(100 + rank)
    tokens_bf16 = torch.randn(local_ntokens, hidden,
                              dtype=torch.bfloat16, device=device)

    # 4. Dispatch (bf16 wire — `a2av_token_bf16_bf16` writes raw bf16, no quantization).
    dispatch_out = allocator.create_tensor(
        [max_tokens_per_rank, hidden], torch.bfloat16,
    )
    dispatch_out = allocator.a2av_token_bf16_bf16(
        tokens_bf16, token_offsets, experts_per_rank, dispatch_out,
    )
    torch.cuda.synchronize()

    # 5. Identity expert: the dispatch output IS the FFN output. Hand it
    #    to combine as a regular torch.Tensor (combine doesn't require
    #    a SymmTensor input).
    expert_outputs = dispatch_out.detach().clone()

    combined = allocator.combine_bf16_bf16(
        expert_outputs,
        token_offsets,
        experts_per_rank,
        max_tokens_per_rank,
        gate_weights=None,           # unweighted sum
    )
    torch.cuda.synchronize()

    # 6. Reference: with identity experts and gate_weights=None,
    #    combined[t] = (#experts t is routed to) × tokens_bf16[t].
    src_start = rank * local_ntokens
    fanout = routing[src_start : src_start + local_ntokens].sum(dim=1).float()  # [local_ntokens]
    ref = tokens_bf16.float() * fanout.unsqueeze(1)

    torch.testing.assert_close(combined.float(), ref, atol=0.02, rtol=0.05)
    if rank == 0:
        print(f"OK: combine_bf16_bf16 round-trip on {world_size} GPUs "
              f"(global_ntokens={global_ntokens}, hidden={hidden}, "
              f"experts_per_rank={experts_per_rank}, topk={topk}).")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
