"""UB-X MoE token dispatch with bf16 → mxfp8 quantization.

Shows the full a2av_token API:
  1. Build a routing matrix that says which experts each token goes to.
  2. Convert it to per-rank slot offsets via `compute_token_offsets`.
  3. Allocate a symmetric mxfp8 buffer big enough for incoming tokens.
  4. Call `a2av_token_bf16_mxfp8` — one kernel handles routing + per-block
     E8M0 quantization to fp8 e4m3.

The example uses a deterministic routing matrix (same seed on every rank)
so all ranks agree on slot assignments without an extra broadcast.

Launch (2 or more GPUs on a single node):

    torchrun --nproc-per-node=2 examples/moe_dispatch.py
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

    # Workload shape — small enough to print, real workloads are much bigger.
    hidden = 128                                 # must be a multiple of 32
    experts_per_rank = 2
    total_experts = world_size * experts_per_rank
    local_ntokens = 8                            # tokens this rank dispatches
    global_ntokens = local_ntokens * world_size

    # 1. Build a deterministic global routing matrix (every token routed to
    #    one expert, seed shared across ranks so all ranks compute the same
    #    slot assignments).
    torch.manual_seed(0)
    routing = torch.zeros(global_ntokens, total_experts, dtype=torch.uint8, device=device)
    expert_ids = torch.randint(0, total_experts, (global_ntokens,), device=device)
    routing[torch.arange(global_ntokens, device=device), expert_ids] = 1

    # 2. Slot assignments for this rank's local tokens.
    token_offsets, max_tokens_per_rank, _, _ = compute_token_offsets(
        routing, experts_per_rank, myrank=rank, nranks=world_size,
    )

    # 3. Allocate input tokens (this rank's slice) and a symmetric mxfp8
    #    output buffer big enough for the worst-case fan-in.
    pool_bytes = 16 * 1024 * 1024
    allocator = SymmAllocator(pool_bytes, device, dist.group.WORLD)

    torch.manual_seed(100 + rank)
    tokens_bf16 = torch.randn(local_ntokens, hidden, dtype=torch.bfloat16, device=device)

    output = allocator.create_tensor(
        [max_tokens_per_rank, hidden], torch.float8_e4m3fn, blocked="mxfp8",
    )

    # 4. Dispatch: routing + bf16 → mxfp8 quantization in one kernel.
    output = allocator.a2av_token_bf16_mxfp8(
        tokens_bf16, token_offsets, experts_per_rank, output,
    )
    torch.cuda.synchronize()

    if rank == 0:
        print(f"OK: dispatched {global_ntokens} tokens across {world_size} ranks "
              f"({experts_per_rank} experts/rank, hidden={hidden}). "
              f"Output buffer: {output.shape} mxfp8.")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
