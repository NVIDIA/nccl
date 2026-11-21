# contrib/custom_algos/

Reference custom collective kernels built on the NCCL Device API.

## Contents

| Directory | Description |
|-----------|-------------|
| [`allreduce/`](allreduce/) | Custom AllReduce kernels using NVLink multicast (MC) on Hopper/Blackwell GPUs, and one-shot push kernels over LSA unicast pointers for PCIe P2P GPUs |
| [`alltoall/`](alltoall/) | Custom AllToAll kernels for GPUs within an NVLink island (LSA team) |

## Maintainers

| GitHub | Areas |
|--------|------|
| @akhillanger | All |
| @Ishannaik | `allreduce/` LSA push kernels (`oneshot_lsa_push*`, `lsa_push_reduce.cuh`) |
