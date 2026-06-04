# contrib/custom_algos/

Reference custom collective kernels built on the NCCL Device API.

## Contents

| Directory | Description |
|-----------|-------------|
| [`allreduce/`](allreduce/) | Custom AllReduce kernels using NVLink multicast (MC) on Hopper/Blackwell GPUs |
| [`alltoall/`](alltoall/) | Custom AllToAll kernels for GPUs within an NVLink island (LSA team) |

## Maintainers

| GitHub | Areas |
|--------|------|
| @akhillanger | All |
