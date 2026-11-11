# nccl4rust all-reduce example

A minimal multi-process, multi-node NCCL program in Rust. Each rank fills a
buffer with `rank + 1`, enqueues a sum all-reduce on a CUDA stream through the
`nccl` crate, and checks that every element equals `1 + 2 + ... + world`.

The example uses `nccl` for the unique ID, the communicator, and the
all-reduce. It uses [`cudarc`](https://crates.io/crates/cudarc) only for the
CUDA context, stream, and device buffers; `nccl` takes the raw stream handle
and device pointers.

## Build

Prepare matching NCCL headers and library as described in the
[nccl4rust README](../../README.md#prepare-matching-nccl-headers-and-library),
then:

```bash
export NCCL_INCLUDE_DIR=/path/to/nccl/include
export NCCL_LIB_DIR=/path/to/nccl/lib
export CUDA_HOME=/path/to/cuda
cargo build --release
```

## Run

Rank and world size come from Slurm (`SLURM_PROCID`, `SLURM_NTASKS`), and each
rank uses the GPU at index `SLURM_LOCALID`. Rank 0 writes the NCCL unique ID to
`NCCL_ID_DIR/nccl-id.$SLURM_JOB_ID`, so `NCCL_ID_DIR` must be on a filesystem
that every node can see:

```bash
export LD_LIBRARY_PATH="$NCCL_LIB_DIR:$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export NCCL_ID_DIR=/shared/scratch
srun -N2 --ntasks-per-node=1 target/release/nccl4rust-allreduce
```

Each rank prints a line such as:

```text
[node01] rank 0/2: sum = 3 (expected 3) OK
```
