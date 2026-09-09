# nccl4rust multi-GPU runtime smoke

This standalone package exercises the Rust host wrappers and the CUDA-Oxide
device boundary on two GPUs in one process. Each rank runs on its own host
thread and current CUDA device. The smoke:

1. initializes a two-rank communicator and verifies a typed host `f32`
   all-reduce;
2. allocates and registers symmetric source and destination windows;
3. creates a public NCCL device communicator with two LSA barriers and copies
   its image to each GPU;
4. validates scalar queries, typed world/LSA/rail teams, and rank translation;
5. validates local, LSA, world, and typed-team self-pointer mappings at two
   window offsets while leaving multimem disabled;
6. runs a partitioned 17-element LSA reduce-sum-copy twice, reusing the barrier
   indices and checking the non-16-byte tail and surrounding guards; and
7. destroys the device communicator before deregistering the windows and
   finalizing the host communicator.

Both visible GPUs must belong to one two-rank LSA team. Runtime multimem is not
covered by this smoke.

Build the matching NCCL library, shim, and CUDA-Oxide cubin first. Then run on
a node with at least two `sm_90` GPUs:

```bash
export CUDA_HOME=/path/to/cuda
export NCCL_INCLUDE_DIR=/path/to/nccl/include
export NCCL_LIB_DIR=/path/to/nccl/lib
export LD_LIBRARY_PATH="$NCCL_LIB_DIR:$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export NCCL4RUST_CUBIN=../cuda-oxide-smoke/nccl4rust_cuda_oxide_smoke_sm_90.cubin

cargo run --release --offline
```

On the local H100 Slurm partition, GPUs 0 and 3 are a direct NVLink pair and
the node has no configured GPU GRES. Allocate the whole node and select those
devices explicitly:

```bash
srun -p h100x8-cicd -N1 -n1 --exclusive -c16 --mem=64G -t 00:10:00 \
  bash -lc 'export CUDA_VISIBLE_DEVICES=0,3; \
    export LD_LIBRARY_PATH="$NCCL_LIB_DIR:$CUDA_HOME/lib64"; \
    timeout 240s cargo run --release --offline'
```

Build NCCL with `-DCMAKE_CUDA_ARCHITECTURES=90` for this node. A library that
contains only another architecture can fall back to embedded PTX, and PTX from
a toolkit newer than the installed driver may fail JIT compilation.

The package deliberately has its own empty workspace so GPU runtime linkage is
not imposed on the main host/device binding workspace.
