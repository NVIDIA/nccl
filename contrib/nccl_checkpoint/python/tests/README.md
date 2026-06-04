# NCCL Checkpoint Testing

The test harness for NCCL Checkpointing is entirely Python-based and is found
in `python/tests`.

The pytest harness is unable to change loader behavior at runtime, so during
test launch `LD_PRELOAD` must be set to include the `libnccl-checkpoint-shim.so`
library.

These tests do not actually use CUDA Checkpoint or CRIU, but they do exercise
the checkpoint and restore paths.  Because the shim frees communicator resources
on prepare and recreates them during restore, this cycle of prepare and restore
without a checkpoint is still an effective test vehicle.

Tests can be launched in one of two modes:

- Pass `--checkpoint-mode=shim` (the default) and launch the test with
  `libnccl-checkpoint-shim.so` in `LD_PRELOAD`. This mode exercises the
  checkpoint/restore path and the shim interposition behavior.

- Pass `--checkpoint-mode=no-shim` and don't set LD_PRELOAD.  This mode can be
  used to ensure the same tests pass on regular NCCL without any checkpointing
  functionality.

The tests support MPI via `mpi4py` and `pytest-mpi`, and support execution in a
variety of multi-host and multi-gpu configurations.  If the testing is launched
using MPI, the user should pass the `--with-mpi` option.

# Preparation

Install the Python test dependencies from `contrib/nccl_checkpoint`:

```bash
cd contrib/nccl_checkpoint
python -m pip install -e "python[test]"
```

It may be useful to set the following variables:

```bash
export NCCL_CHECKPOINT_SHIM="$PWD/build/lib/libnccl-checkpoint-shim.so"
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
```

* `NCCL_CHECKPOINT_SHIM` is set based on the default BUILDDIR from
  contrib/nccl_checkpoint.
* `PYTHONUNBUFFERED=1` makes Python output appear promptly from each launched
  rank.
* `PYTHONFAULTHANDLER=1` prints Python tracebacks for fatal interpreter errors.

# Execution

```bash
# with SLURM:
LD_PRELOAD="$NCCL_CHECKPOINT_SHIM" \
srun python -m pytest -ra --with-mpi --checkpoint-mode=shim python/tests

# with OpenMPI:
mpirun -x LD_PRELOAD="$NCCL_CHECKPOINT_SHIM" \
 python -m pytest -ra --with-mpi --checkpoint-mode=shim python/tests
```

The above commands will execute the full set of tests.

Note that the test harness sets `NCCL_CHECKPOINT_KVS_PATH` at runtime, so it
doesn't have to be set at launch time.  However the test harness cannot change
`LD_PRELOAD`, so it must be set prior to launch.  Consult your MPI's
documentation to determine how to pass environment variables into the execution
environment.
