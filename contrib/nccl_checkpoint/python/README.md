# NCCL Checkpoint Python Bindings

This directory contains pure Python `ctypes` bindings for the NCCL checkpoint
shim.

The `nccl_checkpoint` package binds checkpoint symbols from
`libnccl-checkpoint-shim.so` and exposes:

- `get_version()`, wrapping `ncclCheckpointGetVersion()`
- `checkpoint_prepare()`, wrapping `ncclCheckpointPrepare()`
- `checkpoint_restore()`, wrapping `ncclCheckpointRestore()`

`get_version()` returns a `VersionInfo` object with NCCL-style packed integer
versions. `checkpoint_version` is the checkpoint shim version, and
`nccl_version` is the `NCCL_VERSION_CODE` from the NCCL header used to compile
the shim. For example, checkpoint release `0.1.0` is returned as `100`.

The shim must be loaded before Python starts via `LD_PRELOAD`, otherwise
`get_version()`, `checkpoint_prepare()`, and `checkpoint_restore()` raise
`NCCLCheckpointPreloadError`, a subclass of `NCCLCheckpointError`.

## Install

From this (`contrib/nccl_checkpoint/python`) directory:

```bash
python -m pip install -e .
```

For test dependencies:

```bash
python -m pip install -e ".[test]"
```

Although this python package does not require CUDA or NCCL Python packages to be
installed, the testing does use `cuda.bindings` and `cuda.core`. See the NVIDIA
[CUDA Python installation
docs](https://nvidia.github.io/cuda-python/cuda-bindings/latest/install.html)
for installation details.

See [tests/README.md](tests/README.md) for the recommended test commands.

## Example

```python
# Example.py
import nccl_checkpoint

nccl_checkpoint.get_version()
nccl_checkpoint.checkpoint_prepare()
nccl_checkpoint.checkpoint_restore()
```
