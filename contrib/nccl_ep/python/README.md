# NCCL EP Low-Level Python Bindings

Low-level Python (ctypes) bindings for the NCCL EP C API.

**Package:** Install as `nccl_ep` (pip). Use in Python as `import nccl_ep`.

## Features

- **Pure Python**: No compilation required, uses ctypes for direct C API bindings
- **Low-level access**: Direct wrapper around NCCL EP C API

## Prerequisites

Build NCCL and the nccl_ep library first so that `$NCCL_HOME/lib` contains `libnccl_ep.so` and `libnccl.so`:

```bash
cd /path/to/nccl
make src.build

# Default NCCL output directory is ./build.
# Optional: build into a custom directory instead of ./build
# make src.build BUILDDIR=$PWD/build_rel

# If NCCL was built into ./build (default):
make -C contrib/nccl_ep

# If NCCL was built into a custom directory, pass BUILDDIR as an absolute path.
# Example:
# make -C contrib/nccl_ep BUILDDIR=$PWD/build_rel
```

For custom NCCL build directories, use an absolute `BUILDDIR` path when invoking `make -C contrib/nccl_ep` (for example, `$PWD/build_rel`).

## Installation

```bash
pip install -e /path/to/nccl/nccl_ep/python
# or, for a non-editable install:
# pip install /path/to/nccl/nccl_ep/python
```

## Environment Setup

```bash
# Point to the NCCL build directory (contains lib/libnccl_ep.so and lib/libnccl.so)
export NCCL_HOME=/path/to/nccl/build
export LD_LIBRARY_PATH=$NCCL_HOME/lib:$LD_LIBRARY_PATH

# Optional: Force PyTorch to use this NCCL build (set BEFORE importing torch).
export LD_PRELOAD=$NCCL_HOME/lib/libnccl.so.2${LD_PRELOAD:+:$LD_PRELOAD}
```

## Quick Start

This package provides low-level ctypes bindings to the NCCL EP C API.

```python
from nccl_ep import (
    NCCLLibrary,
    ncclEpAlgorithm_t,
    HAVE_NCCL_EP,
    NCCL_EP_ALGO_HIGH_THROUGHPUT
)

# Check if NCCL EP is available
if HAVE_NCCL_EP:
    nccl_lib = NCCLLibrary()
    # Use nccl_lib.ncclEpDispatch, ncclEpCombine, etc.
    # See nccl_wrapper.py for full C API bindings
```

## API Reference

This package provides low-level ctypes bindings to the NCCL EP C library.
See `nccl_ep/nccl_wrapper.py` for the complete C API interface.

### Constants

- `HAVE_NCCL_EP` - `True` if NCCL EP library is available
- `NCCL_EP_ALGO_LOW_LATENCY` (0) - Low-latency algorithm
- `NCCL_EP_ALGO_HIGH_THROUGHPUT` (1) - High-throughput algorithm

## Usage

This is a low-level bindings package. It provides direct ctypes access to the NCCL EP C API functions such as:

- `ncclEpDispatch` - Dispatch tokens to experts
- `ncclEpCombine` - Combine expert outputs
- `ncclEpHandleCreate` / `ncclEpHandleDestroy` - Handle management
- `ncclEpSetTensor` - Configure input/output tensors

For API documentation, see the docstrings in `nccl_ep/nccl_wrapper.py`.

## License

BSD-3-Clause
