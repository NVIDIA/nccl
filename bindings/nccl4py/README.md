# nccl4py: Python Bindings for NCCL

nccl4py provides low-level Cython bindings and a high-level Python API for the
[NVIDIA Collective Communications Library (NCCL)](https://developer.nvidia.com/nccl).
It supports multi-GPU and multi-node communication from Python.

## Installation

nccl4py supports Linux and Python 3.10 or later. Running NCCL operations
requires a supported NVIDIA GPU and compatible NVIDIA driver. Choose the extra
matching the CUDA major version of the other CUDA packages in your environment:

```bash
python -m pip install "nccl4py[cu12]"
# or
python -m pip install "nccl4py[cu13]"
```

The extras install the corresponding NCCL runtime and CUDA Python dependencies.
Installing a published wheel does not require `CUDA_HOME` or a local CUDA
Toolkit. Compiling nccl4py from source does.

Verify the installation and inspect the loaded component versions:

```python
import nccl.core as nccl

nccl.show_versions()
```

`nccl.core` ships inline type information for
[PEP 561](https://peps.python.org/pep-0561/)-compatible type checkers.

See [`examples/01_basic`](examples/01_basic) for MPI-based collective and
point-to-point examples.

## Experimental Cython Support

The wheel includes `nccl/bindings/cynccl.pxd` as an experimental Cython API:

```cython
from nccl.bindings cimport cynccl
```

This allows Cython extensions to call NCCL functions with minimal Python
overhead.

## Namespace Package

`nccl` is a [PEP 420](https://peps.python.org/pep-0420/) implicit namespace
package. nccl4py provides `nccl.bindings` and `nccl.core`; other NCCL extension
distributions can provide additional `nccl.*` subpackages.

## Building from source

The commands below use `bindings/nccl4py` as the working directory:

```bash
git clone https://github.com/NVIDIA/nccl.git
cd nccl/bindings/nccl4py
```

Building the extension modules requires:

- Linux
- Python 3.10 or later
- a C++ compiler
- CUDA Toolkit 12.x or 13.x, including its headers
- `CUDA_HOME` pointing to that CUDA Toolkit

Set `CUDA_HOME` to the CUDA Toolkit used for the build. For example:

```bash
export CUDA_HOME=/usr/local/cuda
```

### Development with the Makefile

The Makefile uses [uv](https://docs.astral.sh/uv/) to manage the development
environment. `make dev` detects the CUDA version from `CUDA_HOME`, creates
`.venv`, and installs nccl4py in editable mode with the matching `cu12` or
`cu13` extra, test dependencies, PyTorch, and CuPy. `make build` compiles the
Cython extensions.

```bash
# Create the development environment and install nccl4py in editable mode.
make dev

# Build the source distribution and wheel.
make build

# Remove local build artifacts.
make clean
```

### Build with standard Python tools

Create a virtual environment and install nccl4py with standard Python tools:

```bash
python -m venv .venv
source .venv/bin/activate

# Select exactly one CUDA extra.
python -m pip install -e ".[cu12]"
# python -m pip install -e ".[cu13]"
```


## References

- [NCCL documentation](https://docs.nvidia.com/deeplearning/nccl/)
- [NCCL repository](https://github.com/NVIDIA/nccl)
- [nccl4py package documentation](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/nccl4py.html)
