# nccl4py: Python Bindings for NCCL

Python bindings for NVIDIA Collective Communications Library (NCCL), providing high-performance multi-GPU and multi-node communication primitives.

This package provides both low-level Cython bindings and a high-level Pythonic API for NCCL collective operations.

### Experimental Cython Support

```python
from nccl.bindings cimport cynccl
```

This allows Cython code to call NCCL functions with minimal overhead. The `cynccl.pxd` file is included in the package distribution for direct Cython integration.

## Requirements

- **CUDA Toolkit**: CUDA 12.x or 13.x
- **NCCL Library**: Matching CUDA version (nvidia-nccl-cu12 or nvidia-nccl-cu13)
- **Python**: 3.10 or later

## Development Setup

### Prerequisites

**Set CUDA_HOME environment variable:**
```bash
export CUDA_HOME=/usr/local/cuda  # Or your CUDA installation path
```

### Building from Source (using Makefile)

The easiest way to build is using the Makefile, which requires [uv](https://docs.astral.sh/uv/):

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

cd nccl4py

# Create development environment
make dev

# Build package (sdist and wheel)
make build

# Clean build artifacts
make clean
```

The Makefile automatically:
- Detects CUDA version from `$CUDA_HOME`
- Installs appropriate CUDA-specific dependencies (cu12/cu13)
- Builds Cython extensions

### Manual Build (without uv)

If you prefer not to use `uv`, you can build manually with standard Python tools:

```bash
# Set CUDA_HOME
export CUDA_HOME=/usr/local/cuda

cd nccl4py

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install in editable mode with CUDA dependencies
pip install -e .[cu12]  # For CUDA 12.x
# OR
pip install -e .[cu13]  # For CUDA 13.x

# Build distribution packages
pip install build
python -m build
```

## Building Distribution Packages

### Build for current Python version:
```bash
make build
# Output: build/dist/nccl4py-*.tar.gz and nccl4py-*.whl
```

## References

- [NCCL Documentation](https://docs.nvidia.com/deeplearning/nccl/)
- [NCCL Repository](https://github.com/NVIDIA/nccl)
