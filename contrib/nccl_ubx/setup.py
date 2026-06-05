"""Build script for UB-X (Ultra Bandwidth X) CUDA extensions."""

import os
import torch
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

# Source files use relative paths (relative to setup.py, resolved by setuptools).
# include_dirs must be absolute — the compiler receives them directly and ninja
# runs from a temp build directory, so relative paths would resolve incorrectly.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csrc_dir = "csrc"
include_dir = os.path.join(BASE_DIR, "csrc", "include")

# NCCL device-API headers + library are required for the symmetric-window
# allocator and per-kernel pointer resolution (ncclGetLsaPointer,
# ncclGetLsaMultimemPointer, ncclSymPtr, etc.). Default to /usr/local
# (the NGC / nccl-from-source install layout); overridable via env vars.
nccl_home = os.environ.get("NCCL_HOME", "/usr/local")
nccl_include_dir = os.environ.get(
    "NCCL_INCLUDE_DIR", os.path.join(nccl_home, "include"))
nccl_library_dir = os.environ.get(
    "NCCL_LIBRARY_DIR", os.path.join(nccl_home, "lib"))

# Kernel + binding sources.
kernel_sources = [
    os.path.join(csrc_dir, "ubx.cu"),
]
binding_sources = [
    os.path.join(csrc_dir, "bindings.cpp"),
]
all_sources = kernel_sources + binding_sources

# Compiler flags
cxx_flags = ["-O3"]
nvcc_flags = [
    "-O3",
    "--use_fast_math",
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
]

# Optional: compile in kernel-side timeouts on every polling/barrier loop.
# Off by default — adds runtime overhead and `printf` calls. Enable for hang
# triage:  UBX_BUILD_TIMEOUT=1 pip install -e .
# Timeout duration is then runtime-configurable via the UBX_TIMEOUT_SEC env
# var (read once at SymmAllocator construction).
if os.environ.get("UBX_BUILD_TIMEOUT", "0") not in ("0", "", "false", "False"):
    cxx_flags.append("-DUB_TIMEOUT_ENABLED")
    nvcc_flags.append("-DUB_TIMEOUT_ENABLED")

# Optional: compile out cudaGridDependencySynchronize() calls in kernels.
# These are PDL upstream-dependency waits. If a launch is made WITHOUT the
# programmaticStreamSerializationAllowed attribute (PDL off), the intrinsic
# should be a no-op, but some driver/runtime combos still wedge on it.
# Define UBX_SKIP_GRID_DEP_SYNC to compile the calls out entirely.
if os.environ.get("UBX_BUILD_SKIP_GRID_DEP_SYNC", "0") not in ("0", "", "false", "False"):
    cxx_flags.append("-DUBX_SKIP_GRID_DEP_SYNC")
    nvcc_flags.append("-DUBX_SKIP_GRID_DEP_SYNC")

# Detect CUDA architectures.
# Default to the accelerated SM target variants ("9.0a" / "10.0a") to ensure
# access to the full multimem.* instruction set.  Plain "9.0" / "10.0" assemble
# under recent NVCC for the instructions UB-X currently uses, but accelerated-
# only variants of multimem.* are not available on the non-`a` targets, so
# kernels that adopt those variants in future would silently lose performance
# or fail to assemble.  Use the `a` form unless you have a specific reason not to.
# SM 8.0 (A100) has no NVLink multicast hardware at all — exclude it.
cuda_archs = os.environ.get("TORCH_CUDA_ARCH_LIST", None)
if cuda_archs is None:
    cuda_archs = "9.0a;10.0a"

ext_modules = [
    CUDAExtension(
        name="ubx._C",
        sources=all_sources,
        include_dirs=[include_dir, nccl_include_dir],
        # CUDA runtime (cudaLaunchKernelExC, cudaGridDependencySynchronize,
        # etc.) is linked automatically via torch's CUDAExtension. Adding
        # -lnvidia-ml / -lcuda causes linker failures in Docker builds where
        # libcuda.so is only present as a stub on a non-default library path.
        #
        # libnccl is linked explicitly: kernels and the allocator call into
        # NCCL's device + host API for symmetric-window pointer resolution
        # (ncclGetLsaPointer, ncclGetLsaMultimemPointer, ncclDevCommCreate,
        # ncclMemAlloc, etc.). The library lives at the standard NCCL install
        # location ($NCCL_HOME/lib, default /usr/local/lib).
        libraries=["nccl"],
        library_dirs=[nccl_library_dir],
        extra_compile_args={
            "cxx": cxx_flags,
            "nvcc": nvcc_flags,
        },
    ),
]

setup(
    name="ubx",
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
)
