# SPDX-FileCopyrightText: Copyright (c) 2026 Kuaishou Technology. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build: pip install -e . --no-build-isolation --no-deps

against the NCCL tree this directory sits in (`make -j src.build` at its root first), or NCCL_HOME = another
NCCL >= 2.31.2 build dir (include/, lib/). The reduce-scatter's kernels need nvcc (CUDA_HOME, default /usr/local/cuda)."""
import os
import pathlib
import re

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ROOT = pathlib.Path(__file__).resolve().parent
NCCL_HOME = os.environ.get("NCCL_HOME", str(ROOT.parent.parent / "build"))
if not (pathlib.Path(NCCL_HOME) / "include" / "nccl.h").exists():
    raise SystemExit(f"no NCCL build in {NCCL_HOME}: run `make -j src.build` at the NCCL root, or set NCCL_HOME to an "
                     "NCCL >= 2.31.2 build dir containing include/nccl.h and lib/libnccl.so")
CUDA_HOME = os.environ.get("CUDA_HOME", "/usr/local/cuda")
# the kernels need sm_90 (TMA, multimem): drop older targets from an inherited list (NGC containers start at 7.5)
archs = [a for a in re.split(r"[;\s]+", os.environ.get("TORCH_CUDA_ARCH_LIST", "")) if a and not re.match(r"[0-8]\.", a)]
os.environ["TORCH_CUDA_ARCH_LIST"] = ";".join(archs) or "9.0;10.0"
# NCCL's headers as system headers: nccl_device.h defines static device helpers that warn unused in host code
nccl_include = ["-isystem", f"{NCCL_HOME}/include"]
# TIDE_FAULT_INJECTION=1 compiles the test-only fault hooks (TIDE_FAULT=...) in; scripts/build.sh sets it
defines = ["-DTIDE_FAULT_INJECTION"] if os.environ.get("TIDE_FAULT_INJECTION", "0") == "1" else []

sources = [
    "csrc/bindings.cpp",
    "csrc/collective/ag/plan.cpp",
    "csrc/collective/ag/issue.cpp",
    "csrc/collective/rs/plan.cpp",
    "csrc/collective/rs/issue.cpp",
    "csrc/collective/rs/waves.cpp",
    "csrc/collective/rs/kernels.cu",
    "csrc/collective/rs/devctl.cu",
    "csrc/runtime/comm.cpp",
    "csrc/runtime/config.cpp",
]

setup(
    packages=["tide"],
    ext_modules=[
        CUDAExtension(
            name="tide._C",
            sources=sources,  # relative to this file: setuptools rejects absolute source paths
            include_dirs=[str(ROOT / "csrc"), f"{CUDA_HOME}/include"],
            library_dirs=[f"{NCCL_HOME}/lib", f"{CUDA_HOME}/lib64", f"{CUDA_HOME}/lib64/stubs"],
            libraries=["nccl", "cudart", "cuda"],
            extra_compile_args={"cxx": ["-O2", "-fvisibility=hidden", *nccl_include, *defines],
                                "nvcc": ["-O3", *nccl_include, *defines]},
            extra_link_args=[f"-Wl,-rpath,{NCCL_HOME}/lib", f"-Wl,-rpath,{CUDA_HOME}/lib64"],
        )
    ],
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=True)},
)
