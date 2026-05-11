# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# See LICENSE.txt for more license information

"""NCCL4Py EP API: Low-level access to NCCL EP operations.

This module provides the public Python API for NCCL EP ctypes bindings.
"""

import re

# Runtime library discovery
from cuda.pathfinder import DynamicLibNotFoundError, load_nvidia_dynamic_lib

_SUPPORTED_CUDA_MAJOR = "13"
# Match NCCL's own version banner (e.g. "NCCL version 2.30.4 compiled with
# CUDA 13.0"). Embedded by src/init.cc and survives `strip`, unlike nvcc's
# "Cuda compilation tools, release N.N" identification string.
_NCCL_CUDA_VERSION_RE = re.compile(
    rb"NCCL version \S+ compiled with CUDA ([0-9]+)\."
)


def _cuda_major_from_nccl_library(path: str) -> str:
    majors: set[str] = set()
    tail = b""
    try:
        with open(path, "rb") as library:
            while True:
                chunk = library.read(1024 * 1024)
                if not chunk:
                    break

                data = tail + chunk
                for match in _NCCL_CUDA_VERSION_RE.finditer(data):
                    majors.add(match.group(1).decode("ascii"))
                tail = data[-128:]
    except OSError as e:
        raise ImportError(f"nccl.ep failed to read CUDA version from {path}: {e}") from e

    if len(majors) > 1:
        detected = ", ".join(sorted(majors))
        raise ImportError(
            f"nccl.ep found multiple CUDA compiler versions in {path}: {detected}."
        )

    if not majors:
        raise ImportError(f"nccl.ep failed to read CUDA version from {path}.")

    return next(iter(majors))


def _check_cuda_major() -> None:
    try:
        loaded = load_nvidia_dynamic_lib("nccl")
    except DynamicLibNotFoundError as e:
        raise ImportError("nccl.ep failed to load libnccl.so.") from e

    if loaded is None or loaded.abs_path is None:
        raise ImportError("nccl.ep failed to resolve the path to libnccl.so.")

    path = loaded.abs_path
    source = f"libnccl.so resolved by cuda.pathfinder from {path}"
    if loaded.found_via:
        source = f"{source} ({loaded.found_via})"

    major = _cuda_major_from_nccl_library(path)
    if major != _SUPPORTED_CUDA_MAJOR:
        raise ImportError(
            "nccl.ep only supports CUDA 13: the libnccl_ep.so packaged with this "
            "nccl4py wheel was built against CUDA 13 and is binary-incompatible "
            f"with other CUDA major versions. Detected CUDA {major} in {source}. "
            "Reinstall nccl4py with the cu13 extra (pip install 'nccl4py[cu13]'), "
            "or use a CUDA 13 NCCL build."
        )


_check_cuda_major()

from .nccl_wrapper import (
    HAVE_TORCH,
    NCCLLibrary,
    _load_nccl_ep_library,
    get_nccl_comm_from_group,
    ncclEpAlgorithm_t,
    ncclEpDispatchConfig_t,
    ncclEpGroupConfig_t,
    ncclEpHandleConfig_t,
    ncclEpLayout_t,
    ncclEpTensorTag_t,
    ncclNDTensor_t,
    ncclEpAllocFn_t,
    ncclEpFreeFn_t,
    CUDA_SUCCESS,
    CUDA_ERROR_MEMORY_ALLOCATION,
)

try:
    _load_nccl_ep_library()
except Exception as e:
    raise ImportError(
        "nccl.ep failed to load libnccl_ep.so. The library is searched (in "
        "order) under the nccl4py package's nccl/ep/lib/ directory, "
        "$CONDA_PREFIX/lib[64], the dynamic linker's default paths "
        "(LD_LIBRARY_PATH / ld.so.cache / /lib / /usr/lib), and "
        "$CUDA_HOME/lib[64]. Install nccl4py with libnccl_ep.so packaged in, "
        "or place libnccl_ep.so on one of the searched paths."
    ) from e

NCCL_EP_ALGO_LOW_LATENCY = ncclEpAlgorithm_t.NCCL_EP_ALGO_LOW_LATENCY
NCCL_EP_ALGO_HIGH_THROUGHPUT = ncclEpAlgorithm_t.NCCL_EP_ALGO_HIGH_THROUGHPUT

# The following __all__ exports define the stable, public API surface of NCCL4Py EP.
# Semantic versioning guarantees apply only to the symbols explicitly listed below.
# All other modules, functions, and symbols are internal implementation details and are
# subject to change without notice.
__all__ = [
    # Availability
    "HAVE_TORCH",
    # Library wrapper
    "NCCLLibrary",
    "get_nccl_comm_from_group",
    # Types and enums
    "ncclEpAlgorithm_t",
    "ncclEpDispatchConfig_t",
    "ncclEpGroupConfig_t",
    "ncclEpHandleConfig_t",
    "ncclEpLayout_t",
    "ncclEpTensorTag_t",
    "ncclNDTensor_t",
    # Allocator callbacks
    "ncclEpAllocFn_t",
    "ncclEpFreeFn_t",
    # CUDA status constants
    "CUDA_SUCCESS",
    "CUDA_ERROR_MEMORY_ALLOCATION",
    # Algorithm constants
    "NCCL_EP_ALGO_LOW_LATENCY",
    "NCCL_EP_ALGO_HIGH_THROUGHPUT",
]
