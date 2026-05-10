# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# See LICENSE.txt for more license information

"""NCCL EP: Pythonic API for the libnccl_ep.so extension.

The Cython bindings under :mod:`nccl.ep.bindings` are auto-generated from
``contrib/nccl_ep/include/nccl_ep.h`` by cybind. This package provides
hand-written Pythonic wrappers (:class:`EpGroup`, :class:`EpHandle`,
:class:`NDTensor`) on top of those bindings.
"""

import re

from cuda.pathfinder import DynamicLibNotFoundError, load_nvidia_dynamic_lib

from nccl.ep.allocator import EpAllocConfig, ncclEpAllocFn_t, ncclEpFreeFn_t
from nccl.ep.enums import NcclEpAlgorithm, NcclEpLayout
from nccl.ep.group import EpGroup, EpGroupConfig
from nccl.ep.handle import (
    EpCombineConfig,
    EpCombineInputs,
    EpCombineOutputs,
    EpDispatchConfig,
    EpDispatchInputs,
    EpDispatchOutputs,
    EpHandle,
    EpHandleConfig,
    EpLayoutInfo,
)
from nccl.ep.tensor import NDTensor
from nccl.ep.interop.torch import get_nccl_comm_from_group


__all__ = [
    "EpAllocConfig",
    "EpCombineConfig",
    "EpCombineInputs",
    "EpCombineOutputs",
    "EpDispatchConfig",
    "EpDispatchInputs",
    "EpDispatchOutputs",
    "EpGroup",
    "EpGroupConfig",
    "EpHandle",
    "EpHandleConfig",
    "EpLayoutInfo",
    "NDTensor",
    "NcclEpAlgorithm",
    "NcclEpLayout",
    "get_nccl_comm_from_group",
    "ncclEpAllocFn_t",
    "ncclEpFreeFn_t",
]


# Runtime CUDA-major gate. libnccl_ep.so is CUDA-13-only; libnccl_ep loads
# lazily on first call, but we want to fail import-time rather than
# crash-time when the loaded libnccl.so is the wrong CUDA major.
_SUPPORTED_CUDA_MAJOR = "13"
# NCCL's version banner survives `strip`, unlike nvcc's "release N.N" string.
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
