# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# See LICENSE.txt for more license information

"""NCCL EP: Pythonic API for the libnccl_ep.so extension.

The Cython bindings under :mod:`nccl.bindings.nccl_ep` are auto-generated from
``contrib/nccl_ep/include/nccl_ep.h`` by cybind. This package provides
hand-written Pythonic wrappers (:class:`Group`, :class:`Handle`,
:class:`Tensor`) on top of those bindings.
"""

import os as _os
from pathlib import Path as _Path

from packaging.version import Version as _Version

from nccl.bindings import nccl_ep as _ep_bindings

# Defaults for libnccl_ep.so's JIT runtime; either env var can be overridden
# by setting it in the environment before importing nccl.ep.
_PKG_DIR = _Path(__file__).parent
_JIT_SOURCE_DIR = _PKG_DIR / "include"
if _JIT_SOURCE_DIR.is_dir():
    _os.environ.setdefault("NCCL_EP_JIT_SOURCE_DIR", str(_JIT_SOURCE_DIR))

# NCCL public headers (nccl.h, nccl_device/...).
try:
    import nvidia.nccl as _nv_nccl

    _NCCL_INCLUDE_DIR = _Path(_nv_nccl.__path__[0]) / "include"
    if _NCCL_INCLUDE_DIR.is_dir():
        _os.environ.setdefault("NCCL_EP_JIT_BUILD_INCLUDE_DIR", str(_NCCL_INCLUDE_DIR))
except ImportError:
    pass

from nccl.ep.allocator import AllocConfig, AllocFn, FreeFn
from nccl.ep.enums import Algorithm, Layout, PassDir
from nccl.ep.group import Group, GroupConfig
from nccl.ep.handle import (
    CombineConfig,
    CombineInputs,
    CombineOutputs,
    DispatchConfig,
    DispatchInputs,
    DispatchOutputs,
    Handle,
    HandleConfig,
    LayoutInfo,
)
from nccl.ep.tensor import Tensor


__all__ = [
    "Algorithm",
    "AllocConfig",
    "AllocFn",
    "CombineConfig",
    "CombineInputs",
    "CombineOutputs",
    "DispatchConfig",
    "DispatchInputs",
    "DispatchOutputs",
    "FreeFn",
    "get_lib_path",
    "get_lib_version",
    "Group",
    "GroupConfig",
    "Handle",
    "HandleConfig",
    "Layout",
    "LayoutInfo",
    "PassDir",
    "Tensor",
]


def _decode_version(v: int) -> _Version:
    """Decode NCCL_EP_VERSION_CODE (MAJOR*10000 + MINOR*100 + PATCH)."""
    return _Version(f"{v // 10000}.{(v % 10000) // 100}.{v % 100}")


def get_lib_version() -> _Version:
    """Release version of the loaded ``libnccl_ep.so`` (e.g. ``0.1.0``)."""
    return _decode_version(_ep_bindings.get_version())


def get_lib_path() -> _Path | None:
    """Path of the loaded ``libnccl_ep.so``, or None if it cannot be determined."""
    raw = _ep_bindings.get_library_path()
    return _Path(raw) if raw else None


def _check_cuda_consistency() -> None:
    """Raise ImportError (nccl.ep not functional) if libnccl.so and
    libnccl_ep.so were built against different CUDA majors."""
    import mmap
    import re

    from nccl.bindings import nccl as _nccl_bindings

    def cuda_variant(path, pattern):
        """``+cudaA.B`` from a .so's embedded version banner, or None."""
        if path is None:
            return None
        try:
            with open(path, "rb") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                m = pattern.search(mm)
                if m is not None:
                    return _Version(f"{int(m.group(1))}.{int(m.group(2))}")
        except OSError:
            pass
        return None

    # Resolving the paths dlopens both libraries; a dlopen RuntimeError (e.g.
    # libnccl_ep.so absent on a CUDA-12 host) becomes ImportError so callers
    # can `try/except ImportError`.
    try:
        nccl_path = _nccl_bindings.get_library_path()
        ep_path = _ep_bindings.get_library_path()
    except RuntimeError as e:
        raise ImportError(str(e)) from e

    nccl_cuda = cuda_variant(nccl_path, re.compile(rb"NCCL version [^\+\s\x00]+\+cuda(\d+)\.(\d+)"))
    ep_cuda = cuda_variant(ep_path, re.compile(rb"NCCL EP version [^\+\s\x00]+\+cuda(\d+)\.(\d+)"))
    if nccl_cuda is None or ep_cuda is None:
        return

    if nccl_cuda.major != ep_cuda.major:
        raise ImportError(
            "nccl.ep is not functional: CUDA major-version mismatch between "
            "NCCL libraries:\n"
            f"  libnccl.so     +cuda{nccl_cuda}  ({nccl_path})\n"
            f"  libnccl_ep.so  +cuda{ep_cuda}  ({ep_path})\n"
            "Both libraries must be built with the same CUDA major. "
            "Check your LD_PRELOAD setting or installed packages."
        )


_check_cuda_consistency()
