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

# Defaults for libnccl_ep.so's JIT runtime; either env var can be overridden
# by setting it in the environment before importing nccl.ep.
_PKG_DIR = _Path(__file__).parent
_JIT_SOURCE_DIR = _PKG_DIR / "include" / "nccl_ep"
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
    "Group",
    "GroupConfig",
    "Handle",
    "HandleConfig",
    "Layout",
    "LayoutInfo",
    "PassDir",
    "Tensor",
]


def _check_cuda_consistency() -> None:
    """Verify libnccl.so and libnccl_ep.so were built with the same CUDA major.
    """
    from nccl._show_versions import (
        _NCCL_BANNER,
        _NCCL_EP_BANNER,
        _extract_cuda_variant,
        _resolve_so_path,
    )
    from nccl.bindings import nccl as _nccl_bindings
    from nccl.bindings import nccl_ep as _ep_bindings

    # Trigger libnccl_ep.so / libnccl.so dlopen via a cheap binding call.
    # Bindings raise RuntimeError when dlopen fails (e.g. libnccl_ep.so absent
    # on a CUDA-12 host). Convert to ImportError so `import nccl.ep` honors the
    # standard contract and `try/except ImportError` callers (including
    # _nccl_ep_importable) degrade gracefully without disturbing nccl.core.
    # Doing this directly avoids recursing through get_version(), which would
    # re-enter `import nccl.ep` while this module is mid-init.
    try:
        _ep_bindings.get_version()
        _nccl_bindings.get_version()
    except RuntimeError as e:
        raise ImportError(str(e)) from e

    nccl_path = _resolve_so_path("libnccl.so")
    ep_path = _resolve_so_path("libnccl_ep.so")
    nccl_cuda = _extract_cuda_variant(nccl_path, _NCCL_BANNER)
    ep_cuda = _extract_cuda_variant(ep_path, _NCCL_EP_BANNER)
    if nccl_cuda is None or ep_cuda is None:
        return

    if nccl_cuda.major != ep_cuda.major:
        raise ImportError(
            "CUDA major-version mismatch between NCCL libraries:\n"
            f"  libnccl.so     +cuda{nccl_cuda}  ({nccl_path})\n"
            f"  libnccl_ep.so  +cuda{ep_cuda}  ({ep_path})\n"
            "Both libraries must be built with the same CUDA major. "
            "Check your LD_PRELOAD setting or installed packages."
        )


_check_cuda_consistency()
