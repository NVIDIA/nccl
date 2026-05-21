# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# See LICENSE.txt for more license information

"""NCCL EP: Pythonic API for the libnccl_ep.so extension.

The Cython bindings under :mod:`nccl.bindings.nccl_ep` are auto-generated from
``contrib/nccl_ep/include/nccl_ep.h`` by cybind. This package provides
hand-written Pythonic wrappers (:class:`EpGroup`, :class:`EpHandle`,
:class:`NDTensor`) on top of those bindings.
"""

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
    "ncclEpAllocFn_t",
    "ncclEpFreeFn_t",
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
