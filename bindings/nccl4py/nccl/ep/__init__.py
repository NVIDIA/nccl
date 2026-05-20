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
    from nccl._show_versions import get_version

    v = get_version()
    if v.nccl_ep is None:
        return

    nccl_cuda = v.nccl.cuda_variant
    ep_cuda = v.nccl_ep.cuda_variant
    if nccl_cuda is None or ep_cuda is None:
        return

    if nccl_cuda.major != ep_cuda.major:
        raise ImportError(
            "CUDA major-version mismatch between NCCL libraries:\n"
            f"  libnccl.so     +cuda{nccl_cuda}  ({v.nccl.path})\n"
            f"  libnccl_ep.so  +cuda{ep_cuda}  ({v.nccl_ep.path})\n"
            "Both libraries must be built with the same CUDA major. "
            "Check your LD_PRELOAD setting or installed packages."
        )


_check_cuda_consistency()
