# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# See LICENSE.txt for more license information

"""Experimental CuTeDSL bindings for the NCCL device API.

Public device types and operations are available directly from this package.
Use ``runtime.to_cute(host_object)`` for explicit host argument conversion.
Compile-only helpers also live in ``runtime``.

.. warning::

    Experimental and unstable. This subpackage carries no semantic-versioning
    guarantee: symbols may change or be removed in any release before nccl4py
    1.0.

    The selected ``libnccl_device.bc`` must match exactly the NCCL version
    nccl4py was built against -- the ``bindings`` version reported by
    ``nccl.core.show_versions()``. Resolution prefers
    ``$NCCL_HOME/lib/libnccl_device.bc`` and otherwise uses the copy from the
    installed ``nvidia-nccl-cu12`` / ``nvidia-nccl-cu13`` wheel. No version
    check is performed; a mismatch can cause link errors or incorrect kernel
    behavior.

See ``examples/cute/00_basic.py`` for a complete, runnable example.
"""

try:
    import cutlass.cute  # noqa: F401
except ModuleNotFoundError as e:
    if e.name not in ("cutlass", "cutlass.cute"):
        raise
    raise ImportError(
        "nccl.core.device.cute requires the nvidia-cutlass-dsl package. "
        "Install it via the matching extra:\n"
        "    pip install 'nccl4py[cu12]'   # for CUDA 12\n"
        "    pip install 'nccl4py[cu13]'   # for CUDA 13"
    ) from e

from . import runtime
from ._barrier import GinBarrierSession, GinFenceLevel, HybridBarrierSession, LsaBarrierSession
from ._comm import CftTeamMode, DevComm
from ._coop import Coop
from ._gin import (
    Gin,
    GinAllContexts,
    GinBackendMask,
    GinCounterInc,
    GinDescriptorSmem,
    GinOptFlag,
    GinRequest,
    GinResourceSharingMode,
    GinSegmentType,
    GinSignalAdd,
    GinSignalFollowResult,
    GinSignalInc,
    ThreadScope,
)
from ._handles import GinBarrierHandle, LsaBarrierHandle, MultimemHandle
from ._reduce_copy import (
    local_reduce_sum_copy,
    lsa_copy,
    lsa_copy_tma,
    lsa_reduce_sum,
    lsa_reduce_sum_copy,
    multimem_copy,
    multimem_reduce_sum,
    multimem_reduce_sum_copy,
)
from ._team import Team
from ._types import CftLeInfo, MemoryOrder
from ._window import SymmetricPointer, Window

__all__ = [
    "CftLeInfo",
    "CftTeamMode",
    "Coop",
    "DevComm",
    "Gin",
    "GinAllContexts",
    "GinBackendMask",
    "GinBarrierHandle",
    "GinBarrierSession",
    "GinCounterInc",
    "GinDescriptorSmem",
    "GinFenceLevel",
    "GinOptFlag",
    "GinRequest",
    "GinResourceSharingMode",
    "GinSegmentType",
    "GinSignalAdd",
    "GinSignalFollowResult",
    "GinSignalInc",
    "HybridBarrierSession",
    "LsaBarrierHandle",
    "LsaBarrierSession",
    "MemoryOrder",
    "MultimemHandle",
    "SymmetricPointer",
    "Team",
    "ThreadScope",
    "Window",
    "local_reduce_sum_copy",
    "lsa_copy",
    "lsa_copy_tma",
    "lsa_reduce_sum",
    "lsa_reduce_sum_copy",
    "multimem_copy",
    "multimem_reduce_sum",
    "multimem_reduce_sum_copy",
    "runtime",
]
