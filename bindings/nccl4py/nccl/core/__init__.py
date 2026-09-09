# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# See LICENSE.txt for more license information

"""NCCL4Py Core API: Pythonic access to NCCL for multi-GPU communication.

This module provides the main public API for NCCL operations.
"""

from nccl.core._version import __version__  # noqa: I001

# Core types and enums
from nccl.core.typing import *

# Constants
from nccl.core.constants import *

# Team values and operations
from nccl.core.team import *

# Communicator and configuration
from nccl.core.communicator import *

# Resource management
from nccl.core.resources import *

# Group operations
from nccl.core.group import *

# Utilities
from nccl.core.utils import *

# Memory management
from nccl.core.buffer import *

# Parameter access
from nccl.core.params import *

# The following __all__ exports define the stable, public API surface of NCCL4Py.
# Semantic versioning guarantees apply only to the symbols explicitly listed below.
# All other modules, functions, and symbols are internal implementation details and are subject to change without notice.
__all__ = [  # noqa: RUF022
    # Types and specs
    "NcclDataType",
    "NcclRedOp",
    "NcclGinType",
    "NcclGinConnectionType",
    "NcclHostCftMode",
    "NcclCftTeamMode",
    "NcclCftCap",
    "NcclCommMemStat",
    "NcclBufferSpec",
    "NcclScalarSpec",
    "NcclDeviceSpec",
    "NcclStreamSpec",
    # Exceptions
    "NcclInvalid",
    # Data type constants
    "INT8",
    "CHAR",
    "UINT8",
    "INT32",
    "INT",
    "UINT32",
    "INT64",
    "UINT64",
    "FLOAT16",
    "HALF",
    "FLOAT32",
    "FLOAT",
    "FLOAT64",
    "DOUBLE",
    "BFLOAT16",
    "FLOAT8E4M3",
    "FLOAT8E5M2",
    # Reduction op constants
    "SUM",
    "PROD",
    "MAX",
    "MIN",
    "AVG",
    # Constants and enums
    "CTAPolicy",
    "CommShrinkFlag",
    "CommSuspendFlag",
    "WindowFlag",
    # Communicator
    "NCCLConfig",
    "NCCLCollConfig",
    "VendorOption",
    "NCCLCommProperties",
    "WaitSignalDesc",
    "TeamRequirement",
    "LsaBarrierRequirement",
    "GinBarrierRequirement",
    "LLA2ARequirement",
    "NCCLDevCommRequirements",
    "Communicator",
    # Team
    "NCCLTeam",
    # Resources
    "MultimemHandle",
    "LsaBarrierHandle",
    "GinBarrierHandle",
    "LLA2AHandle",
    "RegisteredBufferHandle",
    "RegisteredWindowHandle",
    "CftLeInfo",
    "CustomRedOp",
    "DevCommResource",
    # Group
    "group",
    "group_start",
    "group_end",
    "GroupSimInfo",
    # Utilities
    "__version__",
    "LibraryInfo",
    "VersionInfo",
    "get_version",
    "show_versions",
    "UniqueId",
    "get_unique_id",
    "get_error_string",
    # Memory
    "mem_alloc",
    "mem_free",
    # Parameters
    "params",
    "dump_params",
    # Interop modules (lazy-loaded)
    "cupy",
    "torch",
]


def __getattr__(name):
    """Lazy-loads interop submodules on first access to avoid importing cupy/torch unless needed."""
    if name == "cupy":
        import nccl.core.interop.cupy

        return nccl.core.interop.cupy
    elif name == "torch":
        import nccl.core.interop.torch

        return nccl.core.interop.torch
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
