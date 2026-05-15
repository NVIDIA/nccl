# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# See LICENSE.txt for more license information

"""NCCL constants and Python-only enums.

This module centralizes Python-only NCCL constants and flag enums that wrap
#define constants from the NCCL C headers. C-defined enums are re-exported
from nccl.core.typing.
"""

from enum import IntEnum, IntFlag

__all__ = [
    "NCCL_UNDEF_INT",
    "NCCL_UNDEF_FLOAT",
    "NCCL_SPLIT_NOCOLOR",
    "CTAPolicy",
    "CommShrinkFlag",
    "CommSuspendFlag",
    "WindowFlag",
]

# NCCL sentinel values for undefined config fields
NCCL_UNDEF_INT: int = -2147483648  # INT_MIN
"""NCCL sentinel value for undefined integer configuration fields."""

NCCL_UNDEF_FLOAT: float = -1.0
"""NCCL sentinel value for undefined float fields."""

# Communicator split constants
NCCL_SPLIT_NOCOLOR: int = -1
"""Color value for ncclCommSplit to indicate the rank will not be part of any group."""

# NCCL magic number
NCCL_MAGIC: int = 0xCAFEBEEF
"""Magic number for NCCL configuration structs (used for ABI validation)."""


# CTA (Cooperative Thread Array) Policy flags
class CTAPolicy(IntFlag):
    """NCCL performance policy for CTA scheduling, used by
    :py:attr:`NCCLConfig.cta_policy`.
    """

    DEFAULT = 0x00
    """Default CTA policy."""
    EFFICIENCY = 0x01
    """Optimize for efficiency."""
    ZERO = 0x02
    """Zero-CTA optimization."""

    # Backward-compat aliases (PascalCase forms from the prior public API).
    Default = 0x00
    Efficiency = 0x01
    Zero = 0x02


# Communicator shrink flags
class CommShrinkFlag(IntEnum):
    """Behavior flag for :py:meth:`Communicator.shrink`."""

    DEFAULT = 0x00
    """Shrink the parent communicator normally; outstanding NCCL operations
    must already be quiesced."""
    ABORT = 0x01
    """First terminate ongoing parent operations, then shrink. No resources
    are shared with the parent."""

    # Backward-compat aliases.
    Default = 0x00
    Abort = 0x01


# Communicator suspend flags
class CommSuspendFlag(IntFlag):
    """Behavior flag for :py:meth:`Communicator.suspend`."""

    MEM = 0x01
    """Suspend memory by releasing dynamic GPU memory allocations held by
    the communicator."""

    # Backward-compat alias.
    Mem = 0x01


# Window registration flags
class WindowFlag(IntFlag):
    """Window registration behavior flags for
    :py:meth:`Communicator.register_window`.
    """

    DEFAULT = 0x00
    """Default window registration."""
    COLL_SYMMETRIC = 0x01
    """Collective symmetric window registration."""
    STRICT_ORDERING = 0x02
    """Strict ordering for window operations."""

    # Backward-compat aliases.
    Default = 0x00
    CollSymmetric = 0x01
    StrictOrdering = 0x02
