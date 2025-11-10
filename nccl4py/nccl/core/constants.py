# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
# See LICENSE.txt for license information

"""
NCCL constants and enums.

This module centralizes all NCCL constants for easy access and organization.
"""

from enum import IntEnum

__all__ = [
    "NCCL_UNDEF_INT",
    "NCCL_UNDEF_FLOAT",
    "NCCL_SPLIT_NOCOLOR",
    "CTAPolicy",
    "CommShrinkFlag",
    "WindowFlag",
]

# NCCL sentinel values for undefined config fields
NCCL_UNDEF_INT: int = -2147483648  # INT_MIN
"""NCCL sentinel value for undefined integer configuration fields."""

NCCL_UNDEF_FLOAT: float = -1.0
"""NCCL sentinel value for undefined float fields."""

# Communicator split constants
NCCL_SPLIT_NOCOLOR: int = -1
"""Color value for ncclCommSplit to indicate rank will not be part of any group."""


# CTA (Cooperative Thread Array) Policy flags
class CTAPolicy(IntEnum):
    """
    NCCL performance policy for CTA scheduling.
    """

    Default = 0x00
    """Default CTA policy."""
    Efficiency = 0x01
    """Optimize for efficiency."""
    Zero = 0x02
    """Zero-CTA optimization."""


# Communicator shrink flags
class CommShrinkFlag(IntEnum):
    """
    Flags for ncclCommShrink behavior.
    """

    Default = 0x00
    """Shrink the parent communicator."""
    Abort = 0x01
    """First terminate ongoing parent operations, then shrink the parent communicator."""


# Window registration flags
class WindowFlag(IntEnum):
    """
    Flags for window registration.
    """

    Default = 0x00
    """Default window registration."""
    CollSymmetric = 0x01
    """Collective symmetric window registration."""
