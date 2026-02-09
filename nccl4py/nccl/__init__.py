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
NCCL4Py: Python bindings for NVIDIA Collective Communications Library (NCCL).

NCCL4Py provides Pythonic access to NCCL for efficient multi-GPU and multi-node
communication. It supports all NCCL collective operations, point-to-point
communication, and advanced features like buffer registration and custom reduction
operators.
"""

from nccl._version import __version__

__all__ = [
    "__version__",
]
