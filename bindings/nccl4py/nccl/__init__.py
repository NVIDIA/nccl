#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# See LICENSE.txt for more license information
#

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
