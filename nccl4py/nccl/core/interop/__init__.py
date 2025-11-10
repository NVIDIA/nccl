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
NCCL4Py interop modules: PyTorch and CuPy integration.

This module provides utilities for integrating NCCL4Py with PyTorch and CuPy:
- Memory allocation backed by NCCL allocator
- Buffer resolution utilities
"""

import nccl.core.interop.cupy as cupy
import nccl.core.interop.torch as torch

__all__ = ["cupy", "torch"]
