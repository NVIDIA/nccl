#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# See LICENSE.txt for more license information
#

"""
Internal bindings implementation.

This module preloads the NCCL library using cuda-pathfinder if available,
which provides better library discovery across different environments
(conda, system installs, custom CUDA paths, etc.).

If cuda-pathfinder is not available or fails to find NCCL, the Cython
bindings will fall back to direct dlopen("libnccl.so.2") which works
if the library is in standard system paths.

See documentation for cuda-pathfinder including the search order at:
https://nvidia.github.io/cuda-python/cuda-pathfinder/latest/generated/cuda.pathfinder.load_nvidia_dynamic_lib.html#cuda.pathfinder.load_nvidia_dynamic_lib
"""

# Optional: Preload NCCL library for better discovery
# This runs before the Cython extensions are loaded, allowing
# dlsym(RTLD_DEFAULT, ...) to find the already-loaded library
try:
    from cuda.pathfinder import load_nvidia_dynamic_lib

    load_nvidia_dynamic_lib("nccl")
except ImportError:
    # cuda-python not installed, fall back to Cython's dlopen
    pass
except Exception:
    # Library not found by pathfinder or other error
    # Fall back to Cython's dlopen - it will provide the error message if needed
    pass
