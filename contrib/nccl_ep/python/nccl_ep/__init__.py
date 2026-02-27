# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NCCL EP: Low-level Python bindings for NCCL EP operations.

This package provides low-level ctypes bindings to the NCCL EP C API.

Pure Python (ctypes) implementation - no compilation needed!
Works immediately with LD_PRELOAD setup.
"""

from ._version import __version__

try:
    from .nccl_wrapper import (
        HAVE_TORCH,
        NCCLLibrary,
        get_nccl_comm_from_group,
        ncclDataTypeEnum,
        ncclEpAlgorithm_t,
        ncclEpDispatchConfig_t,
        ncclEpGroupConfig_t,
        ncclEpHandleConfig_t,
        ncclEpTensorTag_t,
        ncclNDTensor_t,
        ncclEpAllocFn_t,
        ncclEpFreeFn_t,
        CUDA_SUCCESS,
        CUDA_ERROR_MEMORY_ALLOCATION,
    )

    HAVE_NCCL_EP = True
    NCCL_EP_ALGO_LOW_LATENCY = ncclEpAlgorithm_t.NCCL_EP_ALGO_LOW_LATENCY
    NCCL_EP_ALGO_HIGH_THROUGHPUT = ncclEpAlgorithm_t.NCCL_EP_ALGO_HIGH_THROUGHPUT

except ImportError as e:
    HAVE_NCCL_EP = False
    HAVE_TORCH = False
    _import_error = str(e)
    NCCL_EP_ALGO_LOW_LATENCY = 0
    NCCL_EP_ALGO_HIGH_THROUGHPUT = 1

__all__ = [
    '__version__',
    'HAVE_NCCL_EP',
    'HAVE_TORCH',
    'NCCLLibrary',
    'get_nccl_comm_from_group',
    'ncclDataTypeEnum',
    'ncclEpAlgorithm_t',
    'ncclEpDispatchConfig_t',
    'ncclEpGroupConfig_t',
    'ncclEpHandleConfig_t',
    'ncclEpTensorTag_t',
    'ncclNDTensor_t',
    'ncclEpAllocFn_t',
    'ncclEpFreeFn_t',
    'CUDA_SUCCESS',
    'CUDA_ERROR_MEMORY_ALLOCATION',
    'NCCL_EP_ALGO_LOW_LATENCY',
    'NCCL_EP_ALGO_HIGH_THROUGHPUT',
]

if not HAVE_NCCL_EP:
    import warnings
    warnings.warn(
        f"NCCL EP bindings are not available. Error: {_import_error}\n"
        "Make sure:\n"
        "  1. NCCL_HOME points to your NCCL EP build\n"
        "  2. LD_PRELOAD is set to force PyTorch to use custom NCCL\n"
        "  3. NCCL library has EP extensions"
    )
