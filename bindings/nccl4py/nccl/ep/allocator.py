# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# See LICENSE.txt for more license information

"""ctypes function-pointer types and ``AllocConfig`` wrapper for the NCCL EP allocator hooks.

Match the C typedefs declared in ``nccl_ep.h``:

.. code-block:: c

    typedef cudaError_t (*ncclEpAllocFn_t)(void** ptr, size_t size, void* context);
    typedef cudaError_t (*ncclEpFreeFn_t)(void* ptr, void* context);

Usage
-----

Use either as a decorator to wrap a Python callable into a C-callable
trampoline, extract the integer address with
``ctypes.cast(fn, ctypes.c_void_p).value``, then drop the addresses into
:class:`AllocConfig` and attach it to :class:`GroupConfig.alloc`::

    import ctypes
    from cuda.bindings import runtime as cudart
    from nccl.ep import (
        AllocConfig, AllocFn, FreeFn, Group, GroupConfig,
    )

    @AllocFn
    def my_alloc(out_ptr, size, context):
        err, ptr = cudart.cudaMalloc(size)
        out_ptr[0] = ctypes.c_void_p(int(ptr))
        return int(err)

    @FreeFn
    def my_free(ptr, context):
        err, = cudart.cudaFree(ptr)
        return int(err)

    alloc_addr = ctypes.cast(my_alloc, ctypes.c_void_p).value
    free_addr  = ctypes.cast(my_free,  ctypes.c_void_p).value
    cfg = GroupConfig(
        ...,
        alloc=AllocConfig(alloc_fn=alloc_addr, free_fn=free_addr),
    )
    group = Group.create(comm, cfg)

Lifetime requirement
--------------------

The trampoline owns the underlying machine-code stub. ``my_alloc`` and
``my_free`` MUST stay alive for at least as long as the resulting
:py:class:`Group`; if either is garbage-collected while NCCL EP still
holds the pointer, the next call from C lands in freed memory and crashes.
Stash them at module scope or on an object that outlives the group.

For allocators that already live in a separate ``.so`` (exported with C
linkage), the addresses can be obtained directly from a ``ctypes.CDLL``
handle — the loaded library object then anchors the lifetime instead::

    my_lib = ctypes.CDLL("libmyalloc.so")
    alloc_addr = ctypes.cast(my_lib.my_alloc, ctypes.c_void_p).value
    free_addr  = ctypes.cast(my_lib.my_free,  ctypes.c_void_p).value
"""

from __future__ import annotations

import ctypes

from nccl.bindings import nccl_ep as _ep_bindings
from nccl.ep._binding_helpers import binding_dataclass


__all__ = ["AllocConfig", "AllocFn", "FreeFn"]


AllocFn = ctypes.CFUNCTYPE(
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_void_p),
    ctypes.c_size_t,
    ctypes.c_void_p,
)
"""C-callable type: ``cudaError_t (*)(void** ptr, size_t size, void* context)``."""

FreeFn = ctypes.CFUNCTYPE(
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_void_p,
)
"""C-callable type: ``cudaError_t (*)(void* ptr, void* context)``."""


@binding_dataclass(_ep_bindings.AllocConfig)
class AllocConfig:
    """Allocator hooks for :py:attr:`GroupConfig.alloc`.

    Mirrors :c:struct:`ncclEpAllocConfig_t`. Leaving every field at 0
    selects NCCL EP's default ``cudaMalloc``/``cudaFree`` path.

    Attributes:
        alloc_fn: Address of a custom ``ncclEpAllocFn_t``-compatible C
            allocator. 0 (default) → ``cudaMalloc``.
        free_fn: Address of the matching deallocator. Same conventions
            as ``alloc_fn``.
        context: Opaque pointer forwarded verbatim to every
            ``alloc_fn``/``free_fn`` call.
    """

    alloc_fn: int = 0
    free_fn: int = 0
    context: int = 0
