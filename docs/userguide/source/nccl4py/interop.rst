*****************
Framework Interop
*****************

Lazy-loaded helpers for allocating CuPy arrays and PyTorch tensors backed by
NCCL-managed memory, plus resolvers that translate framework objects into
the ``(ptr, count, dtype, device_id)`` tuple NCCL expects. The submodules
are imported on first attribute access via ``nccl.core.cupy`` and
``nccl.core.torch``.

CuPy
====

.. py:currentmodule:: nccl.core.interop.cupy

.. autofunction:: empty

.. autofunction:: resolve_array


PyTorch
=======

.. py:currentmodule:: nccl.core.interop.torch

.. autofunction:: empty

.. autofunction:: resolve_tensor
