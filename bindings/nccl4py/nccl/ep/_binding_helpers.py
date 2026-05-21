# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# See LICENSE.txt for more license information

"""``binding_dataclass`` — decorator pairing a Python dataclass with a
binding-layer (cybind-generated, Cython-compiled) value class.

The decorator:

1. Applies :func:`dataclasses.dataclass` to the decorated class.
2. Adds a ``to_binding()`` method that materializes a fresh instance of
   the bound binding class, populated from the dataclass fields. Field
   values are coerced before assignment:

   * :class:`enum.IntEnum` → ``int``
   * ``None`` → ``0`` (NCCL EP treats 0 as the NULL sentinel for handle
     and pointer fields)
   * objects with ``to_binding`` (nested ``binding_dataclass`` instances)
     → recursively materialized and ``memcpy``'d in via the binding's
     own setter
   * objects with a ``.ptr`` integer attribute (e.g. :class:`Tensor`)
     → ``obj.ptr`` (pointer address)

3. If the optional ``size_field_dtype`` argument is supplied, the
   binding's ``size_`` field is set to ``dtype.itemsize`` before any
   user-provided fields are copied in. NCCL EP uses ``size`` for
   layout/ABI validation and rejects ``size == 0``, so any config struct
   that crosses the API boundary needs this set.

Field names on the decorated class must match field names on the
binding class exactly. Mismatches surface as :class:`AttributeError` on
the first ``to_binding`` call.

Note on type-checkers
---------------------

``to_binding`` is installed at decoration time via :func:`setattr`.
Static type-checkers (Pylance / pyright / mypy) can't follow that —
they will flag ``cfg.to_binding()`` as ``reportAttributeAccessIssue``
at call sites. Suppress with ``# type: ignore[attr-defined]`` where
needed. The pattern is correct; the warning is the price of using a
decorator to add methods the type-checker has no static signal for.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING, Any

try:
    from typing import dataclass_transform
except ImportError:
    # 3.10 fallback; typing_extensions is pinned in pyproject.toml for 3.10.
    from typing_extensions import dataclass_transform

if TYPE_CHECKING:
    from typing import Callable, TypeVar

    T = TypeVar("T")


__all__ = ["binding_dataclass"]


def _coerce(val: Any) -> Any:
    if val is None:
        return 0
    if isinstance(val, IntEnum):
        return int(val)
    # Nested binding_dataclass: recurse, then hand the binding object to
    # the parent setter (which memcpy's the sub-struct into place).
    to_binding = getattr(val, "to_binding", None)
    if callable(to_binding):
        return to_binding()
    # Tensor-shaped objects: peel out the underlying pointer.
    if not isinstance(val, (int, float, bool)) and hasattr(val, "ptr"):
        return val.ptr
    return val


@dataclass_transform()
def binding_dataclass(
    binding_cls: type,
    *,
    size_field_dtype: Any = None,
) -> Callable[[type[T]], type[T]]:
    """Class decorator: pair a Python dataclass with a binding-layer class.

    Applies :func:`dataclasses.dataclass` automatically (no need to
    stack ``@dataclass`` separately) and tags itself with
    :func:`typing.dataclass_transform` so mypy and pyright recognize the
    decorated class as a dataclass.

    Args:
        binding_cls: The cybind-generated value class to populate.
        size_field_dtype: Optional ``numpy.dtype`` whose ``itemsize`` is
            written to the binding's ``size_`` field on every
            ``to_binding()`` call. Pass the module-level
            ``<binding_cls_snake>_dtype`` exposed alongside ``binding_cls``.

    Example:
        ::

            from nccl.bindings import nccl_ep as _bindings

            @binding_dataclass(
                _bindings.GroupConfig,
                size_field_dtype=_bindings.group_config_dtype,
            )
            class GroupConfig:
                algorithm: Algorithm = Algorithm.LOW_LATENCY
                num_experts: int = 0

            cfg = GroupConfig(num_experts=8)
            binding = cfg.to_binding()  # type: ignore[attr-defined]
    """
    def decorator(cls: type[T]) -> type[T]:
        cls = dataclass(cls, kw_only=True)

        def to_binding(self) -> object:
            obj = binding_cls()
            if size_field_dtype is not None:
                obj.size_ = size_field_dtype.itemsize
            for name in cls.__dataclass_fields__:
                setattr(obj, name, _coerce(getattr(self, name)))
            return obj

        to_binding.__doc__ = (
            f"Return a fresh ``{binding_cls.__module__}.{binding_cls.__name__}`` "
            f"instance populated from this dataclass's fields."
        )
        # Dynamic attribute install — setattr signals intent and bypasses
        # the "type[T] has no to_binding" static check at the assignment site.
        setattr(cls, "to_binding", to_binding)
        return cls

    return decorator
