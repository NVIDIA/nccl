# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# See LICENSE.txt for more license information

"""Base classes pairing Python facades with cybind lowpp classes.

A "lowpp" is the low-level Python object cybind generates for a C struct
(e.g. ``nccl.bindings.nccl.Config``); it holds a ``.ptr`` to the struct and
exposes its members as properties. Two roles sit above it:

- :class:`LowppSpec` — a user-facing dataclass whose fields are the source of
  truth. ``obj._to_lowpp()`` builds a fresh lowpp struct from the current field
  values, to pass *into* a binding wrapper function.
- :class:`LowppView` — a read-only, live view over a lowpp struct *returned from*
  a binding wrapper. Each field reads the lowpp directly, so a value NCCL writes
  later (e.g. non-blocking completion) is always current. The wrapped struct is
  available as ``obj._lowpp``.
"""

from __future__ import annotations

import reprlib
from dataclasses import field as dataclass_field, fields as dataclass_fields
from enum import Enum, auto
from typing import Any, ClassVar, TypeVar

__all__ = ["PassBy", "Field", "LowppSpec", "LowppView"]

_FIELD_CONFIG_METADATA_KEY = "nccl4py.binding.field_config"

# Non-writable attributes every lowpp class exposes; an implicitly-named field
# colliding with one is treated as Python-only rather than mis-bound.
_RESERVED_LOWPP_ATTRS = frozenset({"ptr"})


class PassBy(Enum):
    """How a field's value is written into the lowpp struct.

    ``VALUE`` assigns the value (or a nested facade's lowpp) by value.
    ``POINTER`` assigns ``int(value.ptr)`` — a borrowed pointer. The referent
    must outlive any native use of the materialization.
    """

    VALUE = auto()
    POINTER = auto()


def Field(
    *,
    lowpp_name: str | None = None,
    pass_by: PassBy = PassBy.VALUE,
    **kwargs: Any,
) -> Any:
    """A :func:`dataclasses.field` carrying lowpp-binding metadata.

    ``lowpp_name`` overrides the lowpp member name (default: the Python field
    name); ``pass_by`` selects value vs ``.ptr`` assignment. Any other keyword
    (``default``, ``default_factory``, ``metadata``, ...) is forwarded to
    :func:`dataclasses.field`. Both are validated here, since a bad ``pass_by``
    would otherwise mis-bind silently rather than fail loudly.
    """
    if lowpp_name is not None and (not isinstance(lowpp_name, str) or not lowpp_name):
        raise TypeError("lowpp_name must be a non-empty string or None")
    if not isinstance(pass_by, PassBy):
        raise TypeError("pass_by must be a PassBy value")
    metadata = dict(kwargs.pop("metadata", None) or {})
    metadata[_FIELD_CONFIG_METADATA_KEY] = (lowpp_name, pass_by)
    return dataclass_field(metadata=metadata, **kwargs)


def _writable_member(lowpp_cls: type, name: str) -> bool:
    return (
        hasattr(lowpp_cls, name)
        and name not in _RESERVED_LOWPP_ATTRS
        and not callable(getattr(lowpp_cls, name))
    )


def _config_binds(cls: type) -> dict[str, tuple[str, PassBy]]:
    """The cached ``field_name -> (lowpp_name, pass_by)`` map for a config class.

    Computed lazily on first ``_to_lowpp`` (``@dataclass`` runs after
    ``__init_subclass__``, so ``fields()`` isn't available earlier).
    """
    cached = cls.__dict__.get("__lowpp_binds__")
    if cached is not None:
        return cached
    lowpp_cls = cls._lowpp_cls
    binds: dict[str, tuple[str, PassBy]] = {}
    for f in dataclass_fields(cls):
        lowpp_name, pass_by = f.metadata.get(_FIELD_CONFIG_METADATA_KEY, (None, PassBy.VALUE))
        name = lowpp_name or f.name
        if _writable_member(lowpp_cls, name):
            binds[f.name] = (name, pass_by)
        elif lowpp_name is not None:
            raise TypeError(
                f"{cls.__name__}.{f.name} maps to missing or non-writable "
                f"{lowpp_cls.__name__}.{name}"
            )
    cls.__lowpp_binds__ = binds
    return binds


class LowppSpec:
    """Base for a dataclass facade that materializes into a lowpp struct.

    Subclass and apply ``@dataclass`` explicitly::

        @dataclass(frozen=True)
        class WaitSignalDesc(LowppSpec, lowpp_cls=_bindings.WaitSignalDesc):
            peer: int
            op_count: int = Field(default=1, lowpp_name="op_cnt")

    ``slots=True`` is unsupported: the dataclass slots rewrite recreates the
    class, re-running ``__init_subclass__`` without the ``lowpp_cls`` keyword.

    ``_to_lowpp()`` builds a fresh ``lowpp_cls`` from the current field values.
    A field whose value is ``None`` — or that has been ``del``-eted — is left at
    the lowpp default, so a facade may mirror a subset of the lowpp members and
    add Python-only fields: an
    implicitly-named field not naming a writable lowpp member is Python-only,
    while an explicit ``lowpp_name`` must identify a writable member. A ``VALUE``
    field carrying a nested ``LowppSpec`` is materialized recursively into an
    embedded struct; a ``POINTER`` field contributes ``int(value.ptr)``, a
    borrowed pointer the caller must keep alive for the duration of native use.
    """

    _lowpp_cls: ClassVar[type[Any]]
    # field_name -> (lowpp_name, pass_by); populated lazily by _config_binds.
    __lowpp_binds__: ClassVar[dict[str, tuple[str, PassBy]]]

    def __init_subclass__(cls, *, lowpp_cls: type, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        cls._lowpp_cls = lowpp_cls

    def _to_lowpp(self) -> Any:
        binds = _config_binds(type(self))
        lowpp = self._lowpp_cls()
        for field_name, (lowpp_name, pass_by) in binds.items():
            # A deleted field falls back to None, left at the lowpp default.
            value = getattr(self, field_name, None)
            if value is None:
                continue
            if pass_by is PassBy.POINTER:
                try:
                    value = int(value.ptr)
                except (AttributeError, TypeError, ValueError) as exc:
                    raise TypeError(
                        f"{type(self).__name__}.{field_name} is passed by pointer "
                        f"but its value does not provide an integer-convertible "
                        f"'.ptr'"
                    ) from exc
            elif isinstance(value, LowppSpec):
                value = value._to_lowpp()  # nested spec -> embedded struct
            setattr(lowpp, lowpp_name, value)
        return lowpp


_ViewT = TypeVar("_ViewT", bound="LowppView")


class LowppView:
    """Base for a read-only, live view over a lowpp struct returned by NCCL.

    Subclass and expose each field as a plain read-only ``@property`` reading
    ``self._lowpp`` directly, so values NCCL writes later (e.g. non-blocking
    completion) are always current::

        class LsaBarrierHandle(LowppView, lowpp_cls=_bindings.LsaBarrierHandle):
            @property
            def buf_handle(self) -> int:
                return self._lowpp.buf_handle

    Explicit getters keep the fields statically read-only instead of relying on
    generated properties, and may read a differently-named lowpp member
    (``mc_base_pointer`` -> ``self._lowpp.mc_base_ptr``). Create with
    ``Cls._from_lowpp(lowpp)``; direct construction is blocked. The wrapped
    struct outlives the view via the held ``self._lowpp`` reference. ``repr()``
    lists every read-only getter (``__lowpp_fields__``, derived below).
    """

    _lowpp_cls: ClassVar[type[Any]]
    __lowpp_fields__: ClassVar[tuple[str, ...]] = ()
    _lowpp: Any  # the wrapped struct, set per-instance by _from_lowpp

    def __init_subclass__(cls, *, lowpp_cls: type, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        cls._lowpp_cls = lowpp_cls
        # Every getter, whether or not its name matches the member it reads.
        cls.__lowpp_fields__ = tuple(
            name
            for name, attr in vars(cls).items()
            if isinstance(attr, property)
        )

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise TypeError(f"{type(self).__name__} objects are returned by NCCL APIs")

    @classmethod
    def _from_lowpp(cls: type[_ViewT], lowpp: Any) -> _ViewT:
        if not isinstance(lowpp, cls._lowpp_cls):
            raise TypeError(
                f"{cls.__name__} wraps {cls._lowpp_cls.__name__}, got "
                f"{type(lowpp).__name__}"
            )
        obj = object.__new__(cls)
        obj._lowpp = lowpp
        return obj

    def __eq__(self, other: object) -> bool:
        return type(other) is type(self) and self._lowpp.ptr == other._lowpp.ptr

    def __hash__(self) -> int:
        return hash(self._lowpp.ptr)

    @reprlib.recursive_repr()
    def __repr__(self) -> str:
        cls = type(self)
        body = ", ".join(f"{n}={getattr(self, n)!r}" for n in cls.__lowpp_fields__)
        return f"{cls.__qualname__}({body})"
