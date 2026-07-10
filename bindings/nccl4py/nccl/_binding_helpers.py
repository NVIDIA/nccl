# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# See LICENSE.txt for more license information

"""Internal helper for pairing Python dataclasses with cybind lowpp classes."""

from __future__ import annotations

import sys
from dataclasses import dataclass, fields
from typing import Any

if sys.version_info >= (3, 11):
    from typing import dataclass_transform
else:
    # 3.10 fallback; typing_extensions is pinned in pyproject.toml for 3.10.
    from typing_extensions import dataclass_transform

__all__ = ["binding_dataclass"]


@dataclass_transform(kw_only_default=True, frozen_default=False)
def binding_dataclass(
    lowpp_cls: type,
    *,
    kw_only: bool = True,
    frozen: bool = False,
):
    """Class decorator: pair a Python dataclass facade with a cybind lowpp class.

    On construction, non-None fields are pushed into a fresh ``lowpp_cls()``
    stored as ``self._lowpp``. The lowpp attribute name matches the
    dataclass field name by default; override per field via
    ``field(metadata={"lowpp": "..."})``. Fields missing from the lowpp are
    silently skipped (so Python-only fields can coexist with C-bound ones).

    ``@dataclass`` is applied implicitly — do not stack a separate
    ``@dataclass`` decorator.

    Args:
        lowpp_cls: The cybind-generated lowpp class to materialize on each
            instance. Must be constructible with no arguments.
        kw_only: Forwarded to ``@dataclass(kw_only=...)``. Defaults to ``True``.
        frozen: Forwarded to ``@dataclass(frozen=...)``. Defaults to ``False``.
    """

    def decorate(cls: type) -> type:
        user_post = cls.__dict__.get("__post_init__")

        field_bindings: list[tuple[str, str]] = []

        def __post_init__(self: Any) -> None:
            if user_post is not None:
                user_post(self)
            lowpp = lowpp_cls()
            for field_name, lowpp_name in field_bindings:
                value = getattr(self, field_name)
                if value is None:
                    continue
                if hasattr(value, "_lowpp"):
                    value = value._lowpp
                elif hasattr(value, "ptr"):
                    value = int(value.ptr)
                setattr(lowpp, lowpp_name, value)
            object.__setattr__(self, "_lowpp", lowpp)

        cls.__post_init__ = __post_init__  # type: ignore[attr-defined]
        cls = dataclass(cls, kw_only=kw_only, frozen=frozen)  # type: ignore[arg-type]

        for f in fields(cls):  # type: ignore[arg-type]
            lowpp_name = f.metadata.get("lowpp", f.name)
            if hasattr(lowpp_cls, lowpp_name):
                field_bindings.append((f.name, lowpp_name))

        return cls

    return decorate
