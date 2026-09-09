# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# See LICENSE.txt for more license information

"""NCCL parameter access.

:py:data:`params` — read-only mapping of NCCL parameter names to their values.
:py:func:`dump_params` — print parameters to stdout.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping

from nccl.bindings import nccl as _nccl_bindings

__all__ = ["params", "dump_params"]


class _NcclParams(Mapping[str, str]):
    """Read-only mapping of NCCL parameter names to their current values."""

    def __getitem__(self, key: str) -> str:
        return _nccl_bindings.param_get_parameter(key)

    def __iter__(self) -> Iterator[str]:
        return iter(_nccl_bindings.param_get_all_keys())

    def __len__(self) -> int:
        return len(_nccl_bindings.param_get_all_keys())


params: _NcclParams = _NcclParams()


def dump_params() -> None:
    """Print NCCL parameters to stdout. Set ``NCCL_PARAM_DUMP_ALL=1`` to include internal parameters."""
    _nccl_bindings.param_dump_all()
