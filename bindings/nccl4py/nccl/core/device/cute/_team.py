# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# See LICENSE.txt for more license information

"""CuTeDSL rank-group values and device operations."""

from dataclasses import dataclass

import cutlass
from cutlass.cutlass_dsl import ir

from . import _nccl_device_ir as _ir

__all__ = ["Team"]


@dataclass(frozen=True, eq=False)
class Team:
    """An immutable rank-group value with ``int`` or ``cutlass.Int32`` fields.

    Construct in host Python or traced code; fields retain their supplied
    values. As an ordinary JIT argument, fields become runtime ``Int32``
    values.

    Convert a host :class:`~nccl.core.NCCLTeam` with
    :func:`~nccl.core.device.cute.runtime.to_cute` before passing it to JIT.
    A Team owns no NCCL resources.
    """

    # Do not cache storage pointers: Teams can be reconstructed in another scope.
    n_ranks: cutlass.Int32 | int
    """Number of ranks in the team."""

    rank: cutlass.Int32 | int
    """The calling rank's index within the team."""

    stride: cutlass.Int32 | int
    """Stride between consecutive team ranks in world-rank numbering."""

    @classmethod
    def _from_ptr(cls, ptr: ir.Value) -> "Team":
        """Read a Team value from initialized native storage."""
        native = _ir._NcclTeam._load(ptr)
        return cls(n_ranks=native.n_ranks, rank=native.rank, stride=native.stride)

    def _to_ptr(self) -> ir.Value:
        native = _ir._NcclTeam(
            n_ranks=self.n_ranks,
            rank=self.rank,
            stride=self.stride,
        )
        ptr = _ir._NcclTeam._allocate(at_entry=True)
        native._store(ptr)
        return ptr
