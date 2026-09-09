# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# See LICENSE.txt for more license information

"""NCCL team values and host-side team operations."""

from __future__ import annotations

from dataclasses import dataclass

from nccl.bindings import nccl as _nccl_bindings

from nccl.core._binding_helpers import LowppSpec

__all__ = ["NCCLTeam"]


@dataclass(frozen=True)
class NCCLTeam(LowppSpec, lowpp_cls=_nccl_bindings.Team):
    """A NCCL team: ``(n_ranks, rank, stride)`` view over a communicator.

    Produced by :py:attr:`~nccl.core.Communicator.team_world`,
    :py:attr:`~nccl.core.Communicator.team_lsa`, and
    :py:attr:`~nccl.core.Communicator.team_rail`.
    """

    n_ranks: int
    rank: int
    stride: int
