# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# See LICENSE.txt for more license information

"""NCCL group operations for batching collective communications.

This module provides APIs for grouping multiple NCCL operations together to enable
efficient batching and overlapping of communication operations. Group calls allow
multiple collective operations to be launched together with better performance and
the ability to simulate operations for performance analysis.

See :ref:`group-calls` for more details.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Generator, Literal, overload

from nccl.bindings import nccl as _nccl_bindings

__all__ = ["GroupSimInfo", "group_start", "group_end", "group"]


@dataclass(frozen=True)
class GroupSimInfo:
    """Result of an NCCL group simulation.

    Returned by :py:func:`group_end` when called with ``simulate=True``.
    """

    estimated_time: float
    """Estimated execution time for the simulated group operations, in seconds."""


def group_start() -> None:
    """Starts a group of NCCL operations.

    All NCCL operations called after this will be batched together and
    executed when :py:func:`group_end` is called. This can improve performance
    by allowing NCCL to optimize the operation sequence.
    """
    return _nccl_bindings.group_start()


@overload
def group_end(*, simulate: Literal[False] = False) -> None: ...
@overload
def group_end(*, simulate: Literal[True]) -> GroupSimInfo: ...
@overload
def group_end(*, simulate: bool) -> GroupSimInfo | None: ...
def group_end(*, simulate: bool = False) -> GroupSimInfo | None:
    """Ends a group of NCCL operations.

    By default, executes all operations queued since the last
    :py:func:`group_start`. When ``simulate=True``, the queued operations
    are simulated instead of executed, and the estimated execution time is
    returned in a :py:class:`GroupSimInfo`.

    Args:
        simulate: When True, simulates the group instead of executing it
            and returns a :py:class:`GroupSimInfo` carrying the estimated
            time. Defaults to False.

    Returns:
        ``None`` when ``simulate=False``; a :py:class:`GroupSimInfo` with
        the simulation result when ``simulate=True``.
    """
    if not simulate:
        _nccl_bindings.group_end()
        return None

    sim_info = _nccl_bindings.group_simulate_end()
    return GroupSimInfo(estimated_time=sim_info.estimated_time)


@contextlib.contextmanager
def group() -> Generator[None, None, None]:
    """Context manager for NCCL group operations.

    Automatically calls :py:func:`group_start` on entry and
    :py:func:`group_end` on exit, ensuring proper cleanup even if an
    exception occurs.

    Simulation mode is not supported here. To simulate, call
    :py:func:`group_start` and :py:func:`group_end` directly and pass
    ``simulate=True`` to :py:func:`group_end`.
    """
    group_start()
    try:
        yield
    finally:
        group_end()
