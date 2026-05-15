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
from typing import Generator

from nccl import bindings as _nccl_bindings

from nccl.core.constants import NCCL_UNDEF_FLOAT

__all__ = ["GroupSimInfo", "group_start", "group_end", "group_simulate_end", "group"]


class GroupSimInfo:
    """Information for NCCL group operation simulation.

    Holds simulation information that can be used to estimate the performance
    of group operations without actually executing them. Pass an instance to
    group_simulate_end and read estimated_time after the call.
    """

    def __init__(self) -> None:
        """Initializes group simulation info with default values."""
        self._sim_info = _nccl_bindings.SimInfo()

        # Apply NCCL_SIM_INFO_INITIALIZER defaults
        self._sim_info.size_ = int(_nccl_bindings.sim_info_dtype.itemsize)
        self._sim_info.magic = 0x74685283  # NCCL protocol magic number for ncclSimInfo_t validation
        self._sim_info.version = _nccl_bindings.get_version()
        self._sim_info.estimated_time = NCCL_UNDEF_FLOAT

    @property
    def ptr(self) -> int:
        """Raw NCCL simulation info pointer."""
        return int(self._sim_info.ptr)

    # Field proxies
    @property
    def estimated_time(self) -> float:
        """Estimated execution time for the group operations, in seconds."""
        return self._sim_info.estimated_time


def group_start() -> None:
    """Starts a group of NCCL operations.

    All NCCL operations called after this will be batched together and
    executed when :py:func:`group_end` is called. This can improve performance
    by allowing NCCL to optimize the operation sequence.
    """
    return _nccl_bindings.group_start()


def group_end() -> None:
    """Ends a group of NCCL operations.

    Executes all operations that were queued since the last
    :py:func:`group_start`. Must be called to actually perform the batched
    operations.
    """
    return _nccl_bindings.group_end()


def group_simulate_end(sim_info: GroupSimInfo | None) -> None:
    """Simulates the end of a group of NCCL operations.

    Estimates the execution time of the queued operations without actually
    executing them. The estimated time is stored in
    :py:attr:`GroupSimInfo.estimated_time`.

    Args:
        sim_info: Simulation info object to store estimated time, or
            ``None`` to discard the result.
    """
    if sim_info is None:
        return _nccl_bindings.group_simulate_end(None)
    return _nccl_bindings.group_simulate_end(sim_info.ptr)


@contextlib.contextmanager
def group() -> Generator[None, None, None]:
    """Context manager for NCCL group operations.

    Automatically calls :py:func:`group_start` on entry and
    :py:func:`group_end` on exit, ensuring proper cleanup even if an
    exception occurs.
    """
    group_start()
    try:
        yield
    finally:
        group_end()
