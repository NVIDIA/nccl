# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
# See LICENSE.txt for license information

"""
NCCL group operations for batching collective communications.

This module provides APIs for grouping multiple NCCL operations together to enable
efficient batching and overlapping of communication operations. Group calls allow
multiple collective operations to be launched together with better performance and
the ability to simulate operations for performance analysis.

See more details at: https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/groups.html
"""

from __future__ import annotations

import contextlib
from typing import Generator

from nccl import bindings as _nccl_bindings

from nccl.core.constants import NCCL_UNDEF_FLOAT

__all__ = ["GroupSimInfo", "group_start", "group_end", "group_simulate_end", "group"]


class GroupSimInfo:
    """
    Information for NCCL group operation simulation.

    This class holds simulation information that can be used to estimate
    the performance of group operations without actually executing them.
    """

    def __init__(self) -> None:
        """
        Initializes group simulation info with default values.
        """
        self._sim_info = _nccl_bindings.SimInfo()

        # Apply NCCL_SIM_INFO_INITIALIZER defaults
        self._sim_info.size_ = int(_nccl_bindings.sim_info_dtype.itemsize)
        self._sim_info.magic = 0x74685283  # NCCL protocol magic number for ncclSimInfo_t validation
        self._sim_info.version = _nccl_bindings.get_version()
        self._sim_info.estimated_time = NCCL_UNDEF_FLOAT

    @property
    def ptr(self) -> int:
        """
        Raw NCCL simulation info pointer.

        Returns:
            ``int``: The simulation info pointer.
        """
        return int(self._sim_info.ptr)

    # Field proxies
    @property
    def estimated_time(self) -> float:
        """
        Estimated execution time for the group operations (in seconds).

        Returns:
            ``float``: Estimated time in seconds.
        """
        return self._sim_info.estimated_time


def group_start() -> None:
    """
    Starts a group of NCCL operations.

    All NCCL operations called after this will be batched together
    and executed when group_end() is called. This can improve
    performance by allowing NCCL to optimize the operation sequence.

    See Also:
        https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/group.html#ncclgroupstart
    """
    return _nccl_bindings.group_start()


def group_end() -> None:
    """
    Ends a group of NCCL operations.

    Executes all operations that were queued since the last group_start().
    This must be called to actually perform the batched operations.

    See Also:
        https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/group.html#ncclgroupend
    """
    return _nccl_bindings.group_end()


def group_simulate_end(sim_info: GroupSimInfo | None) -> None:
    """
    Simulates the end of a group of NCCL operations.

    This estimates the execution time of the queued operations without
    actually executing them. The estimated time is stored in sim_info.

    Args:
        - sim_info (GroupSimInfo, optional): Simulation info object to store estimated time, or None.

    See Also:
        https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/group.html#ncclgroupsimulateend
    """
    if sim_info is None:
        return _nccl_bindings.group_simulate_end(None)
    return _nccl_bindings.group_simulate_end(int(sim_info.ptr))


@contextlib.contextmanager
def group() -> Generator[None, None, None]:
    """
    Context manager for NCCL group operations.

    Automatically calls group_start() on entry and group_end() on exit.
    This ensures proper cleanup even if an exception occurs.

    See Also:
        https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/groups.html
    """
    group_start()
    try:
        yield
    finally:
        group_end()
