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
NCCL communicator creation, management, and operations.

This module provides the core Communicator class and NCCLConfig for NCCL operations.
Communicators manage groups of ranks for collective and point-to-point communication,
with support for buffer registration, custom reduction operators, and resource management.
"""

from __future__ import annotations
from typing import Sequence, Any

import numpy as _np

from cuda.core.experimental import Device

from nccl import bindings as _nccl_bindings

from nccl.core.buffer import NcclBuffer
from nccl.core.constants import (
    NCCL_SPLIT_NOCOLOR,
    NCCL_UNDEF_INT,
    CTAPolicy,
    CommShrinkFlag,
    WindowFlag,
)
from nccl.core.cuda import get_stream_ptr, get_cuda_device
from nccl.core.resources import (
    CommResource,
    RegisteredBufferHandle,
    RegisteredWindowHandle,
    CustomRedOp,
)
from nccl.core.typing import (
    NcclDataType,
    NcclBufferSpec,
    NcclRedOp,
    NcclStreamSpec,
    NcclScalarSpec,
    NcclInvalid,
)
from nccl.core.utils import UniqueId


__all__ = [
    "NCCLConfig",
    "Communicator",
]


class NCCLConfig:
    """
    NCCL configuration for communicator initialization.

    This class provides configuration options for NCCL communicators, allowing
    fine-tuning of performance and behavior characteristics. Based on the official
    NCCL API documentation.
    """

    def __init__(
        self,
        *,
        blocking: bool | None = None,
        cga_cluster_size: int | None = None,
        min_ctas: int | None = None,
        max_ctas: int | None = None,
        net_name: str | None = None,
        split_share: bool | None = None,
        traffic_class: int | None = None,
        comm_name: str | None = None,
        collnet_enable: bool | None = None,
        cta_policy: CTAPolicy | None = None,
        shrink_share: bool | None = None,
        nvls_ctas: int | None = None,
        n_channels_per_net_peer: int | None = None,
        nvlink_centric_sched: bool | None = None,
    ) -> None:
        """
        Initializes NCCL configuration with custom parameters.

        All parameters are optional and default to NCCL's internal defaults (undefined).

        Args:
            - blocking (bool, optional): Blocking (True) or non-blocking (False) communicator behavior. Defaults to True.
            - cga_cluster_size (int, optional): Cooperative Group Array (CGA) size for kernels (0-8). Defaults to 4 for sm90+, 0 otherwise.
            - min_ctas (int, optional): Minimal number of CTAs per kernel. Positive integer up to 32. Defaults to 1.
            - max_ctas (int, optional): Maximal number of CTAs per kernel. Positive integer up to 32. Defaults to 32.
            - net_name (str, optional): Network module name (e.g., "IB", "Socket"). Case-insensitive. Defaults to NCCL auto-selection.
            - split_share (bool, optional): Share resources with child communicator during split. Defaults to False.
            - traffic_class (int, optional): Traffic class (TC) for network operations (>= 0). Network-specific meaning. Defaults to undefined.
            - comm_name (str, optional): User-defined communicator name for logging and profiling. Defaults to undefined.
            - collnet_enable (bool, optional): Enable (True) or disable (False) IB SHARP. Defaults to False.
            - cta_policy (CTAPolicy, optional): CTA scheduling policy. See CTAPolicy enum (Default, Efficiency, Zero). Defaults to CTAPolicy.Default.
            - shrink_share (bool, optional): Share resources with child communicator during shrink. Defaults to False.
            - nvls_ctas (int, optional): Total number of CTAs for NVLS kernels. Positive integer. Defaults to NCCL auto-determined value.
            - n_channels_per_net_peer (int, optional): Number of network channels for pairwise communication. Positive integer, rounded up to power of 2. Defaults to AlltoAll-optimized value.
            - nvlink_centric_sched (bool, optional): Enable (True) NVLink-centric scheduling. Defaults to False.

        Notes:
            Aborting any communicator may affect others in the same family when split_share or shrink_share is enabled.
        """
        self._cfg: _nccl_bindings.Config = _nccl_bindings.Config()
        self._cfg._data.fill(0)

        # Apply NCCL_CONFIG_INITIALIZER defaults
        self._cfg.size_ = int(_nccl_bindings.config_dtype.itemsize)
        self._cfg.magic = 0xCAFEBEEF  # NCCL protocol magic number for ncclConfig_t validation
        self._cfg.version = _nccl_bindings.get_version()

        # Initialize all fields to undef
        self._cfg.blocking = NCCL_UNDEF_INT
        self._cfg.cga_cluster_size = NCCL_UNDEF_INT
        self._cfg.min_ctas = NCCL_UNDEF_INT
        self._cfg.max_ctas = NCCL_UNDEF_INT
        self._cfg.split_share = NCCL_UNDEF_INT
        self._cfg.traffic_class = NCCL_UNDEF_INT
        self._cfg.collnet_enable = NCCL_UNDEF_INT
        self._cfg.cta_policy = NCCL_UNDEF_INT
        self._cfg.shrink_share = NCCL_UNDEF_INT
        self._cfg.nvls_ctas = NCCL_UNDEF_INT
        self._cfg.n_channels_per_net_peer = NCCL_UNDEF_INT
        self._cfg.nvlink_centric_sched = NCCL_UNDEF_INT

        # Use setters for validation - they handle type checking and range validation
        if blocking is not None:
            self.blocking = blocking
        if cga_cluster_size is not None:
            self.cga_cluster_size = cga_cluster_size
        if min_ctas is not None:
            self.min_ctas = min_ctas
        if max_ctas is not None:
            self.max_ctas = max_ctas
        if net_name is not None:
            self.net_name = net_name
        if split_share is not None:
            self.split_share = split_share
        if traffic_class is not None:
            self.traffic_class = traffic_class
        if comm_name is not None:
            self.comm_name = comm_name
        if collnet_enable is not None:
            self.collnet_enable = collnet_enable
        if cta_policy is not None:
            self.cta_policy = cta_policy
        if shrink_share is not None:
            self.shrink_share = shrink_share
        if nvls_ctas is not None:
            self.nvls_ctas = nvls_ctas
        if n_channels_per_net_peer is not None:
            self.n_channels_per_net_peer = n_channels_per_net_peer
        if nvlink_centric_sched is not None:
            self.nvlink_centric_sched = nvlink_centric_sched

    def __repr__(self) -> str:
        """
        Returns string representation showing non-default values.

        Returns:
            ``str``: String showing configured (non-default) values.
        """
        parts = []

        # Check each field and include if not undefined
        if self._cfg.blocking != NCCL_UNDEF_INT:
            parts.append(f"blocking={bool(self._cfg.blocking)}")
        if self._cfg.cga_cluster_size != NCCL_UNDEF_INT:
            parts.append(f"cga_cluster_size={self._cfg.cga_cluster_size}")
        if self._cfg.min_ctas != NCCL_UNDEF_INT:
            parts.append(f"min_ctas={self._cfg.min_ctas}")
        if self._cfg.max_ctas != NCCL_UNDEF_INT:
            parts.append(f"max_ctas={self._cfg.max_ctas}")
        if hasattr(self._cfg, "net_name") and self._cfg.net_name:
            parts.append(f"net_name='{self._cfg.net_name}'")
        if self._cfg.split_share != NCCL_UNDEF_INT:
            parts.append(f"split_share={bool(self._cfg.split_share)}")
        if self._cfg.traffic_class != NCCL_UNDEF_INT:
            parts.append(f"traffic_class={self._cfg.traffic_class}")
        if hasattr(self._cfg, "comm_name") and self._cfg.comm_name:
            parts.append(f"comm_name='{self._cfg.comm_name}'")
        if self._cfg.collnet_enable != NCCL_UNDEF_INT:
            parts.append(f"collnet_enable={bool(self._cfg.collnet_enable)}")
        if self._cfg.cta_policy != NCCL_UNDEF_INT:
            parts.append(f"cta_policy={CTAPolicy(self._cfg.cta_policy).name}")
        if self._cfg.shrink_share != NCCL_UNDEF_INT:
            parts.append(f"shrink_share={bool(self._cfg.shrink_share)}")
        if self._cfg.nvls_ctas != NCCL_UNDEF_INT:
            parts.append(f"nvls_ctas={self._cfg.nvls_ctas}")
        if self._cfg.n_channels_per_net_peer != NCCL_UNDEF_INT:
            parts.append(f"n_channels_per_net_peer={self._cfg.n_channels_per_net_peer}")
        if self._cfg.nvlink_centric_sched != NCCL_UNDEF_INT:
            parts.append(f"nvlink_centric_sched={bool(self._cfg.nvlink_centric_sched)}")

        if parts:
            return f"<NCCLConfig: {', '.join(parts)}>"
        else:
            return "<NCCLConfig: all defaults>"

    @property
    def ptr(self) -> int:
        """
        Raw NCCL config pointer.

        Returns:
            ``int``: The configuration pointer.
        """
        return int(self._cfg.ptr)

    # Field proxies

    @property
    def blocking(self) -> bool:
        """
        Blocking or non-blocking communicator behavior.

        Returns:
            ``bool``: True for blocking, False for non-blocking. Default: True.
        """
        return bool(self._cfg.blocking)

    @blocking.setter
    def blocking(self, val: bool) -> None:
        if not isinstance(val, bool):
            raise NcclInvalid(f"blocking must be bool, got {type(val).__name__}")
        self._cfg.blocking = int(val)

    @property
    def cga_cluster_size(self) -> int:
        """
        Cooperative Group Array size (0-8).

        Returns:
            ``int``: CGA cluster size. Default: 4 for sm90+, 0 for older architectures.
        """
        return self._cfg.cga_cluster_size

    @cga_cluster_size.setter
    def cga_cluster_size(self, val: int) -> None:
        if not isinstance(val, int):
            raise NcclInvalid(f"cga_cluster_size must be int, got {type(val).__name__}")
        if not (0 <= val <= 8):
            raise NcclInvalid(f"cga_cluster_size must be between 0 and 8, got {val}")
        self._cfg.cga_cluster_size = val

    @property
    def min_ctas(self) -> int:
        """
        Minimum number of CTAs.

        Returns:
            ``int``: Positive integer up to 32. Default: 1.
        """
        return self._cfg.min_ctas

    @min_ctas.setter
    def min_ctas(self, val: int) -> None:
        if not isinstance(val, int):
            raise NcclInvalid(f"min_ctas must be int, got {type(val).__name__}")
        if not (0 < val <= 32):
            raise NcclInvalid(f"min_ctas must be a positive integer up to 32, got {val}")
        self._cfg.min_ctas = val

    @property
    def max_ctas(self) -> int:
        """
        Maximum number of CTAs.

        Returns:
            ``int``: Positive integer up to 32. Default: 32.
        """
        return self._cfg.max_ctas

    @max_ctas.setter
    def max_ctas(self, val: int) -> None:
        if not isinstance(val, int):
            raise NcclInvalid(f"max_ctas must be int, got {type(val).__name__}")
        if not (0 < val <= 32):
            raise NcclInvalid(f"max_ctas must be a positive integer up to 32, got {val}")
        self._cfg.max_ctas = val

    @property
    def net_name(self) -> str:
        """
        Network module name.

        Returns:
            ``str``: Network module (e.g., 'IB', 'Socket'). Case-insensitive. Default: auto-selected.
        """
        return self._cfg.net_name

    @net_name.setter
    def net_name(self, val: str) -> None:
        if not isinstance(val, str):
            raise NcclInvalid(f"net_name must be str, got {type(val).__name__}")
        self._cfg.net_name = val

    @property
    def split_share(self) -> bool:
        """
        Share resources with child communicator during split.

        Returns:
            ``bool``: True to share resources. Default: False.
        """
        return bool(self._cfg.split_share)

    @split_share.setter
    def split_share(self, val: bool) -> None:
        if not isinstance(val, bool):
            raise NcclInvalid(f"split_share must be bool, got {type(val).__name__}")
        self._cfg.split_share = int(val)

    @property
    def traffic_class(self) -> int:
        """
        Traffic class for network operations.

        Returns:
            ``int``: Traffic class (>= 0). Meaning is network-specific.
        """
        return self._cfg.traffic_class

    @traffic_class.setter
    def traffic_class(self, val: int) -> None:
        if not isinstance(val, int):
            raise NcclInvalid(f"traffic_class must be int, got {type(val).__name__}")
        if val < 0:
            raise NcclInvalid(f"traffic_class must be >= 0, got {val}")
        self._cfg.traffic_class = val

    @property
    def comm_name(self) -> str:
        """
        User-defined communicator name for logging and profiling.

        Returns:
            ``str``: Communicator name.
        """
        return self._cfg.comm_name

    @comm_name.setter
    def comm_name(self, val: str) -> None:
        if not isinstance(val, str):
            raise NcclInvalid(f"comm_name must be str, got {type(val).__name__}")
        self._cfg.comm_name = val

    @property
    def collnet_enable(self) -> bool:
        """
        Enable IB SHARP.

        Returns:
            ``bool``: True to enable. Default: False.
        """
        return bool(self._cfg.collnet_enable)

    @collnet_enable.setter
    def collnet_enable(self, val: bool) -> None:
        if not isinstance(val, bool):
            raise NcclInvalid(f"collnet_enable must be bool, got {type(val).__name__}")
        self._cfg.collnet_enable = int(val)

    @property
    def cta_policy(self) -> CTAPolicy:
        """
        CTA scheduling policy.

        Returns:
            ``CTAPolicy``: CTA policy enum. Default: CTAPolicy.Default.
        """
        return CTAPolicy(self._cfg.cta_policy)

    @cta_policy.setter
    def cta_policy(self, val: int | CTAPolicy) -> None:
        if not isinstance(val, (int, CTAPolicy)):
            raise NcclInvalid(f"cta_policy must be int or CTAPolicy, got {type(val).__name__}")
        self._cfg.cta_policy = int(val)

    @property
    def shrink_share(self) -> bool:
        """
        Share resources with child communicator during shrink.

        Returns:
            ``bool``: True to share resources. Default: False.
        """
        return bool(self._cfg.shrink_share)

    @shrink_share.setter
    def shrink_share(self, val: bool) -> None:
        if not isinstance(val, bool):
            raise NcclInvalid(f"shrink_share must be bool, got {type(val).__name__}")
        self._cfg.shrink_share = int(val)

    @property
    def nvls_ctas(self) -> int:
        """
        Total number of CTAs for NVLS kernels.

        Returns:
            ``int``: Positive integer. Auto-determined by default.
        """
        return self._cfg.nvls_ctas

    @nvls_ctas.setter
    def nvls_ctas(self, val: int) -> None:
        if not isinstance(val, int):
            raise NcclInvalid(f"nvls_ctas must be int, got {type(val).__name__}")
        if val <= 0:
            raise NcclInvalid(f"nvls_ctas must be a positive integer, got {val}")
        self._cfg.nvls_ctas = val

    @property
    def n_channels_per_net_peer(self) -> int:
        """
        Number of network channels for pairwise communication.

        Returns:
            ``int``: Positive integer, rounded up to power of 2.
        """
        return self._cfg.n_channels_per_net_peer

    @n_channels_per_net_peer.setter
    def n_channels_per_net_peer(self, val: int) -> None:
        if not isinstance(val, int):
            raise NcclInvalid(f"n_channels_per_net_peer must be int, got {type(val).__name__}")
        if val <= 0:
            raise NcclInvalid(f"n_channels_per_net_peer must be a positive integer, got {val}")
        self._cfg.n_channels_per_net_peer = val

    @property
    def nvlink_centric_sched(self) -> bool:
        """
        Enable NVLink-centric scheduling.

        Returns:
            ``bool``: True to enable. Default: False.
        """
        return bool(self._cfg.nvlink_centric_sched)

    @nvlink_centric_sched.setter
    def nvlink_centric_sched(self, val: bool) -> None:
        if not isinstance(val, bool):
            raise NcclInvalid(f"nvlink_centric_sched must be bool, got {type(val).__name__}")
        self._cfg.nvlink_centric_sched = int(val)


class Communicator:
    """
    NCCL Communicator for collective and point-to-point operations.

    A Communicator represents a group of ranks that can perform collective
    operations (like reduce, broadcast) and point-to-point operations (send/recv).
    Each rank in the communicator has a unique ID (0 to nranks-1).

    Attributes:
        ptr (int): Raw NCCL communicator pointer (0 if destroyed)
        nranks (int): Total number of ranks in this communicator
        device (Device): CUDA device object associated with this communicator
        rank (int): This rank's ID within the communicator
    """

    def __init__(self, ptr: int) -> None:
        """
        Initializes communicator with a raw NCCL pointer.

        Args:
            - ptr (int): Integer representing NCCL communicator pointer (0 for sentinel/invalid).

        Raises:
            - ``NcclInvalid``: If ptr is not an integer.

        Notes:
            Unlike the class method ``init()``, this constructor allows ptr=0 for
            creating sentinel communicators (e.g., when ``split()`` excludes a rank).
        """
        self._comm: int = int(ptr)
        self._resources: list[CommResource] = []

        self._nranks = int(_nccl_bindings.comm_count(self._comm)) if ptr != 0 else None
        self._device = Device(int(_nccl_bindings.comm_cu_device(self._comm))) if ptr != 0 else None
        self._rank = int(_nccl_bindings.comm_user_rank(self._comm)) if ptr != 0 else None

    def _check_valid(self, operation: str) -> None:
        """
        Checks if communicator is valid for the given operation.

        Args:
            - operation (str): Name of the operation being performed.

        Raises:
            - ``NcclInvalid``: If communicator is not initialized (ptr == 0).
        """
        if self._comm == 0:
            raise NcclInvalid(f"Cannot {operation}: Communicator not initialized")

    def _validate_buffer_device(self, buffer: NcclBuffer, buffer_name: str = "buffer") -> None:
        """
        Validates that buffer is on the same device as the communicator.

        Args:
            - buffer (NcclBuffer): Resolved buffer to validate.
            - buffer_name (str): Name of the buffer for error messages.

        Raises:
            - ``NcclInvalid``: If buffer device does not match communicator device.
        """
        if buffer.device_id != self.device.device_id:
            raise NcclInvalid(
                f"{buffer_name} is on device {buffer.device_id}, but communicator "
                f"is on device {self.device.device_id}. Buffers must be on the same "
                f"device as the communicator."
            )

    def __repr__(self) -> str:
        """
        Returns string representation of the communicator.

        Returns:
            ``str``: String showing rank/count/device info if valid, or invalid status.
        """
        if self._comm == 0:
            return "<Communicator: invalid (ptr=0)>"
        try:
            return f"<Communicator: rank={self.rank}/{self.nranks}, device={self.device.device_id}, ptr={self._comm:#x}>"
        except RuntimeError:
            # If we can't get properties, just show the pointer
            return f"<Communicator: ptr={self._comm:#x}>"

    @classmethod
    def init(
        cls,
        nranks: int,
        rank: int,
        unique_id: UniqueId | Sequence[UniqueId],
        config: NCCLConfig | None = None,
    ) -> Communicator:
        """
        Initializes a new NCCL communicator.

        Creates a communicator that connects multiple ranks. All ranks must
        call this method with the same nranks and unique_id, but with different rank values.

        Args:
            - nranks (int): Total number of ranks in the communicator.
            - rank (int): This rank (must be between 0 and nranks-1).
            - unique_id (UniqueId | Sequence[UniqueId]): Unique identifier(s) shared by all ranks.
            - config (NCCLConfig, optional): NCCL configuration options. Defaults to None.

        Returns:
            ``Communicator``: A new communicator instance.

        Raises:
            - ``NcclInvalid``: If unique_id has an invalid type.

        Notes:
            - This is a collective operation. All ranks must call this method.
            - See [ncclCommInitRankScalable](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/comms.html#ncclcomminitrankscalable) for when multiple unique_ids are used.
        """
        cfg_ptr = 0 if config is None else config.ptr
        if isinstance(unique_id, UniqueId):
            comm_ptr = _nccl_bindings.comm_init_rank_scalable(
                int(nranks), int(rank), 1, unique_id.ptr, cfg_ptr
            )
        elif isinstance(unique_id, Sequence) and all(
            isinstance(uid, UniqueId) for uid in unique_id
        ):
            # Pack a sequence of UniqueId wrappers into a single C-side UniqueId buffer
            arr = _np.empty(len(unique_id), dtype=_nccl_bindings.unique_id_dtype)
            for i, uid in enumerate(unique_id):
                # Defensive copy to protect against internal structure changes
                arr[i] = uid.as_ndarray[0].copy()
            packed = _nccl_bindings.UniqueId.from_data(arr)
            comm_ptr = _nccl_bindings.comm_init_rank_scalable(
                int(nranks), int(rank), int(len(unique_id)), packed.ptr, cfg_ptr
            )
        else:
            raise NcclInvalid("unique_id must be a UniqueId or a sequence of UniqueIds")

        comm = cls(comm_ptr)
        # reassign the values in case init() is called inside a group
        comm._nranks = int(nranks)
        comm._device = get_cuda_device()
        comm._rank = int(rank)
        return comm

    # --- Communicator APIs ---
    def split(self, color: int, key: int, config: NCCLConfig | None = None) -> Communicator:
        """
        Splits this communicator into sub-communicators based on color values.

        Ranks which pass the same color value will be part of the same group. If color is
        NCCL_SPLIT_NOCOLOR, the rank will not be part of any group and receives a communicator with ptr=0.
        The key value determines rank ordering; smaller key means smaller rank in the new communicator.
        If keys are equal between ranks, the rank in the original communicator determines ordering.

        Args:
            - color (int): Non-negative color value for grouping ranks (use NCCL_SPLIT_NOCOLOR to exclude this rank).
            - key (int): Rank ordering key within each color group (smaller key = smaller rank).
            - config (NCCLConfig, optional): Configuration for the new communicator. If None, inherits parent's configuration. Defaults to None.

        Returns:
            ``Communicator``: New sub-communicator, or sentinel communicator (ptr=0) if color is NCCL_SPLIT_NOCOLOR.

        Raises:
            - ``NcclInvalid``: If communicator is not initialized or has outstanding operations.

        Notes:
            - This is a collective operation. All ranks in the communicator must call this method.
            - There must not be any outstanding NCCL operations on the communicator to avoid deadlock.

        See Also:
            https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/comms.html#ncclcommsplit
        """
        self._check_valid("split")

        if color == NCCL_SPLIT_NOCOLOR:
            # Return a sentinel communicator instead of None for consistent API
            return Communicator(0)

        cfg_ptr = 0 if config is None else config.ptr
        comm_ptr = _nccl_bindings.comm_split(self._comm, int(color), int(key), cfg_ptr)

        return Communicator(comm_ptr)

    def shrink(
        self,
        exclude_ranks: Sequence[int] | None = None,
        config: NCCLConfig | None = None,
        flag: CommShrinkFlag = CommShrinkFlag.Default,
    ) -> Communicator:
        """
        Creates a new communicator by removing specified ranks from the existing communicator.

        Ranks listed in exclude_ranks will be excluded from the new communicator, and ranks within
        the new communicator will be updated to maintain a contiguous set of IDs.

        Args:
            - exclude_ranks (Sequence[int], optional): Ranks to exclude from the new communicator. Defaults to None (no exclusions).
            - config (NCCLConfig, optional): Configuration for the new communicator. If None, inherits parent's configuration. Defaults to None.
            - flag (CommShrinkFlag, optional): Shrink behavior flag. Use CommShrinkFlag.Default for normal operation or CommShrinkFlag.Abort after errors. Defaults to CommShrinkFlag.Default.

        Returns:
            ``Communicator``: New communicator without the excluded ranks.

        Raises:
            - ``NcclInvalid``: If communicator is not initialized.

        Notes:
            - This is a collective operation. All non-excluded ranks must call this method.
            - Excluded ranks must NOT call this function.
            - With CommShrinkFlag.Default: There must not be outstanding NCCL operations to avoid deadlock.
            - With CommShrinkFlag.Default + config.shrink_share=True: Parent communicator resources are reused.
            - With CommShrinkFlag.Abort: Automatically aborts outstanding operations; no resources shared with parent.

        See Also:
            https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/comms.html#ncclcommshrink
        """
        self._check_valid("shrink")
        ranks_to_exclude = list(exclude_ranks) if exclude_ranks is not None else []
        cfg_ptr = 0 if config is None else config.ptr
        comm_ptr = _nccl_bindings.comm_shrink(
            self._comm, ranks_to_exclude, len(ranks_to_exclude), cfg_ptr, int(flag)
        )

        return Communicator(comm_ptr)

    def destroy(self) -> None:
        """
        Destroys the communicator and frees local resources allocated to it.

        This function only frees local resources if ``finalize()`` was previously called;
        otherwise, ``destroy()`` will call ``finalize()`` internally. If ``finalize()``
        is called explicitly, users must ensure the communicator state becomes ncclSuccess
        before calling ``destroy()``. The communicator should not be accessed after
        ``destroy()`` returns.

        Args:
            None

        Raises:
            None (errors during cleanup are suppressed for safety)

        Notes:
            - All resources (registered buffers, windows, custom operators) owned by this
              communicator are automatically closed before destruction.
            - This is an intra-node collective call - all ranks on the same node must call it to avoid hanging.
            - Recommended pattern: Call ``finalize()`` then ``destroy()``.

        See Also:
            https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/comms.html#ncclcommdestroy
        """
        # Close all resources first (best-effort, ignore errors)
        self.close_all_resources()

        if self._comm == 0:
            return

        _nccl_bindings.comm_destroy(self._comm)
        self._comm = 0

    def abort(self) -> None:
        """
        Aborts the communicator and frees resources, terminating all uncompleted operations.

        This should be called when an unrecoverable error occurs. All active ranks are
        required to call this function in order to abort the NCCL communicator successfully.

        Args:
            None

        Raises:
            None (errors during cleanup are suppressed for safety)

        Notes:
            - All resources (registered buffers, windows, custom operators) owned by this
              communicator are automatically closed before aborting.
            - Unlike ``destroy()``, this immediately aborts uncompleted operations.
            - All active ranks must call this function to abort successfully.
            - For more details, see the Fault Tolerance section in NCCL documentation.

        See Also:
            https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/comms.html#ncclcommabort
        """
        # Close all resources first (best-effort, ignore errors)
        self.close_all_resources()

        if self._comm == 0:
            return

        _nccl_bindings.comm_abort(self._comm)
        self._comm = 0

    def finalize(self) -> None:
        """
        Finalizes the communicator, flushing all uncompleted operations and network resources.

        When the communicator is nonblocking, this is a nonblocking function. Successful return
        sets the communicator state to ncclInProgress, indicating finalization is in progress.
        Once all NCCL operations complete, the communicator transitions to ncclSuccess state.
        Users can query the state with ``get_async_error()``.

        Args:
            None

        Returns:
            None

        Notes:
            - This is typically called before ``destroy()`` to ensure all operations complete.
            - For nonblocking communicators, check completion status with ``get_async_error()``.
            - This is a collective operation that must be called by all ranks.

        See Also:
            https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/comms.html#ncclcommfinalize
        """
        if self._comm == 0:
            return

        _nccl_bindings.comm_finalize(self._comm)

    # --- Properties ---
    @property
    def ptr(self) -> int:
        """
        Raw NCCL communicator pointer (0 if destroyed/invalid).

        Returns:
            ``int``: The communicator raw pointer.
        """
        return self._comm

    @property
    def is_valid(self) -> bool:
        """
        Checks if the communicator is valid.

        Returns:
            ``bool``: True if valid, False if destroyed or invalid.
        """
        return self._comm != 0

    @property
    def nranks(self) -> int:
        """
        Total number of ranks in the communicator.

        Returns:
            ``int``: Number of ranks.

        Raises:
            - ``NcclInvalid``: If communicator is not initialized.

        See Also:
            https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/comms.html#ncclcommcount
        """
        self._check_valid("get nranks")
        if self._nranks is None:
            self._nranks = int(_nccl_bindings.comm_count(self._comm))
        return self._nranks

    @property
    def device(self) -> Device:
        """
        CUDA device object associated with this communicator.

        Returns:
            ``Device``: CUDA device object from cuda.core.experimental.

        Raises:
            - ``NcclInvalid``: If communicator is not initialized.

        See Also:
            https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/comms.html#ncclcommcudevice
        """
        self._check_valid("get device")
        if self._device is None:
            self._device = get_cuda_device()
        return self._device

    @property
    def rank(self) -> int:
        """
        This caller's rank within the communicator.

        Returns:
            ``int``: This rank's ID (0 to nranks-1).

        Raises:
            - ``NcclInvalid``: If communicator is not initialized.

        See Also:
            https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/comms.html#ncclcommuserrank
        """
        self._check_valid("get rank")
        if self._rank is None:
            self._rank = int(_nccl_bindings.comm_user_rank(self._comm))
        return self._rank

    # --- Point-to-Point Communication ---
    def send(
        self, sendbuf: NcclBufferSpec, peer: int, *, stream: NcclStreamSpec | None = None
    ) -> None:
        """
        Sends a buffer to a peer rank using this communicator.

        Args:
            - sendbuf (NcclBufferSpec): Source buffer to send.
            - peer (int): Destination rank ID.
            - stream (NcclStreamSpec, optional): CUDA stream for the operation. Defaults to None (uses default stream).

        Raises:
            - ``NcclInvalid``: If the buffer specification is invalid, buffer is on wrong device, or communicator is not initialized.
        """
        self._check_valid("send")
        s = NcclBuffer(sendbuf)
        self._validate_buffer_device(s, "sendbuf")

        _nccl_bindings.send(
            s.ptr, s.count, int(s.dtype), int(peer), int(self._comm), get_stream_ptr(stream)
        )

    def recv(
        self, recvbuf: NcclBufferSpec, peer: int, *, stream: NcclStreamSpec | None = None
    ) -> None:
        """
        Receives data into a buffer from a peer rank using this communicator.

        Args:
            - recvbuf (NcclBufferSpec): Destination buffer to receive into.
            - peer (int): Source rank ID.
            - stream (NcclStreamSpec, optional): CUDA stream for the operation. Defaults to None (uses default stream).

        Raises:
            - ``NcclInvalid``: If the buffer specification is invalid, buffer is on wrong device, or communicator is not initialized.
        """
        self._check_valid("recv")
        r = NcclBuffer(recvbuf)
        self._validate_buffer_device(r, "recvbuf")

        _nccl_bindings.recv(
            r.ptr, r.count, int(r.dtype), int(peer), int(self._comm), get_stream_ptr(stream)
        )

    # --- Collective Communication Operations ---
    def allreduce(
        self,
        sendbuf: NcclBufferSpec,
        recvbuf: NcclBufferSpec,
        op: NcclRedOp | CustomRedOp,
        stream: NcclStreamSpec | None = None,
    ) -> None:
        """
        Reduces data arrays of length count in sendbuf using the specified operation and leaves identical copies of the result in each recvbuf.

        All ranks receive the same reduced result in their receive buffers after this collective operation completes.

        This is a shortcut for ``reduce(sendbuf, recvbuf, op, root=None, stream=stream)``.

        Args:
            - sendbuf (NcclBufferSpec): Source buffer specification containing data to be reduced.
            - recvbuf (NcclBufferSpec): Destination buffer specification that will receive the reduced result.
            - op (NcclRedOp | CustomRedOp): Reduction operator to apply (e.g., SUM, MAX, MIN, AVG, PROD, or custom operator).
            - stream (NcclStreamSpec, optional): CUDA stream for the operation. Defaults to None (uses default stream).

        Raises:
            - ``NcclInvalid``: If send and receive buffers have mismatched dtypes, mismatched counts, buffers on wrong device, invalid buffer specifications, or communicator is not initialized.

        Notes:
            - Both send and receive buffers must have matching data types.
            - Element count is inferred from the sendbuf specification: count = sendcount.
            - Requires recvcount >= sendcount.
            - In-place operation occurs when sendbuf and recvbuf resolve to the same device memory address.

        See Also:
            https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/colls.html#ncclallreduce
        """
        self._check_valid("allreduce")

        self.reduce(sendbuf, recvbuf, op, stream=stream)

    def broadcast(
        self,
        sendbuf: NcclBufferSpec | Any,
        recvbuf: NcclBufferSpec,
        root: int,
        *,
        stream: NcclStreamSpec | None = None,
    ) -> None:
        """
        Copies count elements from sendbuf on the root rank to all ranks' recvbuf.

        The sendbuf is only used on the root rank and is ignored for other ranks.

        Args:
            - sendbuf (NcclBufferSpec | Any): Source buffer specification (only used on root rank).
            - recvbuf (NcclBufferSpec): Destination buffer specification that will receive the broadcast data.
            - root (int): Root rank that broadcasts the data (must be between 0 and nranks-1).
            - stream (NcclStreamSpec, optional): CUDA stream for the operation. Defaults to None (uses default stream).

        Raises:
            - ``NcclInvalid``: If send and receive buffers have mismatched dtypes, mismatched counts, buffers on wrong device, invalid buffer specifications, or communicator is not initialized.

        Notes:
            - On root rank, both send and receive buffers must have matching data types.
            - Element count is inferred from the recvbuf specification: count = recvcount.
            - On root rank, requires sendcount == recvcount.
            - In-place operation occurs when sendbuf and recvbuf resolve to the same device memory address.

        See Also:
            https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/colls.html#ncclbroadcast
        """
        self._check_valid("broadcast")

        s, r = None, None
        if root == self.rank:
            s = NcclBuffer(sendbuf)
            self._validate_buffer_device(s, "sendbuf")
        r = NcclBuffer(recvbuf)
        self._validate_buffer_device(r, "recvbuf")

        if s is not None:
            if s.dtype != r.dtype:
                raise NcclInvalid(
                    f"Dtype mismatch: sendbuf has dtype {s.dtype}, recvbuf has dtype {r.dtype}"
                )
            if r.count != s.count:
                raise NcclInvalid(
                    f"Buffer count mismatch: recvbuf must have exactly {s.count} elements, got {r.count}"
                )

        s_ptr = s.ptr if s is not None else 0
        r_ptr = r.ptr
        count = r.count
        dtype = r.dtype

        _nccl_bindings.broadcast(
            s_ptr, r_ptr, count, int(dtype), int(root), int(self._comm), get_stream_ptr(stream)
        )

    def reduce(
        self,
        sendbuf: NcclBufferSpec,
        recvbuf: NcclBufferSpec | Any,
        op: NcclRedOp | CustomRedOp,
        root: int | None = None,
        *,
        stream: NcclStreamSpec | None = None,
    ) -> None:
        """
        Reduces data arrays of length count in sendbuf using the specified operation.

        This method supports two modes of operation:

        1. **AllReduce Mode** (root=None): Reduces data and leaves identical copies of the result in each rank's recvbuf.
           All ranks receive the same reduced result after the collective operation completes.

        2. **Reduce Mode** (root specified): Reduces data and places the result only in recvbuf on the specified root rank.
           The recvbuf is only used on the root rank and is ignored for other ranks.

        Args:
            - sendbuf (NcclBufferSpec): Source buffer specification containing data to be reduced.
            - recvbuf (NcclBufferSpec | Any): Destination buffer specification that will receive the reduced result.
              In Reduce Mode (root specified), only used on root rank.
            - op (NcclRedOp | CustomRedOp): Reduction operator to apply (e.g., SUM, MAX, MIN, AVG, PROD, or custom operator).
            - root (int | None, optional): Root rank that receives the reduced result (must be between 0 and nranks-1).
              If None, performs an all-reduce where all ranks receive the result. Defaults to None.
            - stream (NcclStreamSpec, optional): CUDA stream for the operation. Defaults to None (uses default stream).

        Raises:
            - ``NcclInvalid``: If send and receive buffers have mismatched dtypes, mismatched counts, buffers on wrong device, invalid buffer specifications, or communicator is not initialized.

        Notes:
            - Both send and receive buffers must have matching data types if receive buffer is used.
            - Element count is inferred from the sendbuf specification: count = sendcount.
            - In All-Reduce Mode: All ranks must have recvcount >= sendcount.
            - In Reduce Mode: Only root rank requires recvcount >= sendcount.
            - In-place operation occurs when sendbuf and recvbuf resolve to the same device memory address.

        See Also:
            - All-Reduce Mode: https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/colls.html#ncclallreduce
            - Reduce Mode: https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/colls.html#ncclreduce
        """
        self._check_valid("reduce")

        s, r = NcclBuffer(sendbuf), None
        self._validate_buffer_device(s, "sendbuf")
        if root is None or root == self.rank:
            r = NcclBuffer(recvbuf)
            self._validate_buffer_device(r, "recvbuf")

        if r is not None:
            if s.dtype != r.dtype:
                raise NcclInvalid(
                    f"Dtype mismatch: sendbuf has dtype {s.dtype}, recvbuf has dtype {r.dtype}"
                )
            if r.count < s.count:
                raise NcclInvalid(
                    f"Buffer count mismatch: recvbuf must have at least {s.count} elements, got {r.count}"
                )

        s_ptr = s.ptr
        r_ptr = r.ptr if r is not None else 0
        count = s.count
        dtype = s.dtype

        if root is None:
            _nccl_bindings.all_reduce(
                s_ptr, r_ptr, count, int(dtype), int(op), int(self._comm), get_stream_ptr(stream)
            )
        else:
            _nccl_bindings.reduce(
                s_ptr,
                r_ptr,
                count,
                int(dtype),
                int(op),
                int(root),
                int(self._comm),
                get_stream_ptr(stream),
            )

    def allgather(
        self,
        sendbuf: NcclBufferSpec,
        recvbuf: NcclBufferSpec,
        stream: NcclStreamSpec | None = None,
    ) -> None:
        """
        Gathers sendcount values from all ranks and leaves identical copies of the result in each recvbuf, receiving data from rank i at offset i*sendcount.

        All ranks receive the same concatenated result containing data from all ranks.

        This is a shortcut for ``gather(sendbuf, recvbuf, root=None, stream=stream)``.

        Args:
            - sendbuf (NcclBufferSpec): Source buffer specification containing sendcount elements.
            - recvbuf (NcclBufferSpec): Destination buffer specification (must have size at least nranks*sendcount elements).
            - stream (NcclStreamSpec, optional): CUDA stream for the operation. Defaults to None (uses default stream).

        Raises:
            - ``NcclInvalid``: If send and receive buffers have mismatched dtypes, recvbuf is too small, buffers on wrong device, invalid buffer specifications, or communicator is not initialized.

        Notes:
            - Both send and receive buffers must have matching data types.
            - Element count is inferred from the sendbuf specification: count = sendcount.
            - Requires recvcount >= nranks * sendcount.
            - Data from rank i is placed at recvbuf + i*sendcount.
            - In-place operation occurs when sendbuf resolves to device memory address: recvbuf_address + rank*sendcount.

        See Also:
            https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/colls.html#ncclallgather
        """
        self._check_valid("allgather")

        self.gather(sendbuf, recvbuf, stream=stream)

    def reduce_scatter(
        self,
        sendbuf: NcclBufferSpec,
        recvbuf: NcclBufferSpec,
        op: NcclRedOp | CustomRedOp,
        *,
        stream: NcclStreamSpec | None = None,
    ) -> None:
        """
        Reduces data in sendbuf from all ranks using the specified operation and leaves the reduced result scattered over the devices so that recvbuf on rank i contains the i-th block of the result.

        Each rank receives a different portion of the reduced result.

        Args:
            - sendbuf (NcclBufferSpec): Source buffer specification (must have size at least nranks*recvcount elements).
            - recvbuf (NcclBufferSpec): Destination buffer specification containing recvcount elements.
            - op (NcclRedOp | CustomRedOp): Reduction operator to apply (e.g., SUM, MAX, MIN, AVG, PROD, or custom operator).
            - stream (NcclStreamSpec, optional): CUDA stream for the operation. Defaults to None (uses default stream).

        Raises:
            - ``NcclInvalid``: If send and receive buffers have mismatched dtypes, sendbuf is too small, buffers on wrong device, invalid buffer specifications, or communicator is not initialized.

        Notes:
            - Both send and receive buffers must have matching data types.
            - Element count is inferred from the sendbuf specification: count = sendcount / nranks.
            - Requires sendcount >= nranks and recvcount >= count.
            - Rank i receives the i-th block of the reduced result in its recvbuf.
            - In-place operation occurs when recvbuf resolves to device memory address: sendbuf_address + rank*count.

        See Also:
            https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/colls.html#ncclreducescatter
        """
        self._check_valid("reduce_scatter")

        s, r = NcclBuffer(sendbuf), NcclBuffer(recvbuf)
        self._validate_buffer_device(s, "sendbuf")
        self._validate_buffer_device(r, "recvbuf")

        if s.dtype != r.dtype:
            raise NcclInvalid(
                f"Dtype mismatch: sendbuf has dtype {s.dtype}, recvbuf has dtype {r.dtype}"
            )
        per_rank_count = s.count // self.nranks
        if per_rank_count < 1:
            raise NcclInvalid(
                f"Buffer count mismatch: sendbuf must have at least {self.nranks} elements (nranks), got {s.count}"
            )
        if r.count < per_rank_count:
            raise NcclInvalid(
                f"Buffer count mismatch: recvbuf must have at least {per_rank_count} elements (sendcount / nranks), got {r.count}"
            )

        s_ptr = s.ptr
        r_ptr = r.ptr
        count = per_rank_count
        dtype = s.dtype

        _nccl_bindings.reduce_scatter(
            s_ptr, r_ptr, count, int(dtype), int(op), int(self._comm), get_stream_ptr(stream)
        )

    def alltoall(
        self,
        sendbuf: NcclBufferSpec,
        recvbuf: NcclBufferSpec,
        *,
        stream: NcclStreamSpec | None = None,
    ) -> None:
        """
        Each rank sends count values to all other ranks and receives count values from all other ranks.

        Data to send to destination rank j is taken from sendbuf+j*count and data received from source rank i is placed at recvbuf+i*count.

        Args:
            - sendbuf (NcclBufferSpec): Source buffer specification (must have size at least nranks*count elements).
            - recvbuf (NcclBufferSpec): Destination buffer specification (must have size at least nranks*count elements).
            - stream (NcclStreamSpec, optional): CUDA stream for the operation. Defaults to None (uses default stream).

        Raises:
            - ``NcclInvalid``: If send and receive buffers have mismatched dtypes, buffer sizes incompatible with nranks, buffers on wrong device, invalid buffer specifications, or communicator is not initialized.

        Notes:
            - Both send and receive buffers must have matching data types.
            - Element count is inferred from the sendbuf specification: count = sendcount / nranks.
            - Requires sendcount >= nranks and recvcount >= sendcount.
            - Data sent to rank j is at sendbuf + j*count and data received from rank i is at recvbuf + i*count.

        See Also:
            https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/colls.html#ncclalltoall
        """
        self._check_valid("alltoall")

        s, r = NcclBuffer(sendbuf), NcclBuffer(recvbuf)
        self._validate_buffer_device(s, "sendbuf")
        self._validate_buffer_device(r, "recvbuf")

        if s.dtype != r.dtype:
            raise NcclInvalid(
                f"Dtype mismatch: sendbuf has dtype {s.dtype}, recvbuf has dtype {r.dtype}"
            )
        per_rank_count = s.count // self.nranks
        if per_rank_count < 1:
            raise NcclInvalid(
                f"Buffer count mismatch: sendbuf must have at least {self.nranks} elements (nranks), got {s.count}"
            )
        if r.count < s.count:
            raise NcclInvalid(
                f"Buffer count mismatch: recvbuf must have at least {s.count} elements (nranks * count), got {r.count}"
            )

        s_ptr = s.ptr
        r_ptr = r.ptr
        count = per_rank_count
        dtype = s.dtype

        _nccl_bindings.allto_all(
            s_ptr, r_ptr, count, int(dtype), int(self._comm), get_stream_ptr(stream)
        )

    def gather(
        self,
        sendbuf: NcclBufferSpec,
        recvbuf: NcclBufferSpec | Any,
        root: int | None = None,
        *,
        stream: NcclStreamSpec | None = None,
    ) -> None:
        """
        Gathers sendcount values from all ranks.

        This method supports two modes of operation:

        1. **AllGather Mode** (root=None): Gathers values from all ranks and leaves identical copies of the result in each recvbuf.
           All ranks receive the same concatenated result containing data from all ranks.

        2. **Gather Mode** (root specified): Gathers values from all ranks to the specified root rank.
           The recvbuf is only used on the root rank and is ignored for other ranks.

        Args:
            - sendbuf (NcclBufferSpec): Source buffer specification containing sendcount elements.
            - recvbuf (NcclBufferSpec | Any): Destination buffer specification (must have size at least nranks*sendcount elements).
              In Gather Mode (root specified), only used on root rank.
            - root (int | None, optional): Root rank that receives the gathered data (must be between 0 and nranks-1).
              If None, performs an all-gather where all ranks receive the result. Defaults to None.
            - stream (NcclStreamSpec, optional): CUDA stream for the operation. Defaults to None (uses default stream).

        Raises:
            - ``NcclInvalid``: If send and receive buffers have mismatched dtypes, recvbuf is too small, buffers on wrong device, invalid buffer specifications, or communicator is not initialized.

        Notes:
            - Both send and receive buffers must have matching data types if receive buffer is used.
            - Element count is inferred from the sendbuf specification: count = sendcount.
            - In AllGather Mode: All ranks must have recvcount >= nranks * sendcount.
            - In Gather Mode: Only root rank requires recvcount >= nranks * sendcount.
            - Data from rank i is placed at recvbuf + i*sendcount.
            - In AllGather Mode, in-place operation occurs when sendbuf resolves to device memory address: recvbuf_address + rank*sendcount.
            - In Gather Mode, in-place operation occurs when sendbuf resolves to device memory address: recvbuf_address + root*sendcount.

        See Also:
            - AllGather Mode: https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/colls.html#ncclallgather
            - Gather Mode: https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/colls.html#ncclgather
        """
        self._check_valid("gather")

        s, r = NcclBuffer(sendbuf), None
        self._validate_buffer_device(s, "sendbuf")
        if root is None or root == self.rank:
            r = NcclBuffer(recvbuf)
            self._validate_buffer_device(r, "recvbuf")

        if r is not None:
            if r.dtype != s.dtype:
                raise NcclInvalid(
                    f"Dtype mismatch: sendbuf has dtype {s.dtype}, recvbuf has dtype {r.dtype}"
                )
            expected_recv_count = self.nranks * s.count
            if r.count < expected_recv_count:
                raise NcclInvalid(
                    f"Buffer count mismatch: recvbuf must have at least {expected_recv_count} elements (nranks * sendcount), got {r.count}"
                )

        s_ptr = s.ptr
        r_ptr = r.ptr if r is not None else 0
        count = s.count
        dtype = s.dtype

        if root is None:
            _nccl_bindings.all_gather(
                s_ptr, r_ptr, count, int(dtype), int(self._comm), get_stream_ptr(stream)
            )
        else:
            _nccl_bindings.gather(
                s_ptr, r_ptr, count, int(dtype), int(root), int(self._comm), get_stream_ptr(stream)
            )

    def scatter(
        self,
        sendbuf: NcclBufferSpec | Any,
        recvbuf: NcclBufferSpec,
        root: int,
        *,
        stream: NcclStreamSpec | None = None,
    ) -> None:
        """
        Each rank receives count elements from the root rank. On the root rank, count elements from sendbuf + i*count are sent to rank i.

        On non-root ranks, sendbuf is not used.

        Args:
            - sendbuf (NcclBufferSpec | Any): Source buffer specification (only used on root rank, must have size at least nranks*count elements).
            - recvbuf (NcclBufferSpec): Destination buffer specification containing count elements.
            - root (int): Root rank that scatters the data (must be between 0 and nranks-1).
            - stream (NcclStreamSpec, optional): CUDA stream for the operation. Defaults to None (uses default stream).

        Raises:
            - ``NcclInvalid``: If send and receive buffers have mismatched dtypes, sendbuf is too small on root rank, buffers on wrong device, invalid buffer specifications, or communicator is not initialized.

        Notes:
            - On root rank, both send and receive buffers must have matching data types.
            - Element count is inferred from the recvbuf specification: count = recvcount.
            - On root rank, requires sendcount >= nranks and sendcount / nranks == recvcount.
            - On root rank, data at sendbuf + i*count is sent to rank i.
            - In-place operation occurs when recvbuf resolves to device memory address: sendbuf_address + root*count.

        See Also:
            https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/colls.html#ncclscatter
        """
        self._check_valid("scatter")

        s, r = None, None
        if root == self.rank:
            s = NcclBuffer(sendbuf)
            self._validate_buffer_device(s, "sendbuf")
        r = NcclBuffer(recvbuf)
        self._validate_buffer_device(r, "recvbuf")

        if s is not None:
            if s.dtype != r.dtype:
                raise NcclInvalid(
                    f"Dtype mismatch: sendbuf has dtype {s.dtype}, recvbuf has dtype {r.dtype}"
                )
            per_rank_count = s.count // self.nranks
            if per_rank_count < 1:
                raise NcclInvalid(
                    f"Buffer count mismatch: sendbuf must have at least {self.nranks} elements (nranks), got {s.count}"
                )
            if r.count != per_rank_count:
                raise NcclInvalid(
                    f"Buffer count mismatch: recvbuf must have exactly {per_rank_count} elements (sendcount / nranks), got {r.count}"
                )

        s_ptr = s.ptr if s is not None else 0
        r_ptr = r.ptr
        count = r.count
        dtype = r.dtype

        _nccl_bindings.scatter(
            s_ptr, r_ptr, count, int(dtype), int(root), int(self._comm), get_stream_ptr(stream)
        )

    # --- Registration ---
    def register_buffer(self, buffer: NcclBufferSpec) -> RegisteredBufferHandle:
        """
        Registers a buffer with this communicator for zero-copy communication.

        Registered buffers can enable performance optimizations in NCCL operations.
        The returned RegisteredBufferHandle must be explicitly closed when no longer needed.

        Args:
            - buffer (NcclBufferSpec): Buffer to register (array, Buffer, or buffer-like object).

        Returns:
            ``RegisteredBufferHandle``: Resource handle that can be closed manually or automatically when the communicator is destroyed / aborted.

        Raises:
            - ``NcclInvalid``: If buffer is on wrong device or communicator is not initialized.

        Notes:
            - Buffer size is automatically derived from buffer count and dtype.

        See Also:
            https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/comms.html#ncclcommregister
        """
        self._check_valid("register_buffer")

        nccl_buf = NcclBuffer(buffer)
        self._validate_buffer_device(nccl_buf, "buffer")
        buffer_ptr = nccl_buf.ptr
        size = nccl_buf.count * nccl_buf.dtype.itemsize

        resource = RegisteredBufferHandle(self._comm, buffer_ptr, size)
        self._resources.append(resource)
        return resource

    def register_window(
        self, buffer: NcclBufferSpec, flags: WindowFlag | None = None
    ) -> RegisteredWindowHandle | None:
        """
        Collectively registers a local buffer into an NCCL window for optimized communication.

        Since this is a collective call, every rank in the communicator must participate,
        and buffer size must be equal among ranks by default. Windows enable optimized
        communication patterns in NCCL.

        Args:
            - buffer (NcclBufferSpec): Local buffer to register as window.
            - flags (WindowFlag, optional): Window registration flags to control behavior. Defaults to None.

        Returns:
            ``RegisteredWindowHandle``: Resource handle that can be closed manually or
            automatically when the communicator is destroyed / aborted, or ``None`` if
            NCCL returns a NULL handle (e.g., window unsupported on a platform).

        Raises:
            - ``NcclInvalid``: If buffer is on wrong device or communicator is not initialized.

        Notes:
            - This is a collective operation. All ranks in the communicator must call this method.
            - Buffer sizes must be equal among ranks by default.
            - Buffer size is automatically derived from buffer count and dtype.
            - If called within a group, the handle value may not be filled until ncclGroupEnd() completes.

        See Also:
            https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/comms.html#ncclcommwindowregister
        """
        self._check_valid("register_window")

        nccl_buf = NcclBuffer(buffer)
        self._validate_buffer_device(nccl_buf, "buffer")
        buffer_ptr = nccl_buf.ptr
        size = nccl_buf.count * nccl_buf.dtype.itemsize

        resource = RegisteredWindowHandle(self._comm, buffer_ptr, size, flags)
        if resource.handle == 0:
            return None

        self._resources.append(resource)
        return resource

    # --- Custom Reduction APIs ---
    def create_pre_mul_sum(
        self,
        scalar: NcclScalarSpec,
        datatype: NcclDataType | None = None,
    ) -> CustomRedOp:
        """
        Creates a PreMulSum custom reduction operator.

        The PreMulSum operator performs: output = scalar * sum(inputs).
        This is useful for averaging (scalar = 1/N) or weighted reductions.
        The returned CustomRedOp must be explicitly closed when no longer needed.

        Args:
            scalar: Scalar multiplier value. Can be:
                - Python int/float: Converted to NumPy array, uses host memory
                - NumPy array: Must contain exactly 1 element, uses host memory
                - NcclSupportedBuffer: Device buffer
            datatype: NCCL data type of the scalar and reduction. If None, inferred from scalar.
                - For Python int/float: inferred from NumPy's natural dtype (int64/float64)
                - For NumPy array: inferred from array dtype
                - For device buffer: inferred from buffer dtype

        Returns:
            CustomRedOp resource that can be closed manually or automatically when the communicator is destroyed / aborted.

        Raises:
            - ``NcclInvalid``: If communicator is not initialized, scalar type is not supported, NumPy array or buffer doesn't contain exactly 1 element, or datatype cannot be inferred or is incompatible.
        """
        self._check_valid("create_pre_mul_sum")

        # Determine residence and prepare scalar pointer
        scalar_array = None  # Will hold host scalar to prevent GC
        residence: _nccl_bindings.ScalarResidence

        if isinstance(scalar, (int, float)):
            # Python scalar: Convert to NumPy array in host memory
            residence = _nccl_bindings.ScalarResidence.HostImmediate

            if datatype is None:
                scalar_array = _np.array([scalar])
                datatype = NcclDataType(scalar_array.dtype)
            else:
                scalar_array = _np.array([scalar], dtype=datatype.numpy_dtype)
            scalar_ptr = scalar_array.ctypes.data
        elif isinstance(scalar, _np.ndarray):
            # NumPy array: Host memory
            residence = _nccl_bindings.ScalarResidence.HostImmediate

            # Validate array has exactly 1 element
            if scalar.size != 1:
                raise NcclInvalid(
                    f"NumPy array must contain exactly 1 element for scalar, got {scalar.size} elements"
                )

            # Ensure contiguous array
            scalar_array = _np.ascontiguousarray(scalar.ravel())
            scalar_ptr = scalar_array.ctypes.data

            if datatype is None:
                datatype = NcclDataType(scalar_array.dtype)
        else:
            # Assume it's NcclSupportedBuffer (device buffer)
            # Use NcclBuffer to handle Buffer, DLPack, CAI, etc.
            residence = _nccl_bindings.ScalarResidence.Device

            try:
                buf = NcclBuffer(scalar)
            except (TypeError, ValueError) as e:
                raise NcclInvalid(
                    f"scalar must be int, float, numpy.ndarray, or NcclSupportedBuffer, "
                    f"got {type(scalar).__name__}: {e}"
                ) from e

            # Validate buffer contains exactly 1 element
            if buf.count != 1:
                raise NcclInvalid(
                    f"Device buffer must contain exactly 1 element for scalar, got {buf.count} elements"
                )

            # Infer datatype from buffer if not provided
            if datatype is None:
                datatype = buf.dtype
            else:
                # Validate buffer datatype matches requested datatype
                if buf.dtype != datatype:
                    raise NcclInvalid(
                        f"Device buffer datatype {buf.dtype} doesn't match requested datatype {datatype}"
                    )

            scalar_ptr = buf.ptr

        # Create the custom reduction operator
        resource = CustomRedOp(self._comm, scalar_ptr, datatype, residence)

        # Keep scalar_array alive by storing in resource to prevent premature GC
        # (only needed for host scalars; device buffers are managed by user)
        if scalar_array is not None:
            resource._scalar_array = scalar_array

        self._resources.append(resource)
        return resource

    def close_all_resources(self) -> None:
        """
        Closes all resources owned by this communicator.

        This method is called automatically during ``destroy()`` and ``abort()`` but can
        be called manually if needed. It performs best-effort cleanup, ignoring
        any errors that occur during resource deallocation.

        Notes:
            This method is idempotent - calling it multiple times is safe.
        """
        for resource in self._resources:
            try:
                resource.close()
            except Exception:
                # Best-effort cleanup - ignore errors to avoid masking
                # user exceptions or preventing communicator destruction
                pass
        self._resources.clear()

    # --- Miscellaneous ---
    def get_last_error(self) -> str:
        """
        Gets the last error string for this communicator.

        Returns:
            ``str``: Error message string.

        Raises:
            - ``NcclInvalid``: If communicator is not initialized.
        """
        self._check_valid("get last error")
        return _nccl_bindings.get_last_error(self._comm)

    def get_async_error(self) -> _nccl_bindings.Result:
        """
        Queries the progress and potential errors of asynchronous NCCL operations.

        Operations without a stream argument (e.g., finalize) are complete when they return ncclSuccess.
        Operations with a stream argument (e.g., reduce) return ncclSuccess when posted but may
        report errors through this method until completed. If any NCCL function returns ncclInProgress,
        users must query communicator state until it becomes ncclSuccess before calling another NCCL function.

        Returns:
            ``Result``: Current state of the communicator (ncclSuccess, ncclInProgress, or error code).

        Raises:
            - ``NcclInvalid``: If communicator is not initialized.

        Notes:
            - Before state becomes ncclSuccess, do not issue CUDA kernels on streams used by NCCL.
            - If an error occurs, destroy the communicator with ``abort()``.
            - Nothing can be assumed about completion or correctness of enqueued operations after an error.

        See Also:
            https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/comms.html#ncclcommgetasyncerror
        """
        self._check_valid("get async error")
        return _nccl_bindings.comm_get_async_error(self._comm)
