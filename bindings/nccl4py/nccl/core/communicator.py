# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# See LICENSE.txt for more license information

"""NCCL communicator creation, management, and operations.

This module provides the core Communicator class, NCCLConfig for communicator
initialization options, and NCCLDevCommRequirements for device-side
communicator configuration. Communicators manage groups of ranks for
collective and point-to-point communication, with support for buffer
registration, custom reduction operators, and resource management.
"""

from __future__ import annotations
from dataclasses import field
from typing import Any, Sequence

import numpy as _np

from cuda.core import Device
from cuda.core import system

from nccl.bindings import nccl as _nccl_bindings

from nccl._binding_helpers import binding_dataclass
from nccl.core.buffer import NcclBuffer
from nccl.core.constants import (
    NCCL_SPLIT_NOCOLOR,
    CTAPolicy,
    CommShrinkFlag,
    CommSuspendFlag,
    WindowFlag,
)
from nccl.core.cuda import get_stream_ptr
from nccl.core.resources import (
    CommResource,
    RegisteredBufferHandle,
    RegisteredWindowHandle,
    CustomRedOp,
    DevCommResource,
)
from nccl.core.typing import (
    NcclDataType,
    NcclBufferSpec,
    NcclRedOp,
    NcclGinType,
    NcclGinConnectionType,
    NcclStreamSpec,
    NcclScalarSpec,
    NcclInvalid,
    NcclCommMemStat,
)
from nccl.core.utils import UniqueId

_PointerBox = _nccl_bindings.PointerBox
_Result = _nccl_bindings.Result


__all__ = [
    "NCCLConfig",
    "WaitSignalDesc",
    "NCCLDevCommRequirements",
    "Communicator",
]


@binding_dataclass(_nccl_bindings.Config)
class NCCLConfig:
    """NCCL configuration for communicator initialization.

    Provides configuration options for NCCL communicators, allowing
    fine-tuning of performance and behavior characteristics. Fields not set
    in the constructor remain at NCCL's internal default; values are
    validated by the C library when the config is consumed.

    See Also:
        :c:type:`ncclConfig_t` for the description of each field.
    """

    blocking: bool | None = None
    """Blocking (True) or non-blocking (False) communicator behavior. If unset, NCCL uses True."""

    cga_cluster_size: int | None = None
    """Cooperative Group Array (CGA) size for kernels (0-8). If unset, NCCL uses 4 for sm90+, 0 otherwise."""

    min_ctas: int | None = None
    """Minimum number of CTAs per kernel; positive integer up to 32. If unset, NCCL uses 1."""

    max_ctas: int | None = None
    """Maximum number of CTAs per kernel; positive integer up to 32. If unset, NCCL uses 32."""

    net_name: str | None = None
    """Network module name (e.g. 'IB', 'Socket'). Case-insensitive. If unset, NCCL auto-selects."""

    split_share: bool | None = None
    """Share resources with the child communicator during split. If unset, NCCL uses False."""

    traffic_class: int | None = None
    """Traffic class (TC) for network operations (>= 0). Network-specific meaning."""

    comm_name: str | None = None
    """User-defined communicator name for logging and profiling."""

    collnet_enable: bool | None = None
    """Enable (True) or disable (False) IB SHARP. If unset, NCCL uses False."""

    cta_policy: CTAPolicy | None = None
    """CTA scheduling policy. If unset, NCCL uses CTAPolicy.DEFAULT."""

    shrink_share: bool | None = None
    """Share resources with the child communicator during shrink. If unset, NCCL uses False."""

    nvls_ctas: int | None = None
    """Total number of CTAs for NVLS kernels (positive integer). If unset, NCCL auto-determines."""

    n_channels_per_net_peer: int | None = None
    """Number of network channels for pairwise communication. Positive integer, rounded up to power of 2. If unset, NCCL uses an AlltoAll-optimized value."""

    nvlink_centric_sched: bool | None = None
    """Enable NVLink-centric scheduling. If unset, NCCL uses False."""

    graph_usage_mode: int | None = None
    """Graph usage mode (NCCL 2.29+). Supported values are 0 (no graphs), 1 (one graph), 2 (multiple graphs or mix of graph and non-graph). If unset, NCCL uses 2."""

    num_rma_ctx: int | None = None
    """Number of RMA contexts (NCCL 2.29+). Positive integer. If unset, NCCL uses 1."""

    max_p2p_peers: int | None = None
    """Maximum number of peers any rank will concurrently communicate with using P2P (NCCL 2.30+). Positive integer. If unset, NCCL uses the communicator size."""

    graph_stream_ordering: int | None = None
    """Whether NCCL preserves stream-ordering semantics for collectives captured into CUDA graphs. Supported values are 0 (disabled) or 1 (enabled). Cannot be combined with ``graph_usage_mode=2``. Also controllable via the ``NCCL_GRAPH_STREAM_ORDERING`` environment variable. If unset, NCCL uses 1."""


@binding_dataclass(_nccl_bindings.WaitSignalDesc, kw_only=False, frozen=True)
class WaitSignalDesc:
    """Descriptor for a wait-signal operation.

    Describes a single signal-wait operation for use with
    :py:meth:`Communicator.wait_signal`. Each descriptor specifies which peer
    to wait for, how many signal operations to wait for, and additional
    context for the wait operation.
    """

    peer: int
    """Target peer rank to wait for signals from."""

    op_count: int = field(default=1, metadata={"lowpp": "op_cnt"})
    """Number of signal operations to wait for from the peer. Defaults to 1."""

    signal_index: int = field(default=0, metadata={"lowpp": "sig_idx"})
    """Signal index identifier. Currently must be 0. Defaults to 0."""

    context: int = field(default=0, metadata={"lowpp": "ctx"})
    """Context identifier. Currently must be 0. Defaults to 0."""


@binding_dataclass(_nccl_bindings.DevCommRequirements)
class NCCLDevCommRequirements:
    """NCCL device communicator requirements configuration.

    Provides configuration for device communicator creation, allowing
    fine-tuning of resource allocation and device-side communication
    behavior. Fields not set in the constructor remain at NCCL's internal
    default; values are validated by the C library when the requirements
    are consumed by :py:meth:`Communicator.create_dev_comm`.

    See Also:
        :c:type:`ncclDevCommRequirements` for the description of each field.
    """

    lsa_multimem: bool | None = None
    """Enable multimem on the LSA team. If unset, NCCL uses False."""

    barrier_count: int | None = None
    """Number of barriers required. If unset, NCCL uses 0."""

    lsa_barrier_count: int | None = None
    """Number of LSA barriers. If unset, NCCL uses 0."""

    rail_gin_barrier_count: int | None = None
    """Number of railed GIN barriers. If unset, NCCL uses 0."""

    lsa_ll_a2a_block_count: int | None = None
    """LSA low-latency all-to-all block count. If unset, NCCL uses 0."""

    lsa_ll_a2a_slot_count: int | None = None
    """LSA low-latency all-to-all slot count. If unset, NCCL uses 0."""

    gin_force_enable: bool | None = None
    """Force-enable GPU Interconnect Network. If unset, NCCL uses False."""

    gin_context_count: int | None = None
    """Number of GIN contexts (hint; actual count may differ). If unset, NCCL uses 4."""

    gin_signal_count: int | None = None
    """Number of GIN signals (guaranteed to start at id=0). If unset, NCCL uses 0."""

    gin_counter_count: int | None = None
    """Number of GIN counters (guaranteed to start at id=0). If unset, NCCL uses 0."""

    gin_connection_type: NcclGinConnectionType | None = None
    """GIN connection type. If unset, NCCL uses NcclGinConnectionType.NONE."""

    gin_exclusive_contexts: bool | None = None
    """Use exclusive GIN contexts. If unset, NCCL uses False."""

    gin_queue_depth: int | None = None
    """GIN queue depth. If unset, NCCL uses 0."""

    world_gin_barrier_count: int | None = None
    """Number of world GIN barriers. If unset, NCCL uses 0."""

    gin_strong_signals_required: bool | None = None
    """Whether GIN strong signals are required by kernels using this devComm.
    When False, using GIN strong signals results in undefined behavior. If unset, NCCL uses True."""

    gin_va_signals_required: bool | None = None
    """Whether GIN VA signals are required by kernels using this devComm.
    When False, using GIN VA signals results in undefined behavior. If unset, NCCL uses True."""


class Communicator:
    """NCCL communicator for collective and point-to-point operations.

    A communicator represents a group of ranks that can perform collective
    operations (e.g. allreduce, broadcast) and point-to-point operations
    (send/recv). Each rank has a unique ID in ``[0, nranks)``.

    Communicator instances expose a number of properties for inspection
    (``ptr``, ``nranks``, ``device``, ``rank``, plus device-API related
    properties like ``cuda_dev``, ``nvml_dev``, ``device_api_support``,
    ``multimem_support``, ``gin_type``, ``n_lsa_teams``,
    ``host_rma_support``, ``railed_gin_type``); see the per-property
    documentation for details.
    """

    def __init__(self, ptr: int = 0) -> None:
        """Initializes a communicator with a raw NCCL pointer.

        Unlike the :py:meth:`init` classmethod, this constructor allows
        ``ptr=0`` for creating null communicators (e.g. when :py:meth:`split`
        excludes a rank). A null communicator can later be initialized via
        :py:meth:`initialize`, or used as the caller for :py:meth:`grow` to
        join an existing communicator.

        Args:
            ptr: Integer representing an NCCL communicator pointer (0 for a
                null communicator). Defaults to 0.
        """
        if isinstance(ptr, _PointerBox):
            self._comm_box = ptr
        else:
            self._comm_box = _PointerBox(ptr)
        self._resources: list[CommResource] = []
        self._nranks: int | None = None
        self._device: Device | None = None
        self._rank: int | None = None
        self._comm_properties: _nccl_bindings.CommProperties | None = None
        self._children_in_progress = []

    def _check_valid(self, operation: str) -> None:
        """Raises if the communicator is not initialized.

        Args:
            operation: Name of the operation being attempted (for the error
                message).

        Raises:
            NcclInvalid: If the communicator is not initialized (ptr == 0).
        """
        if self._comm == 0:
            raise NcclInvalid(f"Cannot {operation}: Communicator not initialized")

    def _validate_buffer_device(self, buffer: NcclBuffer, buffer_name: str = "buffer") -> None:
        """Validates that the buffer is on the same device as the communicator.

        Args:
            buffer: Resolved buffer to validate.
            buffer_name: Name of the buffer for error messages.

        Raises:
            NcclInvalid: If the buffer device does not match the communicator
                device.
        """
        if buffer.device_id != self.device.device_id:
            raise NcclInvalid(
                f"{buffer_name} is on device {buffer.device_id}, but communicator "
                f"is on device {self.device.device_id}. Buffers must be on the same "
                f"device as the communicator."
            )

    def _get_comm_properties(self) -> _nccl_bindings.CommProperties:
        """Queries and caches communicator properties.

        Returns:
            Cached CommProperties object.

        Raises:
            NcclInvalid: If the communicator is not initialized.
        """
        self._check_valid("query properties")
        if self._comm_properties is None:
            self._comm_properties = _nccl_bindings.comm_query_properties(self._comm)
        return self._comm_properties

    def __repr__(self) -> str:
        if self._comm == 0:
            return "<Communicator: null (ptr=0)>"
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
        """Initializes a new NCCL communicator.

        Creates a communicator that connects multiple ranks. This is a
        collective operation: all ranks must call this method with the same
        ``nranks`` and ``unique_id`` but with different ``rank`` values.

        Args:
            nranks: Total number of ranks in the communicator.
            rank: This rank (must be between 0 and ``nranks - 1``).
            unique_id: Unique identifier(s) shared by all ranks. A sequence
                may be passed to use :c:func:`ncclCommInitRankScalable`.
            config: NCCL configuration options. Defaults to ``None``.

        Returns:
            A new Communicator instance.

        Raises:
            NcclInvalid: If ``unique_id`` has an invalid type.
        """
        comm = cls()
        comm.initialize(nranks, rank, unique_id, config)
        return comm

    @classmethod
    def init_all(
        cls,
        devices: int | Sequence[int] | None = None,
    ) -> list[Communicator]:
        """Initializes multiple NCCL communicators for single-process multi-GPU operations.

        Creates an array of NCCL communicators, one for each device, within a
        single process. This is optimized for single-machine scenarios where
        all GPUs are controlled by the same process. Unlike :py:meth:`init`,
        which requires multi-process coordination (e.g. via MPI),
        :py:meth:`init_all` handles all coordination internally.

        Each communicator is bound to its corresponding device and has its
        rank equal to its index in the returned list. The current device
        context is preserved by the underlying NCCL API. All communicators
        must be manually destroyed via :py:meth:`destroy` on each one.

        Args:
            devices: Specifies which devices to initialize. ``None`` (the
                default) initializes all visible CUDA devices. An int
                creates communicators for devices ``[0, 1, ..., devices - 1]``.
                A sequence of ints uses the explicit device IDs. If the
                resulting device list is empty (``devices=0``, an empty
                sequence, or no visible devices), returns an empty list
                without calling into NCCL.

        Returns:
            List of initialized communicators, one per device. Rank ``i``
            uses ``devices[i]`` (or device ``i`` when ``devices`` is an
            int).

        Raises:
            TypeError: If ``devices`` is not an int, sequence of ints, or
                ``None``.
        """
        # Parse devices parameter
        if devices is None:
            devlist = list(range(system.get_num_devices()))
        elif isinstance(devices, int):
            devlist = list(range(devices))
        elif isinstance(devices, (list, tuple, range)):
            devlist = list(devices)
        else:
            raise TypeError(
                f"devices must be an integer, sequence of integers, or None, got {type(devices).__name__}"
            )

        if not devlist:
            return []

        ndev = len(devlist)

        # Call NCCL binding to initialize all communicators
        # Note: ncclCommInitAll preserves the current device internally
        # The binding returns a Cython array containing communicator pointers
        comm_array = _nccl_bindings.comm_init_all(ndev, devlist)

        return [cls(int(comm_ptr)) for comm_ptr in comm_array]

    def initialize(
        self,
        nranks: int,
        rank: int,
        unique_id: UniqueId | Sequence[UniqueId],
        config: NCCLConfig | None = None,
    ) -> None:
        """Initializes this communicator in-place.

        Instance-method counterpart of the :py:meth:`init` classmethod.
        Allows creating a null communicator first (via ``Communicator()``)
        and initializing it later. This is a collective operation; all ranks
        must call this method.

        Args:
            nranks: Total number of ranks in the communicator.
            rank: This rank (must be between 0 and ``nranks - 1``).
            unique_id: Unique identifier(s) shared by all ranks.
            config: NCCL configuration options. Defaults to ``None``.

        Raises:
            NcclInvalid: If ``unique_id`` has an invalid type or this
                communicator is already initialized.
        """
        if self._comm != 0:
            raise NcclInvalid("Communicator is already initialized")

        cfg_ptr = 0 if config is None else config._lowpp.ptr  # type: ignore[attr-defined]
        if isinstance(unique_id, UniqueId):
            unique_id = (unique_id,)
        elif not isinstance(unique_id, (list, tuple)):
            raise NcclInvalid("unique_id must be a UniqueId or a sequence of UniqueIds")

        commIds = bytearray().join(bytes(uid) for uid in unique_id)
        _nccl_bindings.comm_init_rank_scalable(
            self._comm_box.address, int(nranks), int(rank), int(len(unique_id)), commIds, cfg_ptr
        )

        self._resources = []
        self._nranks = None
        self._device = None
        self._rank = None
        self._comm_properties = None

    # --- Communicator APIs ---
    def split(
        self, color: int | None = None, key: int = 0, config: NCCLConfig | None = None
    ) -> Communicator:
        """Splits this communicator into sub-communicators based on color values.

        Ranks that pass the same ``color`` value will be part of the same
        group. If ``color`` is ``None``, the rank will not be part of any
        group and receives a null communicator (a
        :py:class:`Communicator` instance with ``ptr=0``). The ``key``
        value determines rank ordering; smaller ``key`` means smaller rank
        in the new communicator. If keys are equal, the rank in the
        original communicator determines ordering.

        This is a collective operation: all ranks in the communicator must
        call this method, even ranks that pass ``color=None``. There must
        be no outstanding NCCL operations on the communicator to avoid
        deadlock.

        Args:
            color: Non-negative color value for grouping ranks. Pass
                ``None`` to exclude this rank from all groups. Defaults to
                ``None``.
            key: Ordering key within the color group. Defaults to 0.
            config: Configuration for the new communicator. If ``None``,
                inherits the parent's configuration. Defaults to ``None``.

        Returns:
            New sub-communicator, or a null communicator if ``color`` is
            ``None``.

        Raises:
            NcclInvalid: If the communicator is not initialized.

        See Also:
            :c:func:`ncclCommSplit`
        """
        self._check_valid("split")

        if color is None:
            color = NCCL_SPLIT_NOCOLOR
        cfg_ptr = 0 if config is None else config._lowpp.ptr  # type: ignore[attr-defined]
        newcomm = type(self)()
        self._children_in_progress.append(newcomm)
        _nccl_bindings.comm_split(
            self._comm, int(color), int(key), newcomm._comm_box.address, cfg_ptr
        )

        return newcomm

    def shrink(
        self,
        exclude_ranks: Sequence[int] | None = None,
        config: NCCLConfig | None = None,
        flag: CommShrinkFlag = CommShrinkFlag.DEFAULT,
    ) -> Communicator:
        """Creates a new communicator by removing specified ranks from this one.

        Ranks listed in ``exclude_ranks`` are excluded from the new
        communicator; the remaining ranks are renumbered to a contiguous
        ``[0, n)`` range.

        This is a collective operation. All non-excluded ranks must call this
        method; excluded ranks must NOT call it. With
        :py:attr:`~nccl.core.CommShrinkFlag.DEFAULT` there must be no
        outstanding NCCL operations to avoid deadlock; combine with
        ``config.shrink_share=True`` to reuse parent communicator resources.
        With :py:attr:`~nccl.core.CommShrinkFlag.ABORT` outstanding
        operations are automatically aborted and no resources are shared
        with the parent.

        Args:
            exclude_ranks: Ranks to exclude from the new communicator.
                Defaults to ``None`` (no exclusions).
            config: Configuration for the new communicator. If ``None``,
                inherits the parent's configuration. Defaults to ``None``.
            flag: Shrink behavior. Use
                :py:attr:`~nccl.core.CommShrinkFlag.DEFAULT` for normal
                operation or :py:attr:`~nccl.core.CommShrinkFlag.ABORT`
                after errors. Defaults to
                :py:attr:`~nccl.core.CommShrinkFlag.DEFAULT`.

        Returns:
            New communicator without the excluded ranks.

        Raises:
            NcclInvalid: If the communicator is not initialized.

        See Also:
            :c:func:`ncclCommShrink`
        """
        self._check_valid("shrink")
        ranks_to_exclude = list(exclude_ranks) if exclude_ranks is not None else []
        cfg_ptr = 0 if config is None else config._lowpp.ptr  # type: ignore[attr-defined]
        newcomm = type(self)()
        _nccl_bindings.comm_shrink(
            self._comm,
            ranks_to_exclude,
            len(ranks_to_exclude),
            newcomm._comm_box.address,
            cfg_ptr,
            int(flag),
        )

        return newcomm

    def get_unique_id(self) -> UniqueId:
        """Returns a per-communicator unique ID for use with :py:meth:`grow`.

        Generates a unique identifier bound to this communicator that can be
        shared with new ranks joining via :py:meth:`grow`. This is distinct
        from the global :py:func:`~nccl.core.get_unique_id` used for initial
        communicator creation. Only one existing rank (the grow root)
        should call this method.

        A new UID cannot be generated while a previous UID is unconsumed;
        each UID can be used only once and the user must wait for the
        corresponding grow operation to complete before calling again.

        Returns:
            :py:class:`~nccl.core.UniqueId` for grow operations.

        Raises:
            NcclInvalid: If the communicator is not initialized.
        """
        self._check_valid("get_unique_id")
        return UniqueId(_nccl_bindings.comm_get_unique_id(self._comm))

    def grow(
        self,
        nranks: int,
        unique_id: UniqueId | None = None,
        rank: int | None = None,
        config: NCCLConfig | None = None,
    ) -> Communicator:
        """Grows the communicator by adding new ranks.

        Creates a new communicator that includes both existing ranks from
        this communicator and new ranks joining the group. There are three
        roles:

            - Existing root: the one existing rank that called
              :py:meth:`get_unique_id`.
            - Existing non-root: all other existing ranks.
            - New ranks: ranks joining via a null communicator
              (``Communicator()``).

        This is a collective operation. All ranks (existing and new) must
        call this method. Usage by role:

            - Existing root: ``new_comm = existing_comm.grow(nranks, uid)``
            - Existing non-root: ``new_comm = existing_comm.grow(nranks)``
            - New rank: ``new_comm = Communicator().grow(nranks, uid, rank=assigned_rank)``

        The UID is consumed upon successful grow and cannot be reused.

        Args:
            nranks: Total number of ranks in the new communicator (existing
                plus new). All roles must pass the same value.
            unique_id: Unique identifier from :py:meth:`get_unique_id`.
                Existing root and new ranks must pass the
                :py:class:`~nccl.core.UniqueId`; existing non-root must pass
                ``None``. Defaults to ``None``.
            rank: This rank's ID in the new communicator. New ranks must
                pass their assigned rank, which must be ``>=`` the parent
                communicator size. Existing ranks must pass ``None``.
                Defaults to ``None``.
            config: Configuration for the new communicator. Defaults to ``None``.

        Returns:
            New :py:class:`Communicator` containing all ranks.

        Raises:
            NcclInvalid: If a new rank is given an initialized communicator,
                or an existing rank is given a null communicator.
        """
        is_new_rank = rank is not None
        if is_new_rank and self._comm != 0:
            raise NcclInvalid("New ranks must use a null communicator (Communicator())")
        if not is_new_rank and self._comm == 0:
            raise NcclInvalid("Existing ranks must use an initialized communicator")

        uid_ptr = 0 if unique_id is None else unique_id.ptr
        rank_val = -1 if rank is None else int(rank)
        cfg_ptr = 0 if config is None else config._lowpp.ptr  # type: ignore[attr-defined]
        newcomm = type(self)()
        _nccl_bindings.comm_grow(
            self._comm, int(nranks), uid_ptr, rank_val, newcomm._comm_box.address, cfg_ptr
        )

        return newcomm

    def destroy(self) -> None:
        """Destroys the communicator and frees local resources.

        If :py:meth:`finalize` has not been called explicitly,
        :py:meth:`destroy` will call it internally. If :py:meth:`finalize` is
        called explicitly, users must ensure the communicator state becomes
        ``ncclSuccess`` before calling :py:meth:`destroy`. The communicator
        should not be accessed after :py:meth:`destroy` returns.

        All resources (registered buffers, windows, custom operators) owned
        by this communicator are automatically closed before destruction.
        This is an intra-node collective call: all ranks on the same node
        must call it to avoid hanging. The recommended pattern is
        :py:meth:`finalize` followed by :py:meth:`destroy`.

        Errors during cleanup are suppressed for safety.

        See Also:
            :c:func:`ncclCommDestroy`
        """
        # Close all resources first (best-effort, ignore errors)
        self.close_all_resources()

        if self._comm == 0:
            return

        _nccl_bindings.comm_destroy(self._comm)
        self._comm_box.ptr = 0

    def abort(self) -> None:
        """Aborts the communicator and frees resources, terminating in-flight operations.

        Should be called when an unrecoverable error occurs. Unlike
        :py:meth:`destroy`, this immediately aborts uncompleted operations.
        All active ranks must call this function in order to abort the NCCL
        communicator successfully.

        All resources (registered buffers, windows, custom operators) owned
        by this communicator are automatically closed before aborting.
        Errors during cleanup are suppressed for safety. For more details,
        see the Fault Tolerance section in the NCCL documentation.

        See Also:
            :c:func:`ncclCommAbort`
        """
        # Close all resources first (best-effort, ignore errors)
        self.close_all_resources()

        if self._comm == 0:
            return

        _nccl_bindings.comm_abort(self._comm)
        self._comm_box.ptr = 0

    def finalize(self) -> None:
        """Finalizes the communicator, flushing uncompleted operations and network resources.

        Typically called before :py:meth:`destroy` to ensure all operations
        complete. This is a collective operation that must be called by all
        ranks.

        For nonblocking communicators this is itself nonblocking: success
        sets the communicator state to ``ncclInProgress`` to indicate
        finalization is in progress. Once all NCCL operations complete, the
        communicator transitions to ``ncclSuccess``. Users can query the
        state with :py:meth:`get_async_error`.

        See Also:
            :c:func:`ncclCommFinalize`
        """
        if self._comm == 0:
            return

        _nccl_bindings.comm_finalize(self._comm)

    def revoke(self, flags: int = 0) -> None:
        """Revokes the communicator.

        Stops all in-flight operations and marks the communicator state as
        ``ncclInProgress``. The state transitions to ``ncclSuccess`` when
        the communicator becomes quiescent, after which management
        operations (:py:meth:`destroy`, :py:meth:`split`, :py:meth:`shrink`)
        can proceed safely.

        Calling :py:meth:`finalize` after :py:meth:`revoke` is invalid.
        Resource sharing via split-share / shrink-share is disabled while
        revoked.

        Args:
            flags: Reserved for future use. Currently must be 0.

        Raises:
            NcclInvalid: If the communicator is not initialized.
        """
        self._check_valid("revoke")
        _nccl_bindings.comm_revoke(self._comm, flags)

    def suspend(self, flags: CommSuspendFlag = CommSuspendFlag.MEM) -> None:
        """Suspends communicator operations to free resources.

        The communicator cannot be used for communication while suspended.
        Call :py:meth:`resume` to restore it.

        Args:
            flags: Suspend flags controlling what resources to release.
                :py:attr:`~nccl.core.CommSuspendFlag.MEM` releases dynamic
                GPU memory allocations.

        Raises:
            NcclInvalid: If the communicator is not initialized.
        """
        self._check_valid("suspend")
        _nccl_bindings.comm_suspend(self._comm, int(flags))

    def resume(self) -> None:
        """Resumes all previously suspended communicator resources.

        Restores a communicator that was suspended with :py:meth:`suspend`
        so that it can be used for communication again.

        Raises:
            NcclInvalid: If the communicator is not initialized.
        """
        self._check_valid("resume")
        _nccl_bindings.comm_resume(self._comm)

    # --- Properties ---
    @property
    def _comm(self) -> int:
        return int(self._comm_box.ptr)

    @property
    def ptr(self) -> int:
        """Raw pointer to the underlying :c:type:`ncclComm_t` structure
        (0 if destroyed or null).
        """
        return self._comm

    @property
    def is_valid(self) -> bool:
        """Whether the communicator is valid (not destroyed or null)."""
        return self._comm != 0

    @property
    def nranks(self) -> int:
        """Total number of ranks in the communicator.

        Raises:
            NcclInvalid: If the communicator is not initialized.
        """
        self._check_valid("get nranks")
        if self._nranks is None:
            self._nranks = int(_nccl_bindings.comm_count(self._comm))
        return self._nranks

    @property
    def device(self) -> Device:
        """CUDA device associated with this communicator.

        Returns a :py:class:`cuda.core.Device` providing additional
        functionality such as ``to_system_device`` for obtaining the NVML
        device, device properties, and synchronization.

        Raises:
            NcclInvalid: If the communicator is not initialized.
        """
        self._check_valid("get device")
        if self._device is None:
            self._device = Device(int(_nccl_bindings.comm_cu_device(self._comm)))
        return self._device

    @property
    def rank(self) -> int:
        """This caller's rank within the communicator (0 to nranks - 1).

        Raises:
            NcclInvalid: If the communicator is not initialized.
        """
        self._check_valid("get rank")
        if self._rank is None:
            self._rank = int(_nccl_bindings.comm_user_rank(self._comm))
        return self._rank

    @property
    def cuda_dev(self) -> int:
        """CUDA device ID associated with this communicator.

        Raises:
            NcclInvalid: If the communicator is not initialized.

        """
        self._check_valid("get cuda_dev")
        return self._get_comm_properties().cuda_dev

    @property
    def nvml_dev(self) -> int:
        """NVML device ID for the GPU associated with this communicator.

        Uses the NVML indexing space, which may differ from CUDA indexing.

        Raises:
            NcclInvalid: If the communicator is not initialized.

        """
        self._check_valid("get nvml_dev")
        return self._get_comm_properties().nvml_dev

    @property
    def device_api_support(self) -> bool:
        """Whether device-side NCCL operations are supported on this platform.

        If False, a device communicator cannot be created.

        Raises:
            NcclInvalid: If the communicator is not initialized.

        """
        self._check_valid("get device_api_support")
        return bool(self._get_comm_properties().device_api_support)

    @property
    def multimem_support(self) -> bool:
        """Whether ranks in the same LSA team can communicate using multimem.

        If False, a device communicator cannot be created with multimem
        resources.

        Raises:
            NcclInvalid: If the communicator is not initialized.

        """
        self._check_valid("get multimem_support")
        return bool(self._get_comm_properties().multimem_support)

    @property
    def gin_type(self) -> NcclGinType:
        """GPU Interconnect Network (GIN) type.

        If equal to NcclGinType.NONE, a device communicator cannot be
        created with GIN resources.

        Raises:
            NcclInvalid: If the communicator is not initialized.

        """
        self._check_valid("get gin_type")
        return NcclGinType(self._get_comm_properties().gin_type)

    @property
    def n_lsa_teams(self) -> int:
        """Number of Local Shared Array (LSA) teams for this communicator.

        Raises:
            NcclInvalid: If the communicator is not initialized.

        """
        self._check_valid("get n_lsa_teams")
        return self._get_comm_properties().n_lsa_teams

    @property
    def host_rma_support(self) -> bool:
        """Whether host RMA is supported on this communicator.

        Raises:
            NcclInvalid: If the communicator is not initialized.

        """
        self._check_valid("get host_rma_support")
        return bool(self._get_comm_properties().host_rma_support)

    @property
    def railed_gin_type(self) -> NcclGinType:
        """Railed GIN type supported by this communicator.

        If equal to :py:attr:`NcclGinType.NONE`, a device communicator
        cannot be created with GIN connection type
        :py:attr:`NcclGinConnectionType.RAIL`.

        Raises:
            NcclInvalid: If the communicator is not initialized.

        """
        self._check_valid("get railed_gin_type")
        return NcclGinType(self._get_comm_properties().railed_gin_type)

    # --- Point-to-Point Communication ---
    def send(
        self, sendbuf: NcclBufferSpec, peer: int, *, stream: NcclStreamSpec | None = None
    ) -> None:
        """Sends a buffer to a peer rank.

        Args:
            sendbuf: Source buffer to send.
            peer: Destination rank ID.
            stream: CUDA stream for the operation. Defaults to ``None`` (the
                default stream).

        Raises:
            NcclInvalid: If the buffer specification is invalid, the buffer
                is on the wrong device, or the communicator is not
                initialized.

        See Also:
            :c:func:`ncclSend`
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
        """Receives data into a buffer from a peer rank.

        Args:
            recvbuf: Destination buffer to receive into.
            peer: Source rank ID.
            stream: CUDA stream for the operation. Defaults to ``None`` (the
                default stream).

        Raises:
            NcclInvalid: If the buffer specification is invalid, the buffer
                is on the wrong device, or the communicator is not
                initialized.

        See Also:
            :c:func:`ncclRecv`
        """
        self._check_valid("recv")
        r = NcclBuffer(recvbuf)
        self._validate_buffer_device(r, "recvbuf")

        _nccl_bindings.recv(
            r.ptr, r.count, int(r.dtype), int(peer), int(self._comm), get_stream_ptr(stream)
        )

    def wait_signal(
        self,
        descs: WaitSignalDesc | Sequence[WaitSignalDesc],
        *,
        stream: NcclStreamSpec | None = None,
    ) -> None:
        """Waits for signals as described by the signal descriptor(s).

        Enqueues a wait operation on the specified CUDA stream that blocks
        until the required signals from peer ranks are received. Each
        descriptor specifies a peer rank and the number of signal
        operations to wait for from that peer.

        Args:
            descs: One or more :py:class:`WaitSignalDesc` descriptors
                specifying which peers to wait for and how many signals to
                expect from each.
            stream: CUDA stream to enqueue the wait operation on. Defaults
                to ``None`` (the default stream).

        Raises:
            NcclInvalid: If the communicator is not initialized.

        See Also:
            :c:func:`ncclWaitSignal`
        """
        self._check_valid("wait_signal")

        if isinstance(descs, WaitSignalDesc):
            descs = (descs,)

        buf = bytearray().join(bytes(d._lowpp) for d in descs)  # type: ignore[attr-defined]
        _nccl_bindings.wait_signal(len(descs), buf, int(self._comm), get_stream_ptr(stream))

    def signal(
        self,
        peer: int,
        signal_index: int = 0,
        context: int = 0,
        flags: int = 0,
        *,
        stream: NcclStreamSpec | None = None,
    ) -> None:
        """Sends a signal to a peer rank.

        Enqueues a signal operation on the specified CUDA stream that
        notifies the target peer rank. The peer can wait for this signal
        using :py:meth:`wait_signal`.

        Args:
            peer: Target rank to send the signal to.
            signal_index: Signal index identifier. Currently must be 0.
            context: Context identifier. Currently must be 0.
            flags: Reserved for future use. Currently must be 0.
            stream: CUDA stream to enqueue the signal operation on. Defaults
                to ``None`` (the default stream).

        Raises:
            NcclInvalid: If the communicator is not initialized.

        See Also:
            :c:func:`ncclSignal`
        """
        self._check_valid("signal")

        _nccl_bindings.signal(
            peer, signal_index, context, flags, self._comm, get_stream_ptr(stream)
        )

    def put_signal(
        self,
        local_buffer: NcclBufferSpec,
        peer: int,
        peer_window: RegisteredWindowHandle,
        peer_window_offset: int = 0,
        signal_index: int = 0,
        context: int = 0,
        flags: int = 0,
        *,
        stream: NcclStreamSpec | None = None,
    ) -> None:
        """Puts data from a local buffer to a peer's window and sends a signal.

        Enqueues a put-with-signal operation on the specified CUDA stream
        that transfers the local buffer contents to the target peer's
        registered window and notifies that peer. The peer can wait for
        this signal (and thus for the put to complete) using
        :py:meth:`wait_signal`. The peer's memory must be registered with
        :py:meth:`register_window`; pass the peer's window handle as
        ``peer_window`` (e.g. obtained via an allgather of window handles).

        Args:
            local_buffer: Source buffer whose contents are put to the peer.
            peer: Target rank to put the data to and send the signal to.
            peer_window: Peer's :py:class:`~nccl.core.RegisteredWindowHandle`
                (from :py:meth:`register_window`).
            peer_window_offset: Offset in the peer's window in elements.
                Defaults to 0.
            signal_index: Signal index identifier. Currently must be 0.
            context: Context identifier. Currently must be 0.
            flags: Reserved for future use. Currently must be 0.
            stream: CUDA stream to enqueue the put_signal operation on.
                Defaults to ``None`` (the default stream).

        Raises:
            NcclInvalid: If the communicator is not initialized, or if the
                buffer specification is invalid or the buffer is on a
                different device than the communicator.

        See Also:
            :c:func:`ncclPutSignal`
        """
        self._check_valid("put_signal")

        buffer = NcclBuffer(local_buffer)
        self._validate_buffer_device(buffer, "local_buffer")

        _nccl_bindings.put_signal(
            buffer.ptr,
            buffer.count,
            int(buffer.dtype),
            peer,
            peer_window.handle,
            int(peer_window_offset) * buffer.dtype.itemsize,
            signal_index,
            context,
            flags,
            self._comm,
            get_stream_ptr(stream),
        )

    # --- Collective Communication Operations ---

    def allreduce(
        self,
        sendbuf: NcclBufferSpec,
        recvbuf: NcclBufferSpec,
        op: NcclRedOp | CustomRedOp,
        *,
        stream: NcclStreamSpec | None = None,
    ) -> None:
        """All-reduce variant of :py:meth:`reduce`.

        Equivalent to ``reduce(sendbuf, recvbuf, op, root=None, stream=stream)``:
        reduces data across all ranks and stores identical copies in each
        rank's recvbuf. See :py:meth:`reduce` for argument semantics.

        See Also:
            :py:meth:`reduce`, :c:func:`ncclAllReduce`
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
        """Copies data from ``sendbuf`` on the root rank to all ranks' ``recvbuf``.

        ``sendbuf`` is only used on the root rank and is ignored on other
        ranks.

        On the root rank, both buffers must have matching data types and
        ``sendcount == recvcount``. Element count is inferred from
        ``recvbuf``: ``count = recvcount``. In-place operation occurs when
        ``sendbuf`` and ``recvbuf`` resolve to the same device memory
        address.

        Args:
            sendbuf: Source buffer (only used on the root rank).
            recvbuf: Destination buffer that will receive the broadcast data.
            root: Root rank that broadcasts the data (0 to ``nranks - 1``).
            stream: CUDA stream for the operation. Defaults to ``None`` (the
                default stream).

        Raises:
            NcclInvalid: If send and receive buffers have mismatched dtypes,
                mismatched counts, are on the wrong device, are invalid
                specifications, or the communicator is not initialized.

        See Also:
            :c:func:`ncclBroadcast`
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
        """Reduces data from all ranks using the specified operation.

        Supports two modes. In AllReduce mode (``root`` is ``None``) all
        ranks receive the reduced result in ``recvbuf``. In Reduce mode
        (``root`` specified) only the root rank receives the reduced
        result; ``recvbuf`` is ignored on other ranks.

        Both buffers must have matching data types where used. Element
        count is inferred from ``sendbuf``: ``count = sendcount``. In
        AllReduce mode, all ranks must have ``recvcount >= sendcount``;
        in Reduce mode, only the root rank requires
        ``recvcount >= sendcount``. In-place operation occurs when
        ``sendbuf`` and ``recvbuf`` resolve to the same device memory
        address.

        Args:
            sendbuf: Source buffer containing data to be reduced.
            recvbuf: Destination buffer for the reduced result. Only used on
                the root rank in Reduce mode.
            op: Reduction operator (e.g. :py:data:`~nccl.core.SUM`,
                :py:data:`~nccl.core.MAX`, :py:data:`~nccl.core.MIN`,
                :py:data:`~nccl.core.AVG`, :py:data:`~nccl.core.PROD`, or a
                :py:class:`~nccl.core.CustomRedOp`).
            root: Root rank that receives the reduced result (0 to
                ``nranks - 1``). If ``None``, performs an all-reduce.
                Defaults to ``None``.
            stream: CUDA stream for the operation. Defaults to ``None`` (the
                default stream).

        Raises:
            NcclInvalid: If send and receive buffers have mismatched dtypes,
                mismatched counts, are on the wrong device, are invalid
                specifications, or the communicator is not initialized.

        See Also:
            :c:func:`ncclAllReduce`, :c:func:`ncclReduce`
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
        *,
        stream: NcclStreamSpec | None = None,
    ) -> None:
        """All-gather variant of :py:meth:`gather`.

        Equivalent to ``gather(sendbuf, recvbuf, root=None, stream=stream)``:
        gathers ``sendcount`` values from each rank and places identical
        copies of the concatenated result in every rank's recvbuf. See
        :py:meth:`gather` for argument semantics.

        See Also:
            :py:meth:`gather`, :c:func:`ncclAllGather`
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
        """Reduces data from all ranks and scatters the result across ranks.

        Each rank receives a different portion of the reduced result: rank
        ``i`` receives the i-th block in its ``recvbuf``.

        Both buffers must have matching data types. Element count is
        inferred from ``sendbuf``: ``count = sendcount / nranks``.
        ``sendcount`` must be ``>= nranks`` and ``recvcount`` must be
        ``>= count``. In-place operation occurs when ``recvbuf`` resolves
        to ``sendbuf_address + rank * count``.

        Args:
            sendbuf: Source buffer (size ``>= nranks * recvcount`` elements).
            recvbuf: Destination buffer with ``recvcount`` elements.
            op: Reduction operator (e.g. :py:data:`~nccl.core.SUM`,
                :py:data:`~nccl.core.MAX`, :py:data:`~nccl.core.MIN`,
                :py:data:`~nccl.core.AVG`, :py:data:`~nccl.core.PROD`, or a
                :py:class:`~nccl.core.CustomRedOp`).
            stream: CUDA stream for the operation. Defaults to ``None`` (the
                default stream).

        Raises:
            NcclInvalid: If send and receive buffers have mismatched dtypes,
                ``sendbuf`` is too small, are on the wrong device, are
                invalid specifications, or the communicator is not
                initialized.

        See Also:
            :c:func:`ncclReduceScatter`
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
        """Each rank sends and receives ``count`` values to and from every other rank.

        Data sent to destination rank ``j`` is taken from
        ``sendbuf + j * count`` and data received from source rank ``i`` is
        placed at ``recvbuf + i * count``.

        Both buffers must have matching data types. Element count is
        inferred from ``sendbuf``: ``count = sendcount / nranks``.
        ``sendcount`` must be ``>= nranks`` and ``recvcount`` must be
        ``>= sendcount``.

        Args:
            sendbuf: Source buffer (size ``>= nranks * count`` elements).
            recvbuf: Destination buffer (size ``>= nranks * count`` elements).
            stream: CUDA stream for the operation. Defaults to ``None`` (the
                default stream).

        Raises:
            NcclInvalid: If send and receive buffers have mismatched dtypes,
                buffer sizes are incompatible with ``nranks``, are on the
                wrong device, are invalid specifications, or the
                communicator is not initialized.

        See Also:
            :c:func:`ncclAlltoAll`
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
        """Gathers ``sendcount`` values from all ranks.

        Supports two modes. In AllGather mode (``root`` is ``None``) values
        are gathered from all ranks and identical copies of the result are
        placed in each ``recvbuf``. In Gather mode (``root`` specified)
        values are gathered to the specified root rank only; ``recvbuf``
        is ignored on other ranks.

        Both buffers must have matching data types where used. Element
        count is inferred from ``sendbuf``: ``count = sendcount``. Data
        from rank ``i`` is placed at ``recvbuf + i * sendcount``. AllGather
        mode requires ``recvcount >= nranks * sendcount`` on every rank;
        Gather mode requires it only on the root rank.

        In-place operation occurs when ``sendbuf`` resolves to
        ``recvbuf_address + rank * sendcount`` in AllGather mode, or to
        ``recvbuf_address + root * sendcount`` in Gather mode.

        Args:
            sendbuf: Source buffer containing ``sendcount`` elements.
            recvbuf: Destination buffer (size ``>= nranks * sendcount``
                elements). In Gather mode, only used on the root rank.
            root: Root rank that receives the gathered data (0 to
                ``nranks - 1``). If ``None``, performs an all-gather.
                Defaults to ``None``.
            stream: CUDA stream for the operation. Defaults to ``None`` (the
                default stream).

        Raises:
            NcclInvalid: If send and receive buffers have mismatched dtypes,
                ``recvbuf`` is too small, are on the wrong device, are
                invalid specifications, or the communicator is not
                initialized.

        See Also:
            :c:func:`ncclAllGather`, :c:func:`ncclGather`
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
        """Scatters data from the root rank to all ranks.

        Each rank receives ``count`` elements from the root rank. On the
        root rank, ``count`` elements from ``sendbuf + i * count`` are sent
        to rank ``i``. ``sendbuf`` is not used on non-root ranks.

        On the root rank, both buffers must have matching data types.
        Element count is inferred from ``recvbuf``: ``count = recvcount``.
        The root rank requires ``sendcount >= nranks`` and
        ``sendcount / nranks == recvcount``. In-place operation occurs when
        ``recvbuf`` resolves to ``sendbuf_address + root * count``.

        Args:
            sendbuf: Source buffer (only used on the root rank, size
                ``>= nranks * count`` elements).
            recvbuf: Destination buffer with ``count`` elements.
            root: Root rank that scatters the data (0 to ``nranks - 1``).
            stream: CUDA stream for the operation. Defaults to ``None`` (the
                default stream).

        Raises:
            NcclInvalid: If send and receive buffers have mismatched dtypes,
                ``sendbuf`` is too small on the root rank, are on the wrong
                device, are invalid specifications, or the communicator is
                not initialized.

        See Also:
            :c:func:`ncclScatter`
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
        """Registers a buffer with this communicator for zero-copy communication.

        Registered buffers can enable performance optimizations in NCCL
        operations. Buffer size is automatically derived from buffer count
        and dtype. The returned :py:class:`~nccl.core.RegisteredBufferHandle`
        is tracked by the communicator and may be released explicitly via
        its :py:meth:`~nccl.core.RegisteredBufferHandle.close` method, or
        automatically when the communicator is destroyed or aborted.

        Args:
            buffer: Buffer to register (array, Buffer, or buffer-like
                object).

        Returns:
            :py:class:`~nccl.core.RegisteredBufferHandle` for the registered
            buffer.

        Raises:
            NcclInvalid: If the buffer is on the wrong device or the
                communicator is not initialized.

        See Also:
            :c:func:`ncclCommRegister`
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
        """Collectively registers a local buffer into an NCCL window.

        This is a collective call: every rank in the communicator must
        participate, and buffer size must be equal among ranks by default.
        Buffer size is automatically derived from buffer count and dtype.
        If called within a group, the handle value may not be filled until
        ``ncclGroupEnd`` completes. For non-blocking communicators, the handle
        may remain ``0`` until :py:meth:`get_async_error` reports success.

        The returned :py:class:`~nccl.core.RegisteredWindowHandle` is
        tracked by the communicator and may be released explicitly via its
        :py:meth:`~nccl.core.RegisteredWindowHandle.close` method, or
        automatically when the communicator is destroyed or aborted.

        Args:
            buffer: Local buffer to register as a window.
            flags: Window registration flags. Defaults to ``None``
                (:py:attr:`~nccl.core.WindowFlag.DEFAULT`).

        Returns:
            :py:class:`~nccl.core.RegisteredWindowHandle` for the registered
            window, or ``None`` if NCCL returns a NULL handle (e.g. windows
            are unsupported on this platform).

        Raises:
            NcclInvalid: If the buffer is on the wrong device or the
                communicator is not initialized.

        See Also:
            :c:func:`ncclCommWindowRegister`
        """
        self._check_valid("register_window")

        nccl_buf = NcclBuffer(buffer)
        self._validate_buffer_device(nccl_buf, "buffer")
        buffer_ptr = nccl_buf.ptr
        size = nccl_buf.count * nccl_buf.dtype.itemsize

        resource = RegisteredWindowHandle(self._comm, buffer_ptr, size, flags)
        if self.get_async_error() == int(_Result.Success) and resource.handle == 0:
            # A non-blocking communicator only initializes the handle value
            # after InProgress has changed to Success.  Until then the handle
            # will remain NULL, but we cannot assume its invalid.
            #
            # But if status is Success and handle is still null, then we can
            # simply return None.
            resource.close()
            return None

        self._resources.append(resource)
        return resource

    # --- Custom Reduction APIs ---
    def create_pre_mul_sum(
        self,
        scalar: NcclScalarSpec,
        datatype: NcclDataType | None = None,
    ) -> CustomRedOp:
        """Creates a PreMulSum custom reduction operator.

        Performs ``output = scalar * sum(inputs)`` and is useful for
        averaging (``scalar = 1/N``) or weighted reductions. The returned
        :py:class:`~nccl.core.CustomRedOp` is tracked by the communicator
        and may be released explicitly via its
        :py:meth:`~nccl.core.CustomRedOp.close` method, or automatically
        when the communicator is destroyed or aborted.

        Args:
            scalar: Scalar multiplier value. A Python int or float is
                converted to a NumPy array using host memory. A NumPy array
                must contain exactly 1 element and uses host memory. An
                ``NcclSupportedBuffer`` is treated as a device buffer with
                exactly 1 element.
            datatype: NCCL data type of the scalar and reduction. If
                ``None``, it is inferred from ``scalar``: Python ``int``
                becomes ``int64`` and Python ``float`` becomes ``float64``
                (NumPy's natural dtypes); a NumPy array uses the array's
                dtype; a device buffer uses the buffer's dtype.

        Returns:
            :py:class:`~nccl.core.CustomRedOp` for the PreMulSum operator.

        Raises:
            NcclInvalid: If the communicator is not initialized; the scalar
                type is unsupported; the NumPy array or device buffer does
                not contain exactly 1 element; or the requested datatype
                does not match a device buffer's dtype.

        See Also:
            :c:func:`ncclRedOpCreatePreMulSum`
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

    def create_dev_comm(
        self, requirements: NCCLDevCommRequirements | None = None
    ) -> DevCommResource:
        """Creates a device communicator for device-side NCCL operations.

        Device communicators enable direct GPU kernel access to NCCL
        communication primitives. Multiple device communicators can be
        created from one host communicator. The returned
        :py:class:`~nccl.core.DevCommResource` is tracked by the
        communicator and may be released explicitly via its
        :py:meth:`~nccl.core.DevCommResource.close` method, or automatically
        when the communicator is destroyed or aborted. Access the device
        communicator pointer via :py:attr:`DevCommResource.ptr` or
        ``resource.dev_comm.ptr``.

        Args:
            requirements: Configuration for device communicator resource
                allocation. If ``None``, a default
                :py:class:`~nccl.core.NCCLDevCommRequirements` is used.
                Defaults to ``None``.

        Returns:
            :py:class:`~nccl.core.DevCommResource` for the device
            communicator.

        Raises:
            NcclInvalid: If the communicator is not initialized.

        See Also:
            :c:func:`ncclDevCommCreate`
        """
        self._check_valid("create_dev_comm")

        # Create default requirements if none provided
        if requirements is None:
            requirements = NCCLDevCommRequirements()

        resource = DevCommResource(self._comm, requirements._lowpp.ptr)  # type: ignore[attr-defined]
        self._resources.append(resource)
        return resource

    def close_all_resources(self) -> None:
        """Closes all resources owned by this communicator.

        Called automatically during :py:meth:`destroy` and :py:meth:`abort`,
        but can be called manually. Performs best-effort cleanup, ignoring
        any errors that occur during resource deallocation. Idempotent:
        safe to call multiple times.
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
        """Returns the last error string for this communicator.

        Raises:
            NcclInvalid: If the communicator is not initialized.
        """
        self._check_valid("get last error")
        return _nccl_bindings.get_last_error(self._comm)

    def get_async_error(self) -> _nccl_bindings.Result:
        """Queries the progress and potential errors of asynchronous NCCL operations.

        Operations without a stream argument (e.g. :py:meth:`finalize`) are
        complete when they return ``ncclSuccess``. Operations with a stream
        argument (e.g. :py:meth:`reduce`) return ``ncclSuccess`` when posted
        but may report errors through this method until completed. If any
        NCCL function returns ``ncclInProgress``, users must query the
        communicator state until it becomes ``ncclSuccess`` before calling
        another NCCL function.

        Before the state becomes ``ncclSuccess``, do not issue CUDA kernels
        on streams used by NCCL. If an error occurs, destroy the
        communicator with :py:meth:`abort`; nothing can be assumed about
        the completion or correctness of enqueued operations after an
        error.

        Returns:
            Current state of the communicator (``ncclSuccess``,
            ``ncclInProgress``, or an error code).

        Raises:
            NcclInvalid: If the communicator is not initialized.

        See Also:
            :c:func:`ncclCommGetAsyncError`
        """
        self._check_valid("get async error")
        result = _nccl_bindings.comm_get_async_error(self._comm)
        if result != _Result.InProgress:
            self._children_in_progress = []
        return result

    def get_mem_stat(self, stat: NcclCommMemStat) -> int:
        """Queries communicator memory statistics.

        Args:
            stat: The memory statistic to query.

        Returns:
            The memory statistic value (bytes, or 0/1 for GPU_MEM_SUSPENDED).

        Raises:
            NcclInvalid: If the communicator is not initialized.
        """
        self._check_valid("get mem stat")
        return _nccl_bindings.comm_mem_stats(self._comm, stat)
