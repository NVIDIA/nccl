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
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as _np

from cuda.core import Device
from cuda.core import system

from nccl.bindings import nccl as _nccl_bindings

from nccl.core._binding_helpers import LowppSpec, Field
from nccl.core.buffer import NcclBuffer
from nccl.core.constants import (
    CTAPolicy,
    CommShrinkFlag,
    CommSuspendFlag,
    WindowFlag,
)
from nccl.core.cuda import get_stream_ptr
from nccl.core.team import NCCLTeam
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
    NcclHostCftMode,
    NcclCftTeamMode,
    NcclCftCap,
    NcclStreamSpec,
    NcclScalarSpec,
    NcclInvalid,
    NcclCommMemStat,
)
from nccl.core.utils import UniqueId

_Result = _nccl_bindings.Result


__all__ = [
    "NCCLConfig",
    "NCCLCollConfig",
    "VendorOption",
    "NCCLCommProperties",
    "WaitSignalDesc",
    "TeamRequirement",
    "LsaBarrierRequirement",
    "GinBarrierRequirement",
    "LLA2ARequirement",
    "NCCLDevCommRequirements",
    "Communicator",
]


@dataclass(kw_only=True)
class NCCLConfig(LowppSpec, lowpp_cls=_nccl_bindings.Config):
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
    """Whether NCCL preserves stream-ordering semantics for collectives captured into CUDA graphs. Supported values are 0 (disabled) or 1 (enabled). The value 0 cannot be combined with ``graph_usage_mode=2``. Also controllable via the ``NCCL_GRAPH_STREAM_ORDERING`` environment variable. If unset, NCCL uses 1."""

    launch_order_implicit: bool | None = None
    """Whether this communicator takes part in implicit launch ordering (NCCL 2.31+). Within one CUDA context, operations on communicators that enable it must not overlap with operations on communicators that do not. Also controllable via the ``NCCL_LAUNCH_ORDER_IMPLICIT`` environment variable, which takes precedence. If unset, NCCL uses False."""

    num_rma_sig: int | None = None
    """Number of one-sided RMA signal indexes available per context (NCCL 2.31+). Non-negative integer; bounds the ``signal_index`` accepted by the signal and wait-signal operations. If unset, NCCL uses 1."""

    rma_eager_init: bool | None = None
    """Whether the collective one-sided RMA signal setup is initialized at communicator creation rather than at the first window registration (NCCL 2.31+). True is required if the communicator issues signal or wait-signal operations without first registering a symmetric window. Also controllable via the ``NCCL_RMA_EAGER_INIT`` environment variable, which takes precedence. If unset, NCCL uses False."""

    host_cft_mode: NcclHostCftMode | None = None
    """Host-side Compute Fabric Transport mode (NCCL 2.31+). Controls whether the communicator creates the CUDA fabric logical endpoints backing the host-side CFT queries. If unset, NCCL uses :py:attr:`NcclHostCftMode.DEFAULT`."""


@dataclass(frozen=True)
class VendorOption:
    """A single vendor-specific option attached to an :py:class:`NCCLCollConfig`.

    Mirrors one :c:type:`ncclConfigExt_t` node. Options are identified by the
    ``(vendor_id, option_id)`` pair; the official NCCL library ignores every
    extension, so an option only has an effect on a vendor library that
    recognizes its ``vendor_id``. Vendors pick a non-zero ``vendor_id``
    less than 2**24 that is unlikely to collide.

    Exactly one of the three value fields must be set.

    See Also:
        :c:type:`ncclConfigExt_t`
    """

    vendor_id: int
    """Vendor-chosen identifier, unique across vendor libraries."""

    option_id: int
    """Vendor-defined identifier distinguishing options within a vendor."""

    int_value: int | None = None
    """Integer value (``val.i``)."""

    str_value: str | None = None
    """String value (``val.s``), encoded to UTF-8."""

    raw_value: int | None = None
    """Value of any other type (``val.raw``), as an integer. If it is an
    address, the referent must stay valid for the duration of the call."""

    def __post_init__(self):
        set_fields = [
            name for name in ("int_value", "str_value", "raw_value")
            if getattr(self, name) is not None
        ]
        if len(set_fields) != 1:
            raise NcclInvalid(
                "VendorOption requires exactly one of int_value, str_value, "
                f"raw_value to be set, got {set_fields or 'none'}"
            )


def _validate_vendor_options(options: Sequence[Any]) -> None:
    if isinstance(options, (VendorOption, str, bytes)) or not isinstance(options, Sequence):
        raise NcclInvalid(
            "vendor_options must be a sequence of VendorOption, got "
            f"{type(options).__name__}"
        )
    seen = set()
    for opt in options:
        if not isinstance(opt, VendorOption):
            raise NcclInvalid(
                "vendor_options must contain VendorOption instances, got "
                f"{type(opt).__name__}"
            )
        key = (opt.vendor_id, opt.option_id)
        if key in seen:
            raise NcclInvalid(
                f"Duplicate vendor option key (vendor_id={opt.vendor_id}, "
                f"option_id={opt.option_id}); keys must be unique"
            )
        seen.add(key)


@dataclass(kw_only=True)
class NCCLCollConfig(LowppSpec, lowpp_cls=_nccl_bindings.CollConfig):
    """Per-call configuration for a single collective.

    Accepted as the ``config`` argument of every collective on
    :py:class:`Communicator`, tuning that one call. Fields left unset fall
    back to the communicator's value, or to NCCL's own default. The same
    configuration must be set on every rank; NCCL validates it only locally,
    when the call is issued.

    See Also:
        :c:type:`ncclCollConfig_t`
    """

    min_ctas: int | None = None
    """Lower bound on channels/CTAs for this call. Also set by
    ``NCCL_MIN_CTAS``, which takes precedence. If unset, inherits
    :py:attr:`NCCLConfig.min_ctas`."""

    max_ctas: int | None = None
    """Upper bound on channels/CTAs for this call, clamped to the
    communicator's ``max_ctas``. Also set by ``NCCL_MAX_CTAS``, which takes
    precedence. If unset, inherits :py:attr:`NCCLConfig.max_ctas`."""

    nvls_ctas: int | None = None
    """NVLS-pool-specific channel cap for this call. Also set by
    ``NCCL_NVLS_NCHANNELS``, which takes precedence. If unset, inherits
    :py:attr:`NCCLConfig.nvls_ctas`."""

    cga_cluster_size: int | None = None
    """CUDA thread-block-cluster size (0-8, Hopper+). Inconsistent values
    within one group are undefined behavior. Also set by
    ``NCCL_CGA_CLUSTER_SIZE``, which takes precedence. If unset, inherits
    :py:attr:`NCCLConfig.cga_cluster_size`."""

    alg_selection: str | None = None
    """Selection string filtering which algorithms this call may use, e.g.
    ``"ring"``, ``"tree,ring"``, ``"^symk"``. If unset or empty, NCCL selects
    automatically."""

    force_alg_selection: bool | None = None
    """Whether an unsatisfiable :py:attr:`alg_selection` is an error rather
    than a fallback to automatic selection. If unset, NCCL uses True."""

    cta_policy: CTAPolicy | None = None
    """CTA scheduling policy for this call. Also set by ``NCCL_CTA_POLICY``,
    which takes precedence. If unset, inherits
    :py:attr:`NCCLConfig.cta_policy`."""

    user_profiler_tag: int | None = None
    """Opaque value delivered verbatim to profiler plugins with this call's
    profiler events; does not affect execution. Values with the
    most-significant bit set are reserved by NCCL. If unset, NCCL uses 0."""

    vendor_options: tuple[VendorOption, ...] = ()
    """Vendor-specific options; ``(vendor_id, option_id)`` keys must be
    unique."""

    def __post_init__(self):
        _validate_vendor_options(self.vendor_options)


@dataclass(frozen=True, kw_only=True)
class NCCLCommProperties:
    """The properties NCCL reports for a communicator.

    Returned by :py:attr:`Communicator.properties`. These values are fixed
    for the lifetime of the communicator. Fields marked NCCL 2.31+ are
    ``None`` when nccl4py was built against an older NCCL.

    See Also:
        :c:type:`ncclCommProperties` for the description of each field.
    """

    rank: int
    """This caller's rank within the communicator."""

    n_ranks: int
    """Number of ranks in the communicator."""

    cuda_dev: int
    """CUDA device ID associated with the communicator."""

    nvml_dev: int
    """NVML device ID for the GPU. Uses the NVML indexing space, which may
    differ from CUDA indexing."""

    device_api_support: bool
    """Whether device-side NCCL operations are supported."""

    multimem_support: bool
    """Whether ranks in the same LSA team can communicate using multimem."""

    gin_type: NcclGinType
    """GIN transport reaching every rank. ``NONE`` unless
    :py:attr:`gin_connection_type` is ``FULL``, even when a rail-restricted
    transport is available."""

    n_lsa_teams: int
    """Number of LSA teams."""

    host_rma_support: bool
    """Whether host RMA is supported."""

    railed_gin_type: NcclGinType
    """GIN transport reaching ranks within a rail. ``NONE`` only when no GIN
    transport is available at all."""

    comm_hash: int | None = None
    """Hash identifying the communicator, shared by all its ranks (NCCL 2.31+)."""

    gin_min_stride: int | None = None
    """Granularity of the GIN rank stride this communicator supports. A stride
    passed as :py:attr:`NCCLDevCommRequirements.gin_custom_stride` must be a
    multiple of this value, and no larger than the rail team's stride. It is 1
    when :py:attr:`gin_connection_type` is ``FULL`` (NCCL 2.31+)."""

    gin_connection_type: NcclGinConnectionType | None = None
    """Widest GIN connection topology this communicator supports: ``NONE``,
    ``RAIL`` or ``FULL``. A device communicator may request this topology or a
    narrower one via
    :py:attr:`NCCLDevCommRequirements.gin_connection_type` (NCCL 2.31+)."""

    available_gin_types: frozenset[NcclGinType] | None = None
    """The GIN transports this communicator can use, e.g.
    ``NcclGinType.GDAKI in props.available_gin_types``. Empty when GIN is
    unavailable (NCCL 2.31+)."""

    dev_comm_runtime_version_size: int | None = None
    """Size, in bytes, of the device communicator structure in the running NCCL
    library (NCCL 2.31+)."""


@dataclass(frozen=True)
class WaitSignalDesc(LowppSpec, lowpp_cls=_nccl_bindings.WaitSignalDesc):
    """Descriptor for a wait-signal operation.

    Describes a single signal-wait operation for use with
    :py:meth:`Communicator.wait_signal`. Each descriptor specifies which peer
    to wait for, how many signal operations to wait for, and additional
    context for the wait operation.
    """

    peer: int
    """Target peer rank to wait for signals from."""

    op_count: int = Field(default=1, lowpp_name="op_cnt")
    """Number of signal operations to wait for from the peer. Defaults to 1."""

    signal_index: int = Field(default=0, lowpp_name="sig_idx")
    """Signal index identifier. Must lie in ``[0, num_rma_sig)``; see
    :py:attr:`NCCLConfig.num_rma_sig`. Defaults to 0."""

    context: int = Field(default=0, lowpp_name="ctx")
    """Context identifier. Must lie in ``[0, num_rma_ctx)``; see
    :py:attr:`NCCLConfig.num_rma_ctx`. Defaults to 0."""


@dataclass(frozen=True)
class TeamRequirement:
    """A per-team requirement for device communicator creation.

    Pass a tuple of these as :py:attr:`NCCLDevCommRequirements.teams`. When
    ``multimem`` is True, NCCL allocates a multicast handle for the team,
    retrievable afterwards via
    :py:meth:`~nccl.core.DevCommResource.multimem_handle`.
    """

    team: NCCLTeam
    multimem: bool = False


@dataclass(frozen=True)
class LsaBarrierRequirement:
    """Requests an LSA barrier resource on ``team`` with ``n_barriers`` barriers.

    Add to :py:attr:`NCCLDevCommRequirements.resources`; the finalized
    :py:class:`~nccl.core.LsaBarrierHandle` is returned in
    :py:attr:`~nccl.core.DevCommResource.resource_handles`.
    """

    team: NCCLTeam
    n_barriers: int


@dataclass(frozen=True)
class GinBarrierRequirement:
    """Requests a GIN barrier resource on ``team`` with ``n_barriers`` barriers.

    Add to :py:attr:`NCCLDevCommRequirements.resources`; the finalized
    :py:class:`~nccl.core.GinBarrierHandle` is returned in
    :py:attr:`~nccl.core.DevCommResource.resource_handles`.
    """

    team: NCCLTeam
    n_barriers: int


@dataclass(frozen=True)
class LLA2ARequirement:
    """Requests a low-latency all-to-all resource with ``n_blocks`` blocks,
    sized to hold up to ``max_elements`` elements of at most
    ``max_element_size`` bytes each.

    Add to :py:attr:`NCCLDevCommRequirements.resources`; the finalized
    :py:class:`~nccl.core.LLA2AHandle` is returned in
    :py:attr:`~nccl.core.DevCommResource.resource_handles`.
    """

    n_blocks: int
    max_elements: int
    max_element_size: int


@dataclass(kw_only=True)
class NCCLDevCommRequirements(LowppSpec, lowpp_cls=_nccl_bindings.DevCommRequirements):
    """NCCL device communicator requirements configuration.

    This is a reusable high-level Python request consumed by
    :py:meth:`Communicator.create_dev_comm`. Per-team requirements are
    declared through the :py:attr:`teams` tuple. Each call snapshots the
    request into independent low-level ``ncclDevCommRequirements_t`` and linked
    ``ncclTeamRequirements_t`` storage, including separate multimem output
    handles. NCCL copies the requirements and linked-list nodes before the call
    returns; the resulting :class:`DevCommResource` retains the storage
    referenced by each ``outMultimemHandle``. This object may therefore be
    changed between calls without affecting device communicators that were
    already created. Do not mutate it concurrently with
    :meth:`Communicator.create_dev_comm`.

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
    """Force-enable GPU-Initiated Networking (GIN). If unset, NCCL uses False."""

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

    gin_traffic_class: int | None = None
    """GIN traffic class. If unset, NCCL uses its internal default."""

    world_gin_barrier_count: int | None = None
    """Number of world GIN barriers. If unset, NCCL uses 0."""

    gin_strong_signals_required: bool | None = None
    """Whether GIN strong signals are required by kernels using this devComm.
    When False, using GIN strong signals results in undefined behavior. If unset, NCCL uses True."""

    gin_va_signals_required: bool | None = None
    """Whether GIN VA signals are required by kernels using this devComm.
    When False, using GIN VA signals results in undefined behavior. If unset, NCCL uses True."""

    gin_custom_stride: int | None = None
    """Stride of ranks to connect for GIN. Only consulted when
    :py:attr:`gin_connection_type` is
    :py:attr:`NcclGinConnectionType.CUSTOM_STRIDE`, and must be a multiple of
    :py:attr:`NCCLCommProperties.gin_min_stride`. If unset, NCCL uses 1."""

    gin_type: NcclGinType | None = None
    """GIN transport to require. If unset, NCCL uses
    :py:attr:`NcclGinType.NONE`, accepting any available transport."""

    cft_caps: NcclCftCap | None = None
    """Compute Fabric Transport capabilities to request, as a bitmask of
    :py:class:`NcclCftCap` values. Creation fails if CFT resources are
    requested on a communicator where not all ranks support CFT. If unset,
    NCCL uses :py:attr:`NcclCftCap.NONE`."""

    cft_barrier_count: int | None = None
    """Number of CFT barriers to allocate, one per independently addressed
    barrier slot the kernel uses (commonly one per CTA). If unset, NCCL uses 0."""

    teams: tuple[TeamRequirement, ...] = ()
    """Per-team requirements. Entries for the same team (by value) are merged,
    keeping first-appearance order; multimem is requested for a team if any of
    its entries sets it. A team requested with ``multimem=True`` yields a
    multimem handle retrievable via
    :py:meth:`~nccl.core.DevCommResource.multimem_handle`."""

    resources: tuple[LsaBarrierRequirement | GinBarrierRequirement | LLA2ARequirement, ...] = ()
    """Device resource requirements (LSA/GIN barriers, low-latency all-to-all).
    Each entry yields, in order, a handle in
    :py:attr:`~nccl.core.DevCommResource.resource_handles`. Entries are
    kept as-is (not merged): each is a distinct resource."""


def _materialize_coll_config(
    config: NCCLCollConfig | None,
) -> tuple[Any | None, list[Any]]:
    """Builds the lowpp ``CollConfig`` and its vendor-option chain.

    Returns ``(lowpp, keepalive)``. ``keepalive`` holds the ``ConfigExt``
    nodes and the ``val`` views that own encoded string buffers; the caller
    must keep it referenced until the NCCL call returns, since the config
    stores borrowed pointers into them. The call is the whole window: NCCL
    resolves the config -- parsing ``alg_selection`` into an algorithm mask
    -- before the entry point returns, even inside a group. A path that
    defers that resolution must copy the config rather than borrow it.

    Returns ``(None, [])`` when ``config`` is ``None``.
    """
    if config is None:
        return None, []

    # NCCLCollConfig is mutable, so revalidate Python-only extensions at the
    # point of use rather than relying solely on construction-time validation.
    _validate_vendor_options(config.vendor_options)
    cfg = config._to_lowpp()
    keepalive: list[Any] = []

    nodes = []
    for opt in config.vendor_options:
        node = _nccl_bindings.ConfigExt()
        node.key.vendor_id = int(opt.vendor_id)
        node.key.option_id = int(opt.option_id)
        # `node.val` is a fresh view each access, and the string setter parks
        # the encoded bytes on the view it was called on -- hold that same
        # view so the char* it wrote stays alive.
        val = node.val
        if opt.int_value is not None:
            val.i = int(opt.int_value)
        elif opt.str_value is not None:
            val.s = opt.str_value
        else:
            val.raw = int(opt.raw_value)
        nodes.append(node)
        keepalive += [node, val]

    for node, nxt in zip(nodes, nodes[1:]):
        node.next = nxt
    if nodes:
        # A borrowed pointer: the `ext` setter stores no reference of its own.
        cfg.ext = nodes[0].ptr

    return cfg, keepalive


def _materialize_team_requirements(
    teams: tuple[TeamRequirement, ...],
) -> tuple[
    tuple[_nccl_bindings.TeamRequirements, ...],
    dict[NCCLTeam, _nccl_bindings.MultimemHandle],
]:
    """Builds the team-requirements linked list on ``reqs``.

    Entries for the same team (by value) are merged into one node, keeping
    first-appearance order; its multimem is requested if any entry sets it. This
    matches NCCL, which allocates one multicast per unique team (``symTeamObtain``
    caches per team) whenever any requirement for that team sets multimem.

    Returns ``(team_nodes, multimem_handles)``. The caller must keep
    ``team_nodes`` alive until ``ncclDevCommCreate`` returns (``reqs`` and each
    node hold raw pointers to the next node); ``multimem_handles`` is the per-team
    output storage NCCL writes into.
    """
    merged: dict[NCCLTeam, bool] = {}
    for team_req in teams:
        merged[team_req.team] = merged.get(team_req.team, False) or bool(team_req.multimem)

    team_nodes: list[_nccl_bindings.TeamRequirements] = []
    multimem_handles: dict[NCCLTeam, _nccl_bindings.MultimemHandle] = {}
    previous: _nccl_bindings.TeamRequirements | None = None
    for team, multimem in merged.items():
        node = _nccl_bindings.TeamRequirements()
        # The generated setter copies the team POD into this fresh node.
        node.team = team._to_lowpp()
        node.multimem = multimem

        if multimem:
            handle = _nccl_bindings.MultimemHandle()
            node.out_multimem_handle = handle.ptr
            multimem_handles[team] = handle

        if previous is not None:
            previous.next = node.ptr

        team_nodes.append(node)
        previous = node

    return tuple(team_nodes), multimem_handles


def _materialize_resource_requirements(
    comm: _nccl_bindings.Comm,
    resources: tuple[LsaBarrierRequirement | GinBarrierRequirement | LLA2ARequirement, ...],
) -> tuple[
    tuple[_nccl_bindings.DevResourceRequirements, ...],
    tuple[
        _nccl_bindings.LsaBarrierHandle
        | _nccl_bindings.GinBarrierHandle
        | _nccl_bindings.LLA2AHandle,
        ...,
    ],
]:
    """Builds the resource-requirements linked list and its output handles.

    Each resource calls the matching NCCL ``*CreateRequirement`` helper, which
    fills a node and wires it to write into the handle during
    ``ncclDevCommCreate``. Returns ``(resource_nodes, handle_lowpps)`` in
    requirement order; the caller keeps ``resource_nodes`` alive until
    ``ncclDevCommCreate`` returns (each node holds a raw pointer to the next).
    """
    resource_nodes: list[_nccl_bindings.DevResourceRequirements] = []
    resource_handles: list[
        _nccl_bindings.LsaBarrierHandle
        | _nccl_bindings.GinBarrierHandle
        | _nccl_bindings.LLA2AHandle
    ] = []
    previous: _nccl_bindings.DevResourceRequirements | None = None
    for resource in resources:
        node = _nccl_bindings.DevResourceRequirements()
        if isinstance(resource, LsaBarrierRequirement):
            handle = _nccl_bindings.LsaBarrierHandle()
            team_lowpp = resource.team._to_lowpp()
            _nccl_bindings.lsa_barrier_create_requirement(
                team_lowpp.ptr, resource.n_barriers, handle.ptr, node.ptr
            )
        elif isinstance(resource, GinBarrierRequirement):
            handle = _nccl_bindings.GinBarrierHandle()
            team_lowpp = resource.team._to_lowpp()
            _nccl_bindings.gin_barrier_create_requirement(
                comm, team_lowpp.ptr, resource.n_barriers, handle.ptr, node.ptr
            )
        elif isinstance(resource, LLA2ARequirement):
            handle = _nccl_bindings.LLA2AHandle()
            n_slots = _nccl_bindings.ll_a2a_calc_slots(
                resource.max_elements, resource.max_element_size
            )
            _nccl_bindings.ll_a2a_create_requirement(
                resource.n_blocks, n_slots, handle.ptr, node.ptr
            )
        else:
            raise TypeError(f"unknown resource requirement: {type(resource).__name__}")

        if previous is not None:
            previous.next = node.ptr

        resource_nodes.append(node)
        resource_handles.append(handle)
        previous = node

    return tuple(resource_nodes), tuple(resource_handles)


class Communicator:
    """NCCL communicator for collective and point-to-point operations.

    A communicator represents a group of participants that perform NCCL
    operations. Each participant is assigned an integer rank in
    ``[0, nranks)``.

    Most users should create communicators with :py:meth:`init` or
    :py:meth:`init_all`. The constructor is a low-level interoperability
    entry point for wrapping an existing NCCL communicator pointer or creating
    a null communicator for later initialization.

    A communicator instance provides collective and point-to-point operations,
    lifecycle and resource management, and properties describing its rank,
    device, topology, and capabilities.
    """

    _comm: _nccl_bindings.Comm
    _resources: list[CommResource]
    _nranks: int | None
    _device: Device | None
    _rank: int | None
    _comm_properties: NCCLCommProperties | None
    _children_in_progress: list[Communicator]

    def __init__(self, ptr: int | None = None) -> None:
        """Wraps an existing NCCL communicator pointer.

        This is a low-level interoperability entry point for an NCCL
        communicator pointer obtained from another library or framework. Most
        users should create communicators with :py:meth:`init` or
        :py:meth:`init_all` instead.

        Omitting ``ptr`` or passing 0 creates a null communicator. A null
        communicator can later be initialized with :py:meth:`initialize`, or
        used with :py:meth:`grow` to join an existing communicator.

        Args:
            ptr: Address of an existing NCCL communicator, represented as a
                Python integer. ``None`` and 0 create a null communicator.
                Defaults to ``None``.
        """
        self._comm = _nccl_bindings.Comm(0 if ptr is None else ptr)
        self._resources = []
        self._nranks = None
        self._device = None
        self._rank = None
        self._comm_properties = None
        self._children_in_progress = []

    @classmethod
    def _from_comm(cls, comm: _nccl_bindings.Comm) -> Communicator:
        """Constructs a communicator from an internal binding wrapper."""
        obj = cls()
        obj._comm = comm
        return obj

    def _check_valid(self, operation: str) -> None:
        """Validates that this object references an active communicator.

        Args:
            operation: Name of the operation being attempted (for the error
                message).

        Raises:
            NcclInvalid: If this object does not reference an active NCCL
                communicator.
        """
        if not self._comm:
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

    def __repr__(self) -> str:
        if not self._comm:
            return "<Communicator: null (ptr=0)>"
        try:
            return f"<Communicator: rank={self.rank}/{self.nranks}, device={self.device.device_id}, ptr={self.ptr:#x}>"
        except RuntimeError:
            # If we can't get properties, just show the pointer
            return f"<Communicator: ptr={self.ptr:#x}>"

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
        comm_handles = _nccl_bindings.comm_init_all(ndev, devlist)

        return [cls._from_comm(comm) for comm in comm_handles]

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
        if self._comm:
            raise NcclInvalid("Communicator is already initialized")

        config_lowpp = None if config is None else config._to_lowpp()
        cfg_ptr = 0 if config_lowpp is None else config_lowpp.ptr
        if isinstance(unique_id, UniqueId):
            unique_id = (unique_id,)
        elif not isinstance(unique_id, (list, tuple)):
            raise NcclInvalid("unique_id must be a UniqueId or a sequence of UniqueIds")

        commIds = bytearray().join(bytes(uid) for uid in unique_id)
        self._comm = _nccl_bindings.comm_init_rank_scalable(
            int(nranks), int(rank), int(len(unique_id)), commIds, cfg_ptr
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
            color = -1  # NCCL_SPLIT_NOCOLOR from nccl.h
        config_lowpp = None if config is None else config._to_lowpp()
        cfg_ptr = 0 if config_lowpp is None else config_lowpp.ptr
        newcomm = type(self)._from_comm(
            _nccl_bindings.comm_split(self._comm, int(color), int(key), cfg_ptr)
        )
        self._children_in_progress.append(newcomm)

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
        config_lowpp = None if config is None else config._to_lowpp()
        cfg_ptr = 0 if config_lowpp is None else config_lowpp.ptr
        newcomm = type(self)._from_comm(
            _nccl_bindings.comm_shrink(
                self._comm,
                ranks_to_exclude,
                len(ranks_to_exclude),
                cfg_ptr,
                int(flag),
            )
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
        if is_new_rank and self._comm:
            raise NcclInvalid("New ranks must use a null communicator (Communicator())")
        if not is_new_rank and not self._comm:
            raise NcclInvalid("Existing ranks must use an initialized communicator")

        uid_ptr = 0 if unique_id is None else unique_id.ptr
        rank_val = -1 if rank is None else int(rank)
        config_lowpp = None if config is None else config._to_lowpp()
        cfg_ptr = 0 if config_lowpp is None else config_lowpp.ptr
        newcomm = type(self)._from_comm(
            _nccl_bindings.comm_grow(self._comm, int(nranks), uid_ptr, rank_val, cfg_ptr)
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

        if not self._comm:
            return

        _nccl_bindings.comm_destroy(self._comm)
        self._comm = _nccl_bindings.Comm()

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

        if not self._comm:
            return

        _nccl_bindings.comm_abort(self._comm)
        self._comm = _nccl_bindings.Comm()

    def finalize(self) -> None:
        """Finalizes the communicator, flushing uncompleted operations and network resources.

        Typically called before :py:meth:`destroy` to ensure all operations
        complete. This is a collective operation that must be called by all
        ranks. When one thread manages multiple host-local ranks (multiple GPUs
        per thread), calls to :py:meth:`finalize` must be issued within
        :py:func:`~nccl.core.group` so that all local ranks can enter
        finalization together; otherwise they will hang.

        For nonblocking communicators this is itself nonblocking: success
        sets the communicator state to ``ncclInProgress`` to indicate
        finalization is in progress. Once all NCCL operations complete, the
        communicator transitions to ``ncclSuccess``. Users can query the
        state with :py:meth:`get_async_error`.

        See Also:
            :c:func:`ncclCommFinalize`
        """
        if not self._comm:
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
    def ptr(self) -> int:
        """Integer value of the underlying :c:type:`ncclComm_t` (0 if destroyed or null)."""
        return self._comm.handle

    @property
    def is_valid(self) -> bool:
        """Whether the communicator is valid (not destroyed or null)."""
        return bool(self._comm)

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
    def properties(self) -> NCCLCommProperties:
        """All properties NCCL reports for this communicator.

        Use this to read several properties at once, or to reach the fields
        that have no dedicated accessor.

        Returns:
            An :py:class:`NCCLCommProperties` holding every field NCCL reports
            for this communicator.

        Raises:
            NcclInvalid: If the communicator is not initialized.
        """
        self._check_valid("get properties")
        if self._comm_properties is None:
            pod = _nccl_bindings.comm_query_properties(self._comm)
            gin_support = getattr(pod, "gin_support", None)
            gin_connection_type = getattr(pod, "gin_connection_type", None)
            self._comm_properties = NCCLCommProperties(
                rank=pod.rank,
                n_ranks=pod.n_ranks,
                cuda_dev=pod.cuda_dev,
                nvml_dev=pod.nvml_dev,
                device_api_support=bool(pod.device_api_support),
                multimem_support=bool(pod.multimem_support),
                gin_type=NcclGinType(pod.gin_type),
                n_lsa_teams=pod.n_lsa_teams,
                host_rma_support=bool(pod.host_rma_support),
                railed_gin_type=NcclGinType(pod.railed_gin_type),
                comm_hash=getattr(pod, "comm_hash", None),
                gin_min_stride=getattr(pod, "gin_min_stride", None),
                gin_connection_type=(
                    None
                    if gin_connection_type is None
                    else NcclGinConnectionType(gin_connection_type)
                ),
                # Copied out: the lowpp accessor aliases the CommProperties
                # buffer. Iterating the enum skips the array's reserved tail;
                # NONE is not a transport.
                available_gin_types=(
                    None
                    if gin_support is None
                    else frozenset(
                        t for t in NcclGinType if t is not NcclGinType.NONE and gin_support[t]
                    )
                ),
                dev_comm_runtime_version_size=getattr(pod, "dev_comm_runtime_version_size", None),
            )
        return self._comm_properties

    @property
    def cuda_dev(self) -> int:
        """CUDA device ID associated with this communicator.

        Raises:
            NcclInvalid: If the communicator is not initialized.

        """
        self._check_valid("get cuda_dev")
        return self.properties.cuda_dev

    @property
    def nvml_dev(self) -> int:
        """NVML device ID for the GPU associated with this communicator.

        Uses the NVML indexing space, which may differ from CUDA indexing.

        Raises:
            NcclInvalid: If the communicator is not initialized.

        """
        self._check_valid("get nvml_dev")
        return self.properties.nvml_dev

    @property
    def device_api_support(self) -> bool:
        """Whether device-side NCCL operations are supported on this platform.

        If False, a device communicator cannot be created.

        Raises:
            NcclInvalid: If the communicator is not initialized.

        """
        self._check_valid("get device_api_support")
        return self.properties.device_api_support

    @property
    def multimem_support(self) -> bool:
        """Whether ranks in the same LSA team can communicate using multimem.

        If False, a device communicator cannot be created with multimem
        resources.

        Raises:
            NcclInvalid: If the communicator is not initialized.

        """
        self._check_valid("get multimem_support")
        return self.properties.multimem_support

    @property
    def gin_type(self) -> NcclGinType:
        """GPU-Initiated Networking (GIN) type reaching every rank.

        If equal to :py:attr:`NcclGinType.NONE`, a device communicator
        cannot be created with GIN connection type
        :py:attr:`NcclGinConnectionType.FULL`. A rail-restricted transport
        may still be available; see :py:attr:`railed_gin_type`.

        Raises:
            NcclInvalid: If the communicator is not initialized.

        """
        self._check_valid("get gin_type")
        return self.properties.gin_type

    @property
    def n_lsa_teams(self) -> int:
        """Number of Load/Store Accessible (LSA) teams for this communicator.

        Raises:
            NcclInvalid: If the communicator is not initialized.

        """
        self._check_valid("get n_lsa_teams")
        return self.properties.n_lsa_teams

    @property
    def team_world(self) -> NCCLTeam:
        """The world team for this communicator.

        Raises:
            NcclInvalid: If the communicator is not initialized.
        """
        self._check_valid("get team_world")
        pod = _nccl_bindings.team_world(self._comm)
        return NCCLTeam(pod.n_ranks, pod.rank, pod.stride)

    @property
    def team_lsa(self) -> NCCLTeam:
        """The LSA team for this communicator.

        Raises:
            NcclInvalid: If the communicator is not initialized.
        """
        self._check_valid("get team_lsa")
        pod = _nccl_bindings.team_lsa(self._comm)
        return NCCLTeam(pod.n_ranks, pod.rank, pod.stride)

    @property
    def team_rail(self) -> NCCLTeam:
        """The rail team for this communicator.

        Raises:
            NcclInvalid: If the communicator is not initialized.
        """
        self._check_valid("get team_rail")
        pod = _nccl_bindings.team_rail(self._comm)
        return NCCLTeam(pod.n_ranks, pod.rank, pod.stride)

    def team_cft(self, mode: NcclCftTeamMode = NcclCftTeamMode.FLAT) -> NCCLTeam:
        """The CFT team for this communicator, in the requested layout.

        Args:
            mode: Team layout. Defaults to
                :py:attr:`NcclCftTeamMode.FLAT`, matching the C default.

        Raises:
            NcclInvalid: If the communicator is not initialized.

        See Also:
            :c:func:`ncclTeamCft`
        """
        self._check_valid("get team_cft")
        pod = _nccl_bindings.team_cft(self._comm, int(mode))
        return NCCLTeam(pod.n_ranks, pod.rank, pod.stride)

    @property
    def team_cft_multimem(self) -> NCCLTeam:
        """The CFT multimem team for this communicator.

        Raises:
            NcclInvalid: If the communicator is not initialized.

        See Also:
            :c:func:`ncclTeamCftMultimem`
        """
        self._check_valid("get team_cft_multimem")
        pod = _nccl_bindings.team_cft_multimem(self._comm)
        return NCCLTeam(pod.n_ranks, pod.rank, pod.stride)

    @property
    def host_rma_support(self) -> bool:
        """Whether host RMA is supported on this communicator.

        Raises:
            NcclInvalid: If the communicator is not initialized.

        """
        self._check_valid("get host_rma_support")
        return self.properties.host_rma_support

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
        return self.properties.railed_gin_type

    def team_rank_to_world(self, team: NCCLTeam, team_rank: int) -> int:
        """Maps a rank within ``team`` to its rank in this communicator.

        ``team`` is anchored at this rank, so ``team.rank`` maps back to
        :py:attr:`rank` and neighbours are offset by ``team.stride``.

        Args:
            team: The team ``team_rank`` is expressed in, as returned by
                :py:attr:`team_world`, :py:attr:`team_lsa`, or
                :py:attr:`team_rail`.
            team_rank: Rank within ``team``.

        Returns:
            The corresponding rank in this communicator.

        Raises:
            NcclInvalid: If the communicator is not initialized.
        """
        self._check_valid("team_rank_to_world")
        team_lowpp = team._to_lowpp()
        return _nccl_bindings.team_rank_to_world(self._comm, team_lowpp.ptr, team_rank)

    def team_rank_to_lsa(self, team: NCCLTeam, team_rank: int) -> int:
        """Maps a rank within ``team`` to its rank in the LSA team.

        The LSA-relative counterpart of :py:meth:`team_rank_to_world`:
        ``team.rank`` maps back to this rank's index in
        :py:attr:`team_lsa`. Only meaningful when ``team_rank`` names a
        peer that shares this rank's LSA team.

        Args:
            team: The team ``team_rank`` is expressed in, as returned by
                :py:attr:`team_world`, :py:attr:`team_lsa`, or
                :py:attr:`team_rail`.
            team_rank: Rank within ``team``.

        Returns:
            The corresponding rank in the LSA team, or ``-1`` if the
            device resource state could not be initialized.

        Raises:
            NcclInvalid: If the communicator is not initialized.
        """
        self._check_valid("team_rank_to_lsa")
        team_lowpp = team._to_lowpp()
        return _nccl_bindings.team_rank_to_lsa(self._comm, team_lowpp.ptr, team_rank)

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
            s.ptr, s.count, int(s.dtype), int(peer), self._comm, get_stream_ptr(stream)
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
            r.ptr, r.count, int(r.dtype), int(peer), self._comm, get_stream_ptr(stream)
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

        lowpp_descs = [d._to_lowpp() for d in descs]
        buf = bytearray().join(bytes(d) for d in lowpp_descs)
        _nccl_bindings.wait_signal(len(descs), buf, self._comm, get_stream_ptr(stream))

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
            signal_index: Signal index identifier. Must lie in
                ``[0, num_rma_sig)``; see :py:attr:`NCCLConfig.num_rma_sig`.
            context: Context identifier. Must lie in ``[0, num_rma_ctx)``;
                see :py:attr:`NCCLConfig.num_rma_ctx`.
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
        :py:meth:`wait_signal`. Both the peer's memory and ``local_buffer``
        must be registered with :py:meth:`register_window`; pass the peer's
        window handle as ``peer_window`` (e.g. obtained via an allgather of
        window handles).

        Args:
            local_buffer: Source buffer whose contents are put to the peer.
            peer: Target rank to put the data to and send the signal to.
            peer_window: Peer's :py:class:`~nccl.core.RegisteredWindowHandle`
                (from :py:meth:`register_window`).
            peer_window_offset: Offset in the peer's window in elements.
                Defaults to 0.
            signal_index: Signal index identifier. Must lie in
                ``[0, num_rma_sig)``; see :py:attr:`NCCLConfig.num_rma_sig`.
            context: Context identifier. Must lie in ``[0, num_rma_ctx)``;
                see :py:attr:`NCCLConfig.num_rma_ctx`.
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
            peer_window._window,
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
        config: NCCLCollConfig | None = None,
    ) -> None:
        """All-reduce variant of :py:meth:`reduce`.

        Equivalent to ``reduce(sendbuf, recvbuf, op, root=None, stream=stream)``:
        reduces data across all ranks and stores identical copies in each
        rank's recvbuf. ``config`` is forwarded unchanged. See
        :py:meth:`reduce` for argument semantics.

        See Also:
            :py:meth:`reduce`, :c:func:`ncclAllReduce`
        """
        self._check_valid("allreduce")

        self.reduce(sendbuf, recvbuf, op, stream=stream, config=config)

    def broadcast(
        self,
        sendbuf: NcclBufferSpec | Any,
        recvbuf: NcclBufferSpec,
        root: int,
        *,
        stream: NcclStreamSpec | None = None,
        config: NCCLCollConfig | None = None,
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
            config: Per-call :py:class:`NCCLCollConfig`. Must be identical on
                every rank. Defaults to ``None`` (NCCL's own defaults).

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

        cfg, _cfg_keepalive = _materialize_coll_config(config)
        if cfg is None:
            _nccl_bindings.broadcast(
                s_ptr, r_ptr, count, int(dtype), int(root), self._comm, get_stream_ptr(stream)
            )
        else:
            _nccl_bindings.broadcast_config(
                s_ptr, r_ptr, count, int(dtype), int(root), self._comm,
                get_stream_ptr(stream), cfg.ptr,
            )

    def reduce(
        self,
        sendbuf: NcclBufferSpec,
        recvbuf: NcclBufferSpec | Any,
        op: NcclRedOp | CustomRedOp,
        root: int | None = None,
        *,
        stream: NcclStreamSpec | None = None,
        config: NCCLCollConfig | None = None,
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
            op: Reduction operator (e.g. :py:attr:`NcclRedOp.SUM`,
                :py:attr:`NcclRedOp.MAX`, :py:attr:`NcclRedOp.MIN`,
                :py:attr:`NcclRedOp.AVG`, :py:attr:`NcclRedOp.PROD`, or a
                :py:class:`~nccl.core.CustomRedOp`).
            root: Root rank that receives the reduced result (0 to
                ``nranks - 1``). If ``None``, performs an all-reduce.
                Defaults to ``None``.
            stream: CUDA stream for the operation. Defaults to ``None`` (the
                default stream).
            config: Per-call :py:class:`NCCLCollConfig`. Must be identical on
                every rank. Defaults to ``None`` (NCCL's own defaults).

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

        cfg, _cfg_keepalive = _materialize_coll_config(config)
        if root is None and cfg is None:
            _nccl_bindings.all_reduce(
                s_ptr, r_ptr, count, int(dtype), int(op), self._comm, get_stream_ptr(stream)
            )
        elif root is None:
            _nccl_bindings.all_reduce_config(
                s_ptr, r_ptr, count, int(dtype), int(op), self._comm,
                get_stream_ptr(stream), cfg.ptr,
            )
        elif cfg is None:
            _nccl_bindings.reduce(
                s_ptr, r_ptr, count, int(dtype), int(op), int(root), self._comm,
                get_stream_ptr(stream),
            )
        else:
            _nccl_bindings.reduce_config(
                s_ptr, r_ptr, count, int(dtype), int(op), int(root), self._comm,
                get_stream_ptr(stream), cfg.ptr,
            )

    def allgather(
        self,
        sendbuf: NcclBufferSpec,
        recvbuf: NcclBufferSpec,
        *,
        stream: NcclStreamSpec | None = None,
        config: NCCLCollConfig | None = None,
    ) -> None:
        """All-gather variant of :py:meth:`gather`.

        Equivalent to ``gather(sendbuf, recvbuf, root=None, stream=stream)``:
        gathers ``sendcount`` values from each rank and places identical
        copies of the concatenated result in every rank's recvbuf.
        ``config`` is forwarded unchanged. See :py:meth:`gather` for
        argument semantics.

        See Also:
            :py:meth:`gather`, :c:func:`ncclAllGather`
        """
        self._check_valid("allgather")

        self.gather(sendbuf, recvbuf, stream=stream, config=config)

    def reduce_scatter(
        self,
        sendbuf: NcclBufferSpec,
        recvbuf: NcclBufferSpec,
        op: NcclRedOp | CustomRedOp,
        *,
        stream: NcclStreamSpec | None = None,
        config: NCCLCollConfig | None = None,
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
            op: Reduction operator (e.g. :py:attr:`NcclRedOp.SUM`,
                :py:attr:`NcclRedOp.MAX`, :py:attr:`NcclRedOp.MIN`,
                :py:attr:`NcclRedOp.AVG`, :py:attr:`NcclRedOp.PROD`, or a
                :py:class:`~nccl.core.CustomRedOp`).
            stream: CUDA stream for the operation. Defaults to ``None`` (the
                default stream).
            config: Per-call :py:class:`NCCLCollConfig`. Must be identical on
                every rank. Defaults to ``None`` (NCCL's own defaults).

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

        cfg, _cfg_keepalive = _materialize_coll_config(config)
        if cfg is None:
            _nccl_bindings.reduce_scatter(
                s_ptr, r_ptr, count, int(dtype), int(op), self._comm, get_stream_ptr(stream)
            )
        else:
            _nccl_bindings.reduce_scatter_config(
                s_ptr, r_ptr, count, int(dtype), int(op), self._comm,
                get_stream_ptr(stream), cfg.ptr,
            )

    def alltoall(
        self,
        sendbuf: NcclBufferSpec,
        recvbuf: NcclBufferSpec,
        *,
        stream: NcclStreamSpec | None = None,
        config: NCCLCollConfig | None = None,
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
            config: Per-call :py:class:`NCCLCollConfig`. Must be identical on
                every rank. Defaults to ``None`` (NCCL's own defaults).

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

        cfg, _cfg_keepalive = _materialize_coll_config(config)
        if cfg is None:
            _nccl_bindings.allto_all(
                s_ptr, r_ptr, count, int(dtype), self._comm, get_stream_ptr(stream)
            )
        else:
            _nccl_bindings.allto_all_config(
                s_ptr, r_ptr, count, int(dtype), self._comm,
                get_stream_ptr(stream), cfg.ptr,
            )

    def gather(
        self,
        sendbuf: NcclBufferSpec,
        recvbuf: NcclBufferSpec | Any,
        root: int | None = None,
        *,
        stream: NcclStreamSpec | None = None,
        config: NCCLCollConfig | None = None,
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
            config: Per-call :py:class:`NCCLCollConfig`. Must be identical on
                every rank. Defaults to ``None`` (NCCL's own defaults).

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

        cfg, _cfg_keepalive = _materialize_coll_config(config)
        if root is None and cfg is None:
            _nccl_bindings.all_gather(
                s_ptr, r_ptr, count, int(dtype), self._comm, get_stream_ptr(stream)
            )
        elif root is None:
            _nccl_bindings.all_gather_config(
                s_ptr, r_ptr, count, int(dtype), self._comm,
                get_stream_ptr(stream), cfg.ptr,
            )
        elif cfg is None:
            _nccl_bindings.gather(
                s_ptr, r_ptr, count, int(dtype), int(root), self._comm, get_stream_ptr(stream)
            )
        else:
            _nccl_bindings.gather_config(
                s_ptr, r_ptr, count, int(dtype), int(root), self._comm,
                get_stream_ptr(stream), cfg.ptr,
            )

    def scatter(
        self,
        sendbuf: NcclBufferSpec | Any,
        recvbuf: NcclBufferSpec,
        root: int,
        *,
        stream: NcclStreamSpec | None = None,
        config: NCCLCollConfig | None = None,
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
            config: Per-call :py:class:`NCCLCollConfig`. Must be identical on
                every rank. Defaults to ``None`` (NCCL's own defaults).

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

        cfg, _cfg_keepalive = _materialize_coll_config(config)
        if cfg is None:
            _nccl_bindings.scatter(
                s_ptr, r_ptr, count, int(dtype), int(root), self._comm, get_stream_ptr(stream)
            )
        else:
            _nccl_bindings.scatter_config(
                s_ptr, r_ptr, count, int(dtype), int(root), self._comm,
                get_stream_ptr(stream), cfg.ptr,
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
    ) -> RegisteredWindowHandle:
        """Collectively registers a local buffer into an NCCL window.

        This is a collective call: every rank in the communicator must
        participate. Buffer size is automatically derived from buffer count
        and dtype.

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
            window. Its handle remains ``0`` while registration is pending,
            or when windows are unsupported on the platform.

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

        # NCCL itself gates registration on platform support. Always retain and
        # return its handle storage because a null handle can also mean that a
        # grouped or non-blocking registration has not filled it yet.
        resource = RegisteredWindowHandle(self._comm, buffer_ptr, size, flags)
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

        This is a collective call: every rank in the communicator must
        participate. When called inside a group, the result may not be filled
        in until the group completes.

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

        requirements = requirements or NCCLDevCommRequirements()
        multimem_handles: dict[NCCLTeam, _nccl_bindings.MultimemHandle] | None = None
        resource_handles: (
            tuple[
                _nccl_bindings.LsaBarrierHandle
                | _nccl_bindings.GinBarrierHandle
                | _nccl_bindings.LLA2AHandle,
                ...,
            ]
            | None
        ) = None

        reqs = requirements._to_lowpp()
        if requirements.teams:
            team_nodes, multimem_handles = _materialize_team_requirements(requirements.teams)
            reqs.team_requirements_list = team_nodes[0].ptr

        if requirements.resources:
            resource_nodes, resource_handles = _materialize_resource_requirements(
                self._comm, requirements.resources
            )
            reqs.resource_requirements_list = resource_nodes[0].ptr

        # team_nodes and resource_nodes keep the linked-list nodes alive through
        # the resource constructor's synchronous ncclDevCommCreate; the resource
        # nodes also point into the handle storage retained by the resource.
        resource = DevCommResource(self._comm, reqs, multimem_handles, resource_handles)
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
