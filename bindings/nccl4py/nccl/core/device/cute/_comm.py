# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# See LICENSE.txt for more license information

"""Opaque CuTeDSL view over a host-initialized device communicator."""

from enum import IntEnum

import cutlass
from cutlass import cute
from cutlass.cutlass_dsl import ir

from nccl.core.resources import DevCommResource

from . import _nccl_device_ir as _ir
from ._gin import Gin, GinAllContexts, GinBackendMask, GinResourceSharingMode
from ._handles import MultimemHandle
from ._team import Team
from ._types import CftLeInfo

__all__ = [
    "CftTeamMode",
    "DevComm",
]


class CftTeamMode(IntEnum):
    """Team layout selected by :meth:`DevComm.team_cft`."""

    FLAT = _ir._NCCL_CFT_TEAM_FLAT  # every rank in the CFT unicast group
    # Same index across multicast CFT groups.
    HIER_MULTIMEM = _ir._NCCL_CFT_TEAM_HIER_MULTIMEM
    HIER_LSA = _ir._NCCL_CFT_TEAM_HIER_LSA  # same index across LSA groups


class DevComm:
    """CuTeDSL view over a host-owned device communicator.

    Construct from a host ``DevCommResource`` with ``runtime.to_cute(resource)``
    or ``DevComm(resource)``. Keep the resource open until device work completes.
    Queries and operations are available inside CuTeDSL code.
    """

    def __init__(self, resource: DevCommResource) -> None:
        if not isinstance(resource, DevCommResource):
            raise TypeError(
                f"DevComm expects an nccl.core.DevCommResource, got {type(resource).__name__}"
            )
        self._resource = resource
        self._value = None

    def __c_pointers__(self) -> list[int]:
        if self._resource is None:
            raise cutlass.DSLRuntimeError(
                "DevComm.__c_pointers__ is only available on a host-mode DevComm"
            )
        return [self._resource.ptr]

    @staticmethod
    def __get_mlir_types__() -> list[ir.Type]:
        return _ir._NcclIrDevComm.__get_mlir_types__()

    def __extract_mlir_values__(self) -> list[ir.Value] | None:
        if self._value is None:
            return None
        return self._value.__extract_mlir_values__()

    def __new_from_mlir_values__(self, values: list[ir.Value]) -> "DevComm":
        obj = object.__new__(type(self))
        obj._resource = None
        obj._value = _ir._NcclIrDevComm(values[0])
        return obj

    def _to_ptr(self, *, at_entry: bool = False) -> ir.Value:
        """Materialize storage, optionally at entry for calls that do not retain it."""
        if self._value is None:
            raise cutlass.DSLRuntimeError("DevComm device storage is only available inside JIT")
        ptr = _ir._NcclIrDevComm._allocate(at_entry=at_entry)
        self._value._store(ptr)
        return ptr

    @property
    def rank(self) -> cutlass.Int32:
        """The calling rank's index in the world team."""
        return cutlass.Int32(_ir._nccl_ir_dev_comm_rank(self._to_ptr(at_entry=True)))

    @property
    def n_ranks(self) -> cutlass.Int32:
        """The number of ranks in the communicator."""
        return cutlass.Int32(_ir._nccl_ir_dev_comm_n_ranks(self._to_ptr(at_entry=True)))

    @property
    def lsa_rank(self) -> cutlass.Int32:
        """The calling rank's index within its local LSA team."""
        return cutlass.Int32(_ir._nccl_ir_dev_comm_lsa_rank(self._to_ptr(at_entry=True)))

    @property
    def lsa_size(self) -> cutlass.Int32:
        """The number of ranks in the local LSA team."""
        return cutlass.Int32(_ir._nccl_ir_dev_comm_lsa_size(self._to_ptr(at_entry=True)))

    @property
    def gin_context_count(self) -> cutlass.Uint32:
        """The number of GIN contexts created during host setup."""
        return cutlass.Uint32(_ir._nccl_ir_dev_comm_gin_context_count(self._to_ptr(at_entry=True)))

    @property
    def team_world(self) -> Team:
        """The team containing every rank in the communicator."""
        out_team_ptr = _ir._NcclTeam._allocate(at_entry=True)
        _ir._nccl_ir_team_world(self._to_ptr(at_entry=True), out_team_ptr)
        return Team._from_ptr(out_team_ptr)

    @property
    def team_lsa(self) -> Team:
        """The local load/store-accessible team."""
        out_team_ptr = _ir._NcclTeam._allocate(at_entry=True)
        _ir._nccl_ir_team_lsa(self._to_ptr(at_entry=True), out_team_ptr)
        return Team._from_ptr(out_team_ptr)

    @property
    def team_rail(self) -> Team:
        """The team of corresponding ranks across LSA teams."""
        out_team_ptr = _ir._NcclTeam._allocate(at_entry=True)
        _ir._nccl_ir_team_rail(self._to_ptr(at_entry=True), out_team_ptr)
        return Team._from_ptr(out_team_ptr)

    def team_cft(
        self,
        mode: CftTeamMode = CftTeamMode.FLAT,
    ) -> Team:
        """The CFT team, in the requested layout.

        Args:
            mode: Team layout. Defaults to :py:attr:`CftTeamMode.FLAT`.
        """
        out_team_ptr = _ir._NcclTeam._allocate(at_entry=True)
        _ir._nccl_ir_team_cft(
            self._to_ptr(at_entry=True), out_team_ptr, _ir._NcclCftTeamModeT(mode)
        )
        return Team._from_ptr(out_team_ptr)

    @property
    def team_cft_multimem(self) -> Team:
        """The CFT multimem team."""
        out_team_ptr = _ir._NcclTeam._allocate(at_entry=True)
        _ir._nccl_ir_team_cft_multimem(self._to_ptr(at_entry=True), out_team_ptr)
        return Team._from_ptr(out_team_ptr)

    def team_rank_to_world(
        self,
        team: Team,
        rank: int | cutlass.Integer,
    ) -> cutlass.Int32:
        """Translate ``rank`` within ``team`` to its world rank.

        Args:
            team: team the rank is expressed in.
            rank: peer rank within ``team``.

        Returns:
            Corresponding rank in the world team.
        """
        return cutlass.Int32(
            _ir._nccl_ir_team_rank_to_world(
                self._to_ptr(at_entry=True), team._to_ptr(), cutlass.Int32(rank)
            )
        )

    def team_rank_to_lsa(
        self,
        team: Team,
        rank: int | cutlass.Integer,
    ) -> cutlass.Int32:
        """Translate ``rank`` within ``team`` to its LSA rank.

        ``rank`` must identify a member of this communicator's local LSA team.
        NCCL does not validate membership; a rank outside the local LSA
        produces an out-of-range result.

        Args:
            team: team the rank is expressed in.
            rank: peer rank within ``team``.

        Returns:
            Corresponding rank in the LSA team.
        """
        return cutlass.Int32(
            _ir._nccl_ir_team_rank_to_lsa(
                self._to_ptr(at_entry=True), team._to_ptr(), cutlass.Int32(rank)
            )
        )

    # === Resource buffer pointers ===
    def resource_buffer_local_pointer(
        self,
        dtype: type[cutlass.Numeric],
        handle: int | cutlass.Integer,
    ) -> cute.Pointer:
        """Return a typed pointer to the local resource buffer.

        Args:
            dtype: Element type of the returned pointer.
            handle: ``ncclDevResourceHandle`` from ``DevCommResource``.
        """
        ptr = _ir._nccl_ir_get_resource_buffer_local_pointer(
            self._to_ptr(at_entry=True), _ir._NcclDevResourceHandle(handle)
        )
        return cute.make_ptr(dtype, ptr)

    def resource_buffer_lsa_pointer(
        self,
        dtype: type[cutlass.Numeric],
        handle: int | cutlass.Integer,
        peer: int | cutlass.Integer,
    ) -> cute.Pointer:
        """Return a typed pointer to ``peer``'s LSA resource buffer.

        Args:
            dtype: Element type of the returned pointer.
            handle: ``ncclDevResourceHandle`` from ``DevCommResource``.
            peer: LSA-team peer rank.
        """
        ptr = _ir._nccl_ir_get_resource_buffer_lsa_pointer(
            self._to_ptr(at_entry=True),
            _ir._NcclDevResourceHandle(handle),
            cutlass.Int32(peer),
        )
        return cute.make_ptr(dtype, ptr)

    def resource_buffer_peer_pointer(
        self,
        dtype: type[cutlass.Numeric],
        handle: int | cutlass.Integer,
        team: Team,
        peer: int | cutlass.Integer,
    ) -> cute.Pointer:
        """Return a typed pointer to ``peer``'s resource buffer.

        The peer must be accessible through the resource window's LSA mapping.
        An unsupported peer can produce an invalid, non-null address.

        Args:
            dtype: Element type of the returned pointer.
            handle: ``ncclDevResourceHandle`` from ``DevCommResource``.
            team: Team to address within.
            peer: Rank within ``team``.
        """
        ptr = _ir._nccl_ir_get_resource_buffer_peer_pointer(
            self._to_ptr(at_entry=True),
            _ir._NcclDevResourceHandle(handle),
            team._to_ptr(),
            cutlass.Int32(peer),
        )
        return cute.make_ptr(dtype, ptr)

    def resource_buffer_multimem_pointer(
        self,
        dtype: type[cutlass.Numeric],
        handle: int | cutlass.Integer,
        mm_handle: MultimemHandle,
    ) -> cute.Pointer:
        """Return a typed multicast pointer to the resource buffer.

        Args:
            dtype: Element type of the returned pointer.
            handle: ``ncclDevResourceHandle`` from ``DevCommResource``.
            mm_handle: host-passed multimem handle covering the resource
                window. For the built-in LSA mapping, use
                :meth:`resource_buffer_lsa_multimem_pointer` instead.
        """
        ptr = _ir._nccl_ir_get_resource_buffer_multimem_pointer(
            self._to_ptr(at_entry=True),
            _ir._NcclDevResourceHandle(handle),
            mm_handle._to_ptr(),
        )
        return cute.make_ptr(dtype, ptr)

    def resource_buffer_lsa_multimem_pointer(
        self,
        dtype: type[cutlass.Numeric],
        handle: int | cutlass.Integer,
    ) -> cute.Pointer:
        """Return a typed multicast pointer using the resource's LSA mapping.

        Args:
            dtype: Element type of the returned pointer.
            handle: ``ncclDevResourceHandle`` from ``DevCommResource``.
        """
        ptr = _ir._nccl_ir_get_resource_buffer_lsa_multimem_pointer(
            self._to_ptr(at_entry=True), _ir._NcclDevResourceHandle(handle)
        )
        return cute.make_ptr(dtype, ptr)

    # === Resource-buffer CFT logical endpoints ===
    #
    # As Window.*_le_info, but addressing a resource buffer by handle. The CFT
    # variant takes no team: it resolves against the flat CFT team.
    def resource_buffer_cft_le_info(
        self,
        handle: int | cutlass.Integer,
        peer_cft: int | cutlass.Integer,
    ) -> CftLeInfo:
        """Resolve the buffer to the logical endpoint of ``peer_cft``.

        Args:
            handle: ``ncclDevResourceHandle`` from ``DevCommResource``.
            peer_cft: Rank within the flat CFT team.

        Returns:
            The logical endpoint, as a :class:`CftLeInfo`.
        """
        le_id = cutlass.Array(_ir._NcclCftLeId, 1)
        le_offset = cutlass.Array(_ir._SizeT, 1)
        _ir._nccl_ir_get_resource_buffer_cft_le_info(
            self._to_ptr(at_entry=True),
            _ir._NcclDevResourceHandle(handle),
            cutlass.Int32(peer_cft),
            le_id.data_ptr(),
            le_offset.data_ptr(),
        )
        return CftLeInfo(le_id[0], le_offset[0])

    def resource_buffer_peer_le_info(
        self,
        handle: int | cutlass.Integer,
        peer: int | cutlass.Integer,
    ) -> CftLeInfo:
        """Resolve the buffer to the logical endpoint of ``peer``.

        Args:
            handle: ``ncclDevResourceHandle`` from ``DevCommResource``.
            peer: World rank of the peer. Must fall within the flat CFT team;
                the device API does not check, and a peer outside it yields a
                meaningless logical endpoint.

        Returns:
            The logical endpoint, as a :class:`CftLeInfo`.
        """
        le_id = cutlass.Array(_ir._NcclCftLeId, 1)
        le_offset = cutlass.Array(_ir._SizeT, 1)
        _ir._nccl_ir_get_resource_buffer_peer_le_info(
            self._to_ptr(at_entry=True),
            _ir._NcclDevResourceHandle(handle),
            cutlass.Int32(peer),
            le_id.data_ptr(),
            le_offset.data_ptr(),
        )
        return CftLeInfo(le_id[0], le_offset[0])

    def resource_buffer_multimem_le_info(
        self,
        handle: int | cutlass.Integer,
    ) -> CftLeInfo:
        """Resolve the buffer to its multicast logical endpoint.

        Args:
            handle: ``ncclDevResourceHandle`` from ``DevCommResource``.

        Returns:
            The logical endpoint, as a :class:`CftLeInfo`.
        """
        le_id = cutlass.Array(_ir._NcclCftLeId, 1)
        le_offset = cutlass.Array(_ir._SizeT, 1)
        _ir._nccl_ir_get_resource_buffer_multimem_le_info(
            self._to_ptr(at_entry=True),
            _ir._NcclDevResourceHandle(handle),
            le_id.data_ptr(),
            le_offset.data_ptr(),
        )
        return CftLeInfo(le_id[0], le_offset[0])

    def gin(
        self,
        context_id: int | cutlass.Integer = 0,
        *,
        backend_mask: GinBackendMask = GinBackendMask.ALL,
        resource_sharing_mode: GinResourceSharingMode = GinResourceSharingMode.GPU,
    ) -> Gin:
        """Select a GIN context created during host setup.

        Args:
            context_id: Context index in ``[0, gin_context_count)``. The caller
                must ensure it is in range, including for DSL integer values.
            backend_mask: Compile-time backend mask. Only ``ALL`` is supported;
                other masks raise ``NotImplementedError`` during tracing.
                ``ALL`` dispatches to the backend configured during host setup.
            resource_sharing_mode: scope at which GIN network resources are
                shared: across the GPU, within each CTA, or exclusively by
                each thread. Defaults to ``GPU``.

        Views selecting the same context share its resources; the sharing mode
        promises concurrency limits rather than creating separate partitions.
        This operation neither creates contexts nor fences outstanding work.
        The view and its communicator storage belong to the device function
        in which this factory is traced.
        """
        return Gin(
            self,
            context_id,
            backend_mask=backend_mask,
            resource_sharing_mode=resource_sharing_mode,
        )

    def gin_all_contexts(self) -> GinAllContexts:
        """Select every GIN context on this communicator for a barrier.

        The selector retains the communicator value without creating contexts,
        allocating device storage, or fencing work.
        """
        if self._value is None:
            raise cutlass.DSLRuntimeError("DevComm.gin_all_contexts is only available inside JIT")
        return GinAllContexts(self)
