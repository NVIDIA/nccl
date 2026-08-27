"""Pythonic wrappers for ``ncclDevComm`` and ``ncclTeam``.

:class:`DevComm` is an explicit CuTeDSL view over a host-side
``DevCommResource``. Its JIT protocol passes a :py:class:`DevCommValue`
native struct by value and reconstructs the same public type in value mode
while tracing. Pointer storage is materialized lazily for device APIs whose
FFI ABI requires a pointer or reference.
"""

import cutlass
from cutlass.cutlass_dsl import ir
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import dsl_user_op

from ...resources import DevCommResource
from . import _bindings
from ._helpers import _alloca_cft_le_info, _alloca_struct, _load_cft_le_info, _to_value
from ._structs import (
    DevCommValue,
    ncclGin_C,
    ncclTeam as Team,
)
from .gin import Gin
from .handles import GinBarrierHandle, LsaBarrierHandle, MultimemHandle
from .types import CftLeInfo, CftTeamMode, GinBackendMask, GinResourceSharingMode


def _device_function_entry_block():
    """Return the enclosing device function's entry block."""
    op = ir.InsertionPoint.current.block.owner
    while op is not None:
        parent = op.parent
        if parent is not None and parent.name == "gpu.module":
            if len(op.regions) == 0 or len(op.regions[0].blocks) == 0:
                break
            return op.regions[0].blocks[0]
        op = parent
    raise cutlass.DSLRuntimeError(
        "Unable to find the device function entry block for DevComm.ptr"
    )


@dsl_user_op
def _materialize_dev_comm(value, *, loc=None, ip=None) -> ir.Value:
    """Materialize a by-value DevComm in the device function entry block."""
    entry_block = _device_function_entry_block()
    struct_value = value.__extract_mlir_values__()[0]
    # ip is not forwarded: the alloca must land at entry-block begin so the
    # pointer dominates every use, not at the caller's position.
    with ir.InsertionPoint.at_block_begin(entry_block):
        ptr = _alloca_struct(DevCommValue, alignment=8, loc=loc)

    # The store has to follow both operands. The struct is a block argument
    # while it stays within one block, and then it already dominates the
    # alloca. But a DevComm used on both sides of a conditional is threaded
    # through the region by the DSL, making the value an scf.if result — so
    # anchor the store to whatever defines it.
    owner = struct_value.owner
    after = ptr.owner if isinstance(owner, ir.Block) else owner
    with ir.InsertionPoint.after(after):
        llvm.store(struct_value, ptr, loc=loc)
    return ptr


class DevComm:
    """CuTeDSL view over a host-owned, by-value ``ncclDevComm``.

    ``DevComm(resource)`` creates a host-mode JIT argument that keeps the
    resource alive. CuTeDSL reconstructs value-mode instances without calling
    ``__init__``. The native struct is exposed as ``value`` for field reads
    while tracing; mutating it is unsupported. ``ptr`` is materialized once,
    on demand, and is never included in the JIT or kernel ABI.
    """

    def __init__(self, resource: DevCommResource):
        if not isinstance(resource, DevCommResource):
            raise TypeError(
                "DevComm expects an nccl.core.DevCommResource, "
                f"got {type(resource).__name__}"
            )
        self._resource = resource
        self._value = None
        self._materialized_ptr = None

    def __c_pointers__(self):
        if self._resource is None:
            raise cutlass.DSLRuntimeError(
                "DevComm.__c_pointers__ is only available on a host-mode "
                "DevComm"
            )
        return [self._resource.dev_comm.ptr]

    @staticmethod
    def __get_mlir_types__():
        return DevCommValue.__get_mlir_types__()

    def __extract_mlir_values__(self):
        return self.value.__extract_mlir_values__()

    def __new_from_mlir_values__(self, values):
        obj = object.__new__(type(self))
        obj._resource = None
        obj._value = DevCommValue(values[0])
        obj._materialized_ptr = None
        return obj

    @property
    def value(self) -> DevCommValue:
        """Value-mode native struct backing the field reads."""
        if self._value is None:
            raise cutlass.DSLRuntimeError(
                "DevComm fields are only available while tracing CuTeDSL code"
            )
        return self._value

    @property
    def ptr(self) -> ir.Value:
        """Device-local pointer to this value, materialized on first use."""
        if self._materialized_ptr is None:
            self._materialized_ptr = _materialize_dev_comm(self.value)
        return self._materialized_ptr

    # === Scalar fields ===

    @property
    def rank(self) -> cutlass.Int32:
        return self.value.rank

    @property
    def n_ranks(self) -> cutlass.Int32:
        return self.value.n_ranks

    @property
    def lsa_rank(self) -> cutlass.Int32:
        return self.value.lsa_rank

    @property
    def lsa_size(self) -> cutlass.Int32:
        return self.value.lsa_size

    # === Embedded barrier handles ===

    # Wrapped so an embedded handle and a host-passed one are the same type.

    @property
    def lsa_barrier(self) -> LsaBarrierHandle:
        return LsaBarrierHandle.from_native_struct(self.value.lsa_barrier)

    @property
    def rail_gin_barrier(self) -> GinBarrierHandle:
        return GinBarrierHandle.from_native_struct(self.value.rail_gin_barrier)

    @property
    def hybrid_lsa_barrier(self) -> LsaBarrierHandle:
        return LsaBarrierHandle.from_native_struct(
            self.value.hybrid_lsa_barrier)

    @property
    def hybrid_rail_gin_barrier(self) -> GinBarrierHandle:
        return GinBarrierHandle.from_native_struct(
            self.value.hybrid_rail_gin_barrier)

    @property
    def world_gin_barrier(self) -> GinBarrierHandle:
        return GinBarrierHandle.from_native_struct(
            self.value.world_gin_barrier)

    @property
    def lsa_multimem(self) -> MultimemHandle:
        return MultimemHandle.from_native_struct(self.value.lsa_multimem)

    # === Team factories ===

    @property
    def team_world(self) -> Team:
        return Team(_bindings.nccl_team_world(self.ptr))

    @property
    def team_lsa(self) -> Team:
        return Team(_bindings.nccl_team_lsa(self.ptr))

    @property
    def team_rail(self) -> Team:
        return Team(_bindings.nccl_team_rail(self.ptr))

    def team_cft(self, mode: CftTeamMode = CftTeamMode.FLAT) -> Team:
        """The CFT team, in the requested layout.

        Args:
            mode: Team layout. Defaults to :py:attr:`CftTeamMode.FLAT`,
                matching the C default argument.
        """
        return Team(_bindings.nccl_team_cft(self.ptr, cutlass.Int32(int(mode))))

    @property
    def team_cft_multimem(self) -> Team:
        """The CFT multimem team."""
        return Team(_bindings.nccl_team_cft_multimem(self.ptr))

    def team_rank_to_world(self, team: Team, rank: int) -> cutlass.Int32:
        """Translate ``rank`` within ``team`` to its world rank.

        Args:
            team: team the rank is expressed in.
            rank: peer rank within ``team``.

        Returns:
            Corresponding rank in the world team.
        """
        return cutlass.Int32(_bindings.nccl_team_rank_to_world(
            self.ptr, _to_value(team), cutlass.Int32(rank)))

    def team_rank_to_lsa(self, team: Team, rank: int) -> cutlass.Int32:
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
        return cutlass.Int32(_bindings.nccl_team_rank_to_lsa(
            self.ptr, _to_value(team), cutlass.Int32(rank)))

    # === Resource buffer pointers ===

    def resource_buffer_local_pointer(self, handle: int) -> ir.Value:
        """Translate a resource handle to the local buffer address.

        Args:
            handle: ``ncclDevResourceHandle`` from ``DevCommResource``.

        Returns:
            ``!llvm.ptr`` MLIR value.
        """
        return _bindings.nccl_get_resource_buffer_local_pointer(
            self.ptr, cutlass.Uint32(handle))

    def resource_buffer_lsa_pointer(self, handle: int, peer: int) -> ir.Value:
        """Translate a resource handle to ``peer``'s LSA buffer address.

        Args:
            handle: ``ncclDevResourceHandle`` from ``DevCommResource``.
            peer: LSA-team peer rank.

        Returns:
            ``!llvm.ptr`` MLIR value.
        """
        return _bindings.nccl_get_resource_buffer_lsa_pointer(
            self.ptr, cutlass.Uint32(handle), cutlass.Int32(peer))

    def resource_buffer_peer_pointer(
        self, handle: int, team: Team, peer: int
    ) -> ir.Value:
        """Translate a resource handle to ``peer``'s buffer address.

        Args:
            handle: ``ncclDevResourceHandle`` from ``DevCommResource``.
            team: Team to address within.
            peer: Rank within ``team``.

        Returns:
            ``!llvm.ptr`` MLIR value.
        """
        return _bindings.nccl_get_resource_buffer_peer_pointer(
            self.ptr, cutlass.Uint32(handle), _to_value(team),
            cutlass.Int32(peer))

    def resource_buffer_multimem_pointer(
        self, handle: int, mm_handle: MultimemHandle
    ) -> ir.Value:
        """Translate a resource handle to its multimem buffer address.

        Args:
            handle: ``ncclDevResourceHandle`` from ``DevCommResource``.
            mm_handle: Multimem handle covering the resource window —
                :py:attr:`lsa_multimem`, or one passed in from the host for
                a non-LSA team.

        Returns:
            ``!llvm.ptr`` MLIR value.
        """
        return _bindings.nccl_get_resource_buffer_multimem_pointer(
            self.ptr, cutlass.Uint32(handle), _to_value(mm_handle))

    def resource_buffer_lsa_multimem_pointer(self, handle: int) -> ir.Value:
        """Translate a resource handle to its LSA multimem buffer address.

        Args:
            handle: ``ncclDevResourceHandle`` from ``DevCommResource``.

        Returns:
            ``!llvm.ptr`` MLIR value.
        """
        return _bindings.nccl_get_resource_buffer_lsa_multimem_pointer(
            self.ptr, cutlass.Uint32(handle))

    # === Resource-buffer CFT logical endpoints ===
    #
    # As Window.*_le_info, but addressing a resource buffer by handle. The CFT
    # variant takes no team: it resolves against the flat CFT team.

    def resource_buffer_cft_le_info(
        self, handle: int, peer_cft: int
    ) -> CftLeInfo:
        """Resolve the buffer to the logical endpoint of ``peer_cft``.

        Args:
            handle: ``ncclDevResourceHandle`` from ``DevCommResource``.
            peer_cft: Rank within the flat CFT team.

        Returns:
            The logical endpoint, as a :class:`CftLeInfo`.
        """
        le_id_ptr, le_offset_ptr = _alloca_cft_le_info()
        _bindings.nccl_get_resource_buffer_cft_le_info(
            self.ptr, cutlass.Uint32(handle), cutlass.Int32(peer_cft),
            le_id_ptr, le_offset_ptr)
        return _load_cft_le_info(le_id_ptr, le_offset_ptr)

    def resource_buffer_peer_le_info(
        self, handle: int, peer: int
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
        le_id_ptr, le_offset_ptr = _alloca_cft_le_info()
        _bindings.nccl_get_resource_buffer_peer_le_info(
            self.ptr, cutlass.Uint32(handle), cutlass.Int32(peer),
            le_id_ptr, le_offset_ptr)
        return _load_cft_le_info(le_id_ptr, le_offset_ptr)

    def resource_buffer_multimem_le_info(
        self, handle: int
    ) -> CftLeInfo:
        """Resolve the buffer to its multicast logical endpoint.

        Args:
            handle: ``ncclDevResourceHandle`` from ``DevCommResource``.

        Returns:
            The logical endpoint, as a :class:`CftLeInfo`.
        """
        le_id_ptr, le_offset_ptr = _alloca_cft_le_info()
        _bindings.nccl_get_resource_buffer_multimem_le_info(
            self.ptr, cutlass.Uint32(handle), le_id_ptr, le_offset_ptr)
        return _load_cft_le_info(le_id_ptr, le_offset_ptr)

    # === Gin factory ===

    def gin(
        self,
        backend: GinBackendMask,
        context_id: int,
        *,
        resource_sharing_mode: GinResourceSharingMode = GinResourceSharingMode.GPU,
    ) -> Gin:
        """Allocate and initialize a :class:`Gin` rooted on this comm.

        Args:
            backend: backend selection mask.
            context_id: GIN context id.
            resource_sharing_mode: scope at which GIN network resources are
                shared: across the GPU, within each CTA, or exclusively by
                each thread. Default ``GPU`` matches the NCCL C++ constructor
                default.

        Returns:
            Initialized :class:`Gin`.
        """
        storage = _alloca_struct(ncclGin_C)
        _bindings.nccl_gin_c_init(
            storage, cutlass.Int32(int(backend)), self.ptr,
            cutlass.Int32(context_id), cutlass.Uint8(int(resource_sharing_mode)))
        return Gin(ptr=storage)


@cutlass.register_jit_arg_adapter(DevCommResource)
def _adapt_dev_comm_resource(resource: DevCommResource) -> DevComm:
    """Adapt a device communicator resource to a CuTeDSL view."""
    return DevComm(resource)


__all__ = [
    "Team",
    "DevComm",
]
