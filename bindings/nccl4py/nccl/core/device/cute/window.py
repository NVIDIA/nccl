"""CuTeDSL view over NCCL registered windows."""

from __future__ import annotations

from typing import TYPE_CHECKING

import cutlass
import cutlass.cute as cute
from cutlass.cutlass_dsl import ir

from ...resources import RegisteredWindowHandle
from . import _bindings
from ._helpers import _alloca_cft_le_info, _load_cft_le_info, _to_value
from ._structs import _LLVMPtrType, ncclTeam as Team
from .handles import MultimemHandle
from .types import CftLeInfo

if TYPE_CHECKING:
    from .comm import DevComm


class Window:
    """CuTeDSL view over an NCCL :c:type:`ncclWindow_t` handle.

    ``Window(resource)`` creates a host-mode JIT argument from a
    :py:class:`~nccl.core.RegisteredWindowHandle`. During tracing, CuTeDSL
    reconstructs the same class in value mode around an ``!llvm.ptr``.

    Args:
        resource: Registered window resource.

    Attributes:
        ptr: Device-side window handle, available while tracing.
    """

    def __init__(self, resource: RegisteredWindowHandle):
        if not isinstance(resource, RegisteredWindowHandle):
            raise TypeError(
                "Window expects an nccl.core.RegisteredWindowHandle, "
                f"got {type(resource).__name__}"
            )
        self._resource = resource
        self._ptr = None

    def __c_pointers__(self) -> list[int]:
        if self._resource is None:
            raise cutlass.DSLRuntimeError(
                "Window.__c_pointers__ is only available on a host-mode Window"
            )
        if not self._resource.is_valid:
            raise RuntimeError(
                "RegisteredWindowHandle does not reference an active NCCL window"
            )
        return [self._resource._window.handle_ptr]

    @staticmethod
    def __get_mlir_types__() -> list[ir.Type]:
        return [_LLVMPtrType.mlir_type()]

    def __extract_mlir_values__(self) -> list[ir.Value]:
        return [self.ptr]

    def __new_from_mlir_values__(self, values: list[ir.Value]) -> Window:
        obj = object.__new__(type(self))
        obj._resource = None
        obj._ptr = values[0]
        return obj

    @property
    def ptr(self) -> ir.Value:
        """Device-side :c:type:`ncclWindow_t` value."""
        if self._ptr is None:
            raise cutlass.DSLRuntimeError(
                "Window.ptr is only available while tracing CuTeDSL code"
            )
        return self._ptr

    def local_pointer(self, offset: int) -> ir.Value:
        """Translate ``offset`` to the local virtual address.

        Args:
            offset: Byte offset within the window.

        Returns:
            ``!llvm.ptr`` MLIR value.
        """
        return _bindings.nccl_get_local_pointer(self.ptr, cutlass.Int64(offset))

    def lsa_pointer(self, offset: int, peer: int) -> ir.Value:
        """Translate ``offset`` to ``peer``'s LSA virtual address.

        Args:
            offset: Byte offset within the window.
            peer: LSA-team peer rank.

        Returns:
            ``!llvm.ptr`` MLIR value.
        """
        return _bindings.nccl_get_lsa_pointer(
            self.ptr, cutlass.Int64(offset), cutlass.Int32(peer))

    def peer_pointer(
        self, offset: int, peer: int, team: Team | None = None
    ) -> ir.Value:
        """Translate ``offset`` to ``peer``'s virtual address.

        Args:
            offset: Byte offset within the window.
            peer: Rank within ``team``.
            team: Team to address within. Defaults to ``None``.

        Returns:
            ``!llvm.ptr`` MLIR value.
        """
        if team is None:
            return _bindings.nccl_get_peer_pointer(
                self.ptr, cutlass.Int64(offset), cutlass.Int32(peer))
        return _bindings.nccl_get_peer_pointer_team(
            self.ptr, cutlass.Int64(offset), _to_value(team), cutlass.Int32(peer))

    def multimem_pointer(
        self, offset: int, mm_handle: MultimemHandle
    ) -> ir.Value:
        """Translate ``offset`` to its multimem virtual address.

        Args:
            offset: Byte offset within the window.
            mm_handle: Multimem handle covering this window —
                :py:attr:`DevComm.lsa_multimem`, or one passed in from the
                host for a non-LSA team.

        Returns:
            ``!llvm.ptr`` MLIR value.
        """
        return _bindings.nccl_get_multimem_pointer(
            self.ptr, cutlass.Int64(offset), _to_value(mm_handle))

    def lsa_multimem_pointer(self, offset: int, dev_comm: DevComm) -> ir.Value:
        """Translate ``offset`` to the LSA multimem virtual address.

        Args:
            offset: Byte offset within the window.
            dev_comm: Device communicator supplying the LSA multimem handle.

        Returns:
            ``!llvm.ptr`` MLIR value.
        """
        return _bindings.nccl_get_lsa_multimem_pointer(
            self.ptr, cutlass.Int64(offset), dev_comm.ptr)

    # === CFT logical endpoints ===
    #
    # CFT ops address memory by an (le_id, le_offset) pair rather than by
    # pointer, so these mirror the *_pointer methods above.

    def cft_le_info(
        self, offset: int, peer_cft: int, cft_team: Team, dev_comm: DevComm
    ) -> CftLeInfo:
        """Resolve ``offset`` to the logical endpoint of ``peer_cft``.

        Args:
            offset: Byte offset within the window.
            peer_cft: Rank within ``cft_team``.
            cft_team: CFT team, from :meth:`DevComm.team_cft`.
            dev_comm: Device communicator owning this window.

        Returns:
            The logical endpoint, as a :class:`CftLeInfo`.
        """
        le_id_ptr, le_offset_ptr = _alloca_cft_le_info()
        _bindings.nccl_get_cft_le_info(
            self.ptr, cutlass.Int64(offset), cutlass.Int32(peer_cft),
            _to_value(cft_team), dev_comm.ptr, le_id_ptr, le_offset_ptr)
        return _load_cft_le_info(le_id_ptr, le_offset_ptr)

    def peer_le_info(
        self, offset: int, peer: int, dev_comm: DevComm
    ) -> CftLeInfo:
        """Resolve ``offset`` to the logical endpoint of ``peer``.

        Args:
            offset: Byte offset within the window.
            peer: World rank of the peer. Must fall within the flat CFT team;
                the device API does not check, and a peer outside it yields a
                meaningless logical endpoint.
            dev_comm: Device communicator owning this window.

        Returns:
            The logical endpoint, as a :class:`CftLeInfo`.
        """
        le_id_ptr, le_offset_ptr = _alloca_cft_le_info()
        _bindings.nccl_get_peer_le_info(
            self.ptr, cutlass.Int64(offset), cutlass.Int32(peer),
            dev_comm.ptr, le_id_ptr, le_offset_ptr)
        return _load_cft_le_info(le_id_ptr, le_offset_ptr)

    def multimem_le_info(
        self, offset: int, dev_comm: DevComm
    ) -> CftLeInfo:
        """Resolve ``offset`` to the multicast logical endpoint.

        Args:
            offset: Byte offset within the window.
            dev_comm: Device communicator owning this window.

        Returns:
            The logical endpoint, as a :class:`CftLeInfo`.
        """
        le_id_ptr, le_offset_ptr = _alloca_cft_le_info()
        _bindings.nccl_get_multimem_le_info(
            self.ptr, cutlass.Int64(offset), dev_comm.ptr,
            le_id_ptr, le_offset_ptr)
        return _load_cft_le_info(le_id_ptr, le_offset_ptr)

    def tensor(self, dtype, layout, offset: int = 0):
        """Construct a ``cute.Tensor`` view over the registered buffer.

        Canonical input to
        :py:meth:`~nccl.core.device.cute.Gin.put`: byte offset relative to
        the window and transfer size are derived from the tensor's iterator
        address and layout.

        Args:
            dtype: cutlass numeric type, such as ``cutlass.Int64``.
            layout: ``cute.Layout`` from ``cute.make_layout(...)``.
            offset: Byte offset within the window. Defaults to ``0``.

        Returns:
            ``cute.Tensor`` view at ``offset``.
        """
        return cute.make_tensor(
            cute.make_ptr(dtype, self.local_pointer(offset)),
            layout,
        )


@cutlass.register_jit_arg_adapter(RegisteredWindowHandle)
def _adapt_registered_window_handle(resource: RegisteredWindowHandle) -> Window:
    """Adapt a registered window resource to a CuTeDSL view."""
    return Window(resource)


__all__ = ["Window"]
