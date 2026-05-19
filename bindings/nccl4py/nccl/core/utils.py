# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# See LICENSE.txt for more license information

"""Utility functions and classes for NCCL operations.

This module provides version information, unique identifiers for communicator
initialization, and error string utilities for NCCL operations.
"""

from __future__ import annotations

import mmap
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as _np
from packaging.version import Version as _Version

from nccl._version import __version__
from nccl import bindings as _nccl_bindings

# libnccl_ep.so is bundled with nccl4py but is CUDA-13-only; nccl.ep's
# import-time CUDA-major gate raises ImportError on CUDA-12 hosts.
try:
    from nccl.ep import bindings as _ep_bindings
except ImportError:
    _ep_bindings = None

__all__ = [
    "LibraryInfo",
    "Version",
    "get_version",
    "UniqueId",
    "get_unique_id",
    "get_error_string",
]


_version_cache: Version | None = None

_NCCL_BANNER = re.compile(rb"NCCL version [^\+\s\x00]+\+cuda(\d+)\.(\d+)")
_NCCL_EP_BANNER = re.compile(rb"NCCL EP version [^\+\s\x00]+\+cuda(\d+)\.(\d+)")

_SO_BASENAME = re.compile(r"^(?P<soname>lib[\w-]+\.so)(?:\.[\d.]+)?$")


def _decode_version(v: int) -> _Version:
    """Decode an NCCL-style packed version integer (X*10000 + Y*100 + Z, or
    legacy X*1000 + Y*100 + Z) into a packaging Version."""
    if v >= 10000:
        major = v // 10000
        minor = (v % 10000) // 100
        patch = v % 100
    else:
        major = v // 1000
        minor = (v % 1000) // 100
        patch = v % 100
    return _Version(f"{major}.{minor}.{patch}")


def _resolve_so_path(soname: str) -> Path | None:
    """Absolute path of a currently-mapped shared library, by SONAME.

    Scans /proc/self/maps for an entry whose basename, after stripping any
    trailing ``.X.Y.Z`` version suffix, equals ``soname``. So ``libnccl.so``
    matches ``libnccl.so.2.30.0`` but not ``libnccl_ep.so.0.1.0``.

    Returns None if the library is not mapped (e.g. not yet dlopened) or if
    /proc/self/maps is unreadable.
    """
    try:
        with open("/proc/self/maps") as f:
            for line in f:
                parts = line.rstrip().split(maxsplit=5)
                if len(parts) < 6:
                    continue
                p = Path(parts[5])
                m = _SO_BASENAME.match(p.name)
                if m is not None and m.group("soname") == soname:
                    return p
    except OSError:
        return None
    return None


def _extract_cuda_variant(path: Path | None, pattern: re.Pattern[bytes]) -> _Version | None:
    """Recover ``+cudaA.B`` from a .so's embedded version banner via mmap."""
    if path is None:
        return None
    try:
        with open(path, "rb") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            m = pattern.search(mm)
            if m is not None:
                return _Version(f"{int(m.group(1))}.{int(m.group(2))}")
    except OSError:
        pass
    return None


@dataclass(frozen=True)
class LibraryInfo:
    """Build- and load-time info for a single shared library.

    Attributes:
        version: Library release version (e.g. ``2.30.0``).
        cuda_variant: CUDA toolkit major.minor the library was built with
            (e.g. ``12.9``), or None if it could not be recovered from the
            embedded version banner.
        path: Absolute path of the loaded ``.so`` file, or None if the library
            is not mapped into the process.
    """

    version: _Version
    cuda_variant: _Version | None
    path: Path | None

    def __str__(self) -> str:
        out = str(self.version)
        if self.cuda_variant is not None:
            out += f"+cuda{self.cuda_variant}"
        if self.path is not None:
            out += f" @ {self.path}"
        return out


@dataclass(frozen=True)
class Version:
    """Aggregate version info for NCCL4Py and its native libraries.

    Attributes:
        nccl4py_version: NCCL4Py package version.
        nccl: Build/load info for ``libnccl.so``.
        nccl_ep: Build/load info for ``libnccl_ep.so``, or None on CUDA-12
            hosts where ``libnccl_ep.so`` cannot load.
    """

    nccl4py_version: _Version
    nccl: LibraryInfo
    nccl_ep: LibraryInfo | None

    def __str__(self) -> str:
        ep = str(self.nccl_ep) if self.nccl_ep is not None else "not available"
        return (
            "\nVersions:\n"
            f"    NCCL4Py: {self.nccl4py_version}\n"
            f"    NCCL: {self.nccl}\n"
            f"    NCCL EP: {ep}\n"
        )


def _nccl_library_info() -> LibraryInfo:
    version = _decode_version(_nccl_bindings.get_version())
    path = _resolve_so_path("libnccl.so")
    return LibraryInfo(
        version=version,
        cuda_variant=_extract_cuda_variant(path, _NCCL_BANNER),
        path=path,
    )


def _nccl_ep_library_info() -> LibraryInfo | None:
    if _ep_bindings is None:
        return None
    version = _decode_version(_ep_bindings.get_version())
    path = _resolve_so_path("libnccl_ep.so")
    return LibraryInfo(
        version=version,
        cuda_variant=_extract_cuda_variant(path, _NCCL_EP_BANNER),
        path=path,
    )


def get_version() -> Version:
    """Returns the version information for NCCL, NCCL EP, and NCCL4Py.

    The result is cached after the first call.

    Returns:
        :py:class:`Version` object containing NCCL4Py and per-library
        :py:class:`LibraryInfo` for NCCL and NCCL EP. ``nccl_ep`` is None
        on CUDA-12 hosts where ``libnccl_ep.so`` cannot load.
    """
    global _version_cache
    if _version_cache is None:
        _version_cache = Version(
            nccl4py_version=_Version(__version__),
            nccl=_nccl_library_info(),
            nccl_ep=_nccl_ep_library_info(),
        )
    return _version_cache


class UniqueId:
    """NCCL unique identifier for communicator initialization.

    A UniqueId is used to coordinate communicator initialization across
    multiple ranks. All ranks must use the same UniqueId to form a
    communicator. Typically one rank generates the UniqueId via
    :py:func:`get_unique_id` and broadcasts it to all other ranks. Three
    serialization paths are supported:

    * **Bytes**: ``bytes(uid)`` (or :py:attr:`as_bytes`) on the producer,
      :py:meth:`from_bytes` on receivers. The bytes of unique ID can be
      transmitted through any byte-oriented channel — a TCP socket, a
      shared filesystem, etc.
    * **NumPy**: :py:attr:`as_ndarray` returns an in-place view of the
      underlying buffer, suitable for NumPy-aware buffer transports such
      as ``mpi4py.MPI.Comm.Bcast`` (uppercase ``B``).
    * **Pickle**: instances are picklable directly, so higher level
      object broadcast helpers like ``mpi4py.MPI.Comm.bcast`` (lowercase
      ``b``) work out of the box.
    """

    def __init__(self) -> None:
        """Initializes an empty UniqueId.

        Use :py:func:`get_unique_id` to generate a valid unique ID for
        communicator initialization.
        """
        self._internal: _nccl_bindings.UniqueId = _nccl_bindings.UniqueId()

    def __repr__(self) -> str:
        # Show first 8 and last 8 bytes in hex
        bytes_data = self.as_bytes
        if len(bytes_data) <= 32:
            hex_str = bytes_data.hex()
        else:
            hex_str = bytes_data[:8].hex() + "..." + bytes_data[-8:].hex()
        return f"<UniqueId: {hex_str}>"

    def __bytes__(self) -> bytes:
        return bytes(self._internal)

    @staticmethod
    def from_bytes(b: bytes | bytearray | memoryview) -> UniqueId:
        """Reconstructs a UniqueId from a bytes-like buffer.

        Args:
            b: Bytes representation of a UniqueId, typically obtained via
                the :py:attr:`as_bytes` property on the producing rank.

        Returns:
            Reconstructed :py:class:`UniqueId`.
        """
        uid = UniqueId.__new__(UniqueId)
        uid._internal = _nccl_bindings.UniqueId.from_buffer(b)
        return uid

    @property
    def ptr(self) -> int:
        """Raw pointer to the underlying NCCL unique ID structure."""
        return self._internal.ptr

    @property
    def as_ndarray(self) -> _np.ndarray:
        """NumPy array view of the unique ID data."""
        return _np.ndarray((1,), dtype=_nccl_bindings.unique_id_dtype, buffer=self._internal).view(
            _np.recarray
        )

    @property
    def as_bytes(self) -> bytes:
        """Bytes representation of the unique ID, suitable for serialization or broadcast."""
        return bytes(self)


def get_unique_id(empty: bool = False) -> UniqueId:
    """Generates a new NCCL unique identifier for communicator initialization.

    Should be called by one rank (typically rank 0); the resulting
    :py:class:`UniqueId` must then be broadcast (e.g. via MPI) to all other
    ranks.

    Args:
        empty: If True, return an empty :py:class:`UniqueId` without calling
            NCCL. Useful when the bytes will be filled in later via
            :py:meth:`UniqueId.from_bytes`. Defaults to False.

    Returns:
        A new :py:class:`UniqueId` to be shared across ranks.
    """
    uid = UniqueId()
    if empty:
        return uid
    _nccl_bindings.get_unique_id(uid.ptr)
    return uid


def get_error_string(nccl_result: _nccl_bindings.Result | int) -> str:
    """Returns a human-readable error string for an NCCL result code.

    Args:
        nccl_result: NCCL result code.

    Returns:
        Human-readable error message corresponding to the result code.
    """
    return _nccl_bindings.get_error_string(int(nccl_result))
