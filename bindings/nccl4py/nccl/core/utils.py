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
from functools import cache
from pathlib import Path

import numpy as _np
from packaging.version import Version

from nccl.bindings import nccl as _nccl_bindings
from nccl.core._version import __version__

__all__ = [
    "LibraryInfo",
    "UniqueId",
    "VersionInfo",
    "get_error_string",
    "get_unique_id",
    "get_version",
    "show_versions",
]

# VERSION_STRING in src/init.cc embeds the CUDA toolkit version in libnccl.so.
_NCCL_BANNER = re.compile(rb"NCCL version [^\+\s\x00]+\+cuda(\d+)\.(\d+)")


@dataclass(frozen=True)
class LibraryInfo:
    """Version, CUDA build variant, and loaded path for libnccl."""

    version: Version
    """Loaded libnccl release version."""

    cuda_variant: Version | None
    """CUDA toolkit major.minor used to build libnccl, when discoverable."""

    path: Path | None
    """Path of the loaded libnccl, when discoverable."""

    def __str__(self) -> str:
        result = str(self.version)
        if self.cuda_variant is not None:
            result += f"+cuda{self.cuda_variant}"
        if self.path is not None:
            result += f" @ {self.path}"
        return result


@dataclass(frozen=True)
class VersionInfo:
    """Version snapshot of nccl4py, its bindings, and loaded libnccl."""

    nccl4py: Version
    """nccl4py distribution version."""

    nccl_bindings: Version
    """NCCL header version from which the bindings were generated."""

    libnccl: LibraryInfo | None
    """Loaded libnccl information, or None when the library is unavailable."""


def _decode_version(v: int) -> Version:
    """Decode an NCCL packed version integer (X*10000 + Y*100 + Z, or legacy
    X*1000 + Y*100 + Z) into a packaging Version."""
    if v >= 10000:
        major = v // 10000
        minor = (v % 10000) // 100
        patch = v % 100
    else:
        major = v // 1000
        minor = (v % 1000) // 100
        patch = v % 100
    return Version(f"{major}.{minor}.{patch}")


def _get_lib_version() -> Version:
    """Release version of the loaded ``libnccl.so`` (e.g. ``2.30.0``)."""
    return _decode_version(_nccl_bindings.get_version())


def _get_lib_path() -> Path | None:
    """Path of the loaded ``libnccl.so``, or None if it cannot be determined."""
    raw = _nccl_bindings.get_library_path()
    return Path(raw) if raw else None


@cache
def _extract_cuda_variant(path: Path | None) -> Version | None:
    """Recover ``+cudaA.B`` from libnccl's embedded version banner."""
    if path is None:
        return None
    try:
        with open(path, "rb") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            match = _NCCL_BANNER.search(mm)
            if match is not None:
                return Version(f"{int(match.group(1))}.{int(match.group(2))}")
    except (OSError, ValueError):
        # mmap raises ValueError on a zero-length file.
        pass
    return None


def get_version() -> VersionInfo:
    """Return structured nccl4py, binding, and loaded-libnccl versions."""
    libnccl = None
    try:
        path = _get_lib_path()
        libnccl = LibraryInfo(_get_lib_version(), _extract_cuda_variant(path), path)
    except (ImportError, RuntimeError, _nccl_bindings.NCCLError):
        pass

    return VersionInfo(
        nccl4py=Version(__version__),
        nccl_bindings=Version(_nccl_bindings.__version__),
        libnccl=libnccl,
    )


def show_versions() -> None:
    """Print nccl4py, binding, and loaded-libnccl version information."""
    versions = get_version()

    if versions.libnccl is None:
        libnccl = "not available"
    else:
        libnccl = str(versions.libnccl)

    header = "NCCL4Py versions"

    print()
    print(header)
    print("-" * len(header))
    print(f"nccl4py  : {versions.nccl4py}")
    print(f"bindings : {versions.nccl_bindings}")
    print(f"libnccl  : {libnccl}")


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

    def __init__(self, _internal: _nccl_bindings.UniqueId | None = None) -> None:
        """Initializes a UniqueId.

        Use :py:func:`get_unique_id` to generate a valid unique ID for
        communicator initialization.
        """
        if _internal is None:
            _internal = _nccl_bindings.UniqueId()
        self._internal: _nccl_bindings.UniqueId = _internal

    def __repr__(self) -> str:
        # ncclUniqueId is 128 bytes; show first 8 and last 8 in hex.
        b = self.as_bytes
        return f"<UniqueId: {b[:8].hex()}...{b[-8:].hex()}>"

    def __bytes__(self) -> bytes:
        return bytes(self._internal)

    def __getstate__(self) -> bytes:
        return bytes(self)

    def __setstate__(self, state: bytes) -> None:
        self._internal = _nccl_bindings.UniqueId.from_buffer(state)

    @staticmethod
    def from_bytes(b: bytes | bytearray | memoryview) -> UniqueId:
        """Reconstructs a UniqueId from a bytes-like buffer.

        Args:
            b: Bytes representation of a UniqueId, typically obtained via
                the :py:attr:`as_bytes` property on the producing rank.

        Returns:
            Reconstructed :py:class:`UniqueId`.
        """
        return UniqueId(_nccl_bindings.UniqueId.from_buffer(b))

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
    if empty:
        return UniqueId()
    return UniqueId(_nccl_bindings.get_unique_id())


def get_error_string(nccl_result: _nccl_bindings.Result | int) -> str:
    """Returns a human-readable error string for an NCCL result code.

    Args:
        nccl_result: NCCL result code.

    Returns:
        Human-readable error message corresponding to the result code.
    """
    return _nccl_bindings.get_error_string(int(nccl_result))
