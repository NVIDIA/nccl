# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pure Python ctypes wrapper for the NCCL checkpoint shim."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from typing import Any

ncclResult_t = ctypes.c_int


@dataclass
class Function:
    name: str
    restype: Any
    argtypes: list[Any]


@dataclass(frozen=True)
class VersionInfo:
    checkpoint_version: int
    nccl_version: int


class NCCLCheckpointError(RuntimeError):
    """Raised when a checkpoint shim call returns a non-success status."""


class NCCLCheckpointPreloadError(NCCLCheckpointError):
    """Raised when checkpoint shim symbols are unavailable in this process."""


class NCCLCheckpointLibrary:
    exported_functions = [
        Function(
            "ncclCheckpointGetVersion",
            ncclResult_t,
            [ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)],
        ),
        Function("ncclCheckpointPrepare", ncclResult_t, []),
        Function("ncclCheckpointRestore", ncclResult_t, []),
    ]

    _process_library: ctypes.CDLL | None = None

    def __init__(self):
        if self.__class__._process_library is None:
            self.__class__._process_library = ctypes.CDLL(None)
        self.lib = self.__class__._process_library
        self._bind_functions()

    def _bind_functions(self) -> None:
        for fn in self.exported_functions:
            try:
                c_fn = getattr(self.lib, fn.name)
            except AttributeError as err:
                raise NCCLCheckpointPreloadError(
                    f"{fn.name} is not loaded in the current process; launch Python with "
                    "libnccl-checkpoint-shim.so in LD_PRELOAD or link the executable "
                    "against the checkpoint shim"
                ) from err
            c_fn.restype = fn.restype
            c_fn.argtypes = fn.argtypes
            setattr(self, fn.name, c_fn)

    def check_result(self, result: int, fn_name: str) -> None:
        if result != 0:
            raise NCCLCheckpointError(f"{fn_name} failed with ncclResult_t={result}")

    def checkpoint_prepare(self) -> None:
        self.check_result(self.ncclCheckpointPrepare(), "ncclCheckpointPrepare")

    def checkpoint_restore(self) -> None:
        self.check_result(self.ncclCheckpointRestore(), "ncclCheckpointRestore")

    def get_version(self) -> VersionInfo:
        checkpoint_version = ctypes.c_int()
        nccl_version = ctypes.c_int()
        self.check_result(
            self.ncclCheckpointGetVersion(
                ctypes.byref(checkpoint_version),
                ctypes.byref(nccl_version),
            ),
            "ncclCheckpointGetVersion",
        )
        return VersionInfo(
            checkpoint_version=checkpoint_version.value,
            nccl_version=nccl_version.value,
        )
