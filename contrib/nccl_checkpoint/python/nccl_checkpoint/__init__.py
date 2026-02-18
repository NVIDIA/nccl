# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from ._version import __version__
from .nccl_wrapper import (
    NCCLCheckpointError,
    NCCLCheckpointLibrary,
    NCCLCheckpointPreloadError,
    VersionInfo,
)

_library = None


def _get_library() -> NCCLCheckpointLibrary:
    global _library
    if _library is None:
        _library = NCCLCheckpointLibrary()
    return _library


def checkpoint_prepare() -> None:
    _get_library().checkpoint_prepare()


def checkpoint_restore() -> None:
    _get_library().checkpoint_restore()


def get_version() -> VersionInfo:
    return _get_library().get_version()


__all__ = [
    "__version__",
    "NCCLCheckpointError",
    "NCCLCheckpointLibrary",
    "NCCLCheckpointPreloadError",
    "VersionInfo",
    "checkpoint_prepare",
    "checkpoint_restore",
    "get_version",
]
