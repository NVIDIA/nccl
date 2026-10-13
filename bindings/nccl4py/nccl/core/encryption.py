# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# See LICENSE.txt for more license information

"""NCCL Socket Encryption configuration"""

from __future__ import annotations

from dataclasses import dataclass

from nccl.bindings import nccl as _nccl_bindings
from nccl.core._binding_helpers import LowppSpec
from nccl.core.typing import NcclEncryptionMode

# public interface of the package module
__all__ = [
    "NCCLEncryptionConfig",
    "set_encryption",
]

@dataclass(kw_only=True)
class NCCLEncryptionConfig(LowppSpec, lowpp_cls=_nccl_bindings.EncryptionConfig):
    """Global NCCL connection-encryption configuration."""

    mode: NcclEncryptionMode = NcclEncryptionMode.NONE
    """Encryption mode."""

    psk: str | None = None
    """Pre-shared key used by PSK mode."""

def set_encryption(config: NCCLEncryptionConfig | None = None) -> None:
    if config is None:
        _nccl_bindings.set_encryption(0)
        return

    lowpp = config._to_lowpp()
    _nccl_bindings.set_encryption(lowpp)
