# SPDX-FileCopyrightText: Copyright (c) 2026 Kuaishou Technology. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TIDE: layout-aware zero-SM collectives on NCCL public APIs."""

__version__ = "0.0.1"

from tide.comm import AgHandle, RsHandle, TideComm
from tide.table import byte_table

__all__ = ["AgHandle", "RsHandle", "TideComm", "byte_table"]  # tide.fsdp is imported explicitly
