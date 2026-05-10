# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pure-Python enum definitions mirroring the NCCL EP C enums.

Defined here (rather than re-exported from :mod:`nccl.ep.bindings.nccl_ep`)
so the public API does not depend on cybind's enum naming conventions and
so each member can carry a docstring describing its semantics. Values mirror
the corresponding C enums in ``ep_enums.h``.
"""

from enum import IntEnum

__all__ = ["NcclEpAlgorithm", "NcclEpLayout"]


class NcclEpAlgorithm(IntEnum):
    """EP communication algorithm, mirroring :c:type:`ncclEpAlgorithm_t`.

    Set on :py:attr:`EpGroupConfig.algorithm` before calling
    :py:meth:`EpGroup.create` to select the dispatch/combine path.
    """

    LOW_LATENCY = 0
    """Low-Latency (LL) algorithm. Tuned for minimal per-step latency."""

    HIGH_THROUGHPUT = 1
    """High-Throughput (HT) algorithm. Tuned for peak aggregate bandwidth."""


class NcclEpLayout(IntEnum):
    """Receive-buffer layout for dispatch/combine, mirroring :c:type:`ncclEpLayout_t`.

    Set on :py:attr:`EpGroupConfig.layout` before calling
    :py:meth:`EpGroup.create` to control the shape of dispatch output
    tensors and the expected shape of combine input tensors.
    """

    AUTO = 0
    """Auto-select based on algorithm: ``EXPERT_MAJOR`` for LL,
    ``FLAT`` for HT (the only HT-supported layout)."""

    EXPERT_MAJOR = 1
    """LL only. ``recv_x`` shape:
    ``[num_local_experts, max_tokens_per_rank * num_ranks, hidden]``.
    Combine accumulates per-expert contributions on the receive side."""

    RANK_MAJOR = 2
    """LL only. ``recv_x`` shape:
    ``[max_tokens_per_rank * num_ranks, hidden]``. Caller pre-reduces
    across local experts before combine."""

    FLAT = 3
    """HT only. ``recv_x`` shape: ``[N(r), hidden]`` — a single
    contiguous sequence of tokens with no rank or expert structure."""
