# SPDX-FileCopyrightText: Copyright (c) 2026 Kuaishou Technology. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Byte tables: the only contract between a layout and the transport.

A shard is the concatenation of P segments (one per parameter); segment p has
``sizes[p]`` bytes at shard offset ``offsets[p]``. After the all-gather, the
piece of segment p that came from rank r lives at

    region + offsets[p] * nranks + r * sizes[p]

so the nranks pieces of one parameter are contiguous (parameter-major).
"""

from __future__ import annotations

from typing import Sequence


def byte_table(segment_bytes: Sequence[int]) -> tuple[list[int], list[int]]:
    sizes = [int(s) for s in segment_bytes]
    if not sizes or any(s <= 0 for s in sizes):
        raise ValueError("segments must be non-empty and positive")
    offsets, off = [], 0
    for s in sizes:
        offsets.append(off)
        off += s
    return sizes, offsets


def check_alignment(sizes: Sequence[int], offsets: Sequence[int], itemsizes: Sequence[int], nranks: int) -> str | None:
    """Return a reason string if a landing or a packed-shard piece cannot be retyped to its dtype.

    Landing piece p sits at offsets[p]*nranks (+ r*sizes[p]); a packed shard sits at a 4 KiB-aligned
    stage base (+ node*shard) with piece p at offsets[p]."""
    shard = sum(sizes)
    for p, (s, o, it) in enumerate(zip(sizes, offsets, itemsizes)):
        if (o * nranks) % it or s % it or o % it or shard % it:
            return f"segment {p}: offset {o} / landing offset {o * nranks} / size {s} / shard {shard} not multiples of itemsize {it}"
    return None
