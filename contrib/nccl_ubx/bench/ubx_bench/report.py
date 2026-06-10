"""nccl-tests-compatible output formatting."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import List, Dict, Optional


@dataclass
class BenchResult:
    """Result for a single (size, backend) benchmark point."""
    size_bytes: int
    count: int
    dtype: str
    redop: str
    time_us: float
    algbw_gbs: float
    busbw_gbs: float
    errors: int = 0


def compute_bandwidth(size_bytes: int, time_us: float, nranks: int,
                      collective: str) -> tuple[float, float]:
    """Compute algorithm and bus bandwidth matching nccl-tests formulas.

    Returns:
        (algBw_GB/s, busBw_GB/s)
    """
    if time_us <= 0:
        return 0.0, 0.0

    alg_bw = size_bytes / 1e9 / (time_us / 1e6)

    # Bus bandwidth correction factors (matching nccl-tests)
    if collective in ("all_reduce",):
        bus_factor = 2 * (nranks - 1) / nranks
    elif collective in ("reduce_scatter", "all_gather"):
        bus_factor = (nranks - 1) / nranks
    elif collective in ("all_to_all", "alltoallv", "a2av_mxfp8", "a2av_dispatch"):
        bus_factor = (nranks - 1) / nranks
    else:
        bus_factor = 1.0

    bus_bw = alg_bw * bus_factor
    return alg_bw, bus_bw


def format_table(
    collective: str,
    nranks: int,
    backends: List[str],
    results: Dict[str, List[BenchResult]],
) -> str:
    """Format results as nccl-tests-compatible table.

    Args:
        collective: Name of the collective operation.
        nranks: Number of GPUs.
        backends: List of backend names tested.
        results: Map from backend name to list of BenchResult.

    Returns:
        Formatted table string.
    """
    lines = []
    lines.append("#")
    lines.append(f"# ubx_bench {collective}  ({nranks} GPUs, backend={','.join(backends)})")
    lines.append("#")

    # Header
    header_parts = ["#       size         count    type   redop"]
    for backend in backends:
        header_parts.append(f"    time   algbw   busbw")
    header_parts.append("  #wrong")
    lines.append("".join(header_parts))

    unit_parts = ["#        (B)    (elements)                  "]
    for backend in backends:
        unit_parts.append(f"     (us)  (GB/s)  (GB/s)")
    lines.append("".join(unit_parts))

    # Backend name header
    name_parts = ["#" + " " * 44]
    for backend in backends:
        name_parts.append(f"{backend:>25s}")
    lines.append("".join(name_parts))

    # Data rows — iterate by size index
    first_backend = backends[0]
    num_rows = len(results.get(first_backend, []))

    for i in range(num_rows):
        row_parts = []
        # Get size info from first backend's result
        r = results[first_backend][i]
        row_parts.append(f"{r.size_bytes:>12d} {r.count:>12d}    {r.dtype:>4s}     {r.redop:>3s}")

        total_errors = 0
        for backend in backends:
            if backend in results and i < len(results[backend]):
                br = results[backend][i]
                row_parts.append(f"  {br.time_us:>7.2f} {br.algbw_gbs:>6.2f} {br.busbw_gbs:>7.2f}")
                total_errors += br.errors
            else:
                row_parts.append(f"  {'N/A':>7s} {'N/A':>6s} {'N/A':>7s}")

        row_parts.append(f"  {total_errors:>6d}")
        lines.append("".join(row_parts))

    # Average bus bandwidth
    avg_parts = ["# Avg bus bandwidth    :"]
    for backend in backends:
        if backend in results and results[backend]:
            avg_bw = sum(r.busbw_gbs for r in results[backend]) / len(results[backend])
            avg_parts.append(f" {backend}={avg_bw:.1f}")
    lines.append("".join(avg_parts))

    return "\n".join(lines)


def write_json(
    output_file: str,
    collective: str,
    nranks: int,
    backends: List[str],
    results: Dict[str, List[BenchResult]],
):
    """Write results to JSON file."""
    data = {
        "collective": collective,
        "nranks": nranks,
        "backends": backends,
        "results": {},
    }
    for backend, result_list in results.items():
        data["results"][backend] = [
            {
                "size_bytes": r.size_bytes,
                "count": r.count,
                "dtype": r.dtype,
                "redop": r.redop,
                "time_us": r.time_us,
                "algbw_gbs": r.algbw_gbs,
                "busbw_gbs": r.busbw_gbs,
                "errors": r.errors,
            }
            for r in result_list
        ]

    with open(output_file, "w") as f:
        json.dump(data, f, indent=2)
