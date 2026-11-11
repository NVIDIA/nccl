# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# See LICENSE.txt for more license information

"""Extract communication patterns from NCCL Inspector JSON logs."""

from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import sys
from collections import Counter, defaultdict
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import chain
from pathlib import Path
from time import perf_counter

EXTRACTOR_VERSION = "1.2"

OP_EVENT_DATACLASS_OPTIONS = {"frozen": True}
if sys.version_info >= (3, 10):
    OP_EVENT_DATACLASS_OPTIONS["slots"] = True


@dataclass(**OP_EVENT_DATACLASS_OPTIONS)
class OpEvent:
    comm_id: str
    comm_name: str
    rank: int
    n_ranks: int
    nnodes: int
    hostname: str
    pid: int
    op_type: str
    op: str
    msg_size_bytes: int
    seq: int
    dump_ts_us: int
    start_ts_us: int | None
    algo: str | None = None
    proto: str | None = None
    peer: int | None = None

    @property
    def signature(self) -> tuple:
        if self.op_type == "p2p":
            return (self.op, self.msg_size_bytes, self.peer)
        # The plugin null-checks algo and proto independently, so one can be
        # reported while the other is not. Keep whichever survived instead of
        # discarding both, and only fall back to the short form when neither is.
        if self.algo or self.proto:
            return (
                self.op,
                self.msg_size_bytes,
                self.algo or "unknown",
                self.proto or "unknown",
            )
        return (self.op, self.msg_size_bytes)

    @property
    def sort_key(self) -> tuple:
        return (self.start_ts_us or self.dump_ts_us, self.seq, self.op_type)


def smart_open(path: Path, mode: str = "rt"):
    if path.suffix == ".gz":
        return gzip.open(path, mode)
    return open(path, mode)


def intern_optional(value) -> str | None:
    """Intern a low-cardinality algo/proto string, mapping "unknown" to None.

    The plugin writes "unknown" when the core reported no string, which carries
    the same information as an older log that omits the field entirely. None
    therefore means "not reported"; see OpEvent.signature for how that renders.
    """
    if not value or value == "unknown":
        return None
    return sys.intern(str(value))


def comm_key_of(header: dict) -> str:
    comm_id = header.get("id", "unknown")
    comm_name = header.get("comm_name", "unknown")
    rank = header.get("rank", -1)
    return f"{comm_id}:{comm_name}:rank{rank}"


def iter_records(path: Path) -> Iterator[tuple]:
    """Yield ``("op", OpEvent)``, ``("drops", comm_key, coll, p2p)`` and a final
    ``("unparsed", lineno_of_first, count)`` record.

    The *.log glob also matches job stdout that happens to sit next to the dump
    directory, so unparsable lines are counted and reported once per file rather
    than once per line.
    """
    unparsed = 0
    first_bad = 0
    with smart_open(path) as infile:
        for lineno, line in enumerate(infile, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                unparsed += 1
                first_bad = first_bad or lineno
                continue

            header = rec.get("header", {})
            meta = rec.get("metadata", {})

            # Inspector v4.1+ emits one dump_stats marker per comm per dump cycle
            # instead of an operation; the drop counters are cumulative.
            if "dump_stats" in rec:
                stats = rec["dump_stats"]
                yield (
                    "drops",
                    comm_key_of(header),
                    stats.get("coll_dropped_total", 0),
                    stats.get("p2p_dropped_total", 0),
                )
                continue

            base = dict(
                comm_id=sys.intern(str(header.get("id", "unknown"))),
                comm_name=sys.intern(str(header.get("comm_name", "unknown"))),
                rank=header.get("rank", -1),
                n_ranks=header.get("n_ranks", 0),
                nnodes=header.get("nnodes", 0),
                hostname=sys.intern(str(meta.get("hostname", "unknown"))),
                pid=meta.get("pid", 0),
                dump_ts_us=meta.get("dump_timestamp_us", 0),
            )

            if "coll_perf" in rec:
                perf = rec["coll_perf"]
                trace = perf.get("event_trace_ts", {})
                yield "op", OpEvent(
                    **base,
                    op_type="coll",
                    op=sys.intern(str(perf.get("coll", "unknown"))),
                    msg_size_bytes=perf.get("coll_msg_size_bytes", 0),
                    seq=perf.get("coll_sn", 0),
                    start_ts_us=trace.get("coll_start_ts"),
                    algo=intern_optional(perf.get("coll_algo")),
                    proto=intern_optional(perf.get("coll_proto")),
                )
            elif "p2p_perf" in rec:
                perf = rec["p2p_perf"]
                trace = perf.get("event_trace_ts", {})
                yield "op", OpEvent(
                    **base,
                    op_type="p2p",
                    op=sys.intern(str(perf.get("p2p", "unknown"))),
                    msg_size_bytes=perf.get("p2p_msg_size_bytes", 0),
                    seq=perf.get("p2p_sn", 0),
                    start_ts_us=trace.get("p2p_start_ts"),
                    peer=perf.get("p2p_peer"),
                )

    if unparsed:
        yield "unparsed", first_bad, unparsed


def topology_label(n_ranks: int, nnodes: int) -> str:
    if n_ranks <= 1:
        return "single-rank"
    if nnodes <= 1:
        return "nvlink-only"
    if n_ranks == nnodes:
        return "hca-only"
    return "mixed"


def format_signature(sig: tuple) -> str:
    if len(sig) == 2:
        return f"{sig[0]}@{sig[1]}B"
    if len(sig) == 3 and isinstance(sig[2], int):
        return f"{sig[0]}@{sig[1]}B peer={sig[2]}"
    if len(sig) == 4:
        return f"{sig[0]}@{sig[1]}B {sig[2]}_{sig[3]}"
    return str(sig)


def sanitize_filename_component(value: str) -> str:
    cleaned = re.sub(r"[^\w.\-]+", "_", value.strip())
    return cleaned.strip("._") or "unknown"


def infer_job_id(input_dir: Path) -> str | None:
    for var in ("SLURM_JOB_ID", "SLURM_JOBID", "PBS_JOBID", "JOB_ID"):
        value = os.environ.get(var)
        if value:
            return value
    for part in reversed(input_dir.resolve().parts):
        if part.startswith("nccl-inspector-"):
            suffix = part.removeprefix("nccl-inspector-")
            if suffix:
                return suffix
    return None


def infer_cluster_name() -> str | None:
    for var in (
        "SLURM_CLUSTER_NAME",
        "CLUSTER_NAME",
        "NCCL_CLUSTER",
        "CI_CLUSTER",
        "K8S_CLUSTER_NAME",
    ):
        value = os.environ.get(var)
        if value:
            return value
    return None


def resolve_report_metadata(
    input_dir: Path,
    paths: list[Path],
    job_id: str | None,
    cluster: str | None,
) -> dict:
    resolved_job_id = job_id or infer_job_id(input_dir) or "unknown"
    resolved_cluster = cluster or infer_cluster_name() or "unknown-cluster"
    input_resolved = input_dir.resolve()
    source_files = sorted(
        str(p.resolve().relative_to(input_resolved))
        if p.resolve().is_relative_to(input_resolved)
        else str(p.resolve())
        for p in paths
    )
    return {
        "extractor_version": EXTRACTOR_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "job_id": resolved_job_id,
        "cluster": resolved_cluster,
        "input_dir": str(input_resolved),
        "source_log_files": source_files,
        "num_source_files": len(source_files),
    }


def default_output_path(output_dir: Path, job_id: str) -> Path:
    name = f"comm_pattern_analysis_{sanitize_filename_component(job_id)}.json"
    return output_dir / name


def find_repeating_patterns(
    seq: list[tuple], min_len: int = 2, max_len: int = 8
) -> list[tuple[tuple, int]]:
    patterns: Counter = Counter()
    n = len(seq)
    for length in range(min_len, min(max_len + 1, n // 2 + 1)):
        i = 0
        while i <= n - length * 2:
            sub = tuple(seq[i : i + length])
            reps = 1
            j = i + length
            while j + length <= n and tuple(seq[j : j + length]) == sub:
                reps += 1
                j += length
            if reps >= 2:
                patterns[sub] = max(patterns[sub], reps)
                i = j
            else:
                i += 1
    return patterns.most_common(10)


def discover_log_files(input_dir: Path) -> list[Path]:
    patterns = ("*.log", "*.log.gz", "*.jsonl", "*.jsonl.gz")
    return sorted(chain.from_iterable(input_dir.rglob(p) for p in patterns))


def parse_log_file(path: Path) -> tuple:
    by_comm: dict[str, list[OpEvent]] = defaultdict(list)
    op_histogram: Counter = Counter()
    drops: dict[str, tuple[int, int]] = {}
    unparsed = None

    for record in iter_records(path):
        kind = record[0]
        if kind == "drops":
            _, key, coll_dropped, p2p_dropped = record
            prev_coll, prev_p2p = drops.get(key, (0, 0))
            drops[key] = (max(prev_coll, coll_dropped), max(prev_p2p, p2p_dropped))
            continue
        if kind == "unparsed":
            unparsed = (record[1], record[2])
            continue
        event = record[1]
        key = f"{event.comm_id}:{event.comm_name}:rank{event.rank}"
        by_comm[key].append(event)
        op_histogram[event.signature] += 1

    return dict(by_comm), op_histogram, drops, path, unparsed


def analyze_logs(
    paths: list[Path],
    report_metadata: dict,
    output_json: Path | None,
) -> dict:
    by_comm: dict[str, list[OpEvent]] = defaultdict(list)
    op_histogram: Counter = Counter()
    drops: dict[str, tuple[int, int]] = {}

    skipped: list[str] = []

    max_workers = min(len(paths), os.cpu_count() or 1)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for file_by_comm, file_histogram, file_drops, path, unparsed in executor.map(
            parse_log_file, paths
        ):
            for comm_key, events in file_by_comm.items():
                by_comm[comm_key].extend(events)
            op_histogram.update(file_histogram)
            for comm_key, (coll_dropped, p2p_dropped) in file_drops.items():
                prev_coll, prev_p2p = drops.get(comm_key, (0, 0))
                drops[comm_key] = (
                    max(prev_coll, coll_dropped),
                    max(prev_p2p, p2p_dropped),
                )
            if unparsed:
                first_bad, count = unparsed
                skipped.append(f"{path} ({count} line(s), first at line {first_bad})")
                print(
                    f"warning: skipped {count} unparsable line(s) in {path}, "
                    f"first at line {first_bad}; not an inspector JSON log?",
                    file=sys.stderr,
                )

    num_events = sum(len(events) for events in by_comm.values())
    dropped_total = sum(coll + p2p for coll, p2p in drops.values())
    if dropped_total:
        print(
            f"warning: inspector dropped {dropped_total} record(s) from its ring "
            "buffer; the extracted sequences are incomplete. Raise "
            "NCCL_INSPECTOR_DUMP_COLL_RING_SIZE or shorten the dump interval.",
            file=sys.stderr,
        )
    report_metadata = {
        **report_metadata,
        "num_events": num_events,
        "num_dropped_records": dropped_total,
        "files_with_unparsable_lines": sorted(skipped),
    }

    report: dict = {
        "report_metadata": report_metadata,
        "global_histogram": [
            {"signature": sig, "count": count}
            for sig, count in op_histogram.most_common()
        ],
        "communicators": [],
    }

    for comm_key, events in sorted(by_comm.items()):
        events.sort(key=lambda e: e.sort_key)
        sigs = [e.signature for e in events]
        topo = topology_label(events[0].n_ranks, events[0].nnodes)
        coll_dropped, p2p_dropped = drops.get(comm_key, (0, 0))
        comm_report = {
            "comm_key": comm_key,
            "topology": topo,
            "n_ranks": events[0].n_ranks,
            "nnodes": events[0].nnodes,
            "hostname": events[0].hostname,
            "num_ops": len(events),
            "num_dropped_coll_records": coll_dropped,
            "num_dropped_p2p_records": p2p_dropped,
            "op_kinds": sorted({e.op_type for e in events}),
            "sequence_preview": [format_signature(s) for s in sigs[:20]],
            "repeating_patterns": [
                {
                    "repeats": reps,
                    "pattern": [format_signature(step) for step in pat],
                }
                for pat, reps in find_repeating_patterns(sigs)
            ],
        }
        report["communicators"].append(comm_report)

    if output_json:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(output_json, "w") as out:
            json.dump(report, out, indent=2)

    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract communication patterns from NCCL Inspector JSON logs."
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        required=True,
        help="Directory containing NCCL Inspector log files.",
    )
    parser.add_argument(
        "--output_json",
        type=Path,
        default=None,
        help=(
            "Path for the pattern report JSON. When omitted, writes "
            "comm_pattern_analysis_<job_id>.json under --output_dir "
            "(job_id falls back to 'unknown')."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help=(
            "Directory for the auto-generated report (default: --input_dir). "
            "Ignored when --output_json is set."
        ),
    )
    parser.add_argument(
        "--job_id",
        default=None,
        help=(
            "Optional SLURM/PBS job id for the report filename and metadata. "
            "Defaults to SLURM_JOB_ID / SLURM_JOBID / PBS_JOBID or the "
            "nccl-inspector-<id> suffix in --input_dir, then 'unknown'."
        ),
    )
    parser.add_argument(
        "--cluster",
        default=None,
        help=(
            "Optional cluster name for report metadata. Defaults to "
            "SLURM_CLUSTER_NAME, CLUSTER_NAME, or NCCL_CLUSTER, then "
            "'unknown-cluster'."
        ),
    )
    args = parser.parse_args()

    started_at = datetime.now(timezone.utc)
    started = perf_counter()
    print(f"Started at {started_at.isoformat()}")

    if not args.input_dir.is_dir():
        raise SystemExit(f"Input directory not found: {args.input_dir}")

    paths = discover_log_files(args.input_dir)
    if not paths:
        raise SystemExit(f"No inspector logs found under {args.input_dir}")

    report_metadata = resolve_report_metadata(
        args.input_dir, paths, args.job_id, args.cluster
    )

    if args.output_json is not None:
        output_json = args.output_json
    else:
        output_dir = (args.output_dir or args.input_dir).resolve()
        output_json = default_output_path(output_dir, report_metadata["job_id"])

    analyze_logs(paths, report_metadata, output_json)
    elapsed_s = perf_counter() - started
    print(f"Wrote pattern report to {output_json}")
    print(f"Elapsed time: {elapsed_s:.2f}s")


if __name__ == "__main__":
    main()
