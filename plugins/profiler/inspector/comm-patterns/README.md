# NCCL Inspector Communication Pattern Extraction

This directory provides tooling to extract and analyze sequence-level communication patterns directly from NCCL Inspector JSON logs. While Prometheus and Grafana excel at aggregated monitoring, this tool is built for granular workload characterization, deep-dive debugging, and post-mortem analysis.

Key Capabilities:

* **Trace Operation Timelines**: Answer exactly what collectives ran, in what order, and with what specific algorithms, protocols, and message sizes.
* **Identify Bottlenecks**: Isolate point-to-point (P2P) peer traffic and identify the most dominant or expensive operations.
* **Detect Repeating Loops**: Automatically discover and extract repeating n-gram operation sequences to characterize standard training steps.
* **Compare Execution**: Highlight communication differences across varying ranks or entirely separate training runs.

Ultimately, this allows you to build a complete, sequential map of your network traffic—all without requiring a dedicated Prometheus stack.

## Summary

1. **JSON output enrichment** (in the inspector plugin) — collective records
   include algorithm and protocol strings (`coll_algo`, `coll_proto`) that were
   previously captured internally but only emitted in Prometheus and OTLP mode,
   where they are bucketed into a single `algo_proto` label.
2. **Pattern extractor tool** — a standalone Python script in this directory
   parses inspector JSON logs and reports per-communicator, per-rank operation
   sequences, histograms, and repeating sub-patterns.

## Quick Start

### Recommended Inspector Configuration

For pattern extraction, use **JSON mode** (not Prometheus):

```bash
export NCCL_PROFILER_PLUGIN=/path/to/libnccl-profiler-inspector.so
export NCCL_INSPECTOR_ENABLE=1
export NCCL_INSPECTOR_DUMP_VERBOSE=1          # event_trace_ts for ordering
export NCCL_INSPECTOR_DUMP_DIR=/nccl-inspector-dump
export NCCL_INSPECTOR_DUMP_THREAD_INTERVAL_MICROSECONDS=500  # or -1 for teardown-only
export NCCL_INSPECTOR_ENABLE_P2P=1          # if P2P patterns matter
export NCCL_DEBUG=INFO                      # Optional
export NCCL_DEBUG_SUBSYS=PROFILE            # Optional

export NCCL_INSPECTOR_PROM_DUMP=0 # Make sure you are in JSON mode
```

#### Parameter notes

Inspector must be run in verbose mode. `coll_sn` is **per collective type**,
not a global sequence counter. NCCL increments `comm->seqNumber[func]`
separately for each op type. To order mixed collectives (e.g. `AllReduce`
then `AllGather`), use `event_trace_ts.coll_start_ts` from verbose output,
or fall back to `dump_timestamp_us`, sequence number, and operation type.

`NCCL_DEBUG=INFO` enables informational NCCL logging, while
`NCCL_DEBUG_SUBSYS=PROFILE` limits that output to Inspector/profiler messages.
These messages include plugin initialization and configuration, dump-thread
status, output file locations, and errors. The settings are optional and are
primarily useful for troubleshooting.

#### Persisting logs from containerized Slurm jobs

The directory configured by `NCCL_INSPECTOR_DUMP_DIR` must be mounted from the
host when launching a container with `srun`. For durable multi-node output,
create a job-specific directory on a shared filesystem and mount it at the
configured container path:

```bash
HOST_DUMP_DIR=/path/to/shared/inspector-logs/${SLURM_JOB_ID}
CONTAINER_DUMP_DIR=/nccl-inspector-dump

mkdir -p "$HOST_DUMP_DIR"
export NCCL_INSPECTOR_DUMP_DIR="$CONTAINER_DUMP_DIR"

srun \
  --container-image=/path/to/image.sqsh \
  --container-mounts="$HOST_DUMP_DIR:$CONTAINER_DUMP_DIR,..." \
  your_nccl_application
```

Using `--container-mounts=/tmp:/nccl-inspector-dump` preserves files outside
the container, but `/tmp` is generally node-local and may be cleaned after the
job. Prefer shared storage when logs must remain available or be collected from
multiple nodes.

Set `NCCL_INSPECTOR_DUMP_DIR` only once; the last shell assignment takes
precedence. Also place every `#SBATCH` directive, including `#SBATCH --output`,
before the first executable shell command so Slurm processes it.

See the [general NCCL Inspector setup instructions](../README.md#example-usage)
for additional launch and configuration details.



### Example Workflow

If you already have inspector log files:

```bash
python plugins/profiler/inspector/comm-patterns/comm_pattern_extractor.py \
  --input_dir /path/to/nccl-inspector-<jobid>/
```

**Requirements**: Python 3.9+ standard library only (no pandas or external dependencies required). Python 3.10+ is highly recommended when parsing massive logs, as it utilizes dataclass slots to significantly reduce memory consumption.

## Changed Files

| File | Change |
|------|--------|
| `plugins/profiler/inspector/inspector_json.cc` | Emit `coll_algo` and `coll_proto` in `coll_perf` JSON objects |
| `plugins/profiler/inspector/comm-patterns/comm_pattern_extractor.py` | Pattern extractor for collectives and P2P |

## JSON Schema Change

Each collective record in inspector JSON output now includes:

```json
{
  "header": {
    "id": "0x7f8c496ae9f661",
    "comm_name": "DP Group 0",
    "rank": 2,
    "n_ranks": 8,
    "nnodes": 1
  },
  "metadata": {
    "inspector_output_format_version": "v4.2",
    "dump_timestamp_us": 1748030377748202,
    "hostname": "example-hostname",
    "pid": 1639453
  },
  "coll_perf": {
    "coll": "AllReduce",
    "coll_algo": "RING",
    "coll_proto": "LL",
    "coll_sn": 1407,
    "coll_msg_size_bytes": 17179869184,
    "coll_exec_time_us": 61974,
    "coll_timing_source": "kernel_gpu",
    "coll_algobw_gbs": 277.210914,
    "coll_busbw_gbs": 485.119099
  }
}
```

### Version

- **v4.2** — adds `coll_algo` and `coll_proto` to `coll_perf` records (this change)
- **v4.1** — adds the per-dump `dump_stats` record reporting ring-buffer drops
- **v4.0** — prior format without per-collective algo/proto fields

Emitted as `inspector_output_format_version` in each JSON record's `metadata`
block (`inspector_json.cc`).

### Field notes

- **`coll_algo` / `coll_proto`** — strings the NCCL core passes to the profiler in
  `ncclProfileColl` (`eDescr->coll.algo` / `.proto`, populated from
  `ncclAlgoToString()` / `ncclProtoToString()`). Typical values:
  - Algorithms: `TREE`, `RING`, `COLLNET_DIRECT`, `COLLNET_CHAIN`, `NVLS`,
    `NVLS_TREE`, `PAT`
  - Protocols: `LL`, `LL128`, `SIMPLE`
- These are **strings**, not integer enums. NCCL stores algo/proto internally as
  `#define` constants (`NCCL_ALGO_*`, `NCCL_PROTO_*` in `nccl_tuner.h`), but the
  profiler plugin API (v2+) passes `const char*`.
- Prometheus output uses a different string format for labels (title-case algo
  names, e.g. `Ring_ll`). JSON uses the uppercase profiler strings.

P2P records are unchanged; they use a `p2p_perf` block with `p2p`, `p2p_peer`,
`p2p_sn`, etc.

## Pattern Extractor

### Location

```
plugins/profiler/inspector/comm-patterns/comm_pattern_extractor.py
```

Stdlib only — no pandas/duckdb dependency.

### Usage

Run from the NCCL repo root. If `--output_json` is omitted, the report is written
as `comm_pattern_analysis_<job_id>.json` under `--input_dir` (or `--output_dir`).
If no job id can be inferred, the filename uses `unknown`:

```bash
python plugins/profiler/inspector/comm-patterns/comm_pattern_extractor.py \
  --input_dir /path/to/nccl-inspector-12345678/
```

`--cluster` and `--job_id` are optional. The extractor infers them from the
SLURM/PBS environment and the `nccl-inspector-<jobid>` input-directory suffix
when possible. Otherwise, metadata uses `unknown-cluster` and `unknown`;
explicit options can override the inferred values.
The JSON report includes a `report_metadata` block listing the cluster, job id,
input directory, and every source log file parsed.

Searches recursively for `*.log`, `*.log.gz`, `*.jsonl`, and `*.jsonl.gz`.

### Output

**Console:**

- Process start time (UTC)
- Path of the written pattern report
- Elapsed runtime
- Warnings for dropped inspector records and for files that carried no parsable
  JSON

**JSON report:** structured report with global histograms, per-communicator
sequences, topology labels, and repeating sub-patterns for downstream tooling.

### Data-loss reporting

Inspector buffers completed operations in a per-communicator ring and can
overwrite unread entries when operations complete faster than the dump thread
drains them. A pattern report built from that sample looks complete but is
missing operations, so the extractor reads the `dump_stats` records (inspector
v4.1+) and surfaces the loss:

| Field | Location | Meaning |
|-------|----------|---------|
| `num_dropped_records` | `report_metadata` | Total coll + P2P records the inspector dropped across all communicators |
| `num_dropped_coll_records` | per communicator | Cumulative collective drops for that communicator |
| `num_dropped_p2p_records` | per communicator | Cumulative P2P drops for that communicator |
| `files_with_unparsable_lines` | `report_metadata` | Files whose lines were not inspector JSON, summarized once per file |

Non-zero drops also print a warning on stderr. Raise
`NCCL_INSPECTOR_DUMP_COLL_RING_SIZE` / `NCCL_INSPECTOR_DUMP_P2P_RING_SIZE` or
shorten `NCCL_INSPECTOR_DUMP_THREAD_INTERVAL_MICROSECONDS` and re-run before
trusting a sequence.

Because the file glob matches any `*.log`, job stdout sitting next to the dump
directory gets picked up; those files are reported once in
`files_with_unparsable_lines` rather than warning per line.

### Signature format

| Op type | Signature tuple | Example display |
|---------|----------------|-----------------|
| Collective (with algo/proto) | `(op, msg_size, algo, proto)` | `AllReduce@16GB RING_LL` |
| Collective (legacy logs) | `(op, msg_size)` | `AllReduce@16GB` |
| P2P | `(op, msg_size, peer)` | `Send@512MB peer=3` |

## Building the Plugin

The inspector plugin requires CUDA headers and `libcudart`. Build on a Linux
GPU node:

```bash
cd /path/to/nccl
make -C plugins/profiler/inspector

# If CUDA is not at /usr/local/cuda:
make -C plugins/profiler/inspector CUDA_HOME=/path/to/cuda
```

macOS workstations without a CUDA toolkit cannot build the `.so` locally. Edit
on Mac, build on a GPU node.

See also `plugins/profiler/inspector-internal/utils/build-extract-lib.sh` for
containerized NCCL builds.

## JSON Mode vs Prometheus

| Question | JSON + extractor | Prometheus + Grafana |
|----------|------------------|----------------------|
| What ops ran and in what order? | Yes | No (aggregated snapshots) |
| Per-op algo/proto? | Yes (with this change) | Yes (bucketed `algo_proto` label) |
| Live fleet monitoring? | No | Yes |
| Long-running job file size | Grows with op count | Bounded |
| Post-mortem / one-off debug? | Yes | Harder (lossy aggregates) |

Use JSON mode for workload characterization and pattern mining. Use Prometheus
for production dashboards and alerting.

## Limitations

1. **Per-op, not framework-level** — you see `AllReduce`, not "gradient sync
   step". Infer higher-level phases from sequences.
2. **Ordering approximation** — without verbose mode, cross-collective ordering
   relies on dump timestamps, which reflect dump-thread timing rather than op
   completion time.
3. **Repeating-pattern detection** — the extractor uses a consecutive repeat scan
   (min_len=2, max_len=8). While optimized to fast-forward through highly 
   repetitive sequences ($O(N)$ time), irregular or extremely long patterns 
   outside this window will not be detected and may require custom analysis.
4. **Multi-rank merge** — the tool groups by `comm_id:comm_name:rank`. For
   SPMD workloads, compare sequences across ranks manually or extend the tool
   to diff them.

## Related Documentation

- [Inspector README](../README.md) — plugin build, env vars, output formats
- [Performance Exporter](../exporter/example/README.md) — bandwidth analysis and visualizations
- [Grafana Template](../grafana/README.md) — Prometheus dashboard
- [Elastic/Kibana Integration](../../inspector-internal/elastic/README.md) — internal fleet upload wrapper
