# NCCL Inspector Plugin

The NCCL Inspector is a plugin for the NVIDIA Collective Communications Library (NCCL) that provides detailed, per-communicator performance and metadata logging for collective, P2P, and optionally ProxyOp/ProxyStep events. It is designed to help users analyze and debug NCCL communication by generating structured JSON output for each operation.

## Related Documentation

- **[Performance Exporter](exporter/example/README.md)** - Tool for analyzing and visualizing NCCL performance data from inspector logs
- **[Grafana Dashboard Template](grafana/README.md)** - Grafana dashboard for visualizing NCCL Inspector job performance metrics via Prometheus

## Folder Location

The Inspector plugin source is located in:

```
plugins/profiler/inspector/
```

## Building the Inspector Plugin

To build the Inspector plugin, run:

```bash
make
```

The build system will automatically detect CUDA and NCCL installations from your environment. If you need to specify custom paths, you can set `CUDA_HOME` and `NCCL_HOME` environment variables or pass them as make arguments.

### Build Options

The Makefile supports several build options:

- **DEBUG=1**: Enable debug build with additional debugging information
- **ASAN=1**: Enable Address Sanitizer for memory error detection
- **UBSAN=1**: Enable Undefined Behavior Sanitizer

Example debug build:
```bash
make DEBUG=1
```

### Build Output

The build process creates:
- `libnccl-profiler-inspector.so`: The main inspector plugin library
- `version.cc`: Auto-generated version information from git

## Using NCCL Inspector

To obtain output, explicitly set `NCCL_INSPECTOR_DUMP_THREAD_INTERVAL_MICROSECONDS`
to a non-negative value (for example, `500` for periodic dumping) and leave
`NCCL_INSPECTOR_DUMP_THREAD_ENABLE=1` (the default). The interval defaults to `-1`,
which disables the internal dump thread entirely: no output is written, including
at communicator teardown or finalization.

### Key Differences from Normal NCCL Usage

The main difference between running NCCL with the Inspector plugin versus running NCCL normally is the addition of environment variables that enable detailed performance logging:

**Normal NCCL Run:**
```bash
# Standard NCCL execution
./your_nccl_application
```

**NCCL Inspector Run:**
```bash
# NCCL Inspector enabled execution
export NCCL_PROFILER_PLUGIN=/path/to/nccl/plugins/profiler/inspector/libnccl-profiler-inspector.so
export NCCL_INSPECTOR_ENABLE=1
export NCCL_INSPECTOR_DUMP_THREAD_INTERVAL_MICROSECONDS=500
./your_nccl_application
```

### Required Environment Variables

- `NCCL_PROFILER_PLUGIN=/path/to/nccl/plugins/profiler/inspector/libnccl-profiler-inspector.so`
  Loads the Inspector plugin into NCCL.
- `NCCL_INSPECTOR_ENABLE=1`
  Enables the Inspector plugin.

### Optional Environment Variables

- `NCCL_INSPECTOR_ENABLE_P2P=<0|1>` (default: `1`)
  Enables or disables P2P tracking.
- `NCCL_INSPECTOR_ENABLE_PROXY=<0|1>` (default: `0`)
  Enables or disables bounded ProxyOp/ProxyStep trace collection. Proxy traces are currently supported only by the JSON output backend. If Prometheus or OTLP output is selected, Proxy tracking remains inactive and Inspector logs a message. This is a dedicated opt-in and does not require `NCCL_INSPECTOR_DUMP_VERBOSE=1`.
- `NCCL_INSPECTOR_DUMP_THREAD_ENABLE=<0|1>` (default: `1`)
  Enables or disables the internal dump thread.
- `NCCL_INSPECTOR_DUMP_THREAD_INTERVAL_MICROSECONDS=<interval>` (default: `-1`)
  Sets the interval (in microseconds) for the internal dump thread to write output. A value of `-1` (default) disables the internal dump thread entirely, so no output is written; set a non-negative value to obtain output. A value of `0` enables continuous dumping (dumps as fast as possible). Set to a positive value to enable periodic dumps at the specified interval (e.g., `500` for every 500 µs). When Prometheus mode is enabled (`NCCL_INSPECTOR_PROM_DUMP=1`), non-negative intervals are raised to a minimum of `30000000` (30 seconds) to align with the node exporter polling interval; `-1` still disables dumping.
- `NCCL_INSPECTOR_DUMP_DIR=<output_dir>`
  Sets the output directory for logs. If not set, defaults to `nccl-inspector-unknown-jobid` or `nccl-inspector-<slurm_job_id>` if running under SLURM.
- `NCCL_INSPECTOR_DUMP_VERBOSE=<0|1>` (default: `0`)
  Enables verbose output including event trace information.
- `NCCL_INSPECTOR_PROM_DUMP=<0|1>` (default: `0`)
  Enables Prometheus format for textfile node exporter output instead of custom JSON.
- `NCCL_INSPECTOR_PROM_DUMP_STATS=<0|1>` (default: `0`)
  In Prometheus mode, additionally emit the per-device stats metrics (`nccl_collectives_total`, `nccl_collectives_dropped_total`, `nccl_p2p_total`, `nccl_p2p_dropped_total`). Off by default to keep the default Prometheus output minimal; enable it to plumb drop/rate signals to a dashboard. Has no effect on JSON mode (whose per-dump `dump_stats` record is always emitted) or OTLP mode (whose equivalent stats ride under `NCCL_INSPECTOR_OTEL_VERBOSE`).
- `NCCL_INSPECTOR_OTEL_EXPORT=<0|1>` (default: `0`)
  Enables OTLP HTTP metrics export. Accepted values are `0` (disabled) and `1` (enabled). Default OTLP export emits aggregated bucket metrics.
- `NCCL_INSPECTOR_OTEL_VERBOSE=<0|1>` (default: `0`)
  Enables high-cardinality per-operation OTLP data points for temporary investigations. Accepted values are `0` (aggregated) and `1` (per operation).
- `OTEL_EXPORTER_OTLP_METRICS_ENDPOINT=<http-url>` (default: `http://localhost:4318/v1/metrics`)
  Sets the OTLP HTTP metrics endpoint. Only `http://` endpoints are supported (`https://` is not). This does not need to be set when the collector uses the default OTLP HTTP metrics endpoint (`http://localhost:4318/v1/metrics`). Set it only when the collector listens on a different host, port, or path. `OTEL_EXPORTER_OTLP_ENDPOINT` is used as a fallback. If the endpoint has no path, `/v1/metrics` is appended. An unsupported endpoint disables OTLP export at init (no drain/export).
- `OTEL_EXPORTER_OTLP_METRICS_TIMEOUT=<milliseconds>` (default: `2000`)
  Sets the OTLP HTTP export timeout. `OTEL_EXPORTER_OTLP_TIMEOUT` is used as a fallback. Values must be positive milliseconds.
- `OTEL_EXPORTER_OTLP_METRICS_HEADERS=<key=value,...>`
  Optional comma-separated OTLP HTTP headers. `OTEL_EXPORTER_OTLP_HEADERS` is used as a fallback.
  Example: `export OTEL_EXPORTER_OTLP_METRICS_HEADERS='Authorization=Bearer <token>,X-Scope-OrgID=nccl'`
- `OTEL_SERVICE_NAME=<name>` (default: `nccl-inspector`)
  Sets the OTLP `service.name` resource attribute.
- `OTEL_RESOURCE_ATTRIBUTES=<key=value,...>`
  Adds OTLP resource attributes. Explicit `OTEL_SERVICE_NAME` overrides `service.name` from this list.
  Example: `export OTEL_RESOURCE_ATTRIBUTES='deployment.environment=test,cluster=example-gpu-cluster,team=example-team'`
- `NCCL_INSPECTOR_DUMP_MIN_SIZE_BYTES=<bytes>` (default: `8192`)
  Minimum message size (bytes) to be tracked by inspector.
- `NCCL_INSPECTOR_DUMP_COLL_RING_SIZE=<entries>` (default: `1024`)
  Per-communicator completed-collective ring buffer capacity.
- `NCCL_INSPECTOR_DUMP_P2P_RING_SIZE=<entries>` (default: `1024`)
  Per-communicator completed-P2P ring buffer capacity.

  If operations complete faster than the dump thread drains the ring, the oldest
  entries are overwritten before they can be dumped. When this happens the
  Inspector logs a one-time warning and reports drop counts in the output
  (`dropped_total` / `dropped_since_last_dump` in the JSON `dump_stats` record;
  `nccl_collectives_dropped_total` / `nccl_p2p_dropped_total` in Prometheus/OTLP
  stats). Increasing the ring size retains more entries under bursts.
- `NCCL_INSPECTOR_DUMP_PROXY_RING_SIZE=<entries>` (default: `1024`)
  Per-communicator Proxy ring capacity in small records: each Step, Op start, segment metadata and final summary takes one slot. Full rings overwrite the oldest record; `dump_stats` reports record and Step-detail loss. Nested batches enqueue Steps before their parent metadata under one lock, so a retained Step always has its parent metadata in the same drain. See the memory budget below.
- `NCCL_INSPECTOR_PROXY_OUTPUT_MODE=<nested|flat>` (default: `nested`)
  `nested` emits an Op start, arrays of completed Steps, and a final Op summary. The per-Op buffer grows lazily (1/2/4/8 snapshots); eight completed Steps trigger a flush, while an unfinished tail waits for Op finalization. A dump merges this batch's segments by Op, so a JSON array can exceed eight Steps. There is no cross-dump buffering, timeout or periodic tail flush. `flat` retains independent completed Step/Op records, useful for streaming diagnostics. Unknown values warn and fall back to `nested`. This does not enable Proxy tracking or bypass its JSON-only gate.
- `NCCL_INSPECTOR_DUMP_PROXY_STEPS=<0|1>` (default: `1`)
  Set to `0` to suppress Step details. Nested mode still emits the Op start and final summary, but no Step arrays or intermediate segments; flat mode emits only the completed Op summary. ProxyStep callbacks, pools, byte/count aggregation, and parent references remain active. No Step buffer is allocated in Op-only or flat mode. This option requires `NCCL_INSPECTOR_ENABLE_PROXY=1` and does not change the JSON-only gate.
- `NCCL_INSPECTOR_DUMP_PROXY_EVENT_TRACE_SN=<0|1>` (default: `0`)
  Set to `1` to include the diagnostic `event_trace_sn` block in ProxyOp and ProxyStep JSON records. Internal sequence tracking, timestamps, durations, bandwidth, correlation IDs, and drop accounting remain unchanged when this block is omitted. This option is independent of `NCCL_INSPECTOR_DUMP_VERBOSE` and does not affect Collective/P2P trace output. In Op-only mode it controls the Op block only; it does not re-enable Step records.
- `NCCL_INSPECTOR_COLL_POOL_SIZE=<entries>` (default: `256`)
  Collective pool initial size/stride.
- `NCCL_INSPECTOR_P2P_POOL_SIZE=<entries>` (default: `256`)
  P2P pool initial size/stride.
- `NCCL_INSPECTOR_COMM_POOL_SIZE=<entries>` (default: `256`)
  Comm pool initial size/stride.
- `NCCL_INSPECTOR_PROXY_OP_POOL_SIZE=<entries>` (default: `1024`)
  Fixed capacity of the active ProxyOp pool when Proxy tracking is enabled.
- `NCCL_INSPECTOR_PROXY_STEP_POOL_SIZE=<entries>` (default: `4096`)
  Fixed capacity of the active ProxyStep pool when Proxy tracking is enabled.
- `NCCL_INSPECTOR_POOL_GROW=<0|1>` (default: `1`)
  Enables or disables dynamic growth of the collective, P2P, and Comm pools. Proxy pools are always fixed-capacity and are not affected by this setting.
- `NCCL_INSPECTOR_REQUIRE_KERNEL_TIMING=<0|1>` (default: `1`)
  When enabled (default), only events with GPU-based kernel timing (`kernel_gpu`) are recorded. Events that fall back to CPU-measured timing (`kernel_cpu` or `collective_cpu`) are silently discarded. Set to `0` to restore the previous fallback behaviour and retain all events regardless of timing source.

### Debugging

To see detailed Inspector plugin messages, use NCCL's debug subsystem filtering. The Inspector uses the `PROFILE` subsystem:

```bash
# Show only Inspector messages
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=PROFILE

# Show Inspector messages along with other subsystems
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,PROFILE

# Show all debug messages (including Inspector)
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL
```

Inspector messages will appear with your configured NCCL_DEBUG level and will show:
- Plugin initialization and configuration
- Dump thread status and intervals
- File creation and locations (with device UUIDs for Prometheus mode)
- Error conditions and warnings

### Example Usage

**Single Node:**
```bash
export NCCL_PROFILER_PLUGIN=/path/to/nccl/plugins/profiler/inspector/libnccl-profiler-inspector.so
export NCCL_INSPECTOR_ENABLE=1
export NCCL_INSPECTOR_DUMP_THREAD_INTERVAL_MICROSECONDS=500
./build/test/perf/all_reduce_perf -b 8 -e 16G -f 2 -g 8
```

**Multi-Node (SLURM):**
```bash
# Add these environment variables to your SLURM script
export NCCL_PROFILER_PLUGIN=/path/to/nccl/plugins/profiler/inspector/libnccl-profiler-inspector.so
export NCCL_INSPECTOR_ENABLE=1
export NCCL_INSPECTOR_DUMP_THREAD_INTERVAL_MICROSECONDS=500
export NCCL_INSPECTOR_DUMP_DIR=/path/to/logs/${SLURM_JOB_ID}/

# Then run your normal NCCL application
srun your_nccl_application
```

**Prometheus Output Mode (for node exporter)**

**Example Prometheus Setup:**
```bash
export NCCL_PROFILER_PLUGIN=/path/to/nccl/plugins/profiler/inspector/libnccl-profiler-inspector.so
export NCCL_INSPECTOR_ENABLE=1
export NCCL_INSPECTOR_PROM_DUMP=1
export NCCL_INSPECTOR_DUMP_THREAD_INTERVAL_MICROSECONDS=30000000  # 30 seconds
export NCCL_INSPECTOR_DUMP_DIR=/var/lib/node_exporter/nccl_inspector/
```

Note: Prometheus mode enforces a minimum dump interval of 30 seconds (30,000,000 microseconds) to align with the node exporter polling interval.

**Exported Metrics:**
- `nccl_bus_bandwidth_gbs` - NCCL bus bandwidth in GB/s (collectives)
- `nccl_collective_exec_time_microseconds` - Execution time in microseconds (collectives)
- `nccl_p2p_bus_bandwidth_gbs` - NCCL P2P bus bandwidth in GB/s
- `nccl_p2p_exec_time_microseconds` - P2P execution time in microseconds

When P2P tracking is enabled (`NCCL_INSPECTOR_ENABLE_P2P=1`), Prometheus output includes P2P metrics with a `p2p_operation` label (e.g., `Send`, `Recv`).

**Opt-in per-device stats metrics** (emitted only when `NCCL_INSPECTOR_PROM_DUMP_STATS=1`):
- `nccl_collectives_total` - Cumulative collectives enqueued into the ring buffer, per device (counter)
- `nccl_collectives_dropped_total` - Cumulative collective records overwritten before being dumped, per device (counter)
- `nccl_p2p_total` - Cumulative P2P operations enqueued into the ring buffer, per device (counter)
- `nccl_p2p_dropped_total` - Cumulative P2P records overwritten before being dumped, per device (counter)

These are cumulative counters (not per-window gauges), so `rate(nccl_collectives_dropped_total[$__rate_interval]) / rate(nccl_collectives_total[$__rate_interval])` gives the fraction of collectives being lost, robust to scrape/dump interval alignment. They are emitted for every known device each dump so the counters stay continuous.

**Labels:**
- Collectives: `version`, `slurm_job_id`, `node`, `gpu`, `comm_name`, `n_nodes`, `nranks`, `collective`, `message_size`, `algo_proto`
- P2P: `version`, `slurm_job_id`, `node`, `gpu`, `comm_name`, `n_nodes`, `nranks`, `p2p_operation`, `message_size`
- Per-device stats metrics: `version`, `slurm_job_id`, `node`, `gpu`

`message_size` is a bucketed range string (for example `4-5GB`).

**OTLP HTTP Output Mode**

By default, the Inspector sends **aggregated OTLP bucket metrics** from the same aggregation state used by Prometheus textfile mode. This keeps the default OTLP footprint suitable for fleet dashboards and alerting, with low-cardinality series per node.

The Inspector emits OTLP over HTTP using JSON encoding; this is not user-configurable.

Set `NCCL_INSPECTOR_OTEL_VERBOSE=1` only for temporary investigations that need per-completed-collective and per-P2P OTLP data. Verbose mode emits exact operation attributes such as `coll_sn`, `p2p_sn`, exact message size, and peer, and can increase series cardinality by roughly 100x for the duration of the investigation.

OTLP export speaks plaintext HTTP only. Point `OTEL_EXPORTER_OTLP_METRICS_ENDPOINT` at a local collector `http://` receiver (for example `http://127.0.0.1:44318/v1/metrics`). `https://` endpoints are rejected at init and OTLP export is disabled.

For a GPU-node collector with:

```yaml
receivers:
  otlp:
    protocols:
      http:
        endpoint: 127.0.0.1:44318
```

use:

```bash
export NCCL_PROFILER_PLUGIN=/path/to/nccl/plugins/profiler/inspector/libnccl-profiler-inspector.so
export NCCL_INSPECTOR_ENABLE=1
export NCCL_INSPECTOR_OTEL_EXPORT=1

# Optional when the collector listens on the default http://localhost:4318/v1/metrics.
# Required for this example because the collector listens on port 44318.
export OTEL_EXPORTER_OTLP_METRICS_ENDPOINT=http://127.0.0.1:44318/v1/metrics

# Optional: service.name defaults to nccl-inspector.
# export OTEL_SERVICE_NAME=nccl-inspector

# Optional: add deployment-specific resource attributes when the collector or
# backend does not already enrich metrics with this metadata.
# export OTEL_RESOURCE_ATTRIBUTES='deployment.environment=test,cluster=example-gpu-cluster'

# Optional: headers for collectors that require auth or tenant routing
# export OTEL_EXPORTER_OTLP_METRICS_HEADERS='Authorization=Bearer <token>,X-Scope-OrgID=nccl'
export NCCL_INSPECTOR_DUMP_THREAD_INTERVAL_MICROSECONDS=30000000

# Optional: enable high-cardinality per-operation OTLP temporarily
# export NCCL_INSPECTOR_OTEL_VERBOSE=1
```

Recommended default OTLP settings for fleet dashboards are `NCCL_INSPECTOR_OTEL_VERBOSE=0` and `NCCL_INSPECTOR_DUMP_THREAD_INTERVAL_MICROSECONDS=30000000` or larger. Verbose mode should be enabled only for a short investigation window because it emits per-operation series.

`NCCL_INSPECTOR_PROM_DUMP=1` is not required for OTLP export. If both `NCCL_INSPECTOR_OTEL_EXPORT=1` and `NCCL_INSPECTOR_PROM_DUMP=1` are set, OTLP export takes precedence and Prometheus textfile output is skipped.

**OTLP resource attributes (constant per process, sent once per export):**

- `service.name` defaults to `nccl-inspector`, or `OTEL_SERVICE_NAME` if set.
- `slurm.job.id` is filled only if `SLURM_JOB_ID` is set in the NCCL process environment.
- Anything passed via `OTEL_RESOURCE_ATTRIBUTES`. This is optional and is intended for deployment-specific metadata such as environment, cluster, namespace, or tenant when that metadata is not already added by the collector/backend.

**Default OTLP data-point attributes (aggregated):**

- Collectives: `version` (`v5.1`), `node`, `collective`, `message_size`, `algo_proto`
- P2P: `version` (`v5.1`), `node`, `p2p_operation`, `message_size`
- Per-(comm, device) common attributes: `gpu` (for example `GPU0`), `comm_name`, `n_nodes`, `nranks`

**Default OTLP metrics (aggregated):**

- `nccl_bus_bandwidth_gbs`
- `nccl_collective_exec_time_microseconds`
- `nccl_p2p_bus_bandwidth_gbs`
- `nccl_p2p_exec_time_microseconds`

**Verbose OTLP collective data-point attributes (per operation):**

- `version` (`v5.2`), `node`, `collective`, `coll_sn`, `coll_msg_size_bytes`, `algo_proto`
- Per-(comm, device) common attributes: `gpu`, `comm_name`, `comm_id`, `n_nodes`, `nranks`

**Verbose OTLP collective metrics (per operation):**

- `nccl_bus_bandwidth_gbs`
- `nccl_collective_exec_time_microseconds`
- `nccl_collective_algobw_gbs`

**Verbose OTLP P2P attributes/metrics** follow the same per-operation pattern (`version` (`v5.2`), `node`, `p2p_sn`, `p2p_peer`, exact message size, etc.) when P2P tracking is enabled.

OTLP data points include `timeUnixNano`. Verbose mode uses the completed operation timestamp for each per-operation point. Default aggregated mode uses the latest completed operation timestamp in each bucket, so timestamp granularity improves without adding labels or increasing series cardinality.

**Current Metric Format Examples (Prometheus aggregated mode):**
```
nccl_bus_bandwidth_gbs{version="v5.1",slurm_job_id="unknown",node="nvl72004-T01",gpu="GPU0",comm_name="DP Group 0",n_nodes="1",nranks="4",collective="AllReduce",message_size="4-5GB",algo_proto="Ring_ll"} 678.263
nccl_collective_exec_time_microseconds{version="v5.1",slurm_job_id="unknown",node="nvl72004-T01",gpu="GPU0",comm_name="DP Group 0",n_nodes="1",nranks="4",collective="AllReduce",message_size="4-5GB",algo_proto="Ring_ll"} 9498.47
nccl_p2p_bus_bandwidth_gbs{version="v5.1",slurm_job_id="unknown",node="nvl72004-T01",gpu="GPU0",comm_name="DP Group 0",n_nodes="1",nranks="4",p2p_operation="Send",message_size="512-513MB"} 464.9
nccl_p2p_exec_time_microseconds{version="v5.1",slurm_job_id="unknown",node="nvl72004-T01",gpu="GPU0",comm_name="DP Group 0",n_nodes="1",nranks="4",p2p_operation="Send",message_size="512-513MB"} 1154.87
```

## Output Example

JSON v4.4 is a stream of newline-separated objects. Each communicator dump starts
with a `dump_stats` marker carrying `header` and `metadata` once. The following
`coll_records + p2p_records + proxy_records` objects belong to that marker (an
absent `proxy_records` means zero). The writer holds its output lock across the
whole batch, so communicator batches cannot interleave. Consumers must preserve
the marker context when filtering, splitting, or merging these files; individual
operation lines are no longer self-contained. This applies even with Proxy
tracking disabled. A zero-record marker can still report new losses or PXN skips.
The bundled analysis scripts use `inspector_json_reader.py` to restore this
context while accepting older per-record formats. Keep that helper alongside
the Inspector directory when copying the tools elsewhere.

v4.4 introduces segmented Proxy output; the bundled reader accepts both v4.3
and v4.4 marker-based streams, as well as v4.0-v4.2 per-record context. Nested
Steps remain inside their parent record: they do not consume marker counts.

For example, a collective record following its marker looks like:

```json
{
  "coll_perf": {
    "coll": "AllReduce",
    "coll_algo": "RING",
    "coll_proto": "LL",
    "coll_sn": 1407,
    "coll_msg_size_bytes": 17179869184,
    "coll_exec_time_us": 61974,
    "coll_algobw_gbs": 277.210914,
    "coll_busbw_gbs": 485.119099
  }
}
```

### Per-Dump Stats Record

Each non-idle communicator emits one `dump_stats` marker at the start of its dump.
It reports records written and records overwritten in each completed ring.
`*_dropped_total` is cumulative for the communicator; `*_dropped_since_last_dump`
covers only the interval since its previous dump. Stats are sampled even if the
rings are empty. With Proxy tracking enabled, it also reports Ops lost before
record construction, the Step-output setting (`proxy_steps_enabled`, integer 0/1), and process-wide PXN skips.

Because this record has no `coll_perf`/`p2p_perf` body, consumers that iterate per-collective should filter on the presence of `coll_perf`/`p2p_perf` and read `dump_stats` separately rather than assuming every record is an operation.

```json
{
  "header": {
    "id": "0x7f8c496ae9f661",
    "rank": 2,
    "n_ranks": 8,
    "nnodes": 1
  },
  "metadata": {
    "inspector_output_format_version": "v4.4",
    "git_rev": "",
    "rec_mechanism": "nccl_profiler_interface",
    "dump_timestamp_us": 1748030377748202,
    "hostname": "example-hostname",
    "pid": 1639453
  },
  "dump_stats": {
    "coll_records": 1024,
    "coll_dropped_total": 4096,
    "coll_dropped_since_last_dump": 512,
    "p2p_records": 0,
    "p2p_dropped_total": 0,
    "p2p_dropped_since_last_dump": 0,
    "proxy_records": 2,
    "proxy_dropped_total": 0,
    "proxy_dropped_since_last_dump": 0,
    "proxy_steps_overwritten_total": 0,
    "proxy_steps_overwritten_since_last_dump": 0,
    "proxy_output_mode": "nested",
    "proxy_segment_step_capacity": 8,
    "proxy_ops_dropped_total": 3,
    "proxy_ops_dropped_since_last_dump": 1,
    "proxy_pxn_skipped_process_total": 5,
    "proxy_steps_enabled": 1
  }
}
```

## Output Example Verbose

To enable verbose output with event trace information, set the `NCCL_INSPECTOR_DUMP_VERBOSE=1` environment variable:

```bash
export NCCL_INSPECTOR_DUMP_VERBOSE=1
```

This will include additional event trace information in the JSON output, showing the sequence of callbacks and timestamps for each individual event.

```json
{
  "coll_perf": {
    "coll": "ReduceScatter",
    "coll_algo": "RING",
    "coll_proto": "SIMPLE",
    "coll_sn": 1231,
    "coll_msg_size_bytes": 2147483648,
    "coll_exec_time_us": 41057,
    "coll_timing_source": "kernel_gpu",
    "coll_algobw_gbs": 418.439467,
    "coll_busbw_gbs": 366.134533,
    "event_trace_sn": {
      "coll_start_sn": 1,
      "coll_stop_sn": 2,
      "kernel_events": [
        {
          "channel_id": 0,
          "kernel_start_sn": 3,
          "kernel_stop_sn": 48,
          "kernel_record_sn": 47
        }
      ]
    },
    "event_trace_ts": {
      "coll_start_ts": 1752867229235059,
      "coll_stop_ts": 1752867229235064,
      "kernel_events": [
        {
          "channel_id": 0,
          "kernel_start_ts": 1752867229235181,
          "kernel_stop_ts": 1752867229275811,
          "kernel_record_ts": 1752867229275811
        }
      ]
    }
  }
}
```

Multiple such JSON objects are written, one per completed collective or P2P operation per communicator.

## Proxy Trace Output (JSON Only)

Proxy tracing is disabled by default because a single collective may produce many ProxyOp and ProxyStep events. Enable it explicitly when detailed network-progress timing is needed:

```bash
export NCCL_INSPECTOR_ENABLE_PROXY=1
```

Proxy traces are currently emitted only in JSON mode. If `NCCL_INSPECTOR_PROM_DUMP=1` or `NCCL_INSPECTOR_OTEL_EXPORT=1` selects another output backend, Inspector does not activate the Proxy callbacks or allocate Proxy storage. `NCCL_INSPECTOR_ENABLE_PROXY` is independent of `NCCL_INSPECTOR_DUMP_VERBOSE`; the Proxy records always include their own event trace.

Proxy records use the top-level `proxy_trace` field. In the default nested mode,
Steps are grouped under their Op within each dump, not under `coll_perf` or
`p2p_perf`. Proxy work can finish asynchronously after its parent operation.
With `NCCL_INSPECTOR_PROXY_OUTPUT_MODE=flat`, each completed Step and Op remains
a separate record. Use the following fields to correlate records:

- `parent_type` and `parent_sn` identify the parent collective or P2P operation.
- `proxy_op_sn` identifies a ProxyOp within the communicator.
- `proxy_step_sn` identifies a step within its ProxyOp.
- `record_sn` orders ring entries within the communicator. Nested output uses the selected parent record's sequence; merged Steps and segments also consume sequences, so gaps alone do not indicate loss. Use `dump_stats` for overwrite counts.
- `channel_id`, `peer`, and `direction` describe the connection. The rank is in the dump marker's `header.rank`; Step payloads do not duplicate it. Op payloads retain `rank`. The local process PID is in the dump marker's `metadata.pid`; Proxy records do not duplicate it as `origin_pid`.

To associate a Step with an Op, use the restored dump context (hostname, PID,
communicator ID, and `header.rank`) together with `proxy_op_sn`. Do not use
`proxy_op_sn` alone across ranks or communicators. Readers of Step payloads
must read rank from the dump header, not from `proxy_trace.rank`; the header
is available even when the Op summary has not been emitted or was dropped.

With `NCCL_INSPECTOR_DUMP_PROXY_EVENT_TRACE_SN=1`, a completed send step can
produce a `proxy_trace` object like this in **flat mode** (the
`event_trace_sn` block is omitted by default):

```json
{
  "record_type": "proxy_step",
  "record_sn": 41,
  "parent_type": "coll",
  "parent_sn": 1407,
  "proxy_op_sn": 12,
  "channel_id": 0,
  "peer": 3,
  "direction": "send",
  "proxy_step_sn": 1,
  "step": 0,
  "trans_size_bytes": 1048576,
  "timing_source": "proxy_cpu",
  "total_us": 40,
  "gpu_produce_us": 10,
  "peer_credit_us": 10,
  "wire_us": 10,
  "event_trace_sn": {
    "proxy_step_start_sn": 3,
    "send_gpu_wait_sn": 4,
    "send_peer_wait_sn": 5,
    "send_wait_sn": 6,
    "proxy_step_stop_sn": 7
  },
  "event_trace_ts": {
    "proxy_step_start_ts": 1752867229235100,
    "send_gpu_wait_ts": 1752867229235110,
    "send_peer_wait_ts": 1752867229235120,
    "send_wait_ts": 1752867229235130,
    "proxy_step_stop_ts": 1752867229235140
  }
}
```

With the same diagnostic option enabled, the corresponding completed ProxyOp
summary in flat mode can produce:

```json
{
  "record_type": "proxy_op",
  "record_sn": 42,
  "parent_type": "coll",
  "parent_sn": 1407,
  "proxy_op_sn": 12,
  "rank": 2,
  "channel_id": 0,
  "peer": 3,
  "direction": "send",
  "n_steps": 1,
  "chunk_size_bytes": 1048576,
  "n_steps_started": 1,
  "n_steps_completed": 1,
  "n_steps_dropped": 0,
  "trans_size_bytes": 1048576,
  "timing_source": "proxy_cpu",
  "total_us": 70,
  "op_bw_gbs": 14.979657,
  "event_trace_sn": {
    "proxy_op_start_sn": 1,
    "proxy_op_in_progress_sn": 2,
    "proxy_op_stop_sn": 8
  },
  "event_trace_ts": {
    "proxy_op_start_ts": 1752867229235080,
    "proxy_op_in_progress_ts": 1752867229235090,
    "proxy_op_stop_ts": 1752867229235150
  }
}
```

In a ProxyOp record, `n_steps` is the number of network-transfer steps described by
NCCL. `n_steps_started` counts successfully tracked Step starts, and
`n_steps_completed` counts their stops; the two agree when an Op is finalized.
`n_steps_dropped` counts starts rejected by the active Step pool and completed
Steps whose initial snapshot-buffer allocation failed. A completed Step still
contributes to completion/byte accounting even if its detail cannot be saved.
These counters
still operate when `NCCL_INSPECTOR_DUMP_PROXY_STEPS=0`; suppressing output is not a
dropped event.

### Nested Start, Segments and Final Summary

Nested mode buffers completed snapshots per Op, growing lazily through capacities
1/2/4/8. A full buffer is flushed and reused; finalization flushes the tail and
releases the allocation. Active Step objects are recycled immediately after copying.
Flat and Op-only modes allocate no snapshot buffers.

| `record_type` | `is_final` | Contents |
| --- | --- | --- |
| `proxy_op_start` | `false` | Emitted before returning the Op handle: parent/connection metadata, declared Step count, chunk size and start timestamp |
| `proxy_op_segment` | `false` | Parent/connection metadata and retained Steps from this dump; no cumulative Op metrics |
| `proxy_op` | `true` | Original final Op metrics and retained Steps from this dump; emitted after Op stop and release of all tracked children |

Each flush enqueues its Steps **before** its parent metadata under one communicator
lock. All slots are small, fixed-size and pointer-free. FIFO evicts those Steps
before their parent, even if a batch exceeds ring capacity. Dump drains the whole
ring under the same lock, then groups by Op outside the lock. Starts remain separate.

Within a dump, Step arrays are concatenated and final metrics are taken only from
the final record, never added across segments. An output array can exceed eight
Steps. Across dumps, segments are incremental; nothing is cached or repeated.
A final array can be empty if earlier dumps consumed all Steps. Op-only omits
`steps` entirely but still emits start/final records and final statistics.

Nested Steps contain only their own sequence, index, observed bytes, timing and
optional diagnostic sequence numbers. Direction/connection fields come from the
parent; rank comes from the dump header. No `segment_sn` is added.

Abbreviated output (connection fields omitted for readability):

```json
{"proxy_trace":{"record_type":"proxy_op_start","proxy_op_sn":12,"is_final":false,"event_trace_ts":{"proxy_op_start_ts":100}}}
{"proxy_trace":{"record_type":"proxy_op","proxy_op_sn":12,"is_final":true,"trans_size_bytes":1048576,"total_us":70,"op_bw_gbs":14.979657,"steps":[{"proxy_step_sn":1,"step":0,"trans_size_bytes":1048576,"total_us":40,"gpu_produce_us":10,"peer_credit_us":10,"wire_us":10}]}}
```

**Loss and hangs:** `is_final` is a boolean indicating an observed final Op, not
complete Step retention. A start without a final record does not prove a hang:
records can be overwritten or the log truncated. There is no timer, active-Op
scan or abort flush. A stalled tail of at most seven completed Steps remains
buffered until finalization; use flat mode to stream each completed Step.
Neither mode emits unfinished Step state. Ring output still requires normal dumping.

**Allocation failure:** failed growth flushes existing snapshots and reuses the
old buffer. A failed initial allocation drops only the current Step detail and
increments `n_steps_dropped`; Op completion/byte accounting remains intact.

**Memory:** measured x86-64 sizes are 208 B per ring record, 232 B per active Op
and 152 B per snapshot. A buffer uses at most 1216 B per Op (1.1875 MiB for 1024
full-capacity buffers), excluding allocator overhead. The default ring plus
sentinel uses 208.2 KiB per comm; a full drain needs 208 KiB of record scratch per
dump thread plus grouping storage. An eight-Step batch uses nine small slots.
These are allocation-size estimates, not RSS measurements.

`trans_size_bytes` is bytes **observed**, not necessarily all bytes transferred.
It sums the first transfer-size callback for each tracked Step (`send_wait` for
sends, `recv_flush_wait` for receives). Treat it as a complete byte total only
when all Step starts were tracked and the transfer-size callbacks were received.
A snapshot-buffer failure loses detail but does not reduce this total. A Step
without a transfer-size callback still counts as completed, but contributes no
bytes. Do not reconstruct missing bytes from `n_steps * chunk_size_bytes`:
`chunk_size_bytes` is a maximum slice size, not the realized transfer size.

### Proxy Timing and Observed Bandwidth

Proxy records include `timing_source: "proxy_cpu"`. The existing
`event_trace_ts` block is retained. The diagnostic `event_trace_sn` block is
opt-in via `NCCL_INSPECTOR_DUMP_PROXY_EVENT_TRACE_SN=1`; internal sequence
numbers are always tracked, including when this option is unset or `0`.
Elapsed-time fields are computed when completed records are written, not in
the callbacks, and are independent of this output option.
All timestamps and durations are in microseconds and describe CPU Proxy
callbacks, not GPU execution or hardware NIC timestamps. Do not compare these
durations or `op_bw_gbs` directly with GPU-derived collective timing or
`coll_busbw_gbs`.

Each `*_wait_ts` marks **entry** into a phase, rather than its duration. The
orders are:

```text
send: send_gpu_wait -> send_peer_wait -> send_wait -> stop
recv: recv_wait -> recv_flush_wait -> recv_gpu_wait -> stop
```

| Record | Field | Calculation | Meaning |
| --- | --- | --- | --- |
| Op | `total_us` | `proxy_op_stop_ts - proxy_op_start_ts` | Elapsed Op lifetime |
| Step | `total_us` | `proxy_step_stop_ts - proxy_step_start_ts` | Elapsed Step lifetime |
| Send Step | `gpu_produce_us` | `send_peer_wait_ts - send_gpu_wait_ts` | Wait for GPU-produced send data |
| Send Step | `peer_credit_us` | `send_wait_ts - send_peer_wait_ts` | Wait for receiver buffer credit / send request acceptance |
| Send Step | `wire_us` | `proxy_step_stop_ts - send_wait_ts` | Wait for the submitted network send to complete |
| Receive Step | `wire_us` | `recv_flush_wait_ts - recv_wait_ts` | Wait for network receive completion |
| Receive Step | `flush_us` | `recv_gpu_wait_ts - recv_flush_wait_ts` | Wait for received data to become visible to the GPU |
| Receive Step | `gpu_consume_us` | `proxy_step_stop_ts - recv_gpu_wait_ts` | Wait for GPU consumption and buffer-slot release |

The total also includes the interval between Step start and its first wait
phase, so the listed phase durations need not sum to `total_us`.

A nonzero event sequence number (`sn`) identifies an observed callback. An
unfired state's timestamp key is **omitted**, rather than emitted as a zero
placeholder; when diagnostic sequence output is enabled, its sequence-number
entry remains `0`. A duration
is emitted only when both endpoints were observed and the ending timestamp is
not earlier than the starting timestamp. Missing or reversed intervals are
omitted, not replaced with `0`. Equal valid timestamps do produce a duration
of `0`, which is possible at microsecond resolution. Consumers must accept
missing timestamp and duration keys.

An Op with a valid, positive `total_us` also includes:

```text
op_bw_gbs = trans_size_bytes / (total_us * 1000)
```

This is decimal GB/s over the Op's CPU-observed lifetime. Zero or invalid
elapsed time omits `op_bw_gbs`; zero observed bytes over a positive duration
produces `0`. Lost Steps or missing transfer-size callbacks can underestimate
the byte total and therefore this rate. The field remains available in Op-only
mode (`NCCL_INSPECTOR_DUMP_PROXY_STEPS=0`).

Steps expose `wire_us`, not a per-Step bandwidth field. A consumer may derive
a slice rate from observed bytes and a valid, positive wire duration, but
Steps overlap in a pipeline: their rates must not be summed or interpreted as
achieved NIC throughput. Likewise, overlapping Ops' rates are not additive.

### Bounded Proxy Storage and Dropped Events

Proxy tracking uses fixed-capacity active-event pools and a fixed-capacity completed-record ring so that a high-frequency ProxyStep stream cannot grow Inspector memory without limit:

- If the ProxyOp pool is full, the new ProxyOp and its child steps are not recorded. `proxy_ops_dropped_total` / `proxy_ops_dropped_since_last_dump` in `dump_stats` count these lost Ops, including lock-initialization failures. The reserved `proxy_op_sn` also leaves a gap. These counters count Ops, not the unknown number of child Steps lost with them.
- If the ProxyStep pool is full, the new step is not recorded and its parent ProxyOp increments `n_steps_dropped`.
- If the Proxy ring is full, the oldest small record is overwritten. The ring's counters are emitted once per dump as `proxy_dropped_total` / `proxy_dropped_since_last_dump`. Each Step, start, segment metadata and final summary counts as one ring record. In contrast, `proxy_records` counts top-level JSON payloads after grouping, as required by the dump marker protocol.
- `proxy_steps_overwritten_total` / `proxy_steps_overwritten_since_last_dump` count overwritten Step records in either mode; Op records count as zero. This is a subset of `proxy_dropped_*`, not an additional loss to sum with it. Counting happens inside the ring's existing overwrite detection, not in a second overflow check.
- `proxy_output_mode` and `proxy_segment_step_capacity` in `dump_stats` describe the output mode and the per-Op buffer limit, not the maximum JSON array length. `proxy_steps_enabled` retains its independent meaning.
- `n_steps_dropped` reports active ProxyStep allocation failures and initial snapshot-buffer allocation failures. It does not include records later overwritten in the ring, and is not a count of incomplete transfers. Ring loss does not change the byte/count aggregation already performed for an Op.
- Pool exhaustion and Proxy ring overflow each produce a one-shot message per process, not a log line per lost event. Loss counters continue to advance after messages are suppressed.
- `NCCL_INSPECTOR_POOL_GROW` does not apply to either Proxy pool. Increase `NCCL_INSPECTOR_PROXY_OP_POOL_SIZE`, `NCCL_INSPECTOR_PROXY_STEP_POOL_SIZE`, or `NCCL_INSPECTOR_DUMP_PROXY_RING_SIZE` when a workload needs a larger capture window.

### PXN Limitation

Detached PXN ProxyOps can carry a parent pointer from another process's address space. Inspector checks the origin PID before dereferencing that pointer. ProxyOps whose origin PID differs from the current process are currently skipped and do not produce Proxy JSON records; support for correlating detached PXN work is not included yet.

`proxy_pxn_skipped_process_total` in `dump_stats` makes these skips observable.
It is cumulative for the **local Inspector process**, not for the communicator:
the descriptor's profiler context may also be foreign, so neither it nor the
parent pointer is dereferenced to attribute a skip. The same process total can
appear in several communicator markers; group by `metadata.hostname` and
`metadata.pid` and take the latest/max value, **do not sum it across markers**.
A changed total triggers a stats-only dump for each local communicator even
without local operation records. Skips also trigger one informational message.

## Output Directory

- By default, output directory is auto-generated based on:
  - `nccl-inspector-<jobid>` if `SLURM_JOBID` is set
  - `nccl-inspector-unknown-jobid` otherwise
- You can override this with the `NCCL_INSPECTOR_DUMP_DIR` environment variable.
- For Prometheus integration, set it to a directory where Prometheus exporter can scrape it from (e.g., `NCCL_INSPECTOR_DUMP_DIR=/var/lib/node_exporter/nccl_inspector`).

## Output File Size Estimates

The size of output files depends on the output format and usage patterns:

**JSON Mode** (`NCCL_INSPECTOR_PROM_DUMP=0`, default):
- File size **grows continuously** throughout the application lifetime
- Each collective operation adds a new JSON entry to the log file
- With `NCCL_INSPECTOR_ENABLE_PROXY=1`, nested mode emits Op starts, full Step segments and final summaries; flat mode emits each completed Op/Step separately. Set `NCCL_INSPECTOR_DUMP_PROXY_STEPS=0` to omit Step details while retaining Step-based Op statistics. Common header/metadata are emitted once per communicator dump, rather than on each operation line.
- File size is proportional to:
  - Total number of collective operations executed
  - Number of parallel/overlapping communicators the process (PID) participates in
- Estimate: ~200-500 bytes per collective operation
- Example: A workload with 1M collectives across 4 communicators ≈ 200-500 MB per process

**Prometheus Mode** (`NCCL_INSPECTOR_PROM_DUMP=1`):
- File size is **bounded** (does not grow indefinitely)
- Files are rewritten periodically (default: every 30 seconds based on `NCCL_INSPECTOR_DUMP_THREAD_INTERVAL_MICROSECONDS`)
- File size is proportional to:
  - Number of parallel/overlapping communicators using the same GPU device
- Each file contains only the most recent metrics snapshot
- Estimate: ~500-1000 bytes per communicator per metric
- Example: 8 communicators on one GPU with 3 metrics ≈ 12-24 KB per GPU (fixed size)

## Additional Notes

- The plugin is compatible with standard NCCL workflows and can be used in both single-node and multi-node (SLURM) environments.
- For more details, see the source code and comments in `plugins/profiler/inspector/`.
