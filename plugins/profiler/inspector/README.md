# NCCL Inspector Plugin

The NCCL Inspector is a plugin for the NVIDIA Collective Communications Library (NCCL) that provides detailed, per-communicator, per-collective performance and metadata logging. It is designed to help users analyze and debug NCCL collective operations by generating structured JSON output for each operation.

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
- `NCCL_INSPECTOR_DUMP_THREAD_ENABLE=<0|1>` (default: `1`)
  Enables or disables the internal dump thread.
- `NCCL_INSPECTOR_DUMP_THREAD_INTERVAL_MICROSECONDS=<interval>` (default: `-1`)
  Sets the interval (in microseconds) for the internal dump thread to write output. A value of `-1` (default) disables periodic dumping — output is written only at communicator teardown/finalization. A value of `0` enables continuous dumping (dumps as fast as possible). Set to a positive value to enable periodic dumps at the specified interval (e.g., `500` for every 500 µs). When Prometheus mode is enabled (`NCCL_INSPECTOR_PROM_DUMP=1`), a minimum of `30000000` (30 seconds) is enforced to align with the node exporter polling interval.
- `NCCL_INSPECTOR_DUMP_DIR=<output_dir>`
  Sets the output directory for logs. If not set, defaults to `nccl-inspector-unknown-jobid` or `nccl-inspector-<slurm_job_id>` if running under SLURM.
- `NCCL_INSPECTOR_DUMP_VERBOSE=<0|1>` (default: `0`)
  Enables verbose output including event trace information.
- `NCCL_INSPECTOR_PROM_DUMP=<0|1>` (default: `0`)
  Enables Prometheus format for textfile node exporter output instead of custom JSON.
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
- `NCCL_INSPECTOR_COLL_POOL_SIZE=<entries>` (default: `256`)
  Collective pool initial size/stride.
- `NCCL_INSPECTOR_P2P_POOL_SIZE=<entries>` (default: `256`)
  P2P pool initial size/stride.
- `NCCL_INSPECTOR_COMM_POOL_SIZE=<entries>` (default: `256`)
  Comm pool initial size/stride.
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

**Labels:**
- Collectives: `version`, `slurm_job_id`, `node`, `gpu`, `comm_name`, `n_nodes`, `nranks`, `collective`, `message_size`, `algo_proto`
- P2P: `version`, `slurm_job_id`, `node`, `gpu`, `comm_name`, `n_nodes`, `nranks`, `p2p_operation`, `message_size`

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

Each output file contains JSON objects with the following structure:

```json
{
  "header": {
    "id": "0x7f8c496ae9f661",
    "rank": 2,
    "n_ranks": 8,
    "nnodes": 1
  },
  "metadata": {
    "inspector_output_format_version": "v4.0",
    "git_rev": "",
    "rec_mechanism": "profiler_plugin",
    "dump_timestamp_us": 1748030377748202,
    "hostname": "example-hostname",
    "pid": 1639453
  },
  "coll_perf": {
    "coll": "AllReduce",
    "coll_sn": 1407,
    "coll_msg_size_bytes": 17179869184,
    "coll_exec_time_us": 61974,
    "coll_algobw_gbs": 277.210914,
    "coll_busbw_gbs": 485.119099
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
  "header": {
    "id": "0xe62dedaa97644a",
    "rank": 4,
    "n_ranks": 8,
    "nnodes": 1
  },
  "metadata": {
    "inspector_output_format_version": "v4.0",
    "git_rev": "9019a1912-dirty",
    "rec_mechanism": "nccl_profiler_interface",
    "dump_timestamp_us": 1752867229276385,
    "hostname": "example-hostname",
    "pid": 438776
  },
  "coll_perf": {
    "coll": "ReduceScatter",
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

Multiple such JSON objects are written, one per collective operation per communicator.

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
