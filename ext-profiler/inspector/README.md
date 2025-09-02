# NCCL Inspector Plugin

The NCCL Inspector is a plugin for the NVIDIA Collective Communications Library (NCCL) that provides detailed, per-communicator, per-collective performance and metadata logging. It is designed to help users analyze and debug NCCL collective operations by generating structured JSON output for each operation.

## Related Documentation

- **[Performance Exporter](exporter/example/README.md)** - Tool for analyzing and visualizing NCCL performance data from inspector logs

## Folder Location

The Inspector plugin source is located in:

```
ext-profiler/inspector/
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
export NCCL_PROFILER_PLUGIN=/path/to/nccl/ext-profiler/inspector/libnccl-profiler-inspector.so
export NCCL_INSPECTOR_ENABLE=1
export NCCL_INSPECTOR_DUMP_THREAD_INTERVAL_MICROSECONDS=500
./your_nccl_application
```

### Required Environment Variables

- `NCCL_PROFILER_PLUGIN=/path/to/nccl/ext-profiler/inspector/libnccl-profiler-inspector.so`
  Loads the Inspector plugin into NCCL.
- `NCCL_INSPECTOR_ENABLE=1`
  Enables the Inspector plugin.
- `NCCL_INSPECTOR_DUMP_THREAD_INTERVAL_MICROSECONDS=<interval>`
  Sets the interval (in microseconds) for the internal dump thread to write output. Example: `500`.
- `NCCL_INSPECTOR_DUMP_DIR=<output_dir>` (optional)
  Sets the output directory for logs. If not set, defaults to `nccl-inspector-unknown-jobid` or `nccl-inspector-<slurm_job_id>` if running under SLURM.
- `NCCL_INSPECTOR_DUMP_VERBOSE=<0|1>` (optional)
  Enables verbose output including event trace information. Set to `1` to enable, `0` to disable (default).

### Example Usage

**Single Node:**
```bash
export NCCL_PROFILER_PLUGIN=/path/to/nccl/ext-profiler/inspector/libnccl-profiler-inspector.so
export NCCL_INSPECTOR_ENABLE=1
export NCCL_INSPECTOR_DUMP_THREAD_INTERVAL_MICROSECONDS=500
./build/test/perf/all_reduce_perf -b 8 -e 16G -f 2 -g 8
```

**Multi-Node (SLURM):**
```bash
# Add these environment variables to your SLURM script
export NCCL_PROFILER_PLUGIN=/path/to/nccl/ext-profiler/inspector/libnccl-profiler-inspector.so
export NCCL_INSPECTOR_ENABLE=1
export NCCL_INSPECTOR_DUMP_THREAD_INTERVAL_MICROSECONDS=500
export NCCL_INSPECTOR_DUMP_DIR=/path/to/logs/${SLURM_JOB_ID}/

# Then run your normal NCCL application
srun your_nccl_application
```

## Example Scripts

For detailed example scripts showing how to integrate NCCL Inspector with different workloads, see the **[test/examples/](test/examples/)** directory:

- **Single Node Example**: Basic NCCL performance testing with inspector
- **Multi-Node SLURM Example**: Comprehensive multi-node testing with various collective operations
- **Training Workload Example**: Integration with distributed training workloads

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

- By default, output files are written to:
  - `nccl-inspector-unknown-jobid` (if no SLURM job ID is present)
  - `nccl-inspector-<slurm_job_id>` (if running under SLURM)
- You can override this with the `NCCL_INSPECTOR_DUMP_DIR` environment variable.

## Additional Notes

- The plugin is compatible with standard NCCL workflows and can be used in both single-node and multi-node (SLURM) environments.
- For more details, see the source code and comments in `ext-profiler/inspector/`.

